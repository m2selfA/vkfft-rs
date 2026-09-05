//! Backend-neutral multidimensional C2C FFT composition.
//!
//! Each logical axis is packed into contiguous 1D transform batches, executed by
//! `RecursiveFftIr`, and scattered back to the natural row-major tensor layout.
//! This correctness-first path makes strided axes reuse the same one-dimensional
//! Stockham/Rader/Four-step building blocks instead of adding a second FFT engine.

use crate::complex::Complex64;
use crate::config::{DeviceProfile, Direction, FftConfig, Precision, TransformKind};
use crate::error::{Result, VkFftError};
use crate::kernel_ir::{DispatchGeometry, ScalarType, WorkgroupSize};
use crate::one_dim_ir::{OneDimFftIr, execute_one_dim_fft_ir};
use crate::planner::{C2cDeviceAxisClass, FftPlan};
use crate::zero_pad_ir::{NdZeroPadPassIr, execute_nd_zero_pad_pass};

fn supported_external_storage_pair(compute: ScalarType, storage: ScalarType) -> bool {
    matches!(
        (compute, storage),
        (ScalarType::F32, ScalarType::F16)
            | (ScalarType::F64, ScalarType::F32)
            | (ScalarType::DoubleDouble, ScalarType::F64)
    )
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NdPassOperation {
    PackAxis,
    ScatterAxis,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NdExternalTensorLayout {
    pub dimensions: Vec<usize>,
    pub axis_strides: Vec<usize>,
    pub batch_stride: usize,
}

impl NdExternalTensorLayout {
    pub fn validate(&self, tensor_len: usize) -> Result<()> {
        if self.dimensions.is_empty()
            || self.dimensions.len() != self.axis_strides.len()
            || checked_product(&self.dimensions, "ND external layout tensor length")? != tensor_len
            || self.axis_strides.last().copied() != Some(1)
        {
            return Err(VkFftError::InvalidKernelIr(
                "multidimensional external tensor layout metadata is inconsistent",
            ));
        }
        for axis in (0..self.dimensions.len() - 1).rev() {
            let minimum = self.axis_strides[axis + 1]
                .checked_mul(self.dimensions[axis + 1])
                .ok_or(VkFftError::ArithmeticOverflow {
                    operation: "ND external axis stride span",
                })?;
            if self.axis_strides[axis] < minimum {
                return Err(VkFftError::InvalidKernelIr(
                    "multidimensional external tensor layout overlaps a faster axis",
                ));
            }
        }
        let minimum_batch = self.axis_strides[0].checked_mul(self.dimensions[0]).ok_or(
            VkFftError::ArithmeticOverflow {
                operation: "ND external batch stride span",
            },
        )?;
        if self.batch_stride < minimum_batch {
            return Err(VkFftError::InvalidKernelIr(
                "multidimensional external tensor layout overlaps adjacent batches",
            ));
        }
        Ok(())
    }

    pub fn is_dense(&self) -> Result<bool> {
        Ok(self.axis_strides == dense_axis_strides(&self.dimensions)?)
    }

    pub fn is_tightly_packed(&self) -> Result<bool> {
        Ok(self.is_dense()?
            && self.batch_stride
                == checked_product(&self.dimensions, "ND external tight tensor length")?)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NdFormattedCopyOperation {
    GatherExternalToDense,
    ScatterDenseToExternal,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NdFormattedCopyPassIr {
    pub name: String,
    pub scalar: ScalarType,
    pub input_storage_scalar: ScalarType,
    pub output_storage_scalar: ScalarType,
    pub tensor_len: usize,
    pub batch_count: usize,
    pub external_layout: NdExternalTensorLayout,
    pub workgroup_size: WorkgroupSize,
    pub dispatch: DispatchGeometry,
    pub operation: NdFormattedCopyOperation,
}

impl NdFormattedCopyPassIr {
    pub(crate) fn new(
        name: String,
        scalar: ScalarType,
        external_storage_scalar: ScalarType,
        batch_count: usize,
        external_layout: NdExternalTensorLayout,
        operation: NdFormattedCopyOperation,
        device: DeviceProfile,
    ) -> Result<Self> {
        Self::new_with_max_threads(
            name,
            scalar,
            external_storage_scalar,
            batch_count,
            external_layout,
            operation,
            device.max_threads_per_block,
        )
    }

    pub(crate) fn new_with_max_threads(
        name: String,
        scalar: ScalarType,
        external_storage_scalar: ScalarType,
        batch_count: usize,
        external_layout: NdExternalTensorLayout,
        operation: NdFormattedCopyOperation,
        max_threads_per_block: usize,
    ) -> Result<Self> {
        let tensor_len = checked_product(
            &external_layout.dimensions,
            "formatted copy logical tensor length",
        )?;
        external_layout.validate(tensor_len)?;
        if batch_count == 0 || max_threads_per_block == 0 {
            return Err(VkFftError::InvalidKernelIr(
                "formatted copy requires non-zero batch and device workgroup capacity",
            ));
        }
        if external_storage_scalar != scalar
            && !supported_external_storage_pair(scalar, external_storage_scalar)
        {
            return Err(VkFftError::UnsupportedPrecision {
                backend: "multidimensional formatted copy",
                precision: "unsupported compute/storage scalar pair",
            });
        }
        let local_size = tensor_len.min(max_threads_per_block).max(1);
        let (input_storage_scalar, output_storage_scalar) = match operation {
            NdFormattedCopyOperation::GatherExternalToDense => (external_storage_scalar, scalar),
            NdFormattedCopyOperation::ScatterDenseToExternal => (scalar, external_storage_scalar),
        };
        let pass = Self {
            name,
            scalar,
            input_storage_scalar,
            output_storage_scalar,
            tensor_len,
            batch_count,
            external_layout,
            workgroup_size: WorkgroupSize {
                x: u32::try_from(local_size).map_err(|_| VkFftError::ValueOutOfRange {
                    field: "formatted copy workgroup size",
                })?,
                y: 1,
                z: 1,
            },
            dispatch: DispatchGeometry {
                x: u32::try_from(batch_count).map_err(|_| VkFftError::ValueOutOfRange {
                    field: "formatted copy dispatch count",
                })?,
                y: 1,
                z: 1,
            },
            operation,
        };
        pass.validate()?;
        Ok(pass)
    }

    pub fn validate(&self) -> Result<()> {
        self.external_layout.validate(self.tensor_len)?;
        if self.batch_count == 0
            || self.workgroup_size.x == 0
            || self.workgroup_size.y != 1
            || self.workgroup_size.z != 1
            || self.dispatch.x as usize != self.batch_count
            || self.dispatch.y != 1
            || self.dispatch.z != 1
            || (self.input_storage_scalar != self.scalar
                && !supported_external_storage_pair(self.scalar, self.input_storage_scalar))
            || (self.output_storage_scalar != self.scalar
                && !supported_external_storage_pair(self.scalar, self.output_storage_scalar))
        {
            return Err(VkFftError::InvalidKernelIr(
                "multidimensional formatted copy metadata is inconsistent",
            ));
        }
        let expected_storage = match self.operation {
            NdFormattedCopyOperation::GatherExternalToDense => {
                (self.input_storage_scalar, self.scalar)
            }
            NdFormattedCopyOperation::ScatterDenseToExternal => {
                (self.scalar, self.output_storage_scalar)
            }
        };
        if (self.input_storage_scalar, self.output_storage_scalar) != expected_storage {
            return Err(VkFftError::InvalidKernelIr(
                "multidimensional formatted copy storage ownership is inconsistent",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NdPassIr {
    pub name: String,
    pub scalar: ScalarType,
    pub input_storage_scalar: ScalarType,
    pub output_storage_scalar: ScalarType,
    pub direction: Direction,
    pub tensor_len: usize,
    pub axis: usize,
    pub axis_len: usize,
    pub inner_stride: usize,
    pub line_count: usize,
    pub batch_count: usize,
    /// Physical stored-element distance between adjacent input batches. Internal
    /// natural/packed resources remain dense at `tensor_len`.
    pub input_batch_stride: usize,
    /// Physical stored-element distance between adjacent output batches.
    pub output_batch_stride: usize,
    pub input_external_layout: Option<NdExternalTensorLayout>,
    pub output_external_layout: Option<NdExternalTensorLayout>,
    pub grouped_batch: Option<usize>,
    pub workgroup_size: WorkgroupSize,
    pub dispatch: DispatchGeometry,
    pub operation: NdPassOperation,
}

#[derive(Debug, Clone, Copy)]
struct NdPassShape {
    scalar: ScalarType,
    direction: Direction,
    tensor_len: usize,
    axis: usize,
    axis_len: usize,
    inner_stride: usize,
    line_count: usize,
    batch_count: usize,
    device: DeviceProfile,
}

impl NdPassIr {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn for_axis(
        name: String,
        scalar: ScalarType,
        direction: Direction,
        tensor_len: usize,
        axis: usize,
        axis_len: usize,
        inner_stride: usize,
        line_count: usize,
        batch_count: usize,
        operation: NdPassOperation,
        device: DeviceProfile,
    ) -> Result<Self> {
        Self::new(
            name,
            NdPassShape {
                scalar,
                direction,
                tensor_len,
                axis,
                axis_len,
                inner_stride,
                line_count,
                batch_count,
                device,
            },
            operation,
        )
    }

    fn new(name: String, shape: NdPassShape, operation: NdPassOperation) -> Result<Self> {
        if shape.device.max_threads_per_block == 0 {
            return Err(VkFftError::ResourceLimitExceeded {
                resource: "threads per workgroup",
                required: 1,
                available: 0,
            });
        }
        let local_size = shape
            .axis_len
            .min(shape.device.max_threads_per_block)
            .max(1);
        let transform_count = shape.line_count.checked_mul(shape.batch_count).ok_or(
            VkFftError::ArithmeticOverflow {
                operation: "multidimensional axis transform count",
            },
        )?;
        let pass = Self {
            name,
            scalar: shape.scalar,
            input_storage_scalar: shape.scalar,
            output_storage_scalar: shape.scalar,
            direction: shape.direction,
            tensor_len: shape.tensor_len,
            axis: shape.axis,
            axis_len: shape.axis_len,
            inner_stride: shape.inner_stride,
            line_count: shape.line_count,
            batch_count: shape.batch_count,
            input_batch_stride: shape.tensor_len,
            output_batch_stride: shape.tensor_len,
            input_external_layout: None,
            output_external_layout: None,
            grouped_batch: None,
            workgroup_size: WorkgroupSize {
                x: u32::try_from(local_size).map_err(|_| VkFftError::ValueOutOfRange {
                    field: "multidimensional axis workgroup size",
                })?,
                y: 1,
                z: 1,
            },
            dispatch: DispatchGeometry {
                x: u32::try_from(transform_count).map_err(|_| VkFftError::ValueOutOfRange {
                    field: "multidimensional axis dispatch count",
                })?,
                y: 1,
                z: 1,
            },
            operation,
        };
        pass.validate()?;
        Ok(pass)
    }

    pub(crate) fn with_external_input_storage(mut self, storage: ScalarType) -> Result<Self> {
        if self.operation != NdPassOperation::PackAxis
            || (storage != self.scalar && !supported_external_storage_pair(self.scalar, storage))
        {
            return Err(VkFftError::UnsupportedPrecision {
                backend: "multidimensional pack boundary",
                precision: "unsupported compute/storage scalar pair",
            });
        }
        self.input_storage_scalar = storage;
        self.validate()?;
        Ok(self)
    }

    pub(crate) fn with_external_output_storage(mut self, storage: ScalarType) -> Result<Self> {
        if self.operation != NdPassOperation::ScatterAxis
            || (storage != self.scalar && !supported_external_storage_pair(self.scalar, storage))
        {
            return Err(VkFftError::UnsupportedPrecision {
                backend: "multidimensional scatter boundary",
                precision: "unsupported compute/storage scalar pair",
            });
        }
        self.output_storage_scalar = storage;
        self.validate()?;
        Ok(self)
    }

    pub(crate) fn with_external_input_batch_stride(mut self, stride: usize) -> Result<Self> {
        if self.operation != NdPassOperation::PackAxis || stride < self.tensor_len {
            return Err(VkFftError::InvalidKernelIr(
                "multidimensional formatted input batch stride is invalid",
            ));
        }
        self.input_batch_stride = stride;
        self.validate()?;
        Ok(self)
    }

    pub(crate) fn with_external_output_batch_stride(mut self, stride: usize) -> Result<Self> {
        if self.operation != NdPassOperation::ScatterAxis || stride < self.tensor_len {
            return Err(VkFftError::InvalidKernelIr(
                "multidimensional formatted output batch stride is invalid",
            ));
        }
        self.output_batch_stride = stride;
        self.validate()?;
        Ok(self)
    }

    pub(crate) fn with_external_input_tensor_layout(
        mut self,
        layout: NdExternalTensorLayout,
    ) -> Result<Self> {
        if self.operation != NdPassOperation::PackAxis {
            return Err(VkFftError::InvalidKernelIr(
                "multidimensional formatted input tensor layout requires a pack boundary",
            ));
        }
        layout.validate(self.tensor_len)?;
        self.input_batch_stride = layout.batch_stride;
        self.input_external_layout = Some(layout);
        self.validate()?;
        Ok(self)
    }

    pub(crate) fn with_external_output_tensor_layout(
        mut self,
        layout: NdExternalTensorLayout,
    ) -> Result<Self> {
        if self.operation != NdPassOperation::ScatterAxis {
            return Err(VkFftError::InvalidKernelIr(
                "multidimensional formatted output tensor layout requires a scatter boundary",
            ));
        }
        layout.validate(self.tensor_len)?;
        self.output_batch_stride = layout.batch_stride;
        self.output_external_layout = Some(layout);
        self.validate()?;
        Ok(self)
    }

    pub(crate) fn with_grouped_batch(mut self, grouped_batch: usize) -> Result<Self> {
        if grouped_batch == 0 {
            return Err(VkFftError::InvalidKernelIr(
                "multidimensional axis groupedBatch must be non-zero",
            ));
        }
        self.grouped_batch = Some(grouped_batch);
        self.dispatch.x =
            u32::try_from(self.batch_count.div_ceil(grouped_batch)).map_err(|_| {
                VkFftError::ValueOutOfRange {
                    field: "multidimensional axis grouped dispatch count",
                }
            })?;
        self.validate()?;
        Ok(self)
    }

    pub fn transform_count(&self) -> Result<usize> {
        self.line_count
            .checked_mul(self.batch_count)
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "multidimensional pass transform count",
            })
    }

    pub fn validate(&self) -> Result<()> {
        if self.tensor_len == 0
            || self.axis_len == 0
            || self.inner_stride == 0
            || self.line_count == 0
            || self.batch_count == 0
            || self.input_batch_stride < self.tensor_len
            || self.output_batch_stride < self.tensor_len
            || self.grouped_batch == Some(0)
            || !self.tensor_len.is_multiple_of(self.axis_len)
            || self.line_count != self.tensor_len / self.axis_len
        {
            return Err(VkFftError::InvalidKernelIr(
                "multidimensional axis pass dimensions are inconsistent",
            ));
        }
        if let Some(layout) = &self.input_external_layout {
            layout.validate(self.tensor_len)?;
            if self.operation != NdPassOperation::PackAxis
                || layout.batch_stride != self.input_batch_stride
            {
                return Err(VkFftError::InvalidKernelIr(
                    "multidimensional formatted input tensor layout is attached to the wrong boundary",
                ));
            }
        }
        if let Some(layout) = &self.output_external_layout {
            layout.validate(self.tensor_len)?;
            if self.operation != NdPassOperation::ScatterAxis
                || layout.batch_stride != self.output_batch_stride
            {
                return Err(VkFftError::InvalidKernelIr(
                    "multidimensional formatted output tensor layout is attached to the wrong boundary",
                ));
            }
        }
        if self.scalar == ScalarType::F16
            || (self.input_storage_scalar != self.scalar
                && (self.operation != NdPassOperation::PackAxis
                    || !supported_external_storage_pair(self.scalar, self.input_storage_scalar)))
            || (self.output_storage_scalar != self.scalar
                && (self.operation != NdPassOperation::ScatterAxis
                    || !supported_external_storage_pair(self.scalar, self.output_storage_scalar)))
        {
            return Err(VkFftError::InvalidKernelIr(
                "multidimensional axis storage boundary is inconsistent with compute precision",
            ));
        }
        let axis_span =
            self.axis_len
                .checked_mul(self.inner_stride)
                .ok_or(VkFftError::ArithmeticOverflow {
                    operation: "multidimensional axis stride span",
                })?;
        if axis_span > self.tensor_len {
            return Err(VkFftError::InvalidKernelIr(
                "multidimensional axis stride exceeds the tensor extent",
            ));
        }
        let expected_dispatch = if let Some(grouped_batch) = self.grouped_batch {
            self.batch_count.div_ceil(grouped_batch)
        } else {
            self.transform_count()?
        };
        if self.workgroup_size.x == 0
            || self.workgroup_size.y != 1
            || self.workgroup_size.z != 1
            || self.dispatch.x as usize != expected_dispatch
            || self.dispatch.y != 1
            || self.dispatch.z != 1
        {
            return Err(VkFftError::InvalidKernelIr(
                "multidimensional axis pass launch metadata is inconsistent",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct NdAxisIr {
    pub axis: usize,
    pub pack: NdPassIr,
    pub transform: OneDimFftIr,
    pub scatter: NdPassIr,
}

#[derive(Debug, Clone, PartialEq)]
pub struct NdFftIr {
    pub dimensions: Vec<usize>,
    pub tensor_len: usize,
    pub batch_count: usize,
    pub direction: Direction,
    pub scalar: ScalarType,
    pub external_scalar: ScalarType,
    pub input_buffer_axis_strides: Vec<usize>,
    pub output_buffer_axis_strides: Vec<usize>,
    pub input_buffer_batch_stride: usize,
    pub output_buffer_batch_stride: usize,
    pub input_formatted_copy: Option<NdFormattedCopyPassIr>,
    pub output_formatted_copy: Option<NdFormattedCopyPassIr>,
    pub zero_pad_pass: Option<NdZeroPadPassIr>,
    pub omitted_axes: Vec<bool>,
    pub axes: Vec<NdAxisIr>,
}

impl NdFftIr {
    pub fn build(plan: &FftPlan, direction: Direction, device: DeviceProfile) -> Result<Self> {
        if plan.config.transform != TransformKind::ComplexToComplex {
            return Err(VkFftError::UnsupportedKernelPath(
                "multidimensional FFT IR currently supports C2C transforms only",
            ));
        }
        let (scalar, external_scalar) = match plan.config.precision {
            Precision::F16StorageF32Compute => (ScalarType::F32, ScalarType::F16),
            Precision::F32 => (ScalarType::F32, ScalarType::F32),
            Precision::F64 if device.supports_f64 => (ScalarType::F64, ScalarType::F64),
            Precision::F64ComputeF32Storage if device.supports_f64 => {
                (ScalarType::F64, ScalarType::F32)
            }
            other => {
                return Err(VkFftError::UnsupportedPrecision {
                    backend: "multidimensional FFT IR",
                    precision: precision_name(other),
                });
            }
        };
        if plan.axes.len() != plan.config.dimensions.len() {
            return Err(VkFftError::InvalidKernelIr(
                "multidimensional plan axis metadata does not match dimensions",
            ));
        }
        let tensor_len = checked_product(
            &plan.config.dimensions,
            "multidimensional tensor element count",
        )?;
        let input_buffer_axis_strides = plan.config.resolved_input_buffer_axis_strides()?;
        let output_buffer_axis_strides = plan.config.resolved_output_buffer_axis_strides()?;
        let input_buffer_batch_stride = plan.config.resolved_input_buffer_batch_stride()?;
        let output_buffer_batch_stride = plan.config.resolved_output_buffer_batch_stride()?;
        let fastest_axis = plan.config.dimensions.len() - 1;
        let fastest_axis_len = plan.config.dimensions[fastest_axis];
        let omitted_axes = (0..plan.config.dimensions.len())
            .map(|axis| plan.config.axis_is_omitted(axis))
            .collect::<Vec<_>>();
        let upstream_axis1_grouped_batch = plan
            .config
            .grouped_batch_for_axis(fastest_axis.saturating_sub(1));
        let mut axes = Vec::with_capacity(plan.axes.len());
        let input_external_layout = NdExternalTensorLayout {
            dimensions: plan.config.dimensions.clone(),
            axis_strides: input_buffer_axis_strides.clone(),
            batch_stride: input_buffer_batch_stride,
        };
        let output_external_layout = NdExternalTensorLayout {
            dimensions: plan.config.dimensions.clone(),
            axis_strides: output_buffer_axis_strides.clone(),
            batch_stride: output_buffer_batch_stride,
        };
        input_external_layout.validate(tensor_len)?;
        output_external_layout.validate(tensor_len)?;
        let mut zero_pad_pass = plan
            .config
            .zero_padding
            .iter()
            .any(Option::is_some)
            .then(|| {
                NdZeroPadPassIr::build_with_domain(
                    &plan.config.dimensions,
                    plan.config.batch_count,
                    plan.config.precision,
                    direction,
                    &plan.config.zero_padding,
                    plan.config.zero_padding_domain,
                    device,
                )
            })
            .transpose()?;
        let input_boundary_owned_by_pad = zero_pad_pass
            .as_ref()
            .is_some_and(|pass| pass.operation.is_input_boundary());
        let output_boundary_owned_by_pad = zero_pad_pass
            .as_ref()
            .is_some_and(|pass| pass.operation.is_output_boundary());
        let input_formatted_copy =
            if input_boundary_owned_by_pad && !input_external_layout.is_tightly_packed()? {
                Some(NdFormattedCopyPassIr::new(
                    "vkfft_nd_gather_formatted_input_before_zero_pad".to_owned(),
                    scalar,
                    external_scalar,
                    plan.config.batch_count,
                    input_external_layout.clone(),
                    NdFormattedCopyOperation::GatherExternalToDense,
                    device,
                )?)
            } else {
                None
            };
        let output_formatted_copy =
            if output_boundary_owned_by_pad && !output_external_layout.is_tightly_packed()? {
                Some(NdFormattedCopyPassIr::new(
                    "vkfft_nd_scatter_formatted_output_after_zero_pad".to_owned(),
                    scalar,
                    external_scalar,
                    plan.config.batch_count,
                    output_external_layout.clone(),
                    NdFormattedCopyOperation::ScatterDenseToExternal,
                    device,
                )?)
            } else {
                None
            };
        if input_formatted_copy.is_some() || output_formatted_copy.is_some() {
            zero_pad_pass = zero_pad_pass
                .map(NdZeroPadPassIr::with_compute_storage_boundary)
                .transpose()?;
        }

        // Start with the naturally contiguous last axis, then walk outward. The
        // transform is separable, so this ordering changes locality but not math.
        for axis in (0..plan.axes.len()).rev() {
            if omitted_axes[axis] {
                continue;
            }
            let axis_len = plan.axes[axis].effective_fft_len;
            if axis_len != plan.config.dimensions[axis] {
                return Err(VkFftError::UnsupportedKernelPath(
                    "multidimensional C2C path requires an unchanged effective axis length",
                ));
            }
            let inner_stride = checked_product(
                &plan.config.dimensions[axis + 1..],
                "multidimensional inner axis stride",
            )?;
            let line_count = tensor_len / axis_len;
            let grouped_batch_override = plan.config.grouped_batch_for_axis(axis);
            let transform_batch_count = plan.config.batch_count.checked_mul(line_count).ok_or(
                VkFftError::ArithmeticOverflow {
                    operation: "multidimensional recursive transform batch count",
                },
            )?;
            let mut axis_config = FftConfig::new(vec![axis_len])
                .with_batch_count(transform_batch_count)
                .with_precision(plan.config.precision)
                .with_inverse_normalization(plan.config.normalize_inverse)
                .with_tuning(plan.config.tuning)
                .with_bandwidth_boost(plan.config.bandwidth_boost);
            if let Some(grouped_batch) = grouped_batch_override {
                axis_config = axis_config.with_grouped_batch(0, grouped_batch)?;
            }
            let axis_class = if inner_stride == 1 {
                C2cDeviceAxisClass::Contiguous
            } else {
                C2cDeviceAxisClass::Strided
            };
            let axis_plan = FftPlan::build_c2c_child_for_device(axis_config, device, axis_class)?;
            let transform = if external_scalar != scalar {
                OneDimFftIr::build_internal_compute_storage(&axis_plan, direction, device)?
            } else {
                OneDimFftIr::build(&axis_plan, direction, device)?
            };
            // The ND boundary pass owns spatial zero padding. Keep the fastest-axis
            // retag suppressed while padding is active because its axis-0 splitter has
            // a padding-specific bank-conflict swap policy. Higher/strided-axis blocks,
            // however, never use that axis swap and must still inherit the parent's
            // physical grouping (including user groupedBatch) even with ND padding.
            let transform = if inner_stride == 1 {
                if zero_pad_pass.is_none() {
                    transform.with_axis0_single_upload_block(device)?
                } else {
                    transform
                }
            } else if grouped_batch_override.is_some() {
                transform.with_other_axis_single_upload_block_with_grouped_batch(
                    fastest_axis_len,
                    grouped_batch_override,
                    upstream_axis1_grouped_batch,
                    device,
                )?
            } else {
                transform.with_other_axis_single_upload_block(fastest_axis_len, device)?
            };
            let direction_name = match direction {
                Direction::Forward => "forward",
                Direction::Inverse => "inverse",
            };
            let shape = NdPassShape {
                scalar,
                direction,
                tensor_len,
                axis,
                axis_len,
                inner_stride,
                line_count,
                batch_count: plan.config.batch_count,
                device,
            };
            let mut pack = NdPassIr::new(
                format!("vkfft_nd_pack_axis_{axis}_{direction_name}"),
                shape,
                NdPassOperation::PackAxis,
            )?;
            let mut scatter = NdPassIr::new(
                format!("vkfft_nd_scatter_axis_{axis}_{direction_name}"),
                shape,
                NdPassOperation::ScatterAxis,
            )?;
            if let Some(grouped_batch) = grouped_batch_override {
                pack = pack.with_grouped_batch(grouped_batch)?;
                scatter = scatter.with_grouped_batch(grouped_batch)?;
            }
            axes.push(NdAxisIr {
                axis,
                pack,
                transform,
                scatter,
            });
        }

        if axes.is_empty() {
            return Err(VkFftError::InvalidKernelIr(
                "multidimensional FFT requires at least one active axis",
            ));
        }
        let dense_strides = dense_axis_strides(&plan.config.dimensions)?;
        if input_formatted_copy.is_none() {
            axes[0].pack = axes[0]
                .pack
                .clone()
                .with_external_input_batch_stride(input_buffer_batch_stride)?;
            if input_buffer_axis_strides != dense_strides {
                axes[0].pack = axes[0]
                    .pack
                    .clone()
                    .with_external_input_tensor_layout(input_external_layout.clone())?;
            }
        }
        let last_index = axes.len() - 1;
        if output_formatted_copy.is_none() {
            axes[last_index].scatter = axes[last_index]
                .scatter
                .clone()
                .with_external_output_batch_stride(output_buffer_batch_stride)?;
            if output_buffer_axis_strides != dense_strides {
                axes[last_index].scatter = axes[last_index]
                    .scatter
                    .clone()
                    .with_external_output_tensor_layout(output_external_layout.clone())?;
            }
        }

        if external_scalar != scalar {
            if axes.is_empty() {
                return Err(VkFftError::InvalidKernelIr(
                    "multidimensional FFT requires at least one axis",
                ));
            }
            let last_index = axes.len() - 1;
            let input_boundary_owned_by_pad = zero_pad_pass
                .as_ref()
                .is_some_and(|pass| pass.operation.is_input_boundary());
            let output_boundary_owned_by_pad = zero_pad_pass
                .as_ref()
                .is_some_and(|pass| pass.operation.is_output_boundary());
            if !input_boundary_owned_by_pad {
                let pack = axes[0]
                    .pack
                    .clone()
                    .with_external_input_storage(external_scalar)?;
                axes[0].pack = pack;
            }
            if !output_boundary_owned_by_pad {
                let scatter = axes[last_index]
                    .scatter
                    .clone()
                    .with_external_output_storage(external_scalar)?;
                axes[last_index].scatter = scatter;
            }
        }
        let ir = Self {
            dimensions: plan.config.dimensions.clone(),
            tensor_len,
            batch_count: plan.config.batch_count,
            direction,
            scalar,
            external_scalar,
            input_buffer_axis_strides,
            output_buffer_axis_strides,
            input_buffer_batch_stride,
            output_buffer_batch_stride,
            input_formatted_copy,
            output_formatted_copy,
            zero_pad_pass,
            omitted_axes,
            axes,
        };
        ir.validate()?;
        Ok(ir)
    }

    pub fn validate(&self) -> Result<()> {
        if self.dimensions.is_empty() || self.tensor_len == 0 || self.batch_count == 0 {
            return Err(VkFftError::InvalidKernelIr(
                "multidimensional FFT dimensions and batch count must be non-zero",
            ));
        }
        if checked_product(&self.dimensions, "multidimensional validation tensor size")?
            != self.tensor_len
            || self.omitted_axes.len() != self.dimensions.len()
            || self.axes.len() != self.omitted_axes.iter().filter(|&&omit| !omit).count()
        {
            return Err(VkFftError::InvalidKernelIr(
                "multidimensional FFT tensor metadata is inconsistent",
            ));
        }
        if self.external_scalar != self.scalar
            && !supported_external_storage_pair(self.scalar, self.external_scalar)
        {
            return Err(VkFftError::InvalidKernelIr(
                "multidimensional external storage scalar is incompatible with compute precision",
            ));
        }
        let input_external_layout = NdExternalTensorLayout {
            dimensions: self.dimensions.clone(),
            axis_strides: self.input_buffer_axis_strides.clone(),
            batch_stride: self.input_buffer_batch_stride,
        };
        let output_external_layout = NdExternalTensorLayout {
            dimensions: self.dimensions.clone(),
            axis_strides: self.output_buffer_axis_strides.clone(),
            batch_stride: self.output_buffer_batch_stride,
        };
        input_external_layout.validate(self.tensor_len)?;
        output_external_layout.validate(self.tensor_len)?;
        let input_intra_strided = !input_external_layout.is_dense()?;
        let output_intra_strided = !output_external_layout.is_dense()?;
        let input_copy_required = self
            .zero_pad_pass
            .as_ref()
            .is_some_and(|pass| pass.operation.is_input_boundary())
            && !input_external_layout.is_tightly_packed()?;
        let output_copy_required = self
            .zero_pad_pass
            .as_ref()
            .is_some_and(|pass| pass.operation.is_output_boundary())
            && !output_external_layout.is_tightly_packed()?;
        if self.input_formatted_copy.is_some() != input_copy_required
            || self.output_formatted_copy.is_some() != output_copy_required
        {
            return Err(VkFftError::InvalidKernelIr(
                "multidimensional formatted copy ownership does not match the zero-padding caller boundary",
            ));
        }
        for (copy, layout, operation) in [
            (
                self.input_formatted_copy.as_ref(),
                &input_external_layout,
                NdFormattedCopyOperation::GatherExternalToDense,
            ),
            (
                self.output_formatted_copy.as_ref(),
                &output_external_layout,
                NdFormattedCopyOperation::ScatterDenseToExternal,
            ),
        ] {
            if let Some(copy) = copy {
                copy.validate()?;
                let expected_storage = match operation {
                    NdFormattedCopyOperation::GatherExternalToDense => {
                        (self.external_scalar, self.scalar)
                    }
                    NdFormattedCopyOperation::ScatterDenseToExternal => {
                        (self.scalar, self.external_scalar)
                    }
                };
                if copy.external_layout != *layout
                    || copy.tensor_len != self.tensor_len
                    || copy.batch_count != self.batch_count
                    || copy.scalar != self.scalar
                    || copy.operation != operation
                    || (copy.input_storage_scalar, copy.output_storage_scalar) != expected_storage
                {
                    return Err(VkFftError::InvalidKernelIr(
                        "multidimensional formatted copy metadata is inconsistent",
                    ));
                }
            }
        }
        if let Some(pass) = &self.zero_pad_pass {
            pass.validate()?;
            let expected_storage = if pass.operation.is_input_boundary() {
                (
                    if self.input_formatted_copy.is_some() {
                        self.scalar
                    } else {
                        self.external_scalar
                    },
                    self.scalar,
                )
            } else {
                (
                    self.scalar,
                    if self.output_formatted_copy.is_some() {
                        self.scalar
                    } else {
                        self.external_scalar
                    },
                )
            };
            if pass.dimensions != self.dimensions
                || pass.tensor_len != self.tensor_len
                || pass.batch_count != self.batch_count
                || pass.direction != self.direction
                || pass.scalar != self.scalar
                || (pass.input_storage_scalar, pass.output_storage_scalar) != expected_storage
            {
                return Err(VkFftError::InvalidKernelIr(
                    "multidimensional zero-padding boundary does not match FFT metadata",
                ));
            }
        }
        if self.external_scalar != self.scalar {
            let first = self.axes.first().ok_or(VkFftError::InvalidKernelIr(
                "multidimensional FFT requires at least one axis",
            ))?;
            let last = self.axes.last().ok_or(VkFftError::InvalidKernelIr(
                "multidimensional FFT requires at least one axis",
            ))?;
            let input_boundary_owned_by_pad = self
                .zero_pad_pass
                .as_ref()
                .is_some_and(|pass| pass.operation.is_input_boundary());
            let output_boundary_owned_by_pad = self
                .zero_pad_pass
                .as_ref()
                .is_some_and(|pass| pass.operation.is_output_boundary());
            let expected_first_input = if input_boundary_owned_by_pad {
                self.scalar
            } else {
                self.external_scalar
            };
            let expected_last_output = if output_boundary_owned_by_pad {
                self.scalar
            } else {
                self.external_scalar
            };
            if first.pack.input_storage_scalar != expected_first_input
                || first.pack.output_storage_scalar != self.scalar
                || last.scatter.input_storage_scalar != self.scalar
                || last.scatter.output_storage_scalar != expected_last_output
            {
                return Err(VkFftError::InvalidKernelIr(
                    "multidimensional external storage ownership is inconsistent",
                ));
            }
        }
        let expected_axis_order = (0..self.dimensions.len())
            .rev()
            .filter(|&axis| !self.omitted_axes[axis])
            .collect::<Vec<_>>();
        if self
            .axes
            .iter()
            .map(|axis| axis.axis)
            .ne(expected_axis_order.iter().copied())
        {
            return Err(VkFftError::InvalidKernelIr(
                "multidimensional omitted-axis execution order is inconsistent",
            ));
        }
        for (axis_index, axis_ir) in self.axes.iter().enumerate() {
            axis_ir.pack.validate()?;
            axis_ir.transform.validate()?;
            axis_ir.scatter.validate()?;
            let axis_len =
                *self
                    .dimensions
                    .get(axis_ir.axis)
                    .ok_or(VkFftError::InvalidKernelIr(
                        "multidimensional FFT axis index is out of range",
                    ))?;
            if self.omitted_axes[axis_ir.axis] {
                return Err(VkFftError::InvalidKernelIr(
                    "multidimensional FFT materialized an omitted axis",
                ));
            }
            let expected_transform_batches = self
                .batch_count
                .checked_mul(self.tensor_len / axis_len)
                .ok_or(VkFftError::ArithmeticOverflow {
                    operation: "multidimensional validation transform batch count",
                })?;
            let expected_pack_batch_stride =
                if axis_index == 0 && self.input_formatted_copy.is_none() {
                    self.input_buffer_batch_stride
                } else {
                    self.tensor_len
                };
            let expected_scatter_batch_stride =
                if axis_index + 1 == self.axes.len() && self.output_formatted_copy.is_none() {
                    self.output_buffer_batch_stride
                } else {
                    self.tensor_len
                };
            let expected_input_layout =
                if axis_index == 0 && self.input_formatted_copy.is_none() && input_intra_strided {
                    Some(&input_external_layout)
                } else {
                    None
                };
            let expected_output_layout = if axis_index + 1 == self.axes.len()
                && self.output_formatted_copy.is_none()
                && output_intra_strided
            {
                Some(&output_external_layout)
            } else {
                None
            };
            if axis_ir.pack.operation != NdPassOperation::PackAxis
                || axis_ir.scatter.operation != NdPassOperation::ScatterAxis
                || axis_ir.pack.axis != axis_ir.axis
                || axis_ir.scatter.axis != axis_ir.axis
                || axis_ir.pack.tensor_len != self.tensor_len
                || axis_ir.scatter.tensor_len != self.tensor_len
                || axis_ir.pack.batch_count != self.batch_count
                || axis_ir.scatter.batch_count != self.batch_count
                || axis_ir.pack.input_batch_stride != expected_pack_batch_stride
                || axis_ir.pack.output_batch_stride != self.tensor_len
                || axis_ir.scatter.input_batch_stride != self.tensor_len
                || axis_ir.scatter.output_batch_stride != expected_scatter_batch_stride
                || axis_ir.pack.input_external_layout.as_ref() != expected_input_layout
                || axis_ir.pack.output_external_layout.is_some()
                || axis_ir.scatter.input_external_layout.is_some()
                || axis_ir.scatter.output_external_layout.as_ref() != expected_output_layout
                || axis_ir.transform.logical_len() != axis_len
                || axis_ir.transform.batch_count() != expected_transform_batches
                || axis_ir.pack.grouped_batch != axis_ir.scatter.grouped_batch
                || axis_ir
                    .pack
                    .grouped_batch
                    .is_some_and(|grouped_batch| axis_ir.transform.grouped_batch() != grouped_batch)
                || axis_ir.transform.direction() != self.direction
                || axis_ir.transform.scalar() != self.scalar
            {
                return Err(VkFftError::InvalidKernelIr(
                    "multidimensional FFT axis composition metadata is inconsistent",
                ));
            }
        }
        Ok(())
    }
}

impl NdFftIr {
    pub(crate) fn has_formatted_input_tensor_strides(&self) -> Result<bool> {
        Ok(self.input_buffer_axis_strides != dense_axis_strides(&self.dimensions)?)
    }

    pub(crate) fn has_formatted_output_tensor_strides(&self) -> Result<bool> {
        Ok(self.output_buffer_axis_strides != dense_axis_strides(&self.dimensions)?)
    }

    pub(crate) fn pack_formatted_input<T: Copy + Default>(&self, input: &[T]) -> Result<Vec<T>> {
        self.validate()?;
        let expected = self.tensor_len.checked_mul(self.batch_count).ok_or(
            VkFftError::ArithmeticOverflow {
                operation: "formatted ND logical input element count",
            },
        )?;
        if input.len() != expected {
            return Err(VkFftError::InputLengthMismatch {
                expected,
                actual: input.len(),
            });
        }
        if !self.has_formatted_input_tensor_strides()? {
            return Ok(input.to_vec());
        }
        pack_logical_tensor_batches(
            input,
            &self.dimensions,
            &self.input_buffer_axis_strides,
            self.input_buffer_batch_stride,
        )
    }

    pub(crate) fn unpack_formatted_output<T: Copy>(&self, output: &[T]) -> Result<Vec<T>> {
        self.validate()?;
        if !self.has_formatted_output_tensor_strides()? {
            return Ok(output.to_vec());
        }
        unpack_logical_tensor_batches(
            output,
            &self.dimensions,
            &self.output_buffer_axis_strides,
            self.output_buffer_batch_stride,
            self.batch_count,
        )
    }
}

pub(crate) fn dense_axis_strides(dimensions: &[usize]) -> Result<Vec<usize>> {
    if dimensions.is_empty() {
        return Err(VkFftError::EmptyDimensions);
    }
    let mut strides = vec![1usize; dimensions.len()];
    for axis in (0..dimensions.len() - 1).rev() {
        strides[axis] = strides[axis + 1].checked_mul(dimensions[axis + 1]).ok_or(
            VkFftError::ArithmeticOverflow {
                operation: "dense ND axis stride",
            },
        )?;
    }
    Ok(strides)
}

fn logical_local_to_physical_offset(
    local: usize,
    dimensions: &[usize],
    dense_strides: &[usize],
    physical_strides: &[usize],
) -> Result<usize> {
    let mut offset = 0usize;
    for axis in 0..dimensions.len() {
        let coordinate = (local / dense_strides[axis]) % dimensions[axis];
        offset = offset
            .checked_add(coordinate.checked_mul(physical_strides[axis]).ok_or(
                VkFftError::ArithmeticOverflow {
                    operation: "formatted ND axis address",
                },
            )?)
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "formatted ND tensor address",
            })?;
    }
    Ok(offset)
}

pub(crate) fn pack_logical_tensor_batches<T: Copy + Default>(
    input: &[T],
    dimensions: &[usize],
    physical_strides: &[usize],
    batch_stride: usize,
) -> Result<Vec<T>> {
    let tensor_len = checked_product(dimensions, "formatted ND logical tensor length")?;
    if !input.len().is_multiple_of(tensor_len) {
        return Err(VkFftError::InputLengthMismatch {
            expected: tensor_len,
            actual: input.len(),
        });
    }
    let batch_count = input.len() / tensor_len;
    let physical_len =
        batch_stride
            .checked_mul(batch_count)
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "formatted ND physical input length",
            })?;
    let dense_strides = dense_axis_strides(dimensions)?;
    let mut physical = vec![T::default(); physical_len];
    for batch in 0..batch_count {
        for local in 0..tensor_len {
            let physical_local = logical_local_to_physical_offset(
                local,
                dimensions,
                &dense_strides,
                physical_strides,
            )?;
            physical[batch * batch_stride + physical_local] = input[batch * tensor_len + local];
        }
    }
    Ok(physical)
}

pub(crate) fn unpack_logical_tensor_batches<T: Copy>(
    output: &[T],
    dimensions: &[usize],
    physical_strides: &[usize],
    batch_stride: usize,
    batch_count: usize,
) -> Result<Vec<T>> {
    let tensor_len = checked_product(dimensions, "formatted ND logical tensor length")?;
    let expected_physical =
        batch_stride
            .checked_mul(batch_count)
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "formatted ND physical output length",
            })?;
    if output.len() != expected_physical {
        return Err(VkFftError::InputLengthMismatch {
            expected: expected_physical,
            actual: output.len(),
        });
    }
    let dense_strides = dense_axis_strides(dimensions)?;
    let logical_capacity =
        tensor_len
            .checked_mul(batch_count)
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "formatted ND logical output length",
            })?;
    let mut logical = Vec::with_capacity(logical_capacity);
    for batch in 0..batch_count {
        for local in 0..tensor_len {
            let physical_local = logical_local_to_physical_offset(
                local,
                dimensions,
                &dense_strides,
                physical_strides,
            )?;
            logical.push(output[batch * batch_stride + physical_local]);
        }
    }
    Ok(logical)
}

pub fn execute_nd_fft_ir(ir: &NdFftIr, input: &[Complex64]) -> Result<Vec<Complex64>> {
    ir.validate()?;
    let expected =
        ir.tensor_len
            .checked_mul(ir.batch_count)
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "multidimensional FFT input element count",
            })?;
    if input.len() != expected {
        return Err(VkFftError::InputLengthMismatch {
            expected,
            actual: input.len(),
        });
    }

    let boundary_input = if ir.has_formatted_input_tensor_strides()? {
        let physical = ir.pack_formatted_input(input)?;
        unpack_logical_tensor_batches(
            &physical,
            &ir.dimensions,
            &ir.input_buffer_axis_strides,
            ir.input_buffer_batch_stride,
            ir.batch_count,
        )?
    } else {
        input.to_vec()
    };
    let mut current = if let Some(pass) = &ir.zero_pad_pass
        && pass.operation.is_input_boundary()
    {
        execute_nd_zero_pad_pass(pass, &boundary_input)?
    } else {
        boundary_input
    };
    for axis in &ir.axes {
        let pass = &axis.pack;
        let transform_count = pass.transform_count()?;
        let mut packed = vec![Complex64::new(0.0, 0.0); expected];
        for transform in 0..transform_count {
            let batch = transform / pass.line_count;
            let line = transform % pass.line_count;
            let outer = line / pass.inner_stride;
            let inner = line % pass.inner_stride;
            let packed_base = transform * pass.axis_len;
            for axis_index in 0..pass.axis_len {
                let natural = batch * pass.tensor_len
                    + outer * pass.axis_len * pass.inner_stride
                    + axis_index * pass.inner_stride
                    + inner;
                packed[packed_base + axis_index] = current[natural];
            }
        }
        let transformed = execute_one_dim_fft_ir(&axis.transform, &packed)?;
        let mut next = vec![Complex64::new(0.0, 0.0); expected];
        for transform in 0..transform_count {
            let batch = transform / pass.line_count;
            let line = transform % pass.line_count;
            let outer = line / pass.inner_stride;
            let inner = line % pass.inner_stride;
            let packed_base = transform * pass.axis_len;
            for axis_index in 0..pass.axis_len {
                let natural = batch * pass.tensor_len
                    + outer * pass.axis_len * pass.inner_stride
                    + axis_index * pass.inner_stride
                    + inner;
                next[natural] = transformed[packed_base + axis_index];
            }
        }
        current = next;
    }
    let logical_output = if let Some(pass) = &ir.zero_pad_pass
        && pass.operation.is_output_boundary()
    {
        execute_nd_zero_pad_pass(pass, &current)?
    } else {
        current
    };
    if ir.has_formatted_output_tensor_strides()? {
        let physical = pack_logical_tensor_batches(
            &logical_output,
            &ir.dimensions,
            &ir.output_buffer_axis_strides,
            ir.output_buffer_batch_stride,
        )?;
        ir.unpack_formatted_output(&physical)
    } else {
        Ok(logical_output)
    }
}

fn checked_product(values: &[usize], operation: &'static str) -> Result<usize> {
    values.iter().try_fold(1usize, |product, value| {
        product
            .checked_mul(*value)
            .ok_or(VkFftError::ArithmeticOverflow { operation })
    })
}

fn precision_name(precision: Precision) -> &'static str {
    match precision {
        Precision::F16StorageF32Compute => "f16-storage/f32-compute",
        Precision::F32 => "f32",
        Precision::F64 => "f64",
        Precision::F64ComputeF32Storage => "f64-compute/f32-storage",
        Precision::DoubleDouble => "double-double",
        Precision::DoubleDoubleF64Storage => "double-double/f64-storage",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{Backend, GpuVendor, ZeroPaddingDomain};
    use core::f64::consts::TAU;

    fn device() -> DeviceProfile {
        DeviceProfile {
            shared_memory_bytes: 128 * 1024,
            shared_memory_pow2_bytes: 128 * 1024,
            ..DeviceProfile::generic(Backend::Vulkan, GpuVendor::Nvidia)
        }
    }

    fn sample(length: usize, batch_count: usize) -> Vec<Complex64> {
        (0..length * batch_count)
            .map(|index| {
                let x = index as f64;
                Complex64::new(
                    (0.119 * x).sin() + x * 0.001,
                    (0.043 * x).cos() - x * 0.0017,
                )
            })
            .collect()
    }

    fn dft_2d(
        input: &[Complex64],
        rows: usize,
        cols: usize,
        batch_count: usize,
        direction: Direction,
        normalize_inverse: bool,
    ) -> Vec<Complex64> {
        let tensor_len = rows * cols;
        let mut output = vec![Complex64::new(0.0, 0.0); tensor_len * batch_count];
        let sign = direction.exponent_sign();
        let scale = if direction == Direction::Inverse && normalize_inverse {
            1.0 / tensor_len as f64
        } else {
            1.0
        };
        for batch in 0..batch_count {
            let base = batch * tensor_len;
            for k0 in 0..rows {
                for k1 in 0..cols {
                    let mut sum = Complex64::new(0.0, 0.0);
                    for n0 in 0..rows {
                        for n1 in 0..cols {
                            let angle = sign
                                * TAU
                                * (n0 as f64 * k0 as f64 / rows as f64
                                    + n1 as f64 * k1 as f64 / cols as f64);
                            sum += input[base + n0 * cols + n1] * Complex64::exp_i(angle);
                        }
                    }
                    output[base + k0 * cols + k1] = sum.scale(scale);
                }
            }
        }
        output
    }

    fn dft_one_axis_2d(
        input: &[Complex64],
        rows: usize,
        cols: usize,
        axis: usize,
        direction: Direction,
    ) -> Vec<Complex64> {
        let mut output = vec![Complex64::default(); rows * cols];
        let sign = direction.exponent_sign();
        match axis {
            0 => {
                for col in 0..cols {
                    for k in 0..rows {
                        let mut sum = Complex64::default();
                        for n in 0..rows {
                            let angle = sign * TAU * (n * k) as f64 / rows as f64;
                            sum += input[n * cols + col] * Complex64::exp_i(angle);
                        }
                        output[k * cols + col] = sum;
                    }
                }
            }
            1 => {
                for row in 0..rows {
                    for k in 0..cols {
                        let mut sum = Complex64::default();
                        for n in 0..cols {
                            let angle = sign * TAU * (n * k) as f64 / cols as f64;
                            sum += input[row * cols + n] * Complex64::exp_i(angle);
                        }
                        output[row * cols + k] = sum;
                    }
                }
            }
            _ => panic!("2D helper axis out of range"),
        }
        output
    }

    fn max_error(lhs: &[Complex64], rhs: &[Complex64]) -> f64 {
        lhs.iter()
            .zip(rhs)
            .map(|(lhs, rhs)| (*lhs - *rhs).norm_sqr().sqrt())
            .fold(0.0, f64::max)
    }

    fn stockham_kernel(axis: &NdAxisIr) -> &crate::KernelIr {
        let OneDimFftIr::Recursive(recursive) = &axis.transform else {
            panic!("ND Stockham geometry test unexpectedly selected Bluestein");
        };
        let crate::recursive_ir::RecursiveFftNodeIr::Stockham(kernel) = &recursive.root else {
            panic!("ND Stockham geometry test unexpectedly selected Rader");
        };
        kernel
    }

    #[test]
    fn formatted_batch_stride_is_owned_only_by_external_nd_boundaries() {
        let input = sample(12, 2);
        let dense_plan = FftPlan::build(FftConfig::new(vec![3, 4]).with_batch_count(2)).unwrap();
        let dense = NdFftIr::build(&dense_plan, Direction::Forward, device()).unwrap();
        let formatted_plan = FftPlan::build(
            FftConfig::new(vec![3, 4])
                .with_batch_count(2)
                .with_input_buffer_batch_stride(17)
                .with_output_buffer_batch_stride(19),
        )
        .unwrap();
        let formatted = NdFftIr::build(&formatted_plan, Direction::Forward, device()).unwrap();
        assert_eq!(formatted.input_buffer_batch_stride, 17);
        assert_eq!(formatted.output_buffer_batch_stride, 19);
        assert_eq!(formatted.axes[0].pack.input_batch_stride, 17);
        assert_eq!(formatted.axes[0].pack.output_batch_stride, 12);
        assert_eq!(formatted.axes[0].scatter.input_batch_stride, 12);
        assert_eq!(formatted.axes[0].scatter.output_batch_stride, 12);
        assert_eq!(formatted.axes[1].pack.input_batch_stride, 12);
        assert_eq!(formatted.axes[1].pack.output_batch_stride, 12);
        assert_eq!(formatted.axes[1].scatter.input_batch_stride, 12);
        assert_eq!(formatted.axes[1].scatter.output_batch_stride, 19);

        let actual = execute_nd_fft_ir(&formatted, &input).unwrap();
        let expected = execute_nd_fft_ir(&dense, &input).unwrap();
        assert!(max_error(&actual, &expected) <= 2.0e-10);
    }

    #[test]
    fn formatted_batch_stride_follows_active_grouped_axis_after_omission() {
        let input = sample(12, 2);
        let dense_plan = FftPlan::build(
            FftConfig::new(vec![3, 4])
                .with_batch_count(2)
                .with_omit_dimension(1, true)
                .unwrap()
                .with_grouped_batch(0, 2)
                .unwrap(),
        )
        .unwrap();
        let dense = NdFftIr::build(&dense_plan, Direction::Forward, device()).unwrap();
        let formatted_plan = FftPlan::build(
            FftConfig::new(vec![3, 4])
                .with_batch_count(2)
                .with_omit_dimension(1, true)
                .unwrap()
                .with_grouped_batch(0, 2)
                .unwrap()
                .with_input_buffer_batch_stride(17)
                .with_output_buffer_batch_stride(19),
        )
        .unwrap();
        let formatted = NdFftIr::build(&formatted_plan, Direction::Forward, device()).unwrap();
        assert_eq!(formatted.axes.len(), 1);
        assert_eq!(formatted.axes[0].axis, 0);
        assert_eq!(formatted.axes[0].pack.grouped_batch, Some(2));
        assert_eq!(formatted.axes[0].scatter.grouped_batch, Some(2));
        assert_eq!(formatted.axes[0].pack.input_batch_stride, 17);
        assert_eq!(formatted.axes[0].scatter.output_batch_stride, 19);
        let actual = execute_nd_fft_ir(&formatted, &input).unwrap();
        let expected = execute_nd_fft_ir(&dense, &input).unwrap();
        assert!(max_error(&actual, &expected) <= 2.0e-10);
    }

    #[test]
    fn formatted_tensor_strides_pack_unpack_2d_and_3d_physical_layouts() {
        let plan = FftPlan::build(
            FftConfig::new(vec![3, 4])
                .with_batch_count(2)
                .with_input_buffer_axis_stride(0, 7)
                .unwrap()
                .with_output_buffer_axis_stride(0, 9)
                .unwrap(),
        )
        .unwrap();
        let ir = NdFftIr::build(&plan, Direction::Forward, device()).unwrap();
        assert_eq!(ir.input_buffer_axis_strides, vec![7, 1]);
        assert_eq!(ir.output_buffer_axis_strides, vec![9, 1]);
        assert_eq!(ir.input_buffer_batch_stride, 21);
        assert_eq!(ir.output_buffer_batch_stride, 27);
        assert_eq!(
            ir.axes[0]
                .pack
                .input_external_layout
                .as_ref()
                .unwrap()
                .axis_strides,
            vec![7, 1]
        );
        assert_eq!(
            ir.axes
                .last()
                .unwrap()
                .scatter
                .output_external_layout
                .as_ref()
                .unwrap()
                .axis_strides,
            vec![9, 1]
        );

        let logical = (1usize..=24).collect::<Vec<_>>();
        let physical = ir.pack_formatted_input(&logical).unwrap();
        assert_eq!(physical.len(), 42);
        assert_eq!(&physical[0..4], &[1, 2, 3, 4]);
        assert_eq!(&physical[7..11], &[5, 6, 7, 8]);
        assert_eq!(&physical[14..18], &[9, 10, 11, 12]);
        assert_eq!(&physical[21..25], &[13, 14, 15, 16]);
        assert_eq!(&physical[28..32], &[17, 18, 19, 20]);
        assert_eq!(&physical[35..39], &[21, 22, 23, 24]);
        assert_eq!(physical[4], 0);
        assert_eq!(physical[20], 0);

        let output_physical = pack_logical_tensor_batches(
            &logical,
            &ir.dimensions,
            &ir.output_buffer_axis_strides,
            ir.output_buffer_batch_stride,
        )
        .unwrap();
        assert_eq!(output_physical.len(), 54);
        assert_eq!(
            ir.unpack_formatted_output(&output_physical).unwrap(),
            logical
        );

        let plan3 = FftPlan::build(
            FftConfig::new(vec![2, 3, 4])
                .with_batch_count(2)
                .with_input_buffer_axis_stride(1, 6)
                .unwrap()
                .with_input_buffer_axis_stride(0, 20)
                .unwrap()
                .with_output_buffer_axis_stride(1, 7)
                .unwrap()
                .with_output_buffer_axis_stride(0, 24)
                .unwrap(),
        )
        .unwrap();
        let ir3 = NdFftIr::build(&plan3, Direction::Forward, device()).unwrap();
        assert_eq!(ir3.input_buffer_axis_strides, vec![20, 6, 1]);
        assert_eq!(ir3.output_buffer_axis_strides, vec![24, 7, 1]);
        assert_eq!(ir3.input_buffer_batch_stride, 40);
        assert_eq!(ir3.output_buffer_batch_stride, 48);
        let logical3 = (1usize..=48).collect::<Vec<_>>();
        let physical3 = ir3.pack_formatted_input(&logical3).unwrap();
        assert_eq!(physical3.len(), 80);
        assert_eq!(physical3[35], 24);
        assert_eq!(physical3[75], 48);
        assert_eq!(physical3[4], 0);
        assert_eq!(physical3[19], 0);
    }

    #[test]
    fn formatted_tensor_stride_follows_active_grouped_axis_after_omission() {
        let plan = FftPlan::build(
            FftConfig::new(vec![3, 4])
                .with_batch_count(2)
                .with_omit_dimension(1, true)
                .unwrap()
                .with_grouped_batch(0, 2)
                .unwrap()
                .with_input_buffer_axis_stride(0, 7)
                .unwrap()
                .with_output_buffer_axis_stride(0, 9)
                .unwrap(),
        )
        .unwrap();
        let ir = NdFftIr::build(&plan, Direction::Forward, device()).unwrap();
        assert_eq!(ir.axes.len(), 1);
        assert_eq!(ir.axes[0].axis, 0);
        assert_eq!(ir.axes[0].pack.grouped_batch, Some(2));
        assert_eq!(ir.axes[0].scatter.grouped_batch, Some(2));
        assert_eq!(
            ir.axes[0]
                .pack
                .input_external_layout
                .as_ref()
                .unwrap()
                .axis_strides,
            vec![7, 1]
        );
        assert_eq!(
            ir.axes[0]
                .scatter
                .output_external_layout
                .as_ref()
                .unwrap()
                .axis_strides,
            vec![9, 1]
        );
    }

    #[test]
    fn omit_dimension_skips_only_selected_nd_axis_and_preserves_physical_stride() {
        let input = sample(12, 1);
        for omitted_axis in [0usize, 1] {
            let plan = FftPlan::build(
                FftConfig::new(vec![3, 4])
                    .with_omit_dimension(omitted_axis, true)
                    .unwrap(),
            )
            .unwrap();
            let ir = NdFftIr::build(&plan, Direction::Forward, device()).unwrap();
            let active_axis = 1 - omitted_axis;
            assert_eq!(ir.omitted_axes, vec![omitted_axis == 0, omitted_axis == 1]);
            assert_eq!(ir.axes.len(), 1);
            assert_eq!(ir.axes[0].axis, active_axis);
            assert_eq!(
                ir.axes[0].pack.inner_stride,
                if active_axis == 0 { 4 } else { 1 }
            );
            let actual = execute_nd_fft_ir(&ir, &input).unwrap();
            let expected = dft_one_axis_2d(&input, 3, 4, active_axis, Direction::Forward);
            assert!(
                max_error(&actual, &expected) <= 2.0e-10,
                "omit axis {omitted_axis} changed the active-axis transform"
            );
        }

        let plan = FftPlan::build(FftConfig::new(vec![3, 1, 4])).unwrap();
        let ir = NdFftIr::build(&plan, Direction::Forward, device()).unwrap();
        assert_eq!(ir.omitted_axes, vec![false, true, false]);
        assert_eq!(
            ir.axes.iter().map(|axis| axis.axis).collect::<Vec<_>>(),
            vec![2, 0]
        );
    }

    #[test]
    fn row_major_nd_children_preserve_contiguous_and_strided_device_scoring() {
        let profile = DeviceProfile::generic(Backend::Vulkan, GpuVendor::Nvidia);
        let plan = FftPlan::build(FftConfig::new(vec![1153usize, 4001])).unwrap();
        let ir = NdFftIr::build(&plan, Direction::Forward, profile).unwrap();

        assert_eq!(
            ir.axes.iter().map(|axis| axis.axis).collect::<Vec<_>>(),
            vec![1, 0]
        );
        let fastest = ir.axes.iter().find(|axis| axis.axis == 1).unwrap();
        let outer = ir.axes.iter().find(|axis| axis.axis == 0).unwrap();
        assert!(matches!(fastest.transform, OneDimFftIr::Recursive(_)));
        assert!(matches!(outer.transform, OneDimFftIr::Bluestein(_)));
    }

    #[test]
    fn higher_axis_bluestein_consumes_fixed_device_padding() {
        let mut profile = device();
        profile.shared_memory_bytes = 48 * 1024;
        profile.shared_memory_pow2_bytes = 32 * 1024;
        profile.max_threads_per_block = 1024;
        profile.max_workgroup_size = [1024, 1024, 64];

        // The top-level portable plan only establishes tensor shape/algorithm family.
        // NdFftIr must rebuild the strided outer child with fixed-device padding:
        // p4001 portable Bluestein pads to 8064, NVIDIA/F32 fixed-table pads to 8192.
        let plan = FftPlan::build(FftConfig::new(vec![4001usize, 2usize])).unwrap();
        let ir = NdFftIr::build(&plan, Direction::Forward, profile).unwrap();
        let outer = ir.axes.iter().find(|axis| axis.axis == 0).unwrap();
        let OneDimFftIr::Bluestein(pipeline) = &outer.transform else {
            panic!("ND outer p4001 should use device-scored Bluestein");
        };
        assert_eq!(pipeline.logical_len, 4001);
        assert_eq!(pipeline.convolution_len, 8192);
        for child in [&*pipeline.forward_fft, &*pipeline.inverse_fft] {
            let schedule = child.stockham_upload_schedule.as_ref().unwrap();
            assert_eq!(schedule.register_boost, 1);
            assert_eq!(schedule.upload_count, 2);
            assert_eq!(schedule.axis_split, vec![128, 64]);
            assert!(child.four_step_plan.is_some());
        }

        let shaders = crate::backend::vulkan::VulkanGlslBackend
            .lower_nd_fft(&ir)
            .unwrap();
        assert!(!shaders.is_empty());
        for shader in shaders {
            assert_eq!(shader.compile_spirv().unwrap().words[0], 0x0723_0203);
        }
        ir.validate().unwrap();
    }

    #[test]
    fn higher_axis_stockham_consumes_upstream_xy_block_geometry() {
        let mut profile = device();
        profile.shared_memory_bytes = 48 * 1024;
        profile.shared_memory_pow2_bytes = 32 * 1024;
        profile.max_threads_per_block = 1024;
        profile.max_workgroup_size = [1024, 1024, 64];

        let dimensions = vec![3usize, 8];
        let plan = FftPlan::build(FftConfig::new(dimensions.clone())).unwrap();
        let ir = NdFftIr::build(&plan, Direction::Forward, profile).unwrap();
        assert_eq!(
            ir.axes.iter().map(|axis| axis.axis).collect::<Vec<_>>(),
            vec![1, 0]
        );
        let outer = &ir.axes[1];
        let outer_kernel = stockham_kernel(outer);
        assert_eq!(outer.axis, 0);
        assert_eq!(outer_kernel.workgroup_grouping.transforms_per_workgroup, 8);
        assert_eq!(outer_kernel.workgroup_grouping.threads_per_transform, 1);
        assert_eq!(
            outer_kernel.workgroup_grouping.axis_layout,
            crate::kernel_ir::StockhamWorkgroupAxisLayout::TransformsXThreadsY
        );
        assert_eq!(
            [outer_kernel.workgroup_size.x, outer_kernel.workgroup_size.y],
            [8, 1]
        );
        assert_eq!(outer_kernel.dispatch.x, 1);

        let input = sample(24, 1);
        let actual = execute_nd_fft_ir(&ir, &input).unwrap();
        let expected = dft_2d(&input, 3, 8, 1, Direction::Forward, false);
        assert!(max_error(&actual, &expected) < 2.0e-9 * 24.0);
    }

    #[test]
    fn grouped_nd_c2c_separates_tensor_boundary_and_child_ownership() {
        let mut profile = device();
        profile.shared_memory_bytes = 48 * 1024;
        profile.shared_memory_pow2_bytes = 32 * 1024;
        profile.max_threads_per_block = 1024;
        profile.max_workgroup_size = [1024, 1024, 64];

        let plan = FftPlan::build(
            FftConfig::new(vec![16usize, 8])
                .with_batch_count(5)
                .with_grouped_batch(0, 3)
                .unwrap()
                .with_grouped_batch(1, 3)
                .unwrap(),
        )
        .unwrap();
        let ir = NdFftIr::build(&plan, Direction::Forward, profile).unwrap();
        assert_eq!(
            ir.axes.iter().map(|axis| axis.axis).collect::<Vec<_>>(),
            vec![1, 0]
        );

        let fastest = &ir.axes[0];
        let outer = &ir.axes[1];
        for axis in [fastest, outer] {
            assert_eq!(axis.pack.grouped_batch, Some(3));
            assert_eq!(axis.scatter.grouped_batch, Some(3));
            assert_eq!(axis.pack.dispatch.x, 2);
            assert_eq!(axis.scatter.dispatch.x, 2);
            assert_eq!(axis.transform.grouped_batch(), 3);
        }

        let fastest_kernel = stockham_kernel(fastest);
        assert_eq!(fastest_kernel.batch_count, 80);
        assert_eq!(fastest_kernel.dispatch.x, 27);
        assert_eq!(
            fastest_kernel.workgroup_grouping.transforms_per_workgroup,
            3
        );

        let outer_kernel = stockham_kernel(outer);
        assert_eq!(outer_kernel.batch_count, 40);
        assert_eq!(outer_kernel.dispatch.x, 14);
        assert_eq!(outer_kernel.workgroup_grouping.transforms_per_workgroup, 3);
        assert_eq!(
            outer_kernel.workgroup_grouping.axis_layout,
            crate::kernel_ir::StockhamWorkgroupAxisLayout::TransformsXThreadsY
        );
    }

    #[test]
    fn grouped_nd_c2c_partial_tail_matches_dft_and_compiles_spirv() {
        let mut profile = device();
        profile.shared_memory_bytes = 48 * 1024;
        profile.shared_memory_pow2_bytes = 32 * 1024;
        profile.max_threads_per_block = 1024;
        profile.max_workgroup_size = [1024, 1024, 64];
        let dimensions = vec![3usize, 4];
        let tensor_len = 12usize;
        let batch_count = 5usize;

        for direction in [Direction::Forward, Direction::Inverse] {
            let config = FftConfig::new(dimensions.clone())
                .with_batch_count(batch_count)
                .with_grouped_batch(0, 3)
                .unwrap()
                .with_grouped_batch(1, 3)
                .unwrap()
                .with_inverse_normalization(direction == Direction::Inverse);
            let plan = FftPlan::build(config).unwrap();
            let ir = NdFftIr::build(&plan, direction, profile).unwrap();
            let input = sample(tensor_len, batch_count);
            let actual = execute_nd_fft_ir(&ir, &input).unwrap();
            let expected = dft_2d(
                &input,
                dimensions[0],
                dimensions[1],
                batch_count,
                direction,
                direction == Direction::Inverse,
            );
            assert!(max_error(&actual, &expected) < 2.0e-9 * tensor_len as f64);

            let program = crate::ProgramIr::nd_fft(&ir).unwrap();
            program.validate().unwrap();
            assert_eq!(ir.axes[0].pack.dispatch.x, 2);
            assert_eq!(ir.axes[1].pack.dispatch.x, 2);
            let shaders = crate::backend::vulkan::VulkanGlslBackend
                .lower_nd_fft(&ir)
                .unwrap();
            assert!(!shaders.is_empty());
            for shader in shaders {
                assert_eq!(shader.compile_spirv().unwrap().words[0], 0x0723_0203);
            }
        }
    }

    #[test]
    fn nd_strided_bandwidth_boost_reaches_outer_child_scheduler() {
        let mut profile = device();
        profile.shared_memory_bytes = 48 * 1024;
        profile.shared_memory_pow2_bytes = 32 * 1024;
        profile.max_threads_per_block = 1024;
        profile.max_workgroup_size = [1024, 1024, 64];
        let dimensions = vec![2_097_152usize, 2usize];

        let baseline_plan = FftPlan::build(FftConfig::new(dimensions.clone())).unwrap();
        let baseline = NdFftIr::build(&baseline_plan, Direction::Forward, profile).unwrap();
        let baseline_outer = baseline.axes.iter().find(|axis| axis.axis == 0).unwrap();
        let OneDimFftIr::Recursive(baseline_recursive) = &baseline_outer.transform else {
            panic!("baseline large pow2 outer axis should remain recursive Stockham");
        };
        let baseline_schedule = baseline_recursive
            .stockham_upload_schedule
            .as_ref()
            .unwrap();
        assert_eq!(baseline_schedule.upload_count, 3);
        assert_eq!(baseline_schedule.axis_split, vec![128, 128, 128]);

        let boosted_plan =
            FftPlan::build(FftConfig::new(dimensions).with_bandwidth_boost(2)).unwrap();
        let boosted = NdFftIr::build(&boosted_plan, Direction::Forward, profile).unwrap();
        let boosted_outer = boosted.axes.iter().find(|axis| axis.axis == 0).unwrap();
        let OneDimFftIr::Recursive(boosted_recursive) = &boosted_outer.transform else {
            panic!("boosted large pow2 outer axis should remain recursive Stockham");
        };
        let boosted_schedule = boosted_recursive.stockham_upload_schedule.as_ref().unwrap();
        assert_eq!(boosted_schedule.upload_count, 2);
        assert_eq!(boosted_schedule.axis_split, vec![2_048, 1_024]);
        assert!(boosted_recursive.four_step_plan.is_some());
        boosted.validate().unwrap();
    }

    #[test]
    fn f16_higher_axis_rader_bandwidth_boost_preserves_storage_scheduler_precision() {
        let mut profile = DeviceProfile::generic(Backend::Vulkan, GpuVendor::Nvidia);
        profile.shared_memory_bytes = 32 * 1024;
        profile.shared_memory_pow2_bytes = 32 * 1024;
        profile.max_threads_per_block = 1024;
        profile.max_workgroup_size = [1024, 1024, 64];

        for (bandwidth_boost, expected_split, expected_blocks) in [
            (
                0usize,
                vec![64usize, 68, 64],
                vec![
                    (0usize, 64usize, 34_816usize, 8usize, 8usize),
                    (1, 68, 32_768, 5, 8),
                    (2, 64, 34_816, 8, 8),
                ],
            ),
            (
                2usize,
                vec![544usize, 512],
                vec![
                    (0usize, 544usize, 4_096usize, 34usize, 7usize),
                    (1, 512, 4_352, 64, 8),
                ],
            ),
        ] {
            let plan = FftPlan::build(
                FftConfig::new(vec![278_528usize, 8])
                    .with_precision(Precision::F16StorageF32Compute)
                    .with_bandwidth_boost(bandwidth_boost),
            )
            .unwrap();
            let ir = NdFftIr::build(&plan, Direction::Forward, profile).unwrap();
            let outer = ir.axes.iter().find(|axis| axis.axis == 0).unwrap();
            let OneDimFftIr::Recursive(recursive) = &outer.transform else {
                panic!("N278528 higher-axis F16 Rader child should remain recursive");
            };
            let schedule = recursive
                .rader_forced_upload_schedule
                .as_ref()
                .expect("N278528 higher-axis F16 Rader child should retain upload schedule");
            assert_eq!(schedule.axis_split, expected_split);
            let four_step = recursive
                .four_step_plan
                .as_ref()
                .expect("N278528 higher-axis F16 Rader child should retain Four-step plan");
            assert_eq!(four_step.uploads.len(), expected_blocks.len());
            for (axis_upload_id, fft_len, transform_count, threads, grouped) in expected_blocks {
                let upload = four_step
                    .uploads
                    .iter()
                    .find(|upload| upload.axis_upload_id == axis_upload_id)
                    .unwrap_or_else(|| panic!("missing F16 higher-axis upload {axis_upload_id}"));
                assert_eq!(upload.fft_len, fft_len);
                assert_eq!(upload.transform_count, transform_count);
                let block = upload
                    .axis_block
                    .expect("missing F16 higher-axis physical block");
                assert_eq!(block.threads_per_transform, threads);
                assert_eq!(block.grouped_batch, grouped);
                assert_eq!([block.local_size_x, block.local_size_y], [grouped, threads]);
                assert!(block.transforms_on_x);
                assert!(!block.axis_swapped);
            }
            ir.validate().unwrap();
        }
    }

    #[test]
    fn grouped_f16_higher_axis_forced_rader_materializes_upload_nodes_not_shadow_parent() {
        let mut profile = DeviceProfile::generic(Backend::Vulkan, GpuVendor::Nvidia);
        profile.shared_memory_bytes = 32 * 1024;
        profile.shared_memory_pow2_bytes = 32 * 1024;
        profile.max_threads_per_block = 1024;
        profile.max_workgroup_size = [1024, 1024, 64];

        for (bandwidth_boost, expected_split, expected_blocks) in [
            (
                0usize,
                vec![64usize, 68, 64],
                vec![
                    (0usize, 64usize, 174_080usize, 8usize),
                    (1, 68, 163_840, 5),
                    (2, 64, 174_080, 8),
                ],
            ),
            (
                2usize,
                vec![544usize, 512],
                vec![
                    (0usize, 544usize, 20_480usize, 34usize),
                    (1, 512, 21_760, 64),
                ],
            ),
        ] {
            let plan = FftPlan::build(
                FftConfig::new(vec![278_528usize, 8])
                    .with_batch_count(5)
                    .with_grouped_batch(0, 3)
                    .unwrap()
                    .with_precision(Precision::F16StorageF32Compute)
                    .with_bandwidth_boost(bandwidth_boost),
            )
            .unwrap();
            let ir = NdFftIr::build(&plan, Direction::Forward, profile).unwrap();
            let outer = ir.axes.iter().find(|axis| axis.axis == 0).unwrap();
            assert_eq!(outer.pack.grouped_batch, Some(3));
            assert_eq!(outer.scatter.grouped_batch, Some(3));

            let OneDimFftIr::Recursive(recursive) = &outer.transform else {
                panic!("grouped N278528 higher-axis F16 Rader child should remain recursive");
            };
            let schedule = recursive
                .rader_forced_upload_schedule
                .as_ref()
                .expect("grouped N278528 should retain forced-Rader upload schedule");
            assert_eq!(schedule.axis_split, expected_split);
            let four_step = recursive
                .four_step_plan
                .as_ref()
                .expect("grouped N278528 should retain Four-step plan");
            assert_eq!(four_step.uploads.len(), expected_blocks.len());
            for (axis_upload_id, fft_len, transform_count, threads) in expected_blocks {
                let upload = four_step
                    .uploads
                    .iter()
                    .find(|upload| upload.axis_upload_id == axis_upload_id)
                    .unwrap_or_else(|| panic!("missing grouped F16 upload {axis_upload_id}"));
                assert_eq!(upload.fft_len, fft_len);
                assert_eq!(upload.transform_count, transform_count);
                let block = upload
                    .axis_block
                    .expect("grouped F16 upload should have a physical block");
                assert_eq!(block.threads_per_transform, threads);
                assert_eq!(block.grouped_batch, 3);
                assert_eq!([block.local_size_x, block.local_size_y], [3, threads]);
                assert!(block.transforms_on_x);
                assert!(!block.axis_swapped);
            }
            let uploads = recursive
                .four_step_rader_upload_nodes()
                .unwrap()
                .expect("grouped forced-Rader Four-step nodes must materialize");
            assert_eq!(uploads.len(), four_step.uploads.len());
            crate::ProgramIr::nd_fft(&ir).unwrap().validate().unwrap();
            ir.validate().unwrap();
        }
    }

    #[test]
    fn padded_grouped_f16_higher_axis_rader_keeps_strided_upload_blocks() {
        let mut profile = DeviceProfile::generic(Backend::Vulkan, GpuVendor::Nvidia);
        profile.shared_memory_bytes = 32 * 1024;
        profile.shared_memory_pow2_bytes = 32 * 1024;
        profile.max_threads_per_block = 1024;
        profile.max_workgroup_size = [1024, 1024, 64];

        for (bandwidth_boost, expected_split, expected_blocks) in [
            (
                0usize,
                vec![64usize, 68, 64],
                vec![
                    (0usize, 64usize, 174_080usize, 8usize),
                    (1, 68, 163_840, 5),
                    (2, 64, 174_080, 8),
                ],
            ),
            (
                2usize,
                vec![544usize, 512],
                vec![
                    (0usize, 544usize, 20_480usize, 34usize),
                    (1, 512, 21_760, 64),
                ],
            ),
        ] {
            let config = FftConfig::new(vec![278_528usize, 8])
                .with_batch_count(5)
                .with_grouped_batch(0, 3)
                .unwrap()
                .with_precision(Precision::F16StorageF32Compute)
                .with_bandwidth_boost(bandwidth_boost)
                .with_zero_padding(0, 1, 2)
                .unwrap();
            let plan = FftPlan::build(config).unwrap();
            let ir = NdFftIr::build(&plan, Direction::Forward, profile).unwrap();
            assert!(ir.zero_pad_pass.is_some());
            let outer = ir.axes.iter().find(|axis| axis.axis == 0).unwrap();
            assert_eq!(outer.pack.grouped_batch, Some(3));
            assert_eq!(outer.scatter.grouped_batch, Some(3));

            let OneDimFftIr::Recursive(recursive) = &outer.transform else {
                panic!("padded grouped N278528 F16 Rader child should remain recursive");
            };
            let schedule = recursive
                .rader_forced_upload_schedule
                .as_ref()
                .expect("padded grouped N278528 should retain forced-Rader upload schedule");
            assert_eq!(schedule.axis_split, expected_split);
            let four_step = recursive
                .four_step_plan
                .as_ref()
                .expect("padded grouped N278528 should retain Four-step plan");
            assert_eq!(four_step.uploads.len(), expected_blocks.len());
            for (axis_upload_id, fft_len, transform_count, threads) in expected_blocks {
                let upload = four_step
                    .uploads
                    .iter()
                    .find(|upload| upload.axis_upload_id == axis_upload_id)
                    .unwrap_or_else(|| panic!("missing padded grouped upload {axis_upload_id}"));
                assert_eq!(upload.fft_len, fft_len);
                assert_eq!(upload.transform_count, transform_count);
                let block = upload
                    .axis_block
                    .expect("padded grouped upload should have a physical block");
                assert_eq!(block.threads_per_transform, threads);
                assert_eq!(block.grouped_batch, 3);
                assert_eq!([block.local_size_x, block.local_size_y], [3, threads]);
                assert!(block.transforms_on_x);
                assert!(!block.axis_swapped);
            }
            let uploads = recursive
                .four_step_rader_upload_nodes()
                .unwrap()
                .expect("padded grouped forced-Rader upload nodes must materialize");
            assert_eq!(uploads.len(), four_step.uploads.len());
            crate::ProgramIr::nd_fft(&ir).unwrap().validate().unwrap();
            ir.validate().unwrap();
        }
    }

    #[test]
    fn grouped_higher_axis_two_upload_retags_four_step_xy_geometry() {
        let mut profile = device();
        profile.shared_memory_bytes = 32 * 1024;
        profile.shared_memory_pow2_bytes = 32 * 1024;
        profile.max_threads_per_block = 1024;
        profile.max_workgroup_size = [1024, 1024, 64];

        let plan = FftPlan::build(
            FftConfig::new(vec![6_144usize, 8])
                .with_batch_count(5)
                .with_grouped_batch(0, 3)
                .unwrap()
                .with_grouped_batch(1, 3)
                .unwrap(),
        )
        .unwrap();
        let ir = NdFftIr::build(&plan, Direction::Forward, profile).unwrap();
        let outer = ir.axes.iter().find(|axis| axis.axis == 0).unwrap();
        assert_eq!(outer.pack.grouped_batch, Some(3));
        assert_eq!(outer.scatter.grouped_batch, Some(3));
        assert_eq!(outer.pack.dispatch.x, 2);
        assert_eq!(outer.scatter.dispatch.x, 2);

        let OneDimFftIr::Recursive(recursive) = &outer.transform else {
            panic!("N6144 higher axis should remain recursive Stockham");
        };
        let schedule = recursive.stockham_upload_schedule.as_ref().unwrap();
        assert_eq!(schedule.upload_count, 2);
        assert_eq!(schedule.axis_split, vec![96, 64]);
        let four_step = recursive.four_step_plan.as_ref().unwrap();
        assert_eq!(four_step.uploads.len(), 2);
        assert!(four_step.uploads.iter().all(|upload| {
            upload.axis_block.is_some_and(|block| {
                block.grouped_batch == 3
                    && block.transforms_on_x
                    && !block.axis_swapped
                    && block.local_size_x == 3
                    && block.local_size_y == 8
            })
        }));

        let kernels = recursive
            .four_step_stockham_upload_kernels()
            .unwrap()
            .expect("higher-axis two-upload child should materialize fused kernels");
        assert_eq!(kernels.len(), 2);
        assert_eq!(
            kernels
                .iter()
                .map(|kernel| [kernel.workgroup_size.x, kernel.workgroup_size.y])
                .collect::<Vec<_>>(),
            vec![[3, 8], [3, 8]]
        );
        assert_eq!(
            kernels
                .iter()
                .map(|kernel| kernel.dispatch.x)
                .collect::<Vec<_>>(),
            vec![1_280, 854]
        );

        let shaders = crate::backend::vulkan::VulkanGlslBackend
            .lower_recursive_fft(recursive)
            .unwrap();
        assert_eq!(shaders.len(), 2);
        for shader in shaders {
            assert_eq!(shader.compile_spirv().unwrap().words[0], 0x0723_0203);
        }
        ir.validate().unwrap();
    }

    #[test]
    fn grouped_higher_axis_rader_retags_prime_geometry() {
        let mut profile = device();
        profile.shared_memory_bytes = 48 * 1024;
        profile.shared_memory_pow2_bytes = 32 * 1024;
        profile.max_threads_per_block = 1024;
        profile.max_workgroup_size = [1024, 1024, 64];

        for (prime, expected_threads) in [(47usize, 24usize), (257, 17)] {
            let plan = FftPlan::build(
                FftConfig::new(vec![prime, 8])
                    .with_batch_count(5)
                    .with_grouped_batch(0, 3)
                    .unwrap()
                    .with_grouped_batch(1, 3)
                    .unwrap(),
            )
            .unwrap();
            let ir = NdFftIr::build(&plan, Direction::Forward, profile).unwrap();
            let outer = ir.axes.iter().find(|axis| axis.axis == 0).unwrap();
            assert_eq!(outer.pack.grouped_batch, Some(3));
            assert_eq!(outer.scatter.grouped_batch, Some(3));
            assert_eq!(outer.pack.dispatch.x, 2);
            assert_eq!(outer.scatter.dispatch.x, 2);

            let OneDimFftIr::Recursive(recursive) = &outer.transform else {
                panic!("higher-axis p{prime} should remain recursive Rader");
            };
            match (&recursive.root, prime) {
                (crate::recursive_ir::RecursiveFftNodeIr::DirectRader(rader), 47) => {
                    let block = rader.axis_batch_block.expect("p47 higher-axis block");
                    assert_eq!(block.grouped_batch, 3);
                    assert_eq!(block.threads_per_transform, expected_threads);
                    assert_eq!([block.local_size_x, block.local_size_y], [3, 24]);
                    assert!(block.transforms_on_x);
                    assert!(!block.axis_swapped);
                    assert_eq!(rader.dispatch.x, 14);
                }
                (crate::recursive_ir::RecursiveFftNodeIr::FftRader(rader), 257) => {
                    let block = rader.axis_batch_block.expect("p257 higher-axis block");
                    assert_eq!(block.grouped_batch, 3);
                    assert_eq!(block.threads_per_transform, expected_threads);
                    assert_eq!([block.local_size_x, block.local_size_y], [3, 17]);
                    assert!(block.transforms_on_x);
                    assert!(!block.axis_swapped);
                    assert_eq!(
                        rader
                            .internal_register_schedule
                            .as_ref()
                            .unwrap()
                            .container_fft_num,
                        1
                    );
                    for child in [&*rader.forward_fft, &*rader.inverse_fft] {
                        let OneDimFftIr::Recursive(child) = child else {
                            panic!("p257 convolution child should remain recursive Stockham");
                        };
                        let crate::recursive_ir::RecursiveFftNodeIr::Stockham(kernel) = &child.root
                        else {
                            panic!("p257 convolution child should remain one Stockham root");
                        };
                        assert_eq!([kernel.workgroup_size.x, kernel.workgroup_size.y], [3, 17]);
                        assert_eq!(kernel.dispatch.x, 14);
                    }
                }
                _ => panic!("unexpected higher-axis Rader root for p{prime}"),
            }
            let shaders = crate::backend::vulkan::VulkanGlslBackend
                .lower_recursive_fft(recursive)
                .unwrap();
            assert!(!shaders.is_empty());
            for shader in shaders {
                assert_eq!(shader.compile_spirv().unwrap().words[0], 0x0723_0203);
            }
            ir.validate().unwrap();
        }
    }

    #[test]
    fn axis2_grouped_forced_rader_fails_closed_when_scheduler_block_exceeds_shared_capacity() {
        let mut profile = DeviceProfile::generic(Backend::Vulkan, GpuVendor::Nvidia);
        profile.shared_memory_bytes = 2 * 1024;
        profile.shared_memory_pow2_bytes = 2 * 1024;
        profile.max_threads_per_block = 64;
        profile.max_workgroup_size = [64, 64, 64];
        profile.coalesced_memory_bytes = 32;
        let precision = Precision::F64ComputeF32Storage;
        let tuning = crate::PlannerTuning::for_device(profile, precision);
        let plan = FftPlan::build(
            FftConfig::new(vec![139_264usize, 4, 8])
                .with_batch_count(5)
                .with_precision(precision)
                .with_tuning(tuning)
                .with_grouped_batch(0, 3)
                .unwrap(),
        )
        .unwrap();
        let error = NdFftIr::build(&plan, Direction::Forward, profile).unwrap_err();
        assert_eq!(
            error,
            crate::VkFftError::ResourceLimitExceeded {
                resource: "axis-0 grouped Stockham shared memory",
                required: 3_072,
                available: 2_048,
            }
        );
    }

    #[test]
    fn grouped_higher_axis_composite_rader_retags_only_parent_boundary() {
        let mut profile = device();
        profile.shared_memory_bytes = 48 * 1024;
        profile.shared_memory_pow2_bytes = 32 * 1024;
        profile.max_threads_per_block = 1024;
        profile.max_workgroup_size = [1024, 1024, 64];

        let plan = FftPlan::build(
            FftConfig::new(vec![34usize, 8])
                .with_batch_count(5)
                .with_grouped_batch(0, 3)
                .unwrap()
                .with_grouped_batch(1, 3)
                .unwrap(),
        )
        .unwrap();
        let ir = NdFftIr::build(&plan, Direction::Forward, profile).unwrap();
        let outer = ir.axes.iter().find(|axis| axis.axis == 0).unwrap();
        let OneDimFftIr::Recursive(recursive) = &outer.transform else {
            panic!("N34 higher axis should remain recursive composite Rader");
        };
        let crate::recursive_ir::RecursiveFftNodeIr::CooleyTukey(root) = &recursive.root else {
            panic!("N34 higher-axis composite Rader should keep a Cooley root");
        };
        let parent = root
            .pack_right
            .axis_batch_block
            .expect("higher-axis composite Rader parent block");
        assert_eq!(parent.grouped_batch, 3);
        assert_eq!(parent.local_size_x, 3);
        assert_eq!(parent.local_size_y, parent.threads_per_transform);
        assert!(parent.transforms_on_x);
        assert!(!parent.axis_swapped);
        assert_eq!(root.twiddle_transpose.axis_batch_block, Some(parent));
        assert_eq!(root.scatter_output.axis_batch_block, Some(parent));
        assert_eq!(root.pack_right.dispatch.x, 14);
        assert_eq!(root.twiddle_transpose.dispatch.x, 14);
        assert_eq!(root.scatter_output.dispatch.x, 14);

        fn collect_rader_blocks(
            node: &crate::recursive_ir::RecursiveFftNodeIr,
            blocks: &mut Vec<Option<crate::scheduler::StockhamAxisBlockSchedule>>,
        ) {
            match node {
                crate::recursive_ir::RecursiveFftNodeIr::Stockham(_) => {}
                crate::recursive_ir::RecursiveFftNodeIr::DirectRader(rader) => {
                    blocks.push(rader.axis_batch_block);
                }
                crate::recursive_ir::RecursiveFftNodeIr::FftRader(rader) => {
                    blocks.push(rader.axis_batch_block);
                }
                crate::recursive_ir::RecursiveFftNodeIr::CooleyTukey(node) => {
                    collect_rader_blocks(&node.left, blocks);
                    collect_rader_blocks(&node.right, blocks);
                }
            }
        }
        let mut nested = Vec::new();
        collect_rader_blocks(&recursive.root, &mut nested);
        assert!(!nested.is_empty());
        assert!(nested.iter().all(|block| *block != Some(parent)));

        let shaders = crate::backend::vulkan::VulkanGlslBackend
            .lower_recursive_fft(recursive)
            .unwrap();
        assert!(!shaders.is_empty());
        for shader in shaders {
            assert_eq!(shader.compile_spirv().unwrap().words[0], 0x0723_0203);
        }
        ir.validate().unwrap();
    }

    #[test]
    fn f64_higher_axis_direct_rader_keeps_upstream_auto_coalescing_floor() {
        let mut profile = DeviceProfile::generic(Backend::Vulkan, GpuVendor::Nvidia);
        profile.shared_memory_bytes = 48 * 1024;
        profile.shared_memory_pow2_bytes = 32 * 1024;
        profile.max_threads_per_block = 128;
        profile.max_workgroup_size = [1024, 1024, 64];

        let mut tuning = crate::PlannerTuning::portable();
        tuning.min_rader_fft_prime = 53;
        tuning.validate().unwrap();
        let dimensions = vec![47usize, 8];
        let batch_count = 5usize;
        let plan = FftPlan::build(
            FftConfig::new(dimensions.clone())
                .with_batch_count(batch_count)
                .with_precision(Precision::F64)
                .with_tuning(tuning),
        )
        .unwrap();
        let ir = NdFftIr::build(&plan, Direction::Forward, profile).unwrap();
        let outer = ir.axes.iter().find(|axis| axis.axis == 0).unwrap();
        let OneDimFftIr::Recursive(recursive) = &outer.transform else {
            panic!("F64 p47 higher axis should remain recursive Direct-Rader");
        };
        let crate::recursive_ir::RecursiveFftNodeIr::DirectRader(direct) = &recursive.root else {
            panic!("F64 p47 higher axis should materialize one Direct-Rader root");
        };
        let block = direct
            .axis_batch_block
            .expect("F64 p47 higher axis should receive an automatic physical block");
        assert_eq!(direct.batch_count, 40);
        assert_eq!(block.threads_per_transform, 24);
        assert_eq!(block.grouped_batch, 2);
        assert_eq!([block.local_size_x, block.local_size_y], [2, 24]);
        assert!(block.transforms_on_x);
        assert!(!block.axis_swapped);

        let program = crate::ProgramIr::nd_fft(&ir).unwrap();
        program.validate().unwrap();
        let shaders = crate::backend::vulkan::VulkanGlslBackend
            .lower_nd_fft(&ir)
            .unwrap();
        assert!(shaders.iter().any(|shader| {
            shader.workgroup_size.x == 2
                && shader.workgroup_size.y == 24
                && shader.glsl.contains("gl_LocalInvocationID.y")
        }));
        for shader in shaders {
            assert_eq!(shader.compile_spirv().unwrap().words[0], 0x0723_0203);
        }

        let tensor_len = dimensions.iter().product::<usize>();
        let input = sample(tensor_len, batch_count);
        let actual = execute_nd_fft_ir(&ir, &input).unwrap();
        let expected = dft_2d(
            &input,
            dimensions[0],
            dimensions[1],
            batch_count,
            Direction::Forward,
            false,
        );
        assert!(max_error(&actual, &expected) < 5.0e-8 * tensor_len as f64);
        ir.validate().unwrap();
    }

    #[test]
    fn grouped_higher_axis_composite_direct_rader_preserves_type1_thread_floor() {
        let mut profile = device();
        profile.shared_memory_bytes = 48 * 1024;
        profile.shared_memory_pow2_bytes = 32 * 1024;
        profile.max_threads_per_block = 1024;
        profile.max_workgroup_size = [1024, 1024, 64];

        let plan = FftPlan::build(
            FftConfig::new(vec![94usize, 8])
                .with_batch_count(2)
                .with_grouped_batch(0, 3)
                .unwrap()
                .with_grouped_batch(1, 3)
                .unwrap(),
        )
        .unwrap();
        let ir = NdFftIr::build(&plan, Direction::Forward, profile).unwrap();
        let outer = ir.axes.iter().find(|axis| axis.axis == 0).unwrap();
        let OneDimFftIr::Recursive(recursive) = &outer.transform else {
            panic!("N94 higher axis should remain recursive composite direct-Rader");
        };
        let crate::recursive_ir::RecursiveFftNodeIr::CooleyTukey(root) = &recursive.root else {
            panic!("N94 higher-axis composite direct-Rader should keep a Cooley root");
        };
        assert!(matches!(
            root.right,
            crate::recursive_ir::RecursiveFftNodeIr::DirectRader(_)
        ));
        let parent = root
            .pack_right
            .axis_batch_block
            .expect("N94 higher-axis composite direct-Rader parent block");
        assert_eq!(parent.threads_per_transform, 48);
        assert_eq!(parent.grouped_batch, 3);
        assert_eq!([parent.local_size_x, parent.local_size_y], [3, 48]);
        assert!(parent.transforms_on_x);
        assert!(!parent.axis_swapped);
        assert_eq!(root.twiddle_transpose.axis_batch_block, Some(parent));
        assert_eq!(root.scatter_output.axis_batch_block, Some(parent));

        let shaders = crate::backend::vulkan::VulkanGlslBackend
            .lower_recursive_fft(recursive)
            .unwrap();
        assert!(!shaders.is_empty());
        for shader in shaders {
            assert_eq!(shader.compile_spirv().unwrap().words[0], 0x0723_0203);
        }
        ir.validate().unwrap();
    }

    #[test]
    fn grouped_higher_axis_bluestein_retags_wrapper_and_convolution_child() {
        let mut profile = device();
        profile.shared_memory_bytes = 48 * 1024;
        profile.shared_memory_pow2_bytes = 32 * 1024;
        profile.max_threads_per_block = 1024;
        profile.max_workgroup_size = [1024, 1024, 64];
        let mut tuning = crate::PlannerTuning::portable();
        tuning.max_rader_fft_prime = 100;

        let plan = FftPlan::build(
            FftConfig::new(vec![103usize, 8])
                .with_batch_count(5)
                .with_grouped_batch(0, 3)
                .unwrap()
                .with_grouped_batch(1, 3)
                .unwrap()
                .with_tuning(tuning),
        )
        .unwrap();
        let ir = NdFftIr::build(&plan, Direction::Forward, profile).unwrap();
        let outer = ir.axes.iter().find(|axis| axis.axis == 0).unwrap();
        let OneDimFftIr::Bluestein(pipeline) = &outer.transform else {
            panic!("forced p103 higher axis should remain Bluestein");
        };
        assert_eq!(pipeline.logical_len, 103);
        assert_eq!(pipeline.grouped_batch, 3);
        let wrapper = pipeline
            .preprocess
            .axis_batch_block
            .expect("higher-axis Bluestein wrapper block");
        assert_eq!(wrapper.grouped_batch, 3);
        assert_eq!(wrapper.local_size_x, 3);
        assert_eq!(wrapper.local_size_y, 128);
        assert_eq!(wrapper.threads_per_transform, 128);
        assert!(wrapper.transforms_on_x);
        assert!(!wrapper.axis_swapped);
        assert_eq!(pipeline.multiply.axis_batch_block, Some(wrapper));
        assert_eq!(pipeline.postprocess.axis_batch_block, Some(wrapper));
        assert_eq!(
            [
                pipeline.preprocess.workgroup_size.x,
                pipeline.preprocess.workgroup_size.y,
            ],
            [3, 128]
        );

        for child in [&*pipeline.forward_fft, &*pipeline.inverse_fft] {
            let crate::recursive_ir::RecursiveFftNodeIr::Stockham(kernel) = &child.root else {
                panic!(
                    "p103 M{} convolution should remain one Stockham root",
                    pipeline.convolution_len
                );
            };
            assert_eq!(kernel.workgroup_grouping.transforms_per_workgroup, 3);
            assert_eq!(
                kernel.workgroup_grouping.axis_layout,
                crate::kernel_ir::StockhamWorkgroupAxisLayout::TransformsXThreadsY
            );
            assert_eq!(kernel.workgroup_size.x, 3);
        }

        let program = crate::ProgramIr::nd_fft(&ir).unwrap();
        program.validate().unwrap();
        let shaders = crate::backend::vulkan::VulkanGlslBackend
            .lower_nd_fft(&ir)
            .unwrap();
        assert!(!shaders.is_empty());
        assert!(shaders.iter().any(|shader| {
            shader.workgroup_size.x == 3
                && shader.workgroup_size.y == 128
                && shader.glsl.contains("gl_LocalInvocationID.y")
                && shader.glsl.contains("vkfft_transform_slot")
        }));
        for shader in shaders {
            assert_eq!(shader.compile_spirv().unwrap().words[0], 0x0723_0203);
        }
        for backend in [crate::Backend::Cuda, crate::Backend::OpenCl] {
            let native = crate::backend::native::NativeSourceBackend::new(backend)
                .lower_nd_fft(&ir)
                .unwrap();
            assert!(native.iter().any(|shader| {
                shader.workgroup_size.x == 3
                    && shader.workgroup_size.y == 128
                    && match backend {
                        crate::Backend::Cuda => shader.source.contains("threadIdx.y"),
                        crate::Backend::OpenCl => shader.source.contains("get_local_id(1)"),
                        _ => false,
                    }
            }));
        }
        ir.validate().unwrap();
    }

    #[test]
    fn higher_axis_bluestein_auto_groups_wrapper_without_logical_group_override() {
        let mut profile = device();
        profile.shared_memory_bytes = 48 * 1024;
        profile.shared_memory_pow2_bytes = 32 * 1024;
        profile.max_threads_per_block = 1024;
        profile.max_workgroup_size = [1024, 1024, 64];
        let mut tuning = crate::PlannerTuning::portable();
        tuning.max_rader_fft_prime = 100;

        let plan = FftPlan::build(FftConfig::new(vec![103usize, 8]).with_tuning(tuning)).unwrap();
        let ir = NdFftIr::build(&plan, Direction::Forward, profile).unwrap();
        let outer = ir.axes.iter().find(|axis| axis.axis == 0).unwrap();
        let OneDimFftIr::Bluestein(pipeline) = &outer.transform else {
            panic!("forced p103 higher axis should remain Bluestein");
        };
        assert_eq!(pipeline.grouped_batch, 1);
        let wrapper = pipeline
            .preprocess
            .axis_batch_block
            .expect("default higher-axis Bluestein should receive a physical wrapper block");
        assert_eq!(wrapper.grouped_batch, 4);
        assert_eq!(wrapper.threads_per_transform, 128);
        assert_eq!([wrapper.local_size_x, wrapper.local_size_y], [4, 128]);
        assert!(wrapper.transforms_on_x);
        assert_eq!(pipeline.preprocess.dispatch.x, 2);
        assert_eq!(pipeline.multiply.axis_batch_block, Some(wrapper));
        assert_eq!(pipeline.postprocess.axis_batch_block, Some(wrapper));

        for child in [&*pipeline.forward_fft, &*pipeline.inverse_fft] {
            let crate::recursive_ir::RecursiveFftNodeIr::Stockham(kernel) = &child.root else {
                panic!("p103 convolution should remain one Stockham root");
            };
            assert_eq!(kernel.workgroup_grouping.transforms_per_workgroup, 4);
            assert_eq!(
                kernel.workgroup_grouping.axis_layout,
                crate::kernel_ir::StockhamWorkgroupAxisLayout::TransformsXThreadsY
            );
            assert_eq!(kernel.dispatch.x, 2);
        }

        let shaders = crate::backend::vulkan::VulkanGlslBackend
            .lower_nd_fft(&ir)
            .unwrap();
        assert!(shaders.iter().any(|shader| {
            shader.workgroup_size.x == 4
                && shader.workgroup_size.y == 128
                && shader.glsl.contains("vkfft_transform_slot")
        }));
        for shader in shaders {
            assert_eq!(shader.compile_spirv().unwrap().words[0], 0x0723_0203);
        }
        ir.validate().unwrap();
    }

    #[test]
    fn higher_axis_stockham_caps_grouping_by_fastest_physical_dimension() {
        let mut profile = device();
        profile.shared_memory_bytes = 48 * 1024;
        profile.shared_memory_pow2_bytes = 32 * 1024;
        profile.max_threads_per_block = 1024;
        profile.max_workgroup_size = [1024, 1024, 64];

        let square_plan = FftPlan::build(FftConfig::new(vec![64, 64])).unwrap();
        let square = NdFftIr::build(&square_plan, Direction::Forward, profile).unwrap();
        let square_outer = stockham_kernel(&square.axes[1]);
        assert_eq!(square.axes[1].axis, 0);
        assert_eq!(square_outer.workgroup_grouping.transforms_per_workgroup, 16);
        assert_eq!(square_outer.workgroup_grouping.threads_per_transform, 8);
        assert_eq!(
            [square_outer.workgroup_size.x, square_outer.workgroup_size.y],
            [16, 8]
        );
        assert_eq!(square_outer.dispatch.x, 4);

        let cube_plan = FftPlan::build(FftConfig::new(vec![64, 8, 8])).unwrap();
        let cube = NdFftIr::build(&cube_plan, Direction::Forward, profile).unwrap();
        let cube_outer = cube
            .axes
            .iter()
            .find(|axis| axis.axis == 0)
            .expect("3D outer axis");
        let cube_outer_kernel = stockham_kernel(cube_outer);
        assert_eq!(cube_outer_kernel.batch_count, 64);
        assert_eq!(
            cube_outer_kernel
                .workgroup_grouping
                .transforms_per_workgroup,
            8
        );
        assert_eq!(
            cube_outer_kernel.workgroup_grouping.threads_per_transform,
            8
        );
        assert_eq!(
            [
                cube_outer_kernel.workgroup_size.x,
                cube_outer_kernel.workgroup_size.y,
            ],
            [8, 8]
        );
        assert_eq!(cube_outer_kernel.dispatch.x, 8);
    }

    #[test]
    fn two_dimensional_fft_matches_direct_dft() {
        let dimensions = vec![3usize, 4];
        let tensor_len = 12usize;
        let batch_count = 2usize;
        for direction in [Direction::Forward, Direction::Inverse] {
            let config = FftConfig::new(dimensions.clone())
                .with_batch_count(batch_count)
                .with_inverse_normalization(direction == Direction::Inverse);
            let plan = FftPlan::build(config).unwrap();
            let ir = NdFftIr::build(&plan, direction, device()).unwrap();
            assert_eq!(ir.axes.len(), 2);
            assert_eq!(ir.axes[0].axis, 1);
            assert_eq!(ir.axes[1].axis, 0);
            let input = sample(tensor_len, batch_count);
            let actual = execute_nd_fft_ir(&ir, &input).unwrap();
            let expected = dft_2d(
                &input,
                dimensions[0],
                dimensions[1],
                batch_count,
                direction,
                direction == Direction::Inverse,
            );
            assert!(max_error(&actual, &expected) < 2.0e-9 * tensor_len as f64);
        }
    }

    #[test]
    fn multidimensional_spatial_zero_padding_matches_manual_zero_tensor() {
        let dimensions = vec![3usize, 4];
        let tensor_len = 12usize;
        let config = FftConfig::new(dimensions.clone())
            .with_zero_padding(0, 2, 3)
            .unwrap()
            .with_zero_padding(1, 3, 4)
            .unwrap();
        let plan = FftPlan::build(config).unwrap();
        let ir = NdFftIr::build(&plan, Direction::Forward, device()).unwrap();
        let zero_pad = ir.zero_pad_pass.as_ref().expect("ND zero-padding boundary");
        assert!(zero_pad.contains_linear_index(2 * 4));
        assert!(zero_pad.contains_linear_index(3));
        assert!(!zero_pad.contains_linear_index(5));

        let mut input = sample(tensor_len, 1);
        for row in 0..3 {
            for col in 0..4 {
                if row == 2 || col == 3 {
                    input[row * 4 + col] =
                        Complex64::new(10_000.0 + row as f64, -20_000.0 - col as f64);
                }
            }
        }
        let mut manual = input.clone();
        for row in 0..3 {
            for col in 0..4 {
                if row == 2 || col == 3 {
                    manual[row * 4 + col] = Complex64::default();
                }
            }
        }
        let actual = execute_nd_fft_ir(&ir, &input).unwrap();
        let expected = dft_2d(&manual, 3, 4, 1, Direction::Forward, false);
        assert!(max_error(&actual, &expected) < 2.0e-9 * tensor_len as f64);

        let config = FftConfig::new(dimensions)
            .with_inverse_normalization(true)
            .with_zero_padding(0, 2, 3)
            .unwrap()
            .with_zero_padding(1, 3, 4)
            .unwrap();
        let plan = FftPlan::build(config).unwrap();
        let inverse = NdFftIr::build(&plan, Direction::Inverse, device()).unwrap();
        let spectrum = dft_2d(&sample(tensor_len, 1), 3, 4, 1, Direction::Forward, false);
        let actual = execute_nd_fft_ir(&inverse, &spectrum).unwrap();
        let mut expected = dft_2d(&spectrum, 3, 4, 1, Direction::Inverse, true);
        for row in 0..3 {
            for col in 0..4 {
                if row == 2 || col == 3 {
                    expected[row * 4 + col] = Complex64::default();
                }
            }
        }
        assert!(max_error(&actual, &expected) < 2.0e-9 * tensor_len as f64);
    }

    #[test]
    fn formatted_strides_compose_with_spatial_and_frequency_zero_padding() {
        let dimensions = vec![3usize, 4];
        let tensor_len = 12usize;
        let batch_count = 2usize;
        let input = sample(tensor_len, batch_count);
        for (direction, domain, pad_input_boundary) in [
            (Direction::Forward, ZeroPaddingDomain::Spatial, true),
            (Direction::Inverse, ZeroPaddingDomain::Spatial, false),
            (Direction::Forward, ZeroPaddingDomain::Frequency, false),
            (Direction::Inverse, ZeroPaddingDomain::Frequency, true),
        ] {
            let config = FftConfig::new(dimensions.clone())
                .with_batch_count(batch_count)
                .with_inverse_normalization(direction == Direction::Inverse)
                .with_input_buffer_axis_stride(0, 7)
                .unwrap()
                .with_output_buffer_axis_stride(0, 9)
                .unwrap()
                .with_zero_padding(1, 1, 3)
                .unwrap()
                .with_zero_padding_domain(domain);
            let plan = FftPlan::build(config).unwrap();
            let ir = NdFftIr::build(&plan, direction, device()).unwrap();
            assert_eq!(ir.input_formatted_copy.is_some(), pad_input_boundary);
            assert_eq!(ir.output_formatted_copy.is_some(), !pad_input_boundary);
            let zero_pad = ir.zero_pad_pass.as_ref().unwrap();
            assert_eq!(zero_pad.operation.is_input_boundary(), pad_input_boundary);
            assert_eq!(zero_pad.input_storage_scalar, ir.scalar);
            assert_eq!(zero_pad.output_storage_scalar, ir.scalar);
            if pad_input_boundary {
                assert_eq!(ir.axes[0].pack.input_batch_stride, tensor_len);
                assert!(ir.axes[0].pack.input_external_layout.is_none());
                assert!(
                    ir.axes
                        .last()
                        .unwrap()
                        .scatter
                        .output_external_layout
                        .is_some()
                );
            } else {
                assert!(ir.axes[0].pack.input_external_layout.is_some());
                assert_eq!(
                    ir.axes.last().unwrap().scatter.output_batch_stride,
                    tensor_len
                );
                assert!(
                    ir.axes
                        .last()
                        .unwrap()
                        .scatter
                        .output_external_layout
                        .is_none()
                );
            }

            let mut boundary = input.clone();
            if pad_input_boundary {
                for batch in 0..batch_count {
                    let base = batch * tensor_len;
                    for row in 0..3 {
                        for col in 1..3 {
                            boundary[base + row * 4 + col] = Complex64::default();
                        }
                    }
                }
            }
            let mut expected = dft_2d(
                &boundary,
                3,
                4,
                batch_count,
                direction,
                direction == Direction::Inverse,
            );
            if !pad_input_boundary {
                for batch in 0..batch_count {
                    let base = batch * tensor_len;
                    for row in 0..3 {
                        for col in 1..3 {
                            expected[base + row * 4 + col] = Complex64::default();
                        }
                    }
                }
            }
            let actual = execute_nd_fft_ir(&ir, &input).unwrap();
            assert!(
                max_error(&actual, &expected) < 2.0e-9 * tensor_len as f64,
                "{direction:?} {domain:?}"
            );
        }
    }

    #[test]
    fn multidimensional_frequency_zero_padding_matches_manual_frequency_tensor() {
        let dimensions = vec![3usize, 4];
        let tensor_len = 12usize;
        let build = |direction| {
            let config = FftConfig::new(dimensions.clone())
                .with_inverse_normalization(direction == Direction::Inverse)
                .with_zero_padding(0, 2, 3)
                .unwrap()
                .with_zero_padding(1, 3, 4)
                .unwrap()
                .with_zero_padding_domain(ZeroPaddingDomain::Frequency);
            let plan = FftPlan::build(config).unwrap();
            NdFftIr::build(&plan, direction, device()).unwrap()
        };

        let forward = build(Direction::Forward);
        assert!(
            forward
                .zero_pad_pass
                .as_ref()
                .is_some_and(|pass| pass.operation.is_output_boundary())
        );
        let input = sample(tensor_len, 1);
        let actual = execute_nd_fft_ir(&forward, &input).unwrap();
        let mut expected = dft_2d(&input, 3, 4, 1, Direction::Forward, false);
        for row in 0..3 {
            for col in 0..4 {
                if row == 2 || col == 3 {
                    expected[row * 4 + col] = Complex64::default();
                }
            }
        }
        assert!(max_error(&actual, &expected) < 2.0e-9 * tensor_len as f64);

        let inverse = build(Direction::Inverse);
        assert!(
            inverse
                .zero_pad_pass
                .as_ref()
                .is_some_and(|pass| pass.operation.is_input_boundary())
        );
        let spectrum = dft_2d(&sample(tensor_len, 1), 3, 4, 1, Direction::Forward, false);
        let actual = execute_nd_fft_ir(&inverse, &spectrum).unwrap();
        let mut manual_spectrum = spectrum.clone();
        for row in 0..3 {
            for col in 0..4 {
                if row == 2 || col == 3 {
                    manual_spectrum[row * 4 + col] = Complex64::default();
                }
            }
        }
        let expected = dft_2d(&manual_spectrum, 3, 4, 1, Direction::Inverse, true);
        assert!(max_error(&actual, &expected) < 2.0e-9 * tensor_len as f64);
    }

    #[test]
    fn two_dimensional_bluestein_axis_matches_direct_dft() {
        let dimensions = vec![3usize, 103];
        let tensor_len = dimensions.iter().product::<usize>();
        let mut bluestein_tuning = crate::PlannerTuning::portable();
        bluestein_tuning.max_rader_fft_prime = 100;
        for direction in [Direction::Forward, Direction::Inverse] {
            let config = FftConfig::new(dimensions.clone())
                .with_inverse_normalization(direction == Direction::Inverse)
                .with_tuning(bluestein_tuning);
            let plan = FftPlan::build(config).unwrap();
            let ir = NdFftIr::build(&plan, direction, device()).unwrap();
            assert!(matches!(
                ir.axes[0].transform,
                crate::OneDimFftIr::Bluestein(_)
            ));
            let input = sample(tensor_len, 1);
            let actual = execute_nd_fft_ir(&ir, &input).unwrap();
            let expected = dft_2d(
                &input,
                dimensions[0],
                dimensions[1],
                1,
                direction,
                direction == Direction::Inverse,
            );
            assert!(max_error(&actual, &expected) < 5.0e-8 * tensor_len as f64);
        }
    }
}
