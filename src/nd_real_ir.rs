//! Multidimensional real/complex FFT composition.
//!
//! The last row-major axis uses the compact 1D R2C/C2R path. Remaining axes are
//! ordinary complex FFTs over the compact tensor shape, packed to contiguous
//! batches with the same `NdPassIr` used by multidimensional C2C execution.

use crate::complex::Complex64;
use crate::config::{DeviceProfile, Direction, FftConfig, Precision, TransformKind};
use crate::error::{Result, VkFftError};
use crate::kernel_ir::ScalarType;
use crate::nd_ir::{
    NdAxisIr, NdExternalTensorLayout, NdFormattedCopyOperation, NdFormattedCopyPassIr, NdPassIr,
    NdPassOperation, pack_logical_tensor_batches, unpack_logical_tensor_batches,
};
use crate::one_dim_ir::{OneDimFftIr, execute_one_dim_fft_ir};
use crate::planner::{C2cDeviceAxisClass, FftPlan};
use crate::real_ir::{RealFftIr, RealFftKind, RealFftShapePolicy, execute_c2r_ir, execute_r2c_ir};
use crate::zero_pad_ir::{NdZeroPadPassIr, execute_nd_zero_pad_pass};

#[derive(Debug, Clone, PartialEq)]
pub struct NdRealFftIr {
    pub kind: RealFftKind,
    pub dimensions: Vec<usize>,
    pub compact_dimensions: Vec<usize>,
    pub full_tensor_len: usize,
    pub compact_tensor_len: usize,
    pub batch_count: usize,
    pub real_grouped_batch: Option<usize>,
    pub scalar: ScalarType,
    pub external_scalar: ScalarType,
    pub input_boundary_compute_storage: bool,
    pub output_boundary_compute_storage: bool,
    pub input_external_layout: NdExternalTensorLayout,
    pub output_external_layout: NdExternalTensorLayout,
    pub input_formatted_copy: Option<NdFormattedCopyPassIr>,
    pub output_formatted_copy: Option<NdFormattedCopyPassIr>,
    pub real_axis: RealFftIr,
    pub omitted_axes: Vec<bool>,
    pub complex_axes: Vec<NdAxisIr>,
    pub zero_pad_pass: Option<Box<NdZeroPadPassIr>>,
}

impl NdRealFftIr {
    pub fn build(plan: &FftPlan, device: DeviceProfile) -> Result<Self> {
        Self::build_with_real_shape_policy(
            plan,
            device,
            RealFftShapePolicy::PortableHalfSizeAllEven,
        )
    }

    pub(crate) fn build_for_device_plan(plan: &FftPlan, device: DeviceProfile) -> Result<Self> {
        Self::build_with_real_shape_policy(plan, device, RealFftShapePolicy::FixedUpstreamDevice)
    }

    fn build_with_real_shape_policy(
        plan: &FftPlan,
        device: DeviceProfile,
        real_shape_policy: RealFftShapePolicy,
    ) -> Result<Self> {
        if plan.config.dimensions.len() < 2 {
            return Err(VkFftError::UnsupportedKernelPath(
                "multidimensional real FFT IR requires at least two dimensions",
            ));
        }
        let kind = match plan.config.transform {
            TransformKind::RealToComplex => RealFftKind::RealToComplex,
            TransformKind::ComplexToReal => RealFftKind::ComplexToReal,
            _ => {
                return Err(VkFftError::UnsupportedKernelPath(
                    "multidimensional real FFT IR requires R2C or C2R",
                ));
            }
        };
        let direction = match kind {
            RealFftKind::RealToComplex => Direction::Forward,
            RealFftKind::ComplexToReal => Direction::Inverse,
        };
        let (scalar, external_scalar) = match plan.config.precision {
            Precision::F16StorageF32Compute => (ScalarType::F32, ScalarType::F16),
            Precision::F32 => (ScalarType::F32, ScalarType::F32),
            Precision::F64 if device.supports_f64 => (ScalarType::F64, ScalarType::F64),
            Precision::F64ComputeF32Storage if device.supports_f64 => {
                (ScalarType::F64, ScalarType::F32)
            }
            other => {
                return Err(VkFftError::UnsupportedPrecision {
                    backend: "multidimensional real FFT IR",
                    precision: match other {
                        Precision::F64 => "f64",
                        Precision::DoubleDouble => "double-double",
                        Precision::DoubleDoubleF64Storage => "double-double/f64-storage",
                        _ => "unsupported mixed precision",
                    },
                });
            }
        };
        let full_tensor_len = checked_product(
            &plan.config.dimensions,
            "multidimensional real FFT full tensor size",
        )?;
        let execution_batch_count = plan.config.kernel_preparation_system_count()?;
        let last_axis = plan.config.dimensions.len() - 1;
        let omitted_axes = (0..plan.config.dimensions.len())
            .map(|axis| plan.config.axis_is_omitted(axis))
            .collect::<Vec<_>>();
        if omitted_axes[last_axis] {
            return Err(VkFftError::UnsupportedKernelPath(
                "multidimensional real FFT cannot omit the contiguous real axis",
            ));
        }
        let real_len = plan.config.dimensions[last_axis];
        let outer_lines = full_tensor_len / real_len;
        let real_grouped_batch_override = plan.config.grouped_batch_for_axis(last_axis);
        let real_batches = execution_batch_count.checked_mul(outer_lines).ok_or(
            VkFftError::ArithmeticOverflow {
                operation: "multidimensional real FFT last-axis batch count",
            },
        )?;
        let mut real_config = FftConfig::new(vec![real_len])
            .with_batch_count(real_batches)
            .with_precision(plan.config.precision)
            .with_transform(plan.config.transform)
            .with_kernel_convolution(plan.config.kernel_convolution)
            .with_inverse_normalization(plan.config.normalize_inverse)
            .with_tuning(plan.config.tuning)
            .with_bandwidth_boost(plan.config.bandwidth_boost);
        if let Some(grouped_batch) = real_grouped_batch_override {
            real_config = real_config.with_grouped_batch(0, grouped_batch)?;
        }
        let real_plan = FftPlan::build(real_config)?;
        let has_spatial_zero_padding = plan.config.zero_padding.iter().any(Option::is_some);
        let mut real_axis =
            RealFftIr::build_with_shape_policy(&real_plan, device, real_shape_policy)?;

        let mut compact_dimensions = plan.config.dimensions.clone();
        compact_dimensions[last_axis] = real_axis.half_spectrum_len;
        let compact_tensor_len = checked_product(
            &compact_dimensions,
            "multidimensional real FFT compact tensor size",
        )?;
        let (input_dimensions, output_dimensions) = match kind {
            RealFftKind::RealToComplex => (&plan.config.dimensions, &compact_dimensions),
            RealFftKind::ComplexToReal => (&compact_dimensions, &plan.config.dimensions),
        };
        let input_external_layout = NdExternalTensorLayout {
            dimensions: input_dimensions.clone(),
            axis_strides: plan
                .config
                .resolved_input_buffer_axis_strides_for(input_dimensions)?,
            batch_stride: plan
                .config
                .resolved_input_buffer_batch_stride_for(input_dimensions)?,
        };
        let output_external_layout = NdExternalTensorLayout {
            dimensions: output_dimensions.clone(),
            axis_strides: plan
                .config
                .resolved_output_buffer_axis_strides_for(output_dimensions)?,
            batch_stride: plan
                .config
                .resolved_output_buffer_batch_stride_for(output_dimensions)?,
        };
        let input_formatted_copy = (!input_external_layout.is_tightly_packed()?)
            .then(|| {
                NdFormattedCopyPassIr::new(
                    "vkfft_nd_real_gather_formatted_input".to_owned(),
                    scalar,
                    external_scalar,
                    execution_batch_count,
                    input_external_layout.clone(),
                    NdFormattedCopyOperation::GatherExternalToDense,
                    device,
                )
            })
            .transpose()?;
        let output_formatted_copy = (!output_external_layout.is_tightly_packed()?)
            .then(|| {
                NdFormattedCopyPassIr::new(
                    "vkfft_nd_real_scatter_formatted_output".to_owned(),
                    scalar,
                    external_scalar,
                    execution_batch_count,
                    output_external_layout.clone(),
                    NdFormattedCopyOperation::ScatterDenseToExternal,
                    device,
                )
            })
            .transpose()?;
        let mut complex_axes = Vec::with_capacity(last_axis);
        let upstream_axis1_grouped_batch = plan
            .config
            .grouped_batch_for_axis(last_axis.saturating_sub(1));
        for axis in (0..last_axis).rev() {
            if omitted_axes[axis] {
                continue;
            }
            let axis_len = compact_dimensions[axis];
            let inner_stride = checked_product(
                &compact_dimensions[axis + 1..],
                "multidimensional real FFT compact inner stride",
            )?;
            let line_count = compact_tensor_len / axis_len;
            let grouped_batch_override = plan.config.grouped_batch_for_axis(axis);
            let transform_batches = execution_batch_count.checked_mul(line_count).ok_or(
                VkFftError::ArithmeticOverflow {
                    operation: "multidimensional real FFT complex-axis batch count",
                },
            )?;
            let mut axis_config = FftConfig::new(vec![axis_len])
                .with_batch_count(transform_batches)
                .with_precision(plan.config.precision)
                .with_kernel_convolution(plan.config.kernel_convolution)
                .with_inverse_normalization(
                    direction == Direction::Inverse && plan.config.normalize_inverse,
                )
                .with_tuning(plan.config.tuning)
                .with_bandwidth_boost(plan.config.bandwidth_boost);
            if let Some(grouped_batch) = grouped_batch_override {
                axis_config = axis_config.with_grouped_batch(0, grouped_batch)?;
            }
            let axis_plan = FftPlan::build_c2c_child_for_device(
                axis_config,
                device,
                C2cDeviceAxisClass::Strided,
            )?;
            let transform = if external_scalar != scalar {
                OneDimFftIr::build_internal_compute_storage(&axis_plan, direction, device)?
            } else {
                OneDimFftIr::build(&axis_plan, direction, device)?
            };
            let transform = if grouped_batch_override.is_some() {
                transform.with_other_axis_single_upload_block_with_grouped_batch(
                    compact_dimensions[last_axis],
                    grouped_batch_override,
                    upstream_axis1_grouped_batch,
                    device,
                )?
            } else if has_spatial_zero_padding {
                transform
            } else {
                transform
                    .with_other_axis_single_upload_block(compact_dimensions[last_axis], device)?
            };
            let label = match direction {
                Direction::Forward => "forward",
                Direction::Inverse => "inverse",
            };
            let mut pack = NdPassIr::for_axis(
                format!("vkfft_nd_real_pack_axis_{axis}_{label}"),
                transform.scalar(),
                direction,
                compact_tensor_len,
                axis,
                axis_len,
                inner_stride,
                line_count,
                execution_batch_count,
                NdPassOperation::PackAxis,
                device,
            )?;
            let mut scatter = NdPassIr::for_axis(
                format!("vkfft_nd_real_scatter_axis_{axis}_{label}"),
                transform.scalar(),
                direction,
                compact_tensor_len,
                axis,
                axis_len,
                inner_stride,
                line_count,
                execution_batch_count,
                NdPassOperation::ScatterAxis,
                device,
            )?;
            if let Some(grouped_batch) = grouped_batch_override {
                pack = pack.with_grouped_batch(grouped_batch)?;
                scatter = scatter.with_grouped_batch(grouped_batch)?;
            }
            complex_axes.push(NdAxisIr {
                axis,
                pack,
                transform,
                scatter,
            });
        }
        let zero_pad_pass = has_spatial_zero_padding
            .then(|| {
                let mut pass = NdZeroPadPassIr::build(
                    &plan.config.dimensions,
                    execution_batch_count,
                    plan.config.precision,
                    direction,
                    &plan.config.zero_padding,
                    device,
                )?;
                let formatted_boundary_owned_by_copy = match kind {
                    RealFftKind::RealToComplex => input_formatted_copy.is_some(),
                    RealFftKind::ComplexToReal => output_formatted_copy.is_some(),
                };
                if formatted_boundary_owned_by_copy {
                    pass = pass.with_compute_storage_boundary()?;
                }
                if let Some(grouped_batch) = real_grouped_batch_override {
                    pass.with_grouped_batch(grouped_batch)
                } else {
                    Ok(pass)
                }
            })
            .transpose()?
            .map(Box::new);
        if external_scalar != scalar {
            match kind {
                RealFftKind::RealToComplex => {
                    if complex_axes.is_empty() {
                        if zero_pad_pass.is_some() || input_formatted_copy.is_some() {
                            real_axis = real_axis.with_input_storage_scalar(scalar)?;
                        }
                        if output_formatted_copy.is_some() {
                            real_axis = real_axis.with_output_storage_scalar(scalar)?;
                        }
                    } else {
                        real_axis = real_axis.with_output_storage_scalar(scalar)?;
                        if zero_pad_pass.is_some() || input_formatted_copy.is_some() {
                            real_axis = real_axis.with_input_storage_scalar(scalar)?;
                        }
                        if output_formatted_copy.is_none() {
                            let last_index = complex_axes.len() - 1;
                            complex_axes[last_index].scatter = complex_axes[last_index]
                                .scatter
                                .clone()
                                .with_external_output_storage(external_scalar)?;
                        }
                    }
                }
                RealFftKind::ComplexToReal => {
                    if complex_axes.is_empty() {
                        if input_formatted_copy.is_some() {
                            real_axis = real_axis.with_input_storage_scalar(scalar)?;
                        }
                        if zero_pad_pass.is_some() || output_formatted_copy.is_some() {
                            real_axis = real_axis.with_output_storage_scalar(scalar)?;
                        }
                    } else {
                        real_axis = real_axis.with_input_storage_scalar(scalar)?;
                        if zero_pad_pass.is_some() || output_formatted_copy.is_some() {
                            real_axis = real_axis.with_output_storage_scalar(scalar)?;
                        }
                        if input_formatted_copy.is_none() {
                            complex_axes[0].pack = complex_axes[0]
                                .pack
                                .clone()
                                .with_external_input_storage(external_scalar)?;
                        }
                    }
                }
            }
        }
        let ir = Self {
            kind,
            dimensions: plan.config.dimensions.clone(),
            compact_dimensions,
            full_tensor_len,
            compact_tensor_len,
            batch_count: execution_batch_count,
            real_grouped_batch: real_grouped_batch_override,
            scalar,
            external_scalar,
            input_boundary_compute_storage: false,
            output_boundary_compute_storage: false,
            input_external_layout,
            output_external_layout,
            input_formatted_copy,
            output_formatted_copy,
            real_axis,
            omitted_axes,
            complex_axes,
            zero_pad_pass,
        };
        ir.validate()?;
        Ok(ir)
    }

    pub(crate) fn with_internal_output_compute_storage(mut self) -> Result<Self> {
        if self.kind != RealFftKind::RealToComplex || self.output_formatted_copy.is_some() {
            return Err(VkFftError::InvalidKernelIr(
                "multidimensional real internal output storage requires an unformatted R2C boundary",
            ));
        }
        if self.external_scalar != self.scalar {
            if self.complex_axes.is_empty() {
                self.real_axis = self.real_axis.with_output_storage_scalar(self.scalar)?;
            } else {
                let last = self.complex_axes.len() - 1;
                self.complex_axes[last].scatter = self.complex_axes[last]
                    .scatter
                    .clone()
                    .with_external_output_storage(self.scalar)?;
            }
            self.output_boundary_compute_storage = true;
        }
        self.validate()?;
        Ok(self)
    }

    pub(crate) fn with_internal_input_compute_storage(mut self) -> Result<Self> {
        if self.kind != RealFftKind::ComplexToReal || self.input_formatted_copy.is_some() {
            return Err(VkFftError::InvalidKernelIr(
                "multidimensional real internal input storage requires an unformatted C2R boundary",
            ));
        }
        if self.external_scalar != self.scalar {
            if self.complex_axes.is_empty() {
                self.real_axis = self.real_axis.with_input_storage_scalar(self.scalar)?;
            } else {
                self.complex_axes[0].pack = self.complex_axes[0]
                    .pack
                    .clone()
                    .with_external_input_storage(self.scalar)?;
            }
            self.input_boundary_compute_storage = true;
        }
        self.validate()?;
        Ok(self)
    }

    pub fn validate(&self) -> Result<()> {
        if self.dimensions.len() < 2
            || self.compact_dimensions.len() != self.dimensions.len()
            || self.omitted_axes.len() != self.dimensions.len()
            || self.batch_count == 0
            || self.real_grouped_batch == Some(0)
            || checked_product(
                &self.dimensions,
                "multidimensional real validation full size",
            )? != self.full_tensor_len
            || checked_product(
                &self.compact_dimensions,
                "multidimensional real validation compact size",
            )? != self.compact_tensor_len
        {
            return Err(VkFftError::InvalidKernelIr(
                "multidimensional real FFT tensor metadata is inconsistent",
            ));
        }
        if self.external_scalar != self.scalar
            && !matches!(
                (self.scalar, self.external_scalar),
                (ScalarType::F32, ScalarType::F16) | (ScalarType::F64, ScalarType::F32)
            )
        {
            return Err(VkFftError::InvalidKernelIr(
                "multidimensional real external storage scalar is incompatible with compute precision",
            ));
        }
        if (self.input_boundary_compute_storage
            && (self.kind != RealFftKind::ComplexToReal
                || self.external_scalar == self.scalar
                || self.input_formatted_copy.is_some()))
            || (self.output_boundary_compute_storage
                && (self.kind != RealFftKind::RealToComplex
                    || self.external_scalar == self.scalar
                    || self.output_formatted_copy.is_some()))
        {
            return Err(VkFftError::InvalidKernelIr(
                "multidimensional real internal compute-storage boundary is inconsistent",
            ));
        }
        self.real_axis.validate()?;
        let (input_dimensions, input_tensor_len, output_dimensions, output_tensor_len) =
            match self.kind {
                RealFftKind::RealToComplex => (
                    &self.dimensions,
                    self.full_tensor_len,
                    &self.compact_dimensions,
                    self.compact_tensor_len,
                ),
                RealFftKind::ComplexToReal => (
                    &self.compact_dimensions,
                    self.compact_tensor_len,
                    &self.dimensions,
                    self.full_tensor_len,
                ),
            };
        self.input_external_layout.validate(input_tensor_len)?;
        self.output_external_layout.validate(output_tensor_len)?;
        if &self.input_external_layout.dimensions != input_dimensions
            || &self.output_external_layout.dimensions != output_dimensions
        {
            return Err(VkFftError::InvalidKernelIr(
                "multidimensional real formatted tensor dimensions are inconsistent",
            ));
        }
        let input_needs_copy = !self.input_external_layout.is_tightly_packed()?;
        let output_needs_copy = !self.output_external_layout.is_tightly_packed()?;
        if self.input_formatted_copy.is_some() != input_needs_copy
            || self.output_formatted_copy.is_some() != output_needs_copy
        {
            return Err(VkFftError::InvalidKernelIr(
                "multidimensional real formatted copy ownership is inconsistent",
            ));
        }
        if let Some(copy) = &self.input_formatted_copy {
            copy.validate()?;
            if copy.operation != NdFormattedCopyOperation::GatherExternalToDense
                || copy.external_layout != self.input_external_layout
                || copy.batch_count != self.batch_count
                || copy.scalar != self.scalar
                || copy.input_storage_scalar != self.external_scalar
                || copy.output_storage_scalar != self.scalar
            {
                return Err(VkFftError::InvalidKernelIr(
                    "multidimensional real formatted input copy metadata is inconsistent",
                ));
            }
        }
        if let Some(copy) = &self.output_formatted_copy {
            copy.validate()?;
            if copy.operation != NdFormattedCopyOperation::ScatterDenseToExternal
                || copy.external_layout != self.output_external_layout
                || copy.batch_count != self.batch_count
                || copy.scalar != self.scalar
                || copy.input_storage_scalar != self.scalar
                || copy.output_storage_scalar != self.external_scalar
            {
                return Err(VkFftError::InvalidKernelIr(
                    "multidimensional real formatted output copy metadata is inconsistent",
                ));
            }
        }
        if let Some(pass) = &self.zero_pad_pass {
            pass.validate()?;
            let (expected_operation, expected_storage) = match self.kind {
                RealFftKind::RealToComplex => (
                    crate::ZeroPadPassOperation::PrepareForwardInput,
                    (
                        if self.input_formatted_copy.is_some() {
                            self.scalar
                        } else {
                            self.external_scalar
                        },
                        self.scalar,
                    ),
                ),
                RealFftKind::ComplexToReal => (
                    crate::ZeroPadPassOperation::FinalizeInverseOutput,
                    (
                        self.scalar,
                        if self.output_formatted_copy.is_some() {
                            self.scalar
                        } else {
                            self.external_scalar
                        },
                    ),
                ),
            };
            if pass.dimensions != self.dimensions
                || pass.tensor_len != self.full_tensor_len
                || pass.batch_count != self.batch_count
                || pass.grouped_batch != self.real_grouped_batch.unwrap_or(1)
                || pass.scalar != self.scalar
                || pass.operation != expected_operation
                || (pass.input_storage_scalar, pass.output_storage_scalar) != expected_storage
            {
                return Err(VkFftError::InvalidKernelIr(
                    "multidimensional real zero-padding metadata is inconsistent",
                ));
            }
        }
        let last = self.dimensions.len() - 1;
        if self.omitted_axes[last] {
            return Err(VkFftError::InvalidKernelIr(
                "multidimensional real FFT omitted the contiguous real axis",
            ));
        }
        let expected_real_storage = match self.kind {
            RealFftKind::RealToComplex => (
                if self.zero_pad_pass.is_some() || self.input_formatted_copy.is_some() {
                    self.scalar
                } else {
                    self.external_scalar
                },
                if self.complex_axes.is_empty()
                    && self.output_formatted_copy.is_none()
                    && !self.output_boundary_compute_storage
                {
                    self.external_scalar
                } else {
                    self.scalar
                },
            ),
            RealFftKind::ComplexToReal => (
                if self.complex_axes.is_empty()
                    && self.input_formatted_copy.is_none()
                    && !self.input_boundary_compute_storage
                {
                    self.external_scalar
                } else {
                    self.scalar
                },
                if self.zero_pad_pass.is_some() || self.output_formatted_copy.is_some() {
                    self.scalar
                } else {
                    self.external_scalar
                },
            ),
        };
        let real_outer_lines = self.full_tensor_len / self.real_axis.length;
        let expected_real_grouped_batch = self.real_grouped_batch.unwrap_or(1);
        if self.real_axis.length != self.dimensions[last]
            || self.real_axis.half_spectrum_len != self.compact_dimensions[last]
            || self.real_axis.kind != self.kind
            || self.real_axis.scalar != self.scalar
            || self.real_axis.input_storage_scalar() != expected_real_storage.0
            || self.real_axis.output_storage_scalar() != expected_real_storage.1
            || self.real_axis.batch_count != self.batch_count * real_outer_lines
            || self.real_axis.grouped_batch != expected_real_grouped_batch
            || self.complex_axes.len() != (0..last).filter(|&axis| !self.omitted_axes[axis]).count()
        {
            return Err(VkFftError::InvalidKernelIr(
                "multidimensional real FFT last-axis metadata is inconsistent",
            ));
        }
        let direction = match self.kind {
            RealFftKind::RealToComplex => Direction::Forward,
            RealFftKind::ComplexToReal => Direction::Inverse,
        };
        let expected_axis_order = (0..last)
            .rev()
            .filter(|&axis| !self.omitted_axes[axis])
            .collect::<Vec<_>>();
        if self
            .complex_axes
            .iter()
            .map(|axis| axis.axis)
            .ne(expected_axis_order.iter().copied())
        {
            return Err(VkFftError::InvalidKernelIr(
                "multidimensional real omitted-axis execution order is inconsistent",
            ));
        }
        for (axis_index, axis) in self.complex_axes.iter().enumerate() {
            axis.pack.validate()?;
            axis.transform.validate()?;
            axis.scatter.validate()?;
            if self.omitted_axes[axis.axis] {
                return Err(VkFftError::InvalidKernelIr(
                    "multidimensional real FFT materialized an omitted axis",
                ));
            }
            let axis_len = self.compact_dimensions[axis.axis];
            let expected_batches = self
                .batch_count
                .checked_mul(self.compact_tensor_len / axis_len)
                .ok_or(VkFftError::ArithmeticOverflow {
                    operation: "multidimensional real validation complex batch count",
                })?;
            let expected_pack_input = if self.kind == RealFftKind::ComplexToReal
                && axis_index == 0
                && self.input_formatted_copy.is_none()
                && !self.input_boundary_compute_storage
            {
                self.external_scalar
            } else {
                self.scalar
            };
            let expected_scatter_output = if self.kind == RealFftKind::RealToComplex
                && axis_index + 1 == self.complex_axes.len()
                && self.output_formatted_copy.is_none()
                && !self.output_boundary_compute_storage
            {
                self.external_scalar
            } else {
                self.scalar
            };
            if axis.transform.logical_len() != axis_len
                || axis.transform.batch_count() != expected_batches
                || axis.transform.direction() != direction
                || axis.transform.scalar() != self.scalar
                || axis.transform.external_storage_scalar() != self.scalar
                || axis.pack.grouped_batch != axis.scatter.grouped_batch
                || axis
                    .pack
                    .grouped_batch
                    .is_some_and(|grouped_batch| axis.transform.grouped_batch() != grouped_batch)
                || axis.pack.tensor_len != self.compact_tensor_len
                || axis.scatter.tensor_len != self.compact_tensor_len
                || axis.pack.input_storage_scalar != expected_pack_input
                || axis.pack.output_storage_scalar != self.scalar
                || axis.scatter.input_storage_scalar != self.scalar
                || axis.scatter.output_storage_scalar != expected_scatter_output
            {
                return Err(VkFftError::InvalidKernelIr(
                    "multidimensional real FFT complex-axis metadata is inconsistent",
                ));
            }
        }
        Ok(())
    }

    pub(crate) fn input_tensor_len(&self) -> usize {
        match self.kind {
            RealFftKind::RealToComplex => self.full_tensor_len,
            RealFftKind::ComplexToReal => self.compact_tensor_len,
        }
    }

    pub(crate) fn output_tensor_len(&self) -> usize {
        match self.kind {
            RealFftKind::RealToComplex => self.compact_tensor_len,
            RealFftKind::ComplexToReal => self.full_tensor_len,
        }
    }

    pub(crate) fn pack_formatted_input<T: Copy + Default>(&self, input: &[T]) -> Result<Vec<T>> {
        self.validate()?;
        let expected = self
            .input_tensor_len()
            .checked_mul(self.batch_count)
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "formatted ND real logical input element count",
            })?;
        if input.len() != expected {
            return Err(VkFftError::InputLengthMismatch {
                expected,
                actual: input.len(),
            });
        }
        if self.input_formatted_copy.is_none() {
            return Ok(input.to_vec());
        }
        pack_logical_tensor_batches(
            input,
            &self.input_external_layout.dimensions,
            &self.input_external_layout.axis_strides,
            self.input_external_layout.batch_stride,
        )
    }

    pub(crate) fn unpack_formatted_output<T: Copy>(&self, output: &[T]) -> Result<Vec<T>> {
        self.validate()?;
        if self.output_formatted_copy.is_none() {
            return Ok(output.to_vec());
        }
        unpack_logical_tensor_batches(
            output,
            &self.output_external_layout.dimensions,
            &self.output_external_layout.axis_strides,
            self.output_external_layout.batch_stride,
            self.batch_count,
        )
    }
}

pub fn execute_nd_r2c_ir(ir: &NdRealFftIr, input: &[f64]) -> Result<Vec<Complex64>> {
    ir.validate()?;
    if ir.kind != RealFftKind::RealToComplex {
        return Err(VkFftError::UnsupportedKernelPath(
            "multidimensional R2C executor requires an R2C IR",
        ));
    }
    let expected =
        ir.full_tensor_len
            .checked_mul(ir.batch_count)
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "multidimensional R2C input element count",
            })?;
    if input.len() != expected {
        return Err(VkFftError::InputLengthMismatch {
            expected,
            actual: input.len(),
        });
    }
    let logical_input = if ir.input_formatted_copy.is_some() {
        let physical = ir.pack_formatted_input(input)?;
        unpack_logical_tensor_batches(
            &physical,
            &ir.input_external_layout.dimensions,
            &ir.input_external_layout.axis_strides,
            ir.input_external_layout.batch_stride,
            ir.batch_count,
        )?
    } else {
        input.to_vec()
    };
    let padded_storage = if let Some(pass) = &ir.zero_pad_pass {
        let complex = logical_input
            .iter()
            .map(|value| Complex64::new(*value, 0.0))
            .collect::<Vec<_>>();
        Some(
            execute_nd_zero_pad_pass(pass, &complex)?
                .into_iter()
                .map(|value| value.re)
                .collect::<Vec<_>>(),
        )
    } else {
        None
    };
    let input = padded_storage.as_deref().unwrap_or(&logical_input);
    let current = execute_r2c_ir(&ir.real_axis, input)?;
    let logical_output = execute_complex_axes(ir, current)?;
    if ir.output_formatted_copy.is_some() {
        let physical = pack_logical_tensor_batches(
            &logical_output,
            &ir.output_external_layout.dimensions,
            &ir.output_external_layout.axis_strides,
            ir.output_external_layout.batch_stride,
        )?;
        ir.unpack_formatted_output(&physical)
    } else {
        Ok(logical_output)
    }
}

pub fn execute_nd_c2r_ir(ir: &NdRealFftIr, input: &[Complex64]) -> Result<Vec<f64>> {
    ir.validate()?;
    if ir.kind != RealFftKind::ComplexToReal {
        return Err(VkFftError::UnsupportedKernelPath(
            "multidimensional C2R executor requires a C2R IR",
        ));
    }
    let expected = ir.compact_tensor_len.checked_mul(ir.batch_count).ok_or(
        VkFftError::ArithmeticOverflow {
            operation: "multidimensional C2R input element count",
        },
    )?;
    if input.len() != expected {
        return Err(VkFftError::InputLengthMismatch {
            expected,
            actual: input.len(),
        });
    }
    let logical_input = if ir.input_formatted_copy.is_some() {
        let physical = ir.pack_formatted_input(input)?;
        unpack_logical_tensor_batches(
            &physical,
            &ir.input_external_layout.dimensions,
            &ir.input_external_layout.axis_strides,
            ir.input_external_layout.batch_stride,
            ir.batch_count,
        )?
    } else {
        input.to_vec()
    };
    let current = execute_complex_axes(ir, logical_input)?;
    let output = execute_c2r_ir(&ir.real_axis, &current)?;
    let logical_output = if let Some(pass) = &ir.zero_pad_pass {
        let complex = output
            .into_iter()
            .map(|value| Complex64::new(value, 0.0))
            .collect::<Vec<_>>();
        execute_nd_zero_pad_pass(pass, &complex)?
            .into_iter()
            .map(|value| value.re)
            .collect::<Vec<_>>()
    } else {
        output
    };
    if ir.output_formatted_copy.is_some() {
        let physical = pack_logical_tensor_batches(
            &logical_output,
            &ir.output_external_layout.dimensions,
            &ir.output_external_layout.axis_strides,
            ir.output_external_layout.batch_stride,
        )?;
        ir.unpack_formatted_output(&physical)
    } else {
        Ok(logical_output)
    }
}

fn execute_complex_axes(ir: &NdRealFftIr, mut current: Vec<Complex64>) -> Result<Vec<Complex64>> {
    let expected = ir.compact_tensor_len.checked_mul(ir.batch_count).ok_or(
        VkFftError::ArithmeticOverflow {
            operation: "multidimensional real compact execution size",
        },
    )?;
    for axis in &ir.complex_axes {
        let pass = &axis.pack;
        let transform_count = pass.transform_count()?;
        let mut packed = vec![Complex64::new(0.0, 0.0); expected];
        for transform in 0..transform_count {
            let batch = transform / pass.line_count;
            let line = transform % pass.line_count;
            let outer = line / pass.inner_stride;
            let inner = line % pass.inner_stride;
            let packed_base = transform * pass.axis_len;
            for axis_i in 0..pass.axis_len {
                let natural = batch * pass.tensor_len
                    + outer * pass.axis_len * pass.inner_stride
                    + axis_i * pass.inner_stride
                    + inner;
                packed[packed_base + axis_i] = current[natural];
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
            for axis_i in 0..pass.axis_len {
                let natural = batch * pass.tensor_len
                    + outer * pass.axis_len * pass.inner_stride
                    + axis_i * pass.inner_stride
                    + inner;
                next[natural] = transformed[packed_base + axis_i];
            }
        }
        current = next;
    }
    Ok(current)
}

fn checked_product(values: &[usize], operation: &'static str) -> Result<usize> {
    values.iter().try_fold(1usize, |product, value| {
        product
            .checked_mul(*value)
            .ok_or(VkFftError::ArithmeticOverflow { operation })
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{Backend, GpuVendor};

    fn device() -> DeviceProfile {
        DeviceProfile {
            shared_memory_bytes: 128 * 1024,
            shared_memory_pow2_bytes: 128 * 1024,
            ..DeviceProfile::generic(Backend::Vulkan, GpuVendor::Nvidia)
        }
    }

    fn stockham_kernel(transform: &OneDimFftIr) -> &crate::KernelIr {
        let OneDimFftIr::Recursive(recursive) = transform else {
            panic!("ND real axis-block test unexpectedly selected Bluestein");
        };
        let crate::recursive_ir::RecursiveFftNodeIr::Stockham(kernel) = &recursive.root else {
            panic!("ND real axis-block test unexpectedly selected Rader");
        };
        kernel
    }

    #[test]
    fn omit_dimension_skips_nd_real_outer_axis_but_keeps_real_boundary() {
        let dimensions = vec![3usize, 8];
        let input = (0..24)
            .map(|index| {
                let x = index as f64;
                (0.13 * x).sin() + 0.01 * x
            })
            .collect::<Vec<_>>();
        let forward_plan = FftPlan::build(
            FftConfig::new(dimensions.clone())
                .with_transform(TransformKind::RealToComplex)
                .with_omit_dimension(0, true)
                .unwrap(),
        )
        .unwrap();
        let forward = NdRealFftIr::build(&forward_plan, device()).unwrap();
        assert_eq!(forward.omitted_axes, vec![true, false]);
        assert!(forward.complex_axes.is_empty());
        let actual = execute_nd_r2c_ir(&forward, &input).unwrap();
        let expected = execute_r2c_ir(&forward.real_axis, &input).unwrap();
        assert_eq!(actual, expected);

        let inverse_plan = FftPlan::build(
            FftConfig::new(dimensions)
                .with_transform(TransformKind::ComplexToReal)
                .with_inverse_normalization(true)
                .with_omit_dimension(0, true)
                .unwrap(),
        )
        .unwrap();
        let inverse = NdRealFftIr::build(&inverse_plan, device()).unwrap();
        assert_eq!(inverse.omitted_axes, vec![true, false]);
        assert!(inverse.complex_axes.is_empty());
        let restored = execute_nd_c2r_ir(&inverse, &actual).unwrap();
        let max_error = restored
            .iter()
            .zip(&input)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0, f64::max);
        assert!(
            max_error <= 2.0e-10,
            "omitted ND-real outer-axis round trip error {max_error:e}"
        );
    }

    #[test]
    fn formatted_real_strides_wrap_full_and_compact_boundaries_without_changing_cpu_math() {
        let dimensions = vec![3usize, 8];
        let input = (0..48)
            .map(|index| {
                let x = index as f64;
                (0.071 * x).sin() + 0.019 * x - (0.043 * x).cos()
            })
            .collect::<Vec<_>>();
        let forward_config = FftConfig::new(dimensions.clone())
            .with_batch_count(2)
            .with_transform(TransformKind::RealToComplex)
            .with_input_buffer_axis_stride(0, 11)
            .unwrap()
            .with_output_buffer_axis_stride(0, 7)
            .unwrap();
        let forward_plan = FftPlan::build(forward_config.clone()).unwrap();
        let forward = NdRealFftIr::build(&forward_plan, device()).unwrap();
        assert_eq!(forward.input_external_layout.dimensions, dimensions);
        assert_eq!(forward.input_external_layout.axis_strides, vec![11, 1]);
        assert_eq!(forward.input_external_layout.batch_stride, 33);
        assert_eq!(forward.output_external_layout.dimensions, vec![3, 5]);
        assert_eq!(forward.output_external_layout.axis_strides, vec![7, 1]);
        assert_eq!(forward.output_external_layout.batch_stride, 21);
        assert_eq!(
            forward.input_formatted_copy.as_ref().unwrap().operation,
            NdFormattedCopyOperation::GatherExternalToDense
        );
        assert_eq!(
            forward.output_formatted_copy.as_ref().unwrap().operation,
            NdFormattedCopyOperation::ScatterDenseToExternal
        );
        let physical_input = forward.pack_formatted_input(&input).unwrap();
        assert_eq!(physical_input.len(), 66);
        assert_eq!(&physical_input[0..8], &input[0..8]);
        assert_eq!(&physical_input[11..19], &input[8..16]);
        assert_eq!(physical_input[8], 0.0);
        assert_eq!(physical_input[32], 0.0);

        let actual = execute_nd_r2c_ir(&forward, &input).unwrap();
        let dense_plan = FftPlan::build(
            FftConfig::new(vec![3, 8])
                .with_batch_count(2)
                .with_transform(TransformKind::RealToComplex),
        )
        .unwrap();
        let dense_forward = NdRealFftIr::build(&dense_plan, device()).unwrap();
        let expected = execute_nd_r2c_ir(&dense_forward, &input).unwrap();
        let max_spectrum_error = actual
            .iter()
            .zip(&expected)
            .map(|(actual, expected)| {
                let delta = *actual - *expected;
                (delta.re * delta.re + delta.im * delta.im).sqrt()
            })
            .fold(0.0, f64::max);
        assert!(
            max_spectrum_error <= 2.0e-10,
            "formatted ND R2C CPU error {max_spectrum_error:e}"
        );

        let inverse_plan = FftPlan::build(
            FftConfig::new(vec![3, 8])
                .with_batch_count(2)
                .with_transform(TransformKind::ComplexToReal)
                .with_inverse_normalization(true)
                .with_input_buffer_axis_stride(0, 7)
                .unwrap()
                .with_output_buffer_axis_stride(0, 11)
                .unwrap(),
        )
        .unwrap();
        let inverse = NdRealFftIr::build(&inverse_plan, device()).unwrap();
        assert_eq!(inverse.input_external_layout.dimensions, vec![3, 5]);
        assert_eq!(inverse.input_external_layout.axis_strides, vec![7, 1]);
        assert_eq!(inverse.input_external_layout.batch_stride, 21);
        assert_eq!(inverse.output_external_layout.dimensions, vec![3, 8]);
        assert_eq!(inverse.output_external_layout.axis_strides, vec![11, 1]);
        assert_eq!(inverse.output_external_layout.batch_stride, 33);
        let restored = execute_nd_c2r_ir(&inverse, &actual).unwrap();
        let max_round_trip_error = restored
            .iter()
            .zip(&input)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0, f64::max);
        assert!(
            max_round_trip_error <= 3.0e-10,
            "formatted ND real round trip error {max_round_trip_error:e}"
        );
    }

    #[test]
    fn formatted_real_strides_support_three_dimensional_row_and_plane_pitches() {
        let config = FftConfig::new(vec![2, 3, 8])
            .with_batch_count(2)
            .with_transform(TransformKind::RealToComplex)
            .with_input_buffer_axis_stride(1, 11)
            .unwrap()
            .with_input_buffer_axis_stride(0, 40)
            .unwrap()
            .with_output_buffer_axis_stride(1, 7)
            .unwrap()
            .with_output_buffer_axis_stride(0, 25)
            .unwrap();
        let plan = FftPlan::build(config).unwrap();
        let ir = NdRealFftIr::build(&plan, device()).unwrap();
        assert_eq!(ir.input_external_layout.dimensions, vec![2, 3, 8]);
        assert_eq!(ir.input_external_layout.axis_strides, vec![40, 11, 1]);
        assert_eq!(ir.input_external_layout.batch_stride, 80);
        assert_eq!(ir.output_external_layout.dimensions, vec![2, 3, 5]);
        assert_eq!(ir.output_external_layout.axis_strides, vec![25, 7, 1]);
        assert_eq!(ir.output_external_layout.batch_stride, 50);
        assert!(ir.input_formatted_copy.is_some());
        assert!(ir.output_formatted_copy.is_some());
        ir.validate().unwrap();
    }

    #[test]
    fn multidimensional_real_outer_children_keep_strided_device_scoring() {
        let profile = DeviceProfile::generic(Backend::Vulkan, GpuVendor::Nvidia);
        let plan = FftPlan::build(
            FftConfig::new(vec![1153usize, 8]).with_transform(TransformKind::RealToComplex),
        )
        .unwrap();
        let ir = NdRealFftIr::build(&plan, profile).unwrap();
        assert_eq!(ir.complex_axes.len(), 1);
        assert_eq!(ir.complex_axes[0].axis, 0);
        assert!(matches!(
            ir.complex_axes[0].transform,
            OneDimFftIr::Bluestein(_)
        ));

        // The same p1153 sequence is Rader when it is the contiguous half-size
        // child of a one-dimensional real transform; this pair proves that the
        // extracted ND outer axis did not silently become a contiguous 1D child.
        let contiguous_plan = FftPlan::build(
            FftConfig::new(vec![2306usize]).with_transform(TransformKind::RealToComplex),
        )
        .unwrap();
        let contiguous = RealFftIr::build(&contiguous_plan, profile).unwrap();
        assert!(matches!(contiguous.transform, OneDimFftIr::Recursive(_)));
    }

    #[test]
    fn multidimensional_real_bandwidth_boost_reaches_outer_complex_child() {
        let mut profile = device();
        profile.shared_memory_bytes = 48 * 1024;
        profile.shared_memory_pow2_bytes = 32 * 1024;
        profile.max_threads_per_block = 1024;
        profile.max_workgroup_size = [1024, 1024, 64];
        let dimensions = vec![2_097_152usize, 2usize];

        let baseline_plan = FftPlan::build(
            FftConfig::new(dimensions.clone()).with_transform(TransformKind::RealToComplex),
        )
        .unwrap();
        let baseline = NdRealFftIr::build(&baseline_plan, profile).unwrap();
        let OneDimFftIr::Recursive(baseline_recursive) = &baseline.complex_axes[0].transform else {
            panic!("baseline ND real outer axis should remain recursive Stockham");
        };
        assert_eq!(
            baseline_recursive
                .stockham_upload_schedule
                .as_ref()
                .unwrap()
                .axis_split,
            vec![128, 128, 128]
        );

        let boosted_plan = FftPlan::build(
            FftConfig::new(dimensions)
                .with_transform(TransformKind::RealToComplex)
                .with_bandwidth_boost(2),
        )
        .unwrap();
        let boosted = NdRealFftIr::build(&boosted_plan, profile).unwrap();
        let OneDimFftIr::Recursive(boosted_recursive) = &boosted.complex_axes[0].transform else {
            panic!("boosted ND real outer axis should remain recursive Stockham");
        };
        let schedule = boosted_recursive.stockham_upload_schedule.as_ref().unwrap();
        assert_eq!(schedule.upload_count, 2);
        assert_eq!(schedule.axis_split, vec![2_048, 1_024]);
        assert!(boosted_recursive.four_step_plan.is_some());
        boosted.validate().unwrap();
    }

    #[test]
    fn f16_outer_rader_preserves_storage_scheduler_precision_and_physical_blocks() {
        let mut profile = DeviceProfile::generic(Backend::Vulkan, GpuVendor::Nvidia);
        profile.shared_memory_bytes = 32 * 1024;
        profile.shared_memory_pow2_bytes = 32 * 1024;
        profile.max_threads_per_block = 1024;
        profile.max_workgroup_size = [1024, 1024, 64];

        for (bandwidth_boost, expected_split, expected_groups) in [
            (0usize, vec![64usize, 68, 64], vec![8usize, 8, 8]),
            (2usize, vec![544usize, 512], vec![7usize, 8]),
        ] {
            let plan = FftPlan::build(
                FftConfig::new(vec![278_528usize, 14])
                    .with_transform(TransformKind::RealToComplex)
                    .with_precision(Precision::F16StorageF32Compute)
                    .with_bandwidth_boost(bandwidth_boost),
            )
            .unwrap();
            let ir = NdRealFftIr::build(&plan, profile).unwrap();
            assert_eq!(ir.compact_dimensions[1], 8);
            let OneDimFftIr::Recursive(recursive) = &ir.complex_axes[0].transform else {
                panic!("N278528 ND-real outer F16 Rader child should remain recursive");
            };
            let schedule = recursive
                .rader_forced_upload_schedule
                .as_ref()
                .expect("N278528 ND-real outer F16 Rader child should retain upload schedule");
            assert_eq!(schedule.axis_split, expected_split);
            let four_step = recursive
                .four_step_plan
                .as_ref()
                .expect("N278528 ND-real outer F16 Rader child should retain Four-step plan");
            assert_eq!(four_step.uploads.len(), expected_groups.len());
            for (axis_upload_id, expected_grouped) in expected_groups.into_iter().enumerate() {
                let upload = four_step
                    .uploads
                    .iter()
                    .find(|upload| upload.axis_upload_id == axis_upload_id)
                    .unwrap();
                let block = upload
                    .axis_block
                    .expect("missing ND-real higher-axis block");
                assert_eq!(block.grouped_batch, expected_grouped);
                assert_eq!(block.local_size_x, expected_grouped);
                assert_eq!(block.local_size_y, block.threads_per_transform);
                assert!(block.transforms_on_x);
                assert!(!block.axis_swapped);
            }
            ir.validate().unwrap();
        }
    }

    #[test]
    fn multidimensional_real_children_consume_axis_block_geometry() {
        let mut profile = device();
        profile.shared_memory_bytes = 48 * 1024;
        profile.shared_memory_pow2_bytes = 32 * 1024;
        profile.max_threads_per_block = 1024;
        profile.max_workgroup_size = [1024, 1024, 64];
        let plan = FftPlan::build(
            FftConfig::new(vec![3usize, 8]).with_transform(TransformKind::RealToComplex),
        )
        .unwrap();
        let ir = NdRealFftIr::build(&plan, profile).unwrap();

        let outer = &ir.complex_axes[0];
        let outer_kernel = stockham_kernel(&outer.transform);
        assert_eq!(outer.axis, 0);
        assert_eq!(
            outer_kernel.workgroup_grouping.axis_layout,
            crate::kernel_ir::StockhamWorkgroupAxisLayout::TransformsXThreadsY
        );
        assert!(outer_kernel.workgroup_grouping.transforms_per_workgroup > 1);
        assert_eq!(outer_kernel.workgroup_size.y, 1);
    }

    #[test]
    fn formatted_real_strides_compose_with_spatial_zero_padding() {
        let dimensions = vec![3usize, 8];
        let full_len = dimensions.iter().product::<usize>();
        let batch_count = 2usize;
        let input = (0..full_len * batch_count)
            .map(|index| {
                let x = index as f64;
                (0.071 * x).sin() + 0.019 * x - (0.043 * x).cos()
            })
            .collect::<Vec<_>>();
        let mut masked_input = input.clone();
        for batch in 0..batch_count {
            let base = batch * full_len;
            for row in 0..dimensions[0] {
                for col in 0..dimensions[1] {
                    if row == 1 || (2..4).contains(&col) {
                        masked_input[base + row * dimensions[1] + col] = 0.0;
                    }
                }
            }
        }

        let forward_plan = FftPlan::build(
            FftConfig::new(dimensions.clone())
                .with_batch_count(batch_count)
                .with_transform(TransformKind::RealToComplex)
                .with_input_buffer_axis_stride(0, 11)
                .unwrap()
                .with_output_buffer_axis_stride(0, 7)
                .unwrap()
                .with_zero_padding(0, 1, 2)
                .unwrap()
                .with_zero_padding(1, 2, 4)
                .unwrap(),
        )
        .unwrap();
        let forward = NdRealFftIr::build(&forward_plan, device()).unwrap();
        assert!(forward.input_formatted_copy.is_some());
        assert!(forward.output_formatted_copy.is_some());
        let forward_pad = forward.zero_pad_pass.as_ref().unwrap();
        assert_eq!(
            forward_pad.operation,
            crate::ZeroPadPassOperation::PrepareForwardInput
        );
        assert_eq!(forward_pad.input_storage_scalar, forward.scalar);
        assert_eq!(forward_pad.output_storage_scalar, forward.scalar);
        let actual_spectrum = execute_nd_r2c_ir(&forward, &input).unwrap();

        let dense_forward_plan = FftPlan::build(
            FftConfig::new(dimensions.clone())
                .with_batch_count(batch_count)
                .with_transform(TransformKind::RealToComplex),
        )
        .unwrap();
        let dense_forward = NdRealFftIr::build(&dense_forward_plan, device()).unwrap();
        let expected_spectrum = execute_nd_r2c_ir(&dense_forward, &masked_input).unwrap();
        let max_forward_error = actual_spectrum
            .iter()
            .zip(&expected_spectrum)
            .map(|(actual, expected)| (*actual - *expected).norm_sqr().sqrt())
            .fold(0.0, f64::max);
        assert!(
            max_forward_error <= 3.0e-10 * full_len as f64,
            "formatted+padding ND R2C error {max_forward_error:e}"
        );

        let inverse_plan = FftPlan::build(
            FftConfig::new(dimensions.clone())
                .with_batch_count(batch_count)
                .with_transform(TransformKind::ComplexToReal)
                .with_inverse_normalization(true)
                .with_input_buffer_axis_stride(0, 7)
                .unwrap()
                .with_output_buffer_axis_stride(0, 11)
                .unwrap()
                .with_zero_padding(0, 1, 2)
                .unwrap()
                .with_zero_padding(1, 2, 4)
                .unwrap(),
        )
        .unwrap();
        let inverse = NdRealFftIr::build(&inverse_plan, device()).unwrap();
        assert!(inverse.input_formatted_copy.is_some());
        assert!(inverse.output_formatted_copy.is_some());
        let inverse_pad = inverse.zero_pad_pass.as_ref().unwrap();
        assert_eq!(
            inverse_pad.operation,
            crate::ZeroPadPassOperation::FinalizeInverseOutput
        );
        assert_eq!(inverse_pad.input_storage_scalar, inverse.scalar);
        assert_eq!(inverse_pad.output_storage_scalar, inverse.scalar);
        let actual_output = execute_nd_c2r_ir(&inverse, &actual_spectrum).unwrap();

        let dense_inverse_plan = FftPlan::build(
            FftConfig::new(dimensions.clone())
                .with_batch_count(batch_count)
                .with_transform(TransformKind::ComplexToReal)
                .with_inverse_normalization(true),
        )
        .unwrap();
        let dense_inverse = NdRealFftIr::build(&dense_inverse_plan, device()).unwrap();
        let mut expected_output = execute_nd_c2r_ir(&dense_inverse, &actual_spectrum).unwrap();
        for batch in 0..batch_count {
            let base = batch * full_len;
            for row in 0..dimensions[0] {
                for col in 0..dimensions[1] {
                    if row == 1 || (2..4).contains(&col) {
                        expected_output[base + row * dimensions[1] + col] = 0.0;
                    }
                }
            }
        }
        let max_inverse_error = actual_output
            .iter()
            .zip(&expected_output)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0, f64::max);
        assert!(
            max_inverse_error <= 3.0e-10 * full_len as f64,
            "formatted+padding ND C2R error {max_inverse_error:e}"
        );
    }

    #[test]
    fn multidimensional_real_zero_padding_wraps_only_spatial_boundaries() {
        let dimensions = vec![3usize, 8];
        let full_len = dimensions.iter().product::<usize>();
        let mut input = (0..full_len)
            .map(|index| {
                let x = index as f64;
                (0.11 * x).sin() + 0.31 * (0.047 * x).cos()
            })
            .collect::<Vec<_>>();
        let mut manual = input.clone();
        for row in 0..dimensions[0] {
            for col in 0..dimensions[1] {
                let index = row * dimensions[1] + col;
                if row == 1 || (2..4).contains(&col) {
                    input[index] = 1.0e6 + index as f64;
                    manual[index] = 0.0;
                }
            }
        }
        let padded_config = FftConfig::new(dimensions.clone())
            .with_transform(TransformKind::RealToComplex)
            .with_zero_padding(0, 1, 2)
            .unwrap()
            .with_zero_padding(1, 2, 4)
            .unwrap();
        let forward_plan = FftPlan::build(padded_config).unwrap();
        let forward = NdRealFftIr::build(&forward_plan, device()).unwrap();
        assert!(forward.zero_pad_pass.is_some());
        assert!(forward.real_axis.zero_pad_pass.is_none());
        let actual = execute_nd_r2c_ir(&forward, &input).unwrap();

        let baseline_plan = FftPlan::build(
            FftConfig::new(dimensions.clone()).with_transform(TransformKind::RealToComplex),
        )
        .unwrap();
        let baseline = NdRealFftIr::build(&baseline_plan, device()).unwrap();
        let expected = execute_nd_r2c_ir(&baseline, &manual).unwrap();
        let forward_error = actual
            .iter()
            .zip(&expected)
            .map(|(actual, expected)| (*actual - *expected).norm_sqr().sqrt())
            .fold(0.0, f64::max);
        assert!(forward_error < 1.0e-8 * full_len as f64, "{forward_error}");

        let inverse_plan = FftPlan::build(
            FftConfig::new(dimensions.clone())
                .with_transform(TransformKind::ComplexToReal)
                .with_inverse_normalization(true)
                .with_zero_padding(0, 1, 2)
                .unwrap()
                .with_zero_padding(1, 2, 4)
                .unwrap(),
        )
        .unwrap();
        let inverse = NdRealFftIr::build(&inverse_plan, device()).unwrap();
        let restored = execute_nd_c2r_ir(&inverse, &actual).unwrap();
        let inverse_error = restored
            .iter()
            .zip(&manual)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0, f64::max);
        assert!(inverse_error < 1.0e-8 * full_len as f64, "{inverse_error}");
    }

    #[test]
    fn grouped_multidimensional_real_owns_axis_children_and_zero_pad_tail() {
        let dimensions = vec![3usize, 8];
        let tensor_len = dimensions.iter().product::<usize>();
        let batch_count = 7usize;
        let grouped_batch = 3usize;
        let mut input = (0..tensor_len * batch_count)
            .map(|index| {
                let x = index as f64;
                (0.071 * x).sin() + 0.27 * (0.043 * x).cos() + x * 0.0003
            })
            .collect::<Vec<_>>();
        let mut manual = input.clone();
        for batch in 0..batch_count {
            let base = batch * tensor_len;
            for row in 0..dimensions[0] {
                for col in 0..dimensions[1] {
                    let linear = row * dimensions[1] + col;
                    if row == 1 || (2..4).contains(&col) {
                        input[base + linear] = 1.0e5 + (base + linear) as f64;
                        manual[base + linear] = 0.0;
                    }
                }
            }
        }

        let grouped_config = FftConfig::new(dimensions.clone())
            .with_batch_count(batch_count)
            .with_precision(Precision::F16StorageF32Compute)
            .with_grouped_batch(0, grouped_batch)
            .unwrap()
            .with_grouped_batch(1, grouped_batch)
            .unwrap()
            .with_zero_padding(0, 1, 2)
            .unwrap()
            .with_zero_padding(1, 2, 4)
            .unwrap();
        let forward_plan = FftPlan::build(
            grouped_config
                .clone()
                .with_transform(TransformKind::RealToComplex),
        )
        .unwrap();
        let forward = NdRealFftIr::build(&forward_plan, device()).unwrap();
        assert_eq!(forward.real_grouped_batch, Some(grouped_batch));
        assert_eq!(forward.real_axis.batch_count, batch_count * dimensions[0]);
        assert_eq!(forward.real_axis.grouped_batch, grouped_batch);
        let zero_pad = forward.zero_pad_pass.as_deref().unwrap();
        assert_eq!(zero_pad.grouped_batch, grouped_batch);
        assert_eq!(zero_pad.dispatch.x, 3);
        let outer = &forward.complex_axes[0];
        assert_eq!(outer.pack.line_count, 5);
        assert_eq!(outer.pack.grouped_batch, Some(grouped_batch));
        assert_eq!(outer.scatter.grouped_batch, Some(grouped_batch));
        assert_eq!(outer.pack.dispatch.x, 3);
        assert_eq!(outer.scatter.dispatch.x, 3);
        assert_eq!(outer.transform.grouped_batch(), grouped_batch);
        let outer_kernel = stockham_kernel(&outer.transform);
        assert_eq!(
            outer_kernel.workgroup_grouping.axis_layout,
            crate::kernel_ir::StockhamWorkgroupAxisLayout::TransformsXThreadsY
        );
        assert_eq!(
            outer_kernel.workgroup_grouping.transforms_per_workgroup,
            grouped_batch
        );
        assert_eq!(outer_kernel.workgroup_size.x, grouped_batch as u32);
        assert_eq!(outer_kernel.workgroup_size.y, 1);
        let program = crate::ProgramIr::nd_real_fft(&forward).unwrap();
        assert!(program.passes.iter().any(|pass| pass.dispatch.x > 3));

        let actual = execute_nd_r2c_ir(&forward, &input).unwrap();
        let baseline_plan = FftPlan::build(
            FftConfig::new(dimensions.clone())
                .with_batch_count(batch_count)
                .with_precision(Precision::F16StorageF32Compute)
                .with_transform(TransformKind::RealToComplex),
        )
        .unwrap();
        let baseline = NdRealFftIr::build(&baseline_plan, device()).unwrap();
        let expected = execute_nd_r2c_ir(&baseline, &manual).unwrap();
        let forward_error = actual
            .iter()
            .zip(&expected)
            .map(|(actual, expected)| (*actual - *expected).norm_sqr().sqrt())
            .fold(0.0, f64::max);
        assert!(
            forward_error < 2.0e-8 * tensor_len as f64,
            "{forward_error}"
        );

        let inverse_plan = FftPlan::build(
            grouped_config
                .with_transform(TransformKind::ComplexToReal)
                .with_inverse_normalization(true),
        )
        .unwrap();
        let inverse = NdRealFftIr::build(&inverse_plan, device()).unwrap();
        assert_eq!(inverse.real_axis.grouped_batch, grouped_batch);
        assert!(
            crate::ProgramIr::nd_real_fft(&inverse)
                .unwrap()
                .passes
                .iter()
                .any(|pass| pass.dispatch.x > 3)
        );
        let restored = execute_nd_c2r_ir(&inverse, &actual).unwrap();
        let inverse_error = restored
            .iter()
            .zip(&manual)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0, f64::max);
        assert!(
            inverse_error < 2.0e-8 * tensor_len as f64,
            "{inverse_error}"
        );
    }

    #[test]
    fn grouped_multidimensional_real_strided_stockham_uses_upstream_xy_tile() {
        let grouped_batch = 3usize;
        let plan = FftPlan::build(
            FftConfig::new(vec![64usize, 8])
                .with_batch_count(5)
                .with_grouped_batch(0, grouped_batch)
                .unwrap()
                .with_grouped_batch(1, grouped_batch)
                .unwrap()
                .with_transform(TransformKind::RealToComplex),
        )
        .unwrap();
        let ir = NdRealFftIr::build(&plan, device()).unwrap();
        let outer = &ir.complex_axes[0];
        assert_eq!(outer.transform.grouped_batch(), grouped_batch);
        let kernel = stockham_kernel(&outer.transform);
        assert_eq!(
            kernel.workgroup_grouping.axis_layout,
            crate::kernel_ir::StockhamWorkgroupAxisLayout::TransformsXThreadsY
        );
        assert_eq!(
            kernel.workgroup_grouping.transforms_per_workgroup,
            grouped_batch
        );
        assert_eq!([kernel.workgroup_size.x, kernel.workgroup_size.y], [3, 8]);
    }

    #[test]
    fn multidimensional_real_round_trip_including_bluestein_outer_axis() {
        for dimensions in [vec![3usize, 8], vec![103usize, 8]] {
            let full_len = dimensions.iter().product::<usize>();
            let mut bluestein_tuning = crate::PlannerTuning::portable();
            if dimensions[0] == 103 {
                bluestein_tuning.max_rader_fft_prime = 100;
            }
            let input = (0..full_len)
                .map(|index| {
                    let x = index as f64;
                    (0.13 * x).sin() + 0.23 * (0.041 * x).cos() + x * 0.0007
                })
                .collect::<Vec<_>>();
            let forward_plan = FftPlan::build(
                FftConfig::new(dimensions.clone())
                    .with_transform(TransformKind::RealToComplex)
                    .with_tuning(bluestein_tuning),
            )
            .unwrap();
            let forward = NdRealFftIr::build(&forward_plan, device()).unwrap();
            if dimensions[0] == 103 {
                assert!(matches!(
                    forward.complex_axes[0].transform,
                    OneDimFftIr::Bluestein(_)
                ));
            }
            let spectrum = execute_nd_r2c_ir(&forward, &input).unwrap();
            assert_eq!(spectrum.len(), forward.compact_tensor_len);

            let inverse_plan = FftPlan::build(
                FftConfig::new(dimensions)
                    .with_transform(TransformKind::ComplexToReal)
                    .with_inverse_normalization(true)
                    .with_tuning(bluestein_tuning),
            )
            .unwrap();
            let inverse = NdRealFftIr::build(&inverse_plan, device()).unwrap();
            let restored = execute_nd_c2r_ir(&inverse, &spectrum).unwrap();
            let error = restored
                .iter()
                .zip(&input)
                .map(|(actual, expected)| (actual - expected).abs())
                .fold(0.0, f64::max);
            assert!(error < 7.0e-8 * full_len as f64, "{error}");
        }
    }
}
