//! Backend-neutral multi-pass Bluestein execution IR.
//!
//! The first GPU Bluestein path deliberately composes the already-validated
//! one-upload Stockham kernels instead of introducing a second FFT engine. The
//! outer transform is represented as preprocess -> forward FFT -> frequency
//! multiply -> normalized inverse FFT -> postprocess.

use crate::complex::Complex64;
use crate::config::{DeviceProfile, Direction, FftConfig, Precision, TransformKind};
use crate::error::{Result, VkFftError};
use crate::kernel_ir::{
    DispatchGeometry, KernelIr, RealEvenInversePreprocessMapping, RealEvenPackMapping,
    RealEvenPostprocessMapping, RealEvenUnpackMapping, ScalarType, StockhamIoMapping,
    WorkgroupSize, execute_stockham_ir_with_lookup,
};
use crate::lut::BluesteinTable;
use crate::planner::{AxisAlgorithm, C2cDeviceAxisClass, FftPlan};
use crate::recursive_ir::{
    CooleyTukeyInputModifier, RecursiveFftIr, RecursiveFftNodeIr, execute_recursive_fft_ir,
    execute_recursive_fft_ir_with_resources,
};
use crate::scheduler::{StockhamAxisBlockSchedule, plan_gpu_other_axis_bluestein_wrapper_block};
use crate::zero_pad_ir::{ZeroPadPassIr, execute_zero_pad_pass};

#[derive(Debug, Clone, PartialEq)]
pub enum BluesteinPassOperation {
    Preprocess,
    MultiplyKernelSpectrum { spectrum: Vec<Complex64> },
    Postprocess { normalize: bool },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BluesteinInputModifier {
    #[default]
    None,
    RealEvenPack(RealEvenPackMapping),
    RealEvenInversePreprocess(RealEvenInversePreprocessMapping),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BluesteinOutputModifier {
    #[default]
    None,
    RealEvenPostprocess(RealEvenPostprocessMapping),
    RealEvenUnpack(RealEvenUnpackMapping),
}

#[derive(Debug, Clone, PartialEq)]
pub struct BluesteinPassIr {
    pub name: String,
    pub scalar: ScalarType,
    pub input_storage_scalar: ScalarType,
    pub output_storage_scalar: ScalarType,
    pub direction: Direction,
    pub logical_len: usize,
    pub convolution_len: usize,
    pub batch_count: usize,
    pub grouped_batch: usize,
    /// Optional physical X/Y ownership for higher-axis Bluestein wrapper passes.
    /// Logical groupedBatch remains unchanged; one transform slot is assigned to X
    /// while element slots are distributed over the orthogonal FFT-lane dimension.
    pub axis_batch_block: Option<StockhamAxisBlockSchedule>,
    pub workgroup_size: WorkgroupSize,
    pub dispatch: DispatchGeometry,
    pub operation: BluesteinPassOperation,
    pub input_modifier: BluesteinInputModifier,
    pub output_modifier: BluesteinOutputModifier,
}

#[derive(Debug, Clone, Copy)]
struct BluesteinPassShape {
    scalar: ScalarType,
    direction: Direction,
    logical_len: usize,
    convolution_len: usize,
    batch_count: usize,
    grouped_batch: usize,
    device: DeviceProfile,
}

impl BluesteinPassIr {
    fn new(
        name: String,
        shape: BluesteinPassShape,
        operation: BluesteinPassOperation,
    ) -> Result<Self> {
        if shape.device.max_threads_per_block == 0 {
            return Err(VkFftError::ResourceLimitExceeded {
                resource: "threads per workgroup",
                required: 1,
                available: 0,
            });
        }
        let local_size = shape
            .convolution_len
            .min(shape.device.max_threads_per_block)
            .max(1);
        let x = u32::try_from(local_size).map_err(|_| VkFftError::ValueOutOfRange {
            field: "Bluestein workgroup size",
        })?;
        if shape.grouped_batch == 0 {
            return Err(VkFftError::InvalidKernelIr(
                "Bluestein groupedBatch must be non-zero",
            ));
        }
        let dispatch_x =
            u32::try_from(shape.batch_count.div_ceil(shape.grouped_batch)).map_err(|_| {
                VkFftError::ValueOutOfRange {
                    field: "Bluestein dispatch workgroup count",
                }
            })?;
        let pass = Self {
            name,
            scalar: shape.scalar,
            input_storage_scalar: shape.scalar,
            output_storage_scalar: shape.scalar,
            direction: shape.direction,
            logical_len: shape.logical_len,
            convolution_len: shape.convolution_len,
            batch_count: shape.batch_count,
            grouped_batch: shape.grouped_batch,
            axis_batch_block: None,
            workgroup_size: WorkgroupSize { x, y: 1, z: 1 },
            dispatch: DispatchGeometry {
                x: dispatch_x,
                y: 1,
                z: 1,
            },
            operation,
            input_modifier: BluesteinInputModifier::None,
            output_modifier: BluesteinOutputModifier::None,
        };
        pass.validate()?;
        Ok(pass)
    }

    fn with_external_input_storage(mut self, storage: ScalarType) -> Result<Self> {
        if storage != self.scalar
            && (!matches!(
                (self.scalar, storage),
                (ScalarType::F32, ScalarType::F16) | (ScalarType::F64, ScalarType::F32)
            ) || !matches!(self.operation, BluesteinPassOperation::Preprocess))
        {
            return Err(VkFftError::UnsupportedPrecision {
                backend: "Bluestein preprocess input boundary",
                precision: "unsupported compute/storage scalar pair or pass operation",
            });
        }
        self.input_storage_scalar = storage;
        self.validate()?;
        Ok(self)
    }

    fn with_external_output_storage(mut self, storage: ScalarType) -> Result<Self> {
        if storage != self.scalar
            && (!matches!(
                (self.scalar, storage),
                (ScalarType::F32, ScalarType::F16) | (ScalarType::F64, ScalarType::F32)
            ) || !matches!(self.operation, BluesteinPassOperation::Postprocess { .. }))
        {
            return Err(VkFftError::UnsupportedPrecision {
                backend: "Bluestein postprocess output boundary",
                precision: "unsupported compute/storage scalar pair or pass operation",
            });
        }
        self.output_storage_scalar = storage;
        self.validate()?;
        Ok(self)
    }

    pub(crate) fn with_real_even_pack_input(
        mut self,
        mapping: RealEvenPackMapping,
    ) -> Result<Self> {
        if self.input_modifier != BluesteinInputModifier::None
            || !matches!(self.operation, BluesteinPassOperation::Preprocess)
            || self.direction != Direction::Forward
            || self.input_storage_scalar != self.scalar
        {
            return Err(VkFftError::UnsupportedKernelPath(
                "Bluestein even-real pack fusion requires an unfused forward preprocess at compute storage precision",
            ));
        }
        mapping.validate(self.logical_len)?;
        self.input_modifier = BluesteinInputModifier::RealEvenPack(mapping);
        self.name.push_str("_real_even_pack");
        self.validate()?;
        Ok(self)
    }

    pub(crate) fn with_real_even_inverse_preprocess_input(
        mut self,
        mapping: RealEvenInversePreprocessMapping,
    ) -> Result<Self> {
        if self.input_modifier != BluesteinInputModifier::None
            || !matches!(self.operation, BluesteinPassOperation::Preprocess)
            || self.direction != Direction::Inverse
            || self.input_storage_scalar != self.scalar
        {
            return Err(VkFftError::UnsupportedKernelPath(
                "Bluestein even-real inverse-preprocess fusion requires an unfused inverse preprocess at compute storage precision",
            ));
        }
        mapping.validate(self.logical_len)?;
        self.input_modifier = BluesteinInputModifier::RealEvenInversePreprocess(mapping);
        self.name.push_str("_real_even_inverse_preprocess");
        self.validate()?;
        Ok(self)
    }

    pub(crate) fn with_real_even_postprocess_output(
        mut self,
        mapping: RealEvenPostprocessMapping,
    ) -> Result<Self> {
        if self.output_modifier != BluesteinOutputModifier::None
            || !matches!(self.operation, BluesteinPassOperation::Postprocess { .. })
            || self.direction != Direction::Forward
            || self.output_storage_scalar != self.scalar
        {
            return Err(VkFftError::UnsupportedKernelPath(
                "Bluestein even-real postprocess fusion requires an unfused forward postprocess at compute storage precision",
            ));
        }
        mapping.validate(self.logical_len)?;
        self.output_modifier = BluesteinOutputModifier::RealEvenPostprocess(mapping);
        self.name.push_str("_real_even_postprocess");
        self.validate()?;
        Ok(self)
    }

    pub(crate) fn with_real_even_unpack_output(
        mut self,
        mapping: RealEvenUnpackMapping,
    ) -> Result<Self> {
        if self.output_modifier != BluesteinOutputModifier::None
            || !matches!(self.operation, BluesteinPassOperation::Postprocess { .. })
            || self.direction != Direction::Inverse
            || self.output_storage_scalar != self.scalar
        {
            return Err(VkFftError::UnsupportedKernelPath(
                "Bluestein even-real unpack fusion requires an unfused inverse postprocess at compute storage precision",
            ));
        }
        mapping.validate(self.logical_len)?;
        self.output_modifier = BluesteinOutputModifier::RealEvenUnpack(mapping);
        self.name.push_str("_real_even_unpack");
        self.validate()?;
        Ok(self)
    }

    pub(crate) fn with_axis_batch_block(
        mut self,
        block: StockhamAxisBlockSchedule,
        device: DeviceProfile,
    ) -> Result<Self> {
        block.validate(self.batch_count, device)?;
        if self.grouped_batch > 1 && block.grouped_batch != self.grouped_batch {
            return Err(VkFftError::InvalidKernelIr(
                "Bluestein wrapper physical block changed explicit groupedBatch ownership",
            ));
        }
        self.workgroup_size = WorkgroupSize {
            x: u32::try_from(block.local_size_x).map_err(|_| VkFftError::ValueOutOfRange {
                field: "Bluestein wrapper local_size_x",
            })?,
            y: u32::try_from(block.local_size_y).map_err(|_| VkFftError::ValueOutOfRange {
                field: "Bluestein wrapper local_size_y",
            })?,
            z: 1,
        };
        self.axis_batch_block = Some(block);
        self.dispatch.x =
            u32::try_from(self.batch_count.div_ceil(block.grouped_batch)).map_err(|_| {
                VkFftError::ValueOutOfRange {
                    field: "Bluestein wrapper physical dispatch count",
                }
            })?;
        self.validate()?;
        Ok(self)
    }

    pub fn validate(&self) -> Result<()> {
        if self.logical_len == 0 || self.convolution_len == 0 || self.batch_count == 0 {
            return Err(VkFftError::InvalidKernelIr(
                "Bluestein pass dimensions and batch count must be non-zero",
            ));
        }
        let minimum = self
            .logical_len
            .checked_mul(2)
            .and_then(|value| value.checked_sub(1))
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "Bluestein pass minimum convolution length",
            })?;
        if self.convolution_len < minimum {
            return Err(VkFftError::InvalidKernelIr(
                "Bluestein pass convolution length must be at least 2N-1",
            ));
        }
        if self.workgroup_size.x == 0 || self.workgroup_size.y == 0 || self.workgroup_size.z == 0 {
            return Err(VkFftError::InvalidKernelIr(
                "Bluestein pass workgroup dimensions must be non-zero",
            ));
        }
        if self.grouped_batch == 0 || self.dispatch.y != 1 || self.dispatch.z != 1 {
            return Err(VkFftError::InvalidKernelIr(
                "Bluestein pass dispatch does not match grouped padded transforms",
            ));
        }
        if let Some(block) = self.axis_batch_block {
            let expected = if block.transforms_on_x {
                [block.grouped_batch, block.threads_per_transform]
            } else {
                [block.threads_per_transform, block.grouped_batch]
            };
            if (self.grouped_batch > 1 && block.grouped_batch != self.grouped_batch)
                || self.dispatch.x as usize != self.batch_count.div_ceil(block.grouped_batch)
                || [
                    self.workgroup_size.x as usize,
                    self.workgroup_size.y as usize,
                ] != expected
                || [block.local_size_x, block.local_size_y] != expected
            {
                return Err(VkFftError::InvalidKernelIr(
                    "Bluestein wrapper physical block is inconsistent",
                ));
            }
        } else if self.dispatch.x as usize != self.batch_count.div_ceil(self.grouped_batch) {
            return Err(VkFftError::InvalidKernelIr(
                "Bluestein logical groupedBatch dispatch is inconsistent",
            ));
        }
        let supported_storage = |storage: ScalarType| {
            storage == self.scalar
                || matches!(
                    (self.scalar, storage),
                    (ScalarType::F32, ScalarType::F16) | (ScalarType::F64, ScalarType::F32)
                )
        };
        let storage_contract_ok = match self.operation {
            BluesteinPassOperation::Preprocess => {
                supported_storage(self.input_storage_scalar)
                    && self.output_storage_scalar == self.scalar
            }
            BluesteinPassOperation::MultiplyKernelSpectrum { .. } => {
                self.input_storage_scalar == self.scalar
                    && self.output_storage_scalar == self.scalar
            }
            BluesteinPassOperation::Postprocess { .. } => {
                self.input_storage_scalar == self.scalar
                    && supported_storage(self.output_storage_scalar)
            }
        };
        if self.scalar == ScalarType::F16 || !storage_contract_ok {
            return Err(VkFftError::InvalidKernelIr(
                "Bluestein pass storage boundary is inconsistent with its operation",
            ));
        }
        match self.input_modifier {
            BluesteinInputModifier::None => {}
            BluesteinInputModifier::RealEvenPack(mapping) => {
                if !matches!(self.operation, BluesteinPassOperation::Preprocess)
                    || self.direction != Direction::Forward
                    || self.input_storage_scalar != self.scalar
                {
                    return Err(VkFftError::InvalidKernelIr(
                        "Bluestein even-real pack modifier is attached to an incompatible pass",
                    ));
                }
                mapping.validate(self.logical_len)?;
            }
            BluesteinInputModifier::RealEvenInversePreprocess(mapping) => {
                if !matches!(self.operation, BluesteinPassOperation::Preprocess)
                    || self.direction != Direction::Inverse
                    || self.input_storage_scalar != self.scalar
                {
                    return Err(VkFftError::InvalidKernelIr(
                        "Bluestein even-real inverse-preprocess modifier is attached to an incompatible pass",
                    ));
                }
                mapping.validate(self.logical_len)?;
            }
        }
        match self.output_modifier {
            BluesteinOutputModifier::None => {}
            BluesteinOutputModifier::RealEvenPostprocess(mapping) => {
                if !matches!(self.operation, BluesteinPassOperation::Postprocess { .. })
                    || self.direction != Direction::Forward
                    || self.output_storage_scalar != self.scalar
                {
                    return Err(VkFftError::InvalidKernelIr(
                        "Bluestein even-real postprocess modifier is attached to an incompatible pass",
                    ));
                }
                mapping.validate(self.logical_len)?;
            }
            BluesteinOutputModifier::RealEvenUnpack(mapping) => {
                if !matches!(self.operation, BluesteinPassOperation::Postprocess { .. })
                    || self.direction != Direction::Inverse
                    || self.output_storage_scalar != self.scalar
                {
                    return Err(VkFftError::InvalidKernelIr(
                        "Bluestein even-real unpack modifier is attached to an incompatible pass",
                    ));
                }
                mapping.validate(self.logical_len)?;
            }
        }
        if self.input_modifier != BluesteinInputModifier::None
            && self.output_modifier != BluesteinOutputModifier::None
        {
            return Err(VkFftError::InvalidKernelIr(
                "one Bluestein elementwise pass cannot carry both real input and output modifiers",
            ));
        }
        if let BluesteinPassOperation::MultiplyKernelSpectrum { spectrum } = &self.operation {
            if spectrum.len() != self.convolution_len {
                return Err(VkFftError::InvalidKernelIr(
                    "Bluestein frequency kernel length does not match convolution length",
                ));
            }
            if spectrum
                .iter()
                .any(|value| !value.re.is_finite() || !value.im.is_finite())
            {
                return Err(VkFftError::InvalidKernelIr(
                    "Bluestein frequency kernel contains non-finite values",
                ));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct BluesteinPipelineIr {
    pub logical_len: usize,
    pub convolution_len: usize,
    pub batch_count: usize,
    pub grouped_batch: usize,
    pub direction: Direction,
    pub scalar: ScalarType,
    pub external_scalar: ScalarType,
    pub table: BluesteinTable,
    pub preprocess: BluesteinPassIr,
    pub forward_fft: Box<RecursiveFftIr>,
    pub multiply: BluesteinPassIr,
    pub inverse_fft: Box<RecursiveFftIr>,
    pub zero_pad_pass: Option<ZeroPadPassIr>,
    pub postprocess: BluesteinPassIr,
}

fn plan_axis0_bluestein_wrapper_block(
    convolution_len: usize,
    batch_count: usize,
    grouped_batch: usize,
    device: DeviceProfile,
) -> Result<Option<StockhamAxisBlockSchedule>> {
    if convolution_len == 0
        || grouped_batch == 0
        || grouped_batch > batch_count
        || grouped_batch > device.max_workgroup_size[1]
        || grouped_batch > device.max_threads_per_block
    {
        return Ok(None);
    }
    let max_x_by_threads = device.max_threads_per_block / grouped_batch;
    let threads_per_transform = convolution_len
        .min(128)
        .min(device.max_workgroup_size[0])
        .min(max_x_by_threads);
    if threads_per_transform == 0 {
        return Ok(None);
    }
    let block = StockhamAxisBlockSchedule {
        threads_per_transform,
        grouped_batch,
        transforms_on_x: false,
        axis_swapped: false,
        local_size_x: threads_per_transform,
        local_size_y: grouped_batch,
    };
    block.validate(batch_count, device)?;
    Ok(Some(block))
}

impl BluesteinPipelineIr {
    pub fn build(plan: &FftPlan, direction: Direction, device: DeviceProfile) -> Result<Self> {
        Self::build_with_internal_compute_storage(plan, direction, device, false)
    }

    pub(crate) fn build_internal_compute_storage(
        plan: &FftPlan,
        direction: Direction,
        device: DeviceProfile,
    ) -> Result<Self> {
        Self::build_with_internal_compute_storage(plan, direction, device, true)
    }

    fn build_with_internal_compute_storage(
        plan: &FftPlan,
        direction: Direction,
        device: DeviceProfile,
        internal_compute_storage: bool,
    ) -> Result<Self> {
        if plan.config.dimensions.len() != 1 {
            return Err(VkFftError::UnsupportedKernelPath(
                "initial Bluestein pipeline supports one-dimensional plans only",
            ));
        }
        if plan.config.transform != TransformKind::ComplexToComplex {
            return Err(VkFftError::UnsupportedKernelPath(
                "initial Bluestein pipeline supports C2C transforms only",
            ));
        }
        let (scalar, compute_precision, caller_external_scalar) = match plan.config.precision {
            Precision::F16StorageF32Compute => (ScalarType::F32, Precision::F32, ScalarType::F16),
            Precision::F32 => (ScalarType::F32, Precision::F32, ScalarType::F32),
            Precision::F64 if device.supports_f64 => {
                (ScalarType::F64, Precision::F64, ScalarType::F64)
            }
            Precision::F64ComputeF32Storage if device.supports_f64 => {
                (ScalarType::F64, Precision::F64, ScalarType::F32)
            }
            other => {
                return Err(VkFftError::UnsupportedPrecision {
                    backend: "Bluestein pipeline IR",
                    precision: precision_name(other),
                });
            }
        };
        let external_scalar = if internal_compute_storage {
            scalar
        } else {
            caller_external_scalar
        };
        let axis = plan
            .axes
            .first()
            .ok_or(VkFftError::InvalidKernelIr("missing Bluestein axis plan"))?;
        let AxisAlgorithm::Bluestein {
            convolution_len, ..
        } = axis.algorithm
        else {
            return Err(VkFftError::UnsupportedKernelPath(
                "Bluestein pipeline requires a planner-selected Bluestein axis",
            ));
        };
        let logical_len = axis.effective_fft_len;
        let batch_count = plan.config.batch_count;
        let grouped_batch_override = plan.config.grouped_batch_for_axis(0);
        let grouped_batch = grouped_batch_override.unwrap_or(1);
        let table = BluesteinTable::new(logical_len, convolution_len, direction)?;
        let zero_pad_pass = plan
            .config
            .zero_padding_for_axis(0)
            .map(|range| {
                let pass = ZeroPadPassIr::build_with_domain(
                    logical_len,
                    batch_count,
                    plan.config.precision,
                    direction,
                    range,
                    plan.config.zero_padding_domain,
                    device,
                )?;
                if let Some(grouped_batch) = grouped_batch_override {
                    pass.with_grouped_batch(grouped_batch)
                } else {
                    Ok(pass)
                }
            })
            .transpose()?;

        let convolution_axis_class = plan
            .c2c_device_axis_class_override
            .unwrap_or(C2cDeviceAxisClass::Contiguous);
        let mut forward_config = FftConfig::new(vec![convolution_len])
            .with_batch_count(batch_count)
            .with_precision(plan.config.precision)
            .with_tuning(plan.config.tuning)
            .with_bandwidth_boost(plan.config.bandwidth_boost);
        if let Some(grouped_batch) = grouped_batch_override {
            forward_config = forward_config.with_grouped_batch(0, grouped_batch)?;
        }
        let inverse_config = forward_config.clone().with_inverse_normalization(true);
        let forward_plan =
            FftPlan::build_c2c_bluestein_child(forward_config, convolution_axis_class)?;
        let inverse_plan =
            FftPlan::build_c2c_bluestein_child(inverse_config, convolution_axis_class)?;
        let mut forward_ir = RecursiveFftIr::build_internal_compute_storage(
            &forward_plan,
            Direction::Forward,
            device,
        )?;
        let mut inverse_ir = RecursiveFftIr::build_internal_compute_storage(
            &inverse_plan,
            Direction::Inverse,
            device,
        )?;
        if grouped_batch_override.is_some() {
            forward_ir = forward_ir.with_axis0_single_upload_block(device)?;
            inverse_ir = inverse_ir.with_axis0_single_upload_block(device)?;
        }
        // Recursive Bluestein inverse convolution can consume the immutable spectrum
        // directly at its root Cooley pack, exactly like recursive FFT-Rader. Keep
        // Four-step / forced-Rader upload graphs on the explicit multiply path because
        // those materializers intentionally do not accept root boundary resources.
        if inverse_ir.four_step_plan.is_none()
            && inverse_ir.rader_forced_upload_schedule.is_none()
            && let RecursiveFftNodeIr::CooleyTukey(root) = &mut inverse_ir.root
            && root.pack_right.input_modifier == CooleyTukeyInputModifier::None
        {
            root.pack_right = root.pack_right.clone().with_lookup_table_input_multiply()?;
            inverse_ir.validate()?;
        }
        let forward_fft = Box::new(forward_ir);
        let inverse_fft = Box::new(inverse_ir);

        // Materialize the Bluestein convolution-kernel spectrum once with the exact
        // Stockham IR semantics. Backends can upload/cache this immutable LUT without
        // changing the multi-pass algorithm composition.
        let kernel_plan = FftPlan::build(
            FftConfig::new(vec![convolution_len])
                .with_precision(compute_precision)
                .with_tuning(plan.config.tuning),
        )?;
        let kernel_fft = RecursiveFftIr::build(&kernel_plan, Direction::Forward, device)?;
        let spectrum = execute_recursive_fft_ir(&kernel_fft, &table.convolution_kernel)?;

        let direction_name = match direction {
            Direction::Forward => "forward",
            Direction::Inverse => "inverse",
        };
        let pass_shape = BluesteinPassShape {
            scalar,
            direction,
            logical_len,
            convolution_len,
            batch_count,
            grouped_batch,
            device,
        };
        let mut preprocess = BluesteinPassIr::new(
            format!("vkfft_bluestein_pre_{logical_len}_{direction_name}"),
            pass_shape,
            BluesteinPassOperation::Preprocess,
        )?;
        let mut multiply = BluesteinPassIr::new(
            format!("vkfft_bluestein_mul_{logical_len}_{direction_name}"),
            pass_shape,
            BluesteinPassOperation::MultiplyKernelSpectrum { spectrum },
        )?;
        let mut postprocess = BluesteinPassIr::new(
            format!("vkfft_bluestein_post_{logical_len}_{direction_name}"),
            pass_shape,
            BluesteinPassOperation::Postprocess {
                normalize: direction == Direction::Inverse && plan.config.normalize_inverse,
            },
        )?;

        if let Some(grouped_batch) = grouped_batch_override
            && let Some(block) = plan_axis0_bluestein_wrapper_block(
                convolution_len,
                batch_count,
                grouped_batch,
                device,
            )?
        {
            preprocess = preprocess.with_axis_batch_block(block, device)?;
            multiply = multiply.with_axis_batch_block(block, device)?;
            postprocess = postprocess.with_axis_batch_block(block, device)?;
        }

        if external_scalar != scalar {
            let input_boundary_owned_by_pad = zero_pad_pass
                .as_ref()
                .is_some_and(|pass| pass.operation.is_input_boundary());
            let output_boundary_owned_by_pad = zero_pad_pass
                .as_ref()
                .is_some_and(|pass| pass.operation.is_output_boundary());
            if !input_boundary_owned_by_pad {
                preprocess = preprocess.with_external_input_storage(external_scalar)?;
            }
            if !output_boundary_owned_by_pad {
                postprocess = postprocess.with_external_output_storage(external_scalar)?;
            }
        }

        let pipeline = Self {
            logical_len,
            convolution_len,
            batch_count,
            grouped_batch,
            direction,
            scalar,
            table,
            external_scalar,
            preprocess,
            forward_fft,
            multiply,
            inverse_fft,
            zero_pad_pass,
            postprocess,
        };
        pipeline.validate()?;
        Ok(pipeline)
    }

    pub(crate) fn with_other_axis_blocks(
        mut self,
        fastest_axis_len: usize,
        grouped_batch_override: Option<usize>,
        axis1_grouped_batch_override: Option<usize>,
        device: DeviceProfile,
    ) -> Result<Self> {
        self.forward_fft = Box::new(
            (*self.forward_fft).with_other_axis_single_upload_block_with_grouped_batch(
                fastest_axis_len,
                grouped_batch_override,
                axis1_grouped_batch_override,
                device,
            )?,
        );
        self.inverse_fft = Box::new(
            (*self.inverse_fft).with_other_axis_single_upload_block_with_grouped_batch(
                fastest_axis_len,
                grouped_batch_override,
                axis1_grouped_batch_override,
                device,
            )?,
        );
        if let Some(block) = plan_gpu_other_axis_bluestein_wrapper_block(
            self.convolution_len,
            self.batch_count,
            fastest_axis_len,
            self.scalar.complex_bytes(),
            grouped_batch_override,
            axis1_grouped_batch_override,
            device,
        )? {
            self.preprocess = self.preprocess.with_axis_batch_block(block, device)?;
            self.multiply = self.multiply.with_axis_batch_block(block, device)?;
            self.postprocess = self.postprocess.with_axis_batch_block(block, device)?;
        }
        self.validate()?;
        Ok(self)
    }

    pub const fn external_storage_scalar(&self) -> ScalarType {
        self.external_scalar
    }

    pub fn kernel_spectrum(&self) -> Result<&[Complex64]> {
        match &self.multiply.operation {
            BluesteinPassOperation::MultiplyKernelSpectrum { spectrum } => Ok(spectrum),
            _ => Err(VkFftError::InvalidKernelIr(
                "Bluestein multiply pass does not contain a frequency kernel",
            )),
        }
    }

    /// Return a normalized inverse Stockham clone that consumes the forward spectrum
    /// and the immutable Bluestein kernel spectrum in one dispatch.
    pub(crate) fn fused_inverse_stockham_kernel(&self) -> Result<Option<KernelIr>> {
        let RecursiveFftNodeIr::Stockham(kernel) = &self.inverse_fft.root else {
            return Ok(None);
        };
        if kernel.io_mapping != StockhamIoMapping::Contiguous {
            return Ok(None);
        }
        Ok(Some(
            kernel.as_ref().clone().with_lookup_table_input_multiply()?,
        ))
    }

    /// Whether a recursive Cooley inverse consumes the Bluestein spectrum LUT at
    /// its root pack, removing the standalone frequency-domain multiply dispatch.
    pub(crate) fn has_fused_recursive_inverse(&self) -> bool {
        if self.inverse_fft.four_step_plan.is_some()
            || self.inverse_fft.rader_forced_upload_schedule.is_some()
        {
            return false;
        }
        let RecursiveFftNodeIr::CooleyTukey(root) = &self.inverse_fft.root else {
            return false;
        };
        root.pack_right.input_modifier == CooleyTukeyInputModifier::MultiplyLookupTable
    }

    pub fn validate(&self) -> Result<()> {
        self.preprocess.validate()?;
        self.multiply.validate()?;
        self.postprocess.validate()?;
        if self.grouped_batch == 0
            || self.preprocess.grouped_batch != self.grouped_batch
            || self.multiply.grouped_batch != self.grouped_batch
            || self.postprocess.grouped_batch != self.grouped_batch
        {
            return Err(VkFftError::InvalidKernelIr(
                "Bluestein groupedBatch ownership is inconsistent",
            ));
        }
        if self.preprocess.axis_batch_block != self.multiply.axis_batch_block
            || self.preprocess.axis_batch_block != self.postprocess.axis_batch_block
        {
            return Err(VkFftError::InvalidKernelIr(
                "Bluestein wrapper passes disagree on physical X/Y ownership",
            ));
        }
        if self.preprocess.scalar != self.scalar
            || self.multiply.scalar != self.scalar
            || self.postprocess.scalar != self.scalar
        {
            return Err(VkFftError::InvalidKernelIr(
                "Bluestein elementwise pass compute scalar does not match pipeline",
            ));
        }
        let expected_pre_input = if self
            .zero_pad_pass
            .as_ref()
            .is_some_and(|pass| pass.operation.is_input_boundary())
        {
            self.scalar
        } else {
            self.external_scalar
        };
        let expected_post_output = if self
            .zero_pad_pass
            .as_ref()
            .is_some_and(|pass| pass.operation.is_output_boundary())
        {
            self.scalar
        } else {
            self.external_scalar
        };
        if self.preprocess.input_storage_scalar != expected_pre_input
            || self.preprocess.output_storage_scalar != self.scalar
            || self.multiply.input_storage_scalar != self.scalar
            || self.multiply.output_storage_scalar != self.scalar
            || self.postprocess.input_storage_scalar != self.scalar
            || self.postprocess.output_storage_scalar != expected_post_output
        {
            return Err(VkFftError::InvalidKernelIr(
                "Bluestein caller/internal storage boundaries are inconsistent",
            ));
        }
        let real_boundary_pair = match (
            self.preprocess.input_modifier,
            self.postprocess.output_modifier,
        ) {
            (BluesteinInputModifier::None, BluesteinOutputModifier::None) => false,
            (
                BluesteinInputModifier::RealEvenPack(input),
                BluesteinOutputModifier::RealEvenPostprocess(output),
            ) if input.full_len == output.full_len && self.direction == Direction::Forward => true,
            (
                BluesteinInputModifier::RealEvenInversePreprocess(input),
                BluesteinOutputModifier::RealEvenUnpack(output),
            ) if input.full_len == output.full_len && self.direction == Direction::Inverse => true,
            _ => {
                return Err(VkFftError::InvalidKernelIr(
                    "Bluestein even-real input/output modifiers must form a direction-matched pair",
                ));
            }
        };
        if real_boundary_pair
            && (self.external_scalar != self.scalar || self.zero_pad_pass.is_some())
        {
            return Err(VkFftError::InvalidKernelIr(
                "initial Bluestein even-real fusion requires compute-width boundaries and no child-local zero padding",
            ));
        }
        if let Some(pass) = &self.zero_pad_pass {
            pass.validate()?;
            if pass.logical_len != self.logical_len
                || pass.batch_count != self.batch_count
                || pass.direction != self.direction
                || pass.scalar != self.scalar
                || pass.grouped_batch != self.grouped_batch
            {
                return Err(VkFftError::InvalidKernelIr(
                    "Bluestein zero-padding boundary does not match pipeline metadata",
                ));
            }
            let (expected_input, expected_output) = if pass.operation.is_input_boundary() {
                (self.external_scalar, self.scalar)
            } else {
                (self.scalar, self.external_scalar)
            };
            if pass.input_storage_scalar != expected_input
                || pass.output_storage_scalar != expected_output
            {
                return Err(VkFftError::InvalidKernelIr(
                    "Bluestein zero-padding storage boundary is inconsistent",
                ));
            }
        }
        if self.table.length != self.logical_len
            || self.table.convolution_len != self.convolution_len
            || self.table.direction != self.direction
        {
            return Err(VkFftError::InvalidKernelIr(
                "Bluestein table metadata does not match pipeline metadata",
            ));
        }
        for fft in [&*self.forward_fft, &*self.inverse_fft] {
            fft.validate()?;
            if fft.logical_len != self.convolution_len
                || fft.batch_count != self.batch_count
                || fft.scalar != self.scalar
                || (self.grouped_batch > 1
                    && fft.axis0_grouped_batch_override != Some(self.grouped_batch))
            {
                return Err(VkFftError::InvalidKernelIr(
                    "Bluestein internal FFT metadata does not match padded convolution",
                ));
            }
        }
        if self.forward_fft.direction != Direction::Forward
            || self.inverse_fft.direction != Direction::Inverse
        {
            return Err(VkFftError::InvalidKernelIr(
                "Bluestein internal FFT directions are invalid",
            ));
        }
        Ok(())
    }
}

/// Execute the exact multi-pass Bluestein IR on the CPU. This validates the
/// pass composition independently from the general CPU Bluestein reference.
pub fn execute_bluestein_ir(
    pipeline: &BluesteinPipelineIr,
    input: &[Complex64],
) -> Result<Vec<Complex64>> {
    pipeline.validate()?;
    let input_len = match pipeline.preprocess.input_modifier {
        BluesteinInputModifier::None => pipeline.logical_len,
        BluesteinInputModifier::RealEvenPack(mapping) => mapping.full_len,
        BluesteinInputModifier::RealEvenInversePreprocess(mapping) => mapping.compact_len(),
    };
    let expected =
        input_len
            .checked_mul(pipeline.batch_count)
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "Bluestein input element count",
            })?;
    if input.len() != expected {
        return Err(VkFftError::InputLengthMismatch {
            expected,
            actual: input.len(),
        });
    }
    let padded_count = pipeline
        .convolution_len
        .checked_mul(pipeline.batch_count)
        .ok_or(VkFftError::ArithmeticOverflow {
            operation: "Bluestein padded element count",
        })?;
    let mut padded = vec![Complex64::new(0.0, 0.0); padded_count];
    let prepared;
    let effective_input = if let Some(pass) = &pipeline.zero_pad_pass
        && pass.operation.is_input_boundary()
    {
        prepared = execute_zero_pad_pass(pass, input)?;
        prepared.as_slice()
    } else {
        input
    };
    for batch in 0..pipeline.batch_count {
        for index in 0..pipeline.logical_len {
            let source = match pipeline.preprocess.input_modifier {
                BluesteinInputModifier::None => {
                    effective_input[batch * pipeline.logical_len + index]
                }
                BluesteinInputModifier::RealEvenPack(mapping) => {
                    let base = batch * mapping.full_len;
                    Complex64::new(
                        effective_input[base + 2 * index].re,
                        effective_input[base + 2 * index + 1].re,
                    )
                }
                BluesteinInputModifier::RealEvenInversePreprocess(mapping) => {
                    let base = batch * mapping.compact_len();
                    let x = effective_input[base + index];
                    let mirrored = effective_input[base + (pipeline.logical_len - index)].conj();
                    let w_conj = Complex64::exp_i(
                        std::f64::consts::TAU * index as f64 / mapping.full_len as f64,
                    );
                    let rotated = w_conj * (x - mirrored);
                    let i_rotated = Complex64::new(-rotated.im, rotated.re);
                    let reconstruction_scale = if mapping.normalize { 0.5 } else { 1.0 };
                    (x + mirrored + i_rotated).scale(reconstruction_scale)
                }
            };
            padded[batch * pipeline.convolution_len + index] = source * pipeline.table.chirp[index];
        }
    }

    let mut spectrum = execute_recursive_fft_ir(&pipeline.forward_fft, &padded)?;
    let kernel_spectrum = pipeline.kernel_spectrum()?;
    let convolution = if let Some(fused_inverse) = pipeline.fused_inverse_stockham_kernel()? {
        execute_stockham_ir_with_lookup(&fused_inverse, &spectrum, Some(kernel_spectrum))?
    } else if pipeline.has_fused_recursive_inverse() {
        execute_recursive_fft_ir_with_resources(
            &pipeline.inverse_fft,
            &spectrum,
            Some(kernel_spectrum),
            None,
        )?
    } else {
        for batch in 0..pipeline.batch_count {
            for index in 0..pipeline.convolution_len {
                spectrum[batch * pipeline.convolution_len + index] *= kernel_spectrum[index];
            }
        }
        execute_recursive_fft_ir(&pipeline.inverse_fft, &spectrum)?
    };
    let normalize = match pipeline.postprocess.operation {
        BluesteinPassOperation::Postprocess { normalize } => normalize,
        _ => {
            return Err(VkFftError::InvalidKernelIr(
                "Bluestein postprocess pass has the wrong operation",
            ));
        }
    };
    let scale = if normalize {
        1.0 / pipeline.logical_len as f64
    } else {
        1.0
    };
    let transformed = |batch: usize, index: usize| {
        (convolution[batch * pipeline.convolution_len + index] * pipeline.table.chirp[index])
            .scale(scale)
    };
    let mut output = match pipeline.postprocess.output_modifier {
        BluesteinOutputModifier::None => {
            let mut values = Vec::with_capacity(pipeline.logical_len * pipeline.batch_count);
            for batch in 0..pipeline.batch_count {
                for index in 0..pipeline.logical_len {
                    values.push(transformed(batch, index));
                }
            }
            values
        }
        BluesteinOutputModifier::RealEvenPostprocess(mapping) => {
            let compact_len = mapping.full_len / 2 + 1;
            let mut values = vec![Complex64::default(); compact_len * pipeline.batch_count];
            for batch in 0..pipeline.batch_count {
                let base = batch * compact_len;
                let z0 = transformed(batch, 0);
                values[base] = Complex64::new(z0.re + z0.im, 0.0);
                values[base + pipeline.logical_len] = Complex64::new(z0.re - z0.im, 0.0);
                for k in 1..pipeline.logical_len {
                    let a = transformed(batch, k);
                    let b = transformed(batch, pipeline.logical_len - k).conj();
                    let w = Complex64::exp_i(
                        -std::f64::consts::TAU * k as f64 / mapping.full_len as f64,
                    );
                    let rotated = w * (a - b);
                    values[base + k] = Complex64::new(
                        0.5 * (a.re + b.re + rotated.im),
                        0.5 * (a.im + b.im - rotated.re),
                    );
                }
            }
            values
        }
        BluesteinOutputModifier::RealEvenUnpack(mapping) => {
            let mut values = vec![Complex64::default(); mapping.full_len * pipeline.batch_count];
            for batch in 0..pipeline.batch_count {
                let base = batch * mapping.full_len;
                for index in 0..pipeline.logical_len {
                    let value = transformed(batch, index);
                    values[base + 2 * index] = Complex64::new(value.re, 0.0);
                    values[base + 2 * index + 1] = Complex64::new(value.im, 0.0);
                }
            }
            values
        }
    };
    if let Some(pass) = &pipeline.zero_pad_pass
        && pass.operation.is_output_boundary()
    {
        output = execute_zero_pad_pass(pass, &output)?;
    }
    Ok(output)
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
    use crate::config::{Backend, GpuVendor};
    use crate::kernel_ir::BufferRole;
    use crate::reference::dft;

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
                Complex64::new((0.23 * x).sin() + x * 0.002, (0.07 * x).cos() - x * 0.003)
            })
            .collect()
    }

    fn max_error(lhs: &[Complex64], rhs: &[Complex64]) -> f64 {
        lhs.iter()
            .zip(rhs)
            .map(|(lhs, rhs)| (*lhs - *rhs).norm_sqr().sqrt())
            .fold(0.0, f64::max)
    }

    #[test]
    fn bluestein_pipeline_composes_exact_stockham_ir() {
        let length = 103usize;
        let batch_count = 2usize;
        let mut bluestein_tuning = crate::PlannerTuning::portable();
        bluestein_tuning.max_rader_fft_prime = 100;
        for direction in [Direction::Forward, Direction::Inverse] {
            let config = FftConfig::new(vec![length])
                .with_batch_count(batch_count)
                .with_inverse_normalization(direction == Direction::Inverse)
                .with_tuning(bluestein_tuning);
            let plan = FftPlan::build(config).unwrap();
            assert!(matches!(
                plan.axes[0].algorithm,
                AxisAlgorithm::Bluestein { .. }
            ));
            let pipeline = BluesteinPipelineIr::build(&plan, direction, device()).unwrap();
            assert!(pipeline.convolution_len >= 2 * length - 1);
            let input = sample(length, batch_count);
            let actual = execute_bluestein_ir(&pipeline, &input).unwrap();
            let mut expected = Vec::with_capacity(actual.len());
            for batch in 0..batch_count {
                let start = batch * length;
                expected.extend(dft(
                    &input[start..start + length],
                    direction,
                    direction == Direction::Inverse,
                ));
            }
            assert!(max_error(&actual, &expected) < 2.0e-9 * length as f64);
        }
    }

    #[test]
    fn grouped_bluestein_owns_parent_children_and_zero_pad_tail() {
        let length = 103usize;
        let batch_count = 7usize;
        let grouped_batch = 3usize;
        let range = crate::ZeroPaddingRange {
            left: 19,
            right: 31,
        };
        let mut tuning = crate::PlannerTuning::portable();
        tuning.max_rader_fft_prime = 100;

        for direction in [Direction::Forward, Direction::Inverse] {
            let config = FftConfig::new(vec![length])
                .with_batch_count(batch_count)
                .with_grouped_batch(0, grouped_batch)
                .unwrap()
                .with_precision(Precision::F16StorageF32Compute)
                .with_inverse_normalization(direction == Direction::Inverse)
                .with_tuning(tuning)
                .with_zero_padding(0, range.left, range.right)
                .unwrap();
            let plan = FftPlan::build(config).unwrap();
            assert!(matches!(
                plan.axes[0].algorithm,
                AxisAlgorithm::Bluestein { .. }
            ));
            let pipeline = BluesteinPipelineIr::build(&plan, direction, device()).unwrap();
            assert_eq!(pipeline.grouped_batch, grouped_batch);
            assert_eq!(pipeline.preprocess.dispatch.x, 3);
            assert_eq!(pipeline.multiply.dispatch.x, 3);
            assert_eq!(pipeline.postprocess.dispatch.x, 3);
            assert_eq!(pipeline.preprocess.grouped_batch, grouped_batch);
            assert_eq!(pipeline.multiply.grouped_batch, grouped_batch);
            assert_eq!(pipeline.postprocess.grouped_batch, grouped_batch);
            assert_eq!(
                pipeline.forward_fft.axis0_grouped_batch_override,
                Some(grouped_batch)
            );
            assert_eq!(
                pipeline.inverse_fft.axis0_grouped_batch_override,
                Some(grouped_batch)
            );
            let zero = pipeline.zero_pad_pass.as_ref().unwrap();
            assert_eq!(zero.grouped_batch, grouped_batch);
            assert_eq!(zero.dispatch.x, 3);

            let input = sample(length, batch_count);
            let actual = execute_bluestein_ir(&pipeline, &input).unwrap();
            let mut expected = Vec::with_capacity(actual.len());
            for batch in 0..batch_count {
                let start = batch * length;
                let mut values = input[start..start + length].to_vec();
                match direction {
                    Direction::Forward => {
                        values[range.left..range.right].fill(Complex64::default());
                        expected.extend(dft(&values, Direction::Forward, false));
                    }
                    Direction::Inverse => {
                        let mut transformed = dft(&values, Direction::Inverse, true);
                        transformed[range.left..range.right].fill(Complex64::default());
                        expected.extend(transformed);
                    }
                }
            }
            assert!(max_error(&actual, &expected) < 3.0e-8 * length as f64);
            pipeline.validate().unwrap();
        }
    }

    #[test]
    fn mixed_storage_bluestein_keeps_convolution_resources_at_compute_precision() {
        let length = 103usize;
        let mut profile = device();
        profile.supports_f64 = true;
        let mut bluestein_tuning = crate::PlannerTuning::portable();
        bluestein_tuning.max_rader_fft_prime = 100;

        for (precision, compute, storage) in [
            (
                Precision::F16StorageF32Compute,
                ScalarType::F32,
                ScalarType::F16,
            ),
            (
                Precision::F64ComputeF32Storage,
                ScalarType::F64,
                ScalarType::F32,
            ),
        ] {
            for direction in [Direction::Forward, Direction::Inverse] {
                for zero_padded in [false, true] {
                    let mut config = FftConfig::new(vec![length])
                        .with_precision(precision)
                        .with_inverse_normalization(direction == Direction::Inverse)
                        .with_tuning(bluestein_tuning);
                    if zero_padded {
                        config = config.with_zero_padding(0, length / 2, length).unwrap();
                    }
                    let plan = FftPlan::build(config).unwrap();
                    assert!(matches!(
                        plan.axes[0].algorithm,
                        AxisAlgorithm::Bluestein { .. }
                    ));
                    let pipeline = BluesteinPipelineIr::build(&plan, direction, profile).unwrap();
                    assert_eq!(pipeline.scalar, compute);
                    assert_eq!(pipeline.external_storage_scalar(), storage);
                    assert_eq!(pipeline.forward_fft.scalar, compute);
                    assert_eq!(pipeline.forward_fft.external_storage_scalar(), compute);
                    assert_eq!(pipeline.inverse_fft.scalar, compute);
                    assert_eq!(pipeline.inverse_fft.external_storage_scalar(), compute);
                    assert_eq!(pipeline.multiply.input_storage_scalar, compute);
                    assert_eq!(pipeline.multiply.output_storage_scalar, compute);

                    if zero_padded {
                        let zero = pipeline.zero_pad_pass.as_ref().unwrap();
                        match direction {
                            Direction::Forward => {
                                assert_eq!(zero.input_storage_scalar, storage);
                                assert_eq!(zero.output_storage_scalar, compute);
                                assert_eq!(pipeline.preprocess.input_storage_scalar, compute);
                                assert_eq!(pipeline.postprocess.output_storage_scalar, storage);
                            }
                            Direction::Inverse => {
                                assert_eq!(pipeline.preprocess.input_storage_scalar, storage);
                                assert_eq!(pipeline.postprocess.output_storage_scalar, compute);
                                assert_eq!(zero.input_storage_scalar, compute);
                                assert_eq!(zero.output_storage_scalar, storage);
                            }
                        }
                    } else {
                        assert_eq!(pipeline.preprocess.input_storage_scalar, storage);
                        assert_eq!(pipeline.postprocess.output_storage_scalar, storage);
                    }

                    let program = crate::ProgramIr::bluestein(&pipeline).unwrap();
                    assert_eq!(program.input_resource().unwrap().scalar, storage);
                    assert_eq!(program.output_resource().unwrap().scalar, storage);
                    assert!(program.resources.iter().all(|resource| {
                        matches!(
                            resource.kind,
                            crate::ProgramResourceKind::Input | crate::ProgramResourceKind::Output
                        ) || resource.scalar == compute
                    }));
                }
            }
        }
    }

    #[test]
    fn recursive_bluestein_inverse_fuses_spectrum_multiply_into_root_pack() {
        let length = 103usize;
        let profile = DeviceProfile {
            shared_memory_bytes: 128,
            shared_memory_pow2_bytes: 128,
            ..DeviceProfile::generic(Backend::Vulkan, GpuVendor::Other(0xB1E5))
        };
        let mut bluestein_tuning = crate::PlannerTuning::portable();
        bluestein_tuning.max_rader_fft_prime = 100;
        let plan =
            FftPlan::build(FftConfig::new(vec![length]).with_tuning(bluestein_tuning)).unwrap();
        let pipeline = BluesteinPipelineIr::build(&plan, Direction::Forward, profile).unwrap();
        assert!(pipeline.convolution_len > 128);
        assert!(matches!(
            pipeline.forward_fft.root,
            crate::RecursiveFftNodeIr::CooleyTukey(_)
        ));
        assert!(pipeline.has_fused_recursive_inverse());
        let crate::RecursiveFftNodeIr::CooleyTukey(inverse_root) = &pipeline.inverse_fft.root
        else {
            panic!("large Bluestein inverse must keep a recursive Cooley root");
        };
        assert_eq!(
            inverse_root.pack_right.input_modifier,
            CooleyTukeyInputModifier::MultiplyLookupTable
        );
        let program = crate::ProgramIr::bluestein(&pipeline).unwrap();
        assert!(
            !program
                .passes
                .iter()
                .any(|pass| pass.name == pipeline.multiply.name)
        );
        let pack_index = program
            .passes
            .iter()
            .position(|pass| pass.name == inverse_root.pack_right.name)
            .expect("recursive Bluestein inverse root pack must be materialized");
        assert!(
            program.passes[pack_index]
                .bindings
                .iter()
                .any(|binding| binding.role == BufferRole::LookupTable)
        );
        let one_dim = crate::OneDimFftIr::Bluestein(Box::new(pipeline.clone()));
        let shaders = crate::backend::vulkan::VulkanGlslBackend
            .lower_one_dim_fft(&one_dim)
            .unwrap();
        assert_eq!(shaders.len(), program.passes.len());
        assert!(
            shaders[pack_index]
                .descriptors
                .iter()
                .any(|descriptor| descriptor.role == BufferRole::LookupTable)
        );
        for shader in shaders {
            assert_eq!(shader.compile_spirv().unwrap().words[0], 0x0723_0203);
        }
        let input = sample(length, 1);
        let actual = execute_bluestein_ir(&pipeline, &input).unwrap();
        let expected = crate::reference::fft(&input, Direction::Forward, false).unwrap();
        assert!(max_error(&actual, &expected) < 4.0e-8 * length as f64);
    }

    #[test]
    fn bluestein_pipeline_forces_boost_one_in_power_of_two_convolution_child() {
        let mut profile = device();
        profile.shared_memory_bytes = 48 * 1024;
        profile.shared_memory_pow2_bytes = 32 * 1024;
        profile.max_threads_per_block = 1024;
        profile.max_workgroup_size = [1024, 1024, 64];
        let mut tuning = crate::PlannerTuning::portable();
        tuning.max_rader_fft_prime = 100;

        // 2*4051-1=8101 lies just above the last smaller 2/3/5/7-smooth
        // candidate (8100), so the portable Bluestein convolution is exactly 8192.
        let plan = FftPlan::build(FftConfig::new(vec![4051]).with_tuning(tuning)).unwrap();
        assert!(matches!(
            plan.axes[0].algorithm,
            AxisAlgorithm::Bluestein { .. }
        ));
        let pipeline = BluesteinPipelineIr::build(&plan, Direction::Forward, profile).unwrap();
        assert_eq!(pipeline.convolution_len, 8192);
        for fft in [&pipeline.forward_fft, &pipeline.inverse_fft] {
            let schedule = fft
                .stockham_upload_schedule
                .as_ref()
                .expect("8192-point Bluestein convolution should retain Stockham upload metadata");
            assert_eq!(schedule.register_boost, 1);
            assert_eq!(schedule.upload_count, 2);
            assert_eq!(schedule.axis_split, vec![128, 64]);
            assert!(fft.four_step_plan.is_some());
        }
        assert!(!pipeline.has_fused_recursive_inverse());
        let program = crate::ProgramIr::bluestein(&pipeline).unwrap();
        assert!(
            program
                .passes
                .iter()
                .any(|pass| pass.name == pipeline.multiply.name)
        );
        pipeline.validate().unwrap();
    }

    #[test]
    fn device_scored_strided_bluestein_pipeline_uses_fixed_padding() {
        let mut profile = device();
        profile.shared_memory_bytes = 48 * 1024;
        profile.shared_memory_pow2_bytes = 32 * 1024;
        profile.max_threads_per_block = 1024;
        profile.max_workgroup_size = [1024, 1024, 64];

        let plan = FftPlan::build_c2c_child_for_device(
            FftConfig::new(vec![4001]),
            profile,
            C2cDeviceAxisClass::Strided,
        )
        .unwrap();
        let AxisAlgorithm::Bluestein {
            convolution_len, ..
        } = plan.axes[0].algorithm
        else {
            panic!("device-scored strided p4001 should use Bluestein");
        };
        assert_eq!(convolution_len, 8192);

        let pipeline = BluesteinPipelineIr::build(&plan, Direction::Forward, profile).unwrap();
        assert_eq!(pipeline.convolution_len, 8192);
        for fft in [&pipeline.forward_fft, &pipeline.inverse_fft] {
            let schedule = fft.stockham_upload_schedule.as_ref().unwrap();
            assert_eq!(schedule.register_boost, 1);
            assert_eq!(schedule.upload_count, 2);
            assert_eq!(schedule.axis_split, vec![128, 64]);
            let shaders = crate::backend::vulkan::VulkanGlslBackend
                .lower_recursive_fft(fft)
                .unwrap();
            assert_eq!(shaders.len(), 2);
            for shader in shaders {
                assert_eq!(shader.compile_spirv().unwrap().words[0], 0x0723_0203);
            }
        }
        pipeline.validate().unwrap();
    }

    #[test]
    fn grouped_large_bluestein_propagates_into_recursive_convolution() {
        let length = 2053usize;
        let batch_count = 5usize;
        let grouped_batch = 3usize;
        let mut limited = device();
        limited.shared_memory_bytes = 16 * 128;
        limited.shared_memory_pow2_bytes = limited.shared_memory_bytes;
        let mut tuning = crate::PlannerTuning::portable();
        tuning.max_rader_fft_prime = 2000;
        let plan = FftPlan::build(
            FftConfig::new(vec![length])
                .with_batch_count(batch_count)
                .with_grouped_batch(0, grouped_batch)
                .unwrap()
                .with_tuning(tuning),
        )
        .unwrap();
        let pipeline = BluesteinPipelineIr::build(&plan, Direction::Forward, limited).unwrap();
        assert_eq!(pipeline.grouped_batch, grouped_batch);
        assert!(pipeline.convolution_len > 128);
        assert!(matches!(
            pipeline.forward_fft.root,
            crate::RecursiveFftNodeIr::CooleyTukey(_)
        ));
        assert_eq!(
            pipeline.forward_fft.axis0_grouped_batch_override,
            Some(grouped_batch)
        );
        assert_eq!(
            pipeline.inverse_fft.axis0_grouped_batch_override,
            Some(grouped_batch)
        );
        assert!(pipeline.forward_fft.four_step_plan.is_some());
        assert!(pipeline.inverse_fft.four_step_plan.is_some());
        let forward_uploads = pipeline
            .forward_fft
            .four_step_stockham_upload_kernels()
            .unwrap()
            .unwrap();
        let inverse_uploads = pipeline
            .inverse_fft
            .four_step_stockham_upload_kernels()
            .unwrap()
            .unwrap();
        let mut expected_dispatches = vec![2u32];
        expected_dispatches.extend(forward_uploads.iter().map(|kernel| kernel.dispatch.x));
        expected_dispatches.push(2);
        expected_dispatches.extend(inverse_uploads.iter().map(|kernel| kernel.dispatch.x));
        expected_dispatches.push(2);
        let program = crate::ProgramIr::bluestein(&pipeline).unwrap();
        assert_eq!(
            program
                .passes
                .iter()
                .map(|pass| pass.dispatch.x)
                .collect::<Vec<_>>(),
            expected_dispatches
        );
        let input = sample(length, batch_count);
        let actual = execute_bluestein_ir(&pipeline, &input).unwrap();
        let mut expected = Vec::with_capacity(input.len());
        for batch in 0..batch_count {
            let start = batch * length;
            expected.extend(
                crate::reference::fft(&input[start..start + length], Direction::Forward, false)
                    .unwrap(),
            );
        }
        assert!(max_error(&actual, &expected) < 4.0e-8 * length as f64);
        pipeline.validate().unwrap();
    }

    #[test]
    fn bluestein_pipeline_rejects_rader_axis() {
        let plan = FftPlan::build(FftConfig::new(vec![17])).unwrap();
        let error = BluesteinPipelineIr::build(&plan, Direction::Forward, device()).unwrap_err();
        assert!(matches!(error, VkFftError::UnsupportedKernelPath(_)));
    }
}
