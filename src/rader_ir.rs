//! Backend-neutral correctness-first direct-multiplication Rader IR.
//!
//! This initial slice handles a one-dimensional prime C2C transform whose planner
//! selects `RaderMode::DirectMultiplication`. It deliberately keeps the O(p^2)
//! direct Rader sum explicit before FFT-convolution Rader and mixed composite axes
//! are introduced.

use crate::complex::Complex64;
use crate::config::{
    Backend, DeviceProfile, Direction, FftConfig, GpuVendor, Precision, TransformKind,
};
use crate::error::{Result, VkFftError};
use crate::kernel_ir::{
    CooleyRightInputMapping, DispatchGeometry, KernelIr, RaderFourStepInputMapping,
    RaderGeneratorCooleyRightMapping, RaderGeneratorFourStepMapping, RaderGeneratorMapping,
    RaderScatterMapping, ScalarType, StockhamIoMapping, WorkgroupSize, execute_stockham_ir,
    execute_stockham_ir_with_lookup, execute_stockham_ir_with_resources,
};
use crate::lut::RaderTable;
use crate::one_dim_ir::{OneDimFftIr, execute_one_dim_fft_ir};
use crate::planner::{AxisAlgorithm, C2cDeviceAxisClass, FftPlan, RaderMode};
use crate::recursive_ir::{
    CooleyTukeyInputModifier, CooleyTukeyOutputModifier, RecursiveFftIr, RecursiveFftNodeIr,
    execute_recursive_fft_ir_with_resources,
};
use crate::scheduler::{
    FourStepAxisBlockRequest, RaderFftRegisterSchedule, StockhamAxisBlockSchedule,
    has_specialized_gpu_scheduler_policy, plan_gpu_axis0_direct_rader_batch_block,
    plan_gpu_axis0_fft_rader_batch_block_with_grouped_batch,
    plan_gpu_axis0_four_step_default_block_from_shape_for_precision,
    plan_gpu_rader_fft_registers_for_containers,
};

#[derive(Debug, Clone, PartialEq)]
pub struct RaderDirectIr {
    pub name: String,
    pub scalar: ScalarType,
    /// Caller-visible input storage scalar. Direct Rader is one kernel, so the
    /// conversion boundary is represented directly on binding 0.
    pub input_storage_scalar: ScalarType,
    /// Caller-visible output storage scalar on binding 1.
    pub output_storage_scalar: ScalarType,
    pub direction: Direction,
    pub prime: usize,
    pub batch_count: usize,
    pub axis_batch_block: Option<StockhamAxisBlockSchedule>,
    /// Physical caller-boundary mapping used when this direct-Rader transform is one
    /// component of a fused two/three-upload Four-step axis.
    pub io_mapping: StockhamIoMapping,
    pub workgroup_size: WorkgroupSize,
    pub dispatch: DispatchGeometry,
    pub normalize: bool,
    pub table: RaderTable,
}

impl RaderDirectIr {
    pub fn build(plan: &FftPlan, direction: Direction, device: DeviceProfile) -> Result<Self> {
        if plan.config.dimensions.len() != 1 {
            return Err(VkFftError::UnsupportedKernelPath(
                "initial direct Rader IR supports one-dimensional plans only",
            ));
        }
        if plan.config.transform != TransformKind::ComplexToComplex {
            return Err(VkFftError::UnsupportedKernelPath(
                "initial direct Rader IR supports C2C transforms only",
            ));
        }
        let scalar = match plan.config.precision {
            Precision::F32 => ScalarType::F32,
            Precision::F64 if device.supports_f64 => ScalarType::F64,
            other => {
                return Err(VkFftError::UnsupportedPrecision {
                    backend: "direct Rader IR",
                    precision: precision_name(other),
                });
            }
        };
        if device.max_threads_per_block == 0 {
            return Err(VkFftError::ResourceLimitExceeded {
                resource: "threads per workgroup",
                required: 1,
                available: 0,
            });
        }

        let axis = plan.axes.first().ok_or(VkFftError::InvalidKernelIr(
            "missing direct Rader axis plan",
        ))?;
        let AxisAlgorithm::Rader { stockham, primes } = &axis.algorithm else {
            return Err(VkFftError::UnsupportedKernelPath(
                "direct Rader IR requires a planner-selected Rader axis",
            ));
        };
        if !stockham.prime_factors.is_empty()
            || !stockham.merged_radices.is_empty()
            || primes.len() != 1
            || primes[0].multiplicity != 1
            || primes[0].prime != axis.effective_fft_len
        {
            return Err(VkFftError::UnsupportedKernelPath(
                "initial direct Rader IR supports a single prime axis without Stockham factors",
            ));
        }
        if !matches!(primes[0].mode, RaderMode::DirectMultiplication) {
            return Err(VkFftError::UnsupportedKernelPath(
                "planner selected FFT-convolution Rader; direct Rader IR is not applicable",
            ));
        }

        let prime = primes[0].prime;
        let table = RaderTable::from_prime_plan(&primes[0], direction)?;
        let grouped_batch_override = plan.config.grouped_batch_for_axis(0);
        let axis_batch_block = plan_gpu_axis0_direct_rader_batch_block(
            prime,
            plan.config.batch_count,
            scalar.complex_bytes(),
            plan.config.zero_padding_for_axis(0).is_some(),
            grouped_batch_override,
            device,
        )?;
        if grouped_batch_override.is_some() && axis_batch_block.is_none() {
            return Err(VkFftError::UnsupportedKernelPath(
                "groupedBatch override is not executable for this direct Rader axis",
            ));
        }
        let (workgroup_size, dispatch) = if let Some(block) = axis_batch_block {
            (
                WorkgroupSize {
                    x: u32::try_from(block.local_size_x).map_err(|_| {
                        VkFftError::ValueOutOfRange {
                            field: "direct Rader workgroup x size",
                        }
                    })?,
                    y: u32::try_from(block.local_size_y).map_err(|_| {
                        VkFftError::ValueOutOfRange {
                            field: "direct Rader workgroup y size",
                        }
                    })?,
                    z: 1,
                },
                DispatchGeometry {
                    x: u32::try_from(plan.config.batch_count.div_ceil(block.grouped_batch))
                        .map_err(|_| VkFftError::ValueOutOfRange {
                            field: "direct Rader dispatch workgroup count",
                        })?,
                    y: 1,
                    z: 1,
                },
            )
        } else {
            let local_size = prime.min(device.max_threads_per_block).max(1);
            (
                WorkgroupSize {
                    x: u32::try_from(local_size).map_err(|_| VkFftError::ValueOutOfRange {
                        field: "direct Rader workgroup size",
                    })?,
                    y: 1,
                    z: 1,
                },
                DispatchGeometry {
                    x: u32::try_from(plan.config.batch_count).map_err(|_| {
                        VkFftError::ValueOutOfRange {
                            field: "direct Rader dispatch workgroup count",
                        }
                    })?,
                    y: 1,
                    z: 1,
                },
            )
        };
        let direction_name = match direction {
            Direction::Forward => "forward",
            Direction::Inverse => "inverse",
        };
        let ir = Self {
            name: format!("vkfft_rader_direct_{prime}_{direction_name}"),
            scalar,
            input_storage_scalar: scalar,
            output_storage_scalar: scalar,
            direction,
            prime,
            batch_count: plan.config.batch_count,
            axis_batch_block,
            io_mapping: StockhamIoMapping::Contiguous,
            workgroup_size,
            dispatch,
            normalize: direction == Direction::Inverse && plan.config.normalize_inverse,
            table,
        };
        ir.validate()?;
        Ok(ir)
    }

    pub(crate) fn with_external_input_storage_scalar(
        mut self,
        storage: ScalarType,
    ) -> Result<Self> {
        if storage != self.scalar
            && !matches!(
                (self.scalar, storage),
                (ScalarType::F32, ScalarType::F16) | (ScalarType::F64, ScalarType::F32)
            )
        {
            return Err(VkFftError::UnsupportedPrecision {
                backend: "direct Rader input-storage boundary",
                precision: "unsupported compute/storage scalar pair",
            });
        }
        self.input_storage_scalar = storage;
        self.validate()?;
        Ok(self)
    }

    pub(crate) fn with_external_output_storage_scalar(
        mut self,
        storage: ScalarType,
    ) -> Result<Self> {
        if storage != self.scalar
            && !matches!(
                (self.scalar, storage),
                (ScalarType::F32, ScalarType::F16) | (ScalarType::F64, ScalarType::F32)
            )
        {
            return Err(VkFftError::UnsupportedPrecision {
                backend: "direct Rader output-storage boundary",
                precision: "unsupported compute/storage scalar pair",
            });
        }
        self.output_storage_scalar = storage;
        self.validate()?;
        Ok(self)
    }

    pub(crate) fn with_stockham_io_mapping(mut self, mapping: StockhamIoMapping) -> Result<Self> {
        if !matches!(
            mapping,
            StockhamIoMapping::Contiguous
                | StockhamIoMapping::FourStepRight(_)
                | StockhamIoMapping::FourStepLeft(_)
                | StockhamIoMapping::FourStepThreeUpload2(_)
                | StockhamIoMapping::FourStepThreeUpload1(_)
                | StockhamIoMapping::FourStepThreeUpload0(_)
        ) {
            return Err(VkFftError::UnsupportedKernelPath(
                "direct Rader only accepts contiguous or Four-step caller mappings",
            ));
        }
        mapping.validate_kernel(self.prime, self.batch_count)?;
        self.io_mapping = mapping;
        self.validate()?;
        Ok(self)
    }

    pub(crate) fn with_axis0_batch_block(
        mut self,
        block: StockhamAxisBlockSchedule,
        device: DeviceProfile,
    ) -> Result<Self> {
        block.validate(self.batch_count, device)?;
        let expected_threads = self
            .axis_batch_block
            .map_or(self.prime.div_ceil(2), |current| {
                current.threads_per_transform
            });
        if block.threads_per_transform != expected_threads {
            return Err(VkFftError::UnsupportedKernelPath(
                "direct Rader Four-step axis block changed the per-transform thread floor",
            ));
        }
        self.axis_batch_block = Some(block);
        self.workgroup_size = WorkgroupSize {
            x: u32::try_from(block.local_size_x).map_err(|_| VkFftError::ValueOutOfRange {
                field: "direct Rader remapped workgroup x size",
            })?,
            y: u32::try_from(block.local_size_y).map_err(|_| VkFftError::ValueOutOfRange {
                field: "direct Rader remapped workgroup y size",
            })?,
            z: 1,
        };
        self.dispatch = DispatchGeometry {
            x: u32::try_from(self.batch_count.div_ceil(block.grouped_batch)).map_err(|_| {
                VkFftError::ValueOutOfRange {
                    field: "direct Rader remapped dispatch workgroup count",
                }
            })?,
            y: 1,
            z: 1,
        };
        self.validate()?;
        Ok(self)
    }

    pub fn validate(&self) -> Result<()> {
        if self.scalar == ScalarType::F16 {
            return Err(VkFftError::InvalidKernelIr(
                "binary16 cannot be used as direct-Rader compute scalar",
            ));
        }
        let supported_storage = |storage: ScalarType| {
            storage == self.scalar
                || matches!(
                    (self.scalar, storage),
                    (ScalarType::F32, ScalarType::F16) | (ScalarType::F64, ScalarType::F32)
                )
        };
        if !supported_storage(self.input_storage_scalar)
            || !supported_storage(self.output_storage_scalar)
        {
            return Err(VkFftError::InvalidKernelIr(
                "direct Rader external storage scalar is inconsistent with compute precision",
            ));
        }
        if self.prime < 2 || self.batch_count == 0 {
            return Err(VkFftError::InvalidKernelIr(
                "direct Rader prime and batch count must be non-zero",
            ));
        }
        if !matches!(
            self.io_mapping,
            StockhamIoMapping::Contiguous
                | StockhamIoMapping::FourStepRight(_)
                | StockhamIoMapping::FourStepLeft(_)
                | StockhamIoMapping::FourStepThreeUpload2(_)
                | StockhamIoMapping::FourStepThreeUpload1(_)
                | StockhamIoMapping::FourStepThreeUpload0(_)
        ) {
            return Err(VkFftError::InvalidKernelIr(
                "direct Rader caller mapping is not a supported contiguous/Four-step form",
            ));
        }
        self.io_mapping
            .validate_kernel(self.prime, self.batch_count)?;
        if self.table.prime != self.prime || self.table.direction != self.direction {
            return Err(VkFftError::InvalidKernelIr(
                "direct Rader table metadata does not match IR metadata",
            ));
        }
        if self.table.permutation.len() != self.prime - 1
            || self.table.twiddles_by_generator_power.len() != self.prime - 1
        {
            return Err(VkFftError::InvalidKernelIr(
                "direct Rader table length does not match prime-1",
            ));
        }
        if self.workgroup_size.x == 0 || self.workgroup_size.y == 0 || self.workgroup_size.z == 0 {
            return Err(VkFftError::InvalidKernelIr(
                "direct Rader workgroup dimensions must be non-zero",
            ));
        }
        if let Some(block) = self.axis_batch_block {
            if block.grouped_batch == 0
                || block.grouped_batch > self.batch_count
                || [
                    self.workgroup_size.x as usize,
                    self.workgroup_size.y as usize,
                ] != [block.local_size_x, block.local_size_y]
                || self.dispatch.x as usize != self.batch_count.div_ceil(block.grouped_batch)
                || self.dispatch.y != 1
                || self.dispatch.z != 1
            {
                return Err(VkFftError::InvalidKernelIr(
                    "direct Rader axis-batch geometry is inconsistent",
                ));
            }
        } else if self.dispatch.x as usize != self.batch_count
            || self.dispatch.y != 1
            || self.dispatch.z != 1
        {
            return Err(VkFftError::InvalidKernelIr(
                "direct Rader fallback requires one workgroup per transform",
            ));
        }
        Ok(())
    }

    pub fn twiddle_lut(&self) -> &[Complex64] {
        &self.table.twiddles_by_generator_power
    }
}

pub fn execute_rader_direct_ir(ir: &RaderDirectIr, input: &[Complex64]) -> Result<Vec<Complex64>> {
    ir.validate()?;
    let expected = ir
        .prime
        .checked_mul(ir.batch_count)
        .ok_or(VkFftError::ArithmeticOverflow {
            operation: "direct Rader input element count",
        })?;
    if input.len() != expected {
        return Err(VkFftError::InputLengthMismatch {
            expected,
            actual: input.len(),
        });
    }

    let count = ir.prime - 1;
    let scale = if ir.normalize {
        1.0 / ir.prime as f64
    } else {
        1.0
    };
    let sign = match ir.direction {
        Direction::Forward => -1.0,
        Direction::Inverse => 1.0,
    };
    let mut output = vec![Complex64::new(0.0, 0.0); expected];
    for batch in 0..ir.batch_count {
        let read =
            |local_index: usize| input[ir.io_mapping.input_index(ir.prime, batch, local_index)];

        let dc = (0..ir.prime)
            .map(read)
            .fold(Complex64::new(0.0, 0.0), |sum, value| sum + value)
            .scale(scale);
        let (dc_index, dc_value) = ir.io_mapping.map_output(ir.prime, batch, 0, dc, sign);
        output[dc_index] = dc_value;

        for output_exponent in 0..count {
            let mut sum = read(0);
            for input_exponent in 0..count {
                let input_index = ir.table.permutation[input_exponent];
                let twiddle_index = (input_exponent + output_exponent) % count;
                sum += read(input_index) * ir.table.twiddles_by_generator_power[twiddle_index];
            }
            let local_output_index = ir.table.permutation[output_exponent];
            let (output_index, value) = ir.io_mapping.map_output(
                ir.prime,
                batch,
                local_output_index,
                sum.scale(scale),
                sign,
            );
            output[output_index] = value;
        }
    }
    Ok(output)
}

#[derive(Debug, Clone, PartialEq)]
pub enum RaderFftPassOperation {
    GatherReverse,
    MultiplyKernelSpectrum { spectrum: Vec<Complex64> },
    Scatter { normalize: bool },
}

#[derive(Debug, Clone, PartialEq)]
pub struct RaderFftPassIr {
    pub name: String,
    pub scalar: ScalarType,
    pub input_storage_scalar: ScalarType,
    pub output_storage_scalar: ScalarType,
    pub auxiliary_storage_scalar: ScalarType,
    pub direction: Direction,
    pub prime: usize,
    pub convolution_len: usize,
    pub batch_count: usize,
    pub workgroup_size: WorkgroupSize,
    pub dispatch: DispatchGeometry,
    pub axis_batch_block: Option<StockhamAxisBlockSchedule>,
    pub operation: RaderFftPassOperation,
}

#[derive(Debug, Clone, Copy)]
struct RaderFftPassShape {
    scalar: ScalarType,
    direction: Direction,
    prime: usize,
    convolution_len: usize,
    batch_count: usize,
    device: DeviceProfile,
}

impl RaderFftPassIr {
    fn new(
        name: String,
        shape: RaderFftPassShape,
        operation: RaderFftPassOperation,
    ) -> Result<Self> {
        if shape.device.max_threads_per_block == 0 {
            return Err(VkFftError::ResourceLimitExceeded {
                resource: "threads per workgroup",
                required: 1,
                available: 0,
            });
        }
        let logical_work = match operation {
            RaderFftPassOperation::GatherReverse
            | RaderFftPassOperation::MultiplyKernelSpectrum { .. } => shape.convolution_len,
            RaderFftPassOperation::Scatter { .. } => shape.prime,
        };
        let local_size = logical_work.min(shape.device.max_threads_per_block).max(1);
        let workgroup_x = u32::try_from(local_size).map_err(|_| VkFftError::ValueOutOfRange {
            field: "FFT Rader workgroup size",
        })?;
        let dispatch_x =
            u32::try_from(shape.batch_count).map_err(|_| VkFftError::ValueOutOfRange {
                field: "FFT Rader dispatch workgroup count",
            })?;
        let pass = Self {
            name,
            scalar: shape.scalar,
            input_storage_scalar: shape.scalar,
            output_storage_scalar: shape.scalar,
            auxiliary_storage_scalar: shape.scalar,
            direction: shape.direction,
            prime: shape.prime,
            convolution_len: shape.convolution_len,
            batch_count: shape.batch_count,
            workgroup_size: WorkgroupSize {
                x: workgroup_x,
                y: 1,
                z: 1,
            },
            dispatch: DispatchGeometry {
                x: dispatch_x,
                y: 1,
                z: 1,
            },
            axis_batch_block: None,
            operation,
        };
        pass.validate()?;
        Ok(pass)
    }

    fn with_external_input_storage(mut self, storage: ScalarType) -> Result<Self> {
        if storage != self.scalar
            && (!matches!(
                (self.scalar, storage),
                (ScalarType::F32, ScalarType::F16) | (ScalarType::F64, ScalarType::F32)
            ) || !matches!(self.operation, RaderFftPassOperation::GatherReverse))
        {
            return Err(VkFftError::UnsupportedPrecision {
                backend: "FFT Rader gather input boundary",
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
            ) || !matches!(self.operation, RaderFftPassOperation::Scatter { .. }))
        {
            return Err(VkFftError::UnsupportedPrecision {
                backend: "FFT Rader scatter output boundary",
                precision: "unsupported compute/storage scalar pair or pass operation",
            });
        }
        self.output_storage_scalar = storage;
        self.validate()?;
        Ok(self)
    }

    fn with_axis0_batch_block(
        mut self,
        block: StockhamAxisBlockSchedule,
        device: DeviceProfile,
    ) -> Result<Self> {
        if !matches!(
            self.operation,
            RaderFftPassOperation::GatherReverse | RaderFftPassOperation::Scatter { .. }
        ) {
            return Err(VkFftError::UnsupportedKernelPath(
                "FFT Rader grouped caller ownership is limited to gather/scatter boundaries",
            ));
        }
        block.validate(self.batch_count, device)?;
        self.axis_batch_block = Some(block);
        self.workgroup_size = WorkgroupSize {
            x: u32::try_from(block.local_size_x).map_err(|_| VkFftError::ValueOutOfRange {
                field: "FFT Rader grouped workgroup x size",
            })?,
            y: u32::try_from(block.local_size_y).map_err(|_| VkFftError::ValueOutOfRange {
                field: "FFT Rader grouped workgroup y size",
            })?,
            z: 1,
        };
        self.dispatch = DispatchGeometry {
            x: u32::try_from(self.batch_count.div_ceil(block.grouped_batch)).map_err(|_| {
                VkFftError::ValueOutOfRange {
                    field: "FFT Rader grouped dispatch count",
                }
            })?,
            y: 1,
            z: 1,
        };
        self.validate()?;
        Ok(self)
    }

    fn with_auxiliary_storage(mut self, storage: ScalarType) -> Result<Self> {
        if storage != self.scalar
            && (!matches!(
                (self.scalar, storage),
                (ScalarType::F32, ScalarType::F16) | (ScalarType::F64, ScalarType::F32)
            ) || !matches!(self.operation, RaderFftPassOperation::Scatter { .. }))
        {
            return Err(VkFftError::UnsupportedPrecision {
                backend: "FFT Rader scatter auxiliary boundary",
                precision: "unsupported compute/storage scalar pair or pass operation",
            });
        }
        self.auxiliary_storage_scalar = storage;
        self.validate()?;
        Ok(self)
    }

    pub fn validate(&self) -> Result<()> {
        if self.prime < 2 || self.convolution_len + 1 != self.prime || self.batch_count == 0 {
            return Err(VkFftError::InvalidKernelIr(
                "FFT Rader pass requires convolution_len = prime - 1 and a non-zero batch count",
            ));
        }
        if self.workgroup_size.x == 0 || self.workgroup_size.y == 0 || self.workgroup_size.z == 0 {
            return Err(VkFftError::InvalidKernelIr(
                "FFT Rader workgroup dimensions must be non-zero",
            ));
        }
        if let Some(block) = self.axis_batch_block {
            if self.workgroup_size.x as usize != block.local_size_x
                || self.workgroup_size.y as usize != block.local_size_y
                || self.workgroup_size.z != 1
                || self.dispatch.x as usize != self.batch_count.div_ceil(block.grouped_batch)
                || self.dispatch.y != 1
                || self.dispatch.z != 1
            {
                return Err(VkFftError::InvalidKernelIr(
                    "FFT Rader grouped caller geometry is inconsistent",
                ));
            }
        } else if self.dispatch.x as usize != self.batch_count
            || self.dispatch.y != 1
            || self.dispatch.z != 1
        {
            return Err(VkFftError::InvalidKernelIr(
                "FFT Rader requires one workgroup per transform",
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
            RaderFftPassOperation::GatherReverse => {
                supported_storage(self.input_storage_scalar)
                    && self.output_storage_scalar == self.scalar
                    && self.auxiliary_storage_scalar == self.scalar
            }
            RaderFftPassOperation::MultiplyKernelSpectrum { .. } => {
                self.input_storage_scalar == self.scalar
                    && self.output_storage_scalar == self.scalar
                    && self.auxiliary_storage_scalar == self.scalar
            }
            RaderFftPassOperation::Scatter { .. } => {
                self.input_storage_scalar == self.scalar
                    && supported_storage(self.output_storage_scalar)
                    && supported_storage(self.auxiliary_storage_scalar)
            }
        };
        if self.scalar == ScalarType::F16 || !storage_contract_ok {
            return Err(VkFftError::InvalidKernelIr(
                "FFT Rader pass storage boundary is inconsistent with its operation",
            ));
        }
        if let RaderFftPassOperation::MultiplyKernelSpectrum { spectrum } = &self.operation {
            if spectrum.len() != self.convolution_len {
                return Err(VkFftError::InvalidKernelIr(
                    "FFT Rader kernel spectrum length must equal prime - 1",
                ));
            }
            if spectrum
                .iter()
                .any(|value| !value.re.is_finite() || !value.im.is_finite())
            {
                return Err(VkFftError::InvalidKernelIr(
                    "FFT Rader kernel spectrum contains non-finite values",
                ));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RaderFftInputStrategy {
    /// Materialize a separate `(p - 1)` generator-order buffer before the FFT.
    GatherReversePass,
    /// Read the original prime-length input directly from the forward Stockham
    /// stage in reversed primitive-root order, eliminating the gather dispatch.
    GeneratorOrderStockham,
    /// Read the original prime-length input directly from the root recursive
    /// Cooley-Tukey pack pass in reversed primitive-root order.
    GeneratorOrderRecursive,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RaderFftPipelineIr {
    pub prime: usize,
    pub convolution_len: usize,
    pub batch_count: usize,
    pub direction: Direction,
    pub scalar: ScalarType,
    /// Physical shared-memory budget of the device profile used to materialize this
    /// pipeline. Small composite FFT-Rader fusion uses this retained budget to prove
    /// that its parent-owned shared stripes fit before replacing multiple launches.
    pub device_shared_memory_bytes: usize,
    /// Caller-visible prime-length input storage scalar. Internal convolution
    /// buffers and the Rader kernel spectrum always use `scalar`.
    pub input_storage_scalar: ScalarType,
    /// Caller-visible prime-length output storage scalar.
    pub output_storage_scalar: ScalarType,
    /// Caller-side prime-length mapping. This stays independent of the internal
    /// Rader generator permutation used by the `(p - 1)` convolution.
    pub io_mapping: StockhamIoMapping,
    pub input_strategy: RaderFftInputStrategy,
    /// Exact register/container schedule scored against the physical device profile.
    /// This remains visible even when a backend-specific compiler envelope requires
    /// a second execution rescore with a lower thread ceiling.
    pub device_register_schedule: Option<RaderFftRegisterSchedule>,
    /// Upstream Rader-container register plan for power-of-two `(p - 1)` internal
    /// FFTs. `None` keeps the generic recursive Stockham/Rader fallback.
    pub internal_register_schedule: Option<RaderFftRegisterSchedule>,
    /// Axis-level independent-batch block from `VkFFTSplitAxisBlock`. This is kept
    /// separate from `internal_register_schedule.container_fft_num`: independent
    /// batches must never be reinterpreted as Rader containers.
    pub axis_batch_block: Option<StockhamAxisBlockSchedule>,
    /// Forced-upload parent ownership for the explicit prime-length caller
    /// boundaries. Internal convolution children keep their own scheduler grouping.
    pub caller_axis_batch_block: Option<StockhamAxisBlockSchedule>,
    pub table: RaderTable,
    pub gather: RaderFftPassIr,
    pub forward_fft: Box<OneDimFftIr>,
    pub multiply: RaderFftPassIr,
    pub inverse_fft: Box<OneDimFftIr>,
    pub scatter: RaderFftPassIr,
}

/// Backend code-generation envelope for a monolithic register-Rader kernel. Vulkan
/// on NVIDIA still rejects some large non-power-of-two pipelines above the legacy
/// 256-thread compiler envelope even when the physical device allows wider groups.
/// This cap is execution-only: physical upstream scoring remains separate below.
fn rader_register_codegen_thread_budget(device: DeviceProfile, convolution_len: usize) -> usize {
    if device.backend == Backend::Vulkan
        && device.vendor == GpuVendor::Nvidia
        && !convolution_len.is_power_of_two()
    {
        device.max_threads_per_block.min(256)
    } else {
        device.max_threads_per_block
    }
}

/// Build the correctness fallback against the verified compiler envelope. This is
/// intentionally separate from physical scheduler scoring: a large physical register
/// schedule remains recorded even when Vulkan must execute the recursive fallback.
fn rader_recursive_fallback_device(
    mut device: DeviceProfile,
    convolution_len: usize,
) -> DeviceProfile {
    let budget = rader_register_codegen_thread_budget(device, convolution_len);
    device.max_threads_per_block = budget;
    device.max_workgroup_size[0] = device.max_workgroup_size[0].min(budget);
    device
}

/// Build an internal Rader-convolution C2C plan with the same physical device
/// scoring used by extracted real/R2R/ND children. The convolution buffer is a
/// naturally contiguous temporary, so it must never inherit a strided parent axis.
///
/// The returned plan is the physical source of truth for the whole `(p - 1)` child:
/// Stockham, Rader, and Bluestein classifications are all preserved into execution
/// materialization instead of being rebuilt with portable thresholds.
fn build_rader_convolution_child_plan(config: FftConfig, device: DeviceProfile) -> Result<FftPlan> {
    FftPlan::build_c2c_child_for_device(config, device, C2cDeviceAxisClass::Contiguous)
}

#[derive(Debug, Clone, Copy)]
struct RaderOuterFourStepAxisBlockContext {
    upload_count: usize,
    axis_upload_id: usize,
    stage_start_size: usize,
    outer_batch_count: usize,
    perform_zero_padding: bool,
    grouped_batch_override: Option<usize>,
    precision: Precision,
}

impl RaderFftPipelineIr {
    pub fn build(plan: &FftPlan, direction: Direction, device: DeviceProfile) -> Result<Self> {
        let outer_fft_len = plan.config.dimensions.first().copied().unwrap_or(0);
        Self::build_with_container_context(plan, direction, device, outer_fft_len, 1)
    }

    pub(crate) fn build_with_container_context(
        plan: &FftPlan,
        direction: Direction,
        device: DeviceProfile,
        outer_fft_len: usize,
        container_fft_num: usize,
    ) -> Result<Self> {
        Self::build_with_container_context_impl(
            plan,
            direction,
            device,
            outer_fft_len,
            container_fft_num,
            None,
        )
    }

    pub(crate) fn build_with_container_context_and_outer_four_step(
        plan: &FftPlan,
        direction: Direction,
        device: DeviceProfile,
        outer_fft_len: usize,
        container_fft_num: usize,
        upload_count: usize,
        axis_upload_id: usize,
        stage_start_size: usize,
        outer_batch_count: usize,
        perform_zero_padding: bool,
        grouped_batch_override: Option<usize>,
        precision: Precision,
    ) -> Result<Self> {
        Self::build_with_container_context_impl(
            plan,
            direction,
            device,
            outer_fft_len,
            container_fft_num,
            Some(RaderOuterFourStepAxisBlockContext {
                upload_count,
                axis_upload_id,
                stage_start_size,
                outer_batch_count,
                perform_zero_padding,
                grouped_batch_override,
                precision,
            }),
        )
    }

    fn build_with_container_context_impl(
        plan: &FftPlan,
        direction: Direction,
        device: DeviceProfile,
        outer_fft_len: usize,
        container_fft_num: usize,
        outer_four_step_context: Option<RaderOuterFourStepAxisBlockContext>,
    ) -> Result<Self> {
        if plan.config.dimensions.len() != 1 {
            return Err(VkFftError::UnsupportedKernelPath(
                "initial FFT Rader pipeline supports one-dimensional plans only",
            ));
        }
        if plan.config.transform != TransformKind::ComplexToComplex {
            return Err(VkFftError::UnsupportedKernelPath(
                "initial FFT Rader pipeline supports C2C transforms only",
            ));
        }
        let scalar = match plan.config.precision {
            Precision::F32 => ScalarType::F32,
            Precision::F64 if device.supports_f64 => ScalarType::F64,
            other => {
                return Err(VkFftError::UnsupportedPrecision {
                    backend: "FFT Rader pipeline IR",
                    precision: precision_name(other),
                });
            }
        };
        let axis = plan
            .axes
            .first()
            .ok_or(VkFftError::InvalidKernelIr("missing FFT Rader axis plan"))?;
        let AxisAlgorithm::Rader { stockham, primes } = &axis.algorithm else {
            return Err(VkFftError::UnsupportedKernelPath(
                "FFT Rader pipeline requires a planner-selected Rader axis",
            ));
        };
        if !stockham.prime_factors.is_empty()
            || !stockham.merged_radices.is_empty()
            || primes.len() != 1
            || primes[0].multiplicity != 1
            || primes[0].prime != axis.effective_fft_len
        {
            return Err(VkFftError::UnsupportedKernelPath(
                "initial FFT Rader pipeline supports a single prime axis without Stockham factors",
            ));
        }
        if !matches!(primes[0].mode, RaderMode::FftConvolution { .. }) {
            return Err(VkFftError::UnsupportedKernelPath(
                "planner selected direct-multiplication Rader; FFT Rader pipeline is not applicable",
            ));
        }

        let prime = primes[0].prime;
        let convolution_len = prime - 1;
        let batch_count = plan.config.batch_count;
        let table = RaderTable::from_prime_plan(&primes[0], direction)?;
        let forward_config = FftConfig::new(vec![convolution_len])
            .with_batch_count(batch_count)
            .with_precision(plan.config.precision)
            .with_tuning(plan.config.tuning);
        let inverse_config = forward_config.clone().with_inverse_normalization(true);
        let forward_plan = build_rader_convolution_child_plan(forward_config, device)?;
        let inverse_plan = build_rader_convolution_child_plan(inverse_config, device)?;
        let recursive_device = rader_recursive_fallback_device(device, convolution_len);
        let register_codegen_thread_budget =
            rader_register_codegen_thread_budget(device, convolution_len);
        let device_register_schedule = if has_specialized_gpu_scheduler_policy(device) {
            match plan_gpu_rader_fft_registers_for_containers(
                prime,
                batch_count,
                outer_fft_len,
                container_fft_num,
                device,
            ) {
                Ok(schedule) => Some(schedule),
                Err(error) if is_rader_register_schedule_fallback(&error) => None,
                Err(error) => return Err(error),
            }
        } else {
            None
        };
        let execution_register_schedule = device_register_schedule
            .as_ref()
            .filter(|schedule| {
                schedule.execution_threads_per_workgroup <= register_codegen_thread_budget
            })
            .cloned();
        let mut forward_fft =
            OneDimFftIr::build(&forward_plan, Direction::Forward, recursive_device)?;
        let mut inverse_fft =
            OneDimFftIr::build(&inverse_plan, Direction::Inverse, recursive_device)?;
        let mut internal_register_schedule = None;
        let mut axis_batch_block = None;
        if let Some(schedule) = execution_register_schedule {
            let execution_containers = schedule.execution_container_fft_num;
            let transpose = schedule
                .rader_transpose
                .clone()
                .filter(|_| schedule.upstream_grouping_is_executable());
            let apply_schedule = |fft: RecursiveFftIr| -> Result<RecursiveFftIr> {
                if let Some(transpose) = transpose.clone() {
                    fft.with_stockham_rader_transpose_schedule(
                        schedule.internal_fft.clone(),
                        transpose,
                        device,
                    )
                } else {
                    fft.with_stockham_register_schedule_grouping(
                        schedule.internal_fft.clone(),
                        execution_containers,
                        device,
                    )
                }
            };
            if let (OneDimFftIr::Recursive(forward), OneDimFftIr::Recursive(inverse)) =
                (&forward_fft, &inverse_fft)
            {
                match apply_schedule(forward.as_ref().clone()) {
                    Ok(scheduled_forward) => match apply_schedule(inverse.as_ref().clone()) {
                        Ok(scheduled_inverse) => {
                            forward_fft = OneDimFftIr::Recursive(Box::new(scheduled_forward));
                            inverse_fft = OneDimFftIr::Recursive(Box::new(scheduled_inverse));
                            internal_register_schedule = Some(schedule);
                        }
                        Err(error) if is_rader_register_schedule_fallback(&error) => {}
                        Err(error) => return Err(error),
                    },
                    Err(error) if is_rader_register_schedule_fallback(&error) => {}
                    Err(error) => return Err(error),
                }
            }
        }
        let grouped_batch_override = plan.config.grouped_batch_for_axis(0);
        let mut axis_block_candidate = if let Some(schedule) = internal_register_schedule.as_ref() {
            plan_gpu_axis0_fft_rader_batch_block_with_grouped_batch(
                prime,
                batch_count,
                schedule,
                scalar.complex_bytes(),
                plan.config.zero_padding_for_axis(0).is_some(),
                grouped_batch_override,
                device,
            )?
        } else {
            None
        };
        if let (Some(context), Some(current)) = (outer_four_step_context, axis_block_candidate) {
            axis_block_candidate = plan_gpu_axis0_four_step_default_block_from_shape_for_precision(
                context.upload_count,
                prime,
                current.threads_per_transform,
                FourStepAxisBlockRequest {
                    axis_upload_id: context.axis_upload_id,
                    stage_start_size: context.stage_start_size,
                    transform_count: batch_count,
                    outer_batch_count: context.outer_batch_count,
                    perform_zero_padding: context.perform_zero_padding,
                    grouped_batch_override: context.grouped_batch_override,
                },
                context.precision,
                scalar.complex_bytes(),
                device,
            )?;
        }
        if grouped_batch_override.is_some() && axis_block_candidate.is_none() {
            return Err(VkFftError::UnsupportedKernelPath(
                "groupedBatch override is not executable for this FFT-Rader axis",
            ));
        }
        if let (Some(schedule), Some(block)) =
            (internal_register_schedule.as_ref(), axis_block_candidate)
        {
            let apply_axis_block = |fft: &mut OneDimFftIr| -> Result<()> {
                let OneDimFftIr::Recursive(recursive) = fft else {
                    return Err(VkFftError::UnsupportedKernelPath(
                        "axis-level Rader batching requires a recursive Stockham convolution child",
                    ));
                };
                let RecursiveFftNodeIr::Stockham(root) = &mut recursive.root else {
                    return Err(VkFftError::UnsupportedKernelPath(
                        "axis-level Rader batching requires a single Stockham convolution root",
                    ));
                };
                **root = root.as_ref().clone().with_axis0_batch_block(
                    schedule.internal_fft.clone(),
                    block,
                    device,
                )?;
                recursive.validate()
            };
            apply_axis_block(&mut forward_fft)?;
            apply_axis_block(&mut inverse_fft)?;
            axis_batch_block = Some(block);
        }
        let generator_mapping = RaderGeneratorMapping {
            prime,
            generator: table.generator,
        };
        let normalize_prime = direction == Direction::Inverse && plan.config.normalize_inverse;
        let scatter_mapping = RaderScatterMapping {
            prime,
            generator: table.generator,
            normalize_prime,
            auxiliary_input: None,
        };
        let mut input_strategy = RaderFftInputStrategy::GatherReversePass;
        if let OneDimFftIr::Recursive(forward) = &mut forward_fft {
            if let RecursiveFftNodeIr::Stockham(kernel) = &mut forward.root {
                let mapping = StockhamIoMapping::RaderGeneratorReverse(generator_mapping);
                **kernel = kernel.as_ref().clone().with_stockham_io_mapping(mapping)?;
                input_strategy = RaderFftInputStrategy::GeneratorOrderStockham;
            } else if forward.four_step_plan.is_none()
                && let RecursiveFftNodeIr::CooleyTukey(node) = &mut forward.root
            {
                node.pack_right = node
                    .pack_right
                    .clone()
                    .with_rader_generator_input(generator_mapping)?;
                input_strategy = RaderFftInputStrategy::GeneratorOrderRecursive;
            }
        }
        if input_strategy == RaderFftInputStrategy::GeneratorOrderRecursive
            && let OneDimFftIr::Recursive(inverse) = &mut inverse_fft
            && inverse.four_step_plan.is_none()
            && let RecursiveFftNodeIr::CooleyTukey(node) = &mut inverse.root
        {
            node.pack_right = node.pack_right.clone().with_lookup_table_input_multiply()?;
            node.scatter_output = node
                .scatter_output
                .clone()
                .with_rader_scatter_output(scatter_mapping)?;
        }
        let forward_fft = Box::new(forward_fft);
        let inverse_fft = Box::new(inverse_fft);

        let kernel_plan = build_rader_convolution_child_plan(
            FftConfig::new(vec![convolution_len])
                .with_precision(plan.config.precision)
                .with_tuning(plan.config.tuning),
            device,
        )?;
        let kernel_fft = OneDimFftIr::build(&kernel_plan, Direction::Forward, recursive_device)?;
        let spectrum = execute_one_dim_fft_ir(&kernel_fft, &table.twiddles_by_generator_power)?;
        let direction_name = match direction {
            Direction::Forward => "forward",
            Direction::Inverse => "inverse",
        };
        let shape = RaderFftPassShape {
            scalar,
            direction,
            prime,
            convolution_len,
            batch_count,
            device,
        };
        let gather = RaderFftPassIr::new(
            format!("vkfft_rader_fft_gather_{prime}_{direction_name}"),
            shape,
            RaderFftPassOperation::GatherReverse,
        )?;
        let multiply = RaderFftPassIr::new(
            format!("vkfft_rader_fft_mul_{prime}_{direction_name}"),
            shape,
            RaderFftPassOperation::MultiplyKernelSpectrum { spectrum },
        )?;
        let scatter = RaderFftPassIr::new(
            format!("vkfft_rader_fft_scatter_{prime}_{direction_name}"),
            shape,
            RaderFftPassOperation::Scatter {
                normalize: direction == Direction::Inverse && plan.config.normalize_inverse,
            },
        )?;
        let pipeline = Self {
            prime,
            convolution_len,
            batch_count,
            direction,
            scalar,
            device_shared_memory_bytes: device.shared_memory_bytes,
            input_storage_scalar: scalar,
            output_storage_scalar: scalar,
            io_mapping: StockhamIoMapping::Contiguous,
            input_strategy,
            device_register_schedule,
            internal_register_schedule,
            axis_batch_block,
            caller_axis_batch_block: None,
            table,
            gather,
            forward_fft,
            multiply,
            inverse_fft,
            scatter,
        };
        pipeline.validate()?;
        Ok(pipeline)
    }

    pub(crate) fn forward_recursive(&self) -> Option<&RecursiveFftIr> {
        match self.forward_fft.as_ref() {
            OneDimFftIr::Recursive(ir) => Some(ir.as_ref()),
            OneDimFftIr::Bluestein(_) => None,
        }
    }

    pub(crate) fn inverse_recursive(&self) -> Option<&RecursiveFftIr> {
        match self.inverse_fft.as_ref() {
            OneDimFftIr::Recursive(ir) => Some(ir.as_ref()),
            OneDimFftIr::Bluestein(_) => None,
        }
    }

    fn forward_recursive_mut(&mut self) -> Option<&mut RecursiveFftIr> {
        match self.forward_fft.as_mut() {
            OneDimFftIr::Recursive(ir) => Some(ir.as_mut()),
            OneDimFftIr::Bluestein(_) => None,
        }
    }

    fn inverse_recursive_mut(&mut self) -> Option<&mut RecursiveFftIr> {
        match self.inverse_fft.as_mut() {
            OneDimFftIr::Recursive(ir) => Some(ir.as_mut()),
            OneDimFftIr::Bluestein(_) => None,
        }
    }

    pub(crate) fn with_external_input_storage_scalar(
        mut self,
        storage: ScalarType,
    ) -> Result<Self> {
        if storage != self.scalar
            && !matches!(
                (self.scalar, storage),
                (ScalarType::F32, ScalarType::F16) | (ScalarType::F64, ScalarType::F32)
            )
        {
            return Err(VkFftError::UnsupportedPrecision {
                backend: "FFT Rader input-storage boundary",
                precision: "unsupported compute/storage scalar pair",
            });
        }
        if storage != self.scalar {
            match self.input_strategy {
                RaderFftInputStrategy::GeneratorOrderStockham => {
                    let forward =
                        self.forward_recursive_mut()
                            .ok_or(VkFftError::InvalidKernelIr(
                                "generator-order Stockham strategy lost its recursive child",
                            ))?;
                    let RecursiveFftNodeIr::Stockham(kernel) = &mut forward.root else {
                        return Err(VkFftError::InvalidKernelIr(
                            "generator-order Stockham strategy lost its Stockham root",
                        ));
                    };
                    **kernel = kernel
                        .as_ref()
                        .clone()
                        .with_external_input_storage_scalar(storage)?;
                }
                RaderFftInputStrategy::GeneratorOrderRecursive => {
                    let forward =
                        self.forward_recursive_mut()
                            .ok_or(VkFftError::InvalidKernelIr(
                                "generator-order recursive strategy lost its recursive child",
                            ))?;
                    let RecursiveFftNodeIr::CooleyTukey(node) = &mut forward.root else {
                        return Err(VkFftError::InvalidKernelIr(
                            "generator-order recursive strategy lost its Cooley-Tukey root",
                        ));
                    };
                    node.pack_right = node
                        .pack_right
                        .clone()
                        .with_external_input_storage(storage)?;
                    node.validate()?;
                }
                RaderFftInputStrategy::GatherReversePass => {
                    self.gather = self.gather.clone().with_external_input_storage(storage)?;
                }
            }
        }
        if self.has_fused_recursive_inverse() {
            let inverse = self
                .inverse_recursive_mut()
                .ok_or(VkFftError::InvalidKernelIr(
                    "fused recursive Rader inverse lost its recursive child",
                ))?;
            let RecursiveFftNodeIr::CooleyTukey(node) = &mut inverse.root else {
                return Err(VkFftError::InvalidKernelIr(
                    "fused recursive Rader inverse lost its Cooley-Tukey root",
                ));
            };
            node.scatter_output = node
                .scatter_output
                .clone()
                .with_rader_auxiliary_storage(storage)?;
            node.validate()?;
        } else if self.fused_inverse_stockham_kernel()?.is_none() {
            self.scatter = self.scatter.clone().with_auxiliary_storage(storage)?;
        }
        self.input_storage_scalar = storage;
        self.validate()?;
        Ok(self)
    }

    pub(crate) fn with_external_output_storage_scalar(
        mut self,
        storage: ScalarType,
    ) -> Result<Self> {
        if storage != self.scalar
            && !matches!(
                (self.scalar, storage),
                (ScalarType::F32, ScalarType::F16) | (ScalarType::F64, ScalarType::F32)
            )
        {
            return Err(VkFftError::UnsupportedPrecision {
                backend: "FFT Rader output-storage boundary",
                precision: "unsupported compute/storage scalar pair",
            });
        }
        if storage != self.scalar {
            if self.has_fused_recursive_inverse() {
                let inverse = self
                    .inverse_recursive_mut()
                    .ok_or(VkFftError::InvalidKernelIr(
                        "fused recursive Rader inverse lost its recursive child",
                    ))?;
                let RecursiveFftNodeIr::CooleyTukey(node) = &mut inverse.root else {
                    return Err(VkFftError::InvalidKernelIr(
                        "fused recursive Rader inverse lost its Cooley-Tukey root",
                    ));
                };
                node.scatter_output = node
                    .scatter_output
                    .clone()
                    .with_external_output_storage(storage)?;
                node.validate()?;
            } else if self.fused_inverse_stockham_kernel()?.is_none() {
                self.scatter = self.scatter.clone().with_external_output_storage(storage)?;
            }
        }
        self.output_storage_scalar = storage;
        self.validate()?;
        Ok(self)
    }

    /// Install an outer Four-step caller mapping on a prime-length FFT-Rader
    /// component. A Stockham forward convolution keeps generator-order loads fused
    /// by composing the primitive-root permutation with the outer caller mapping;
    /// recursive/non-Stockham children retain the explicit gather fallback.
    pub(crate) fn with_stockham_io_mapping(mut self, mapping: StockhamIoMapping) -> Result<Self> {
        if !matches!(
            mapping,
            StockhamIoMapping::Contiguous
                | StockhamIoMapping::FourStepRight(_)
                | StockhamIoMapping::FourStepLeft(_)
                | StockhamIoMapping::FourStepThreeUpload2(_)
                | StockhamIoMapping::FourStepThreeUpload1(_)
                | StockhamIoMapping::FourStepThreeUpload0(_)
        ) {
            return Err(VkFftError::UnsupportedKernelPath(
                "FFT Rader caller mapping must be contiguous or Four-step",
            ));
        }
        mapping.validate_kernel(self.prime, self.batch_count)?;
        if mapping != StockhamIoMapping::Contiguous {
            match self.input_strategy {
                RaderFftInputStrategy::GeneratorOrderStockham => {
                    let caller = RaderFourStepInputMapping::from_stockham(mapping).ok_or(
                        VkFftError::UnsupportedKernelPath(
                            "generator-order FFT Rader requires a Four-step caller mapping",
                        ),
                    )?;
                    let composed =
                        StockhamIoMapping::RaderGeneratorFourStep(RaderGeneratorFourStepMapping {
                            rader: RaderGeneratorMapping {
                                prime: self.prime,
                                generator: self.table.generator,
                            },
                            caller,
                        });
                    let forward =
                        self.forward_recursive_mut()
                            .ok_or(VkFftError::InvalidKernelIr(
                                "generator-order Stockham FFT Rader lost its forward child",
                            ))?;
                    let RecursiveFftNodeIr::Stockham(kernel) = &mut forward.root else {
                        return Err(VkFftError::InvalidKernelIr(
                            "generator-order Stockham FFT Rader lost its Stockham root",
                        ));
                    };
                    **kernel = kernel.as_ref().clone().with_stockham_io_mapping(composed)?;
                }
                RaderFftInputStrategy::GeneratorOrderRecursive => {
                    let forward =
                        self.forward_recursive_mut()
                            .ok_or(VkFftError::InvalidKernelIr(
                                "generator-order recursive FFT Rader lost its forward child",
                            ))?;
                    let RecursiveFftNodeIr::CooleyTukey(node) = &mut forward.root else {
                        return Err(VkFftError::InvalidKernelIr(
                            "generator-order recursive FFT Rader lost its Cooley root",
                        ));
                    };
                    node.pack_right.input_modifier = CooleyTukeyInputModifier::None;
                    node.validate()?;

                    let inverse =
                        self.inverse_recursive_mut()
                            .ok_or(VkFftError::InvalidKernelIr(
                                "generator-order recursive FFT Rader lost its inverse child",
                            ))?;
                    if let RecursiveFftNodeIr::CooleyTukey(node) = &mut inverse.root {
                        if matches!(
                            node.pack_right.input_modifier,
                            CooleyTukeyInputModifier::MultiplyLookupTable
                        ) {
                            node.pack_right.input_modifier = CooleyTukeyInputModifier::None;
                        }
                        if matches!(
                            node.scatter_output.output_modifier,
                            CooleyTukeyOutputModifier::RaderScatter(_)
                        ) {
                            node.scatter_output.output_modifier = CooleyTukeyOutputModifier::None;
                        }
                        node.validate()?;
                    }
                    self.input_strategy = RaderFftInputStrategy::GatherReversePass;
                }
                RaderFftInputStrategy::GatherReversePass => {}
            }
        }
        self.io_mapping = mapping;
        self.validate()?;
        Ok(self)
    }

    /// Compose the generator-order forward convolution load with the natural source
    /// layout of a prime-length right child inside a Cooley-Tukey parent. This keeps
    /// the caller-visible Rader mapping contiguous so the already-fused inverse path
    /// remains available; its x0/DC auxiliary reads are remapped from the same caller.
    pub(crate) fn with_cooley_right_input_mapping(
        mut self,
        caller: CooleyRightInputMapping,
    ) -> Result<Self> {
        if self.input_strategy != RaderFftInputStrategy::GeneratorOrderStockham
            || self.io_mapping != StockhamIoMapping::Contiguous
            || self.input_storage_scalar != self.scalar
        {
            return Err(VkFftError::UnsupportedKernelPath(
                "Cooley-right FFT Rader input fusion requires same-scalar contiguous generator-order Stockham input",
            ));
        }
        caller.validate(self.prime, self.batch_count)?;
        if self.fused_inverse_stockham_kernel()?.is_none() {
            return Err(VkFftError::UnsupportedKernelPath(
                "Cooley-right FFT Rader input fusion requires a fused Stockham inverse boundary",
            ));
        }
        let rader = RaderGeneratorMapping {
            prime: self.prime,
            generator: self.table.generator,
        };
        let mapping =
            StockhamIoMapping::RaderGeneratorCooleyRight(RaderGeneratorCooleyRightMapping {
                rader,
                caller,
            });
        let forward = self
            .forward_recursive_mut()
            .ok_or(VkFftError::InvalidKernelIr(
                "Cooley-right FFT Rader input fusion lost its forward child",
            ))?;
        let RecursiveFftNodeIr::Stockham(kernel) = &mut forward.root else {
            return Err(VkFftError::UnsupportedKernelPath(
                "Cooley-right FFT Rader input fusion requires a Stockham forward root",
            ));
        };
        **kernel = kernel.as_ref().clone().with_stockham_io_mapping(mapping)?;
        self.validate()?;
        Ok(self)
    }

    fn cooley_right_auxiliary_mapping(&self) -> Option<CooleyRightInputMapping> {
        let forward = self.forward_recursive()?;
        let RecursiveFftNodeIr::Stockham(kernel) = &forward.root else {
            return None;
        };
        match kernel.io_mapping {
            StockhamIoMapping::RaderGeneratorCooleyRight(mapping) => Some(mapping.caller),
            _ => None,
        }
    }

    /// Replace the independent-batch physical block for a standalone FFT-Rader
    /// axis without changing its internal Rader-container count. Higher ND axes use
    /// this to turn the axis-0-shaped caller block into the fixed-upstream
    /// transforms-X / FFT-threads-Y layout while retaining the same `(p-1)`
    /// register schedule in both convolution children.
    pub(crate) fn with_standalone_axis_batch_block(
        mut self,
        block: StockhamAxisBlockSchedule,
        device: DeviceProfile,
    ) -> Result<Self> {
        block.validate(self.batch_count, device)?;
        let schedule =
            self.internal_register_schedule
                .clone()
                .ok_or(VkFftError::UnsupportedKernelPath(
                    "standalone FFT-Rader axis remapping requires a physical register schedule",
                ))?;
        if schedule.container_fft_num != 1 || schedule.execution_container_fft_num != 1 {
            return Err(VkFftError::UnsupportedKernelPath(
                "standalone FFT-Rader axis remapping cannot reinterpret Rader containers as independent batches",
            ));
        }
        if block.threads_per_transform < schedule.min_rader_fft_thread_num {
            return Err(VkFftError::UnsupportedKernelPath(
                "standalone FFT-Rader axis remapping fell below the prime-level thread floor",
            ));
        }
        let apply = |fft: &mut OneDimFftIr| -> Result<()> {
            let OneDimFftIr::Recursive(recursive) = fft else {
                return Err(VkFftError::UnsupportedKernelPath(
                    "standalone FFT-Rader axis remapping requires recursive Stockham convolution children",
                ));
            };
            let RecursiveFftNodeIr::Stockham(root) = &mut recursive.root else {
                return Err(VkFftError::UnsupportedKernelPath(
                    "standalone FFT-Rader axis remapping requires one Stockham convolution root",
                ));
            };
            **root = root.as_ref().clone().with_axis_batch_block_preserving_io(
                schedule.internal_fft.clone(),
                block,
                device,
            )?;
            recursive.validate()
        };
        apply(&mut self.forward_fft)?;
        apply(&mut self.inverse_fft)?;
        self.axis_batch_block = Some(block);
        self.validate()?;
        Ok(self)
    }

    pub(crate) fn with_caller_axis0_batch_block(
        mut self,
        block: StockhamAxisBlockSchedule,
        device: DeviceProfile,
    ) -> Result<Self> {
        block.validate(self.batch_count, device)?;
        if self.input_strategy == RaderFftInputStrategy::GeneratorOrderStockham {
            let remap_result = {
                let forward = self
                    .forward_recursive_mut()
                    .ok_or(VkFftError::InvalidKernelIr(
                        "mapped generator-order FFT Rader lost its forward child",
                    ))?;
                let RecursiveFftNodeIr::Stockham(root) = &mut forward.root else {
                    return Err(VkFftError::InvalidKernelIr(
                        "mapped generator-order FFT Rader lost its forward Stockham root",
                    ));
                };
                let schedule = root
                    .scheduler_hint
                    .clone()
                    .ok_or(VkFftError::UnsupportedKernelPath(
                        "mapped generator-order FFT Rader requires forward Stockham scheduler metadata",
                    ))?;
                match root
                    .as_ref()
                    .clone()
                    .with_axis_batch_block_preserving_io(schedule, block, device)
                {
                    Ok(mapped) => {
                        **root = mapped;
                        forward.validate()
                    }
                    Err(error) => Err(error),
                }
            };
            if let Err(error) = remap_result {
                if matches!(
                    error,
                    VkFftError::UnsupportedKernelPath(_) | VkFftError::ResourceLimitExceeded { .. }
                ) {
                    let forward =
                        self.forward_recursive_mut()
                            .ok_or(VkFftError::InvalidKernelIr(
                                "mapped FFT Rader fallback lost its forward child",
                            ))?;
                    let RecursiveFftNodeIr::Stockham(root) = &mut forward.root else {
                        return Err(VkFftError::InvalidKernelIr(
                            "mapped FFT Rader fallback lost its forward Stockham root",
                        ));
                    };
                    **root = root
                        .as_ref()
                        .clone()
                        .with_stockham_io_mapping(StockhamIoMapping::Contiguous)?;
                    forward.validate()?;
                    self.input_strategy = RaderFftInputStrategy::GatherReversePass;
                } else {
                    return Err(error);
                }
            }
        }
        self.gather = self.gather.clone().with_axis0_batch_block(block, device)?;
        self.scatter = self.scatter.clone().with_axis0_batch_block(block, device)?;
        self.caller_axis_batch_block = Some(block);
        self.validate()?;
        Ok(self)
    }

    pub fn validate(&self) -> Result<()> {
        self.gather.validate()?;
        self.multiply.validate()?;
        self.scatter.validate()?;
        if let Some(block) = self.caller_axis_batch_block {
            if self.gather.axis_batch_block != Some(block)
                || self.scatter.axis_batch_block != Some(block)
                || self.multiply.axis_batch_block.is_some()
            {
                return Err(VkFftError::InvalidKernelIr(
                    "FFT Rader caller grouped ownership is inconsistent",
                ));
            }
        } else if self.gather.axis_batch_block.is_some() || self.scatter.axis_batch_block.is_some()
        {
            return Err(VkFftError::InvalidKernelIr(
                "FFT Rader caller boundary unexpectedly carries grouped ownership",
            ));
        }
        let supported_storage = |storage: ScalarType| {
            storage == self.scalar
                || matches!(
                    (self.scalar, storage),
                    (ScalarType::F32, ScalarType::F16) | (ScalarType::F64, ScalarType::F32)
                )
        };
        if !supported_storage(self.input_storage_scalar)
            || !supported_storage(self.output_storage_scalar)
        {
            return Err(VkFftError::InvalidKernelIr(
                "FFT Rader external storage scalar is inconsistent with compute precision",
            ));
        }
        self.io_mapping
            .validate_kernel(self.prime, self.batch_count)?;
        if self.io_mapping != StockhamIoMapping::Contiguous
            && !matches!(
                self.input_strategy,
                RaderFftInputStrategy::GatherReversePass
                    | RaderFftInputStrategy::GeneratorOrderStockham
            )
        {
            return Err(VkFftError::InvalidKernelIr(
                "mapped FFT Rader caller boundary requires explicit gather or composed Stockham generator loads",
            ));
        }
        if self.convolution_len + 1 != self.prime
            || self.table.prime != self.prime
            || self.table.direction != self.direction
            || self.table.permutation.len() != self.convolution_len
        {
            return Err(VkFftError::InvalidKernelIr(
                "FFT Rader pipeline/table metadata is inconsistent",
            ));
        }
        if let Some(schedule) = &self.device_register_schedule {
            schedule.validate()?;
            if schedule.prime != self.prime
                || schedule.convolution_len != self.convolution_len
                || schedule.internal_fft.rhs_transform_count != self.batch_count
            {
                return Err(VkFftError::InvalidKernelIr(
                    "FFT Rader device register schedule does not match the pipeline",
                ));
            }
        }
        if self.internal_register_schedule.is_some() && self.device_register_schedule.is_none() {
            return Err(VkFftError::InvalidKernelIr(
                "FFT Rader execution register schedule is missing its physical device score",
            ));
        }
        if let Some(schedule) = &self.internal_register_schedule {
            schedule.validate()?;
            if schedule.prime != self.prime
                || schedule.convolution_len != self.convolution_len
                || schedule.internal_fft.rhs_transform_count != self.batch_count
            {
                return Err(VkFftError::InvalidKernelIr(
                    "FFT Rader register schedule does not match the pipeline",
                ));
            }
            if let Some(device_schedule) = &self.device_register_schedule
                && (schedule.outer_fft_len != device_schedule.outer_fft_len
                    || schedule.container_fft_num != device_schedule.container_fft_num
                    || schedule.internal_fft.fft_len != device_schedule.internal_fft.fft_len)
            {
                return Err(VkFftError::InvalidKernelIr(
                    "FFT Rader execution rescore changed the transform/container decomposition",
                ));
            }
        }
        for (is_forward, fft) in [(true, &*self.forward_fft), (false, &*self.inverse_fft)] {
            fft.validate()?;
            if fft.logical_len() != self.convolution_len
                || fft.batch_count() != self.batch_count
                || fft.scalar() != self.scalar
            {
                return Err(VkFftError::InvalidKernelIr(
                    "FFT Rader internal FFT metadata does not match prime - 1",
                ));
            }
            if let Some(schedule) = &self.internal_register_schedule {
                let OneDimFftIr::Recursive(recursive) = fft else {
                    return Err(VkFftError::InvalidKernelIr(
                        "scheduled FFT Rader internal FFT must remain recursive",
                    ));
                };
                let RecursiveFftNodeIr::Stockham(kernel) = &recursive.root else {
                    return Err(VkFftError::InvalidKernelIr(
                        "scheduled FFT Rader internal FFT must remain a single Stockham root",
                    ));
                };
                let physical_block = if is_forward
                    && self.input_strategy == RaderFftInputStrategy::GeneratorOrderStockham
                {
                    self.caller_axis_batch_block.or(self.axis_batch_block)
                } else {
                    self.axis_batch_block
                };
                let expected_grouped_batch = physical_block
                    .map_or(schedule.execution_container_fft_num, |block| {
                        block.grouped_batch
                    });
                let expected_dispatch = self.batch_count.div_ceil(expected_grouped_batch);
                let expected_workgroup = physical_block.map_or(
                    (schedule.execution_threads_per_workgroup, 1usize),
                    |block| (block.local_size_x, block.local_size_y),
                );
                if kernel.scheduler_hint.as_ref() != Some(&schedule.internal_fft)
                    || kernel.workgroup_grouping.transforms_per_workgroup != expected_grouped_batch
                    || kernel.dispatch.x as usize != expected_dispatch
                    || (
                        kernel.workgroup_size.x as usize,
                        kernel.workgroup_size.y as usize,
                    ) != expected_workgroup
                {
                    return Err(VkFftError::InvalidKernelIr(
                        "FFT Rader internal Stockham kernel lost its Rader register/grouping schedule",
                    ));
                }
                if let Some(block) = physical_block
                    && (schedule.container_fft_num != 1
                        || schedule.execution_container_fft_num != 1
                        || kernel.workgroup_grouping.threads_per_transform
                            != block.threads_per_transform)
                {
                    return Err(VkFftError::InvalidKernelIr(
                        "axis-level FFT Rader batching was conflated with Rader-container grouping",
                    ));
                }
            }
        }
        match self.input_strategy {
            RaderFftInputStrategy::GatherReversePass => {
                if let Some(forward) = self.forward_recursive()
                    && let RecursiveFftNodeIr::Stockham(kernel) = &forward.root
                    && matches!(
                        kernel.io_mapping,
                        StockhamIoMapping::RaderGeneratorReverse(_)
                    )
                {
                    return Err(VkFftError::InvalidKernelIr(
                        "separate-gather Rader pipeline unexpectedly carries generator-order Stockham input",
                    ));
                }
                if self.gather.input_storage_scalar != self.input_storage_scalar
                    || self.gather.output_storage_scalar != self.scalar
                    || self.gather.auxiliary_storage_scalar != self.scalar
                {
                    return Err(VkFftError::InvalidKernelIr(
                        "explicit FFT Rader gather lost its caller-input storage boundary",
                    ));
                }
            }
            RaderFftInputStrategy::GeneratorOrderStockham => {
                let rader = RaderGeneratorMapping {
                    prime: self.prime,
                    generator: self.table.generator,
                };
                let forward = self.forward_recursive().ok_or(VkFftError::InvalidKernelIr(
                    "fused Rader generator-order input requires a recursive forward child",
                ))?;
                let RecursiveFftNodeIr::Stockham(kernel) = &forward.root else {
                    return Err(VkFftError::InvalidKernelIr(
                        "fused Rader generator-order input requires a Stockham forward root",
                    ));
                };
                let mapping_matches = match kernel.io_mapping {
                    StockhamIoMapping::RaderGeneratorCooleyRight(mapping) => {
                        self.io_mapping == StockhamIoMapping::Contiguous
                            && mapping.rader == rader
                            && mapping
                                .caller
                                .validate(self.prime, self.batch_count)
                                .is_ok()
                    }
                    actual => {
                        let expected = if self.io_mapping == StockhamIoMapping::Contiguous {
                            StockhamIoMapping::RaderGeneratorReverse(rader)
                        } else {
                            let caller = RaderFourStepInputMapping::from_stockham(self.io_mapping)
                                .ok_or(VkFftError::InvalidKernelIr(
                                    "fused Rader generator-order input lost its Four-step caller mapping",
                                ))?;
                            StockhamIoMapping::RaderGeneratorFourStep(
                                RaderGeneratorFourStepMapping { rader, caller },
                            )
                        };
                        actual == expected
                    }
                };
                if !mapping_matches || kernel.bindings[0].scalar != self.input_storage_scalar {
                    return Err(VkFftError::InvalidKernelIr(
                        "fused Rader generator-order input does not match its generator/storage boundary",
                    ));
                }
            }
            RaderFftInputStrategy::GeneratorOrderRecursive => {
                let expected = RaderGeneratorMapping {
                    prime: self.prime,
                    generator: self.table.generator,
                };
                let forward_ir = self.forward_recursive().ok_or(VkFftError::InvalidKernelIr(
                    "recursive generator-order Rader input requires a recursive forward child",
                ))?;
                let RecursiveFftNodeIr::CooleyTukey(forward) = &forward_ir.root else {
                    return Err(VkFftError::InvalidKernelIr(
                        "recursive generator-order Rader input requires a Cooley-Tukey forward root",
                    ));
                };
                if forward.pack_right.input_modifier
                    != CooleyTukeyInputModifier::RaderGeneratorReverse(expected)
                    || forward.pack_right.input_storage_scalar != self.input_storage_scalar
                {
                    return Err(VkFftError::InvalidKernelIr(
                        "recursive generator-order Rader input does not match its generator/storage boundary",
                    ));
                }
                let inverse_ir = self.inverse_recursive().ok_or(VkFftError::InvalidKernelIr(
                    "recursive fused Rader inverse requires a recursive inverse child",
                ))?;
                let RecursiveFftNodeIr::CooleyTukey(inverse) = &inverse_ir.root else {
                    return Err(VkFftError::InvalidKernelIr(
                        "recursive fused Rader inverse requires a Cooley-Tukey root",
                    ));
                };
                let normalize_prime = match self.scatter.operation {
                    RaderFftPassOperation::Scatter { normalize } => normalize,
                    _ => {
                        return Err(VkFftError::InvalidKernelIr(
                            "FFT Rader scatter pass has the wrong operation",
                        ));
                    }
                };
                let expected_scatter = RaderScatterMapping {
                    prime: self.prime,
                    generator: self.table.generator,
                    normalize_prime,
                    auxiliary_input: None,
                };
                if inverse.pack_right.input_modifier
                    != CooleyTukeyInputModifier::MultiplyLookupTable
                    || inverse.scatter_output.output_modifier
                        != CooleyTukeyOutputModifier::RaderScatter(expected_scatter)
                    || inverse.scatter_output.output_storage_scalar != self.output_storage_scalar
                    || inverse.scatter_output.auxiliary_storage_scalar != self.input_storage_scalar
                {
                    return Err(VkFftError::InvalidKernelIr(
                        "recursive fused Rader inverse lost its LUT/scatter storage boundary",
                    ));
                }
            }
        }
        let explicit_scatter =
            !self.has_fused_recursive_inverse() && self.fused_inverse_stockham_kernel()?.is_none();
        if explicit_scatter {
            if self.scatter.input_storage_scalar != self.scalar
                || self.scatter.output_storage_scalar != self.output_storage_scalar
                || self.scatter.auxiliary_storage_scalar != self.input_storage_scalar
            {
                return Err(VkFftError::InvalidKernelIr(
                    "explicit FFT Rader scatter lost its output/auxiliary storage boundaries",
                ));
            }
        } else if self.scatter.input_storage_scalar != self.scalar
            || self.scatter.output_storage_scalar != self.scalar
            || self.scatter.auxiliary_storage_scalar != self.scalar
        {
            return Err(VkFftError::InvalidKernelIr(
                "fused FFT Rader pipeline unexpectedly retagged its unused explicit scatter pass",
            ));
        }
        if self.forward_fft.direction() != Direction::Forward
            || self.inverse_fft.direction() != Direction::Inverse
        {
            return Err(VkFftError::InvalidKernelIr(
                "FFT Rader internal FFT directions are invalid",
            ));
        }
        Ok(())
    }

    pub fn kernel_spectrum(&self) -> Result<&[Complex64]> {
        match &self.multiply.operation {
            RaderFftPassOperation::MultiplyKernelSpectrum { spectrum } => Ok(spectrum),
            _ => Err(VkFftError::InvalidKernelIr(
                "FFT Rader multiply pass does not contain a kernel spectrum",
            )),
        }
    }

    /// Whether the inverse convolution tree has fused the kernel-spectrum multiply
    /// into its root pack and the natural-order Rader scatter/DC reconstruction into
    /// its root scatter pass.
    pub(crate) fn has_fused_recursive_inverse(&self) -> bool {
        if self.io_mapping != StockhamIoMapping::Contiguous {
            return false;
        }
        let Some(inverse) = self.inverse_recursive() else {
            return false;
        };
        let RecursiveFftNodeIr::CooleyTukey(node) = &inverse.root else {
            return false;
        };
        matches!(
            node.pack_right.input_modifier,
            CooleyTukeyInputModifier::MultiplyLookupTable
        ) && matches!(
            node.scatter_output.output_modifier,
            CooleyTukeyOutputModifier::RaderScatter(_)
        )
    }

    /// Return an inverse Stockham clone that consumes the forward spectrum and the
    /// immutable Rader kernel spectrum in one dispatch. Recursive/non-Stockham inverse
    /// trees retain the explicit multiply pass.
    pub(crate) fn fused_inverse_stockham_kernel(&self) -> Result<Option<KernelIr>> {
        if self.io_mapping != StockhamIoMapping::Contiguous {
            return Ok(None);
        }
        let Some(inverse) = self.inverse_recursive() else {
            return Ok(None);
        };
        let RecursiveFftNodeIr::Stockham(kernel) = &inverse.root else {
            return Ok(None);
        };
        if kernel.io_mapping != StockhamIoMapping::Contiguous {
            return Ok(None);
        }
        Ok(Some(
            kernel.as_ref().clone().with_lookup_table_input_multiply()?,
        ))
    }

    /// Return the fully fused inverse Rader kernel: stage-0 spectrum multiply plus
    /// final primitive-root scatter/DC/x0 reconstruction. Unsupported recursive
    /// inverse trees retain the explicit multiply/inverse/scatter pipeline.
    pub(crate) fn fused_inverse_rader_kernel(&self) -> Result<Option<KernelIr>> {
        let Some(kernel) = self.fused_inverse_stockham_kernel()? else {
            return Ok(None);
        };
        let normalize_prime = match self.scatter.operation {
            RaderFftPassOperation::Scatter { normalize } => normalize,
            _ => {
                return Err(VkFftError::InvalidKernelIr(
                    "FFT Rader scatter pass has the wrong operation",
                ));
            }
        };
        let kernel = kernel.with_rader_scatter_output(RaderScatterMapping {
            prime: self.prime,
            generator: self.table.generator,
            normalize_prime,
            auxiliary_input: self.cooley_right_auxiliary_mapping(),
        })?;
        let kernel = if self.output_storage_scalar != self.scalar {
            kernel.with_external_output_storage_scalar(self.output_storage_scalar)?
        } else {
            kernel
        };
        let kernel = if self.input_storage_scalar != self.scalar {
            kernel.with_rader_auxiliary_storage_scalar(self.input_storage_scalar)?
        } else {
            kernel
        };
        Ok(Some(kernel))
    }
}

pub fn execute_rader_fft_ir(
    pipeline: &RaderFftPipelineIr,
    input: &[Complex64],
) -> Result<Vec<Complex64>> {
    pipeline.validate()?;
    let expected =
        pipeline
            .prime
            .checked_mul(pipeline.batch_count)
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "FFT Rader input element count",
            })?;
    if input.len() != expected {
        return Err(VkFftError::InputLengthMismatch {
            expected,
            actual: input.len(),
        });
    }
    let convolution_count = pipeline
        .convolution_len
        .checked_mul(pipeline.batch_count)
        .ok_or(VkFftError::ArithmeticOverflow {
            operation: "FFT Rader convolution element count",
        })?;
    let mut spectrum = match pipeline.input_strategy {
        RaderFftInputStrategy::GatherReversePass => {
            let mut gathered = vec![Complex64::new(0.0, 0.0); convolution_count];
            for batch in 0..pipeline.batch_count {
                let convolution_base = batch * pipeline.convolution_len;
                for slot in 0..pipeline.convolution_len {
                    let exponent = (pipeline.convolution_len - slot) % pipeline.convolution_len;
                    let input_index = pipeline.table.permutation[exponent];
                    let mapped_index =
                        pipeline
                            .io_mapping
                            .input_index(pipeline.prime, batch, input_index);
                    gathered[convolution_base + slot] = input[mapped_index];
                }
            }
            execute_one_dim_fft_ir(&pipeline.forward_fft, &gathered)?
        }
        RaderFftInputStrategy::GeneratorOrderStockham => {
            let forward = pipeline
                .forward_recursive()
                .ok_or(VkFftError::InvalidKernelIr(
                    "fused Rader generator-order CPU execution requires a recursive child",
                ))?;
            let RecursiveFftNodeIr::Stockham(kernel) = &forward.root else {
                return Err(VkFftError::InvalidKernelIr(
                    "fused Rader generator-order CPU execution requires a Stockham root",
                ));
            };
            execute_stockham_ir(kernel, input)?
        }
        RaderFftInputStrategy::GeneratorOrderRecursive => {
            let forward = pipeline
                .forward_recursive()
                .ok_or(VkFftError::InvalidKernelIr(
                    "recursive generator-order CPU execution requires a recursive child",
                ))?;
            execute_recursive_fft_ir_with_resources(forward, input, None, None)?
        }
    };
    let kernel_spectrum = pipeline.kernel_spectrum()?;
    if let Some(fused_inverse) = pipeline.fused_inverse_rader_kernel()? {
        return execute_stockham_ir_with_resources(
            &fused_inverse,
            &spectrum,
            Some(kernel_spectrum),
            Some(input),
        );
    }
    if pipeline.has_fused_recursive_inverse() {
        let inverse = pipeline
            .inverse_recursive()
            .ok_or(VkFftError::InvalidKernelIr(
                "fused recursive CPU execution requires a recursive inverse child",
            ))?;
        return execute_recursive_fft_ir_with_resources(
            inverse,
            &spectrum,
            Some(kernel_spectrum),
            Some(input),
        );
    }
    let convolution = if let Some(fused_inverse) = pipeline.fused_inverse_stockham_kernel()? {
        execute_stockham_ir_with_lookup(&fused_inverse, &spectrum, Some(kernel_spectrum))?
    } else {
        for batch in 0..pipeline.batch_count {
            let base = batch * pipeline.convolution_len;
            for index in 0..pipeline.convolution_len {
                spectrum[base + index] *= kernel_spectrum[index];
            }
        }
        execute_one_dim_fft_ir(&pipeline.inverse_fft, &spectrum)?
    };
    let normalize = match pipeline.scatter.operation {
        RaderFftPassOperation::Scatter { normalize } => normalize,
        _ => {
            return Err(VkFftError::InvalidKernelIr(
                "FFT Rader scatter pass has the wrong operation",
            ));
        }
    };
    let scale = if normalize {
        1.0 / pipeline.prime as f64
    } else {
        1.0
    };
    let mut output = vec![Complex64::new(0.0, 0.0); expected];
    for batch in 0..pipeline.batch_count {
        let convolution_base = batch * pipeline.convolution_len;
        let read = |local_index: usize| {
            input[pipeline
                .io_mapping
                .input_index(pipeline.prime, batch, local_index)]
        };
        let dc = (0..pipeline.prime)
            .map(read)
            .fold(Complex64::new(0.0, 0.0), |sum, value| sum + value)
            .scale(scale);
        let sign = match pipeline.direction {
            Direction::Forward => -1.0,
            Direction::Inverse => 1.0,
        };
        let (dc_index, dc_value) =
            pipeline
                .io_mapping
                .map_output(pipeline.prime, batch, 0, dc, sign);
        output[dc_index] = dc_value;
        for exponent in 0..pipeline.convolution_len {
            let local_output_index = pipeline.table.permutation[exponent];
            let value = (read(0) + convolution[convolution_base + exponent]).scale(scale);
            let (output_index, value) = pipeline.io_mapping.map_output(
                pipeline.prime,
                batch,
                local_output_index,
                value,
                sign,
            );
            output[output_index] = value;
        }
    }
    Ok(output)
}

fn is_rader_register_schedule_fallback(error: &VkFftError) -> bool {
    matches!(
        error,
        VkFftError::UnsupportedKernelPath(_) | VkFftError::ResourceLimitExceeded { .. }
    )
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
    use crate::FftConfig;
    use crate::config::{Backend, GpuVendor};
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
                Complex64::new((0.19 * x).sin() + x * 0.001, (0.05 * x).cos() - x * 0.002)
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
    fn direct_rader_ir_matches_dft() {
        let length = 47usize;
        let batch_count = 32usize;
        for direction in [Direction::Forward, Direction::Inverse] {
            let config = FftConfig::new(vec![length])
                .with_batch_count(batch_count)
                .with_inverse_normalization(direction == Direction::Inverse);
            let plan = FftPlan::build(config).unwrap();
            let ir = RaderDirectIr::build(&plan, direction, device()).unwrap();
            let block = ir
                .axis_batch_block
                .expect("standalone p47 should use the direct-Rader axis block");
            assert_eq!(block.threads_per_transform, 24);
            assert_eq!(block.grouped_batch, 5);
            assert_eq!([ir.workgroup_size.x, ir.workgroup_size.y], [24, 5]);
            assert_eq!(ir.dispatch.x, 7);
            let input = sample(length, batch_count);
            let actual = execute_rader_direct_ir(&ir, &input).unwrap();
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
    fn fft_rader_cooley_right_input_matches_explicit_parent_pack_and_dc() {
        let parent_left_len = 3usize;
        let prime = 17usize;
        let parent_logical_len = parent_left_len * prime;
        let parent_batch_count = 2usize;
        let rader_batch_count = parent_batch_count * parent_left_len;
        let natural = sample(parent_logical_len, parent_batch_count);
        let mut packed = vec![Complex64::new(0.0, 0.0); natural.len()];
        for parent_batch in 0..parent_batch_count {
            for n1 in 0..parent_left_len {
                let rader_batch = parent_batch * parent_left_len + n1;
                for n2 in 0..prime {
                    packed[rader_batch * prime + n2] =
                        natural[parent_batch * parent_logical_len + n1 + parent_left_len * n2];
                }
            }
        }
        let caller = crate::kernel_ir::CooleyRightInputMapping {
            parent_logical_len,
            parent_left_len,
            parent_right_len: prime,
            parent_batch_count,
        };

        for direction in [Direction::Forward, Direction::Inverse] {
            let config = FftConfig::new(vec![prime])
                .with_batch_count(rader_batch_count)
                .with_inverse_normalization(direction == Direction::Inverse);
            let plan = FftPlan::build(config).unwrap();
            let baseline = RaderFftPipelineIr::build(&plan, direction, device()).unwrap();
            assert_eq!(
                baseline.input_strategy,
                RaderFftInputStrategy::GeneratorOrderStockham
            );
            let expected = execute_rader_fft_ir(&baseline, &packed).unwrap();

            let mapped = baseline
                .clone()
                .with_cooley_right_input_mapping(caller)
                .unwrap();
            let forward = mapped.forward_recursive().unwrap();
            let RecursiveFftNodeIr::Stockham(forward_root) = &forward.root else {
                panic!("p17 mapped forward convolution should remain Stockham");
            };
            assert!(matches!(
                forward_root.io_mapping,
                StockhamIoMapping::RaderGeneratorCooleyRight(mapping)
                    if mapping.caller == caller
            ));
            let inverse = mapped
                .fused_inverse_rader_kernel()
                .unwrap()
                .expect("p17 mapped pipeline should retain fused inverse scatter");
            assert!(matches!(
                inverse.output_modifier,
                crate::kernel_ir::StockhamOutputModifier::RaderScatter(mapping)
                    if mapping.auxiliary_input == Some(caller)
            ));
            let actual = execute_rader_fft_ir(&mapped, &natural).unwrap();
            assert!(max_error(&actual, &expected) < 2.0e-9 * prime as f64);
        }
    }

    #[test]
    fn direct_rader_grouped_batch_override_matches_upstream_user_branch() {
        let config = FftConfig::new(vec![47])
            .with_batch_count(32)
            .with_grouped_batch(0, 3)
            .unwrap();
        let plan = FftPlan::build(config).unwrap();
        let ir = RaderDirectIr::build(&plan, Direction::Forward, device()).unwrap();
        let block = ir.axis_batch_block.unwrap();
        assert_eq!(block.threads_per_transform, 24);
        assert_eq!(block.grouped_batch, 1);
        assert_eq!([ir.workgroup_size.x, ir.workgroup_size.y], [24, 1]);
        assert_eq!(ir.dispatch.x, 32);
    }

    #[test]
    fn fft_rader_ir_matches_dft() {
        let batch_count = 2usize;
        for length in [17usize, 19, 29, 31, 37, 41, 43, 257] {
            for direction in [Direction::Forward, Direction::Inverse] {
                let config = FftConfig::new(vec![length])
                    .with_batch_count(batch_count)
                    .with_inverse_normalization(direction == Direction::Inverse);
                let plan = FftPlan::build(config).unwrap();
                let pipeline = RaderFftPipelineIr::build(&plan, direction, device()).unwrap();
                assert_eq!(pipeline.convolution_len, length - 1);
                assert_eq!(pipeline.kernel_spectrum().unwrap().len(), length - 1);
                assert!(pipeline.internal_register_schedule.is_some());
                let input = sample(length, batch_count);
                let actual = execute_rader_fft_ir(&pipeline, &input).unwrap();
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
    }

    #[test]
    fn nested_rader_convolution_child_uses_physical_contiguous_device_scoring() {
        let tuning = crate::PlannerTuning::portable().with_recursive_fft_rader(true);
        let child_config = FftConfig::new(vec![106]).with_tuning(tuning);

        let roomy = device();
        let roomy_child = build_rader_convolution_child_plan(child_config.clone(), roomy).unwrap();
        let AxisAlgorithm::Rader { primes, .. } = &roomy_child.axes[0].algorithm else {
            panic!("roomy 106-point convolution should stay a Rader composite");
        };
        let p53 = primes
            .iter()
            .find(|prime| prime.prime == 53)
            .expect("106-point convolution should contain p53");
        assert!(matches!(p53.mode, RaderMode::FftConvolution { .. }));

        let mut constrained = roomy;
        constrained.shared_memory_bytes = 256;
        constrained.shared_memory_pow2_bytes = 256;
        let constrained_child =
            build_rader_convolution_child_plan(child_config, constrained).unwrap();
        let AxisAlgorithm::Rader { primes, .. } = &constrained_child.axes[0].algorithm else {
            panic!("constrained 106-point convolution should stay a Rader composite");
        };
        let p53 = primes
            .iter()
            .find(|prime| prime.prime == 53)
            .expect("106-point convolution should contain p53");
        assert_eq!(p53.mode, RaderMode::DirectMultiplication);

        let outer_plan = FftPlan::build(
            FftConfig::new(vec![107])
                .with_tuning(crate::PlannerTuning::portable().with_recursive_fft_rader(true)),
        )
        .unwrap();
        for (profile, expect_fft_child) in [(roomy, true), (constrained, false)] {
            let pipeline =
                RaderFftPipelineIr::build(&outer_plan, Direction::Forward, profile).unwrap();
            let RecursiveFftNodeIr::CooleyTukey(root) = &pipeline.forward_recursive().unwrap().root
            else {
                panic!("p107 convolution should split 106 into a recursive tree");
            };
            assert_eq!((root.left_len, root.right_len), (2, 53));
            assert_eq!(
                matches!(root.right, RecursiveFftNodeIr::FftRader(_)),
                expect_fft_child
            );
            assert_eq!(
                matches!(root.right, RecursiveFftNodeIr::DirectRader(_)),
                !expect_fft_child
            );

            let input = sample(107, 1);
            let actual = execute_rader_fft_ir(&pipeline, &input).unwrap();
            let expected = dft(&input, Direction::Forward, false);
            assert!(max_error(&actual, &expected) < 2.0e-9 * 107.0);
        }
    }

    #[test]
    fn generator_stockham_four_step_composition_matches_explicit_gather_for_all_upload_positions() {
        use crate::kernel_ir::{FourStepMapping, ThreeUploadFourStepMapping};

        let cases = [
            (
                2usize,
                StockhamIoMapping::FourStepRight(FourStepMapping {
                    logical_len: 62,
                    left_len: 2,
                    right_len: 31,
                    outer_batch_count: 1,
                }),
            ),
            (
                2usize,
                StockhamIoMapping::FourStepLeft(FourStepMapping {
                    logical_len: 62,
                    left_len: 31,
                    right_len: 2,
                    outer_batch_count: 1,
                }),
            ),
            (
                6usize,
                StockhamIoMapping::FourStepThreeUpload2(ThreeUploadFourStepMapping {
                    logical_len: 186,
                    axis_split: [2, 3, 31],
                    outer_batch_count: 1,
                }),
            ),
            (
                6usize,
                StockhamIoMapping::FourStepThreeUpload1(ThreeUploadFourStepMapping {
                    logical_len: 186,
                    axis_split: [2, 31, 3],
                    outer_batch_count: 1,
                }),
            ),
            (
                6usize,
                StockhamIoMapping::FourStepThreeUpload0(ThreeUploadFourStepMapping {
                    logical_len: 186,
                    axis_split: [31, 2, 3],
                    outer_batch_count: 1,
                }),
            ),
        ];

        for (batch_count, mapping) in cases {
            let plan =
                FftPlan::build(FftConfig::new(vec![31]).with_batch_count(batch_count)).unwrap();
            let base = RaderFftPipelineIr::build(&plan, Direction::Forward, device()).unwrap();
            assert_eq!(
                base.input_strategy,
                RaderFftInputStrategy::GeneratorOrderStockham
            );
            let fused = base.clone().with_stockham_io_mapping(mapping).unwrap();
            assert_eq!(
                fused.input_strategy,
                RaderFftInputStrategy::GeneratorOrderStockham
            );
            let forward = fused.forward_recursive().unwrap();
            let RecursiveFftNodeIr::Stockham(kernel) = &forward.root else {
                panic!("p31 forward convolution should remain Stockham");
            };
            assert!(matches!(
                kernel.io_mapping,
                StockhamIoMapping::RaderGeneratorFourStep(_)
            ));

            let mut explicit = fused.clone();
            explicit.input_strategy = RaderFftInputStrategy::GatherReversePass;
            let explicit_forward = explicit.forward_recursive_mut().unwrap();
            let RecursiveFftNodeIr::Stockham(explicit_kernel) = &mut explicit_forward.root else {
                panic!("explicit p31 forward convolution should remain Stockham");
            };
            **explicit_kernel = explicit_kernel
                .as_ref()
                .clone()
                .with_stockham_io_mapping(StockhamIoMapping::Contiguous)
                .unwrap();
            explicit.validate().unwrap();

            let input = sample(31, batch_count);
            let fused_output = execute_rader_fft_ir(&fused, &input).unwrap();
            let explicit_output = execute_rader_fft_ir(&explicit, &input).unwrap();
            assert!(
                max_error(&fused_output, &explicit_output) < 2.0e-11,
                "composed generator/Four-step mapping {mapping:?} diverged from explicit gather"
            );
        }
    }

    #[test]
    fn fft_rader_can_embed_a_bluestein_convolution_child() {
        let mut tuning = crate::PlannerTuning::portable();
        tuning.min_rader_direct_prime = 29;
        tuning.min_rader_fft_prime = 29;
        tuning.validate().unwrap();

        let child_plan = build_rader_convolution_child_plan(
            FftConfig::new(vec![102]).with_tuning(tuning),
            device(),
        )
        .unwrap();
        assert!(matches!(
            child_plan.axes[0].algorithm,
            AxisAlgorithm::Bluestein { .. }
        ));

        let forward_plan = FftPlan::build(FftConfig::new(vec![103]).with_tuning(tuning)).unwrap();
        let AxisAlgorithm::Rader { primes, .. } = &forward_plan.axes[0].algorithm else {
            panic!("custom p103 should still enter FFT-Rader through the safe-prime gate");
        };
        assert!(matches!(primes[0].mode, RaderMode::FftConvolution { .. }));
        let forward =
            RaderFftPipelineIr::build(&forward_plan, Direction::Forward, device()).unwrap();
        assert_eq!(
            forward.input_strategy,
            RaderFftInputStrategy::GatherReversePass
        );
        assert!(matches!(
            forward.forward_fft.as_ref(),
            OneDimFftIr::Bluestein(_)
        ));
        assert!(matches!(
            forward.inverse_fft.as_ref(),
            OneDimFftIr::Bluestein(_)
        ));
        assert!(forward.internal_register_schedule.is_none());
        assert!(!forward.has_fused_recursive_inverse());
        assert!(forward.fused_inverse_stockham_kernel().unwrap().is_none());

        let input = sample(103, 1);
        let spectrum = execute_rader_fft_ir(&forward, &input).unwrap();
        let expected = dft(&input, Direction::Forward, false);
        assert!(max_error(&spectrum, &expected) < 2.0e-9 * 103.0);

        let inverse_plan = FftPlan::build(
            FftConfig::new(vec![103])
                .with_tuning(tuning)
                .with_inverse_normalization(true),
        )
        .unwrap();
        let inverse =
            RaderFftPipelineIr::build(&inverse_plan, Direction::Inverse, device()).unwrap();
        let restored = execute_rader_fft_ir(&inverse, &spectrum).unwrap();
        assert!(max_error(&restored, &input) < 4.0e-9 * 103.0);

        crate::ProgramIr::rader_fft(&forward)
            .unwrap()
            .validate()
            .unwrap();
    }

    #[test]
    fn power_of_two_rader_container_installs_upstream_register_schedule() {
        let batch_count = 2usize;
        let plan = FftPlan::build(FftConfig::new(vec![257]).with_batch_count(batch_count)).unwrap();
        let pipeline = RaderFftPipelineIr::build(&plan, Direction::Forward, device()).unwrap();
        let schedule = pipeline
            .internal_register_schedule
            .as_ref()
            .expect("257-point FFT Rader should have a power-of-two container schedule");
        assert_eq!(schedule.internal_fft.stage_radices, vec![16, 16]);
        assert_eq!(schedule.container_fft_num, 1);
        assert_eq!(schedule.min_rader_fft_thread_num, 16);
        assert_eq!(schedule.execution_container_fft_num, 1);
        assert_eq!(schedule.execution_threads_per_workgroup, 16);
        assert!(schedule.upstream_grouping_is_executable());
        let axis_block = pipeline
            .axis_batch_block
            .expect("batched p257 should carry the independent axis-level Rader block");
        assert_eq!(axis_block.threads_per_transform, 17);
        assert_eq!(axis_block.grouped_batch, 2);
        assert_eq!([axis_block.local_size_x, axis_block.local_size_y], [17, 2]);
        for fft in [
            pipeline.forward_recursive().unwrap(),
            pipeline.inverse_recursive().unwrap(),
        ] {
            assert!(fft.stockham_upload_schedule.is_none());
            assert!(fft.four_step_plan.is_none());
            let RecursiveFftNodeIr::Stockham(kernel) = &fft.root else {
                panic!("scheduled Rader container must stay a Stockham root");
            };
            assert_eq!(
                kernel.execution_layout,
                crate::StockhamExecutionLayout::RegisterSingleShared
            );
            assert_eq!([kernel.workgroup_size.x, kernel.workgroup_size.y], [17, 2]);
            assert_eq!(kernel.dispatch.x, 1);
            assert_eq!(kernel.workgroup_grouping.transforms_per_workgroup, 2);
            assert_eq!(kernel.workgroup_grouping.threads_per_transform, 17);
            assert_eq!(kernel.scheduler_hint.as_ref(), Some(&schedule.internal_fft));
            let stages = kernel.register_stockham_stages().unwrap().unwrap();
            assert_eq!(stages.len(), 2);
            assert!(stages.iter().all(|stage| stage.virtual_thread_count == 16));
        }
    }

    #[test]
    fn batch_one_fft_rader_keeps_prime_caller_floor() {
        let mut profile = device();
        profile.max_threads_per_block = 1024;
        profile.max_workgroup_size = [1024, 1024, 64];
        let plan = FftPlan::build(FftConfig::new(vec![257])).unwrap();
        let pipeline = RaderFftPipelineIr::build(&plan, Direction::Forward, profile).unwrap();
        let schedule = pipeline.internal_register_schedule.as_ref().unwrap();
        assert_eq!(schedule.execution_threads_per_workgroup, 16);
        let block = pipeline
            .axis_batch_block
            .expect("batch-one p257 must preserve the 17-lane prime caller floor");
        assert_eq!(block.threads_per_transform, 17);
        assert_eq!(block.grouped_batch, 1);
        assert_eq!([block.local_size_x, block.local_size_y], [17, 1]);
        for fft in [
            pipeline.forward_recursive().unwrap(),
            pipeline.inverse_recursive().unwrap(),
        ] {
            let RecursiveFftNodeIr::Stockham(kernel) = &fft.root else {
                panic!("batch-one p257 convolution must remain a Stockham root");
            };
            assert_eq!([kernel.workgroup_size.x, kernel.workgroup_size.y], [17, 1]);
            assert_eq!(kernel.dispatch.x, 1);
            assert_eq!(kernel.workgroup_grouping.transforms_per_workgroup, 1);
            assert_eq!(kernel.workgroup_grouping.threads_per_transform, 17);
            assert!(
                kernel
                    .register_stockham_stages()
                    .unwrap()
                    .unwrap()
                    .iter()
                    .all(|stage| stage.virtual_thread_count == 16)
            );
        }
    }

    #[test]
    fn p7681_batch_one_native_path_keeps_769_lane_caller_floor() {
        let mut profile = device();
        profile.backend = Backend::Cuda;
        profile.shared_memory_bytes = 64 * 1024;
        profile.shared_memory_pow2_bytes = 64 * 1024;
        profile.max_threads_per_block = 1024;
        profile.max_workgroup_size = [1024, 1024, 64];
        let length = 7_681usize;
        let plan = FftPlan::build_for_device(FftConfig::new(vec![length]), profile).unwrap();
        let AxisAlgorithm::Rader { .. } = plan.axes[0].algorithm else {
            panic!("64 KiB device-scored p7681 must remain FFT-Rader");
        };
        let pipeline = RaderFftPipelineIr::build(&plan, Direction::Forward, profile).unwrap();
        let schedule = pipeline
            .internal_register_schedule
            .as_ref()
            .expect("native p7681 must consume its 768-lane convolution schedule");
        assert_eq!(schedule.execution_threads_per_workgroup, 768);
        let block = pipeline
            .axis_batch_block
            .expect("batch-one p7681 must preserve the 769-lane prime caller floor");
        assert_eq!(block.threads_per_transform, 769);
        assert_eq!(block.grouped_batch, 1);
        assert_eq!([block.local_size_x, block.local_size_y], [769, 1]);
        assert_eq!(
            pipeline.input_strategy,
            RaderFftInputStrategy::GeneratorOrderStockham
        );
        for fft in [
            pipeline.forward_recursive().unwrap(),
            pipeline.inverse_recursive().unwrap(),
        ] {
            let RecursiveFftNodeIr::Stockham(kernel) = &fft.root else {
                panic!("native p7681 convolution must remain a Stockham root");
            };
            assert_eq!([kernel.workgroup_size.x, kernel.workgroup_size.y], [769, 1]);
            assert_eq!(kernel.dispatch.x, 1);
            assert_eq!(kernel.workgroup_grouping.threads_per_transform, 769);
            assert!(
                kernel
                    .register_stockham_stages()
                    .unwrap()
                    .unwrap()
                    .iter()
                    .all(|stage| stage.virtual_thread_count <= 768)
            );
        }
        crate::ProgramIr::rader_fft(&pipeline)
            .unwrap()
            .validate()
            .unwrap();
    }

    #[test]
    fn p257_axis_batching_keeps_container_count_one_and_allows_partial_final_group() {
        let mut profile = device();
        profile.shared_memory_bytes = 48 * 1024;
        profile.shared_memory_pow2_bytes = 32 * 1024;
        profile.max_threads_per_block = 1024;
        profile.max_workgroup_size = [1024, 1024, 64];
        let batch_count = 32usize;
        let plan = FftPlan::build(FftConfig::new(vec![257]).with_batch_count(batch_count)).unwrap();
        let pipeline = RaderFftPipelineIr::build(&plan, Direction::Forward, profile).unwrap();
        let schedule = pipeline.internal_register_schedule.as_ref().unwrap();
        assert_eq!(schedule.container_fft_num, 1);
        assert_eq!(schedule.execution_container_fft_num, 1);
        assert!(schedule.rader_transpose.is_none());
        let block = pipeline.axis_batch_block.unwrap();
        assert_eq!(block.threads_per_transform, 17);
        assert_eq!(block.grouped_batch, 7);
        assert_eq!([block.local_size_x, block.local_size_y], [17, 7]);
        for fft in [
            pipeline.forward_recursive().unwrap(),
            pipeline.inverse_recursive().unwrap(),
        ] {
            let RecursiveFftNodeIr::Stockham(kernel) = &fft.root else {
                panic!("p257 axis batching should keep a Stockham convolution root");
            };
            assert_eq!(kernel.workgroup_grouping.transforms_per_workgroup, 7);
            assert_eq!(kernel.workgroup_grouping.threads_per_transform, 17);
            assert_eq!([kernel.workgroup_size.x, kernel.workgroup_size.y], [17, 7]);
            assert_eq!(kernel.dispatch.x, 5);
            assert!(kernel.rader_transpose.is_none());
        }

        let input = sample(257, batch_count);
        let actual = execute_rader_fft_ir(&pipeline, &input).unwrap();
        for batch in 0..batch_count {
            let start = batch * 257;
            let expected = dft(&input[start..start + 257], Direction::Forward, false);
            assert!(max_error(&actual[start..start + 257], &expected) < 2.0e-9 * 257.0);
        }
    }

    #[test]
    fn power_of_two_rader_register_schedule_scales_to_device_thread_limit() {
        let mut constrained = device();
        constrained.max_threads_per_block = 8;
        let plan = FftPlan::build(FftConfig::new(vec![257])).unwrap();
        let pipeline = RaderFftPipelineIr::build(&plan, Direction::Forward, constrained).unwrap();
        let schedule = pipeline
            .internal_register_schedule
            .as_ref()
            .expect("p257 should scale register storage until its internal FFT fits eight threads");
        assert_eq!(schedule.execution_threads_per_workgroup, 8);
        assert_eq!(schedule.internal_fft.stage_radices, vec![16, 16]);
        assert!(schedule.internal_fft.registers_per_thread_per_radix[16] >= 32);
        let input = sample(257, 1);
        let actual = execute_rader_fft_ir(&pipeline, &input).unwrap();
        let expected = dft(&input, Direction::Forward, false);
        assert!(max_error(&actual, &expected) < 2.0e-9 * 257.0);
    }

    #[test]
    fn explicit_gather_scatter_mixed_storage_tracks_input_output_and_auxiliary() {
        let mut profile = device();
        profile.shared_memory_bytes = 8 * 1024;
        profile.shared_memory_pow2_bytes = 8 * 1024;
        profile.supports_f64 = true;

        for (precision, compute, storage) in [
            (Precision::F32, ScalarType::F32, ScalarType::F16),
            (Precision::F64, ScalarType::F64, ScalarType::F32),
        ] {
            let plan =
                FftPlan::build(FftConfig::new(vec![12_289]).with_precision(precision)).unwrap();
            let pipeline = RaderFftPipelineIr::build(&plan, Direction::Forward, profile)
                .unwrap()
                .with_external_input_storage_scalar(storage)
                .unwrap()
                .with_external_output_storage_scalar(storage)
                .unwrap();
            assert_eq!(
                pipeline.input_strategy,
                RaderFftInputStrategy::GatherReversePass
            );
            assert!(
                pipeline
                    .forward_recursive()
                    .unwrap()
                    .four_step_plan
                    .is_some()
            );
            assert!(
                pipeline
                    .inverse_recursive()
                    .unwrap()
                    .four_step_plan
                    .is_some()
            );
            assert!(pipeline.fused_inverse_stockham_kernel().unwrap().is_none());
            assert!(!pipeline.has_fused_recursive_inverse());
            assert_eq!(pipeline.scalar, compute);
            assert_eq!(pipeline.input_storage_scalar, storage);
            assert_eq!(pipeline.output_storage_scalar, storage);
            assert_eq!(pipeline.gather.input_storage_scalar, storage);
            assert_eq!(pipeline.gather.output_storage_scalar, compute);
            assert_eq!(pipeline.gather.auxiliary_storage_scalar, compute);
            assert_eq!(pipeline.multiply.input_storage_scalar, compute);
            assert_eq!(pipeline.multiply.output_storage_scalar, compute);
            assert_eq!(pipeline.scatter.input_storage_scalar, compute);
            assert_eq!(pipeline.scatter.output_storage_scalar, storage);
            assert_eq!(pipeline.scatter.auxiliary_storage_scalar, storage);
            pipeline.validate().unwrap();

            let program = crate::ProgramIr::rader_fft(&pipeline).unwrap();
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

    #[test]
    fn large_fft_rader_convolution_splits_internal_fft() {
        let length = 4001usize;
        let mut limited = device();
        limited.shared_memory_bytes = 16 * 128;
        limited.shared_memory_pow2_bytes = limited.shared_memory_bytes;
        let plan = FftPlan::build(FftConfig::new(vec![length])).unwrap();
        let pipeline = RaderFftPipelineIr::build(&plan, Direction::Forward, limited).unwrap();
        assert_eq!(pipeline.convolution_len, 4000);
        assert!(matches!(
            pipeline.forward_recursive().unwrap().root,
            crate::RecursiveFftNodeIr::CooleyTukey(_)
        ));
        assert_eq!(
            pipeline.input_strategy,
            RaderFftInputStrategy::GeneratorOrderRecursive
        );
        let RecursiveFftNodeIr::CooleyTukey(forward) = &pipeline.forward_recursive().unwrap().root
        else {
            unreachable!();
        };
        assert!(matches!(
            forward.pack_right.input_modifier,
            CooleyTukeyInputModifier::RaderGeneratorReverse(_)
        ));
        assert!(pipeline.fused_inverse_stockham_kernel().unwrap().is_none());
        assert!(pipeline.fused_inverse_rader_kernel().unwrap().is_none());
        assert!(pipeline.has_fused_recursive_inverse());
        let input = sample(length, 1);
        let actual = execute_rader_fft_ir(&pipeline, &input).unwrap();
        let expected = crate::reference::fft(&input, Direction::Forward, false).unwrap();
        assert!(max_error(&actual, &expected) < 5.0e-8 * length as f64);
    }

    #[test]
    fn fft_rader_mixed_storage_keeps_convolution_resources_in_compute_precision() {
        for (precision, compute, storage) in [
            (Precision::F32, ScalarType::F32, ScalarType::F16),
            (Precision::F64, ScalarType::F64, ScalarType::F32),
        ] {
            let plan = FftPlan::build(FftConfig::new(vec![19]).with_precision(precision)).unwrap();
            let pipeline = RaderFftPipelineIr::build(&plan, Direction::Forward, device())
                .unwrap()
                .with_external_input_storage_scalar(storage)
                .unwrap()
                .with_external_output_storage_scalar(storage)
                .unwrap();
            assert_eq!(pipeline.scalar, compute);
            assert_eq!(pipeline.input_storage_scalar, storage);
            assert_eq!(pipeline.output_storage_scalar, storage);
            assert_eq!(
                pipeline.input_strategy,
                RaderFftInputStrategy::GeneratorOrderStockham
            );
            let RecursiveFftNodeIr::Stockham(forward) = &pipeline.forward_recursive().unwrap().root
            else {
                panic!("p19 mixed-storage forward Rader convolution should remain Stockham");
            };
            assert_eq!(forward.bindings[0].scalar, storage);
            assert_eq!(forward.bindings[1].scalar, compute);
            assert!(
                forward
                    .bindings
                    .iter()
                    .skip(1)
                    .all(|binding| binding.scalar == compute)
            );

            let inverse = pipeline
                .fused_inverse_rader_kernel()
                .unwrap()
                .expect("p19 mixed-storage inverse should keep fused Stockham scatter");
            assert_eq!(inverse.bindings[0].scalar, compute);
            assert_eq!(inverse.bindings[1].scalar, storage);
            assert!(inverse.bindings.iter().any(|binding| {
                binding.role == crate::BufferRole::LookupTable && binding.scalar == compute
            }));
            assert!(inverse.bindings.iter().any(|binding| {
                binding.role == crate::BufferRole::Auxiliary && binding.scalar == storage
            }));

            let program = crate::ProgramIr::rader_fft(&pipeline).unwrap();
            assert_eq!(program.input_resource().unwrap().scalar, storage);
            assert_eq!(program.output_resource().unwrap().scalar, storage);
            assert!(program.resources.iter().all(|resource| {
                matches!(
                    resource.kind,
                    crate::ProgramResourceKind::Input | crate::ProgramResourceKind::Output
                ) || resource.scalar == compute
            }));
        }

        let length = 4001usize;
        let mut limited = device();
        limited.shared_memory_bytes = 16 * 128;
        limited.shared_memory_pow2_bytes = limited.shared_memory_bytes;
        let plan = FftPlan::build(FftConfig::new(vec![length])).unwrap();
        let pipeline = RaderFftPipelineIr::build(&plan, Direction::Forward, limited)
            .unwrap()
            .with_external_input_storage_scalar(ScalarType::F16)
            .unwrap()
            .with_external_output_storage_scalar(ScalarType::F16)
            .unwrap();
        assert_eq!(
            pipeline.input_strategy,
            RaderFftInputStrategy::GeneratorOrderRecursive
        );
        let RecursiveFftNodeIr::CooleyTukey(forward) = &pipeline.forward_recursive().unwrap().root
        else {
            panic!("p4001 mixed-storage forward should remain recursive");
        };
        let RecursiveFftNodeIr::CooleyTukey(inverse) = &pipeline.inverse_recursive().unwrap().root
        else {
            panic!("p4001 mixed-storage inverse should remain recursive");
        };
        assert_eq!(forward.pack_right.input_storage_scalar, ScalarType::F16);
        assert_eq!(forward.pack_right.output_storage_scalar, ScalarType::F32);
        assert_eq!(inverse.scatter_output.input_storage_scalar, ScalarType::F32);
        assert_eq!(
            inverse.scatter_output.output_storage_scalar,
            ScalarType::F16
        );
    }

    #[test]
    fn large_mixed_rader_internal_fft_consumes_physical_register_schedule() {
        let length = 4001usize;
        let mut wide = device();
        wide.max_threads_per_block = 1024;
        wide.max_workgroup_size = [1024, 1024, 64];
        wide.shared_memory_bytes = 48 * 1024;
        wide.shared_memory_pow2_bytes = 32 * 1024;
        let plan = FftPlan::build(FftConfig::new(vec![length])).unwrap();
        let scheduled =
            plan_gpu_rader_fft_registers_for_containers(length, 1, length, 1, wide).unwrap();
        assert!(scheduled.execution_threads_per_workgroup > 256);
        assert!(scheduled.execution_threads_per_workgroup <= wide.max_threads_per_block);
        let pipeline = RaderFftPipelineIr::build(&plan, Direction::Forward, wide).unwrap();
        assert_eq!(pipeline.convolution_len, 4000);
        let device_scored = pipeline
            .device_register_schedule
            .as_ref()
            .expect("p4001 should retain its physical-device register-Rader score");
        assert_eq!(device_scored, &scheduled);
        assert!(pipeline.internal_register_schedule.is_none());
        assert!(matches!(
            pipeline.forward_recursive().unwrap().root,
            crate::RecursiveFftNodeIr::CooleyTukey(_)
        ));
        assert!(matches!(
            pipeline.inverse_recursive().unwrap().root,
            crate::RecursiveFftNodeIr::CooleyTukey(_)
        ));
        assert_eq!(
            pipeline.input_strategy,
            RaderFftInputStrategy::GeneratorOrderRecursive
        );
        crate::ProgramIr::rader_fft(&pipeline)
            .unwrap()
            .validate()
            .unwrap();
        let backend = crate::backend::vulkan::VulkanGlslBackend;
        for recursive in [
            pipeline.forward_recursive().unwrap(),
            pipeline.inverse_recursive().unwrap(),
        ] {
            for shader in backend.lower_recursive_fft(recursive).unwrap() {
                assert_eq!(shader.compile_spirv().unwrap().words[0], 0x0723_0203);
            }
        }
    }

    #[test]
    fn fft_rader_ir_rejects_direct_prime() {
        let plan = FftPlan::build(FftConfig::new(vec![47])).unwrap();
        let error = RaderFftPipelineIr::build(&plan, Direction::Forward, device()).unwrap_err();
        assert!(matches!(error, VkFftError::UnsupportedKernelPath(_)));
    }

    #[test]
    fn direct_rader_ir_rejects_fft_convolution_prime() {
        let plan = FftPlan::build(FftConfig::new(vec![17])).unwrap();
        let error = RaderDirectIr::build(&plan, Direction::Forward, device()).unwrap_err();
        assert!(matches!(error, VkFftError::UnsupportedKernelPath(_)));
    }
}
