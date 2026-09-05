//! Executable Stockham IR for VkFFT's double-double compute modes.
//!
//! This layer stays backend-neutral: it reuses the planner's Stockham radix
//! decomposition and stage/indexing contract while carrying real double-double
//! twiddles and arithmetic. GPU lowering must not alias this mode to plain F64.

use crate::complex::Complex64;
use crate::config::{
    DctType, DeviceProfile, Direction, DstType, FftConfig, PlannerTuning, Precision,
    PrecisionStorage, TransformKind, ZeroPaddingDomain, ZeroPaddingRange,
};
use crate::double_double::{ComplexDoubleDouble, DoubleDouble, unit_root};
use crate::double_double_recursive_ir::{
    DoubleDoubleRecursiveFftIr, execute_double_double_recursive_ir,
    execute_double_double_recursive_ir_f64_storage,
};
use crate::error::{Result, VkFftError};
use crate::kernel_ir::{ScalarType, SharedBuffer, StockhamIoMapping, StockhamStage};
use crate::lut::{
    DoubleDoubleBluesteinTable, DoubleDoubleRaderTable, DoubleDoubleStockhamTwiddleStage,
    DoubleDoubleStockhamTwiddleTable,
};
use crate::nd_ir::{
    NdExternalTensorLayout, NdFormattedCopyOperation, NdFormattedCopyPassIr,
    pack_logical_tensor_batches, unpack_logical_tensor_batches,
};
use crate::planner::{AxisAlgorithm, C2cDeviceAxisClass, FftPlan, RaderMode, RadixPlan};
use crate::r2r_ir::{R2rTransform, inverse_partner};
use crate::real_ir::{
    RealFftAlgorithm, RealFftKind, RealFftShapePolicy, real_fft_algorithm_for_shape_policy,
};
use crate::scheduler::{
    StockhamAxisBlockSchedule, StockhamUploadAxisContext, plan_gpu_axis0_direct_rader_batch_block,
    plan_gpu_double_double_axis0_bluestein_stockham_block,
    plan_gpu_double_double_axis0_fft_rader_user_grouped_batch_block_with_zero_padding_and_tuning,
    plan_gpu_double_double_axis0_stockham_block,
    plan_gpu_double_double_axis0_user_grouped_stockham_block_with_zero_padding,
    plan_gpu_double_double_other_axis_bluestein_stockham_block,
    plan_gpu_double_double_other_axis_fft_rader_batch_block_with_tuning,
    plan_gpu_double_double_other_axis_stockham_block,
    plan_gpu_double_double_other_axis_user_grouped_stockham_block,
    plan_gpu_double_double_stockham_uploads_for_batches_with_axis_context,
    plan_gpu_other_axis_bluestein_wrapper_block, plan_gpu_other_axis_direct_rader_batch_block,
};
use crate::zero_pad_ir::ZeroPadPassIr;

fn double_double_storage_scalar(storage: PrecisionStorage) -> Result<ScalarType> {
    match storage {
        PrecisionStorage::DoubleDouble => Ok(ScalarType::DoubleDouble),
        PrecisionStorage::F64 => Ok(ScalarType::F64),
        _ => Err(VkFftError::InvalidKernelIr(
            "double-double zero-pad boundary requires DD or F64 external storage",
        )),
    }
}

fn build_double_double_zero_pad_pass(
    range: Option<ZeroPaddingRange>,
    logical_len: usize,
    batch_count: usize,
    grouped_batch: usize,
    direction: Direction,
    external_storage: PrecisionStorage,
    domain: ZeroPaddingDomain,
) -> Result<Option<ZeroPadPassIr>> {
    range
        .map(|range| {
            ZeroPadPassIr::build_storage_preserving_with_domain(
                logical_len,
                batch_count,
                double_double_storage_scalar(external_storage)?,
                direction,
                range,
                grouped_batch,
                domain,
            )
        })
        .transpose()
}

fn validate_double_double_zero_pad_pass(
    pass: Option<&ZeroPadPassIr>,
    logical_len: usize,
    batch_count: usize,
    grouped_batch: usize,
    direction: Direction,
    external_storage: PrecisionStorage,
) -> Result<()> {
    let Some(pass) = pass else {
        return Ok(());
    };
    pass.validate()?;
    let scalar = double_double_storage_scalar(external_storage)?;
    if pass.logical_len != logical_len
        || pass.batch_count != batch_count
        || pass.grouped_batch != grouped_batch
        || pass.direction != direction
        || pass.scalar != scalar
        || pass.input_storage_scalar != scalar
        || pass.output_storage_scalar != scalar
    {
        return Err(VkFftError::InvalidKernelIr(
            "double-double zero-pad boundary does not match its one-dimensional transform",
        ));
    }
    Ok(())
}

fn double_double_zero_padded_input(
    pass: Option<&ZeroPadPassIr>,
    logical_len: usize,
    batch_count: usize,
    input: &[ComplexDoubleDouble],
) -> Option<Vec<ComplexDoubleDouble>> {
    pass.filter(|pass| pass.operation.is_input_boundary())
        .map(|pass| {
            let mut values = input.to_vec();
            for batch in 0..batch_count {
                let base = batch * logical_len;
                values[base + pass.range.left..base + pass.range.right]
                    .fill(ComplexDoubleDouble::default());
            }
            values
        })
}

fn apply_double_double_output_zero_padding(
    pass: Option<&ZeroPadPassIr>,
    logical_len: usize,
    batch_count: usize,
    output: &mut [ComplexDoubleDouble],
) {
    if let Some(pass) = pass.filter(|pass| pass.operation.is_output_boundary()) {
        for batch in 0..batch_count {
            let base = batch * logical_len;
            output[base + pass.range.left..base + pass.range.right]
                .fill(ComplexDoubleDouble::default());
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct DoubleDoubleStockhamIr {
    pub name: String,
    pub direction: Direction,
    pub sequence_len: usize,
    pub batch_count: usize,
    pub grouped_batch: usize,
    /// Fixed-upstream physical batching for the executable DD localSizeY slice.
    /// Low-level builders keep this `None`; high-level device-aware materialization may
    /// install it for explicitly grouped 2/3/5/7-smooth Stockham kernels when the
    /// exact Quad workgroup geometry fits the device.
    pub axis_batch_block: Option<StockhamAxisBlockSchedule>,
    pub normalize: bool,
    pub external_storage: PrecisionStorage,
    pub zero_pad_pass: Option<ZeroPadPassIr>,
    pub stages: Vec<StockhamStage>,
    pub twiddles: DoubleDoubleStockhamTwiddleTable,
}

impl DoubleDoubleStockhamIr {
    pub fn build(plan: &FftPlan, direction: Direction) -> Result<Self> {
        if plan.config.dimensions.len() != 1 {
            return Err(VkFftError::UnsupportedKernelPath(
                "double-double Stockham IR supports one-dimensional plans only",
            ));
        }
        if plan.config.transform != TransformKind::ComplexToComplex {
            return Err(VkFftError::UnsupportedKernelPath(
                "double-double Stockham IR supports C2C transforms only",
            ));
        }
        let external_storage = match plan.config.precision {
            Precision::DoubleDouble => PrecisionStorage::DoubleDouble,
            Precision::DoubleDoubleF64Storage => PrecisionStorage::F64,
            other => {
                return Err(VkFftError::UnsupportedPrecision {
                    backend: "double-double Stockham IR",
                    precision: precision_name(other),
                });
            }
        };
        let axis = plan.axes.first().ok_or(VkFftError::InvalidKernelIr(
            "missing double-double Stockham axis",
        ))?;
        let AxisAlgorithm::Stockham { radix } = &axis.algorithm else {
            return Err(VkFftError::UnsupportedKernelPath(
                "double-double executable IR currently covers planner-selected Stockham axes",
            ));
        };
        let sequence_len = axis.effective_fft_len;
        let grouped_batch = plan.config.grouped_batch_for_axis(0).unwrap_or(1);
        let zero_pad_pass = build_double_double_zero_pad_pass(
            plan.config.zero_padding_for_axis(0),
            sequence_len,
            plan.config.batch_count,
            grouped_batch,
            direction,
            external_storage,
            plan.config.zero_padding_domain,
        )?;
        let radices = stockham_schedule(radix, sequence_len)?;
        let mut stages = Vec::with_capacity(radices.len());
        let mut source = SharedBuffer::A;
        let mut stage_size = 1usize;
        for (index, radix) in radices.into_iter().enumerate() {
            let output = source.alternate();
            stages.push(StockhamStage {
                index,
                radix,
                stage_size,
                butterflies: sequence_len / radix,
                input: source,
                output,
            });
            stage_size = stage_size
                .checked_mul(radix)
                .ok_or(VkFftError::ArithmeticOverflow {
                    operation: "double-double Stockham stage size",
                })?;
            source = output;
        }
        if stage_size != sequence_len {
            return Err(VkFftError::InvalidKernelIr(
                "double-double Stockham stages do not cover the sequence",
            ));
        }
        let twiddles = build_twiddle_table(sequence_len, direction, &stages)?;
        let ir = Self {
            name: format!(
                "vkfft_dd_stockham_{}_{}",
                sequence_len,
                match direction {
                    Direction::Forward => "forward",
                    Direction::Inverse => "inverse",
                }
            ),
            direction,
            sequence_len,
            batch_count: plan.config.batch_count,
            grouped_batch,
            axis_batch_block: None,
            normalize: direction == Direction::Inverse && plan.config.normalize_inverse,
            external_storage,
            zero_pad_pass,
            stages,
            twiddles,
        };
        ir.validate()?;
        Ok(ir)
    }

    pub fn validate(&self) -> Result<()> {
        if self.sequence_len == 0 || self.batch_count == 0 || self.grouped_batch == 0 {
            return Err(VkFftError::InvalidKernelIr(
                "double-double Stockham dimensions and batch count must be non-zero",
            ));
        }
        if let Some(block) = self.axis_batch_block {
            let expected = if block.transforms_on_x {
                [block.grouped_batch, block.threads_per_transform]
            } else {
                [block.threads_per_transform, block.grouped_batch]
            };
            if block.grouped_batch == 0
                || block.grouped_batch > self.batch_count
                || block.threads_per_transform == 0
                || [block.local_size_x, block.local_size_y] != expected
            {
                return Err(VkFftError::InvalidKernelIr(
                    "double-double Stockham axis-batch block is inconsistent with logical grouped ownership",
                ));
            }
        }
        if !matches!(
            self.external_storage,
            PrecisionStorage::DoubleDouble | PrecisionStorage::F64
        ) {
            return Err(VkFftError::InvalidKernelIr(
                "double-double Stockham external storage must be DD or F64",
            ));
        }
        validate_double_double_zero_pad_pass(
            self.zero_pad_pass.as_ref(),
            self.sequence_len,
            self.batch_count,
            self.grouped_batch,
            self.direction,
            self.external_storage,
        )?;
        if self.twiddles.sequence_len != self.sequence_len
            || self.twiddles.direction != self.direction
            || self.twiddles.stages.len() != self.stages.len()
        {
            return Err(VkFftError::InvalidKernelIr(
                "double-double Stockham twiddle metadata does not match stages",
            ));
        }
        let mut expected_stage_size = 1usize;
        let mut expected_source = SharedBuffer::A;
        for (stage, twiddles) in self.stages.iter().zip(&self.twiddles.stages) {
            if stage.stage_size != expected_stage_size
                || stage.input != expected_source
                || stage.output != expected_source.alternate()
                || stage.butterflies != self.sequence_len / stage.radix
                || twiddles.index != stage.index
                || twiddles.radix != stage.radix
                || twiddles.stage_size != stage.stage_size
                || twiddles.values.len() != stage.stage_size * stage.radix
            {
                return Err(VkFftError::InvalidKernelIr(
                    "double-double Stockham stage/twiddle metadata is inconsistent",
                ));
            }
            expected_stage_size = expected_stage_size.checked_mul(stage.radix).ok_or(
                VkFftError::ArithmeticOverflow {
                    operation: "double-double Stockham validation stage size",
                },
            )?;
            expected_source = stage.output;
        }
        if expected_stage_size != self.sequence_len {
            return Err(VkFftError::InvalidKernelIr(
                "double-double Stockham validation found incomplete stages",
            ));
        }
        Ok(())
    }

    pub fn with_axis0_grouped_stockham_block(mut self, device: DeviceProfile) -> Result<Self> {
        let block = plan_gpu_double_double_axis0_user_grouped_stockham_block_with_zero_padding(
            self.sequence_len,
            self.batch_count,
            Some(self.grouped_batch),
            self.zero_pad_pass.is_some(),
            device,
        )?;
        self.axis_batch_block = block.filter(|block| block.grouped_batch == self.grouped_batch);
        self.validate()?;
        Ok(self)
    }

    /// Materialize the automatic fixed-upstream axis-0 block for a single-upload
    /// Stockham transform. Independent batches remain separate physical groups on this
    /// path. It is used by large smooth DD transforms that exceed the historical
    /// recursive 4096 leaf gate while still fitting one shared-memory upload on the
    /// concrete device.
    pub fn with_axis0_device_stockham_block(mut self, device: DeviceProfile) -> Result<Self> {
        self.axis_batch_block = plan_gpu_double_double_axis0_stockham_block(
            self.sequence_len,
            self.batch_count,
            device,
        )?;
        self.validate()?;
        Ok(self)
    }

    fn with_axis0_device_bluestein_stockham_block(mut self, device: DeviceProfile) -> Result<Self> {
        self.axis_batch_block = plan_gpu_double_double_axis0_bluestein_stockham_block(
            self.sequence_len,
            self.batch_count,
            device,
        )?;
        self.validate()?;
        Ok(self)
    }

    fn with_other_axis_device_bluestein_stockham_block(
        mut self,
        device: DeviceProfile,
    ) -> Result<Self> {
        self.axis_batch_block = plan_gpu_double_double_other_axis_bluestein_stockham_block(
            self.sequence_len,
            self.batch_count,
            device,
        )?;
        self.validate()?;
        Ok(self)
    }

    /// Materialize a recursive Stockham leaf with the resource-fitting physical
    /// subgroup selected by the Quad scheduler. `grouped_batch` remains the
    /// recursive tree's logical ownership and may exceed the physical subgroup.
    pub fn with_axis0_recursive_leaf_stockham_block(
        mut self,
        device: DeviceProfile,
    ) -> Result<Self> {
        self.axis_batch_block =
            plan_gpu_double_double_axis0_user_grouped_stockham_block_with_zero_padding(
                self.sequence_len,
                self.batch_count,
                Some(self.grouped_batch),
                self.zero_pad_pass.is_some(),
                device,
            )?;
        self.validate()?;
        Ok(self)
    }

    pub fn with_other_axis_grouped_stockham_block(
        mut self,
        fastest_axis_len: usize,
        grouped_batch_override: Option<usize>,
        axis1_grouped_batch_override: Option<usize>,
        device: DeviceProfile,
    ) -> Result<Self> {
        self.axis_batch_block = plan_gpu_double_double_other_axis_user_grouped_stockham_block(
            self.sequence_len,
            self.batch_count,
            fastest_axis_len,
            grouped_batch_override,
            axis1_grouped_batch_override,
            device,
        )?;
        if self.axis_batch_block.is_none()
            && grouped_batch_override.is_none()
            && axis1_grouped_batch_override.is_none()
        {
            self.axis_batch_block = plan_gpu_double_double_other_axis_stockham_block(
                self.sequence_len,
                self.batch_count,
                device,
            )?;
        }
        self.validate()?;
        Ok(self)
    }

    /// Number of workgroups used by the executable Stockham program. Recursive
    /// leaves may use fewer physical transforms per workgroup than logical ownership.
    pub fn batch_group_count(&self) -> usize {
        let execution_grouped_batch = self
            .axis_batch_block
            .map_or(self.grouped_batch, |block| block.grouped_batch);
        self.batch_count.div_ceil(execution_grouped_batch)
    }

    /// Logical grouped ownership used by recursive Cooley-Tukey composition.
    pub fn logical_batch_group_count(&self) -> usize {
        self.batch_count.div_ceil(self.grouped_batch)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct DoubleDoubleDirectRaderIr {
    pub name: String,
    pub direction: Direction,
    pub prime: usize,
    pub batch_count: usize,
    pub grouped_batch: usize,
    /// Fixed-upstream top-level direct-Rader physical batch block. Low-level
    /// builders keep this `None`; high-level device materialization may install it.
    pub axis_batch_block: Option<StockhamAxisBlockSchedule>,
    pub normalize: bool,
    pub external_storage: PrecisionStorage,
    pub zero_pad_pass: Option<ZeroPadPassIr>,
    /// Recursive Four-step components may read caller storage and write full-DD
    /// scratch (or the reverse on the final upload), so the two edges are typed
    /// independently while `external_storage` retains the public plan precision.
    pub input_storage: PrecisionStorage,
    pub output_storage: PrecisionStorage,
    pub io_mapping: StockhamIoMapping,
    pub table: DoubleDoubleRaderTable,
}

impl DoubleDoubleDirectRaderIr {
    pub fn build(plan: &FftPlan, direction: Direction) -> Result<Self> {
        if plan.config.dimensions.len() != 1
            || plan.config.transform != TransformKind::ComplexToComplex
        {
            return Err(VkFftError::UnsupportedKernelPath(
                "double-double direct Rader IR supports one-dimensional C2C only",
            ));
        }
        let external_storage = double_double_external_storage(plan.config.precision)?;
        let axis = plan.axes.first().ok_or(VkFftError::InvalidKernelIr(
            "missing double-double direct Rader axis",
        ))?;
        let AxisAlgorithm::Rader { stockham, primes } = &axis.algorithm else {
            return Err(VkFftError::UnsupportedKernelPath(
                "double-double direct Rader requires a planner-selected Rader axis",
            ));
        };
        if !stockham.prime_factors.is_empty()
            || !stockham.merged_radices.is_empty()
            || primes.len() != 1
            || primes[0].multiplicity != 1
            || primes[0].prime != axis.effective_fft_len
            || !matches!(primes[0].mode, RaderMode::DirectMultiplication)
        {
            return Err(VkFftError::UnsupportedKernelPath(
                "double-double direct Rader currently requires one direct prime root",
            ));
        }
        let prime = primes[0].prime;
        let grouped_batch = plan.config.grouped_batch_for_axis(0).unwrap_or(1);
        let zero_pad_pass = build_double_double_zero_pad_pass(
            plan.config.zero_padding_for_axis(0),
            prime,
            plan.config.batch_count,
            grouped_batch,
            direction,
            external_storage,
            plan.config.zero_padding_domain,
        )?;
        let ir = Self {
            name: format!(
                "vkfft_dd_rader_direct_{}_{}",
                prime,
                match direction {
                    Direction::Forward => "forward",
                    Direction::Inverse => "inverse",
                }
            ),
            direction,
            prime,
            batch_count: plan.config.batch_count,
            grouped_batch,
            axis_batch_block: None,
            normalize: direction == Direction::Inverse && plan.config.normalize_inverse,
            external_storage,
            zero_pad_pass,
            input_storage: external_storage,
            output_storage: external_storage,
            io_mapping: StockhamIoMapping::Contiguous,
            table: DoubleDoubleRaderTable::from_prime_plan(&primes[0], direction)?,
        };
        ir.validate()?;
        Ok(ir)
    }

    pub fn validate(&self) -> Result<()> {
        if self.prime < 2 || self.batch_count == 0 || self.grouped_batch == 0 {
            return Err(VkFftError::InvalidKernelIr(
                "double-double direct Rader prime and batch count must be non-zero",
            ));
        }
        if let Some(block) = self.axis_batch_block {
            let expected = if block.transforms_on_x {
                [block.grouped_batch, block.threads_per_transform]
            } else {
                [block.threads_per_transform, block.grouped_batch]
            };
            if block.grouped_batch == 0
                || block.grouped_batch > self.batch_count
                || block.threads_per_transform < self.prime.div_ceil(2)
                || [block.local_size_x, block.local_size_y] != expected
            {
                return Err(VkFftError::InvalidKernelIr(
                    "double-double direct Rader axis-batch block is inconsistent with grouped ownership",
                ));
            }
        }
        if !matches!(
            self.external_storage,
            PrecisionStorage::DoubleDouble | PrecisionStorage::F64
        ) || !matches!(
            self.input_storage,
            PrecisionStorage::DoubleDouble | PrecisionStorage::F64
        ) || !matches!(
            self.output_storage,
            PrecisionStorage::DoubleDouble | PrecisionStorage::F64
        ) {
            return Err(VkFftError::InvalidKernelIr(
                "double-double direct Rader caller storage must be DD or F64",
            ));
        }
        validate_double_double_zero_pad_pass(
            self.zero_pad_pass.as_ref(),
            self.prime,
            self.batch_count,
            self.grouped_batch,
            self.direction,
            self.external_storage,
        )?;
        self.io_mapping
            .validate_kernel(self.prime, self.batch_count)?;
        let count = self.prime - 1;
        if self.table.prime != self.prime
            || self.table.direction != self.direction
            || self.table.permutation.len() != count
            || self.table.permutation_inverse.len() != count
            || self.table.twiddles_by_generator_power.len() != count
        {
            return Err(VkFftError::InvalidKernelIr(
                "double-double direct Rader table metadata does not match the IR",
            ));
        }
        let mut seen = vec![false; self.prime];
        for &index in &self.table.permutation {
            if index == 0 || index >= self.prime || seen[index] {
                return Err(VkFftError::InvalidKernelIr(
                    "double-double direct Rader permutation is not bijective",
                ));
            }
            seen[index] = true;
        }
        if seen.iter().skip(1).any(|seen| !seen) {
            return Err(VkFftError::InvalidKernelIr(
                "double-double direct Rader permutation misses a prime residue",
            ));
        }
        Ok(())
    }

    pub fn with_axis0_grouped_direct_rader_block(mut self, device: DeviceProfile) -> Result<Self> {
        const DD_COMPLEX_BYTES: usize = 32;
        let block = plan_gpu_axis0_direct_rader_batch_block(
            self.prime,
            self.batch_count,
            DD_COMPLEX_BYTES,
            self.zero_pad_pass.is_some(),
            Some(self.grouped_batch),
            device,
        )?;
        self.axis_batch_block = block.filter(|block| block.grouped_batch == self.grouped_batch);
        self.validate()?;
        Ok(self)
    }

    pub fn with_other_axis_grouped_direct_rader_block(
        mut self,
        fastest_axis_len: usize,
        grouped_batch_override: Option<usize>,
        axis1_grouped_batch_override: Option<usize>,
        device: DeviceProfile,
    ) -> Result<Self> {
        const DD_COMPLEX_BYTES: usize = 32;
        self.axis_batch_block = plan_gpu_other_axis_direct_rader_batch_block(
            self.prime,
            self.batch_count,
            fastest_axis_len,
            DD_COMPLEX_BYTES,
            grouped_batch_override,
            axis1_grouped_batch_override,
            device,
        )?;
        self.validate()?;
        Ok(self)
    }

    pub(crate) fn with_four_step_io_mapping(
        mut self,
        io_mapping: StockhamIoMapping,
        input_storage: PrecisionStorage,
        output_storage: PrecisionStorage,
    ) -> Result<Self> {
        self.io_mapping = io_mapping;
        self.input_storage = input_storage;
        self.output_storage = output_storage;
        self.validate()?;
        Ok(self)
    }

    pub fn batch_group_count(&self) -> usize {
        let execution_grouped_batch = self
            .axis_batch_block
            .map_or(self.grouped_batch, |block| block.grouped_batch);
        self.batch_count.div_ceil(execution_grouped_batch)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct DoubleDoubleFftRaderIr {
    pub name: String,
    pub direction: Direction,
    pub prime: usize,
    pub convolution_len: usize,
    pub batch_count: usize,
    pub grouped_batch: usize,
    /// Planner thresholds used to classify nested Rader containers. Caller physical
    /// scheduling must consume the same policy as the convolution tree.
    pub planner_tuning: PlannerTuning,
    /// Prime-length caller ownership from the DD FFT-Rader axis block. This is
    /// intentionally distinct from the `(p-1)` Stockham child physical block.
    pub caller_axis_batch_block: Option<StockhamAxisBlockSchedule>,
    pub normalize: bool,
    pub external_storage: PrecisionStorage,
    pub zero_pad_pass: Option<ZeroPadPassIr>,
    /// Recursive Four-step component caller edges are typed independently.
    pub input_storage: PrecisionStorage,
    pub output_storage: PrecisionStorage,
    pub io_mapping: StockhamIoMapping,
    pub table: DoubleDoubleRaderTable,
    pub forward_fft: DoubleDoubleBluesteinConvolutionIr,
    pub inverse_fft: DoubleDoubleBluesteinConvolutionIr,
    pub kernel_spectrum: Vec<ComplexDoubleDouble>,
}

impl DoubleDoubleFftRaderIr {
    pub fn build(plan: &FftPlan, direction: Direction) -> Result<Self> {
        Self::build_impl(plan, direction, None)
    }

    pub fn build_for_device(
        plan: &FftPlan,
        direction: Direction,
        device: DeviceProfile,
    ) -> Result<Self> {
        Self::build_impl(plan, direction, Some(device))
    }

    fn build_impl(
        plan: &FftPlan,
        direction: Direction,
        device: Option<DeviceProfile>,
    ) -> Result<Self> {
        if plan.config.dimensions.len() != 1
            || plan.config.transform != TransformKind::ComplexToComplex
        {
            return Err(VkFftError::UnsupportedKernelPath(
                "double-double FFT Rader IR supports one-dimensional C2C only",
            ));
        }
        let external_storage = double_double_external_storage(plan.config.precision)?;
        let axis = plan.axes.first().ok_or(VkFftError::InvalidKernelIr(
            "missing double-double FFT Rader axis",
        ))?;
        let AxisAlgorithm::Rader { stockham, primes } = &axis.algorithm else {
            return Err(VkFftError::UnsupportedKernelPath(
                "double-double FFT Rader requires a planner-selected Rader axis",
            ));
        };
        if !stockham.prime_factors.is_empty()
            || !stockham.merged_radices.is_empty()
            || primes.len() != 1
            || primes[0].multiplicity != 1
            || primes[0].prime != axis.effective_fft_len
            || !matches!(primes[0].mode, RaderMode::FftConvolution { .. })
        {
            return Err(VkFftError::UnsupportedKernelPath(
                "double-double FFT Rader currently requires one FFT-convolution prime root",
            ));
        }
        let prime = primes[0].prime;
        let convolution_len = prime - 1;
        let grouped_batch_override = plan.config.grouped_batch_for_axis(0);
        let grouped_batch = grouped_batch_override.unwrap_or(1);
        let zero_pad_pass = build_double_double_zero_pad_pass(
            plan.config.zero_padding_for_axis(0),
            prime,
            plan.config.batch_count,
            grouped_batch,
            direction,
            external_storage,
            plan.config.zero_padding_domain,
        )?;
        let table = DoubleDoubleRaderTable::from_prime_plan(&primes[0], direction)?;
        let mut forward_config = crate::FftConfig::new(vec![convolution_len])
            .with_batch_count(plan.config.batch_count)
            .with_precision(Precision::DoubleDouble)
            .with_tuning(plan.config.tuning);
        if let Some(grouped_batch) = grouped_batch_override {
            forward_config = forward_config.with_grouped_batch(0, grouped_batch)?;
        }
        let inverse_config = forward_config.clone().with_inverse_normalization(true);
        let forward_plan = if let Some(device) = device {
            FftPlan::build_c2c_child_for_device(
                forward_config,
                device,
                C2cDeviceAxisClass::Contiguous,
            )?
        } else {
            FftPlan::build(forward_config)?
        };
        let inverse_plan = if let Some(device) = device {
            FftPlan::build_c2c_child_for_device(
                inverse_config,
                device,
                C2cDeviceAxisClass::Contiguous,
            )?
        } else {
            FftPlan::build(inverse_config)?
        };
        let kernel_config = crate::FftConfig::new(vec![convolution_len])
            .with_precision(Precision::DoubleDouble)
            .with_tuning(plan.config.tuning);
        let kernel_plan = if let Some(device) = device {
            FftPlan::build_c2c_child_for_device(
                kernel_config,
                device,
                C2cDeviceAxisClass::Contiguous,
            )?
        } else {
            FftPlan::build(kernel_config)?
        };
        let forward_fft = if let Some(device) = device {
            DoubleDoubleBluesteinConvolutionIr::build_rader_convolution_for_device(
                &forward_plan,
                Direction::Forward,
                device,
            )?
        } else {
            DoubleDoubleBluesteinConvolutionIr::build_rader_convolution(
                &forward_plan,
                Direction::Forward,
            )?
        };
        let inverse_fft = if let Some(device) = device {
            DoubleDoubleBluesteinConvolutionIr::build_rader_convolution_for_device(
                &inverse_plan,
                Direction::Inverse,
                device,
            )?
        } else {
            DoubleDoubleBluesteinConvolutionIr::build_rader_convolution(
                &inverse_plan,
                Direction::Inverse,
            )?
        };
        let kernel_fft = if let Some(device) = device {
            DoubleDoubleBluesteinConvolutionIr::build_rader_convolution_for_device(
                &kernel_plan,
                Direction::Forward,
                device,
            )?
        } else {
            DoubleDoubleBluesteinConvolutionIr::build_rader_convolution(
                &kernel_plan,
                Direction::Forward,
            )?
        };
        let kernel_spectrum = kernel_fft.execute(&table.twiddles_by_generator_power)?;
        let ir = Self {
            name: format!(
                "vkfft_dd_rader_fft_{}_{}",
                prime,
                match direction {
                    Direction::Forward => "forward",
                    Direction::Inverse => "inverse",
                }
            ),
            direction,
            prime,
            convolution_len,
            batch_count: plan.config.batch_count,
            grouped_batch,
            planner_tuning: plan.config.tuning,
            caller_axis_batch_block: None,
            normalize: direction == Direction::Inverse && plan.config.normalize_inverse,
            external_storage,
            zero_pad_pass,
            input_storage: external_storage,
            output_storage: external_storage,
            io_mapping: StockhamIoMapping::Contiguous,
            table,
            forward_fft,
            inverse_fft,
            kernel_spectrum,
        };
        ir.validate()?;
        Ok(ir)
    }

    pub fn validate(&self) -> Result<()> {
        if self.prime < 3
            || self.convolution_len != self.prime - 1
            || self.batch_count == 0
            || self.grouped_batch == 0
        {
            return Err(VkFftError::InvalidKernelIr(
                "double-double FFT Rader dimensions are inconsistent",
            ));
        }
        if !matches!(
            self.external_storage,
            PrecisionStorage::DoubleDouble | PrecisionStorage::F64
        ) || !matches!(
            self.input_storage,
            PrecisionStorage::DoubleDouble | PrecisionStorage::F64
        ) || !matches!(
            self.output_storage,
            PrecisionStorage::DoubleDouble | PrecisionStorage::F64
        ) {
            return Err(VkFftError::InvalidKernelIr(
                "double-double FFT Rader caller storage must be DD or F64",
            ));
        }
        validate_double_double_zero_pad_pass(
            self.zero_pad_pass.as_ref(),
            self.prime,
            self.batch_count,
            self.grouped_batch,
            self.direction,
            self.external_storage,
        )?;
        self.io_mapping
            .validate_kernel(self.prime, self.batch_count)?;
        if let Some(block) = self.caller_axis_batch_block {
            let expected = if block.transforms_on_x {
                [block.grouped_batch, block.threads_per_transform]
            } else {
                [block.threads_per_transform, block.grouped_batch]
            };
            if block.grouped_batch == 0
                || block.grouped_batch > self.batch_count
                || [block.local_size_x, block.local_size_y] != expected
            {
                return Err(VkFftError::InvalidKernelIr(
                    "double-double FFT Rader caller axis-batch block is inconsistent",
                ));
            }
        }
        if self.table.prime != self.prime
            || self.table.direction != self.direction
            || self.table.permutation.len() != self.convolution_len
            || self.table.twiddles_by_generator_power.len() != self.convolution_len
            || self.kernel_spectrum.len() != self.convolution_len
        {
            return Err(VkFftError::InvalidKernelIr(
                "double-double FFT Rader table/spectrum metadata is inconsistent",
            ));
        }
        self.forward_fft.validate()?;
        self.inverse_fft.validate()?;
        if self.forward_fft.stockham_axis_batch_block()
            != self.inverse_fft.stockham_axis_batch_block()
        {
            return Err(VkFftError::InvalidKernelIr(
                "double-double FFT Rader convolution children disagree on physical grouping",
            ));
        }
        if self
            .forward_fft
            .stockham_axis_batch_block()
            .is_some_and(|block| block.grouped_batch > self.grouped_batch)
        {
            return Err(VkFftError::InvalidKernelIr(
                "double-double FFT Rader convolution child changed groupedBatch ownership",
            ));
        }
        if self.forward_fft.sequence_len() != self.convolution_len
            || self.inverse_fft.sequence_len() != self.convolution_len
            || self.forward_fft.batch_count() != self.batch_count
            || self.inverse_fft.batch_count() != self.batch_count
            || self.forward_fft.grouped_batch() != self.grouped_batch
            || self.inverse_fft.grouped_batch() != self.grouped_batch
            || self.forward_fft.direction() != Direction::Forward
            || self.inverse_fft.direction() != Direction::Inverse
            || self.forward_fft.normalize()
            || !self.inverse_fft.normalize()
            || self.forward_fft.external_storage() != PrecisionStorage::DoubleDouble
            || self.inverse_fft.external_storage() != PrecisionStorage::DoubleDouble
        {
            return Err(VkFftError::InvalidKernelIr(
                "double-double FFT Rader internal convolution metadata is inconsistent",
            ));
        }
        Ok(())
    }

    pub fn with_axis0_grouped_fft_rader_blocks(mut self, device: DeviceProfile) -> Result<Self> {
        let caller = plan_gpu_double_double_axis0_fft_rader_user_grouped_batch_block_with_zero_padding_and_tuning(
            self.prime,
            self.batch_count,
            Some(self.grouped_batch),
            self.zero_pad_pass.is_some(),
            self.planner_tuning,
            device,
        )?;
        self.caller_axis_batch_block =
            caller.filter(|block| block.grouped_batch == self.grouped_batch);
        self.forward_fft = self.forward_fft.with_axis0_grouped_stockham_block(device)?;
        self.inverse_fft = self.inverse_fft.with_axis0_grouped_stockham_block(device)?;
        self.validate()?;
        Ok(self)
    }

    pub fn with_other_axis_grouped_fft_rader_block(
        mut self,
        fastest_axis_len: usize,
        grouped_batch_override: Option<usize>,
        axis1_grouped_batch_override: Option<usize>,
        device: DeviceProfile,
    ) -> Result<Self> {
        self.caller_axis_batch_block =
            plan_gpu_double_double_other_axis_fft_rader_batch_block_with_tuning(
                self.prime,
                self.batch_count,
                fastest_axis_len,
                grouped_batch_override,
                axis1_grouped_batch_override,
                self.planner_tuning,
                device,
            )?;
        self.validate()?;
        Ok(self)
    }

    pub(crate) fn with_four_step_io_mapping(
        mut self,
        io_mapping: StockhamIoMapping,
        input_storage: PrecisionStorage,
        output_storage: PrecisionStorage,
    ) -> Result<Self> {
        self.io_mapping = io_mapping;
        self.input_storage = input_storage;
        self.output_storage = output_storage;
        self.validate()?;
        Ok(self)
    }

    pub fn batch_group_count(&self) -> usize {
        let execution_grouped_batch = self
            .caller_axis_batch_block
            .map_or(self.grouped_batch, |block| block.grouped_batch);
        self.batch_count.div_ceil(execution_grouped_batch)
    }
}

fn double_double_stockham_requires_multi_upload(
    plan: &FftPlan,
    axis_len: usize,
    device: DeviceProfile,
) -> Result<bool> {
    let axis_context = StockhamUploadAxisContext {
        strided_axis: plan.c2c_device_axis_class_override == Some(C2cDeviceAxisClass::Strided),
        bandwidth_boost: plan.config.bandwidth_boost,
        use_bluestein_fft: plan.c2c_device_use_bluestein_fft_override,
        perform_convolution: false,
    };
    match plan_gpu_double_double_stockham_uploads_for_batches_with_axis_context(
        axis_len,
        plan.config.batch_count,
        device,
        axis_context,
    ) {
        Ok(schedule) => Ok(schedule.upload_count > 1),
        Err(VkFftError::UnsupportedKernelPath(_))
        | Err(VkFftError::ResourceLimitExceeded { .. }) => Ok(false),
        Err(error) => Err(error),
    }
}

// Keep convolution variants inline: these IR values are cloned/materialized as
// scheduler state, and boxing the large variants would add allocation to hot planning paths.
#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone, PartialEq)]
pub enum DoubleDoubleBluesteinConvolutionIr {
    Stockham(DoubleDoubleStockhamIr),
    Recursive(DoubleDoubleRecursiveFftIr),
    Bluestein(Box<DoubleDoubleBluesteinIr>),
}

impl DoubleDoubleBluesteinConvolutionIr {
    fn build_rader_convolution(plan: &FftPlan, direction: Direction) -> Result<Self> {
        let axis = plan.axes.first().ok_or(VkFftError::InvalidKernelIr(
            "missing double-double FFT Rader convolution axis",
        ))?;
        match &axis.algorithm {
            AxisAlgorithm::Stockham { .. } => Ok(Self::Stockham(DoubleDoubleStockhamIr::build(
                plan, direction,
            )?)),
            AxisAlgorithm::Rader { .. } => Ok(Self::Recursive(DoubleDoubleRecursiveFftIr::build(
                plan, direction,
            )?)),
            AxisAlgorithm::Bluestein { .. } => Ok(Self::Bluestein(Box::new(
                DoubleDoubleBluesteinIr::build(plan, direction)?,
            ))),
        }
    }

    fn build_rader_convolution_for_device(
        plan: &FftPlan,
        direction: Direction,
        device: DeviceProfile,
    ) -> Result<Self> {
        Self::build_impl(plan, direction, Some(device))
    }

    fn build(plan: &FftPlan, direction: Direction) -> Result<Self> {
        Self::build_impl(plan, direction, None)
    }

    fn build_for_device(
        plan: &FftPlan,
        direction: Direction,
        device: DeviceProfile,
    ) -> Result<Self> {
        Self::build_impl(plan, direction, Some(device))
    }

    fn build_impl(
        plan: &FftPlan,
        direction: Direction,
        device: Option<DeviceProfile>,
    ) -> Result<Self> {
        let axis = plan.axes.first().ok_or(VkFftError::InvalidKernelIr(
            "missing double-double Bluestein convolution axis",
        ))?;
        match &axis.algorithm {
            AxisAlgorithm::Stockham { .. } => {
                if let Some(device) = device {
                    if !double_double_stockham_requires_multi_upload(
                        plan,
                        axis.effective_fft_len,
                        device,
                    )? {
                        if axis.effective_fft_len <= 4096 {
                            return Ok(Self::Stockham(DoubleDoubleStockhamIr::build(
                                plan, direction,
                            )?));
                        }
                        let axis_class = plan
                            .c2c_device_axis_class_override
                            .unwrap_or(C2cDeviceAxisClass::Contiguous);
                        let stockham = match axis_class {
                            C2cDeviceAxisClass::Contiguous => {
                                DoubleDoubleStockhamIr::build(plan, direction)?
                                    .with_axis0_device_bluestein_stockham_block(device)?
                            }
                            C2cDeviceAxisClass::Strided => {
                                DoubleDoubleStockhamIr::build(plan, direction)?
                                    .with_other_axis_device_bluestein_stockham_block(device)?
                            }
                        };
                        if stockham.axis_batch_block.is_some() {
                            return Ok(Self::Stockham(stockham));
                        }
                    }
                    Ok(Self::Recursive(
                        DoubleDoubleRecursiveFftIr::build_for_device(plan, direction, device)?,
                    ))
                } else if axis.effective_fft_len <= 4096 {
                    Ok(Self::Stockham(DoubleDoubleStockhamIr::build(
                        plan, direction,
                    )?))
                } else {
                    Ok(Self::Recursive(DoubleDoubleRecursiveFftIr::build(
                        plan, direction,
                    )?))
                }
            }
            AxisAlgorithm::Rader { .. } => {
                let ir = if let Some(device) = device {
                    DoubleDoubleRecursiveFftIr::build_for_device(plan, direction, device)?
                } else {
                    DoubleDoubleRecursiveFftIr::build(plan, direction)?
                };
                Ok(Self::Recursive(ir))
            }
            AxisAlgorithm::Bluestein { .. } => {
                let ir = if let Some(device) = device {
                    DoubleDoubleBluesteinIr::build_for_device(plan, direction, device)?
                } else {
                    DoubleDoubleBluesteinIr::build(plan, direction)?
                };
                Ok(Self::Bluestein(Box::new(ir)))
            }
        }
    }

    pub fn with_axis0_grouped_stockham_block(self, device: DeviceProfile) -> Result<Self> {
        match self {
            Self::Stockham(ir) => Ok(Self::Stockham(
                ir.with_axis0_grouped_stockham_block(device)?,
            )),
            Self::Recursive(ir) => Ok(Self::Recursive(ir.with_device_physical_blocks(device)?)),
            Self::Bluestein(ir) => Ok(Self::Bluestein(Box::new(
                (*ir).with_axis0_grouped_bluestein_blocks(device)?,
            ))),
        }
    }

    fn with_axis0_device_bluestein_child_block(self, device: DeviceProfile) -> Result<Self> {
        match self {
            Self::Stockham(ir) if ir.axis_batch_block.is_some() => Ok(Self::Stockham(ir)),
            Self::Stockham(ir) => Ok(Self::Stockham(
                ir.with_axis0_grouped_stockham_block(device)?,
            )),
            // `build_for_device` has already materialized recursive children with the
            // parent Bluestein scheduling context. Re-running the generic materializer
            // here would drop `useBluesteinFFT` and overwrite the non-reorder blocks.
            Self::Recursive(ir) => Ok(Self::Recursive(ir)),
            Self::Bluestein(ir) => Ok(Self::Bluestein(Box::new(
                (*ir).with_axis0_device_bluestein_blocks(device)?,
            ))),
        }
    }

    pub fn with_other_axis_grouped_blocks(
        self,
        fastest_axis_len: usize,
        grouped_batch_override: Option<usize>,
        axis1_grouped_batch_override: Option<usize>,
        device: DeviceProfile,
    ) -> Result<Self> {
        match self {
            Self::Stockham(ir)
                if grouped_batch_override.is_none()
                    && axis1_grouped_batch_override.is_none()
                    && ir
                        .axis_batch_block
                        .is_some_and(|block| block.transforms_on_x && !block.axis_swapped) =>
            {
                Ok(Self::Stockham(ir))
            }
            Self::Stockham(ir) => Ok(Self::Stockham(ir.with_other_axis_grouped_stockham_block(
                fastest_axis_len,
                grouped_batch_override,
                axis1_grouped_batch_override,
                device,
            )?)),
            Self::Recursive(ir) => Ok(Self::Recursive(ir.with_other_axis_physical_blocks(
                fastest_axis_len,
                grouped_batch_override,
                axis1_grouped_batch_override,
                device,
            )?)),
            Self::Bluestein(ir) => Ok(Self::Bluestein(Box::new((*ir).with_other_axis_blocks(
                fastest_axis_len,
                grouped_batch_override,
                axis1_grouped_batch_override,
                device,
            )?))),
        }
    }

    pub fn stockham_axis_batch_block(&self) -> Option<StockhamAxisBlockSchedule> {
        match self {
            Self::Stockham(ir) => ir.axis_batch_block,
            Self::Recursive(_) | Self::Bluestein(_) => None,
        }
    }

    pub fn zero_pad_pass(&self) -> Option<&ZeroPadPassIr> {
        match self {
            Self::Stockham(ir) => ir.zero_pad_pass.as_ref(),
            Self::Recursive(ir) => ir.zero_pad_pass.as_ref(),
            Self::Bluestein(_) => None,
        }
    }

    pub fn validate(&self) -> Result<()> {
        match self {
            Self::Stockham(ir) => ir.validate(),
            Self::Recursive(ir) => ir.validate(),
            Self::Bluestein(ir) => {
                ir.validate()?;
                if ir.zero_padding.is_some() {
                    return Err(VkFftError::InvalidKernelIr(
                        "nested double-double Bluestein convolution child must not own spatial padding",
                    ));
                }
                Ok(())
            }
        }
    }

    pub fn sequence_len(&self) -> usize {
        match self {
            Self::Stockham(ir) => ir.sequence_len,
            Self::Recursive(ir) => ir.logical_len,
            Self::Bluestein(ir) => ir.logical_len,
        }
    }

    pub fn batch_count(&self) -> usize {
        match self {
            Self::Stockham(ir) => ir.batch_count,
            Self::Recursive(ir) => ir.batch_count,
            Self::Bluestein(ir) => ir.batch_count,
        }
    }

    pub fn grouped_batch(&self) -> usize {
        match self {
            Self::Stockham(ir) => ir.grouped_batch,
            Self::Recursive(ir) => ir.grouped_batch,
            Self::Bluestein(ir) => ir.grouped_batch,
        }
    }

    pub fn direction(&self) -> Direction {
        match self {
            Self::Stockham(ir) => ir.direction,
            Self::Recursive(ir) => ir.direction,
            Self::Bluestein(ir) => ir.direction,
        }
    }

    pub fn normalize(&self) -> bool {
        match self {
            Self::Stockham(ir) => ir.normalize,
            Self::Recursive(ir) => ir.normalize,
            Self::Bluestein(ir) => ir.normalize,
        }
    }

    pub fn external_storage(&self) -> PrecisionStorage {
        match self {
            Self::Stockham(ir) => ir.external_storage,
            Self::Recursive(ir) => ir.external_storage,
            Self::Bluestein(ir) => ir.external_storage,
        }
    }

    pub fn execute(&self, input: &[ComplexDoubleDouble]) -> Result<Vec<ComplexDoubleDouble>> {
        match self {
            Self::Stockham(ir) => execute_double_double_stockham_ir(ir, input),
            Self::Recursive(ir) => execute_double_double_recursive_ir(ir, input),
            Self::Bluestein(ir) => execute_double_double_bluestein_ir(ir, input),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct DoubleDoubleBluesteinIr {
    pub name: String,
    pub direction: Direction,
    pub logical_len: usize,
    pub convolution_len: usize,
    pub batch_count: usize,
    pub grouped_batch: usize,
    /// Physical ownership for Bluestein caller-boundary wrapper passes.
    /// Stockham children reuse their exact Quad block; recursive convolution
    /// children receive an independent slot-parallel wrapper block.
    pub wrapper_axis_batch_block: Option<StockhamAxisBlockSchedule>,
    pub normalize: bool,
    pub external_storage: PrecisionStorage,
    pub zero_padding: Option<ZeroPaddingRange>,
    pub zero_padding_domain: ZeroPaddingDomain,
    pub table: DoubleDoubleBluesteinTable,
    pub forward_fft: DoubleDoubleBluesteinConvolutionIr,
    pub inverse_fft: DoubleDoubleBluesteinConvolutionIr,
    pub kernel_spectrum: Vec<ComplexDoubleDouble>,
}

impl DoubleDoubleBluesteinIr {
    pub fn build(plan: &FftPlan, direction: Direction) -> Result<Self> {
        Self::build_impl(plan, direction, None)
    }

    pub fn build_for_device(
        plan: &FftPlan,
        direction: Direction,
        device: DeviceProfile,
    ) -> Result<Self> {
        Self::build_impl(plan, direction, Some(device))?.with_axis0_device_bluestein_blocks(device)
    }

    fn build_impl(
        plan: &FftPlan,
        direction: Direction,
        device: Option<DeviceProfile>,
    ) -> Result<Self> {
        if plan.config.dimensions.len() != 1
            || plan.config.transform != TransformKind::ComplexToComplex
        {
            return Err(VkFftError::UnsupportedKernelPath(
                "double-double Bluestein IR supports one-dimensional C2C only",
            ));
        }
        let external_storage = double_double_external_storage(plan.config.precision)?;
        let zero_padding = plan.config.zero_padding_for_axis(0);
        let axis = plan.axes.first().ok_or(VkFftError::InvalidKernelIr(
            "missing double-double Bluestein axis",
        ))?;
        let AxisAlgorithm::Bluestein {
            convolution_len, ..
        } = axis.algorithm
        else {
            return Err(VkFftError::UnsupportedKernelPath(
                "double-double Bluestein requires a planner-selected Bluestein axis",
            ));
        };
        let logical_len = axis.effective_fft_len;
        let grouped_batch_override = plan.config.grouped_batch_for_axis(0);
        let grouped_batch = grouped_batch_override.unwrap_or(1);
        let table = DoubleDoubleBluesteinTable::new(logical_len, convolution_len, direction)?;
        let convolution_axis_class = plan
            .c2c_device_axis_class_override
            .unwrap_or(C2cDeviceAxisClass::Contiguous);
        let mut forward_config = crate::FftConfig::new(vec![convolution_len])
            .with_batch_count(plan.config.batch_count)
            .with_precision(Precision::DoubleDouble)
            .with_tuning(plan.config.tuning)
            .with_bandwidth_boost(plan.config.bandwidth_boost);
        if let Some(grouped_batch) = grouped_batch_override {
            forward_config = forward_config.with_grouped_batch(0, grouped_batch)?;
        }
        let inverse_config = forward_config.clone().with_inverse_normalization(true);
        let forward_plan = if let Some(device) = device {
            FftPlan::build_c2c_bluestein_child_for_device(
                forward_config,
                device,
                convolution_axis_class,
            )?
        } else {
            FftPlan::build(forward_config)?
        };
        let inverse_plan = if let Some(device) = device {
            FftPlan::build_c2c_bluestein_child_for_device(
                inverse_config,
                device,
                convolution_axis_class,
            )?
        } else {
            FftPlan::build(inverse_config)?
        };
        let kernel_plan = FftPlan::build(
            crate::FftConfig::new(vec![convolution_len])
                .with_precision(Precision::DoubleDouble)
                .with_tuning(plan.config.tuning),
        )?;
        let forward_fft = if let Some(device) = device {
            DoubleDoubleBluesteinConvolutionIr::build_for_device(
                &forward_plan,
                Direction::Forward,
                device,
            )?
        } else {
            DoubleDoubleBluesteinConvolutionIr::build(&forward_plan, Direction::Forward)?
        };
        let inverse_fft = if let Some(device) = device {
            DoubleDoubleBluesteinConvolutionIr::build_for_device(
                &inverse_plan,
                Direction::Inverse,
                device,
            )?
        } else {
            DoubleDoubleBluesteinConvolutionIr::build(&inverse_plan, Direction::Inverse)?
        };
        let kernel_fft =
            DoubleDoubleBluesteinConvolutionIr::build(&kernel_plan, Direction::Forward)?;
        let kernel_spectrum = kernel_fft.execute(&table.convolution_kernel)?;
        let ir = Self {
            name: format!(
                "vkfft_dd_bluestein_{}_{}",
                logical_len,
                match direction {
                    Direction::Forward => "forward",
                    Direction::Inverse => "inverse",
                }
            ),
            direction,
            logical_len,
            convolution_len,
            batch_count: plan.config.batch_count,
            grouped_batch,
            wrapper_axis_batch_block: None,
            normalize: direction == Direction::Inverse && plan.config.normalize_inverse,
            external_storage,
            zero_padding,
            zero_padding_domain: plan.config.zero_padding_domain,
            table,
            forward_fft,
            inverse_fft,
            kernel_spectrum,
        };
        ir.validate()?;
        Ok(ir)
    }

    pub fn validate(&self) -> Result<()> {
        if self.logical_len == 0
            || self.convolution_len < self.logical_len.saturating_mul(2).saturating_sub(1)
            || self.batch_count == 0
            || self.grouped_batch == 0
        {
            return Err(VkFftError::InvalidKernelIr(
                "double-double Bluestein dimensions are inconsistent",
            ));
        }
        if !matches!(
            self.external_storage,
            PrecisionStorage::DoubleDouble | PrecisionStorage::F64
        ) {
            return Err(VkFftError::InvalidKernelIr(
                "double-double Bluestein external storage must be DD or F64",
            ));
        }
        if let Some(range) = self.zero_padding
            && (range.left > range.right || range.right > self.logical_len)
        {
            return Err(VkFftError::InvalidZeroPaddingRange {
                axis: 0,
                left: range.left,
                right: range.right,
                length: self.logical_len,
            });
        }
        if self.table.length != self.logical_len
            || self.table.convolution_len != self.convolution_len
            || self.table.direction != self.direction
            || self.table.chirp.len() != self.logical_len
            || self.table.convolution_kernel.len() != self.convolution_len
            || self.kernel_spectrum.len() != self.convolution_len
        {
            return Err(VkFftError::InvalidKernelIr(
                "double-double Bluestein table/spectrum metadata is inconsistent",
            ));
        }
        if let Some(block) = self.wrapper_axis_batch_block {
            let expected = if block.transforms_on_x {
                [block.grouped_batch, block.threads_per_transform]
            } else {
                [block.threads_per_transform, block.grouped_batch]
            };
            if block.grouped_batch == 0
                || block.grouped_batch > self.batch_count
                || block.threads_per_transform == 0
                || block.local_size_x == 0
                || block.local_size_y == 0
                || [block.local_size_x, block.local_size_y] != expected
            {
                return Err(VkFftError::InvalidKernelIr(
                    "double-double Bluestein wrapper physical block is inconsistent",
                ));
            }
        }
        self.forward_fft.validate()?;
        self.inverse_fft.validate()?;
        if self.forward_fft.stockham_axis_batch_block()
            != self.inverse_fft.stockham_axis_batch_block()
        {
            return Err(VkFftError::InvalidKernelIr(
                "double-double Bluestein convolution children disagree on physical grouping",
            ));
        }
        if self
            .forward_fft
            .stockham_axis_batch_block()
            .is_some_and(|block| block.grouped_batch > self.batch_count)
        {
            return Err(VkFftError::InvalidKernelIr(
                "double-double Bluestein convolution child changed groupedBatch ownership",
            ));
        }
        if self.forward_fft.sequence_len() != self.convolution_len
            || self.inverse_fft.sequence_len() != self.convolution_len
            || self.forward_fft.batch_count() != self.batch_count
            || self.inverse_fft.batch_count() != self.batch_count
            || self.forward_fft.grouped_batch() != self.grouped_batch
            || self.inverse_fft.grouped_batch() != self.grouped_batch
            || self.forward_fft.direction() != Direction::Forward
            || self.inverse_fft.direction() != Direction::Inverse
            || self.forward_fft.normalize()
            || !self.inverse_fft.normalize()
            || self.forward_fft.external_storage() != PrecisionStorage::DoubleDouble
            || self.inverse_fft.external_storage() != PrecisionStorage::DoubleDouble
        {
            return Err(VkFftError::InvalidKernelIr(
                "double-double Bluestein internal convolution FFT metadata is inconsistent",
            ));
        }
        Ok(())
    }

    fn with_axis0_device_bluestein_blocks(mut self, device: DeviceProfile) -> Result<Self> {
        self.forward_fft = self
            .forward_fft
            .with_axis0_device_bluestein_child_block(device)?;
        self.inverse_fft = self
            .inverse_fft
            .with_axis0_device_bluestein_child_block(device)?;
        self.wrapper_axis_batch_block =
            if let Some(block) = self.forward_fft.stockham_axis_batch_block() {
                Some(block)
            } else {
                plan_double_double_bluestein_wrapper_block(
                    self.convolution_len,
                    self.batch_count,
                    self.grouped_batch,
                    device,
                )?
            };
        self.validate()?;
        Ok(self)
    }

    pub fn with_axis0_grouped_bluestein_blocks(mut self, device: DeviceProfile) -> Result<Self> {
        self.forward_fft = self.forward_fft.with_axis0_grouped_stockham_block(device)?;
        self.inverse_fft = self.inverse_fft.with_axis0_grouped_stockham_block(device)?;
        self.wrapper_axis_batch_block =
            if let Some(block) = self.forward_fft.stockham_axis_batch_block() {
                Some(block)
            } else {
                plan_double_double_bluestein_wrapper_block(
                    self.convolution_len,
                    self.batch_count,
                    self.grouped_batch,
                    device,
                )?
            };
        self.validate()?;
        Ok(self)
    }

    pub fn with_other_axis_blocks(
        mut self,
        fastest_axis_len: usize,
        grouped_batch_override: Option<usize>,
        axis1_grouped_batch_override: Option<usize>,
        device: DeviceProfile,
    ) -> Result<Self> {
        self.forward_fft = self.forward_fft.with_other_axis_grouped_blocks(
            fastest_axis_len,
            grouped_batch_override,
            axis1_grouped_batch_override,
            device,
        )?;
        self.inverse_fft = self.inverse_fft.with_other_axis_grouped_blocks(
            fastest_axis_len,
            grouped_batch_override,
            axis1_grouped_batch_override,
            device,
        )?;
        self.wrapper_axis_batch_block = plan_gpu_other_axis_bluestein_wrapper_block(
            self.convolution_len,
            self.batch_count,
            fastest_axis_len,
            32,
            grouped_batch_override,
            axis1_grouped_batch_override,
            device,
        )?;
        self.validate()?;
        Ok(self)
    }

    pub fn stockham_convolution_axis_batch_block(&self) -> Option<StockhamAxisBlockSchedule> {
        self.forward_fft.stockham_axis_batch_block()
    }

    pub fn wrapper_axis_batch_block(&self) -> Option<StockhamAxisBlockSchedule> {
        self.wrapper_axis_batch_block
    }

    pub fn has_zero_padding(&self) -> bool {
        self.zero_padding.is_some()
    }

    pub fn has_spatial_zero_padding(&self) -> bool {
        self.zero_padding_domain == ZeroPaddingDomain::Spatial && self.has_zero_padding()
    }

    pub fn has_frequency_zero_padding(&self) -> bool {
        self.zero_padding_domain == ZeroPaddingDomain::Frequency && self.has_zero_padding()
    }

    pub fn zero_padding_is_input_boundary(&self) -> bool {
        self.has_zero_padding()
            && matches!(
                (self.direction, self.zero_padding_domain),
                (Direction::Forward, ZeroPaddingDomain::Spatial)
                    | (Direction::Inverse, ZeroPaddingDomain::Frequency)
            )
    }

    pub fn zero_padding_is_output_boundary(&self) -> bool {
        self.has_zero_padding()
            && matches!(
                (self.direction, self.zero_padding_domain),
                (Direction::Inverse, ZeroPaddingDomain::Spatial)
                    | (Direction::Forward, ZeroPaddingDomain::Frequency)
            )
    }

    pub fn contains_zero_index(&self, index: usize) -> bool {
        self.zero_padding.is_some_and(|range| range.contains(index))
    }

    pub fn contains_spatial_zero_index(&self, index: usize) -> bool {
        self.has_spatial_zero_padding() && self.contains_zero_index(index)
    }

    pub fn batch_group_count(&self) -> usize {
        let execution_grouped_batch = self
            .wrapper_axis_batch_block
            .map_or(self.grouped_batch, |block| block.grouped_batch);
        self.batch_count.div_ceil(execution_grouped_batch)
    }
}

fn plan_double_double_bluestein_wrapper_block(
    convolution_len: usize,
    batch_count: usize,
    grouped_batch: usize,
    device: DeviceProfile,
) -> Result<Option<StockhamAxisBlockSchedule>> {
    if grouped_batch == 0
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

#[derive(Debug, Clone, PartialEq)]
pub struct DoubleDoubleRealFftIr {
    pub kind: RealFftKind,
    pub length: usize,
    pub half_spectrum_len: usize,
    pub batch_count: usize,
    pub grouped_batch: usize,
    pub direction: Direction,
    pub normalize: bool,
    pub external_storage: PrecisionStorage,
    pub zero_padding: Option<ZeroPaddingRange>,
    pub even_half_size: bool,
    pub even_roots: Vec<ComplexDoubleDouble>,
    pub transform: DoubleDoubleOneDimIr,
}

impl DoubleDoubleRealFftIr {
    pub fn build(plan: &FftPlan) -> Result<Self> {
        Self::build_impl(plan, None)
    }

    pub fn build_for_device(plan: &FftPlan, device: DeviceProfile) -> Result<Self> {
        Self::build_impl(plan, Some(device))
    }

    fn build_impl(plan: &FftPlan, device: Option<DeviceProfile>) -> Result<Self> {
        if plan.config.dimensions.len() != 1 {
            return Err(VkFftError::UnsupportedKernelPath(
                "initial double-double real FFT IR supports one dimension only",
            ));
        }
        let (kind, direction) = match plan.config.transform {
            TransformKind::RealToComplex => (RealFftKind::RealToComplex, Direction::Forward),
            TransformKind::ComplexToReal => (RealFftKind::ComplexToReal, Direction::Inverse),
            _ => {
                return Err(VkFftError::UnsupportedKernelPath(
                    "double-double real FFT IR requires R2C or C2R",
                ));
            }
        };
        let external_storage = double_double_external_storage(plan.config.precision)?;
        let axis = plan.axes.first().ok_or(VkFftError::InvalidKernelIr(
            "missing double-double real FFT axis",
        ))?;
        let length = axis.logical_len;
        if axis.effective_fft_len != length {
            return Err(VkFftError::InvalidKernelIr(
                "double-double real planner changed the effective transform length",
            ));
        }
        if length == 0 {
            return Err(VkFftError::InvalidKernelIr(
                "double-double real FFT length must be non-zero",
            ));
        }
        let algorithm = if let Some(device) = device {
            real_fft_algorithm_for_shape_policy(
                length,
                plan.config.batch_count,
                plan.config.precision,
                plan.config.tuning,
                device,
                RealFftShapePolicy::FixedUpstreamDevice,
            )?
        } else if length.is_multiple_of(2) {
            RealFftAlgorithm::EvenHalfSize
        } else {
            RealFftAlgorithm::FullComplex
        };
        let even_half_size = algorithm == RealFftAlgorithm::EvenHalfSize;
        let grouped_batch_override = plan.config.grouped_batch_for_axis(0);
        let grouped_batch = grouped_batch_override.unwrap_or(1);
        let child_len = if even_half_size { length / 2 } else { length };
        let mut child_config = FftConfig::new(vec![child_len])
            .with_batch_count(plan.config.batch_count)
            .with_precision(Precision::DoubleDouble)
            .with_inverse_normalization(
                direction == Direction::Inverse && plan.config.normalize_inverse,
            )
            .with_tuning(plan.config.tuning)
            .with_bandwidth_boost(plan.config.bandwidth_boost);
        if let Some(grouped_batch) = grouped_batch_override {
            child_config = child_config.with_grouped_batch(0, grouped_batch)?;
        }
        let child_plan = if let Some(device) = device {
            FftPlan::build_c2c_child_for_device(
                child_config,
                device,
                C2cDeviceAxisClass::Contiguous,
            )?
        } else {
            FftPlan::build(child_config)?
        };
        let transform = if let Some(device) = device {
            DoubleDoubleOneDimIr::build_for_device(&child_plan, direction, device)?
        } else {
            DoubleDoubleOneDimIr::build(&child_plan, direction)?
        };
        let even_roots = if even_half_size {
            (0..=child_len)
                .map(|index| unit_root(index, length, Direction::Forward))
                .collect::<Result<Vec<_>>>()?
        } else {
            Vec::new()
        };
        let ir = Self {
            kind,
            length,
            half_spectrum_len: length / 2 + 1,
            batch_count: plan.config.batch_count,
            grouped_batch,
            direction,
            normalize: direction == Direction::Inverse && plan.config.normalize_inverse,
            external_storage,
            zero_padding: plan.config.zero_padding_for_axis(0),
            even_half_size,
            even_roots,
            transform,
        };
        ir.validate()?;
        Ok(ir)
    }

    pub fn with_grouped_stockham_child_block(mut self, device: DeviceProfile) -> Result<Self> {
        self.transform = self.transform.with_axis0_grouped_stockham_block(device)?;
        self.validate()?;
        Ok(self)
    }

    pub fn stockham_axis_batch_block(&self) -> Option<StockhamAxisBlockSchedule> {
        self.transform.stockham_axis_batch_block()
    }

    pub fn validate(&self) -> Result<()> {
        if self.length == 0
            || self.half_spectrum_len != self.length / 2 + 1
            || self.batch_count == 0
            || self.grouped_batch == 0
        {
            return Err(VkFftError::InvalidKernelIr(
                "double-double real metadata is inconsistent",
            ));
        }
        if !matches!(
            self.external_storage,
            PrecisionStorage::DoubleDouble | PrecisionStorage::F64
        ) {
            return Err(VkFftError::InvalidKernelIr(
                "double-double real external storage must be DD or F64",
            ));
        }
        if let Some(range) = self.zero_padding
            && (range.left > range.right || range.right > self.length)
        {
            return Err(VkFftError::InvalidZeroPaddingRange {
                axis: 0,
                left: range.left,
                right: range.right,
                length: self.length,
            });
        }
        let expected_child_len = if self.even_half_size {
            self.length / 2
        } else {
            self.length
        };
        if self.transform.sequence_len() != expected_child_len
            || self.transform.batch_count() != self.batch_count
            || self.transform.grouped_batch() != self.grouped_batch
            || self.transform.direction() != self.direction
            || self.transform.external_storage() != PrecisionStorage::DoubleDouble
        {
            return Err(VkFftError::InvalidKernelIr(
                "double-double real child transform metadata is inconsistent",
            ));
        }
        let expected_kind = match self.direction {
            Direction::Forward => RealFftKind::RealToComplex,
            Direction::Inverse => RealFftKind::ComplexToReal,
        };
        if self.kind != expected_kind {
            return Err(VkFftError::InvalidKernelIr(
                "double-double real kind/direction pair is inconsistent",
            ));
        }
        let roots_are_consistent = if self.even_half_size {
            self.length.is_multiple_of(2) && self.even_roots.len() == self.length / 2 + 1
        } else {
            self.even_roots.is_empty()
        };
        if !roots_are_consistent {
            return Err(VkFftError::InvalidKernelIr(
                "double-double real half-size/root metadata is inconsistent",
            ));
        }
        if self.normalize != (self.direction == Direction::Inverse && self.transform_normalizes()) {
            return Err(VkFftError::InvalidKernelIr(
                "double-double real inverse normalization metadata is inconsistent",
            ));
        }
        Ok(())
    }

    pub fn has_spatial_zero_padding(&self) -> bool {
        self.zero_padding.is_some()
    }

    pub fn contains_spatial_zero_index(&self, index: usize) -> bool {
        self.zero_padding.is_some_and(|range| range.contains(index))
    }

    pub fn batch_group_count(&self) -> usize {
        self.batch_count.div_ceil(self.grouped_batch)
    }

    fn transform_normalizes(&self) -> bool {
        match &self.transform {
            DoubleDoubleOneDimIr::Stockham(ir) => ir.normalize,
            DoubleDoubleOneDimIr::DirectRader(ir) => ir.normalize,
            DoubleDoubleOneDimIr::FftRader(ir) => ir.normalize,
            DoubleDoubleOneDimIr::Bluestein(ir) => ir.normalize,
            DoubleDoubleOneDimIr::Recursive(ir) => ir.normalize,
        }
    }
}

pub fn execute_double_double_r2c_ir(
    ir: &DoubleDoubleRealFftIr,
    input: &[DoubleDouble],
) -> Result<Vec<ComplexDoubleDouble>> {
    if ir.kind != RealFftKind::RealToComplex {
        return Err(VkFftError::UnsupportedKernelPath(
            "double-double R2C executor requires an R2C IR",
        ));
    }
    if ir.external_storage != PrecisionStorage::DoubleDouble {
        return Err(VkFftError::UnsupportedPrecision {
            backend: "double-double R2C DD-storage executor",
            precision: "IR uses F64 external storage",
        });
    }
    execute_double_double_r2c_compute(ir, input)
}

pub fn execute_double_double_r2c_ir_f64_storage(
    ir: &DoubleDoubleRealFftIr,
    input: &[f64],
) -> Result<Vec<Complex64>> {
    if ir.external_storage != PrecisionStorage::F64 {
        return Err(VkFftError::UnsupportedPrecision {
            backend: "double-double R2C F64-storage executor",
            precision: "IR uses double-double external storage",
        });
    }
    let promoted = input
        .iter()
        .copied()
        .map(DoubleDouble::from_f64)
        .collect::<Vec<_>>();
    execute_double_double_r2c_compute(ir, &promoted).map(|values| {
        values
            .into_iter()
            .map(ComplexDoubleDouble::to_complex64)
            .collect()
    })
}

fn execute_double_double_r2c_compute(
    ir: &DoubleDoubleRealFftIr,
    input: &[DoubleDouble],
) -> Result<Vec<ComplexDoubleDouble>> {
    ir.validate()?;
    let expected = ir
        .length
        .checked_mul(ir.batch_count)
        .ok_or(VkFftError::ArithmeticOverflow {
            operation: "double-double R2C input element count",
        })?;
    if input.len() != expected {
        return Err(VkFftError::InputLengthMismatch {
            expected,
            actual: input.len(),
        });
    }
    let padded_storage = if ir.has_spatial_zero_padding() {
        let mut values = input.to_vec();
        for batch in 0..ir.batch_count {
            let base = batch * ir.length;
            if let Some(range) = ir.zero_padding {
                for index in range.left..range.right {
                    values[base + index] = DoubleDouble::ZERO;
                }
            }
        }
        Some(values)
    } else {
        None
    };
    let input = padded_storage.as_deref().unwrap_or(input);
    if ir.even_half_size {
        let half = ir.length / 2;
        let mut packed = Vec::with_capacity(half * ir.batch_count);
        for batch in 0..ir.batch_count {
            let base = batch * ir.length;
            for index in 0..half {
                packed.push(ComplexDoubleDouble::new(
                    input[base + 2 * index],
                    input[base + 2 * index + 1],
                ));
            }
        }
        let spectrum = execute_double_double_one_dim_ir(&ir.transform, &packed)?;
        let half_scale = DoubleDouble::from_f64(0.5);
        let mut output = Vec::with_capacity(ir.half_spectrum_len * ir.batch_count);
        for batch in 0..ir.batch_count {
            let base = batch * half;
            for k in 0..=half {
                let a = spectrum[base + (k % half)];
                let b = spectrum[base + ((half - (k % half)) % half)].conj();
                let sum = a + b;
                let diff = a - b;
                let rotated = ir.even_roots[k] * diff;
                let minus_i = ComplexDoubleDouble::new(rotated.im, -rotated.re);
                output.push((sum + minus_i).scale_dd(half_scale));
            }
        }
        return Ok(output);
    }

    let full = input
        .iter()
        .copied()
        .map(|value| ComplexDoubleDouble::new(value, DoubleDouble::ZERO))
        .collect::<Vec<_>>();
    let transformed = execute_double_double_one_dim_ir(&ir.transform, &full)?;
    let mut output = Vec::with_capacity(ir.half_spectrum_len * ir.batch_count);
    for batch in 0..ir.batch_count {
        let base = batch * ir.length;
        output.extend_from_slice(&transformed[base..base + ir.half_spectrum_len]);
    }
    Ok(output)
}

pub fn execute_double_double_c2r_ir(
    ir: &DoubleDoubleRealFftIr,
    input: &[ComplexDoubleDouble],
) -> Result<Vec<DoubleDouble>> {
    if ir.kind != RealFftKind::ComplexToReal {
        return Err(VkFftError::UnsupportedKernelPath(
            "double-double C2R executor requires a C2R IR",
        ));
    }
    if ir.external_storage != PrecisionStorage::DoubleDouble {
        return Err(VkFftError::UnsupportedPrecision {
            backend: "double-double C2R DD-storage executor",
            precision: "IR uses F64 external storage",
        });
    }
    execute_double_double_c2r_compute(ir, input)
}

pub fn execute_double_double_c2r_ir_f64_storage(
    ir: &DoubleDoubleRealFftIr,
    input: &[Complex64],
) -> Result<Vec<f64>> {
    if ir.external_storage != PrecisionStorage::F64 {
        return Err(VkFftError::UnsupportedPrecision {
            backend: "double-double C2R F64-storage executor",
            precision: "IR uses double-double external storage",
        });
    }
    let promoted = input
        .iter()
        .copied()
        .map(ComplexDoubleDouble::from_complex64)
        .collect::<Vec<_>>();
    execute_double_double_c2r_compute(ir, &promoted)
        .map(|values| values.into_iter().map(DoubleDouble::to_f64).collect())
}

fn execute_double_double_c2r_compute(
    ir: &DoubleDoubleRealFftIr,
    input: &[ComplexDoubleDouble],
) -> Result<Vec<DoubleDouble>> {
    ir.validate()?;
    let expected =
        ir.half_spectrum_len
            .checked_mul(ir.batch_count)
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "double-double C2R input element count",
            })?;
    if input.len() != expected {
        return Err(VkFftError::InputLengthMismatch {
            expected,
            actual: input.len(),
        });
    }
    if ir.even_half_size {
        let half = ir.length / 2;
        let preprocess_scale = if ir.normalize {
            DoubleDouble::from_f64(0.5)
        } else {
            DoubleDouble::ONE
        };
        let mut packed_spectrum = Vec::with_capacity(half * ir.batch_count);
        for batch in 0..ir.batch_count {
            let compact_base = batch * ir.half_spectrum_len;
            for k in 0..half {
                let xk = input[compact_base + k];
                let c = input[compact_base + half - k].conj();
                let sum = xk + c;
                let diff = xk - c;
                let rotated = ir.even_roots[k].conj() * diff;
                let plus_i = ComplexDoubleDouble::new(-rotated.im, rotated.re);
                packed_spectrum.push((sum + plus_i).scale_dd(preprocess_scale));
            }
        }
        let packed_time = execute_double_double_one_dim_ir(&ir.transform, &packed_spectrum)?;
        let mut output = Vec::with_capacity(ir.length * ir.batch_count);
        for value in packed_time {
            output.push(value.re);
            output.push(value.im);
        }
        apply_double_double_real_zero_padding(ir, &mut output);
        return Ok(output);
    }

    let mut full = vec![ComplexDoubleDouble::default(); ir.length * ir.batch_count];
    for batch in 0..ir.batch_count {
        let compact_base = batch * ir.half_spectrum_len;
        let full_base = batch * ir.length;
        full[full_base..full_base + ir.half_spectrum_len]
            .copy_from_slice(&input[compact_base..compact_base + ir.half_spectrum_len]);
        for index in ir.half_spectrum_len..ir.length {
            full[full_base + index] = input[compact_base + ir.length - index].conj();
        }
    }
    let transformed = execute_double_double_one_dim_ir(&ir.transform, &full)?;
    let mut output = transformed
        .into_iter()
        .map(|value| value.re)
        .collect::<Vec<_>>();
    apply_double_double_real_zero_padding(ir, &mut output);
    Ok(output)
}

fn apply_double_double_real_zero_padding(ir: &DoubleDoubleRealFftIr, output: &mut [DoubleDouble]) {
    if let Some(range) = ir.zero_padding {
        for batch in 0..ir.batch_count {
            let base = batch * ir.length;
            for index in range.left..range.right {
                output[base + index] = DoubleDouble::ZERO;
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct DoubleDoubleNdFormattedIo {
    pub input_external_layout: NdExternalTensorLayout,
    pub output_external_layout: NdExternalTensorLayout,
    pub input_formatted_copy: Option<NdFormattedCopyPassIr>,
    pub output_formatted_copy: Option<NdFormattedCopyPassIr>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DoubleDoubleNdRealFftIr {
    pub kind: RealFftKind,
    pub dimensions: Vec<usize>,
    pub compact_dimensions: Vec<usize>,
    pub full_tensor_len: usize,
    pub compact_tensor_len: usize,
    pub batch_count: usize,
    pub real_grouped_batch: usize,
    pub external_storage: PrecisionStorage,
    pub formatted_io: Box<DoubleDoubleNdFormattedIo>,
    pub zero_padding: Vec<Option<ZeroPaddingRange>>,
    pub real_axis: DoubleDoubleRealFftIr,
    pub omitted_axes: Vec<bool>,
    pub complex_axes: Vec<DoubleDoubleNdAxisIr>,
}

impl DoubleDoubleNdRealFftIr {
    pub fn build(plan: &FftPlan) -> Result<Self> {
        Self::build_impl(plan, None)
    }

    pub fn build_for_device(plan: &FftPlan, device: DeviceProfile) -> Result<Self> {
        Self::build_impl(plan, Some(device))
    }

    fn build_impl(plan: &FftPlan, device: Option<DeviceProfile>) -> Result<Self> {
        if plan.config.dimensions.len() < 2 {
            return Err(VkFftError::UnsupportedKernelPath(
                "double-double multidimensional real FFT IR requires at least two dimensions",
            ));
        }
        let (kind, direction) = match plan.config.transform {
            TransformKind::RealToComplex => (RealFftKind::RealToComplex, Direction::Forward),
            TransformKind::ComplexToReal => (RealFftKind::ComplexToReal, Direction::Inverse),
            _ => {
                return Err(VkFftError::UnsupportedKernelPath(
                    "double-double multidimensional real FFT IR requires R2C or C2R",
                ));
            }
        };
        let external_storage = double_double_external_storage(plan.config.precision)?;
        let full_tensor_len =
            plan.config
                .dimensions
                .iter()
                .try_fold(1usize, |product, value| {
                    product
                        .checked_mul(*value)
                        .ok_or(VkFftError::ArithmeticOverflow {
                            operation: "double-double multidimensional real full tensor size",
                        })
                })?;
        let last_axis = plan.config.dimensions.len() - 1;
        let real_len = plan.config.dimensions[last_axis];
        let omitted_axes = (0..plan.config.dimensions.len())
            .map(|axis| plan.config.axis_is_omitted(axis))
            .collect::<Vec<_>>();
        if omitted_axes[last_axis] {
            return Err(VkFftError::UnsupportedKernelPath(
                "double-double multidimensional real FFT cannot omit the contiguous real axis",
            ));
        }
        let outer_lines = full_tensor_len / real_len;
        let real_grouped_batch_override = plan.config.grouped_batch_for_axis(last_axis);
        let real_grouped_batch = real_grouped_batch_override.unwrap_or(1);
        let real_batches = plan.config.batch_count.checked_mul(outer_lines).ok_or(
            VkFftError::ArithmeticOverflow {
                operation: "double-double multidimensional real last-axis batch count",
            },
        )?;
        let mut real_config = FftConfig::new(vec![real_len])
            .with_batch_count(real_batches)
            .with_precision(Precision::DoubleDouble)
            .with_transform(plan.config.transform)
            .with_inverse_normalization(plan.config.normalize_inverse)
            .with_tuning(plan.config.tuning)
            .with_bandwidth_boost(plan.config.bandwidth_boost);
        if let Some(grouped_batch) = real_grouped_batch_override {
            real_config = real_config.with_grouped_batch(0, grouped_batch)?;
        }
        let real_plan = if let Some(device) = device {
            FftPlan::build_for_device(real_config, device)?
        } else {
            FftPlan::build(real_config)?
        };
        let real_axis = if let Some(device) = device {
            DoubleDoubleRealFftIr::build_for_device(&real_plan, device)?
        } else {
            DoubleDoubleRealFftIr::build(&real_plan)?
        };
        let mut compact_dimensions = plan.config.dimensions.clone();
        compact_dimensions[last_axis] = real_axis.half_spectrum_len;
        let compact_tensor_len = compact_dimensions
            .iter()
            .try_fold(1usize, |product, value| {
                product
                    .checked_mul(*value)
                    .ok_or(VkFftError::ArithmeticOverflow {
                        operation: "double-double multidimensional real compact tensor size",
                    })
            })?;
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
        input_external_layout.validate(match kind {
            RealFftKind::RealToComplex => full_tensor_len,
            RealFftKind::ComplexToReal => compact_tensor_len,
        })?;
        output_external_layout.validate(match kind {
            RealFftKind::RealToComplex => compact_tensor_len,
            RealFftKind::ComplexToReal => full_tensor_len,
        })?;
        let external_scalar = double_double_storage_scalar(external_storage)?;
        let max_threads_per_block = device.map_or(256, |profile| profile.max_threads_per_block);
        let input_formatted_copy = (!input_external_layout.is_tightly_packed()?)
            .then(|| {
                NdFormattedCopyPassIr::new_with_max_threads(
                    "vkfft_dd_nd_real_gather_formatted_input".to_owned(),
                    ScalarType::DoubleDouble,
                    external_scalar,
                    plan.config.batch_count,
                    input_external_layout.clone(),
                    NdFormattedCopyOperation::GatherExternalToDense,
                    max_threads_per_block,
                )
            })
            .transpose()?;
        let output_formatted_copy = (!output_external_layout.is_tightly_packed()?)
            .then(|| {
                NdFormattedCopyPassIr::new_with_max_threads(
                    "vkfft_dd_nd_real_scatter_formatted_output".to_owned(),
                    ScalarType::DoubleDouble,
                    external_scalar,
                    plan.config.batch_count,
                    output_external_layout.clone(),
                    NdFormattedCopyOperation::ScatterDenseToExternal,
                    max_threads_per_block,
                )
            })
            .transpose()?;
        let mut complex_axes = Vec::with_capacity(last_axis);
        for axis in (0..last_axis).rev() {
            if omitted_axes[axis] {
                continue;
            }
            let axis_len = compact_dimensions[axis];
            let inner_stride =
                compact_dimensions[axis + 1..]
                    .iter()
                    .try_fold(1usize, |product, value| {
                        product
                        .checked_mul(*value)
                        .ok_or(VkFftError::ArithmeticOverflow {
                            operation: "double-double multidimensional real compact inner stride",
                        })
                    })?;
            let line_count = compact_tensor_len / axis_len;
            let grouped_batch_override = plan.config.grouped_batch_for_axis(axis);
            let grouped_batch = grouped_batch_override.unwrap_or(1);
            let transform_batch_count = plan.config.batch_count.checked_mul(line_count).ok_or(
                VkFftError::ArithmeticOverflow {
                    operation: "double-double multidimensional real complex-axis batch count",
                },
            )?;
            let mut axis_config = FftConfig::new(vec![axis_len])
                .with_batch_count(transform_batch_count)
                .with_precision(Precision::DoubleDouble)
                .with_inverse_normalization(
                    direction == Direction::Inverse && plan.config.normalize_inverse,
                )
                .with_tuning(plan.config.tuning)
                .with_bandwidth_boost(plan.config.bandwidth_boost);
            if let Some(grouped_batch) = grouped_batch_override {
                axis_config = axis_config.with_grouped_batch(0, grouped_batch)?;
            }
            let axis_plan = if let Some(device) = device {
                FftPlan::build_c2c_child_for_device(
                    axis_config,
                    device,
                    C2cDeviceAxisClass::Strided,
                )?
            } else {
                FftPlan::build(axis_config)?
            };
            let transform = if let Some(device) = device {
                DoubleDoubleOneDimIr::build_for_device(&axis_plan, direction, device)?
            } else {
                DoubleDoubleOneDimIr::build(&axis_plan, direction)?
            };
            complex_axes.push(DoubleDoubleNdAxisIr {
                axis,
                axis_len,
                inner_stride,
                line_count,
                grouped_batch,
                grouped_batch_override,
                transform,
            });
        }
        let ir = Self {
            kind,
            dimensions: plan.config.dimensions.clone(),
            compact_dimensions,
            full_tensor_len,
            compact_tensor_len,
            batch_count: plan.config.batch_count,
            real_grouped_batch,
            external_storage,
            formatted_io: Box::new(DoubleDoubleNdFormattedIo {
                input_external_layout,
                output_external_layout,
                input_formatted_copy,
                output_formatted_copy,
            }),
            zero_padding: plan.config.zero_padding.clone(),
            real_axis,
            omitted_axes,
            complex_axes,
        };
        ir.validate()?;
        Ok(ir)
    }

    pub fn with_grouped_stockham_axis_blocks(
        mut self,
        axis1_grouped_batch_override: Option<usize>,
        device: DeviceProfile,
    ) -> Result<Self> {
        self.real_axis = self.real_axis.with_grouped_stockham_child_block(device)?;
        self =
            self.with_grouped_stockham_complex_axis_blocks(axis1_grouped_batch_override, device)?;
        self.validate()?;
        Ok(self)
    }

    pub fn with_grouped_stockham_complex_axis_blocks(
        mut self,
        axis1_grouped_batch_override: Option<usize>,
        device: DeviceProfile,
    ) -> Result<Self> {
        let fastest_axis_len =
            *self
                .compact_dimensions
                .last()
                .ok_or(VkFftError::InvalidKernelIr(
                    "double-double ND real compact dimensions are empty",
                ))?;
        for axis in &mut self.complex_axes {
            axis.transform = axis
                .transform
                .clone()
                .with_other_axis_grouped_physical_block(
                    fastest_axis_len,
                    axis.grouped_batch_override,
                    axis1_grouped_batch_override,
                    device,
                )?;
        }
        self.validate()?;
        Ok(self)
    }

    pub fn validate(&self) -> Result<()> {
        if self.dimensions.len() < 2
            || self.compact_dimensions.len() != self.dimensions.len()
            || self.zero_padding.len() != self.dimensions.len()
            || self.omitted_axes.len() != self.dimensions.len()
            || self.batch_count == 0
            || self.real_grouped_batch == 0
        {
            return Err(VkFftError::InvalidKernelIr(
                "double-double multidimensional real tensor metadata is inconsistent",
            ));
        }
        let full_tensor_len = self.dimensions.iter().try_fold(1usize, |product, value| {
            product
                .checked_mul(*value)
                .ok_or(VkFftError::ArithmeticOverflow {
                    operation: "double-double multidimensional real validation full size",
                })
        })?;
        let compact_tensor_len =
            self.compact_dimensions
                .iter()
                .try_fold(1usize, |product, value| {
                    product
                    .checked_mul(*value)
                    .ok_or(VkFftError::ArithmeticOverflow {
                        operation: "double-double multidimensional real validation compact size",
                    })
                })?;
        if full_tensor_len != self.full_tensor_len || compact_tensor_len != self.compact_tensor_len
        {
            return Err(VkFftError::InvalidKernelIr(
                "double-double multidimensional real tensor lengths are inconsistent",
            ));
        }
        if !matches!(
            self.external_storage,
            PrecisionStorage::DoubleDouble | PrecisionStorage::F64
        ) {
            return Err(VkFftError::InvalidKernelIr(
                "double-double multidimensional real external storage must be DD or F64",
            ));
        }
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
        let formatted_io = self.formatted_io.as_ref();
        formatted_io
            .input_external_layout
            .validate(input_tensor_len)?;
        formatted_io
            .output_external_layout
            .validate(output_tensor_len)?;
        if &formatted_io.input_external_layout.dimensions != input_dimensions
            || &formatted_io.output_external_layout.dimensions != output_dimensions
        {
            return Err(VkFftError::InvalidKernelIr(
                "double-double multidimensional real formatted tensor dimensions are inconsistent",
            ));
        }
        let input_needs_copy = !formatted_io.input_external_layout.is_tightly_packed()?;
        let output_needs_copy = !formatted_io.output_external_layout.is_tightly_packed()?;
        if formatted_io.input_formatted_copy.is_some() != input_needs_copy
            || formatted_io.output_formatted_copy.is_some() != output_needs_copy
        {
            return Err(VkFftError::InvalidKernelIr(
                "double-double multidimensional real formatted copy ownership is inconsistent",
            ));
        }
        let external_scalar = double_double_storage_scalar(self.external_storage)?;
        if let Some(copy) = &formatted_io.input_formatted_copy {
            copy.validate()?;
            if copy.operation != NdFormattedCopyOperation::GatherExternalToDense
                || copy.external_layout != formatted_io.input_external_layout
                || copy.batch_count != self.batch_count
                || copy.scalar != ScalarType::DoubleDouble
                || copy.input_storage_scalar != external_scalar
                || copy.output_storage_scalar != ScalarType::DoubleDouble
            {
                return Err(VkFftError::InvalidKernelIr(
                    "double-double multidimensional real formatted input copy metadata is inconsistent",
                ));
            }
        }
        if let Some(copy) = &formatted_io.output_formatted_copy {
            copy.validate()?;
            if copy.operation != NdFormattedCopyOperation::ScatterDenseToExternal
                || copy.external_layout != formatted_io.output_external_layout
                || copy.batch_count != self.batch_count
                || copy.scalar != ScalarType::DoubleDouble
                || copy.input_storage_scalar != ScalarType::DoubleDouble
                || copy.output_storage_scalar != external_scalar
            {
                return Err(VkFftError::InvalidKernelIr(
                    "double-double multidimensional real formatted output copy metadata is inconsistent",
                ));
            }
        }
        for (axis, (length, range)) in self
            .dimensions
            .iter()
            .copied()
            .zip(self.zero_padding.iter().copied())
            .enumerate()
        {
            if let Some(range) = range
                && (range.left > range.right || range.right > length)
            {
                return Err(VkFftError::InvalidZeroPaddingRange {
                    axis,
                    left: range.left,
                    right: range.right,
                    length,
                });
            }
        }
        self.real_axis.validate()?;
        let last_axis = self.dimensions.len() - 1;
        if self.omitted_axes[last_axis] {
            return Err(VkFftError::InvalidKernelIr(
                "double-double multidimensional real FFT omitted the contiguous real axis",
            ));
        }
        let direction = match self.kind {
            RealFftKind::RealToComplex => Direction::Forward,
            RealFftKind::ComplexToReal => Direction::Inverse,
        };
        if self.real_axis.kind != self.kind
            || self.real_axis.length != self.dimensions[last_axis]
            || self.real_axis.half_spectrum_len != self.compact_dimensions[last_axis]
            || self.real_axis.batch_count
                != self.batch_count * (self.full_tensor_len / self.real_axis.length)
            || self.real_axis.grouped_batch != self.real_grouped_batch
            || self.real_axis.direction != direction
            || self.real_axis.external_storage != PrecisionStorage::DoubleDouble
            || self.complex_axes.len()
                != (0..last_axis)
                    .filter(|&axis| !self.omitted_axes[axis])
                    .count()
        {
            return Err(VkFftError::InvalidKernelIr(
                "double-double multidimensional real last-axis metadata is inconsistent",
            ));
        }
        let expected_axis_order = (0..last_axis)
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
                "double-double multidimensional real omitted-axis execution order is inconsistent",
            ));
        }
        for axis in &self.complex_axes {
            axis.transform.validate()?;
            if axis.axis >= last_axis {
                return Err(VkFftError::InvalidKernelIr(
                    "double-double multidimensional real complex axis is out of range",
                ));
            }
            if self.omitted_axes[axis.axis] {
                return Err(VkFftError::InvalidKernelIr(
                    "double-double multidimensional real materialized an omitted axis",
                ));
            }
            let expected_stride = self.compact_dimensions[axis.axis + 1..]
                .iter()
                .product::<usize>();
            if axis.axis_len != self.compact_dimensions[axis.axis]
                || axis.inner_stride != expected_stride
                || axis.line_count != self.compact_tensor_len / axis.axis_len
                || axis.grouped_batch == 0
                || axis.grouped_batch_override.unwrap_or(1) != axis.grouped_batch
                || axis.transform.sequence_len() != axis.axis_len
                || axis.transform.batch_count() != self.batch_count * axis.line_count
                || axis.transform.grouped_batch() != axis.grouped_batch
                || axis.transform.direction() != direction
                || axis.transform.external_storage() != PrecisionStorage::DoubleDouble
            {
                return Err(VkFftError::InvalidKernelIr(
                    "double-double multidimensional real complex-axis metadata is inconsistent",
                ));
            }
        }
        Ok(())
    }
}

impl DoubleDoubleNdRealFftIr {
    pub fn batch_group_count(&self) -> usize {
        self.batch_count.div_ceil(self.real_grouped_batch)
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
                operation: "double-double formatted ND real logical input element count",
            })?;
        if input.len() != expected {
            return Err(VkFftError::InputLengthMismatch {
                expected,
                actual: input.len(),
            });
        }
        let layout = &self.formatted_io.input_external_layout;
        if layout.is_tightly_packed()? {
            return Ok(input.to_vec());
        }
        pack_logical_tensor_batches(
            input,
            &layout.dimensions,
            &layout.axis_strides,
            layout.batch_stride,
        )
    }

    pub(crate) fn unpack_formatted_output<T: Copy>(&self, output: &[T]) -> Result<Vec<T>> {
        self.validate()?;
        let layout = &self.formatted_io.output_external_layout;
        if layout.is_tightly_packed()? {
            return Ok(output.to_vec());
        }
        unpack_logical_tensor_batches(
            output,
            &layout.dimensions,
            &layout.axis_strides,
            layout.batch_stride,
            self.batch_count,
        )
    }

    fn logicalize_formatted_input<T: Copy + Default>(&self, input: &[T]) -> Result<Vec<T>> {
        let physical = self.pack_formatted_input(input)?;
        let layout = &self.formatted_io.input_external_layout;
        if layout.is_tightly_packed()? {
            return Ok(physical);
        }
        unpack_logical_tensor_batches(
            &physical,
            &layout.dimensions,
            &layout.axis_strides,
            layout.batch_stride,
            self.batch_count,
        )
    }

    fn round_trip_formatted_output<T: Copy + Default>(&self, output: &[T]) -> Result<Vec<T>> {
        let layout = &self.formatted_io.output_external_layout;
        if layout.is_tightly_packed()? {
            return Ok(output.to_vec());
        }
        let physical = pack_logical_tensor_batches(
            output,
            &layout.dimensions,
            &layout.axis_strides,
            layout.batch_stride,
        )?;
        self.unpack_formatted_output(&physical)
    }

    pub fn has_spatial_zero_padding(&self) -> bool {
        self.zero_padding.iter().any(Option::is_some)
    }

    pub fn contains_spatial_zero_linear_index(&self, mut index: usize) -> bool {
        for axis in (0..self.dimensions.len()).rev() {
            let length = self.dimensions[axis];
            let coordinate = index % length;
            index /= length;
            if self.zero_padding[axis].is_some_and(|range| range.contains(coordinate)) {
                return true;
            }
        }
        false
    }
}

pub fn execute_double_double_nd_r2c_ir(
    ir: &DoubleDoubleNdRealFftIr,
    input: &[DoubleDouble],
) -> Result<Vec<ComplexDoubleDouble>> {
    if ir.external_storage != PrecisionStorage::DoubleDouble {
        return Err(VkFftError::UnsupportedPrecision {
            backend: "double-double ND R2C DD-storage executor",
            precision: "IR uses F64 external storage",
        });
    }
    let logical_input = ir.logicalize_formatted_input(input)?;
    let logical_output = execute_double_double_nd_r2c_compute(ir, &logical_input)?;
    ir.round_trip_formatted_output(&logical_output)
}

pub fn execute_double_double_nd_r2c_ir_f64_storage(
    ir: &DoubleDoubleNdRealFftIr,
    input: &[f64],
) -> Result<Vec<Complex64>> {
    if ir.external_storage != PrecisionStorage::F64 {
        return Err(VkFftError::UnsupportedPrecision {
            backend: "double-double ND R2C F64-storage executor",
            precision: "IR uses double-double external storage",
        });
    }
    let logical_input = ir.logicalize_formatted_input(input)?;
    let promoted = logical_input
        .iter()
        .copied()
        .map(DoubleDouble::from_f64)
        .collect::<Vec<_>>();
    let logical_output = execute_double_double_nd_r2c_compute(ir, &promoted)?
        .into_iter()
        .map(ComplexDoubleDouble::to_complex64)
        .collect::<Vec<_>>();
    ir.round_trip_formatted_output(&logical_output)
}

fn execute_double_double_nd_r2c_compute(
    ir: &DoubleDoubleNdRealFftIr,
    input: &[DoubleDouble],
) -> Result<Vec<ComplexDoubleDouble>> {
    ir.validate()?;
    if ir.kind != RealFftKind::RealToComplex {
        return Err(VkFftError::UnsupportedKernelPath(
            "double-double multidimensional R2C executor requires an R2C IR",
        ));
    }
    let expected =
        ir.full_tensor_len
            .checked_mul(ir.batch_count)
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "double-double multidimensional R2C input element count",
            })?;
    if input.len() != expected {
        return Err(VkFftError::InputLengthMismatch {
            expected,
            actual: input.len(),
        });
    }
    let padded_storage = if ir.has_spatial_zero_padding() {
        let mut values = input.to_vec();
        for batch in 0..ir.batch_count {
            let base = batch * ir.full_tensor_len;
            for linear in 0..ir.full_tensor_len {
                if ir.contains_spatial_zero_linear_index(linear) {
                    values[base + linear] = DoubleDouble::ZERO;
                }
            }
        }
        Some(values)
    } else {
        None
    };
    let input = padded_storage.as_deref().unwrap_or(input);
    let current = execute_double_double_r2c_ir(&ir.real_axis, input)?;
    execute_double_double_nd_real_complex_axes(ir, current)
}

pub fn execute_double_double_nd_c2r_ir(
    ir: &DoubleDoubleNdRealFftIr,
    input: &[ComplexDoubleDouble],
) -> Result<Vec<DoubleDouble>> {
    if ir.external_storage != PrecisionStorage::DoubleDouble {
        return Err(VkFftError::UnsupportedPrecision {
            backend: "double-double ND C2R DD-storage executor",
            precision: "IR uses F64 external storage",
        });
    }
    let logical_input = ir.logicalize_formatted_input(input)?;
    let logical_output = execute_double_double_nd_c2r_compute(ir, &logical_input)?;
    ir.round_trip_formatted_output(&logical_output)
}

pub fn execute_double_double_nd_c2r_ir_f64_storage(
    ir: &DoubleDoubleNdRealFftIr,
    input: &[Complex64],
) -> Result<Vec<f64>> {
    if ir.external_storage != PrecisionStorage::F64 {
        return Err(VkFftError::UnsupportedPrecision {
            backend: "double-double ND C2R F64-storage executor",
            precision: "IR uses double-double external storage",
        });
    }
    let logical_input = ir.logicalize_formatted_input(input)?;
    let promoted = logical_input
        .iter()
        .copied()
        .map(ComplexDoubleDouble::from_complex64)
        .collect::<Vec<_>>();
    let logical_output = execute_double_double_nd_c2r_compute(ir, &promoted)?
        .into_iter()
        .map(DoubleDouble::to_f64)
        .collect::<Vec<_>>();
    ir.round_trip_formatted_output(&logical_output)
}

fn execute_double_double_nd_c2r_compute(
    ir: &DoubleDoubleNdRealFftIr,
    input: &[ComplexDoubleDouble],
) -> Result<Vec<DoubleDouble>> {
    ir.validate()?;
    if ir.kind != RealFftKind::ComplexToReal {
        return Err(VkFftError::UnsupportedKernelPath(
            "double-double multidimensional C2R executor requires a C2R IR",
        ));
    }
    let expected = ir.compact_tensor_len.checked_mul(ir.batch_count).ok_or(
        VkFftError::ArithmeticOverflow {
            operation: "double-double multidimensional C2R input element count",
        },
    )?;
    if input.len() != expected {
        return Err(VkFftError::InputLengthMismatch {
            expected,
            actual: input.len(),
        });
    }
    let current = execute_double_double_nd_real_complex_axes(ir, input.to_vec())?;
    let mut output = execute_double_double_c2r_ir(&ir.real_axis, &current)?;
    if ir.has_spatial_zero_padding() {
        for batch in 0..ir.batch_count {
            let base = batch * ir.full_tensor_len;
            for linear in 0..ir.full_tensor_len {
                if ir.contains_spatial_zero_linear_index(linear) {
                    output[base + linear] = DoubleDouble::ZERO;
                }
            }
        }
    }
    Ok(output)
}

fn execute_double_double_nd_real_complex_axes(
    ir: &DoubleDoubleNdRealFftIr,
    mut current: Vec<ComplexDoubleDouble>,
) -> Result<Vec<ComplexDoubleDouble>> {
    let expected = ir.compact_tensor_len.checked_mul(ir.batch_count).ok_or(
        VkFftError::ArithmeticOverflow {
            operation: "double-double multidimensional real compact execution size",
        },
    )?;
    for axis in &ir.complex_axes {
        let outer_count = ir.compact_tensor_len / (axis.axis_len * axis.inner_stride);
        let transform_batch_count = ir.batch_count * axis.line_count;
        let mut packed = Vec::with_capacity(transform_batch_count * axis.axis_len);
        for batch in 0..ir.batch_count {
            let batch_base = batch * ir.compact_tensor_len;
            for outer in 0..outer_count {
                let outer_base = batch_base + outer * axis.axis_len * axis.inner_stride;
                for inner in 0..axis.inner_stride {
                    for lane in 0..axis.axis_len {
                        packed.push(current[outer_base + lane * axis.inner_stride + inner]);
                    }
                }
            }
        }
        let transformed = execute_double_double_one_dim_ir(&axis.transform, &packed)?;
        let mut next = vec![ComplexDoubleDouble::default(); expected];
        let mut cursor = 0usize;
        for batch in 0..ir.batch_count {
            let batch_base = batch * ir.compact_tensor_len;
            for outer in 0..outer_count {
                let outer_base = batch_base + outer * axis.axis_len * axis.inner_stride;
                for inner in 0..axis.inner_stride {
                    for lane in 0..axis.axis_len {
                        next[outer_base + lane * axis.inner_stride + inner] = transformed[cursor];
                        cursor += 1;
                    }
                }
            }
        }
        current = next;
    }
    Ok(current)
}

#[derive(Debug, Clone, PartialEq)]
pub struct DoubleDoubleNdAxisIr {
    pub axis: usize,
    pub axis_len: usize,
    pub inner_stride: usize,
    pub line_count: usize,
    pub grouped_batch: usize,
    pub grouped_batch_override: Option<usize>,
    pub transform: DoubleDoubleOneDimIr,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DoubleDoubleNdFftIr {
    pub dimensions: Vec<usize>,
    pub tensor_len: usize,
    pub batch_count: usize,
    pub direction: Direction,
    pub external_storage: PrecisionStorage,
    pub input_external_layout: NdExternalTensorLayout,
    pub output_external_layout: NdExternalTensorLayout,
    pub input_formatted_copy: Option<NdFormattedCopyPassIr>,
    pub output_formatted_copy: Option<NdFormattedCopyPassIr>,
    pub zero_padding: Vec<Option<ZeroPaddingRange>>,
    pub zero_padding_domain: ZeroPaddingDomain,
    pub omitted_axes: Vec<bool>,
    pub axes: Vec<DoubleDoubleNdAxisIr>,
}

impl DoubleDoubleNdFftIr {
    pub fn build(plan: &FftPlan, direction: Direction) -> Result<Self> {
        Self::build_impl(plan, direction, None)
    }

    pub fn build_for_device(
        plan: &FftPlan,
        direction: Direction,
        device: DeviceProfile,
    ) -> Result<Self> {
        Self::build_impl(plan, direction, Some(device))
    }

    fn build_impl(
        plan: &FftPlan,
        direction: Direction,
        device: Option<DeviceProfile>,
    ) -> Result<Self> {
        if plan.config.transform != TransformKind::ComplexToComplex
            || plan.config.dimensions.len() < 2
        {
            return Err(VkFftError::UnsupportedKernelPath(
                "double-double ND FFT IR requires a multidimensional C2C plan",
            ));
        }
        let external_storage = double_double_external_storage(plan.config.precision)?;
        if plan.axes.len() != plan.config.dimensions.len() {
            return Err(VkFftError::InvalidKernelIr(
                "double-double ND plan axis metadata does not match dimensions",
            ));
        }
        let tensor_len = plan
            .config
            .dimensions
            .iter()
            .try_fold(1usize, |product, value| {
                product
                    .checked_mul(*value)
                    .ok_or(VkFftError::ArithmeticOverflow {
                        operation: "double-double ND tensor element count",
                    })
            })?;
        let input_external_layout = NdExternalTensorLayout {
            dimensions: plan.config.dimensions.clone(),
            axis_strides: plan.config.resolved_input_buffer_axis_strides()?,
            batch_stride: plan.config.resolved_input_buffer_batch_stride()?,
        };
        let output_external_layout = NdExternalTensorLayout {
            dimensions: plan.config.dimensions.clone(),
            axis_strides: plan.config.resolved_output_buffer_axis_strides()?,
            batch_stride: plan.config.resolved_output_buffer_batch_stride()?,
        };
        input_external_layout.validate(tensor_len)?;
        output_external_layout.validate(tensor_len)?;
        let external_scalar = double_double_storage_scalar(external_storage)?;
        let max_threads_per_block = device.map_or(256, |profile| profile.max_threads_per_block);
        let input_formatted_copy = (!input_external_layout.is_tightly_packed()?)
            .then(|| {
                NdFormattedCopyPassIr::new_with_max_threads(
                    "vkfft_dd_nd_gather_formatted_input".to_owned(),
                    ScalarType::DoubleDouble,
                    external_scalar,
                    plan.config.batch_count,
                    input_external_layout.clone(),
                    NdFormattedCopyOperation::GatherExternalToDense,
                    max_threads_per_block,
                )
            })
            .transpose()?;
        let output_formatted_copy = (!output_external_layout.is_tightly_packed()?)
            .then(|| {
                NdFormattedCopyPassIr::new_with_max_threads(
                    "vkfft_dd_nd_scatter_formatted_output".to_owned(),
                    ScalarType::DoubleDouble,
                    external_scalar,
                    plan.config.batch_count,
                    output_external_layout.clone(),
                    NdFormattedCopyOperation::ScatterDenseToExternal,
                    max_threads_per_block,
                )
            })
            .transpose()?;
        let omitted_axes = (0..plan.config.dimensions.len())
            .map(|axis| plan.config.axis_is_omitted(axis))
            .collect::<Vec<_>>();
        let mut axes = Vec::with_capacity(plan.axes.len());
        for axis in (0..plan.axes.len()).rev() {
            if omitted_axes[axis] {
                continue;
            }
            let axis_len = plan.axes[axis].effective_fft_len;
            if axis_len != plan.config.dimensions[axis] {
                return Err(VkFftError::UnsupportedKernelPath(
                    "double-double ND requires unchanged effective axis lengths",
                ));
            }
            let inner_stride =
                plan.config.dimensions[axis + 1..]
                    .iter()
                    .try_fold(1usize, |product, value| {
                        product
                            .checked_mul(*value)
                            .ok_or(VkFftError::ArithmeticOverflow {
                                operation: "double-double ND inner stride",
                            })
                    })?;
            let line_count = tensor_len / axis_len;
            let grouped_batch_override = plan.config.grouped_batch_for_axis(axis);
            let grouped_batch = grouped_batch_override.unwrap_or(1);
            let transform_batch_count = plan.config.batch_count.checked_mul(line_count).ok_or(
                VkFftError::ArithmeticOverflow {
                    operation: "double-double ND transform batch count",
                },
            )?;
            let mut axis_config = FftConfig::new(vec![axis_len])
                .with_batch_count(transform_batch_count)
                .with_precision(Precision::DoubleDouble)
                .with_inverse_normalization(plan.config.normalize_inverse)
                .with_tuning(plan.config.tuning)
                .with_bandwidth_boost(plan.config.bandwidth_boost);
            if let Some(grouped_batch) = grouped_batch_override {
                axis_config = axis_config.with_grouped_batch(0, grouped_batch)?;
            }
            let axis_plan = if let Some(device) = device {
                let axis_class = if inner_stride == 1 {
                    C2cDeviceAxisClass::Contiguous
                } else {
                    C2cDeviceAxisClass::Strided
                };
                FftPlan::build_c2c_child_for_device(axis_config, device, axis_class)?
            } else {
                FftPlan::build(axis_config)?
            };
            let transform = if let Some(device) = device {
                DoubleDoubleOneDimIr::build_for_device(&axis_plan, direction, device)?
            } else {
                DoubleDoubleOneDimIr::build(&axis_plan, direction)?
            };
            axes.push(DoubleDoubleNdAxisIr {
                axis,
                axis_len,
                inner_stride,
                line_count,
                grouped_batch,
                grouped_batch_override,
                transform,
            });
        }
        let ir = Self {
            dimensions: plan.config.dimensions.clone(),
            tensor_len,
            batch_count: plan.config.batch_count,
            direction,
            external_storage,
            input_external_layout,
            output_external_layout,
            input_formatted_copy,
            output_formatted_copy,
            zero_padding: plan.config.zero_padding.clone(),
            zero_padding_domain: plan.config.zero_padding_domain,
            omitted_axes,
            axes,
        };
        ir.validate()?;
        Ok(ir)
    }

    pub fn with_grouped_stockham_axis_blocks(
        mut self,
        axis1_grouped_batch_override: Option<usize>,
        device: DeviceProfile,
    ) -> Result<Self> {
        let last_axis = self
            .dimensions
            .len()
            .checked_sub(1)
            .ok_or(VkFftError::InvalidKernelIr(
                "double-double ND dimensions are empty",
            ))?;
        let fastest_axis_len = self.dimensions[last_axis];
        for axis in &mut self.axes {
            axis.transform = if axis.inner_stride == 1 {
                axis.transform
                    .clone()
                    .with_axis0_grouped_physical_block(device)?
            } else {
                axis.transform
                    .clone()
                    .with_other_axis_grouped_physical_block(
                        fastest_axis_len,
                        axis.grouped_batch_override,
                        axis1_grouped_batch_override,
                        device,
                    )?
            };
        }
        self.validate()?;
        Ok(self)
    }

    pub fn validate(&self) -> Result<()> {
        if self.dimensions.len() < 2
            || self.zero_padding.len() != self.dimensions.len()
            || self.omitted_axes.len() != self.dimensions.len()
            || self.tensor_len == 0
            || self.batch_count == 0
        {
            return Err(VkFftError::InvalidKernelIr(
                "double-double ND dimensions, padding, omitted axes, and batch count must be consistent",
            ));
        }
        let tensor_len = self.dimensions.iter().try_fold(1usize, |product, value| {
            product
                .checked_mul(*value)
                .ok_or(VkFftError::ArithmeticOverflow {
                    operation: "double-double ND validation tensor size",
                })
        })?;
        if tensor_len != self.tensor_len
            || self.axes.len() != self.omitted_axes.iter().filter(|&&omit| !omit).count()
        {
            return Err(VkFftError::InvalidKernelIr(
                "double-double ND tensor metadata is inconsistent",
            ));
        }
        if !matches!(
            self.external_storage,
            PrecisionStorage::DoubleDouble | PrecisionStorage::F64
        ) {
            return Err(VkFftError::InvalidKernelIr(
                "double-double ND external storage must be DD or F64",
            ));
        }
        let external_scalar = double_double_storage_scalar(self.external_storage)?;
        for (layout, copy, operation) in [
            (
                &self.input_external_layout,
                self.input_formatted_copy.as_ref(),
                NdFormattedCopyOperation::GatherExternalToDense,
            ),
            (
                &self.output_external_layout,
                self.output_formatted_copy.as_ref(),
                NdFormattedCopyOperation::ScatterDenseToExternal,
            ),
        ] {
            layout.validate(self.tensor_len)?;
            let tightly_packed = layout.is_tightly_packed()?;
            if tightly_packed != copy.is_none() {
                return Err(VkFftError::InvalidKernelIr(
                    "double-double ND formatted copy ownership does not match the external layout",
                ));
            }
            if let Some(copy) = copy {
                copy.validate()?;
                let expected_storage = match operation {
                    NdFormattedCopyOperation::GatherExternalToDense => {
                        (external_scalar, ScalarType::DoubleDouble)
                    }
                    NdFormattedCopyOperation::ScatterDenseToExternal => {
                        (ScalarType::DoubleDouble, external_scalar)
                    }
                };
                if copy.scalar != ScalarType::DoubleDouble
                    || copy.tensor_len != self.tensor_len
                    || copy.batch_count != self.batch_count
                    || copy.external_layout != *layout
                    || copy.operation != operation
                    || (copy.input_storage_scalar, copy.output_storage_scalar) != expected_storage
                {
                    return Err(VkFftError::InvalidKernelIr(
                        "double-double ND formatted copy metadata is inconsistent",
                    ));
                }
            }
        }
        for (axis, (length, range)) in self
            .dimensions
            .iter()
            .copied()
            .zip(self.zero_padding.iter().copied())
            .enumerate()
        {
            if let Some(range) = range
                && (range.left > range.right || range.right > length)
            {
                return Err(VkFftError::InvalidZeroPaddingRange {
                    axis,
                    left: range.left,
                    right: range.right,
                    length,
                });
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
                "double-double ND omitted-axis execution order is inconsistent",
            ));
        }
        for axis in &self.axes {
            axis.transform.validate()?;
            if axis.axis >= self.dimensions.len() {
                return Err(VkFftError::InvalidKernelIr(
                    "double-double ND axis index is out of range",
                ));
            }
            if self.omitted_axes[axis.axis] {
                return Err(VkFftError::InvalidKernelIr(
                    "double-double ND materialized an omitted axis",
                ));
            }
            let expected_stride = self.dimensions[axis.axis + 1..].iter().product::<usize>();
            if axis.axis_len != self.dimensions[axis.axis]
                || axis.inner_stride != expected_stride
                || axis.line_count != self.tensor_len / axis.axis_len
                || axis.grouped_batch == 0
                || axis.grouped_batch_override.unwrap_or(1) != axis.grouped_batch
                || axis.transform.sequence_len() != axis.axis_len
                || axis.transform.batch_count() != self.batch_count * axis.line_count
                || axis.transform.grouped_batch() != axis.grouped_batch
                || axis.transform.direction() != self.direction
                || axis.transform.external_storage() != PrecisionStorage::DoubleDouble
            {
                return Err(VkFftError::InvalidKernelIr(
                    "double-double ND axis metadata is inconsistent",
                ));
            }
        }
        Ok(())
    }

    pub(crate) fn pack_formatted_input<T: Copy + Default>(&self, input: &[T]) -> Result<Vec<T>> {
        self.validate()?;
        let expected = self.tensor_len.checked_mul(self.batch_count).ok_or(
            VkFftError::ArithmeticOverflow {
                operation: "double-double formatted ND logical input element count",
            },
        )?;
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

    fn unpack_formatted_input<T: Copy>(&self, input: &[T]) -> Result<Vec<T>> {
        if self.input_formatted_copy.is_none() {
            return Ok(input.to_vec());
        }
        unpack_logical_tensor_batches(
            input,
            &self.input_external_layout.dimensions,
            &self.input_external_layout.axis_strides,
            self.input_external_layout.batch_stride,
            self.batch_count,
        )
    }

    fn pack_formatted_output<T: Copy + Default>(&self, output: &[T]) -> Result<Vec<T>> {
        if self.output_formatted_copy.is_none() {
            return Ok(output.to_vec());
        }
        pack_logical_tensor_batches(
            output,
            &self.output_external_layout.dimensions,
            &self.output_external_layout.axis_strides,
            self.output_external_layout.batch_stride,
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

    pub fn has_zero_padding(&self) -> bool {
        self.zero_padding.iter().any(Option::is_some)
    }

    pub fn has_spatial_zero_padding(&self) -> bool {
        self.zero_padding_domain == ZeroPaddingDomain::Spatial && self.has_zero_padding()
    }

    pub fn has_frequency_zero_padding(&self) -> bool {
        self.zero_padding_domain == ZeroPaddingDomain::Frequency && self.has_zero_padding()
    }

    pub fn zero_padding_is_input_boundary(&self) -> bool {
        self.has_zero_padding()
            && matches!(
                (self.direction, self.zero_padding_domain),
                (Direction::Forward, ZeroPaddingDomain::Spatial)
                    | (Direction::Inverse, ZeroPaddingDomain::Frequency)
            )
    }

    pub fn zero_padding_is_output_boundary(&self) -> bool {
        self.has_zero_padding()
            && matches!(
                (self.direction, self.zero_padding_domain),
                (Direction::Inverse, ZeroPaddingDomain::Spatial)
                    | (Direction::Forward, ZeroPaddingDomain::Frequency)
            )
    }

    pub fn contains_zero_linear_index(&self, mut index: usize) -> bool {
        for axis in (0..self.dimensions.len()).rev() {
            let length = self.dimensions[axis];
            let coordinate = index % length;
            index /= length;
            if self.zero_padding[axis].is_some_and(|range| range.contains(coordinate)) {
                return true;
            }
        }
        false
    }

    pub fn contains_spatial_zero_linear_index(&self, index: usize) -> bool {
        self.has_spatial_zero_padding() && self.contains_zero_linear_index(index)
    }
}

pub fn execute_double_double_nd_ir(
    ir: &DoubleDoubleNdFftIr,
    input: &[ComplexDoubleDouble],
) -> Result<Vec<ComplexDoubleDouble>> {
    if ir.external_storage != PrecisionStorage::DoubleDouble {
        return Err(VkFftError::UnsupportedPrecision {
            backend: "double-double ND DD-storage executor",
            precision: "IR uses F64 external storage",
        });
    }
    let physical_input = ir.pack_formatted_input(input)?;
    let dense_input = ir.unpack_formatted_input(&physical_input)?;
    let dense_output = execute_double_double_nd_compute(ir, &dense_input)?;
    let physical_output = ir.pack_formatted_output(&dense_output)?;
    ir.unpack_formatted_output(&physical_output)
}

pub fn execute_double_double_nd_ir_f64_storage(
    ir: &DoubleDoubleNdFftIr,
    input: &[Complex64],
) -> Result<Vec<Complex64>> {
    if ir.external_storage != PrecisionStorage::F64 {
        return Err(VkFftError::UnsupportedPrecision {
            backend: "double-double ND F64-storage executor",
            precision: "IR uses double-double external storage",
        });
    }
    let physical_input = ir.pack_formatted_input(input)?;
    let dense_input = ir.unpack_formatted_input(&physical_input)?;
    let promoted = dense_input
        .iter()
        .copied()
        .map(ComplexDoubleDouble::from_complex64)
        .collect::<Vec<_>>();
    let dense_output = execute_double_double_nd_compute(ir, &promoted)?
        .into_iter()
        .map(ComplexDoubleDouble::to_complex64)
        .collect::<Vec<_>>();
    let physical_output = ir.pack_formatted_output(&dense_output)?;
    ir.unpack_formatted_output(&physical_output)
}

fn execute_double_double_nd_compute(
    ir: &DoubleDoubleNdFftIr,
    input: &[ComplexDoubleDouble],
) -> Result<Vec<ComplexDoubleDouble>> {
    ir.validate()?;
    let expected =
        ir.tensor_len
            .checked_mul(ir.batch_count)
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "double-double ND input element count",
            })?;
    if input.len() != expected {
        return Err(VkFftError::InputLengthMismatch {
            expected,
            actual: input.len(),
        });
    }
    let mut tensor = input.to_vec();
    if ir.zero_padding_is_input_boundary() {
        for batch in 0..ir.batch_count {
            let base = batch * ir.tensor_len;
            for linear in 0..ir.tensor_len {
                if ir.contains_zero_linear_index(linear) {
                    tensor[base + linear] = ComplexDoubleDouble::default();
                }
            }
        }
    }
    for axis in &ir.axes {
        let outer_count = ir.tensor_len / (axis.axis_len * axis.inner_stride);
        let transform_batch_count = ir.batch_count * axis.line_count;
        let mut packed = Vec::with_capacity(transform_batch_count * axis.axis_len);
        for batch in 0..ir.batch_count {
            let batch_base = batch * ir.tensor_len;
            for outer in 0..outer_count {
                let outer_base = batch_base + outer * axis.axis_len * axis.inner_stride;
                for inner in 0..axis.inner_stride {
                    for lane in 0..axis.axis_len {
                        packed.push(tensor[outer_base + lane * axis.inner_stride + inner]);
                    }
                }
            }
        }
        let transformed = execute_double_double_one_dim_ir(&axis.transform, &packed)?;
        let mut cursor = 0usize;
        for batch in 0..ir.batch_count {
            let batch_base = batch * ir.tensor_len;
            for outer in 0..outer_count {
                let outer_base = batch_base + outer * axis.axis_len * axis.inner_stride;
                for inner in 0..axis.inner_stride {
                    for lane in 0..axis.axis_len {
                        tensor[outer_base + lane * axis.inner_stride + inner] = transformed[cursor];
                        cursor += 1;
                    }
                }
            }
        }
    }
    if ir.zero_padding_is_output_boundary() {
        for batch in 0..ir.batch_count {
            let base = batch * ir.tensor_len;
            for linear in 0..ir.tensor_len {
                if ir.contains_zero_linear_index(linear) {
                    tensor[base + linear] = ComplexDoubleDouble::default();
                }
            }
        }
    }
    Ok(tensor)
}

// Preserve the public one-dimensional IR shape and avoid per-plan heap allocation
// solely to equalize enum variant sizes.
#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone, PartialEq)]
pub enum DoubleDoubleOneDimIr {
    Stockham(DoubleDoubleStockhamIr),
    DirectRader(DoubleDoubleDirectRaderIr),
    FftRader(DoubleDoubleFftRaderIr),
    Bluestein(DoubleDoubleBluesteinIr),
    Recursive(DoubleDoubleRecursiveFftIr),
}

impl DoubleDoubleOneDimIr {
    pub fn build(plan: &FftPlan, direction: Direction) -> Result<Self> {
        let axis = plan.axes.first().ok_or(VkFftError::InvalidKernelIr(
            "missing double-double one-dimensional axis",
        ))?;
        match &axis.algorithm {
            AxisAlgorithm::Stockham { .. } if axis.effective_fft_len > 4096 => Ok(Self::Recursive(
                DoubleDoubleRecursiveFftIr::build(plan, direction)?,
            )),
            AxisAlgorithm::Stockham { .. } => Ok(Self::Stockham(DoubleDoubleStockhamIr::build(
                plan, direction,
            )?)),
            AxisAlgorithm::Rader { stockham, primes }
                if stockham.prime_factors.is_empty()
                    && stockham.merged_radices.is_empty()
                    && primes.len() == 1
                    && primes[0].multiplicity == 1
                    && matches!(primes[0].mode, RaderMode::DirectMultiplication) =>
            {
                Ok(Self::DirectRader(DoubleDoubleDirectRaderIr::build(
                    plan, direction,
                )?))
            }
            AxisAlgorithm::Rader { stockham, primes }
                if stockham.prime_factors.is_empty()
                    && stockham.merged_radices.is_empty()
                    && primes.len() == 1
                    && primes[0].multiplicity == 1
                    && matches!(primes[0].mode, RaderMode::FftConvolution { .. }) =>
            {
                Ok(Self::FftRader(DoubleDoubleFftRaderIr::build(
                    plan, direction,
                )?))
            }
            AxisAlgorithm::Rader { .. } => Ok(Self::Recursive(DoubleDoubleRecursiveFftIr::build(
                plan, direction,
            )?)),
            AxisAlgorithm::Bluestein { .. } => Ok(Self::Bluestein(DoubleDoubleBluesteinIr::build(
                plan, direction,
            )?)),
        }
    }

    /// GPU-aware 1D builder. Large smooth Stockham and composite Rader axes use the
    /// device-aware recursive builder so upload splitting can participate in tree
    /// construction. Bluestein keeps its caller wrapper but builds large smooth
    /// convolution children device-aware; standalone Rader leaves retain portable
    /// construction and receive their physical blocks afterward.
    pub fn build_for_device(
        plan: &FftPlan,
        direction: Direction,
        device: DeviceProfile,
    ) -> Result<Self> {
        let axis = plan.axes.first().ok_or(VkFftError::InvalidKernelIr(
            "missing double-double one-dimensional axis",
        ))?;
        let composite_rader = match &axis.algorithm {
            AxisAlgorithm::Rader { stockham, primes } => {
                !(stockham.prime_factors.is_empty()
                    && stockham.merged_radices.is_empty()
                    && primes.len() == 1
                    && primes[0].multiplicity == 1)
            }
            _ => false,
        };
        let stockham_multi_upload = if matches!(axis.algorithm, AxisAlgorithm::Stockham { .. }) {
            double_double_stockham_requires_multi_upload(plan, axis.effective_fft_len, device)?
        } else {
            false
        };
        if matches!(axis.algorithm, AxisAlgorithm::Stockham { .. }) && stockham_multi_upload
            || composite_rader
        {
            return Ok(Self::Recursive(
                DoubleDoubleRecursiveFftIr::build_for_device(plan, direction, device)?,
            ));
        }
        if matches!(axis.algorithm, AxisAlgorithm::Stockham { .. }) && axis.effective_fft_len > 4096
        {
            let stockham = DoubleDoubleStockhamIr::build(plan, direction)?
                .with_axis0_device_stockham_block(device)?;
            if stockham.axis_batch_block.is_some() {
                return Ok(Self::Stockham(stockham));
            }
            return Ok(Self::Recursive(
                DoubleDoubleRecursiveFftIr::build_for_device(plan, direction, device)?,
            ));
        }
        if matches!(axis.algorithm, AxisAlgorithm::Bluestein { .. }) {
            return Ok(Self::Bluestein(DoubleDoubleBluesteinIr::build_for_device(
                plan, direction, device,
            )?));
        }
        if matches!(
            axis.algorithm,
            AxisAlgorithm::Rader { ref stockham, ref primes }
                if stockham.prime_factors.is_empty()
                    && stockham.merged_radices.is_empty()
                    && primes.len() == 1
                    && primes[0].multiplicity == 1
                    && matches!(primes[0].mode, RaderMode::FftConvolution { .. })
        ) {
            return Ok(Self::FftRader(
                DoubleDoubleFftRaderIr::build_for_device(plan, direction, device)?
                    .with_axis0_grouped_fft_rader_blocks(device)?,
            ));
        }
        Self::build(plan, direction)?.with_axis0_grouped_physical_block(device)
    }

    pub fn with_axis0_grouped_physical_block(self, device: DeviceProfile) -> Result<Self> {
        match self {
            Self::Stockham(ir) => Ok(Self::Stockham(
                ir.with_axis0_grouped_stockham_block(device)?,
            )),
            Self::DirectRader(ir) => Ok(Self::DirectRader(
                ir.with_axis0_grouped_direct_rader_block(device)?,
            )),
            Self::FftRader(ir) => Ok(Self::FftRader(
                ir.with_axis0_grouped_fft_rader_blocks(device)?,
            )),
            Self::Bluestein(ir) => Ok(Self::Bluestein(
                ir.with_axis0_grouped_bluestein_blocks(device)?,
            )),
            Self::Recursive(ir) => Ok(Self::Recursive(ir.with_device_physical_blocks(device)?)),
        }
    }

    pub fn with_axis0_grouped_stockham_block(self, device: DeviceProfile) -> Result<Self> {
        match self {
            Self::Stockham(ir) => Ok(Self::Stockham(
                ir.with_axis0_grouped_stockham_block(device)?,
            )),
            other => Ok(other),
        }
    }

    pub fn with_other_axis_grouped_physical_block(
        self,
        fastest_axis_len: usize,
        grouped_batch_override: Option<usize>,
        axis1_grouped_batch_override: Option<usize>,
        device: DeviceProfile,
    ) -> Result<Self> {
        match self {
            Self::Stockham(ir) => Ok(Self::Stockham(ir.with_other_axis_grouped_stockham_block(
                fastest_axis_len,
                grouped_batch_override,
                axis1_grouped_batch_override,
                device,
            )?)),
            Self::DirectRader(ir) => Ok(Self::DirectRader(
                ir.with_other_axis_grouped_direct_rader_block(
                    fastest_axis_len,
                    grouped_batch_override,
                    axis1_grouped_batch_override,
                    device,
                )?,
            )),
            Self::FftRader(ir) => Ok(Self::FftRader(ir.with_other_axis_grouped_fft_rader_block(
                fastest_axis_len,
                grouped_batch_override,
                axis1_grouped_batch_override,
                device,
            )?)),
            Self::Bluestein(ir) => Ok(Self::Bluestein(ir.with_other_axis_blocks(
                fastest_axis_len,
                grouped_batch_override,
                axis1_grouped_batch_override,
                device,
            )?)),
            Self::Recursive(ir) => Ok(Self::Recursive(ir.with_other_axis_physical_blocks(
                fastest_axis_len,
                grouped_batch_override,
                axis1_grouped_batch_override,
                device,
            )?)),
        }
    }

    pub fn stockham_axis_batch_block(&self) -> Option<StockhamAxisBlockSchedule> {
        match self {
            Self::Stockham(stockham) => stockham.axis_batch_block,
            _ => None,
        }
    }

    pub fn validate(&self) -> Result<()> {
        match self {
            Self::Stockham(ir) => ir.validate(),
            Self::DirectRader(ir) => ir.validate(),
            Self::FftRader(ir) => ir.validate(),
            Self::Bluestein(ir) => ir.validate(),
            Self::Recursive(ir) => ir.validate(),
        }
    }

    pub fn sequence_len(&self) -> usize {
        match self {
            Self::Stockham(ir) => ir.sequence_len,
            Self::DirectRader(ir) => ir.prime,
            Self::FftRader(ir) => ir.prime,
            Self::Bluestein(ir) => ir.logical_len,
            Self::Recursive(ir) => ir.logical_len,
        }
    }

    pub fn direction(&self) -> Direction {
        match self {
            Self::Stockham(ir) => ir.direction,
            Self::DirectRader(ir) => ir.direction,
            Self::FftRader(ir) => ir.direction,
            Self::Bluestein(ir) => ir.direction,
            Self::Recursive(ir) => ir.direction,
        }
    }

    pub fn batch_count(&self) -> usize {
        match self {
            Self::Stockham(ir) => ir.batch_count,
            Self::DirectRader(ir) => ir.batch_count,
            Self::FftRader(ir) => ir.batch_count,
            Self::Bluestein(ir) => ir.batch_count,
            Self::Recursive(ir) => ir.batch_count,
        }
    }

    pub fn grouped_batch(&self) -> usize {
        match self {
            Self::Stockham(ir) => ir.grouped_batch,
            Self::DirectRader(ir) => ir.grouped_batch,
            Self::FftRader(ir) => ir.grouped_batch,
            Self::Bluestein(ir) => ir.grouped_batch,
            Self::Recursive(ir) => ir.grouped_batch,
        }
    }

    pub fn external_storage(&self) -> PrecisionStorage {
        match self {
            Self::Stockham(ir) => ir.external_storage,
            Self::DirectRader(ir) => ir.external_storage,
            Self::FftRader(ir) => ir.external_storage,
            Self::Bluestein(ir) => ir.external_storage,
            Self::Recursive(ir) => ir.external_storage,
        }
    }
}

pub fn execute_double_double_one_dim_ir(
    ir: &DoubleDoubleOneDimIr,
    input: &[ComplexDoubleDouble],
) -> Result<Vec<ComplexDoubleDouble>> {
    match ir {
        DoubleDoubleOneDimIr::Stockham(ir) => execute_double_double_stockham_ir(ir, input),
        DoubleDoubleOneDimIr::DirectRader(ir) => execute_double_double_direct_rader_ir(ir, input),
        DoubleDoubleOneDimIr::FftRader(ir) => execute_double_double_fft_rader_ir(ir, input),
        DoubleDoubleOneDimIr::Bluestein(ir) => execute_double_double_bluestein_ir(ir, input),
        DoubleDoubleOneDimIr::Recursive(ir) => execute_double_double_recursive_ir(ir, input),
    }
}

pub fn execute_double_double_one_dim_ir_f64_storage(
    ir: &DoubleDoubleOneDimIr,
    input: &[Complex64],
) -> Result<Vec<Complex64>> {
    match ir {
        DoubleDoubleOneDimIr::Stockham(ir) => {
            execute_double_double_stockham_ir_f64_storage(ir, input)
        }
        DoubleDoubleOneDimIr::DirectRader(ir) => {
            execute_double_double_direct_rader_ir_f64_storage(ir, input)
        }
        DoubleDoubleOneDimIr::FftRader(ir) => {
            execute_double_double_fft_rader_ir_f64_storage(ir, input)
        }
        DoubleDoubleOneDimIr::Bluestein(ir) => {
            execute_double_double_bluestein_ir_f64_storage(ir, input)
        }
        DoubleDoubleOneDimIr::Recursive(ir) => {
            execute_double_double_recursive_ir_f64_storage(ir, input)
        }
    }
}

pub fn execute_double_double_bluestein_ir(
    ir: &DoubleDoubleBluesteinIr,
    input: &[ComplexDoubleDouble],
) -> Result<Vec<ComplexDoubleDouble>> {
    if ir.external_storage != PrecisionStorage::DoubleDouble {
        return Err(VkFftError::UnsupportedPrecision {
            backend: "double-double Bluestein DD-storage executor",
            precision: "IR uses F64 external storage",
        });
    }
    execute_bluestein_compute(ir, input)
}

pub fn execute_double_double_bluestein_ir_f64_storage(
    ir: &DoubleDoubleBluesteinIr,
    input: &[Complex64],
) -> Result<Vec<Complex64>> {
    if ir.external_storage != PrecisionStorage::F64 {
        return Err(VkFftError::UnsupportedPrecision {
            backend: "double-double Bluestein F64-storage executor",
            precision: "IR uses double-double external storage",
        });
    }
    let promoted = input
        .iter()
        .copied()
        .map(ComplexDoubleDouble::from_complex64)
        .collect::<Vec<_>>();
    execute_bluestein_compute(ir, &promoted).map(|values| {
        values
            .into_iter()
            .map(ComplexDoubleDouble::to_complex64)
            .collect()
    })
}

fn execute_bluestein_compute(
    ir: &DoubleDoubleBluesteinIr,
    input: &[ComplexDoubleDouble],
) -> Result<Vec<ComplexDoubleDouble>> {
    ir.validate()?;
    let expected =
        ir.logical_len
            .checked_mul(ir.batch_count)
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "double-double Bluestein input element count",
            })?;
    if input.len() != expected {
        return Err(VkFftError::InputLengthMismatch {
            expected,
            actual: input.len(),
        });
    }
    let padded_count =
        ir.convolution_len
            .checked_mul(ir.batch_count)
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "double-double Bluestein padded element count",
            })?;
    let mut padded = vec![ComplexDoubleDouble::default(); padded_count];
    for batch in 0..ir.batch_count {
        let input_base = batch * ir.logical_len;
        let padded_base = batch * ir.convolution_len;
        for index in 0..ir.logical_len {
            let value = if ir.zero_padding_is_input_boundary() && ir.contains_zero_index(index) {
                ComplexDoubleDouble::default()
            } else {
                input[input_base + index]
            };
            padded[padded_base + index] = value * ir.table.chirp[index];
        }
    }
    let mut spectrum = ir.forward_fft.execute(&padded)?;
    for batch in 0..ir.batch_count {
        let base = batch * ir.convolution_len;
        for index in 0..ir.convolution_len {
            spectrum[base + index] *= ir.kernel_spectrum[index];
        }
    }
    let convolution = ir.inverse_fft.execute(&spectrum)?;
    let scale = if ir.normalize {
        DoubleDouble::ONE / DoubleDouble::from_f64(ir.logical_len as f64)
    } else {
        DoubleDouble::ONE
    };
    let mut output = Vec::with_capacity(expected);
    for batch in 0..ir.batch_count {
        let base = batch * ir.convolution_len;
        for index in 0..ir.logical_len {
            let value = (convolution[base + index] * ir.table.chirp[index]).scale_dd(scale);
            output.push(
                if ir.zero_padding_is_output_boundary() && ir.contains_zero_index(index) {
                    ComplexDoubleDouble::default()
                } else {
                    value
                },
            );
        }
    }
    Ok(output)
}

pub fn execute_double_double_fft_rader_ir(
    ir: &DoubleDoubleFftRaderIr,
    input: &[ComplexDoubleDouble],
) -> Result<Vec<ComplexDoubleDouble>> {
    if ir.external_storage != PrecisionStorage::DoubleDouble {
        return Err(VkFftError::UnsupportedPrecision {
            backend: "double-double FFT Rader DD-storage executor",
            precision: "IR uses F64 external storage",
        });
    }
    execute_fft_rader_compute(ir, input)
}

pub fn execute_double_double_fft_rader_ir_f64_storage(
    ir: &DoubleDoubleFftRaderIr,
    input: &[Complex64],
) -> Result<Vec<Complex64>> {
    if ir.external_storage != PrecisionStorage::F64 {
        return Err(VkFftError::UnsupportedPrecision {
            backend: "double-double FFT Rader F64-storage executor",
            precision: "IR uses double-double external storage",
        });
    }
    let promoted = input
        .iter()
        .copied()
        .map(ComplexDoubleDouble::from_complex64)
        .collect::<Vec<_>>();
    execute_fft_rader_compute(ir, &promoted).map(|values| {
        values
            .into_iter()
            .map(ComplexDoubleDouble::to_complex64)
            .collect()
    })
}

fn execute_fft_rader_compute(
    ir: &DoubleDoubleFftRaderIr,
    input: &[ComplexDoubleDouble],
) -> Result<Vec<ComplexDoubleDouble>> {
    ir.validate()?;
    let expected = ir
        .prime
        .checked_mul(ir.batch_count)
        .ok_or(VkFftError::ArithmeticOverflow {
            operation: "double-double FFT Rader input element count",
        })?;
    if input.len() != expected {
        return Err(VkFftError::InputLengthMismatch {
            expected,
            actual: input.len(),
        });
    }
    let padded_input =
        double_double_zero_padded_input(ir.zero_pad_pass.as_ref(), ir.prime, ir.batch_count, input);
    let transform_input = padded_input.as_deref().unwrap_or(input);
    let convolution_count =
        ir.convolution_len
            .checked_mul(ir.batch_count)
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "double-double FFT Rader convolution element count",
            })?;
    let mut gathered = vec![ComplexDoubleDouble::default(); convolution_count];
    for batch in 0..ir.batch_count {
        let input_base = batch * ir.prime;
        let convolution_base = batch * ir.convolution_len;
        for slot in 0..ir.convolution_len {
            let exponent = (ir.convolution_len - slot) % ir.convolution_len;
            let input_index = ir.table.permutation[exponent];
            gathered[convolution_base + slot] = transform_input[input_base + input_index];
        }
    }
    let mut spectrum = ir.forward_fft.execute(&gathered)?;
    for batch in 0..ir.batch_count {
        let base = batch * ir.convolution_len;
        for index in 0..ir.convolution_len {
            spectrum[base + index] *= ir.kernel_spectrum[index];
        }
    }
    let convolution = ir.inverse_fft.execute(&spectrum)?;
    let scale = if ir.normalize {
        DoubleDouble::ONE / DoubleDouble::from_f64(ir.prime as f64)
    } else {
        DoubleDouble::ONE
    };
    let mut output = vec![ComplexDoubleDouble::default(); expected];
    for batch in 0..ir.batch_count {
        let input_base = batch * ir.prime;
        let convolution_base = batch * ir.convolution_len;
        let source = &transform_input[input_base..input_base + ir.prime];
        let destination = &mut output[input_base..input_base + ir.prime];
        let dc = source
            .iter()
            .copied()
            .fold(ComplexDoubleDouble::default(), |sum, value| sum + value);
        destination[0] = dc.scale_dd(scale);
        for exponent in 0..ir.convolution_len {
            let output_index = ir.table.permutation[exponent];
            destination[output_index] =
                (source[0] + convolution[convolution_base + exponent]).scale_dd(scale);
        }
    }
    apply_double_double_output_zero_padding(
        ir.zero_pad_pass.as_ref(),
        ir.prime,
        ir.batch_count,
        &mut output,
    );
    Ok(output)
}

pub fn execute_double_double_direct_rader_ir(
    ir: &DoubleDoubleDirectRaderIr,
    input: &[ComplexDoubleDouble],
) -> Result<Vec<ComplexDoubleDouble>> {
    if ir.external_storage != PrecisionStorage::DoubleDouble {
        return Err(VkFftError::UnsupportedPrecision {
            backend: "double-double direct Rader DD-storage executor",
            precision: "IR uses F64 external storage",
        });
    }
    execute_direct_rader_compute(ir, input)
}

pub fn execute_double_double_direct_rader_ir_f64_storage(
    ir: &DoubleDoubleDirectRaderIr,
    input: &[Complex64],
) -> Result<Vec<Complex64>> {
    if ir.external_storage != PrecisionStorage::F64 {
        return Err(VkFftError::UnsupportedPrecision {
            backend: "double-double direct Rader F64-storage executor",
            precision: "IR uses double-double external storage",
        });
    }
    let promoted = input
        .iter()
        .copied()
        .map(ComplexDoubleDouble::from_complex64)
        .collect::<Vec<_>>();
    execute_direct_rader_compute(ir, &promoted).map(|values| {
        values
            .into_iter()
            .map(ComplexDoubleDouble::to_complex64)
            .collect()
    })
}

fn execute_direct_rader_compute(
    ir: &DoubleDoubleDirectRaderIr,
    input: &[ComplexDoubleDouble],
) -> Result<Vec<ComplexDoubleDouble>> {
    ir.validate()?;
    let expected = ir
        .prime
        .checked_mul(ir.batch_count)
        .ok_or(VkFftError::ArithmeticOverflow {
            operation: "double-double direct Rader input element count",
        })?;
    if input.len() != expected {
        return Err(VkFftError::InputLengthMismatch {
            expected,
            actual: input.len(),
        });
    }
    let padded_input =
        double_double_zero_padded_input(ir.zero_pad_pass.as_ref(), ir.prime, ir.batch_count, input);
    let transform_input = padded_input.as_deref().unwrap_or(input);
    let count = ir.prime - 1;
    let scale = if ir.normalize {
        DoubleDouble::ONE / DoubleDouble::from_f64(ir.prime as f64)
    } else {
        DoubleDouble::ONE
    };
    let mut output = vec![ComplexDoubleDouble::default(); expected];
    for batch in 0..ir.batch_count {
        let base = batch * ir.prime;
        let source = &transform_input[base..base + ir.prime];
        let destination = &mut output[base..base + ir.prime];
        let dc = source
            .iter()
            .copied()
            .fold(ComplexDoubleDouble::default(), |sum, value| sum + value);
        destination[0] = dc.scale_dd(scale);
        for output_exponent in 0..count {
            let mut sum = source[0];
            for input_exponent in 0..count {
                let input_index = ir.table.permutation[input_exponent];
                let twiddle_index = (input_exponent + output_exponent) % count;
                sum += source[input_index] * ir.table.twiddles_by_generator_power[twiddle_index];
            }
            let output_index = ir.table.permutation[output_exponent];
            destination[output_index] = sum.scale_dd(scale);
        }
    }
    apply_double_double_output_zero_padding(
        ir.zero_pad_pass.as_ref(),
        ir.prime,
        ir.batch_count,
        &mut output,
    );
    Ok(output)
}

pub fn execute_double_double_stockham_ir(
    ir: &DoubleDoubleStockhamIr,
    input: &[ComplexDoubleDouble],
) -> Result<Vec<ComplexDoubleDouble>> {
    if ir.external_storage != PrecisionStorage::DoubleDouble {
        return Err(VkFftError::UnsupportedPrecision {
            backend: "double-double Stockham DD-storage executor",
            precision: "IR uses F64 external storage",
        });
    }
    execute_compute(ir, input)
}

pub fn execute_double_double_stockham_ir_f64_storage(
    ir: &DoubleDoubleStockhamIr,
    input: &[Complex64],
) -> Result<Vec<Complex64>> {
    if ir.external_storage != PrecisionStorage::F64 {
        return Err(VkFftError::UnsupportedPrecision {
            backend: "double-double Stockham F64-storage executor",
            precision: "IR uses double-double external storage",
        });
    }
    let promoted = input
        .iter()
        .copied()
        .map(ComplexDoubleDouble::from_complex64)
        .collect::<Vec<_>>();
    execute_compute(ir, &promoted).map(|values| {
        values
            .into_iter()
            .map(ComplexDoubleDouble::to_complex64)
            .collect()
    })
}

fn execute_compute(
    ir: &DoubleDoubleStockhamIr,
    input: &[ComplexDoubleDouble],
) -> Result<Vec<ComplexDoubleDouble>> {
    ir.validate()?;
    let expected =
        ir.sequence_len
            .checked_mul(ir.batch_count)
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "double-double Stockham input element count",
            })?;
    if input.len() != expected {
        return Err(VkFftError::InputLengthMismatch {
            expected,
            actual: input.len(),
        });
    }

    let padded_input = double_double_zero_padded_input(
        ir.zero_pad_pass.as_ref(),
        ir.sequence_len,
        ir.batch_count,
        input,
    );
    let transform_input = padded_input.as_deref().unwrap_or(input);
    let mut output = vec![ComplexDoubleDouble::default(); expected];
    for batch in 0..ir.batch_count {
        let base = batch * ir.sequence_len;
        let mut source = transform_input[base..base + ir.sequence_len].to_vec();
        for (stage, twiddles) in ir.stages.iter().zip(&ir.twiddles.stages) {
            let radix = stage.radix;
            let butterflies = stage.butterflies;
            let mut destination = vec![ComplexDoubleDouble::default(); ir.sequence_len];
            for butterfly in 0..butterflies {
                let stage_invocation = butterfly % stage.stage_size;
                let mut values = Vec::with_capacity(radix);
                for lane in 0..radix {
                    let input_index = butterfly + lane * butterflies;
                    let twiddle =
                        twiddles
                            .get(stage_invocation, lane)
                            .ok_or(VkFftError::InvalidKernelIr(
                                "double-double Stockham twiddle index is out of range",
                            ))?;
                    values.push(source[input_index] * twiddle);
                }
                for output_lane in 0..radix {
                    let mut sum = ComplexDoubleDouble::default();
                    for (input_lane, value) in values.iter().copied().enumerate() {
                        let root_index = output_lane.checked_mul(input_lane).ok_or(
                            VkFftError::ArithmeticOverflow {
                                operation: "double-double radix DFT root index",
                            },
                        )?;
                        sum += value * unit_root(root_index, radix, ir.direction)?;
                    }
                    let output_index = stage_invocation
                        + (butterfly - stage_invocation) * radix
                        + output_lane * stage.stage_size;
                    destination[output_index] = sum;
                }
            }
            source = destination;
        }
        if ir.normalize {
            let scale = DoubleDouble::ONE / DoubleDouble::from_f64(ir.sequence_len as f64);
            for value in &mut source {
                *value = value.scale_dd(scale);
            }
        }
        output[base..base + ir.sequence_len].copy_from_slice(&source);
    }
    apply_double_double_output_zero_padding(
        ir.zero_pad_pass.as_ref(),
        ir.sequence_len,
        ir.batch_count,
        &mut output,
    );
    Ok(output)
}

fn build_twiddle_table(
    sequence_len: usize,
    direction: Direction,
    stages: &[StockhamStage],
) -> Result<DoubleDoubleStockhamTwiddleTable> {
    let mut twiddle_stages = Vec::with_capacity(stages.len());
    for stage in stages {
        let denominator =
            stage
                .stage_size
                .checked_mul(stage.radix)
                .ok_or(VkFftError::ArithmeticOverflow {
                    operation: "double-double Stockham twiddle denominator",
                })?;
        let mut values = Vec::with_capacity(denominator);
        for stage_invocation in 0..stage.stage_size {
            for lane in 0..stage.radix {
                let root_index =
                    stage_invocation
                        .checked_mul(lane)
                        .ok_or(VkFftError::ArithmeticOverflow {
                            operation: "double-double Stockham twiddle root index",
                        })?;
                values.push(unit_root(root_index, denominator, direction)?);
            }
        }
        twiddle_stages.push(DoubleDoubleStockhamTwiddleStage {
            index: stage.index,
            radix: stage.radix,
            stage_size: stage.stage_size,
            values,
        });
    }
    Ok(DoubleDoubleStockhamTwiddleTable {
        sequence_len,
        direction,
        stages: twiddle_stages,
    })
}

fn stockham_schedule(radix: &RadixPlan, sequence_len: usize) -> Result<Vec<usize>> {
    if sequence_len == 1 {
        return Ok(Vec::new());
    }
    let merged_product = checked_product(&radix.merged_radices)?;
    if !radix.merged_radices.is_empty() && merged_product == sequence_len {
        // A literal DD radix-32 butterfly expands to a very large fp64 shader and
        // causes pathological driver pipeline compilation on current Vulkan/CUDA/OpenCL
        // runtimes. Keep the planner-visible VkFFT merged-radix choice intact, but
        // factor only the executable DD IR stage into mathematically equivalent 16x2
        // Stockham stages. Smaller merged radices remain unchanged.
        let mut executable = Vec::with_capacity(radix.merged_radices.len() + 1);
        for &stage_radix in &radix.merged_radices {
            if stage_radix == 32 {
                executable.extend([16, 2]);
            } else {
                executable.push(stage_radix);
            }
        }
        return Ok(executable);
    }
    let prime_product = checked_product(&radix.prime_factors)?;
    if prime_product == sequence_len {
        return Ok(radix.prime_factors.clone());
    }
    Err(VkFftError::InvalidKernelIr(
        "planner did not provide a complete double-double Stockham radix schedule",
    ))
}

fn checked_product(values: &[usize]) -> Result<usize> {
    values.iter().try_fold(1usize, |product, value| {
        product
            .checked_mul(*value)
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "double-double Stockham radix product",
            })
    })
}

fn double_double_external_storage(precision: Precision) -> Result<PrecisionStorage> {
    match precision {
        Precision::DoubleDouble => Ok(PrecisionStorage::DoubleDouble),
        Precision::DoubleDoubleF64Storage => Ok(PrecisionStorage::F64),
        other => Err(VkFftError::UnsupportedPrecision {
            backend: "double-double one-dimensional IR",
            precision: precision_name(other),
        }),
    }
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

#[derive(Debug, Clone, PartialEq)]
pub enum DoubleDoubleR2rAlgorithm {
    Direct {
        coefficients: Vec<DoubleDouble>,
    },
    FftReduction {
        fft_len: usize,
        fft: Box<DoubleDoubleOneDimIr>,
        phases: Vec<ComplexDoubleDouble>,
    },
    EvenTypeIvHalfSize {
        fft_len: usize,
        fft: Box<DoubleDoubleOneDimIr>,
        pack_phases: Vec<ComplexDoubleDouble>,
        extract_phases: Vec<ComplexDoubleDouble>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct DoubleDoubleR2rIr {
    pub transform: R2rTransform,
    pub effective_transform: R2rTransform,
    pub direction: Direction,
    pub length: usize,
    pub batch_count: usize,
    pub grouped_batch: usize,
    /// Upstream spatial zero interval, fused into the external R2R boundary.
    pub zero_padding: Option<ZeroPaddingRange>,
    pub normalize: bool,
    pub normalization_scale: DoubleDouble,
    pub external_storage: PrecisionStorage,
    pub algorithm: DoubleDoubleR2rAlgorithm,
}

impl DoubleDoubleR2rIr {
    pub fn build(plan: &FftPlan, direction: Direction) -> Result<Self> {
        Self::build_impl(plan, direction, None, C2cDeviceAxisClass::Contiguous)
    }

    pub fn build_for_device(
        plan: &FftPlan,
        direction: Direction,
        device: DeviceProfile,
    ) -> Result<Self> {
        Self::build_impl(
            plan,
            direction,
            Some(device),
            C2cDeviceAxisClass::Contiguous,
        )
    }

    pub(crate) fn build_for_device_with_axis_class(
        plan: &FftPlan,
        direction: Direction,
        device: DeviceProfile,
        axis_class: C2cDeviceAxisClass,
    ) -> Result<Self> {
        Self::build_impl(plan, direction, Some(device), axis_class)
    }

    fn build_impl(
        plan: &FftPlan,
        direction: Direction,
        device: Option<DeviceProfile>,
        axis_class: C2cDeviceAxisClass,
    ) -> Result<Self> {
        if plan.config.dimensions.len() != 1 {
            return Err(VkFftError::UnsupportedKernelPath(
                "initial double-double DCT/DST IR supports one dimension only",
            ));
        }
        let transform = match plan.config.transform {
            TransformKind::Dct(kind) => R2rTransform::Dct(kind),
            TransformKind::Dst(kind) => R2rTransform::Dst(kind),
            _ => {
                return Err(VkFftError::UnsupportedKernelPath(
                    "double-double R2R IR requires a DCT or DST transform",
                ));
            }
        };
        let external_storage = match plan.config.precision {
            Precision::DoubleDouble => PrecisionStorage::DoubleDouble,
            Precision::DoubleDoubleF64Storage => PrecisionStorage::F64,
            other => {
                return Err(VkFftError::UnsupportedPrecision {
                    backend: "double-double R2R IR",
                    precision: match other {
                        Precision::F16StorageF32Compute => "f16-storage/f32-compute",
                        Precision::F32 => "f32",
                        Precision::F64 => "f64",
                        Precision::F64ComputeF32Storage => "f64-compute/f32-storage",
                        Precision::DoubleDouble | Precision::DoubleDoubleF64Storage => {
                            unreachable!()
                        }
                    },
                });
            }
        };
        let length = plan.config.dimensions[0];
        let zero_padding = plan.config.zero_padding_for_axis(0);
        let grouped_batch_override = plan.config.grouped_batch_for_axis(0);
        let grouped_batch = grouped_batch_override.unwrap_or(1);
        if matches!(transform, R2rTransform::Dct(DctType::I)) && length < 2 {
            return Err(VkFftError::InvalidTransformLength {
                axis: 0,
                transform: "DCT-I",
                length,
            });
        }
        let effective_transform = if direction == Direction::Inverse {
            inverse_partner(transform)
        } else {
            transform
        };
        let normalize = direction == Direction::Inverse && plan.config.normalize_inverse;
        let normalization_scale = if normalize {
            let denominator = double_double_r2r_inverse_denominator(transform, length)?;
            DoubleDouble::ONE / DoubleDouble::from_f64(denominator as f64)
        } else {
            DoubleDouble::ONE
        };
        let algorithm = match effective_transform {
            R2rTransform::Dct(DctType::IV) | R2rTransform::Dst(DstType::IV)
                if length.is_multiple_of(2) =>
            {
                let fft_len = length / 2;
                let mut internal_config = FftConfig::new(vec![fft_len])
                    .with_batch_count(plan.config.batch_count)
                    .with_precision(Precision::DoubleDouble)
                    .with_inverse_normalization(false)
                    .with_tuning(plan.config.tuning)
                    .with_bandwidth_boost(plan.config.bandwidth_boost);
                if let Some(grouped_batch) = grouped_batch_override {
                    internal_config = internal_config.with_grouped_batch(0, grouped_batch)?;
                }
                let internal_plan = if let Some(device) = device {
                    FftPlan::build_c2c_child_for_device(internal_config, device, axis_class)?
                } else {
                    FftPlan::build(internal_config)?
                };
                let fft = if let Some(device) = device {
                    DoubleDoubleOneDimIr::build_for_device(
                        &internal_plan,
                        Direction::Inverse,
                        device,
                    )?
                } else {
                    DoubleDoubleOneDimIr::build(&internal_plan, Direction::Inverse)?
                };
                let n = DoubleDouble::from_f64(length as f64);
                let pack_phases = (0..fft_len)
                    .map(|j| {
                        let angle = DoubleDouble::PI * DoubleDouble::from_f64(j as f64) / n;
                        let (sin, cos) = angle.sin_cos();
                        ComplexDoubleDouble::new(cos, sin)
                    })
                    .collect();
                let four_n = DoubleDouble::from_f64(4.0 * length as f64);
                let extract_phases = (0..length)
                    .map(|k| {
                        let angle =
                            DoubleDouble::PI * DoubleDouble::from_f64((2 * k + 1) as f64) / four_n;
                        let (sin, cos) = angle.sin_cos();
                        ComplexDoubleDouble::new(cos, sin)
                    })
                    .collect();
                DoubleDoubleR2rAlgorithm::EvenTypeIvHalfSize {
                    fft_len,
                    fft: Box::new(fft),
                    pack_phases,
                    extract_phases,
                }
            }

            R2rTransform::Dct(_) | R2rTransform::Dst(_) => {
                let fft_len = match effective_transform {
                    R2rTransform::Dct(DctType::I) => length
                        .checked_sub(1)
                        .and_then(|value| value.checked_mul(2))
                        .ok_or(VkFftError::ArithmeticOverflow {
                            operation: "double-double DCT-I FFT reduction length",
                        })?,
                    R2rTransform::Dst(DstType::I) => length
                        .checked_add(1)
                        .and_then(|value| value.checked_mul(2))
                        .ok_or(VkFftError::ArithmeticOverflow {
                            operation: "double-double DST-I FFT reduction length",
                        })?,
                    R2rTransform::Dct(DctType::II | DctType::III)
                    | R2rTransform::Dst(DstType::II | DstType::III) => length,
                    R2rTransform::Dct(DctType::IV) | R2rTransform::Dst(DstType::IV) => length
                        .checked_mul(2)
                        .ok_or(VkFftError::ArithmeticOverflow {
                            operation: "double-double odd DCT/DST-IV FFT reduction length",
                        })?,
                };
                let mut internal_config = FftConfig::new(vec![fft_len])
                    .with_batch_count(plan.config.batch_count)
                    .with_precision(Precision::DoubleDouble)
                    .with_inverse_normalization(false)
                    .with_tuning(plan.config.tuning)
                    .with_bandwidth_boost(plan.config.bandwidth_boost);
                if let Some(grouped_batch) = grouped_batch_override {
                    internal_config = internal_config.with_grouped_batch(0, grouped_batch)?;
                }
                let internal_plan = if let Some(device) = device {
                    FftPlan::build_c2c_child_for_device(internal_config, device, axis_class)?
                } else {
                    FftPlan::build(internal_config)?
                };
                let fft_direction = match effective_transform {
                    R2rTransform::Dct(DctType::I | DctType::II | DctType::IV)
                    | R2rTransform::Dst(DstType::I | DstType::II | DstType::IV) => {
                        Direction::Forward
                    }
                    R2rTransform::Dct(DctType::III) | R2rTransform::Dst(DstType::III) => {
                        Direction::Inverse
                    }
                };
                let fft = if let Some(device) = device {
                    DoubleDoubleOneDimIr::build_for_device(&internal_plan, fft_direction, device)?
                } else {
                    DoubleDoubleOneDimIr::build(&internal_plan, fft_direction)?
                };
                let phases = match effective_transform {
                    R2rTransform::Dct(DctType::II)
                    | R2rTransform::Dst(DstType::II)
                    | R2rTransform::Dct(DctType::III)
                    | R2rTransform::Dst(DstType::III) => {
                        let sign = match effective_transform {
                            R2rTransform::Dct(DctType::II) | R2rTransform::Dst(DstType::II) => -1.0,
                            R2rTransform::Dct(DctType::III) | R2rTransform::Dst(DstType::III) => {
                                1.0
                            }
                            _ => unreachable!(),
                        };
                        let denominator = DoubleDouble::from_f64(2.0 * length as f64);
                        (0..length)
                            .map(|k| {
                                let angle = DoubleDouble::PI * DoubleDouble::from_f64(k as f64)
                                    / denominator
                                    * DoubleDouble::from_f64(sign);
                                let (sin, cos) = angle.sin_cos();
                                ComplexDoubleDouble::new(cos, sin)
                            })
                            .collect()
                    }
                    R2rTransform::Dct(DctType::IV) | R2rTransform::Dst(DstType::IV) => {
                        let denominator = DoubleDouble::from_f64(2.0 * fft_len as f64);
                        (0..length)
                            .map(|k| {
                                let angle = -DoubleDouble::PI
                                    * DoubleDouble::from_f64((2 * k + 1) as f64)
                                    / denominator;
                                let (sin, cos) = angle.sin_cos();
                                ComplexDoubleDouble::new(cos, sin)
                            })
                            .collect()
                    }
                    R2rTransform::Dct(DctType::I) | R2rTransform::Dst(DstType::I) => Vec::new(),
                };
                DoubleDoubleR2rAlgorithm::FftReduction {
                    fft_len,
                    fft: Box::new(fft),
                    phases,
                }
            }
        };
        let ir = Self {
            transform,
            effective_transform,
            direction,
            length,
            batch_count: plan.config.batch_count,
            grouped_batch,
            zero_padding,
            normalize,
            normalization_scale,
            external_storage,
            algorithm,
        };
        ir.validate()?;
        Ok(ir)
    }

    pub fn with_grouped_stockham_fft_block(
        mut self,
        fastest_axis: bool,
        fastest_axis_len: usize,
        grouped_batch_override: Option<usize>,
        axis1_grouped_batch_override: Option<usize>,
        device: DeviceProfile,
    ) -> Result<Self> {
        let fft = match &mut self.algorithm {
            DoubleDoubleR2rAlgorithm::FftReduction { fft, .. }
            | DoubleDoubleR2rAlgorithm::EvenTypeIvHalfSize { fft, .. } => Some(fft),
            DoubleDoubleR2rAlgorithm::Direct { .. } => None,
        };
        if let Some(fft) = fft {
            let materialized = if fastest_axis {
                (**fft).clone().with_axis0_grouped_stockham_block(device)?
            } else {
                (**fft).clone().with_other_axis_grouped_physical_block(
                    fastest_axis_len,
                    grouped_batch_override,
                    axis1_grouped_batch_override,
                    device,
                )?
            };
            **fft = materialized;
        }
        self.validate()?;
        Ok(self)
    }

    pub fn validate(&self) -> Result<()> {
        if self.length == 0 || self.batch_count == 0 || self.grouped_batch == 0 {
            return Err(VkFftError::InvalidKernelIr(
                "double-double R2R requires non-zero length and batch count",
            ));
        }
        if let Some(range) = self.zero_padding
            && (range.left > range.right || range.right > self.length)
        {
            return Err(VkFftError::InvalidKernelIr(
                "double-double R2R zero-padding interval is out of range",
            ));
        }
        if matches!(self.transform, R2rTransform::Dct(DctType::I)) && self.length < 2 {
            return Err(VkFftError::InvalidKernelIr(
                "double-double DCT-I requires length at least two",
            ));
        }
        let expected_effective = if self.direction == Direction::Inverse {
            inverse_partner(self.transform)
        } else {
            self.transform
        };
        if self.effective_transform != expected_effective {
            return Err(VkFftError::InvalidKernelIr(
                "double-double R2R effective transform does not match direction",
            ));
        }
        match &self.algorithm {
            DoubleDoubleR2rAlgorithm::Direct { coefficients } => {
                if matches!(
                    self.effective_transform,
                    R2rTransform::Dct(_) | R2rTransform::Dst(_)
                ) {
                    return Err(VkFftError::InvalidKernelIr(
                        "double-double DCT/DST I-IV must use an FFT reduction",
                    ));
                }
                if coefficients.len()
                    != self.length.checked_mul(self.length).ok_or(
                        VkFftError::ArithmeticOverflow {
                            operation: "double-double R2R coefficient validation count",
                        },
                    )?
                {
                    return Err(VkFftError::InvalidKernelIr(
                        "double-double direct R2R coefficient matrix has the wrong size",
                    ));
                }
            }
            DoubleDoubleR2rAlgorithm::FftReduction {
                fft_len,
                fft,
                phases,
            } => {
                if matches!(
                    self.effective_transform,
                    R2rTransform::Dct(DctType::IV) | R2rTransform::Dst(DstType::IV)
                ) && self.length.is_multiple_of(2)
                {
                    return Err(VkFftError::InvalidKernelIr(
                        "even double-double DCT/DST-IV must use the half-size FFT reduction",
                    ));
                }
                if !matches!(
                    self.effective_transform,
                    R2rTransform::Dct(_) | R2rTransform::Dst(_)
                ) {
                    return Err(VkFftError::InvalidKernelIr(
                        "double-double R2R FFT reduction is only valid for DCT/DST I-IV",
                    ));
                }
                let expected_fft_len = match self.effective_transform {
                    R2rTransform::Dct(DctType::I) => self
                        .length
                        .checked_sub(1)
                        .and_then(|value| value.checked_mul(2))
                        .ok_or(VkFftError::ArithmeticOverflow {
                            operation: "double-double DCT-I FFT validation length",
                        })?,
                    R2rTransform::Dst(DstType::I) => self
                        .length
                        .checked_add(1)
                        .and_then(|value| value.checked_mul(2))
                        .ok_or(VkFftError::ArithmeticOverflow {
                            operation: "double-double DST-I FFT validation length",
                        })?,
                    R2rTransform::Dct(DctType::II | DctType::III)
                    | R2rTransform::Dst(DstType::II | DstType::III) => self.length,
                    R2rTransform::Dct(DctType::IV) | R2rTransform::Dst(DstType::IV) => self
                        .length
                        .checked_mul(2)
                        .ok_or(VkFftError::ArithmeticOverflow {
                            operation: "double-double odd DCT/DST-IV FFT validation length",
                        })?,
                };
                let expected_phase_len = if matches!(
                    self.effective_transform,
                    R2rTransform::Dct(DctType::II | DctType::III | DctType::IV)
                        | R2rTransform::Dst(DstType::II | DstType::III | DstType::IV)
                ) {
                    self.length
                } else {
                    0
                };
                if *fft_len != expected_fft_len
                    || fft.sequence_len() != expected_fft_len
                    || fft.batch_count() != self.batch_count
                    || fft.grouped_batch() != self.grouped_batch
                    || phases.len() != expected_phase_len
                    || fft.direction()
                        != match self.effective_transform {
                            R2rTransform::Dct(DctType::I | DctType::II | DctType::IV)
                            | R2rTransform::Dst(DstType::I | DstType::II | DstType::IV) => {
                                Direction::Forward
                            }
                            R2rTransform::Dct(DctType::III) | R2rTransform::Dst(DstType::III) => {
                                Direction::Inverse
                            }
                        }
                {
                    return Err(VkFftError::InvalidKernelIr(
                        "double-double R2R FFT reduction metadata is inconsistent",
                    ));
                }
            }
            DoubleDoubleR2rAlgorithm::EvenTypeIvHalfSize {
                fft_len,
                fft,
                pack_phases,
                extract_phases,
            } => {
                if !matches!(
                    self.effective_transform,
                    R2rTransform::Dct(DctType::IV) | R2rTransform::Dst(DstType::IV)
                ) || !self.length.is_multiple_of(2)
                    || *fft_len != self.length / 2
                    || fft.sequence_len() != *fft_len
                    || fft.batch_count() != self.batch_count
                    || fft.grouped_batch() != self.grouped_batch
                    || fft.direction() != Direction::Inverse
                    || pack_phases.len() != *fft_len
                    || extract_phases.len() != self.length
                {
                    return Err(VkFftError::InvalidKernelIr(
                        "double-double even DCT/DST-IV half-size metadata is inconsistent",
                    ));
                }
            }
        }
        Ok(())
    }

    pub fn batch_group_count(&self) -> usize {
        self.batch_count.div_ceil(self.grouped_batch)
    }
}

pub fn execute_double_double_r2r_ir(
    ir: &DoubleDoubleR2rIr,
    input: &[DoubleDouble],
) -> Result<Vec<DoubleDouble>> {
    ir.validate()?;
    let expected = ir
        .length
        .checked_mul(ir.batch_count)
        .ok_or(VkFftError::ArithmeticOverflow {
            operation: "double-double R2R input element count",
        })?;
    if input.len() != expected {
        return Err(VkFftError::InputLengthMismatch {
            expected,
            actual: input.len(),
        });
    }
    let padded_input = if ir.direction == Direction::Forward {
        ir.zero_padding.map(|range| {
            let mut values = input.to_vec();
            for batch in 0..ir.batch_count {
                let base = batch * ir.length;
                values[base + range.left..base + range.right].fill(DoubleDouble::ZERO);
            }
            values
        })
    } else {
        None
    };
    let transform_input = padded_input.as_deref().unwrap_or(input);
    let mut output = match &ir.algorithm {
        DoubleDoubleR2rAlgorithm::Direct { coefficients } => {
            let mut output = vec![DoubleDouble::ZERO; expected];
            for batch in 0..ir.batch_count {
                let base = batch * ir.length;
                for k in 0..ir.length {
                    let mut sum = DoubleDouble::ZERO;
                    let coeff_base = k * ir.length;
                    for j in 0..ir.length {
                        sum += transform_input[base + j] * coefficients[coeff_base + j];
                    }
                    output[base + k] = sum * ir.normalization_scale;
                }
            }
            output
        }
        DoubleDoubleR2rAlgorithm::FftReduction {
            fft_len,
            fft,
            phases,
        } => execute_double_double_r2r_fft_reduction(ir, transform_input, *fft_len, fft, phases)?,
        DoubleDoubleR2rAlgorithm::EvenTypeIvHalfSize {
            fft_len,
            fft,
            pack_phases,
            extract_phases,
        } => execute_double_double_r2r_even_type_iv_half_size(
            ir,
            transform_input,
            *fft_len,
            fft,
            pack_phases,
            extract_phases,
        )?,
    };
    if ir.direction == Direction::Inverse
        && let Some(range) = ir.zero_padding
    {
        for batch in 0..ir.batch_count {
            let base = batch * ir.length;
            output[base + range.left..base + range.right].fill(DoubleDouble::ZERO);
        }
    }
    Ok(output)
}

pub fn execute_double_double_r2r_ir_f64_storage(
    ir: &DoubleDoubleR2rIr,
    input: &[f64],
) -> Result<Vec<f64>> {
    if ir.external_storage != PrecisionStorage::F64 {
        return Err(VkFftError::UnsupportedPrecision {
            backend: "double-double R2R CPU executor",
            precision: "F64-storage wrapper requires DoubleDoubleF64Storage",
        });
    }
    let promoted = input
        .iter()
        .copied()
        .map(DoubleDouble::from_f64)
        .collect::<Vec<_>>();
    Ok(execute_double_double_r2r_ir(ir, &promoted)?
        .into_iter()
        .map(DoubleDouble::to_f64)
        .collect())
}

fn execute_double_double_r2r_even_type_iv_half_size(
    ir: &DoubleDoubleR2rIr,
    input: &[DoubleDouble],
    fft_len: usize,
    fft: &DoubleDoubleOneDimIr,
    pack_phases: &[ComplexDoubleDouble],
    extract_phases: &[ComplexDoubleDouble],
) -> Result<Vec<DoubleDouble>> {
    let n = ir.length;
    let mut fft_input = vec![ComplexDoubleDouble::default(); fft_len * ir.batch_count];
    let is_dst = matches!(ir.effective_transform, R2rTransform::Dst(DstType::IV));
    for batch in 0..ir.batch_count {
        let source_base = batch * n;
        let fft_base = batch * fft_len;
        for j in 0..fft_len {
            let left = input[source_base + 2 * j];
            let right = input[source_base + n - 1 - 2 * j];
            let pair = ComplexDoubleDouble::new(left, if is_dst { right } else { -right });
            fft_input[fft_base + j] = pair * pack_phases[j];
        }
    }
    let transformed = execute_double_double_one_dim_ir(fft, &fft_input)?;
    let scale = ir.normalization_scale * DoubleDouble::from_f64(2.0);
    let mut output = vec![DoubleDouble::ZERO; n * ir.batch_count];
    for batch in 0..ir.batch_count {
        let fft_base = batch * fft_len;
        let output_base = batch * n;
        for k in 0..n {
            let bin = if k.is_multiple_of(2) {
                transformed[fft_base + k / 2]
            } else {
                transformed[fft_base + fft_len - 1 - k / 2].conj()
            };
            let rotated = bin * extract_phases[k];
            output[output_base + k] = match ir.effective_transform {
                R2rTransform::Dct(DctType::IV) => rotated.re * scale,
                R2rTransform::Dst(DstType::IV) => rotated.im * scale,
                _ => unreachable!(),
            };
        }
    }
    Ok(output)
}

fn execute_double_double_r2r_fft_reduction(
    ir: &DoubleDoubleR2rIr,
    input: &[DoubleDouble],
    fft_len: usize,
    fft: &DoubleDoubleOneDimIr,
    phases: &[ComplexDoubleDouble],
) -> Result<Vec<DoubleDouble>> {
    let n = ir.length;
    let mut fft_input = vec![ComplexDoubleDouble::default(); fft_len * ir.batch_count];
    match ir.effective_transform {
        R2rTransform::Dct(DctType::I) => {
            for batch in 0..ir.batch_count {
                let source_base = batch * n;
                let fft_base = batch * fft_len;
                fft_input[fft_base] =
                    ComplexDoubleDouble::new(input[source_base], DoubleDouble::ZERO);
                fft_input[fft_base + n - 1] =
                    ComplexDoubleDouble::new(input[source_base + n - 1], DoubleDouble::ZERO);
                for j in 1..n - 1 {
                    let value =
                        ComplexDoubleDouble::new(input[source_base + j], DoubleDouble::ZERO);
                    fft_input[fft_base + j] = value;
                    fft_input[fft_base + fft_len - j] = value;
                }
            }
        }
        R2rTransform::Dst(DstType::I) => {
            for batch in 0..ir.batch_count {
                let source_base = batch * n;
                let fft_base = batch * fft_len;
                for j in 0..n {
                    let value =
                        ComplexDoubleDouble::new(input[source_base + j], DoubleDouble::ZERO);
                    fft_input[fft_base + j + 1] = value;
                    fft_input[fft_base + fft_len - j - 1] = -value;
                }
            }
        }
        R2rTransform::Dct(DctType::II) | R2rTransform::Dst(DstType::II) => {
            let is_dst = matches!(ir.effective_transform, R2rTransform::Dst(DstType::II));
            let split = n.div_ceil(2);
            for batch in 0..ir.batch_count {
                let source_base = batch * n;
                let fft_base = batch * fft_len;
                for j in 0..n {
                    let source = if j < split { 2 * j } else { 2 * (n - j) - 1 };
                    let mut value = input[source_base + source];
                    if is_dst && !source.is_multiple_of(2) {
                        value = -value;
                    }
                    fft_input[fft_base + j] = ComplexDoubleDouble::new(value, DoubleDouble::ZERO);
                }
            }
        }
        R2rTransform::Dct(DctType::III) | R2rTransform::Dst(DstType::III) => {
            let is_dst = matches!(ir.effective_transform, R2rTransform::Dst(DstType::III));
            for batch in 0..ir.batch_count {
                let source_base = batch * n;
                let fft_base = batch * fft_len;
                for k in 0..n {
                    let (a, b) = if is_dst {
                        (
                            input[source_base + n - 1 - k],
                            if k == 0 {
                                DoubleDouble::ZERO
                            } else {
                                input[source_base + k - 1]
                            },
                        )
                    } else {
                        (
                            input[source_base + k],
                            if k == 0 {
                                DoubleDouble::ZERO
                            } else {
                                input[source_base + n - k]
                            },
                        )
                    };
                    fft_input[fft_base + k] = ComplexDoubleDouble::new(a, -b) * phases[k];
                }
            }
        }
        R2rTransform::Dct(DctType::IV) | R2rTransform::Dst(DstType::IV) => {
            let phase_zero_conj = phases[0].conj();
            for batch in 0..ir.batch_count {
                let source_base = batch * n;
                let fft_base = batch * fft_len;
                for j in 0..n {
                    let phase = phases[j] * phase_zero_conj;
                    fft_input[fft_base + j] = phase.scale_dd(input[source_base + j]);
                }
            }
        }
    }
    let transformed = execute_double_double_one_dim_ir(fft, &fft_input)?;
    let mut output = vec![DoubleDouble::ZERO; n * ir.batch_count];
    match ir.effective_transform {
        R2rTransform::Dct(DctType::I) => {
            for batch in 0..ir.batch_count {
                let fft_base = batch * fft_len;
                let output_base = batch * n;
                for k in 0..n {
                    output[output_base + k] = transformed[fft_base + k].re * ir.normalization_scale;
                }
            }
        }
        R2rTransform::Dst(DstType::I) => {
            for batch in 0..ir.batch_count {
                let fft_base = batch * fft_len;
                let output_base = batch * n;
                for k in 0..n {
                    output[output_base + k] =
                        -transformed[fft_base + k + 1].im * ir.normalization_scale;
                }
            }
        }
        R2rTransform::Dct(DctType::II) | R2rTransform::Dst(DstType::II) => {
            let is_dst = matches!(ir.effective_transform, R2rTransform::Dst(DstType::II));
            let scale = ir.normalization_scale * DoubleDouble::from_f64(2.0);
            for batch in 0..ir.batch_count {
                let fft_base = batch * fft_len;
                let output_base = batch * n;
                for k in 0..n {
                    let q = if is_dst { n - 1 - k } else { k };
                    let rotated = transformed[fft_base + q] * phases[q];
                    output[output_base + k] = rotated.re * scale;
                }
            }
        }
        R2rTransform::Dct(DctType::III) | R2rTransform::Dst(DstType::III) => {
            let is_dst = matches!(ir.effective_transform, R2rTransform::Dst(DstType::III));
            for batch in 0..ir.batch_count {
                let fft_base = batch * fft_len;
                let output_base = batch * n;
                for k in 0..n {
                    let source = if k.is_multiple_of(2) {
                        k / 2
                    } else {
                        n - 1 - k / 2
                    };
                    let mut value = transformed[fft_base + source].re;
                    if is_dst && !k.is_multiple_of(2) {
                        value = -value;
                    }
                    output[output_base + k] = value * ir.normalization_scale;
                }
            }
        }
        R2rTransform::Dct(DctType::IV) | R2rTransform::Dst(DstType::IV) => {
            let scale = ir.normalization_scale * DoubleDouble::from_f64(2.0);
            for batch in 0..ir.batch_count {
                let fft_base = batch * fft_len;
                let output_base = batch * n;
                for k in 0..n {
                    let value = transformed[fft_base + k] * phases[k];
                    output[output_base + k] = match ir.effective_transform {
                        R2rTransform::Dct(DctType::IV) => value.re * scale,
                        R2rTransform::Dst(DstType::IV) => -value.im * scale,
                        _ => unreachable!(),
                    };
                }
            }
        }
    }
    Ok(output)
}

fn double_double_r2r_inverse_denominator(transform: R2rTransform, length: usize) -> Result<usize> {
    match transform {
        R2rTransform::Dct(DctType::I) => length
            .checked_sub(1)
            .and_then(|value| value.checked_mul(2))
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "double-double DCT-I inverse normalization denominator",
            }),
        R2rTransform::Dst(DstType::I) => length
            .checked_add(1)
            .and_then(|value| value.checked_mul(2))
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "double-double DST-I inverse normalization denominator",
            }),
        R2rTransform::Dct(DctType::II | DctType::III | DctType::IV)
        | R2rTransform::Dst(DstType::II | DstType::III | DstType::IV) => {
            length.checked_mul(2).ok_or(VkFftError::ArithmeticOverflow {
                operation: "double-double DCT/DST inverse normalization denominator",
            })
        }
    }
}

#[cfg(test)]
fn double_double_r2r_coefficient(
    transform: R2rTransform,
    n: usize,
    j: usize,
    k: usize,
) -> Result<DoubleDouble> {
    let pi = DoubleDouble::PI;
    let two = DoubleDouble::from_f64(2.0);
    let jf = DoubleDouble::from_f64(j as f64);
    let kf = DoubleDouble::from_f64(k as f64);
    let nf = DoubleDouble::from_f64(n as f64);
    let half = DoubleDouble::from_f64(0.5);
    let coefficient = match transform {
        R2rTransform::Dct(DctType::I) => {
            if j == 0 {
                DoubleDouble::ONE
            } else if j == n - 1 {
                if k.is_multiple_of(2) {
                    DoubleDouble::ONE
                } else {
                    -DoubleDouble::ONE
                }
            } else {
                let denominator = DoubleDouble::from_f64((n - 1) as f64);
                let (_, cosine) = (pi * jf * kf / denominator).sin_cos();
                two * cosine
            }
        }
        R2rTransform::Dct(DctType::IV) => {
            let (_, cosine) = (pi * (jf + half) * (kf + half) / nf).sin_cos();
            two * cosine
        }
        R2rTransform::Dst(DstType::I) => {
            let denominator = DoubleDouble::from_f64((n + 1) as f64);
            let (sine, _) = (pi
                * DoubleDouble::from_f64((j + 1) as f64)
                * DoubleDouble::from_f64((k + 1) as f64)
                / denominator)
                .sin_cos();
            two * sine
        }
        R2rTransform::Dst(DstType::IV) => {
            let (sine, _) = (pi * (jf + half) * (kf + half) / nf).sin_cos();
            two * sine
        }
        R2rTransform::Dct(DctType::II | DctType::III)
        | R2rTransform::Dst(DstType::II | DstType::III) => {
            return Err(VkFftError::InvalidKernelIr(
                "double-double DCT/DST-II/III coefficients are owned by the FFT reduction",
            ));
        }
    };
    Ok(coefficient)
}

#[derive(Debug, Clone, PartialEq)]
pub struct DoubleDoubleNdR2rAxisIr {
    pub axis: usize,
    pub axis_len: usize,
    pub inner_stride: usize,
    pub line_count: usize,
    pub grouped_batch: usize,
    pub grouped_batch_override: Option<usize>,
    pub transform: DoubleDoubleR2rIr,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DoubleDoubleNdR2rIr {
    pub dimensions: Vec<usize>,
    pub tensor_len: usize,
    pub batch_count: usize,
    pub direction: Direction,
    pub external_storage: PrecisionStorage,
    pub input_external_layout: NdExternalTensorLayout,
    pub output_external_layout: NdExternalTensorLayout,
    pub input_formatted_copy: Option<NdFormattedCopyPassIr>,
    pub output_formatted_copy: Option<NdFormattedCopyPassIr>,
    pub zero_padding: Vec<Option<ZeroPaddingRange>>,
    pub omitted_axes: Vec<bool>,
    pub axes: Vec<DoubleDoubleNdR2rAxisIr>,
}

impl DoubleDoubleNdR2rIr {
    pub fn build(plan: &FftPlan, direction: Direction) -> Result<Self> {
        Self::build_impl(plan, direction, None)
    }

    pub fn build_for_device(
        plan: &FftPlan,
        direction: Direction,
        device: DeviceProfile,
    ) -> Result<Self> {
        Self::build_impl(plan, direction, Some(device))
    }

    fn build_impl(
        plan: &FftPlan,
        direction: Direction,
        device: Option<DeviceProfile>,
    ) -> Result<Self> {
        if !matches!(
            plan.config.transform,
            TransformKind::Dct(_) | TransformKind::Dst(_)
        ) || plan.config.dimensions.len() < 2
        {
            return Err(VkFftError::UnsupportedKernelPath(
                "double-double ND R2R IR requires a multidimensional DCT/DST plan",
            ));
        }
        let external_storage = double_double_external_storage(plan.config.precision)?;
        let tensor_len = plan
            .config
            .dimensions
            .iter()
            .try_fold(1usize, |product, value| {
                product
                    .checked_mul(*value)
                    .ok_or(VkFftError::ArithmeticOverflow {
                        operation: "double-double ND R2R tensor element count",
                    })
            })?;
        let input_external_layout = NdExternalTensorLayout {
            dimensions: plan.config.dimensions.clone(),
            axis_strides: plan.config.resolved_input_buffer_axis_strides()?,
            batch_stride: plan.config.resolved_input_buffer_batch_stride()?,
        };
        let output_external_layout = NdExternalTensorLayout {
            dimensions: plan.config.dimensions.clone(),
            axis_strides: plan.config.resolved_output_buffer_axis_strides()?,
            batch_stride: plan.config.resolved_output_buffer_batch_stride()?,
        };
        input_external_layout.validate(tensor_len)?;
        output_external_layout.validate(tensor_len)?;
        let external_scalar = double_double_storage_scalar(external_storage)?;
        let max_threads_per_block = device.map_or(256, |profile| profile.max_threads_per_block);
        let input_formatted_copy = (!input_external_layout.is_tightly_packed()?)
            .then(|| {
                NdFormattedCopyPassIr::new_with_max_threads(
                    "vkfft_dd_nd_r2r_gather_formatted_input".to_owned(),
                    ScalarType::DoubleDouble,
                    external_scalar,
                    plan.config.batch_count,
                    input_external_layout.clone(),
                    NdFormattedCopyOperation::GatherExternalToDense,
                    max_threads_per_block,
                )
            })
            .transpose()?;
        let output_formatted_copy = (!output_external_layout.is_tightly_packed()?)
            .then(|| {
                NdFormattedCopyPassIr::new_with_max_threads(
                    "vkfft_dd_nd_r2r_scatter_formatted_output".to_owned(),
                    ScalarType::DoubleDouble,
                    external_scalar,
                    plan.config.batch_count,
                    output_external_layout.clone(),
                    NdFormattedCopyOperation::ScatterDenseToExternal,
                    max_threads_per_block,
                )
            })
            .transpose()?;
        let omitted_axes = (0..plan.config.dimensions.len())
            .map(|axis| plan.config.axis_is_omitted(axis))
            .collect::<Vec<_>>();
        let mut axes = Vec::with_capacity(plan.config.dimensions.len());
        for axis in (0..plan.config.dimensions.len()).rev() {
            if omitted_axes[axis] {
                continue;
            }
            let axis_len = plan.config.dimensions[axis];
            let inner_stride =
                plan.config.dimensions[axis + 1..]
                    .iter()
                    .try_fold(1usize, |product, value| {
                        product
                            .checked_mul(*value)
                            .ok_or(VkFftError::ArithmeticOverflow {
                                operation: "double-double ND R2R inner stride",
                            })
                    })?;
            let line_count = tensor_len / axis_len;
            let grouped_batch_override = plan.config.grouped_batch_for_axis(axis);
            let grouped_batch = grouped_batch_override.unwrap_or(1);
            let transform_batch_count = plan.config.batch_count.checked_mul(line_count).ok_or(
                VkFftError::ArithmeticOverflow {
                    operation: "double-double ND R2R transform batch count",
                },
            )?;
            let axis_config = FftConfig::new(vec![axis_len])
                .with_batch_count(transform_batch_count)
                .with_precision(Precision::DoubleDouble)
                .with_transform(plan.config.transform)
                .with_inverse_normalization(plan.config.normalize_inverse)
                .with_tuning(plan.config.tuning)
                .with_bandwidth_boost(plan.config.bandwidth_boost)
                .with_grouped_batch(0, grouped_batch)?;
            let axis_plan = if let Some(device) = device {
                FftPlan::build_for_device(axis_config, device)?
            } else {
                FftPlan::build(axis_config)?
            };
            let transform = if let Some(device) = device {
                let axis_class = if inner_stride == 1 {
                    C2cDeviceAxisClass::Contiguous
                } else {
                    C2cDeviceAxisClass::Strided
                };
                DoubleDoubleR2rIr::build_for_device_with_axis_class(
                    &axis_plan, direction, device, axis_class,
                )?
            } else {
                DoubleDoubleR2rIr::build(&axis_plan, direction)?
            };
            axes.push(DoubleDoubleNdR2rAxisIr {
                axis,
                axis_len,
                inner_stride,
                line_count,
                grouped_batch,
                grouped_batch_override,
                transform,
            });
        }
        let ir = Self {
            dimensions: plan.config.dimensions.clone(),
            tensor_len,
            batch_count: plan.config.batch_count,
            direction,
            external_storage,
            input_external_layout,
            output_external_layout,
            input_formatted_copy,
            output_formatted_copy,
            zero_padding: plan.config.zero_padding.clone(),
            omitted_axes,
            axes,
        };
        ir.validate()?;
        Ok(ir)
    }

    pub fn with_grouped_stockham_axis_blocks(
        mut self,
        axis1_grouped_batch_override: Option<usize>,
        device: DeviceProfile,
    ) -> Result<Self> {
        let last_axis = self
            .dimensions
            .len()
            .checked_sub(1)
            .ok_or(VkFftError::InvalidKernelIr(
                "double-double ND R2R dimensions are empty",
            ))?;
        let fastest_axis_len = self.dimensions[last_axis];
        for axis in &mut self.axes {
            axis.transform = axis.transform.clone().with_grouped_stockham_fft_block(
                axis.axis == last_axis,
                fastest_axis_len,
                axis.grouped_batch_override,
                axis1_grouped_batch_override,
                device,
            )?;
        }
        self.validate()?;
        Ok(self)
    }

    pub fn validate(&self) -> Result<()> {
        if self.dimensions.len() < 2
            || self.zero_padding.len() != self.dimensions.len()
            || self.omitted_axes.len() != self.dimensions.len()
            || self.tensor_len == 0
            || self.batch_count == 0
        {
            return Err(VkFftError::InvalidKernelIr(
                "double-double ND R2R dimensions, padding, and batch count must be consistent",
            ));
        }
        let tensor_len = self.dimensions.iter().try_fold(1usize, |product, value| {
            product
                .checked_mul(*value)
                .ok_or(VkFftError::ArithmeticOverflow {
                    operation: "double-double ND R2R validation tensor size",
                })
        })?;
        if tensor_len != self.tensor_len
            || self.axes.len() != self.omitted_axes.iter().filter(|&&omit| !omit).count()
        {
            return Err(VkFftError::InvalidKernelIr(
                "double-double ND R2R tensor metadata is inconsistent",
            ));
        }
        if !matches!(
            self.external_storage,
            PrecisionStorage::DoubleDouble | PrecisionStorage::F64
        ) {
            return Err(VkFftError::InvalidKernelIr(
                "double-double ND R2R external storage must be DD or F64",
            ));
        }
        self.input_external_layout.validate(self.tensor_len)?;
        self.output_external_layout.validate(self.tensor_len)?;
        if self.input_external_layout.dimensions != self.dimensions
            || self.output_external_layout.dimensions != self.dimensions
        {
            return Err(VkFftError::InvalidKernelIr(
                "double-double ND R2R formatted tensor dimensions are inconsistent",
            ));
        }
        let input_needs_copy = !self.input_external_layout.is_tightly_packed()?;
        let output_needs_copy = !self.output_external_layout.is_tightly_packed()?;
        if self.input_formatted_copy.is_some() != input_needs_copy
            || self.output_formatted_copy.is_some() != output_needs_copy
        {
            return Err(VkFftError::InvalidKernelIr(
                "double-double ND R2R formatted copy ownership is inconsistent",
            ));
        }
        let external_scalar = double_double_storage_scalar(self.external_storage)?;
        if let Some(copy) = &self.input_formatted_copy {
            copy.validate()?;
            if copy.operation != NdFormattedCopyOperation::GatherExternalToDense
                || copy.external_layout != self.input_external_layout
                || copy.batch_count != self.batch_count
                || copy.scalar != ScalarType::DoubleDouble
                || copy.input_storage_scalar != external_scalar
                || copy.output_storage_scalar != ScalarType::DoubleDouble
            {
                return Err(VkFftError::InvalidKernelIr(
                    "double-double ND R2R formatted input copy metadata is inconsistent",
                ));
            }
        }
        if let Some(copy) = &self.output_formatted_copy {
            copy.validate()?;
            if copy.operation != NdFormattedCopyOperation::ScatterDenseToExternal
                || copy.external_layout != self.output_external_layout
                || copy.batch_count != self.batch_count
                || copy.scalar != ScalarType::DoubleDouble
                || copy.input_storage_scalar != ScalarType::DoubleDouble
                || copy.output_storage_scalar != external_scalar
            {
                return Err(VkFftError::InvalidKernelIr(
                    "double-double ND R2R formatted output copy metadata is inconsistent",
                ));
            }
        }
        for (axis, (length, range)) in self
            .dimensions
            .iter()
            .copied()
            .zip(self.zero_padding.iter().copied())
            .enumerate()
        {
            if let Some(range) = range
                && (range.left > range.right || range.right > length)
            {
                return Err(VkFftError::InvalidZeroPaddingRange {
                    axis,
                    left: range.left,
                    right: range.right,
                    length,
                });
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
                "double-double ND R2R omitted-axis execution order is inconsistent",
            ));
        }
        for axis in &self.axes {
            axis.transform.validate()?;
            if axis.axis >= self.dimensions.len() {
                return Err(VkFftError::InvalidKernelIr(
                    "double-double ND R2R axis index is out of range",
                ));
            }
            if self.omitted_axes[axis.axis] {
                return Err(VkFftError::InvalidKernelIr(
                    "double-double ND R2R materialized an omitted axis",
                ));
            }
            let expected_stride = self.dimensions[axis.axis + 1..].iter().product::<usize>();
            if axis.axis_len != self.dimensions[axis.axis]
                || axis.inner_stride != expected_stride
                || axis.line_count != self.tensor_len / axis.axis_len
                || axis.grouped_batch == 0
                || axis
                    .grouped_batch_override
                    .is_some_and(|requested| requested != axis.grouped_batch)
                || axis.transform.length != axis.axis_len
                || axis.transform.batch_count != self.batch_count * axis.line_count
                || axis.transform.grouped_batch != axis.grouped_batch
                || axis.transform.direction != self.direction
                || axis.transform.external_storage != PrecisionStorage::DoubleDouble
            {
                return Err(VkFftError::InvalidKernelIr(
                    "double-double ND R2R axis metadata is inconsistent",
                ));
            }
        }
        Ok(())
    }

    pub fn has_spatial_zero_padding(&self) -> bool {
        self.zero_padding.iter().any(Option::is_some)
    }

    pub fn contains_spatial_zero_linear_index(&self, mut index: usize) -> bool {
        for axis in (0..self.dimensions.len()).rev() {
            let length = self.dimensions[axis];
            let coordinate = index % length;
            index /= length;
            if self.zero_padding[axis].is_some_and(|range| range.contains(coordinate)) {
                return true;
            }
        }
        false
    }
    pub(crate) fn pack_formatted_input<T: Copy + Default>(&self, input: &[T]) -> Result<Vec<T>> {
        self.validate()?;
        let expected = self.tensor_len.checked_mul(self.batch_count).ok_or(
            VkFftError::ArithmeticOverflow {
                operation: "formatted double-double ND R2R logical input element count",
            },
        )?;
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

    fn logicalize_formatted_input<T: Copy + Default>(&self, input: &[T]) -> Result<Vec<T>> {
        let physical = self.pack_formatted_input(input)?;
        if self.input_formatted_copy.is_none() {
            return Ok(physical);
        }
        unpack_logical_tensor_batches(
            &physical,
            &self.input_external_layout.dimensions,
            &self.input_external_layout.axis_strides,
            self.input_external_layout.batch_stride,
            self.batch_count,
        )
    }

    fn round_trip_formatted_output<T: Copy + Default>(&self, output: &[T]) -> Result<Vec<T>> {
        if self.output_formatted_copy.is_none() {
            return Ok(output.to_vec());
        }
        let physical = pack_logical_tensor_batches(
            output,
            &self.output_external_layout.dimensions,
            &self.output_external_layout.axis_strides,
            self.output_external_layout.batch_stride,
        )?;
        self.unpack_formatted_output(&physical)
    }
}

pub fn execute_double_double_nd_r2r_ir(
    ir: &DoubleDoubleNdR2rIr,
    input: &[DoubleDouble],
) -> Result<Vec<DoubleDouble>> {
    if ir.external_storage != PrecisionStorage::DoubleDouble {
        return Err(VkFftError::UnsupportedPrecision {
            backend: "double-double ND R2R DD-storage executor",
            precision: "IR uses F64 external storage",
        });
    }
    let logical_input = ir.logicalize_formatted_input(input)?;
    let logical_output = execute_double_double_nd_r2r_compute(ir, &logical_input)?;
    ir.round_trip_formatted_output(&logical_output)
}

pub fn execute_double_double_nd_r2r_ir_f64_storage(
    ir: &DoubleDoubleNdR2rIr,
    input: &[f64],
) -> Result<Vec<f64>> {
    if ir.external_storage != PrecisionStorage::F64 {
        return Err(VkFftError::UnsupportedPrecision {
            backend: "double-double ND R2R F64-storage executor",
            precision: "IR uses double-double external storage",
        });
    }
    let logical_input = ir.logicalize_formatted_input(input)?;
    let promoted = logical_input
        .iter()
        .copied()
        .map(DoubleDouble::from_f64)
        .collect::<Vec<_>>();
    let logical_output = execute_double_double_nd_r2r_compute(ir, &promoted)?
        .into_iter()
        .map(DoubleDouble::to_f64)
        .collect::<Vec<_>>();
    ir.round_trip_formatted_output(&logical_output)
}

fn execute_double_double_nd_r2r_compute(
    ir: &DoubleDoubleNdR2rIr,
    input: &[DoubleDouble],
) -> Result<Vec<DoubleDouble>> {
    ir.validate()?;
    let expected =
        ir.tensor_len
            .checked_mul(ir.batch_count)
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "double-double ND R2R input element count",
            })?;
    if input.len() != expected {
        return Err(VkFftError::InputLengthMismatch {
            expected,
            actual: input.len(),
        });
    }
    let mut tensor = input.to_vec();
    if ir.direction == Direction::Forward && ir.has_spatial_zero_padding() {
        for batch in 0..ir.batch_count {
            let base = batch * ir.tensor_len;
            for linear in 0..ir.tensor_len {
                if ir.contains_spatial_zero_linear_index(linear) {
                    tensor[base + linear] = DoubleDouble::ZERO;
                }
            }
        }
    }
    for axis in &ir.axes {
        let outer_count = ir.tensor_len / (axis.axis_len * axis.inner_stride);
        let transform_batch_count = ir.batch_count * axis.line_count;
        let mut packed = Vec::with_capacity(transform_batch_count * axis.axis_len);
        for batch in 0..ir.batch_count {
            let batch_base = batch * ir.tensor_len;
            for outer in 0..outer_count {
                let outer_base = batch_base + outer * axis.axis_len * axis.inner_stride;
                for inner in 0..axis.inner_stride {
                    for lane in 0..axis.axis_len {
                        packed.push(tensor[outer_base + lane * axis.inner_stride + inner]);
                    }
                }
            }
        }
        let transformed = execute_double_double_r2r_ir(&axis.transform, &packed)?;
        let mut cursor = 0usize;
        for batch in 0..ir.batch_count {
            let batch_base = batch * ir.tensor_len;
            for outer in 0..outer_count {
                let outer_base = batch_base + outer * axis.axis_len * axis.inner_stride;
                for inner in 0..axis.inner_stride {
                    for lane in 0..axis.axis_len {
                        tensor[outer_base + lane * axis.inner_stride + inner] = transformed[cursor];
                        cursor += 1;
                    }
                }
            }
        }
    }
    if ir.direction == Direction::Inverse && ir.has_spatial_zero_padding() {
        for batch in 0..ir.batch_count {
            let base = batch * ir.tensor_len;
            for linear in 0..ir.tensor_len {
                if ir.contains_spatial_zero_linear_index(linear) {
                    tensor[base + linear] = DoubleDouble::ZERO;
                }
            }
        }
    }
    Ok(tensor)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::FftConfig;
    use crate::double_double::dft;

    fn dd_error(lhs: ComplexDoubleDouble, rhs: ComplexDoubleDouble) -> f64 {
        let re = (lhs.re - rhs.re).abs();
        let im = (lhs.im - rhs.im).abs();
        re.hi.abs() + re.lo.abs() + im.hi.abs() + im.lo.abs()
    }

    fn dd_scalar_error(lhs: DoubleDouble, rhs: DoubleDouble) -> f64 {
        let delta = (lhs - rhs).abs();
        delta.hi.abs() + delta.lo.abs()
    }

    fn recursive_bluestein_child(
        node: &crate::DoubleDoubleRecursiveFftNodeIr,
    ) -> Option<&DoubleDoubleBluesteinIr> {
        match node {
            crate::DoubleDoubleRecursiveFftNodeIr::Bluestein(ir) => Some(ir),
            crate::DoubleDoubleRecursiveFftNodeIr::CooleyTukey(ir) => {
                recursive_bluestein_child(&ir.right).or_else(|| recursive_bluestein_child(&ir.left))
            }
            _ => None,
        }
    }

    fn one_dim_bluestein_child(ir: &DoubleDoubleOneDimIr) -> Option<&DoubleDoubleBluesteinIr> {
        match ir {
            DoubleDoubleOneDimIr::Bluestein(ir) => Some(ir),
            DoubleDoubleOneDimIr::Recursive(ir) => recursive_bluestein_child(&ir.root),
            _ => None,
        }
    }

    #[test]
    fn dd_executable_schedule_splits_merged_radix32() {
        let plan32 =
            FftPlan::build(FftConfig::new(vec![32]).with_precision(Precision::DoubleDouble))
                .unwrap();
        let ir32 = DoubleDoubleStockhamIr::build(&plan32, Direction::Forward).unwrap();
        assert_eq!(
            ir32.stages
                .iter()
                .map(|stage| stage.radix)
                .collect::<Vec<_>>(),
            vec![16, 2]
        );

        let plan64 =
            FftPlan::build(FftConfig::new(vec![64]).with_precision(Precision::DoubleDouble))
                .unwrap();
        let ir64 = DoubleDoubleStockhamIr::build(&plan64, Direction::Forward).unwrap();
        assert_eq!(
            ir64.stages
                .iter()
                .map(|stage| stage.radix)
                .collect::<Vec<_>>(),
            vec![16, 2, 2]
        );
        assert_eq!(ir64.twiddles.stages.len(), 3);
        ir64.validate().unwrap();
    }

    #[test]
    fn even_real_n16_half_size_matches_dd_dft_and_round_trips() {
        let length = 16usize;
        let batch_count = 2usize;
        let r2c_plan = FftPlan::build(
            FftConfig::new(vec![length])
                .with_batch_count(batch_count)
                .with_transform(TransformKind::RealToComplex)
                .with_precision(Precision::DoubleDouble),
        )
        .unwrap();
        let c2r_plan = FftPlan::build(
            FftConfig::new(vec![length])
                .with_batch_count(batch_count)
                .with_transform(TransformKind::ComplexToReal)
                .with_precision(Precision::DoubleDouble)
                .with_inverse_normalization(true),
        )
        .unwrap();
        let r2c = DoubleDoubleRealFftIr::build(&r2c_plan).unwrap();
        let c2r = DoubleDoubleRealFftIr::build(&c2r_plan).unwrap();
        assert_eq!(r2c.kind, RealFftKind::RealToComplex);
        assert_eq!(c2r.kind, RealFftKind::ComplexToReal);
        assert_eq!(r2c.half_spectrum_len, 9);
        assert!(matches!(r2c.transform, DoubleDoubleOneDimIr::Stockham(_)));

        let input = (0..length * batch_count)
            .map(|index| {
                let x = index as f64;
                DoubleDouble::from_parts((0.13 * x).sin() + 0.002 * x, (index + 1) as f64 * 7.0e-32)
            })
            .collect::<Vec<_>>();
        let actual = execute_double_double_r2c_ir(&r2c, &input).unwrap();
        let mut expected = Vec::with_capacity(actual.len());
        for batch in 0..batch_count {
            let base = batch * length;
            let complex = input[base..base + length]
                .iter()
                .copied()
                .map(|value| ComplexDoubleDouble::new(value, DoubleDouble::ZERO))
                .collect::<Vec<_>>();
            let spectrum = dft(&complex, Direction::Forward, false).unwrap();
            expected.extend_from_slice(&spectrum[..r2c.half_spectrum_len]);
        }
        let error = actual
            .iter()
            .copied()
            .zip(expected.iter().copied())
            .map(|(actual, expected)| dd_error(actual, expected))
            .fold(0.0, f64::max);
        assert!(error < 2.0e-27, "DD even R2C forward error {error:e}");

        let restored = execute_double_double_c2r_ir(&c2r, &actual).unwrap();
        let round_trip_error = restored
            .iter()
            .copied()
            .zip(input.iter().copied())
            .map(|(actual, expected)| {
                let delta = (actual - expected).abs();
                delta.hi.abs() + delta.lo.abs()
            })
            .fold(0.0, f64::max);
        assert!(
            round_trip_error < 2.0e-27,
            "DD even real round trip error {round_trip_error:e}"
        );

        let c2r_unnormalized_plan = FftPlan::build(
            FftConfig::new(vec![length])
                .with_batch_count(batch_count)
                .with_transform(TransformKind::ComplexToReal)
                .with_precision(Precision::DoubleDouble),
        )
        .unwrap();
        let c2r_unnormalized = DoubleDoubleRealFftIr::build(&c2r_unnormalized_plan).unwrap();
        assert!(c2r_unnormalized.even_half_size);
        assert!(!c2r_unnormalized.normalize);
        let unnormalized = execute_double_double_c2r_ir(&c2r_unnormalized, &actual).unwrap();
        let n_scale = DoubleDouble::from_f64(length as f64);
        let unnormalized_error = unnormalized
            .iter()
            .copied()
            .zip(input.iter().copied())
            .map(|(actual, expected)| {
                let delta = (actual - expected * n_scale).abs();
                delta.hi.abs() + delta.lo.abs()
            })
            .fold(0.0, f64::max);
        assert!(
            unnormalized_error < 4.0e-26,
            "DD even real unnormalized inverse scale error {unnormalized_error:e}"
        );

        let r2c_f64_plan = FftPlan::build(
            FftConfig::new(vec![length])
                .with_batch_count(batch_count)
                .with_transform(TransformKind::RealToComplex)
                .with_precision(Precision::DoubleDoubleF64Storage),
        )
        .unwrap();
        let c2r_f64_plan = FftPlan::build(
            FftConfig::new(vec![length])
                .with_batch_count(batch_count)
                .with_transform(TransformKind::ComplexToReal)
                .with_precision(Precision::DoubleDoubleF64Storage)
                .with_inverse_normalization(true),
        )
        .unwrap();
        let r2c_f64 = DoubleDoubleRealFftIr::build(&r2c_f64_plan).unwrap();
        let c2r_f64 = DoubleDoubleRealFftIr::build(&c2r_f64_plan).unwrap();
        let f64_input = input
            .iter()
            .copied()
            .map(DoubleDouble::to_f64)
            .collect::<Vec<_>>();
        let f64_spectrum = execute_double_double_r2c_ir_f64_storage(&r2c_f64, &f64_input).unwrap();
        let promoted = f64_input
            .iter()
            .copied()
            .map(DoubleDouble::from_f64)
            .collect::<Vec<_>>();
        let promoted_expected = execute_double_double_r2c_compute(&r2c_f64, &promoted)
            .unwrap()
            .into_iter()
            .map(ComplexDoubleDouble::to_complex64)
            .collect::<Vec<_>>();
        assert_eq!(f64_spectrum, promoted_expected);
        let f64_restored =
            execute_double_double_c2r_ir_f64_storage(&c2r_f64, &f64_spectrum).unwrap();
        let f64_error = f64_restored
            .iter()
            .zip(&f64_input)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0, f64::max);
        assert!(
            f64_error < 2.0e-14,
            "DD/F64 even real round trip error {f64_error:e}"
        );
    }

    #[test]
    fn real_spatial_zero_padding_odd_even_matches_manual_dd_boundary() {
        let batch_count = 7usize;
        let grouped_batch = 3usize;
        for length in [15usize, 16usize] {
            let input = (0..length * batch_count)
                .map(|index| {
                    let x = index as f64;
                    DoubleDouble::from_parts(
                        (0.113 * x).sin() + 0.17 * (0.037 * x).cos() + 0.0011 * x,
                        (index + 1) as f64 * 6.0e-32,
                    )
                })
                .collect::<Vec<_>>();
            let mut manual = input.clone();
            for batch in 0..batch_count {
                let base = batch * length;
                for index in 3..7 {
                    manual[base + index] = DoubleDouble::ZERO;
                }
            }

            let build = |precision, transform| {
                let config = FftConfig::new(vec![length])
                    .with_batch_count(batch_count)
                    .with_precision(precision)
                    .with_transform(transform)
                    .with_inverse_normalization(transform == TransformKind::ComplexToReal)
                    .with_grouped_batch(0, grouped_batch)
                    .unwrap()
                    .with_zero_padding(0, 3, 7)
                    .unwrap();
                DoubleDoubleRealFftIr::build(&FftPlan::build(config).unwrap()).unwrap()
            };
            let r2c = build(Precision::DoubleDouble, TransformKind::RealToComplex);
            let c2r = build(Precision::DoubleDouble, TransformKind::ComplexToReal);
            assert_eq!(r2c.even_half_size, length.is_multiple_of(2));
            assert_eq!(r2c.grouped_batch, grouped_batch);
            assert_eq!(c2r.grouped_batch, grouped_batch);
            assert_eq!(r2c.transform.grouped_batch(), grouped_batch);
            assert_eq!(c2r.transform.grouped_batch(), grouped_batch);
            assert_eq!(r2c.batch_group_count(), 3);
            assert_eq!(c2r.batch_group_count(), 3);
            assert!(r2c.has_spatial_zero_padding());
            assert!(r2c.contains_spatial_zero_index(3));
            assert!(!r2c.contains_spatial_zero_index(7));
            assert_eq!(
                r2c.transform.external_storage(),
                PrecisionStorage::DoubleDouble
            );

            let baseline = DoubleDoubleRealFftIr::build(
                &FftPlan::build(
                    FftConfig::new(vec![length])
                        .with_batch_count(batch_count)
                        .with_precision(Precision::DoubleDouble)
                        .with_transform(TransformKind::RealToComplex),
                )
                .unwrap(),
            )
            .unwrap();
            let expected = execute_double_double_r2c_ir(&baseline, &manual).unwrap();
            let actual = execute_double_double_r2c_ir(&r2c, &input).unwrap();
            let forward_error = actual
                .iter()
                .copied()
                .zip(expected.iter().copied())
                .map(|(actual, expected)| dd_error(actual, expected))
                .fold(0.0, f64::max);
            assert!(
                forward_error < 5.0e-26,
                "DD padded real N={length} forward error {forward_error:e}"
            );
            let restored = execute_double_double_c2r_ir(&c2r, &actual).unwrap();
            let round_trip_error = restored
                .iter()
                .copied()
                .zip(manual.iter().copied())
                .map(|(actual, expected)| {
                    let delta = (actual - expected).abs();
                    delta.hi.abs() + delta.lo.abs()
                })
                .fold(0.0, f64::max);
            assert!(
                round_trip_error < 5.0e-25,
                "DD padded real N={length} round-trip error {round_trip_error:e}"
            );

            let r2c_f64 = build(
                Precision::DoubleDoubleF64Storage,
                TransformKind::RealToComplex,
            );
            let c2r_f64 = build(
                Precision::DoubleDoubleF64Storage,
                TransformKind::ComplexToReal,
            );
            let f64_input = input
                .iter()
                .copied()
                .map(DoubleDouble::to_f64)
                .collect::<Vec<_>>();
            let f64_manual = manual
                .iter()
                .copied()
                .map(DoubleDouble::to_f64)
                .collect::<Vec<_>>();
            let f64_spectrum =
                execute_double_double_r2c_ir_f64_storage(&r2c_f64, &f64_input).unwrap();
            let f64_restored =
                execute_double_double_c2r_ir_f64_storage(&c2r_f64, &f64_spectrum).unwrap();
            let f64_error = f64_restored
                .iter()
                .zip(&f64_manual)
                .map(|(actual, expected)| (actual - expected).abs())
                .fold(0.0, f64::max);
            assert!(
                f64_error < 5.0e-13,
                "DD/F64 padded real N={length} round-trip error {f64_error:e}"
            );
        }
    }

    #[test]
    fn odd_real_n15_matches_independent_dd_dft_and_round_trips() {
        let length = 15usize;
        let batch_count = 2usize;
        let r2c_plan = FftPlan::build(
            FftConfig::new(vec![length])
                .with_batch_count(batch_count)
                .with_transform(TransformKind::RealToComplex)
                .with_precision(Precision::DoubleDouble),
        )
        .unwrap();
        let c2r_plan = FftPlan::build(
            FftConfig::new(vec![length])
                .with_batch_count(batch_count)
                .with_transform(TransformKind::ComplexToReal)
                .with_precision(Precision::DoubleDouble)
                .with_inverse_normalization(true),
        )
        .unwrap();
        let r2c = DoubleDoubleRealFftIr::build(&r2c_plan).unwrap();
        let c2r = DoubleDoubleRealFftIr::build(&c2r_plan).unwrap();
        assert_eq!(r2c.kind, RealFftKind::RealToComplex);
        assert_eq!(c2r.kind, RealFftKind::ComplexToReal);
        assert_eq!(r2c.half_spectrum_len, 8);
        assert!(matches!(r2c.transform, DoubleDoubleOneDimIr::Stockham(_)));

        let input = (0..length * batch_count)
            .map(|index| {
                let x = index as f64;
                DoubleDouble::from_parts((0.13 * x).sin() + 0.002 * x, (index + 1) as f64 * 7.0e-32)
            })
            .collect::<Vec<_>>();
        let actual = execute_double_double_r2c_ir(&r2c, &input).unwrap();
        let mut expected = Vec::with_capacity(actual.len());
        for batch in 0..batch_count {
            let base = batch * length;
            let complex = input[base..base + length]
                .iter()
                .copied()
                .map(|value| ComplexDoubleDouble::new(value, DoubleDouble::ZERO))
                .collect::<Vec<_>>();
            let spectrum = dft(&complex, Direction::Forward, false).unwrap();
            expected.extend_from_slice(&spectrum[..r2c.half_spectrum_len]);
        }
        let error = actual
            .iter()
            .copied()
            .zip(expected.iter().copied())
            .map(|(actual, expected)| dd_error(actual, expected))
            .fold(0.0, f64::max);
        assert!(error < 2.0e-27, "DD odd R2C forward error {error:e}");

        let restored = execute_double_double_c2r_ir(&c2r, &actual).unwrap();
        let round_trip_error = restored
            .iter()
            .copied()
            .zip(input.iter().copied())
            .map(|(actual, expected)| {
                let delta = (actual - expected).abs();
                delta.hi.abs() + delta.lo.abs()
            })
            .fold(0.0, f64::max);
        assert!(
            round_trip_error < 2.0e-27,
            "DD odd real round trip error {round_trip_error:e}"
        );

        let r2c_f64_plan = FftPlan::build(
            FftConfig::new(vec![length])
                .with_batch_count(batch_count)
                .with_transform(TransformKind::RealToComplex)
                .with_precision(Precision::DoubleDoubleF64Storage),
        )
        .unwrap();
        let c2r_f64_plan = FftPlan::build(
            FftConfig::new(vec![length])
                .with_batch_count(batch_count)
                .with_transform(TransformKind::ComplexToReal)
                .with_precision(Precision::DoubleDoubleF64Storage)
                .with_inverse_normalization(true),
        )
        .unwrap();
        let r2c_f64 = DoubleDoubleRealFftIr::build(&r2c_f64_plan).unwrap();
        let c2r_f64 = DoubleDoubleRealFftIr::build(&c2r_f64_plan).unwrap();
        let f64_input = input
            .iter()
            .copied()
            .map(DoubleDouble::to_f64)
            .collect::<Vec<_>>();
        let f64_spectrum = execute_double_double_r2c_ir_f64_storage(&r2c_f64, &f64_input).unwrap();
        let promoted = f64_input
            .iter()
            .copied()
            .map(DoubleDouble::from_f64)
            .collect::<Vec<_>>();
        let promoted_expected = execute_double_double_r2c_compute(&r2c_f64, &promoted)
            .unwrap()
            .into_iter()
            .map(ComplexDoubleDouble::to_complex64)
            .collect::<Vec<_>>();
        assert_eq!(f64_spectrum, promoted_expected);
        let f64_restored =
            execute_double_double_c2r_ir_f64_storage(&c2r_f64, &f64_spectrum).unwrap();
        let f64_error = f64_restored
            .iter()
            .zip(&f64_input)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0, f64::max);
        assert!(
            f64_error < 2.0e-14,
            "DD/F64 odd real round trip error {f64_error:e}"
        );
    }

    #[test]
    fn nd_real_3x7_and_3x8_match_independent_dd_dft_and_round_trip() {
        let batch_count = 7usize;
        let grouped_batch = 3usize;
        for last_len in [7usize, 8usize] {
            let dimensions = [3usize, last_len];
            let full_tensor_len = dimensions.iter().product::<usize>();
            let compact_last = last_len / 2 + 1;
            let compact_tensor_len = dimensions[0] * compact_last;
            let r2c_plan = FftPlan::build(
                FftConfig::new(dimensions.to_vec())
                    .with_batch_count(batch_count)
                    .with_transform(TransformKind::RealToComplex)
                    .with_precision(Precision::DoubleDouble)
                    .with_grouped_batch(0, grouped_batch)
                    .unwrap()
                    .with_grouped_batch(1, grouped_batch)
                    .unwrap(),
            )
            .unwrap();
            let c2r_plan = FftPlan::build(
                FftConfig::new(dimensions.to_vec())
                    .with_batch_count(batch_count)
                    .with_transform(TransformKind::ComplexToReal)
                    .with_precision(Precision::DoubleDouble)
                    .with_inverse_normalization(true)
                    .with_grouped_batch(0, grouped_batch)
                    .unwrap()
                    .with_grouped_batch(1, grouped_batch)
                    .unwrap(),
            )
            .unwrap();
            let r2c = DoubleDoubleNdRealFftIr::build(&r2c_plan).unwrap();
            let c2r = DoubleDoubleNdRealFftIr::build(&c2r_plan).unwrap();
            assert_eq!(r2c.full_tensor_len, full_tensor_len);
            assert_eq!(r2c.compact_tensor_len, compact_tensor_len);
            assert_eq!(r2c.compact_dimensions, vec![3, compact_last]);
            assert_eq!(r2c.real_axis.even_half_size, last_len.is_multiple_of(2));
            assert_eq!(r2c.real_grouped_batch, grouped_batch);
            assert_eq!(c2r.real_grouped_batch, grouped_batch);
            assert_eq!(r2c.real_axis.grouped_batch, grouped_batch);
            assert_eq!(c2r.real_axis.grouped_batch, grouped_batch);
            assert_eq!(r2c.batch_group_count(), 3);
            assert_eq!(c2r.batch_group_count(), 3);
            assert_eq!(r2c.complex_axes.len(), 1);
            assert_eq!(r2c.complex_axes[0].grouped_batch, grouped_batch);
            assert_eq!(r2c.complex_axes[0].transform.grouped_batch(), grouped_batch);
            assert_eq!(
                r2c.real_axis.external_storage,
                PrecisionStorage::DoubleDouble
            );
            assert!(r2c
                .complex_axes
                .iter()
                .all(|axis| axis.transform.external_storage() == PrecisionStorage::DoubleDouble));

            let input = (0..full_tensor_len * batch_count)
                .map(|index| {
                    let x = index as f64;
                    DoubleDouble::from_parts(
                        (0.09 * x).sin() + 0.21 * (0.04 * x).cos() + 0.0015 * x,
                        (index + 1) as f64 * 6.0e-32,
                    )
                })
                .collect::<Vec<_>>();
            let actual = execute_double_double_nd_r2c_ir(&r2c, &input).unwrap();
            let mut expected = vec![ComplexDoubleDouble::default(); actual.len()];
            for batch in 0..batch_count {
                let input_base = batch * full_tensor_len;
                let output_base = batch * compact_tensor_len;
                for k0 in 0..dimensions[0] {
                    for k1 in 0..compact_last {
                        let mut sum = ComplexDoubleDouble::default();
                        for n0 in 0..dimensions[0] {
                            let root0 =
                                unit_root(n0 * k0, dimensions[0], Direction::Forward).unwrap();
                            for n1 in 0..last_len {
                                let root1 =
                                    unit_root(n1 * k1, last_len, Direction::Forward).unwrap();
                                let value = input[input_base + n0 * last_len + n1];
                                sum += ComplexDoubleDouble::new(value, DoubleDouble::ZERO)
                                    * root0
                                    * root1;
                            }
                        }
                        expected[output_base + k0 * compact_last + k1] = sum;
                    }
                }
            }
            let forward_error = actual
                .iter()
                .copied()
                .zip(expected.iter().copied())
                .map(|(actual, expected)| dd_error(actual, expected))
                .fold(0.0, f64::max);
            assert!(
                forward_error < 5.0e-26,
                "DD ND R2C [3,{last_len}] forward error {forward_error:e}"
            );
            let restored = execute_double_double_nd_c2r_ir(&c2r, &actual).unwrap();
            let round_trip_error = restored
                .iter()
                .copied()
                .zip(input.iter().copied())
                .map(|(actual, expected)| {
                    let delta = (actual - expected).abs();
                    delta.hi.abs() + delta.lo.abs()
                })
                .fold(0.0, f64::max);
            assert!(
                round_trip_error < 5.0e-26,
                "DD ND real [3,{last_len}] round-trip error {round_trip_error:e}"
            );

            let r2c_f64_plan = FftPlan::build(
                FftConfig::new(dimensions.to_vec())
                    .with_batch_count(batch_count)
                    .with_transform(TransformKind::RealToComplex)
                    .with_precision(Precision::DoubleDoubleF64Storage)
                    .with_grouped_batch(0, grouped_batch)
                    .unwrap()
                    .with_grouped_batch(1, grouped_batch)
                    .unwrap(),
            )
            .unwrap();
            let c2r_f64_plan = FftPlan::build(
                FftConfig::new(dimensions.to_vec())
                    .with_batch_count(batch_count)
                    .with_transform(TransformKind::ComplexToReal)
                    .with_precision(Precision::DoubleDoubleF64Storage)
                    .with_inverse_normalization(true)
                    .with_grouped_batch(0, grouped_batch)
                    .unwrap()
                    .with_grouped_batch(1, grouped_batch)
                    .unwrap(),
            )
            .unwrap();
            let r2c_f64 = DoubleDoubleNdRealFftIr::build(&r2c_f64_plan).unwrap();
            let c2r_f64 = DoubleDoubleNdRealFftIr::build(&c2r_f64_plan).unwrap();
            assert_eq!(r2c_f64.external_storage, PrecisionStorage::F64);
            assert_eq!(
                r2c_f64.real_axis.external_storage,
                PrecisionStorage::DoubleDouble
            );
            let f64_input = input
                .iter()
                .copied()
                .map(DoubleDouble::to_f64)
                .collect::<Vec<_>>();
            let f64_spectrum =
                execute_double_double_nd_r2c_ir_f64_storage(&r2c_f64, &f64_input).unwrap();
            let promoted = f64_input
                .iter()
                .copied()
                .map(DoubleDouble::from_f64)
                .collect::<Vec<_>>();
            let promoted_expected = execute_double_double_nd_r2c_compute(&r2c_f64, &promoted)
                .unwrap()
                .into_iter()
                .map(ComplexDoubleDouble::to_complex64)
                .collect::<Vec<_>>();
            assert_eq!(f64_spectrum, promoted_expected);
            let f64_restored =
                execute_double_double_nd_c2r_ir_f64_storage(&c2r_f64, &f64_spectrum).unwrap();
            let f64_error = f64_restored
                .iter()
                .zip(&f64_input)
                .map(|(actual, expected)| (actual - expected).abs())
                .fold(0.0, f64::max);
            assert!(
                f64_error < 5.0e-14,
                "DD/F64 ND real [3,{last_len}] round-trip error {f64_error:e}"
            );
        }
    }

    #[test]
    fn nd_real_spatial_zero_padding_matches_manual_dd_boundary() {
        let dimensions = [3usize, 8usize];
        let batch_count = 7usize;
        let grouped_batch = 3usize;
        let full_tensor_len = dimensions.iter().product::<usize>();
        let input = (0..full_tensor_len * batch_count)
            .map(|index| {
                let x = index as f64;
                DoubleDouble::from_parts(
                    (0.071 * x).sin() + 0.17 * (0.033 * x).cos() + 0.0009 * x,
                    (index + 1) as f64 * 4.0e-32,
                )
            })
            .collect::<Vec<_>>();
        let mut manual = input.clone();
        for batch in 0..batch_count {
            let base = batch * full_tensor_len;
            for n0 in 0..dimensions[0] {
                for n1 in 0..dimensions[1] {
                    if (1..2).contains(&n0) || (2..4).contains(&n1) {
                        manual[base + n0 * dimensions[1] + n1] = DoubleDouble::ZERO;
                    }
                }
            }
        }

        let r2c_config = FftConfig::new(dimensions.to_vec())
            .with_batch_count(batch_count)
            .with_transform(TransformKind::RealToComplex)
            .with_precision(Precision::DoubleDouble)
            .with_grouped_batch(0, grouped_batch)
            .unwrap()
            .with_grouped_batch(1, grouped_batch)
            .unwrap()
            .with_zero_padding(0, 1, 2)
            .unwrap()
            .with_zero_padding(1, 2, 4)
            .unwrap();
        let c2r_config = FftConfig::new(dimensions.to_vec())
            .with_batch_count(batch_count)
            .with_transform(TransformKind::ComplexToReal)
            .with_precision(Precision::DoubleDouble)
            .with_inverse_normalization(true)
            .with_grouped_batch(0, grouped_batch)
            .unwrap()
            .with_grouped_batch(1, grouped_batch)
            .unwrap()
            .with_zero_padding(0, 1, 2)
            .unwrap()
            .with_zero_padding(1, 2, 4)
            .unwrap();
        let r2c = DoubleDoubleNdRealFftIr::build(&FftPlan::build(r2c_config).unwrap()).unwrap();
        let c2r = DoubleDoubleNdRealFftIr::build(&FftPlan::build(c2r_config).unwrap()).unwrap();
        assert!(r2c.has_spatial_zero_padding());
        assert_eq!(r2c.real_grouped_batch, grouped_batch);
        assert_eq!(c2r.real_grouped_batch, grouped_batch);
        assert_eq!(r2c.real_axis.grouped_batch, grouped_batch);
        assert_eq!(r2c.complex_axes[0].transform.grouped_batch(), grouped_batch);
        assert_eq!(r2c.batch_group_count(), 3);
        assert!(r2c.contains_spatial_zero_linear_index(dimensions[1]));
        assert!(r2c.contains_spatial_zero_linear_index(2));
        assert!(!r2c.contains_spatial_zero_linear_index(0));
        assert_eq!(
            r2c.real_axis.external_storage,
            PrecisionStorage::DoubleDouble
        );

        let baseline_plan = FftPlan::build(
            FftConfig::new(dimensions.to_vec())
                .with_batch_count(batch_count)
                .with_transform(TransformKind::RealToComplex)
                .with_precision(Precision::DoubleDouble),
        )
        .unwrap();
        let baseline = DoubleDoubleNdRealFftIr::build(&baseline_plan).unwrap();
        let expected = execute_double_double_nd_r2c_ir(&baseline, &manual).unwrap();
        let actual = execute_double_double_nd_r2c_ir(&r2c, &input).unwrap();
        let forward_error = actual
            .iter()
            .copied()
            .zip(expected.iter().copied())
            .map(|(actual, expected)| dd_error(actual, expected))
            .fold(0.0, f64::max);
        assert!(
            forward_error < 5.0e-26,
            "DD ND padded R2C error {forward_error:e}"
        );

        let restored = execute_double_double_nd_c2r_ir(&c2r, &actual).unwrap();
        let round_trip_error = restored
            .iter()
            .copied()
            .zip(manual.iter().copied())
            .map(|(actual, expected)| {
                let delta = (actual - expected).abs();
                delta.hi.abs() + delta.lo.abs()
            })
            .fold(0.0, f64::max);
        assert!(
            round_trip_error < 5.0e-26,
            "DD ND padded real round-trip error {round_trip_error:e}"
        );

        let f64_r2c_config = FftConfig::new(dimensions.to_vec())
            .with_batch_count(batch_count)
            .with_transform(TransformKind::RealToComplex)
            .with_precision(Precision::DoubleDoubleF64Storage)
            .with_zero_padding(0, 1, 2)
            .unwrap()
            .with_zero_padding(1, 2, 4)
            .unwrap();
        let f64_c2r_config = FftConfig::new(dimensions.to_vec())
            .with_batch_count(batch_count)
            .with_transform(TransformKind::ComplexToReal)
            .with_precision(Precision::DoubleDoubleF64Storage)
            .with_inverse_normalization(true)
            .with_zero_padding(0, 1, 2)
            .unwrap()
            .with_zero_padding(1, 2, 4)
            .unwrap();
        let f64_r2c =
            DoubleDoubleNdRealFftIr::build(&FftPlan::build(f64_r2c_config).unwrap()).unwrap();
        let f64_c2r =
            DoubleDoubleNdRealFftIr::build(&FftPlan::build(f64_c2r_config).unwrap()).unwrap();
        let f64_input = input
            .iter()
            .copied()
            .map(DoubleDouble::to_f64)
            .collect::<Vec<_>>();
        let f64_manual = manual
            .iter()
            .copied()
            .map(DoubleDouble::to_f64)
            .collect::<Vec<_>>();
        let f64_spectrum =
            execute_double_double_nd_r2c_ir_f64_storage(&f64_r2c, &f64_input).unwrap();
        let f64_restored =
            execute_double_double_nd_c2r_ir_f64_storage(&f64_c2r, &f64_spectrum).unwrap();
        let f64_error = f64_restored
            .iter()
            .zip(&f64_manual)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0, f64::max);
        assert!(
            f64_error < 5.0e-14,
            "DD/F64 ND padded real error {f64_error:e}"
        );
    }

    #[test]
    fn nd_prime_axes_compose_direct_rader_and_bluestein() {
        for (prime, force_bluestein) in [(47usize, false), (103usize, true)] {
            let dimensions = [2usize, prime];
            let tensor_len = dimensions.iter().product::<usize>();
            let mut tuning = crate::PlannerTuning::portable();
            if force_bluestein {
                tuning.max_rader_fft_prime = 100;
            }
            let forward_plan = FftPlan::build(
                FftConfig::new(dimensions.to_vec())
                    .with_precision(Precision::DoubleDouble)
                    .with_tuning(tuning),
            )
            .unwrap();
            let inverse_plan = FftPlan::build(
                FftConfig::new(dimensions.to_vec())
                    .with_precision(Precision::DoubleDouble)
                    .with_inverse_normalization(true)
                    .with_tuning(tuning),
            )
            .unwrap();
            let forward = DoubleDoubleNdFftIr::build(&forward_plan, Direction::Forward).unwrap();
            let inverse = DoubleDoubleNdFftIr::build(&inverse_plan, Direction::Inverse).unwrap();
            match (&forward.axes[0].transform, force_bluestein) {
                (DoubleDoubleOneDimIr::DirectRader(rader), false) => {
                    assert_eq!(rader.prime, 47);
                }
                (DoubleDoubleOneDimIr::Bluestein(bluestein), true) => {
                    assert_eq!(bluestein.logical_len, 103);
                    assert_eq!(bluestein.convolution_len, 210);
                }
                (other, _) => panic!("unexpected DD ND prime-axis child: {other:?}"),
            }

            let input = (0..tensor_len)
                .map(|index| {
                    let x = index as f64;
                    ComplexDoubleDouble::new(
                        DoubleDouble::from_parts(
                            (0.037 * x).sin() + 0.0003 * x,
                            (index + 1) as f64 * 8.0e-32,
                        ),
                        DoubleDouble::from_parts(
                            (0.019 * x).cos() - 0.0002 * x,
                            -(index as f64 + 1.0) * 4.0e-32,
                        ),
                    )
                })
                .collect::<Vec<_>>();
            let actual = execute_double_double_nd_ir(&forward, &input).unwrap();
            let mut expected = vec![ComplexDoubleDouble::default(); tensor_len];
            for k0 in 0..dimensions[0] {
                for k1 in 0..dimensions[1] {
                    let mut sum = ComplexDoubleDouble::default();
                    for n0 in 0..dimensions[0] {
                        let root0 = unit_root(n0 * k0, dimensions[0], Direction::Forward).unwrap();
                        for n1 in 0..dimensions[1] {
                            let root1 =
                                unit_root(n1 * k1, dimensions[1], Direction::Forward).unwrap();
                            sum += input[n0 * dimensions[1] + n1] * root0 * root1;
                        }
                    }
                    expected[k0 * dimensions[1] + k1] = sum;
                }
            }
            let error = actual
                .iter()
                .copied()
                .zip(expected.iter().copied())
                .map(|(actual, expected)| dd_error(actual, expected))
                .fold(0.0, f64::max);
            assert!(
                error < 1.0e-22,
                "DD ND [2,{prime}] prime-axis forward error {error:e}"
            );
            let restored = execute_double_double_nd_ir(&inverse, &actual).unwrap();
            let round_trip_error = restored
                .iter()
                .copied()
                .zip(input.iter().copied())
                .map(|(actual, expected)| dd_error(actual, expected))
                .fold(0.0, f64::max);
            assert!(
                round_trip_error < 1.0e-20,
                "DD ND [2,{prime}] prime-axis round-trip error {round_trip_error:e}"
            );
        }
    }

    #[test]
    fn nd_bluestein_p2053_propagates_recursive_convolution_child() {
        let dimensions = [2usize, 2_053usize];
        let tensor_len = dimensions.iter().product::<usize>();
        let mut tuning = crate::PlannerTuning::portable();
        tuning.max_rader_fft_prime = 100;
        let forward_plan = FftPlan::build(
            FftConfig::new(dimensions.to_vec())
                .with_precision(Precision::DoubleDouble)
                .with_tuning(tuning),
        )
        .unwrap();
        let inverse_plan = FftPlan::build(
            FftConfig::new(dimensions.to_vec())
                .with_precision(Precision::DoubleDouble)
                .with_inverse_normalization(true)
                .with_tuning(tuning),
        )
        .unwrap();
        let forward = DoubleDoubleNdFftIr::build(&forward_plan, Direction::Forward).unwrap();
        let inverse = DoubleDoubleNdFftIr::build(&inverse_plan, Direction::Inverse).unwrap();
        assert_eq!(forward.axes[0].axis, 1);
        assert_eq!(forward.axes[0].line_count, 2);
        assert_eq!(forward.axes[0].transform.batch_count(), 2);
        assert_eq!(forward.axes[0].transform.grouped_batch(), 1);
        let DoubleDoubleOneDimIr::Bluestein(bluestein) = &forward.axes[0].transform else {
            panic!("forced [2,2053] DD ND long axis must use Bluestein");
        };
        assert_eq!(bluestein.logical_len, 2_053);
        assert_eq!(bluestein.convolution_len, 4_116);
        assert!(matches!(
            bluestein.forward_fft,
            DoubleDoubleBluesteinConvolutionIr::Recursive(_)
        ));
        assert!(matches!(
            bluestein.inverse_fft,
            DoubleDoubleBluesteinConvolutionIr::Recursive(_)
        ));
        let program = crate::ProgramIr::double_double_nd(&forward).unwrap();
        let shaders = crate::backend::vulkan::VulkanGlslBackend
            .lower_double_double_nd(&forward)
            .unwrap();
        assert_eq!(shaders.len(), program.passes.len());
        assert!(program.passes.len() > 20);
        assert!(program.passes.iter().any(|pass| pass.dispatch.x == 1));
        assert!(program.passes.iter().any(|pass| pass.dispatch.x == 2));
        assert!(
            shaders
                .iter()
                .zip(&program.passes)
                .all(|(shader, pass)| shader.dispatch == pass.dispatch)
        );
        for shader in &shaders {
            assert_eq!(shader.compile_spirv().unwrap().words[0], 0x0723_0203);
        }

        let n0 = 1usize;
        let n1 = 137usize;
        let impulse = ComplexDoubleDouble::new(
            DoubleDouble::from_parts(1.25, 3.0e-31),
            DoubleDouble::from_parts(-0.75, -2.0e-31),
        );
        let mut input = vec![ComplexDoubleDouble::default(); tensor_len];
        input[n0 * dimensions[1] + n1] = impulse;
        let actual = execute_double_double_nd_ir(&forward, &input).unwrap();
        let mut forward_error = 0.0f64;
        for k0 in 0..dimensions[0] {
            let root0 = unit_root(n0 * k0, dimensions[0], Direction::Forward).unwrap();
            for k1 in 0..dimensions[1] {
                let root1 = unit_root(n1 * k1, dimensions[1], Direction::Forward).unwrap();
                let expected = impulse * root0 * root1;
                forward_error =
                    forward_error.max(dd_error(actual[k0 * dimensions[1] + k1], expected));
            }
        }
        assert!(
            forward_error < 5.0e-21,
            "DD ND [2,2053] recursive-Bluestein impulse error {forward_error:e}"
        );
        let restored = execute_double_double_nd_ir(&inverse, &actual).unwrap();
        let round_trip_error = restored
            .iter()
            .copied()
            .zip(input.iter().copied())
            .map(|(actual, expected)| dd_error(actual, expected))
            .fold(0.0, f64::max);
        assert!(
            round_trip_error < 5.0e-19,
            "DD ND [2,2053] recursive-Bluestein round-trip error {round_trip_error:e}"
        );
    }

    #[test]
    fn nd_intel_strided_dd_axis_preserves_upstream_register_boost_single_upload() {
        let dimensions = vec![1_024usize, 2usize];
        let device = DeviceProfile {
            shared_memory_bytes: 32 * 1024,
            shared_memory_pow2_bytes: 32 * 1024,
            max_threads_per_block: 1024,
            max_workgroup_size: [1024, 1024, 64],
            supports_f64: true,
            ..DeviceProfile::generic(crate::Backend::Vulkan, crate::GpuVendor::Intel)
        };

        let baseline_config = FftConfig::new(dimensions.clone())
            .with_precision(Precision::DoubleDouble)
            .resolve_tuning_for_device(device);
        let baseline_plan = FftPlan::build_for_device(baseline_config, device).unwrap();
        let baseline =
            DoubleDoubleNdFftIr::build_for_device(&baseline_plan, Direction::Forward, device)
                .unwrap();
        let outer = baseline.axes.iter().find(|axis| axis.axis == 0).unwrap();
        let DoubleDoubleOneDimIr::Stockham(stockham) = &outer.transform else {
            panic!(
                "Intel strided DD N1024 must retain upstream registerBoost=2 single-upload Stockham"
            );
        };
        assert_eq!(stockham.sequence_len, 1_024);
        assert!(stockham.axis_batch_block.is_none());
        let program = crate::ProgramIr::double_double_nd(&baseline).unwrap();
        program.validate().unwrap();
        let shaders = crate::backend::vulkan::VulkanGlslBackend
            .lower_double_double_nd(&baseline)
            .unwrap();
        assert_eq!(shaders.len(), program.passes.len());
        for shader in &shaders {
            assert_eq!(shader.compile_spirv().unwrap().words[0], 0x0723_0203);
        }
        let mut input = vec![ComplexDoubleDouble::default(); 1_024 * 2];
        input[0] = ComplexDoubleDouble::new(DoubleDouble::ONE, DoubleDouble::ZERO);
        let actual = execute_double_double_nd_ir(&baseline, &input).unwrap();
        let expected = ComplexDoubleDouble::new(DoubleDouble::ONE, DoubleDouble::ZERO);
        let impulse_error = actual
            .iter()
            .copied()
            .map(|value| dd_error(value, expected))
            .fold(0.0f64, f64::max);
        assert!(
            impulse_error < 5.0e-20,
            "Intel strided DD N1024 ND impulse error {impulse_error:e}"
        );

        let boosted_config = FftConfig::new(dimensions)
            .with_precision(Precision::DoubleDouble)
            .with_bandwidth_boost(2)
            .resolve_tuning_for_device(device);
        let boosted_plan = FftPlan::build_for_device(boosted_config, device).unwrap();
        let boosted =
            DoubleDoubleNdFftIr::build_for_device(&boosted_plan, Direction::Forward, device)
                .unwrap();
        let boosted_outer = boosted.axes.iter().find(|axis| axis.axis == 0).unwrap();
        let DoubleDoubleOneDimIr::Stockham(boosted_stockham) = &boosted_outer.transform else {
            panic!("bandwidthBoost=2 should preserve the Intel DD N1024 single-upload child");
        };
        assert_eq!(boosted_stockham.sequence_len, 1_024);
        assert!(boosted_stockham.axis_batch_block.is_none());
        boosted.validate().unwrap();
    }

    #[test]
    fn nd_higher_axis_dd_three_upload_blocks_keep_transforms_on_x() {
        let device = DeviceProfile {
            shared_memory_bytes: 1024,
            shared_memory_pow2_bytes: 1024,
            max_threads_per_block: 1024,
            max_workgroup_size: [1024, 1024, 64],
            supports_f64: true,
            ..DeviceProfile::generic(crate::Backend::Vulkan, crate::GpuVendor::Nvidia)
        };
        let config = FftConfig::new(vec![8_192usize, 2])
            .with_precision(Precision::DoubleDouble)
            .resolve_tuning_for_device(device);
        let transform = crate::TransformIr::build(config, Direction::Forward, device).unwrap();
        let crate::TransformIr::ComplexNdDoubleDouble(nd) = transform else {
            panic!("DD [8192,2] should remain ND C2C");
        };
        let outer = nd.axes.iter().find(|axis| axis.axis == 0).unwrap();
        let DoubleDoubleOneDimIr::Recursive(child) = &outer.transform else {
            panic!("constrained higher-axis N8192 should use recursive DD IR");
        };
        let schedule = child
            .stockham_upload_schedule
            .as_ref()
            .expect("constrained higher-axis N8192 should retain upload metadata");
        assert_eq!(schedule.upload_count, 3);
        assert_eq!(schedule.axis_split, vec![32, 16, 16]);
        let four_step = child
            .three_upload_four_step_plan
            .expect("constrained higher-axis N8192 should retain three-upload metadata");
        assert_eq!(four_step.axis_split, [32, 16, 16]);
        for block in [
            four_step.upload2_axis_block,
            four_step.upload1_axis_block,
            four_step.upload0_axis_block,
        ] {
            assert!(block.transforms_on_x);
            assert!(!block.axis_swapped);
            assert_eq!(
                [block.local_size_x, block.local_size_y],
                [block.grouped_batch, block.threads_per_transform]
            );
        }
        let crate::double_double_recursive_ir::DoubleDoubleRecursiveFftNodeIr::CooleyTukey(root) =
            &child.root
        else {
            panic!("constrained higher-axis N8192 should keep a Cooley root");
        };
        let crate::double_double_recursive_ir::DoubleDoubleRecursiveFftNodeIr::Stockham(low) =
            &root.left
        else {
            panic!("constrained higher-axis N8192 should keep a low Stockham leaf");
        };
        let crate::double_double_recursive_ir::DoubleDoubleRecursiveFftNodeIr::CooleyTukey(upper) =
            &root.right
        else {
            panic!("constrained higher-axis N8192 should keep an upper Cooley node");
        };
        let (
            crate::double_double_recursive_ir::DoubleDoubleRecursiveFftNodeIr::Stockham(middle),
            crate::double_double_recursive_ir::DoubleDoubleRecursiveFftNodeIr::Stockham(high),
        ) = (&upper.left, &upper.right)
        else {
            panic!("constrained higher-axis N8192 should keep middle/high Stockham leaves");
        };
        assert_eq!(low.axis_batch_block, Some(four_step.upload0_axis_block));
        assert_eq!(middle.axis_batch_block, Some(four_step.upload1_axis_block));
        assert_eq!(high.axis_batch_block, Some(four_step.upload2_axis_block));

        let program = crate::ProgramIr::double_double_recursive(child).unwrap();
        assert_eq!(program.passes.len(), 3);
        let shaders = crate::backend::vulkan::VulkanGlslBackend
            .lower_double_double_recursive(child)
            .unwrap();
        assert_eq!(shaders.len(), 3);
        assert_eq!(
            shaders
                .iter()
                .map(|shader| shader.sequence_len)
                .collect::<Vec<_>>(),
            vec![16, 16, 32]
        );
        assert!(shaders.iter().all(|shader| shader.compile_spirv().is_ok()));
    }

    #[test]
    fn nd_grouped_padded_n102272_forced_rader_three_upload_keeps_exact_blocks() {
        fn block_of(
            node: &crate::double_double_recursive_ir::DoubleDoubleRecursiveFftNodeIr,
        ) -> Option<crate::scheduler::StockhamAxisBlockSchedule> {
            use crate::double_double_recursive_ir::DoubleDoubleRecursiveFftNodeIr;
            match node {
                DoubleDoubleRecursiveFftNodeIr::Stockham(stockham) => stockham.axis_batch_block,
                DoubleDoubleRecursiveFftNodeIr::CooleyTukey(cooley) => {
                    cooley.pack_right.axis_batch_block
                }
                DoubleDoubleRecursiveFftNodeIr::DirectRader(rader) => rader.axis_batch_block,
                DoubleDoubleRecursiveFftNodeIr::FftRader(rader) => rader.caller_axis_batch_block,
                DoubleDoubleRecursiveFftNodeIr::Bluestein(_) => None,
            }
        }

        for vendor in [crate::GpuVendor::Nvidia, crate::GpuVendor::Amd] {
            let device = DeviceProfile {
                shared_memory_bytes: 8 * 1024,
                shared_memory_pow2_bytes: 8 * 1024,
                max_threads_per_block: 1024,
                max_workgroup_size: [1024, 1024, 64],
                supports_f64: true,
                ..DeviceProfile::generic(crate::Backend::Vulkan, vendor)
            };
            let config = FftConfig::new(vec![102_272usize, 8])
                .with_batch_count(5)
                .with_grouped_batch(0, 3)
                .unwrap()
                .with_zero_padding(0, 1, 2)
                .unwrap()
                .with_precision(Precision::DoubleDouble)
                .with_tuning(PlannerTuning::portable());
            let transform = crate::TransformIr::build(config, Direction::Forward, device).unwrap();
            let crate::TransformIr::ComplexNdDoubleDouble(nd) = transform else {
                panic!("grouped padded DD N102272 probe did not build ND C2C");
            };
            let outer = nd.axes.iter().find(|axis| axis.axis == 0).unwrap();
            let DoubleDoubleOneDimIr::Recursive(recursive) = &outer.transform else {
                panic!("grouped padded DD N102272 must retain recursive Rader IR");
            };
            let schedule = recursive
                .rader_forced_upload_schedule
                .as_ref()
                .expect("DD N102272 must retain forced-Rader upload metadata");
            assert_eq!(schedule.upload_count, 3);
            assert_eq!(schedule.axis_split, vec![64, 47, 34]);
            assert!(recursive.three_upload_four_step_plan.is_none());

            let crate::double_double_recursive_ir::DoubleDoubleRecursiveFftNodeIr::CooleyTukey(
                root,
            ) = &recursive.root
            else {
                panic!("DD N102272 forced-Rader root must remain Cooley-Tukey");
            };
            let crate::double_double_recursive_ir::DoubleDoubleRecursiveFftNodeIr::CooleyTukey(
                upper,
            ) = &root.right
            else {
                panic!("DD N102272 forced-Rader root must retain upper Cooley node");
            };
            let nodes = [&root.left, &upper.left, &upper.right];
            let expected = [
                (64usize, 63_920usize, 8usize),
                (47, 87_040, 24),
                (34, 120_320, 3),
            ];
            for (node, (len, batch, lanes)) in nodes.into_iter().zip(expected) {
                assert_eq!(node.logical_len(), len);
                assert_eq!(node.batch_count(), batch);
                let block = block_of(node).expect("DD N102272 component physical block");
                assert_eq!(block.threads_per_transform, lanes);
                assert_eq!(block.grouped_batch, 3);
                assert_eq!([block.local_size_x, block.local_size_y], [3, lanes]);
                assert!(block.transforms_on_x);
                assert!(!block.axis_swapped);
            }

            let mapped = recursive
                .forced_rader_three_upload_mapped_components()
                .unwrap()
                .expect("DD N102272 must materialize mapped three-upload components");
            assert_eq!(
                mapped
                    .iter()
                    .map(|component| component.upload_id())
                    .collect::<Vec<_>>(),
                vec![2, 1, 0]
            );
            assert_eq!(
                mapped
                    .iter()
                    .map(|component| component.logical_len())
                    .collect::<Vec<_>>(),
                vec![34, 47, 64]
            );

            let program = crate::ProgramIr::double_double_nd(&nd).unwrap();
            program.validate().unwrap();
            assert!(
                program
                    .passes
                    .iter()
                    .any(|pass| pass.name.contains("forced_rader_three_upload_0"))
            );
            let shaders = crate::backend::vulkan::VulkanGlslBackend
                .lower_double_double_nd(&nd)
                .unwrap();
            assert_eq!(shaders.len(), program.passes.len());
            for shader in &shaders {
                assert_eq!(shader.compile_spirv().unwrap().words[0], 0x0723_0203);
            }
        }
    }

    #[test]
    fn nd_device_aware_higher_axis_large_stockham_uses_upload_schedule() {
        let dimensions = vec![4_116usize, 2];
        let device = DeviceProfile {
            shared_memory_bytes: 48 * 1024,
            shared_memory_pow2_bytes: 32 * 1024,
            max_threads_per_block: 1024,
            max_workgroup_size: [1024, 1024, 64],
            supports_f64: true,
            ..DeviceProfile::generic(crate::Backend::Vulkan, crate::GpuVendor::Nvidia)
        };
        for precision in [Precision::DoubleDouble, Precision::DoubleDoubleF64Storage] {
            let config = FftConfig::new(dimensions.clone())
                .with_precision(precision)
                .resolve_tuning_for_device(device);
            let plan = FftPlan::build_for_device(config.clone(), device).unwrap();
            let portable = DoubleDoubleNdFftIr::build(&plan, Direction::Forward).unwrap();
            let portable_axis = portable.axes.iter().find(|axis| axis.axis == 0).unwrap();
            let DoubleDoubleOneDimIr::Recursive(portable_child) = &portable_axis.transform else {
                panic!("portable higher-axis N4116 child should remain recursive");
            };
            assert!(portable_child.stockham_upload_schedule.is_none());

            let scheduled =
                DoubleDoubleNdFftIr::build_for_device(&plan, Direction::Forward, device)
                    .unwrap()
                    .with_grouped_stockham_axis_blocks(None, device)
                    .unwrap();
            let scheduled_axis = scheduled.axes.iter().find(|axis| axis.axis == 0).unwrap();
            let DoubleDoubleOneDimIr::Recursive(child) = &scheduled_axis.transform else {
                panic!("device-aware higher-axis N4116 child should remain recursive");
            };
            assert_eq!(
                child
                    .stockham_upload_schedule
                    .as_ref()
                    .expect("higher-axis N4116 should retain device upload metadata")
                    .axis_split,
                vec![84, 49]
            );
            let four_step = child
                .two_upload_four_step_plan
                .expect("higher-axis N4116 should retain executable two-upload metadata");
            for block in [four_step.right_axis_block, four_step.left_axis_block] {
                assert!(block.transforms_on_x);
                assert!(!block.axis_swapped);
                assert_eq!(
                    [block.local_size_x, block.local_size_y],
                    [block.grouped_batch, block.threads_per_transform]
                );
            }
            let crate::double_double_recursive_ir::DoubleDoubleRecursiveFftNodeIr::CooleyTukey(
                root,
            ) = &child.root
            else {
                panic!("higher-axis N4116 should retain a two-upload Cooley root");
            };
            let (
                crate::double_double_recursive_ir::DoubleDoubleRecursiveFftNodeIr::Stockham(left),
                crate::double_double_recursive_ir::DoubleDoubleRecursiveFftNodeIr::Stockham(right),
            ) = (&root.left, &root.right)
            else {
                panic!("higher-axis N4116 two-upload root should keep Stockham leaves");
            };
            assert_eq!(left.axis_batch_block, Some(four_step.left_axis_block));
            assert_eq!(right.axis_batch_block, Some(four_step.right_axis_block));
            assert_eq!(
                crate::ProgramIr::double_double_one_dim(&scheduled_axis.transform)
                    .unwrap()
                    .passes
                    .len(),
                2
            );
            assert_eq!(
                scheduled.external_storage,
                double_double_external_storage(precision).unwrap()
            );

            let high_level = crate::TransformIr::build(config, Direction::Forward, device).unwrap();
            let crate::TransformIr::ComplexNdDoubleDouble(high_level) = high_level else {
                panic!("high-level DD N4116x2 should remain ND C2C");
            };
            let high_axis = high_level.axes.iter().find(|axis| axis.axis == 0).unwrap();
            let DoubleDoubleOneDimIr::Recursive(high_child) = &high_axis.transform else {
                panic!("high-level higher-axis N4116 should use recursive device schedule");
            };
            assert_eq!(
                high_child
                    .stockham_upload_schedule
                    .as_ref()
                    .expect("high-level higher-axis N4116 schedule")
                    .axis_split,
                vec![84, 49]
            );
        }
    }

    #[test]
    fn nd_higher_axis_dd_stockham_auto_groups_physical_batches() {
        let device = DeviceProfile {
            shared_memory_bytes: 48 * 1024,
            shared_memory_pow2_bytes: 32 * 1024,
            max_threads_per_block: 1024,
            max_workgroup_size: [1024, 1024, 64],
            supports_f64: true,
            ..DeviceProfile::generic(crate::Backend::Vulkan, crate::GpuVendor::Nvidia)
        };
        let config = FftConfig::new(vec![64usize, 8usize])
            .with_precision(Precision::DoubleDouble)
            .resolve_tuning_for_device(device);
        assert_eq!(config.grouped_batch_for_axis(0), None);
        let transform = crate::TransformIr::build(config, Direction::Forward, device).unwrap();
        let crate::TransformIr::ComplexNdDoubleDouble(nd) = transform else {
            panic!("DD [64,8] should remain ND C2C");
        };
        let outer = nd.axes.iter().find(|axis| axis.axis == 0).unwrap();
        assert_eq!(outer.grouped_batch, 1);
        assert_eq!(outer.grouped_batch_override, None);
        let DoubleDoubleOneDimIr::Stockham(stockham) = &outer.transform else {
            panic!("DD [64,8] higher axis should use Stockham");
        };
        assert_eq!(stockham.grouped_batch, 1);
        let block = stockham
            .axis_batch_block
            .expect("default DD higher-axis Stockham should auto-group physical batches");
        assert_eq!(block.grouped_batch, 8);
        assert_eq!(block.threads_per_transform, 8);
        assert!(block.transforms_on_x);
        assert!(!block.axis_swapped);
        assert_eq!([block.local_size_x, block.local_size_y], [8, 8]);
        assert_eq!(stockham.batch_group_count(), 1);

        let program = crate::ProgramIr::double_double_one_dim(&outer.transform).unwrap();
        assert!(program.passes.iter().all(|pass| pass.dispatch.x == 1));
        let shaders = crate::backend::vulkan::VulkanGlslBackend
            .lower_double_double_stockham_program(stockham)
            .unwrap();
        assert_eq!(shaders.len(), program.passes.len());
        assert!(shaders.iter().all(|shader| shader.dispatch.x == 1));
        assert!(shaders.iter().all(|shader| shader.compile_spirv().is_ok()));
    }

    #[test]
    fn nd_higher_axis_dd_standalone_rader_callers_use_transforms_on_x() {
        let device = DeviceProfile {
            shared_memory_bytes: 48 * 1024,
            shared_memory_pow2_bytes: 32 * 1024,
            max_threads_per_block: 1024,
            max_workgroup_size: [1024, 1024, 64],
            supports_f64: true,
            ..DeviceProfile::generic(crate::Backend::Vulkan, crate::GpuVendor::Nvidia)
        };
        for (length, expected_threads) in [(47usize, 24usize), (257usize, 17usize)] {
            let config = FftConfig::new(vec![length, 8usize])
                .with_precision(Precision::DoubleDouble)
                .with_tuning(crate::PlannerTuning::portable())
                .with_grouped_batch(0, 3)
                .unwrap();
            let transform = crate::TransformIr::build(config, Direction::Forward, device).unwrap();
            let crate::TransformIr::ComplexNdDoubleDouble(nd) = transform else {
                panic!("DD [{length},8] should remain ND C2C");
            };
            let outer = nd.axes.iter().find(|axis| axis.axis == 0).unwrap();
            assert_eq!(outer.grouped_batch, 3);
            let block = match &outer.transform {
                DoubleDoubleOneDimIr::DirectRader(rader) => {
                    assert_eq!(length, 47);
                    rader.axis_batch_block
                }
                DoubleDoubleOneDimIr::FftRader(rader) => {
                    assert_eq!(length, 257);
                    let child_block = rader.forward_fft.stockham_axis_batch_block();
                    assert_eq!(child_block, rader.inverse_fft.stockham_axis_batch_block());
                    rader.caller_axis_batch_block
                }
                _ => panic!("unexpected DD higher-axis standalone Rader kind for N={length}"),
            }
            .expect("DD higher-axis standalone Rader caller should own a physical block");
            assert_eq!(block.threads_per_transform, expected_threads);
            assert_eq!(block.grouped_batch, 3);
            assert!(block.transforms_on_x);
            assert!(!block.axis_swapped);
            assert_eq!(
                [block.local_size_x, block.local_size_y],
                [3, expected_threads]
            );

            let program = crate::ProgramIr::double_double_nd(&nd).unwrap();
            let shaders = crate::backend::vulkan::VulkanGlslBackend
                .lower_double_double_nd(&nd)
                .unwrap();
            assert_eq!(shaders.len(), program.passes.len());
            assert!(shaders.iter().all(|shader| shader.compile_spirv().is_ok()));
        }
    }

    #[test]
    fn nd_higher_axis_dd_rader_auto_groups_physical_batches() {
        let device = DeviceProfile {
            shared_memory_bytes: 48 * 1024,
            shared_memory_pow2_bytes: 32 * 1024,
            max_threads_per_block: 1024,
            max_workgroup_size: [1024, 1024, 64],
            supports_f64: true,
            ..DeviceProfile::generic(crate::Backend::Vulkan, crate::GpuVendor::Nvidia)
        };
        for (length, expected_threads, expected_group) in [
            (47usize, 24usize, 8usize),
            (94usize, 48usize, 4usize),
            (257usize, 17usize, 4usize),
        ] {
            let config = FftConfig::new(vec![length, 8usize])
                .with_precision(Precision::DoubleDouble)
                .with_tuning(crate::PlannerTuning::portable());
            assert_eq!(config.grouped_batch_for_axis(0), None);
            let transform = crate::TransformIr::build(config, Direction::Forward, device).unwrap();
            let crate::TransformIr::ComplexNdDoubleDouble(nd) = transform else {
                panic!("DD [{length},8] should remain ND C2C");
            };
            let outer = nd.axes.iter().find(|axis| axis.axis == 0).unwrap();
            assert_eq!(outer.grouped_batch, 1);
            assert_eq!(outer.grouped_batch_override, None);
            let (block, batch_groups) = match &outer.transform {
                DoubleDoubleOneDimIr::DirectRader(rader) => {
                    assert_eq!(length, 47);
                    (rader.axis_batch_block, rader.batch_group_count())
                }
                DoubleDoubleOneDimIr::FftRader(rader) => {
                    assert_eq!(length, 257);
                    (rader.caller_axis_batch_block, rader.batch_group_count())
                }
                DoubleDoubleOneDimIr::Recursive(recursive) => {
                    assert_eq!(length, 94);
                    let crate::double_double_recursive_ir::DoubleDoubleRecursiveFftNodeIr::CooleyTukey(root) =
                        &recursive.root
                    else {
                        panic!("DD higher-axis N94 should keep a Cooley root");
                    };
                    assert!(matches!(
                        root.right,
                        crate::double_double_recursive_ir::DoubleDoubleRecursiveFftNodeIr::DirectRader(_)
                    ));
                    let block = root.pack_right.axis_batch_block;
                    let batch_groups = block.map_or_else(
                        || {
                            root.pack_right
                                .batch_count
                                .div_ceil(root.pack_right.grouped_batch)
                        },
                        |block| root.pack_right.batch_count.div_ceil(block.grouped_batch),
                    );
                    (block, batch_groups)
                }
                _ => panic!("unexpected DD higher-axis Rader kind for N={length}"),
            };
            let block =
                block.expect("default DD higher-axis Rader should auto-group physical batches");
            assert_eq!(block.threads_per_transform, expected_threads);
            assert_eq!(block.grouped_batch, expected_group);
            assert!(block.transforms_on_x);
            assert!(!block.axis_swapped);
            assert_eq!(
                [block.local_size_x, block.local_size_y],
                [expected_group, expected_threads]
            );
            assert_eq!(batch_groups, 8usize.div_ceil(expected_group));
        }
    }

    #[test]
    fn nd_dd_n192_higher_axis_preserves_vendor_specific_automatic_grouping() {
        for (vendor, expected_grouped) in [
            (crate::GpuVendor::Nvidia, 4usize),
            (crate::GpuVendor::Amd, 8usize),
            (crate::GpuVendor::Intel, 8usize),
        ] {
            let mut device = DeviceProfile::generic(crate::Backend::Vulkan, vendor);
            device.shared_memory_bytes = 48 * 1024;
            device.shared_memory_pow2_bytes = 32 * 1024;
            device.max_threads_per_block = 1024;
            device.max_workgroup_size = [1024, 1024, 64];
            let config = FftConfig::new(vec![192, 8])
                .with_batch_count(5)
                .with_precision(Precision::DoubleDouble);
            let plan = FftPlan::build_for_device(config, device).unwrap();
            let scheduled =
                DoubleDoubleNdFftIr::build_for_device(&plan, Direction::Forward, device)
                    .unwrap()
                    .with_grouped_stockham_axis_blocks(None, device)
                    .unwrap();
            let axis = scheduled.axes.iter().find(|axis| axis.axis == 0).unwrap();
            assert_eq!(axis.axis_len, 192, "{vendor:?}");
            assert_eq!(axis.transform.batch_count(), 40, "{vendor:?}");
            let DoubleDoubleOneDimIr::Stockham(stockham) = &axis.transform else {
                panic!("{vendor:?} DD N192 higher axis should remain Stockham");
            };
            let block = stockham
                .axis_batch_block
                .expect("DD N192 higher axis should retain an automatic physical block");
            assert_eq!(block.threads_per_transform, 32, "{vendor:?}");
            assert_eq!(block.grouped_batch, expected_grouped, "{vendor:?}");
            assert_eq!(
                [block.local_size_x, block.local_size_y],
                [expected_grouped, 32],
                "{vendor:?}"
            );
            assert!(block.transforms_on_x, "{vendor:?}");
            assert!(!block.axis_swapped, "{vendor:?}");
            scheduled.validate().unwrap();
        }
    }

    #[test]
    fn nd_device_aware_higher_axis_composite_rader_uses_forced_upload_split() {
        let dimensions = vec![5_100usize, 2];
        let device = DeviceProfile {
            shared_memory_bytes: 48 * 1024,
            shared_memory_pow2_bytes: 32 * 1024,
            max_threads_per_block: 128,
            max_workgroup_size: [1024, 1024, 64],
            supports_f64: true,
            ..DeviceProfile::generic(crate::Backend::Vulkan, crate::GpuVendor::Nvidia)
        };
        let config = FftConfig::new(dimensions)
            .with_precision(Precision::DoubleDouble)
            .resolve_tuning_for_device(device);
        let plan = FftPlan::build_for_device(config.clone(), device).unwrap();
        let scheduled = DoubleDoubleNdFftIr::build_for_device(&plan, Direction::Forward, device)
            .unwrap()
            .with_grouped_stockham_axis_blocks(None, device)
            .unwrap();
        let axis = scheduled.axes.iter().find(|axis| axis.axis == 0).unwrap();
        let DoubleDoubleOneDimIr::Recursive(child) = &axis.transform else {
            panic!("higher-axis N5100 should use recursive composite-Rader IR");
        };
        assert_eq!(
            child
                .rader_forced_upload_schedule
                .as_ref()
                .expect("higher-axis N5100 should retain forced-Rader split")
                .axis_split,
            vec![68, 75]
        );
        let (mapped_high, _) = child
            .forced_rader_two_upload_mapped_high_stockham()
            .unwrap()
            .expect("higher-axis N5100 mapped high Stockham component");
        let high_block = mapped_high
            .axis_batch_block
            .expect("higher-axis N5100 mapped high Stockham should own a block");
        assert!(high_block.transforms_on_x);
        assert_eq!(
            [high_block.local_size_x, high_block.local_size_y],
            [high_block.grouped_batch, high_block.threads_per_transform]
        );
        let mapped_low = child
            .forced_rader_two_upload_mapped_low_component()
            .unwrap()
            .expect("higher-axis N5100 mapped low component");
        let crate::double_double_recursive_ir::DoubleDoubleRecursiveFftNodeIr::CooleyTukey(
            mapped_low,
        ) = mapped_low
        else {
            panic!("higher-axis N5100 mapped low component should remain Cooley-Tukey");
        };
        for pass in [
            &mapped_low.pack_right,
            &mapped_low.twiddle_transpose,
            &mapped_low.scatter_output,
        ] {
            let block = pass
                .axis_batch_block
                .expect("higher-axis N5100 mapped low Cooley pass should own a block");
            assert!(block.transforms_on_x);
            assert_eq!(
                [
                    pass.workgroup_size.x as usize,
                    pass.workgroup_size.y as usize
                ],
                [block.grouped_batch, block.threads_per_transform]
            );
        }
        let program = crate::ProgramIr::double_double_recursive(child).unwrap();
        let shaders = crate::backend::vulkan::VulkanGlslBackend
            .lower_double_double_recursive(child)
            .unwrap();
        assert_eq!(shaders.len(), program.passes.len());
        assert!(shaders.iter().all(|shader| shader.compile_spirv().is_ok()));

        let high_level = crate::TransformIr::build(config, Direction::Forward, device).unwrap();
        let crate::TransformIr::ComplexNdDoubleDouble(high_level) = high_level else {
            panic!("high-level DD N5100x2 should remain ND C2C");
        };
        let high_axis = high_level.axes.iter().find(|axis| axis.axis == 0).unwrap();
        let DoubleDoubleOneDimIr::Recursive(high_child) = &high_axis.transform else {
            panic!("high-level higher-axis N5100 should use recursive device schedule");
        };
        assert_eq!(
            high_child
                .rader_forced_upload_schedule
                .as_ref()
                .expect("high-level higher-axis N5100 forced split")
                .axis_split,
            vec![68, 75]
        );
    }

    #[test]
    fn nd_intel_strided_dd_bluestein_child_uses_axis_context_and_bandwidth_boost() {
        let device = DeviceProfile {
            shared_memory_bytes: 32 * 1024,
            shared_memory_pow2_bytes: 32 * 1024,
            max_threads_per_block: 1024,
            max_workgroup_size: [1024, 1024, 64],
            supports_f64: true,
            ..DeviceProfile::generic(crate::Backend::Vulkan, crate::GpuVendor::Intel)
        };

        let standalone_config = FftConfig::new(vec![263usize])
            .with_precision(Precision::DoubleDouble)
            .resolve_tuning_for_device(device);
        let standalone_plan = FftPlan::build_for_device(standalone_config, device).unwrap();
        let standalone =
            DoubleDoubleOneDimIr::build_for_device(&standalone_plan, Direction::Forward, device)
                .unwrap();
        let DoubleDoubleOneDimIr::Bluestein(standalone) = standalone else {
            panic!("Intel DD p263 should use Bluestein");
        };
        assert_eq!(standalone.convolution_len, 625);
        assert!(matches!(
            standalone.forward_fft,
            DoubleDoubleBluesteinConvolutionIr::Stockham(_)
        ));

        let nd_config = FftConfig::new(vec![263usize, 2usize])
            .with_precision(Precision::DoubleDouble)
            .resolve_tuning_for_device(device);
        let nd_plan = FftPlan::build_for_device(nd_config.clone(), device).unwrap();
        let probe_plan = FftPlan::build_c2c_bluestein_child_for_device(
            FftConfig::new(vec![625usize])
                .with_batch_count(2)
                .with_precision(Precision::DoubleDouble)
                .with_tuning(nd_plan.config.tuning),
            device,
            C2cDeviceAxisClass::Strided,
        )
        .unwrap();
        assert_eq!(
            probe_plan.c2c_device_axis_class_override,
            Some(C2cDeviceAxisClass::Strided)
        );
        assert!(probe_plan.c2c_device_use_bluestein_fft_override);
        assert!(double_double_stockham_requires_multi_upload(&probe_plan, 625, device).unwrap());
        assert!(matches!(
            DoubleDoubleBluesteinConvolutionIr::build_for_device(
                &probe_plan,
                Direction::Forward,
                device
            )
            .unwrap(),
            DoubleDoubleBluesteinConvolutionIr::Recursive(_)
        ));
        let nd =
            DoubleDoubleNdFftIr::build_for_device(&nd_plan, Direction::Forward, device).unwrap();
        let outer = nd.axes.iter().find(|axis| axis.axis == 0).unwrap();
        let DoubleDoubleOneDimIr::Bluestein(bluestein) = &outer.transform else {
            panic!("Intel DD ND p263 outer axis should use Bluestein");
        };
        assert_eq!(bluestein.convolution_len, 625);
        let DoubleDoubleBluesteinConvolutionIr::Recursive(forward_child) = &bluestein.forward_fft
        else {
            panic!("strided Intel DD M625 Bluestein child should require two uploads");
        };
        assert_eq!(
            forward_child
                .stockham_upload_schedule
                .as_ref()
                .expect("strided Intel DD M625 upload schedule")
                .axis_split,
            vec![25, 25]
        );
        assert!(forward_child.two_upload_four_step_plan.is_some());
        let program = crate::ProgramIr::double_double_nd(&nd).unwrap();
        program.validate().unwrap();
        let shaders = crate::backend::vulkan::VulkanGlslBackend
            .lower_double_double_nd(&nd)
            .unwrap();
        assert_eq!(shaders.len(), program.passes.len());
        for shader in &shaders {
            assert_eq!(shader.compile_spirv().unwrap().words[0], 0x0723_0203);
        }
        let mut input = vec![ComplexDoubleDouble::default(); 263 * 2];
        input[0] = ComplexDoubleDouble::new(DoubleDouble::ONE, DoubleDouble::ZERO);
        let actual = execute_double_double_nd_ir(&nd, &input).unwrap();
        let expected = ComplexDoubleDouble::new(DoubleDouble::ONE, DoubleDouble::ZERO);
        let impulse_error = actual
            .iter()
            .copied()
            .map(|value| dd_error(value, expected))
            .fold(0.0f64, f64::max);
        assert!(
            impulse_error < 5.0e-19,
            "Intel DD ND p263/M625 strided Bluestein impulse error {impulse_error:e}"
        );

        let boosted_config = FftConfig::new(vec![263usize, 2usize])
            .with_precision(Precision::DoubleDouble)
            .with_bandwidth_boost(2)
            .resolve_tuning_for_device(device);
        let boosted_plan = FftPlan::build_for_device(boosted_config, device).unwrap();
        let boosted =
            DoubleDoubleNdFftIr::build_for_device(&boosted_plan, Direction::Forward, device)
                .unwrap();
        let boosted_outer = boosted.axes.iter().find(|axis| axis.axis == 0).unwrap();
        let DoubleDoubleOneDimIr::Bluestein(boosted_bluestein) = &boosted_outer.transform else {
            panic!("boosted Intel DD ND p263 outer axis should keep Bluestein");
        };
        assert_eq!(boosted_bluestein.convolution_len, 625);
        assert!(matches!(
            boosted_bluestein.forward_fft,
            DoubleDoubleBluesteinConvolutionIr::Stockham(_)
        ));
        boosted.validate().unwrap();
    }

    #[test]
    fn nd_higher_axis_dd_bluestein_wrapper_and_stockham_child_use_transforms_on_x() {
        let device = DeviceProfile {
            shared_memory_bytes: 48 * 1024,
            shared_memory_pow2_bytes: 32 * 1024,
            max_threads_per_block: 1024,
            max_workgroup_size: [1024, 1024, 64],
            supports_f64: true,
            ..DeviceProfile::generic(crate::Backend::Vulkan, crate::GpuVendor::Nvidia)
        };
        let config = FftConfig::new(vec![103usize, 8usize])
            .with_precision(Precision::DoubleDouble)
            .with_tuning(crate::PlannerTuning::portable())
            .with_grouped_batch(0, 3)
            .unwrap();
        let transform = crate::TransformIr::build(config, Direction::Forward, device).unwrap();
        let crate::TransformIr::ComplexNdDoubleDouble(nd) = transform else {
            panic!("DD [103,8] should remain ND C2C");
        };
        let outer = nd.axes.iter().find(|axis| axis.axis == 0).unwrap();
        let DoubleDoubleOneDimIr::Bluestein(bluestein) = &outer.transform else {
            panic!("DD higher-axis p103 should use Bluestein");
        };
        assert_eq!(bluestein.convolution_len, 256);
        let wrapper = bluestein
            .wrapper_axis_batch_block()
            .expect("higher-axis DD Bluestein wrapper should own a physical block");
        assert_eq!(wrapper.grouped_batch, 3);
        assert_eq!(wrapper.threads_per_transform, 128);
        assert!(wrapper.transforms_on_x);
        assert_eq!([wrapper.local_size_x, wrapper.local_size_y], [3, 128]);

        let child = bluestein
            .stockham_convolution_axis_batch_block()
            .expect("higher-axis DD M256 Stockham child should own a physical block");
        assert_eq!(child.grouped_batch, 3);
        assert_eq!(child.threads_per_transform, 32);
        assert!(child.transforms_on_x);
        assert_eq!([child.local_size_x, child.local_size_y], [3, 32]);

        let program = crate::ProgramIr::double_double_nd(&nd).unwrap();
        let shaders = crate::backend::vulkan::VulkanGlslBackend
            .lower_double_double_nd(&nd)
            .unwrap();
        assert_eq!(shaders.len(), program.passes.len());
        assert!(shaders.iter().all(|shader| shader.compile_spirv().is_ok()));
    }

    #[test]
    fn nd_higher_axis_dd_bluestein_auto_groups_wrapper_and_child() {
        let device = DeviceProfile {
            shared_memory_bytes: 48 * 1024,
            shared_memory_pow2_bytes: 32 * 1024,
            max_threads_per_block: 1024,
            max_workgroup_size: [1024, 1024, 64],
            supports_f64: true,
            ..DeviceProfile::generic(crate::Backend::Vulkan, crate::GpuVendor::Nvidia)
        };
        let tuning = crate::PlannerTuning {
            min_rader_direct_prime: 17,
            max_rader_direct_prime: 17,
            min_rader_fft_prime: 17,
            max_rader_fft_prime: 1024,
            allow_recursive_fft_rader: false,
        };
        let config = FftConfig::new(vec![11usize, 8usize])
            .with_precision(Precision::DoubleDouble)
            .with_tuning(tuning);
        assert_eq!(config.grouped_batch_for_axis(0), None);
        let transform = crate::TransformIr::build(config, Direction::Forward, device).unwrap();
        let crate::TransformIr::ComplexNdDoubleDouble(nd) = transform else {
            panic!("forced-Bluestein DD [11,8] should remain ND C2C");
        };
        let outer = nd.axes.iter().find(|axis| axis.axis == 0).unwrap();
        assert_eq!(outer.grouped_batch, 1);
        assert_eq!(outer.grouped_batch_override, None);
        let DoubleDoubleOneDimIr::Bluestein(bluestein) = &outer.transform else {
            panic!("forced DD higher-axis p11 should use Bluestein");
        };
        assert_eq!(bluestein.grouped_batch, 1);
        assert_eq!(bluestein.convolution_len, 32);
        let wrapper = bluestein
            .wrapper_axis_batch_block()
            .expect("default higher-axis DD Bluestein wrapper should auto-group");
        assert_eq!(wrapper.grouped_batch, 4);
        assert_eq!(wrapper.threads_per_transform, 32);
        assert!(wrapper.transforms_on_x);
        assert_eq!([wrapper.local_size_x, wrapper.local_size_y], [4, 32]);
        assert_eq!(bluestein.batch_group_count(), 2);

        let child = bluestein
            .stockham_convolution_axis_batch_block()
            .expect("default higher-axis DD M32 child should auto-group independently");
        assert!(child.grouped_batch > 1);
        assert!(child.transforms_on_x);
        assert_eq!(
            [child.local_size_x, child.local_size_y],
            [child.grouped_batch, child.threads_per_transform]
        );

        let program = crate::ProgramIr::double_double_nd(&nd).unwrap();
        let wrapper_dispatches = program
            .passes
            .iter()
            .filter(|pass| {
                pass.name.contains("bluestein")
                    && (pass.name.contains("preprocess")
                        || pass.name.contains("multiply")
                        || pass.name.contains("postprocess"))
            })
            .map(|pass| pass.dispatch.x)
            .collect::<Vec<_>>();
        assert!(wrapper_dispatches.iter().all(|dispatch| *dispatch == 2));
        let shaders = crate::backend::vulkan::VulkanGlslBackend
            .lower_double_double_nd(&nd)
            .unwrap();
        assert_eq!(shaders.len(), program.passes.len());
        assert!(shaders.iter().all(|shader| shader.compile_spirv().is_ok()));
    }

    #[test]
    fn nd_device_aware_higher_axis_bluestein_installs_wrapper_block() {
        fn contains_direct_p13(
            node: &crate::double_double_recursive_ir::DoubleDoubleRecursiveFftNodeIr,
        ) -> bool {
            use crate::double_double_recursive_ir::DoubleDoubleRecursiveFftNodeIr;
            match node {
                DoubleDoubleRecursiveFftNodeIr::DirectRader(rader) => rader.prime == 13,
                DoubleDoubleRecursiveFftNodeIr::CooleyTukey(cooley) => {
                    contains_direct_p13(&cooley.left) || contains_direct_p13(&cooley.right)
                }
                DoubleDoubleRecursiveFftNodeIr::Stockham(_)
                | DoubleDoubleRecursiveFftNodeIr::FftRader(_)
                | DoubleDoubleRecursiveFftNodeIr::Bluestein(_) => false,
            }
        }

        fn assert_higher_axis_parent(
            node: &crate::double_double_recursive_ir::DoubleDoubleRecursiveFftNodeIr,
        ) {
            let crate::double_double_recursive_ir::DoubleDoubleRecursiveFftNodeIr::CooleyTukey(
                cooley,
            ) = node
            else {
                panic!("higher-axis M4368 recursive root should remain Cooley-Tukey");
            };
            let block = cooley
                .pack_right
                .axis_batch_block
                .expect("higher-axis M4368 Cooley parent should own a physical block");
            assert!(block.transforms_on_x);
            assert!(!block.axis_swapped);
            assert_eq!(cooley.twiddle_transpose.axis_batch_block, Some(block));
            assert_eq!(cooley.scatter_output.axis_batch_block, Some(block));
            assert_eq!(
                [block.local_size_x, block.local_size_y],
                [block.grouped_batch, block.threads_per_transform]
            );
        }

        let dimensions = vec![2_053usize, 2];
        let device = DeviceProfile {
            shared_memory_bytes: 48 * 1024,
            shared_memory_pow2_bytes: 32 * 1024,
            max_threads_per_block: 1024,
            max_workgroup_size: [1024, 1024, 64],
            supports_f64: true,
            ..DeviceProfile::generic(crate::Backend::Vulkan, crate::GpuVendor::Nvidia)
        };
        let config = FftConfig::new(dimensions)
            .with_precision(Precision::DoubleDouble)
            .resolve_tuning_for_device(device);
        let plan = FftPlan::build_for_device(config.clone(), device).unwrap();
        let portable = DoubleDoubleNdFftIr::build(&plan, Direction::Forward).unwrap();
        let portable_axis = portable.axes.iter().find(|axis| axis.axis == 0).unwrap();
        let DoubleDoubleOneDimIr::Bluestein(portable_bluestein) = &portable_axis.transform else {
            panic!("portable higher-axis p2053 should remain Bluestein");
        };
        assert!(portable_bluestein.wrapper_axis_batch_block().is_none());

        let scheduled = DoubleDoubleNdFftIr::build_for_device(&plan, Direction::Forward, device)
            .unwrap()
            .with_grouped_stockham_axis_blocks(None, device)
            .unwrap();
        let axis = scheduled.axes.iter().find(|axis| axis.axis == 0).unwrap();
        let DoubleDoubleOneDimIr::Bluestein(bluestein) = &axis.transform else {
            panic!("device-aware higher-axis p2053 should remain Bluestein");
        };
        assert_eq!(bluestein.logical_len, 2_053);
        assert_eq!(bluestein.convolution_len, 4_368);
        let wrapper = bluestein
            .wrapper_axis_batch_block()
            .expect("device-aware higher-axis p2053 should own a wrapper block");
        assert!(wrapper.threads_per_transform > 0 && wrapper.grouped_batch > 0);
        let DoubleDoubleBluesteinConvolutionIr::Recursive(forward_child) = &bluestein.forward_fft
        else {
            panic!("device-padded M4368 Bluestein convolution should be recursive");
        };
        assert!(
            forward_child.stockham_upload_schedule.is_none(),
            "device-padded M4368 is mixed Stockham + p13 Rader, not a pure Stockham upload tree"
        );
        assert!(
            contains_direct_p13(&forward_child.root),
            "device-padded M4368 must preserve the DD p13 direct-Rader leaf"
        );
        assert_higher_axis_parent(&forward_child.root);
        let DoubleDoubleBluesteinConvolutionIr::Recursive(inverse_child) = &bluestein.inverse_fft
        else {
            panic!("inverse device-padded M4368 Bluestein convolution should be recursive");
        };
        assert!(contains_direct_p13(&inverse_child.root));
        assert_higher_axis_parent(&inverse_child.root);
        let program = crate::ProgramIr::double_double_nd(&scheduled).unwrap();
        program.validate().unwrap();
        let shaders = crate::backend::vulkan::VulkanGlslBackend
            .lower_double_double_nd(&scheduled)
            .unwrap();
        assert_eq!(shaders.len(), program.passes.len());
        for shader in &shaders {
            assert_eq!(shader.compile_spirv().unwrap().words[0], 0x0723_0203);
        }

        let high_level = crate::TransformIr::build(config, Direction::Forward, device).unwrap();
        let crate::TransformIr::ComplexNdDoubleDouble(high_level) = high_level else {
            panic!("high-level DD p2053x2 should remain ND C2C");
        };
        let high_axis = high_level.axes.iter().find(|axis| axis.axis == 0).unwrap();
        let DoubleDoubleOneDimIr::Bluestein(high_bluestein) = &high_axis.transform else {
            panic!("high-level higher-axis p2053 should use Bluestein");
        };
        assert!(high_bluestein.wrapper_axis_batch_block().is_some());
        assert_eq!(high_bluestein.convolution_len, 4_368);
        let DoubleDoubleBluesteinConvolutionIr::Recursive(high_child) = &high_bluestein.forward_fft
        else {
            panic!("high-level device-padded M4368 child should be recursive");
        };
        assert!(contains_direct_p13(&high_child.root));
        assert_higher_axis_parent(&high_child.root);
    }

    #[test]
    fn nd_grouped_padded_p2053_bluestein_recursive_child_keeps_upstream_upload_split() {
        fn contains_direct_prime(
            node: &crate::double_double_recursive_ir::DoubleDoubleRecursiveFftNodeIr,
            prime: usize,
        ) -> bool {
            use crate::double_double_recursive_ir::DoubleDoubleRecursiveFftNodeIr;
            match node {
                DoubleDoubleRecursiveFftNodeIr::DirectRader(rader) => rader.prime == prime,
                DoubleDoubleRecursiveFftNodeIr::CooleyTukey(cooley) => {
                    contains_direct_prime(&cooley.left, prime)
                        || contains_direct_prime(&cooley.right, prime)
                }
                _ => false,
            }
        }

        for (vendor, expected_m, expected_split, direct_prime, expected_blocks) in [
            (
                crate::GpuVendor::Nvidia,
                4_368usize,
                vec![78usize, 56usize],
                13usize,
                [(2_240usize, 42usize), (3_120usize, 8usize)],
            ),
            (
                crate::GpuVendor::Amd,
                4_224usize,
                vec![66usize, 64usize],
                11usize,
                [(2_560usize, 36usize), (2_640usize, 8usize)],
            ),
        ] {
            let device = DeviceProfile {
                shared_memory_bytes: 32 * 1024,
                shared_memory_pow2_bytes: 32 * 1024,
                max_threads_per_block: 1024,
                max_workgroup_size: [1024, 1024, 64],
                supports_f64: true,
                ..DeviceProfile::generic(crate::Backend::Vulkan, vendor)
            };
            for bandwidth_boost in [0usize, 2] {
                let config = FftConfig::new(vec![2_053usize, 8])
                    .with_batch_count(5)
                    .with_grouped_batch(0, 3)
                    .unwrap()
                    .with_bandwidth_boost(bandwidth_boost)
                    .with_zero_padding(0, 1, 2)
                    .unwrap()
                    .with_precision(Precision::DoubleDouble)
                    .resolve_tuning_for_device(device);
                let transform =
                    crate::TransformIr::build(config, Direction::Forward, device).unwrap();
                let crate::TransformIr::ComplexNdDoubleDouble(nd) = transform else {
                    panic!("grouped padded DD p2053 probe did not build ND C2C");
                };
                let outer = nd.axes.iter().find(|axis| axis.axis == 0).unwrap();
                let DoubleDoubleOneDimIr::Bluestein(bluestein) = &outer.transform else {
                    panic!("grouped padded DD p2053 higher axis did not select Bluestein");
                };
                assert_eq!(bluestein.convolution_len, expected_m);
                assert_eq!(bluestein.batch_count, 40);
                assert_eq!(bluestein.grouped_batch, 3);
                let wrapper = bluestein
                    .wrapper_axis_batch_block()
                    .expect("grouped padded DD p2053 wrapper block");
                assert!(wrapper.grouped_batch > 0);
                assert!(wrapper.grouped_batch <= bluestein.grouped_batch);
                assert!(wrapper.transforms_on_x);
                assert!(!wrapper.axis_swapped);

                for child in [&bluestein.forward_fft, &bluestein.inverse_fft] {
                    let DoubleDoubleBluesteinConvolutionIr::Recursive(child) = child else {
                        panic!("grouped padded DD p2053 convolution child must be recursive");
                    };
                    assert_eq!(child.logical_len, expected_m);
                    assert_eq!(child.batch_count, 40);
                    assert_eq!(child.grouped_batch, 3);
                    assert!(contains_direct_prime(&child.root, direct_prime));
                    assert_eq!(
                        child
                            .rader_forced_upload_schedule
                            .as_ref()
                            .map(|schedule| schedule.axis_split.clone()),
                        Some(expected_split.clone()),
                        "{vendor:?} B{bandwidth_boost} recursive child lost pinned-upstream upload split"
                    );
                    let crate::double_double_recursive_ir::DoubleDoubleRecursiveFftNodeIr::CooleyTukey(root) =
                        &child.root
                    else {
                        panic!("grouped padded DD p2053 child must retain two-upload Cooley root");
                    };
                    for (upload_id, node) in [&root.left, &root.right].into_iter().enumerate() {
                        let (expected_batch, expected_threads) = expected_blocks[upload_id];
                        assert_eq!(node.logical_len(), expected_split[upload_id]);
                        assert_eq!(node.batch_count(), expected_batch);
                        let block = match node {
                            crate::double_double_recursive_ir::DoubleDoubleRecursiveFftNodeIr::CooleyTukey(cooley) => {
                                cooley.pack_right.axis_batch_block
                            }
                            crate::double_double_recursive_ir::DoubleDoubleRecursiveFftNodeIr::Stockham(stockham) => {
                                stockham.axis_batch_block
                            }
                            _ => None,
                        }
                        .expect("grouped padded DD p2053 recursive component block");
                        assert_eq!(block.threads_per_transform, expected_threads);
                        assert_eq!(block.grouped_batch, 3);
                        assert_eq!(
                            [block.local_size_x, block.local_size_y],
                            [3, expected_threads]
                        );
                        assert!(block.transforms_on_x);
                        assert!(!block.axis_swapped);
                    }
                }
                crate::ProgramIr::double_double_nd(&nd)
                    .unwrap()
                    .validate()
                    .unwrap();
            }
        }
    }

    #[test]
    fn nd_grouped_padded_p2503_bluestein_cross_vendor_child_topology_matches_upstream() {
        fn contains_direct_prime(
            node: &crate::double_double_recursive_ir::DoubleDoubleRecursiveFftNodeIr,
            prime: usize,
        ) -> bool {
            use crate::double_double_recursive_ir::DoubleDoubleRecursiveFftNodeIr;
            match node {
                DoubleDoubleRecursiveFftNodeIr::DirectRader(rader) => rader.prime == prime,
                DoubleDoubleRecursiveFftNodeIr::CooleyTukey(cooley) => {
                    contains_direct_prime(&cooley.left, prime)
                        || contains_direct_prime(&cooley.right, prime)
                }
                _ => false,
            }
        }

        for (
            vendor,
            expected_m,
            expected_split,
            expected_blocks,
            expected_direct_primes,
            expect_rader_schedule,
        ) in [
            (
                crate::GpuVendor::Nvidia,
                5_184usize,
                [72usize, 72usize],
                [(2_880usize, 12usize), (2_880usize, 12usize)],
                [0usize, 0usize],
                false,
            ),
            (
                crate::GpuVendor::Amd,
                5_005usize,
                [77usize, 65usize],
                [(2_600usize, 42usize), (3_080usize, 35usize)],
                [11usize, 13usize],
                true,
            ),
        ] {
            let device = DeviceProfile {
                shared_memory_bytes: 32 * 1024,
                shared_memory_pow2_bytes: 32 * 1024,
                max_threads_per_block: 1024,
                max_workgroup_size: [1024, 1024, 64],
                supports_f64: true,
                ..DeviceProfile::generic(crate::Backend::Vulkan, vendor)
            };
            for bandwidth_boost in [0usize, 2] {
                let config = FftConfig::new(vec![2_503usize, 8])
                    .with_batch_count(5)
                    .with_grouped_batch(0, 3)
                    .unwrap()
                    .with_bandwidth_boost(bandwidth_boost)
                    .with_zero_padding(0, 1, 2)
                    .unwrap()
                    .with_precision(Precision::DoubleDouble)
                    .resolve_tuning_for_device(device);
                let transform =
                    crate::TransformIr::build(config, Direction::Forward, device).unwrap();
                let crate::TransformIr::ComplexNdDoubleDouble(nd) = transform else {
                    panic!("grouped padded DD p2503 probe did not build ND C2C");
                };
                let outer = nd.axes.iter().find(|axis| axis.axis == 0).unwrap();
                let DoubleDoubleOneDimIr::Bluestein(bluestein) = &outer.transform else {
                    panic!("grouped padded DD p2503 higher axis did not select Bluestein");
                };
                assert_eq!(bluestein.convolution_len, expected_m);
                assert_eq!(bluestein.batch_count, 40);
                assert_eq!(bluestein.grouped_batch, 3);
                let wrapper = bluestein
                    .wrapper_axis_batch_block()
                    .expect("grouped padded DD p2503 wrapper block");
                assert!(wrapper.grouped_batch > 0);
                assert!(wrapper.grouped_batch <= bluestein.grouped_batch);
                assert!(wrapper.transforms_on_x);
                assert!(!wrapper.axis_swapped);

                for child in [&bluestein.forward_fft, &bluestein.inverse_fft] {
                    let DoubleDoubleBluesteinConvolutionIr::Recursive(child) = child else {
                        panic!("grouped padded DD p2503 convolution child must be recursive");
                    };
                    assert_eq!(child.logical_len, expected_m);
                    assert_eq!(child.batch_count, 40);
                    assert_eq!(child.grouped_batch, 3);
                    let actual_split = if expect_rader_schedule {
                        assert!(child.stockham_upload_schedule.is_none());
                        child
                            .rader_forced_upload_schedule
                            .as_ref()
                            .expect("AMD p2503/M5005 child must retain forced-Rader uploads")
                            .axis_split
                            .clone()
                    } else {
                        assert!(child.rader_forced_upload_schedule.is_none());
                        child
                            .stockham_upload_schedule
                            .as_ref()
                            .expect("NVIDIA p2503/M5184 child must retain Stockham uploads")
                            .axis_split
                            .clone()
                    };
                    assert_eq!(actual_split, expected_split);
                    let crate::double_double_recursive_ir::DoubleDoubleRecursiveFftNodeIr::CooleyTukey(root) =
                        &child.root
                    else {
                        panic!("grouped padded DD p2503 child must retain two-upload Cooley root");
                    };
                    for (upload_id, node) in [&root.left, &root.right].into_iter().enumerate() {
                        let (expected_batch, expected_threads) = expected_blocks[upload_id];
                        assert_eq!(node.logical_len(), expected_split[upload_id]);
                        assert_eq!(node.batch_count(), expected_batch);
                        if expected_direct_primes[upload_id] != 0 {
                            assert!(contains_direct_prime(
                                node,
                                expected_direct_primes[upload_id]
                            ));
                        }
                        let block = match node {
                            crate::double_double_recursive_ir::DoubleDoubleRecursiveFftNodeIr::CooleyTukey(cooley) => {
                                cooley.pack_right.axis_batch_block
                            }
                            crate::double_double_recursive_ir::DoubleDoubleRecursiveFftNodeIr::Stockham(stockham) => {
                                stockham.axis_batch_block
                            }
                            _ => None,
                        }
                        .expect("grouped padded DD p2503 recursive component block");
                        assert_eq!(block.threads_per_transform, expected_threads);
                        assert_eq!(block.grouped_batch, 3);
                        assert_eq!(
                            [block.local_size_x, block.local_size_y],
                            [3, expected_threads]
                        );
                        assert!(block.transforms_on_x);
                        assert!(!block.axis_swapped);
                    }
                }
                crate::ProgramIr::double_double_nd(&nd)
                    .unwrap()
                    .validate()
                    .unwrap();
            }
        }
    }

    #[test]
    fn nd_grouped_padded_n391_preserves_nested_fft_rader_bluestein_tree() {
        fn find_fft_rader(
            node: &crate::double_double_recursive_ir::DoubleDoubleRecursiveFftNodeIr,
            prime: usize,
        ) -> Option<&DoubleDoubleFftRaderIr> {
            use crate::double_double_recursive_ir::DoubleDoubleRecursiveFftNodeIr;
            match node {
                DoubleDoubleRecursiveFftNodeIr::FftRader(rader) if rader.prime == prime => {
                    Some(rader)
                }
                DoubleDoubleRecursiveFftNodeIr::CooleyTukey(cooley) => {
                    find_fft_rader(&cooley.left, prime)
                        .or_else(|| find_fft_rader(&cooley.right, prime))
                }
                _ => None,
            }
        }

        let mut tuning = PlannerTuning::portable();
        tuning.min_rader_direct_prime = 17;
        tuning.min_rader_fft_prime = 17;
        tuning.validate().unwrap();
        for vendor in [crate::GpuVendor::Nvidia, crate::GpuVendor::Amd] {
            let device = DeviceProfile {
                shared_memory_bytes: 32 * 1024,
                shared_memory_pow2_bytes: 32 * 1024,
                max_threads_per_block: 1024,
                max_workgroup_size: [1024, 1024, 64],
                supports_f64: true,
                ..DeviceProfile::generic(crate::Backend::Vulkan, vendor)
            };
            let config = FftConfig::new(vec![391usize, 8])
                .with_batch_count(5)
                .with_grouped_batch(0, 3)
                .unwrap()
                .with_zero_padding(0, 1, 2)
                .unwrap()
                .with_precision(Precision::DoubleDouble)
                .with_tuning(tuning);
            let transform = crate::TransformIr::build(config, Direction::Forward, device).unwrap();
            let crate::TransformIr::ComplexNdDoubleDouble(nd) = transform else {
                panic!("grouped padded DD N391 probe did not build ND C2C");
            };
            let outer = nd.axes.iter().find(|axis| axis.axis == 0).unwrap();
            let DoubleDoubleOneDimIr::Recursive(recursive) = &outer.transform else {
                panic!("grouped padded DD N391 must retain recursive Rader IR");
            };
            assert_eq!(recursive.logical_len, 391);
            assert_eq!(recursive.batch_count, 40);
            assert_eq!(recursive.grouped_batch, 3);
            assert!(recursive.rader_forced_upload_schedule.is_none());
            assert!(recursive.stockham_upload_schedule.is_none());
            let crate::double_double_recursive_ir::DoubleDoubleRecursiveFftNodeIr::CooleyTukey(
                root,
            ) = &recursive.root
            else {
                panic!("DD N391 nested-Rader root must remain Cooley-Tukey");
            };
            let block = root
                .pack_right
                .axis_batch_block
                .expect("DD N391 nested-Rader root physical block");
            assert_eq!(block.threads_per_transform, 25);
            assert_eq!(block.grouped_batch, 2);
            assert_eq!([block.local_size_x, block.local_size_y], [2, 25]);
            assert!(block.transforms_on_x);
            assert!(!block.axis_swapped);
            assert_eq!(root.twiddle_transpose.axis_batch_block, Some(block));
            assert_eq!(root.scatter_output.axis_batch_block, Some(block));

            let p17 =
                find_fft_rader(&recursive.root, 17).expect("DD N391 must retain p17 FFT-Rader");
            let p23 =
                find_fft_rader(&recursive.root, 23).expect("DD N391 must retain p23 FFT-Rader");
            assert_eq!(p17.convolution_len, 16);
            assert_eq!(p23.convolution_len, 22);
            for child in [&p23.forward_fft, &p23.inverse_fft] {
                let DoubleDoubleBluesteinConvolutionIr::Bluestein(nested) = child else {
                    panic!("DD N391 p23 convolution must retain nested Bluestein");
                };
                assert_eq!(nested.logical_len, 22);
                assert_eq!(nested.convolution_len, 64);
                assert_eq!(nested.batch_count, 680);
                assert_eq!(nested.grouped_batch, 51);
            }

            let program = crate::ProgramIr::double_double_nd(&nd).unwrap();
            program.validate().unwrap();
            let shaders = crate::backend::vulkan::VulkanGlslBackend
                .lower_double_double_nd(&nd)
                .unwrap();
            assert_eq!(shaders.len(), program.passes.len());
            for shader in &shaders {
                assert_eq!(shader.compile_spirv().unwrap().words[0], 0x0723_0203);
            }
        }
    }

    #[test]
    fn nd_real_grouped_padded_n391_compaction_preserves_nested_fft_rader_bluestein_tree() {
        fn find_fft_rader(
            node: &crate::double_double_recursive_ir::DoubleDoubleRecursiveFftNodeIr,
            prime: usize,
        ) -> Option<&DoubleDoubleFftRaderIr> {
            use crate::double_double_recursive_ir::DoubleDoubleRecursiveFftNodeIr;
            match node {
                DoubleDoubleRecursiveFftNodeIr::FftRader(rader) if rader.prime == prime => {
                    Some(rader)
                }
                DoubleDoubleRecursiveFftNodeIr::CooleyTukey(cooley) => {
                    find_fft_rader(&cooley.left, prime)
                        .or_else(|| find_fft_rader(&cooley.right, prime))
                }
                _ => None,
            }
        }

        let mut tuning = PlannerTuning::portable();
        tuning.min_rader_direct_prime = 17;
        tuning.min_rader_fft_prime = 17;
        tuning.validate().unwrap();
        for (raw_fastest_axis_len, compact_fastest_axis_len, expected_batch, nested_batch) in [
            (14usize, 8usize, 40usize, 680usize),
            (126usize, 64usize, 320usize, 5_440usize),
        ] {
            for vendor in [crate::GpuVendor::Nvidia, crate::GpuVendor::Amd] {
                let device = DeviceProfile {
                    shared_memory_bytes: 32 * 1024,
                    shared_memory_pow2_bytes: 32 * 1024,
                    max_threads_per_block: 1024,
                    max_workgroup_size: [1024, 1024, 64],
                    supports_f64: true,
                    ..DeviceProfile::generic(crate::Backend::Vulkan, vendor)
                };
                let config = FftConfig::new(vec![391usize, raw_fastest_axis_len])
                    .with_batch_count(5)
                    .with_grouped_batch(0, 3)
                    .unwrap()
                    .with_zero_padding(0, 1, 2)
                    .unwrap()
                    .with_transform(TransformKind::RealToComplex)
                    .with_precision(Precision::DoubleDouble)
                    .with_tuning(tuning);
                let transform =
                    crate::TransformIr::build(config, Direction::Forward, device).unwrap();
                let crate::TransformIr::RealNdDoubleDouble(nd) = transform else {
                    panic!("grouped padded DD N391 ND-real probe did not build ND R2C");
                };
                assert_eq!(nd.compact_dimensions, vec![391, compact_fastest_axis_len]);
                let outer = nd.complex_axes.iter().find(|axis| axis.axis == 0).unwrap();
                assert_eq!(outer.line_count, compact_fastest_axis_len);
                let DoubleDoubleOneDimIr::Recursive(recursive) = &outer.transform else {
                    panic!("grouped padded DD N391 ND-real must retain recursive Rader IR");
                };
                assert_eq!(recursive.logical_len, 391);
                assert_eq!(recursive.batch_count, expected_batch);
                assert_eq!(recursive.grouped_batch, 3);
                assert!(recursive.rader_forced_upload_schedule.is_none());
                assert!(recursive.stockham_upload_schedule.is_none());
                let crate::double_double_recursive_ir::DoubleDoubleRecursiveFftNodeIr::CooleyTukey(
                    root,
                ) = &recursive.root
                else {
                    panic!("DD N391 ND-real nested-Rader root must remain Cooley-Tukey");
                };
                let block = root
                    .pack_right
                    .axis_batch_block
                    .expect("DD N391 ND-real nested-Rader root physical block");
                assert_eq!(block.threads_per_transform, 25);
                assert_eq!(block.grouped_batch, 2);
                assert_eq!([block.local_size_x, block.local_size_y], [2, 25]);
                assert!(block.transforms_on_x);
                assert!(!block.axis_swapped);
                assert_eq!(root.twiddle_transpose.axis_batch_block, Some(block));
                assert_eq!(root.scatter_output.axis_batch_block, Some(block));

                let p17 = find_fft_rader(&recursive.root, 17)
                    .expect("DD N391 ND-real must retain p17 FFT-Rader");
                let p23 = find_fft_rader(&recursive.root, 23)
                    .expect("DD N391 ND-real must retain p23 FFT-Rader");
                assert_eq!(p17.convolution_len, 16);
                assert_eq!(p23.convolution_len, 22);
                for child in [&p23.forward_fft, &p23.inverse_fft] {
                    let DoubleDoubleBluesteinConvolutionIr::Bluestein(nested) = child else {
                        panic!("DD N391 ND-real p23 convolution must retain nested Bluestein");
                    };
                    assert_eq!(nested.logical_len, 22);
                    assert_eq!(nested.convolution_len, 64);
                    assert_eq!(nested.batch_count, nested_batch);
                    assert_eq!(nested.grouped_batch, 51);
                }

                let program = crate::ProgramIr::double_double_nd_real(&nd).unwrap();
                program.validate().unwrap();
                let shaders = crate::backend::vulkan::VulkanGlslBackend
                    .lower_double_double_nd_real(&nd)
                    .unwrap();
                assert_eq!(shaders.len(), program.passes.len());
                for shader in &shaders {
                    assert_eq!(shader.compile_spirv().unwrap().words[0], 0x0723_0203);
                }
            }
        }
    }

    #[test]
    fn nd_r2r_grouped_padded_n392_dct1_reduction_preserves_nested_fft_rader_bluestein_tree() {
        fn find_fft_rader(
            node: &crate::double_double_recursive_ir::DoubleDoubleRecursiveFftNodeIr,
            prime: usize,
        ) -> Option<&DoubleDoubleFftRaderIr> {
            use crate::double_double_recursive_ir::DoubleDoubleRecursiveFftNodeIr;
            match node {
                DoubleDoubleRecursiveFftNodeIr::FftRader(rader) if rader.prime == prime => {
                    Some(rader)
                }
                DoubleDoubleRecursiveFftNodeIr::CooleyTukey(cooley) => {
                    find_fft_rader(&cooley.left, prime)
                        .or_else(|| find_fft_rader(&cooley.right, prime))
                }
                _ => None,
            }
        }

        fn block_of(
            node: &crate::double_double_recursive_ir::DoubleDoubleRecursiveFftNodeIr,
        ) -> Option<crate::scheduler::StockhamAxisBlockSchedule> {
            use crate::double_double_recursive_ir::DoubleDoubleRecursiveFftNodeIr;
            match node {
                DoubleDoubleRecursiveFftNodeIr::Stockham(stockham) => stockham.axis_batch_block,
                DoubleDoubleRecursiveFftNodeIr::CooleyTukey(cooley) => {
                    cooley.pack_right.axis_batch_block
                }
                DoubleDoubleRecursiveFftNodeIr::DirectRader(rader) => rader.axis_batch_block,
                DoubleDoubleRecursiveFftNodeIr::FftRader(rader) => rader.caller_axis_batch_block,
                DoubleDoubleRecursiveFftNodeIr::Bluestein(_) => None,
            }
        }

        let mut tuning = PlannerTuning::portable();
        tuning.min_rader_direct_prime = 17;
        tuning.min_rader_fft_prime = 17;
        tuning.validate().unwrap();

        for vendor in [crate::GpuVendor::Nvidia, crate::GpuVendor::Amd] {
            let device = DeviceProfile {
                shared_memory_bytes: 32 * 1024,
                shared_memory_pow2_bytes: 32 * 1024,
                max_threads_per_block: 1024,
                max_workgroup_size: [1024, 1024, 64],
                supports_f64: true,
                ..DeviceProfile::generic(crate::Backend::Vulkan, vendor)
            };
            let config = FftConfig::new(vec![392usize, 8])
                .with_batch_count(5)
                .with_grouped_batch(0, 3)
                .unwrap()
                .with_zero_padding(0, 1, 2)
                .unwrap()
                .with_transform(TransformKind::Dct(DctType::I))
                .with_precision(Precision::DoubleDouble)
                .with_tuning(tuning);
            let transform = crate::TransformIr::build(config, Direction::Forward, device).unwrap();
            let crate::TransformIr::RealToRealNdDoubleDouble(nd) = transform else {
                panic!("grouped padded DD N392 ND DCT-I probe did not build ND R2R");
            };
            let outer = nd.axes.iter().find(|axis| axis.axis == 0).unwrap();
            assert_eq!(outer.axis_len, 392);
            assert_eq!(outer.line_count, 8);
            assert_eq!(outer.grouped_batch, 3);
            let DoubleDoubleR2rAlgorithm::FftReduction { fft_len, fft, .. } =
                &outer.transform.algorithm
            else {
                panic!("grouped padded DD N392 DCT-I must use FFT reduction");
            };
            assert_eq!(*fft_len, 782);
            let DoubleDoubleOneDimIr::Recursive(recursive) = fft.as_ref() else {
                panic!("grouped padded DD N392 DCT-I must retain recursive N782 child");
            };
            assert_eq!(recursive.logical_len, 782);
            assert_eq!(recursive.batch_count, 40);
            assert_eq!(recursive.grouped_batch, 3);
            assert!(recursive.rader_forced_upload_schedule.is_none());
            assert!(recursive.stockham_upload_schedule.is_none());

            let crate::double_double_recursive_ir::DoubleDoubleRecursiveFftNodeIr::CooleyTukey(
                root,
            ) = &recursive.root
            else {
                panic!("DD N392 DCT-I N782 reduction child must remain Cooley-Tukey");
            };
            let block = root
                .pack_right
                .axis_batch_block
                .expect("DD N392 DCT-I N782 root physical block");
            assert_eq!(block.threads_per_transform, 49);
            assert_eq!(block.grouped_batch, 1);
            assert_eq!([block.local_size_x, block.local_size_y], [1, 49]);
            assert!(block.transforms_on_x);
            assert!(!block.axis_swapped);
            assert_eq!(root.twiddle_transpose.axis_batch_block, Some(block));
            assert_eq!(root.scatter_output.axis_batch_block, Some(block));

            let p17 = find_fft_rader(&recursive.root, 17)
                .expect("DD N392 DCT-I N782 child must retain p17 FFT-Rader");
            let p23 = find_fft_rader(&recursive.root, 23)
                .expect("DD N392 DCT-I N782 child must retain p23 FFT-Rader");
            assert_eq!(p17.convolution_len, 16);
            assert_eq!(p23.convolution_len, 22);
            for child in [&p23.forward_fft, &p23.inverse_fft] {
                let DoubleDoubleBluesteinConvolutionIr::Bluestein(nested) = child else {
                    panic!("DD N392 DCT-I p23 convolution must retain nested Bluestein");
                };
                assert_eq!(nested.logical_len, 22);
                assert_eq!(nested.convolution_len, 64);
            }

            let program = crate::ProgramIr::double_double_nd_r2r(&nd).unwrap();
            program.validate().unwrap();
            let shaders = crate::backend::vulkan::VulkanGlslBackend
                .lower_double_double_nd_r2r(&nd)
                .unwrap();
            assert_eq!(shaders.len(), program.passes.len());
            for shader in &shaders {
                assert_eq!(shader.compile_spirv().unwrap().words[0], 0x0723_0203);
            }
        }

        for backend in [crate::Backend::OpenCl, crate::Backend::LevelZero] {
            let device = DeviceProfile {
                shared_memory_bytes: 32 * 1024,
                shared_memory_pow2_bytes: 32 * 1024,
                max_threads_per_block: 512,
                max_workgroup_size: [512, 512, 64],
                coalesced_memory_bytes: 64,
                supports_f64: true,
                ..DeviceProfile::generic(backend, crate::GpuVendor::Intel)
            };
            let config = FftConfig::new(vec![392usize, 8])
                .with_batch_count(5)
                .with_grouped_batch(0, 3)
                .unwrap()
                .with_zero_padding(0, 1, 2)
                .unwrap()
                .with_transform(TransformKind::Dct(DctType::I))
                .with_precision(Precision::DoubleDouble)
                .with_tuning(tuning);
            let transform = crate::TransformIr::build(config, Direction::Forward, device).unwrap();
            let crate::TransformIr::RealToRealNdDoubleDouble(nd) = transform else {
                panic!("Intel grouped padded DD N392 ND DCT-I probe did not build ND R2R");
            };
            let outer = nd.axes.iter().find(|axis| axis.axis == 0).unwrap();
            let DoubleDoubleR2rAlgorithm::FftReduction { fft_len, fft, .. } =
                &outer.transform.algorithm
            else {
                panic!("Intel grouped padded DD N392 DCT-I must use FFT reduction");
            };
            assert_eq!(*fft_len, 782);
            let DoubleDoubleOneDimIr::Recursive(recursive) = fft.as_ref() else {
                panic!("Intel grouped padded DD N392 DCT-I must retain recursive N782 child");
            };
            assert_eq!(recursive.logical_len, 782);
            assert_eq!(recursive.batch_count, 40);
            assert_eq!(recursive.grouped_batch, 3);
            assert!(recursive.stockham_upload_schedule.is_none());
            let schedule = recursive
                .rader_forced_upload_schedule
                .as_ref()
                .expect("Intel N782 DCT-I child must retain capacity two-upload schedule");
            assert_eq!(schedule.upload_count, 2);
            assert_eq!(schedule.axis_split.as_slice(), [34, 23]);
            assert_eq!(
                schedule.reason,
                crate::scheduler::RaderUploadReason::CapacityOrBandwidth
            );

            let crate::double_double_recursive_ir::DoubleDoubleRecursiveFftNodeIr::CooleyTukey(
                root,
            ) = &recursive.root
            else {
                panic!("Intel DD N392 DCT-I N782 child must retain Cooley upload root");
            };
            for (node, expected_len, expected_batch) in
                [(&root.left, 34usize, 920usize), (&root.right, 23, 1360)]
            {
                assert_eq!(node.logical_len(), expected_len);
                assert_eq!(node.batch_count(), expected_batch);
                let block = block_of(node).expect("Intel N782 upload component block");
                assert_eq!(block.threads_per_transform, 3);
                assert_eq!(block.grouped_batch, 3);
                assert_eq!([block.local_size_x, block.local_size_y], [3, 3]);
                assert!(block.transforms_on_x);
                assert!(!block.axis_swapped);
            }

            let p17 = find_fft_rader(&recursive.root, 17)
                .expect("Intel DD N782 child must retain p17 FFT-Rader");
            let p23 = find_fft_rader(&recursive.root, 23)
                .expect("Intel DD N782 child must retain p23 FFT-Rader");
            assert_eq!(p17.convolution_len, 16);
            assert_eq!(p23.convolution_len, 22);
            for child in [&p23.forward_fft, &p23.inverse_fft] {
                let DoubleDoubleBluesteinConvolutionIr::Bluestein(nested) = child else {
                    panic!("Intel DD N782 p23 convolution must retain nested Bluestein");
                };
                assert_eq!(nested.logical_len, 22);
                assert_eq!(nested.convolution_len, 64);
            }

            let program = crate::ProgramIr::double_double_nd_r2r(&nd).unwrap();
            program.validate().unwrap();
            let shaders = crate::backend::vulkan::VulkanGlslBackend
                .lower_double_double_nd_r2r(&nd)
                .unwrap();
            assert_eq!(shaders.len(), program.passes.len());
            for shader in &shaders {
                assert_eq!(shader.compile_spirv().unwrap().words[0], 0x0723_0203);
            }
        }
    }

    #[test]
    fn nd_r2r_grouped_padded_n390_dst1_reduction_preserves_nested_fft_rader_bluestein_tree() {
        fn find_fft_rader(
            node: &crate::double_double_recursive_ir::DoubleDoubleRecursiveFftNodeIr,
            prime: usize,
        ) -> Option<&DoubleDoubleFftRaderIr> {
            use crate::double_double_recursive_ir::DoubleDoubleRecursiveFftNodeIr;
            match node {
                DoubleDoubleRecursiveFftNodeIr::FftRader(rader) if rader.prime == prime => {
                    Some(rader)
                }
                DoubleDoubleRecursiveFftNodeIr::CooleyTukey(cooley) => {
                    find_fft_rader(&cooley.left, prime)
                        .or_else(|| find_fft_rader(&cooley.right, prime))
                }
                _ => None,
            }
        }

        let mut tuning = PlannerTuning::portable();
        tuning.min_rader_direct_prime = 17;
        tuning.min_rader_fft_prime = 17;
        tuning.validate().unwrap();

        for vendor in [crate::GpuVendor::Nvidia, crate::GpuVendor::Amd] {
            let device = DeviceProfile {
                shared_memory_bytes: 32 * 1024,
                shared_memory_pow2_bytes: 32 * 1024,
                max_threads_per_block: 1024,
                max_workgroup_size: [1024, 1024, 64],
                supports_f64: true,
                ..DeviceProfile::generic(crate::Backend::Vulkan, vendor)
            };
            let config = FftConfig::new(vec![390usize, 8])
                .with_batch_count(5)
                .with_grouped_batch(0, 3)
                .unwrap()
                .with_zero_padding(0, 1, 2)
                .unwrap()
                .with_transform(TransformKind::Dst(DstType::I))
                .with_precision(Precision::DoubleDouble)
                .with_tuning(tuning);
            let transform = crate::TransformIr::build(config, Direction::Forward, device).unwrap();
            let crate::TransformIr::RealToRealNdDoubleDouble(nd) = transform else {
                panic!("grouped padded DD N390 ND DST-I probe did not build ND R2R");
            };
            let outer = nd.axes.iter().find(|axis| axis.axis == 0).unwrap();
            assert_eq!(outer.axis_len, 390);
            assert_eq!(outer.line_count, 8);
            assert_eq!(outer.grouped_batch, 3);
            let DoubleDoubleR2rAlgorithm::FftReduction { fft_len, fft, .. } =
                &outer.transform.algorithm
            else {
                panic!("grouped padded DD N390 DST-I must use FFT reduction");
            };
            assert_eq!(*fft_len, 782);
            let DoubleDoubleOneDimIr::Recursive(recursive) = fft.as_ref() else {
                panic!("grouped padded DD N390 DST-I must retain recursive N782 child");
            };
            assert_eq!(recursive.logical_len, 782);
            assert_eq!(recursive.batch_count, 40);
            assert_eq!(recursive.grouped_batch, 3);
            assert!(recursive.rader_forced_upload_schedule.is_none());
            assert!(recursive.stockham_upload_schedule.is_none());

            let crate::double_double_recursive_ir::DoubleDoubleRecursiveFftNodeIr::CooleyTukey(
                root,
            ) = &recursive.root
            else {
                panic!("DD N390 DST-I N782 reduction child must remain Cooley-Tukey");
            };
            let block = root
                .pack_right
                .axis_batch_block
                .expect("DD N390 DST-I N782 root physical block");
            assert_eq!(block.threads_per_transform, 49);
            assert_eq!(block.grouped_batch, 1);
            assert_eq!([block.local_size_x, block.local_size_y], [1, 49]);
            assert!(block.transforms_on_x);
            assert!(!block.axis_swapped);
            assert_eq!(root.twiddle_transpose.axis_batch_block, Some(block));
            assert_eq!(root.scatter_output.axis_batch_block, Some(block));

            let p17 = find_fft_rader(&recursive.root, 17)
                .expect("DD N390 DST-I N782 child must retain p17 FFT-Rader");
            let p23 = find_fft_rader(&recursive.root, 23)
                .expect("DD N390 DST-I N782 child must retain p23 FFT-Rader");
            assert_eq!(p17.convolution_len, 16);
            assert_eq!(p23.convolution_len, 22);
            for child in [&p23.forward_fft, &p23.inverse_fft] {
                let DoubleDoubleBluesteinConvolutionIr::Bluestein(nested) = child else {
                    panic!("DD N390 DST-I p23 convolution must retain nested Bluestein");
                };
                assert_eq!(nested.logical_len, 22);
                assert_eq!(nested.convolution_len, 64);
            }

            let program = crate::ProgramIr::double_double_nd_r2r(&nd).unwrap();
            program.validate().unwrap();
            let shaders = crate::backend::vulkan::VulkanGlslBackend
                .lower_double_double_nd_r2r(&nd)
                .unwrap();
            assert_eq!(shaders.len(), program.passes.len());
            for shader in &shaders {
                assert_eq!(shader.compile_spirv().unwrap().words[0], 0x0723_0203);
            }
        }
    }

    #[test]
    fn even_dct4_dst4_use_half_size_and_match_direct_double_double_formula() {
        let length = 10usize;
        let input = (0..length)
            .map(|index| {
                let x = index as f64;
                DoubleDouble::from_parts(
                    (0.19 * x).sin() + 0.07 * (0.31 * x).cos() - 0.009 * x,
                    (index + 1) as f64 * 1.7e-31,
                )
            })
            .collect::<Vec<_>>();

        for transform in [
            R2rTransform::Dct(DctType::IV),
            R2rTransform::Dst(DstType::IV),
        ] {
            let kind = match transform {
                R2rTransform::Dct(kind) => TransformKind::Dct(kind),
                R2rTransform::Dst(kind) => TransformKind::Dst(kind),
            };
            let plan = FftPlan::build(
                FftConfig::new(vec![length])
                    .with_transform(kind)
                    .with_precision(Precision::DoubleDouble),
            )
            .unwrap();
            let ir = DoubleDoubleR2rIr::build(&plan, Direction::Forward).unwrap();
            let DoubleDoubleR2rAlgorithm::EvenTypeIvHalfSize {
                fft_len,
                fft,
                pack_phases,
                extract_phases,
            } = &ir.algorithm
            else {
                panic!("even {transform:?} must use the DD half-size FFT reduction");
            };
            assert_eq!(*fft_len, length / 2);
            assert_eq!(fft.sequence_len(), length / 2);
            assert_eq!(fft.direction(), Direction::Inverse);
            assert_eq!(pack_phases.len(), length / 2);
            assert_eq!(extract_phases.len(), length);

            let output = execute_double_double_r2r_ir(&ir, &input).unwrap();
            for (k, actual) in output.iter().copied().enumerate() {
                let expected = input.iter().copied().enumerate().fold(
                    DoubleDouble::ZERO,
                    |sum, (j, value)| {
                        sum + value
                            * double_double_r2r_coefficient(transform, length, j, k).unwrap()
                    },
                );
                let error = (actual - expected).abs().to_f64().abs();
                assert!(
                    error < 5.0e-25,
                    "{transform:?} even length bin {k} mismatch: actual={actual:?} expected={expected:?} error={error:e}"
                );
            }
        }
    }

    #[test]
    fn nd_r2r_grouped_padded_n782_dct4_half_size_preserves_nested_fft_rader_bluestein_tree() {
        fn find_fft_rader(
            node: &crate::double_double_recursive_ir::DoubleDoubleRecursiveFftNodeIr,
            prime: usize,
        ) -> Option<&DoubleDoubleFftRaderIr> {
            use crate::double_double_recursive_ir::DoubleDoubleRecursiveFftNodeIr;
            match node {
                DoubleDoubleRecursiveFftNodeIr::FftRader(rader) if rader.prime == prime => {
                    Some(rader)
                }
                DoubleDoubleRecursiveFftNodeIr::CooleyTukey(cooley) => {
                    find_fft_rader(&cooley.left, prime)
                        .or_else(|| find_fft_rader(&cooley.right, prime))
                }
                _ => None,
            }
        }

        let mut tuning = PlannerTuning::portable();
        tuning.min_rader_direct_prime = 17;
        tuning.min_rader_fft_prime = 17;
        tuning.validate().unwrap();

        for vendor in [crate::GpuVendor::Nvidia, crate::GpuVendor::Amd] {
            let device = DeviceProfile {
                shared_memory_bytes: 32 * 1024,
                shared_memory_pow2_bytes: 32 * 1024,
                max_threads_per_block: 1024,
                max_workgroup_size: [1024, 1024, 64],
                supports_f64: true,
                ..DeviceProfile::generic(crate::Backend::Vulkan, vendor)
            };
            let config = FftConfig::new(vec![782usize, 8])
                .with_batch_count(5)
                .with_grouped_batch(0, 3)
                .unwrap()
                .with_zero_padding(0, 1, 2)
                .unwrap()
                .with_transform(TransformKind::Dct(DctType::IV))
                .with_precision(Precision::DoubleDouble)
                .with_tuning(tuning);
            let transform = crate::TransformIr::build(config, Direction::Forward, device).unwrap();
            let crate::TransformIr::RealToRealNdDoubleDouble(nd) = transform else {
                panic!("grouped padded DD N782 ND DCT-IV probe did not build ND R2R");
            };
            let outer = nd.axes.iter().find(|axis| axis.axis == 0).unwrap();
            assert_eq!(outer.axis_len, 782);
            assert_eq!(outer.line_count, 8);
            assert_eq!(outer.grouped_batch, 3);
            let DoubleDoubleR2rAlgorithm::EvenTypeIvHalfSize {
                fft_len,
                fft,
                pack_phases,
                extract_phases,
            } = &outer.transform.algorithm
            else {
                panic!("grouped padded DD N782 DCT-IV must use half-size FFT reduction");
            };
            assert_eq!(*fft_len, 391);
            assert_eq!(pack_phases.len(), 391);
            assert_eq!(extract_phases.len(), 782);
            let DoubleDoubleOneDimIr::Recursive(recursive) = fft.as_ref() else {
                panic!("grouped padded DD N782 DCT-IV must retain recursive N391 child");
            };
            assert_eq!(recursive.logical_len, 391);
            assert_eq!(recursive.batch_count, 40);
            assert_eq!(recursive.grouped_batch, 3);
            assert_eq!(recursive.direction, Direction::Inverse);
            assert!(recursive.rader_forced_upload_schedule.is_none());
            assert!(recursive.stockham_upload_schedule.is_none());

            let crate::double_double_recursive_ir::DoubleDoubleRecursiveFftNodeIr::CooleyTukey(
                root,
            ) = &recursive.root
            else {
                panic!("DD N782 DCT-IV N391 half-size child must remain Cooley-Tukey");
            };
            let block = root
                .pack_right
                .axis_batch_block
                .expect("DD N782 DCT-IV N391 root physical block");
            assert_eq!(block.threads_per_transform, 25);
            assert_eq!(block.grouped_batch, 2);
            assert_eq!([block.local_size_x, block.local_size_y], [2, 25]);
            assert!(block.transforms_on_x);
            assert!(!block.axis_swapped);
            assert_eq!(root.twiddle_transpose.axis_batch_block, Some(block));
            assert_eq!(root.scatter_output.axis_batch_block, Some(block));

            let p17 = find_fft_rader(&recursive.root, 17)
                .expect("DD N782 DCT-IV N391 child must retain p17 FFT-Rader");
            let p23 = find_fft_rader(&recursive.root, 23)
                .expect("DD N782 DCT-IV N391 child must retain p23 FFT-Rader");
            assert_eq!(p17.convolution_len, 16);
            assert_eq!(p23.convolution_len, 22);
            for child in [&p23.forward_fft, &p23.inverse_fft] {
                let DoubleDoubleBluesteinConvolutionIr::Bluestein(nested) = child else {
                    panic!("DD N782 DCT-IV p23 convolution must retain nested Bluestein");
                };
                assert_eq!(nested.logical_len, 22);
                assert_eq!(nested.convolution_len, 64);
                assert_eq!(nested.batch_count, 680);
                assert_eq!(nested.grouped_batch, 51);
            }

            let program = crate::ProgramIr::double_double_nd_r2r(&nd).unwrap();
            program.validate().unwrap();
            let shaders = crate::backend::vulkan::VulkanGlslBackend
                .lower_double_double_nd_r2r(&nd)
                .unwrap();
            assert_eq!(shaders.len(), program.passes.len());
            for shader in &shaders {
                assert_eq!(shader.compile_spirv().unwrap().words[0], 0x0723_0203);
            }
        }
    }

    #[test]
    fn nd_r2r_grouped_padded_n204544_dct4_half_size_preserves_mapped_three_upload_child() {
        fn block_of(
            node: &crate::double_double_recursive_ir::DoubleDoubleRecursiveFftNodeIr,
        ) -> Option<crate::scheduler::StockhamAxisBlockSchedule> {
            use crate::double_double_recursive_ir::DoubleDoubleRecursiveFftNodeIr;
            match node {
                DoubleDoubleRecursiveFftNodeIr::Stockham(stockham) => stockham.axis_batch_block,
                DoubleDoubleRecursiveFftNodeIr::CooleyTukey(cooley) => {
                    cooley.pack_right.axis_batch_block
                }
                DoubleDoubleRecursiveFftNodeIr::DirectRader(rader) => rader.axis_batch_block,
                DoubleDoubleRecursiveFftNodeIr::FftRader(rader) => rader.caller_axis_batch_block,
                DoubleDoubleRecursiveFftNodeIr::Bluestein(_) => None,
            }
        }

        for vendor in [crate::GpuVendor::Nvidia, crate::GpuVendor::Amd] {
            let device = DeviceProfile {
                shared_memory_bytes: 8 * 1024,
                shared_memory_pow2_bytes: 8 * 1024,
                max_threads_per_block: 1024,
                max_workgroup_size: [1024, 1024, 64],
                supports_f64: true,
                ..DeviceProfile::generic(crate::Backend::Vulkan, vendor)
            };
            let config = FftConfig::new(vec![204_544usize, 8])
                .with_batch_count(5)
                .with_grouped_batch(0, 3)
                .unwrap()
                .with_zero_padding(0, 1, 2)
                .unwrap()
                .with_transform(TransformKind::Dct(DctType::IV))
                .with_precision(Precision::DoubleDouble)
                .with_tuning(PlannerTuning::portable());
            let transform = crate::TransformIr::build(config, Direction::Forward, device).unwrap();
            let crate::TransformIr::RealToRealNdDoubleDouble(nd) = transform else {
                panic!("grouped padded DD N204544 ND DCT-IV probe did not build ND R2R");
            };
            let outer = nd.axes.iter().find(|axis| axis.axis == 0).unwrap();
            assert_eq!(outer.axis_len, 204_544);
            assert_eq!(outer.line_count, 8);
            assert_eq!(outer.grouped_batch, 3);
            let DoubleDoubleR2rAlgorithm::EvenTypeIvHalfSize { fft_len, fft, .. } =
                &outer.transform.algorithm
            else {
                panic!("grouped padded DD N204544 DCT-IV must use half-size FFT reduction");
            };
            assert_eq!(*fft_len, 102_272);
            let DoubleDoubleOneDimIr::Recursive(recursive) = fft.as_ref() else {
                panic!("grouped padded DD N204544 DCT-IV must retain recursive N102272 child");
            };
            assert_eq!(recursive.logical_len, 102_272);
            assert_eq!(recursive.direction, Direction::Inverse);
            assert_eq!(recursive.batch_count, 40);
            assert_eq!(recursive.grouped_batch, 3);
            let schedule = recursive
                .rader_forced_upload_schedule
                .as_ref()
                .expect("DCT-IV N102272 child must retain forced-Rader schedule");
            assert_eq!(schedule.upload_count, 3);
            assert_eq!(schedule.axis_split, vec![64, 47, 34]);
            assert!(recursive.three_upload_four_step_plan.is_none());

            let crate::double_double_recursive_ir::DoubleDoubleRecursiveFftNodeIr::CooleyTukey(
                root,
            ) = &recursive.root
            else {
                panic!("DCT-IV N102272 child must retain Cooley-Tukey root");
            };
            let crate::double_double_recursive_ir::DoubleDoubleRecursiveFftNodeIr::CooleyTukey(
                upper,
            ) = &root.right
            else {
                panic!("DCT-IV N102272 child must retain upper Cooley node");
            };
            let nodes = [&root.left, &upper.left, &upper.right];
            let expected = [
                (64usize, 63_920usize, 8usize),
                (47, 87_040, 24),
                (34, 120_320, 3),
            ];
            for (node, (len, batch, lanes)) in nodes.into_iter().zip(expected) {
                assert_eq!(node.logical_len(), len);
                assert_eq!(node.batch_count(), batch);
                let block = block_of(node).expect("DCT-IV N102272 child physical block");
                assert_eq!(block.threads_per_transform, lanes);
                assert_eq!(block.grouped_batch, 3);
                assert_eq!([block.local_size_x, block.local_size_y], [3, lanes]);
                assert!(block.transforms_on_x);
                assert!(!block.axis_swapped);
            }

            let mapped = recursive
                .forced_rader_three_upload_mapped_components()
                .unwrap()
                .expect("DCT-IV N102272 child must materialize mapped three-upload components");
            assert_eq!(
                mapped
                    .iter()
                    .map(|component| component.upload_id())
                    .collect::<Vec<_>>(),
                vec![2, 1, 0]
            );
            assert_eq!(
                mapped
                    .iter()
                    .map(|component| component.logical_len())
                    .collect::<Vec<_>>(),
                vec![34, 47, 64]
            );

            let program = crate::ProgramIr::double_double_nd_r2r(&nd).unwrap();
            program.validate().unwrap();
            assert!(
                program
                    .passes
                    .iter()
                    .any(|pass| pass.name.contains("forced_rader_three_upload_0"))
            );
            let shaders = crate::backend::vulkan::VulkanGlslBackend
                .lower_double_double_nd_r2r(&nd)
                .unwrap();
            assert_eq!(shaders.len(), program.passes.len());
            for shader in &shaders {
                assert_eq!(shader.compile_spirv().unwrap().words[0], 0x0723_0203);
            }
        }
    }

    #[test]
    fn nd_r2r_grouped_padded_n102272_dct2_npoint_preserves_mapped_three_upload_child() {
        fn block_of(
            node: &crate::double_double_recursive_ir::DoubleDoubleRecursiveFftNodeIr,
        ) -> Option<crate::scheduler::StockhamAxisBlockSchedule> {
            use crate::double_double_recursive_ir::DoubleDoubleRecursiveFftNodeIr;
            match node {
                DoubleDoubleRecursiveFftNodeIr::Stockham(stockham) => stockham.axis_batch_block,
                DoubleDoubleRecursiveFftNodeIr::CooleyTukey(cooley) => {
                    cooley.pack_right.axis_batch_block
                }
                DoubleDoubleRecursiveFftNodeIr::DirectRader(rader) => rader.axis_batch_block,
                DoubleDoubleRecursiveFftNodeIr::FftRader(rader) => rader.caller_axis_batch_block,
                DoubleDoubleRecursiveFftNodeIr::Bluestein(_) => None,
            }
        }

        for vendor in [crate::GpuVendor::Nvidia, crate::GpuVendor::Amd] {
            let device = DeviceProfile {
                shared_memory_bytes: 8 * 1024,
                shared_memory_pow2_bytes: 8 * 1024,
                max_threads_per_block: 1024,
                max_workgroup_size: [1024, 1024, 64],
                supports_f64: true,
                ..DeviceProfile::generic(crate::Backend::Vulkan, vendor)
            };
            let config = FftConfig::new(vec![102_272usize, 8])
                .with_batch_count(5)
                .with_grouped_batch(0, 3)
                .unwrap()
                .with_zero_padding(0, 1, 2)
                .unwrap()
                .with_transform(TransformKind::Dct(DctType::II))
                .with_precision(Precision::DoubleDouble)
                .with_tuning(PlannerTuning::portable());
            let transform = crate::TransformIr::build(config, Direction::Forward, device).unwrap();
            let crate::TransformIr::RealToRealNdDoubleDouble(nd) = transform else {
                panic!("grouped padded DD N102272 ND DCT-II probe did not build ND R2R");
            };
            let outer = nd.axes.iter().find(|axis| axis.axis == 0).unwrap();
            assert_eq!(outer.axis_len, 102_272);
            assert_eq!(outer.line_count, 8);
            assert_eq!(outer.grouped_batch, 3);
            let DoubleDoubleR2rAlgorithm::FftReduction { fft_len, fft, .. } =
                &outer.transform.algorithm
            else {
                panic!("grouped padded DD N102272 DCT-II must use FFT reduction");
            };
            assert_eq!(*fft_len, 102_272);
            let DoubleDoubleOneDimIr::Recursive(recursive) = fft.as_ref() else {
                panic!("grouped padded DD N102272 DCT-II must retain recursive N102272 child");
            };
            assert_eq!(recursive.logical_len, 102_272);
            assert_eq!(recursive.direction, Direction::Forward);
            assert_eq!(recursive.batch_count, 40);
            assert_eq!(recursive.grouped_batch, 3);
            let schedule = recursive
                .rader_forced_upload_schedule
                .as_ref()
                .expect("DCT-II N102272 child must retain forced-Rader schedule");
            assert_eq!(schedule.upload_count, 3);
            assert_eq!(schedule.axis_split, vec![64, 47, 34]);
            assert!(recursive.three_upload_four_step_plan.is_none());

            let crate::double_double_recursive_ir::DoubleDoubleRecursiveFftNodeIr::CooleyTukey(
                root,
            ) = &recursive.root
            else {
                panic!("DCT-II N102272 child must retain Cooley-Tukey root");
            };
            let crate::double_double_recursive_ir::DoubleDoubleRecursiveFftNodeIr::CooleyTukey(
                upper,
            ) = &root.right
            else {
                panic!("DCT-II N102272 child must retain upper Cooley node");
            };
            let nodes = [&root.left, &upper.left, &upper.right];
            let expected = [
                (64usize, 63_920usize, 8usize),
                (47, 87_040, 24),
                (34, 120_320, 3),
            ];
            for (node, (len, batch, lanes)) in nodes.into_iter().zip(expected) {
                assert_eq!(node.logical_len(), len);
                assert_eq!(node.batch_count(), batch);
                let block = block_of(node).expect("DCT-II N102272 child physical block");
                assert_eq!(block.threads_per_transform, lanes);
                assert_eq!(block.grouped_batch, 3);
                assert_eq!([block.local_size_x, block.local_size_y], [3, lanes]);
                assert!(block.transforms_on_x);
                assert!(!block.axis_swapped);
            }

            let mapped = recursive
                .forced_rader_three_upload_mapped_components()
                .unwrap()
                .expect("DCT-II N102272 child must materialize mapped three-upload components");
            assert_eq!(
                mapped
                    .iter()
                    .map(|component| component.upload_id())
                    .collect::<Vec<_>>(),
                vec![2, 1, 0]
            );
            assert_eq!(
                mapped
                    .iter()
                    .map(|component| component.logical_len())
                    .collect::<Vec<_>>(),
                vec![34, 47, 64]
            );

            let Some(
                crate::double_double_recursive_ir::DoubleDoubleForcedRaderThreeUploadComponentIr::Recursive {
                    upload_id: 2,
                    ir: crate::DoubleDoubleRecursiveFftNodeIr::CooleyTukey(high),
                },
            ) = mapped.first()
            else {
                panic!("DCT-II N102272 upload2 must remain recursive Cooley-Tukey");
            };
            assert!(matches!(
                high.pack_right.input_modifier,
                crate::double_double_recursive_ir::DoubleDoubleCooleyTukeyInputModifier::FourStepThreeUpload2(_)
            ));
            let child_program = crate::ProgramIr::double_double_recursive(recursive).unwrap();
            let fused_axis_program = crate::ProgramIr::double_double_r2r(&outer.transform).unwrap();
            assert_eq!(fused_axis_program.passes.len(), child_program.passes.len());
            assert!(
                fused_axis_program
                    .passes
                    .first()
                    .unwrap()
                    .name
                    .contains("r2r_dct2_forced_rader_three_upload_2")
            );
            assert!(
                fused_axis_program
                    .passes
                    .last()
                    .unwrap()
                    .name
                    .contains("r2r_dct2_forced_rader_three_upload_0")
            );
            assert!(fused_axis_program.resources.iter().any(|resource| {
                resource.name == "double_double_r2r_forced_three_upload_exchange_0"
                    && resource.kind == crate::program_ir::ProgramResourceKind::Scratch
                    && resource.scalar == crate::kernel_ir::ScalarType::DoubleDouble
            }));
            assert!(fused_axis_program.resources.iter().all(|resource| {
                !resource.name.contains("double_double_r2r_fft_input")
                    && !resource.name.contains("double_double_r2r_fft_output")
            }));
            let fused_axis_shaders = crate::backend::vulkan::VulkanGlslBackend
                .lower_double_double_r2r(&outer.transform)
                .unwrap();
            assert_eq!(fused_axis_shaders.len(), fused_axis_program.passes.len());
            assert!(
                fused_axis_shaders
                    .first()
                    .unwrap()
                    .glsl
                    .contains("logical_source")
            );
            assert!(
                fused_axis_shaders
                    .last()
                    .unwrap()
                    .glsl
                    .contains("logical_output")
            );
            for shader in &fused_axis_shaders {
                assert_eq!(shader.compile_spirv().unwrap().words[0], 0x0723_0203);
            }

            for direction in [Direction::Forward, Direction::Inverse] {
                let one_dim_f64 = crate::TransformIr::build(
                    FftConfig::new(vec![102_272usize])
                        .with_batch_count(5)
                        .with_grouped_batch(0, 3)
                        .unwrap()
                        .with_transform(TransformKind::Dct(DctType::II))
                        .with_precision(Precision::DoubleDoubleF64Storage)
                        .with_inverse_normalization(direction == Direction::Inverse)
                        .with_tuning(PlannerTuning::portable()),
                    direction,
                    device,
                )
                .unwrap();
                let crate::TransformIr::RealToRealDoubleDouble(one_dim_f64) = one_dim_f64 else {
                    panic!("DD/F64 N102272 DCT-II/III probe did not build 1D R2R");
                };
                let DoubleDoubleR2rAlgorithm::FftReduction { fft, .. } = &one_dim_f64.algorithm
                else {
                    panic!("DD/F64 N102272 DCT-II/III must use FFT reduction");
                };
                let DoubleDoubleOneDimIr::Recursive(f64_recursive) = fft.as_ref() else {
                    panic!("DD/F64 N102272 DCT-II/III must retain recursive child");
                };
                assert_eq!(
                    f64_recursive
                        .rader_forced_upload_schedule
                        .as_ref()
                        .expect("DD/F64 N102272 forced-Rader schedule")
                        .axis_split,
                    vec![64, 47, 34]
                );
                let f64_components = f64_recursive
                    .forced_rader_three_upload_mapped_components()
                    .unwrap()
                    .expect("DD/F64 N102272 mapped forced-three components");
                assert!(matches!(
                    f64_components.first(),
                    Some(
                        crate::double_double_recursive_ir::DoubleDoubleForcedRaderThreeUploadComponentIr::Recursive {
                            upload_id: 2,
                            ir: crate::DoubleDoubleRecursiveFftNodeIr::CooleyTukey(_),
                        }
                    )
                ));
                let f64_child_program =
                    crate::ProgramIr::double_double_recursive(f64_recursive).unwrap();
                let f64_program = crate::ProgramIr::double_double_r2r(&one_dim_f64).unwrap();
                assert_eq!(f64_program.passes.len(), f64_child_program.passes.len());
                assert_eq!(
                    f64_program.input_resource().unwrap().scalar,
                    crate::kernel_ir::ScalarType::F64
                );
                assert_eq!(
                    f64_program.output_resource().unwrap().scalar,
                    crate::kernel_ir::ScalarType::F64
                );
                let f64_shaders = crate::backend::vulkan::VulkanGlslBackend
                    .lower_double_double_r2r(&one_dim_f64)
                    .unwrap();
                assert_eq!(f64_shaders.len(), f64_program.passes.len());
                assert!(f64_shaders.first().unwrap().glsl.contains("double data[]"));
                if direction == Direction::Forward {
                    assert!(f64_shaders.first().unwrap().glsl.contains("logical_source"));
                } else {
                    assert!(f64_shaders.first().unwrap().glsl.contains("a_index"));
                    assert!(f64_shaders.first().unwrap().glsl.contains("binding = 2"));
                }
                for shader in &f64_shaders {
                    assert_eq!(shader.compile_spirv().unwrap().words[0], 0x0723_0203);
                }
            }

            let program = crate::ProgramIr::double_double_nd_r2r(&nd).unwrap();
            program.validate().unwrap();
            assert!(
                program
                    .passes
                    .iter()
                    .any(|pass| pass.name.contains("forced_rader_three_upload_0"))
            );
            let shaders = crate::backend::vulkan::VulkanGlslBackend
                .lower_double_double_nd_r2r(&nd)
                .unwrap();
            assert_eq!(shaders.len(), program.passes.len());
            for shader in &shaders {
                assert_eq!(shader.compile_spirv().unwrap().words[0], 0x0723_0203);
            }
        }
    }

    #[test]
    fn r2r_forced_three_upload_stockham_boundaries_fuse_without_parent_dispatches() {
        let length = 17usize * 65_536;
        let device = DeviceProfile {
            shared_memory_bytes: 32 * 1024,
            shared_memory_pow2_bytes: 32 * 1024,
            max_threads_per_block: 1024,
            max_workgroup_size: [1024, 1024, 64],
            supports_f64: true,
            ..DeviceProfile::generic(crate::Backend::Vulkan, crate::GpuVendor::Nvidia)
        };
        for precision in [Precision::DoubleDouble, Precision::DoubleDoubleF64Storage] {
            for direction in [Direction::Forward, Direction::Inverse] {
                let transform = crate::TransformIr::build(
                    FftConfig::new(vec![length])
                        .with_transform(TransformKind::Dct(DctType::II))
                        .with_precision(precision)
                        .with_inverse_normalization(direction == Direction::Inverse),
                    direction,
                    device,
                )
                .unwrap();
                let crate::TransformIr::RealToRealDoubleDouble(r2r) = transform else {
                    panic!("DD N1114112 DCT-II/III probe did not build 1D R2R");
                };
                let DoubleDoubleR2rAlgorithm::FftReduction { fft, .. } = &r2r.algorithm else {
                    panic!("DD N1114112 DCT-II/III must use FFT reduction");
                };
                let DoubleDoubleOneDimIr::Recursive(recursive) = fft.as_ref() else {
                    panic!("DD N1114112 DCT-II/III must retain recursive child");
                };
                assert_eq!(
                    recursive
                        .rader_forced_upload_schedule
                        .as_ref()
                        .expect("DD N1114112 forced-Rader schedule")
                        .axis_split,
                    vec![128, 68, 128]
                );
                let components = recursive
                    .forced_rader_three_upload_mapped_components()
                    .unwrap()
                    .expect("DD N1114112 mapped forced-three components");
                assert!(matches!(
                    components.first(),
                    Some(
                        crate::double_double_recursive_ir::DoubleDoubleForcedRaderThreeUploadComponentIr::Stockham {
                            upload_id: 2,
                            ..
                        }
                    )
                ));
                assert!(matches!(
                    components.last(),
                    Some(
                        crate::double_double_recursive_ir::DoubleDoubleForcedRaderThreeUploadComponentIr::Stockham {
                            upload_id: 0,
                            ..
                        }
                    )
                ));
                let child_program = crate::ProgramIr::double_double_recursive(recursive).unwrap();
                let program = crate::ProgramIr::double_double_r2r(&r2r).unwrap();
                assert_eq!(program.passes.len(), child_program.passes.len());
                assert!(program.passes.first().unwrap().name.contains("r2r_dct"));
                assert!(
                    program
                        .passes
                        .first()
                        .unwrap()
                        .name
                        .contains("forced_rader_three_upload_2")
                );
                assert!(
                    program
                        .passes
                        .last()
                        .unwrap()
                        .name
                        .contains("forced_rader_three_upload_0")
                );
                assert!(program.resources.iter().all(|resource| {
                    !resource.name.contains("double_double_r2r_fft_input")
                        && !resource.name.contains("double_double_r2r_fft_output")
                }));
                assert!(program.resources.iter().any(|resource| {
                    resource.name == "double_double_r2r_forced_three_upload_exchange_0"
                        && resource.kind == crate::program_ir::ProgramResourceKind::Scratch
                        && resource.scalar == crate::kernel_ir::ScalarType::DoubleDouble
                }));
                let expected_scalar = match precision {
                    Precision::DoubleDouble => crate::kernel_ir::ScalarType::DoubleDouble,
                    Precision::DoubleDoubleF64Storage => crate::kernel_ir::ScalarType::F64,
                    _ => unreachable!(),
                };
                assert_eq!(program.input_resource().unwrap().scalar, expected_scalar);
                assert_eq!(program.output_resource().unwrap().scalar, expected_scalar);
                let shaders = crate::backend::vulkan::VulkanGlslBackend
                    .lower_double_double_r2r(&r2r)
                    .unwrap();
                assert_eq!(shaders.len(), program.passes.len());
                match direction {
                    Direction::Forward => {
                        assert!(shaders.first().unwrap().glsl.contains("logical_source"));
                    }
                    Direction::Inverse => {
                        let first = &shaders.first().unwrap().glsl;
                        assert!(first.contains("a_index"));
                        assert!(first.contains("vkfft_r2r_phases"));
                    }
                }
                assert!(shaders.last().unwrap().glsl.contains("logical_output"));
                for shader in &shaders {
                    assert_eq!(shader.compile_spirv().unwrap().words[0], 0x0723_0203);
                }
            }
        }
    }

    #[test]
    fn r2r_forced_two_upload_recursive_high_stockham_low_fuses_boundaries() {
        let length = 17usize * 256;
        let device = DeviceProfile {
            shared_memory_bytes: 48 * 1024,
            shared_memory_pow2_bytes: 32 * 1024,
            max_threads_per_block: 128,
            max_workgroup_size: [1024, 1024, 64],
            supports_f64: true,
            ..DeviceProfile::generic(crate::Backend::Vulkan, crate::GpuVendor::Nvidia)
        };
        for precision in [Precision::DoubleDouble, Precision::DoubleDoubleF64Storage] {
            for direction in [Direction::Forward, Direction::Inverse] {
                let transform = crate::TransformIr::build(
                    FftConfig::new(vec![length])
                        .with_transform(TransformKind::Dct(DctType::II))
                        .with_precision(precision)
                        .with_inverse_normalization(direction == Direction::Inverse),
                    direction,
                    device,
                )
                .unwrap();
                let crate::TransformIr::RealToRealDoubleDouble(r2r) = transform else {
                    panic!("DD N4352 DCT-II/III probe did not build 1D R2R");
                };
                let DoubleDoubleR2rAlgorithm::FftReduction { fft, .. } = &r2r.algorithm else {
                    panic!("DD N4352 DCT-II/III must use FFT reduction");
                };
                let DoubleDoubleOneDimIr::Recursive(recursive) = fft.as_ref() else {
                    panic!("DD N4352 DCT-II/III must retain recursive child");
                };
                assert_eq!(
                    recursive
                        .rader_forced_upload_schedule
                        .as_ref()
                        .expect("DD N4352 forced-Rader schedule")
                        .axis_split,
                    vec![64, 68]
                );
                let mapped_high = recursive
                    .forced_rader_two_upload_mapped_high_component()
                    .unwrap()
                    .expect("DD N4352 mapped recursive high component");
                let crate::DoubleDoubleRecursiveFftNodeIr::CooleyTukey(high) = mapped_high else {
                    panic!("DD N4352 high upload must remain mapped recursive Cooley-Tukey");
                };
                let crate::double_double_recursive_ir::DoubleDoubleCooleyTukeyInputModifier::FourStepRight(
                    high_mapping,
                ) = high.pack_right.input_modifier
                else {
                    panic!("DD N4352 high Cooley pack must own FourStepRight input");
                };
                let (low, low_mapping) = recursive
                    .forced_rader_two_upload_mapped_low_stockham()
                    .unwrap()
                    .expect("DD N4352 N64 Stockham low boundary");
                assert_eq!(low.sequence_len, 64);
                assert!(low.axis_batch_block.is_some());
                assert_eq!(high_mapping, low_mapping);

                let child_program = crate::ProgramIr::double_double_recursive(recursive).unwrap();
                let program = crate::ProgramIr::double_double_r2r(&r2r).unwrap();
                assert_eq!(program.passes.len(), child_program.passes.len());
                assert!(
                    program
                        .passes
                        .first()
                        .unwrap()
                        .name
                        .contains("forced_rader_two_upload_1")
                );
                assert!(
                    program
                        .passes
                        .last()
                        .unwrap()
                        .name
                        .contains("forced_rader_two_upload_0")
                );
                let expected_scalar = match precision {
                    Precision::DoubleDouble => crate::kernel_ir::ScalarType::DoubleDouble,
                    Precision::DoubleDoubleF64Storage => crate::kernel_ir::ScalarType::F64,
                    _ => unreachable!(),
                };
                assert_eq!(program.input_resource().unwrap().scalar, expected_scalar);
                assert_eq!(program.output_resource().unwrap().scalar, expected_scalar);
                assert!(program.resources.iter().any(|resource| {
                    resource.name == "double_double_rader_four_step_exchange"
                        && resource.kind == crate::program_ir::ProgramResourceKind::Scratch
                        && resource.scalar == crate::kernel_ir::ScalarType::DoubleDouble
                }));
                assert!(program.resources.iter().all(|resource| {
                    !resource.name.contains("double_double_r2r_fft_input")
                        && !resource.name.contains("double_double_r2r_fft_output")
                }));
                let shaders = crate::backend::vulkan::VulkanGlslBackend
                    .lower_double_double_r2r(&r2r)
                    .unwrap();
                assert_eq!(shaders.len(), program.passes.len());
                match direction {
                    Direction::Forward => {
                        assert!(shaders.first().unwrap().glsl.contains("logical_source"));
                    }
                    Direction::Inverse => {
                        let first = &shaders.first().unwrap().glsl;
                        assert!(first.contains("a_index"));
                        assert!(first.contains("vkfft_r2r_phases"));
                        assert!(first.contains("binding = 2"));
                    }
                }
                assert!(shaders.last().unwrap().glsl.contains("logical_output"));
                for shader in &shaders {
                    assert_eq!(shader.compile_spirv().unwrap().words[0], 0x0723_0203);
                }
            }
        }
    }

    #[test]
    fn r2r_forced_two_upload_stockham_high_recursive_low_fuses_output_boundary() {
        let length = 17usize * 300;
        let device = DeviceProfile {
            shared_memory_bytes: 48 * 1024,
            shared_memory_pow2_bytes: 32 * 1024,
            max_threads_per_block: 128,
            max_workgroup_size: [1024, 1024, 64],
            supports_f64: true,
            ..DeviceProfile::generic(crate::Backend::Vulkan, crate::GpuVendor::Nvidia)
        };
        for precision in [Precision::DoubleDouble, Precision::DoubleDoubleF64Storage] {
            for direction in [Direction::Forward, Direction::Inverse] {
                let transform = crate::TransformIr::build(
                    FftConfig::new(vec![length])
                        .with_transform(TransformKind::Dct(DctType::II))
                        .with_precision(precision)
                        .with_inverse_normalization(direction == Direction::Inverse),
                    direction,
                    device,
                )
                .unwrap();
                let crate::TransformIr::RealToRealDoubleDouble(r2r) = transform else {
                    panic!("DD N5100 DCT-II/III probe did not build 1D R2R");
                };
                let DoubleDoubleR2rAlgorithm::FftReduction { fft, .. } = &r2r.algorithm else {
                    panic!("DD N5100 DCT-II/III must use FFT reduction");
                };
                let DoubleDoubleOneDimIr::Recursive(recursive) = fft.as_ref() else {
                    panic!("DD N5100 DCT-II/III must retain recursive child");
                };
                assert_eq!(
                    recursive
                        .rader_forced_upload_schedule
                        .as_ref()
                        .expect("DD N5100 forced-Rader schedule")
                        .axis_split,
                    vec![68, 75]
                );
                let (high, mapping) = recursive
                    .forced_rader_two_upload_mapped_high_stockham()
                    .unwrap()
                    .expect("DD N5100 N75 Stockham high boundary");
                assert_eq!(high.sequence_len, 75);
                assert!(high.axis_batch_block.is_some());
                let mapped_low = recursive
                    .forced_rader_two_upload_mapped_low_component()
                    .unwrap()
                    .expect("DD N5100 mapped recursive low boundary");
                let crate::DoubleDoubleRecursiveFftNodeIr::CooleyTukey(low) = mapped_low else {
                    panic!("DD N5100 low upload must remain mapped recursive Cooley-Tukey");
                };
                assert_eq!(low.logical_len, 68);
                assert!(matches!(
                    low.scatter_output.output_modifier,
                    crate::double_double_recursive_ir::DoubleDoubleCooleyTukeyOutputModifier::FourStepLeft(low_mapping)
                        if low_mapping == mapping
                ));

                let child_program = crate::ProgramIr::double_double_recursive(recursive).unwrap();
                let program = crate::ProgramIr::double_double_r2r(&r2r).unwrap();
                assert_eq!(program.passes.len(), child_program.passes.len());
                assert!(
                    program
                        .passes
                        .first()
                        .unwrap()
                        .name
                        .contains("forced_rader_two_upload_1")
                );
                assert!(
                    program
                        .passes
                        .last()
                        .unwrap()
                        .name
                        .contains("forced_rader_two_upload_0_recursive")
                );
                let expected_scalar = match precision {
                    Precision::DoubleDouble => crate::kernel_ir::ScalarType::DoubleDouble,
                    Precision::DoubleDoubleF64Storage => crate::kernel_ir::ScalarType::F64,
                    _ => unreachable!(),
                };
                assert_eq!(program.input_resource().unwrap().scalar, expected_scalar);
                assert_eq!(program.output_resource().unwrap().scalar, expected_scalar);
                assert!(program.resources.iter().any(|resource| {
                    resource.name == "double_double_rader_four_step_exchange"
                        && resource.kind == crate::program_ir::ProgramResourceKind::Scratch
                        && resource.scalar == crate::kernel_ir::ScalarType::DoubleDouble
                }));
                assert!(program.resources.iter().all(|resource| {
                    !resource.name.contains("double_double_r2r_fft_input")
                        && !resource.name.contains("double_double_r2r_fft_output")
                }));

                let shaders = crate::backend::vulkan::VulkanGlslBackend
                    .lower_double_double_r2r(&r2r)
                    .unwrap();
                assert_eq!(shaders.len(), program.passes.len());
                match direction {
                    Direction::Forward => {
                        assert!(shaders.first().unwrap().glsl.contains("logical_source"));
                        let last = &shaders.last().unwrap().glsl;
                        assert!(last.contains("vkfft_r2r_destination"));
                        assert!(last.contains("vkfft_r2r_phases"));
                        assert!(last.contains("binding = 2"));
                    }
                    Direction::Inverse => {
                        let first = &shaders.first().unwrap().glsl;
                        assert!(first.contains("a_index"));
                        assert!(first.contains("vkfft_r2r_phases"));
                        assert!(first.contains("binding = 4"));
                        let last = &shaders.last().unwrap().glsl;
                        assert!(last.contains("logical_output"));
                        assert!(last.contains("vkfft_r2r_destination"));
                    }
                }
                for shader in &shaders {
                    assert_eq!(shader.compile_spirv().unwrap().words[0], 0x0723_0203);
                }
            }
        }
    }

    #[test]
    fn r2r_forced_two_upload_direct_rader_high_recursive_low_fuses_both_boundaries() {
        let length = 11usize * 17 * 47;
        let device = DeviceProfile {
            shared_memory_bytes: 8 * 1024,
            shared_memory_pow2_bytes: 8 * 1024,
            max_threads_per_block: 1024,
            max_workgroup_size: [1024, 1024, 64],
            supports_f64: true,
            ..DeviceProfile::generic(crate::Backend::Vulkan, crate::GpuVendor::Nvidia)
        };
        for precision in [Precision::DoubleDouble, Precision::DoubleDoubleF64Storage] {
            for direction in [Direction::Forward, Direction::Inverse] {
                let mut tuning = PlannerTuning::portable();
                tuning.min_rader_direct_prime = 11;
                tuning.max_rader_direct_prime = 89;
                tuning.validate().unwrap();
                let transform = crate::TransformIr::build(
                    FftConfig::new(vec![length])
                        .with_transform(TransformKind::Dct(DctType::II))
                        .with_precision(precision)
                        .with_inverse_normalization(direction == Direction::Inverse)
                        .with_tuning(tuning),
                    direction,
                    device,
                )
                .unwrap();
                let crate::TransformIr::RealToRealDoubleDouble(r2r) = transform else {
                    panic!("DD N8789 DCT-II/III probe did not build 1D R2R");
                };
                let DoubleDoubleR2rAlgorithm::FftReduction { fft, .. } = &r2r.algorithm else {
                    panic!("DD N8789 DCT-II/III must use FFT reduction");
                };
                let DoubleDoubleOneDimIr::Recursive(recursive) = fft.as_ref() else {
                    panic!("DD N8789 DCT-II/III must retain recursive child");
                };
                assert_eq!(
                    recursive
                        .rader_forced_upload_schedule
                        .as_ref()
                        .expect("DD N8789 forced-Rader schedule")
                        .axis_split,
                    vec![187, 47]
                );
                let mapped_high = recursive
                    .forced_rader_two_upload_mapped_high_component()
                    .unwrap()
                    .expect("DD N8789 mapped Direct-Rader high boundary");
                let crate::DoubleDoubleRecursiveFftNodeIr::DirectRader(high) = mapped_high else {
                    panic!("DD N8789 high upload must remain p47 Direct-Rader");
                };
                assert_eq!(high.prime, 47);
                let crate::StockhamIoMapping::FourStepRight(mapping) = high.io_mapping else {
                    panic!("DD N8789 p47 high upload must own FourStepRight input");
                };
                assert_eq!(mapping.logical_len, length);
                assert_eq!([mapping.left_len, mapping.right_len], [187, 47]);
                let high_block = high
                    .axis_batch_block
                    .expect("DD N8789 p47 high caller block");
                assert_eq!(high_block.threads_per_transform, 24);
                assert_eq!(high_block.grouped_batch, 4);
                assert_eq!([high_block.local_size_x, high_block.local_size_y], [4, 24]);
                assert!(high_block.transforms_on_x);
                assert!(!high_block.axis_swapped);

                let mapped_low = recursive
                    .forced_rader_two_upload_mapped_low_component()
                    .unwrap()
                    .expect("DD N8789 mapped recursive low boundary");
                let crate::DoubleDoubleRecursiveFftNodeIr::CooleyTukey(low) = mapped_low else {
                    panic!("DD N8789 low upload must remain recursive N187 Cooley-Tukey");
                };
                assert_eq!(low.logical_len, 187);
                assert!(matches!(
                    low.scatter_output.output_modifier,
                    crate::double_double_recursive_ir::DoubleDoubleCooleyTukeyOutputModifier::FourStepLeft(low_mapping)
                        if low_mapping == mapping
                ));

                let child_program = crate::ProgramIr::double_double_recursive(recursive).unwrap();
                let program = crate::ProgramIr::double_double_r2r(&r2r).unwrap();
                assert_eq!(program.passes.len(), child_program.passes.len());
                assert!(
                    program
                        .passes
                        .first()
                        .unwrap()
                        .name
                        .contains("forced_rader_two_upload_1")
                );
                assert!(
                    program
                        .passes
                        .last()
                        .unwrap()
                        .name
                        .contains("forced_rader_two_upload_0_recursive")
                );
                let expected_scalar = match precision {
                    Precision::DoubleDouble => crate::kernel_ir::ScalarType::DoubleDouble,
                    Precision::DoubleDoubleF64Storage => crate::kernel_ir::ScalarType::F64,
                    _ => unreachable!(),
                };
                assert_eq!(program.input_resource().unwrap().scalar, expected_scalar);
                assert_eq!(program.output_resource().unwrap().scalar, expected_scalar);
                assert!(program.resources.iter().any(|resource| {
                    resource.name == "double_double_rader_four_step_exchange"
                        && resource.kind == crate::program_ir::ProgramResourceKind::Scratch
                        && resource.scalar == crate::kernel_ir::ScalarType::DoubleDouble
                }));
                assert!(program.resources.iter().all(|resource| {
                    !resource.name.contains("double_double_r2r_fft_input")
                        && !resource.name.contains("double_double_r2r_fft_output")
                }));
                let first_bindings = &program.passes.first().unwrap().bindings;
                assert!(first_bindings.iter().any(|binding| binding.binding == 2));
                assert!(first_bindings.iter().any(|binding| binding.binding == 3));
                if direction == Direction::Inverse {
                    assert!(first_bindings.iter().any(|binding| binding.binding == 4));
                }

                let shaders = crate::backend::vulkan::VulkanGlslBackend
                    .lower_double_double_r2r(&r2r)
                    .unwrap();
                assert_eq!(shaders.len(), program.passes.len());
                let first = &shaders.first().unwrap().glsl;
                assert!(first.contains("DoubleDoubleDirectRaderIr"));
                match precision {
                    Precision::DoubleDouble => assert!(first.contains("dvec2 data[]")),
                    Precision::DoubleDoubleF64Storage => assert!(first.contains("double data[]")),
                    _ => unreachable!(),
                }
                match direction {
                    Direction::Forward => {
                        assert!(first.contains("_logical_source"));
                        let last = &shaders.last().unwrap().glsl;
                        assert!(last.contains("vkfft_r2r_destination"));
                        assert!(last.contains("vkfft_r2r_phases"));
                        assert!(last.contains("binding = 2"));
                    }
                    Direction::Inverse => {
                        assert!(first.contains("_a_index"));
                        assert!(first.contains("vkfft_r2r_phases"));
                        assert!(first.contains("binding = 4"));
                        let last = &shaders.last().unwrap().glsl;
                        assert!(last.contains("logical_output"));
                        assert!(last.contains("vkfft_r2r_destination"));
                    }
                }
                for shader in &shaders {
                    assert_eq!(shader.compile_spirv().unwrap().words[0], 0x0723_0203);
                }
            }
        }
    }

    #[test]
    fn r2r_forced_two_upload_fft_rader_high_recursive_low_fuses_both_boundaries() {
        let length = 11usize * 17 * 31;
        let device = DeviceProfile {
            shared_memory_bytes: 24 * 1024,
            shared_memory_pow2_bytes: 24 * 1024,
            max_threads_per_block: 128,
            max_workgroup_size: [1024, 1024, 64],
            supports_f64: true,
            ..DeviceProfile::generic(crate::Backend::Vulkan, crate::GpuVendor::Nvidia)
        };
        for precision in [Precision::DoubleDouble, Precision::DoubleDoubleF64Storage] {
            for direction in [Direction::Forward, Direction::Inverse] {
                let transform = crate::TransformIr::build(
                    FftConfig::new(vec![length])
                        .with_transform(TransformKind::Dct(DctType::II))
                        .with_precision(precision)
                        .with_inverse_normalization(direction == Direction::Inverse),
                    direction,
                    device,
                )
                .unwrap();
                let crate::TransformIr::RealToRealDoubleDouble(r2r) = transform else {
                    panic!("DD N5797 DCT-II/III probe did not build 1D R2R");
                };
                let DoubleDoubleR2rAlgorithm::FftReduction { fft, .. } = &r2r.algorithm else {
                    panic!("DD N5797 DCT-II/III must use FFT reduction");
                };
                let DoubleDoubleOneDimIr::Recursive(recursive) = fft.as_ref() else {
                    panic!("DD N5797 DCT-II/III must retain recursive child");
                };
                assert_eq!(
                    recursive
                        .rader_forced_upload_schedule
                        .as_ref()
                        .expect("DD N5797 forced-Rader schedule")
                        .axis_split,
                    vec![187, 31]
                );
                let mapped_high = recursive
                    .forced_rader_two_upload_mapped_high_component()
                    .unwrap()
                    .expect("DD N5797 mapped FFT-Rader high boundary");
                let crate::DoubleDoubleRecursiveFftNodeIr::FftRader(high) = mapped_high else {
                    panic!("DD N5797 high upload must remain p31 FFT-Rader");
                };
                assert_eq!(high.prime, 31);
                let crate::StockhamIoMapping::FourStepRight(mapping) = high.io_mapping else {
                    panic!("DD N5797 p31 high upload must own FourStepRight input");
                };
                assert_eq!(mapping.logical_len, length);
                assert_eq!([mapping.left_len, mapping.right_len], [187, 31]);
                let high_block = high
                    .caller_axis_batch_block
                    .expect("DD N5797 p31 high caller block");
                assert_eq!(high_block.threads_per_transform, 7);
                assert_eq!(high_block.grouped_batch, 16);
                assert_eq!([high_block.local_size_x, high_block.local_size_y], [16, 7]);
                assert!(high_block.transforms_on_x);
                assert!(!high_block.axis_swapped);

                let mapped_low = recursive
                    .forced_rader_two_upload_mapped_low_component()
                    .unwrap()
                    .expect("DD N5797 mapped recursive low boundary");
                let crate::DoubleDoubleRecursiveFftNodeIr::CooleyTukey(low) = mapped_low else {
                    panic!("DD N5797 low upload must remain recursive N187 Cooley-Tukey");
                };
                assert_eq!(low.logical_len, 187);
                assert!(matches!(
                    low.scatter_output.output_modifier,
                    crate::double_double_recursive_ir::DoubleDoubleCooleyTukeyOutputModifier::FourStepLeft(low_mapping)
                        if low_mapping == mapping
                ));

                let child_program = crate::ProgramIr::double_double_recursive(recursive).unwrap();
                let program = crate::ProgramIr::double_double_r2r(&r2r).unwrap();
                assert_eq!(program.passes.len(), child_program.passes.len());
                let expected_scalar = match precision {
                    Precision::DoubleDouble => crate::kernel_ir::ScalarType::DoubleDouble,
                    Precision::DoubleDoubleF64Storage => crate::kernel_ir::ScalarType::F64,
                    _ => unreachable!(),
                };
                assert_eq!(program.input_resource().unwrap().scalar, expected_scalar);
                assert_eq!(program.output_resource().unwrap().scalar, expected_scalar);
                assert!(program.resources.iter().any(|resource| {
                    resource.name == "double_double_rader_four_step_exchange"
                        && resource.kind == crate::program_ir::ProgramResourceKind::Scratch
                        && resource.scalar == crate::kernel_ir::ScalarType::DoubleDouble
                }));
                assert!(program.resources.iter().all(|resource| {
                    !resource.name.contains("double_double_r2r_fft_input")
                        && !resource.name.contains("double_double_r2r_fft_output")
                }));
                let input_id = program.input_resource().unwrap().id;
                let first = program.passes.first().unwrap();
                assert!(
                    first
                        .bindings
                        .iter()
                        .any(|binding| { binding.binding == 0 && binding.resource == input_id })
                );
                let scatter_suffix = format!("{}_scatter", high.name);
                let high_scatter = program
                    .passes
                    .iter()
                    .find(|pass| pass.name.ends_with(&scatter_suffix))
                    .expect("DD N5797 high FFT-Rader scatter pass");
                assert!(
                    high_scatter
                        .bindings
                        .iter()
                        .any(|binding| { binding.binding == 0 && binding.resource == input_id })
                );
                assert!(
                    high_scatter
                        .bindings
                        .iter()
                        .any(|binding| binding.binding == 2)
                );
                assert!(
                    high_scatter
                        .bindings
                        .iter()
                        .any(|binding| binding.binding == 3)
                );
                let last = program.passes.last().unwrap();
                assert!(last.name.contains("forced_rader_two_upload_0_recursive"));
                match direction {
                    Direction::Forward => {
                        assert!(!first.bindings.iter().any(|binding| binding.binding == 2));
                        assert!(
                            !high_scatter
                                .bindings
                                .iter()
                                .any(|binding| binding.binding == 4)
                        );
                        assert!(last.bindings.iter().any(|binding| binding.binding == 2));
                    }
                    Direction::Inverse => {
                        assert!(first.bindings.iter().any(|binding| binding.binding == 2));
                        assert!(
                            high_scatter
                                .bindings
                                .iter()
                                .any(|binding| binding.binding == 4)
                        );
                    }
                }

                let shaders = crate::backend::vulkan::VulkanGlslBackend
                    .lower_double_double_r2r(&r2r)
                    .unwrap();
                assert_eq!(shaders.len(), program.passes.len());
                let gather = shaders
                    .iter()
                    .find(|shader| shader.glsl.contains("DD FFT-Rader generator gather"))
                    .expect("DD N5797 R2R FFT-Rader gather shader");
                let scatter = shaders
                    .iter()
                    .find(|shader| shader.glsl.contains("DD FFT-Rader natural-order scatter"))
                    .expect("DD N5797 R2R FFT-Rader scatter shader");
                match precision {
                    Precision::DoubleDouble => {
                        assert!(gather.glsl.contains("VkFftInput { dvec2 data[]; }"));
                        assert!(scatter.glsl.contains("VkFftInput { dvec2 data[]; }"));
                    }
                    Precision::DoubleDoubleF64Storage => {
                        assert!(gather.glsl.contains("VkFftInput { double data[]; }"));
                        assert!(scatter.glsl.contains("VkFftInput { double data[]; }"));
                    }
                    _ => unreachable!(),
                }
                match direction {
                    Direction::Forward => {
                        assert!(gather.glsl.contains("_logical_source"));
                        assert!(scatter.glsl.contains("_logical_source"));
                        assert!(!gather.glsl.contains("VkFftDdR2rPhases"));
                        assert!(!scatter.glsl.contains("VkFftDdR2rPhases"));
                        let low_output = &shaders.last().unwrap().glsl;
                        assert!(low_output.contains("vkfft_r2r_phases"));
                        assert!(low_output.contains("binding = 2"));
                    }
                    Direction::Inverse => {
                        assert!(gather.glsl.contains("_a_index"));
                        assert!(scatter.glsl.contains("_a_index"));
                        assert!(gather.glsl.contains("VkFftDdR2rPhases"));
                        assert!(gather.glsl.contains("binding = 2"));
                        assert!(scatter.glsl.contains("VkFftDdR2rPhases"));
                        assert!(scatter.glsl.contains("binding = 4"));
                    }
                }
                for shader in &shaders {
                    assert_eq!(shader.compile_spirv().unwrap().words[0], 0x0723_0203);
                }
            }
        }
    }

    #[test]
    fn r2r_forced_two_upload_stockham_high_direct_rader_low_fuses_both_boundaries() {
        let length = 345usize;
        let device = DeviceProfile {
            shared_memory_bytes: 8 * 1024,
            shared_memory_pow2_bytes: 8 * 1024,
            max_threads_per_block: 1024,
            max_workgroup_size: [1024, 1024, 64],
            supports_f64: true,
            ..DeviceProfile::generic(crate::Backend::Vulkan, crate::GpuVendor::Nvidia)
        };
        for precision in [Precision::DoubleDouble, Precision::DoubleDoubleF64Storage] {
            for direction in [Direction::Forward, Direction::Inverse] {
                let transform = crate::TransformIr::build(
                    FftConfig::new(vec![length])
                        .with_transform(TransformKind::Dct(DctType::II))
                        .with_precision(precision)
                        .with_inverse_normalization(direction == Direction::Inverse),
                    direction,
                    device,
                )
                .unwrap();
                let crate::TransformIr::RealToRealDoubleDouble(r2r) = transform else {
                    panic!("DD N345 DCT-II/III probe did not build 1D R2R");
                };
                let DoubleDoubleR2rAlgorithm::FftReduction { fft, .. } = &r2r.algorithm else {
                    panic!("DD N345 DCT-II/III must use FFT reduction");
                };
                let DoubleDoubleOneDimIr::Recursive(recursive) = fft.as_ref() else {
                    panic!("DD N345 DCT-II/III must retain recursive child");
                };
                assert_eq!(
                    recursive
                        .rader_forced_upload_schedule
                        .as_ref()
                        .expect("DD N345 forced-Rader schedule")
                        .axis_split,
                    vec![23, 15]
                );
                let (high, mapping) = recursive
                    .forced_rader_two_upload_mapped_high_stockham()
                    .unwrap()
                    .expect("DD N345 mapped N15 Stockham high boundary");
                assert_eq!(high.sequence_len, 15);
                assert_eq!([mapping.left_len, mapping.right_len], [23, 15]);
                assert_eq!(mapping.logical_len, length);
                let high_block = high.axis_batch_block.expect("DD N345 N15 high block");
                assert_eq!(high_block.threads_per_transform, 5);
                assert_eq!(high_block.grouped_batch, 4);
                assert_eq!([high_block.local_size_x, high_block.local_size_y], [4, 5]);
                assert!(high_block.transforms_on_x);
                assert!(!high_block.axis_swapped);

                let mapped_low = recursive
                    .forced_rader_two_upload_mapped_low_component()
                    .unwrap()
                    .expect("DD N345 mapped p23 Direct-Rader low boundary");
                let crate::DoubleDoubleRecursiveFftNodeIr::DirectRader(low) = mapped_low else {
                    panic!("DD N345 low upload must remain p23 Direct-Rader");
                };
                assert_eq!(low.prime, 23);
                assert!(matches!(
                    low.io_mapping,
                    crate::StockhamIoMapping::FourStepLeft(low_mapping) if low_mapping == mapping
                ));
                let low_block = low.axis_batch_block.expect("DD N345 p23 low caller block");
                assert_eq!(low_block.threads_per_transform, 12);
                assert_eq!(low_block.grouped_batch, 4);
                assert_eq!([low_block.local_size_x, low_block.local_size_y], [12, 4]);
                assert!(!low_block.transforms_on_x);
                assert!(!low_block.axis_swapped);

                let child_program = crate::ProgramIr::double_double_recursive(recursive).unwrap();
                let program = crate::ProgramIr::double_double_r2r(&r2r).unwrap();
                assert_eq!(program.passes.len(), child_program.passes.len());
                let expected_scalar = match precision {
                    Precision::DoubleDouble => crate::kernel_ir::ScalarType::DoubleDouble,
                    Precision::DoubleDoubleF64Storage => crate::kernel_ir::ScalarType::F64,
                    _ => unreachable!(),
                };
                assert_eq!(program.input_resource().unwrap().scalar, expected_scalar);
                assert_eq!(program.output_resource().unwrap().scalar, expected_scalar);
                assert!(program.resources.iter().any(|resource| {
                    resource.name == "double_double_rader_four_step_exchange"
                        && resource.kind == crate::program_ir::ProgramResourceKind::Scratch
                        && resource.scalar == crate::kernel_ir::ScalarType::DoubleDouble
                }));
                assert!(program.resources.iter().all(|resource| {
                    !resource.name.contains("double_double_r2r_fft_input")
                        && !resource.name.contains("double_double_r2r_fft_output")
                }));
                let first = program.passes.first().unwrap();
                let last = program.passes.last().unwrap();
                assert!(first.name.contains("forced_rader_two_upload_1"));
                assert!(last.name.contains("forced_rader_two_upload_0_recursive"));
                assert!(last.bindings.iter().any(|binding| binding.binding == 2));
                match direction {
                    Direction::Forward => {
                        assert!(last.bindings.iter().any(|binding| binding.binding == 3));
                    }
                    Direction::Inverse => {
                        assert!(first.bindings.iter().any(|binding| binding.binding == 4));
                        assert!(!last.bindings.iter().any(|binding| binding.binding == 3));
                    }
                }

                let shaders = crate::backend::vulkan::VulkanGlslBackend
                    .lower_double_double_r2r(&r2r)
                    .unwrap();
                assert_eq!(shaders.len(), program.passes.len());
                let high_shader = &shaders.first().unwrap().glsl;
                let low_shader = &shaders.last().unwrap().glsl;
                assert!(low_shader.contains("DoubleDoubleDirectRaderIr"));
                match precision {
                    Precision::DoubleDouble => {
                        assert!(high_shader.contains("VkFftInput { dvec2 data[]; }"));
                        assert!(low_shader.contains("VkFftOutput { dvec2 data[]; }"));
                    }
                    Precision::DoubleDoubleF64Storage => {
                        assert!(high_shader.contains("VkFftInput { double data[]; }"));
                        assert!(low_shader.contains("VkFftOutput { double data[]; }"));
                    }
                    _ => unreachable!(),
                }
                match direction {
                    Direction::Forward => {
                        assert!(high_shader.contains("logical_source"));
                        assert!(low_shader.contains("VkFftDdR2rPhases"));
                        assert!(low_shader.contains("binding = 3"));
                        assert!(low_shader.contains("vkfft_r2r_dc_destination"));
                        assert!(low_shader.contains("vkfft_r2r_output_destination"));
                    }
                    Direction::Inverse => {
                        assert!(high_shader.contains("VkFftDdR2rPhases"));
                        assert!(high_shader.contains("binding = 4"));
                        assert!(!low_shader.contains("VkFftDdR2rPhases"));
                        assert!(low_shader.contains("vkfft_r2r_dc_logical_output"));
                        assert!(low_shader.contains("vkfft_r2r_output_logical_output"));
                    }
                }
                for shader in &shaders {
                    assert_eq!(shader.compile_spirv().unwrap().words[0], 0x0723_0203);
                }
            }
        }
    }

    #[test]
    fn r2r_forced_two_upload_stockham_high_fft_rader_low_fuses_output_boundary() {
        let length = 285usize;
        let device = DeviceProfile {
            shared_memory_bytes: 8 * 1024,
            shared_memory_pow2_bytes: 8 * 1024,
            max_threads_per_block: 1024,
            max_workgroup_size: [1024, 1024, 64],
            supports_f64: true,
            ..DeviceProfile::generic(crate::Backend::Vulkan, crate::GpuVendor::Nvidia)
        };
        for precision in [Precision::DoubleDouble, Precision::DoubleDoubleF64Storage] {
            for direction in [Direction::Forward, Direction::Inverse] {
                let transform = crate::TransformIr::build(
                    FftConfig::new(vec![length])
                        .with_transform(TransformKind::Dct(DctType::II))
                        .with_precision(precision)
                        .with_inverse_normalization(direction == Direction::Inverse),
                    direction,
                    device,
                )
                .unwrap();
                let crate::TransformIr::RealToRealDoubleDouble(r2r) = transform else {
                    panic!("DD N285 DCT-II/III probe did not build 1D R2R");
                };
                let DoubleDoubleR2rAlgorithm::FftReduction { fft, .. } = &r2r.algorithm else {
                    panic!("DD N285 DCT-II/III must use FFT reduction");
                };
                let DoubleDoubleOneDimIr::Recursive(recursive) = fft.as_ref() else {
                    panic!("DD N285 DCT-II/III must retain recursive child");
                };
                assert_eq!(
                    recursive
                        .rader_forced_upload_schedule
                        .as_ref()
                        .expect("DD N285 forced-Rader schedule")
                        .axis_split,
                    vec![19, 15]
                );
                let (high, mapping) = recursive
                    .forced_rader_two_upload_mapped_high_stockham()
                    .unwrap()
                    .expect("DD N285 mapped N15 Stockham high boundary");
                assert_eq!(high.sequence_len, 15);
                assert_eq!([mapping.left_len, mapping.right_len], [19, 15]);
                assert_eq!(mapping.logical_len, length);
                assert!(high.axis_batch_block.is_some());

                let mapped_low = recursive
                    .forced_rader_two_upload_mapped_low_component()
                    .unwrap()
                    .expect("DD N285 mapped p19 FFT-Rader low boundary");
                let crate::DoubleDoubleRecursiveFftNodeIr::FftRader(low) = mapped_low else {
                    panic!("DD N285 low upload must remain p19 FFT-Rader");
                };
                assert_eq!(low.prime, 19);
                assert!(matches!(
                    low.io_mapping,
                    crate::StockhamIoMapping::FourStepLeft(low_mapping) if low_mapping == mapping
                ));
                assert!(low.caller_axis_batch_block.is_some());

                let child_program = crate::ProgramIr::double_double_recursive(recursive).unwrap();
                let program = crate::ProgramIr::double_double_r2r(&r2r).unwrap();
                assert_eq!(program.passes.len(), child_program.passes.len());
                let expected_scalar = match precision {
                    Precision::DoubleDouble => crate::kernel_ir::ScalarType::DoubleDouble,
                    Precision::DoubleDoubleF64Storage => crate::kernel_ir::ScalarType::F64,
                    _ => unreachable!(),
                };
                assert_eq!(program.input_resource().unwrap().scalar, expected_scalar);
                assert_eq!(program.output_resource().unwrap().scalar, expected_scalar);
                assert!(program.resources.iter().all(|resource| {
                    !resource.name.contains("double_double_r2r_fft_input")
                        && !resource.name.contains("double_double_r2r_fft_output")
                }));
                let first = program.passes.first().unwrap();
                let last = program.passes.last().unwrap();
                assert!(first.name.contains("forced_rader_two_upload_1"));
                assert!(last.name.contains("forced_rader_two_upload_0_recursive"));
                assert!(last.bindings.iter().any(|binding| binding.binding == 2));
                match direction {
                    Direction::Forward => {
                        assert!(last.bindings.iter().any(|binding| binding.binding == 3));
                    }
                    Direction::Inverse => {
                        assert!(first.bindings.iter().any(|binding| binding.binding == 4));
                        assert!(!last.bindings.iter().any(|binding| binding.binding == 3));
                    }
                }

                let shaders = crate::backend::vulkan::VulkanGlslBackend
                    .lower_double_double_r2r(&r2r)
                    .unwrap();
                assert_eq!(shaders.len(), program.passes.len());
                let low_shader = shaders
                    .iter()
                    .find(|shader| shader.glsl.contains("DD FFT-Rader natural-order scatter"))
                    .expect("DD N285 FFT-Rader low scatter shader");
                match precision {
                    Precision::DoubleDouble => {
                        assert!(low_shader.glsl.contains("VkFftOutput { dvec2 data[]; }"));
                    }
                    Precision::DoubleDoubleF64Storage => {
                        assert!(low_shader.glsl.contains("VkFftOutput { double data[]; }"));
                    }
                    _ => unreachable!(),
                }
                match direction {
                    Direction::Forward => {
                        assert!(low_shader.glsl.contains("VkFftDdR2rPhases"));
                        assert!(low_shader.glsl.contains("binding = 3"));
                        assert!(low_shader.glsl.contains("vkfft_r2r_dc_output_destination"));
                        assert!(low_shader.glsl.contains("vkfft_r2r_output_destination"));
                    }
                    Direction::Inverse => {
                        assert!(!low_shader.glsl.contains("VkFftDdR2rPhases"));
                        assert!(
                            low_shader
                                .glsl
                                .contains("vkfft_r2r_dc_output_logical_output")
                        );
                        assert!(low_shader.glsl.contains("vkfft_r2r_output_logical_output"));
                    }
                }
                for shader in &shaders {
                    assert_eq!(shader.compile_spirv().unwrap().words[0], 0x0723_0203);
                }
            }
        }
    }

    #[test]
    fn nd_real_and_r2r_propagate_intel_strided_dd_bandwidth_context() {
        let device = DeviceProfile {
            shared_memory_bytes: 32 * 1024,
            shared_memory_pow2_bytes: 32 * 1024,
            max_threads_per_block: 1024,
            max_workgroup_size: [1024, 1024, 64],
            supports_f64: true,
            ..DeviceProfile::generic(crate::Backend::Vulkan, crate::GpuVendor::Intel)
        };

        let real_config = FftConfig::new(vec![1_024usize, 2usize])
            .with_transform(TransformKind::RealToComplex)
            .with_precision(Precision::DoubleDouble)
            .resolve_tuning_for_device(device);
        let real_plan = FftPlan::build_for_device(real_config, device).unwrap();
        let real = DoubleDoubleNdRealFftIr::build_for_device(&real_plan, device)
            .unwrap()
            .with_grouped_stockham_axis_blocks(None, device)
            .unwrap();
        let real_outer = real
            .complex_axes
            .iter()
            .find(|axis| axis.axis == 0)
            .unwrap();
        let DoubleDoubleOneDimIr::Stockham(real_child) = &real_outer.transform else {
            panic!(
                "Intel ND-real strided DD N1024 axis should retain upstream single-upload Stockham"
            );
        };
        assert_eq!(real_child.sequence_len, 1_024);
        let real_block = real_child
            .axis_batch_block
            .expect("Intel ND-real strided DD N1024 should retain a higher-axis block");
        assert!(real_block.transforms_on_x);
        assert!(!real_block.axis_swapped);
        assert_eq!([real_block.local_size_x, real_block.local_size_y], [1, 128]);
        let real_program = crate::ProgramIr::double_double_nd_real(&real).unwrap();
        real_program.validate().unwrap();
        let real_shaders = crate::backend::vulkan::VulkanGlslBackend
            .lower_double_double_nd_real(&real)
            .unwrap();
        assert_eq!(real_shaders.len(), real_program.passes.len());
        for shader in &real_shaders {
            assert_eq!(shader.compile_spirv().unwrap().words[0], 0x0723_0203);
        }
        let mut real_input = vec![DoubleDouble::ZERO; real.full_tensor_len];
        real_input[0] = DoubleDouble::ONE;
        let real_spectrum = execute_double_double_nd_r2c_ir(&real, &real_input).unwrap();
        let expected = ComplexDoubleDouble::new(DoubleDouble::ONE, DoubleDouble::ZERO);
        let real_impulse_error = real_spectrum
            .iter()
            .copied()
            .map(|actual| dd_error(actual, expected))
            .fold(0.0f64, f64::max);
        assert!(
            real_impulse_error < 5.0e-20,
            "Intel ND-real strided DD N1024 impulse error {real_impulse_error:e}"
        );

        let real_boosted_config = FftConfig::new(vec![1_024usize, 2usize])
            .with_transform(TransformKind::RealToComplex)
            .with_precision(Precision::DoubleDouble)
            .with_bandwidth_boost(2)
            .resolve_tuning_for_device(device);
        let real_boosted_plan = FftPlan::build_for_device(real_boosted_config, device).unwrap();
        let real_boosted =
            DoubleDoubleNdRealFftIr::build_for_device(&real_boosted_plan, device).unwrap();
        let real_boosted_outer = real_boosted
            .complex_axes
            .iter()
            .find(|axis| axis.axis == 0)
            .unwrap();
        assert!(matches!(
            real_boosted_outer.transform,
            DoubleDoubleOneDimIr::Stockham(_)
        ));
        real_boosted.validate().unwrap();

        let transform = TransformKind::Dct(DctType::II);
        let r2r_config = FftConfig::new(vec![512usize, 2usize])
            .with_transform(transform)
            .with_precision(Precision::DoubleDouble)
            .resolve_tuning_for_device(device);
        let r2r_plan = FftPlan::build_for_device(r2r_config, device).unwrap();
        let r2r = DoubleDoubleNdR2rIr::build_for_device(&r2r_plan, Direction::Forward, device)
            .unwrap()
            .with_grouped_stockham_axis_blocks(None, device)
            .unwrap();
        let r2r_outer = r2r.axes.iter().find(|axis| axis.axis == 0).unwrap();
        let DoubleDoubleR2rAlgorithm::FftReduction { fft, fft_len, .. } =
            &r2r_outer.transform.algorithm
        else {
            panic!("Intel ND DCT-II outer axis should use FFT reduction");
        };
        assert_eq!(*fft_len, 512);
        let DoubleDoubleOneDimIr::Stockham(r2r_child) = fft.as_ref() else {
            panic!(
                "Intel ND DCT-II strided N512 reduction should retain upstream single-upload Stockham"
            );
        };
        assert_eq!(r2r_child.sequence_len, 512);
        let r2r_block = r2r_child
            .axis_batch_block
            .expect("Intel ND DCT-II strided N512 should retain a higher-axis block");
        assert!(r2r_block.transforms_on_x);
        assert!(!r2r_block.axis_swapped);
        assert_eq!([r2r_block.local_size_x, r2r_block.local_size_y], [2, 64]);
        assert_eq!(r2r_outer.grouped_batch_override, None);
        let r2r_program = crate::ProgramIr::double_double_nd_r2r(&r2r).unwrap();
        r2r_program.validate().unwrap();
        let r2r_shaders = crate::backend::vulkan::VulkanGlslBackend
            .lower_double_double_nd_r2r(&r2r)
            .unwrap();
        assert_eq!(r2r_shaders.len(), r2r_program.passes.len());
        for shader in &r2r_shaders {
            assert_eq!(shader.compile_spirv().unwrap().words[0], 0x0723_0203);
        }

        let r2r_boosted_config = FftConfig::new(vec![512usize, 2usize])
            .with_transform(transform)
            .with_precision(Precision::DoubleDouble)
            .with_bandwidth_boost(2)
            .resolve_tuning_for_device(device);
        let r2r_boosted_plan = FftPlan::build_for_device(r2r_boosted_config, device).unwrap();
        let r2r_boosted =
            DoubleDoubleNdR2rIr::build_for_device(&r2r_boosted_plan, Direction::Forward, device)
                .unwrap();
        let boosted_outer = r2r_boosted.axes.iter().find(|axis| axis.axis == 0).unwrap();
        let DoubleDoubleR2rAlgorithm::FftReduction { fft, fft_len, .. } =
            &boosted_outer.transform.algorithm
        else {
            panic!("boosted Intel ND DCT-II outer axis should keep FFT reduction");
        };
        assert_eq!(*fft_len, 512);
        assert!(matches!(fft.as_ref(), DoubleDoubleOneDimIr::Stockham(_)));
        r2r_boosted.validate().unwrap();
    }

    #[test]
    fn real_device_aware_children_use_large_stockham_upload_schedule() {
        let device = DeviceProfile {
            shared_memory_bytes: 48 * 1024,
            shared_memory_pow2_bytes: 32 * 1024,
            max_threads_per_block: 1024,
            max_workgroup_size: [1024, 1024, 64],
            supports_f64: true,
            ..DeviceProfile::generic(crate::Backend::Vulkan, crate::GpuVendor::Nvidia)
        };

        let one_dim_config = FftConfig::new(vec![8_232usize])
            .with_transform(TransformKind::RealToComplex)
            .with_precision(Precision::DoubleDouble)
            .resolve_tuning_for_device(device);
        let one_dim_plan = FftPlan::build_for_device(one_dim_config.clone(), device).unwrap();
        let portable = DoubleDoubleRealFftIr::build(&one_dim_plan).unwrap();
        let DoubleDoubleOneDimIr::Recursive(portable_child) = &portable.transform else {
            panic!("portable real N8232 half-size child should be recursive N4116");
        };
        assert!(portable_child.stockham_upload_schedule.is_none());
        let scheduled = DoubleDoubleRealFftIr::build_for_device(&one_dim_plan, device)
            .unwrap()
            .with_grouped_stockham_child_block(device)
            .unwrap();
        let DoubleDoubleOneDimIr::Recursive(child) = &scheduled.transform else {
            panic!("device-aware real N8232 half-size child should be recursive N4116");
        };
        assert_eq!(child.logical_len, 4_116);
        assert_eq!(
            child
                .stockham_upload_schedule
                .as_ref()
                .expect("real N8232 child should retain device upload schedule")
                .axis_split,
            vec![84, 49]
        );
        assert!(child.two_upload_four_step_plan.is_some());
        let high_level =
            crate::TransformIr::build(one_dim_config, Direction::Forward, device).unwrap();
        let crate::TransformIr::RealDoubleDouble(high_level) = high_level else {
            panic!("high-level real N8232 should use DD real IR");
        };
        let DoubleDoubleOneDimIr::Recursive(high_child) = &high_level.transform else {
            panic!("high-level real N8232 child should be recursive N4116");
        };
        assert_eq!(
            high_child
                .stockham_upload_schedule
                .as_ref()
                .expect("high-level real N8232 upload schedule")
                .axis_split,
            vec![84, 49]
        );

        let nd_config = FftConfig::new(vec![4_116usize, 2])
            .with_transform(TransformKind::RealToComplex)
            .with_precision(Precision::DoubleDouble)
            .resolve_tuning_for_device(device);
        let nd_plan = FftPlan::build_for_device(nd_config.clone(), device).unwrap();
        let portable_nd = DoubleDoubleNdRealFftIr::build(&nd_plan).unwrap();
        let portable_axis = portable_nd
            .complex_axes
            .iter()
            .find(|axis| axis.axis == 0)
            .unwrap();
        let DoubleDoubleOneDimIr::Recursive(portable_child) = &portable_axis.transform else {
            panic!("portable ND-real higher N4116 axis should be recursive");
        };
        assert!(portable_child.stockham_upload_schedule.is_none());
        let scheduled_nd = DoubleDoubleNdRealFftIr::build_for_device(&nd_plan, device)
            .unwrap()
            .with_grouped_stockham_axis_blocks(None, device)
            .unwrap();
        let axis = scheduled_nd
            .complex_axes
            .iter()
            .find(|axis| axis.axis == 0)
            .unwrap();
        let DoubleDoubleOneDimIr::Recursive(child) = &axis.transform else {
            panic!("device-aware ND-real higher N4116 axis should be recursive");
        };
        assert_eq!(
            child
                .stockham_upload_schedule
                .as_ref()
                .expect("ND-real higher N4116 upload schedule")
                .axis_split,
            vec![84, 49]
        );
        assert!(child.two_upload_four_step_plan.is_some());
        let high_level = crate::TransformIr::build(nd_config, Direction::Forward, device).unwrap();
        let crate::TransformIr::RealNdDoubleDouble(high_level) = high_level else {
            panic!("high-level [4116,2] R2C should use DD ND-real IR");
        };
        let high_axis = high_level
            .complex_axes
            .iter()
            .find(|axis| axis.axis == 0)
            .unwrap();
        let DoubleDoubleOneDimIr::Recursive(high_child) = &high_axis.transform else {
            panic!("high-level ND-real higher N4116 axis should be recursive");
        };
        assert_eq!(
            high_child
                .stockham_upload_schedule
                .as_ref()
                .expect("high-level ND-real higher N4116 upload schedule")
                .axis_split,
            vec![84, 49]
        );

        let nd_f64 = crate::TransformIr::build(
            FftConfig::new(vec![4_116usize, 2])
                .with_transform(TransformKind::RealToComplex)
                .with_precision(Precision::DoubleDoubleF64Storage)
                .resolve_tuning_for_device(device),
            Direction::Forward,
            device,
        )
        .unwrap();
        let crate::TransformIr::RealNdDoubleDouble(nd_f64) = nd_f64 else {
            panic!("high-level DD/F64 [4116,2] R2C should use ND-real IR");
        };
        assert_eq!(nd_f64.external_storage, PrecisionStorage::F64);
        let f64_axis = nd_f64
            .complex_axes
            .iter()
            .find(|axis| axis.axis == 0)
            .unwrap();
        let DoubleDoubleOneDimIr::Recursive(f64_child) = &f64_axis.transform else {
            panic!("DD/F64 ND-real N4116 axis should be recursive");
        };
        assert_eq!(
            f64_child
                .stockham_upload_schedule
                .as_ref()
                .expect("DD/F64 ND-real N4116 upload schedule")
                .axis_split,
            vec![84, 49]
        );
    }

    #[test]
    fn r2r_device_aware_children_keep_n_point_dct2_topology() {
        fn assert_n_point(ir: &DoubleDoubleR2rIr, expected_len: usize) {
            let DoubleDoubleR2rAlgorithm::FftReduction { fft, fft_len, .. } = &ir.algorithm else {
                panic!("DCT-II must use FFT reduction");
            };
            assert_eq!(*fft_len, expected_len);
            assert_eq!(fft.sequence_len(), expected_len);
        }

        let device = DeviceProfile {
            shared_memory_bytes: 48 * 1024,
            shared_memory_pow2_bytes: 32 * 1024,
            max_threads_per_block: 1024,
            max_workgroup_size: [1024, 1024, 64],
            supports_f64: true,
            ..DeviceProfile::generic(crate::Backend::Vulkan, crate::GpuVendor::Nvidia)
        };
        let transform = TransformKind::Dct(DctType::II);
        let one_dim_config = FftConfig::new(vec![2_058usize])
            .with_transform(transform)
            .with_precision(Precision::DoubleDouble)
            .resolve_tuning_for_device(device);
        let one_dim_plan = FftPlan::build_for_device(one_dim_config.clone(), device).unwrap();
        let portable = DoubleDoubleR2rIr::build(&one_dim_plan, Direction::Forward).unwrap();
        assert_n_point(&portable, 2_058);
        let scheduled =
            DoubleDoubleR2rIr::build_for_device(&one_dim_plan, Direction::Forward, device).unwrap();
        assert_n_point(&scheduled, 2_058);
        scheduled.validate().unwrap();

        let high_level =
            crate::TransformIr::build(one_dim_config, Direction::Forward, device).unwrap();
        let crate::TransformIr::RealToRealDoubleDouble(high_level) = high_level else {
            panic!("high-level DCT-II N2058 should use DD R2R IR");
        };
        assert_n_point(&high_level, 2_058);

        let nd_config = FftConfig::new(vec![2_058usize, 2])
            .with_transform(transform)
            .with_precision(Precision::DoubleDouble)
            .resolve_tuning_for_device(device);
        let nd_plan = FftPlan::build_for_device(nd_config.clone(), device).unwrap();
        let portable_nd = DoubleDoubleNdR2rIr::build(&nd_plan, Direction::Forward).unwrap();
        let portable_axis = portable_nd.axes.iter().find(|axis| axis.axis == 0).unwrap();
        assert_n_point(&portable_axis.transform, 2_058);

        let scheduled_nd =
            DoubleDoubleNdR2rIr::build_for_device(&nd_plan, Direction::Forward, device)
                .unwrap()
                .with_grouped_stockham_axis_blocks(None, device)
                .unwrap();
        let axis = scheduled_nd
            .axes
            .iter()
            .find(|axis| axis.axis == 0)
            .unwrap();
        assert_n_point(&axis.transform, 2_058);
        scheduled_nd.validate().unwrap();
        let axis_program = crate::ProgramIr::double_double_r2r(&axis.transform).unwrap();
        assert_eq!(axis_program.passes.len(), 2);
        assert!(
            axis_program
                .passes
                .iter()
                .all(|pass| pass.name.contains("four_step_upload_"))
        );
        assert!(axis_program.resources.iter().all(|resource| {
            !resource.name.contains("double_double_r2r_fft_input")
                && !resource.name.contains("double_double_r2r_fft_output")
        }));
        let axis_shaders = crate::backend::vulkan::VulkanGlslBackend
            .lower_double_double_r2r(&axis.transform)
            .unwrap();
        assert_eq!(axis_shaders.len(), 2);
        assert_eq!(axis_shaders.len(), axis_program.passes.len());
        assert!(axis_shaders[0].glsl.contains("logical_source"));
        assert!(axis_shaders[1].glsl.contains("logical_output"));
        for shader in &axis_shaders {
            assert_eq!(shader.compile_spirv().unwrap().words[0], 0x0723_0203);
        }
        let program = crate::ProgramIr::double_double_nd_r2r(&scheduled_nd).unwrap();
        program.validate().unwrap();
        let shaders = crate::backend::vulkan::VulkanGlslBackend
            .lower_double_double_nd_r2r(&scheduled_nd)
            .unwrap();
        assert_eq!(shaders.len(), program.passes.len());
        for shader in &shaders {
            assert_eq!(shader.compile_spirv().unwrap().words[0], 0x0723_0203);
        }

        let high_level = crate::TransformIr::build(nd_config, Direction::Forward, device).unwrap();
        let crate::TransformIr::RealToRealNdDoubleDouble(high_level) = high_level else {
            panic!("high-level ND DCT-II [2058,2] should use DD ND-R2R IR");
        };
        let high_axis = high_level.axes.iter().find(|axis| axis.axis == 0).unwrap();
        assert_n_point(&high_axis.transform, 2_058);

        let nd_f64 = crate::TransformIr::build(
            FftConfig::new(vec![2_058usize, 2])
                .with_transform(transform)
                .with_precision(Precision::DoubleDoubleF64Storage)
                .resolve_tuning_for_device(device),
            Direction::Forward,
            device,
        )
        .unwrap();
        let crate::TransformIr::RealToRealNdDoubleDouble(nd_f64) = nd_f64 else {
            panic!("high-level DD/F64 ND DCT-II [2058,2] should use ND-R2R IR");
        };
        assert_eq!(nd_f64.external_storage, PrecisionStorage::F64);
        let f64_axis = nd_f64.axes.iter().find(|axis| axis.axis == 0).unwrap();
        assert_n_point(&f64_axis.transform, 2_058);
        assert_eq!(
            f64_axis.transform.external_storage,
            PrecisionStorage::DoubleDouble
        );
        let f64_axis_program = crate::ProgramIr::double_double_r2r(&f64_axis.transform).unwrap();
        assert_eq!(f64_axis_program.passes.len(), 2);
        assert_eq!(
            f64_axis_program.input_resource().unwrap().scalar,
            crate::kernel_ir::ScalarType::DoubleDouble
        );

        let one_dim_f64 = crate::TransformIr::build(
            FftConfig::new(vec![2_058usize])
                .with_transform(transform)
                .with_precision(Precision::DoubleDoubleF64Storage)
                .resolve_tuning_for_device(device),
            Direction::Forward,
            device,
        )
        .unwrap();
        let crate::TransformIr::RealToRealDoubleDouble(one_dim_f64) = one_dim_f64 else {
            panic!("high-level DD/F64 DCT-II N2058 should use 1D R2R IR");
        };
        assert_eq!(one_dim_f64.external_storage, PrecisionStorage::F64);
        assert_n_point(&one_dim_f64, 2_058);
        let f64_program = crate::ProgramIr::double_double_r2r(&one_dim_f64).unwrap();
        assert_eq!(f64_program.passes.len(), 2);
        assert_eq!(
            f64_program.input_resource().unwrap().scalar,
            crate::kernel_ir::ScalarType::F64
        );
        assert_eq!(
            f64_program.output_resource().unwrap().scalar,
            crate::kernel_ir::ScalarType::F64
        );
        assert!(f64_program.resources.iter().all(|resource| {
            !resource.name.contains("double_double_r2r_fft_input")
                && !resource.name.contains("double_double_r2r_fft_output")
        }));
        let f64_shaders = crate::backend::vulkan::VulkanGlslBackend
            .lower_double_double_r2r(&one_dim_f64)
            .unwrap();
        assert_eq!(f64_shaders.len(), 2);
        for shader in &f64_shaders {
            assert_eq!(shader.compile_spirv().unwrap().words[0], 0x0723_0203);
        }
    }

    #[test]
    fn real_r2r_device_aware_preserve_rader_and_bluestein_children() {
        let device = DeviceProfile {
            shared_memory_bytes: 48 * 1024,
            shared_memory_pow2_bytes: 32 * 1024,
            max_threads_per_block: 128,
            max_workgroup_size: [1024, 1024, 64],
            supports_f64: true,
            ..DeviceProfile::generic(crate::Backend::Vulkan, crate::GpuVendor::Nvidia)
        };

        let nd_real_plan = FftPlan::build_for_device(
            FftConfig::new(vec![5_100usize, 2])
                .with_transform(TransformKind::RealToComplex)
                .with_precision(Precision::DoubleDouble)
                .resolve_tuning_for_device(device),
            device,
        )
        .unwrap();
        let nd_real = DoubleDoubleNdRealFftIr::build_for_device(&nd_real_plan, device).unwrap();
        let rader_axis = nd_real
            .complex_axes
            .iter()
            .find(|axis| axis.axis == 0)
            .unwrap();
        let DoubleDoubleOneDimIr::Recursive(rader) = &rader_axis.transform else {
            panic!("device-aware ND-real N5100 axis should use recursive Rader");
        };
        assert_eq!(
            rader
                .rader_forced_upload_schedule
                .as_ref()
                .expect("ND-real N5100 should retain forced-Rader split")
                .axis_split,
            vec![68, 75]
        );

        let dct_rader_plan = FftPlan::build_for_device(
            FftConfig::new(vec![2_550usize])
                .with_transform(TransformKind::Dct(DctType::II))
                .with_precision(Precision::DoubleDouble)
                .resolve_tuning_for_device(device),
            device,
        )
        .unwrap();
        let dct_rader =
            DoubleDoubleR2rIr::build_for_device(&dct_rader_plan, Direction::Forward, device)
                .unwrap();
        let DoubleDoubleR2rAlgorithm::FftReduction { fft, fft_len, .. } = &dct_rader.algorithm
        else {
            panic!("device-aware DCT-II N2550 should use FFT reduction");
        };
        assert_eq!(*fft_len, 2_550);
        let DoubleDoubleOneDimIr::Recursive(rader) = fft.as_ref() else {
            panic!("device-aware DCT-II N2550 should use recursive N2550 child");
        };
        assert_eq!(
            rader
                .rader_forced_upload_schedule
                .as_ref()
                .expect("DCT-II N2550 should retain N-point forced-Rader split")
                .axis_split,
            vec![50, 51]
        );

        let mut tuning = crate::PlannerTuning::for_device(device, Precision::DoubleDouble);
        tuning.max_rader_fft_prime = 100;
        let real_bluestein_plan = FftPlan::build_for_device(
            FftConfig::new(vec![2_053usize])
                .with_transform(TransformKind::RealToComplex)
                .with_precision(Precision::DoubleDouble)
                .with_tuning(tuning),
            device,
        )
        .unwrap();
        let real_bluestein =
            DoubleDoubleRealFftIr::build_for_device(&real_bluestein_plan, device).unwrap();
        let bluestein = one_dim_bluestein_child(&real_bluestein.transform)
            .expect("device-aware real p2053 child should contain Bluestein");
        assert_eq!(bluestein.logical_len, 2_053);
        assert_eq!(bluestein.convolution_len, 4_368);
        assert!(bluestein.wrapper_axis_batch_block().is_some());
        let DoubleDoubleBluesteinConvolutionIr::Recursive(real_child) = &bluestein.forward_fft
        else {
            panic!("device-aware real p2053 should use recursive M4368 convolution");
        };
        assert!(real_child.stockham_upload_schedule.is_none());
        let recursive_program = crate::ProgramIr::double_double_bluestein(bluestein).unwrap();
        assert!(
            recursive_program
                .resources
                .iter()
                .any(|resource| { resource.name == "double_double_bluestein_inverse_input" })
        );
        assert!(
            recursive_program
                .passes
                .iter()
                .any(|pass| pass.name.ends_with("_multiply"))
        );

        let dct_bluestein_plan = FftPlan::build_for_device(
            FftConfig::new(vec![2_053usize])
                .with_transform(TransformKind::Dct(DctType::II))
                .with_precision(Precision::DoubleDouble)
                .with_tuning(tuning),
            device,
        )
        .unwrap();
        let dct_bluestein =
            DoubleDoubleR2rIr::build_for_device(&dct_bluestein_plan, Direction::Forward, device)
                .unwrap();
        let DoubleDoubleR2rAlgorithm::FftReduction { fft, fft_len, .. } = &dct_bluestein.algorithm
        else {
            panic!("device-aware DCT-II p2053 should use FFT reduction");
        };
        assert_eq!(*fft_len, 2_053);
        let bluestein = one_dim_bluestein_child(fft)
            .expect("device-aware DCT-II p2053 N-point child should contain Bluestein");
        assert_eq!(bluestein.logical_len, 2_053);
        assert_eq!(bluestein.convolution_len, 4_368);
        assert!(bluestein.wrapper_axis_batch_block().is_some());
    }

    #[test]
    fn r2r_whole_axis_bluestein_fuses_real_boundaries_into_chirp_passes() {
        let device = DeviceProfile {
            shared_memory_bytes: 48 * 1024,
            shared_memory_pow2_bytes: 32 * 1024,
            max_threads_per_block: 1024,
            max_workgroup_size: [1024, 1024, 64],
            supports_f64: true,
            ..DeviceProfile::generic(crate::Backend::Vulkan, crate::GpuVendor::Nvidia)
        };
        let length = 103usize;
        for precision in [Precision::DoubleDouble, Precision::DoubleDoubleF64Storage] {
            for direction in [Direction::Forward, Direction::Inverse] {
                let mut tuning = crate::PlannerTuning::for_device(device, Precision::DoubleDouble);
                tuning.max_rader_fft_prime = 100;
                let transform = crate::TransformIr::build(
                    FftConfig::new(vec![length])
                        .with_transform(TransformKind::Dct(DctType::II))
                        .with_precision(precision)
                        .with_tuning(tuning)
                        .with_inverse_normalization(direction == Direction::Inverse),
                    direction,
                    device,
                )
                .unwrap();
                let crate::TransformIr::RealToRealDoubleDouble(r2r) = transform else {
                    panic!("DD N103 DCT-II/III probe did not build R2R");
                };
                let DoubleDoubleR2rAlgorithm::FftReduction { fft, .. } = &r2r.algorithm else {
                    panic!("DD N103 DCT-II/III must use FFT reduction");
                };
                let DoubleDoubleOneDimIr::Bluestein(bluestein) = fft.as_ref() else {
                    panic!("DD N103 DCT-II/III must keep whole-axis Bluestein");
                };
                assert_eq!(bluestein.logical_len, length);
                assert_eq!(bluestein.convolution_len, 256);
                assert_eq!(
                    bluestein.external_storage,
                    crate::PrecisionStorage::DoubleDouble
                );
                assert!(bluestein.zero_padding.is_none());

                let child_program = crate::ProgramIr::double_double_bluestein(bluestein).unwrap();
                let program = crate::ProgramIr::double_double_r2r(&r2r).unwrap();
                assert_eq!(program.passes.len(), child_program.passes.len());
                assert!(program.resources.iter().all(|resource| {
                    !resource.name.contains("double_double_r2r_fft_input")
                        && !resource.name.contains("double_double_r2r_fft_output")
                }));
                let expected_scalar = match precision {
                    Precision::DoubleDouble => crate::kernel_ir::ScalarType::DoubleDouble,
                    Precision::DoubleDoubleF64Storage => crate::kernel_ir::ScalarType::F64,
                    _ => unreachable!(),
                };
                assert_eq!(program.input_resource().unwrap().scalar, expected_scalar);
                assert_eq!(program.output_resource().unwrap().scalar, expected_scalar);
                let first = program.passes.first().unwrap();
                let last = program.passes.last().unwrap();
                assert!(first.name.contains("bluestein_preprocess"));
                assert!(last.name.contains("bluestein_postprocess"));
                match direction {
                    Direction::Forward => {
                        assert!(!first.bindings.iter().any(|binding| binding.binding == 3));
                        assert!(last.bindings.iter().any(|binding| binding.binding == 3));
                    }
                    Direction::Inverse => {
                        assert!(first.bindings.iter().any(|binding| binding.binding == 3));
                        assert!(!last.bindings.iter().any(|binding| binding.binding == 3));
                    }
                }

                let shaders = crate::backend::vulkan::VulkanGlslBackend
                    .lower_double_double_r2r(&r2r)
                    .unwrap();
                assert_eq!(shaders.len(), program.passes.len());
                let first_shader = &shaders.first().unwrap().glsl;
                let last_shader = &shaders.last().unwrap().glsl;
                match direction {
                    Direction::Forward => {
                        assert!(first_shader.contains("logical_source"));
                        assert!(last_shader.contains("VkFftDdR2rPhases"));
                        assert!(last_shader.contains("binding = 3"));
                        assert!(last_shader.contains("vkfft_r2r_bluestein_output_destination"));
                    }
                    Direction::Inverse => {
                        assert!(first_shader.contains("a_index"));
                        assert!(first_shader.contains("VkFftDdR2rPhases"));
                        assert!(first_shader.contains("binding = 3"));
                        assert!(last_shader.contains("logical_output"));
                        assert!(!last_shader.contains("VkFftDdR2rPhases"));
                    }
                }
                for shader in &shaders {
                    assert_eq!(shader.compile_spirv().unwrap().words[0], 0x0723_0203);
                }
            }
        }
    }

    #[test]
    fn r2r_whole_axis_bluestein_dst_ii_iii_fuses_sign_boundaries() {
        let device = DeviceProfile {
            shared_memory_bytes: 48 * 1024,
            shared_memory_pow2_bytes: 32 * 1024,
            max_threads_per_block: 1024,
            max_workgroup_size: [1024, 1024, 64],
            supports_f64: true,
            ..DeviceProfile::generic(crate::Backend::Vulkan, crate::GpuVendor::Nvidia)
        };
        let length = 103usize;
        for precision in [Precision::DoubleDouble, Precision::DoubleDoubleF64Storage] {
            for direction in [Direction::Forward, Direction::Inverse] {
                let mut tuning = crate::PlannerTuning::for_device(device, Precision::DoubleDouble);
                tuning.max_rader_fft_prime = 100;
                let transform = crate::TransformIr::build(
                    FftConfig::new(vec![length])
                        .with_transform(TransformKind::Dst(DstType::II))
                        .with_precision(precision)
                        .with_tuning(tuning)
                        .with_inverse_normalization(direction == Direction::Inverse),
                    direction,
                    device,
                )
                .unwrap();
                let crate::TransformIr::RealToRealDoubleDouble(r2r) = transform else {
                    panic!("DD N103 DST-II/III probe did not build R2R");
                };
                assert_eq!(
                    r2r.effective_transform,
                    match direction {
                        Direction::Forward => R2rTransform::Dst(DstType::II),
                        Direction::Inverse => R2rTransform::Dst(DstType::III),
                    }
                );
                let DoubleDoubleR2rAlgorithm::FftReduction { fft, .. } = &r2r.algorithm else {
                    panic!("DD N103 DST-II/III must use FFT reduction");
                };
                let DoubleDoubleOneDimIr::Bluestein(bluestein) = fft.as_ref() else {
                    panic!("DD N103 DST-II/III must keep whole-axis Bluestein");
                };
                assert_eq!(bluestein.logical_len, length);
                assert_eq!(bluestein.convolution_len, 256);
                assert!(bluestein.zero_padding.is_none());

                let child_program = crate::ProgramIr::double_double_bluestein(bluestein).unwrap();
                let program = crate::ProgramIr::double_double_r2r(&r2r).unwrap();
                assert_eq!(program.passes.len(), child_program.passes.len());
                assert!(program.resources.iter().all(|resource| {
                    !resource.name.contains("double_double_r2r_fft_input")
                        && !resource.name.contains("double_double_r2r_fft_output")
                }));

                let shaders = crate::backend::vulkan::VulkanGlslBackend
                    .lower_double_double_r2r(&r2r)
                    .unwrap();
                assert_eq!(shaders.len(), program.passes.len());
                let first_shader = &shaders.first().unwrap().glsl;
                let last_shader = &shaders.last().unwrap().glsl;
                match direction {
                    Direction::Forward => {
                        assert!(first_shader.contains("logical_source"));
                        assert!(first_shader.contains("vkfft_r2r_bluestein_input_logical_source"));
                        assert!(first_shader.contains("vkfft_r2r_bluestein_input_scalar = vkfft_dd_neg(vkfft_r2r_bluestein_input_scalar)"));
                        assert!(last_shader.contains("VkFftDdR2rPhases"));
                    }
                    Direction::Inverse => {
                        assert!(first_shader.contains("VkFftDdR2rPhases"));
                        assert!(last_shader.contains("logical_output"));
                        assert!(last_shader.contains("vkfft_r2r_bluestein_output_logical_output"));
                        assert!(last_shader.contains("vkfft_r2r_bluestein_output_real_value = vkfft_dd_neg(vkfft_r2r_bluestein_output_real_value)"));
                    }
                }
                for shader in &shaders {
                    assert_eq!(shader.compile_spirv().unwrap().words[0], 0x0723_0203);
                }
            }
        }
    }

    #[test]
    fn r2r_whole_axis_bluestein_preserves_grouped_parent_padding_boundary() {
        let device = DeviceProfile {
            shared_memory_bytes: 48 * 1024,
            shared_memory_pow2_bytes: 32 * 1024,
            max_threads_per_block: 1024,
            max_workgroup_size: [1024, 1024, 64],
            supports_f64: true,
            ..DeviceProfile::generic(crate::Backend::Vulkan, crate::GpuVendor::Nvidia)
        };
        let length = 103usize;
        let mut tuning = crate::PlannerTuning::for_device(device, Precision::DoubleDouble);
        tuning.max_rader_fft_prime = 100;
        let transform = crate::TransformIr::build(
            FftConfig::new(vec![length])
                .with_batch_count(5)
                .with_transform(TransformKind::Dct(DctType::II))
                .with_precision(Precision::DoubleDouble)
                .with_tuning(tuning)
                .with_zero_padding(0, 20, 25)
                .unwrap()
                .with_grouped_batch(0, 3)
                .unwrap(),
            Direction::Forward,
            device,
        )
        .unwrap();
        let crate::TransformIr::RealToRealDoubleDouble(r2r) = transform else {
            panic!("grouped padded DD N103 probe did not build R2R");
        };
        assert_eq!(r2r.batch_count, 5);
        assert_eq!(r2r.grouped_batch, 3);
        assert_eq!(r2r.zero_padding.unwrap().left, 20);
        assert_eq!(r2r.zero_padding.unwrap().right, 25);
        let DoubleDoubleR2rAlgorithm::FftReduction { fft, .. } = &r2r.algorithm else {
            panic!("grouped padded DD N103 must use FFT reduction");
        };
        let DoubleDoubleOneDimIr::Bluestein(bluestein) = fft.as_ref() else {
            panic!("grouped padded DD N103 child must remain whole-axis Bluestein");
        };
        assert_eq!(bluestein.logical_len, length);
        assert_eq!(bluestein.convolution_len, 256);
        assert_eq!(bluestein.batch_count, 5);
        assert_eq!(bluestein.grouped_batch, 3);
        assert!(bluestein.zero_padding.is_none());

        let child_program = crate::ProgramIr::double_double_bluestein(bluestein).unwrap();
        let program = crate::ProgramIr::double_double_r2r(&r2r).unwrap();
        assert_eq!(program.passes.len(), child_program.passes.len());
        assert!(program.resources.iter().all(|resource| {
            !resource.name.contains("double_double_r2r_fft_input")
                && !resource.name.contains("double_double_r2r_fft_output")
        }));
        let shaders = crate::backend::vulkan::VulkanGlslBackend
            .lower_double_double_r2r(&r2r)
            .unwrap();
        assert_eq!(shaders.len(), program.passes.len());
        let first = &shaders.first().unwrap().glsl;
        assert!(first.contains("vkfft_r2r_bluestein_input_logical_source"));
        assert!(first.contains("vkfft_r2r_bluestein_input_logical_source >= 20u"));
        assert!(first.contains("vkfft_r2r_bluestein_input_logical_source < 25u"));
        for shader in &shaders {
            assert_eq!(shader.compile_spirv().unwrap().words[0], 0x0723_0203);
        }
    }

    #[test]
    fn nd_r2r_higher_axis_whole_axis_bluestein_reuses_fused_child_boundaries() {
        let device = DeviceProfile {
            shared_memory_bytes: 48 * 1024,
            shared_memory_pow2_bytes: 32 * 1024,
            max_threads_per_block: 1024,
            max_workgroup_size: [1024, 1024, 64],
            supports_f64: true,
            ..DeviceProfile::generic(crate::Backend::Vulkan, crate::GpuVendor::Nvidia)
        };
        let dimensions = vec![103usize, 8usize];
        for precision in [Precision::DoubleDouble, Precision::DoubleDoubleF64Storage] {
            for direction in [Direction::Forward, Direction::Inverse] {
                let mut tuning = crate::PlannerTuning::for_device(device, Precision::DoubleDouble);
                tuning.max_rader_fft_prime = 100;
                let transform = crate::TransformIr::build(
                    FftConfig::new(dimensions.clone())
                        .with_batch_count(2)
                        .with_transform(TransformKind::Dct(DctType::II))
                        .with_precision(precision)
                        .with_tuning(tuning)
                        .with_grouped_batch(0, 2)
                        .unwrap()
                        .with_grouped_batch(1, 2)
                        .unwrap()
                        .with_inverse_normalization(direction == Direction::Inverse),
                    direction,
                    device,
                )
                .unwrap();
                let crate::TransformIr::RealToRealNdDoubleDouble(nd) = transform else {
                    panic!("DD ND [103,8] DCT-II/III probe did not build ND R2R");
                };
                let outer = nd.axes.iter().find(|axis| axis.axis == 0).unwrap();
                assert_eq!(outer.axis_len, 103);
                assert_eq!(outer.inner_stride, 8);
                assert_eq!(outer.line_count, 8);
                assert_eq!(outer.grouped_batch, 2);
                assert_eq!(outer.transform.batch_count, 16);
                assert_eq!(outer.transform.grouped_batch, 2);
                let DoubleDoubleR2rAlgorithm::FftReduction { fft, .. } = &outer.transform.algorithm
                else {
                    panic!("DD ND [103,8] higher axis must use FFT reduction");
                };
                let DoubleDoubleOneDimIr::Bluestein(bluestein) = fft.as_ref() else {
                    panic!("DD ND [103,8] higher axis must remain whole-axis Bluestein");
                };
                assert_eq!(bluestein.logical_len, 103);
                assert_eq!(bluestein.convolution_len, 256);
                assert_eq!(bluestein.batch_count, 16);
                assert_eq!(bluestein.grouped_batch, 2);
                assert_eq!(
                    bluestein.external_storage,
                    crate::PrecisionStorage::DoubleDouble
                );
                assert!(bluestein.zero_padding.is_none());

                let child_program = crate::ProgramIr::double_double_r2r(&outer.transform).unwrap();
                assert!(child_program.resources.iter().all(|resource| {
                    !resource.name.contains("double_double_r2r_fft_input")
                        && !resource.name.contains("double_double_r2r_fft_output")
                }));
                let program = crate::ProgramIr::double_double_nd_r2r(&nd).unwrap();
                let expected_external_scalar = match precision {
                    Precision::DoubleDouble => crate::kernel_ir::ScalarType::DoubleDouble,
                    Precision::DoubleDoubleF64Storage => crate::kernel_ir::ScalarType::F64,
                    _ => unreachable!(),
                };
                assert_eq!(
                    program.input_resource().unwrap().scalar,
                    expected_external_scalar
                );
                assert_eq!(
                    program.output_resource().unwrap().scalar,
                    expected_external_scalar
                );
                let axis0_passes = program
                    .passes
                    .iter()
                    .filter(|pass| pass.name.contains("vkfft_dd_nd_r2r_axis_0_"))
                    .count();
                assert_eq!(axis0_passes, child_program.passes.len() + 2);
                assert!(program.resources.iter().all(|resource| {
                    !resource
                        .name
                        .contains("double_double_nd_r2r_axis_0_double_double_r2r_fft_input")
                        && !resource
                            .name
                            .contains("double_double_nd_r2r_axis_0_double_double_r2r_fft_output")
                }));
                assert!(program.passes.iter().any(|pass| {
                    pass.name
                        .contains("axis_0_vkfft_dd_r2r_dct2_bluestein_preprocess")
                        || pass
                            .name
                            .contains("axis_0_vkfft_dd_r2r_dct3_bluestein_preprocess")
                }));
                assert!(program.passes.iter().any(|pass| {
                    pass.name
                        .contains("axis_0_vkfft_dd_r2r_dct2_bluestein_postprocess")
                        || pass
                            .name
                            .contains("axis_0_vkfft_dd_r2r_dct3_bluestein_postprocess")
                }));

                let shaders = crate::backend::vulkan::VulkanGlslBackend
                    .lower_double_double_nd_r2r(&nd)
                    .unwrap();
                assert_eq!(shaders.len(), program.passes.len());
                for shader in &shaders {
                    assert_eq!(shader.compile_spirv().unwrap().words[0], 0x0723_0203);
                }
            }
        }
    }

    #[test]
    fn nd_r2r_higher_axis_bluestein_keeps_padding_at_tensor_boundary() {
        let device = DeviceProfile {
            shared_memory_bytes: 48 * 1024,
            shared_memory_pow2_bytes: 32 * 1024,
            max_threads_per_block: 1024,
            max_workgroup_size: [1024, 1024, 64],
            supports_f64: true,
            ..DeviceProfile::generic(crate::Backend::Vulkan, crate::GpuVendor::Nvidia)
        };
        let dimensions = vec![103usize, 8usize];
        let pad_left = 20usize;
        let pad_right = 25usize;
        for precision in [Precision::DoubleDouble, Precision::DoubleDoubleF64Storage] {
            for direction in [Direction::Forward, Direction::Inverse] {
                let mut tuning = crate::PlannerTuning::for_device(device, Precision::DoubleDouble);
                tuning.max_rader_fft_prime = 100;
                let transform = crate::TransformIr::build(
                    FftConfig::new(dimensions.clone())
                        .with_batch_count(2)
                        .with_transform(TransformKind::Dct(DctType::II))
                        .with_precision(precision)
                        .with_tuning(tuning)
                        .with_grouped_batch(0, 2)
                        .unwrap()
                        .with_grouped_batch(1, 2)
                        .unwrap()
                        .with_zero_padding(0, pad_left, pad_right)
                        .unwrap()
                        .with_inverse_normalization(direction == Direction::Inverse),
                    direction,
                    device,
                )
                .unwrap();
                let crate::TransformIr::RealToRealNdDoubleDouble(nd) = transform else {
                    panic!("padded DD ND [103,8] probe did not build ND R2R");
                };
                assert_eq!(nd.zero_padding[0].unwrap().left, pad_left);
                assert_eq!(nd.zero_padding[0].unwrap().right, pad_right);
                assert!(nd.zero_padding[1].is_none());
                let outer = nd.axes.iter().find(|axis| axis.axis == 0).unwrap();
                let DoubleDoubleR2rAlgorithm::FftReduction { fft, .. } = &outer.transform.algorithm
                else {
                    panic!("padded DD ND [103,8] higher axis must use FFT reduction");
                };
                let DoubleDoubleOneDimIr::Bluestein(bluestein) = fft.as_ref() else {
                    panic!("padded DD ND [103,8] higher axis must remain whole-axis Bluestein");
                };
                assert_eq!(bluestein.convolution_len, 256);
                assert_eq!(bluestein.batch_count, 16);
                assert_eq!(bluestein.grouped_batch, 2);
                assert!(bluestein.zero_padding.is_none());
                assert!(outer.transform.zero_padding.is_none());

                let child_program = crate::ProgramIr::double_double_r2r(&outer.transform).unwrap();
                let program = crate::ProgramIr::double_double_nd_r2r(&nd).unwrap();
                let axis0_passes = program
                    .passes
                    .iter()
                    .filter(|pass| pass.name.contains("vkfft_dd_nd_r2r_axis_0_"))
                    .count();
                assert_eq!(axis0_passes, child_program.passes.len() + 2);
                assert!(program.resources.iter().all(|resource| {
                    !resource
                        .name
                        .contains("double_double_nd_r2r_axis_0_double_double_r2r_fft_input")
                        && !resource
                            .name
                            .contains("double_double_nd_r2r_axis_0_double_double_r2r_fft_output")
                }));

                let shaders = crate::backend::vulkan::VulkanGlslBackend
                    .lower_double_double_nd_r2r(&nd)
                    .unwrap();
                assert_eq!(shaders.len(), program.passes.len());
                let padded_boundaries = shaders
                    .iter()
                    .filter(|shader| {
                        shader
                            .glsl
                            .contains("fused spatial zero padding at DD ND R2R true boundary")
                    })
                    .collect::<Vec<_>>();
                assert_eq!(padded_boundaries.len(), 1);
                let boundary = &padded_boundaries[0].glsl;
                assert!(boundary.contains("% 103u) >= 20u"));
                assert!(boundary.contains("% 103u) < 25u"));
                match direction {
                    Direction::Forward => assert!(boundary.contains("axis 1 pack")),
                    Direction::Inverse => assert!(boundary.contains("axis 0 scatter")),
                }
                for shader in &shaders {
                    assert_eq!(shader.compile_spirv().unwrap().words[0], 0x0723_0203);
                }
            }
        }
    }

    #[test]
    fn nd_recursive_rader_axes_compose_and_match_independent_dd_dft() {
        for axis_len in [17usize * 17, 17usize * 19] {
            let dimensions = [2usize, axis_len];
            let tensor_len = dimensions.iter().product::<usize>();
            let forward_plan = FftPlan::build(
                FftConfig::new(dimensions.to_vec()).with_precision(Precision::DoubleDouble),
            )
            .unwrap();
            let inverse_plan = FftPlan::build(
                FftConfig::new(dimensions.to_vec())
                    .with_precision(Precision::DoubleDouble)
                    .with_inverse_normalization(true),
            )
            .unwrap();
            let forward = DoubleDoubleNdFftIr::build(&forward_plan, Direction::Forward).unwrap();
            let inverse = DoubleDoubleNdFftIr::build(&inverse_plan, Direction::Inverse).unwrap();
            assert!(matches!(
                forward.axes[0].transform,
                DoubleDoubleOneDimIr::Recursive(_)
            ));
            let program = crate::ProgramIr::double_double_nd(&forward).unwrap();
            let shaders = crate::backend::vulkan::VulkanGlslBackend
                .lower_double_double_nd(&forward)
                .unwrap();
            assert_eq!(shaders.len(), program.passes.len());
            for shader in &shaders {
                assert_eq!(shader.compile_spirv().unwrap().words[0], 0x0723_0203);
            }

            let input = (0..tensor_len)
                .map(|index| {
                    let x = index as f64;
                    ComplexDoubleDouble::new(
                        DoubleDouble::from_parts(
                            (0.031 * x).sin() + 0.00011 * x,
                            (index + 1) as f64 * 7.0e-32,
                        ),
                        DoubleDouble::from_parts(
                            (0.017 * x).cos() - 0.00009 * x,
                            -(index as f64 + 1.0) * 3.0e-32,
                        ),
                    )
                })
                .collect::<Vec<_>>();
            let actual = execute_double_double_nd_ir(&forward, &input).unwrap();
            let mut expected = vec![ComplexDoubleDouble::default(); tensor_len];
            for k0 in 0..dimensions[0] {
                for k1 in 0..dimensions[1] {
                    let mut sum = ComplexDoubleDouble::default();
                    for n0 in 0..dimensions[0] {
                        let root0 = unit_root(n0 * k0, dimensions[0], Direction::Forward).unwrap();
                        for n1 in 0..dimensions[1] {
                            let root1 =
                                unit_root(n1 * k1, dimensions[1], Direction::Forward).unwrap();
                            sum += input[n0 * dimensions[1] + n1] * root0 * root1;
                        }
                    }
                    expected[k0 * dimensions[1] + k1] = sum;
                }
            }
            let error = actual
                .iter()
                .copied()
                .zip(expected.iter().copied())
                .map(|(actual, expected)| dd_error(actual, expected))
                .fold(0.0, f64::max);
            assert!(
                error < 2.0e-20,
                "DD ND [2,{axis_len}] recursive-axis forward error {error:e}"
            );
            let restored = execute_double_double_nd_ir(&inverse, &actual).unwrap();
            let round_trip_error = restored
                .iter()
                .copied()
                .zip(input.iter().copied())
                .map(|(actual, expected)| dd_error(actual, expected))
                .fold(0.0, f64::max);
            assert!(
                round_trip_error < 2.0e-18,
                "DD ND [2,{axis_len}] recursive-axis round-trip error {round_trip_error:e}"
            );

            let f64_plan = FftPlan::build(
                FftConfig::new(dimensions.to_vec())
                    .with_precision(Precision::DoubleDoubleF64Storage),
            )
            .unwrap();
            let f64_ir = DoubleDoubleNdFftIr::build(&f64_plan, Direction::Forward).unwrap();
            assert!(matches!(
                f64_ir.axes[0].transform,
                DoubleDoubleOneDimIr::Recursive(_)
            ));
            let f64_program = crate::ProgramIr::double_double_nd(&f64_ir).unwrap();
            assert_eq!(
                f64_program.input_resource().unwrap().scalar,
                crate::ScalarType::F64
            );
            assert_eq!(
                f64_program.output_resource().unwrap().scalar,
                crate::ScalarType::F64
            );
            assert!(f64_program.resources.iter().any(|resource| {
                resource.kind == crate::ProgramResourceKind::Scratch
                    && resource.scalar == crate::ScalarType::DoubleDouble
            }));
        }
    }

    #[test]
    fn nd_grouped_tail_ownership_propagates_into_recursive_axis_children() {
        let dimensions = [2usize, 17usize * 17];
        let tensor_len = dimensions.iter().product::<usize>();
        let batch_count = 7usize;
        let grouped_batch = 3usize;
        let grouped_config = FftConfig::new(dimensions.to_vec())
            .with_batch_count(batch_count)
            .with_precision(Precision::DoubleDouble)
            .with_grouped_batch(0, grouped_batch)
            .unwrap()
            .with_grouped_batch(1, grouped_batch)
            .unwrap();
        let forward_plan = FftPlan::build(grouped_config.clone()).unwrap();
        let inverse_plan = FftPlan::build(grouped_config.with_inverse_normalization(true)).unwrap();
        let forward = DoubleDoubleNdFftIr::build(&forward_plan, Direction::Forward).unwrap();
        let inverse = DoubleDoubleNdFftIr::build(&inverse_plan, Direction::Inverse).unwrap();

        assert_eq!(forward.axes.len(), 2);
        assert_eq!(forward.axes[0].axis, 1);
        assert_eq!(forward.axes[0].line_count, 2);
        assert_eq!(forward.axes[0].grouped_batch, grouped_batch);
        assert_eq!(forward.axes[0].transform.grouped_batch(), grouped_batch);
        assert_eq!(forward.axes[0].transform.batch_count(), batch_count * 2);
        assert!(matches!(
            forward.axes[0].transform,
            DoubleDoubleOneDimIr::Recursive(_)
        ));
        assert_eq!(forward.axes[1].axis, 0);
        assert_eq!(forward.axes[1].line_count, 17 * 17);
        assert_eq!(forward.axes[1].grouped_batch, grouped_batch);
        assert_eq!(forward.axes[1].transform.grouped_batch(), grouped_batch);
        assert_eq!(
            forward.axes[1].transform.batch_count(),
            batch_count * 17 * 17
        );
        assert_eq!(
            forward.axes[0]
                .transform
                .batch_count()
                .div_ceil(forward.axes[0].transform.grouped_batch()),
            5
        );
        assert_eq!(
            forward.axes[1]
                .transform
                .batch_count()
                .div_ceil(forward.axes[1].transform.grouped_batch()),
            675
        );

        let program = crate::ProgramIr::double_double_nd(&forward).unwrap();
        assert!(program.passes.iter().any(|pass| pass.dispatch.x == 3));
        assert!(program.passes.iter().any(|pass| pass.dispatch.x == 5));
        assert!(program.passes.iter().any(|pass| pass.dispatch.x == 675));
        let shaders = crate::backend::vulkan::VulkanGlslBackend
            .lower_double_double_nd(&forward)
            .unwrap();
        assert_eq!(shaders.len(), program.passes.len());
        assert!(
            shaders
                .iter()
                .zip(&program.passes)
                .all(|(shader, pass)| shader.dispatch == pass.dispatch)
        );
        assert!(shaders.iter().any(|shader| {
            shader.glsl.contains(
                "3 tensor batches/workgroup, 6 boundary packed transforms/workgroup; child groupedBatch 3",
            )
        }));
        assert!(shaders.iter().any(|shader| {
            shader.glsl.contains(
                "3 tensor batches/workgroup, 867 boundary packed transforms/workgroup; child groupedBatch 3",
            )
        }));
        for shader in &shaders {
            assert_eq!(shader.compile_spirv().unwrap().words[0], 0x0723_0203);
        }

        let impulses = [
            (0usize, 0usize),
            (1, 7),
            (0, 31),
            (1, 59),
            (0, 113),
            (1, 173),
            (0, 251),
        ];
        let mut input = vec![ComplexDoubleDouble::default(); tensor_len * batch_count];
        for (batch, (n0, n1)) in impulses.into_iter().enumerate() {
            let scale = (batch + 1) as f64;
            input[batch * tensor_len + n0 * dimensions[1] + n1] = ComplexDoubleDouble::new(
                DoubleDouble::from_parts(0.25 * scale, scale * 3.0e-31),
                DoubleDouble::from_parts(-0.125 * scale, -scale * 2.0e-31),
            );
        }
        let actual = execute_double_double_nd_ir(&forward, &input).unwrap();
        let mut forward_error = 0.0f64;
        for (batch, (n0, n1)) in impulses.into_iter().enumerate() {
            let impulse = input[batch * tensor_len + n0 * dimensions[1] + n1];
            let base = batch * tensor_len;
            for k0 in 0..dimensions[0] {
                let root0 = unit_root(n0 * k0, dimensions[0], Direction::Forward).unwrap();
                for k1 in 0..dimensions[1] {
                    let root1 = unit_root(n1 * k1, dimensions[1], Direction::Forward).unwrap();
                    let expected = impulse * root0 * root1;
                    forward_error = forward_error
                        .max(dd_error(actual[base + k0 * dimensions[1] + k1], expected));
                }
            }
        }
        assert!(
            forward_error < 2.0e-20,
            "DD ND grouped [2,289] impulse forward error {forward_error:e}"
        );
        let restored = execute_double_double_nd_ir(&inverse, &actual).unwrap();
        let round_trip_error = restored
            .iter()
            .copied()
            .zip(input.iter().copied())
            .map(|(actual, expected)| dd_error(actual, expected))
            .fold(0.0, f64::max);
        assert!(
            round_trip_error < 2.0e-18,
            "DD ND grouped [2,289] round-trip error {round_trip_error:e}"
        );
    }

    #[test]
    fn nd_3x17_composes_fft_rader_axis_and_matches_independent_dd_dft() {
        let dimensions = [3usize, 17usize];
        let tensor_len = dimensions.iter().product::<usize>();
        let forward_plan = FftPlan::build(
            FftConfig::new(dimensions.to_vec()).with_precision(Precision::DoubleDouble),
        )
        .unwrap();
        let inverse_plan = FftPlan::build(
            FftConfig::new(dimensions.to_vec())
                .with_precision(Precision::DoubleDouble)
                .with_inverse_normalization(true),
        )
        .unwrap();
        let forward = DoubleDoubleNdFftIr::build(&forward_plan, Direction::Forward).unwrap();
        let inverse = DoubleDoubleNdFftIr::build(&inverse_plan, Direction::Inverse).unwrap();
        assert_eq!(
            forward
                .axes
                .iter()
                .map(|axis| axis.axis)
                .collect::<Vec<_>>(),
            vec![1, 0]
        );
        let DoubleDoubleOneDimIr::FftRader(rader) = &forward.axes[0].transform else {
            panic!("DD ND axis length 17 must preserve planner-selected FFT Rader");
        };
        assert_eq!(rader.prime, 17);
        assert_eq!(rader.convolution_len, 16);
        assert!(matches!(
            forward.axes[1].transform,
            DoubleDoubleOneDimIr::Stockham(_)
        ));

        let input = (0..tensor_len)
            .map(|index| {
                let x = index as f64;
                ComplexDoubleDouble::new(
                    DoubleDouble::from_parts(
                        (0.071 * x).sin() + 0.0007 * x,
                        (index + 1) as f64 * 1.0e-31,
                    ),
                    DoubleDouble::from_parts(
                        (0.043 * x).cos() - 0.0004 * x,
                        -(index as f64 + 1.0) * 6.0e-32,
                    ),
                )
            })
            .collect::<Vec<_>>();
        let actual = execute_double_double_nd_ir(&forward, &input).unwrap();
        let mut expected = vec![ComplexDoubleDouble::default(); tensor_len];
        for k0 in 0..dimensions[0] {
            for k1 in 0..dimensions[1] {
                let mut sum = ComplexDoubleDouble::default();
                for n0 in 0..dimensions[0] {
                    for n1 in 0..dimensions[1] {
                        let root0 = unit_root(n0 * k0, dimensions[0], Direction::Forward).unwrap();
                        let root1 = unit_root(n1 * k1, dimensions[1], Direction::Forward).unwrap();
                        sum += input[n0 * dimensions[1] + n1] * root0 * root1;
                    }
                }
                expected[k0 * dimensions[1] + k1] = sum;
            }
        }
        let error = actual
            .iter()
            .copied()
            .zip(expected.iter().copied())
            .map(|(actual, expected)| dd_error(actual, expected))
            .fold(0.0, f64::max);
        assert!(error < 5.0e-25, "DD ND [3,17] forward error {error:e}");

        let restored = execute_double_double_nd_ir(&inverse, &actual).unwrap();
        let round_trip_error = restored
            .iter()
            .copied()
            .zip(input.iter().copied())
            .map(|(actual, expected)| dd_error(actual, expected))
            .fold(0.0, f64::max);
        assert!(
            round_trip_error < 5.0e-23,
            "DD ND [3,17] round-trip error {round_trip_error:e}"
        );
    }

    #[test]
    fn nd_omit_dimension_matches_independent_single_axis_dd_dft() {
        let dimensions = [3usize, 4usize];
        let tensor_len = dimensions.iter().product::<usize>();
        let input = (0..tensor_len)
            .map(|index| {
                let x = index as f64;
                ComplexDoubleDouble::new(
                    DoubleDouble::from_parts(
                        (0.13 * x).sin() + 0.002 * x,
                        (index + 1) as f64 * 6.0e-32,
                    ),
                    DoubleDouble::from_parts(
                        (0.09 * x).cos() - 0.001 * x,
                        -(index as f64 + 1.0) * 3.0e-32,
                    ),
                )
            })
            .collect::<Vec<_>>();

        for omitted_axis in [0usize, 1] {
            let plan = FftPlan::build(
                FftConfig::new(dimensions.to_vec())
                    .with_precision(Precision::DoubleDouble)
                    .with_omit_dimension(omitted_axis, true)
                    .unwrap(),
            )
            .unwrap();
            let ir = DoubleDoubleNdFftIr::build(&plan, Direction::Forward).unwrap();
            let active_axis = 1 - omitted_axis;
            assert_eq!(ir.omitted_axes, vec![omitted_axis == 0, omitted_axis == 1]);
            assert_eq!(ir.axes.len(), 1);
            assert_eq!(ir.axes[0].axis, active_axis);
            assert_eq!(
                ir.axes[0].inner_stride,
                if active_axis == 0 { 4 } else { 1 }
            );

            let actual = execute_double_double_nd_ir(&ir, &input).unwrap();
            let mut expected = vec![ComplexDoubleDouble::default(); tensor_len];
            match active_axis {
                0 => {
                    for col in 0..dimensions[1] {
                        for k in 0..dimensions[0] {
                            let mut sum = ComplexDoubleDouble::default();
                            for n in 0..dimensions[0] {
                                sum += input[n * dimensions[1] + col]
                                    * unit_root(n * k, dimensions[0], Direction::Forward).unwrap();
                            }
                            expected[k * dimensions[1] + col] = sum;
                        }
                    }
                }
                1 => {
                    for row in 0..dimensions[0] {
                        for k in 0..dimensions[1] {
                            let mut sum = ComplexDoubleDouble::default();
                            for n in 0..dimensions[1] {
                                sum += input[row * dimensions[1] + n]
                                    * unit_root(n * k, dimensions[1], Direction::Forward).unwrap();
                            }
                            expected[row * dimensions[1] + k] = sum;
                        }
                    }
                }
                _ => unreachable!(),
            }
            let error = actual
                .iter()
                .copied()
                .zip(expected.iter().copied())
                .map(|(actual, expected)| dd_error(actual, expected))
                .fold(0.0, f64::max);
            assert!(
                error <= 2.0e-27,
                "DD omit axis {omitted_axis} mismatch: {error:e}"
            );
        }
    }

    #[test]
    fn nd_real_omit_dimension_keeps_only_true_real_axis_in_double_double() {
        let dimensions = vec![3usize, 8usize];
        let input = (0..24)
            .map(|index| {
                let x = index as f64;
                DoubleDouble::from_parts(
                    (0.113 * x).sin() + 0.002 * x,
                    (index + 1) as f64 * 4.0e-32,
                )
            })
            .collect::<Vec<_>>();
        let forward_plan = FftPlan::build(
            FftConfig::new(dimensions.clone())
                .with_transform(TransformKind::RealToComplex)
                .with_precision(Precision::DoubleDouble)
                .with_omit_dimension(0, true)
                .unwrap(),
        )
        .unwrap();
        let forward = DoubleDoubleNdRealFftIr::build(&forward_plan).unwrap();
        assert_eq!(forward.omitted_axes, vec![true, false]);
        assert!(forward.complex_axes.is_empty());
        let actual = execute_double_double_nd_r2c_ir(&forward, &input).unwrap();
        let expected = execute_double_double_r2c_ir(&forward.real_axis, &input).unwrap();
        assert_eq!(actual.len(), expected.len());
        let forward_error = actual
            .iter()
            .copied()
            .zip(expected.iter().copied())
            .map(|(actual, expected)| dd_error(actual, expected))
            .fold(0.0, f64::max);
        assert!(
            forward_error <= 2.0e-27,
            "DD ND-real omit forward mismatch {forward_error:e}"
        );

        let inverse_plan = FftPlan::build(
            FftConfig::new(dimensions)
                .with_transform(TransformKind::ComplexToReal)
                .with_precision(Precision::DoubleDouble)
                .with_inverse_normalization(true)
                .with_omit_dimension(0, true)
                .unwrap(),
        )
        .unwrap();
        let inverse = DoubleDoubleNdRealFftIr::build(&inverse_plan).unwrap();
        assert_eq!(inverse.omitted_axes, vec![true, false]);
        assert!(inverse.complex_axes.is_empty());
        let restored = execute_double_double_nd_c2r_ir(&inverse, &actual).unwrap();
        let round_trip_error = restored
            .iter()
            .copied()
            .zip(input.iter().copied())
            .map(|(actual, expected)| dd_scalar_error(actual, expected))
            .fold(0.0, f64::max);
        assert!(
            round_trip_error <= 2.0e-27,
            "DD ND-real omit round-trip mismatch {round_trip_error:e}"
        );
    }

    #[test]
    fn nd_r2r_omit_dimension_executes_only_active_double_double_axis() {
        let dimensions = [3usize, 4usize];
        let input = (0..12)
            .map(|index| {
                let x = index as f64;
                DoubleDouble::from_parts(
                    (0.173 * x).cos() - 0.021 * x,
                    -(index as f64 + 1.0) * 5.0e-32,
                )
            })
            .collect::<Vec<_>>();
        for omitted_axis in [0usize, 1] {
            let plan = FftPlan::build(
                FftConfig::new(dimensions.to_vec())
                    .with_transform(TransformKind::Dct(crate::DctType::II))
                    .with_precision(Precision::DoubleDouble)
                    .with_omit_dimension(omitted_axis, true)
                    .unwrap(),
            )
            .unwrap();
            let ir = DoubleDoubleNdR2rIr::build(&plan, Direction::Forward).unwrap();
            let active_axis = 1 - omitted_axis;
            assert_eq!(ir.omitted_axes, vec![omitted_axis == 0, omitted_axis == 1]);
            assert_eq!(ir.axes.len(), 1);
            assert_eq!(ir.axes[0].axis, active_axis);
            assert_eq!(
                ir.axes[0].inner_stride,
                if active_axis == 0 { 4 } else { 1 }
            );

            let mut packed = Vec::with_capacity(input.len());
            if active_axis == 1 {
                packed.extend_from_slice(&input);
            } else {
                for column in 0..dimensions[1] {
                    for row in 0..dimensions[0] {
                        packed.push(input[row * dimensions[1] + column]);
                    }
                }
            }
            let transformed = execute_double_double_r2r_ir(&ir.axes[0].transform, &packed).unwrap();
            let mut expected = vec![DoubleDouble::ZERO; input.len()];
            if active_axis == 1 {
                expected.copy_from_slice(&transformed);
            } else {
                for column in 0..dimensions[1] {
                    for row in 0..dimensions[0] {
                        expected[row * dimensions[1] + column] =
                            transformed[column * dimensions[0] + row];
                    }
                }
            }
            let actual = execute_double_double_nd_r2r_ir(&ir, &input).unwrap();
            let error = actual
                .iter()
                .copied()
                .zip(expected.iter().copied())
                .map(|(actual, expected)| dd_scalar_error(actual, expected))
                .fold(0.0, f64::max);
            assert!(
                error <= 2.0e-27,
                "DD ND-R2R omit axis {omitted_axis} mismatch {error:e}"
            );
        }
    }

    #[test]
    fn formatted_nd_c2c_strides_preserve_full_dd_and_f64_storage() {
        let cases = [
            (
                vec![3usize, 4],
                vec![(0usize, 7usize)],
                vec![(0usize, 9usize)],
                21usize,
                27usize,
            ),
            (
                vec![2usize, 3, 4],
                vec![(1usize, 6usize), (0usize, 20usize)],
                vec![(1usize, 7usize), (0usize, 24usize)],
                40usize,
                48usize,
            ),
        ];
        for (dimensions, input_strides, output_strides, input_batch, output_batch) in cases {
            let tensor_len = dimensions.iter().product::<usize>();
            for precision in [Precision::DoubleDouble, Precision::DoubleDoubleF64Storage] {
                let mut formatted_config = FftConfig::new(dimensions.clone())
                    .with_batch_count(2)
                    .with_precision(precision);
                for &(axis, stride) in &input_strides {
                    formatted_config = formatted_config
                        .with_input_buffer_axis_stride(axis, stride)
                        .unwrap();
                }
                for &(axis, stride) in &output_strides {
                    formatted_config = formatted_config
                        .with_output_buffer_axis_stride(axis, stride)
                        .unwrap();
                }
                let dense_config = FftConfig::new(dimensions.clone())
                    .with_batch_count(2)
                    .with_precision(precision);
                let formatted_plan = FftPlan::build(formatted_config).unwrap();
                let dense_plan = FftPlan::build(dense_config).unwrap();
                let formatted =
                    DoubleDoubleNdFftIr::build(&formatted_plan, Direction::Forward).unwrap();
                let dense = DoubleDoubleNdFftIr::build(&dense_plan, Direction::Forward).unwrap();
                assert_eq!(formatted.input_external_layout.batch_stride, input_batch);
                assert_eq!(formatted.output_external_layout.batch_stride, output_batch);
                assert!(formatted.input_formatted_copy.is_some());
                assert!(formatted.output_formatted_copy.is_some());
                assert!(dense.input_formatted_copy.is_none());
                assert!(dense.output_formatted_copy.is_none());

                match precision {
                    Precision::DoubleDouble => {
                        let input = (0..2 * tensor_len)
                            .map(|index| {
                                let x = index as f64;
                                ComplexDoubleDouble::new(
                                    DoubleDouble::from_parts(
                                        (0.17 * x).sin() + 0.003 * x,
                                        (index + 1) as f64 * 1.0e-31,
                                    ),
                                    DoubleDouble::from_parts(
                                        (0.11 * x).cos() - 0.002 * x,
                                        -(index as f64 + 1.0) * 7.0e-32,
                                    ),
                                )
                            })
                            .collect::<Vec<_>>();
                        let physical = formatted.pack_formatted_input(&input).unwrap();
                        assert_eq!(physical.len(), 2 * input_batch);
                        assert_ne!(input[1].re.lo, 0.0);
                        assert_eq!(physical[1], input[1]);
                        let actual = execute_double_double_nd_ir(&formatted, &input).unwrap();
                        let expected = execute_double_double_nd_ir(&dense, &input).unwrap();
                        let error = actual
                            .iter()
                            .copied()
                            .zip(expected.iter().copied())
                            .map(|(actual, expected)| dd_error(actual, expected))
                            .fold(0.0, f64::max);
                        assert!(error <= 2.0e-27, "formatted full-DD mismatch {error:e}");
                    }
                    Precision::DoubleDoubleF64Storage => {
                        let input = (0..2 * tensor_len)
                            .map(|index| {
                                let x = index as f64;
                                Complex64::new(
                                    (0.17 * x).sin() + 0.003 * x,
                                    (0.11 * x).cos() - 0.002 * x,
                                )
                            })
                            .collect::<Vec<_>>();
                        let physical = formatted.pack_formatted_input(&input).unwrap();
                        assert_eq!(physical.len(), 2 * input_batch);
                        let actual =
                            execute_double_double_nd_ir_f64_storage(&formatted, &input).unwrap();
                        let expected =
                            execute_double_double_nd_ir_f64_storage(&dense, &input).unwrap();
                        let error = actual
                            .iter()
                            .zip(&expected)
                            .map(|(actual, expected)| {
                                ((actual.re - expected.re).powi(2)
                                    + (actual.im - expected.im).powi(2))
                                .sqrt()
                            })
                            .fold(0.0, f64::max);
                        assert!(error <= 2.0e-13, "formatted DD/F64 mismatch {error:e}");
                    }
                    _ => unreachable!(),
                }
            }
        }
    }

    #[test]
    fn formatted_nd_real_and_r2r_strides_preserve_dd_and_f64_storage() {
        let real_dimensions = vec![3usize, 8];
        let full_len = real_dimensions.iter().product::<usize>();
        for precision in [Precision::DoubleDouble, Precision::DoubleDoubleF64Storage] {
            let formatted_forward_plan = FftPlan::build(
                FftConfig::new(real_dimensions.clone())
                    .with_batch_count(2)
                    .with_transform(TransformKind::RealToComplex)
                    .with_precision(precision)
                    .with_input_buffer_axis_stride(0, 11)
                    .unwrap()
                    .with_output_buffer_axis_stride(0, 7)
                    .unwrap(),
            )
            .unwrap();
            let dense_forward_plan = FftPlan::build(
                FftConfig::new(real_dimensions.clone())
                    .with_batch_count(2)
                    .with_transform(TransformKind::RealToComplex)
                    .with_precision(precision),
            )
            .unwrap();
            let formatted_forward =
                DoubleDoubleNdRealFftIr::build(&formatted_forward_plan).unwrap();
            let dense_forward = DoubleDoubleNdRealFftIr::build(&dense_forward_plan).unwrap();
            assert_eq!(
                formatted_forward
                    .formatted_io
                    .input_external_layout
                    .batch_stride,
                33
            );
            assert_eq!(
                formatted_forward
                    .formatted_io
                    .output_external_layout
                    .batch_stride,
                21
            );
            assert!(
                formatted_forward
                    .formatted_io
                    .input_formatted_copy
                    .is_some()
            );
            assert!(
                formatted_forward
                    .formatted_io
                    .output_formatted_copy
                    .is_some()
            );

            match precision {
                Precision::DoubleDouble => {
                    let input = (0..2 * full_len)
                        .map(|index| {
                            let x = index as f64;
                            DoubleDouble::from_parts(
                                (0.13 * x).sin() + 0.002 * x,
                                (index + 1) as f64 * 4.0e-32,
                            )
                        })
                        .collect::<Vec<_>>();
                    assert_ne!(input[1].lo, 0.0);
                    let actual =
                        execute_double_double_nd_r2c_ir(&formatted_forward, &input).unwrap();
                    let expected = execute_double_double_nd_r2c_ir(&dense_forward, &input).unwrap();
                    let error = actual
                        .iter()
                        .copied()
                        .zip(expected.iter().copied())
                        .map(|(actual, expected)| dd_error(actual, expected))
                        .fold(0.0, f64::max);
                    assert!(
                        error <= 2.0e-27,
                        "formatted DD ND-real forward mismatch {error:e}"
                    );

                    let inverse_plan = FftPlan::build(
                        FftConfig::new(real_dimensions.clone())
                            .with_batch_count(2)
                            .with_transform(TransformKind::ComplexToReal)
                            .with_precision(precision)
                            .with_inverse_normalization(true)
                            .with_input_buffer_axis_stride(0, 7)
                            .unwrap()
                            .with_output_buffer_axis_stride(0, 11)
                            .unwrap(),
                    )
                    .unwrap();
                    let inverse = DoubleDoubleNdRealFftIr::build(&inverse_plan).unwrap();
                    let restored = execute_double_double_nd_c2r_ir(&inverse, &actual).unwrap();
                    let round_trip = restored
                        .iter()
                        .copied()
                        .zip(input.iter().copied())
                        .map(|(actual, expected)| dd_scalar_error(actual, expected))
                        .fold(0.0, f64::max);
                    assert!(
                        round_trip <= 3.0e-27,
                        "formatted DD ND-real round-trip {round_trip:e}"
                    );
                }
                Precision::DoubleDoubleF64Storage => {
                    let input = (0..2 * full_len)
                        .map(|index| {
                            let x = index as f64;
                            (0.13 * x).sin() + 0.002 * x
                        })
                        .collect::<Vec<_>>();
                    let actual =
                        execute_double_double_nd_r2c_ir_f64_storage(&formatted_forward, &input)
                            .unwrap();
                    let expected =
                        execute_double_double_nd_r2c_ir_f64_storage(&dense_forward, &input)
                            .unwrap();
                    let error = actual
                        .iter()
                        .zip(&expected)
                        .map(|(actual, expected)| {
                            (actual.re - expected.re).abs() + (actual.im - expected.im).abs()
                        })
                        .fold(0.0, f64::max);
                    assert!(
                        error <= 2.0e-13,
                        "formatted DD/F64 ND-real forward mismatch {error:e}"
                    );

                    let inverse_plan = FftPlan::build(
                        FftConfig::new(real_dimensions.clone())
                            .with_batch_count(2)
                            .with_transform(TransformKind::ComplexToReal)
                            .with_precision(precision)
                            .with_inverse_normalization(true)
                            .with_input_buffer_axis_stride(0, 7)
                            .unwrap()
                            .with_output_buffer_axis_stride(0, 11)
                            .unwrap(),
                    )
                    .unwrap();
                    let inverse = DoubleDoubleNdRealFftIr::build(&inverse_plan).unwrap();
                    let restored =
                        execute_double_double_nd_c2r_ir_f64_storage(&inverse, &actual).unwrap();
                    let round_trip = restored
                        .iter()
                        .zip(&input)
                        .map(|(actual, expected)| (actual - expected).abs())
                        .fold(0.0, f64::max);
                    assert!(
                        round_trip <= 3.0e-13,
                        "formatted DD/F64 ND-real round-trip {round_trip:e}"
                    );
                }
                _ => unreachable!(),
            }
        }

        let r2r_dimensions = vec![3usize, 4];
        let r2r_len = r2r_dimensions.iter().product::<usize>();
        for precision in [Precision::DoubleDouble, Precision::DoubleDoubleF64Storage] {
            let formatted_plan = FftPlan::build(
                FftConfig::new(r2r_dimensions.clone())
                    .with_batch_count(2)
                    .with_transform(TransformKind::Dct(crate::DctType::II))
                    .with_precision(precision)
                    .with_input_buffer_axis_stride(0, 7)
                    .unwrap()
                    .with_output_buffer_axis_stride(0, 9)
                    .unwrap(),
            )
            .unwrap();
            let dense_plan = FftPlan::build(
                FftConfig::new(r2r_dimensions.clone())
                    .with_batch_count(2)
                    .with_transform(TransformKind::Dct(crate::DctType::II))
                    .with_precision(precision),
            )
            .unwrap();
            let formatted =
                DoubleDoubleNdR2rIr::build(&formatted_plan, Direction::Forward).unwrap();
            let dense = DoubleDoubleNdR2rIr::build(&dense_plan, Direction::Forward).unwrap();
            assert_eq!(formatted.input_external_layout.batch_stride, 21);
            assert_eq!(formatted.output_external_layout.batch_stride, 27);
            assert!(formatted.input_formatted_copy.is_some());
            assert!(formatted.output_formatted_copy.is_some());
            match precision {
                Precision::DoubleDouble => {
                    let input = (0..2 * r2r_len)
                        .map(|index| {
                            let x = index as f64;
                            DoubleDouble::from_parts(
                                (0.17 * x).cos() - 0.003 * x,
                                -(index as f64 + 1.0) * 5.0e-32,
                            )
                        })
                        .collect::<Vec<_>>();
                    let actual = execute_double_double_nd_r2r_ir(&formatted, &input).unwrap();
                    let expected = execute_double_double_nd_r2r_ir(&dense, &input).unwrap();
                    let error = actual
                        .iter()
                        .copied()
                        .zip(expected.iter().copied())
                        .map(|(actual, expected)| dd_scalar_error(actual, expected))
                        .fold(0.0, f64::max);
                    assert!(error <= 3.0e-27, "formatted DD ND-R2R mismatch {error:e}");
                }
                Precision::DoubleDoubleF64Storage => {
                    let input = (0..2 * r2r_len)
                        .map(|index| {
                            let x = index as f64;
                            (0.17 * x).cos() - 0.003 * x
                        })
                        .collect::<Vec<_>>();
                    let actual =
                        execute_double_double_nd_r2r_ir_f64_storage(&formatted, &input).unwrap();
                    let expected =
                        execute_double_double_nd_r2r_ir_f64_storage(&dense, &input).unwrap();
                    let error = actual
                        .iter()
                        .zip(&expected)
                        .map(|(actual, expected)| (actual - expected).abs())
                        .fold(0.0, f64::max);
                    assert!(
                        error <= 3.0e-13,
                        "formatted DD/F64 ND-R2R mismatch {error:e}"
                    );
                }
                _ => unreachable!(),
            }
        }
    }

    #[test]
    fn nd_3x4_matches_independent_dd_dft_and_round_trips() {
        let dimensions = [3usize, 4usize];
        let tensor_len = dimensions.iter().product::<usize>();
        let batch_count = 2usize;
        let forward_plan = FftPlan::build(
            FftConfig::new(dimensions.to_vec())
                .with_batch_count(batch_count)
                .with_precision(Precision::DoubleDouble),
        )
        .unwrap();
        let inverse_plan = FftPlan::build(
            FftConfig::new(dimensions.to_vec())
                .with_batch_count(batch_count)
                .with_precision(Precision::DoubleDouble)
                .with_inverse_normalization(true),
        )
        .unwrap();
        let forward = DoubleDoubleNdFftIr::build(&forward_plan, Direction::Forward).unwrap();
        let inverse = DoubleDoubleNdFftIr::build(&inverse_plan, Direction::Inverse).unwrap();
        assert_eq!(
            forward
                .axes
                .iter()
                .map(|axis| axis.axis)
                .collect::<Vec<_>>(),
            vec![1, 0]
        );
        let input = (0..tensor_len * batch_count)
            .map(|index| {
                let x = index as f64;
                ComplexDoubleDouble::new(
                    DoubleDouble::from_parts(
                        (0.17 * x).sin() + 0.003 * x,
                        (index + 1) as f64 * 1.0e-31,
                    ),
                    DoubleDouble::from_parts(
                        (0.11 * x).cos() - 0.002 * x,
                        -(index as f64 + 1.0) * 7.0e-32,
                    ),
                )
            })
            .collect::<Vec<_>>();
        let actual = execute_double_double_nd_ir(&forward, &input).unwrap();
        let mut expected = vec![ComplexDoubleDouble::default(); input.len()];
        for batch in 0..batch_count {
            let base = batch * tensor_len;
            for k0 in 0..dimensions[0] {
                for k1 in 0..dimensions[1] {
                    let mut sum = ComplexDoubleDouble::default();
                    for n0 in 0..dimensions[0] {
                        for n1 in 0..dimensions[1] {
                            let root0 =
                                unit_root(n0 * k0, dimensions[0], Direction::Forward).unwrap();
                            let root1 =
                                unit_root(n1 * k1, dimensions[1], Direction::Forward).unwrap();
                            sum += input[base + n0 * dimensions[1] + n1] * root0 * root1;
                        }
                    }
                    expected[base + k0 * dimensions[1] + k1] = sum;
                }
            }
        }
        let error = actual
            .iter()
            .copied()
            .zip(expected.iter().copied())
            .map(|(actual, expected)| dd_error(actual, expected))
            .fold(0.0, f64::max);
        assert!(error <= 2.0e-27, "DD ND forward mismatch: {error:e}");
        let restored = execute_double_double_nd_ir(&inverse, &actual).unwrap();
        let round_trip_error = restored
            .iter()
            .copied()
            .zip(input.iter().copied())
            .map(|(actual, expected)| dd_error(actual, expected))
            .fold(0.0, f64::max);
        assert!(
            round_trip_error <= 2.0e-27,
            "DD ND round trip mismatch: {round_trip_error:e}"
        );

        let f64_plan = FftPlan::build(
            FftConfig::new(dimensions.to_vec())
                .with_batch_count(batch_count)
                .with_precision(Precision::DoubleDoubleF64Storage),
        )
        .unwrap();
        let f64_ir = DoubleDoubleNdFftIr::build(&f64_plan, Direction::Forward).unwrap();
        let f64_input = input
            .iter()
            .copied()
            .map(ComplexDoubleDouble::to_complex64)
            .collect::<Vec<_>>();
        let f64_actual = execute_double_double_nd_ir_f64_storage(&f64_ir, &f64_input).unwrap();
        let f64_expected = execute_double_double_nd_compute(
            &f64_ir,
            &f64_input
                .iter()
                .copied()
                .map(ComplexDoubleDouble::from_complex64)
                .collect::<Vec<_>>(),
        )
        .unwrap()
        .into_iter()
        .map(ComplexDoubleDouble::to_complex64)
        .collect::<Vec<_>>();
        assert_eq!(f64_actual, f64_expected);
    }

    #[test]
    fn nd_c2c_spatial_zero_padding_matches_manual_dd_boundary() {
        let dimensions = [3usize, 4usize];
        let batch_count = 2usize;
        let tensor_len = dimensions.iter().product::<usize>();
        let input = (0..tensor_len * batch_count)
            .map(|index| {
                let x = index as f64;
                ComplexDoubleDouble::new(
                    DoubleDouble::from_parts(
                        (0.101 * x).sin() + 0.0017 * x,
                        (index + 1) as f64 * 8.0e-32,
                    ),
                    DoubleDouble::from_parts(
                        (0.067 * x).cos() - 0.0011 * x,
                        -(index as f64 + 1.0) * 5.0e-32,
                    ),
                )
            })
            .collect::<Vec<_>>();
        let mut manual = input.clone();
        for batch in 0..batch_count {
            let base = batch * tensor_len;
            for n0 in 0..dimensions[0] {
                for n1 in 0..dimensions[1] {
                    if (1..2).contains(&n0) || (1..3).contains(&n1) {
                        manual[base + n0 * dimensions[1] + n1] = ComplexDoubleDouble::default();
                    }
                }
            }
        }

        let padded = |precision, inverse| {
            FftConfig::new(dimensions.to_vec())
                .with_batch_count(batch_count)
                .with_precision(precision)
                .with_inverse_normalization(inverse)
                .with_zero_padding(0, 1, 2)
                .unwrap()
                .with_zero_padding(1, 1, 3)
                .unwrap()
        };
        let forward = DoubleDoubleNdFftIr::build(
            &FftPlan::build(padded(Precision::DoubleDouble, false)).unwrap(),
            Direction::Forward,
        )
        .unwrap();
        let inverse = DoubleDoubleNdFftIr::build(
            &FftPlan::build(padded(Precision::DoubleDouble, true)).unwrap(),
            Direction::Inverse,
        )
        .unwrap();
        assert!(forward.has_spatial_zero_padding());
        assert!(forward.contains_spatial_zero_linear_index(1));
        assert!(forward.contains_spatial_zero_linear_index(dimensions[1]));
        assert!(!forward.contains_spatial_zero_linear_index(0));
        assert!(
            forward
                .axes
                .iter()
                .all(|axis| axis.transform.external_storage() == PrecisionStorage::DoubleDouble)
        );

        let baseline = DoubleDoubleNdFftIr::build(
            &FftPlan::build(
                FftConfig::new(dimensions.to_vec())
                    .with_batch_count(batch_count)
                    .with_precision(Precision::DoubleDouble),
            )
            .unwrap(),
            Direction::Forward,
        )
        .unwrap();
        let expected = execute_double_double_nd_ir(&baseline, &manual).unwrap();
        let actual = execute_double_double_nd_ir(&forward, &input).unwrap();
        let forward_error = actual
            .iter()
            .copied()
            .zip(expected.iter().copied())
            .map(|(actual, expected)| dd_error(actual, expected))
            .fold(0.0, f64::max);
        assert!(
            forward_error <= 2.0e-27,
            "DD ND padded C2C forward mismatch: {forward_error:e}"
        );
        let restored = execute_double_double_nd_ir(&inverse, &actual).unwrap();
        let round_trip_error = restored
            .iter()
            .copied()
            .zip(manual.iter().copied())
            .map(|(actual, expected)| dd_error(actual, expected))
            .fold(0.0, f64::max);
        assert!(
            round_trip_error <= 2.0e-27,
            "DD ND padded C2C round-trip mismatch: {round_trip_error:e}"
        );

        let f64_forward = DoubleDoubleNdFftIr::build(
            &FftPlan::build(padded(Precision::DoubleDoubleF64Storage, false)).unwrap(),
            Direction::Forward,
        )
        .unwrap();
        let f64_inverse = DoubleDoubleNdFftIr::build(
            &FftPlan::build(padded(Precision::DoubleDoubleF64Storage, true)).unwrap(),
            Direction::Inverse,
        )
        .unwrap();
        let f64_input = input
            .iter()
            .copied()
            .map(ComplexDoubleDouble::to_complex64)
            .collect::<Vec<_>>();
        let f64_manual = manual
            .iter()
            .copied()
            .map(ComplexDoubleDouble::to_complex64)
            .collect::<Vec<_>>();
        let f64_actual = execute_double_double_nd_ir_f64_storage(&f64_forward, &f64_input).unwrap();
        let f64_restored =
            execute_double_double_nd_ir_f64_storage(&f64_inverse, &f64_actual).unwrap();
        let f64_error = f64_restored
            .iter()
            .zip(&f64_manual)
            .map(|(actual, expected)| (*actual - *expected).norm_sqr().sqrt())
            .fold(0.0, f64::max);
        assert!(
            f64_error <= 3.0e-14,
            "DD/F64 ND padded C2C mismatch: {f64_error:e}"
        );
    }

    #[test]
    fn nd_c2c_frequency_zero_padding_matches_manual_dd_frequency_boundary() {
        let dimensions = [3usize, 4usize];
        let tensor_len = dimensions.iter().product::<usize>();
        let input = (0..tensor_len)
            .map(|index| {
                let x = index as f64;
                ComplexDoubleDouble::new(
                    DoubleDouble::from_parts(
                        (0.113 * x).sin() + 0.003 * x,
                        (index + 1) as f64 * 7.0e-32,
                    ),
                    DoubleDouble::from_parts(
                        (0.071 * x).cos() - 0.002 * x,
                        -(index as f64 + 1.0) * 4.0e-32,
                    ),
                )
            })
            .collect::<Vec<_>>();
        let base_config = |inverse| {
            FftConfig::new(dimensions.to_vec())
                .with_precision(Precision::DoubleDouble)
                .with_inverse_normalization(inverse)
        };
        let frequency_config = |inverse| {
            base_config(inverse)
                .with_zero_padding(0, 2, 3)
                .unwrap()
                .with_zero_padding(1, 3, 4)
                .unwrap()
                .with_zero_padding_domain(ZeroPaddingDomain::Frequency)
        };
        let forward = DoubleDoubleNdFftIr::build(
            &FftPlan::build(frequency_config(false)).unwrap(),
            Direction::Forward,
        )
        .unwrap();
        let inverse = DoubleDoubleNdFftIr::build(
            &FftPlan::build(frequency_config(true)).unwrap(),
            Direction::Inverse,
        )
        .unwrap();
        assert!(forward.has_frequency_zero_padding());
        assert!(!forward.has_spatial_zero_padding());
        assert!(forward.zero_padding_is_output_boundary());
        assert!(inverse.zero_padding_is_input_boundary());

        let baseline_forward = DoubleDoubleNdFftIr::build(
            &FftPlan::build(base_config(false)).unwrap(),
            Direction::Forward,
        )
        .unwrap();
        let baseline_inverse = DoubleDoubleNdFftIr::build(
            &FftPlan::build(base_config(true)).unwrap(),
            Direction::Inverse,
        )
        .unwrap();
        let spectrum = execute_double_double_nd_ir(&baseline_forward, &input).unwrap();
        let mut masked_spectrum = spectrum.clone();
        for n0 in 0..dimensions[0] {
            for n1 in 0..dimensions[1] {
                if n0 == 2 || n1 == 3 {
                    masked_spectrum[n0 * dimensions[1] + n1] = ComplexDoubleDouble::default();
                }
            }
        }

        let actual_forward = execute_double_double_nd_ir(&forward, &input).unwrap();
        let forward_error = actual_forward
            .iter()
            .copied()
            .zip(masked_spectrum.iter().copied())
            .map(|(actual, expected)| dd_error(actual, expected))
            .fold(0.0, f64::max);
        assert!(
            forward_error <= 2.0e-27,
            "DD ND frequency-forward mismatch: {forward_error:e}"
        );

        let actual_inverse = execute_double_double_nd_ir(&inverse, &spectrum).unwrap();
        let expected_inverse =
            execute_double_double_nd_ir(&baseline_inverse, &masked_spectrum).unwrap();
        let inverse_error = actual_inverse
            .iter()
            .copied()
            .zip(expected_inverse.iter().copied())
            .map(|(actual, expected)| dd_error(actual, expected))
            .fold(0.0, f64::max);
        assert!(
            inverse_error <= 2.0e-27,
            "DD ND frequency-inverse mismatch: {inverse_error:e}"
        );
    }

    #[test]
    fn bluestein_p103_matches_independent_dd_dft_and_round_trips() {
        let length = 103usize;
        let mut tuning = crate::PlannerTuning::portable();
        tuning.max_rader_fft_prime = 100;
        let forward_plan = FftPlan::build(
            FftConfig::new(vec![length])
                .with_precision(Precision::DoubleDouble)
                .with_tuning(tuning),
        )
        .unwrap();
        let inverse_plan = FftPlan::build(
            FftConfig::new(vec![length])
                .with_precision(Precision::DoubleDouble)
                .with_inverse_normalization(true)
                .with_tuning(tuning),
        )
        .unwrap();
        let forward = DoubleDoubleOneDimIr::build(&forward_plan, Direction::Forward).unwrap();
        let inverse = DoubleDoubleOneDimIr::build(&inverse_plan, Direction::Inverse).unwrap();
        let DoubleDoubleOneDimIr::Bluestein(forward_bluestein) = &forward else {
            panic!("forced p103 DD plan must preserve Bluestein selection");
        };
        assert_eq!(forward_bluestein.logical_len, length);
        assert_eq!(forward_bluestein.convolution_len, 210);
        assert_eq!(forward_bluestein.forward_fft.sequence_len(), 210);
        assert_eq!(forward_bluestein.inverse_fft.sequence_len(), 210);
        assert_eq!(forward_bluestein.kernel_spectrum.len(), 210);
        assert!(
            forward_bluestein
                .table
                .chirp
                .iter()
                .any(|value| { value.re.lo != 0.0 || value.im.lo != 0.0 })
        );

        let input = (0..length)
            .map(|index| {
                let x = index as f64;
                ComplexDoubleDouble::new(
                    DoubleDouble::from_parts(
                        (0.017 * x).sin() + 0.00009 * x,
                        (index + 1) as f64 * 1.0e-31,
                    ),
                    DoubleDouble::from_parts(
                        (0.011 * x).cos() - 0.00004 * x,
                        -(index as f64 + 1.0) * 5.0e-32,
                    ),
                )
            })
            .collect::<Vec<_>>();
        let spectrum = execute_double_double_one_dim_ir(&forward, &input).unwrap();
        let expected = dft(&input, Direction::Forward, false).unwrap();
        let error = spectrum
            .iter()
            .copied()
            .zip(expected)
            .map(|(actual, expected)| dd_error(actual, expected))
            .fold(0.0, f64::max);
        assert!(
            error < 2.0e-24,
            "double-double p103 Bluestein error {error:e}"
        );

        let restored = execute_double_double_one_dim_ir(&inverse, &spectrum).unwrap();
        let round_trip_error = restored
            .iter()
            .copied()
            .zip(input.iter().copied())
            .map(|(actual, expected)| dd_error(actual, expected))
            .fold(0.0, f64::max);
        assert!(
            round_trip_error < 2.0e-22,
            "double-double p103 Bluestein round-trip error {round_trip_error:e}"
        );
    }

    #[test]
    fn bluestein_p103_zero_padding_composes_with_grouped_tail_batch() {
        let length = 103usize;
        let batch_count = 7usize;
        let grouped_batch = 3usize;
        let zero_left = 19usize;
        let zero_right = 31usize;
        let mut tuning = crate::PlannerTuning::portable();
        tuning.max_rader_fft_prime = 100;
        let forward_plan = FftPlan::build(
            FftConfig::new(vec![length])
                .with_batch_count(batch_count)
                .with_precision(Precision::DoubleDouble)
                .with_tuning(tuning)
                .with_grouped_batch(0, grouped_batch)
                .unwrap()
                .with_zero_padding(0, zero_left, zero_right)
                .unwrap(),
        )
        .unwrap();
        let inverse_plan = FftPlan::build(
            FftConfig::new(vec![length])
                .with_batch_count(batch_count)
                .with_precision(Precision::DoubleDouble)
                .with_inverse_normalization(true)
                .with_tuning(tuning)
                .with_grouped_batch(0, grouped_batch)
                .unwrap()
                .with_zero_padding(0, zero_left, zero_right)
                .unwrap(),
        )
        .unwrap();
        let forward = DoubleDoubleOneDimIr::build(&forward_plan, Direction::Forward).unwrap();
        let inverse = DoubleDoubleOneDimIr::build(&inverse_plan, Direction::Inverse).unwrap();
        let DoubleDoubleOneDimIr::Bluestein(forward_bluestein) = &forward else {
            panic!("padded grouped p103 DD plan must preserve Bluestein selection");
        };
        let DoubleDoubleOneDimIr::Bluestein(inverse_bluestein) = &inverse else {
            panic!("padded grouped inverse p103 DD plan must preserve Bluestein selection");
        };
        assert_eq!(forward_bluestein.grouped_batch, grouped_batch);
        assert_eq!(forward_bluestein.batch_group_count(), 3);
        assert!(forward_bluestein.has_spatial_zero_padding());
        assert!(forward_bluestein.contains_spatial_zero_index(zero_left));
        assert!(forward_bluestein.contains_spatial_zero_index(zero_right - 1));
        assert!(!forward_bluestein.contains_spatial_zero_index(zero_right));
        assert_eq!(
            inverse_bluestein.zero_padding,
            forward_bluestein.zero_padding
        );

        let input = (0..length * batch_count)
            .map(|index| {
                let x = index as f64;
                ComplexDoubleDouble::new(
                    DoubleDouble::from_parts(
                        (0.013 * x).sin() + 0.00007 * x + 17.0,
                        (index + 1) as f64 * 2.0e-31,
                    ),
                    DoubleDouble::from_parts(
                        (0.009 * x).cos() - 0.00003 * x - 11.0,
                        -(index as f64 + 1.0) * 1.0e-31,
                    ),
                )
            })
            .collect::<Vec<_>>();
        let mut manual_zeroed = input.clone();
        for batch in 0..batch_count {
            let base = batch * length;
            for index in zero_left..zero_right {
                manual_zeroed[base + index] = ComplexDoubleDouble::default();
            }
        }
        let actual = execute_double_double_one_dim_ir(&forward, &input).unwrap();
        let mut expected = Vec::with_capacity(actual.len());
        for batch in 0..batch_count {
            let base = batch * length;
            expected.extend(
                dft(
                    &manual_zeroed[base..base + length],
                    Direction::Forward,
                    false,
                )
                .unwrap(),
            );
        }
        let forward_error = actual
            .iter()
            .copied()
            .zip(expected.iter().copied())
            .map(|(actual, expected)| dd_error(actual, expected))
            .fold(0.0, f64::max);
        assert!(
            forward_error < 2.0e-22,
            "double-double grouped padded p103 Bluestein forward error {forward_error:e}"
        );

        let restored = execute_double_double_one_dim_ir(&inverse, &actual).unwrap();
        let round_trip_error = restored
            .iter()
            .copied()
            .zip(manual_zeroed.iter().copied())
            .map(|(actual, expected)| dd_error(actual, expected))
            .fold(0.0, f64::max);
        assert!(
            round_trip_error < 2.0e-20,
            "double-double grouped padded p103 Bluestein round-trip error {round_trip_error:e}"
        );
    }

    #[test]
    fn bluestein_p2053_large_shared_device_keeps_m4116_single_upload_stockham_child() {
        let length = 2_053usize;
        let mut tuning = crate::PlannerTuning::portable();
        tuning.max_rader_fft_prime = 100;
        let plan = FftPlan::build(
            FftConfig::new(vec![length])
                .with_precision(Precision::DoubleDouble)
                .with_tuning(tuning),
        )
        .unwrap();
        let portable = DoubleDoubleBluesteinIr::build(&plan, Direction::Forward).unwrap();
        assert_eq!(portable.convolution_len, 4_116);
        assert!(matches!(
            portable.forward_fft,
            DoubleDoubleBluesteinConvolutionIr::Recursive(_)
        ));

        let mut device = DeviceProfile::generic(crate::Backend::Vulkan, crate::GpuVendor::Nvidia);
        device.shared_memory_bytes = 164 * 1024;
        device.shared_memory_pow2_bytes = 164 * 1024;
        device.max_threads_per_block = 1024;
        device.max_workgroup_size = [1024, 1024, 64];
        device.supports_f64 = true;

        let scheduled =
            DoubleDoubleBluesteinIr::build_for_device(&plan, Direction::Forward, device).unwrap();
        assert_eq!(scheduled.logical_len, length);
        assert_eq!(scheduled.convolution_len, 4_116);
        for child in [&scheduled.forward_fft, &scheduled.inverse_fft] {
            let DoubleDoubleBluesteinConvolutionIr::Stockham(child) = child else {
                panic!("164KiB p2053 M4116 convolution should stay single-upload Stockham");
            };
            let block = child
                .axis_batch_block
                .expect("large Bluestein Stockham child must retain its physical block");
            assert_eq!(block.threads_per_transform, 686);
            assert_eq!(block.grouped_batch, 1);
            assert_eq!([block.local_size_x, block.local_size_y], [686, 1]);
            assert!(!block.transforms_on_x);
            assert!(!block.axis_swapped);
        }
        assert_eq!(
            scheduled.wrapper_axis_batch_block(),
            scheduled.stockham_convolution_axis_batch_block()
        );
        assert_eq!(scheduled.kernel_spectrum, portable.kernel_spectrum);

        let program = crate::ProgramIr::double_double_bluestein(&scheduled).unwrap();
        program.validate().unwrap();
        assert!(program.passes.iter().all(|pass| {
            !pass.name.contains("pack_right") && !pass.name.contains("twiddle_transpose")
        }));
        let shaders = crate::backend::vulkan::VulkanGlslBackend
            .lower_double_double_bluestein(&scheduled)
            .unwrap();
        assert_eq!(shaders.len(), program.passes.len());
        assert!(shaders.iter().all(|shader| shader.compile_spirv().is_ok()));

        let impulse_index = 137usize;
        let impulse = ComplexDoubleDouble::new(
            DoubleDouble::from_parts(1.25, 3.0e-31),
            DoubleDouble::from_parts(-0.75, -2.0e-31),
        );
        let mut input = vec![ComplexDoubleDouble::default(); length];
        input[impulse_index] = impulse;
        let actual = execute_double_double_bluestein_ir(&scheduled, &input).unwrap();
        let max_error = actual
            .iter()
            .copied()
            .enumerate()
            .map(|(k, actual)| {
                let expected =
                    impulse * unit_root(impulse_index * k, length, Direction::Forward).unwrap();
                dd_error(actual, expected)
            })
            .fold(0.0, f64::max);
        assert!(
            max_error < 2.0e-22,
            "large-shared DD p2053/M4116 impulse error {max_error:e}"
        );
    }

    #[test]
    fn bluestein_p2203_device_padding_uses_single_upload_m4608_and_upstream_block() {
        let length = 2_203usize;
        let mut device = DeviceProfile::generic(crate::Backend::Vulkan, crate::GpuVendor::Nvidia);
        device.shared_memory_bytes = 164 * 1024;
        device.shared_memory_pow2_bytes = 164 * 1024;
        device.max_threads_per_block = 1024;
        device.max_workgroup_size = [1024, 1024, 64];
        device.supports_f64 = true;

        let mut tuning = crate::PlannerTuning::for_device(device, Precision::DoubleDouble);
        tuning.max_rader_fft_prime = 100;
        let config = FftConfig::new(vec![length])
            .with_precision(Precision::DoubleDouble)
            .with_tuning(tuning);
        let plan = FftPlan::build_for_device(config, device).unwrap();
        let AxisAlgorithm::Bluestein {
            convolution_len, ..
        } = plan.axes[0].algorithm
        else {
            panic!("164KiB tuned DD p2203 should select Bluestein");
        };
        assert_eq!(convolution_len, 4_608);

        let ir = DoubleDoubleOneDimIr::build_for_device(&plan, Direction::Forward, device).unwrap();
        let DoubleDoubleOneDimIr::Bluestein(bluestein) = &ir else {
            panic!("164KiB tuned DD p2203 should remain a top-level Bluestein transform");
        };
        assert_eq!(bluestein.convolution_len, 4_608);
        for child in [&bluestein.forward_fft, &bluestein.inverse_fft] {
            let DoubleDoubleBluesteinConvolutionIr::Stockham(child) = child else {
                panic!("p2203 M4608 convolution should stay single-upload Stockham on 164KiB");
            };
            let block = child
                .axis_batch_block
                .expect("p2203 M4608 child must retain the upstream physical block");
            assert_eq!(block.threads_per_transform, 768);
            assert_eq!(block.grouped_batch, 1);
            assert_eq!([block.local_size_x, block.local_size_y], [768, 1]);
            assert!(!block.transforms_on_x);
            assert!(!block.axis_swapped);
        }
        assert_eq!(
            bluestein.wrapper_axis_batch_block(),
            bluestein.stockham_convolution_axis_batch_block()
        );

        let program = crate::ProgramIr::double_double_one_dim(&ir).unwrap();
        program.validate().unwrap();
        assert!(program.passes.iter().all(|pass| {
            !pass.name.contains("pack_right") && !pass.name.contains("twiddle_transpose")
        }));
        let shaders = crate::backend::vulkan::VulkanGlslBackend
            .lower_double_double_one_dim_program(&ir)
            .unwrap();
        assert_eq!(shaders.len(), program.passes.len());
        assert!(shaders.iter().all(|shader| {
            shader.workgroup_size.x == 768
                && shader.workgroup_size.y == 1
                && shader.compile_spirv().is_ok()
        }));

        let impulse_index = 173usize;
        let impulse = ComplexDoubleDouble::new(
            DoubleDouble::from_parts(1.125, 2.0e-31),
            DoubleDouble::from_parts(-0.625, -3.0e-31),
        );
        let mut input = vec![ComplexDoubleDouble::default(); length];
        input[impulse_index] = impulse;
        let actual = execute_double_double_one_dim_ir(&ir, &input).unwrap();
        let max_error = actual
            .iter()
            .copied()
            .enumerate()
            .map(|(k, actual)| {
                let expected =
                    impulse * unit_root(impulse_index * k, length, Direction::Forward).unwrap();
                dd_error(actual, expected)
            })
            .fold(0.0, f64::max);
        assert!(
            max_error < 2.0e-22,
            "large-shared DD p2203/M4608 impulse error {max_error:e}"
        );
    }

    #[test]
    fn large_shared_bluestein_strided_child_uses_one_upload_stockham_block() {
        let mut device = DeviceProfile::generic(crate::Backend::Vulkan, crate::GpuVendor::Nvidia);
        device.shared_memory_bytes = 164 * 1024;
        device.shared_memory_pow2_bytes = 164 * 1024;
        device.max_threads_per_block = 1024;
        device.max_workgroup_size = [1024, 1024, 64];
        device.supports_f64 = true;

        let plan = FftPlan::build_c2c_bluestein_child_for_device(
            FftConfig::new(vec![4_608])
                .with_batch_count(2)
                .with_precision(Precision::DoubleDouble),
            device,
            C2cDeviceAxisClass::Strided,
        )
        .unwrap();
        assert_eq!(
            plan.c2c_device_axis_class_override,
            Some(C2cDeviceAxisClass::Strided)
        );
        assert!(plan.c2c_device_use_bluestein_fft_override);
        assert!(!double_double_stockham_requires_multi_upload(&plan, 4_608, device).unwrap());

        let child =
            DoubleDoubleBluesteinConvolutionIr::build_for_device(&plan, Direction::Forward, device)
                .unwrap();
        let DoubleDoubleBluesteinConvolutionIr::Stockham(child) = child else {
            panic!("large strided M4608 Bluestein child should remain one-upload Stockham");
        };
        let block = child
            .axis_batch_block
            .expect("large strided M4608 Bluestein child must retain its higher-axis block");
        assert_eq!(block.threads_per_transform, 768);
        assert_eq!(block.grouped_batch, 1);
        assert!(block.transforms_on_x);
        assert!(!block.axis_swapped);
        assert_eq!([block.local_size_x, block.local_size_y], [1, 768]);
        child.validate().unwrap();
    }

    #[test]
    fn nd_p2203_large_shared_bluestein_keeps_m4608_higher_axis_stockham_block() {
        let mut device = DeviceProfile::generic(crate::Backend::Vulkan, crate::GpuVendor::Nvidia);
        device.shared_memory_bytes = 164 * 1024;
        device.shared_memory_pow2_bytes = 164 * 1024;
        device.max_threads_per_block = 1024;
        device.max_workgroup_size = [1024, 1024, 64];
        device.supports_f64 = true;

        let mut tuning = crate::PlannerTuning::for_device(device, Precision::DoubleDouble);
        tuning.max_rader_fft_prime = 100;
        let dimensions = [2_203usize, 2usize];
        let forward_plan = FftPlan::build_for_device(
            FftConfig::new(dimensions.to_vec())
                .with_precision(Precision::DoubleDouble)
                .with_tuning(tuning),
            device,
        )
        .unwrap();
        let inverse_plan = FftPlan::build_for_device(
            FftConfig::new(dimensions.to_vec())
                .with_precision(Precision::DoubleDouble)
                .with_inverse_normalization(true)
                .with_tuning(tuning),
            device,
        )
        .unwrap();
        let scheduled =
            DoubleDoubleNdFftIr::build_for_device(&forward_plan, Direction::Forward, device)
                .unwrap()
                .with_grouped_stockham_axis_blocks(None, device)
                .unwrap();
        let inverse =
            DoubleDoubleNdFftIr::build_for_device(&inverse_plan, Direction::Inverse, device)
                .unwrap()
                .with_grouped_stockham_axis_blocks(None, device)
                .unwrap();
        let outer = scheduled.axes.iter().find(|axis| axis.axis == 0).unwrap();
        assert_eq!(outer.axis_len, 2_203);
        assert_eq!(outer.line_count, 2);
        let DoubleDoubleOneDimIr::Bluestein(bluestein) = &outer.transform else {
            panic!("164KiB DD [2203,2] outer axis should use Bluestein");
        };
        assert_eq!(bluestein.convolution_len, 4_608);
        for child in [&bluestein.forward_fft, &bluestein.inverse_fft] {
            let DoubleDoubleBluesteinConvolutionIr::Stockham(child) = child else {
                panic!("higher-axis p2203 M4608 convolution should stay one-upload Stockham");
            };
            let block = child
                .axis_batch_block
                .expect("higher-axis p2203 M4608 child must preserve the Bluestein-specific block");
            assert_eq!(child.batch_count, 2);
            assert_eq!(block.threads_per_transform, 768);
            assert_eq!(block.grouped_batch, 1);
            assert!(block.transforms_on_x);
            assert!(!block.axis_swapped);
            assert_eq!([block.local_size_x, block.local_size_y], [1, 768]);
        }

        let program = crate::ProgramIr::double_double_nd(&scheduled).unwrap();
        program.validate().unwrap();
        let shaders = crate::backend::vulkan::VulkanGlslBackend
            .lower_double_double_nd(&scheduled)
            .unwrap();
        assert_eq!(shaders.len(), program.passes.len());
        for shader in shaders {
            assert_eq!(shader.compile_spirv().unwrap().words[0], 0x0723_0203);
        }

        let n0 = 137usize;
        let n1 = 1usize;
        let impulse = ComplexDoubleDouble::new(
            DoubleDouble::from_parts(1.125, 4.0e-31),
            DoubleDouble::from_parts(-0.875, -3.0e-31),
        );
        let mut input = vec![ComplexDoubleDouble::default(); dimensions.iter().product()];
        input[n0 * dimensions[1] + n1] = impulse;
        let actual = execute_double_double_nd_ir(&scheduled, &input).unwrap();
        let mut forward_error = 0.0f64;
        for k0 in 0..dimensions[0] {
            let root0 = unit_root(n0 * k0, dimensions[0], Direction::Forward).unwrap();
            for k1 in 0..dimensions[1] {
                let root1 = unit_root(n1 * k1, dimensions[1], Direction::Forward).unwrap();
                let expected = impulse * root0 * root1;
                forward_error =
                    forward_error.max(dd_error(actual[k0 * dimensions[1] + k1], expected));
            }
        }
        assert!(
            forward_error < 5.0e-20,
            "DD ND [2203,2] higher-axis M4608 impulse error {forward_error:e}"
        );
        let restored = execute_double_double_nd_ir(&inverse, &actual).unwrap();
        let round_trip_error = restored
            .iter()
            .copied()
            .zip(input.iter().copied())
            .map(|(actual, expected)| dd_error(actual, expected))
            .fold(0.0, f64::max);
        assert!(
            round_trip_error < 5.0e-18,
            "DD ND [2203,2] higher-axis M4608 round-trip error {round_trip_error:e}"
        );
        scheduled.validate().unwrap();
        inverse.validate().unwrap();
    }

    #[test]
    fn bluestein_p2053_device_aware_convolution_uses_fixed_padding_and_p13_rader() {
        fn contains_direct_p13(
            node: &crate::double_double_recursive_ir::DoubleDoubleRecursiveFftNodeIr,
        ) -> bool {
            use crate::double_double_recursive_ir::DoubleDoubleRecursiveFftNodeIr;
            match node {
                DoubleDoubleRecursiveFftNodeIr::DirectRader(rader) => rader.prime == 13,
                DoubleDoubleRecursiveFftNodeIr::CooleyTukey(cooley) => {
                    contains_direct_p13(&cooley.left) || contains_direct_p13(&cooley.right)
                }
                DoubleDoubleRecursiveFftNodeIr::Stockham(_)
                | DoubleDoubleRecursiveFftNodeIr::FftRader(_)
                | DoubleDoubleRecursiveFftNodeIr::Bluestein(_) => false,
            }
        }

        let length = 2_053usize;
        let device = DeviceProfile {
            shared_memory_bytes: 48 * 1024,
            shared_memory_pow2_bytes: 32 * 1024,
            max_threads_per_block: 1024,
            max_workgroup_size: [1024, 1024, 64],
            supports_f64: true,
            ..DeviceProfile::generic(crate::Backend::Vulkan, crate::GpuVendor::Nvidia)
        };
        let mut tuning = crate::PlannerTuning::for_device(device, Precision::DoubleDouble);
        tuning.max_rader_fft_prime = 100;
        let config = FftConfig::new(vec![length])
            .with_precision(Precision::DoubleDouble)
            .with_tuning(tuning);
        let portable_plan = FftPlan::build(config.clone()).unwrap();
        let device_plan = FftPlan::build_for_device(config.clone(), device).unwrap();

        let portable = DoubleDoubleBluesteinIr::build(&portable_plan, Direction::Forward).unwrap();
        assert_eq!(portable.convolution_len, 4_116);
        let DoubleDoubleBluesteinConvolutionIr::Recursive(portable_child) = &portable.forward_fft
        else {
            panic!("portable p2053 convolution should remain recursive M4116");
        };
        assert!(portable_child.stockham_upload_schedule.is_none());

        let scheduled =
            DoubleDoubleBluesteinIr::build_for_device(&device_plan, Direction::Forward, device)
                .unwrap();
        assert_eq!(scheduled.logical_len, length);
        assert_eq!(scheduled.convolution_len, 4_368);
        assert!(scheduled.wrapper_axis_batch_block().is_some());
        for child in [&scheduled.forward_fft, &scheduled.inverse_fft] {
            let DoubleDoubleBluesteinConvolutionIr::Recursive(child) = child else {
                panic!("device-aware p2053 convolution should use recursive M4368");
            };
            assert_eq!(child.logical_len, 4_368);
            assert!(child.stockham_upload_schedule.is_none());
            assert!(contains_direct_p13(&child.root));
            crate::ProgramIr::double_double_recursive(child)
                .unwrap()
                .validate()
                .unwrap();
        }

        let program = crate::ProgramIr::double_double_bluestein(&scheduled).unwrap();
        program.validate().unwrap();
        let shaders = crate::backend::vulkan::VulkanGlslBackend
            .lower_double_double_bluestein(&scheduled)
            .unwrap();
        assert_eq!(shaders.len(), program.passes.len());
        assert!(shaders.iter().all(|shader| shader.compile_spirv().is_ok()));

        let f64 = crate::TransformIr::build(
            FftConfig::new(vec![length])
                .with_precision(Precision::DoubleDoubleF64Storage)
                .with_tuning(tuning),
            Direction::Forward,
            device,
        )
        .unwrap();
        let crate::TransformIr::Complex1dDoubleDouble(DoubleDoubleOneDimIr::Bluestein(f64)) = f64
        else {
            panic!("high-level DD/F64 p2053 should remain Bluestein");
        };
        assert_eq!(f64.external_storage, PrecisionStorage::F64);
        assert_eq!(f64.convolution_len, 4_368);
        let DoubleDoubleBluesteinConvolutionIr::Recursive(f64_child) = &f64.forward_fft else {
            panic!("high-level DD/F64 p2053 should use recursive M4368 child");
        };
        assert!(f64_child.stockham_upload_schedule.is_none());
        assert!(contains_direct_p13(&f64_child.root));
    }

    #[test]
    fn bluestein_n3196_device_child_matches_upstream_two_upload_blocks_and_spirv() {
        fn contains_direct_p11(node: &crate::DoubleDoubleRecursiveFftNodeIr) -> bool {
            match node {
                crate::DoubleDoubleRecursiveFftNodeIr::DirectRader(rader) => rader.prime == 11,
                crate::DoubleDoubleRecursiveFftNodeIr::CooleyTukey(cooley) => {
                    contains_direct_p11(&cooley.left) || contains_direct_p11(&cooley.right)
                }
                _ => false,
            }
        }

        let device = DeviceProfile {
            shared_memory_bytes: 48 * 1024,
            shared_memory_pow2_bytes: 32 * 1024,
            max_threads_per_block: 1024,
            max_workgroup_size: [1024, 1024, 64],
            coalesced_memory_bytes: 32,
            supports_f64: true,
            ..DeviceProfile::generic(crate::Backend::Vulkan, crate::GpuVendor::Amd)
        };
        let plan = FftPlan::build_for_device(
            FftConfig::new(vec![3_196]).with_precision(Precision::DoubleDouble),
            device,
        )
        .unwrap();
        let AxisAlgorithm::Bluestein {
            convolution_len, ..
        } = plan.axes[0].algorithm
        else {
            panic!("AMD48 DD N3196 should use device-default Bluestein");
        };
        assert_eq!(convolution_len, 6_875);

        let bluestein =
            DoubleDoubleBluesteinIr::build_for_device(&plan, Direction::Forward, device).unwrap();
        assert_eq!(bluestein.convolution_len, 6_875);
        for child in [&bluestein.forward_fft, &bluestein.inverse_fft] {
            let DoubleDoubleBluesteinConvolutionIr::Recursive(child) = child else {
                panic!("DD N3196 M6875 convolution should use recursive device scheduling");
            };
            assert_eq!(
                child
                    .rader_forced_upload_schedule
                    .as_ref()
                    .expect("M6875 must retain the capacity-driven two-upload schedule")
                    .axis_split,
                vec![125, 55]
            );
            let (low, _) = child
                .forced_rader_two_upload_mapped_low_stockham()
                .unwrap()
                .expect("M6875 upload0 must materialize N125 Stockham");
            let low_block = low
                .axis_batch_block
                .expect("M6875 N125 upload0 must own a physical block");
            assert_eq!([low_block.local_size_x, low_block.local_size_y], [25, 5]);
            assert!(!low_block.transforms_on_x);
            assert!(!low_block.axis_swapped);

            let high = child
                .forced_rader_two_upload_mapped_high_component()
                .unwrap()
                .expect("M6875 upload1 must materialize N55 recursive component");
            assert!(contains_direct_p11(&high));
            let crate::DoubleDoubleRecursiveFftNodeIr::CooleyTukey(high) = high else {
                panic!("M6875 N55 upload1 should remain Cooley-Tukey 5xp11");
            };
            let high_block = high
                .pack_right
                .axis_batch_block
                .expect("M6875 N55 upload1 must own a physical block");
            assert_eq!([high_block.local_size_x, high_block.local_size_y], [24, 30]);
            assert!(high_block.transforms_on_x);
            assert!(!high_block.axis_swapped);
        }

        let program = crate::ProgramIr::double_double_bluestein(&bluestein).unwrap();
        program.validate().unwrap();
        let shaders = crate::backend::vulkan::VulkanGlslBackend
            .lower_double_double_bluestein(&bluestein)
            .unwrap();
        assert_eq!(shaders.len(), program.passes.len());
        assert!(shaders.iter().all(|shader| shader.compile_spirv().is_ok()));
    }

    #[test]
    fn bluestein_table_miss_uses_quad_generic_padding_and_device_uploads() {
        let length = 4_106usize;
        let device = DeviceProfile {
            shared_memory_bytes: 48 * 1024,
            shared_memory_pow2_bytes: 32 * 1024,
            max_threads_per_block: 1024,
            max_workgroup_size: [1024, 1024, 64],
            supports_f64: true,
            ..DeviceProfile::generic(crate::Backend::Vulkan, crate::GpuVendor::Nvidia)
        };
        let mut tuning = crate::PlannerTuning::portable();
        tuning.max_rader_fft_prime = 100;
        let plan = FftPlan::build_for_device(
            FftConfig::new(vec![length])
                .with_precision(Precision::DoubleDouble)
                .with_tuning(tuning),
            device,
        )
        .unwrap();
        let AxisAlgorithm::Bluestein {
            convolution_len, ..
        } = plan.axes[0].algorithm
        else {
            panic!("forced DD N4106 should use Bluestein");
        };
        assert_eq!(convolution_len, 8_232);

        let bluestein =
            DoubleDoubleBluesteinIr::build_for_device(&plan, Direction::Forward, device).unwrap();
        assert_eq!(bluestein.logical_len, length);
        assert_eq!(bluestein.convolution_len, 8_232);
        for child in [&bluestein.forward_fft, &bluestein.inverse_fft] {
            let DoubleDoubleBluesteinConvolutionIr::Recursive(child) = child else {
                panic!("DD M8232 convolution should use recursive device scheduling");
            };
            let schedule = child
                .stockham_upload_schedule
                .as_ref()
                .expect("DD M8232 child should retain a device upload schedule");
            assert_eq!(schedule.upload_count, 2);
            // Bluestein owns a unit-stride first upload, so the generic divisor
            // search keeps the shared-capacity quotient in locAxisSplit[0].
            assert_eq!(schedule.axis_split, vec![98, 84]);
            assert!(child.two_upload_four_step_plan.is_some());
        }
        let program = crate::ProgramIr::double_double_bluestein(&bluestein).unwrap();
        program.validate().unwrap();
        let shaders = crate::backend::vulkan::VulkanGlslBackend
            .lower_double_double_bluestein(&bluestein)
            .unwrap();
        assert_eq!(shaders.len(), program.passes.len());
        assert!(shaders.iter().all(|shader| shader.compile_spirv().is_ok()));
    }

    #[test]
    fn bluestein_p2053_uses_recursive_4116_point_convolution() {
        let length = 2_053usize;
        let mut tuning = crate::PlannerTuning::portable();
        tuning.max_rader_fft_prime = 100;
        let plan = FftPlan::build(
            FftConfig::new(vec![length])
                .with_precision(Precision::DoubleDouble)
                .with_tuning(tuning),
        )
        .unwrap();
        let ir = DoubleDoubleOneDimIr::build(&plan, Direction::Forward).unwrap();
        let DoubleDoubleOneDimIr::Bluestein(bluestein) = ir else {
            panic!("forced p2053 DD plan must use Bluestein");
        };
        assert_eq!(bluestein.logical_len, length);
        assert_eq!(bluestein.convolution_len, 4_116);
        assert!(matches!(
            bluestein.forward_fft,
            DoubleDoubleBluesteinConvolutionIr::Recursive(_)
        ));
        assert!(matches!(
            bluestein.inverse_fft,
            DoubleDoubleBluesteinConvolutionIr::Recursive(_)
        ));
        assert_eq!(bluestein.kernel_spectrum.len(), 4_116);
        let program = crate::ProgramIr::double_double_bluestein(&bluestein).unwrap();
        let shaders = crate::backend::vulkan::VulkanGlslBackend
            .lower_double_double_bluestein(&bluestein)
            .unwrap();
        assert_eq!(shaders.len(), program.passes.len());
        assert!(shaders.len() > 10);
        assert!(shaders.iter().all(|shader| shader.compile_spirv().is_ok()));

        let impulse_index = 137usize;
        let impulse = ComplexDoubleDouble::new(
            DoubleDouble::from_parts(1.25, 3.0e-31),
            DoubleDouble::from_parts(-0.75, -2.0e-31),
        );
        let mut input = vec![ComplexDoubleDouble::default(); length];
        input[impulse_index] = impulse;
        let actual = execute_double_double_bluestein_ir(&bluestein, &input).unwrap();
        let max_error = actual
            .iter()
            .copied()
            .enumerate()
            .map(|(k, actual)| {
                let expected =
                    impulse * unit_root(impulse_index * k, length, Direction::Forward).unwrap();
                dd_error(actual, expected)
            })
            .fold(0.0, f64::max);
        assert!(
            max_error < 2.0e-22,
            "recursive-convolution DD Bluestein impulse error {max_error:e}"
        );

        let inverse_plan = FftPlan::build(
            FftConfig::new(vec![length])
                .with_precision(Precision::DoubleDouble)
                .with_inverse_normalization(true)
                .with_tuning(tuning),
        )
        .unwrap();
        let inverse = DoubleDoubleBluesteinIr::build(&inverse_plan, Direction::Inverse).unwrap();
        assert!(matches!(
            inverse.inverse_fft,
            DoubleDoubleBluesteinConvolutionIr::Recursive(_)
        ));
        let restored = execute_double_double_bluestein_ir(&inverse, &actual).unwrap();
        let round_trip_error = restored
            .iter()
            .copied()
            .zip(input.iter().copied())
            .map(|(actual, expected)| dd_error(actual, expected))
            .fold(0.0, f64::max);
        assert!(
            round_trip_error < 2.0e-20,
            "recursive-convolution DD Bluestein round-trip error {round_trip_error:e}"
        );
    }

    #[test]
    fn one_dim_dd_c2c_spatial_zero_padding_covers_stockham_direct_and_fft_rader() {
        #[derive(Clone, Copy)]
        enum ExpectedKind {
            Stockham,
            DirectRader,
            FftRader,
        }

        let cases = [
            (16usize, ExpectedKind::Stockham),
            (47usize, ExpectedKind::DirectRader),
            (83usize, ExpectedKind::DirectRader),
            (257usize, ExpectedKind::FftRader),
        ];
        let batch_count = 5usize;
        let grouped_batch = 3usize;
        for precision in [Precision::DoubleDouble, Precision::DoubleDoubleF64Storage] {
            for (length, expected_kind) in cases {
                let left = length / 4;
                let right = length / 2;
                let base = FftConfig::new(vec![length])
                    .with_batch_count(batch_count)
                    .with_precision(precision)
                    .with_tuning(crate::PlannerTuning::portable())
                    .with_grouped_batch(0, grouped_batch)
                    .unwrap();
                let padded_config = base.clone().with_zero_padding(0, left, right).unwrap();
                let padded = DoubleDoubleOneDimIr::build(
                    &FftPlan::build(padded_config).unwrap(),
                    Direction::Forward,
                )
                .unwrap();
                match (&padded, expected_kind) {
                    (DoubleDoubleOneDimIr::Stockham(ir), ExpectedKind::Stockham) => {
                        assert!(ir.zero_pad_pass.is_some());
                    }
                    (DoubleDoubleOneDimIr::DirectRader(ir), ExpectedKind::DirectRader) => {
                        assert!(ir.zero_pad_pass.is_some());
                    }
                    (DoubleDoubleOneDimIr::FftRader(ir), ExpectedKind::FftRader) => {
                        assert!(ir.zero_pad_pass.is_some());
                        assert!(ir.forward_fft.zero_pad_pass().is_none());
                        assert!(ir.inverse_fft.zero_pad_pass().is_none());
                    }
                    _ => panic!("unexpected DD zero-padded algorithm for N={length}"),
                }
                let baseline = DoubleDoubleOneDimIr::build(
                    &FftPlan::build(base.clone()).unwrap(),
                    Direction::Forward,
                )
                .unwrap();
                let baseline_program = crate::ProgramIr::double_double_one_dim(&baseline).unwrap();
                let program = crate::ProgramIr::double_double_one_dim(&padded).unwrap();
                assert_eq!(program.passes.len(), baseline_program.passes.len() + 1);
                assert!(
                    program.passes[0]
                        .name
                        .contains("zero_pad_storage_forward_input")
                );
                let shaders = crate::backend::vulkan::VulkanGlslBackend
                    .lower_double_double_one_dim_program(&padded)
                    .unwrap();
                assert_eq!(shaders.len(), program.passes.len());
                assert!(shaders[0].glsl.contains("VKFFT_ZERO_PAD_LEFT"));
                for shader in shaders {
                    assert_eq!(shader.compile_spirv().unwrap().words[0], 0x0723_0203);
                }

                match precision {
                    Precision::DoubleDouble => {
                        let input = (0..length * batch_count)
                            .map(|index| {
                                let x = index as f64;
                                ComplexDoubleDouble::new(
                                    DoubleDouble::from_parts(
                                        (0.071 * x).sin() + 0.0007 * x,
                                        (index + 1) as f64 * 8.0e-32,
                                    ),
                                    DoubleDouble::from_parts(
                                        (0.039 * x).cos() - 0.0003 * x,
                                        -(index as f64 + 1.0) * 5.0e-32,
                                    ),
                                )
                            })
                            .collect::<Vec<_>>();
                        let mut manual = input.clone();
                        for batch in 0..batch_count {
                            let base = batch * length;
                            manual[base + left..base + right].fill(ComplexDoubleDouble::default());
                        }
                        let expected =
                            execute_double_double_one_dim_ir(&baseline, &manual).unwrap();
                        let actual = execute_double_double_one_dim_ir(&padded, &input).unwrap();
                        assert_eq!(actual, expected);

                        let inverse_base = base.clone().with_inverse_normalization(true);
                        let inverse = DoubleDoubleOneDimIr::build(
                            &FftPlan::build(
                                inverse_base
                                    .clone()
                                    .with_zero_padding(0, left, right)
                                    .unwrap(),
                            )
                            .unwrap(),
                            Direction::Inverse,
                        )
                        .unwrap();
                        let baseline_inverse = DoubleDoubleOneDimIr::build(
                            &FftPlan::build(inverse_base).unwrap(),
                            Direction::Inverse,
                        )
                        .unwrap();
                        let mut inverse_expected =
                            execute_double_double_one_dim_ir(&baseline_inverse, &actual).unwrap();
                        for batch in 0..batch_count {
                            let base = batch * length;
                            inverse_expected[base + left..base + right]
                                .fill(ComplexDoubleDouble::default());
                        }
                        let inverse_actual =
                            execute_double_double_one_dim_ir(&inverse, &actual).unwrap();
                        assert_eq!(inverse_actual, inverse_expected);
                    }
                    Precision::DoubleDoubleF64Storage => {
                        let input = (0..length * batch_count)
                            .map(|index| {
                                let x = index as f64;
                                Complex64::new(
                                    (0.071 * x).sin() + 0.0007 * x,
                                    (0.039 * x).cos() - 0.0003 * x,
                                )
                            })
                            .collect::<Vec<_>>();
                        let mut manual = input.clone();
                        for batch in 0..batch_count {
                            let base = batch * length;
                            manual[base + left..base + right].fill(Complex64::default());
                        }
                        let expected =
                            execute_double_double_one_dim_ir_f64_storage(&baseline, &manual)
                                .unwrap();
                        let actual =
                            execute_double_double_one_dim_ir_f64_storage(&padded, &input).unwrap();
                        assert_eq!(actual, expected);
                        assert!(program.resources.iter().any(|resource| {
                            resource.kind == crate::ProgramResourceKind::Scratch
                                && resource.scalar == ScalarType::F64
                                && resource.name.contains("zero_padded_input")
                        }));
                    }
                    _ => unreachable!(),
                }
            }
        }
    }

    #[test]
    fn direct_rader_p47_matches_independent_dd_dft_and_round_trips() {
        let length = 47usize;
        let batch_count = 2usize;
        let forward_plan = FftPlan::build(
            FftConfig::new(vec![length])
                .with_batch_count(batch_count)
                .with_precision(Precision::DoubleDouble),
        )
        .unwrap();
        let inverse_plan = FftPlan::build(
            FftConfig::new(vec![length])
                .with_batch_count(batch_count)
                .with_precision(Precision::DoubleDouble)
                .with_inverse_normalization(true),
        )
        .unwrap();
        let forward = DoubleDoubleOneDimIr::build(&forward_plan, Direction::Forward).unwrap();
        let inverse = DoubleDoubleOneDimIr::build(&inverse_plan, Direction::Inverse).unwrap();
        let DoubleDoubleOneDimIr::DirectRader(forward_rader) = &forward else {
            panic!("p47 DD plan must preserve direct Rader selection");
        };
        assert_eq!(forward_rader.prime, length);
        assert_eq!(
            forward_rader.table.twiddles_by_generator_power.len(),
            length - 1
        );

        let input = (0..length * batch_count)
            .map(|index| {
                let x = index as f64;
                ComplexDoubleDouble::new(
                    DoubleDouble::from_parts(
                        (0.031 * x).sin() + 0.0002 * x,
                        (index + 1) as f64 * 1.0e-31,
                    ),
                    DoubleDouble::from_parts(
                        (0.017 * x).cos() - 0.0001 * x,
                        -(index as f64 + 1.0) * 5.0e-32,
                    ),
                )
            })
            .collect::<Vec<_>>();
        let spectrum = execute_double_double_one_dim_ir(&forward, &input).unwrap();
        for batch in 0..batch_count {
            let start = batch * length;
            let expected = dft(&input[start..start + length], Direction::Forward, false).unwrap();
            let error = spectrum[start..start + length]
                .iter()
                .copied()
                .zip(expected)
                .map(|(actual, expected)| dd_error(actual, expected))
                .fold(0.0, f64::max);
            assert!(error < 8.0e-27, "double-double p47 Rader error {error:e}");
        }
        let restored = execute_double_double_one_dim_ir(&inverse, &spectrum).unwrap();
        let error = restored
            .iter()
            .copied()
            .zip(input.iter().copied())
            .map(|(actual, expected)| dd_error(actual, expected))
            .fold(0.0, f64::max);
        assert!(
            error < 2.0e-25,
            "double-double p47 Rader round-trip error {error:e}"
        );
    }

    #[test]
    fn fft_rader_p4159_crosses_old_4096_lowering_gate() {
        let length = 4159usize;
        let convolution_len = length - 1;
        let batch_count = 2usize;
        let grouped_batch = 2usize;
        let left = 1024usize;
        let right = 1536usize;
        for precision in [Precision::DoubleDouble, Precision::DoubleDoubleF64Storage] {
            let config = FftConfig::new(vec![length])
                .with_batch_count(batch_count)
                .with_precision(precision)
                .with_tuning(crate::PlannerTuning::portable())
                .with_grouped_batch(0, grouped_batch)
                .unwrap()
                .with_zero_padding(0, left, right)
                .unwrap();
            let plan = FftPlan::build(config).unwrap();
            let ir = DoubleDoubleOneDimIr::build(&plan, Direction::Forward).unwrap();
            let DoubleDoubleOneDimIr::FftRader(rader) = &ir else {
                panic!("p4159 DD plan must remain standalone FFT-Rader");
            };
            assert_eq!(rader.prime, length);
            assert_eq!(rader.convolution_len, convolution_len);
            assert_eq!(rader.forward_fft.sequence_len(), convolution_len);
            assert_eq!(rader.inverse_fft.sequence_len(), convolution_len);
            assert!(rader.forward_fft.sequence_len() > 4096);
            assert!(rader.forward_fft.zero_pad_pass().is_none());
            assert!(rader.inverse_fft.zero_pad_pass().is_none());
            let zero_pad = rader
                .zero_pad_pass
                .as_ref()
                .expect("p4159 should keep one external zero-pad boundary");
            assert_eq!(zero_pad.range, ZeroPaddingRange::new(left, right));
            assert_eq!(zero_pad.grouped_batch, grouped_batch);

            let program = crate::ProgramIr::double_double_one_dim(&ir).unwrap();
            assert!(
                program.passes[0]
                    .name
                    .contains("zero_pad_storage_forward_input")
            );
            let shaders = crate::backend::vulkan::VulkanGlslBackend
                .lower_double_double_one_dim_program(&ir)
                .unwrap();
            assert_eq!(shaders.len(), program.passes.len());
            assert!(shaders[0].glsl.contains("VKFFT_ZERO_PAD_LEFT"));
            assert!(shaders.iter().skip(1).any(|shader| {
                shader.sequence_len == convolution_len && shader.glsl.contains("Stockham stage")
            }));
            for shader in shaders {
                assert_eq!(shader.compile_spirv().unwrap().words[0], 0x0723_0203);
            }
            if precision == Precision::DoubleDoubleF64Storage {
                assert!(program.resources.iter().any(|resource| {
                    resource.kind == crate::ProgramResourceKind::Scratch
                        && resource.scalar == ScalarType::F64
                        && resource.name.contains("zero_padded_input")
                }));
            }
        }
    }

    #[test]
    fn device_aware_large_shared_stockham_bypasses_old_4096_recursive_gate() {
        let mut device = DeviceProfile::generic(crate::Backend::Vulkan, crate::GpuVendor::Nvidia);
        device.shared_memory_bytes = 164 * 1024;
        device.shared_memory_pow2_bytes = 164 * 1024;
        device.max_threads_per_block = 1024;
        device.max_workgroup_size = [1024, 1024, 64];
        device.supports_f64 = true;

        for (length, expected_threads) in [
            (4_800usize, 960usize),
            (5_000usize, 625usize),
            (5_120usize, 640usize),
        ] {
            let config = FftConfig::new(vec![length]).with_precision(Precision::DoubleDouble);
            let portable_plan = FftPlan::build(config.clone()).unwrap();
            let portable = DoubleDoubleOneDimIr::build(&portable_plan, Direction::Forward).unwrap();
            assert!(matches!(portable, DoubleDoubleOneDimIr::Recursive(_)));

            let device_plan = FftPlan::build_for_device(config, device).unwrap();
            let device_ir =
                DoubleDoubleOneDimIr::build_for_device(&device_plan, Direction::Forward, device)
                    .unwrap();
            let DoubleDoubleOneDimIr::Stockham(stockham) = &device_ir else {
                panic!("N{length} should stay a single-upload DD Stockham program on 164KiB");
            };
            let block = stockham
                .axis_batch_block
                .expect("large single-upload DD Stockham must retain its physical block");
            assert_eq!(block.threads_per_transform, expected_threads);
            assert_eq!(block.grouped_batch, 1);
            assert_eq!(
                [block.local_size_x, block.local_size_y],
                [expected_threads, 1]
            );
            let program = crate::ProgramIr::double_double_one_dim(&device_ir).unwrap();
            assert!(program.passes.len() >= stockham.stages.len() + 2);
            assert!(
                program
                    .passes
                    .iter()
                    .all(|pass| !pass.name.contains("pack_right"))
            );
            assert!(
                program
                    .passes
                    .iter()
                    .all(|pass| !pass.name.contains("twiddle_transpose"))
            );

            let batched_config = FftConfig::new(vec![length])
                .with_batch_count(32)
                .with_precision(Precision::DoubleDouble);
            let batched_plan = FftPlan::build_for_device(batched_config, device).unwrap();
            let batched_ir =
                DoubleDoubleOneDimIr::build_for_device(&batched_plan, Direction::Forward, device)
                    .unwrap();
            let DoubleDoubleOneDimIr::Stockham(batched_stockham) = &batched_ir else {
                panic!("N{length}/batch32 should remain single-upload DD Stockham on 164KiB");
            };
            let batched_block = batched_stockham
                .axis_batch_block
                .expect("large batch32 DD Stockham must retain its physical block");
            assert_eq!(batched_stockham.batch_count, 32);
            assert_eq!(batched_stockham.batch_group_count(), 32);
            assert_eq!(batched_block.threads_per_transform, expected_threads);
            assert_eq!(batched_block.grouped_batch, 1);
            assert_eq!(
                [batched_block.local_size_x, batched_block.local_size_y],
                [expected_threads, 1]
            );
            let batched_program = crate::ProgramIr::double_double_one_dim(&batched_ir).unwrap();
            assert!(
                batched_program
                    .passes
                    .iter()
                    .all(|pass| pass.dispatch.x == 32)
            );

            let nd_config = FftConfig::new(vec![length, 8])
                .with_batch_count(5)
                .with_precision(Precision::DoubleDouble);
            let nd = crate::TransformIr::build(nd_config, Direction::Forward, device).unwrap();
            let crate::TransformIr::ComplexNdDoubleDouble(nd) = nd else {
                panic!("DD [N{length},8] should remain ND C2C");
            };
            let outer = nd.axes.iter().find(|axis| axis.axis == 0).unwrap();
            let DoubleDoubleOneDimIr::Stockham(outer_stockham) = &outer.transform else {
                panic!("DD [N{length},8] outer axis should remain one-upload Stockham on 164KiB");
            };
            assert_eq!(outer_stockham.batch_count, 40);
            assert_eq!(outer_stockham.batch_group_count(), 40);
            let outer_block = outer_stockham
                .axis_batch_block
                .expect("large higher-axis DD Stockham should retain an automatic physical block");
            assert_eq!(outer_block.threads_per_transform, expected_threads);
            assert_eq!(outer_block.grouped_batch, 1);
            assert!(outer_block.transforms_on_x);
            assert!(!outer_block.axis_swapped);
            assert_eq!(
                [outer_block.local_size_x, outer_block.local_size_y],
                [1, expected_threads]
            );
            let outer_program = crate::ProgramIr::double_double_one_dim(&outer.transform).unwrap();
            assert!(
                outer_program
                    .passes
                    .iter()
                    .all(|pass| pass.dispatch.x == 40)
            );
        }
    }

    #[test]
    fn fft_rader_p67_device_scores_its_contiguous_convolution_child() {
        let length = 67usize;
        let convolution_len = length - 1;
        let device = DeviceProfile {
            shared_memory_bytes: 48 * 1024,
            shared_memory_pow2_bytes: 32 * 1024,
            max_threads_per_block: 1024,
            max_workgroup_size: [1024, 1024, 64],
            supports_f64: true,
            ..DeviceProfile::generic(crate::Backend::Vulkan, crate::GpuVendor::Nvidia)
        };
        let tuning = crate::PlannerTuning::portable();
        let config = FftConfig::new(vec![length])
            .with_precision(Precision::DoubleDouble)
            .with_tuning(tuning);
        let outer_plan = FftPlan::build(config.clone()).unwrap();
        let portable = DoubleDoubleFftRaderIr::build(&outer_plan, Direction::Forward).unwrap();
        assert!(matches!(
            portable.forward_fft,
            DoubleDoubleBluesteinConvolutionIr::Stockham(_)
        ));

        let device_forward =
            DoubleDoubleFftRaderIr::build_for_device(&outer_plan, Direction::Forward, device)
                .unwrap();
        assert_eq!(device_forward.convolution_len, convolution_len);
        let DoubleDoubleBluesteinConvolutionIr::Bluestein(child) = &device_forward.forward_fft
        else {
            panic!("fixed DD device scoring should reclassify the 66-point p67 convolution child");
        };
        assert_eq!(child.logical_len, convolution_len);
        assert!(child.zero_padding.is_none());
        assert!(matches!(
            device_forward.inverse_fft,
            DoubleDoubleBluesteinConvolutionIr::Bluestein(_)
        ));

        let high_level =
            crate::TransformIr::build(config.clone(), Direction::Forward, device).unwrap();
        let crate::TransformIr::Complex1dDoubleDouble(DoubleDoubleOneDimIr::FftRader(rader)) =
            &high_level
        else {
            panic!("public device-aware p67 route should remain an FFT-Rader caller");
        };
        assert!(matches!(
            rader.forward_fft,
            DoubleDoubleBluesteinConvolutionIr::Bluestein(_)
        ));

        let program = crate::ProgramIr::double_double_fft_rader(&device_forward).unwrap();
        let shaders = crate::backend::vulkan::VulkanGlslBackend
            .lower_double_double_fft_rader(&device_forward)
            .unwrap();
        assert_eq!(shaders.len(), program.passes.len());
        assert!(
            shaders
                .iter()
                .any(|shader| shader.glsl.contains("DD Bluestein"))
        );
        assert!(shaders.iter().all(|shader| shader.compile_spirv().is_ok()));

        let input = (0..length)
            .map(|index| {
                let x = index as f64;
                ComplexDoubleDouble::new(
                    DoubleDouble::from_parts((0.019 * x).sin() + 0.0002 * x, 3.0e-31),
                    DoubleDouble::from_parts((0.013 * x).cos() - 0.0001 * x, -2.0e-31),
                )
            })
            .collect::<Vec<_>>();
        let actual = execute_double_double_fft_rader_ir(&device_forward, &input).unwrap();
        let expected = dft(&input, Direction::Forward, false).unwrap();
        let error = actual
            .iter()
            .copied()
            .zip(expected)
            .map(|(actual, expected)| dd_error(actual, expected))
            .fold(0.0, f64::max);
        assert!(error < 2.0e-23, "device-scored DD p67 error {error:e}");

        let inverse_plan = FftPlan::build(config.clone().with_inverse_normalization(true)).unwrap();
        let device_inverse =
            DoubleDoubleFftRaderIr::build_for_device(&inverse_plan, Direction::Inverse, device)
                .unwrap();
        let restored = execute_double_double_fft_rader_ir(&device_inverse, &actual).unwrap();
        let round_trip_error = restored
            .iter()
            .copied()
            .zip(input)
            .map(|(actual, expected)| dd_error(actual, expected))
            .fold(0.0, f64::max);
        assert!(
            round_trip_error < 2.0e-21,
            "device-scored DD p67 round-trip error {round_trip_error:e}"
        );
    }

    #[test]
    fn fft_rader_p257_matches_independent_dd_dft_and_round_trips() {
        let length = 257usize;
        let forward_plan =
            FftPlan::build(FftConfig::new(vec![length]).with_precision(Precision::DoubleDouble))
                .unwrap();
        let inverse_plan = FftPlan::build(
            FftConfig::new(vec![length])
                .with_precision(Precision::DoubleDouble)
                .with_inverse_normalization(true),
        )
        .unwrap();
        let forward = DoubleDoubleOneDimIr::build(&forward_plan, Direction::Forward).unwrap();
        let inverse = DoubleDoubleOneDimIr::build(&inverse_plan, Direction::Inverse).unwrap();
        let DoubleDoubleOneDimIr::FftRader(forward_rader) = &forward else {
            panic!("p257 DD plan must preserve FFT-convolution Rader selection");
        };
        assert_eq!(forward_rader.prime, length);
        assert_eq!(forward_rader.convolution_len, 256);
        assert_eq!(forward_rader.forward_fft.sequence_len(), 256);
        assert_eq!(forward_rader.inverse_fft.sequence_len(), 256);
        assert_eq!(forward_rader.kernel_spectrum.len(), 256);
        assert!(
            forward_rader
                .kernel_spectrum
                .iter()
                .any(|value| { value.re.lo != 0.0 || value.im.lo != 0.0 })
        );

        let input = (0..length)
            .map(|index| {
                let x = index as f64;
                ComplexDoubleDouble::new(
                    DoubleDouble::from_parts(
                        (0.013 * x).sin() + 0.00007 * x,
                        (index + 1) as f64 * 2.0e-31,
                    ),
                    DoubleDouble::from_parts(
                        (0.009 * x).cos() - 0.00003 * x,
                        -(index as f64 + 1.0) * 1.0e-31,
                    ),
                )
            })
            .collect::<Vec<_>>();
        let spectrum = execute_double_double_one_dim_ir(&forward, &input).unwrap();
        let expected = dft(&input, Direction::Forward, false).unwrap();
        let error = spectrum
            .iter()
            .copied()
            .zip(expected)
            .map(|(actual, expected)| dd_error(actual, expected))
            .fold(0.0, f64::max);
        assert!(
            error < 2.0e-24,
            "double-double p257 FFT-Rader error {error:e}"
        );

        let restored = execute_double_double_one_dim_ir(&inverse, &spectrum).unwrap();
        let round_trip_error = restored
            .iter()
            .copied()
            .zip(input.iter().copied())
            .map(|(actual, expected)| dd_error(actual, expected))
            .fold(0.0, f64::max);
        assert!(
            round_trip_error < 2.0e-22,
            "double-double p257 FFT-Rader round-trip error {round_trip_error:e}"
        );
    }

    #[test]
    fn fft_rader_p107_uses_recursive_convolution_child_when_opted_in() {
        fn contains_p53_fft_rader(
            node: &crate::double_double_recursive_ir::DoubleDoubleRecursiveFftNodeIr,
        ) -> bool {
            use crate::double_double_recursive_ir::DoubleDoubleRecursiveFftNodeIr;
            match node {
                DoubleDoubleRecursiveFftNodeIr::FftRader(rader) => rader.prime == 53,
                DoubleDoubleRecursiveFftNodeIr::CooleyTukey(node) => {
                    contains_p53_fft_rader(&node.left) || contains_p53_fft_rader(&node.right)
                }
                _ => false,
            }
        }

        let length = 107usize;
        let batch_count = 3usize;
        let grouped_batch = 2usize;
        let left = 17usize;
        let right = 29usize;
        let tuning = crate::PlannerTuning::portable().with_recursive_fft_rader(true);
        let base = FftConfig::new(vec![length])
            .with_batch_count(batch_count)
            .with_precision(Precision::DoubleDouble)
            .with_tuning(tuning)
            .with_grouped_batch(0, grouped_batch)
            .unwrap();
        let forward_plan =
            FftPlan::build(base.clone().with_zero_padding(0, left, right).unwrap()).unwrap();
        let forward = DoubleDoubleOneDimIr::build(&forward_plan, Direction::Forward).unwrap();
        let DoubleDoubleOneDimIr::FftRader(rader) = &forward else {
            panic!("p107 recursive-Rader opt-in should keep an FFT-Rader caller");
        };
        assert_eq!(rader.prime, length);
        assert_eq!(rader.convolution_len, length - 1);
        let DoubleDoubleBluesteinConvolutionIr::Recursive(child) = &rader.forward_fft else {
            panic!("p107 DD FFT-Rader should use a recursive 106-point convolution child");
        };
        assert_eq!(child.logical_len, 106);
        assert_eq!(child.grouped_batch, grouped_batch);
        assert!(child.zero_pad_pass.is_none());
        assert!(contains_p53_fft_rader(&child.root));
        assert!(matches!(
            rader.inverse_fft,
            DoubleDoubleBluesteinConvolutionIr::Recursive(_)
        ));

        let program = crate::ProgramIr::double_double_one_dim(&forward).unwrap();
        assert!(
            program.passes[0]
                .name
                .contains("zero_pad_storage_forward_input")
        );
        assert!(
            program
                .passes
                .iter()
                .any(|pass| pass.name.contains("fft_rader_forward"))
        );
        let shaders = crate::backend::vulkan::VulkanGlslBackend
            .lower_double_double_one_dim_program(&forward)
            .unwrap();
        assert_eq!(shaders.len(), program.passes.len());
        for shader in shaders {
            assert_eq!(shader.compile_spirv().unwrap().words[0], 0x0723_0203);
        }

        let input = (0..length * batch_count)
            .map(|index| {
                let x = index as f64;
                ComplexDoubleDouble::new(
                    DoubleDouble::from_parts(
                        (0.021 * x).sin() + 0.00011 * x,
                        (index + 1) as f64 * 4.0e-32,
                    ),
                    DoubleDouble::from_parts(
                        (0.017 * x).cos() - 0.00007 * x,
                        -(index as f64 + 1.0) * 3.0e-32,
                    ),
                )
            })
            .collect::<Vec<_>>();
        let mut manual = input.clone();
        for batch in 0..batch_count {
            let base = batch * length;
            manual[base + left..base + right].fill(ComplexDoubleDouble::default());
        }
        let actual = execute_double_double_one_dim_ir(&forward, &input).unwrap();
        for batch in 0..batch_count {
            let base = batch * length;
            let expected = dft(&manual[base..base + length], Direction::Forward, false).unwrap();
            let error = actual[base..base + length]
                .iter()
                .copied()
                .zip(expected)
                .map(|(actual, expected)| dd_error(actual, expected))
                .fold(0.0, f64::max);
            assert!(error < 5.0e-24, "nested DD p107 FFT-Rader error {error:e}");
        }

        let inverse_plan = FftPlan::build(
            base.with_inverse_normalization(true)
                .with_zero_padding(0, left, right)
                .unwrap(),
        )
        .unwrap();
        let inverse = DoubleDoubleOneDimIr::build(&inverse_plan, Direction::Inverse).unwrap();
        let restored = execute_double_double_one_dim_ir(&inverse, &actual).unwrap();
        let round_trip_error = restored
            .iter()
            .copied()
            .zip(manual.iter().copied())
            .map(|(actual, expected)| dd_error(actual, expected))
            .fold(0.0, f64::max);
        assert!(
            round_trip_error < 2.0e-22,
            "nested DD p107 FFT-Rader round-trip error {round_trip_error:e}"
        );
    }

    #[test]
    fn prime_convolutions_above_256_match_independent_dd_dft() {
        for (length, force_bluestein, expected_convolution) in
            [(401usize, false, 400usize), (257usize, true, 525usize)]
        {
            let mut tuning = crate::PlannerTuning::portable();
            if force_bluestein {
                tuning.max_rader_fft_prime = 100;
            }
            let forward_plan = FftPlan::build(
                FftConfig::new(vec![length])
                    .with_precision(Precision::DoubleDouble)
                    .with_tuning(tuning),
            )
            .unwrap();
            let inverse_plan = FftPlan::build(
                FftConfig::new(vec![length])
                    .with_precision(Precision::DoubleDouble)
                    .with_inverse_normalization(true)
                    .with_tuning(tuning),
            )
            .unwrap();
            let forward = DoubleDoubleOneDimIr::build(&forward_plan, Direction::Forward).unwrap();
            let inverse = DoubleDoubleOneDimIr::build(&inverse_plan, Direction::Inverse).unwrap();
            match &forward {
                DoubleDoubleOneDimIr::FftRader(rader) if !force_bluestein => {
                    assert_eq!(rader.convolution_len, expected_convolution);
                }
                DoubleDoubleOneDimIr::Bluestein(bluestein) if force_bluestein => {
                    assert_eq!(bluestein.convolution_len, expected_convolution);
                }
                other => panic!("unexpected DD prime algorithm above old 256 limit: {other:?}"),
            }
            let input = (0..length)
                .map(|index| {
                    let x = index as f64;
                    ComplexDoubleDouble::new(
                        DoubleDouble::from_parts(
                            (0.013 * x).sin() + 0.00007 * x,
                            (index + 1) as f64 * 2.0e-31,
                        ),
                        DoubleDouble::from_parts(
                            (0.009 * x).cos() - 0.00003 * x,
                            -(index as f64 + 1.0) * 1.0e-31,
                        ),
                    )
                })
                .collect::<Vec<_>>();
            let spectrum = execute_double_double_one_dim_ir(&forward, &input).unwrap();
            let expected = dft(&input, Direction::Forward, false).unwrap();
            let error = spectrum
                .iter()
                .copied()
                .zip(expected)
                .map(|(actual, expected)| dd_error(actual, expected))
                .fold(0.0, f64::max);
            assert!(
                error < 5.0e-23,
                "DD prime length {length} forward error above old convolution limit: {error:e}"
            );
            let restored = execute_double_double_one_dim_ir(&inverse, &spectrum).unwrap();
            let round_trip_error = restored
                .iter()
                .copied()
                .zip(input.iter().copied())
                .map(|(actual, expected)| dd_error(actual, expected))
                .fold(0.0, f64::max);
            assert!(
                round_trip_error < 5.0e-21,
                "DD prime length {length} round-trip error above old convolution limit: {round_trip_error:e}"
            );
        }
    }

    #[test]
    fn direct_rader_f64_storage_and_fft_rader_boundary_are_explicit() {
        let length = 47usize;
        let plan = FftPlan::build(
            FftConfig::new(vec![length]).with_precision(Precision::DoubleDoubleF64Storage),
        )
        .unwrap();
        let ir = DoubleDoubleOneDimIr::build(&plan, Direction::Forward).unwrap();
        let DoubleDoubleOneDimIr::DirectRader(rader) = &ir else {
            panic!("p47 DD/F64 plan must select direct Rader");
        };
        assert_eq!(rader.external_storage, PrecisionStorage::F64);
        let input = (0..length)
            .map(|index| {
                let x = index as f64;
                Complex64::new((0.043 * x).sin() + 0.0003 * x, (0.027 * x).cos())
            })
            .collect::<Vec<_>>();
        let actual = execute_double_double_one_dim_ir_f64_storage(&ir, &input).unwrap();
        let expected = dft(
            &input
                .iter()
                .copied()
                .map(ComplexDoubleDouble::from_complex64)
                .collect::<Vec<_>>(),
            Direction::Forward,
            false,
        )
        .unwrap()
        .into_iter()
        .map(ComplexDoubleDouble::to_complex64)
        .collect::<Vec<_>>();
        let error = actual
            .iter()
            .zip(&expected)
            .map(|(actual, expected)| (*actual - *expected).norm_sqr().sqrt())
            .fold(0.0, f64::max);
        assert!(
            error < 5.0e-13 * length as f64,
            "DD/F64 p47 Rader error {error:e}"
        );

        let fft_rader =
            FftPlan::build(FftConfig::new(vec![257]).with_precision(Precision::DoubleDouble))
                .unwrap();
        let fft_ir = DoubleDoubleOneDimIr::build(&fft_rader, Direction::Forward).unwrap();
        assert!(matches!(fft_ir, DoubleDoubleOneDimIr::FftRader(_)));
    }

    #[test]
    fn mixed_radix_double_double_stockham_matches_independent_dd_dft() {
        let length = 15usize;
        let batch_count = 2usize;
        let plan = FftPlan::build(
            FftConfig::new(vec![length])
                .with_batch_count(batch_count)
                .with_precision(Precision::DoubleDouble),
        )
        .unwrap();
        let ir = DoubleDoubleStockhamIr::build(&plan, Direction::Forward).unwrap();
        assert_eq!(ir.external_storage, PrecisionStorage::DoubleDouble);
        assert_eq!(
            ir.stages.iter().map(|stage| stage.radix).product::<usize>(),
            length
        );
        assert_eq!(ir.twiddles.stages.len(), ir.stages.len());

        let input = (0..length * batch_count)
            .map(|index| {
                let x = index as f64;
                ComplexDoubleDouble::new(
                    DoubleDouble::from_parts((0.071 * x).sin(), (index as f64 + 1.0) * 1.0e-30),
                    DoubleDouble::from_parts((0.043 * x).cos(), -(index as f64 + 1.0) * 5.0e-31),
                )
            })
            .collect::<Vec<_>>();
        let actual = execute_double_double_stockham_ir(&ir, &input).unwrap();
        for batch in 0..batch_count {
            let start = batch * length;
            let expected = dft(&input[start..start + length], Direction::Forward, false).unwrap();
            let error = actual[start..start + length]
                .iter()
                .copied()
                .zip(expected)
                .map(|(actual, expected)| dd_error(actual, expected))
                .fold(0.0, f64::max);
            assert!(error < 2.0e-28, "double-double Stockham error {error:e}");
        }
    }

    #[test]
    fn normalized_round_trip_retains_residual_lost_by_plain_f64() {
        let length = 8usize;
        let forward_plan =
            FftPlan::build(FftConfig::new(vec![length]).with_precision(Precision::DoubleDouble))
                .unwrap();
        let inverse_plan = FftPlan::build(
            FftConfig::new(vec![length])
                .with_precision(Precision::DoubleDouble)
                .with_inverse_normalization(true),
        )
        .unwrap();
        let forward = DoubleDoubleStockhamIr::build(&forward_plan, Direction::Forward).unwrap();
        let inverse = DoubleDoubleStockhamIr::build(&inverse_plan, Direction::Inverse).unwrap();
        let mut input = vec![ComplexDoubleDouble::default(); length];
        input[0] =
            ComplexDoubleDouble::new(DoubleDouble::from_parts(1.0e16, 1.0), DoubleDouble::ZERO);
        for (index, value) in input.iter_mut().enumerate().skip(1) {
            value.re = DoubleDouble::from_f64(index as f64 * 0.125);
            value.im = DoubleDouble::from_f64(-(index as f64) * 0.0625);
        }
        assert_eq!(input[0].re.to_f64(), 1.0e16);
        assert_ne!(input[0].re.lo, 0.0);

        let spectrum = execute_double_double_stockham_ir(&forward, &input).unwrap();
        let restored = execute_double_double_stockham_ir(&inverse, &spectrum).unwrap();
        let error = restored
            .iter()
            .copied()
            .zip(input.iter().copied())
            .map(|(actual, expected)| dd_error(actual, expected))
            .fold(0.0, f64::max);
        assert!(error < 5.0e-13, "double-double round-trip error {error:e}");
        assert!((restored[0].re.lo - input[0].re.lo).abs() < 5.0e-13);
    }

    #[test]
    fn f64_storage_mode_promotes_and_narrows_only_at_external_boundary() {
        let length = 12usize;
        let plan = FftPlan::build(
            FftConfig::new(vec![length]).with_precision(Precision::DoubleDoubleF64Storage),
        )
        .unwrap();
        let ir = DoubleDoubleStockhamIr::build(&plan, Direction::Forward).unwrap();
        assert_eq!(ir.external_storage, PrecisionStorage::F64);
        let input = (0..length)
            .map(|index| {
                let x = index as f64;
                Complex64::new((0.17 * x).sin() + x * 0.003, (0.11 * x).cos() - x * 0.002)
            })
            .collect::<Vec<_>>();
        let actual = execute_double_double_stockham_ir_f64_storage(&ir, &input).unwrap();
        let promoted = input
            .iter()
            .copied()
            .map(ComplexDoubleDouble::from_complex64)
            .collect::<Vec<_>>();
        let expected = dft(&promoted, Direction::Forward, false)
            .unwrap()
            .into_iter()
            .map(ComplexDoubleDouble::to_complex64)
            .collect::<Vec<_>>();
        let error = actual
            .iter()
            .zip(&expected)
            .map(|(actual, expected)| (*actual - *expected).norm_sqr().sqrt())
            .fold(0.0, f64::max);
        assert!(
            error < 2.0e-14 * length as f64,
            "F64-storage DD error {error:e}"
        );
    }

    #[test]
    fn all_double_double_dct_dst_types_round_trip_with_inverse_normalization() {
        let length = 9usize;
        let batch_count = 7usize;
        let grouped_batch = 3usize;
        let transforms = [
            TransformKind::Dct(DctType::I),
            TransformKind::Dct(DctType::II),
            TransformKind::Dct(DctType::III),
            TransformKind::Dct(DctType::IV),
            TransformKind::Dst(DstType::I),
            TransformKind::Dst(DstType::II),
            TransformKind::Dst(DstType::III),
            TransformKind::Dst(DstType::IV),
        ];
        let input = (0..length * batch_count)
            .map(|index| {
                let x = index as f64;
                DoubleDouble::from_parts(
                    (0.071 * x).sin() + 0.23 * (0.041 * x).cos() + 0.0007 * x,
                    (index as f64 + 1.0) * 1.0e-31,
                )
            })
            .collect::<Vec<_>>();
        for transform in transforms {
            let forward_plan = FftPlan::build(
                FftConfig::new(vec![length])
                    .with_batch_count(batch_count)
                    .with_precision(Precision::DoubleDouble)
                    .with_transform(transform)
                    .with_grouped_batch(0, grouped_batch)
                    .unwrap(),
            )
            .unwrap();
            let inverse_plan = FftPlan::build(
                FftConfig::new(vec![length])
                    .with_batch_count(batch_count)
                    .with_precision(Precision::DoubleDouble)
                    .with_transform(transform)
                    .with_inverse_normalization(true)
                    .with_grouped_batch(0, grouped_batch)
                    .unwrap(),
            )
            .unwrap();
            let forward = DoubleDoubleR2rIr::build(&forward_plan, Direction::Forward).unwrap();
            let inverse = DoubleDoubleR2rIr::build(&inverse_plan, Direction::Inverse).unwrap();
            assert_eq!(forward.grouped_batch, grouped_batch);
            assert_eq!(inverse.grouped_batch, grouped_batch);
            assert_eq!(forward.batch_group_count(), 3);
            assert_eq!(inverse.batch_group_count(), 3);
            assert!(matches!(
                forward.algorithm,
                DoubleDoubleR2rAlgorithm::FftReduction { .. }
            ));
            assert!(matches!(
                inverse.algorithm,
                DoubleDoubleR2rAlgorithm::FftReduction { .. }
            ));
            let DoubleDoubleR2rAlgorithm::FftReduction { fft, .. } = &forward.algorithm else {
                unreachable!();
            };
            assert_eq!(fft.grouped_batch(), grouped_batch);
            let spectrum = execute_double_double_r2r_ir(&forward, &input).unwrap();
            let restored = execute_double_double_r2r_ir(&inverse, &spectrum).unwrap();
            let error = restored
                .iter()
                .zip(&input)
                .map(|(actual, expected)| (*actual - *expected).abs().to_f64().abs())
                .fold(0.0, f64::max);
            assert!(
                error < 2.0e-25,
                "{transform:?} DD R2R round-trip error {error:e}"
            );
        }
    }

    #[test]
    fn double_double_dct_dst_ii_iii_n_point_reductions_match_true_dd_direct_oracle() {
        let length = 9usize;
        let batch_count = 2usize;
        let input = (0..length * batch_count)
            .map(|index| {
                let x = index as f64;
                DoubleDouble::from_parts(
                    (0.079 * x).sin() + 0.13 * (0.031 * x).cos() + 0.0009 * x,
                    (index as f64 + 1.0) * 7.0e-32,
                )
            })
            .collect::<Vec<_>>();

        for (kind, transform) in [
            (
                TransformKind::Dct(DctType::II),
                R2rTransform::Dct(DctType::II),
            ),
            (
                TransformKind::Dct(DctType::III),
                R2rTransform::Dct(DctType::III),
            ),
            (
                TransformKind::Dst(DstType::II),
                R2rTransform::Dst(DstType::II),
            ),
            (
                TransformKind::Dst(DstType::III),
                R2rTransform::Dst(DstType::III),
            ),
        ] {
            let plan = FftPlan::build(
                FftConfig::new(vec![length])
                    .with_batch_count(batch_count)
                    .with_precision(Precision::DoubleDouble)
                    .with_transform(kind),
            )
            .unwrap();
            let ir = DoubleDoubleR2rIr::build(&plan, Direction::Forward).unwrap();
            let DoubleDoubleR2rAlgorithm::FftReduction {
                fft_len,
                fft,
                phases,
            } = &ir.algorithm
            else {
                panic!("{kind:?} must use an FFT reduction");
            };
            assert_eq!(*fft_len, length);
            assert_eq!(fft.sequence_len(), length);
            assert_eq!(phases.len(), length);
            assert_eq!(
                fft.direction(),
                if matches!(
                    kind,
                    TransformKind::Dct(DctType::III) | TransformKind::Dst(DstType::III)
                ) {
                    Direction::Inverse
                } else {
                    Direction::Forward
                }
            );

            let actual = execute_double_double_r2r_ir(&ir, &input).unwrap();
            let mut expected = vec![DoubleDouble::ZERO; actual.len()];
            for batch in 0..batch_count {
                let base = batch * length;
                for k in 0..length {
                    let mut sum = DoubleDouble::ZERO;
                    for j in 0..length {
                        let jf = DoubleDouble::from_f64(j as f64);
                        let kf = DoubleDouble::from_f64(k as f64);
                        let nf = DoubleDouble::from_f64(length as f64);
                        let half = DoubleDouble::from_f64(0.5);
                        let two = DoubleDouble::from_f64(2.0);
                        let coefficient = match transform {
                            R2rTransform::Dct(DctType::II) => {
                                let (_, cosine) =
                                    (DoubleDouble::PI * (jf + half) * kf / nf).sin_cos();
                                two * cosine
                            }
                            R2rTransform::Dct(DctType::III) => {
                                if j == 0 {
                                    DoubleDouble::ONE
                                } else {
                                    let (_, cosine) =
                                        (DoubleDouble::PI * jf * (kf + half) / nf).sin_cos();
                                    two * cosine
                                }
                            }
                            R2rTransform::Dst(DstType::II) => {
                                let (sine, _) = (DoubleDouble::PI
                                    * (jf + half)
                                    * DoubleDouble::from_f64((k + 1) as f64)
                                    / nf)
                                    .sin_cos();
                                two * sine
                            }
                            R2rTransform::Dst(DstType::III) => {
                                if j + 1 == length {
                                    if k.is_multiple_of(2) {
                                        DoubleDouble::ONE
                                    } else {
                                        -DoubleDouble::ONE
                                    }
                                } else {
                                    let (sine, _) = (DoubleDouble::PI
                                        * DoubleDouble::from_f64((j + 1) as f64)
                                        * (kf + half)
                                        / nf)
                                        .sin_cos();
                                    two * sine
                                }
                            }
                            _ => unreachable!(),
                        };
                        sum += input[base + j] * coefficient;
                    }
                    expected[base + k] = sum;
                }
            }
            let error = actual
                .iter()
                .zip(&expected)
                .map(|(actual, expected)| (*actual - *expected).abs().to_f64().abs())
                .fold(0.0, f64::max);
            assert!(
                error < 5.0e-25,
                "{kind:?} N-point direct-oracle error {error:e}"
            );
        }
    }

    #[test]
    fn double_double_dct4_dst4_fft_reductions_match_true_dd_direct_oracle() {
        let length = 9usize;
        let batch_count = 2usize;
        let input = (0..length * batch_count)
            .map(|index| {
                let x = index as f64;
                DoubleDouble::from_parts(
                    (0.083 * x).sin() + 0.17 * (0.029 * x).cos() + 0.0011 * x,
                    (index as f64 + 1.0) * 9.0e-32,
                )
            })
            .collect::<Vec<_>>();

        for (kind, transform) in [
            (
                TransformKind::Dct(DctType::IV),
                R2rTransform::Dct(DctType::IV),
            ),
            (
                TransformKind::Dst(DstType::IV),
                R2rTransform::Dst(DstType::IV),
            ),
        ] {
            let plan = FftPlan::build(
                FftConfig::new(vec![length])
                    .with_batch_count(batch_count)
                    .with_precision(Precision::DoubleDouble)
                    .with_transform(kind),
            )
            .unwrap();
            let ir = DoubleDoubleR2rIr::build(&plan, Direction::Forward).unwrap();
            assert!(matches!(
                ir.algorithm,
                DoubleDoubleR2rAlgorithm::FftReduction { .. }
            ));
            let actual = execute_double_double_r2r_ir(&ir, &input).unwrap();
            let mut expected = vec![DoubleDouble::ZERO; actual.len()];
            for batch in 0..batch_count {
                let base = batch * length;
                for k in 0..length {
                    let mut sum = DoubleDouble::ZERO;
                    for j in 0..length {
                        sum += input[base + j]
                            * double_double_r2r_coefficient(transform, length, j, k).unwrap();
                    }
                    expected[base + k] = sum;
                }
            }
            let error = actual
                .iter()
                .zip(&expected)
                .map(|(actual, expected)| (*actual - *expected).abs().to_f64().abs())
                .fold(0.0, f64::max);
            assert!(error < 5.0e-25, "{kind:?} direct-oracle error {error:e}");
        }
    }

    #[test]
    fn double_double_dct_ii_fft_reduction_preserves_sub_f64_low_word() {
        let length = 8usize;
        let plan = FftPlan::build(
            FftConfig::new(vec![length])
                .with_precision(Precision::DoubleDouble)
                .with_transform(TransformKind::Dct(DctType::II)),
        )
        .unwrap();
        let ir = DoubleDoubleR2rIr::build(&plan, Direction::Forward).unwrap();
        let DoubleDoubleR2rAlgorithm::FftReduction { fft_len, .. } = &ir.algorithm else {
            panic!("DD DCT-II must use the N-point FFT reduction");
        };
        assert_eq!(*fft_len, length);
        let value = DoubleDouble::from_parts(1.0e16, 1.0);
        assert_eq!(value.to_f64(), 1.0e16);
        assert_ne!(value.lo, 0.0);
        let input = vec![value; length];
        let output = execute_double_double_r2r_ir(&ir, &input).unwrap();
        let expected_dc = value * DoubleDouble::from_f64((2 * length) as f64);
        let dc_error = (output[0] - expected_dc).abs();
        assert!(
            dc_error.to_f64().abs() < 1.0e-20,
            "DD DCT-II DC lost its low word: actual={:?} expected={:?} error={:?}",
            output[0],
            expected_dc,
            dc_error
        );
        assert_ne!(
            output[0].lo, 0.0,
            "DD DCT-II must retain a non-zero residual word"
        );
    }

    #[test]
    fn double_double_r2r_spatial_zero_padding_matches_manual_boundary() {
        let length = 9usize;
        let batch_count = 5usize;
        let left = 3usize;
        let right = 6usize;
        let input = (0..length * batch_count)
            .map(|index| {
                let x = index as f64;
                DoubleDouble::from_parts(
                    (0.091 * x).sin() + 0.17 * (0.043 * x).cos() + 0.0007 * x,
                    (index + 1) as f64 * 7.0e-32,
                )
            })
            .collect::<Vec<_>>();
        for transform in [
            TransformKind::Dct(DctType::II),
            TransformKind::Dst(DstType::IV),
        ] {
            let base_config = FftConfig::new(vec![length])
                .with_batch_count(batch_count)
                .with_precision(Precision::DoubleDouble)
                .with_transform(transform)
                .with_grouped_batch(0, 3)
                .unwrap();
            let padded_config = base_config
                .clone()
                .with_zero_padding(0, left, right)
                .unwrap();
            let forward = DoubleDoubleR2rIr::build(
                &FftPlan::build(padded_config).unwrap(),
                Direction::Forward,
            )
            .unwrap();
            assert_eq!(
                forward.zero_padding,
                Some(ZeroPaddingRange::new(left, right))
            );
            assert_eq!(forward.grouped_batch, 3);

            let mut manual = input.clone();
            for batch in 0..batch_count {
                let base = batch * length;
                manual[base + left..base + right].fill(DoubleDouble::ZERO);
            }
            let baseline = DoubleDoubleR2rIr::build(
                &FftPlan::build(base_config.clone()).unwrap(),
                Direction::Forward,
            )
            .unwrap();
            let expected = execute_double_double_r2r_ir(&baseline, &manual).unwrap();
            let actual = execute_double_double_r2r_ir(&forward, &input).unwrap();
            let forward_error = actual
                .iter()
                .copied()
                .zip(expected.iter().copied())
                .map(|(actual, expected)| {
                    let delta = (actual - expected).abs();
                    delta.hi.abs() + delta.lo.abs()
                })
                .fold(0.0, f64::max);
            assert!(
                forward_error < 5.0e-22,
                "DD R2R padded {transform:?} forward error {forward_error:e}"
            );

            let inverse_base = base_config.clone().with_inverse_normalization(true);
            let inverse = DoubleDoubleR2rIr::build(
                &FftPlan::build(
                    inverse_base
                        .clone()
                        .with_zero_padding(0, left, right)
                        .unwrap(),
                )
                .unwrap(),
                Direction::Inverse,
            )
            .unwrap();
            let baseline_inverse = DoubleDoubleR2rIr::build(
                &FftPlan::build(inverse_base).unwrap(),
                Direction::Inverse,
            )
            .unwrap();
            let mut inverse_expected =
                execute_double_double_r2r_ir(&baseline_inverse, &actual).unwrap();
            for batch in 0..batch_count {
                let base = batch * length;
                inverse_expected[base + left..base + right].fill(DoubleDouble::ZERO);
            }
            let inverse_actual = execute_double_double_r2r_ir(&inverse, &actual).unwrap();
            let inverse_error = inverse_actual
                .iter()
                .copied()
                .zip(inverse_expected.iter().copied())
                .map(|(actual, expected)| {
                    let delta = (actual - expected).abs();
                    delta.hi.abs() + delta.lo.abs()
                })
                .fold(0.0, f64::max);
            assert!(
                inverse_error < 5.0e-22,
                "DD R2R padded {transform:?} inverse error {inverse_error:e}"
            );
            for batch in 0..batch_count {
                let base = batch * length;
                assert!(
                    inverse_actual[base + left..base + right]
                        .iter()
                        .all(|value| *value == DoubleDouble::ZERO)
                );
            }

            let f64_base = FftConfig::new(vec![length])
                .with_batch_count(batch_count)
                .with_precision(Precision::DoubleDoubleF64Storage)
                .with_transform(transform);
            let f64_forward = DoubleDoubleR2rIr::build(
                &FftPlan::build(f64_base.clone().with_zero_padding(0, left, right).unwrap())
                    .unwrap(),
                Direction::Forward,
            )
            .unwrap();
            let f64_baseline =
                DoubleDoubleR2rIr::build(&FftPlan::build(f64_base).unwrap(), Direction::Forward)
                    .unwrap();
            let f64_input = input.iter().map(|value| value.to_f64()).collect::<Vec<_>>();
            let mut f64_manual = f64_input.clone();
            for batch in 0..batch_count {
                let base = batch * length;
                f64_manual[base + left..base + right].fill(0.0);
            }
            let f64_expected =
                execute_double_double_r2r_ir_f64_storage(&f64_baseline, &f64_manual).unwrap();
            let f64_actual =
                execute_double_double_r2r_ir_f64_storage(&f64_forward, &f64_input).unwrap();
            assert_eq!(f64_actual, f64_expected);
        }
    }

    #[test]
    fn double_double_r2r_f64_storage_narrows_only_at_external_boundary() {
        let length = 9usize;
        let plan = FftPlan::build(
            FftConfig::new(vec![length])
                .with_precision(Precision::DoubleDoubleF64Storage)
                .with_transform(TransformKind::Dct(DctType::II)),
        )
        .unwrap();
        let ir = DoubleDoubleR2rIr::build(&plan, Direction::Forward).unwrap();
        assert_eq!(ir.external_storage, PrecisionStorage::F64);
        let input = (0..length)
            .map(|index| {
                let x = index as f64;
                (0.11 * x).sin() + 0.03 * x
            })
            .collect::<Vec<_>>();
        let actual = execute_double_double_r2r_ir_f64_storage(&ir, &input).unwrap();
        let promoted = input
            .iter()
            .copied()
            .map(DoubleDouble::from_f64)
            .collect::<Vec<_>>();
        let expected = execute_double_double_r2r_ir(&ir, &promoted)
            .unwrap()
            .into_iter()
            .map(DoubleDouble::to_f64)
            .collect::<Vec<_>>();
        assert_eq!(actual, expected);
    }

    #[test]
    fn nd_r2r_3x4_fft_families_round_trip_in_true_dd() {
        let dimensions = vec![3usize, 4usize];
        let batch_count = 2usize;
        let tensor_len = dimensions.iter().product::<usize>();
        let input = (0..tensor_len * batch_count)
            .map(|index| {
                let x = index as f64;
                DoubleDouble::from_parts(
                    (0.09 * x).sin() + 0.21 * (0.037 * x).cos() + 0.0003 * x,
                    (index + 1) as f64 * 3.0e-32,
                )
            })
            .collect::<Vec<_>>();
        for transform in [
            TransformKind::Dct(DctType::I),
            TransformKind::Dst(DstType::I),
            TransformKind::Dct(DctType::II),
            TransformKind::Dst(DstType::III),
            TransformKind::Dct(DctType::IV),
            TransformKind::Dst(DstType::IV),
        ] {
            let forward_plan = FftPlan::build(
                FftConfig::new(dimensions.clone())
                    .with_batch_count(batch_count)
                    .with_precision(Precision::DoubleDouble)
                    .with_transform(transform),
            )
            .unwrap();
            let inverse_plan = FftPlan::build(
                FftConfig::new(dimensions.clone())
                    .with_batch_count(batch_count)
                    .with_precision(Precision::DoubleDouble)
                    .with_transform(transform)
                    .with_inverse_normalization(true),
            )
            .unwrap();
            let forward = DoubleDoubleNdR2rIr::build(&forward_plan, Direction::Forward).unwrap();
            let inverse = DoubleDoubleNdR2rIr::build(&inverse_plan, Direction::Inverse).unwrap();
            assert_eq!(forward.axes.len(), 2);
            assert_eq!(forward.axes[0].axis, 1);
            assert_eq!(forward.axes[1].axis, 0);
            for axis in &forward.axes {
                let even_type_iv = matches!(
                    transform,
                    TransformKind::Dct(DctType::IV) | TransformKind::Dst(DstType::IV)
                ) && axis.axis_len.is_multiple_of(2);
                if even_type_iv {
                    assert!(matches!(
                        &axis.transform.algorithm,
                        DoubleDoubleR2rAlgorithm::EvenTypeIvHalfSize { fft_len, .. }
                            if *fft_len == axis.axis_len / 2
                    ));
                } else {
                    assert!(matches!(
                        axis.transform.algorithm,
                        DoubleDoubleR2rAlgorithm::FftReduction { .. }
                    ));
                }
            }
            let spectrum = execute_double_double_nd_r2r_ir(&forward, &input).unwrap();
            let restored = execute_double_double_nd_r2r_ir(&inverse, &spectrum).unwrap();
            let error = restored
                .iter()
                .copied()
                .zip(input.iter().copied())
                .map(|(actual, expected)| {
                    let delta = (actual - expected).abs();
                    delta.hi.abs() + delta.lo.abs()
                })
                .fold(0.0, f64::max);
            assert!(
                error < 5.0e-24,
                "DD ND R2R {transform:?} round-trip error {error:e}"
            );
        }

        let f64_plan = FftPlan::build(
            FftConfig::new(dimensions)
                .with_batch_count(batch_count)
                .with_precision(Precision::DoubleDoubleF64Storage)
                .with_transform(TransformKind::Dct(DctType::II)),
        )
        .unwrap();
        let f64_ir = DoubleDoubleNdR2rIr::build(&f64_plan, Direction::Forward).unwrap();
        let f64_input = input
            .iter()
            .copied()
            .map(DoubleDouble::to_f64)
            .collect::<Vec<_>>();
        let f64_actual = execute_double_double_nd_r2r_ir_f64_storage(&f64_ir, &f64_input).unwrap();
        let promoted = f64_input
            .iter()
            .copied()
            .map(DoubleDouble::from_f64)
            .collect::<Vec<_>>();
        let expected = execute_double_double_nd_r2r_compute(&f64_ir, &promoted)
            .unwrap()
            .into_iter()
            .map(DoubleDouble::to_f64)
            .collect::<Vec<_>>();
        assert_eq!(f64_actual, expected);
    }

    #[test]
    fn real_and_dct2_p2053_propagate_recursive_bluestein_children() {
        let length = 2_053usize;
        let impulse_index = 137usize;
        let mut tuning = crate::PlannerTuning::portable();
        tuning.max_rader_fft_prime = 100;

        let real_plan = |transform, inverse| {
            FftPlan::build(
                FftConfig::new(vec![length])
                    .with_precision(Precision::DoubleDouble)
                    .with_transform(transform)
                    .with_inverse_normalization(inverse)
                    .with_tuning(tuning),
            )
            .unwrap()
        };
        let r2c =
            DoubleDoubleRealFftIr::build(&real_plan(TransformKind::RealToComplex, false)).unwrap();
        let c2r =
            DoubleDoubleRealFftIr::build(&real_plan(TransformKind::ComplexToReal, true)).unwrap();
        let real_bluestein = one_dim_bluestein_child(&r2c.transform)
            .expect("forced DD real p2053 child must contain Bluestein");
        assert_eq!(real_bluestein.logical_len, length);
        assert_eq!(real_bluestein.convolution_len, 4_116);
        assert!(matches!(
            real_bluestein.forward_fft,
            DoubleDoubleBluesteinConvolutionIr::Recursive(_)
        ));
        assert!(matches!(
            real_bluestein.inverse_fft,
            DoubleDoubleBluesteinConvolutionIr::Recursive(_)
        ));
        let real_program = crate::ProgramIr::double_double_real(&r2c).unwrap();
        let real_shaders = crate::backend::vulkan::VulkanGlslBackend
            .lower_double_double_real(&r2c)
            .unwrap();
        assert_eq!(real_shaders.len(), real_program.passes.len());
        assert!(real_program.passes.len() > 10);
        for shader in &real_shaders {
            assert_eq!(shader.compile_spirv().unwrap().words[0], 0x0723_0203);
        }

        let impulse = DoubleDouble::from_parts(1.25, 3.0e-31);
        let mut real_input = vec![DoubleDouble::ZERO; length];
        real_input[impulse_index] = impulse;
        let spectrum = execute_double_double_r2c_ir(&r2c, &real_input).unwrap();
        let real_forward_error = spectrum
            .iter()
            .copied()
            .enumerate()
            .map(|(k, actual)| {
                let expected = unit_root(impulse_index * k, length, Direction::Forward)
                    .unwrap()
                    .scale_dd(impulse);
                dd_error(actual, expected)
            })
            .fold(0.0, f64::max);
        assert!(
            real_forward_error < 5.0e-21,
            "DD real p2053 recursive-Bluestein impulse error {real_forward_error:e}"
        );
        let real_restored = execute_double_double_c2r_ir(&c2r, &spectrum).unwrap();
        let real_round_trip_error = real_restored
            .iter()
            .copied()
            .zip(real_input.iter().copied())
            .map(|(actual, expected)| {
                let delta = (actual - expected).abs();
                delta.hi.abs() + delta.lo.abs()
            })
            .fold(0.0, f64::max);
        assert!(
            real_round_trip_error < 5.0e-19,
            "DD real p2053 recursive-Bluestein round-trip error {real_round_trip_error:e}"
        );

        let dct_plan = |inverse| {
            FftPlan::build(
                FftConfig::new(vec![length])
                    .with_precision(Precision::DoubleDouble)
                    .with_transform(TransformKind::Dct(DctType::II))
                    .with_inverse_normalization(inverse)
                    .with_tuning(tuning),
            )
            .unwrap()
        };
        let dct_forward = DoubleDoubleR2rIr::build(&dct_plan(false), Direction::Forward).unwrap();
        let dct_inverse = DoubleDoubleR2rIr::build(&dct_plan(true), Direction::Inverse).unwrap();
        let DoubleDoubleR2rAlgorithm::FftReduction { fft_len, fft, .. } = &dct_forward.algorithm
        else {
            panic!("DD DCT-II p2053 must use FFT reduction");
        };
        assert_eq!(*fft_len, 2_053);
        let dct_bluestein = one_dim_bluestein_child(fft)
            .expect("DD DCT-II p2053 N-point child must contain Bluestein");
        assert_eq!(dct_bluestein.logical_len, 2_053);
        assert_eq!(dct_bluestein.convolution_len, 4_116);
        assert!(matches!(
            dct_bluestein.forward_fft,
            DoubleDoubleBluesteinConvolutionIr::Recursive(_)
        ));
        assert!(matches!(
            dct_bluestein.inverse_fft,
            DoubleDoubleBluesteinConvolutionIr::Recursive(_)
        ));
        let dct_program = crate::ProgramIr::double_double_r2r(&dct_forward).unwrap();
        let dct_shaders = crate::backend::vulkan::VulkanGlslBackend
            .lower_double_double_r2r(&dct_forward)
            .unwrap();
        assert_eq!(dct_shaders.len(), dct_program.passes.len());
        assert!(dct_program.passes.len() > 10);
        for shader in &dct_shaders {
            assert_eq!(shader.compile_spirv().unwrap().words[0], 0x0723_0203);
        }

        let mut dct_input = vec![DoubleDouble::ZERO; length];
        dct_input[impulse_index] = impulse;
        let dct_spectrum = execute_double_double_r2r_ir(&dct_forward, &dct_input).unwrap();
        let two = DoubleDouble::from_f64(2.0);
        let half = DoubleDouble::from_f64(0.5);
        for k in [0usize, 1, 17, length / 2, length - 1] {
            let angle = DoubleDouble::PI
                * (DoubleDouble::from_f64(impulse_index as f64) + half)
                * DoubleDouble::from_f64(k as f64)
                / DoubleDouble::from_f64(length as f64);
            let (_, cosine) = angle.sin_cos();
            let expected = impulse * two * cosine;
            let delta = (dct_spectrum[k] - expected).abs();
            let error = delta.hi.abs() + delta.lo.abs();
            assert!(
                error < 5.0e-21,
                "DD DCT-II p2053 recursive-Bluestein bin {k} error {error:e}"
            );
        }
        let dct_restored = execute_double_double_r2r_ir(&dct_inverse, &dct_spectrum).unwrap();
        let dct_round_trip_error = dct_restored
            .iter()
            .copied()
            .zip(dct_input.iter().copied())
            .map(|(actual, expected)| {
                let delta = (actual - expected).abs();
                delta.hi.abs() + delta.lo.abs()
            })
            .fold(0.0, f64::max);
        assert!(
            dct_round_trip_error < 5.0e-18,
            "DD DCT-II/III p2053 recursive-Bluestein round-trip error {dct_round_trip_error:e}"
        );
    }

    #[test]
    fn nd_real_and_dct2_p2053_propagate_recursive_bluestein_children() {
        let dimensions = [2usize, 2_053usize];
        let tensor_len = dimensions.iter().product::<usize>();
        let compact_last = dimensions[1] / 2 + 1;
        let compact_tensor_len = dimensions[0] * compact_last;
        let impulse_n0 = 1usize;
        let impulse_n1 = 137usize;
        let impulse = DoubleDouble::from_parts(1.25, 3.0e-31);
        let mut tuning = crate::PlannerTuning::portable();
        tuning.max_rader_fft_prime = 100;

        let real_plan = |transform, inverse| {
            FftPlan::build(
                FftConfig::new(dimensions.to_vec())
                    .with_precision(Precision::DoubleDouble)
                    .with_transform(transform)
                    .with_inverse_normalization(inverse)
                    .with_tuning(tuning),
            )
            .unwrap()
        };
        let r2c = DoubleDoubleNdRealFftIr::build(&real_plan(TransformKind::RealToComplex, false))
            .unwrap();
        let c2r =
            DoubleDoubleNdRealFftIr::build(&real_plan(TransformKind::ComplexToReal, true)).unwrap();
        let real_bluestein = one_dim_bluestein_child(&r2c.real_axis.transform)
            .expect("forced DD ND-real [2,2053] fastest axis must contain Bluestein");
        assert_eq!(real_bluestein.logical_len, 2_053);
        assert_eq!(real_bluestein.convolution_len, 4_116);
        assert!(matches!(
            real_bluestein.forward_fft,
            DoubleDoubleBluesteinConvolutionIr::Recursive(_)
        ));
        assert!(matches!(
            real_bluestein.inverse_fft,
            DoubleDoubleBluesteinConvolutionIr::Recursive(_)
        ));
        let real_program = crate::ProgramIr::double_double_nd_real(&r2c).unwrap();
        let real_shaders = crate::backend::vulkan::VulkanGlslBackend
            .lower_double_double_nd_real(&r2c)
            .unwrap();
        assert_eq!(real_shaders.len(), real_program.passes.len());
        assert!(real_program.passes.len() > 20);
        for shader in &real_shaders {
            assert_eq!(shader.compile_spirv().unwrap().words[0], 0x0723_0203);
        }

        let mut real_input = vec![DoubleDouble::ZERO; tensor_len];
        real_input[impulse_n0 * dimensions[1] + impulse_n1] = impulse;
        let real_spectrum = execute_double_double_nd_r2c_ir(&r2c, &real_input).unwrap();
        assert_eq!(real_spectrum.len(), compact_tensor_len);
        let mut real_forward_error = 0.0f64;
        for k0 in 0..dimensions[0] {
            let root0 = unit_root(impulse_n0 * k0, dimensions[0], Direction::Forward).unwrap();
            for k1 in 0..compact_last {
                let root1 = unit_root(impulse_n1 * k1, dimensions[1], Direction::Forward).unwrap();
                let expected = (root0 * root1).scale_dd(impulse);
                real_forward_error = real_forward_error
                    .max(dd_error(real_spectrum[k0 * compact_last + k1], expected));
            }
        }
        assert!(
            real_forward_error < 5.0e-20,
            "DD ND-real [2,2053] recursive-Bluestein impulse error {real_forward_error:e}"
        );
        let real_restored = execute_double_double_nd_c2r_ir(&c2r, &real_spectrum).unwrap();
        let real_round_trip_error = real_restored
            .iter()
            .copied()
            .zip(real_input.iter().copied())
            .map(|(actual, expected)| {
                let delta = (actual - expected).abs();
                delta.hi.abs() + delta.lo.abs()
            })
            .fold(0.0, f64::max);
        assert!(
            real_round_trip_error < 5.0e-18,
            "DD ND-real [2,2053] recursive-Bluestein round-trip error {real_round_trip_error:e}"
        );

        let dct_plan = |inverse| {
            FftPlan::build(
                FftConfig::new(dimensions.to_vec())
                    .with_precision(Precision::DoubleDouble)
                    .with_transform(TransformKind::Dct(DctType::II))
                    .with_inverse_normalization(inverse)
                    .with_tuning(tuning),
            )
            .unwrap()
        };
        let dct_forward = DoubleDoubleNdR2rIr::build(&dct_plan(false), Direction::Forward).unwrap();
        let dct_inverse = DoubleDoubleNdR2rIr::build(&dct_plan(true), Direction::Inverse).unwrap();
        assert_eq!(dct_forward.axes[0].axis, 1);
        let DoubleDoubleR2rAlgorithm::FftReduction { fft, fft_len, .. } =
            &dct_forward.axes[0].transform.algorithm
        else {
            panic!("DD ND DCT-II [2,2053] fastest axis must use FFT reduction");
        };
        assert_eq!(*fft_len, 2_053);
        let dct_bluestein = one_dim_bluestein_child(fft)
            .expect("DD ND DCT-II [2,2053] fastest N-point child must contain Bluestein");
        assert_eq!(dct_bluestein.logical_len, 2_053);
        assert_eq!(dct_bluestein.convolution_len, 4_116);
        assert!(matches!(
            dct_bluestein.forward_fft,
            DoubleDoubleBluesteinConvolutionIr::Recursive(_)
        ));
        assert!(matches!(
            dct_bluestein.inverse_fft,
            DoubleDoubleBluesteinConvolutionIr::Recursive(_)
        ));
        let dct_program = crate::ProgramIr::double_double_nd_r2r(&dct_forward).unwrap();
        let dct_shaders = crate::backend::vulkan::VulkanGlslBackend
            .lower_double_double_nd_r2r(&dct_forward)
            .unwrap();
        assert_eq!(dct_shaders.len(), dct_program.passes.len());
        assert!(dct_program.passes.len() > 20);
        for shader in &dct_shaders {
            assert_eq!(shader.compile_spirv().unwrap().words[0], 0x0723_0203);
        }

        let mut dct_input = vec![DoubleDouble::ZERO; tensor_len];
        dct_input[impulse_n0 * dimensions[1] + impulse_n1] = impulse;
        let dct_spectrum = execute_double_double_nd_r2r_ir(&dct_forward, &dct_input).unwrap();
        let coefficient = |n: usize, j: usize, k: usize| {
            let angle = DoubleDouble::PI
                * (DoubleDouble::from_f64(j as f64) + DoubleDouble::from_f64(0.5))
                * DoubleDouble::from_f64(k as f64)
                / DoubleDouble::from_f64(n as f64);
            let (_, cosine) = angle.sin_cos();
            DoubleDouble::from_f64(2.0) * cosine
        };
        for k0 in 0..dimensions[0] {
            for k1 in [0usize, 1, 17, dimensions[1] / 2, dimensions[1] - 1] {
                let expected = impulse
                    * coefficient(dimensions[0], impulse_n0, k0)
                    * coefficient(dimensions[1], impulse_n1, k1);
                let delta = (dct_spectrum[k0 * dimensions[1] + k1] - expected).abs();
                let error = delta.hi.abs() + delta.lo.abs();
                assert!(
                    error < 5.0e-19,
                    "DD ND DCT-II [2,2053] bin ({k0},{k1}) error {error:e}"
                );
            }
        }
        let dct_restored = execute_double_double_nd_r2r_ir(&dct_inverse, &dct_spectrum).unwrap();
        let dct_round_trip_error = dct_restored
            .iter()
            .copied()
            .zip(dct_input.iter().copied())
            .map(|(actual, expected)| {
                let delta = (actual - expected).abs();
                delta.hi.abs() + delta.lo.abs()
            })
            .fold(0.0, f64::max);
        assert!(
            dct_round_trip_error < 5.0e-17,
            "DD ND DCT-II/III [2,2053] recursive-Bluestein round-trip error {dct_round_trip_error:e}"
        );
    }

    #[test]
    fn nd_r2r_spatial_zero_padding_matches_manual_dd_boundary() {
        let dimensions = vec![3usize, 4usize];
        let batch_count = 2usize;
        let tensor_len = dimensions.iter().product::<usize>();
        let input = (0..tensor_len * batch_count)
            .map(|index| {
                let x = index as f64;
                DoubleDouble::from_parts(
                    (0.083 * x).sin() + 0.19 * (0.041 * x).cos() + 0.0004 * x,
                    (index + 1) as f64 * 4.0e-32,
                )
            })
            .collect::<Vec<_>>();
        let mut manual = input.clone();
        for batch in 0..batch_count {
            let base = batch * tensor_len;
            for n0 in 0..dimensions[0] {
                for n1 in 0..dimensions[1] {
                    if (1..2).contains(&n0) || (1..3).contains(&n1) {
                        manual[base + n0 * dimensions[1] + n1] = DoubleDouble::ZERO;
                    }
                }
            }
        }
        for transform in [
            TransformKind::Dct(DctType::II),
            TransformKind::Dst(DstType::III),
        ] {
            let padded = |precision, inverse| {
                FftConfig::new(dimensions.clone())
                    .with_batch_count(batch_count)
                    .with_precision(precision)
                    .with_transform(transform)
                    .with_inverse_normalization(inverse)
                    .with_zero_padding(0, 1, 2)
                    .unwrap()
                    .with_zero_padding(1, 1, 3)
                    .unwrap()
            };
            let forward = DoubleDoubleNdR2rIr::build(
                &FftPlan::build(padded(Precision::DoubleDouble, false)).unwrap(),
                Direction::Forward,
            )
            .unwrap();
            let inverse = DoubleDoubleNdR2rIr::build(
                &FftPlan::build(padded(Precision::DoubleDouble, true)).unwrap(),
                Direction::Inverse,
            )
            .unwrap();
            assert!(forward.has_spatial_zero_padding());
            assert!(forward.contains_spatial_zero_linear_index(1));
            assert!(forward.contains_spatial_zero_linear_index(dimensions[1]));
            assert!(!forward.contains_spatial_zero_linear_index(0));
            assert!(
                forward.axes.iter().all(|axis| {
                    axis.transform.external_storage == PrecisionStorage::DoubleDouble
                })
            );
            let baseline = DoubleDoubleNdR2rIr::build(
                &FftPlan::build(
                    FftConfig::new(dimensions.clone())
                        .with_batch_count(batch_count)
                        .with_precision(Precision::DoubleDouble)
                        .with_transform(transform),
                )
                .unwrap(),
                Direction::Forward,
            )
            .unwrap();
            let expected = execute_double_double_nd_r2r_ir(&baseline, &manual).unwrap();
            let actual = execute_double_double_nd_r2r_ir(&forward, &input).unwrap();
            let forward_error = actual
                .iter()
                .copied()
                .zip(expected.iter().copied())
                .map(|(actual, expected)| {
                    let delta = (actual - expected).abs();
                    delta.hi.abs() + delta.lo.abs()
                })
                .fold(0.0, f64::max);
            assert!(
                forward_error < 5.0e-24,
                "DD ND R2R padded {transform:?} forward error {forward_error:e}"
            );
            let restored = execute_double_double_nd_r2r_ir(&inverse, &actual).unwrap();
            let round_trip_error = restored
                .iter()
                .copied()
                .zip(manual.iter().copied())
                .map(|(actual, expected)| {
                    let delta = (actual - expected).abs();
                    delta.hi.abs() + delta.lo.abs()
                })
                .fold(0.0, f64::max);
            assert!(
                round_trip_error < 5.0e-23,
                "DD ND R2R padded {transform:?} round-trip error {round_trip_error:e}"
            );

            let f64_forward = DoubleDoubleNdR2rIr::build(
                &FftPlan::build(padded(Precision::DoubleDoubleF64Storage, false)).unwrap(),
                Direction::Forward,
            )
            .unwrap();
            let f64_inverse = DoubleDoubleNdR2rIr::build(
                &FftPlan::build(padded(Precision::DoubleDoubleF64Storage, true)).unwrap(),
                Direction::Inverse,
            )
            .unwrap();
            let f64_input = input
                .iter()
                .copied()
                .map(DoubleDouble::to_f64)
                .collect::<Vec<_>>();
            let f64_manual = manual
                .iter()
                .copied()
                .map(DoubleDouble::to_f64)
                .collect::<Vec<_>>();
            let f64_spectrum =
                execute_double_double_nd_r2r_ir_f64_storage(&f64_forward, &f64_input).unwrap();
            let f64_restored =
                execute_double_double_nd_r2r_ir_f64_storage(&f64_inverse, &f64_spectrum).unwrap();
            let f64_error = f64_restored
                .iter()
                .zip(&f64_manual)
                .map(|(actual, expected)| (actual - expected).abs())
                .fold(0.0, f64::max);
            assert!(
                f64_error < 5.0e-13,
                "DD/F64 ND R2R padded {transform:?} round-trip error {f64_error:e}"
            );
        }
    }
}
