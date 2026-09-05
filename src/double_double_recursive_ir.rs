//! Recursive mixed-radix composition for double-double one-dimensional FFTs.
//!
//! Leaf transforms reuse the dedicated DD Stockham/Rader/Bluestein IRs. Composite
//! axes are expressed as explicit Cooley-Tukey pack, true-DD twiddle/transpose,
//! child transforms, and natural-order scatter passes. Caller-visible F64 storage
//! is converted only at the root boundary; every recursive scratch edge remains
//! full double-double.

use crate::complex::Complex64;
use crate::config::{
    DeviceProfile, Direction, FftConfig, PlannerTuning, Precision, PrecisionStorage, TransformKind,
    upstream_coalesced_memory_bytes_for_precision, upstream_effective_rader_tuning,
};
use crate::double_double::ComplexDoubleDouble;
use crate::double_double_ir::{
    DoubleDoubleBluesteinConvolutionIr, DoubleDoubleBluesteinIr, DoubleDoubleDirectRaderIr,
    DoubleDoubleFftRaderIr, DoubleDoubleStockhamIr, execute_double_double_bluestein_ir,
    execute_double_double_direct_rader_ir, execute_double_double_fft_rader_ir,
    execute_double_double_stockham_ir,
};
use crate::error::{Result, VkFftError};
use crate::kernel_ir::{
    DispatchGeometry, FourStepMapping, ScalarType, StockhamIoMapping, ThreeUploadFourStepMapping,
    WorkgroupSize,
};
use crate::planner::{
    AxisAlgorithm, C2cDeviceAxisClass, FftPlan, RaderMode, RaderPrimePlan, RadixPlan,
    merged_radix_schedule, primitive_root,
};
use crate::scheduler::{
    Axis0RaderSplitState, DoubleDoubleStockhamUploadSchedule, FourStepAxisBlockRequest,
    OtherAxisFourStepPhysicalContext, RaderUploadSchedule, StockhamAxisBlockSchedule,
    StockhamUploadAxisContext, plan_gpu_axis0_bluestein_four_step_default_block_from_shape,
    plan_gpu_axis0_bluestein_rader_four_step_default_block_from_split_state,
    plan_gpu_axis0_four_step_default_block_from_shape,
    plan_gpu_axis0_four_step_grouped_block_from_shape_for_precision,
    plan_gpu_axis0_rader_four_step_default_block_from_split_state,
    plan_gpu_axis0_rader_four_step_grouped_block_from_split_state_for_precision,
    plan_gpu_double_double_axis0_composite_direct_rader_split_state_for_prime_multiplicities_with_max_batch_coalesced,
    plan_gpu_double_double_axis0_composite_direct_rader_threads_for_prime_multiplicities,
    plan_gpu_double_double_axis0_composite_direct_rader_threads_for_prime_multiplicities_with_max_batch_coalesced,
    plan_gpu_double_double_axis0_fft_rader_threads_for_forced_upload_with_tuning,
    plan_gpu_double_double_axis0_fft_rader_threads_for_prime_multiplicities_with_tuning,
    plan_gpu_double_double_axis0_mixed_direct_multi_fft_rader_split_state_for_forced_upload_with_tuning,
    plan_gpu_double_double_axis0_mixed_direct_multi_fft_rader_threads_for_forced_upload_with_tuning,
    plan_gpu_double_double_axis0_mixed_direct_multi_fft_rader_threads_with_tuning,
    plan_gpu_double_double_other_axis_composite_direct_rader_threads_for_prime_multiplicities,
    plan_gpu_double_double_other_axis_four_step_upload_block,
    plan_gpu_double_double_quad_registers,
    plan_gpu_double_double_stockham_uploads_for_batches_with_axis_context,
    plan_gpu_other_axis_composite_rader_batch_block,
    plan_gpu_other_axis_four_step_block_from_shape_for_precision,
    plan_gpu_rader_upload_split_with_axis_context,
};
use crate::zero_pad_ir::ZeroPadPassIr;

const DD_RECURSIVE_STOCKHAM_LEAF_LIMIT: usize = 4096;
const DD_TWO_UPLOAD_FOUR_STEP_LEAF_LIMIT: usize = DD_RECURSIVE_STOCKHAM_LEAF_LIMIT;
const DD_THREE_UPLOAD_FOUR_STEP_LEAF_LIMIT: usize = DD_RECURSIVE_STOCKHAM_LEAF_LIMIT;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DoubleDoubleCooleyTukeyPassOperation {
    PackRightInput,
    TwiddleTranspose,
    ScatterOutput,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DoubleDoubleCooleyTukeyInputModifier {
    #[default]
    None,
    FourStepRight(FourStepMapping),
    FourStepThreeUpload2(ThreeUploadFourStepMapping),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DoubleDoubleCooleyTukeyOutputModifier {
    #[default]
    None,
    FourStepRight(FourStepMapping),
    FourStepLeft(FourStepMapping),
    FourStepThreeUpload2(ThreeUploadFourStepMapping),
    FourStepThreeUpload1(ThreeUploadFourStepMapping),
    FourStepThreeUpload0(ThreeUploadFourStepMapping),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DoubleDoubleCooleyTukeyPassIr {
    pub name: String,
    pub direction: Direction,
    pub logical_len: usize,
    pub left_len: usize,
    pub right_len: usize,
    pub batch_count: usize,
    pub grouped_batch: usize,
    pub input_storage: PrecisionStorage,
    pub output_storage: PrecisionStorage,
    pub operation: DoubleDoubleCooleyTukeyPassOperation,
    pub workgroup_size: WorkgroupSize,
    pub axis_batch_block: Option<StockhamAxisBlockSchedule>,
    pub(crate) device_shared_memory_bytes: Option<usize>,
    pub input_modifier: DoubleDoubleCooleyTukeyInputModifier,
    pub output_modifier: DoubleDoubleCooleyTukeyOutputModifier,

    pub dispatch: DispatchGeometry,
}

impl DoubleDoubleCooleyTukeyPassIr {
    fn new(
        name: String,
        direction: Direction,
        logical_len: usize,
        left_len: usize,
        right_len: usize,
        batch_count: usize,
        grouped_batch: usize,
        input_storage: PrecisionStorage,
        output_storage: PrecisionStorage,
        operation: DoubleDoubleCooleyTukeyPassOperation,
    ) -> Result<Self> {
        let dispatch = DispatchGeometry {
            x: u32::try_from(batch_count.div_ceil(grouped_batch)).map_err(|_| {
                VkFftError::ValueOutOfRange {
                    field: "double-double recursive Cooley-Tukey dispatch count",
                }
            })?,
            y: 1,
            z: 1,
        };
        let pass = Self {
            name,
            direction,
            logical_len,
            left_len,
            right_len,
            batch_count,
            grouped_batch,
            input_storage,
            output_storage,
            operation,
            input_modifier: DoubleDoubleCooleyTukeyInputModifier::None,
            output_modifier: DoubleDoubleCooleyTukeyOutputModifier::None,
            workgroup_size: WorkgroupSize { x: 1, y: 1, z: 1 },
            axis_batch_block: None,
            device_shared_memory_bytes: None,
            dispatch,
        };
        pass.validate()?;
        Ok(pass)
    }

    pub fn with_device_physical_block(self, device: DeviceProfile) -> Result<Self> {
        self.validate()?;
        // A user groupedBatch may legitimately exceed this recursive shadow pass's
        // current batch count (for example parent batch2/group3 before a forced-Rader
        // upload is rematerialized with many component transforms). It still controls
        // logical dispatch ownership, but it cannot be installed as a physical block
        // whose grouped dimension exceeds the transforms owned by this pass. Fail soft
        // here and let the executed upload/component scorer materialize its own block.
        if self.grouped_batch > self.batch_count
            || self.grouped_batch > device.max_workgroup_size[1]
            || self.grouped_batch > device.max_threads_per_block
        {
            return Ok(self);
        }
        let max_x_by_threads = device.max_threads_per_block / self.grouped_batch;
        let slot_threads = self
            .logical_len
            .min(128)
            .min(device.max_workgroup_size[0])
            .min(max_x_by_threads);
        if slot_threads == 0 {
            return Ok(self);
        }
        let grouped_batch = self.grouped_batch;
        self.with_axis_batch_block(
            StockhamAxisBlockSchedule {
                threads_per_transform: slot_threads,
                grouped_batch,
                transforms_on_x: false,
                axis_swapped: false,
                local_size_x: slot_threads,
                local_size_y: grouped_batch,
            },
            device,
        )
    }

    pub fn with_axis_batch_block(
        mut self,
        block: StockhamAxisBlockSchedule,
        device: DeviceProfile,
    ) -> Result<Self> {
        block.validate(self.batch_count, device)?;
        self.workgroup_size = WorkgroupSize {
            x: u32::try_from(block.local_size_x).map_err(|_| VkFftError::ValueOutOfRange {
                field: "double-double recursive Cooley-Tukey local_size_x",
            })?,
            y: u32::try_from(block.local_size_y).map_err(|_| VkFftError::ValueOutOfRange {
                field: "double-double recursive Cooley-Tukey local_size_y",
            })?,
            z: 1,
        };
        self.dispatch.x =
            u32::try_from(self.batch_count.div_ceil(block.grouped_batch)).map_err(|_| {
                VkFftError::ValueOutOfRange {
                    field: "double-double recursive Cooley-Tukey physical dispatch count",
                }
            })?;
        self.axis_batch_block = Some(block);
        self.device_shared_memory_bytes = Some(device.shared_memory_bytes);
        self.validate()?;
        Ok(self)
    }

    fn with_external_input_storage(mut self, storage: PrecisionStorage) -> Result<Self> {
        self.input_storage = storage;
        self.validate()?;
        Ok(self)
    }

    fn with_external_output_storage(mut self, storage: PrecisionStorage) -> Result<Self> {
        self.output_storage = storage;
        self.validate()?;
        Ok(self)
    }

    fn with_four_step_input_modifier(
        mut self,
        modifier: DoubleDoubleCooleyTukeyInputModifier,
    ) -> Result<Self> {
        if self.operation != DoubleDoubleCooleyTukeyPassOperation::PackRightInput {
            return Err(VkFftError::InvalidKernelIr(
                "double-double Four-step input mapping requires a Cooley pack boundary",
            ));
        }
        self.input_modifier = modifier;
        self.validate()?;
        Ok(self)
    }

    fn with_four_step_output_modifier(
        mut self,
        modifier: DoubleDoubleCooleyTukeyOutputModifier,
    ) -> Result<Self> {
        if self.operation != DoubleDoubleCooleyTukeyPassOperation::ScatterOutput {
            return Err(VkFftError::InvalidKernelIr(
                "double-double Four-step output mapping requires a Cooley scatter boundary",
            ));
        }
        self.output_modifier = modifier;
        self.validate()?;
        Ok(self)
    }

    pub fn validate(&self) -> Result<()> {
        let default_workgroup = WorkgroupSize { x: 1, y: 1, z: 1 };
        let execution_grouped_batch = self
            .axis_batch_block
            .map_or(self.grouped_batch, |block| block.grouped_batch);
        if self.axis_batch_block.is_some() != self.device_shared_memory_bytes.is_some()
            || self.device_shared_memory_bytes == Some(0)
        {
            return Err(VkFftError::InvalidKernelIr(
                "double-double recursive Cooley-Tukey physical block must retain its device shared-memory budget",
            ));
        }
        let physical_workgroup_valid = if let Some(block) = self.axis_batch_block {
            let expected = if block.transforms_on_x {
                [block.grouped_batch, block.threads_per_transform]
            } else {
                [block.threads_per_transform, block.grouped_batch]
            };
            block.grouped_batch > 0
                && block.grouped_batch <= self.batch_count
                && block.threads_per_transform > 0
                && [block.local_size_x, block.local_size_y] == expected
                && usize::try_from(self.workgroup_size.x).ok() == Some(block.local_size_x)
                && usize::try_from(self.workgroup_size.y).ok() == Some(block.local_size_y)
                && self.workgroup_size.z == 1
        } else {
            self.workgroup_size == default_workgroup
                || (self.workgroup_size.x > 0
                    && self.workgroup_size.z == 1
                    && usize::try_from(self.workgroup_size.y).ok() == Some(self.grouped_batch))
        };
        if self.left_len < 2
            || self.right_len < 2
            || self.batch_count == 0
            || self.grouped_batch == 0
            || self.left_len.checked_mul(self.right_len) != Some(self.logical_len)
            || !physical_workgroup_valid
            || self.dispatch.y != 1
            || self.dispatch.z != 1
            || usize::try_from(self.dispatch.x).ok()
                != Some(self.batch_count.div_ceil(execution_grouped_batch))
        {
            return Err(VkFftError::InvalidKernelIr(
                "double-double recursive Cooley-Tukey pass dimensions are inconsistent",
            ));
        }
        if !matches!(
            self.input_storage,
            PrecisionStorage::DoubleDouble | PrecisionStorage::F64
        ) || !matches!(
            self.output_storage,
            PrecisionStorage::DoubleDouble | PrecisionStorage::F64
        ) {
            return Err(VkFftError::InvalidKernelIr(
                "double-double recursive pass storage must be DD or F64",
            ));
        }
        if let DoubleDoubleCooleyTukeyInputModifier::FourStepRight(mapping) = self.input_modifier {
            mapping.validate()?;
            let expected_batch = mapping
                .outer_batch_count
                .checked_mul(mapping.left_len)
                .ok_or(VkFftError::ArithmeticOverflow {
                    operation: "double-double mapped right-upload input batch count",
                })?;
            if self.operation != DoubleDoubleCooleyTukeyPassOperation::PackRightInput
                || self.logical_len != mapping.right_len
                || self.batch_count != expected_batch
            {
                return Err(VkFftError::InvalidKernelIr(
                    "double-double Four-step right-input mapping does not match its component",
                ));
            }
        }
        if let DoubleDoubleCooleyTukeyInputModifier::FourStepThreeUpload2(mapping) =
            self.input_modifier
        {
            mapping.validate()?;
            let [a, b, c] = mapping.axis_split;
            let expected_batch = mapping
                .outer_batch_count
                .checked_mul(a)
                .and_then(|value| value.checked_mul(b))
                .ok_or(VkFftError::ArithmeticOverflow {
                    operation: "double-double mapped three-upload-2 input batch count",
                })?;
            if self.operation != DoubleDoubleCooleyTukeyPassOperation::PackRightInput
                || self.logical_len != c
                || self.batch_count != expected_batch
            {
                return Err(VkFftError::InvalidKernelIr(
                    "double-double three-upload-2 input mapping does not match its component",
                ));
            }
        }
        if let DoubleDoubleCooleyTukeyOutputModifier::FourStepRight(mapping) = self.output_modifier
        {
            mapping.validate()?;
            let expected_batch = mapping
                .outer_batch_count
                .checked_mul(mapping.left_len)
                .ok_or(VkFftError::ArithmeticOverflow {
                    operation: "double-double mapped right-upload output batch count",
                })?;
            if self.operation != DoubleDoubleCooleyTukeyPassOperation::ScatterOutput
                || self.logical_len != mapping.right_len
                || self.batch_count != expected_batch
                || self.output_storage != PrecisionStorage::DoubleDouble
            {
                return Err(VkFftError::InvalidKernelIr(
                    "double-double Four-step right-output mapping does not match its component",
                ));
            }
        }
        if let DoubleDoubleCooleyTukeyOutputModifier::FourStepLeft(mapping) = self.output_modifier {
            mapping.validate()?;
            let expected_batch = mapping
                .outer_batch_count
                .checked_mul(mapping.right_len)
                .ok_or(VkFftError::ArithmeticOverflow {
                    operation: "double-double mapped left-upload output batch count",
                })?;
            if self.operation != DoubleDoubleCooleyTukeyPassOperation::ScatterOutput
                || self.logical_len != mapping.left_len
                || self.batch_count != expected_batch
            {
                return Err(VkFftError::InvalidKernelIr(
                    "double-double Four-step left-output mapping does not match its component",
                ));
            }
        }
        for (upload_id, modifier_mapping) in [
            (
                2usize,
                match self.output_modifier {
                    DoubleDoubleCooleyTukeyOutputModifier::FourStepThreeUpload2(mapping) => {
                        Some(mapping)
                    }
                    _ => None,
                },
            ),
            (
                1usize,
                match self.output_modifier {
                    DoubleDoubleCooleyTukeyOutputModifier::FourStepThreeUpload1(mapping) => {
                        Some(mapping)
                    }
                    _ => None,
                },
            ),
            (
                0usize,
                match self.output_modifier {
                    DoubleDoubleCooleyTukeyOutputModifier::FourStepThreeUpload0(mapping) => {
                        Some(mapping)
                    }
                    _ => None,
                },
            ),
        ] {
            let Some(mapping) = modifier_mapping else {
                continue;
            };
            mapping.validate()?;
            let [a, b, c] = mapping.axis_split;
            let (expected_len, expected_batch) = match upload_id {
                2 => (
                    c,
                    mapping
                        .outer_batch_count
                        .checked_mul(a)
                        .and_then(|value| value.checked_mul(b)),
                ),
                1 => (
                    b,
                    mapping
                        .outer_batch_count
                        .checked_mul(c)
                        .and_then(|value| value.checked_mul(a)),
                ),
                0 => (
                    a,
                    mapping
                        .outer_batch_count
                        .checked_mul(c)
                        .and_then(|value| value.checked_mul(b)),
                ),
                _ => unreachable!(),
            };
            let expected_batch = expected_batch.ok_or(VkFftError::ArithmeticOverflow {
                operation: "double-double mapped three-upload output batch count",
            })?;
            if self.operation != DoubleDoubleCooleyTukeyPassOperation::ScatterOutput
                || self.logical_len != expected_len
                || self.batch_count != expected_batch
                || (upload_id != 0 && self.output_storage != PrecisionStorage::DoubleDouble)
            {
                return Err(VkFftError::InvalidKernelIr(
                    "double-double three-upload output mapping does not match its component",
                ));
            }
        }
        if self.operation != DoubleDoubleCooleyTukeyPassOperation::PackRightInput
            && self.input_modifier != DoubleDoubleCooleyTukeyInputModifier::None
        {
            return Err(VkFftError::InvalidKernelIr(
                "double-double Cooley input modifier is attached to the wrong boundary",
            ));
        }
        if self.operation != DoubleDoubleCooleyTukeyPassOperation::ScatterOutput
            && self.output_modifier != DoubleDoubleCooleyTukeyOutputModifier::None
        {
            return Err(VkFftError::InvalidKernelIr(
                "double-double Cooley output modifier is attached to the wrong boundary",
            ));
        }
        match self.operation {
            DoubleDoubleCooleyTukeyPassOperation::PackRightInput
                if self.output_storage != PrecisionStorage::DoubleDouble =>
            {
                Err(VkFftError::InvalidKernelIr(
                    "double-double recursive pack must write full-DD scratch",
                ))
            }
            DoubleDoubleCooleyTukeyPassOperation::TwiddleTranspose
                if self.input_storage != PrecisionStorage::DoubleDouble
                    || self.output_storage != PrecisionStorage::DoubleDouble =>
            {
                Err(VkFftError::InvalidKernelIr(
                    "double-double recursive twiddle/transpose must stay full-DD",
                ))
            }
            DoubleDoubleCooleyTukeyPassOperation::ScatterOutput
                if self.input_storage != PrecisionStorage::DoubleDouble =>
            {
                Err(VkFftError::InvalidKernelIr(
                    "double-double recursive scatter must read full-DD scratch",
                ))
            }
            _ => Ok(()),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum DoubleDoubleRecursiveFftNodeIr {
    Stockham(Box<DoubleDoubleStockhamIr>),
    DirectRader(Box<DoubleDoubleDirectRaderIr>),
    FftRader(Box<DoubleDoubleFftRaderIr>),
    Bluestein(Box<DoubleDoubleBluesteinIr>),
    CooleyTukey(Box<DoubleDoubleRecursiveCooleyTukeyIr>),
}

// This short-lived mapped-upload value stays inline so materializing a physical
// Stockham component does not introduce a heap allocation purely for enum sizing.
#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum DoubleDoubleForcedRaderThreeUploadComponentIr {
    Stockham {
        upload_id: usize,
        ir: DoubleDoubleStockhamIr,
    },
    Recursive {
        upload_id: usize,
        ir: DoubleDoubleRecursiveFftNodeIr,
    },
}

impl DoubleDoubleForcedRaderThreeUploadComponentIr {
    pub(crate) fn upload_id(&self) -> usize {
        match self {
            Self::Stockham { upload_id, .. } | Self::Recursive { upload_id, .. } => *upload_id,
        }
    }

    pub(crate) fn logical_len(&self) -> usize {
        match self {
            Self::Stockham { ir, .. } => ir.sequence_len,
            Self::Recursive { ir, .. } => ir.logical_len(),
        }
    }
}

impl DoubleDoubleRecursiveFftNodeIr {
    pub fn logical_len(&self) -> usize {
        match self {
            Self::Stockham(ir) => ir.sequence_len,
            Self::DirectRader(ir) => ir.prime,
            Self::FftRader(ir) => ir.prime,
            Self::Bluestein(ir) => ir.logical_len,
            Self::CooleyTukey(ir) => ir.logical_len,
        }
    }

    pub fn batch_count(&self) -> usize {
        match self {
            Self::Stockham(ir) => ir.batch_count,
            Self::DirectRader(ir) => ir.batch_count,
            Self::FftRader(ir) => ir.batch_count,
            Self::Bluestein(ir) => ir.batch_count,
            Self::CooleyTukey(ir) => ir.batch_count,
        }
    }

    pub fn grouped_batch(&self) -> usize {
        match self {
            Self::Stockham(ir) => ir.grouped_batch,
            Self::DirectRader(ir) => ir.grouped_batch,
            Self::FftRader(ir) => ir.grouped_batch,
            Self::Bluestein(ir) => ir.grouped_batch,
            Self::CooleyTukey(ir) => ir.grouped_batch,
        }
    }

    pub fn logical_batch_group_count(&self) -> usize {
        self.batch_count().div_ceil(self.grouped_batch())
    }

    pub fn batch_group_count(&self) -> usize {
        match self {
            Self::Stockham(ir) => ir.logical_batch_group_count(),
            Self::DirectRader(ir) => ir.batch_group_count(),
            Self::FftRader(ir) => ir.batch_group_count(),
            Self::Bluestein(ir) => ir.batch_group_count(),
            Self::CooleyTukey(ir) => ir.batch_count.div_ceil(ir.grouped_batch),
        }
    }

    pub fn direction(&self) -> Direction {
        match self {
            Self::Stockham(ir) => ir.direction,
            Self::DirectRader(ir) => ir.direction,
            Self::FftRader(ir) => ir.direction,
            Self::Bluestein(ir) => ir.direction,
            Self::CooleyTukey(ir) => ir.direction,
        }
    }

    pub fn input_storage(&self) -> PrecisionStorage {
        match self {
            Self::Stockham(ir) => ir.external_storage,
            Self::DirectRader(ir) => ir.input_storage,
            Self::FftRader(ir) => ir.input_storage,
            Self::Bluestein(ir) => ir.external_storage,
            Self::CooleyTukey(ir) => ir.pack_right.input_storage,
        }
    }

    pub fn output_storage(&self) -> PrecisionStorage {
        match self {
            Self::Stockham(ir) => ir.external_storage,
            Self::DirectRader(ir) => ir.output_storage,
            Self::FftRader(ir) => ir.output_storage,
            Self::Bluestein(ir) => ir.external_storage,
            Self::CooleyTukey(ir) => ir.scatter_output.output_storage,
        }
    }

    pub fn with_device_physical_blocks(self, device: DeviceProfile) -> Result<Self> {
        self.with_device_physical_blocks_with_tuning(PlannerTuning::portable(), device)
    }

    fn with_device_physical_blocks_with_tuning(
        self,
        tuning: PlannerTuning,
        device: DeviceProfile,
    ) -> Result<Self> {
        let tuning = upstream_effective_rader_tuning(tuning, device, Precision::DoubleDouble);
        let exact_parent_threads = self.exact_rader_parent_threads_with_tuning(tuning, device)?;
        match self {
            Self::Stockham(ir) => Ok(Self::Stockham(Box::new(
                (*ir).with_axis0_recursive_leaf_stockham_block(device)?,
            ))),
            Self::CooleyTukey(mut ir) => {
                ir.pack_right = ir.pack_right.with_device_physical_block(device)?;
                ir.twiddle_transpose = ir.twiddle_transpose.with_device_physical_block(device)?;
                ir.scatter_output = ir.scatter_output.with_device_physical_block(device)?;
                ir.right = ir
                    .right
                    .with_device_physical_blocks_with_tuning(tuning, device)?;
                ir.left = ir
                    .left
                    .with_device_physical_blocks_with_tuning(tuning, device)?;
                if let Some(threads_per_transform) = exact_parent_threads
                    && let Some(current) = ir.pack_right.axis_batch_block
                {
                    // Upstream clamps the physical coalesced batch after the exact
                    // register/Rader lane floor is known. Logical grouped ownership is
                    // unchanged; only the number of transforms sharing one workgroup is
                    // reduced when `threads_per_transform * grouped_batch` would exceed
                    // the device limits.
                    let thread_dimension_limit = if current.transforms_on_x {
                        device.max_workgroup_size[1]
                    } else {
                        device.max_workgroup_size[0]
                    };
                    let grouped_dimension_limit = if current.transforms_on_x {
                        device.max_workgroup_size[0]
                    } else {
                        device.max_workgroup_size[1]
                    };
                    if threads_per_transform <= thread_dimension_limit
                        && threads_per_transform <= device.max_threads_per_block
                    {
                        let grouped_by_threads =
                            device.max_threads_per_block / threads_per_transform;
                        let grouped_batch = current
                            .grouped_batch
                            .min(grouped_dimension_limit)
                            .min(grouped_by_threads);
                        if grouped_batch > 0 {
                            let (local_size_x, local_size_y) = if current.transforms_on_x {
                                (grouped_batch, threads_per_transform)
                            } else {
                                (threads_per_transform, grouped_batch)
                            };
                            let exact = StockhamAxisBlockSchedule {
                                threads_per_transform,
                                grouped_batch,
                                transforms_on_x: current.transforms_on_x,
                                axis_swapped: current.axis_swapped,
                                local_size_x,
                                local_size_y,
                            };
                            ir.pack_right = ir.pack_right.with_axis_batch_block(exact, device)?;
                            ir.twiddle_transpose =
                                ir.twiddle_transpose.with_axis_batch_block(exact, device)?;
                            ir.scatter_output =
                                ir.scatter_output.with_axis_batch_block(exact, device)?;
                        }
                    }
                }
                ir.validate()?;
                Ok(Self::CooleyTukey(ir))
            }
            other => Ok(other),
        }
    }

    fn contains_rader(&self) -> bool {
        match self {
            Self::DirectRader(_) | Self::FftRader(_) => true,
            Self::CooleyTukey(ir) => ir.left.contains_rader() || ir.right.contains_rader(),
            Self::Stockham(_) | Self::Bluestein(_) => false,
        }
    }

    /// Collect the exact all-Rader parent slice as upstream `raderContainer`
    /// metadata. Repeated direct and FFT-Rader leaves both increment one
    /// `(prime,multiplicity)` entry; Stockham leaves are the smooth outer factor and
    /// Bluestein leaves leave this exact Rader-only scorer.
    fn collect_rader_parent_components(
        &self,
        direct_primes: &mut Vec<(usize, usize)>,
        fft_primes: &mut Vec<(usize, usize)>,
    ) -> bool {
        match self {
            Self::Stockham(_) => true,
            Self::DirectRader(ir) => {
                if let Some((_, multiplicity)) = direct_primes
                    .iter_mut()
                    .find(|(prime, _)| *prime == ir.prime)
                {
                    *multiplicity += 1;
                } else {
                    direct_primes.push((ir.prime, 1));
                }
                true
            }
            Self::FftRader(ir) => {
                if let Some((_, multiplicity)) =
                    fft_primes.iter_mut().find(|(prime, _)| *prime == ir.prime)
                {
                    *multiplicity += 1;
                } else {
                    fft_primes.push((ir.prime, 1));
                }
                true
            }
            Self::CooleyTukey(ir) => {
                ir.left
                    .collect_rader_parent_components(direct_primes, fft_primes)
                    && ir
                        .right
                        .collect_rader_parent_components(direct_primes, fft_primes)
            }
            Self::Bluestein(_) => false,
        }
    }

    fn exact_rader_parent_threads_with_tuning(
        &self,
        tuning: PlannerTuning,
        device: DeviceProfile,
    ) -> Result<Option<usize>> {
        if !matches!(self, Self::CooleyTukey(_)) {
            return Ok(None);
        }
        let mut direct_prime_multiplicities = Vec::new();
        let mut fft_primes = Vec::new();
        if !self.collect_rader_parent_components(&mut direct_prime_multiplicities, &mut fft_primes)
        {
            return Ok(None);
        }
        let sequence_len = self.logical_len();
        let batch_count = self.batch_count();
        if direct_prime_multiplicities.is_empty() && !fft_primes.is_empty() {
            plan_gpu_double_double_axis0_fft_rader_threads_for_prime_multiplicities_with_tuning(
                sequence_len,
                &fft_primes,
                batch_count,
                tuning,
                device,
            )
        } else if !direct_prime_multiplicities.is_empty() && fft_primes.is_empty() {
            plan_gpu_double_double_axis0_composite_direct_rader_threads_for_prime_multiplicities(
                sequence_len,
                &direct_prime_multiplicities,
                batch_count,
                device,
            )
        } else if !direct_prime_multiplicities.is_empty() && !fft_primes.is_empty() {
            plan_gpu_double_double_axis0_mixed_direct_multi_fft_rader_threads_with_tuning(
                sequence_len,
                &direct_prime_multiplicities,
                &fft_primes,
                batch_count,
                tuning,
                device,
            )
        } else {
            Ok(None)
        }
    }

    fn exact_rader_parent_threads_for_forced_upload_with_tuning(
        &self,
        tuning: PlannerTuning,
        max_batch_coalesced: usize,
        axis_upload_id: usize,
        device: DeviceProfile,
    ) -> Result<Option<usize>> {
        if max_batch_coalesced == 0 {
            return Ok(None);
        }
        if let Self::FftRader(ir) = self {
            return plan_gpu_double_double_axis0_fft_rader_threads_for_forced_upload_with_tuning(
                ir.prime,
                &[(ir.prime, 1)],
                self.batch_count(),
                max_batch_coalesced,
                axis_upload_id,
                tuning,
                device,
            );
        }
        if !matches!(self, Self::CooleyTukey(_)) {
            return Ok(None);
        }
        let mut direct_prime_multiplicities = Vec::new();
        let mut fft_primes = Vec::new();
        if !self.collect_rader_parent_components(&mut direct_prime_multiplicities, &mut fft_primes)
        {
            return Ok(None);
        }
        if direct_prime_multiplicities.is_empty() && !fft_primes.is_empty() {
            plan_gpu_double_double_axis0_fft_rader_threads_for_forced_upload_with_tuning(
                self.logical_len(),
                &fft_primes,
                self.batch_count(),
                max_batch_coalesced,
                axis_upload_id,
                tuning,
                device,
            )
        } else if !direct_prime_multiplicities.is_empty() && fft_primes.is_empty() {
            plan_gpu_double_double_axis0_composite_direct_rader_threads_for_prime_multiplicities_with_max_batch_coalesced(
                self.logical_len(),
                &direct_prime_multiplicities,
                self.batch_count(),
                max_batch_coalesced,
                device,
            )
        } else if !direct_prime_multiplicities.is_empty() && !fft_primes.is_empty() {
            plan_gpu_double_double_axis0_mixed_direct_multi_fft_rader_threads_for_forced_upload_with_tuning(
                self.logical_len(),
                &direct_prime_multiplicities,
                &fft_primes,
                self.batch_count(),
                max_batch_coalesced,
                axis_upload_id,
                tuning,
                device,
            )
        } else {
            Ok(None)
        }
    }

    fn exact_rader_parent_split_state_for_forced_upload_with_tuning(
        &self,
        tuning: PlannerTuning,
        max_batch_coalesced: usize,
        axis_upload_id: usize,
        device: DeviceProfile,
    ) -> Result<Option<Axis0RaderSplitState>> {
        if max_batch_coalesced == 0 {
            return Ok(None);
        }
        if let Self::DirectRader(ir) = self {
            return plan_gpu_double_double_axis0_composite_direct_rader_split_state_for_prime_multiplicities_with_max_batch_coalesced(
                ir.prime,
                &[(ir.prime, 1)],
                self.batch_count(),
                max_batch_coalesced,
                device,
            );
        }
        if let Self::FftRader(ir) = self {
            let Some(threads) =
                plan_gpu_double_double_axis0_fft_rader_threads_for_forced_upload_with_tuning(
                    ir.prime,
                    &[(ir.prime, 1)],
                    self.batch_count(),
                    max_batch_coalesced,
                    axis_upload_id,
                    tuning,
                    device,
                )?
            else {
                return Ok(None);
            };
            return Ok(Some(Axis0RaderSplitState {
                base_axis_threads: threads,
                rader_min_registers: 1,
                min_rader_fft_thread_num: threads,
                direct_prime_multiplicities: Vec::new(),
            }));
        }
        if !matches!(self, Self::CooleyTukey(_)) {
            return Ok(None);
        }
        let mut direct_prime_multiplicities = Vec::new();
        let mut fft_primes = Vec::new();
        if !self.collect_rader_parent_components(&mut direct_prime_multiplicities, &mut fft_primes)
        {
            return Ok(None);
        }
        if !direct_prime_multiplicities.is_empty() && fft_primes.is_empty() {
            plan_gpu_double_double_axis0_composite_direct_rader_split_state_for_prime_multiplicities_with_max_batch_coalesced(
                self.logical_len(),
                &direct_prime_multiplicities,
                self.batch_count(),
                max_batch_coalesced,
                device,
            )
        } else if direct_prime_multiplicities.is_empty() && !fft_primes.is_empty() {
            let Some(threads) =
                plan_gpu_double_double_axis0_fft_rader_threads_for_forced_upload_with_tuning(
                    self.logical_len(),
                    &fft_primes,
                    self.batch_count(),
                    max_batch_coalesced,
                    axis_upload_id,
                    tuning,
                    device,
                )?
            else {
                return Ok(None);
            };
            Ok(Some(Axis0RaderSplitState {
                base_axis_threads: threads,
                rader_min_registers: 1,
                min_rader_fft_thread_num: threads,
                direct_prime_multiplicities: Vec::new(),
            }))
        } else if !direct_prime_multiplicities.is_empty() && !fft_primes.is_empty() {
            plan_gpu_double_double_axis0_mixed_direct_multi_fft_rader_split_state_for_forced_upload_with_tuning(
                self.logical_len(),
                &direct_prime_multiplicities,
                &fft_primes,
                self.batch_count(),
                max_batch_coalesced,
                axis_upload_id,
                tuning,
                device,
            )
        } else {
            Ok(None)
        }
    }

    fn direct_rader_parent_base_threads_for_other_axis(
        &self,
        device: DeviceProfile,
    ) -> Result<Option<usize>> {
        let mut direct_prime_multiplicities = Vec::new();
        let mut fft_primes = Vec::new();
        if !self.collect_rader_parent_components(&mut direct_prime_multiplicities, &mut fft_primes)
            || direct_prime_multiplicities.is_empty()
            || !fft_primes.is_empty()
        {
            return Ok(None);
        }
        plan_gpu_double_double_other_axis_composite_direct_rader_threads_for_prime_multiplicities(
            self.logical_len(),
            &direct_prime_multiplicities,
            self.batch_count(),
            device,
        )
    }

    fn axis0_forced_rader_leaf_threads(&self) -> Option<usize> {
        match self {
            Self::DirectRader(ir) => Some(
                ir.axis_batch_block
                    .map_or(ir.prime.div_ceil(2), |block| block.threads_per_transform),
            ),
            Self::FftRader(ir) => Some(
                ir.caller_axis_batch_block
                    .map_or(ir.prime, |block| block.threads_per_transform),
            ),
            _ => None,
        }
    }

    fn apply_axis0_parent_block(
        &mut self,
        block: StockhamAxisBlockSchedule,
        device: DeviceProfile,
    ) -> Result<()> {
        match self {
            Self::CooleyTukey(ir) => {
                ir.pack_right = ir.pack_right.clone().with_axis_batch_block(block, device)?;
                ir.twiddle_transpose = ir
                    .twiddle_transpose
                    .clone()
                    .with_axis_batch_block(block, device)?;
                ir.scatter_output = ir
                    .scatter_output
                    .clone()
                    .with_axis_batch_block(block, device)?;
                ir.validate()
            }
            Self::DirectRader(ir) => {
                block.validate(ir.batch_count, device)?;
                ir.axis_batch_block = Some(block);
                ir.validate()
            }
            Self::FftRader(ir) => {
                block.validate(ir.batch_count, device)?;
                ir.caller_axis_batch_block = Some(block);
                ir.validate()
            }
            Self::Stockham(ir) => {
                block.validate(ir.batch_count, device)?;
                ir.axis_batch_block = Some(block);
                ir.validate()
            }
            Self::Bluestein(_) => Ok(()),
        }
    }

    fn with_other_axis_forced_upload_component_block(
        mut self,
        upload_count: usize,
        axis_upload_id: usize,
        fastest_axis_len: usize,
        grouped_batch_override: Option<usize>,
        axis1_grouped_batch_override: Option<usize>,
        device: DeviceProfile,
    ) -> Result<Self> {
        if matches!(self, Self::Bluestein(_)) {
            return self.with_other_axis_component_block(
                fastest_axis_len,
                grouped_batch_override,
                axis1_grouped_batch_override,
                device,
            );
        }
        let threads_per_transform = match &self {
            Self::Stockham(ir) => ir.axis_batch_block.map(|block| block.threads_per_transform),
            Self::DirectRader(ir) => Some(
                ir.axis_batch_block
                    .map_or(ir.prime.div_ceil(2), |block| block.threads_per_transform),
            ),
            Self::FftRader(ir) => Some(
                ir.caller_axis_batch_block
                    .map_or(ir.prime, |block| block.threads_per_transform),
            ),
            Self::CooleyTukey(ir) => self
                .direct_rader_parent_base_threads_for_other_axis(device)?
                .or_else(|| {
                    Some(
                        ir.pack_right
                            .axis_batch_block
                            .map_or(ir.pack_right.workgroup_size.x as usize, |block| {
                                block.threads_per_transform
                            })
                            .max(1),
                    )
                }),
            Self::Bluestein(_) => unreachable!("handled above"),
        };
        let Some(threads_per_transform) = threads_per_transform else {
            return Ok(self);
        };
        let mut direct_prime_multiplicities = Vec::new();
        let mut fft_primes = Vec::new();
        self.collect_rader_parent_components(&mut direct_prime_multiplicities, &mut fft_primes);
        let direct_prime = direct_prime_multiplicities
            .iter()
            .map(|(prime, _)| *prime)
            .max()
            .unwrap_or(1);
        let reserve = direct_prime.saturating_sub(1).checked_mul(32).ok_or(
            VkFftError::ArithmeticOverflow {
                operation: "higher-axis DD forced-Rader shared-memory reserve",
            },
        )?;
        let Some(available_shared_memory_bytes) = device.shared_memory_bytes.checked_sub(reserve)
        else {
            return Ok(self);
        };
        if let Some(block) = plan_gpu_other_axis_four_step_block_from_shape_for_precision(
            upload_count,
            axis_upload_id,
            self.logical_len(),
            self.batch_count(),
            fastest_axis_len,
            threads_per_transform,
            Precision::DoubleDouble,
            32,
            grouped_batch_override,
            axis1_grouped_batch_override,
            OtherAxisFourStepPhysicalContext {
                has_direct_rader: !direct_prime_multiplicities.is_empty(),
                available_shared_memory_bytes,
            },
            device,
        )? {
            self.apply_axis0_parent_block(block, device)?;
        }
        Ok(self)
    }

    fn with_other_axis_component_block(
        self,
        fastest_axis_len: usize,
        grouped_batch_override: Option<usize>,
        axis1_grouped_batch_override: Option<usize>,
        device: DeviceProfile,
    ) -> Result<Self> {
        match self {
            Self::Stockham(ir) => {
                let mut ir = *ir;
                let Some(current) = ir.axis_batch_block else {
                    return Ok(Self::Stockham(Box::new(ir)));
                };
                if let Some(block) = plan_gpu_other_axis_composite_rader_batch_block(
                    ir.sequence_len,
                    ir.batch_count,
                    fastest_axis_len,
                    32,
                    current.threads_per_transform,
                    grouped_batch_override,
                    axis1_grouped_batch_override,
                    device,
                )? {
                    ir.axis_batch_block = Some(block);
                    ir.validate()?;
                }
                Ok(Self::Stockham(Box::new(ir)))
            }
            Self::DirectRader(ir) => Ok(Self::DirectRader(Box::new(
                (*ir).with_other_axis_grouped_direct_rader_block(
                    fastest_axis_len,
                    grouped_batch_override,
                    axis1_grouped_batch_override,
                    device,
                )?,
            ))),
            Self::FftRader(ir) => Ok(Self::FftRader(Box::new(
                (*ir).with_other_axis_grouped_fft_rader_block(
                    fastest_axis_len,
                    grouped_batch_override,
                    axis1_grouped_batch_override,
                    device,
                )?,
            ))),
            Self::Bluestein(ir) => Ok(Self::Bluestein(Box::new((*ir).with_other_axis_blocks(
                fastest_axis_len,
                grouped_batch_override,
                axis1_grouped_batch_override,
                device,
            )?))),
            Self::CooleyTukey(mut ir) => {
                let node = Self::CooleyTukey(ir.clone());
                let threads_per_transform = node
                    .direct_rader_parent_base_threads_for_other_axis(device)?
                    .unwrap_or_else(|| {
                        ir.pack_right
                            .axis_batch_block
                            .map_or(ir.pack_right.workgroup_size.x as usize, |block| {
                                block.threads_per_transform
                            })
                            .max(1)
                    });
                if let Some(block) = plan_gpu_other_axis_composite_rader_batch_block(
                    ir.logical_len,
                    ir.batch_count,
                    fastest_axis_len,
                    32,
                    threads_per_transform,
                    grouped_batch_override,
                    axis1_grouped_batch_override,
                    device,
                )? {
                    ir.pack_right = ir.pack_right.with_axis_batch_block(block, device)?;
                    ir.twiddle_transpose =
                        ir.twiddle_transpose.with_axis_batch_block(block, device)?;
                    ir.scatter_output = ir.scatter_output.with_axis_batch_block(block, device)?;
                }
                ir.validate()?;
                Ok(Self::CooleyTukey(ir))
            }
        }
    }

    pub fn validate(&self) -> Result<()> {
        match self {
            Self::Stockham(ir) => ir.validate(),
            Self::DirectRader(ir) => ir.validate(),
            Self::FftRader(ir) => ir.validate(),
            Self::Bluestein(ir) => ir.validate(),
            Self::CooleyTukey(ir) => ir.validate(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct DoubleDoubleFusedSmallDirectRaderStockhamIr<'a> {
    pub parent: &'a DoubleDoubleRecursiveCooleyTukeyIr,
    pub stockham: &'a DoubleDoubleStockhamIr,
    pub direct: &'a DoubleDoubleDirectRaderIr,
    pub axis_block: StockhamAxisBlockSchedule,
}

impl DoubleDoubleFusedSmallDirectRaderStockhamIr<'_> {
    pub(crate) fn name(self) -> String {
        format!(
            "vkfft_dd_composite_direct_rader_stockham_{}_{}x{}_{:?}",
            self.parent.logical_len,
            self.parent.left_len,
            self.parent.right_len,
            self.parent.direction
        )
    }

    pub(crate) fn required_shared_memory_bytes(self) -> Result<usize> {
        self.parent
            .logical_len
            .checked_mul(self.axis_block.grouped_batch)
            .and_then(|value| value.checked_mul(32))
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "double-double fused Direct-Rader Stockham shared-memory size",
            })
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct DoubleDoubleFusedSmallFftRaderStockhamIr<'a> {
    pub parent: &'a DoubleDoubleRecursiveCooleyTukeyIr,
    pub stockham: &'a DoubleDoubleStockhamIr,
    pub rader: &'a DoubleDoubleFftRaderIr,
    pub axis_block: StockhamAxisBlockSchedule,
}

impl DoubleDoubleFusedSmallFftRaderStockhamIr<'_> {
    pub(crate) fn name(self) -> String {
        format!(
            "vkfft_dd_composite_fft_rader_stockham_{}_{}x{}_{:?}",
            self.parent.logical_len,
            self.parent.left_len,
            self.parent.right_len,
            self.parent.direction
        )
    }

    pub(crate) fn required_shared_memory_bytes(self) -> Result<usize> {
        let convolution_elements = self
            .parent
            .left_len
            .checked_mul(self.rader.convolution_len)
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "double-double fused FFT-Rader convolution shared elements",
            })?;
        let left_exchange_elements = if self.stockham.stages.len() > 1 {
            self.parent.logical_len
        } else {
            0
        };
        convolution_elements
            .checked_add(self.parent.logical_len)
            .and_then(|value| value.checked_add(left_exchange_elements))
            .and_then(|value| value.checked_mul(self.axis_block.grouped_batch))
            .and_then(|value| value.checked_mul(32))
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "double-double fused FFT-Rader shared-memory size",
            })
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct DoubleDoubleFusedDualFftRaderIr<'a> {
    pub parent: &'a DoubleDoubleRecursiveCooleyTukeyIr,
    pub left: &'a DoubleDoubleFftRaderIr,
    pub right: &'a DoubleDoubleFftRaderIr,
    pub axis_block: StockhamAxisBlockSchedule,
}

impl DoubleDoubleFusedDualFftRaderIr<'_> {
    pub(crate) fn name(self) -> String {
        format!(
            "vkfft_dd_composite_dual_fft_rader_{}_{}x{}_{:?}",
            self.parent.logical_len,
            self.parent.left_len,
            self.parent.right_len,
            self.parent.direction
        )
    }

    pub(crate) fn required_shared_memory_bytes(self) -> Result<usize> {
        let right_convolution_elements = self
            .parent
            .left_len
            .checked_mul(self.right.convolution_len)
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "double-double fused dual FFT-Rader right convolution shared elements",
            })?;
        let left_convolution_elements = self
            .parent
            .right_len
            .checked_mul(self.left.convolution_len)
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "double-double fused dual FFT-Rader left convolution shared elements",
            })?;
        let convolution_elements = right_convolution_elements.max(left_convolution_elements);
        let retained_left_scalars =
            self.parent
                .right_len
                .checked_mul(2)
                .ok_or(VkFftError::ArithmeticOverflow {
                    operation: "double-double fused dual FFT-Rader retained left values",
                })?;
        convolution_elements
            .checked_add(self.parent.logical_len)
            .and_then(|value| value.checked_add(retained_left_scalars))
            .and_then(|value| value.checked_mul(self.axis_block.grouped_batch))
            .and_then(|value| value.checked_mul(32))
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "double-double fused dual FFT-Rader shared-memory size",
            })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct DoubleDoubleRecursiveCooleyTukeyIr {
    pub logical_len: usize,
    pub left_len: usize,
    pub right_len: usize,
    pub batch_count: usize,
    pub grouped_batch: usize,
    pub direction: Direction,
    pub pack_right: DoubleDoubleCooleyTukeyPassIr,
    pub right: DoubleDoubleRecursiveFftNodeIr,
    pub twiddle_transpose: DoubleDoubleCooleyTukeyPassIr,
    /// One full-period true-DD unit-root table. The twiddle pass indexes it by
    /// `(n1 * k2) % logical_len`, avoiding an F64 trigonometric narrowing step.
    pub twiddles: Vec<ComplexDoubleDouble>,
    pub left: DoubleDoubleRecursiveFftNodeIr,
    pub scatter_output: DoubleDoubleCooleyTukeyPassIr,
}

impl DoubleDoubleRecursiveCooleyTukeyIr {
    pub(crate) fn fused_small_direct_rader_stockham(
        &self,
    ) -> Result<Option<DoubleDoubleFusedSmallDirectRaderStockhamIr<'_>>> {
        self.validate()?;
        if self.left_len > 12
            || self.logical_len > 576
            || self.pack_right.input_storage != PrecisionStorage::DoubleDouble
            || self.pack_right.output_storage != PrecisionStorage::DoubleDouble
            || self.scatter_output.input_storage != PrecisionStorage::DoubleDouble
            || self.scatter_output.output_storage != PrecisionStorage::DoubleDouble
            || self.pack_right.input_modifier != DoubleDoubleCooleyTukeyInputModifier::None
            || self.scatter_output.output_modifier != DoubleDoubleCooleyTukeyOutputModifier::None
        {
            return Ok(None);
        }
        let Some(axis_block) = self.pack_right.axis_batch_block else {
            return Ok(None);
        };
        let Some(device_shared_memory_bytes) = self.pack_right.device_shared_memory_bytes else {
            return Ok(None);
        };
        if self.twiddle_transpose.axis_batch_block != Some(axis_block)
            || self.scatter_output.axis_batch_block != Some(axis_block)
            || self.twiddle_transpose.device_shared_memory_bytes != Some(device_shared_memory_bytes)
            || self.scatter_output.device_shared_memory_bytes != Some(device_shared_memory_bytes)
        {
            return Ok(None);
        }
        let DoubleDoubleRecursiveFftNodeIr::Stockham(stockham) = &self.left else {
            return Ok(None);
        };
        let DoubleDoubleRecursiveFftNodeIr::DirectRader(direct) = &self.right else {
            return Ok(None);
        };
        if stockham.sequence_len != self.left_len
            || stockham.external_storage != PrecisionStorage::DoubleDouble
            || stockham.zero_pad_pass.is_some()
            || direct.prime != self.right_len
            || direct.external_storage != PrecisionStorage::DoubleDouble
            || direct.input_storage != PrecisionStorage::DoubleDouble
            || direct.output_storage != PrecisionStorage::DoubleDouble
            || direct.io_mapping != StockhamIoMapping::Contiguous
            || direct.zero_pad_pass.is_some()
        {
            return Ok(None);
        }
        let fused = DoubleDoubleFusedSmallDirectRaderStockhamIr {
            parent: self,
            stockham,
            direct,
            axis_block,
        };
        if fused.required_shared_memory_bytes()? > device_shared_memory_bytes {
            return Ok(None);
        }
        Ok(Some(fused))
    }

    pub(crate) fn fused_small_fft_rader_stockham(
        &self,
    ) -> Result<Option<DoubleDoubleFusedSmallFftRaderStockhamIr<'_>>> {
        self.validate()?;
        if !((3..=16).contains(&self.left_len)
            || matches!(
                self.left_len,
                18 | 20 | 21 | 22 | 24 | 25 | 26 | 27 | 28 | 30 | 32
            ))
            || self.left_len.checked_mul(17) != Some(self.logical_len)
            || self.right_len != 17
            || self.batch_count != 1
            || self.grouped_batch != 1
            || self.pack_right.input_storage != PrecisionStorage::DoubleDouble
            || self.pack_right.output_storage != PrecisionStorage::DoubleDouble
            || self.scatter_output.input_storage != PrecisionStorage::DoubleDouble
            || self.scatter_output.output_storage != PrecisionStorage::DoubleDouble
            || self.pack_right.input_modifier != DoubleDoubleCooleyTukeyInputModifier::None
            || self.scatter_output.output_modifier != DoubleDoubleCooleyTukeyOutputModifier::None
        {
            return Ok(None);
        }
        let Some(axis_block) = self.pack_right.axis_batch_block else {
            return Ok(None);
        };
        let Some(device_shared_memory_bytes) = self.pack_right.device_shared_memory_bytes else {
            return Ok(None);
        };
        if axis_block.grouped_batch != 1
            || axis_block.threads_per_transform < self.left_len
            || self.twiddle_transpose.axis_batch_block != Some(axis_block)
            || self.scatter_output.axis_batch_block != Some(axis_block)
            || self.twiddle_transpose.device_shared_memory_bytes != Some(device_shared_memory_bytes)
            || self.scatter_output.device_shared_memory_bytes != Some(device_shared_memory_bytes)
        {
            return Ok(None);
        }
        let DoubleDoubleRecursiveFftNodeIr::Stockham(stockham) = &self.left else {
            return Ok(None);
        };
        let DoubleDoubleRecursiveFftNodeIr::FftRader(rader) = &self.right else {
            return Ok(None);
        };
        let single_stage_left = stockham.stages.len() == 1
            && stockham.stages[0].index == 0
            && stockham.stages[0].radix == self.left_len
            && stockham.stages[0].stage_size == 1
            && stockham.stages[0].butterflies == 1;
        let two_stage_left = stockham.stages.len() == 2
            && stockham.stages[0].index == 0
            && stockham.stages[0].stage_size == 1
            && stockham.stages[1].index == 1
            && stockham.stages[1].stage_size == stockham.stages[0].radix
            && stockham.stages[0]
                .radix
                .checked_mul(stockham.stages[1].radix)
                == Some(self.left_len)
            && stockham.stages[0].butterflies == self.left_len / stockham.stages[0].radix
            && stockham.stages[1].butterflies == self.left_len / stockham.stages[1].radix;
        if stockham.sequence_len != self.left_len
            || stockham.external_storage != PrecisionStorage::DoubleDouble
            || stockham.zero_pad_pass.is_some()
            || !(single_stage_left || two_stage_left)
            || rader.prime != 17
            || rader.convolution_len != 16
            || rader.external_storage != PrecisionStorage::DoubleDouble
            || rader.input_storage != PrecisionStorage::DoubleDouble
            || rader.output_storage != PrecisionStorage::DoubleDouble
            || rader.zero_pad_pass.is_some()
            || rader.io_mapping != StockhamIoMapping::Contiguous
            || rader.caller_axis_batch_block.is_some()
        {
            return Ok(None);
        }
        let DoubleDoubleBluesteinConvolutionIr::Stockham(forward) = &rader.forward_fft else {
            return Ok(None);
        };
        let DoubleDoubleBluesteinConvolutionIr::Stockham(inverse) = &rader.inverse_fft else {
            return Ok(None);
        };
        let valid_convolution_child = |child: &DoubleDoubleStockhamIr, normalize: bool| {
            child.sequence_len == 16
                && child.batch_count == self.left_len
                && child.grouped_batch == self.left_len
                && child.external_storage == PrecisionStorage::DoubleDouble
                && child.zero_pad_pass.is_none()
                && child.axis_batch_block.is_none()
                && child.normalize == normalize
                && child.stages.len() == 1
                && child.stages[0].radix == 16
                && child.stages[0].stage_size == 1
                && child.stages[0].butterflies == 1
        };
        if !valid_convolution_child(forward, false)
            || !valid_convolution_child(inverse, true)
            || rader.kernel_spectrum.len() != 16
            || rader.table.permutation.len() != 16
        {
            return Ok(None);
        }
        let fused = DoubleDoubleFusedSmallFftRaderStockhamIr {
            parent: self,
            stockham,
            rader,
            axis_block,
        };
        if fused.required_shared_memory_bytes()? > device_shared_memory_bytes {
            return Ok(None);
        }
        Ok(Some(fused))
    }

    fn fused_dual_fft_rader_strict(
        &self,
        left_prime: usize,
        right_prime: usize,
        left_convolution_len: usize,
        right_convolution_len: usize,
        left_stage_radices: &[usize],
        right_stage_radices: &[usize],
    ) -> Result<Option<DoubleDoubleFusedDualFftRaderIr<'_>>> {
        self.validate()?;
        if self.left_len != left_prime
            || self.right_len != right_prime
            || self.left_len.checked_mul(self.right_len) != Some(self.logical_len)
            || self.batch_count != 1
            || self.grouped_batch != 1
            || self.pack_right.input_storage != PrecisionStorage::DoubleDouble
            || self.pack_right.output_storage != PrecisionStorage::DoubleDouble
            || self.scatter_output.input_storage != PrecisionStorage::DoubleDouble
            || self.scatter_output.output_storage != PrecisionStorage::DoubleDouble
            || self.pack_right.input_modifier != DoubleDoubleCooleyTukeyInputModifier::None
            || self.scatter_output.output_modifier != DoubleDoubleCooleyTukeyOutputModifier::None
        {
            return Ok(None);
        }
        let Some(axis_block) = self.pack_right.axis_batch_block else {
            return Ok(None);
        };
        let Some(device_shared_memory_bytes) = self.pack_right.device_shared_memory_bytes else {
            return Ok(None);
        };
        if axis_block.grouped_batch != 1
            || axis_block.threads_per_transform < self.left_len.max(self.right_len)
            || self.twiddle_transpose.axis_batch_block != Some(axis_block)
            || self.scatter_output.axis_batch_block != Some(axis_block)
            || self.twiddle_transpose.device_shared_memory_bytes != Some(device_shared_memory_bytes)
            || self.scatter_output.device_shared_memory_bytes != Some(device_shared_memory_bytes)
        {
            return Ok(None);
        }
        let DoubleDoubleRecursiveFftNodeIr::FftRader(left) = &self.left else {
            return Ok(None);
        };
        let DoubleDoubleRecursiveFftNodeIr::FftRader(right) = &self.right else {
            return Ok(None);
        };
        let valid_rader = |rader: &DoubleDoubleFftRaderIr,
                           prime: usize,
                           convolution_len: usize,
                           batch_count: usize,
                           stage_radices: &[usize]|
         -> bool {
            if rader.prime != prime
                || rader.convolution_len != convolution_len
                || rader.batch_count != batch_count
                || rader.grouped_batch != batch_count
                || rader.direction != self.direction
                || rader.external_storage != PrecisionStorage::DoubleDouble
                || rader.input_storage != PrecisionStorage::DoubleDouble
                || rader.output_storage != PrecisionStorage::DoubleDouble
                || rader.zero_pad_pass.is_some()
                || rader.io_mapping != StockhamIoMapping::Contiguous
                || rader.caller_axis_batch_block.is_some()
                || rader.kernel_spectrum.len() != convolution_len
                || rader.table.permutation.len() != convolution_len
            {
                return false;
            }
            let DoubleDoubleBluesteinConvolutionIr::Stockham(forward) = &rader.forward_fft else {
                return false;
            };
            let DoubleDoubleBluesteinConvolutionIr::Stockham(inverse) = &rader.inverse_fft else {
                return false;
            };
            let valid_child = |child: &DoubleDoubleStockhamIr, normalize: bool| {
                child.sequence_len == convolution_len
                    && child.batch_count == batch_count
                    && child.grouped_batch == batch_count
                    && child.external_storage == PrecisionStorage::DoubleDouble
                    && child.zero_pad_pass.is_none()
                    && child.axis_batch_block.is_none()
                    && child.normalize == normalize
                    && child.stages.len() == stage_radices.len()
                    && child
                        .stages
                        .iter()
                        .zip(stage_radices.iter().copied())
                        .all(|(stage, radix)| stage.radix == radix)
            };
            valid_child(forward, false) && valid_child(inverse, true)
        };
        if !valid_rader(
            left,
            left_prime,
            left_convolution_len,
            self.right_len,
            left_stage_radices,
        ) || !valid_rader(
            right,
            right_prime,
            right_convolution_len,
            self.left_len,
            right_stage_radices,
        ) {
            return Ok(None);
        }
        let fused = DoubleDoubleFusedDualFftRaderIr {
            parent: self,
            left,
            right,
            axis_block,
        };
        if fused.required_shared_memory_bytes()? > device_shared_memory_bytes {
            return Ok(None);
        }
        Ok(Some(fused))
    }

    pub(crate) fn fused_dual_fft_rader_n289(
        &self,
    ) -> Result<Option<DoubleDoubleFusedDualFftRaderIr<'_>>> {
        if self.logical_len != 289 {
            return Ok(None);
        }
        self.fused_dual_fft_rader_strict(17, 17, 16, 16, &[16], &[16])
    }

    pub(crate) fn fused_dual_fft_rader_n323(
        &self,
    ) -> Result<Option<DoubleDoubleFusedDualFftRaderIr<'_>>> {
        if self.logical_len != 323 {
            return Ok(None);
        }
        self.fused_dual_fft_rader_strict(17, 19, 16, 18, &[16], &[9, 2])
    }

    pub(crate) fn fused_dual_fft_rader_n493(
        &self,
    ) -> Result<Option<DoubleDoubleFusedDualFftRaderIr<'_>>> {
        if self.logical_len != 493 {
            return Ok(None);
        }
        self.fused_dual_fft_rader_strict(17, 29, 16, 28, &[16], &[14, 2])
    }

    pub(crate) fn fused_dual_fft_rader_n527(
        &self,
    ) -> Result<Option<DoubleDoubleFusedDualFftRaderIr<'_>>> {
        if self.logical_len != 527 {
            return Ok(None);
        }
        self.fused_dual_fft_rader_strict(17, 31, 16, 30, &[16], &[15, 2])
    }

    pub(crate) fn fused_dual_fft_rader_supported(
        &self,
    ) -> Result<Option<DoubleDoubleFusedDualFftRaderIr<'_>>> {
        if let Some(fused) = self.fused_dual_fft_rader_n289()? {
            return Ok(Some(fused));
        }
        if let Some(fused) = self.fused_dual_fft_rader_n323()? {
            return Ok(Some(fused));
        }
        if let Some(fused) = self.fused_dual_fft_rader_n493()? {
            return Ok(Some(fused));
        }
        self.fused_dual_fft_rader_n527()
    }

    pub fn validate(&self) -> Result<()> {
        if self.left_len < 2
            || self.right_len < 2
            || self.batch_count == 0
            || self.grouped_batch == 0
            || self.left_len.checked_mul(self.right_len) != Some(self.logical_len)
            || self.twiddles.len() != self.logical_len
        {
            return Err(VkFftError::InvalidKernelIr(
                "double-double recursive Cooley-Tukey node dimensions are inconsistent",
            ));
        }
        self.pack_right.validate()?;
        self.right.validate()?;
        self.twiddle_transpose.validate()?;
        self.left.validate()?;
        self.scatter_output.validate()?;
        let group_count = self.batch_count.div_ceil(self.grouped_batch);
        let expected_right_grouped = self.grouped_batch.checked_mul(self.left_len).ok_or(
            VkFftError::ArithmeticOverflow {
                operation: "double-double recursive right-child grouped batch ownership",
            },
        )?;
        let expected_left_grouped = self.grouped_batch.checked_mul(self.right_len).ok_or(
            VkFftError::ArithmeticOverflow {
                operation: "double-double recursive left-child grouped batch ownership",
            },
        )?;
        if self.pack_right.grouped_batch != self.grouped_batch
            || self.twiddle_transpose.grouped_batch != self.grouped_batch
            || self.scatter_output.grouped_batch != self.grouped_batch
            || self.right.grouped_batch() != expected_right_grouped
            || self.left.grouped_batch() != expected_left_grouped
            || self.right.logical_batch_group_count() != group_count
            || self.left.logical_batch_group_count() != group_count
        {
            return Err(VkFftError::InvalidKernelIr(
                "double-double recursive Cooley-Tukey grouped ownership is inconsistent",
            ));
        }
        for pass in [
            &self.pack_right,
            &self.twiddle_transpose,
            &self.scatter_output,
        ] {
            if pass.logical_len != self.logical_len
                || pass.left_len != self.left_len
                || pass.right_len != self.right_len
                || pass.batch_count != self.batch_count
                || pass.direction != self.direction
            {
                return Err(VkFftError::InvalidKernelIr(
                    "double-double recursive Cooley-Tukey pass metadata is inconsistent",
                ));
            }
        }
        let right_batch =
            self.batch_count
                .checked_mul(self.left_len)
                .ok_or(VkFftError::ArithmeticOverflow {
                    operation: "double-double recursive right-child batch count",
                })?;
        let left_batch =
            self.batch_count
                .checked_mul(self.right_len)
                .ok_or(VkFftError::ArithmeticOverflow {
                    operation: "double-double recursive left-child batch count",
                })?;
        if self.pack_right.operation != DoubleDoubleCooleyTukeyPassOperation::PackRightInput
            || self.twiddle_transpose.operation
                != DoubleDoubleCooleyTukeyPassOperation::TwiddleTranspose
            || self.scatter_output.operation != DoubleDoubleCooleyTukeyPassOperation::ScatterOutput
            || self.right.logical_len() != self.right_len
            || self.right.batch_count() != right_batch
            || self.left.logical_len() != self.left_len
            || self.left.batch_count() != left_batch
            || self.right.direction() != self.direction
            || self.left.direction() != self.direction
            || self.right.input_storage() != PrecisionStorage::DoubleDouble
            || self.right.output_storage() != PrecisionStorage::DoubleDouble
            || self.left.input_storage() != PrecisionStorage::DoubleDouble
            || self.left.output_storage() != PrecisionStorage::DoubleDouble
        {
            return Err(VkFftError::InvalidKernelIr(
                "double-double recursive Cooley-Tukey child metadata is inconsistent",
            ));
        }
        if self.twiddles.iter().any(|value| {
            !value.re.hi.is_finite()
                || !value.re.lo.is_finite()
                || !value.im.hi.is_finite()
                || !value.im.lo.is_finite()
        }) {
            return Err(VkFftError::InvalidKernelIr(
                "double-double recursive Cooley-Tukey twiddle table is not finite",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DoubleDoubleTwoUploadFourStepPlanIr {
    pub logical_len: usize,
    pub batch_count: usize,
    pub left_len: usize,
    pub right_len: usize,
    pub right_axis_block: StockhamAxisBlockSchedule,
    pub left_axis_block: StockhamAxisBlockSchedule,
}

impl DoubleDoubleTwoUploadFourStepPlanIr {
    pub fn validate(&self) -> Result<()> {
        if self.logical_len == 0
            || self.batch_count == 0
            || self.left_len < 2
            || self.right_len < 2
            || self.left_len.checked_mul(self.right_len) != Some(self.logical_len)
            || self.left_len > DD_TWO_UPLOAD_FOUR_STEP_LEAF_LIMIT
            || self.right_len > DD_TWO_UPLOAD_FOUR_STEP_LEAF_LIMIT
            || self.right_axis_block.grouped_batch == 0
            || self.right_axis_block.threads_per_transform == 0
            || self.left_axis_block.grouped_batch == 0
            || self.left_axis_block.threads_per_transform == 0
        {
            return Err(VkFftError::InvalidKernelIr(
                "double-double two-upload Four-step metadata is inconsistent",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DoubleDoubleThreeUploadFourStepPlanIr {
    pub logical_len: usize,
    pub batch_count: usize,
    pub axis_split: [usize; 3],
    pub upload2_axis_block: StockhamAxisBlockSchedule,
    pub upload1_axis_block: StockhamAxisBlockSchedule,
    pub upload0_axis_block: StockhamAxisBlockSchedule,
}

impl DoubleDoubleThreeUploadFourStepPlanIr {
    pub fn validate(&self) -> Result<()> {
        let [a, b, c] = self.axis_split;
        let product = a.checked_mul(b).and_then(|value| value.checked_mul(c));
        if self.logical_len == 0
            || self.batch_count == 0
            || a < 2
            || b < 2
            || c < 2
            || product != Some(self.logical_len)
            || self
                .axis_split
                .iter()
                .any(|length| *length > DD_THREE_UPLOAD_FOUR_STEP_LEAF_LIMIT)
            || self.upload2_axis_block.grouped_batch == 0
            || self.upload2_axis_block.threads_per_transform == 0
            || self.upload1_axis_block.grouped_batch == 0
            || self.upload1_axis_block.threads_per_transform == 0
            || self.upload0_axis_block.grouped_batch == 0
            || self.upload0_axis_block.threads_per_transform == 0
        {
            return Err(VkFftError::InvalidKernelIr(
                "double-double three-upload Four-step metadata is inconsistent",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct DoubleDoubleRecursiveFftIr {
    pub name: String,
    pub logical_len: usize,
    pub batch_count: usize,
    pub grouped_batch: usize,
    /// Preserve whether groupedBatch was explicitly configured. The logical grouped
    /// width alone cannot distinguish automatic ownership from an explicit value of 1,
    /// while upstream `VkFFTSplitAxisBlock` takes a different early physical branch
    /// whenever the configuration entry is non-zero.
    pub grouped_batch_override: Option<usize>,
    pub direction: Direction,
    pub normalize: bool,
    pub external_storage: PrecisionStorage,
    /// Caller-visible spatial zero-padding. The pass wraps only the recursive
    /// root boundary; recursive leaves never inherit this range.
    pub zero_pad_pass: Option<ZeroPadPassIr>,
    /// Fixed-upstream boost-1 Quad upload split used to construct this tree when
    /// a concrete GPU profile is available. Portable builds retain `None` and
    /// use the correctness-first balanced chunking fallback.
    pub stockham_upload_schedule: Option<DoubleDoubleStockhamUploadSchedule>,
    /// Fixed-upstream non-power-of-two upload split selected when FFT-Rader
    /// container pressure forces the axis across multiple uploads. The schedule
    /// controls the recursive root cuts even before generic DD Four-step component
    /// boundary fusion is attached.
    pub rader_forced_upload_schedule: Option<RaderUploadSchedule>,
    /// Executable two-upload Four-step path. When present, the root pack,
    /// twiddle/transpose, and scatter boundaries are fused into the two scheduled
    /// Stockham upload kernels instead of being emitted as standalone passes.
    pub two_upload_four_step_plan: Option<DoubleDoubleTwoUploadFourStepPlanIr>,
    /// Executable three-upload Four-step path. This mirrors upstream launch order
    /// [2, 1, 0] and keeps all intermediate storage in full double-double precision.
    pub three_upload_four_step_plan: Option<DoubleDoubleThreeUploadFourStepPlanIr>,
    pub root: DoubleDoubleRecursiveFftNodeIr,
}

#[derive(Debug, Clone)]
enum FactorKind {
    Stockham,
    Rader(RaderMode),
}

#[derive(Debug, Clone)]
struct FactorSpec {
    len: usize,
    kind: FactorKind,
}

impl DoubleDoubleRecursiveFftIr {
    pub fn build(plan: &FftPlan, direction: Direction) -> Result<Self> {
        Self::build_with_upload_schedules(plan, direction, None, None, plan.config.tuning, None)
    }

    pub fn build_for_device(
        plan: &FftPlan,
        direction: Direction,
        device: DeviceProfile,
    ) -> Result<Self> {
        let axis = plan.axes.first().ok_or(VkFftError::InvalidKernelIr(
            "missing double-double recursive axis plan",
        ))?;
        let effective_rader_tuning =
            upstream_effective_rader_tuning(plan.config.tuning, device, plan.config.precision);
        let axis_context = StockhamUploadAxisContext {
            strided_axis: plan.c2c_device_axis_class_override == Some(C2cDeviceAxisClass::Strided),
            bandwidth_boost: plan.config.bandwidth_boost,
            use_bluestein_fft: plan.c2c_device_use_bluestein_fft_override,
            perform_convolution: false,
        };
        let stockham_upload_schedule = if matches!(axis.algorithm, AxisAlgorithm::Stockham { .. })
            && axis.effective_fft_len > 1
        {
            match plan_gpu_double_double_stockham_uploads_for_batches_with_axis_context(
                axis.effective_fft_len,
                plan.config.batch_count,
                device,
                axis_context,
            ) {
                Ok(schedule)
                    if schedule.upload_count > 1
                        && schedule
                            .axis_split
                            .iter()
                            .all(|length| *length <= DD_RECURSIVE_STOCKHAM_LEAF_LIMIT) =>
                {
                    Some(schedule)
                }
                Ok(_) => None,
                Err(VkFftError::UnsupportedKernelPath(_))
                | Err(VkFftError::ResourceLimitExceeded { .. }) => None,
                Err(error) => return Err(error),
            }
        } else {
            None
        };
        let rader_forced_upload_schedule =
            if let AxisAlgorithm::Rader { primes, .. } = &axis.algorithm {
                let fft_rader_primes = primes
                    .iter()
                    .filter(|prime| matches!(prime.mode, RaderMode::FftConvolution { .. }))
                    .map(|prime| prime.prime)
                    .collect::<Vec<_>>();
                let direct_rader_primes = primes
                    .iter()
                    .filter(|prime| matches!(prime.mode, RaderMode::DirectMultiplication))
                    .map(|prime| prime.prime)
                    .collect::<Vec<_>>();
                match plan_gpu_rader_upload_split_with_axis_context(
                    axis.effective_fft_len,
                    &fft_rader_primes,
                    &direct_rader_primes,
                    plan.config.precision,
                    device,
                    axis_context,
                ) {
                    Ok(schedule) => schedule,
                    Err(VkFftError::UnsupportedKernelPath(_))
                    | Err(VkFftError::ResourceLimitExceeded { .. }) => None,
                    Err(error) => return Err(error),
                }
            } else {
                None
            };
        Self::build_with_upload_schedules(
            plan,
            direction,
            stockham_upload_schedule,
            rader_forced_upload_schedule,
            effective_rader_tuning,
            Some(device),
        )?
        .with_device_physical_blocks_with_tuning(
            effective_rader_tuning,
            axis_context.use_bluestein_fft,
            device,
        )?
        .with_two_upload_four_step_plan(device)?
        .with_three_upload_four_step_plan(device)
    }

    fn build_with_upload_schedules(
        plan: &FftPlan,
        direction: Direction,
        stockham_upload_schedule: Option<DoubleDoubleStockhamUploadSchedule>,
        rader_forced_upload_schedule: Option<RaderUploadSchedule>,
        tuning: PlannerTuning,
        device: Option<DeviceProfile>,
    ) -> Result<Self> {
        if plan.config.dimensions.len() != 1
            || plan.config.transform != TransformKind::ComplexToComplex
        {
            return Err(VkFftError::UnsupportedKernelPath(
                "double-double recursive FFT supports one-dimensional C2C plans only",
            ));
        }
        let external_storage = match plan.config.precision {
            Precision::DoubleDouble => PrecisionStorage::DoubleDouble,
            Precision::DoubleDoubleF64Storage => PrecisionStorage::F64,
            other => {
                return Err(VkFftError::UnsupportedPrecision {
                    backend: "double-double recursive FFT IR",
                    precision: precision_name(other),
                });
            }
        };
        let grouped_batch_override = plan.config.grouped_batch_for_axis(0);
        let grouped_batch = grouped_batch_override.unwrap_or(1);
        let axis = plan.axes.first().ok_or(VkFftError::InvalidKernelIr(
            "missing double-double recursive axis plan",
        ))?;
        let mut factors = Vec::new();
        match &axis.algorithm {
            AxisAlgorithm::Stockham { radix } => {
                if let Some(schedule) = &stockham_upload_schedule {
                    schedule.validate()?;
                    if schedule.sequence_len != axis.effective_fft_len
                        || schedule.batch_count != plan.config.batch_count
                        || schedule.upload_count < 2
                    {
                        return Err(VkFftError::InvalidKernelIr(
                            "double-double Stockham upload schedule does not match recursive root",
                        ));
                    }
                    factors.extend(schedule.axis_split.iter().copied().map(|len| FactorSpec {
                        len,
                        kind: FactorKind::Stockham,
                    }));
                } else {
                    push_stockham_chunks(&mut factors, &radix.prime_factors)?;
                }
            }
            AxisAlgorithm::Rader { stockham, primes } => {
                if let Some(schedule) = &rader_forced_upload_schedule {
                    schedule.validate()?;
                    if schedule.sequence_len != axis.effective_fft_len {
                        return Err(VkFftError::InvalidKernelIr(
                            "double-double forced-Rader upload schedule does not match recursive root",
                        ));
                    }
                    let (scheduled_factors, _) = rader_factor_specs_for_upload_split(
                        &stockham.prime_factors,
                        primes,
                        &schedule.axis_split,
                    )?;
                    factors = scheduled_factors;
                } else {
                    push_stockham_chunks(&mut factors, &stockham.prime_factors)?;
                    for prime in primes {
                        for _ in 0..prime.multiplicity {
                            factors.push(FactorSpec {
                                len: prime.prime,
                                kind: FactorKind::Rader(prime.mode.clone()),
                            });
                        }
                    }
                }
            }
            AxisAlgorithm::Bluestein { .. } => {
                return Err(VkFftError::UnsupportedKernelPath(
                    "double-double recursive FFT composes smooth Stockham and Rader factor trees, not Bluestein roots",
                ));
            }
        }
        if factors.len() < 2 {
            return Err(VkFftError::InvalidKernelIr(
                "double-double recursive FFT requires at least two executable factor chunks",
            ));
        }
        let factor_product = checked_factor_product(&factors)?;
        if factor_product != axis.effective_fft_len {
            return Err(VkFftError::InvalidKernelIr(
                "double-double recursive factor chunks do not cover the planned axis",
            ));
        }
        let component_factor_counts = if let Some(schedule) = stockham_upload_schedule.as_ref() {
            Some(vec![1usize; schedule.upload_count])
        } else if let Some(schedule) = rader_forced_upload_schedule.as_ref() {
            let AxisAlgorithm::Rader { stockham, primes } = &axis.algorithm else {
                unreachable!("forced Rader schedule is only created for Rader axes");
            };
            let (_, counts) = rader_factor_specs_for_upload_split(
                &stockham.prime_factors,
                primes,
                &schedule.axis_split,
            )?;
            Some(counts)
        } else {
            None
        };
        let root = build_node(
            &factors,
            plan.config.batch_count,
            grouped_batch,
            component_factor_counts.as_deref(),
            direction,
            plan.config.normalize_inverse,
            tuning,
            external_storage,
            external_storage,
            device,
        )?;
        let ir = Self {
            name: format!(
                "vkfft_dd_recursive_{}_{}",
                axis.effective_fft_len,
                match direction {
                    Direction::Forward => "forward",
                    Direction::Inverse => "inverse",
                }
            ),
            logical_len: axis.effective_fft_len,
            batch_count: plan.config.batch_count,
            grouped_batch,
            grouped_batch_override,
            direction,
            normalize: direction == Direction::Inverse && plan.config.normalize_inverse,
            external_storage,
            zero_pad_pass: plan
                .config
                .zero_padding_for_axis(0)
                .map(|range| {
                    let storage_scalar = match external_storage {
                        PrecisionStorage::DoubleDouble => ScalarType::DoubleDouble,
                        PrecisionStorage::F64 => ScalarType::F64,
                        _ => unreachable!("validated DD recursive external storage"),
                    };
                    ZeroPadPassIr::build_storage_preserving_with_domain(
                        axis.effective_fft_len,
                        plan.config.batch_count,
                        storage_scalar,
                        direction,
                        range,
                        grouped_batch,
                        plan.config.zero_padding_domain,
                    )
                })
                .transpose()?,
            stockham_upload_schedule,
            rader_forced_upload_schedule,
            root,
            two_upload_four_step_plan: None,
            three_upload_four_step_plan: None,
        };
        ir.validate()?;
        Ok(ir)
    }

    pub fn with_device_physical_blocks(self, device: DeviceProfile) -> Result<Self> {
        self.with_device_physical_blocks_with_tuning(PlannerTuning::portable(), false, device)
    }

    fn with_device_physical_blocks_with_tuning(
        mut self,
        tuning: PlannerTuning,
        use_bluestein_fft: bool,
        device: DeviceProfile,
    ) -> Result<Self> {
        let tuning = upstream_effective_rader_tuning(tuning, device, Precision::DoubleDouble);
        let exact_rader_threads = {
            let mut direct_prime_multiplicities = Vec::new();
            let mut fft_primes = Vec::new();
            if self
                .root
                .collect_rader_parent_components(&mut direct_prime_multiplicities, &mut fft_primes)
            {
                if !direct_prime_multiplicities.is_empty() && fft_primes.is_empty() {
                    plan_gpu_double_double_axis0_composite_direct_rader_threads_for_prime_multiplicities(
                        self.logical_len,
                        &direct_prime_multiplicities,
                        self.batch_count,
                        device,
                    )?
                } else if direct_prime_multiplicities.is_empty() && !fft_primes.is_empty() {
                    plan_gpu_double_double_axis0_fft_rader_threads_for_prime_multiplicities_with_tuning(
                        self.logical_len,
                        &fft_primes,
                        self.batch_count,
                        tuning,
                        device,
                    )?
                } else if !direct_prime_multiplicities.is_empty() && !fft_primes.is_empty() {
                    plan_gpu_double_double_axis0_mixed_direct_multi_fft_rader_threads_with_tuning(
                        self.logical_len,
                        &direct_prime_multiplicities,
                        &fft_primes,
                        self.batch_count,
                        tuning,
                        device,
                    )?
                } else {
                    None
                }
            } else {
                None
            }
        };

        self.root = self
            .root
            .with_device_physical_blocks_with_tuning(tuning, device)?;
        if let Some(threads_per_transform) = exact_rader_threads
            && let DoubleDoubleRecursiveFftNodeIr::CooleyTukey(root) = &mut self.root
            && let Some(current) = root.pack_right.axis_batch_block
        {
            // Preserve the already-materialized physical transform grouping/orientation.
            // If replacing only the FFT-lane dimension no longer fits the device, keep
            // the conservative existing block; grouped-pressure parity is a separate
            // scheduler slice rather than silently changing logical ownership here.
            let (local_size_x, local_size_y) = if current.transforms_on_x {
                (current.grouped_batch, threads_per_transform)
            } else {
                (threads_per_transform, current.grouped_batch)
            };
            let exact = StockhamAxisBlockSchedule {
                threads_per_transform,
                grouped_batch: current.grouped_batch,
                transforms_on_x: current.transforms_on_x,
                axis_swapped: current.axis_swapped,
                local_size_x,
                local_size_y,
            };
            if exact.validate(root.batch_count, device).is_ok() {
                root.pack_right = root
                    .pack_right
                    .clone()
                    .with_axis_batch_block(exact, device)?;
                root.twiddle_transpose = root
                    .twiddle_transpose
                    .clone()
                    .with_axis_batch_block(exact, device)?;
                root.scatter_output = root
                    .scatter_output
                    .clone()
                    .with_axis_batch_block(exact, device)?;
                root.validate()?;
            }
        }
        self.apply_stockham_axis0_default_blocks(device)?;
        self.apply_forced_rader_axis0_default_blocks(tuning, use_bluestein_fft, device)?;
        self.validate()?;
        Ok(self)
    }

    fn apply_stockham_axis0_default_blocks(&mut self, device: DeviceProfile) -> Result<()> {
        let Some(schedule) = self.stockham_upload_schedule.clone() else {
            return Ok(());
        };
        if !matches!(schedule.upload_count, 2 | 3) {
            return Ok(());
        }
        let perform_zero_padding = self.zero_pad_pass.is_some();
        let score_component = |node: &DoubleDoubleRecursiveFftNodeIr,
                               axis_upload_id: usize,
                               stage_start_size: usize|
         -> Result<Option<StockhamAxisBlockSchedule>> {
            let DoubleDoubleRecursiveFftNodeIr::Stockham(stockham) = node else {
                return Ok(None);
            };
            let quad =
                plan_gpu_double_double_quad_registers(stockham.sequence_len, stockham.batch_count)?;
            let threads_per_transform = stockham
                .sequence_len
                .div_ceil(quad.min_registers_per_thread)
                .max(1);
            let block = plan_gpu_axis0_four_step_default_block_from_shape(
                schedule.upload_count,
                stockham.sequence_len,
                threads_per_transform,
                FourStepAxisBlockRequest {
                    axis_upload_id,
                    stage_start_size,
                    transform_count: stockham.batch_count,
                    outer_batch_count: self.batch_count,
                    perform_zero_padding,
                    grouped_batch_override: None,
                },
                32,
                device,
            )?;
            if block.is_some() {
                return Ok(block);
            }
            // The shared shape scorer intentionally omits singleton non-Rader blocks
            // because most callers only need a physical-grouping override. A mapped
            // Four-step Stockham leaf must still publish that singleton when pass-local
            // rescoring shrinks a previously wider block: upload 0 keeps FFT lanes on X,
            // while higher uploads keep batches on X and FFT lanes on Y.
            if stockham.axis_batch_block.is_some() {
                let transforms_on_x = axis_upload_id > 0;
                let singleton = StockhamAxisBlockSchedule {
                    threads_per_transform,
                    grouped_batch: 1,
                    transforms_on_x,
                    axis_swapped: false,
                    local_size_x: if transforms_on_x {
                        1
                    } else {
                        threads_per_transform
                    },
                    local_size_y: if transforms_on_x {
                        threads_per_transform
                    } else {
                        1
                    },
                };
                if singleton.validate(stockham.batch_count, device).is_ok() {
                    return Ok(Some(singleton));
                }
            }
            Ok(None)
        };

        match schedule.axis_split.as_slice() {
            [a, _b] if schedule.upload_count == 2 => {
                let (upload1_block, upload0_block) = {
                    let DoubleDoubleRecursiveFftNodeIr::CooleyTukey(root) = &self.root else {
                        return Ok(());
                    };
                    (
                        score_component(&root.right, 1, *a)?,
                        score_component(&root.left, 0, 1)?,
                    )
                };
                let DoubleDoubleRecursiveFftNodeIr::CooleyTukey(root) = &mut self.root else {
                    return Ok(());
                };
                if let Some(block) = upload1_block {
                    root.right.apply_axis0_parent_block(block, device)?;
                }
                if let Some(block) = upload0_block {
                    root.left.apply_axis0_parent_block(block, device)?;
                }
                root.validate()?;
            }
            [a, b, _c] if schedule.upload_count == 3 => {
                let ab = a.checked_mul(*b).ok_or(VkFftError::ArithmeticOverflow {
                    operation: "double-double Stockham automatic A*B stage size",
                })?;
                let (upload2_block, upload1_block, upload0_block) = {
                    let DoubleDoubleRecursiveFftNodeIr::CooleyTukey(root) = &self.root else {
                        return Ok(());
                    };
                    let DoubleDoubleRecursiveFftNodeIr::CooleyTukey(upper) = &root.right else {
                        return Ok(());
                    };
                    (
                        score_component(&upper.right, 2, ab)?,
                        score_component(&upper.left, 1, *a)?,
                        score_component(&root.left, 0, 1)?,
                    )
                };
                let DoubleDoubleRecursiveFftNodeIr::CooleyTukey(root) = &mut self.root else {
                    return Ok(());
                };
                let DoubleDoubleRecursiveFftNodeIr::CooleyTukey(upper) = &mut root.right else {
                    return Ok(());
                };
                if let Some(block) = upload2_block {
                    upper.right.apply_axis0_parent_block(block, device)?;
                }
                if let Some(block) = upload1_block {
                    upper.left.apply_axis0_parent_block(block, device)?;
                }
                if let Some(block) = upload0_block {
                    root.left.apply_axis0_parent_block(block, device)?;
                }
                upper.validate()?;
                root.validate()?;
            }
            _ => {}
        }
        self.validate()?;
        Ok(())
    }

    fn apply_forced_rader_axis0_default_blocks(
        &mut self,
        tuning: PlannerTuning,
        use_bluestein_fft: bool,
        device: DeviceProfile,
    ) -> Result<()> {
        let Some(schedule) = self.rader_forced_upload_schedule.clone() else {
            return Ok(());
        };
        let perform_zero_padding = self.zero_pad_pass.is_some();
        let max_batch_coalesced =
            (upstream_coalesced_memory_bytes_for_precision(device, Precision::DoubleDouble) / 32)
                .max(1);
        let score_component = |node: &DoubleDoubleRecursiveFftNodeIr,
                               axis_upload_id: usize,
                               stage_start_size: usize|
         -> Result<Option<StockhamAxisBlockSchedule>> {
            let request = FourStepAxisBlockRequest {
                axis_upload_id,
                stage_start_size,
                transform_count: node.batch_count(),
                outer_batch_count: self.batch_count,
                perform_zero_padding,
                grouped_batch_override: self.grouped_batch_override,
            };
            // Default splitters deliberately reject a configured groupedBatch because
            // upstream takes the user-grouped branch first. When that branch collapses
            // to a singleton forced-Rader upload, keep a cleared copy for the physical
            // group1 fall-through.
            let default_request = FourStepAxisBlockRequest {
                grouped_batch_override: None,
                ..request
            };
            let pass_max_batch_coalesced = if use_bluestein_fft && axis_upload_id == 0 {
                1
            } else {
                max_batch_coalesced
            };

            let grouped_block = |fft_len: usize,
                                 threads_per_transform: usize|
             -> Result<Option<StockhamAxisBlockSchedule>> {
                if self.grouped_batch_override.is_none() {
                    return Ok(None);
                }
                plan_gpu_axis0_four_step_grouped_block_from_shape_for_precision(
                    schedule.upload_count,
                    fft_len,
                    threads_per_transform,
                    request,
                    Precision::DoubleDouble,
                    32,
                    device,
                )
            };

            if let DoubleDoubleRecursiveFftNodeIr::Stockham(stockham) = node {
                let quad = plan_gpu_double_double_quad_registers(
                    stockham.sequence_len,
                    stockham.batch_count,
                )?;
                let threads_per_transform = stockham
                    .sequence_len
                    .div_ceil(quad.min_registers_per_thread)
                    .max(1);
                if let Some(block) = grouped_block(stockham.sequence_len, threads_per_transform)? {
                    return Ok(Some(block));
                }
                let block = if use_bluestein_fft {
                    plan_gpu_axis0_bluestein_four_step_default_block_from_shape(
                        schedule.upload_count,
                        stockham.sequence_len,
                        threads_per_transform,
                        default_request,
                        32,
                        device,
                    )
                } else {
                    plan_gpu_axis0_four_step_default_block_from_shape(
                        schedule.upload_count,
                        stockham.sequence_len,
                        threads_per_transform,
                        default_request,
                        32,
                        device,
                    )
                };
                return match block {
                    Ok(Some(block)) => Ok(Some(block)),
                    Ok(None) if !use_bluestein_fft => {
                        // The generic non-Rader scorer deliberately suppresses groupedBatch=1.
                        // Inside a forced-Rader upload, however, a Stockham component still owns
                        // a real VkFFTSplitAxisBlock pass and upstream may select that singleton.
                        let transforms_on_x = axis_upload_id > 0;
                        let block = StockhamAxisBlockSchedule {
                            threads_per_transform,
                            grouped_batch: 1,
                            transforms_on_x,
                            axis_swapped: false,
                            local_size_x: if transforms_on_x {
                                1
                            } else {
                                threads_per_transform
                            },
                            local_size_y: if transforms_on_x {
                                threads_per_transform
                            } else {
                                1
                            },
                        };
                        block.validate(node.batch_count(), device)?;
                        Ok(Some(block))
                    }
                    Ok(None) => Ok(None),
                    Err(VkFftError::UnsupportedKernelPath(_)) => Ok(None),
                    Err(error) => Err(error),
                };
            }

            let rader_state = node.exact_rader_parent_split_state_for_forced_upload_with_tuning(
                tuning,
                pass_max_batch_coalesced,
                axis_upload_id,
                device,
            )?;
            if self.grouped_batch_override.is_some() {
                if let Some(state) = rader_state.as_ref() {
                    if let Some(block) =
                        plan_gpu_axis0_rader_four_step_grouped_block_from_split_state_for_precision(
                            schedule.upload_count,
                            node.logical_len(),
                            state,
                            request,
                            Precision::DoubleDouble,
                            32,
                            device,
                        )?
                    {
                        return Ok(Some(block));
                    }
                } else {
                    let threads_per_transform = if let Some(threads) = node
                        .exact_rader_parent_threads_for_forced_upload_with_tuning(
                            tuning,
                            pass_max_batch_coalesced,
                            axis_upload_id,
                            device,
                        )? {
                        threads
                    } else if let Some(threads) = node.axis0_forced_rader_leaf_threads() {
                        threads
                    } else {
                        return Ok(None);
                    };
                    if let Some(block) = grouped_block(node.logical_len(), threads_per_transform)? {
                        return Ok(Some(block));
                    }
                }
                // The user-grouped scorer intentionally suppresses a singleton higher
                // upload. Forced-Rader still launches it, so fall through to the default
                // split-state scorer to materialize upstream group1.
            }

            if let Some(state) = rader_state {
                let block = if use_bluestein_fft {
                    plan_gpu_axis0_bluestein_rader_four_step_default_block_from_split_state(
                        schedule.upload_count,
                        node.logical_len(),
                        &state,
                        default_request,
                        32,
                        device,
                    )
                } else {
                    plan_gpu_axis0_rader_four_step_default_block_from_split_state(
                        schedule.upload_count,
                        node.logical_len(),
                        &state,
                        default_request,
                        32,
                        device,
                    )
                };
                return match block {
                    Ok(block) => Ok(block),
                    Err(VkFftError::UnsupportedKernelPath(_)) => Ok(None),
                    Err(error) => Err(error),
                };
            }
            let threads_per_transform = if let Some(threads) = node
                .exact_rader_parent_threads_for_forced_upload_with_tuning(
                    tuning,
                    pass_max_batch_coalesced,
                    axis_upload_id,
                    device,
                )? {
                threads
            } else if let Some(threads) = node.axis0_forced_rader_leaf_threads() {
                threads
            } else {
                return Ok(None);
            };
            let block = if use_bluestein_fft {
                plan_gpu_axis0_bluestein_four_step_default_block_from_shape(
                    schedule.upload_count,
                    node.logical_len(),
                    threads_per_transform,
                    default_request,
                    32,
                    device,
                )
            } else {
                plan_gpu_axis0_four_step_default_block_from_shape(
                    schedule.upload_count,
                    node.logical_len(),
                    threads_per_transform,
                    default_request,
                    32,
                    device,
                )
            };
            match block {
                Ok(block) => Ok(block),
                Err(VkFftError::UnsupportedKernelPath(_)) => Ok(None),
                Err(error) => Err(error),
            }
        };

        match schedule.axis_split.as_slice() {
            [a, _b] if schedule.upload_count == 2 => {
                let (right_block, left_block) = {
                    let DoubleDoubleRecursiveFftNodeIr::CooleyTukey(root) = &self.root else {
                        return Ok(());
                    };
                    (
                        score_component(&root.right, 1, *a)?,
                        score_component(&root.left, 0, 1)?,
                    )
                };
                let DoubleDoubleRecursiveFftNodeIr::CooleyTukey(root) = &mut self.root else {
                    return Ok(());
                };
                if let Some(block) = right_block {
                    root.right.apply_axis0_parent_block(block, device)?;
                }
                if let Some(block) = left_block {
                    root.left.apply_axis0_parent_block(block, device)?;
                }
                root.validate()?;
            }
            [a, b, _c] if schedule.upload_count == 3 => {
                let ab = a.checked_mul(*b).ok_or(VkFftError::ArithmeticOverflow {
                    operation: "double-double forced-Rader automatic A*B stage size",
                })?;
                let (upload2_block, upload1_block, upload0_block) = {
                    let DoubleDoubleRecursiveFftNodeIr::CooleyTukey(root) = &self.root else {
                        return Ok(());
                    };
                    let DoubleDoubleRecursiveFftNodeIr::CooleyTukey(upper) = &root.right else {
                        return Ok(());
                    };
                    (
                        score_component(&upper.right, 2, ab)?,
                        score_component(&upper.left, 1, *a)?,
                        score_component(&root.left, 0, 1)?,
                    )
                };
                let DoubleDoubleRecursiveFftNodeIr::CooleyTukey(root) = &mut self.root else {
                    return Ok(());
                };
                if let Some(block) = upload0_block {
                    root.left.apply_axis0_parent_block(block, device)?;
                }
                let DoubleDoubleRecursiveFftNodeIr::CooleyTukey(upper) = &mut root.right else {
                    return Ok(());
                };
                if let Some(block) = upload1_block {
                    upper.left.apply_axis0_parent_block(block, device)?;
                }
                if let Some(block) = upload2_block {
                    upper.right.apply_axis0_parent_block(block, device)?;
                }
                upper.validate()?;
                root.validate()?;
            }
            _ => {}
        }
        Ok(())
    }

    pub fn with_other_axis_physical_blocks(
        mut self,
        fastest_axis_len: usize,
        grouped_batch_override: Option<usize>,
        axis1_grouped_batch_override: Option<usize>,
        device: DeviceProfile,
    ) -> Result<Self> {
        if let Some(upload) = self.stockham_upload_schedule.clone() {
            if upload.upload_count == 2 && self.two_upload_four_step_plan.is_some() {
                let DoubleDoubleRecursiveFftNodeIr::CooleyTukey(root) = &self.root else {
                    return Ok(self);
                };
                let (
                    DoubleDoubleRecursiveFftNodeIr::Stockham(left),
                    DoubleDoubleRecursiveFftNodeIr::Stockham(right),
                ) = (&root.left, &root.right)
                else {
                    return Ok(self);
                };
                let left_block = plan_gpu_double_double_other_axis_four_step_upload_block(
                    &upload,
                    0,
                    left.batch_count,
                    fastest_axis_len,
                    grouped_batch_override,
                    axis1_grouped_batch_override,
                    device,
                )?;
                let right_block = plan_gpu_double_double_other_axis_four_step_upload_block(
                    &upload,
                    1,
                    right.batch_count,
                    fastest_axis_len,
                    grouped_batch_override,
                    axis1_grouped_batch_override,
                    device,
                )?;
                if let (Some(left_block), Some(right_block)) = (left_block, right_block) {
                    let DoubleDoubleRecursiveFftNodeIr::CooleyTukey(root) = &mut self.root else {
                        unreachable!("validated two-upload DD root");
                    };
                    let DoubleDoubleRecursiveFftNodeIr::Stockham(left) = &mut root.left else {
                        unreachable!("validated two-upload DD left leaf");
                    };
                    let DoubleDoubleRecursiveFftNodeIr::Stockham(right) = &mut root.right else {
                        unreachable!("validated two-upload DD right leaf");
                    };
                    left.axis_batch_block = Some(left_block);
                    right.axis_batch_block = Some(right_block);
                    left.validate()?;
                    right.validate()?;
                    root.validate()?;
                    if let Some(plan) = self.two_upload_four_step_plan.as_mut() {
                        plan.left_axis_block = left_block;
                        plan.right_axis_block = right_block;
                        plan.validate()?;
                    }
                }
                self.validate()?;
                return Ok(self);
            }

            if upload.upload_count == 3 && self.three_upload_four_step_plan.is_some() {
                let DoubleDoubleRecursiveFftNodeIr::CooleyTukey(root) = &self.root else {
                    return Ok(self);
                };
                let DoubleDoubleRecursiveFftNodeIr::Stockham(low) = &root.left else {
                    return Ok(self);
                };
                let DoubleDoubleRecursiveFftNodeIr::CooleyTukey(upper) = &root.right else {
                    return Ok(self);
                };
                let (
                    DoubleDoubleRecursiveFftNodeIr::Stockham(middle),
                    DoubleDoubleRecursiveFftNodeIr::Stockham(high),
                ) = (&upper.left, &upper.right)
                else {
                    return Ok(self);
                };
                let upload0_block = plan_gpu_double_double_other_axis_four_step_upload_block(
                    &upload,
                    0,
                    low.batch_count,
                    fastest_axis_len,
                    grouped_batch_override,
                    axis1_grouped_batch_override,
                    device,
                )?;
                let upload1_block = plan_gpu_double_double_other_axis_four_step_upload_block(
                    &upload,
                    1,
                    middle.batch_count,
                    fastest_axis_len,
                    grouped_batch_override,
                    axis1_grouped_batch_override,
                    device,
                )?;
                let upload2_block = plan_gpu_double_double_other_axis_four_step_upload_block(
                    &upload,
                    2,
                    high.batch_count,
                    fastest_axis_len,
                    grouped_batch_override,
                    axis1_grouped_batch_override,
                    device,
                )?;
                if let (Some(upload0_block), Some(upload1_block), Some(upload2_block)) =
                    (upload0_block, upload1_block, upload2_block)
                {
                    let DoubleDoubleRecursiveFftNodeIr::CooleyTukey(root) = &mut self.root else {
                        unreachable!("validated three-upload DD root");
                    };
                    let DoubleDoubleRecursiveFftNodeIr::Stockham(low) = &mut root.left else {
                        unreachable!("validated three-upload DD low leaf");
                    };
                    let DoubleDoubleRecursiveFftNodeIr::CooleyTukey(upper) = &mut root.right else {
                        unreachable!("validated three-upload DD upper node");
                    };
                    let DoubleDoubleRecursiveFftNodeIr::Stockham(middle) = &mut upper.left else {
                        unreachable!("validated three-upload DD middle leaf");
                    };
                    let DoubleDoubleRecursiveFftNodeIr::Stockham(high) = &mut upper.right else {
                        unreachable!("validated three-upload DD high leaf");
                    };
                    low.axis_batch_block = Some(upload0_block);
                    middle.axis_batch_block = Some(upload1_block);
                    high.axis_batch_block = Some(upload2_block);
                    low.validate()?;
                    middle.validate()?;
                    high.validate()?;
                    upper.validate()?;
                    root.validate()?;
                    if let Some(plan) = self.three_upload_four_step_plan.as_mut() {
                        plan.upload0_axis_block = upload0_block;
                        plan.upload1_axis_block = upload1_block;
                        plan.upload2_axis_block = upload2_block;
                        plan.validate()?;
                    }
                }
                self.validate()?;
                return Ok(self);
            }
        }

        if let Some(schedule) = self.rader_forced_upload_schedule.as_ref()
            && schedule.upload_count == 2
            && schedule.axis_split.len() == 2
        {
            let DoubleDoubleRecursiveFftNodeIr::CooleyTukey(root) = &mut self.root else {
                return Ok(self);
            };
            let parent_threads = root
                .pack_right
                .axis_batch_block
                .map_or(root.pack_right.workgroup_size.x as usize, |block| {
                    block.threads_per_transform
                })
                .max(1);
            if let Some(block) = plan_gpu_other_axis_composite_rader_batch_block(
                root.logical_len,
                root.batch_count,
                fastest_axis_len,
                32,
                parent_threads,
                grouped_batch_override,
                axis1_grouped_batch_override,
                device,
            )? {
                root.pack_right = root
                    .pack_right
                    .clone()
                    .with_axis_batch_block(block, device)?;
                root.twiddle_transpose = root
                    .twiddle_transpose
                    .clone()
                    .with_axis_batch_block(block, device)?;
                root.scatter_output = root
                    .scatter_output
                    .clone()
                    .with_axis_batch_block(block, device)?;
            }
            root.right = root
                .right
                .clone()
                .with_other_axis_forced_upload_component_block(
                    2,
                    1,
                    fastest_axis_len,
                    grouped_batch_override,
                    axis1_grouped_batch_override,
                    device,
                )?;
            root.left = root
                .left
                .clone()
                .with_other_axis_forced_upload_component_block(
                    2,
                    0,
                    fastest_axis_len,
                    grouped_batch_override,
                    axis1_grouped_batch_override,
                    device,
                )?;
            root.validate()?;
            self.validate()?;
            return Ok(self);
        }

        if let Some(schedule) = self.rader_forced_upload_schedule.as_ref()
            && schedule.upload_count == 3
            && schedule.axis_split.len() == 3
        {
            let [low_len, middle_len, high_len]: [usize; 3] =
                schedule.axis_split.as_slice().try_into().map_err(|_| {
                    VkFftError::InvalidKernelIr(
                        "double-double forced-Rader three-upload split must contain three components",
                    )
                })?;
            let DoubleDoubleRecursiveFftNodeIr::CooleyTukey(root) = &mut self.root else {
                return Ok(self);
            };
            let parent_threads = root
                .pack_right
                .axis_batch_block
                .map_or(root.pack_right.workgroup_size.x as usize, |block| {
                    block.threads_per_transform
                })
                .max(1);
            if let Some(block) = plan_gpu_other_axis_composite_rader_batch_block(
                root.logical_len,
                root.batch_count,
                fastest_axis_len,
                32,
                parent_threads,
                grouped_batch_override,
                axis1_grouped_batch_override,
                device,
            )? {
                root.pack_right = root
                    .pack_right
                    .clone()
                    .with_axis_batch_block(block, device)?;
                root.twiddle_transpose = root
                    .twiddle_transpose
                    .clone()
                    .with_axis_batch_block(block, device)?;
                root.scatter_output = root
                    .scatter_output
                    .clone()
                    .with_axis_batch_block(block, device)?;
            }
            let DoubleDoubleRecursiveFftNodeIr::CooleyTukey(upper) = &mut root.right else {
                return Ok(self);
            };
            if root.left.logical_len() != low_len
                || upper.left.logical_len() != middle_len
                || upper.right.logical_len() != high_len
            {
                return Ok(self);
            }
            upper.right = upper
                .right
                .clone()
                .with_other_axis_forced_upload_component_block(
                    3,
                    2,
                    fastest_axis_len,
                    grouped_batch_override,
                    axis1_grouped_batch_override,
                    device,
                )?;
            upper.left = upper
                .left
                .clone()
                .with_other_axis_forced_upload_component_block(
                    3,
                    1,
                    fastest_axis_len,
                    grouped_batch_override,
                    axis1_grouped_batch_override,
                    device,
                )?;
            root.left = root
                .left
                .clone()
                .with_other_axis_forced_upload_component_block(
                    3,
                    0,
                    fastest_axis_len,
                    grouped_batch_override,
                    axis1_grouped_batch_override,
                    device,
                )?;
            upper.validate()?;
            root.validate()?;
            self.validate()?;
            return Ok(self);
        }

        if self.root.contains_rader() {
            self.root = self.root.with_other_axis_component_block(
                fastest_axis_len,
                grouped_batch_override,
                axis1_grouped_batch_override,
                device,
            )?;
        }
        self.validate()?;
        Ok(self)
    }

    fn with_two_upload_four_step_plan(mut self, device: DeviceProfile) -> Result<Self> {
        let Some(schedule) = self.stockham_upload_schedule.as_ref() else {
            return Ok(self);
        };
        if schedule.upload_count != 2 || schedule.axis_split.len() != 2 {
            return Ok(self);
        }
        let [left_len, right_len]: [usize; 2] =
            schedule.axis_split.as_slice().try_into().map_err(|_| {
                VkFftError::InvalidKernelIr(
                    "double-double two-upload schedule must contain two axis factors",
                )
            })?;
        if left_len > DD_TWO_UPLOAD_FOUR_STEP_LEAF_LIMIT
            || right_len > DD_TWO_UPLOAD_FOUR_STEP_LEAF_LIMIT
        {
            return Ok(self);
        }
        let DoubleDoubleRecursiveFftNodeIr::CooleyTukey(root) = &self.root else {
            return Ok(self);
        };
        let (
            DoubleDoubleRecursiveFftNodeIr::Stockham(left),
            DoubleDoubleRecursiveFftNodeIr::Stockham(right),
        ) = (&root.left, &root.right)
        else {
            return Ok(self);
        };
        if left.sequence_len != left_len || right.sequence_len != right_len {
            return Ok(self);
        }
        let (Some(left_axis_block), Some(right_axis_block)) =
            (left.axis_batch_block, right.axis_batch_block)
        else {
            return Ok(self);
        };
        let right_shared = right
            .sequence_len
            .checked_mul(right_axis_block.grouped_batch)
            .and_then(|value| value.checked_mul(32))
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "double-double Four-step right shared-memory bytes",
            })?;
        let left_shared = left
            .sequence_len
            .checked_mul(left_axis_block.grouped_batch)
            .and_then(|value| value.checked_mul(32))
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "double-double Four-step left shared-memory bytes",
            })?;
        if right_shared > device.shared_memory_bytes || left_shared > device.shared_memory_bytes {
            return Ok(self);
        }
        let plan = DoubleDoubleTwoUploadFourStepPlanIr {
            logical_len: self.logical_len,
            batch_count: self.batch_count,
            left_len,
            right_len,
            right_axis_block,
            left_axis_block,
        };
        plan.validate()?;
        self.two_upload_four_step_plan = Some(plan);
        self.validate()?;
        Ok(self)
    }

    fn with_three_upload_four_step_plan(mut self, device: DeviceProfile) -> Result<Self> {
        let Some(schedule) = self.stockham_upload_schedule.as_ref() else {
            return Ok(self);
        };
        let [a, b, c]: [usize; 3] = match schedule.axis_split.as_slice().try_into() {
            Ok(split) if schedule.upload_count == 3 => split,
            _ => return Ok(self),
        };
        if [a, b, c]
            .iter()
            .any(|length| *length > DD_THREE_UPLOAD_FOUR_STEP_LEAF_LIMIT)
        {
            return Ok(self);
        }
        let DoubleDoubleRecursiveFftNodeIr::CooleyTukey(root) = &self.root else {
            return Ok(self);
        };
        let DoubleDoubleRecursiveFftNodeIr::Stockham(low) = &root.left else {
            return Ok(self);
        };
        let DoubleDoubleRecursiveFftNodeIr::CooleyTukey(upper) = &root.right else {
            return Ok(self);
        };
        let (
            DoubleDoubleRecursiveFftNodeIr::Stockham(middle),
            DoubleDoubleRecursiveFftNodeIr::Stockham(high),
        ) = (&upper.left, &upper.right)
        else {
            return Ok(self);
        };
        if low.sequence_len != a || middle.sequence_len != b || high.sequence_len != c {
            return Ok(self);
        }
        let (Some(upload0_axis_block), Some(upload1_axis_block), Some(upload2_axis_block)) = (
            low.axis_batch_block,
            middle.axis_batch_block,
            high.axis_batch_block,
        ) else {
            return Ok(self);
        };
        let upload2_shared = high
            .sequence_len
            .checked_mul(upload2_axis_block.grouped_batch)
            .and_then(|value| value.checked_mul(32))
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "double-double three-upload high shared-memory bytes",
            })?;
        let upload1_shared = middle
            .sequence_len
            .checked_mul(upload1_axis_block.grouped_batch)
            .and_then(|value| value.checked_mul(32))
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "double-double three-upload middle shared-memory bytes",
            })?;
        let upload0_shared = low
            .sequence_len
            .checked_mul(upload0_axis_block.grouped_batch)
            .and_then(|value| value.checked_mul(32))
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "double-double three-upload low shared-memory bytes",
            })?;
        if upload2_shared > device.shared_memory_bytes
            || upload1_shared > device.shared_memory_bytes
            || upload0_shared > device.shared_memory_bytes
        {
            return Ok(self);
        }
        let plan = DoubleDoubleThreeUploadFourStepPlanIr {
            logical_len: self.logical_len,
            batch_count: self.batch_count,
            axis_split: [a, b, c],
            upload2_axis_block,
            upload1_axis_block,
            upload0_axis_block,
        };
        plan.validate()?;
        self.three_upload_four_step_plan = Some(plan);
        self.validate()?;
        Ok(self)
    }

    pub(crate) fn forced_rader_two_upload_mapped_high_component(
        &self,
    ) -> Result<Option<DoubleDoubleRecursiveFftNodeIr>> {
        if self.stockham_upload_schedule.is_some()
            || self.two_upload_four_step_plan.is_some()
            || self.three_upload_four_step_plan.is_some()
        {
            return Ok(None);
        }
        let Some(schedule) = self.rader_forced_upload_schedule.as_ref() else {
            return Ok(None);
        };
        let [left_len, right_len]: [usize; 2] = match schedule.axis_split.as_slice().try_into() {
            Ok(split) if schedule.upload_count == 2 => split,
            _ => return Ok(None),
        };
        let DoubleDoubleRecursiveFftNodeIr::CooleyTukey(root) = &self.root else {
            return Ok(None);
        };
        if root.left_len != left_len || root.right_len != right_len {
            return Ok(None);
        }
        let mapping = FourStepMapping {
            logical_len: self.logical_len,
            left_len,
            right_len,
            outer_batch_count: self.batch_count,
        };
        mapping.validate()?;
        let mapped = match &root.right {
            DoubleDoubleRecursiveFftNodeIr::CooleyTukey(high) => {
                let mut mapped = (**high).clone();
                mapped.pack_right = mapped
                    .pack_right
                    .with_external_input_storage(self.external_storage)?
                    .with_four_step_input_modifier(
                        DoubleDoubleCooleyTukeyInputModifier::FourStepRight(mapping),
                    )?;
                mapped.scatter_output = mapped.scatter_output.with_four_step_output_modifier(
                    DoubleDoubleCooleyTukeyOutputModifier::FourStepRight(mapping),
                )?;
                mapped.validate()?;
                DoubleDoubleRecursiveFftNodeIr::CooleyTukey(Box::new(mapped))
            }
            DoubleDoubleRecursiveFftNodeIr::DirectRader(high) if high.prime == right_len => {
                DoubleDoubleRecursiveFftNodeIr::DirectRader(Box::new(
                    high.as_ref().clone().with_four_step_io_mapping(
                        StockhamIoMapping::FourStepRight(mapping),
                        self.external_storage,
                        PrecisionStorage::DoubleDouble,
                    )?,
                ))
            }
            DoubleDoubleRecursiveFftNodeIr::FftRader(high) if high.prime == right_len => {
                DoubleDoubleRecursiveFftNodeIr::FftRader(Box::new(
                    high.as_ref().clone().with_four_step_io_mapping(
                        StockhamIoMapping::FourStepRight(mapping),
                        self.external_storage,
                        PrecisionStorage::DoubleDouble,
                    )?,
                ))
            }
            _ => return Ok(None),
        };
        Ok(Some(mapped))
    }

    pub(crate) fn forced_rader_two_upload_mapped_high_stockham(
        &self,
    ) -> Result<Option<(&DoubleDoubleStockhamIr, FourStepMapping)>> {
        if self.stockham_upload_schedule.is_some()
            || self.two_upload_four_step_plan.is_some()
            || self.three_upload_four_step_plan.is_some()
        {
            return Ok(None);
        }
        let Some(schedule) = self.rader_forced_upload_schedule.as_ref() else {
            return Ok(None);
        };
        let [left_len, right_len]: [usize; 2] = match schedule.axis_split.as_slice().try_into() {
            Ok(split) if schedule.upload_count == 2 => split,
            _ => return Ok(None),
        };
        let DoubleDoubleRecursiveFftNodeIr::CooleyTukey(root) = &self.root else {
            return Ok(None);
        };
        let DoubleDoubleRecursiveFftNodeIr::Stockham(high) = &root.right else {
            return Ok(None);
        };
        let expected_batch =
            self.batch_count
                .checked_mul(left_len)
                .ok_or(VkFftError::ArithmeticOverflow {
                    operation: "double-double forced-Rader high Stockham batch count",
                })?;
        if root.left_len != left_len
            || root.right_len != right_len
            || high.sequence_len != right_len
            || high.batch_count != expected_batch
            || high.axis_batch_block.is_none()
        {
            return Ok(None);
        }
        let mapping = FourStepMapping {
            logical_len: self.logical_len,
            left_len,
            right_len,
            outer_batch_count: self.batch_count,
        };
        mapping.validate()?;
        Ok(Some((high, mapping)))
    }

    pub(crate) fn forced_rader_two_upload_mapped_low_stockham(
        &self,
    ) -> Result<Option<(&DoubleDoubleStockhamIr, FourStepMapping)>> {
        if self.two_upload_four_step_plan.is_some() || self.three_upload_four_step_plan.is_some() {
            return Ok(None);
        }
        let Some(schedule) = self.rader_forced_upload_schedule.as_ref() else {
            return Ok(None);
        };
        let [left_len, right_len]: [usize; 2] = match schedule.axis_split.as_slice().try_into() {
            Ok(split) if schedule.upload_count == 2 => split,
            _ => return Ok(None),
        };
        let DoubleDoubleRecursiveFftNodeIr::CooleyTukey(root) = &self.root else {
            return Ok(None);
        };
        let DoubleDoubleRecursiveFftNodeIr::Stockham(low) = &root.left else {
            return Ok(None);
        };
        if root.left_len != left_len
            || root.right_len != right_len
            || low.sequence_len != left_len
            || (low.sequence_len <= 64 && low.axis_batch_block.is_none())
            || low.batch_count
                != self.batch_count.checked_mul(right_len).ok_or(
                    VkFftError::ArithmeticOverflow {
                        operation: "double-double forced-Rader low upload batch count",
                    },
                )?
        {
            return Ok(None);
        }
        let mapping = FourStepMapping {
            logical_len: self.logical_len,
            left_len,
            right_len,
            outer_batch_count: self.batch_count,
        };
        mapping.validate()?;
        Ok(Some((low, mapping)))
    }

    pub(crate) fn forced_rader_two_upload_mapped_low_component(
        &self,
    ) -> Result<Option<DoubleDoubleRecursiveFftNodeIr>> {
        if self.two_upload_four_step_plan.is_some() || self.three_upload_four_step_plan.is_some() {
            return Ok(None);
        }
        let Some(schedule) = self.rader_forced_upload_schedule.as_ref() else {
            return Ok(None);
        };
        let [left_len, right_len]: [usize; 2] = match schedule.axis_split.as_slice().try_into() {
            Ok(split) if schedule.upload_count == 2 => split,
            _ => return Ok(None),
        };
        let DoubleDoubleRecursiveFftNodeIr::CooleyTukey(root) = &self.root else {
            return Ok(None);
        };
        if root.left_len != left_len || root.right_len != right_len {
            return Ok(None);
        }
        let expected_batch =
            self.batch_count
                .checked_mul(right_len)
                .ok_or(VkFftError::ArithmeticOverflow {
                    operation: "double-double forced-Rader mapped low component batch count",
                })?;
        if root.left.batch_count() != expected_batch {
            return Ok(None);
        }
        let mapping = FourStepMapping {
            logical_len: self.logical_len,
            left_len,
            right_len,
            outer_batch_count: self.batch_count,
        };
        mapping.validate()?;
        let mapped = match &root.left {
            DoubleDoubleRecursiveFftNodeIr::CooleyTukey(low) => {
                let mut mapped = (**low).clone();
                mapped.scatter_output = mapped
                    .scatter_output
                    .with_external_output_storage(self.external_storage)?
                    .with_four_step_output_modifier(
                        DoubleDoubleCooleyTukeyOutputModifier::FourStepLeft(mapping),
                    )?;
                mapped.validate()?;
                DoubleDoubleRecursiveFftNodeIr::CooleyTukey(Box::new(mapped))
            }
            DoubleDoubleRecursiveFftNodeIr::DirectRader(low) if low.prime == left_len => {
                DoubleDoubleRecursiveFftNodeIr::DirectRader(Box::new(
                    low.as_ref().clone().with_four_step_io_mapping(
                        StockhamIoMapping::FourStepLeft(mapping),
                        PrecisionStorage::DoubleDouble,
                        self.external_storage,
                    )?,
                ))
            }
            DoubleDoubleRecursiveFftNodeIr::FftRader(low) if low.prime == left_len => {
                DoubleDoubleRecursiveFftNodeIr::FftRader(Box::new(
                    low.as_ref().clone().with_four_step_io_mapping(
                        StockhamIoMapping::FourStepLeft(mapping),
                        PrecisionStorage::DoubleDouble,
                        self.external_storage,
                    )?,
                ))
            }
            _ => return Ok(None),
        };
        Ok(Some(mapped))
    }

    pub(crate) fn forced_rader_three_upload_mapped_components(
        &self,
    ) -> Result<Option<Vec<DoubleDoubleForcedRaderThreeUploadComponentIr>>> {
        if self.stockham_upload_schedule.is_some()
            || self.two_upload_four_step_plan.is_some()
            || self.three_upload_four_step_plan.is_some()
        {
            return Ok(None);
        }
        let Some(schedule) = self.rader_forced_upload_schedule.as_ref() else {
            return Ok(None);
        };
        let [a, b, c]: [usize; 3] = match schedule.axis_split.as_slice().try_into() {
            Ok(split) if schedule.upload_count == 3 => split,
            _ => return Ok(None),
        };
        let DoubleDoubleRecursiveFftNodeIr::CooleyTukey(root) = &self.root else {
            return Ok(None);
        };
        let DoubleDoubleRecursiveFftNodeIr::CooleyTukey(upper) = &root.right else {
            return Ok(None);
        };
        if root.left_len != a
            || root.right_len
                != b.checked_mul(c).ok_or(VkFftError::ArithmeticOverflow {
                    operation: "double-double forced-Rader three-upload upper length",
                })?
            || upper.left_len != b
            || upper.right_len != c
        {
            return Ok(None);
        }
        let mapping = ThreeUploadFourStepMapping {
            logical_len: self.logical_len,
            axis_split: [a, b, c],
            outer_batch_count: self.batch_count,
        };
        mapping.validate()?;

        let map_component =
            |node: &DoubleDoubleRecursiveFftNodeIr,
             upload_id: usize|
             -> Result<Option<DoubleDoubleForcedRaderThreeUploadComponentIr>> {
                let expected_len = match upload_id {
                    2 => c,
                    1 => b,
                    0 => a,
                    _ => return Ok(None),
                };
                if node.logical_len() != expected_len {
                    return Ok(None);
                }
                match node {
                    DoubleDoubleRecursiveFftNodeIr::Stockham(stockham) => Ok(Some(
                        DoubleDoubleForcedRaderThreeUploadComponentIr::Stockham {
                            upload_id,
                            ir: stockham.as_ref().clone(),
                        },
                    )),
                    DoubleDoubleRecursiveFftNodeIr::CooleyTukey(cooley) => {
                        let mut mapped = cooley.as_ref().clone();
                        match upload_id {
                            2 => {
                                mapped.pack_right = mapped
                                    .pack_right
                                    .with_external_input_storage(self.external_storage)?
                                    .with_four_step_input_modifier(
                                        DoubleDoubleCooleyTukeyInputModifier::FourStepThreeUpload2(
                                            mapping,
                                        ),
                                    )?;
                                mapped.scatter_output =
                                    mapped.scatter_output.with_four_step_output_modifier(
                                        DoubleDoubleCooleyTukeyOutputModifier::FourStepThreeUpload2(
                                            mapping,
                                        ),
                                    )?;
                            }
                            1 => {
                                mapped.scatter_output =
                                    mapped.scatter_output.with_four_step_output_modifier(
                                        DoubleDoubleCooleyTukeyOutputModifier::FourStepThreeUpload1(
                                            mapping,
                                        ),
                                    )?;
                            }
                            0 => {
                                mapped.scatter_output = mapped
                                    .scatter_output
                                    .with_external_output_storage(self.external_storage)?
                                    .with_four_step_output_modifier(
                                        DoubleDoubleCooleyTukeyOutputModifier::FourStepThreeUpload0(
                                            mapping,
                                        ),
                                    )?;
                            }
                            _ => return Ok(None),
                        }
                        mapped.validate()?;
                        Ok(Some(
                            DoubleDoubleForcedRaderThreeUploadComponentIr::Recursive {
                                upload_id,
                                ir: DoubleDoubleRecursiveFftNodeIr::CooleyTukey(Box::new(mapped)),
                            },
                        ))
                    }
                    DoubleDoubleRecursiveFftNodeIr::DirectRader(direct) => {
                        let (io_mapping, input_storage, output_storage) = match upload_id {
                            2 => (
                                StockhamIoMapping::FourStepThreeUpload2(mapping),
                                self.external_storage,
                                PrecisionStorage::DoubleDouble,
                            ),
                            1 => (
                                StockhamIoMapping::FourStepThreeUpload1(mapping),
                                PrecisionStorage::DoubleDouble,
                                PrecisionStorage::DoubleDouble,
                            ),
                            0 => (
                                StockhamIoMapping::FourStepThreeUpload0(mapping),
                                PrecisionStorage::DoubleDouble,
                                self.external_storage,
                            ),
                            _ => return Ok(None),
                        };
                        let mapped = direct.as_ref().clone().with_four_step_io_mapping(
                            io_mapping,
                            input_storage,
                            output_storage,
                        )?;
                        Ok(Some(
                            DoubleDoubleForcedRaderThreeUploadComponentIr::Recursive {
                                upload_id,
                                ir: DoubleDoubleRecursiveFftNodeIr::DirectRader(Box::new(mapped)),
                            },
                        ))
                    }
                    DoubleDoubleRecursiveFftNodeIr::FftRader(rader) => {
                        let (io_mapping, input_storage, output_storage) = match upload_id {
                            2 => (
                                StockhamIoMapping::FourStepThreeUpload2(mapping),
                                self.external_storage,
                                PrecisionStorage::DoubleDouble,
                            ),
                            1 => (
                                StockhamIoMapping::FourStepThreeUpload1(mapping),
                                PrecisionStorage::DoubleDouble,
                                PrecisionStorage::DoubleDouble,
                            ),
                            0 => (
                                StockhamIoMapping::FourStepThreeUpload0(mapping),
                                PrecisionStorage::DoubleDouble,
                                self.external_storage,
                            ),
                            _ => return Ok(None),
                        };
                        let mapped = rader.as_ref().clone().with_four_step_io_mapping(
                            io_mapping,
                            input_storage,
                            output_storage,
                        )?;
                        Ok(Some(
                            DoubleDoubleForcedRaderThreeUploadComponentIr::Recursive {
                                upload_id,
                                ir: DoubleDoubleRecursiveFftNodeIr::FftRader(Box::new(mapped)),
                            },
                        ))
                    }
                    _ => Ok(None),
                }
            };

        let Some(high) = map_component(&upper.right, 2)? else {
            return Ok(None);
        };
        let Some(middle) = map_component(&upper.left, 1)? else {
            return Ok(None);
        };
        let Some(low) = map_component(&root.left, 0)? else {
            return Ok(None);
        };
        Ok(Some(vec![high, middle, low]))
    }

    pub fn validate(&self) -> Result<()> {
        if self.logical_len == 0
            || self.batch_count == 0
            || self.grouped_batch == 0
            || self.grouped_batch_override == Some(0)
            || self
                .grouped_batch_override
                .is_some_and(|grouped| grouped != self.grouped_batch)
        {
            return Err(VkFftError::InvalidKernelIr(
                "double-double recursive FFT dimensions/grouped ownership must be consistent",
            ));
        }
        if !matches!(
            self.external_storage,
            PrecisionStorage::DoubleDouble | PrecisionStorage::F64
        ) {
            return Err(VkFftError::InvalidKernelIr(
                "double-double recursive FFT external storage must be DD or F64",
            ));
        }
        if let Some(pass) = &self.zero_pad_pass {
            pass.validate()?;
            let storage_scalar = match self.external_storage {
                PrecisionStorage::DoubleDouble => ScalarType::DoubleDouble,
                PrecisionStorage::F64 => ScalarType::F64,
                _ => unreachable!("validated DD recursive external storage"),
            };
            if pass.logical_len != self.logical_len
                || pass.batch_count != self.batch_count
                || pass.grouped_batch != self.grouped_batch
                || pass.direction != self.direction
                || pass.scalar != storage_scalar
                || pass.input_storage_scalar != storage_scalar
                || pass.output_storage_scalar != storage_scalar
            {
                return Err(VkFftError::InvalidKernelIr(
                    "double-double recursive zero-pad boundary does not match its root",
                ));
            }
        }
        self.root.validate()?;
        if self.normalize != (self.direction == Direction::Inverse && node_normalizes(&self.root)) {
            return Err(VkFftError::InvalidKernelIr(
                "double-double recursive inverse normalization metadata is inconsistent",
            ));
        }
        if self.root.logical_len() != self.logical_len
            || self.root.batch_count() != self.batch_count
            || self.root.grouped_batch() != self.grouped_batch
            || self.root.logical_batch_group_count()
                != self.batch_count.div_ceil(self.grouped_batch)
            || self.root.direction() != self.direction
            || self.root.input_storage() != self.external_storage
            || self.root.output_storage() != self.external_storage
            || !matches!(self.root, DoubleDoubleRecursiveFftNodeIr::CooleyTukey(_))
        {
            return Err(VkFftError::InvalidKernelIr(
                "double-double recursive FFT root metadata is inconsistent",
            ));
        }
        if let Some(schedule) = &self.stockham_upload_schedule {
            schedule.validate()?;
            if schedule.sequence_len != self.logical_len
                || schedule.batch_count != self.batch_count
                || schedule.upload_count < 2
            {
                return Err(VkFftError::InvalidKernelIr(
                    "double-double recursive upload schedule does not match root dimensions",
                ));
            }
            let mut leaf_lengths = Vec::new();
            collect_stockham_leaf_lengths(&self.root, &mut leaf_lengths)?;
            if leaf_lengths != schedule.axis_split {
                return Err(VkFftError::InvalidKernelIr(
                    "double-double recursive Stockham leaves do not match upload schedule",
                ));
            }
        }
        if let Some(schedule) = &self.rader_forced_upload_schedule {
            schedule.validate()?;
            if schedule.sequence_len != self.logical_len
                || !rader_upload_split_matches_root(&self.root, &schedule.axis_split)
            {
                return Err(VkFftError::InvalidKernelIr(
                    "double-double forced-Rader upload schedule does not match recursive root cuts",
                ));
            }
        }
        if self.stockham_upload_schedule.is_some() && self.rader_forced_upload_schedule.is_some() {
            return Err(VkFftError::InvalidKernelIr(
                "double-double recursive root cannot own both Stockham and forced-Rader upload schedules",
            ));
        }
        if let Some(plan) = self.two_upload_four_step_plan {
            plan.validate()?;
            let Some(schedule) = self.stockham_upload_schedule.as_ref() else {
                return Err(VkFftError::InvalidKernelIr(
                    "double-double Four-step plan requires upload scheduler metadata",
                ));
            };
            if schedule.upload_count != 2
                || schedule.axis_split.as_slice() != [plan.left_len, plan.right_len]
                || plan.logical_len != self.logical_len
                || plan.batch_count != self.batch_count
            {
                return Err(VkFftError::InvalidKernelIr(
                    "double-double Four-step plan does not match upload scheduler metadata",
                ));
            }
            let DoubleDoubleRecursiveFftNodeIr::CooleyTukey(root) = &self.root else {
                return Err(VkFftError::InvalidKernelIr(
                    "double-double Four-step plan requires a Cooley-Tukey root",
                ));
            };
            let (
                DoubleDoubleRecursiveFftNodeIr::Stockham(left),
                DoubleDoubleRecursiveFftNodeIr::Stockham(right),
            ) = (&root.left, &root.right)
            else {
                return Err(VkFftError::InvalidKernelIr(
                    "double-double two-upload Four-step requires Stockham upload leaves",
                ));
            };
            if left.sequence_len != plan.left_len
                || right.sequence_len != plan.right_len
                || left.axis_batch_block != Some(plan.left_axis_block)
                || right.axis_batch_block != Some(plan.right_axis_block)
            {
                return Err(VkFftError::InvalidKernelIr(
                    "double-double Four-step leaf geometry does not match the executable plan",
                ));
            }
        }
        if self.two_upload_four_step_plan.is_some() && self.three_upload_four_step_plan.is_some() {
            return Err(VkFftError::InvalidKernelIr(
                "double-double recursive root cannot own both two- and three-upload Four-step plans",
            ));
        }
        if let Some(plan) = self.three_upload_four_step_plan {
            plan.validate()?;
            let Some(schedule) = self.stockham_upload_schedule.as_ref() else {
                return Err(VkFftError::InvalidKernelIr(
                    "double-double three-upload Four-step requires scheduler metadata",
                ));
            };
            if schedule.upload_count != 3
                || schedule.axis_split.as_slice() != plan.axis_split
                || plan.logical_len != self.logical_len
                || plan.batch_count != self.batch_count
            {
                return Err(VkFftError::InvalidKernelIr(
                    "double-double three-upload Four-step plan does not match scheduler metadata",
                ));
            }
            let DoubleDoubleRecursiveFftNodeIr::CooleyTukey(root) = &self.root else {
                return Err(VkFftError::InvalidKernelIr(
                    "double-double three-upload Four-step requires a Cooley-Tukey root",
                ));
            };
            let DoubleDoubleRecursiveFftNodeIr::Stockham(low) = &root.left else {
                return Err(VkFftError::InvalidKernelIr(
                    "double-double three-upload Four-step is missing its low Stockham leaf",
                ));
            };
            let DoubleDoubleRecursiveFftNodeIr::CooleyTukey(upper) = &root.right else {
                return Err(VkFftError::InvalidKernelIr(
                    "double-double three-upload Four-step is missing its upper Cooley node",
                ));
            };
            let (
                DoubleDoubleRecursiveFftNodeIr::Stockham(middle),
                DoubleDoubleRecursiveFftNodeIr::Stockham(high),
            ) = (&upper.left, &upper.right)
            else {
                return Err(VkFftError::InvalidKernelIr(
                    "double-double three-upload Four-step is missing middle/high Stockham leaves",
                ));
            };
            let [a, b, c] = plan.axis_split;
            if low.sequence_len != a
                || middle.sequence_len != b
                || high.sequence_len != c
                || low.axis_batch_block != Some(plan.upload0_axis_block)
                || middle.axis_batch_block != Some(plan.upload1_axis_block)
                || high.axis_batch_block != Some(plan.upload2_axis_block)
            {
                return Err(VkFftError::InvalidKernelIr(
                    "double-double three-upload Four-step leaf geometry is inconsistent",
                ));
            }
        }
        Ok(())
    }
}

fn collect_stockham_leaf_lengths(
    node: &DoubleDoubleRecursiveFftNodeIr,
    output: &mut Vec<usize>,
) -> Result<()> {
    match node {
        DoubleDoubleRecursiveFftNodeIr::Stockham(ir) => output.push(ir.sequence_len),
        DoubleDoubleRecursiveFftNodeIr::CooleyTukey(ir) => {
            collect_stockham_leaf_lengths(&ir.left, output)?;
            collect_stockham_leaf_lengths(&ir.right, output)?;
        }
        _ => {
            return Err(VkFftError::InvalidKernelIr(
                "double-double Stockham upload schedule contains a non-Stockham leaf",
            ));
        }
    }
    Ok(())
}

fn node_normalizes(node: &DoubleDoubleRecursiveFftNodeIr) -> bool {
    match node {
        DoubleDoubleRecursiveFftNodeIr::Stockham(ir) => ir.normalize,
        DoubleDoubleRecursiveFftNodeIr::DirectRader(ir) => ir.normalize,
        DoubleDoubleRecursiveFftNodeIr::FftRader(ir) => ir.normalize,
        DoubleDoubleRecursiveFftNodeIr::Bluestein(ir) => ir.normalize,
        DoubleDoubleRecursiveFftNodeIr::CooleyTukey(ir) => {
            node_normalizes(&ir.left) && node_normalizes(&ir.right)
        }
    }
}

fn push_stockham_chunks(output: &mut Vec<FactorSpec>, prime_factors: &[usize]) -> Result<()> {
    let mut current = 1usize;
    for &factor in prime_factors {
        if !(2..=DD_RECURSIVE_STOCKHAM_LEAF_LIMIT).contains(&factor) {
            return Err(VkFftError::InvalidKernelIr(
                "double-double recursive Stockham factor is outside the executable leaf range",
            ));
        }
        let combined = current
            .checked_mul(factor)
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "double-double recursive Stockham chunk product",
            })?;
        if current > 1 && combined > DD_RECURSIVE_STOCKHAM_LEAF_LIMIT {
            output.push(FactorSpec {
                len: current,
                kind: FactorKind::Stockham,
            });
            current = factor;
        } else {
            current = combined;
        }
    }
    if current > 1 {
        output.push(FactorSpec {
            len: current,
            kind: FactorKind::Stockham,
        });
    }
    Ok(())
}

fn prime_factorization(mut value: usize) -> Vec<usize> {
    let mut factors = Vec::new();
    let mut divisor = 2usize;
    while divisor <= value / divisor {
        while value.is_multiple_of(divisor) {
            factors.push(divisor);
            value /= divisor;
        }
        divisor += if divisor == 2 { 1 } else { 2 };
    }
    if value > 1 {
        factors.push(value);
    }
    factors
}

fn rader_factor_specs_for_upload_split(
    stockham_prime_factors: &[usize],
    rader_primes: &[RaderPrimePlan],
    axis_split: &[usize],
) -> Result<(Vec<FactorSpec>, Vec<usize>)> {
    if !matches!(axis_split.len(), 2 | 3) || axis_split.contains(&0) {
        return Err(VkFftError::InvalidKernelIr(
            "double-double forced-Rader upload split must contain two or three non-zero factors",
        ));
    }
    let mut remaining_stockham = stockham_prime_factors.to_vec();
    let mut remaining_rader = rader_primes
        .iter()
        .map(|prime| (prime.prime, prime.mode.clone(), prime.multiplicity))
        .collect::<Vec<_>>();
    let mut output = Vec::new();
    let mut component_factor_counts = Vec::with_capacity(axis_split.len());

    for &component in axis_split {
        if component < 2 {
            return Err(VkFftError::UnsupportedKernelPath(
                "double-double forced-Rader upload split requires non-trivial factors",
            ));
        }
        let component_start = output.len();
        let mut pending_stockham = Vec::new();
        for prime in prime_factorization(component) {
            if let Some((_, mode, remaining)) = remaining_rader
                .iter_mut()
                .find(|(candidate, _, remaining)| *candidate == prime && *remaining > 0)
            {
                if !pending_stockham.is_empty() {
                    push_stockham_chunks(&mut output, &pending_stockham)?;
                    pending_stockham.clear();
                }
                output.push(FactorSpec {
                    len: prime,
                    kind: FactorKind::Rader(mode.clone()),
                });
                *remaining -= 1;
            } else if let Some(position) = remaining_stockham
                .iter()
                .position(|candidate| *candidate == prime)
            {
                remaining_stockham.remove(position);
                pending_stockham.push(prime);
            } else {
                return Err(VkFftError::InvalidKernelIr(
                    "double-double forced-Rader split contains a factor absent from the planner axis",
                ));
            }
        }
        if !pending_stockham.is_empty() {
            push_stockham_chunks(&mut output, &pending_stockham)?;
        }
        if checked_factor_product(&output[component_start..])? != component {
            return Err(VkFftError::InvalidKernelIr(
                "double-double forced-Rader factor bin does not match its axis split",
            ));
        }
        let factor_count = output.len() - component_start;
        if factor_count == 0 {
            return Err(VkFftError::InvalidKernelIr(
                "double-double forced-Rader upload factor bin is empty",
            ));
        }
        component_factor_counts.push(factor_count);
    }

    if !remaining_stockham.is_empty()
        || remaining_rader
            .iter()
            .any(|(_, _, remaining)| *remaining != 0)
    {
        return Err(VkFftError::InvalidKernelIr(
            "double-double forced-Rader factor bins did not consume the planner factorization",
        ));
    }
    Ok((output, component_factor_counts))
}

fn rader_upload_split_matches_root(
    node: &DoubleDoubleRecursiveFftNodeIr,
    axis_split: &[usize],
) -> bool {
    if axis_split.len() < 2 {
        return false;
    }
    let Some(right_len) = axis_split[1..]
        .iter()
        .try_fold(1usize, |product, factor| product.checked_mul(*factor))
    else {
        return false;
    };
    let DoubleDoubleRecursiveFftNodeIr::CooleyTukey(root) = node else {
        return false;
    };
    if root.left_len != axis_split[0] || root.right_len != right_len {
        return false;
    }
    axis_split.len() == 2 || rader_upload_split_matches_root(&root.right, &axis_split[1..])
}

fn checked_factor_product(factors: &[FactorSpec]) -> Result<usize> {
    factors.iter().try_fold(1usize, |product, factor| {
        product
            .checked_mul(factor.len)
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "double-double recursive factor product",
            })
    })
}

fn balanced_factor_split_index(factors: &[FactorSpec]) -> Result<usize> {
    if factors.len() < 2 {
        return Err(VkFftError::InvalidKernelIr(
            "double-double recursive balanced split requires at least two factors",
        ));
    }
    let total = checked_factor_product(factors)?;
    let mut left = 1usize;
    let mut best_index = 1usize;
    let mut best_larger_child = usize::MAX;
    let mut best_difference = usize::MAX;
    for (index, factor) in factors.iter().enumerate().take(factors.len() - 1) {
        left = left
            .checked_mul(factor.len)
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "double-double recursive left factor product",
            })?;
        let right = total / left;
        let larger_child = left.max(right);
        let difference = left.abs_diff(right);
        if larger_child < best_larger_child
            || (larger_child == best_larger_child && difference < best_difference)
        {
            best_index = index + 1;
            best_larger_child = larger_child;
            best_difference = difference;
        }
    }
    Ok(best_index)
}

fn preserve_factor_algorithm(plan: &mut FftPlan, factor: &FactorSpec) -> Result<()> {
    let axis = plan.axes.first_mut().ok_or(VkFftError::InvalidKernelIr(
        "double-double recursive leaf plan is missing axis metadata",
    ))?;
    if axis.effective_fft_len != factor.len {
        return Err(VkFftError::InvalidKernelIr(
            "double-double recursive leaf plan changed the parent factor length",
        ));
    }
    axis.algorithm = match &factor.kind {
        FactorKind::Stockham => AxisAlgorithm::Stockham {
            radix: RadixPlan {
                prime_factors: prime_factorization(factor.len),
                merged_radices: merged_radix_schedule(factor.len),
            },
        },
        FactorKind::Rader(mode) => AxisAlgorithm::Rader {
            stockham: RadixPlan {
                prime_factors: Vec::new(),
                merged_radices: Vec::new(),
            },
            primes: vec![RaderPrimePlan {
                prime: factor.len,
                multiplicity: 1,
                generator: primitive_root(factor.len).ok_or(VkFftError::InvalidKernelIr(
                    "double-double recursive Rader leaf lost its primitive root",
                ))?,
                mode: mode.clone(),
            }],
        },
    };
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn build_node(
    factors: &[FactorSpec],
    batch_count: usize,
    grouped_batch: usize,
    component_factor_counts: Option<&[usize]>,
    direction: Direction,
    normalize_inverse: bool,
    tuning: crate::config::PlannerTuning,
    input_storage: PrecisionStorage,
    output_storage: PrecisionStorage,
    device: Option<DeviceProfile>,
) -> Result<DoubleDoubleRecursiveFftNodeIr> {
    if factors.len() == 1 {
        if input_storage != PrecisionStorage::DoubleDouble
            || output_storage != PrecisionStorage::DoubleDouble
        {
            return Err(VkFftError::InvalidKernelIr(
                "double-double recursive leaves require full-DD internal storage",
            ));
        }
        let factor = factors[0].clone();
        let config = FftConfig::new(vec![factor.len])
            .with_batch_count(batch_count)
            .with_precision(Precision::DoubleDouble)
            .with_inverse_normalization(normalize_inverse)
            .with_tuning(tuning)
            .with_grouped_batch(0, grouped_batch)?;
        let mut plan = if let Some(device) = device {
            FftPlan::build_c2c_child_for_device(config, device, C2cDeviceAxisClass::Contiguous)?
        } else {
            FftPlan::build(config)?
        };
        preserve_factor_algorithm(&mut plan, &factor)?;
        return match factor.kind {
            FactorKind::Stockham => Ok(DoubleDoubleRecursiveFftNodeIr::Stockham(Box::new(
                DoubleDoubleStockhamIr::build(&plan, direction)?,
            ))),
            FactorKind::Rader(RaderMode::DirectMultiplication) => {
                Ok(DoubleDoubleRecursiveFftNodeIr::DirectRader(Box::new(
                    DoubleDoubleDirectRaderIr::build(&plan, direction)?,
                )))
            }
            FactorKind::Rader(RaderMode::FftConvolution { .. }) => {
                let ir = if let Some(device) = device {
                    DoubleDoubleFftRaderIr::build_for_device(&plan, direction, device)?
                } else {
                    DoubleDoubleFftRaderIr::build(&plan, direction)?
                };
                Ok(DoubleDoubleRecursiveFftNodeIr::FftRader(Box::new(ir)))
            }
        };
    }

    let split = match component_factor_counts {
        Some(counts)
            if counts.len() >= 2
                && counts.iter().all(|count| *count > 0)
                && counts.iter().sum::<usize>() == factors.len() =>
        {
            counts[0]
        }
        Some(_) => {
            return Err(VkFftError::InvalidKernelIr(
                "double-double recursive upload component factor counts are inconsistent",
            ));
        }
        None => balanced_factor_split_index(factors)?,
    };
    let left_len = checked_factor_product(&factors[..split])?;
    let right_len = checked_factor_product(&factors[split..])?;
    let logical_len = left_len
        .checked_mul(right_len)
        .ok_or(VkFftError::ArithmeticOverflow {
            operation: "double-double recursive Cooley-Tukey length",
        })?;
    let right_grouped_batch =
        grouped_batch
            .checked_mul(left_len)
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "double-double recursive right child grouped batch",
            })?;
    let left_grouped_batch =
        grouped_batch
            .checked_mul(right_len)
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "double-double recursive left child grouped batch",
            })?;
    let right_batch = batch_count
        .checked_mul(left_len)
        .ok_or(VkFftError::ArithmeticOverflow {
            operation: "double-double recursive right-child batch count",
        })?;
    let left_batch = batch_count
        .checked_mul(right_len)
        .ok_or(VkFftError::ArithmeticOverflow {
            operation: "double-double recursive left-child batch count",
        })?;
    let right = build_node(
        &factors[split..],
        right_batch,
        right_grouped_batch,
        component_factor_counts.and_then(|counts| (counts.len() > 2).then_some(&counts[1..])),
        direction,
        normalize_inverse,
        tuning,
        PrecisionStorage::DoubleDouble,
        PrecisionStorage::DoubleDouble,
        device,
    )?;
    let left = build_node(
        &factors[..split],
        left_batch,
        left_grouped_batch,
        None,
        direction,
        normalize_inverse,
        tuning,
        PrecisionStorage::DoubleDouble,
        PrecisionStorage::DoubleDouble,
        device,
    )?;
    let direction_name = match direction {
        Direction::Forward => "forward",
        Direction::Inverse => "inverse",
    };
    let twiddles = crate::lut::unit_root_table_double_double(logical_len, direction)?;
    let node = DoubleDoubleRecursiveCooleyTukeyIr {
        logical_len,
        left_len,
        right_len,
        batch_count,
        grouped_batch,
        direction,
        pack_right: DoubleDoubleCooleyTukeyPassIr::new(
            format!("vkfft_dd_recursive_pack_{logical_len}_{direction_name}"),
            direction,
            logical_len,
            left_len,
            right_len,
            batch_count,
            grouped_batch,
            input_storage,
            PrecisionStorage::DoubleDouble,
            DoubleDoubleCooleyTukeyPassOperation::PackRightInput,
        )?,
        right,
        twiddle_transpose: DoubleDoubleCooleyTukeyPassIr::new(
            format!("vkfft_dd_recursive_twiddle_{logical_len}_{direction_name}"),
            direction,
            logical_len,
            left_len,
            right_len,
            batch_count,
            grouped_batch,
            PrecisionStorage::DoubleDouble,
            PrecisionStorage::DoubleDouble,
            DoubleDoubleCooleyTukeyPassOperation::TwiddleTranspose,
        )?,
        twiddles,
        left,
        scatter_output: DoubleDoubleCooleyTukeyPassIr::new(
            format!("vkfft_dd_recursive_scatter_{logical_len}_{direction_name}"),
            direction,
            logical_len,
            left_len,
            right_len,
            batch_count,
            grouped_batch,
            PrecisionStorage::DoubleDouble,
            output_storage,
            DoubleDoubleCooleyTukeyPassOperation::ScatterOutput,
        )?,
    };
    node.validate()?;
    Ok(DoubleDoubleRecursiveFftNodeIr::CooleyTukey(Box::new(node)))
}

pub fn execute_double_double_recursive_ir(
    ir: &DoubleDoubleRecursiveFftIr,
    input: &[ComplexDoubleDouble],
) -> Result<Vec<ComplexDoubleDouble>> {
    if ir.external_storage != PrecisionStorage::DoubleDouble {
        return Err(VkFftError::UnsupportedPrecision {
            backend: "double-double recursive DD-storage executor",
            precision: "IR uses F64 external storage",
        });
    }
    execute_double_double_recursive_compute(ir, input)
}

pub fn execute_double_double_recursive_ir_f64_storage(
    ir: &DoubleDoubleRecursiveFftIr,
    input: &[Complex64],
) -> Result<Vec<Complex64>> {
    if ir.external_storage != PrecisionStorage::F64 {
        return Err(VkFftError::UnsupportedPrecision {
            backend: "double-double recursive F64-storage executor",
            precision: "IR uses double-double external storage",
        });
    }
    let promoted = input
        .iter()
        .copied()
        .map(ComplexDoubleDouble::from_complex64)
        .collect::<Vec<_>>();
    execute_double_double_recursive_compute(ir, &promoted).map(|values| {
        values
            .into_iter()
            .map(ComplexDoubleDouble::to_complex64)
            .collect()
    })
}

fn execute_double_double_recursive_compute(
    ir: &DoubleDoubleRecursiveFftIr,
    input: &[ComplexDoubleDouble],
) -> Result<Vec<ComplexDoubleDouble>> {
    ir.validate()?;
    let expected =
        ir.logical_len
            .checked_mul(ir.batch_count)
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "double-double recursive input element count",
            })?;
    if input.len() != expected {
        return Err(VkFftError::InputLengthMismatch {
            expected,
            actual: input.len(),
        });
    }
    let padded_input = ir
        .zero_pad_pass
        .as_ref()
        .filter(|pass| pass.operation.is_input_boundary())
        .map(|pass| {
            let mut values = input.to_vec();
            for batch in 0..ir.batch_count {
                let base = batch * ir.logical_len;
                values[base + pass.range.left..base + pass.range.right]
                    .fill(ComplexDoubleDouble::default());
            }
            values
        });
    let transform_input = padded_input.as_deref().unwrap_or(input);
    let mut output = execute_node(&ir.root, transform_input)?;
    if let Some(pass) = ir
        .zero_pad_pass
        .as_ref()
        .filter(|pass| pass.operation.is_output_boundary())
    {
        for batch in 0..ir.batch_count {
            let base = batch * ir.logical_len;
            output[base + pass.range.left..base + pass.range.right]
                .fill(ComplexDoubleDouble::default());
        }
    }
    Ok(output)
}

fn execute_node(
    node: &DoubleDoubleRecursiveFftNodeIr,
    input: &[ComplexDoubleDouble],
) -> Result<Vec<ComplexDoubleDouble>> {
    node.validate()?;
    let expected = node.logical_len().checked_mul(node.batch_count()).ok_or(
        VkFftError::ArithmeticOverflow {
            operation: "double-double recursive node input element count",
        },
    )?;
    if input.len() != expected {
        return Err(VkFftError::InputLengthMismatch {
            expected,
            actual: input.len(),
        });
    }
    match node {
        DoubleDoubleRecursiveFftNodeIr::Stockham(ir) => {
            execute_double_double_stockham_ir(ir, input)
        }
        DoubleDoubleRecursiveFftNodeIr::DirectRader(ir) => {
            execute_double_double_direct_rader_ir(ir, input)
        }
        DoubleDoubleRecursiveFftNodeIr::FftRader(ir) => {
            execute_double_double_fft_rader_ir(ir, input)
        }
        DoubleDoubleRecursiveFftNodeIr::Bluestein(ir) => {
            execute_double_double_bluestein_ir(ir, input)
        }
        DoubleDoubleRecursiveFftNodeIr::CooleyTukey(ir) => execute_cooley_tukey(ir, input),
    }
}

fn execute_cooley_tukey(
    ir: &DoubleDoubleRecursiveCooleyTukeyIr,
    input: &[ComplexDoubleDouble],
) -> Result<Vec<ComplexDoubleDouble>> {
    ir.validate()?;
    let total =
        ir.logical_len
            .checked_mul(ir.batch_count)
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "double-double recursive Cooley-Tukey element count",
            })?;
    let mut right_input = vec![ComplexDoubleDouble::default(); total];
    for batch in 0..ir.batch_count {
        for n1 in 0..ir.left_len {
            let destination_base = (batch * ir.left_len + n1) * ir.right_len;
            for n2 in 0..ir.right_len {
                let source = batch * ir.logical_len + n1 + ir.left_len * n2;
                right_input[destination_base + n2] = input[source];
            }
        }
    }
    let right_output = execute_node(&ir.right, &right_input)?;

    let mut left_input = vec![ComplexDoubleDouble::default(); total];
    for batch in 0..ir.batch_count {
        for k2 in 0..ir.right_len {
            let destination_base = (batch * ir.right_len + k2) * ir.left_len;
            for n1 in 0..ir.left_len {
                let source = (batch * ir.left_len + n1) * ir.right_len + k2;
                let twiddle_index = n1.checked_mul(k2).ok_or(VkFftError::ArithmeticOverflow {
                    operation: "double-double recursive twiddle index",
                })? % ir.logical_len;
                left_input[destination_base + n1] =
                    right_output[source] * ir.twiddles[twiddle_index];
            }
        }
    }
    let left_output = execute_node(&ir.left, &left_input)?;

    let mut output = vec![ComplexDoubleDouble::default(); total];
    for batch in 0..ir.batch_count {
        for k2 in 0..ir.right_len {
            let source_base = (batch * ir.right_len + k2) * ir.left_len;
            for k1 in 0..ir.left_len {
                output[batch * ir.logical_len + k2 + ir.right_len * k1] =
                    left_output[source_base + k1];
            }
        }
    }
    Ok(output)
}

fn precision_name(precision: Precision) -> &'static str {
    match precision {
        Precision::F16StorageF32Compute => "F16 storage / F32 compute",
        Precision::F32 => "F32",
        Precision::F64 => "F64",
        Precision::F64ComputeF32Storage => "F64 compute / F32 storage",
        Precision::DoubleDouble => "DoubleDouble",
        Precision::DoubleDoubleF64Storage => "DoubleDouble compute / F64 storage",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DoubleDoubleBluesteinConvolutionIr;
    use crate::double_double::{DoubleDouble, dft};

    fn sample(length: usize, batch_count: usize) -> Vec<ComplexDoubleDouble> {
        (0..length * batch_count)
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
            .collect()
    }

    fn max_error(a: &[ComplexDoubleDouble], b: &[ComplexDoubleDouble]) -> f64 {
        a.iter()
            .copied()
            .zip(b.iter().copied())
            .map(|(a, b)| {
                let dr = (a.re - b.re).to_f64();
                let di = (a.im - b.im).to_f64();
                dr.hypot(di)
            })
            .fold(0.0, f64::max)
    }

    #[test]
    fn smooth_stockham_above_leaf_limit_composes_into_recursive_tree() {
        let length = 4_116usize;
        let plan =
            FftPlan::build(FftConfig::new(vec![length]).with_precision(Precision::DoubleDouble))
                .unwrap();
        assert!(matches!(
            plan.axes[0].algorithm,
            AxisAlgorithm::Stockham { .. }
        ));
        let ir = DoubleDoubleRecursiveFftIr::build(&plan, Direction::Forward).unwrap();
        let DoubleDoubleRecursiveFftNodeIr::CooleyTukey(root) = &ir.root else {
            panic!("smooth DD length above the leaf limit must form a Cooley-Tukey tree");
        };
        assert_eq!(root.logical_len, length);
        assert_eq!(root.left_len * root.right_len, length);
        assert!(root.left_len <= DD_RECURSIVE_STOCKHAM_LEAF_LIMIT);
        assert!(root.right_len <= DD_RECURSIVE_STOCKHAM_LEAF_LIMIT);
        assert!(matches!(
            root.left,
            DoubleDoubleRecursiveFftNodeIr::Stockham(_)
        ));
        assert!(matches!(
            root.right,
            DoubleDoubleRecursiveFftNodeIr::Stockham(_)
        ));

        let program = crate::ProgramIr::double_double_recursive(&ir).unwrap();
        let shaders = crate::backend::vulkan::VulkanGlslBackend
            .lower_double_double_recursive(&ir)
            .unwrap();
        assert_eq!(shaders.len(), program.passes.len());
        assert!(shaders.iter().all(|shader| shader.compile_spirv().is_ok()));

        let one_dim = crate::DoubleDoubleOneDimIr::build(&plan, Direction::Forward).unwrap();
        assert!(matches!(one_dim, crate::DoubleDoubleOneDimIr::Recursive(_)));
    }

    #[test]
    fn composite_fft_rader_n51_has_strict_dd_fusion_candidate() {
        let mut device = DeviceProfile::generic(crate::Backend::Vulkan, crate::GpuVendor::Nvidia);
        device.shared_memory_bytes = 48 * 1024;
        device.shared_memory_pow2_bytes = 32 * 1024;
        device.max_threads_per_block = 1024;
        device.max_workgroup_size = [1024, 1024, 64];
        device.supports_f64 = true;
        let mut tuning = crate::PlannerTuning::portable();
        tuning.min_rader_direct_prime = 29;
        tuning.min_rader_fft_prime = 17;
        tuning.validate().unwrap();
        let plan = FftPlan::build(
            FftConfig::new(vec![3usize * 17])
                .with_precision(Precision::DoubleDouble)
                .with_tuning(tuning),
        )
        .unwrap();
        let ir = DoubleDoubleRecursiveFftIr::build_for_device(&plan, Direction::Forward, device)
            .unwrap();
        let DoubleDoubleRecursiveFftNodeIr::CooleyTukey(root) = &ir.root else {
            panic!("DD N51 FFT-Rader probe should keep a 3 x p17 Cooley root");
        };
        assert_eq!((root.left_len, root.right_len), (3, 17));
        assert!(matches!(
            root.left,
            DoubleDoubleRecursiveFftNodeIr::Stockham(_)
        ));
        let DoubleDoubleRecursiveFftNodeIr::FftRader(rader) = &root.right else {
            panic!("DD N51 right child should be p17 FFT-Rader");
        };
        assert_eq!(rader.prime, 17);
        assert_eq!(rader.convolution_len, 16);
        let DoubleDoubleBluesteinConvolutionIr::Stockham(forward) = &rader.forward_fft else {
            panic!("DD N51 p17 forward convolution should be Stockham");
        };
        let DoubleDoubleBluesteinConvolutionIr::Stockham(inverse) = &rader.inverse_fft else {
            panic!("DD N51 p17 inverse convolution should be Stockham");
        };
        assert_eq!(forward.sequence_len, 16);
        assert_eq!(inverse.sequence_len, 16);
        assert!(!forward.normalize);
        assert!(inverse.normalize);
        assert_eq!(
            root.pack_right.axis_batch_block,
            Some(StockhamAxisBlockSchedule {
                threads_per_transform: 4,
                grouped_batch: 1,
                transforms_on_x: false,
                axis_swapped: false,
                local_size_x: 4,
                local_size_y: 1,
            })
        );
        assert!(rader.caller_axis_batch_block.is_none());
        assert!(forward.axis_batch_block.is_none());
        assert!(inverse.axis_batch_block.is_none());
        assert_eq!(forward.stages.len(), 1);
        assert_eq!(forward.stages[0].radix, 16);
        assert_eq!(forward.stages[0].stage_size, 1);
        assert_eq!(forward.stages[0].butterflies, 1);
        assert_eq!(inverse.stages, forward.stages);
        let fused = root
            .fused_small_fft_rader_stockham()
            .unwrap()
            .expect("DD N51 should expose the strict p17 FFT-Rader fusion candidate");
        assert_eq!(fused.required_shared_memory_bytes().unwrap(), 3_168);
        let program = crate::ProgramIr::double_double_recursive(&ir).unwrap();
        assert_eq!(program.passes.len(), 1);
        assert_eq!(program.passes[0].name, fused.name());
        assert_eq!(program.passes[0].bindings.len(), 4);
        assert_eq!(
            program.passes[0].bindings[2].role,
            crate::BufferRole::LookupTable
        );
        assert_eq!(
            program.passes[0].bindings[3].role,
            crate::BufferRole::TwiddleLookupTable
        );

        let shaders = crate::backend::vulkan::VulkanGlslBackend
            .lower_double_double_recursive(&ir)
            .unwrap();
        assert_eq!(shaders.len(), 1);
        assert_eq!(shaders[0].required_shared_memory_bytes, 3_168);
        assert_eq!(shaders[0].glsl.matches("barrier();").count(), 4);
        assert!(
            shaders[0]
                .glsl
                .contains("fused double-double Stockham x FFT-Rader Cooley IR")
        );
        assert!(shaders[0].glsl.contains("vkfft_dd_fft16_forward_root"));
        assert!(shaders[0].glsl.contains("vkfft_dd_fft16_inverse_root"));
        shaders[0].compile_spirv().unwrap();

        let length = 51usize;
        let mut input = vec![ComplexDoubleDouble::default(); length];
        input[1] = ComplexDoubleDouble::new(DoubleDouble::from_f64(1.0), DoubleDouble::default());
        let actual = execute_double_double_recursive_ir(&ir, &input).unwrap();
        let expected = (0..length)
            .map(|k| {
                let angle = -std::f64::consts::TAU * k as f64 / length as f64;
                ComplexDoubleDouble::new(
                    DoubleDouble::from_f64(angle.cos()),
                    DoubleDouble::from_f64(angle.sin()),
                )
            })
            .collect::<Vec<_>>();
        let error = max_error(&actual, &expected);
        assert!(
            error <= 2.0e-15,
            "DD N51 FFT-Rader impulse mismatch: {error:e}"
        );
        ir.validate().unwrap();
    }

    #[test]
    fn composite_fft_rader_n68_radix4_has_strict_dd_fusion_candidate() {
        let mut device = DeviceProfile::generic(crate::Backend::Vulkan, crate::GpuVendor::Nvidia);
        device.shared_memory_bytes = 48 * 1024;
        device.shared_memory_pow2_bytes = 32 * 1024;
        device.max_threads_per_block = 1024;
        device.max_workgroup_size = [1024, 1024, 64];
        device.supports_f64 = true;
        let mut tuning = crate::PlannerTuning::portable();
        tuning.min_rader_direct_prime = 29;
        tuning.min_rader_fft_prime = 17;
        tuning.validate().unwrap();
        let plan = FftPlan::build(
            FftConfig::new(vec![4usize * 17])
                .with_precision(Precision::DoubleDouble)
                .with_tuning(tuning),
        )
        .unwrap();
        let ir = DoubleDoubleRecursiveFftIr::build_for_device(&plan, Direction::Forward, device)
            .unwrap();
        let DoubleDoubleRecursiveFftNodeIr::CooleyTukey(root) = &ir.root else {
            panic!("DD N68 FFT-Rader probe should keep a 4 x p17 Cooley root");
        };
        assert_eq!((root.left_len, root.right_len), (4, 17));
        let DoubleDoubleRecursiveFftNodeIr::Stockham(stockham) = &root.left else {
            panic!("DD N68 left child should be radix-4 Stockham");
        };
        assert_eq!(stockham.sequence_len, 4);
        assert_eq!(stockham.stages.len(), 1);
        assert_eq!(stockham.stages[0].radix, 4);
        let DoubleDoubleRecursiveFftNodeIr::FftRader(rader) = &root.right else {
            panic!("DD N68 right child should be p17 FFT-Rader");
        };
        assert_eq!(rader.prime, 17);
        assert_eq!(rader.convolution_len, 16);
        let DoubleDoubleBluesteinConvolutionIr::Stockham(forward) = &rader.forward_fft else {
            panic!("DD N68 p17 forward convolution should be Stockham");
        };
        let DoubleDoubleBluesteinConvolutionIr::Stockham(inverse) = &rader.inverse_fft else {
            panic!("DD N68 p17 inverse convolution should be Stockham");
        };
        assert_eq!((forward.batch_count, forward.grouped_batch), (4, 4));
        assert_eq!((inverse.batch_count, inverse.grouped_batch), (4, 4));
        assert_eq!(forward.sequence_len, 16);
        assert_eq!(inverse.sequence_len, 16);
        assert_eq!(forward.stages.len(), 1);
        assert_eq!(forward.stages[0].radix, 16);
        assert_eq!(inverse.stages, forward.stages);
        assert_eq!(
            root.pack_right.axis_batch_block,
            Some(StockhamAxisBlockSchedule {
                threads_per_transform: 5,
                grouped_batch: 1,
                transforms_on_x: false,
                axis_swapped: false,
                local_size_x: 5,
                local_size_y: 1,
            })
        );
        let fused = root
            .fused_small_fft_rader_stockham()
            .unwrap()
            .expect("DD N68 should expose the strict radix-4 p17 FFT-Rader fusion candidate");
        assert_eq!(fused.required_shared_memory_bytes().unwrap(), 4_224);
        let program = crate::ProgramIr::double_double_recursive(&ir).unwrap();
        assert_eq!(program.passes.len(), 1);
        assert_eq!(program.passes[0].name, fused.name());
        assert_eq!(program.passes[0].bindings.len(), 4);

        let shaders = crate::backend::vulkan::VulkanGlslBackend
            .lower_double_double_recursive(&ir)
            .unwrap();
        assert_eq!(shaders.len(), 1);
        assert_eq!(shaders[0].required_shared_memory_bytes, 4_224);
        assert_eq!(shaders[0].glsl.matches("barrier();").count(), 4);
        assert!(shaders[0].glsl.contains("const uint VKFFT_A = 4u;"));
        assert!(shaders[0].glsl.contains("vkfft_dd_fft16_forward_root"));
        assert!(shaders[0].glsl.contains("vkfft_dd_fft16_inverse_root"));
        shaders[0].compile_spirv().unwrap();

        let length = 68usize;
        let mut input = vec![ComplexDoubleDouble::default(); length];
        input[1] = ComplexDoubleDouble::new(DoubleDouble::from_f64(1.0), DoubleDouble::default());
        let actual = execute_double_double_recursive_ir(&ir, &input).unwrap();
        let expected = (0..length)
            .map(|k| {
                let angle = -std::f64::consts::TAU * k as f64 / length as f64;
                ComplexDoubleDouble::new(
                    DoubleDouble::from_f64(angle.cos()),
                    DoubleDouble::from_f64(angle.sin()),
                )
            })
            .collect::<Vec<_>>();
        let error = max_error(&actual, &expected);
        assert!(
            error <= 2.0e-15,
            "DD N68 radix-4 FFT-Rader impulse mismatch: {error:e}"
        );
        ir.validate().unwrap();
    }

    #[test]
    fn composite_fft_rader_n85_radix5_has_strict_dd_fusion_candidate() {
        let mut device = DeviceProfile::generic(crate::Backend::Vulkan, crate::GpuVendor::Nvidia);
        device.shared_memory_bytes = 48 * 1024;
        device.shared_memory_pow2_bytes = 32 * 1024;
        device.max_threads_per_block = 1024;
        device.max_workgroup_size = [1024, 1024, 64];
        device.supports_f64 = true;
        let mut tuning = crate::PlannerTuning::portable();
        tuning.min_rader_direct_prime = 29;
        tuning.min_rader_fft_prime = 17;
        tuning.validate().unwrap();
        let plan = FftPlan::build(
            FftConfig::new(vec![5usize * 17])
                .with_precision(Precision::DoubleDouble)
                .with_tuning(tuning),
        )
        .unwrap();
        let ir = DoubleDoubleRecursiveFftIr::build_for_device(&plan, Direction::Forward, device)
            .unwrap();
        let DoubleDoubleRecursiveFftNodeIr::CooleyTukey(root) = &ir.root else {
            panic!("DD N85 FFT-Rader probe should keep a 5 x p17 Cooley root");
        };
        assert_eq!((root.left_len, root.right_len), (5, 17));
        let DoubleDoubleRecursiveFftNodeIr::Stockham(stockham) = &root.left else {
            panic!("DD N85 left child should be radix-5 Stockham");
        };
        assert_eq!(stockham.sequence_len, 5);
        assert_eq!(stockham.stages.len(), 1);
        assert_eq!(stockham.stages[0].radix, 5);
        let DoubleDoubleRecursiveFftNodeIr::FftRader(rader) = &root.right else {
            panic!("DD N85 right child should be p17 FFT-Rader");
        };
        assert_eq!(rader.prime, 17);
        assert_eq!(rader.convolution_len, 16);
        let DoubleDoubleBluesteinConvolutionIr::Stockham(forward) = &rader.forward_fft else {
            panic!("DD N85 p17 forward convolution should be Stockham");
        };
        let DoubleDoubleBluesteinConvolutionIr::Stockham(inverse) = &rader.inverse_fft else {
            panic!("DD N85 p17 inverse convolution should be Stockham");
        };
        assert_eq!((forward.batch_count, forward.grouped_batch), (5, 5));
        assert_eq!((inverse.batch_count, inverse.grouped_batch), (5, 5));
        assert_eq!(forward.sequence_len, 16);
        assert_eq!(inverse.sequence_len, 16);
        assert_eq!(forward.stages.len(), 1);
        assert_eq!(forward.stages[0].radix, 16);
        assert_eq!(inverse.stages, forward.stages);
        assert!(rader.caller_axis_batch_block.is_none());
        assert!(forward.axis_batch_block.is_none());
        assert!(inverse.axis_batch_block.is_none());
        assert_eq!(
            root.pack_right.axis_batch_block,
            Some(StockhamAxisBlockSchedule {
                threads_per_transform: 6,
                grouped_batch: 1,
                transforms_on_x: false,
                axis_swapped: false,
                local_size_x: 6,
                local_size_y: 1,
            })
        );
        let fused = root
            .fused_small_fft_rader_stockham()
            .unwrap()
            .expect("DD N85 should expose the strict radix-5 p17 FFT-Rader fusion candidate");
        assert_eq!(fused.required_shared_memory_bytes().unwrap(), 5_280);
        let program = crate::ProgramIr::double_double_recursive(&ir).unwrap();
        assert_eq!(program.passes.len(), 1);
        assert_eq!(program.passes[0].name, fused.name());
        assert_eq!(program.passes[0].bindings.len(), 4);

        let shaders = crate::backend::vulkan::VulkanGlslBackend
            .lower_double_double_recursive(&ir)
            .unwrap();
        assert_eq!(shaders.len(), 1);
        assert_eq!(shaders[0].required_shared_memory_bytes, 5_280);
        assert_eq!(shaders[0].glsl.matches("barrier();").count(), 4);
        assert!(shaders[0].glsl.contains("const uint VKFFT_A = 5u;"));
        assert!(shaders[0].glsl.contains("vkfft_dd_fft16_forward_root"));
        assert!(shaders[0].glsl.contains("vkfft_dd_fft16_inverse_root"));
        shaders[0].compile_spirv().unwrap();

        let length = 85usize;
        let mut input = vec![ComplexDoubleDouble::default(); length];
        input[1] = ComplexDoubleDouble::new(DoubleDouble::from_f64(1.0), DoubleDouble::default());
        let actual = execute_double_double_recursive_ir(&ir, &input).unwrap();
        let expected = (0..length)
            .map(|k| {
                let angle = -std::f64::consts::TAU * k as f64 / length as f64;
                ComplexDoubleDouble::new(
                    DoubleDouble::from_f64(angle.cos()),
                    DoubleDouble::from_f64(angle.sin()),
                )
            })
            .collect::<Vec<_>>();
        let error = max_error(&actual, &expected);
        assert!(
            error <= 2.0e-15,
            "DD N85 radix-5 FFT-Rader impulse mismatch: {error:e}"
        );
        ir.validate().unwrap();
    }

    #[test]
    fn composite_fft_rader_n102_radix6_has_strict_dd_fusion_candidate() {
        let mut device = DeviceProfile::generic(crate::Backend::Vulkan, crate::GpuVendor::Nvidia);
        device.shared_memory_bytes = 48 * 1024;
        device.shared_memory_pow2_bytes = 32 * 1024;
        device.max_threads_per_block = 1024;
        device.max_workgroup_size = [1024, 1024, 64];
        device.supports_f64 = true;
        let mut tuning = crate::PlannerTuning::portable();
        tuning.min_rader_direct_prime = 29;
        tuning.min_rader_fft_prime = 17;
        tuning.validate().unwrap();
        let plan = FftPlan::build(
            FftConfig::new(vec![6usize * 17])
                .with_precision(Precision::DoubleDouble)
                .with_tuning(tuning),
        )
        .unwrap();
        let ir = DoubleDoubleRecursiveFftIr::build_for_device(&plan, Direction::Forward, device)
            .unwrap();
        let DoubleDoubleRecursiveFftNodeIr::CooleyTukey(root) = &ir.root else {
            panic!("DD N102 FFT-Rader probe should keep a 6 x p17 Cooley root");
        };
        assert_eq!((root.left_len, root.right_len), (6, 17));
        let DoubleDoubleRecursiveFftNodeIr::Stockham(stockham) = &root.left else {
            panic!("DD N102 left child should be radix-6 Stockham");
        };
        assert_eq!(stockham.sequence_len, 6);
        assert_eq!(stockham.stages.len(), 1);
        assert_eq!(stockham.stages[0].radix, 6);
        let DoubleDoubleRecursiveFftNodeIr::FftRader(rader) = &root.right else {
            panic!("DD N102 right child should be p17 FFT-Rader");
        };
        assert_eq!(rader.prime, 17);
        assert_eq!(rader.convolution_len, 16);
        let DoubleDoubleBluesteinConvolutionIr::Stockham(forward) = &rader.forward_fft else {
            panic!("DD N102 p17 forward convolution should be Stockham");
        };
        let DoubleDoubleBluesteinConvolutionIr::Stockham(inverse) = &rader.inverse_fft else {
            panic!("DD N102 p17 inverse convolution should be Stockham");
        };
        assert_eq!((forward.batch_count, forward.grouped_batch), (6, 6));
        assert_eq!((inverse.batch_count, inverse.grouped_batch), (6, 6));
        assert_eq!(forward.sequence_len, 16);
        assert_eq!(inverse.sequence_len, 16);
        assert_eq!(forward.stages.len(), 1);
        assert_eq!(forward.stages[0].radix, 16);
        assert_eq!(inverse.stages, forward.stages);
        assert!(rader.caller_axis_batch_block.is_none());
        assert!(forward.axis_batch_block.is_none());
        assert!(inverse.axis_batch_block.is_none());
        assert_eq!(
            root.pack_right.axis_batch_block,
            Some(StockhamAxisBlockSchedule {
                threads_per_transform: 9,
                grouped_batch: 1,
                transforms_on_x: false,
                axis_swapped: false,
                local_size_x: 9,
                local_size_y: 1,
            })
        );
        let fused = root
            .fused_small_fft_rader_stockham()
            .unwrap()
            .expect("DD N102 should expose the strict radix-6 p17 FFT-Rader fusion candidate");
        assert_eq!(fused.required_shared_memory_bytes().unwrap(), 6_336);
        let program = crate::ProgramIr::double_double_recursive(&ir).unwrap();
        assert_eq!(program.passes.len(), 1);
        assert_eq!(program.passes[0].name, fused.name());
        assert_eq!(program.passes[0].bindings.len(), 4);

        let shaders = crate::backend::vulkan::VulkanGlslBackend
            .lower_double_double_recursive(&ir)
            .unwrap();
        assert_eq!(shaders.len(), 1);
        assert_eq!(shaders[0].required_shared_memory_bytes, 6_336);
        assert_eq!(shaders[0].glsl.matches("barrier();").count(), 4);
        assert!(shaders[0].glsl.contains("const uint VKFFT_A = 6u;"));
        assert!(shaders[0].glsl.contains("vkfft_dd_fft16_forward_root"));
        assert!(shaders[0].glsl.contains("vkfft_dd_fft16_inverse_root"));
        shaders[0].compile_spirv().unwrap();

        let length = 102usize;
        let mut input = vec![ComplexDoubleDouble::default(); length];
        input[1] = ComplexDoubleDouble::new(DoubleDouble::from_f64(1.0), DoubleDouble::default());
        let actual = execute_double_double_recursive_ir(&ir, &input).unwrap();
        let expected = (0..length)
            .map(|k| {
                let angle = -std::f64::consts::TAU * k as f64 / length as f64;
                ComplexDoubleDouble::new(
                    DoubleDouble::from_f64(angle.cos()),
                    DoubleDouble::from_f64(angle.sin()),
                )
            })
            .collect::<Vec<_>>();
        let error = max_error(&actual, &expected);
        assert!(
            error <= 2.0e-15,
            "DD N102 radix-6 FFT-Rader impulse mismatch: {error:e}"
        );
        ir.validate().unwrap();
    }

    #[test]
    fn composite_fft_rader_p17_radix7_through16_have_strict_dd_fusion_candidates() {
        let mut device = DeviceProfile::generic(crate::Backend::Vulkan, crate::GpuVendor::Nvidia);
        device.shared_memory_bytes = 48 * 1024;
        device.shared_memory_pow2_bytes = 32 * 1024;
        device.max_threads_per_block = 1024;
        device.max_workgroup_size = [1024, 1024, 64];
        device.supports_f64 = true;
        let mut tuning = crate::PlannerTuning::portable();
        tuning.min_rader_direct_prime = 29;
        tuning.min_rader_fft_prime = 17;
        tuning.validate().unwrap();
        let cases = [
            (7usize, 9usize),
            (8, 9),
            (9, 17),
            (10, 11),
            (11, 128),
            (12, 17),
            (13, 128),
            (14, 17),
            (15, 17),
            (16, 17),
        ];

        for (radix, threads_per_transform) in cases {
            let length = radix * 17;
            let plan = FftPlan::build(
                FftConfig::new(vec![length])
                    .with_precision(Precision::DoubleDouble)
                    .with_tuning(tuning),
            )
            .unwrap();
            let ir =
                DoubleDoubleRecursiveFftIr::build_for_device(&plan, Direction::Forward, device)
                    .unwrap();
            let DoubleDoubleRecursiveFftNodeIr::CooleyTukey(root) = &ir.root else {
                panic!("DD N{length} p17 surface should keep a {radix} x 17 Cooley root");
            };
            assert_eq!((root.left_len, root.right_len), (radix, 17));
            let DoubleDoubleRecursiveFftNodeIr::Stockham(stockham) = &root.left else {
                panic!("DD N{length} left child should be single-stage Stockham");
            };
            assert_eq!(stockham.sequence_len, radix);
            assert_eq!(stockham.stages.len(), 1);
            assert_eq!(stockham.stages[0].radix, radix);
            assert_eq!(stockham.stages[0].stage_size, 1);
            assert_eq!(stockham.stages[0].butterflies, 1);
            let DoubleDoubleRecursiveFftNodeIr::FftRader(rader) = &root.right else {
                panic!("DD N{length} right child should be p17 FFT-Rader");
            };
            assert_eq!(rader.prime, 17);
            assert_eq!(rader.convolution_len, 16);
            let DoubleDoubleBluesteinConvolutionIr::Stockham(forward) = &rader.forward_fft else {
                panic!("DD N{length} p17 forward convolution should be Stockham");
            };
            let DoubleDoubleBluesteinConvolutionIr::Stockham(inverse) = &rader.inverse_fft else {
                panic!("DD N{length} p17 inverse convolution should be Stockham");
            };
            assert_eq!((forward.batch_count, forward.grouped_batch), (radix, radix));
            assert_eq!((inverse.batch_count, inverse.grouped_batch), (radix, radix));
            assert_eq!(forward.sequence_len, 16);
            assert_eq!(inverse.sequence_len, 16);
            assert_eq!(forward.stages.len(), 1);
            assert_eq!(forward.stages[0].radix, 16);
            assert_eq!(inverse.stages, forward.stages);
            assert!(rader.caller_axis_batch_block.is_none());
            assert!(forward.axis_batch_block.is_none());
            assert!(inverse.axis_batch_block.is_none());
            assert_eq!(
                root.pack_right.axis_batch_block,
                Some(StockhamAxisBlockSchedule {
                    threads_per_transform,
                    grouped_batch: 1,
                    transforms_on_x: false,
                    axis_swapped: false,
                    local_size_x: threads_per_transform,
                    local_size_y: 1,
                })
            );
            let fused = root
                .fused_small_fft_rader_stockham()
                .unwrap()
                .expect("DD p17 single-stage parent should expose the strict fusion candidate");
            let expected_shared = 1_056usize * radix;
            assert_eq!(
                fused.required_shared_memory_bytes().unwrap(),
                expected_shared
            );
            let program = crate::ProgramIr::double_double_recursive(&ir).unwrap();
            assert_eq!(program.passes.len(), 1);
            assert_eq!(program.passes[0].name, fused.name());
            assert_eq!(program.passes[0].bindings.len(), 4);

            let shaders = crate::backend::vulkan::VulkanGlslBackend
                .lower_double_double_recursive(&ir)
                .unwrap();
            assert_eq!(shaders.len(), 1);
            assert_eq!(shaders[0].required_shared_memory_bytes, expected_shared);
            assert_eq!(shaders[0].glsl.matches("barrier();").count(), 4);
            assert!(
                shaders[0]
                    .glsl
                    .contains(&format!("const uint VKFFT_A = {radix}u;"))
            );
            shaders[0].compile_spirv().unwrap();

            let mut input = vec![ComplexDoubleDouble::default(); length];
            input[1] =
                ComplexDoubleDouble::new(DoubleDouble::from_f64(1.0), DoubleDouble::default());
            let actual = execute_double_double_recursive_ir(&ir, &input).unwrap();
            let expected = (0..length)
                .map(|k| {
                    let angle = -std::f64::consts::TAU * k as f64 / length as f64;
                    ComplexDoubleDouble::new(
                        DoubleDouble::from_f64(angle.cos()),
                        DoubleDouble::from_f64(angle.sin()),
                    )
                })
                .collect::<Vec<_>>();
            let error = max_error(&actual, &expected);
            assert!(
                error <= 2.0e-15,
                "DD N{length} radix-{radix} p17 FFT-Rader impulse mismatch: {error:e}"
            );
            ir.validate().unwrap();
        }
    }

    #[test]
    fn composite_fft_rader_n289_has_strict_dual_rader_fusion_candidate() {
        let mut device = DeviceProfile::generic(crate::Backend::Vulkan, crate::GpuVendor::Nvidia);
        device.shared_memory_bytes = 48 * 1024;
        device.shared_memory_pow2_bytes = 32 * 1024;
        device.max_threads_per_block = 1024;
        device.max_workgroup_size = [1024, 1024, 64];
        device.supports_f64 = true;
        let mut tuning = crate::PlannerTuning::portable();
        tuning.min_rader_direct_prime = 29;
        tuning.min_rader_fft_prime = 17;
        tuning.validate().unwrap();
        let plan = FftPlan::build(
            FftConfig::new(vec![17usize * 17])
                .with_precision(Precision::DoubleDouble)
                .with_tuning(tuning),
        )
        .unwrap();
        let ir = DoubleDoubleRecursiveFftIr::build_for_device(&plan, Direction::Forward, device)
            .unwrap();
        let DoubleDoubleRecursiveFftNodeIr::CooleyTukey(root) = &ir.root else {
            panic!("DD N289 dual-Rader probe should remain Cooley-Tukey");
        };
        assert_eq!((root.left_len, root.right_len), (17, 17));
        let block = root.pack_right.axis_batch_block.unwrap();
        assert_eq!(block.threads_per_transform, 19);
        assert_eq!(block.grouped_batch, 1);
        assert_eq!([block.local_size_x, block.local_size_y], [19, 1]);
        let DoubleDoubleRecursiveFftNodeIr::FftRader(left) = &root.left else {
            panic!("DD N289 left child should be p17 FFT-Rader");
        };
        let DoubleDoubleRecursiveFftNodeIr::FftRader(right) = &root.right else {
            panic!("DD N289 right child should be p17 FFT-Rader");
        };
        assert_eq!((left.prime, right.prime), (17, 17));
        assert!(root.fused_small_fft_rader_stockham().unwrap().is_none());
        let fused = root
            .fused_dual_fft_rader_n289()
            .unwrap()
            .expect("DD N289 should expose the strict dual FFT-Rader fused candidate");
        assert_eq!(fused.required_shared_memory_bytes().unwrap(), 19_040);
        let program = crate::ProgramIr::double_double_recursive(&ir).unwrap();
        assert_eq!(program.passes.len(), 1);
        assert_eq!(program.passes[0].bindings.len(), 5);
        let shaders = crate::backend::vulkan::VulkanGlslBackend
            .lower_double_double_recursive(&ir)
            .unwrap();
        assert_eq!(shaders.len(), 1);
        assert_eq!(shaders[0].required_shared_memory_bytes, 19_040);
        assert_eq!(shaders[0].glsl.matches("barrier();").count(), 7);
        assert!(
            shaders[0]
                .glsl
                .contains("VKFFT_THREADS_PER_TRANSFORM = 19u")
        );
        shaders[0].compile_spirv().unwrap();

        let mut input = vec![ComplexDoubleDouble::default(); 289];
        input[1] = ComplexDoubleDouble::new(DoubleDouble::from_f64(1.0), DoubleDouble::default());
        let actual = execute_double_double_recursive_ir(&ir, &input).unwrap();
        let expected = (0..289)
            .map(|k| {
                let angle = -std::f64::consts::TAU * k as f64 / 289.0;
                ComplexDoubleDouble::new(
                    DoubleDouble::from_f64(angle.cos()),
                    DoubleDouble::from_f64(angle.sin()),
                )
            })
            .collect::<Vec<_>>();
        let error = max_error(&actual, &expected);
        assert!(
            error <= 2.0e-15,
            "DD N289 dual-Rader impulse mismatch: {error:e}"
        );
        ir.validate().unwrap();
    }

    #[test]
    fn composite_fft_rader_n289_dual_rader_is_capacity_gated() {
        let mut device = DeviceProfile::generic(crate::Backend::Vulkan, crate::GpuVendor::Nvidia);
        device.shared_memory_bytes = 18 * 1024;
        device.shared_memory_pow2_bytes = 18 * 1024;
        device.max_threads_per_block = 1024;
        device.max_workgroup_size = [1024, 1024, 64];
        device.supports_f64 = true;
        let mut tuning = crate::PlannerTuning::portable();
        tuning.min_rader_direct_prime = 29;
        tuning.min_rader_fft_prime = 17;
        tuning.validate().unwrap();
        let plan = FftPlan::build(
            FftConfig::new(vec![289])
                .with_precision(Precision::DoubleDouble)
                .with_tuning(tuning),
        )
        .unwrap();
        let ir = DoubleDoubleRecursiveFftIr::build_for_device(&plan, Direction::Forward, device)
            .unwrap();
        let DoubleDoubleRecursiveFftNodeIr::CooleyTukey(root) = &ir.root else {
            panic!("DD N289 18 KiB capacity probe should remain Cooley-Tukey");
        };
        assert_eq!((root.left_len, root.right_len), (17, 17));
        assert!(root.fused_dual_fft_rader_n289().unwrap().is_none());
        assert!(
            crate::ProgramIr::double_double_recursive(&ir)
                .unwrap()
                .passes
                .len()
                > 1
        );

        let mut input = vec![ComplexDoubleDouble::default(); 289];
        input[1] = ComplexDoubleDouble::new(DoubleDouble::from_f64(1.0), DoubleDouble::default());
        let actual = execute_double_double_recursive_ir(&ir, &input).unwrap();
        let expected = (0..289)
            .map(|k| {
                let angle = -std::f64::consts::TAU * k as f64 / 289.0;
                ComplexDoubleDouble::new(
                    DoubleDouble::from_f64(angle.cos()),
                    DoubleDouble::from_f64(angle.sin()),
                )
            })
            .collect::<Vec<_>>();
        let error = max_error(&actual, &expected);
        assert!(
            error <= 2.0e-15,
            "DD N289 18 KiB fallback impulse mismatch: {error:e}"
        );
        ir.validate().unwrap();
    }

    #[test]
    fn composite_fft_rader_n323_has_independent_p17_p19_fusion_candidate() {
        let mut device = DeviceProfile::generic(crate::Backend::Vulkan, crate::GpuVendor::Nvidia);
        device.shared_memory_bytes = 48 * 1024;
        device.shared_memory_pow2_bytes = 32 * 1024;
        device.max_threads_per_block = 1024;
        device.max_workgroup_size = [1024, 1024, 64];
        device.supports_f64 = true;
        let mut tuning = crate::PlannerTuning::portable();
        tuning.min_rader_direct_prime = 29;
        tuning.min_rader_fft_prime = 17;
        tuning.validate().unwrap();
        let plan = FftPlan::build(
            FftConfig::new(vec![323])
                .with_precision(Precision::DoubleDouble)
                .with_tuning(tuning),
        )
        .unwrap();
        let ir = DoubleDoubleRecursiveFftIr::build_for_device(&plan, Direction::Forward, device)
            .unwrap();
        let DoubleDoubleRecursiveFftNodeIr::CooleyTukey(root) = &ir.root else {
            panic!("DD N323 dual-Rader probe should remain Cooley-Tukey");
        };
        assert_eq!((root.left_len, root.right_len), (17, 19));
        let block = root.pack_right.axis_batch_block.unwrap();
        assert_eq!(block.threads_per_transform, 21);
        assert_eq!([block.local_size_x, block.local_size_y], [21, 1]);
        let DoubleDoubleRecursiveFftNodeIr::FftRader(left) = &root.left else {
            panic!("DD N323 left child should be p17 FFT-Rader");
        };
        let DoubleDoubleRecursiveFftNodeIr::FftRader(right) = &root.right else {
            panic!("DD N323 right child should be p19 FFT-Rader");
        };
        assert_eq!((left.prime, left.convolution_len), (17, 16));
        assert_eq!((right.prime, right.convolution_len), (19, 18));
        let DoubleDoubleBluesteinConvolutionIr::Stockham(right_forward) = &right.forward_fft else {
            panic!("DD N323 p19 forward convolution should remain Stockham");
        };
        assert_eq!(
            right_forward
                .stages
                .iter()
                .map(|stage| stage.radix)
                .collect::<Vec<_>>(),
            vec![9, 2]
        );
        let fused = root
            .fused_dual_fft_rader_n323()
            .unwrap()
            .expect("DD N323 should expose the independent p17 x p19 fused candidate");
        assert_eq!(fused.required_shared_memory_bytes().unwrap(), 21_344);
        let program = crate::ProgramIr::double_double_recursive(&ir).unwrap();
        assert_eq!(program.passes.len(), 1);
        assert_eq!(program.passes[0].bindings.len(), 5);
        let shaders = crate::backend::vulkan::VulkanGlslBackend
            .lower_double_double_recursive(&ir)
            .unwrap();
        assert_eq!(shaders.len(), 1);
        assert_eq!(shaders[0].required_shared_memory_bytes, 21_344);
        assert_eq!(shaders[0].glsl.matches("barrier();").count(), 7);
        assert!(shaders[0].glsl.contains("VKFFT_COUNT = 18u"));
        assert!(shaders[0].glsl.contains("VKFFT_LEFT_COUNT = 16u"));
        shaders[0].compile_spirv().unwrap();

        let mut input = vec![ComplexDoubleDouble::default(); 323];
        input[1] = ComplexDoubleDouble::new(DoubleDouble::from_f64(1.0), DoubleDouble::default());
        let actual = execute_double_double_recursive_ir(&ir, &input).unwrap();
        let expected = (0..323)
            .map(|k| {
                let angle = -std::f64::consts::TAU * k as f64 / 323.0;
                ComplexDoubleDouble::new(
                    DoubleDouble::from_f64(angle.cos()),
                    DoubleDouble::from_f64(angle.sin()),
                )
            })
            .collect::<Vec<_>>();
        let error = max_error(&actual, &expected);
        assert!(
            error <= 2.0e-15,
            "DD N323 dual-Rader impulse mismatch: {error:e}"
        );
        ir.validate().unwrap();
    }

    #[test]
    fn composite_fft_rader_n323_dual_rader_is_capacity_gated() {
        let mut device = DeviceProfile::generic(crate::Backend::Vulkan, crate::GpuVendor::Nvidia);
        device.shared_memory_bytes = 20 * 1024;
        device.shared_memory_pow2_bytes = 20 * 1024;
        device.max_threads_per_block = 1024;
        device.max_workgroup_size = [1024, 1024, 64];
        device.supports_f64 = true;
        let mut tuning = crate::PlannerTuning::portable();
        tuning.min_rader_direct_prime = 29;
        tuning.min_rader_fft_prime = 17;
        tuning.validate().unwrap();
        let plan = FftPlan::build(
            FftConfig::new(vec![323])
                .with_precision(Precision::DoubleDouble)
                .with_tuning(tuning),
        )
        .unwrap();
        let ir = DoubleDoubleRecursiveFftIr::build_for_device(&plan, Direction::Forward, device)
            .unwrap();
        let DoubleDoubleRecursiveFftNodeIr::CooleyTukey(root) = &ir.root else {
            panic!("DD N323 20 KiB capacity probe should remain Cooley-Tukey");
        };
        assert_eq!((root.left_len, root.right_len), (17, 19));
        assert!(root.fused_dual_fft_rader_n323().unwrap().is_none());
        assert_eq!(
            crate::ProgramIr::double_double_recursive(&ir)
                .unwrap()
                .passes
                .len(),
            13
        );

        let mut input = vec![ComplexDoubleDouble::default(); 323];
        input[1] = ComplexDoubleDouble::new(DoubleDouble::from_f64(1.0), DoubleDouble::default());
        let actual = execute_double_double_recursive_ir(&ir, &input).unwrap();
        let expected = (0..323)
            .map(|k| {
                let angle = -std::f64::consts::TAU * k as f64 / 323.0;
                ComplexDoubleDouble::new(
                    DoubleDouble::from_f64(angle.cos()),
                    DoubleDouble::from_f64(angle.sin()),
                )
            })
            .collect::<Vec<_>>();
        let error = max_error(&actual, &expected);
        assert!(
            error <= 2.0e-15,
            "DD N323 20 KiB fallback impulse mismatch: {error:e}"
        );
        ir.validate().unwrap();
    }

    #[test]
    fn composite_fft_rader_n391_is_nested_bluestein_boundary() {
        let mut device = DeviceProfile::generic(crate::Backend::Vulkan, crate::GpuVendor::Nvidia);
        device.shared_memory_bytes = 48 * 1024;
        device.shared_memory_pow2_bytes = 32 * 1024;
        device.max_threads_per_block = 1024;
        device.max_workgroup_size = [1024, 1024, 64];
        device.supports_f64 = true;
        let mut tuning = crate::PlannerTuning::portable();
        tuning.min_rader_direct_prime = 29;
        tuning.min_rader_fft_prime = 17;
        tuning.validate().unwrap();
        let plan = FftPlan::build(
            FftConfig::new(vec![391])
                .with_precision(Precision::DoubleDouble)
                .with_tuning(tuning),
        )
        .unwrap();
        let ir = DoubleDoubleRecursiveFftIr::build_for_device(&plan, Direction::Forward, device)
            .unwrap();
        let DoubleDoubleRecursiveFftNodeIr::CooleyTukey(root) = &ir.root else {
            panic!("DD N391 dual-Rader probe should remain Cooley-Tukey");
        };
        assert_eq!((root.left_len, root.right_len), (17, 23));
        let block = root.pack_right.axis_batch_block.unwrap();
        assert_eq!(block.threads_per_transform, 25);
        assert_eq!([block.local_size_x, block.local_size_y], [25, 1]);
        let DoubleDoubleRecursiveFftNodeIr::FftRader(left) = &root.left else {
            panic!("DD N391 left child should be p17 FFT-Rader");
        };
        let DoubleDoubleRecursiveFftNodeIr::FftRader(right) = &root.right else {
            panic!("DD N391 right child should be p23 FFT-Rader");
        };
        assert_eq!((left.prime, left.convolution_len), (17, 16));
        assert_eq!((right.prime, right.convolution_len), (23, 22));
        let DoubleDoubleBluesteinConvolutionIr::Bluestein(right_forward) = &right.forward_fft
        else {
            panic!("DD N391 p23 M22 convolution should enter nested Bluestein");
        };
        let DoubleDoubleBluesteinConvolutionIr::Bluestein(right_inverse) = &right.inverse_fft
        else {
            panic!("DD N391 p23 inverse M22 convolution should enter nested Bluestein");
        };
        assert_eq!(
            (right_forward.logical_len, right_forward.convolution_len),
            (22, 64)
        );
        assert_eq!(
            (right_inverse.logical_len, right_inverse.convolution_len),
            (22, 64)
        );
        assert!(matches!(
            right_forward.forward_fft,
            DoubleDoubleBluesteinConvolutionIr::Stockham(ref child) if child.sequence_len == 64
        ));
        assert!(matches!(
            right_forward.inverse_fft,
            DoubleDoubleBluesteinConvolutionIr::Stockham(ref child) if child.sequence_len == 64
        ));
        assert!(root.fused_dual_fft_rader_supported().unwrap().is_none());
        let program = crate::ProgramIr::double_double_recursive(&ir).unwrap();
        assert_eq!(program.passes.len(), 21);
        let shaders = crate::backend::vulkan::VulkanGlslBackend
            .lower_double_double_recursive(&ir)
            .unwrap();
        assert_eq!(shaders.len(), 21);
        for shader in &shaders {
            shader.compile_spirv().unwrap();
        }

        let mut input = vec![ComplexDoubleDouble::default(); 391];
        input[1] = ComplexDoubleDouble::new(DoubleDouble::from_f64(1.0), DoubleDouble::default());
        let actual = execute_double_double_recursive_ir(&ir, &input).unwrap();
        let expected = (0..391)
            .map(|k| {
                let angle = -std::f64::consts::TAU * k as f64 / 391.0;
                ComplexDoubleDouble::new(
                    DoubleDouble::from_f64(angle.cos()),
                    DoubleDouble::from_f64(angle.sin()),
                )
            })
            .collect::<Vec<_>>();
        let error = max_error(&actual, &expected);
        assert!(
            error <= 2.0e-15,
            "DD N391 dual-Rader impulse mismatch: {error:e}"
        );
        ir.validate().unwrap();
    }

    #[test]
    fn composite_fft_rader_n391_nested_bluestein_remains_fail_soft_at_25k() {
        let mut device = DeviceProfile::generic(crate::Backend::Vulkan, crate::GpuVendor::Nvidia);
        device.shared_memory_bytes = 25 * 1024;
        device.shared_memory_pow2_bytes = 25 * 1024;
        device.max_threads_per_block = 1024;
        device.max_workgroup_size = [1024, 1024, 64];
        device.supports_f64 = true;
        let mut tuning = crate::PlannerTuning::portable();
        tuning.min_rader_direct_prime = 29;
        tuning.min_rader_fft_prime = 17;
        tuning.validate().unwrap();
        let plan = FftPlan::build(
            FftConfig::new(vec![391])
                .with_precision(Precision::DoubleDouble)
                .with_tuning(tuning),
        )
        .unwrap();
        let ir = DoubleDoubleRecursiveFftIr::build_for_device(&plan, Direction::Forward, device)
            .unwrap();
        let DoubleDoubleRecursiveFftNodeIr::CooleyTukey(root) = &ir.root else {
            panic!("DD N391 25 KiB capacity probe should remain Cooley-Tukey");
        };
        assert_eq!((root.left_len, root.right_len), (17, 23));
        let DoubleDoubleRecursiveFftNodeIr::FftRader(right) = &root.right else {
            panic!("DD N391 25 KiB right child should remain p23 FFT-Rader");
        };
        assert!(matches!(
            right.forward_fft,
            DoubleDoubleBluesteinConvolutionIr::Bluestein(ref nested)
                if nested.logical_len == 22 && nested.convolution_len == 64
        ));
        assert!(root.fused_dual_fft_rader_supported().unwrap().is_none());
        assert_eq!(
            crate::ProgramIr::double_double_recursive(&ir)
                .unwrap()
                .passes
                .len(),
            21
        );

        let mut input = vec![ComplexDoubleDouble::default(); 391];
        input[1] = ComplexDoubleDouble::new(DoubleDouble::from_f64(1.0), DoubleDouble::default());
        let actual = execute_double_double_recursive_ir(&ir, &input).unwrap();
        let expected = (0..391)
            .map(|k| {
                let angle = -std::f64::consts::TAU * k as f64 / 391.0;
                ComplexDoubleDouble::new(
                    DoubleDouble::from_f64(angle.cos()),
                    DoubleDouble::from_f64(angle.sin()),
                )
            })
            .collect::<Vec<_>>();
        let error = max_error(&actual, &expected);
        assert!(
            error <= 2.0e-15,
            "DD N391 25 KiB fallback impulse mismatch: {error:e}"
        );
        ir.validate().unwrap();
    }

    #[test]
    fn composite_fft_rader_n493_has_independent_p17_p29_fusion_candidate() {
        let mut device = DeviceProfile::generic(crate::Backend::Vulkan, crate::GpuVendor::Nvidia);
        device.shared_memory_bytes = 48 * 1024;
        device.shared_memory_pow2_bytes = 32 * 1024;
        device.max_threads_per_block = 1024;
        device.max_workgroup_size = [1024, 1024, 64];
        device.supports_f64 = true;
        let mut tuning = crate::PlannerTuning::portable();
        tuning.min_rader_direct_prime = 29;
        tuning.min_rader_fft_prime = 17;
        tuning.validate().unwrap();
        let plan = FftPlan::build(
            FftConfig::new(vec![493])
                .with_precision(Precision::DoubleDouble)
                .with_tuning(tuning),
        )
        .unwrap();
        let ir = DoubleDoubleRecursiveFftIr::build_for_device(&plan, Direction::Forward, device)
            .unwrap();
        let DoubleDoubleRecursiveFftNodeIr::CooleyTukey(root) = &ir.root else {
            panic!("DD N493 dual-Rader probe should remain Cooley-Tukey");
        };
        assert_eq!((root.left_len, root.right_len), (17, 29));
        let block = root.pack_right.axis_batch_block.unwrap();
        assert_eq!(block.threads_per_transform, 31);
        assert_eq!([block.local_size_x, block.local_size_y], [31, 1]);
        let DoubleDoubleRecursiveFftNodeIr::FftRader(left) = &root.left else {
            panic!("DD N493 left child should be p17 FFT-Rader");
        };
        let DoubleDoubleRecursiveFftNodeIr::FftRader(right) = &root.right else {
            panic!("DD N493 right child should be p29 FFT-Rader");
        };
        assert_eq!((left.prime, left.convolution_len), (17, 16));
        assert_eq!((right.prime, right.convolution_len), (29, 28));
        let DoubleDoubleBluesteinConvolutionIr::Stockham(right_forward) = &right.forward_fft else {
            panic!("DD N493 p29 forward convolution should remain Stockham");
        };
        assert_eq!(
            right_forward
                .stages
                .iter()
                .map(|stage| stage.radix)
                .collect::<Vec<_>>(),
            vec![14, 2]
        );
        let fused = root
            .fused_dual_fft_rader_n493()
            .unwrap()
            .expect("DD N493 should expose the independent p17 x p29 fused candidate");
        assert_eq!(fused.required_shared_memory_bytes().unwrap(), 32_864);
        let program = crate::ProgramIr::double_double_recursive(&ir).unwrap();
        assert_eq!(program.passes.len(), 1);
        assert_eq!(program.passes[0].bindings.len(), 5);
        let shaders = crate::backend::vulkan::VulkanGlslBackend
            .lower_double_double_recursive(&ir)
            .unwrap();
        assert_eq!(shaders.len(), 1);
        assert_eq!(shaders[0].required_shared_memory_bytes, 32_864);
        assert_eq!(shaders[0].glsl.matches("barrier();").count(), 7);
        assert!(shaders[0].glsl.contains("VKFFT_COUNT = 28u"));
        assert!(shaders[0].glsl.contains("VKFFT_LEFT_COUNT = 16u"));
        shaders[0].compile_spirv().unwrap();

        let mut input = vec![ComplexDoubleDouble::default(); 493];
        input[1] = ComplexDoubleDouble::new(DoubleDouble::from_f64(1.0), DoubleDouble::default());
        let actual = execute_double_double_recursive_ir(&ir, &input).unwrap();
        let expected = (0..493)
            .map(|k| {
                let angle = -std::f64::consts::TAU * k as f64 / 493.0;
                ComplexDoubleDouble::new(
                    DoubleDouble::from_f64(angle.cos()),
                    DoubleDouble::from_f64(angle.sin()),
                )
            })
            .collect::<Vec<_>>();
        let error = max_error(&actual, &expected);
        assert!(
            error <= 2.0e-15,
            "DD N493 dual-Rader impulse mismatch: {error:e}"
        );
        ir.validate().unwrap();
    }

    #[test]
    fn composite_fft_rader_n493_dual_rader_is_capacity_gated() {
        let mut device = DeviceProfile::generic(crate::Backend::Vulkan, crate::GpuVendor::Nvidia);
        device.shared_memory_bytes = 32 * 1024;
        device.shared_memory_pow2_bytes = 32 * 1024;
        device.max_threads_per_block = 1024;
        device.max_workgroup_size = [1024, 1024, 64];
        device.supports_f64 = true;
        let mut tuning = crate::PlannerTuning::portable();
        tuning.min_rader_direct_prime = 29;
        tuning.min_rader_fft_prime = 17;
        tuning.validate().unwrap();
        let plan = FftPlan::build(
            FftConfig::new(vec![493])
                .with_precision(Precision::DoubleDouble)
                .with_tuning(tuning),
        )
        .unwrap();
        let ir = DoubleDoubleRecursiveFftIr::build_for_device(&plan, Direction::Forward, device)
            .unwrap();
        let DoubleDoubleRecursiveFftNodeIr::CooleyTukey(root) = &ir.root else {
            panic!("DD N493 32 KiB capacity probe should remain Cooley-Tukey");
        };
        assert_eq!((root.left_len, root.right_len), (17, 29));
        assert!(root.fused_dual_fft_rader_n493().unwrap().is_none());
        assert_eq!(
            crate::ProgramIr::double_double_recursive(&ir)
                .unwrap()
                .passes
                .len(),
            13
        );

        let mut input = vec![ComplexDoubleDouble::default(); 493];
        input[1] = ComplexDoubleDouble::new(DoubleDouble::from_f64(1.0), DoubleDouble::default());
        let actual = execute_double_double_recursive_ir(&ir, &input).unwrap();
        let expected = (0..493)
            .map(|k| {
                let angle = -std::f64::consts::TAU * k as f64 / 493.0;
                ComplexDoubleDouble::new(
                    DoubleDouble::from_f64(angle.cos()),
                    DoubleDouble::from_f64(angle.sin()),
                )
            })
            .collect::<Vec<_>>();
        let error = max_error(&actual, &expected);
        assert!(
            error <= 2.0e-15,
            "DD N493 32 KiB fallback impulse mismatch: {error:e}"
        );
        ir.validate().unwrap();
    }

    #[test]
    fn composite_fft_rader_n527_has_independent_p17_p31_fusion_candidate() {
        let mut device = DeviceProfile::generic(crate::Backend::Vulkan, crate::GpuVendor::Nvidia);
        device.shared_memory_bytes = 48 * 1024;
        device.shared_memory_pow2_bytes = 32 * 1024;
        device.max_threads_per_block = 1024;
        device.max_workgroup_size = [1024, 1024, 64];
        device.supports_f64 = true;
        let mut tuning = crate::PlannerTuning::portable();
        tuning.min_rader_direct_prime = 29;
        tuning.min_rader_fft_prime = 17;
        tuning.validate().unwrap();
        let plan = FftPlan::build(
            FftConfig::new(vec![527])
                .with_precision(Precision::DoubleDouble)
                .with_tuning(tuning),
        )
        .unwrap();
        let ir = DoubleDoubleRecursiveFftIr::build_for_device(&plan, Direction::Forward, device)
            .unwrap();
        let DoubleDoubleRecursiveFftNodeIr::CooleyTukey(root) = &ir.root else {
            panic!("DD N527 dual-Rader probe should remain Cooley-Tukey");
        };
        assert_eq!((root.left_len, root.right_len), (17, 31));
        let block = root.pack_right.axis_batch_block.unwrap();
        assert_eq!(block.threads_per_transform, 33);
        assert_eq!([block.local_size_x, block.local_size_y], [33, 1]);
        let DoubleDoubleRecursiveFftNodeIr::FftRader(left) = &root.left else {
            panic!("DD N527 left child should be p17 FFT-Rader");
        };
        let DoubleDoubleRecursiveFftNodeIr::FftRader(right) = &root.right else {
            panic!("DD N527 right child should be p31 FFT-Rader");
        };
        assert_eq!((left.prime, left.convolution_len), (17, 16));
        assert_eq!((right.prime, right.convolution_len), (31, 30));
        let DoubleDoubleBluesteinConvolutionIr::Stockham(right_forward) = &right.forward_fft else {
            panic!("DD N527 p31 forward convolution should remain Stockham");
        };
        assert_eq!(
            right_forward
                .stages
                .iter()
                .map(|stage| stage.radix)
                .collect::<Vec<_>>(),
            vec![15, 2]
        );
        let fused = root
            .fused_dual_fft_rader_n527()
            .unwrap()
            .expect("DD N527 should expose the independent p17 x p31 fused candidate");
        assert_eq!(fused.required_shared_memory_bytes().unwrap(), 35_168);
        let program = crate::ProgramIr::double_double_recursive(&ir).unwrap();
        assert_eq!(program.passes.len(), 1);
        assert_eq!(program.passes[0].bindings.len(), 5);
        let shaders = crate::backend::vulkan::VulkanGlslBackend
            .lower_double_double_recursive(&ir)
            .unwrap();
        assert_eq!(shaders.len(), 1);
        assert_eq!(shaders[0].required_shared_memory_bytes, 35_168);
        assert_eq!(shaders[0].glsl.matches("barrier();").count(), 7);
        assert!(shaders[0].glsl.contains("VKFFT_COUNT = 30u"));
        assert!(shaders[0].glsl.contains("VKFFT_LEFT_COUNT = 16u"));
        shaders[0].compile_spirv().unwrap();

        let mut input = vec![ComplexDoubleDouble::default(); 527];
        input[1] = ComplexDoubleDouble::new(DoubleDouble::from_f64(1.0), DoubleDouble::default());
        let actual = execute_double_double_recursive_ir(&ir, &input).unwrap();
        let expected = (0..527)
            .map(|k| {
                let angle = -std::f64::consts::TAU * k as f64 / 527.0;
                ComplexDoubleDouble::new(
                    DoubleDouble::from_f64(angle.cos()),
                    DoubleDouble::from_f64(angle.sin()),
                )
            })
            .collect::<Vec<_>>();
        let error = max_error(&actual, &expected);
        assert!(
            error <= 2.0e-15,
            "DD N527 dual-Rader impulse mismatch: {error:e}"
        );
        ir.validate().unwrap();
    }

    #[test]
    fn composite_fft_rader_n527_dual_rader_is_capacity_gated() {
        let mut device = DeviceProfile::generic(crate::Backend::Vulkan, crate::GpuVendor::Nvidia);
        device.shared_memory_bytes = 34 * 1024;
        device.shared_memory_pow2_bytes = 34 * 1024;
        device.max_threads_per_block = 1024;
        device.max_workgroup_size = [1024, 1024, 64];
        device.supports_f64 = true;
        let mut tuning = crate::PlannerTuning::portable();
        tuning.min_rader_direct_prime = 29;
        tuning.min_rader_fft_prime = 17;
        tuning.validate().unwrap();
        let plan = FftPlan::build(
            FftConfig::new(vec![527])
                .with_precision(Precision::DoubleDouble)
                .with_tuning(tuning),
        )
        .unwrap();
        let ir = DoubleDoubleRecursiveFftIr::build_for_device(&plan, Direction::Forward, device)
            .unwrap();
        let DoubleDoubleRecursiveFftNodeIr::CooleyTukey(root) = &ir.root else {
            panic!("DD N527 34 KiB capacity probe should remain Cooley-Tukey");
        };
        assert_eq!((root.left_len, root.right_len), (17, 31));
        assert!(root.fused_dual_fft_rader_n527().unwrap().is_none());
        assert_eq!(
            crate::ProgramIr::double_double_recursive(&ir)
                .unwrap()
                .passes
                .len(),
            13
        );

        let mut input = vec![ComplexDoubleDouble::default(); 527];
        input[1] = ComplexDoubleDouble::new(DoubleDouble::from_f64(1.0), DoubleDouble::default());
        let actual = execute_double_double_recursive_ir(&ir, &input).unwrap();
        let expected = (0..527)
            .map(|k| {
                let angle = -std::f64::consts::TAU * k as f64 / 527.0;
                ComplexDoubleDouble::new(
                    DoubleDouble::from_f64(angle.cos()),
                    DoubleDouble::from_f64(angle.sin()),
                )
            })
            .collect::<Vec<_>>();
        let error = max_error(&actual, &expected);
        assert!(
            error <= 2.0e-15,
            "DD N527 34 KiB fallback impulse mismatch: {error:e}"
        );
        ir.validate().unwrap();
    }

    #[test]
    fn composite_fft_rader_n306_multistage_has_strict_dd_fusion_candidate() {
        let mut device = DeviceProfile::generic(crate::Backend::Vulkan, crate::GpuVendor::Nvidia);
        device.shared_memory_bytes = 48 * 1024;
        device.shared_memory_pow2_bytes = 32 * 1024;
        device.max_threads_per_block = 1024;
        device.max_workgroup_size = [1024, 1024, 64];
        device.supports_f64 = true;
        let mut tuning = crate::PlannerTuning::portable();
        tuning.min_rader_direct_prime = 29;
        tuning.min_rader_fft_prime = 17;
        tuning.validate().unwrap();
        let plan = FftPlan::build(
            FftConfig::new(vec![18usize * 17])
                .with_precision(Precision::DoubleDouble)
                .with_tuning(tuning),
        )
        .unwrap();
        let ir = DoubleDoubleRecursiveFftIr::build_for_device(&plan, Direction::Forward, device)
            .unwrap();
        let DoubleDoubleRecursiveFftNodeIr::CooleyTukey(root) = &ir.root else {
            panic!("DD N306 FFT-Rader probe should keep an 18 x p17 Cooley root");
        };
        assert_eq!((root.left_len, root.right_len), (18, 17));
        let DoubleDoubleRecursiveFftNodeIr::Stockham(stockham) = &root.left else {
            panic!("DD N306 left child should be multi-stage Stockham");
        };
        assert_eq!(
            stockham
                .stages
                .iter()
                .map(|stage| (
                    stage.index,
                    stage.radix,
                    stage.stage_size,
                    stage.butterflies
                ))
                .collect::<Vec<_>>(),
            vec![(0, 9, 1, 2), (1, 2, 9, 9)]
        );
        let DoubleDoubleRecursiveFftNodeIr::FftRader(rader) = &root.right else {
            panic!("DD N306 right child should be p17 FFT-Rader");
        };
        assert_eq!(rader.prime, 17);
        assert_eq!(rader.convolution_len, 16);
        let DoubleDoubleBluesteinConvolutionIr::Stockham(forward) = &rader.forward_fft else {
            panic!("DD N306 p17 forward convolution should be Stockham");
        };
        let DoubleDoubleBluesteinConvolutionIr::Stockham(inverse) = &rader.inverse_fft else {
            panic!("DD N306 p17 inverse convolution should be Stockham");
        };
        assert_eq!((forward.batch_count, forward.grouped_batch), (18, 18));
        assert_eq!((inverse.batch_count, inverse.grouped_batch), (18, 18));
        assert_eq!(
            root.pack_right.axis_batch_block,
            Some(StockhamAxisBlockSchedule {
                threads_per_transform: 26,
                grouped_batch: 1,
                transforms_on_x: false,
                axis_swapped: false,
                local_size_x: 26,
                local_size_y: 1,
            })
        );
        let fused = root
            .fused_small_fft_rader_stockham()
            .unwrap()
            .expect("DD N306 should expose the strict multi-stage p17 fusion candidate");
        assert_eq!(fused.required_shared_memory_bytes().unwrap(), 28_800);
        let program = crate::ProgramIr::double_double_recursive(&ir).unwrap();
        assert_eq!(program.passes.len(), 1);
        assert_eq!(program.passes[0].name, fused.name());
        assert_eq!(program.passes[0].bindings.len(), 4);

        let shaders = crate::backend::vulkan::VulkanGlslBackend
            .lower_double_double_recursive(&ir)
            .unwrap();
        assert_eq!(shaders.len(), 1);
        assert_eq!(shaders[0].required_shared_memory_bytes, 28_800);
        assert_eq!(shaders[0].glsl.matches("barrier();").count(), 6);
        assert!(
            shaders[0]
                .glsl
                .contains("fused left Stockham stage 0: radix=9")
        );
        assert!(
            shaders[0]
                .glsl
                .contains("fused left Stockham stage 1: radix=2")
        );
        shaders[0].compile_spirv().unwrap();

        let length = 306usize;
        let mut input = vec![ComplexDoubleDouble::default(); length];
        input[1] = ComplexDoubleDouble::new(DoubleDouble::from_f64(1.0), DoubleDouble::default());
        let actual = execute_double_double_recursive_ir(&ir, &input).unwrap();
        let expected = (0..length)
            .map(|k| {
                let angle = -std::f64::consts::TAU * k as f64 / length as f64;
                ComplexDoubleDouble::new(
                    DoubleDouble::from_f64(angle.cos()),
                    DoubleDouble::from_f64(angle.sin()),
                )
            })
            .collect::<Vec<_>>();
        let error = max_error(&actual, &expected);
        assert!(
            error <= 2.0e-15,
            "DD N306 FFT-Rader impulse mismatch: {error:e}"
        );
        ir.validate().unwrap();
    }

    #[test]
    fn composite_fft_rader_n306_multistage_capacity_fails_soft() {
        fn build(
            shared_memory_bytes: usize,
            shared_memory_pow2_bytes: usize,
        ) -> DoubleDoubleRecursiveFftIr {
            let mut device =
                DeviceProfile::generic(crate::Backend::Vulkan, crate::GpuVendor::Nvidia);
            device.shared_memory_bytes = shared_memory_bytes;
            device.shared_memory_pow2_bytes = shared_memory_pow2_bytes;
            device.max_threads_per_block = 1024;
            device.max_workgroup_size = [1024, 1024, 64];
            device.supports_f64 = true;
            let mut tuning = crate::PlannerTuning::portable();
            tuning.min_rader_direct_prime = 29;
            tuning.min_rader_fft_prime = 17;
            tuning.validate().unwrap();
            let plan = FftPlan::build(
                FftConfig::new(vec![18usize * 17])
                    .with_precision(Precision::DoubleDouble)
                    .with_tuning(tuning),
            )
            .unwrap();
            DoubleDoubleRecursiveFftIr::build_for_device(&plan, Direction::Forward, device).unwrap()
        }

        let roomy = build(32 * 1024, 32 * 1024);
        let DoubleDoubleRecursiveFftNodeIr::CooleyTukey(roomy_root) = &roomy.root else {
            panic!("DD N306 32 KiB capacity witness should remain Cooley-Tukey");
        };
        let roomy_fused = roomy_root
            .fused_small_fft_rader_stockham()
            .unwrap()
            .expect("DD N306 should fit its 28.8 KiB fused footprint in 32 KiB");
        assert_eq!(roomy_fused.required_shared_memory_bytes().unwrap(), 28_800);
        assert_eq!(
            crate::ProgramIr::double_double_recursive(&roomy)
                .unwrap()
                .passes
                .len(),
            1
        );

        let tight = build(24 * 1024, 16 * 1024);
        let DoubleDoubleRecursiveFftNodeIr::CooleyTukey(tight_root) = &tight.root else {
            panic!("DD N306 24 KiB capacity witness should remain Cooley-Tukey");
        };
        assert_eq!((tight_root.left_len, tight_root.right_len), (18, 17));
        assert!(
            tight_root
                .fused_small_fft_rader_stockham()
                .unwrap()
                .is_none()
        );
        let tight_program = crate::ProgramIr::double_double_recursive(&tight).unwrap();
        assert!(tight_program.passes.len() > 1);
        let tight_shaders = crate::backend::vulkan::VulkanGlslBackend
            .lower_double_double_recursive(&tight)
            .unwrap();
        assert!(tight_shaders.len() > 1);
        for shader in &tight_shaders {
            assert!(shader.required_shared_memory_bytes <= 24 * 1024);
            shader.compile_spirv().unwrap();
        }
        roomy.validate().unwrap();
        tight.validate().unwrap();
    }

    #[test]
    fn composite_fft_rader_n340_multistage_has_strict_dd_fusion_candidate() {
        let mut device = DeviceProfile::generic(crate::Backend::Vulkan, crate::GpuVendor::Nvidia);
        device.shared_memory_bytes = 48 * 1024;
        device.shared_memory_pow2_bytes = 32 * 1024;
        device.max_threads_per_block = 1024;
        device.max_workgroup_size = [1024, 1024, 64];
        device.supports_f64 = true;
        let mut tuning = crate::PlannerTuning::portable();
        tuning.min_rader_direct_prime = 29;
        tuning.min_rader_fft_prime = 17;
        tuning.validate().unwrap();
        let plan = FftPlan::build(
            FftConfig::new(vec![20usize * 17])
                .with_precision(Precision::DoubleDouble)
                .with_tuning(tuning),
        )
        .unwrap();
        let ir = DoubleDoubleRecursiveFftIr::build_for_device(&plan, Direction::Forward, device)
            .unwrap();
        let DoubleDoubleRecursiveFftNodeIr::CooleyTukey(root) = &ir.root else {
            panic!("DD N340 FFT-Rader probe should keep a 20 x p17 Cooley root");
        };
        assert_eq!((root.left_len, root.right_len), (20, 17));
        let DoubleDoubleRecursiveFftNodeIr::Stockham(stockham) = &root.left else {
            panic!("DD N340 left child should be multi-stage Stockham");
        };
        assert_eq!(
            stockham
                .stages
                .iter()
                .map(|stage| (
                    stage.index,
                    stage.radix,
                    stage.stage_size,
                    stage.butterflies
                ))
                .collect::<Vec<_>>(),
            vec![(0, 10, 1, 2), (1, 2, 10, 10)]
        );
        let DoubleDoubleRecursiveFftNodeIr::FftRader(rader) = &root.right else {
            panic!("DD N340 right child should be p17 FFT-Rader");
        };
        assert_eq!(rader.prime, 17);
        assert_eq!(rader.convolution_len, 16);
        let DoubleDoubleBluesteinConvolutionIr::Stockham(forward) = &rader.forward_fft else {
            panic!("DD N340 p17 forward convolution should be Stockham");
        };
        let DoubleDoubleBluesteinConvolutionIr::Stockham(inverse) = &rader.inverse_fft else {
            panic!("DD N340 p17 inverse convolution should be Stockham");
        };
        assert_eq!((forward.batch_count, forward.grouped_batch), (20, 20));
        assert_eq!((inverse.batch_count, inverse.grouped_batch), (20, 20));
        assert_eq!(
            root.pack_right.axis_batch_block,
            Some(StockhamAxisBlockSchedule {
                threads_per_transform: 22,
                grouped_batch: 1,
                transforms_on_x: false,
                axis_swapped: false,
                local_size_x: 22,
                local_size_y: 1,
            })
        );
        let fused = root
            .fused_small_fft_rader_stockham()
            .unwrap()
            .expect("DD N340 should expose the strict multi-stage p17 fusion candidate");
        assert_eq!(fused.required_shared_memory_bytes().unwrap(), 32_000);
        let program = crate::ProgramIr::double_double_recursive(&ir).unwrap();
        assert_eq!(program.passes.len(), 1);
        assert_eq!(program.passes[0].name, fused.name());
        assert_eq!(program.passes[0].bindings.len(), 4);

        let shaders = crate::backend::vulkan::VulkanGlslBackend
            .lower_double_double_recursive(&ir)
            .unwrap();
        assert_eq!(shaders.len(), 1);
        assert_eq!(shaders[0].required_shared_memory_bytes, 32_000);
        assert_eq!(shaders[0].glsl.matches("barrier();").count(), 6);
        assert!(
            shaders[0]
                .glsl
                .contains("fused left Stockham stage 0: radix=10")
        );
        assert!(
            shaders[0]
                .glsl
                .contains("fused left Stockham stage 1: radix=2")
        );
        shaders[0].compile_spirv().unwrap();

        let length = 340usize;
        let mut input = vec![ComplexDoubleDouble::default(); length];
        input[1] = ComplexDoubleDouble::new(DoubleDouble::from_f64(1.0), DoubleDouble::default());
        let actual = execute_double_double_recursive_ir(&ir, &input).unwrap();
        let expected = (0..length)
            .map(|k| {
                let angle = -std::f64::consts::TAU * k as f64 / length as f64;
                ComplexDoubleDouble::new(
                    DoubleDouble::from_f64(angle.cos()),
                    DoubleDouble::from_f64(angle.sin()),
                )
            })
            .collect::<Vec<_>>();
        let error = max_error(&actual, &expected);
        assert!(
            error <= 2.0e-15,
            "DD N340 FFT-Rader impulse mismatch: {error:e}"
        );
        ir.validate().unwrap();
    }

    #[test]
    fn composite_fft_rader_n340_multistage_capacity_fails_soft() {
        fn build(
            shared_memory_bytes: usize,
            shared_memory_pow2_bytes: usize,
        ) -> DoubleDoubleRecursiveFftIr {
            let mut device =
                DeviceProfile::generic(crate::Backend::Vulkan, crate::GpuVendor::Nvidia);
            device.shared_memory_bytes = shared_memory_bytes;
            device.shared_memory_pow2_bytes = shared_memory_pow2_bytes;
            device.max_threads_per_block = 1024;
            device.max_workgroup_size = [1024, 1024, 64];
            device.supports_f64 = true;
            let mut tuning = crate::PlannerTuning::portable();
            tuning.min_rader_direct_prime = 29;
            tuning.min_rader_fft_prime = 17;
            tuning.validate().unwrap();
            let plan = FftPlan::build(
                FftConfig::new(vec![20usize * 17])
                    .with_precision(Precision::DoubleDouble)
                    .with_tuning(tuning),
            )
            .unwrap();
            DoubleDoubleRecursiveFftIr::build_for_device(&plan, Direction::Forward, device).unwrap()
        }

        let roomy = build(32 * 1024, 32 * 1024);
        let DoubleDoubleRecursiveFftNodeIr::CooleyTukey(roomy_root) = &roomy.root else {
            panic!("DD N340 32 KiB capacity witness should remain Cooley-Tukey");
        };
        assert_eq!((roomy_root.left_len, roomy_root.right_len), (20, 17));
        let roomy_fused = roomy_root
            .fused_small_fft_rader_stockham()
            .unwrap()
            .expect("DD N340 should fit its 32,000 B fused footprint in 32 KiB");
        assert_eq!(roomy_fused.required_shared_memory_bytes().unwrap(), 32_000);
        assert_eq!(
            crate::ProgramIr::double_double_recursive(&roomy)
                .unwrap()
                .passes
                .len(),
            1
        );

        let tight = build(30 * 1024, 16 * 1024);
        let DoubleDoubleRecursiveFftNodeIr::CooleyTukey(tight_root) = &tight.root else {
            panic!("DD N340 30 KiB capacity witness should remain Cooley-Tukey");
        };
        assert_eq!((tight_root.left_len, tight_root.right_len), (20, 17));
        assert!(
            tight_root
                .fused_small_fft_rader_stockham()
                .unwrap()
                .is_none()
        );
        let tight_program = crate::ProgramIr::double_double_recursive(&tight).unwrap();
        assert!(tight_program.passes.len() > 1);
        let tight_shaders = crate::backend::vulkan::VulkanGlslBackend
            .lower_double_double_recursive(&tight)
            .unwrap();
        assert!(tight_shaders.len() > 1);
        for shader in &tight_shaders {
            assert!(shader.required_shared_memory_bytes <= 30 * 1024);
            shader.compile_spirv().unwrap();
        }
        roomy.validate().unwrap();
        tight.validate().unwrap();
    }

    #[test]
    fn composite_fft_rader_p17_multistage_surface_n340_through_n510() {
        let mut device = DeviceProfile::generic(crate::Backend::Vulkan, crate::GpuVendor::Nvidia);
        device.shared_memory_bytes = 48 * 1024;
        device.shared_memory_pow2_bytes = 32 * 1024;
        device.max_threads_per_block = 1024;
        device.max_workgroup_size = [1024, 1024, 64];
        device.supports_f64 = true;
        let mut tuning = crate::PlannerTuning::portable();
        tuning.min_rader_direct_prime = 29;
        tuning.min_rader_fft_prime = 17;
        tuning.validate().unwrap();

        let cases = [
            (20usize, 10usize, 2usize, 22usize, 32_000usize),
            (21, 7, 3, 30, 33_600),
            (22, 11, 2, 128, 35_200),
            (24, 12, 2, 34, 38_400),
            (25, 5, 5, 29, 40_000),
            (26, 13, 2, 128, 41_600),
            (27, 9, 3, 51, 43_200),
            (28, 14, 2, 34, 44_800),
            (30, 15, 2, 34, 48_000),
        ];
        for (left_len, radix0, radix1, lanes, expected_shared) in cases {
            let length = left_len * 17;
            let plan = FftPlan::build(
                FftConfig::new(vec![length])
                    .with_precision(Precision::DoubleDouble)
                    .with_tuning(tuning),
            )
            .unwrap();
            let ir =
                DoubleDoubleRecursiveFftIr::build_for_device(&plan, Direction::Forward, device)
                    .unwrap();
            let DoubleDoubleRecursiveFftNodeIr::CooleyTukey(root) = &ir.root else {
                panic!("DD N{length} p17 multi-stage surface should remain Cooley-Tukey");
            };
            assert_eq!((root.left_len, root.right_len), (left_len, 17));
            let DoubleDoubleRecursiveFftNodeIr::Stockham(stockham) = &root.left else {
                panic!("DD N{length} left child should remain Stockham");
            };
            assert_eq!(
                stockham
                    .stages
                    .iter()
                    .map(|stage| (
                        stage.index,
                        stage.radix,
                        stage.stage_size,
                        stage.butterflies
                    ))
                    .collect::<Vec<_>>(),
                vec![
                    (0, radix0, 1, left_len / radix0),
                    (1, radix1, radix0, left_len / radix1),
                ]
            );
            assert!(matches!(
                root.right,
                DoubleDoubleRecursiveFftNodeIr::FftRader(ref rader)
                    if rader.prime == 17 && rader.convolution_len == 16
            ));
            assert_eq!(
                root.pack_right
                    .axis_batch_block
                    .expect("DD p17 multi-stage surface should retain a physical axis block")
                    .threads_per_transform,
                lanes
            );
            let fused = root
                .fused_small_fft_rader_stockham()
                .unwrap()
                .expect("DD p17 multi-stage surface should expose the strict fusion candidate");
            assert_eq!(
                fused.required_shared_memory_bytes().unwrap(),
                expected_shared
            );
            let program = crate::ProgramIr::double_double_recursive(&ir).unwrap();
            assert_eq!(program.passes.len(), 1);
            assert_eq!(program.passes[0].bindings.len(), 4);
            let shaders = crate::backend::vulkan::VulkanGlslBackend
                .lower_double_double_recursive(&ir)
                .unwrap();
            assert_eq!(shaders.len(), 1);
            assert_eq!(shaders[0].required_shared_memory_bytes, expected_shared);
            assert_eq!(shaders[0].glsl.matches("barrier();").count(), 6);
            shaders[0].compile_spirv().unwrap();

            let mut input = vec![ComplexDoubleDouble::default(); length];
            input[1] =
                ComplexDoubleDouble::new(DoubleDouble::from_f64(1.0), DoubleDouble::default());
            let actual = execute_double_double_recursive_ir(&ir, &input).unwrap();
            let expected = (0..length)
                .map(|k| {
                    let angle = -std::f64::consts::TAU * k as f64 / length as f64;
                    ComplexDoubleDouble::new(
                        DoubleDouble::from_f64(angle.cos()),
                        DoubleDouble::from_f64(angle.sin()),
                    )
                })
                .collect::<Vec<_>>();
            let error = max_error(&actual, &expected);
            assert!(
                error <= 2.0e-15,
                "DD N{length} p17 multi-stage surface impulse mismatch: {error:e}"
            );
            ir.validate().unwrap();
        }
    }

    #[test]
    fn composite_fft_rader_n544_multistage_is_capacity_gated() {
        fn build(shared_memory_bytes: usize) -> DoubleDoubleRecursiveFftIr {
            let mut device =
                DeviceProfile::generic(crate::Backend::Vulkan, crate::GpuVendor::Nvidia);
            device.shared_memory_bytes = shared_memory_bytes;
            device.shared_memory_pow2_bytes = if shared_memory_bytes.is_power_of_two() {
                shared_memory_bytes
            } else {
                shared_memory_bytes.next_power_of_two() / 2
            };
            device.max_threads_per_block = 1024;
            device.max_workgroup_size = [1024, 1024, 64];
            device.supports_f64 = true;
            let mut tuning = crate::PlannerTuning::portable();
            tuning.min_rader_direct_prime = 29;
            tuning.min_rader_fft_prime = 17;
            tuning.validate().unwrap();
            let plan = FftPlan::build(
                FftConfig::new(vec![32usize * 17])
                    .with_precision(Precision::DoubleDouble)
                    .with_tuning(tuning),
            )
            .unwrap();
            DoubleDoubleRecursiveFftIr::build_for_device(&plan, Direction::Forward, device).unwrap()
        }

        let tight = build(48 * 1024);
        let DoubleDoubleRecursiveFftNodeIr::CooleyTukey(tight_root) = &tight.root else {
            panic!("DD N544 48 KiB capacity witness should remain Cooley-Tukey");
        };
        assert_eq!((tight_root.left_len, tight_root.right_len), (32, 17));
        let DoubleDoubleRecursiveFftNodeIr::Stockham(tight_stockham) = &tight_root.left else {
            panic!("DD N544 left child should remain Stockham");
        };
        assert_eq!(
            tight_stockham
                .stages
                .iter()
                .map(|stage| stage.radix)
                .collect::<Vec<_>>(),
            vec![16, 2]
        );
        assert!(
            tight_root
                .fused_small_fft_rader_stockham()
                .unwrap()
                .is_none()
        );
        let tight_program = crate::ProgramIr::double_double_recursive(&tight).unwrap();
        assert!(tight_program.passes.len() > 1);
        let tight_shaders = crate::backend::vulkan::VulkanGlslBackend
            .lower_double_double_recursive(&tight)
            .unwrap();
        assert!(tight_shaders.len() > 1);
        for shader in &tight_shaders {
            assert!(shader.required_shared_memory_bytes <= 48 * 1024);
            shader.compile_spirv().unwrap();
        }

        let roomy = build(64 * 1024);
        let DoubleDoubleRecursiveFftNodeIr::CooleyTukey(roomy_root) = &roomy.root else {
            panic!("DD N544 64 KiB capacity witness should remain Cooley-Tukey");
        };
        assert_eq!((roomy_root.left_len, roomy_root.right_len), (32, 17));
        let fused = roomy_root
            .fused_small_fft_rader_stockham()
            .unwrap()
            .expect("DD N544 should fuse when its 51,200 B footprint fits");
        assert_eq!(fused.required_shared_memory_bytes().unwrap(), 51_200);
        let roomy_program = crate::ProgramIr::double_double_recursive(&roomy).unwrap();
        assert_eq!(roomy_program.passes.len(), 1);
        let roomy_shaders = crate::backend::vulkan::VulkanGlslBackend
            .lower_double_double_recursive(&roomy)
            .unwrap();
        assert_eq!(roomy_shaders.len(), 1);
        assert_eq!(roomy_shaders[0].required_shared_memory_bytes, 51_200);
        assert_eq!(roomy_shaders[0].glsl.matches("barrier();").count(), 6);
        roomy_shaders[0].compile_spirv().unwrap();
        tight.validate().unwrap();
        roomy.validate().unwrap();
    }

    #[test]
    fn composite_fft_rader_n51_strict_scope_falls_back_fail_soft() {
        fn tuning() -> crate::PlannerTuning {
            let mut tuning = crate::PlannerTuning::portable();
            tuning.min_rader_direct_prime = 29;
            tuning.min_rader_fft_prime = 17;
            tuning.validate().unwrap();
            tuning
        }

        fn build(
            precision: Precision,
            batch_count: usize,
            device: DeviceProfile,
        ) -> DoubleDoubleRecursiveFftIr {
            let plan = FftPlan::build(
                FftConfig::new(vec![3usize * 17])
                    .with_batch_count(batch_count)
                    .with_precision(precision)
                    .with_tuning(tuning()),
            )
            .unwrap();
            DoubleDoubleRecursiveFftIr::build_for_device(&plan, Direction::Forward, device).unwrap()
        }

        fn assert_unfused(ir: &DoubleDoubleRecursiveFftIr, shared_budget: usize) {
            let DoubleDoubleRecursiveFftNodeIr::CooleyTukey(root) = &ir.root else {
                panic!("DD N51 strict-scope fallback should keep a Cooley root");
            };
            assert_eq!((root.left_len, root.right_len), (3, 17));
            assert!(root.fused_small_fft_rader_stockham().unwrap().is_none());
            let program = crate::ProgramIr::double_double_recursive(ir).unwrap();
            assert!(program.passes.len() > 1);
            let shaders = crate::backend::vulkan::VulkanGlslBackend
                .lower_double_double_recursive(ir)
                .unwrap();
            assert_eq!(shaders.len(), program.passes.len());
            assert!(
                shaders
                    .iter()
                    .all(|shader| shader.required_shared_memory_bytes <= shared_budget)
            );
            for shader in shaders {
                shader.compile_spirv().unwrap();
            }
        }

        let mut roomy = DeviceProfile::generic(crate::Backend::Vulkan, crate::GpuVendor::Nvidia);
        roomy.shared_memory_bytes = 48 * 1024;
        roomy.shared_memory_pow2_bytes = 32 * 1024;
        roomy.max_threads_per_block = 1024;
        roomy.max_workgroup_size = [1024, 1024, 64];
        roomy.supports_f64 = true;

        let f64_storage = build(Precision::DoubleDoubleF64Storage, 1, roomy);
        assert_unfused(&f64_storage, roomy.shared_memory_bytes);

        let batched = build(Precision::DoubleDouble, 2, roomy);
        assert_unfused(&batched, roomy.shared_memory_bytes);

        let mut constrained = roomy;
        constrained.shared_memory_bytes = 3 * 1024;
        constrained.shared_memory_pow2_bytes = 3 * 1024;
        let capacity_limited = build(Precision::DoubleDouble, 1, constrained);
        assert_unfused(&capacity_limited, constrained.shared_memory_bytes);
    }

    #[test]
    fn device_default_min_direct_11_keeps_nested_p13_pressure() {
        let mut device = DeviceProfile::generic(crate::Backend::Vulkan, crate::GpuVendor::Nvidia);
        device.shared_memory_bytes = 48 * 1024;
        device.shared_memory_pow2_bytes = 32 * 1024;
        device.max_threads_per_block = 1024;
        device.max_workgroup_size = [1024, 1024, 64];
        device.supports_f64 = true;
        let length = 2usize * 53;
        let batch_count = 2usize;
        let tuning = crate::PlannerTuning::for_device(device, Precision::DoubleDouble)
            .with_recursive_fft_rader(true);
        assert_eq!(tuning.min_rader_direct_prime, 11);
        assert_eq!(tuning.max_rader_direct_prime, 29);
        let plan = FftPlan::build_for_device(
            FftConfig::new(vec![length])
                .with_batch_count(batch_count)
                .with_precision(Precision::DoubleDouble)
                .with_tuning(tuning),
            device,
        )
        .unwrap();
        let crate::planner::AxisAlgorithm::Rader { primes, .. } = &plan.axes[0].algorithm else {
            panic!("DD N106 should retain p53 recursive FFT-Rader under explicit recursive tuning");
        };
        assert!(matches!(
            primes.as_slice(),
            [crate::planner::RaderPrimePlan {
                prime: 53,
                mode: crate::planner::RaderMode::FftConvolution { .. },
                ..
            }]
        ));

        let ir = DoubleDoubleRecursiveFftIr::build_for_device(&plan, Direction::Forward, device)
            .unwrap();
        let DoubleDoubleRecursiveFftNodeIr::CooleyTukey(root) = &ir.root else {
            panic!("DD N106 should keep smooth-2 x p53 Cooley root");
        };
        let block = root.pack_right.axis_batch_block.unwrap();
        assert_eq!(block.threads_per_transform, 9);
        assert_eq!(block.grouped_batch, 1);
        assert_eq!([block.local_size_x, block.local_size_y], [9, 1]);
        assert_eq!(root.twiddle_transpose.axis_batch_block, Some(block));
        assert_eq!(root.scatter_output.axis_batch_block, Some(block));

        let outer = match (&root.left, &root.right) {
            (DoubleDoubleRecursiveFftNodeIr::FftRader(rader), _) if rader.prime == 53 => rader,
            (_, DoubleDoubleRecursiveFftNodeIr::FftRader(rader)) if rader.prime == 53 => rader,
            _ => panic!("DD N106 should contain p53 FFT-Rader child"),
        };
        let DoubleDoubleBluesteinConvolutionIr::Recursive(convolution) = &outer.forward_fft else {
            panic!("DD p53 convolution should remain recursive 52=4x13");
        };
        fn contains_direct_prime(node: &DoubleDoubleRecursiveFftNodeIr, prime: usize) -> bool {
            match node {
                DoubleDoubleRecursiveFftNodeIr::DirectRader(direct) => direct.prime == prime,
                DoubleDoubleRecursiveFftNodeIr::CooleyTukey(cooley) => {
                    contains_direct_prime(&cooley.left, prime)
                        || contains_direct_prime(&cooley.right, prime)
                }
                DoubleDoubleRecursiveFftNodeIr::Stockham(_)
                | DoubleDoubleRecursiveFftNodeIr::FftRader(_)
                | DoubleDoubleRecursiveFftNodeIr::Bluestein(_) => false,
            }
        }
        assert!(contains_direct_prime(&convolution.root, 13));

        let shaders = crate::backend::vulkan::VulkanGlslBackend
            .lower_double_double_recursive(&ir)
            .unwrap();
        for shader in shaders {
            shader.compile_spirv().unwrap();
        }
        let mut input = vec![ComplexDoubleDouble::default(); length * batch_count];
        for batch in 0..batch_count {
            input[batch * length + 1] =
                ComplexDoubleDouble::new(DoubleDouble::from_f64(1.0), DoubleDouble::default());
        }
        let actual = execute_double_double_recursive_ir(&ir, &input).unwrap();
        let error = actual
            .iter()
            .enumerate()
            .map(|(index, value)| {
                let k = index % length;
                let angle = -std::f64::consts::TAU * k as f64 / length as f64;
                let expected = ComplexDoubleDouble::new(
                    DoubleDouble::from_f64(angle.cos()),
                    DoubleDouble::from_f64(angle.sin()),
                );
                let dr = (value.re - expected.re).to_f64();
                let di = (value.im - expected.im).to_f64();
                dr.hypot(di)
            })
            .fold(0.0f64, f64::max);
        assert!(
            error <= 3.0e-14,
            "DD N106 minDirect=11 nested p13 impulse mismatch: {error:e}"
        );
        ir.validate().unwrap();
    }

    #[test]
    fn device_default_nested_fft_then_direct_order_reaches_n886_parent() {
        let mut device = DeviceProfile::generic(crate::Backend::Vulkan, crate::GpuVendor::Nvidia);
        device.shared_memory_bytes = 48 * 1024;
        device.shared_memory_pow2_bytes = 32 * 1024;
        device.max_threads_per_block = 1024;
        device.max_workgroup_size = [1024, 1024, 64];
        device.supports_f64 = true;
        let length = 2usize * 443;
        let tuning = crate::PlannerTuning::for_device(device, Precision::DoubleDouble)
            .with_recursive_fft_rader(true);
        let plan = FftPlan::build_for_device(
            FftConfig::new(vec![length])
                .with_precision(Precision::DoubleDouble)
                .with_tuning(tuning),
            device,
        )
        .unwrap();
        let ir = DoubleDoubleRecursiveFftIr::build_for_device(&plan, Direction::Forward, device)
            .unwrap();
        let DoubleDoubleRecursiveFftNodeIr::CooleyTukey(root) = &ir.root else {
            panic!("DD N886 should keep smooth-2 x p443 Cooley root");
        };
        let block = root.pack_right.axis_batch_block.unwrap();
        assert_eq!(block.threads_per_transform, 74);
        assert_eq!(block.grouped_batch, 1);
        assert_eq!([block.local_size_x, block.local_size_y], [74, 1]);
        assert_eq!(root.twiddle_transpose.axis_batch_block, Some(block));
        assert_eq!(root.scatter_output.axis_batch_block, Some(block));

        let outer = match (&root.left, &root.right) {
            (DoubleDoubleRecursiveFftNodeIr::FftRader(rader), _) if rader.prime == 443 => rader,
            (_, DoubleDoubleRecursiveFftNodeIr::FftRader(rader)) if rader.prime == 443 => rader,
            _ => panic!("DD N886 should contain p443 FFT-Rader child"),
        };
        let DoubleDoubleBluesteinConvolutionIr::Recursive(convolution) = &outer.forward_fft else {
            panic!("DD p443 convolution should remain recursive 442=2x13x17");
        };
        fn contains_fft_prime(node: &DoubleDoubleRecursiveFftNodeIr, prime: usize) -> bool {
            match node {
                DoubleDoubleRecursiveFftNodeIr::FftRader(rader) => rader.prime == prime,
                DoubleDoubleRecursiveFftNodeIr::CooleyTukey(cooley) => {
                    contains_fft_prime(&cooley.left, prime)
                        || contains_fft_prime(&cooley.right, prime)
                }
                _ => false,
            }
        }
        fn contains_direct_prime(node: &DoubleDoubleRecursiveFftNodeIr, prime: usize) -> bool {
            match node {
                DoubleDoubleRecursiveFftNodeIr::DirectRader(rader) => rader.prime == prime,
                DoubleDoubleRecursiveFftNodeIr::CooleyTukey(cooley) => {
                    contains_direct_prime(&cooley.left, prime)
                        || contains_direct_prime(&cooley.right, prime)
                }
                _ => false,
            }
        }
        assert!(contains_fft_prime(&convolution.root, 17));
        assert!(contains_direct_prime(&convolution.root, 13));

        let shaders = crate::backend::vulkan::VulkanGlslBackend
            .lower_double_double_recursive(&ir)
            .unwrap();
        for shader in shaders {
            shader.compile_spirv().unwrap();
        }
        let impulse = ComplexDoubleDouble::new(
            DoubleDouble::from_parts(1.25, 3.0e-31),
            DoubleDouble::from_parts(-0.75, -2.0e-31),
        );
        let impulse_index = 3usize;
        let mut input = vec![ComplexDoubleDouble::default(); length];
        input[impulse_index] = impulse;
        let actual = execute_double_double_recursive_ir(&ir, &input).unwrap();
        let error = actual
            .iter()
            .copied()
            .enumerate()
            .map(|(k, actual)| {
                let expected = impulse
                    * crate::double_double_unit_root(impulse_index * k, length, Direction::Forward)
                        .unwrap();
                let dr = (actual.re - expected.re).to_f64();
                let di = (actual.im - expected.im).to_f64();
                dr.hypot(di)
            })
            .fold(0.0f64, f64::max);
        assert!(
            error <= 2.0e-13,
            "DD N886 nested Rader impulse mismatch: {error:e}"
        );
        ir.validate().unwrap();
    }

    #[test]
    fn nested_sub_rader_composite_parent_uses_recursive_quad_type0_floor() {
        let mut device = DeviceProfile::generic(crate::Backend::Vulkan, crate::GpuVendor::Nvidia);
        device.shared_memory_bytes = 48 * 1024;
        device.shared_memory_pow2_bytes = 32 * 1024;
        device.max_threads_per_block = 1024;
        device.max_workgroup_size = [1024, 1024, 64];
        device.supports_f64 = true;
        let length = 2usize * 107;
        let batch_count = 2usize;
        let tuning = crate::PlannerTuning::portable().with_recursive_fft_rader(true);
        let plan = FftPlan::build(
            FftConfig::new(vec![length])
                .with_batch_count(batch_count)
                .with_precision(Precision::DoubleDouble)
                .with_tuning(tuning),
        )
        .unwrap();
        let ir = DoubleDoubleRecursiveFftIr::build_for_device(&plan, Direction::Forward, device)
            .unwrap();
        assert!(ir.rader_forced_upload_schedule.is_none());
        let DoubleDoubleRecursiveFftNodeIr::CooleyTukey(root) = &ir.root else {
            panic!("DD N214 should keep a smooth-2 x p107 Cooley root");
        };
        let block = root.pack_right.axis_batch_block.unwrap();
        assert_eq!(block.threads_per_transform, 18);
        assert_eq!(block.grouped_batch, 1);
        assert_eq!([block.local_size_x, block.local_size_y], [18, 1]);
        assert_eq!(root.twiddle_transpose.axis_batch_block, Some(block));
        assert_eq!(root.scatter_output.axis_batch_block, Some(block));

        let outer = match (&root.left, &root.right) {
            (DoubleDoubleRecursiveFftNodeIr::FftRader(rader), _) if rader.prime == 107 => rader,
            (_, DoubleDoubleRecursiveFftNodeIr::FftRader(rader)) if rader.prime == 107 => rader,
            _ => panic!("DD N214 should contain p107 FFT-Rader child"),
        };
        let DoubleDoubleBluesteinConvolutionIr::Recursive(convolution) = &outer.forward_fft else {
            panic!("DD p107 convolution should remain recursive 106=2x53");
        };
        let DoubleDoubleRecursiveFftNodeIr::CooleyTukey(convolution_root) = &convolution.root
        else {
            panic!("DD p107 convolution should keep a Cooley root");
        };
        assert!(matches!(
            convolution_root.right,
            DoubleDoubleRecursiveFftNodeIr::FftRader(ref sub) if sub.prime == 53
        ));

        let shaders = crate::backend::vulkan::VulkanGlslBackend
            .lower_double_double_recursive(&ir)
            .unwrap();
        for shader in shaders {
            shader.compile_spirv().unwrap();
        }
        let mut input = vec![ComplexDoubleDouble::default(); length * batch_count];
        for batch in 0..batch_count {
            input[batch * length + 1] =
                ComplexDoubleDouble::new(DoubleDouble::from_f64(1.0), DoubleDouble::default());
        }
        let actual = execute_double_double_recursive_ir(&ir, &input).unwrap();
        let error = actual
            .iter()
            .enumerate()
            .map(|(index, value)| {
                let k = index % length;
                let angle = -std::f64::consts::TAU * k as f64 / length as f64;
                let expected = ComplexDoubleDouble::new(
                    DoubleDouble::from_f64(angle.cos()),
                    DoubleDouble::from_f64(angle.sin()),
                );
                let dr = (value.re - expected.re).to_f64();
                let di = (value.im - expected.im).to_f64();
                dr.hypot(di)
            })
            .fold(0.0f64, f64::max);
        assert!(
            error <= 3.0e-14,
            "DD N214 nested Rader impulse mismatch: {error:e}"
        );
        ir.validate().unwrap();
    }

    #[test]
    fn nested_rader_parent_uses_actual_planner_tuning_for_double_double() {
        let mut device = DeviceProfile::generic(crate::Backend::Vulkan, crate::GpuVendor::Nvidia);
        device.shared_memory_bytes = 48 * 1024;
        device.shared_memory_pow2_bytes = 32 * 1024;
        device.supports_f64 = true;
        let length = 2usize * 283;
        let batch_count = 2usize;
        let mut tuning = crate::PlannerTuning::portable().with_recursive_fft_rader(true);
        tuning.max_rader_direct_prime = 47;
        tuning.validate().unwrap();
        let plan = FftPlan::build(
            FftConfig::new(vec![length])
                .with_batch_count(batch_count)
                .with_precision(Precision::DoubleDouble)
                .with_tuning(tuning),
        )
        .unwrap();
        let ir = DoubleDoubleRecursiveFftIr::build_for_device(&plan, Direction::Forward, device)
            .unwrap();
        let DoubleDoubleRecursiveFftNodeIr::CooleyTukey(root) = &ir.root else {
            panic!("custom DD N566 should keep smooth-2 x p283 Cooley root");
        };
        let block = root.pack_right.axis_batch_block.unwrap();
        assert_eq!(block.threads_per_transform, 57);
        assert_eq!(block.grouped_batch, 1);
        assert_eq!([block.local_size_x, block.local_size_y], [57, 1]);
        assert_eq!(root.twiddle_transpose.axis_batch_block, Some(block));
        assert_eq!(root.scatter_output.axis_batch_block, Some(block));

        let outer = match (&root.left, &root.right) {
            (DoubleDoubleRecursiveFftNodeIr::FftRader(rader), _) if rader.prime == 283 => rader,
            (_, DoubleDoubleRecursiveFftNodeIr::FftRader(rader)) if rader.prime == 283 => rader,
            _ => panic!("custom DD N566 should contain p283 FFT-Rader child"),
        };
        let DoubleDoubleBluesteinConvolutionIr::Recursive(convolution) = &outer.forward_fft else {
            panic!("custom DD p283 convolution should remain recursive");
        };
        fn contains_fft_prime(node: &DoubleDoubleRecursiveFftNodeIr, prime: usize) -> bool {
            match node {
                DoubleDoubleRecursiveFftNodeIr::FftRader(rader) => rader.prime == prime,
                DoubleDoubleRecursiveFftNodeIr::CooleyTukey(cooley) => {
                    contains_fft_prime(&cooley.left, prime)
                        || contains_fft_prime(&cooley.right, prime)
                }
                _ => false,
            }
        }
        fn contains_direct_prime(node: &DoubleDoubleRecursiveFftNodeIr, prime: usize) -> bool {
            match node {
                DoubleDoubleRecursiveFftNodeIr::DirectRader(direct) => direct.prime == prime,
                DoubleDoubleRecursiveFftNodeIr::CooleyTukey(cooley) => {
                    contains_direct_prime(&cooley.left, prime)
                        || contains_direct_prime(&cooley.right, prime)
                }
                _ => false,
            }
        }
        assert!(contains_fft_prime(&convolution.root, 47));
        assert!(!contains_direct_prime(&convolution.root, 47));

        let shaders = crate::backend::vulkan::VulkanGlslBackend
            .lower_double_double_recursive(&ir)
            .unwrap();
        for shader in shaders {
            shader.compile_spirv().unwrap();
        }
        let mut input = vec![ComplexDoubleDouble::default(); length * batch_count];
        for batch in 0..batch_count {
            input[batch * length + 1] =
                ComplexDoubleDouble::new(DoubleDouble::from_f64(1.0), DoubleDouble::default());
        }
        let actual = execute_double_double_recursive_ir(&ir, &input).unwrap();
        let error = actual
            .iter()
            .enumerate()
            .map(|(index, value)| {
                let k = index % length;
                let angle = -std::f64::consts::TAU * k as f64 / length as f64;
                let expected = ComplexDoubleDouble::new(
                    DoubleDouble::from_f64(angle.cos()),
                    DoubleDouble::from_f64(angle.sin()),
                );
                let dr = (value.re - expected.re).to_f64();
                let di = (value.im - expected.im).to_f64();
                dr.hypot(di)
            })
            .fold(0.0f64, f64::max);
        assert!(
            error <= 8.0e-14,
            "DD custom N566 impulse mismatch: {error:e}"
        );
        ir.validate().unwrap();
    }

    #[test]
    fn nested_sub_rader_and_sibling_type0_composite_uses_joint_quad_floor() {
        let mut device = DeviceProfile::generic(crate::Backend::Vulkan, crate::GpuVendor::Nvidia);
        device.shared_memory_bytes = 128 * 1024;
        device.shared_memory_pow2_bytes = 128 * 1024;
        device.max_threads_per_block = 1024;
        device.max_workgroup_size = [1024, 1024, 64];
        device.supports_f64 = true;
        let length = 19usize * 107;
        let batch_count = 2usize;
        let tuning = crate::PlannerTuning::portable().with_recursive_fft_rader(true);
        let plan = FftPlan::build(
            FftConfig::new(vec![length])
                .with_batch_count(batch_count)
                .with_precision(Precision::DoubleDouble)
                .with_tuning(tuning),
        )
        .unwrap();
        let ir = DoubleDoubleRecursiveFftIr::build_for_device(&plan, Direction::Forward, device)
            .unwrap();
        assert!(ir.rader_forced_upload_schedule.is_none());
        let DoubleDoubleRecursiveFftNodeIr::CooleyTukey(root) = &ir.root else {
            panic!("DD N2033 should keep p19 x p107 Cooley root");
        };
        let block = root.pack_right.axis_batch_block.unwrap();
        assert_eq!(block.threads_per_transform, 214);
        assert_eq!(block.grouped_batch, 1);
        assert_eq!([block.local_size_x, block.local_size_y], [214, 1]);
        assert_eq!(root.twiddle_transpose.axis_batch_block, Some(block));
        assert_eq!(root.scatter_output.axis_batch_block, Some(block));
        assert!(matches!(
            (&root.left, &root.right),
            (
                DoubleDoubleRecursiveFftNodeIr::FftRader(left),
                DoubleDoubleRecursiveFftNodeIr::FftRader(right)
            ) if [left.prime, right.prime] == [19, 107]
                || [left.prime, right.prime] == [107, 19]
        ));
        let outer = match (&root.left, &root.right) {
            (DoubleDoubleRecursiveFftNodeIr::FftRader(rader), _) if rader.prime == 107 => rader,
            (_, DoubleDoubleRecursiveFftNodeIr::FftRader(rader)) if rader.prime == 107 => rader,
            _ => unreachable!("asserted DD p107 sibling"),
        };
        let DoubleDoubleBluesteinConvolutionIr::Recursive(convolution) = &outer.forward_fft else {
            panic!("DD p107 sibling convolution should remain recursive 106=2x53");
        };
        let DoubleDoubleRecursiveFftNodeIr::CooleyTukey(convolution_root) = &convolution.root
        else {
            panic!("DD p107 sibling convolution should keep a Cooley root");
        };
        assert!(matches!(
            convolution_root.right,
            DoubleDoubleRecursiveFftNodeIr::FftRader(ref sub) if sub.prime == 53
        ));
        let shaders = crate::backend::vulkan::VulkanGlslBackend
            .lower_double_double_recursive(&ir)
            .unwrap();
        for shader in shaders {
            shader.compile_spirv().unwrap();
        }
        ir.validate().unwrap();
    }

    #[test]
    fn composite_direct_rader_parent_uses_quad_type1_thread_floor() {
        let mut device = DeviceProfile::generic(crate::Backend::Vulkan, crate::GpuVendor::Nvidia);
        device.shared_memory_bytes = 48 * 1024;
        device.shared_memory_pow2_bytes = 32 * 1024;
        device.max_threads_per_block = 1024;
        device.max_workgroup_size = [1024, 1024, 64];
        device.supports_f64 = true;
        let length = 2usize * 47;
        let batch_count = 5usize;
        let config = FftConfig::new(vec![length])
            .with_batch_count(batch_count)
            .with_grouped_batch(0, 3)
            .unwrap()
            .with_precision(Precision::DoubleDouble);
        let plan = FftPlan::build(config).unwrap();
        let ir = DoubleDoubleRecursiveFftIr::build_for_device(&plan, Direction::Forward, device)
            .unwrap();
        assert!(ir.rader_forced_upload_schedule.is_none());
        let DoubleDoubleRecursiveFftNodeIr::CooleyTukey(root) = &ir.root else {
            panic!("DD N94 composite direct-Rader should keep a Cooley root");
        };
        assert!(matches!(
            root.right,
            DoubleDoubleRecursiveFftNodeIr::DirectRader(_)
        ));
        let block = root.pack_right.axis_batch_block.unwrap();
        assert_eq!(block.threads_per_transform, 48);
        assert_eq!(block.grouped_batch, 3);
        assert_eq!([block.local_size_x, block.local_size_y], [48, 3]);
        assert_eq!(root.twiddle_transpose.axis_batch_block, Some(block));
        assert_eq!(root.scatter_output.axis_batch_block, Some(block));
        let fused = root
            .fused_small_direct_rader_stockham()
            .unwrap()
            .expect("DD N94 should fuse Stockham x Direct-Rader into one component kernel");
        assert_eq!(fused.required_shared_memory_bytes().unwrap(), 9_024);

        let program = crate::ProgramIr::double_double_recursive(&ir).unwrap();
        assert_eq!(program.passes.len(), 1);
        assert_eq!(program.passes[0].name, fused.name());
        assert_eq!(program.passes[0].bindings.len(), 4);
        assert_eq!(
            program.passes[0].bindings[2].role,
            crate::BufferRole::LookupTable
        );
        assert_eq!(
            program.passes[0].bindings[3].role,
            crate::BufferRole::TwiddleLookupTable
        );

        let shaders = crate::backend::vulkan::VulkanGlslBackend
            .lower_double_double_recursive(&ir)
            .unwrap();
        assert_eq!(shaders.len(), 1);
        assert_eq!(shaders[0].required_shared_memory_bytes, 9_024);
        assert_eq!(shaders[0].glsl.matches("barrier();").count(), 1);
        assert!(
            shaders[0]
                .glsl
                .contains("fused double-double Stockham x Direct-Rader Cooley IR")
        );
        assert!(shaders[0].glsl.contains("vkfft_twiddle_lut.data"));
        shaders[0].compile_spirv().unwrap();
        let mut input = vec![ComplexDoubleDouble::default(); length * batch_count];
        for batch in 0..batch_count {
            input[batch * length + 1] =
                ComplexDoubleDouble::new(DoubleDouble::from_f64(1.0), DoubleDouble::default());
        }
        let actual = execute_double_double_recursive_ir(&ir, &input).unwrap();
        let mut expected = Vec::with_capacity(actual.len());
        for _batch in 0..batch_count {
            for k in 0..length {
                let angle = -std::f64::consts::TAU * k as f64 / length as f64;
                expected.push(ComplexDoubleDouble::new(
                    DoubleDouble::from_f64(angle.cos()),
                    DoubleDouble::from_f64(angle.sin()),
                ));
            }
        }
        let error = max_error(&actual, &expected);
        assert!(
            error <= 2.0e-15,
            "DD N94 composite direct-Rader impulse mismatch: {error:e}"
        );
        ir.validate().unwrap();
    }

    #[test]
    fn composite_direct_rader_n141_crosses_old_small_fusion_limit() {
        let mut device = DeviceProfile::generic(crate::Backend::Vulkan, crate::GpuVendor::Nvidia);
        device.shared_memory_bytes = 48 * 1024;
        device.shared_memory_pow2_bytes = 32 * 1024;
        device.max_threads_per_block = 1024;
        device.max_workgroup_size = [1024, 1024, 64];
        device.supports_f64 = true;
        let length = 3usize * 47;
        let batch_count = 5usize;
        let config = FftConfig::new(vec![length])
            .with_batch_count(batch_count)
            .with_grouped_batch(0, 3)
            .unwrap()
            .with_precision(Precision::DoubleDouble);
        let plan = FftPlan::build(config).unwrap();
        let ir = DoubleDoubleRecursiveFftIr::build_for_device(&plan, Direction::Forward, device)
            .unwrap();
        assert!(ir.rader_forced_upload_schedule.is_none());
        let DoubleDoubleRecursiveFftNodeIr::CooleyTukey(root) = &ir.root else {
            panic!("DD N141 composite direct-Rader should keep a Cooley root");
        };
        assert_eq!((root.left_len, root.right_len), (3, 47));
        assert!(matches!(
            root.left,
            DoubleDoubleRecursiveFftNodeIr::Stockham(_)
        ));
        assert!(matches!(
            root.right,
            DoubleDoubleRecursiveFftNodeIr::DirectRader(_)
        ));
        let block = root.pack_right.axis_batch_block.unwrap();
        assert_eq!(block.threads_per_transform, 72);
        assert_eq!(block.grouped_batch, 3);
        assert_eq!([block.local_size_x, block.local_size_y], [72, 3]);
        assert_eq!(root.twiddle_transpose.axis_batch_block, Some(block));
        assert_eq!(root.scatter_output.axis_batch_block, Some(block));
        let fused = root
            .fused_small_direct_rader_stockham()
            .unwrap()
            .expect("DD N141 should fuse Stockham x Direct-Rader beyond the old N<=128 gate");
        assert_eq!(fused.required_shared_memory_bytes().unwrap(), 13_536);

        let program = crate::ProgramIr::double_double_recursive(&ir).unwrap();
        assert_eq!(program.passes.len(), 1);
        assert_eq!(program.passes[0].name, fused.name());
        assert_eq!(program.passes[0].bindings.len(), 4);

        let shaders = crate::backend::vulkan::VulkanGlslBackend
            .lower_double_double_recursive(&ir)
            .unwrap();
        assert_eq!(shaders.len(), 1);
        assert_eq!(shaders[0].required_shared_memory_bytes, 13_536);
        assert_eq!(shaders[0].glsl.matches("barrier();").count(), 1);
        shaders[0].compile_spirv().unwrap();

        let mut input = vec![ComplexDoubleDouble::default(); length * batch_count];
        for batch in 0..batch_count {
            input[batch * length + 1] =
                ComplexDoubleDouble::new(DoubleDouble::from_f64(1.0), DoubleDouble::default());
        }
        let actual = execute_double_double_recursive_ir(&ir, &input).unwrap();
        let mut expected = Vec::with_capacity(actual.len());
        for _batch in 0..batch_count {
            for k in 0..length {
                let angle = -std::f64::consts::TAU * k as f64 / length as f64;
                expected.push(ComplexDoubleDouble::new(
                    DoubleDouble::from_f64(angle.cos()),
                    DoubleDouble::from_f64(angle.sin()),
                ));
            }
        }
        let error = max_error(&actual, &expected);
        assert!(
            error <= 2.0e-15,
            "DD N141 composite direct-Rader impulse mismatch: {error:e}"
        );
        ir.validate().unwrap();
    }

    #[test]
    fn composite_direct_rader_n188_fuses_radix4_stockham_parent() {
        let mut device = DeviceProfile::generic(crate::Backend::Vulkan, crate::GpuVendor::Nvidia);
        device.shared_memory_bytes = 48 * 1024;
        device.shared_memory_pow2_bytes = 32 * 1024;
        device.max_threads_per_block = 1024;
        device.max_workgroup_size = [1024, 1024, 64];
        device.supports_f64 = true;
        let length = 4usize * 47;
        let batch_count = 5usize;
        let plan = FftPlan::build(
            FftConfig::new(vec![length])
                .with_batch_count(batch_count)
                .with_grouped_batch(0, 3)
                .unwrap()
                .with_precision(Precision::DoubleDouble),
        )
        .unwrap();
        let ir = DoubleDoubleRecursiveFftIr::build_for_device(&plan, Direction::Forward, device)
            .unwrap();
        assert!(ir.rader_forced_upload_schedule.is_none());
        let DoubleDoubleRecursiveFftNodeIr::CooleyTukey(root) = &ir.root else {
            panic!("DD N188 composite direct-Rader should keep a Cooley root");
        };
        assert_eq!((root.left_len, root.right_len), (4, 47));
        assert!(matches!(
            root.left,
            DoubleDoubleRecursiveFftNodeIr::Stockham(_)
        ));
        assert!(matches!(
            root.right,
            DoubleDoubleRecursiveFftNodeIr::DirectRader(_)
        ));
        let block = root.pack_right.axis_batch_block.unwrap();
        assert_eq!(block.threads_per_transform, 96);
        assert_eq!(block.grouped_batch, 3);
        assert_eq!([block.local_size_x, block.local_size_y], [96, 3]);
        assert_eq!(root.twiddle_transpose.axis_batch_block, Some(block));
        assert_eq!(root.scatter_output.axis_batch_block, Some(block));
        let fused = root
            .fused_small_direct_rader_stockham()
            .unwrap()
            .expect("DD N188 should deliberately extend fused ownership to radix-4 Stockham");
        assert_eq!(fused.required_shared_memory_bytes().unwrap(), 18_048);
        let program = crate::ProgramIr::double_double_recursive(&ir).unwrap();
        assert_eq!(program.passes.len(), 1);
        assert_eq!(program.passes[0].name, fused.name());

        let shaders = crate::backend::vulkan::VulkanGlslBackend
            .lower_double_double_recursive(&ir)
            .unwrap();
        assert_eq!(shaders.len(), 1);
        assert_eq!(shaders[0].required_shared_memory_bytes, 18_048);
        assert_eq!(shaders[0].glsl.matches("barrier();").count(), 1);
        shaders[0].compile_spirv().unwrap();

        let mut input = vec![ComplexDoubleDouble::default(); length * batch_count];
        for batch in 0..batch_count {
            input[batch * length + 1] =
                ComplexDoubleDouble::new(DoubleDouble::from_f64(1.0), DoubleDouble::default());
        }
        let actual = execute_double_double_recursive_ir(&ir, &input).unwrap();
        let mut expected = Vec::with_capacity(actual.len());
        for _batch in 0..batch_count {
            for k in 0..length {
                let angle = -std::f64::consts::TAU * k as f64 / length as f64;
                expected.push(ComplexDoubleDouble::new(
                    DoubleDouble::from_f64(angle.cos()),
                    DoubleDouble::from_f64(angle.sin()),
                ));
            }
        }
        let error = max_error(&actual, &expected);
        assert!(
            error <= 2.0e-15,
            "DD N188 composite direct-Rader impulse mismatch: {error:e}"
        );
        ir.validate().unwrap();
    }

    #[test]
    fn composite_direct_rader_n235_fuses_radix5_stockham_parent() {
        let mut device = DeviceProfile::generic(crate::Backend::Vulkan, crate::GpuVendor::Nvidia);
        device.shared_memory_bytes = 48 * 1024;
        device.shared_memory_pow2_bytes = 32 * 1024;
        device.max_threads_per_block = 1024;
        device.max_workgroup_size = [1024, 1024, 64];
        device.supports_f64 = true;
        let length = 5usize * 47;
        let plan = FftPlan::build(
            FftConfig::new(vec![length])
                .with_batch_count(5)
                .with_grouped_batch(0, 3)
                .unwrap()
                .with_precision(Precision::DoubleDouble),
        )
        .unwrap();
        let ir = DoubleDoubleRecursiveFftIr::build_for_device(&plan, Direction::Forward, device)
            .unwrap();
        let DoubleDoubleRecursiveFftNodeIr::CooleyTukey(root) = &ir.root else {
            panic!("DD N235 composite direct-Rader should keep a Cooley root");
        };
        assert_eq!((root.left_len, root.right_len), (5, 47));
        assert!(matches!(
            root.left,
            DoubleDoubleRecursiveFftNodeIr::Stockham(_)
        ));
        assert!(matches!(
            root.right,
            DoubleDoubleRecursiveFftNodeIr::DirectRader(_)
        ));
        let block = root.pack_right.axis_batch_block.unwrap();
        assert_eq!(block.threads_per_transform, 120);
        assert_eq!(block.grouped_batch, 3);
        assert_eq!([block.local_size_x, block.local_size_y], [120, 3]);
        assert_eq!(root.twiddle_transpose.axis_batch_block, Some(block));
        assert_eq!(root.scatter_output.axis_batch_block, Some(block));
        let fused = root
            .fused_small_direct_rader_stockham()
            .unwrap()
            .expect("DD N235 should deliberately extend fused ownership to radix-5 Stockham");
        assert_eq!(fused.required_shared_memory_bytes().unwrap(), 22_560);
        let program = crate::ProgramIr::double_double_recursive(&ir).unwrap();
        assert_eq!(program.passes.len(), 1);
        assert_eq!(program.passes[0].name, fused.name());

        let shaders = crate::backend::vulkan::VulkanGlslBackend
            .lower_double_double_recursive(&ir)
            .unwrap();
        assert_eq!(shaders.len(), 1);
        assert_eq!(shaders[0].required_shared_memory_bytes, 22_560);
        assert_eq!(shaders[0].glsl.matches("barrier();").count(), 1);
        shaders[0].compile_spirv().unwrap();

        let batch_count = 5usize;
        let mut input = vec![ComplexDoubleDouble::default(); length * batch_count];
        for batch in 0..batch_count {
            input[batch * length + 1] =
                ComplexDoubleDouble::new(DoubleDouble::from_f64(1.0), DoubleDouble::default());
        }
        let actual = execute_double_double_recursive_ir(&ir, &input).unwrap();
        let mut expected = Vec::with_capacity(actual.len());
        for _batch in 0..batch_count {
            for k in 0..length {
                let angle = -std::f64::consts::TAU * k as f64 / length as f64;
                expected.push(ComplexDoubleDouble::new(
                    DoubleDouble::from_f64(angle.cos()),
                    DoubleDouble::from_f64(angle.sin()),
                ));
            }
        }
        let error = max_error(&actual, &expected);
        assert!(
            error <= 2.0e-15,
            "DD N235 composite direct-Rader impulse mismatch: {error:e}"
        );
        ir.validate().unwrap();
    }

    #[test]
    fn composite_direct_rader_n282_fuses_radix6_stockham_parent() {
        let mut device = DeviceProfile::generic(crate::Backend::Vulkan, crate::GpuVendor::Nvidia);
        device.shared_memory_bytes = 48 * 1024;
        device.shared_memory_pow2_bytes = 32 * 1024;
        device.max_threads_per_block = 1024;
        device.max_workgroup_size = [1024, 1024, 64];
        device.supports_f64 = true;
        let length = 6usize * 47;
        let plan = FftPlan::build(
            FftConfig::new(vec![length])
                .with_batch_count(5)
                .with_grouped_batch(0, 3)
                .unwrap()
                .with_precision(Precision::DoubleDouble),
        )
        .unwrap();
        let ir = DoubleDoubleRecursiveFftIr::build_for_device(&plan, Direction::Forward, device)
            .unwrap();
        let DoubleDoubleRecursiveFftNodeIr::CooleyTukey(root) = &ir.root else {
            panic!("DD N282 composite direct-Rader should keep a Cooley root");
        };
        assert_eq!((root.left_len, root.right_len), (6, 47));
        assert!(matches!(
            root.left,
            DoubleDoubleRecursiveFftNodeIr::Stockham(_)
        ));
        assert!(matches!(
            root.right,
            DoubleDoubleRecursiveFftNodeIr::DirectRader(_)
        ));
        let block = root.pack_right.axis_batch_block.unwrap();
        assert_eq!(block.threads_per_transform, 144);
        assert_eq!(block.grouped_batch, 3);
        assert_eq!([block.local_size_x, block.local_size_y], [144, 3]);
        assert_eq!(root.twiddle_transpose.axis_batch_block, Some(block));
        assert_eq!(root.scatter_output.axis_batch_block, Some(block));
        let fused = root
            .fused_small_direct_rader_stockham()
            .unwrap()
            .expect("DD N282 should deliberately extend fused ownership to radix-6 Stockham");
        assert_eq!(fused.required_shared_memory_bytes().unwrap(), 27_072);
        let program = crate::ProgramIr::double_double_recursive(&ir).unwrap();
        assert_eq!(program.passes.len(), 1);
        assert_eq!(program.passes[0].name, fused.name());

        let shaders = crate::backend::vulkan::VulkanGlslBackend
            .lower_double_double_recursive(&ir)
            .unwrap();
        assert_eq!(shaders.len(), 1);
        assert_eq!(shaders[0].required_shared_memory_bytes, 27_072);
        assert_eq!(shaders[0].glsl.matches("barrier();").count(), 1);
        shaders[0].compile_spirv().unwrap();

        let batch_count = 5usize;
        let mut input = vec![ComplexDoubleDouble::default(); length * batch_count];
        for batch in 0..batch_count {
            input[batch * length + 1] =
                ComplexDoubleDouble::new(DoubleDouble::from_f64(1.0), DoubleDouble::default());
        }
        let actual = execute_double_double_recursive_ir(&ir, &input).unwrap();
        let mut expected = Vec::with_capacity(actual.len());
        for _batch in 0..batch_count {
            for k in 0..length {
                let angle = -std::f64::consts::TAU * k as f64 / length as f64;
                expected.push(ComplexDoubleDouble::new(
                    DoubleDouble::from_f64(angle.cos()),
                    DoubleDouble::from_f64(angle.sin()),
                ));
            }
        }
        let error = max_error(&actual, &expected);
        assert!(
            error <= 2.0e-15,
            "DD N282 composite direct-Rader impulse mismatch: {error:e}"
        );
        ir.validate().unwrap();
    }

    #[test]
    fn composite_direct_rader_n329_n376_fuse_small_stockham_parents() {
        let mut device = DeviceProfile::generic(crate::Backend::Vulkan, crate::GpuVendor::Nvidia);
        device.shared_memory_bytes = 48 * 1024;
        device.shared_memory_pow2_bytes = 32 * 1024;
        device.max_threads_per_block = 1024;
        device.max_workgroup_size = [1024, 1024, 64];
        device.supports_f64 = true;
        for (left_len, length, threads_per_transform, shared_bytes) in [
            (7usize, 7usize * 47, 168usize, 31_584usize),
            (8usize, 8usize * 47, 192usize, 36_096usize),
        ] {
            let batch_count = 5usize;
            let plan = FftPlan::build(
                FftConfig::new(vec![length])
                    .with_batch_count(batch_count)
                    .with_grouped_batch(0, 3)
                    .unwrap()
                    .with_precision(Precision::DoubleDouble),
            )
            .unwrap();
            let ir =
                DoubleDoubleRecursiveFftIr::build_for_device(&plan, Direction::Forward, device)
                    .unwrap();
            let DoubleDoubleRecursiveFftNodeIr::CooleyTukey(root) = &ir.root else {
                panic!("DD N{length} composite direct-Rader should keep a Cooley root");
            };
            assert_eq!((root.left_len, root.right_len), (left_len, 47));
            assert!(matches!(
                root.left,
                DoubleDoubleRecursiveFftNodeIr::Stockham(_)
            ));
            assert!(matches!(
                root.right,
                DoubleDoubleRecursiveFftNodeIr::DirectRader(_)
            ));
            let block = root.pack_right.axis_batch_block.unwrap();
            assert_eq!(block.threads_per_transform, threads_per_transform);
            assert_eq!(block.grouped_batch, 3);
            assert_eq!(
                [block.local_size_x, block.local_size_y],
                [threads_per_transform, 3]
            );
            assert_eq!(root.twiddle_transpose.axis_batch_block, Some(block));
            assert_eq!(root.scatter_output.axis_batch_block, Some(block));
            let fused = root
                .fused_small_direct_rader_stockham()
                .unwrap()
                .unwrap_or_else(|| panic!("DD N{length} should fuse the small Stockham parent"));
            assert_eq!(fused.required_shared_memory_bytes().unwrap(), shared_bytes);
            let program = crate::ProgramIr::double_double_recursive(&ir).unwrap();
            assert_eq!(program.passes.len(), 1);
            assert_eq!(program.passes[0].name, fused.name());

            let shaders = crate::backend::vulkan::VulkanGlslBackend
                .lower_double_double_recursive(&ir)
                .unwrap();
            assert_eq!(shaders.len(), 1);
            assert_eq!(shaders[0].required_shared_memory_bytes, shared_bytes);
            assert_eq!(shaders[0].glsl.matches("barrier();").count(), 1);
            shaders[0].compile_spirv().unwrap();

            let mut input = vec![ComplexDoubleDouble::default(); length * batch_count];
            for batch in 0..batch_count {
                input[batch * length + 1] =
                    ComplexDoubleDouble::new(DoubleDouble::from_f64(1.0), DoubleDouble::default());
            }
            let actual = execute_double_double_recursive_ir(&ir, &input).unwrap();
            let mut expected = Vec::with_capacity(actual.len());
            for _batch in 0..batch_count {
                for k in 0..length {
                    let angle = -std::f64::consts::TAU * k as f64 / length as f64;
                    expected.push(ComplexDoubleDouble::new(
                        DoubleDouble::from_f64(angle.cos()),
                        DoubleDouble::from_f64(angle.sin()),
                    ));
                }
            }
            let error = max_error(&actual, &expected);
            assert!(
                error <= 2.0e-15,
                "DD N{length} composite direct-Rader impulse mismatch: {error:e}"
            );
            ir.validate().unwrap();
        }
    }

    #[test]
    fn composite_direct_rader_n423_n470_fuse_radix9_10_stockham_parents() {
        let mut device = DeviceProfile::generic(crate::Backend::Vulkan, crate::GpuVendor::Nvidia);
        device.shared_memory_bytes = 48 * 1024;
        device.shared_memory_pow2_bytes = 32 * 1024;
        device.max_threads_per_block = 1024;
        device.max_workgroup_size = [1024, 1024, 64];
        device.supports_f64 = true;
        for (left_len, length, threads_per_transform, shared_bytes) in [
            (9usize, 9usize * 47, 216usize, 40_608usize),
            (10usize, 10usize * 47, 240usize, 45_120usize),
        ] {
            let batch_count = 5usize;
            let plan = FftPlan::build(
                FftConfig::new(vec![length])
                    .with_batch_count(batch_count)
                    .with_grouped_batch(0, 3)
                    .unwrap()
                    .with_precision(Precision::DoubleDouble),
            )
            .unwrap();
            let ir =
                DoubleDoubleRecursiveFftIr::build_for_device(&plan, Direction::Forward, device)
                    .unwrap();
            let DoubleDoubleRecursiveFftNodeIr::CooleyTukey(root) = &ir.root else {
                panic!("DD N{length} composite direct-Rader should keep a Cooley root");
            };
            assert_eq!((root.left_len, root.right_len), (left_len, 47));
            assert!(matches!(
                root.left,
                DoubleDoubleRecursiveFftNodeIr::Stockham(_)
            ));
            assert!(matches!(
                root.right,
                DoubleDoubleRecursiveFftNodeIr::DirectRader(_)
            ));
            let block = root.pack_right.axis_batch_block.unwrap();
            assert_eq!(block.threads_per_transform, threads_per_transform);
            assert_eq!(block.grouped_batch, 3);
            assert_eq!(
                [block.local_size_x, block.local_size_y],
                [threads_per_transform, 3]
            );
            assert_eq!(root.twiddle_transpose.axis_batch_block, Some(block));
            assert_eq!(root.scatter_output.axis_batch_block, Some(block));
            let fused = root
                .fused_small_direct_rader_stockham()
                .unwrap()
                .unwrap_or_else(|| panic!("DD N{length} should fuse on a 48 KiB device"));
            assert_eq!(fused.required_shared_memory_bytes().unwrap(), shared_bytes);
            let program = crate::ProgramIr::double_double_recursive(&ir).unwrap();
            assert_eq!(program.passes.len(), 1);
            assert_eq!(program.passes[0].name, fused.name());

            let shaders = crate::backend::vulkan::VulkanGlslBackend
                .lower_double_double_recursive(&ir)
                .unwrap();
            assert_eq!(shaders.len(), 1);
            assert_eq!(shaders[0].required_shared_memory_bytes, shared_bytes);
            assert_eq!(shaders[0].glsl.matches("barrier();").count(), 1);
            shaders[0].compile_spirv().unwrap();

            let mut input = vec![ComplexDoubleDouble::default(); length * batch_count];
            for batch in 0..batch_count {
                input[batch * length + 1] =
                    ComplexDoubleDouble::new(DoubleDouble::from_f64(1.0), DoubleDouble::default());
            }
            let actual = execute_double_double_recursive_ir(&ir, &input).unwrap();
            let mut expected = Vec::with_capacity(actual.len());
            for _batch in 0..batch_count {
                for k in 0..length {
                    let angle = -std::f64::consts::TAU * k as f64 / length as f64;
                    expected.push(ComplexDoubleDouble::new(
                        DoubleDouble::from_f64(angle.cos()),
                        DoubleDouble::from_f64(angle.sin()),
                    ));
                }
            }
            let error = max_error(&actual, &expected);
            assert!(
                error <= 2.0e-15,
                "DD N{length} composite direct-Rader impulse mismatch: {error:e}"
            );
            ir.validate().unwrap();
        }
    }

    #[test]
    fn composite_direct_rader_n517_is_gated_by_48k_shared_capacity() {
        let length = 11usize * 47;
        let plan = FftPlan::build(
            FftConfig::new(vec![length])
                .with_batch_count(5)
                .with_grouped_batch(0, 3)
                .unwrap()
                .with_precision(Precision::DoubleDouble),
        )
        .unwrap();

        let mut device_48k =
            DeviceProfile::generic(crate::Backend::Vulkan, crate::GpuVendor::Nvidia);
        device_48k.shared_memory_bytes = 48 * 1024;
        device_48k.shared_memory_pow2_bytes = 32 * 1024;
        device_48k.max_threads_per_block = 1024;
        device_48k.max_workgroup_size = [1024, 1024, 64];
        device_48k.supports_f64 = true;
        let ir_48k =
            DoubleDoubleRecursiveFftIr::build_for_device(&plan, Direction::Forward, device_48k)
                .unwrap();
        let DoubleDoubleRecursiveFftNodeIr::CooleyTukey(root_48k) = &ir_48k.root else {
            panic!("DD N517 capacity witness should keep a Cooley root");
        };
        assert_eq!((root_48k.left_len, root_48k.right_len), (11, 47));
        let block = root_48k.pack_right.axis_batch_block.unwrap();
        assert_eq!(block.threads_per_transform, 128);
        assert_eq!(block.grouped_batch, 3);
        assert_eq!([block.local_size_x, block.local_size_y], [128, 3]);
        assert!(
            root_48k
                .fused_small_direct_rader_stockham()
                .unwrap()
                .is_none()
        );
        assert_eq!(
            crate::ProgramIr::double_double_recursive(&ir_48k)
                .unwrap()
                .passes
                .len(),
            5
        );

        let mut device_64k = device_48k;
        device_64k.shared_memory_bytes = 64 * 1024;
        device_64k.shared_memory_pow2_bytes = 64 * 1024;
        let ir_64k =
            DoubleDoubleRecursiveFftIr::build_for_device(&plan, Direction::Forward, device_64k)
                .unwrap();
        let DoubleDoubleRecursiveFftNodeIr::CooleyTukey(root_64k) = &ir_64k.root else {
            panic!("DD N517 high-capacity witness should keep a Cooley root");
        };
        let block_64k = root_64k.pack_right.axis_batch_block.unwrap();
        assert_eq!(block_64k.threads_per_transform, 128);
        assert_eq!(block_64k.grouped_batch, 3);
        assert_eq!([block_64k.local_size_x, block_64k.local_size_y], [128, 3]);
        let fused = root_64k
            .fused_small_direct_rader_stockham()
            .unwrap()
            .expect("DD N517 should fuse once shared capacity exceeds its 49,632-byte footprint");
        assert_eq!(fused.required_shared_memory_bytes().unwrap(), 49_632);
        assert_eq!(
            crate::ProgramIr::double_double_recursive(&ir_64k)
                .unwrap()
                .passes
                .len(),
            1
        );
        let shaders = crate::backend::vulkan::VulkanGlslBackend
            .lower_double_double_recursive(&ir_64k)
            .unwrap();
        assert_eq!(shaders.len(), 1);
        assert_eq!(shaders[0].required_shared_memory_bytes, 49_632);
        shaders[0].compile_spirv().unwrap();

        let mut input = vec![ComplexDoubleDouble::default(); length * 5];
        for batch in 0..5 {
            input[batch * length + 1] =
                ComplexDoubleDouble::new(DoubleDouble::from_f64(1.0), DoubleDouble::default());
        }
        let actual = execute_double_double_recursive_ir(&ir_64k, &input).unwrap();
        let mut expected = Vec::with_capacity(actual.len());
        for _batch in 0..5 {
            for k in 0..length {
                let angle = -std::f64::consts::TAU * k as f64 / length as f64;
                expected.push(ComplexDoubleDouble::new(
                    DoubleDouble::from_f64(angle.cos()),
                    DoubleDouble::from_f64(angle.sin()),
                ));
            }
        }
        let error = max_error(&actual, &expected);
        assert!(
            error <= 2.0e-15,
            "DD N517 high-capacity composite direct-Rader impulse mismatch: {error:e}"
        );
        ir_48k.validate().unwrap();
        ir_64k.validate().unwrap();
    }

    #[test]
    fn composite_direct_rader_n564_extends_left_stockham_and_capacity_gate() {
        let length = 12usize * 47;
        let mut device_48k =
            DeviceProfile::generic(crate::Backend::Vulkan, crate::GpuVendor::Nvidia);
        device_48k.shared_memory_bytes = 48 * 1024;
        device_48k.shared_memory_pow2_bytes = 32 * 1024;
        device_48k.max_threads_per_block = 1024;
        device_48k.max_workgroup_size = [1024, 1024, 64];
        device_48k.supports_f64 = true;

        let single_plan =
            FftPlan::build(FftConfig::new(vec![length]).with_precision(Precision::DoubleDouble))
                .unwrap();
        let single = DoubleDoubleRecursiveFftIr::build_for_device(
            &single_plan,
            Direction::Forward,
            device_48k,
        )
        .unwrap();
        let DoubleDoubleRecursiveFftNodeIr::CooleyTukey(single_root) = &single.root else {
            panic!("DD N564 batch1 witness should keep a Cooley root");
        };
        assert_eq!((single_root.left_len, single_root.right_len), (12, 47));
        let single_block = single_root.pack_right.axis_batch_block.unwrap();
        assert_eq!(single_block.threads_per_transform, 96);
        assert_eq!(single_block.grouped_batch, 1);
        assert_eq!(
            [single_block.local_size_x, single_block.local_size_y],
            [96, 1]
        );
        let single_fused = single_root
            .fused_small_direct_rader_stockham()
            .unwrap()
            .expect("DD N564 batch1 should fit the 48 KiB fused kernel");
        assert_eq!(single_fused.required_shared_memory_bytes().unwrap(), 18_048);
        let single_program = crate::ProgramIr::double_double_recursive(&single).unwrap();
        assert_eq!(single_program.passes.len(), 1);
        let single_shaders = crate::backend::vulkan::VulkanGlslBackend
            .lower_double_double_recursive(&single)
            .unwrap();
        assert_eq!(single_shaders.len(), 1);
        assert_eq!(single_shaders[0].required_shared_memory_bytes, 18_048);
        single_shaders[0].compile_spirv().unwrap();

        let mut input = vec![ComplexDoubleDouble::default(); length];
        input[1] = ComplexDoubleDouble::new(DoubleDouble::from_f64(1.0), DoubleDouble::default());
        let actual = execute_double_double_recursive_ir(&single, &input).unwrap();
        let expected = (0..length)
            .map(|k| {
                let angle = -std::f64::consts::TAU * k as f64 / length as f64;
                ComplexDoubleDouble::new(
                    DoubleDouble::from_f64(angle.cos()),
                    DoubleDouble::from_f64(angle.sin()),
                )
            })
            .collect::<Vec<_>>();
        let error = max_error(&actual, &expected);
        assert!(
            error <= 2.0e-15,
            "DD N564 batch1 impulse mismatch: {error:e}"
        );

        let grouped_plan = FftPlan::build(
            FftConfig::new(vec![length])
                .with_batch_count(5)
                .with_grouped_batch(0, 3)
                .unwrap()
                .with_precision(Precision::DoubleDouble),
        )
        .unwrap();
        let grouped_48k = DoubleDoubleRecursiveFftIr::build_for_device(
            &grouped_plan,
            Direction::Forward,
            device_48k,
        )
        .unwrap();
        let DoubleDoubleRecursiveFftNodeIr::CooleyTukey(grouped_48k_root) = &grouped_48k.root
        else {
            panic!("DD N564 G3/48KiB witness should keep a Cooley root");
        };
        let grouped_48k_block = grouped_48k_root.pack_right.axis_batch_block.unwrap();
        assert_eq!(grouped_48k_block.threads_per_transform, 96);
        assert_eq!(grouped_48k_block.grouped_batch, 3);
        assert_eq!(
            [
                grouped_48k_block.local_size_x,
                grouped_48k_block.local_size_y
            ],
            [96, 3]
        );
        assert!(
            grouped_48k_root
                .fused_small_direct_rader_stockham()
                .unwrap()
                .is_none()
        );
        assert_eq!(
            crate::ProgramIr::double_double_recursive(&grouped_48k)
                .unwrap()
                .passes
                .len(),
            5
        );

        let mut device_64k = device_48k;
        device_64k.shared_memory_bytes = 64 * 1024;
        device_64k.shared_memory_pow2_bytes = 64 * 1024;
        let grouped_64k = DoubleDoubleRecursiveFftIr::build_for_device(
            &grouped_plan,
            Direction::Forward,
            device_64k,
        )
        .unwrap();
        let DoubleDoubleRecursiveFftNodeIr::CooleyTukey(grouped_64k_root) = &grouped_64k.root
        else {
            panic!("DD N564 G3/64KiB witness should keep a Cooley root");
        };
        let grouped_64k_block = grouped_64k_root.pack_right.axis_batch_block.unwrap();
        assert_eq!(grouped_64k_block.threads_per_transform, 96);
        assert_eq!(grouped_64k_block.grouped_batch, 3);
        let grouped_fused = grouped_64k_root
            .fused_small_direct_rader_stockham()
            .unwrap()
            .expect("DD N564 G3 should fuse once 54,144 bytes fit");
        assert_eq!(
            grouped_fused.required_shared_memory_bytes().unwrap(),
            54_144
        );
        assert_eq!(
            crate::ProgramIr::double_double_recursive(&grouped_64k)
                .unwrap()
                .passes
                .len(),
            1
        );
        let grouped_shaders = crate::backend::vulkan::VulkanGlslBackend
            .lower_double_double_recursive(&grouped_64k)
            .unwrap();
        assert_eq!(grouped_shaders.len(), 1);
        assert_eq!(grouped_shaders[0].required_shared_memory_bytes, 54_144);
        grouped_shaders[0].compile_spirv().unwrap();

        single.validate().unwrap();
        grouped_48k.validate().unwrap();
        grouped_64k.validate().unwrap();
    }

    #[test]
    fn n611_portable_and_device_default_algorithms_remain_distinct() {
        let length = 13usize * 47;
        let mut device = DeviceProfile::generic(crate::Backend::Vulkan, crate::GpuVendor::Nvidia);
        device.shared_memory_bytes = 48 * 1024;
        device.shared_memory_pow2_bytes = 32 * 1024;
        device.max_threads_per_block = 1024;
        device.max_workgroup_size = [1024, 1024, 64];
        device.supports_f64 = true;

        let portable_plan = FftPlan::build(
            FftConfig::new(vec![length])
                .with_precision(Precision::DoubleDouble)
                .with_tuning(crate::PlannerTuning::portable()),
        )
        .unwrap();
        let portable = DoubleDoubleRecursiveFftIr::build_for_device(
            &portable_plan,
            Direction::Forward,
            device,
        )
        .unwrap();
        let DoubleDoubleRecursiveFftNodeIr::CooleyTukey(portable_root) = &portable.root else {
            panic!("portable DD N611 should retain a Cooley root");
        };
        assert_eq!((portable_root.left_len, portable_root.right_len), (13, 47));
        assert!(matches!(
            portable_root.left,
            DoubleDoubleRecursiveFftNodeIr::Stockham(_)
        ));
        assert!(matches!(
            portable_root.right,
            DoubleDoubleRecursiveFftNodeIr::DirectRader(_)
        ));
        let portable_block = portable_root.pack_right.axis_batch_block.unwrap();
        assert_eq!(portable_block.threads_per_transform, 128);
        assert_eq!(portable_block.grouped_batch, 1);
        assert_eq!(
            [portable_block.local_size_x, portable_block.local_size_y],
            [128, 1]
        );
        assert_eq!(
            crate::ProgramIr::double_double_recursive(&portable)
                .unwrap()
                .passes
                .len(),
            5
        );
        assert!(
            portable_root
                .fused_small_direct_rader_stockham()
                .unwrap()
                .is_none()
        );

        let device_plan = FftPlan::build_for_device(
            FftConfig::new(vec![length]).with_precision(Precision::DoubleDouble),
            device,
        )
        .unwrap();
        let AxisAlgorithm::Bluestein {
            convolution_len, ..
        } = &device_plan.axes[0].algorithm
        else {
            panic!("device-default DD N611 should select Bluestein");
        };
        let scheduled = crate::double_double_ir::DoubleDoubleOneDimIr::build_for_device(
            &device_plan,
            Direction::Forward,
            device,
        )
        .unwrap();
        let crate::double_double_ir::DoubleDoubleOneDimIr::Bluestein(device_bluestein) = scheduled
        else {
            panic!("device-default DD N611 should materialize a Bluestein pipeline");
        };
        assert_eq!(*convolution_len, 1_225);
        let wrapper = device_bluestein.wrapper_axis_batch_block.unwrap();
        assert_eq!(wrapper.threads_per_transform, 128);
        assert_eq!(wrapper.grouped_batch, 1);
        assert_eq!([wrapper.local_size_x, wrapper.local_size_y], [128, 1]);
        let DoubleDoubleBluesteinConvolutionIr::Stockham(forward_child) =
            &device_bluestein.forward_fft
        else {
            panic!("device-default DD N611 forward M1225 child should be Stockham");
        };
        let DoubleDoubleBluesteinConvolutionIr::Stockham(inverse_child) =
            &device_bluestein.inverse_fft
        else {
            panic!("device-default DD N611 inverse M1225 child should be Stockham");
        };
        assert_eq!(forward_child.sequence_len, 1_225);
        assert_eq!(inverse_child.sequence_len, 1_225);
        assert_eq!(forward_child.stages.len(), 4);
        assert_eq!(inverse_child.stages.len(), 4);
        let program = crate::ProgramIr::double_double_bluestein(&device_bluestein).unwrap();
        assert_eq!(program.passes.len(), 10);
        assert!(
            !program
                .resources
                .iter()
                .any(|resource| { resource.name == "double_double_bluestein_inverse_input" })
        );
        assert!(program.passes.iter().all(|pass| {
            !pass.name.contains("_promote")
                && !pass.name.contains("_finalize")
                && !pass.name.ends_with("_multiply")
        }));
        let inverse_stage0 = program
            .passes
            .iter()
            .find(|pass| {
                pass.name.contains("vkfft_dd_bluestein_inverse") && pass.name.contains("stage_0")
            })
            .expect("device-default DD N611 must retain inverse Stockham stage0");
        assert!(inverse_stage0.name.ends_with("_input_mul_lut"));
        assert!(inverse_stage0.bindings.iter().any(|binding| {
            binding.binding == 3
                && binding.role == crate::BufferRole::LookupTable
                && binding.access == crate::BufferAccess::ReadOnly
        }));
        let shaders = crate::backend::vulkan::VulkanGlslBackend
            .lower_double_double_bluestein(&device_bluestein)
            .unwrap();
        assert_eq!(shaders.len(), program.passes.len());
        let fused_shader = shaders
            .iter()
            .find(|shader| {
                shader.descriptors.iter().any(|descriptor| {
                    descriptor.binding == 3 && descriptor.role == crate::BufferRole::LookupTable
                })
            })
            .expect("device-default DD N611 inverse stage0 must bind kernel spectrum");
        assert!(fused_shader.glsl.contains("vkfft_lut"));
        for shader in &shaders {
            shader.compile_spirv().unwrap();
        }

        portable.validate().unwrap();
        device_bluestein.validate().unwrap();
    }

    #[test]
    fn composite_direct_rader_n329_n376_respect_32k_shared_capacity() {
        let mut device = DeviceProfile::generic(crate::Backend::Vulkan, crate::GpuVendor::Nvidia);
        device.shared_memory_bytes = 32 * 1024;
        device.shared_memory_pow2_bytes = 32 * 1024;
        device.max_threads_per_block = 1024;
        device.max_workgroup_size = [1024, 1024, 64];
        device.supports_f64 = true;
        for (length, should_fuse) in [(7usize * 47, true), (8usize * 47, false)] {
            let plan = FftPlan::build(
                FftConfig::new(vec![length])
                    .with_batch_count(5)
                    .with_grouped_batch(0, 3)
                    .unwrap()
                    .with_precision(Precision::DoubleDouble),
            )
            .unwrap();
            let ir =
                DoubleDoubleRecursiveFftIr::build_for_device(&plan, Direction::Forward, device)
                    .unwrap();
            let DoubleDoubleRecursiveFftNodeIr::CooleyTukey(root) = &ir.root else {
                panic!("DD N{length} capacity witness should keep a Cooley root");
            };
            let fused = root.fused_small_direct_rader_stockham().unwrap();
            assert_eq!(fused.is_some(), should_fuse);
            let program = crate::ProgramIr::double_double_recursive(&ir).unwrap();
            assert_eq!(program.passes.len(), if should_fuse { 1 } else { 5 });
            if let Some(fused) = fused {
                assert_eq!(fused.required_shared_memory_bytes().unwrap(), 31_584);
            }
            ir.validate().unwrap();
        }
    }

    #[test]
    fn composite_direct_rader_f64_storage_stays_unfused() {
        let mut device = DeviceProfile::generic(crate::Backend::Vulkan, crate::GpuVendor::Nvidia);
        device.shared_memory_bytes = 48 * 1024;
        device.shared_memory_pow2_bytes = 32 * 1024;
        device.max_threads_per_block = 1024;
        device.max_workgroup_size = [1024, 1024, 64];
        device.supports_f64 = true;
        let length = 2usize * 47;
        let plan = FftPlan::build(
            FftConfig::new(vec![length])
                .with_batch_count(5)
                .with_grouped_batch(0, 3)
                .unwrap()
                .with_precision(Precision::DoubleDoubleF64Storage),
        )
        .unwrap();
        let ir = DoubleDoubleRecursiveFftIr::build_for_device(&plan, Direction::Forward, device)
            .unwrap();
        let DoubleDoubleRecursiveFftNodeIr::CooleyTukey(root) = &ir.root else {
            panic!("DD/F64 N94 composite direct-Rader should keep a Cooley root");
        };
        assert!(root.fused_small_direct_rader_stockham().unwrap().is_none());
        let program = crate::ProgramIr::double_double_recursive(&ir).unwrap();
        assert_eq!(program.passes.len(), 5);
        let shaders = crate::backend::vulkan::VulkanGlslBackend
            .lower_double_double_recursive(&ir)
            .unwrap();
        assert_eq!(shaders.len(), 5);
        for shader in shaders {
            shader.compile_spirv().unwrap();
        }
    }

    #[test]
    fn composite_direct_rader_small_shared_budget_stays_unfused() {
        let mut device = DeviceProfile::generic(crate::Backend::Vulkan, crate::GpuVendor::Nvidia);
        device.shared_memory_bytes = 8 * 1024;
        device.shared_memory_pow2_bytes = 8 * 1024;
        device.max_threads_per_block = 1024;
        device.max_workgroup_size = [1024, 1024, 64];
        device.supports_f64 = true;
        let plan = FftPlan::build(
            FftConfig::new(vec![2usize * 47])
                .with_batch_count(5)
                .with_grouped_batch(0, 3)
                .unwrap()
                .with_precision(Precision::DoubleDouble),
        )
        .unwrap();
        let ir = DoubleDoubleRecursiveFftIr::build_for_device(&plan, Direction::Forward, device)
            .unwrap();
        let DoubleDoubleRecursiveFftNodeIr::CooleyTukey(root) = &ir.root else {
            panic!("DD N94 constrained-shared witness should keep a Cooley root");
        };
        assert_eq!(root.pack_right.axis_batch_block.unwrap().grouped_batch, 3);
        assert_eq!(root.pack_right.device_shared_memory_bytes, Some(8 * 1024));
        assert!(root.fused_small_direct_rader_stockham().unwrap().is_none());
        let program = crate::ProgramIr::double_double_recursive(&ir).unwrap();
        assert_eq!(program.passes.len(), 5);
        let shaders = crate::backend::vulkan::VulkanGlslBackend
            .lower_double_double_recursive(&ir)
            .unwrap();
        assert_eq!(shaders.len(), 5);
        assert!(
            shaders
                .iter()
                .all(|shader| shader.required_shared_memory_bytes <= device.shared_memory_bytes)
        );
        for shader in shaders {
            shader.compile_spirv().unwrap();
        }
    }

    #[test]
    fn mixed_direct_fft_rader_parent_uses_zero_smooth_quad_slice() {
        let mut device = DeviceProfile::generic(crate::Backend::Vulkan, crate::GpuVendor::Nvidia);
        device.shared_memory_bytes = 48 * 1024;
        device.shared_memory_pow2_bytes = 32 * 1024;
        device.max_threads_per_block = 256;
        device.max_workgroup_size = [256, 256, 64];
        device.supports_f64 = true;
        let length = 17usize * 31;
        let batch_count = 2usize;
        let mut tuning = crate::PlannerTuning::portable();
        tuning.min_rader_fft_prime = 19;
        tuning.validate().unwrap();
        let plan = FftPlan::build(
            FftConfig::new(vec![length])
                .with_batch_count(batch_count)
                .with_precision(Precision::DoubleDouble)
                .with_tuning(tuning),
        )
        .unwrap();
        let ir = DoubleDoubleRecursiveFftIr::build_for_device(&plan, Direction::Forward, device)
            .unwrap();
        assert!(ir.rader_forced_upload_schedule.is_none());
        let DoubleDoubleRecursiveFftNodeIr::CooleyTukey(root) = &ir.root else {
            panic!("DD N527 mixed Rader should keep a Cooley root");
        };
        assert_eq!((root.left_len, root.right_len), (17, 31));
        assert!(matches!(
            root.left,
            DoubleDoubleRecursiveFftNodeIr::DirectRader(_)
        ));
        assert!(matches!(
            root.right,
            DoubleDoubleRecursiveFftNodeIr::FftRader(_)
        ));
        let block = root.pack_right.axis_batch_block.unwrap();
        assert_eq!(block.threads_per_transform, 108);
        assert_eq!(block.grouped_batch, 1);
        assert_eq!([block.local_size_x, block.local_size_y], [108, 1]);
        assert_eq!(root.twiddle_transpose.axis_batch_block, Some(block));
        assert_eq!(root.scatter_output.axis_batch_block, Some(block));

        let shaders = crate::backend::vulkan::VulkanGlslBackend
            .lower_double_double_recursive(&ir)
            .unwrap();
        for shader in shaders {
            shader.compile_spirv().unwrap();
        }
        let mut input = vec![ComplexDoubleDouble::default(); length * batch_count];
        for batch in 0..batch_count {
            input[batch * length + 1] =
                ComplexDoubleDouble::new(DoubleDouble::from_f64(1.0), DoubleDouble::default());
        }
        let actual = execute_double_double_recursive_ir(&ir, &input).unwrap();
        let error = actual
            .iter()
            .enumerate()
            .map(|(index, value)| {
                let k = index % length;
                let angle = -std::f64::consts::TAU * k as f64 / length as f64;
                let expected = ComplexDoubleDouble::new(
                    DoubleDouble::from_f64(angle.cos()),
                    DoubleDouble::from_f64(angle.sin()),
                );
                let dr = (value.re - expected.re).to_f64();
                let di = (value.im - expected.im).to_f64();
                dr.hypot(di)
            })
            .fold(0.0f64, f64::max);
        assert!(
            error <= 1.0e-14,
            "DD N527 mixed Direct+FFT-Rader impulse mismatch: {error:e}"
        );
        ir.validate().unwrap();
    }

    #[test]
    fn mixed_rader_parent_uses_quad_outer_register_floor_with_smooth_radix() {
        let mut device = DeviceProfile::generic(crate::Backend::Vulkan, crate::GpuVendor::Nvidia);
        device.shared_memory_bytes = 512 * 1024;
        device.shared_memory_pow2_bytes = 512 * 1024;
        device.max_threads_per_block = 1024;
        device.max_workgroup_size = [1024, 1024, 64];
        device.supports_f64 = true;
        let length = 15usize * 17 * 31;
        let mut tuning = crate::PlannerTuning::portable();
        tuning.min_rader_fft_prime = 19;
        tuning.validate().unwrap();
        let plan = FftPlan::build(
            FftConfig::new(vec![length])
                .with_precision(Precision::DoubleDouble)
                .with_tuning(tuning),
        )
        .unwrap();
        let ir = DoubleDoubleRecursiveFftIr::build_for_device(&plan, Direction::Forward, device)
            .unwrap();
        assert!(ir.rader_forced_upload_schedule.is_none());
        let DoubleDoubleRecursiveFftNodeIr::CooleyTukey(root) = &ir.root else {
            panic!("DD N7905 mixed Rader should keep a Cooley root");
        };
        let block = root.pack_right.axis_batch_block.unwrap();
        assert_eq!(block.threads_per_transform, 990);
        assert_eq!(block.grouped_batch, 1);
        assert_eq!([block.local_size_x, block.local_size_y], [990, 1]);
        assert_eq!(root.twiddle_transpose.axis_batch_block, Some(block));
        assert_eq!(root.scatter_output.axis_batch_block, Some(block));

        let shaders = crate::backend::vulkan::VulkanGlslBackend
            .lower_double_double_recursive(&ir)
            .unwrap();
        for shader in shaders {
            shader.compile_spirv().unwrap();
        }
        let mut input = vec![ComplexDoubleDouble::default(); length];
        input[1] = ComplexDoubleDouble::new(DoubleDouble::from_f64(1.0), DoubleDouble::default());
        let actual = execute_double_double_recursive_ir(&ir, &input).unwrap();
        let error = actual
            .iter()
            .enumerate()
            .map(|(k, value)| {
                let angle = -std::f64::consts::TAU * k as f64 / length as f64;
                let expected = ComplexDoubleDouble::new(
                    DoubleDouble::from_f64(angle.cos()),
                    DoubleDouble::from_f64(angle.sin()),
                );
                let dr = (value.re - expected.re).to_f64();
                let di = (value.im - expected.im).to_f64();
                dr.hypot(di)
            })
            .fold(0.0f64, f64::max);
        assert!(
            error <= 2.0e-13,
            "DD N7905 mixed Rader impulse mismatch: {error:e}"
        );
        ir.validate().unwrap();
    }

    #[test]
    fn mixed_multi_fft_rader_global_scaling_reaches_dd_recursive_parent() {
        let mut device = DeviceProfile::generic(crate::Backend::Vulkan, crate::GpuVendor::Nvidia);
        device.shared_memory_bytes = 512 * 1024;
        device.shared_memory_pow2_bytes = 512 * 1024;
        device.max_threads_per_block = 1024;
        device.max_workgroup_size = [1024, 1024, 64];
        device.supports_f64 = true;
        let length = 17usize * 19 * 29;
        let mut tuning = crate::PlannerTuning::portable();
        tuning.min_rader_direct_prime = 17;
        tuning.min_rader_fft_prime = 19;
        tuning.validate().unwrap();
        let plan = FftPlan::build(
            FftConfig::new(vec![length])
                .with_precision(Precision::DoubleDouble)
                .with_tuning(tuning),
        )
        .unwrap();
        let AxisAlgorithm::Rader { primes, .. } = &plan.axes[0].algorithm else {
            panic!("DD N9367 should retain mixed multi-Rader metadata");
        };
        assert_eq!(primes.len(), 3);
        assert_eq!(primes[0].prime, 17);
        assert!(matches!(primes[0].mode, RaderMode::DirectMultiplication));
        assert_eq!(primes[1].prime, 19);
        assert!(matches!(primes[1].mode, RaderMode::FftConvolution { .. }));
        assert_eq!(primes[2].prime, 29);
        assert!(matches!(primes[2].mode, RaderMode::FftConvolution { .. }));
        assert!(
            !crate::scheduler::plan_gpu_force_rader_two_upload(length, &[19, 29], device).unwrap()
        );

        let ir = DoubleDoubleRecursiveFftIr::build_for_device(&plan, Direction::Forward, device)
            .unwrap();
        assert!(ir.rader_forced_upload_schedule.is_none());
        assert!(ir.two_upload_four_step_plan.is_none());
        assert!(ir.three_upload_four_step_plan.is_none());
        let DoubleDoubleRecursiveFftNodeIr::CooleyTukey(root) = &ir.root else {
            panic!("DD N9367 mixed multi-Rader should keep a Cooley root");
        };
        let block = root.pack_right.axis_batch_block.unwrap();
        assert_eq!(block.threads_per_transform, 999);
        assert_eq!(block.grouped_batch, 1);
        assert_eq!([block.local_size_x, block.local_size_y], [999, 1]);
        assert_eq!(root.twiddle_transpose.axis_batch_block, Some(block));
        assert_eq!(root.scatter_output.axis_batch_block, Some(block));

        fn count_modes(node: &DoubleDoubleRecursiveFftNodeIr) -> (usize, usize) {
            match node {
                DoubleDoubleRecursiveFftNodeIr::DirectRader(_) => (1, 0),
                DoubleDoubleRecursiveFftNodeIr::FftRader(_) => (0, 1),
                DoubleDoubleRecursiveFftNodeIr::CooleyTukey(ir) => {
                    let left = count_modes(&ir.left);
                    let right = count_modes(&ir.right);
                    (left.0 + right.0, left.1 + right.1)
                }
                DoubleDoubleRecursiveFftNodeIr::Stockham(_)
                | DoubleDoubleRecursiveFftNodeIr::Bluestein(_) => (0, 0),
            }
        }
        assert_eq!(count_modes(&ir.root), (1, 2));

        let shaders = crate::backend::vulkan::VulkanGlslBackend
            .lower_double_double_recursive(&ir)
            .unwrap();
        for shader in shaders {
            shader.compile_spirv().unwrap();
        }
        let mut input = vec![ComplexDoubleDouble::default(); length];
        input[1] = ComplexDoubleDouble::new(DoubleDouble::from_f64(1.0), DoubleDouble::default());
        let actual = execute_double_double_recursive_ir(&ir, &input).unwrap();
        let error = actual
            .iter()
            .enumerate()
            .map(|(k, value)| {
                let angle = -std::f64::consts::TAU * k as f64 / length as f64;
                let expected = ComplexDoubleDouble::new(
                    DoubleDouble::from_f64(angle.cos()),
                    DoubleDouble::from_f64(angle.sin()),
                );
                let dr = (value.re - expected.re).to_f64();
                let di = (value.im - expected.im).to_f64();
                dr.hypot(di)
            })
            .fold(0.0f64, f64::max);
        assert!(
            error <= 3.0e-13,
            "DD N9367 mixed multi-Rader impulse mismatch: {error:e}"
        );
        ir.validate().unwrap();
    }

    #[test]
    fn mixed_multi_fft_rader_forced256_uses_pass_local_quad_floor() {
        let mut device = DeviceProfile::generic(crate::Backend::Vulkan, crate::GpuVendor::Nvidia);
        device.shared_memory_bytes = 48 * 1024;
        device.shared_memory_pow2_bytes = 32 * 1024;
        device.max_threads_per_block = 256;
        device.max_workgroup_size = [1024, 1024, 64];
        device.supports_f64 = true;
        let length = 17usize * 19 * 29;
        let mut tuning = crate::PlannerTuning::portable();
        tuning.min_rader_direct_prime = 17;
        tuning.min_rader_fft_prime = 19;
        tuning.validate().unwrap();
        let plan = FftPlan::build(
            FftConfig::new(vec![length])
                .with_precision(Precision::DoubleDouble)
                .with_tuning(tuning),
        )
        .unwrap();
        assert!(
            crate::scheduler::plan_gpu_force_rader_two_upload(length, &[19, 29], device).unwrap()
        );

        let ir = DoubleDoubleRecursiveFftIr::build_for_device(&plan, Direction::Forward, device)
            .unwrap();
        let schedule = ir
            .rader_forced_upload_schedule
            .as_ref()
            .expect("DD N9367 should force two uploads under a 256-thread cap");
        assert_eq!(schedule.axis_split, vec![323, 29]);
        let DoubleDoubleRecursiveFftNodeIr::CooleyTukey(root) = &ir.root else {
            panic!("DD N9367 forced256 should keep a Cooley root");
        };
        assert_eq!((root.left_len, root.right_len), (323, 29));
        assert!(matches!(
            root.right,
            DoubleDoubleRecursiveFftNodeIr::FftRader(ref rader) if rader.prime == 29
        ));
        let DoubleDoubleRecursiveFftNodeIr::CooleyTukey(low) = &root.left else {
            panic!("DD N9367 forced256 upload0 should remain p17+p19 Cooley");
        };
        let block = low.pack_right.axis_batch_block.unwrap();
        assert_eq!(block.threads_per_transform, 162);
        assert_eq!(block.grouped_batch, 1);
        assert!(!block.axis_swapped);
        assert_eq!([block.local_size_x, block.local_size_y], [162, 1]);
        assert_eq!(low.twiddle_transpose.axis_batch_block, Some(block));
        assert_eq!(low.scatter_output.axis_batch_block, Some(block));
        let mapped_high = ir
            .forced_rader_two_upload_mapped_high_component()
            .unwrap()
            .expect("DD N9367 p29 upload1 should own the Four-step input boundary");
        let DoubleDoubleRecursiveFftNodeIr::FftRader(high_rader) = mapped_high else {
            panic!("DD N9367 mapped upload1 should remain p29 FFT-Rader");
        };
        assert_eq!(high_rader.prime, 29);
        let high_block = high_rader.caller_axis_batch_block.unwrap();
        assert_eq!(high_block.threads_per_transform, 5);
        assert_eq!(high_block.grouped_batch, 24);
        assert!(!high_block.axis_swapped);
        assert_eq!([high_block.local_size_x, high_block.local_size_y], [24, 5]);
        let mapped_low = ir
            .forced_rader_two_upload_mapped_low_component()
            .unwrap()
            .expect("DD N9367 p17+p19 upload0 should own the final Four-step scatter");
        let DoubleDoubleRecursiveFftNodeIr::CooleyTukey(mapped_low) = mapped_low else {
            panic!("DD N9367 mapped forced256 upload0 should remain Cooley");
        };
        assert_eq!(mapped_low.pack_right.axis_batch_block, Some(block));
        assert_eq!(mapped_low.twiddle_transpose.axis_batch_block, Some(block));
        assert_eq!(mapped_low.scatter_output.axis_batch_block, Some(block));

        let shaders = crate::backend::vulkan::VulkanGlslBackend
            .lower_double_double_recursive(&ir)
            .unwrap();
        for shader in shaders {
            shader.compile_spirv().unwrap();
        }
        let mut input = vec![ComplexDoubleDouble::default(); length];
        input[1] = ComplexDoubleDouble::new(DoubleDouble::from_f64(1.0), DoubleDouble::default());
        let actual = execute_double_double_recursive_ir(&ir, &input).unwrap();
        let error = actual
            .iter()
            .enumerate()
            .map(|(k, value)| {
                let angle = -std::f64::consts::TAU * k as f64 / length as f64;
                let expected = ComplexDoubleDouble::new(
                    DoubleDouble::from_f64(angle.cos()),
                    DoubleDouble::from_f64(angle.sin()),
                );
                let dr = (value.re - expected.re).to_f64();
                let di = (value.im - expected.im).to_f64();
                dr.hypot(di)
            })
            .fold(0.0f64, f64::max);
        assert!(
            error <= 3.0e-13,
            "DD N9367 forced256 impulse mismatch: {error:e}"
        );
        ir.validate().unwrap();
    }

    #[test]
    fn forced_rader_upload0_uses_pass_local_dd_coalescing_on_intel() {
        let mut device = DeviceProfile::generic(crate::Backend::OpenCl, crate::GpuVendor::Intel);
        device.shared_memory_bytes = 48 * 1024;
        device.shared_memory_pow2_bytes = 32 * 1024;
        device.max_threads_per_block = 32;
        device.max_workgroup_size = [32, 32, 64];
        device.supports_f64 = true;
        let length = 2usize * 17 * 31;
        let mut tuning = crate::PlannerTuning::portable();
        tuning.min_rader_direct_prime = 17;
        tuning.max_rader_direct_prime = 31;
        tuning.min_rader_fft_prime = 19;
        tuning.validate().unwrap();
        let plan = FftPlan::build(
            FftConfig::new(vec![length])
                .with_precision(Precision::DoubleDouble)
                .with_tuning(tuning),
        )
        .unwrap();
        assert!(crate::scheduler::plan_gpu_force_rader_two_upload(length, &[31], device).unwrap());

        let ir = DoubleDoubleRecursiveFftIr::build_for_device(&plan, Direction::Forward, device)
            .unwrap();
        let schedule = ir
            .rader_forced_upload_schedule
            .as_ref()
            .expect("DD Intel N1054 should force two uploads under a 32-thread cap");
        assert_eq!(schedule.axis_split, vec![34, 31]);
        let DoubleDoubleRecursiveFftNodeIr::CooleyTukey(root) = &ir.root else {
            panic!("DD Intel N1054 should keep a Cooley root");
        };
        assert_eq!((root.left_len, root.right_len), (34, 31));
        assert!(matches!(
            root.right,
            DoubleDoubleRecursiveFftNodeIr::FftRader(ref rader) if rader.prime == 31
        ));
        let DoubleDoubleRecursiveFftNodeIr::CooleyTukey(low) = &root.left else {
            panic!("DD Intel N1054 upload0 should remain 2 x p17 Cooley");
        };
        let block = low.pack_right.axis_batch_block.unwrap();
        assert_eq!(block.threads_per_transform, 9);
        assert_eq!(block.grouped_batch, 2);
        assert!(block.transforms_on_x);
        assert!(block.axis_swapped);
        assert_eq!([block.local_size_x, block.local_size_y], [2, 9]);
        assert_eq!(low.twiddle_transpose.axis_batch_block, Some(block));
        assert_eq!(low.scatter_output.axis_batch_block, Some(block));
        let mapped_high = ir
            .forced_rader_two_upload_mapped_high_component()
            .unwrap()
            .expect("DD Intel N1054 p31 upload1 should own the Four-step input boundary");
        let DoubleDoubleRecursiveFftNodeIr::FftRader(high_rader) = mapped_high else {
            panic!("DD Intel N1054 mapped upload1 should remain p31 FFT-Rader");
        };
        assert_eq!(high_rader.prime, 31);
        let high_block = high_rader.caller_axis_batch_block.unwrap();
        assert_eq!(high_block.threads_per_transform, 7);
        assert_eq!(high_block.grouped_batch, 4);
        assert!(!high_block.axis_swapped);
        assert_eq!([high_block.local_size_x, high_block.local_size_y], [4, 7]);
        let mapped_low = ir
            .forced_rader_two_upload_mapped_low_component()
            .unwrap()
            .expect("DD Intel N1054 N34 upload0 should own the final Four-step scatter");
        let DoubleDoubleRecursiveFftNodeIr::CooleyTukey(mapped_low) = mapped_low else {
            panic!("DD Intel N1054 mapped upload0 should remain Cooley");
        };
        assert_eq!(mapped_low.pack_right.axis_batch_block, Some(block));
        assert_eq!(mapped_low.twiddle_transpose.axis_batch_block, Some(block));
        assert_eq!(mapped_low.scatter_output.axis_batch_block, Some(block));

        let shaders = crate::backend::vulkan::VulkanGlslBackend
            .lower_double_double_recursive(&ir)
            .unwrap();
        for shader in shaders {
            shader.compile_spirv().unwrap();
        }
        let mut input = vec![ComplexDoubleDouble::default(); length];
        input[1] = ComplexDoubleDouble::new(DoubleDouble::from_f64(1.0), DoubleDouble::default());
        let actual = execute_double_double_recursive_ir(&ir, &input).unwrap();
        let error = actual
            .iter()
            .enumerate()
            .map(|(k, value)| {
                let angle = -std::f64::consts::TAU * k as f64 / length as f64;
                let expected = ComplexDoubleDouble::new(
                    DoubleDouble::from_f64(angle.cos()),
                    DoubleDouble::from_f64(angle.sin()),
                );
                let dr = (value.re - expected.re).to_f64();
                let di = (value.im - expected.im).to_f64();
                dr.hypot(di)
            })
            .fold(0.0f64, f64::max);
        assert!(
            error <= 3.0e-13,
            "DD Intel N1054 impulse mismatch: {error:e}"
        );
        ir.validate().unwrap();
    }

    #[test]
    fn repeated_fft_rader_global_scaling_reaches_dd_recursive_parent() {
        let mut device = DeviceProfile::generic(crate::Backend::Vulkan, crate::GpuVendor::Nvidia);
        device.shared_memory_bytes = 512 * 1024;
        device.shared_memory_pow2_bytes = 512 * 1024;
        device.max_threads_per_block = 1024;
        device.max_workgroup_size = [1024, 1024, 64];
        device.supports_f64 = true;
        let length = 19usize * 19 * 23;
        let mut tuning = crate::PlannerTuning::portable();
        tuning.min_rader_fft_prime = 19;
        tuning.validate().unwrap();
        let plan = FftPlan::build(
            FftConfig::new(vec![length])
                .with_precision(Precision::DoubleDouble)
                .with_tuning(tuning),
        )
        .unwrap();
        let AxisAlgorithm::Rader { primes, .. } = &plan.axes[0].algorithm else {
            panic!("DD N8303 should retain repeated FFT-Rader metadata");
        };
        assert_eq!(primes.len(), 2);
        assert_eq!((primes[0].prime, primes[0].multiplicity), (19, 2));
        assert_eq!((primes[1].prime, primes[1].multiplicity), (23, 1));
        assert!(
            primes
                .iter()
                .all(|prime| matches!(prime.mode, RaderMode::FftConvolution { .. }))
        );

        let ir = DoubleDoubleRecursiveFftIr::build_for_device(&plan, Direction::Forward, device)
            .unwrap();
        assert!(ir.rader_forced_upload_schedule.is_none());
        assert!(ir.two_upload_four_step_plan.is_none());
        assert!(ir.three_upload_four_step_plan.is_none());
        let DoubleDoubleRecursiveFftNodeIr::CooleyTukey(root) = &ir.root else {
            panic!("DD N8303 repeated FFT-Rader should keep a Cooley root");
        };
        let block = root.pack_right.axis_batch_block.unwrap();
        assert_eq!(block.threads_per_transform, 437);
        assert_eq!(block.grouped_batch, 1);
        assert_eq!([block.local_size_x, block.local_size_y], [437, 1]);
        assert_eq!(root.twiddle_transpose.axis_batch_block, Some(block));
        assert_eq!(root.scatter_output.axis_batch_block, Some(block));

        fn count_fft_rader_primes(node: &DoubleDoubleRecursiveFftNodeIr, prime: usize) -> usize {
            match node {
                DoubleDoubleRecursiveFftNodeIr::FftRader(ir) => usize::from(ir.prime == prime),
                DoubleDoubleRecursiveFftNodeIr::CooleyTukey(ir) => {
                    count_fft_rader_primes(&ir.left, prime)
                        + count_fft_rader_primes(&ir.right, prime)
                }
                DoubleDoubleRecursiveFftNodeIr::Stockham(_)
                | DoubleDoubleRecursiveFftNodeIr::DirectRader(_)
                | DoubleDoubleRecursiveFftNodeIr::Bluestein(_) => 0,
            }
        }
        assert_eq!(count_fft_rader_primes(&ir.root, 19), 2);
        assert_eq!(count_fft_rader_primes(&ir.root, 23), 1);

        let shaders = crate::backend::vulkan::VulkanGlslBackend
            .lower_double_double_recursive(&ir)
            .unwrap();
        for shader in shaders {
            shader.compile_spirv().unwrap();
        }
        let mut input = vec![ComplexDoubleDouble::default(); length];
        input[1] = ComplexDoubleDouble::new(DoubleDouble::from_f64(1.0), DoubleDouble::default());
        let actual = execute_double_double_recursive_ir(&ir, &input).unwrap();
        let error = actual
            .iter()
            .enumerate()
            .map(|(k, value)| {
                let angle = -std::f64::consts::TAU * k as f64 / length as f64;
                let expected = ComplexDoubleDouble::new(
                    DoubleDouble::from_f64(angle.cos()),
                    DoubleDouble::from_f64(angle.sin()),
                );
                let dr = (value.re - expected.re).to_f64();
                let di = (value.im - expected.im).to_f64();
                dr.hypot(di)
            })
            .fold(0.0f64, f64::max);
        assert!(
            error <= 3.0e-13,
            "DD N8303 repeated FFT-Rader impulse mismatch: {error:e}"
        );
        ir.validate().unwrap();
    }

    #[test]
    fn repeated_direct_rader_parent_uses_quad_container_multiplier() {
        let mut device = DeviceProfile::generic(crate::Backend::Vulkan, crate::GpuVendor::Nvidia);
        device.shared_memory_bytes = 48 * 1024;
        device.shared_memory_pow2_bytes = 32 * 1024;
        device.max_threads_per_block = 256;
        device.max_workgroup_size = [256, 256, 64];
        device.supports_f64 = true;
        let length = 17usize * 17;
        let batch_count = 2usize;
        let mut tuning = crate::PlannerTuning::portable();
        tuning.min_rader_fft_prime = 29;
        tuning.validate().unwrap();
        let plan = FftPlan::build(
            FftConfig::new(vec![length])
                .with_batch_count(batch_count)
                .with_precision(Precision::DoubleDouble)
                .with_tuning(tuning),
        )
        .unwrap();
        let ir = DoubleDoubleRecursiveFftIr::build_for_device(&plan, Direction::Forward, device)
            .unwrap();
        assert!(ir.rader_forced_upload_schedule.is_none());
        let DoubleDoubleRecursiveFftNodeIr::CooleyTukey(root) = &ir.root else {
            panic!("DD N289 repeated direct-Rader should keep a Cooley root");
        };
        assert_eq!((root.left_len, root.right_len), (17, 17));
        assert!(matches!(
            root.left,
            DoubleDoubleRecursiveFftNodeIr::DirectRader(_)
        ));
        assert!(matches!(
            root.right,
            DoubleDoubleRecursiveFftNodeIr::DirectRader(_)
        ));
        let block = root.pack_right.axis_batch_block.unwrap();
        assert_eq!(block.threads_per_transform, 153);
        assert_eq!(block.grouped_batch, 1);
        assert_eq!([block.local_size_x, block.local_size_y], [153, 1]);
        assert_eq!(root.twiddle_transpose.axis_batch_block, Some(block));
        assert_eq!(root.scatter_output.axis_batch_block, Some(block));

        let shaders = crate::backend::vulkan::VulkanGlslBackend
            .lower_double_double_recursive(&ir)
            .unwrap();
        for shader in shaders {
            shader.compile_spirv().unwrap();
        }
        let mut input = vec![ComplexDoubleDouble::default(); length * batch_count];
        for batch in 0..batch_count {
            input[batch * length + 1] =
                ComplexDoubleDouble::new(DoubleDouble::from_f64(1.0), DoubleDouble::default());
        }
        let actual = execute_double_double_recursive_ir(&ir, &input).unwrap();
        let error = actual
            .iter()
            .enumerate()
            .map(|(index, value)| {
                let k = index % length;
                let angle = -std::f64::consts::TAU * k as f64 / length as f64;
                let expected = ComplexDoubleDouble::new(
                    DoubleDouble::from_f64(angle.cos()),
                    DoubleDouble::from_f64(angle.sin()),
                );
                let dr = (value.re - expected.re).to_f64();
                let di = (value.im - expected.im).to_f64();
                dr.hypot(di)
            })
            .fold(0.0f64, f64::max);
        assert!(
            error <= 5.0e-15,
            "DD N289 repeated direct-Rader impulse mismatch: {error:e}"
        );
        ir.validate().unwrap();
    }

    #[test]
    fn multi_direct_rader_parent_uses_quad_shared_type1_scale() {
        let mut device = DeviceProfile::generic(crate::Backend::Vulkan, crate::GpuVendor::Nvidia);
        device.shared_memory_bytes = 128 * 1024;
        device.shared_memory_pow2_bytes = 128 * 1024;
        device.max_threads_per_block = 256;
        device.max_workgroup_size = [256, 256, 64];
        device.supports_f64 = true;
        let length = 47usize * 53;
        let batch_count = 2usize;
        let mut tuning = crate::PlannerTuning::portable();
        tuning.min_rader_fft_prime = 89;
        tuning.validate().unwrap();
        let plan = FftPlan::build(
            FftConfig::new(vec![length])
                .with_batch_count(batch_count)
                .with_precision(Precision::DoubleDouble)
                .with_tuning(tuning),
        )
        .unwrap();
        let ir = DoubleDoubleRecursiveFftIr::build_for_device(&plan, Direction::Forward, device)
            .unwrap();
        assert!(ir.rader_forced_upload_schedule.is_none());
        let DoubleDoubleRecursiveFftNodeIr::CooleyTukey(root) = &ir.root else {
            panic!("DD p47*p53 all-direct Rader axis should keep a Cooley root");
        };
        assert!(matches!(
            root.left,
            DoubleDoubleRecursiveFftNodeIr::DirectRader(_)
        ));
        assert!(matches!(
            root.right,
            DoubleDoubleRecursiveFftNodeIr::DirectRader(_)
        ));
        let block = root.pack_right.axis_batch_block.unwrap();
        assert_eq!(block.threads_per_transform, 216);
        assert_eq!(block.grouped_batch, 1);
        assert_eq!([block.local_size_x, block.local_size_y], [216, 1]);
        assert_eq!(root.twiddle_transpose.axis_batch_block, Some(block));
        assert_eq!(root.scatter_output.axis_batch_block, Some(block));

        let shaders = crate::backend::vulkan::VulkanGlslBackend
            .lower_double_double_recursive(&ir)
            .unwrap();
        for shader in shaders {
            shader.compile_spirv().unwrap();
        }
        let mut input = vec![ComplexDoubleDouble::default(); length * batch_count];
        for batch in 0..batch_count {
            input[batch * length + 1] =
                ComplexDoubleDouble::new(DoubleDouble::from_f64(1.0), DoubleDouble::default());
        }
        let actual = execute_double_double_recursive_ir(&ir, &input).unwrap();
        let error = actual
            .iter()
            .enumerate()
            .map(|(index, value)| {
                let k = index % length;
                let angle = -std::f64::consts::TAU * k as f64 / length as f64;
                let expected = ComplexDoubleDouble::new(
                    DoubleDouble::from_f64(angle.cos()),
                    DoubleDouble::from_f64(angle.sin()),
                );
                let dr = (value.re - expected.re).to_f64();
                let di = (value.im - expected.im).to_f64();
                dr.hypot(di)
            })
            .fold(0.0f64, f64::max);
        assert!(
            error <= 5.0e-14,
            "DD p47*p53 composite direct-Rader impulse mismatch: {error:e}"
        );
        ir.validate().unwrap();
    }

    #[test]
    fn forced_rader_bandwidth_context_reaches_dd_root_cut() {
        let mut intel = DeviceProfile::generic(crate::Backend::Vulkan, crate::GpuVendor::Intel);
        intel.shared_memory_bytes = 16 * 1024;
        intel.shared_memory_pow2_bytes = 16 * 1024;
        intel.max_threads_per_block = 1024;
        intel.max_workgroup_size = [1024, 1024, 64];
        intel.supports_f64 = true;
        let length = 17usize * 4_096;

        let strided_config = FftConfig::new(vec![length])
            .with_precision(Precision::DoubleDouble)
            .resolve_tuning_for_device(intel);
        let strided_plan =
            FftPlan::build_c2c_child_for_device(strided_config, intel, C2cDeviceAxisClass::Strided)
                .unwrap();
        let strided =
            DoubleDoubleRecursiveFftIr::build_for_device(&strided_plan, Direction::Forward, intel)
                .unwrap();
        let strided_schedule = strided
            .rader_forced_upload_schedule
            .as_ref()
            .expect("Intel strided DD composite p17 must retain forced-Rader scheduling");
        assert_eq!(strided_schedule.upload_count, 3);
        assert_eq!(strided_schedule.axis_split, vec![64, 34, 32]);
        assert!(rader_upload_split_matches_root(
            &strided.root,
            &strided_schedule.axis_split
        ));
        strided.validate().unwrap();

        let boosted_config = FftConfig::new(vec![length])
            .with_precision(Precision::DoubleDouble)
            .with_bandwidth_boost(2)
            .resolve_tuning_for_device(intel);
        let boosted_plan =
            FftPlan::build_c2c_child_for_device(boosted_config, intel, C2cDeviceAxisClass::Strided)
                .unwrap();
        let boosted =
            DoubleDoubleRecursiveFftIr::build_for_device(&boosted_plan, Direction::Forward, intel)
                .unwrap();
        let boosted_schedule = boosted
            .rader_forced_upload_schedule
            .as_ref()
            .expect("B=2 Intel DD composite p17 must keep forced-Rader scheduling");
        assert_eq!(boosted_schedule.upload_count, 2);
        assert_eq!(boosted_schedule.axis_split, vec![272, 256]);
        assert!(rader_upload_split_matches_root(
            &boosted.root,
            &boosted_schedule.axis_split
        ));
        assert!(
            boosted
                .forced_rader_two_upload_mapped_high_component()
                .unwrap()
                .is_none(),
            "256-point high upload is monolithic Stockham"
        );
        let program = crate::ProgramIr::double_double_recursive(&boosted).unwrap();
        program.validate().unwrap();
        let shaders = crate::backend::vulkan::VulkanGlslBackend
            .lower_double_double_recursive(&boosted)
            .unwrap();
        assert_eq!(shaders.len(), program.passes.len());
        for shader in shaders {
            assert_eq!(shader.compile_spirv().unwrap().words[0], 0x0723_0203);
        }
    }

    #[test]
    fn direct_rader_normal_capacity_materializes_dd_two_upload_mapping() {
        let length = 4usize * 11 * 23;
        let device = DeviceProfile {
            shared_memory_bytes: 32 * 1024,
            shared_memory_pow2_bytes: 32 * 1024,
            max_threads_per_block: 1024,
            max_workgroup_size: [1024, 1024, 64],
            supports_f64: true,
            ..DeviceProfile::generic(crate::Backend::Vulkan, crate::GpuVendor::Nvidia)
        };
        let plan = FftPlan::build_for_device(
            FftConfig::new(vec![length]).with_precision(Precision::DoubleDouble),
            device,
        )
        .unwrap();
        let AxisAlgorithm::Rader { primes, .. } = &plan.axes[0].algorithm else {
            panic!("DD N1012 should retain direct-Rader metadata");
        };
        assert_eq!(primes.len(), 2);
        assert_eq!(primes[0].prime, 11);
        assert_eq!(primes[1].prime, 23);
        assert!(
            primes
                .iter()
                .all(|prime| matches!(prime.mode, RaderMode::DirectMultiplication))
        );

        let ir = DoubleDoubleRecursiveFftIr::build_for_device(&plan, Direction::Forward, device)
            .unwrap();
        let schedule = ir
            .rader_forced_upload_schedule
            .as_ref()
            .expect("DD N1012 normal capacity must materialize Rader multi-upload metadata");
        assert_eq!(schedule.axis_split, vec![44, 23]);
        assert_eq!(schedule.upload_count, 2);
        assert!(ir.two_upload_four_step_plan.is_none());
        assert!(ir.three_upload_four_step_plan.is_none());
        assert!(matches!(
            ir.forced_rader_two_upload_mapped_high_component().unwrap(),
            Some(DoubleDoubleRecursiveFftNodeIr::DirectRader(ref rader)) if rader.prime == 23
        ));
        assert!(
            ir.forced_rader_two_upload_mapped_low_component()
                .unwrap()
                .is_some()
        );

        let program = crate::ProgramIr::double_double_recursive(&ir).unwrap();
        program.validate().unwrap();
        let shaders = crate::backend::vulkan::VulkanGlslBackend
            .lower_double_double_recursive(&ir)
            .unwrap();
        assert_eq!(shaders.len(), program.passes.len());
        for shader in shaders {
            shader.compile_spirv().unwrap();
        }
        let mut input = vec![ComplexDoubleDouble::default(); length];
        input[1] = ComplexDoubleDouble::new(DoubleDouble::from_f64(1.0), DoubleDouble::default());
        let actual = execute_double_double_recursive_ir(&ir, &input).unwrap();
        let error = actual
            .iter()
            .enumerate()
            .map(|(k, value)| {
                let angle = -std::f64::consts::TAU * k as f64 / length as f64;
                let dr = value.re.to_f64() - angle.cos();
                let di = value.im.to_f64() - angle.sin();
                dr.hypot(di)
            })
            .fold(0.0f64, f64::max);
        assert!(error <= 3.0e-13, "DD N1012 impulse mismatch: {error:e}");
        ir.validate().unwrap();
    }

    #[test]
    fn forced_rader_two_upload_controls_dd_root_cut_and_preserves_results() {
        let length = 17usize * 300;
        let device = DeviceProfile {
            shared_memory_bytes: 48 * 1024,
            shared_memory_pow2_bytes: 32 * 1024,
            max_threads_per_block: 128,
            max_workgroup_size: [1024, 1024, 64],
            supports_f64: true,
            ..DeviceProfile::generic(crate::Backend::Vulkan, crate::GpuVendor::Nvidia)
        };
        let plan =
            FftPlan::build(FftConfig::new(vec![length]).with_precision(Precision::DoubleDouble))
                .unwrap();
        assert!(matches!(
            plan.axes[0].algorithm,
            AxisAlgorithm::Rader { .. }
        ));

        let portable = DoubleDoubleRecursiveFftIr::build(&plan, Direction::Forward).unwrap();
        assert!(portable.rader_forced_upload_schedule.is_none());
        let scheduled =
            DoubleDoubleRecursiveFftIr::build_for_device(&plan, Direction::Forward, device)
                .unwrap();
        assert!(scheduled.stockham_upload_schedule.is_none());
        let upload = scheduled
            .rader_forced_upload_schedule
            .as_ref()
            .expect("thread-limited DD p17 axis should retain forced-Rader upload metadata");
        assert_eq!(upload.upload_count, 2);
        assert_eq!(upload.axis_split, vec![68, 75]);
        assert!(scheduled.two_upload_four_step_plan.is_none());
        assert!(scheduled.three_upload_four_step_plan.is_none());

        let DoubleDoubleRecursiveFftNodeIr::CooleyTukey(root) = &scheduled.root else {
            panic!("forced-Rader DD axis should have a Cooley-Tukey upload root");
        };
        assert_eq!((root.left_len, root.right_len), (68, 75));
        let DoubleDoubleRecursiveFftNodeIr::CooleyTukey(low) = &root.left else {
            panic!("68-point DD upload0 should compose 4 x p17");
        };
        let DoubleDoubleRecursiveFftNodeIr::Stockham(high) = &root.right else {
            panic!("75-point DD upload1 should remain Stockham");
        };
        let root_pack_name = root.pack_right.name.clone();
        let root_twiddle_name = root.twiddle_transpose.name.clone();
        let root_scatter_name = root.scatter_output.name.clone();
        let low_scatter_name = low.scatter_output.name.clone();
        assert_eq!((low.left_len, low.right_len), (4, 17));
        assert!(matches!(
            low.left,
            DoubleDoubleRecursiveFftNodeIr::Stockham(ref stockham) if stockham.sequence_len == 4
        ));
        assert!(matches!(
            low.right,
            DoubleDoubleRecursiveFftNodeIr::FftRader(ref rader) if rader.prime == 17
        ));
        assert_eq!(high.sequence_len, 75);
        let low_block = low
            .pack_right
            .axis_batch_block
            .expect("DD 68-point p17 FFT-Rader parent must consume Quad pass-local scoring");
        assert_eq!(low_block.threads_per_transform, 5);
        assert_eq!(low.twiddle_transpose.axis_batch_block, Some(low_block));
        assert_eq!(low.scatter_output.axis_batch_block, Some(low_block));
        scheduled.validate().unwrap();

        let input = sample(length, 1);
        let expected = execute_double_double_recursive_ir(&portable, &input).unwrap();
        let actual = execute_double_double_recursive_ir(&scheduled, &input).unwrap();
        assert!(max_error(&actual, &expected) <= 2.0e-11);

        let (mapped_high, mapping) = scheduled
            .forced_rader_two_upload_mapped_high_stockham()
            .unwrap()
            .expect("75-point high Stockham upload should own FourStepRight");
        assert_eq!(mapped_high.sequence_len, 75);
        assert_eq!(
            mapping,
            FourStepMapping {
                logical_len: length,
                left_len: 68,
                right_len: 75,
                outer_batch_count: 1,
            }
        );
        let mapped_low = scheduled
            .forced_rader_two_upload_mapped_low_component()
            .unwrap()
            .expect("68-point recursive low upload should own FourStepLeft");
        let DoubleDoubleRecursiveFftNodeIr::CooleyTukey(mapped_low) = mapped_low else {
            panic!("mapped 68-point low upload should remain Cooley-Tukey");
        };
        assert_eq!(mapped_low.logical_len, 68);
        assert_eq!(
            mapped_low.scatter_output.output_modifier,
            DoubleDoubleCooleyTukeyOutputModifier::FourStepLeft(mapping)
        );
        assert!(
            mapped_low
                .fused_small_fft_rader_stockham()
                .unwrap()
                .is_none(),
            "mapped N68 FourStepLeft child must not consume the standalone radix-4 fusion"
        );

        let program = crate::ProgramIr::double_double_recursive(&scheduled).unwrap();
        assert!(program.name.contains("mapped_rader_four_step"));
        assert!(
            program
                .passes
                .iter()
                .all(|pass| pass.name != root_pack_name)
        );
        assert!(
            program
                .passes
                .iter()
                .all(|pass| pass.name != root_twiddle_name)
        );
        assert!(
            program
                .passes
                .iter()
                .all(|pass| pass.name != root_scatter_name)
        );
        assert!(
            program
                .resources
                .iter()
                .all(|resource| resource.name != "double_double_rader_four_step_left_output")
        );
        assert!(
            program
                .passes
                .iter()
                .any(|pass| pass.name == low_scatter_name)
        );
        assert!(
            program.passes[0]
                .name
                .ends_with("forced_rader_two_upload_1")
        );

        let one_dim =
            crate::DoubleDoubleOneDimIr::build_for_device(&plan, Direction::Forward, device)
                .unwrap();
        let crate::DoubleDoubleOneDimIr::Recursive(one_dim) = one_dim else {
            panic!("device-aware composite Rader must route through recursive DD builder");
        };
        assert_eq!(
            one_dim
                .rader_forced_upload_schedule
                .as_ref()
                .expect("high-level DD route must retain forced-Rader upload metadata")
                .axis_split,
            vec![68, 75]
        );
        assert!(
            one_dim
                .forced_rader_two_upload_mapped_high_stockham()
                .unwrap()
                .is_some()
        );
        assert!(
            one_dim
                .forced_rader_two_upload_mapped_low_component()
                .unwrap()
                .is_some()
        );

        let shaders = crate::backend::vulkan::VulkanGlslBackend
            .lower_double_double_recursive(&scheduled)
            .unwrap();
        assert_eq!(shaders.len(), program.passes.len());
        assert!(shaders.iter().all(|shader| shader.compile_spirv().is_ok()));
        assert!(shaders.iter().all(|shader| shader.compile_spirv().is_ok()));
    }

    #[test]
    fn forced_rader_two_upload_maps_stockham_high_component() {
        let length = 17usize * 1024;
        let device = DeviceProfile {
            shared_memory_bytes: 48 * 1024,
            shared_memory_pow2_bytes: 32 * 1024,
            max_threads_per_block: 1024,
            max_workgroup_size: [1024, 1024, 64],
            supports_f64: true,
            ..DeviceProfile::generic(crate::Backend::Vulkan, crate::GpuVendor::Nvidia)
        };

        for precision in [Precision::DoubleDouble, Precision::DoubleDoubleF64Storage] {
            let plan =
                FftPlan::build(FftConfig::new(vec![length]).with_precision(precision)).unwrap();
            assert!(matches!(
                plan.axes[0].algorithm,
                AxisAlgorithm::Rader { .. }
            ));
            let scheduled =
                DoubleDoubleRecursiveFftIr::build_for_device(&plan, Direction::Forward, device)
                    .unwrap();
            let upload = scheduled
                .rader_forced_upload_schedule
                .as_ref()
                .expect("DD 17x1024 axis should retain forced-Rader upload metadata");
            assert_eq!(upload.upload_count, 2);
            assert_eq!(upload.axis_split, vec![136, 128]);
            assert!(scheduled.stockham_upload_schedule.is_none());
            assert!(scheduled.two_upload_four_step_plan.is_none());
            assert!(scheduled.three_upload_four_step_plan.is_none());

            let DoubleDoubleRecursiveFftNodeIr::CooleyTukey(root) = &scheduled.root else {
                panic!("DD 17x1024 forced upload should keep a Cooley-Tukey root");
            };
            assert_eq!((root.left_len, root.right_len), (136, 128));
            let DoubleDoubleRecursiveFftNodeIr::Stockham(high) = &root.right else {
                panic!("128-point high upload should remain Stockham");
            };
            assert_eq!(high.sequence_len, 128);
            assert_eq!(high.batch_count, 136);
            assert!(high.axis_batch_block.is_some());
            assert!(matches!(
                root.left,
                DoubleDoubleRecursiveFftNodeIr::CooleyTukey(_)
            ));
            assert!(
                scheduled
                    .forced_rader_two_upload_mapped_high_component()
                    .unwrap()
                    .is_none()
            );

            let (mapped_high, mapping) = scheduled
                .forced_rader_two_upload_mapped_high_stockham()
                .unwrap()
                .expect("128-point high Stockham upload should own FourStepRight");
            assert_eq!(mapped_high.sequence_len, 128);
            assert_eq!(
                mapping,
                FourStepMapping {
                    logical_len: length,
                    left_len: 136,
                    right_len: 128,
                    outer_batch_count: 1,
                }
            );
            let mapped_low = scheduled
                .forced_rader_two_upload_mapped_low_component()
                .unwrap()
                .expect("136-point recursive low upload should own FourStepLeft");
            let DoubleDoubleRecursiveFftNodeIr::CooleyTukey(mapped_low) = mapped_low else {
                panic!("mapped 136-point low upload should remain Cooley-Tukey");
            };
            assert_eq!(mapped_low.logical_len, 136);
            assert_eq!(
                mapped_low.scatter_output.output_storage,
                scheduled.external_storage
            );
            assert_eq!(
                mapped_low.scatter_output.output_modifier,
                DoubleDoubleCooleyTukeyOutputModifier::FourStepLeft(mapping)
            );
            let mapped_low_scatter_name = mapped_low.scatter_output.name.clone();

            let program = crate::ProgramIr::double_double_recursive(&scheduled).unwrap();
            assert!(program.name.contains("mapped_rader_four_step"));
            assert_eq!(
                program.resources[0].scalar,
                match precision {
                    Precision::DoubleDouble => crate::ScalarType::DoubleDouble,
                    Precision::DoubleDoubleF64Storage => crate::ScalarType::F64,
                    _ => unreachable!(),
                }
            );
            assert_eq!(program.resources[1].scalar, program.resources[0].scalar);
            assert!(
                program.passes[0]
                    .name
                    .ends_with("forced_rader_two_upload_1")
            );
            assert!(program.passes[0].bindings.iter().any(|binding| {
                binding.binding == 3
                    && binding.role == crate::BufferRole::Auxiliary
                    && program.resources[binding.resource.0].elements == length
            }));
            assert!(
                program
                    .passes
                    .iter()
                    .all(|pass| pass.name != root.pack_right.name)
            );
            assert!(
                program
                    .passes
                    .iter()
                    .all(|pass| pass.name != root.twiddle_transpose.name)
            );
            assert!(
                program
                    .passes
                    .iter()
                    .all(|pass| pass.name != root.scatter_output.name)
            );
            assert!(
                program
                    .resources
                    .iter()
                    .all(|resource| resource.name != "double_double_rader_four_step_left_output")
            );
            assert!(
                program
                    .passes
                    .iter()
                    .any(|pass| pass.name == mapped_low_scatter_name)
            );

            let shaders = crate::backend::vulkan::VulkanGlslBackend
                .lower_double_double_recursive(&scheduled)
                .unwrap();
            assert_eq!(shaders.len(), program.passes.len());
            let high_shader = shaders.first().expect("mapped high Stockham shader");
            assert_eq!(high_shader.sequence_len, 128);
            assert!(high_shader.glsl.contains("fused DD Four-step upload 1"));
            assert!(high_shader.glsl.contains(
                "source = outer_batch * VKFFT_DD_FOUR_STEP_N + n1 + VKFFT_DD_FOUR_STEP_A * i"
            ));
            assert!(high_shader.glsl.contains("vkfft_aux.data[twiddle]"));
            if precision == Precision::DoubleDoubleF64Storage {
                assert!(high_shader.glsl.contains("VkFftInput { dvec2 data[]; }"));
            }
            let low_scatter_shader = shaders.last().expect("mapped low scatter shader");
            assert!(
                low_scatter_shader
                    .glsl
                    .contains("VKFFT_DD_FOUR_STEP_RIGHT * component_index")
            );
            if precision == Precision::DoubleDoubleF64Storage {
                assert!(
                    low_scatter_shader
                        .glsl
                        .contains("VkFftOutput { dvec2 data[]; }")
                );
            }
            for shader in &shaders {
                assert_eq!(shader.compile_spirv().unwrap().words[0], 0x0723_0203);
            }
        }
    }

    #[test]
    fn forced_rader_two_upload_maps_monolithic_stockham_low_component() {
        let length = 17usize * 32;
        let device = DeviceProfile {
            shared_memory_bytes: 48 * 1024,
            shared_memory_pow2_bytes: 32 * 1024,
            max_threads_per_block: 16,
            max_workgroup_size: [16, 16, 64],
            supports_f64: true,
            ..DeviceProfile::generic(crate::Backend::Vulkan, crate::GpuVendor::Nvidia)
        };

        for precision in [Precision::DoubleDouble, Precision::DoubleDoubleF64Storage] {
            let plan =
                FftPlan::build(FftConfig::new(vec![length]).with_precision(precision)).unwrap();
            let scheduled =
                DoubleDoubleRecursiveFftIr::build_for_device(&plan, Direction::Forward, device)
                    .unwrap();
            let upload = scheduled
                .rader_forced_upload_schedule
                .as_ref()
                .expect("DD N544 should force a two-upload Rader split under a 16-thread cap");
            assert_eq!(upload.upload_count, 2);
            assert_eq!(upload.axis_split, vec![32, 17]);

            let DoubleDoubleRecursiveFftNodeIr::CooleyTukey(root) = &scheduled.root else {
                panic!("DD N544 forced upload should keep a Cooley-Tukey root");
            };
            assert_eq!((root.left_len, root.right_len), (32, 17));
            let DoubleDoubleRecursiveFftNodeIr::Stockham(low) = &root.left else {
                panic!("N544 low upload should be a monolithic N32 Stockham leaf");
            };
            assert_eq!(low.sequence_len, 32);
            assert_eq!(low.batch_count, 17);
            assert!(low.axis_batch_block.is_some());
            assert!(
                scheduled
                    .forced_rader_two_upload_mapped_high_component()
                    .unwrap()
                    .is_some()
            );
            let (mapped_low, mapping) = scheduled
                .forced_rader_two_upload_mapped_low_stockham()
                .unwrap()
                .expect("N32 low Stockham should own the FourStepLeft caller boundary");
            assert_eq!(mapped_low.sequence_len, 32);
            assert_eq!(
                mapping,
                FourStepMapping {
                    logical_len: length,
                    left_len: 32,
                    right_len: 17,
                    outer_batch_count: 1,
                }
            );

            let program = crate::ProgramIr::double_double_recursive(&scheduled).unwrap();
            assert!(program.name.contains("mapped_rader_four_step"));
            assert!(
                program
                    .passes
                    .iter()
                    .all(|pass| pass.name != root.scatter_output.name)
            );
            assert!(
                program
                    .resources
                    .iter()
                    .all(|resource| resource.name != "double_double_rader_four_step_left_output")
            );
            let final_pass = program.passes.last().expect("mapped N32 low Stockham pass");
            assert_eq!(final_pass.name, format!("{}_four_step_left", low.name));
            assert!(final_pass.bindings.iter().any(|binding| {
                binding.binding == 1 && binding.resource == crate::ProgramResourceId(1)
            }));

            let shaders = crate::backend::vulkan::VulkanGlslBackend
                .lower_double_double_recursive(&scheduled)
                .unwrap();
            assert_eq!(shaders.len(), program.passes.len());
            let low_shader = shaders
                .last()
                .expect("mapped monolithic low Stockham shader");
            assert_eq!(low_shader.sequence_len, 32);
            assert!(
                low_shader
                    .glsl
                    .contains("forced-Rader FourStepLeft mapped monolithic finalize")
            );
            assert!(
                low_shader
                    .glsl
                    .contains("uint destination = outer_batch * 544u + k2 + 17u * i;")
            );
            assert!(low_shader.glsl.contains("VkFftInput { dvec4 data[]; }"));
            if precision == Precision::DoubleDoubleF64Storage {
                assert!(low_shader.glsl.contains("VkFftOutput { dvec2 data[]; }"));
            }
            for shader in &shaders {
                assert_eq!(shader.compile_spirv().unwrap().words[0], 0x0723_0203);
            }
        }
    }

    #[test]
    fn forced_rader_three_upload_materializes_narrow_singleton_axis_block() {
        let length = 33_728usize;
        let device = DeviceProfile {
            shared_memory_bytes: 2 * 1024,
            shared_memory_pow2_bytes: 2 * 1024,
            max_threads_per_block: 64,
            max_workgroup_size: [64, 64, 64],
            supports_f64: true,
            ..DeviceProfile::generic(crate::Backend::Vulkan, crate::GpuVendor::Nvidia)
        };
        let plan = FftPlan::build_for_device(
            FftConfig::new(vec![length]).with_precision(Precision::DoubleDouble),
            device,
        )
        .unwrap();
        let ir = DoubleDoubleRecursiveFftIr::build_for_device(&plan, Direction::Forward, device)
            .unwrap();
        let schedule = ir
            .rader_forced_upload_schedule
            .as_ref()
            .expect("narrow DD N33728 must retain forced-Rader metadata");
        assert_eq!(schedule.axis_split, vec![32, 34, 31]);

        let components = ir
            .forced_rader_three_upload_mapped_components()
            .unwrap()
            .expect("narrow DD N33728 must materialize all three upload components");
        assert_eq!(components.len(), 3);

        let DoubleDoubleForcedRaderThreeUploadComponentIr::Recursive {
            upload_id: 2,
            ir: DoubleDoubleRecursiveFftNodeIr::FftRader(high),
        } = &components[0]
        else {
            panic!("upload2 must remain the p31 FFT-Rader component");
        };
        let high_block = high
            .caller_axis_batch_block
            .expect("upload2 p31 must retain its physical caller block");
        assert_eq!(high.prime, 31);
        assert_eq!(high_block.threads_per_transform, 7);
        assert_eq!(high_block.grouped_batch, 2);
        assert_eq!([high_block.local_size_x, high_block.local_size_y], [2, 7]);
        assert!(high_block.transforms_on_x);
        assert!(!high_block.axis_swapped);

        let DoubleDoubleForcedRaderThreeUploadComponentIr::Recursive {
            upload_id: 1,
            ir: DoubleDoubleRecursiveFftNodeIr::CooleyTukey(middle),
        } = &components[1]
        else {
            panic!("upload1 must remain the 34-point Cooley component");
        };
        let middle_block = middle
            .pack_right
            .axis_batch_block
            .expect("upload1 must publish the upstream singleton block");
        assert_eq!((middle.left_len, middle.right_len), (2, 17));
        assert_eq!(middle_block.threads_per_transform, 3);
        assert_eq!(middle_block.grouped_batch, 1);
        assert_eq!(
            [middle_block.local_size_x, middle_block.local_size_y],
            [1, 3]
        );
        assert!(middle_block.transforms_on_x);
        assert!(!middle_block.axis_swapped);
        assert_eq!(
            middle.twiddle_transpose.axis_batch_block,
            Some(middle_block)
        );
        assert_eq!(middle.scatter_output.axis_batch_block, Some(middle_block));

        let DoubleDoubleForcedRaderThreeUploadComponentIr::Stockham {
            upload_id: 0,
            ir: low,
        } = &components[2]
        else {
            panic!("upload0 must remain the 32-point Stockham component");
        };
        let low_block = low
            .axis_batch_block
            .expect("upload0 Stockham must retain its physical block");
        assert_eq!(low.sequence_len, 32);
        assert_eq!(low_block.threads_per_transform, 4);
        assert_eq!(low_block.grouped_batch, 2);
        assert_eq!([low_block.local_size_x, low_block.local_size_y], [4, 2]);
        assert!(!low_block.transforms_on_x);
        assert!(!low_block.axis_swapped);
        ir.validate().unwrap();
    }

    #[test]
    fn forced_rader_three_upload_maps_middle_cooley_component() {
        let length = 17usize * 65_536;
        let device = DeviceProfile {
            shared_memory_bytes: 32 * 1024,
            shared_memory_pow2_bytes: 32 * 1024,
            max_threads_per_block: 1024,
            max_workgroup_size: [1024, 1024, 64],
            supports_f64: true,
            ..DeviceProfile::generic(crate::Backend::Vulkan, crate::GpuVendor::Nvidia)
        };
        let plan =
            FftPlan::build(FftConfig::new(vec![length]).with_precision(Precision::DoubleDouble))
                .unwrap();
        let scheduled =
            DoubleDoubleRecursiveFftIr::build_for_device(&plan, Direction::Forward, device)
                .unwrap();
        let upload = scheduled
            .rader_forced_upload_schedule
            .as_ref()
            .expect("DD promoted p17 axis should retain forced-Rader upload metadata");
        assert_eq!(upload.upload_count, 3);
        assert_eq!(upload.axis_split, vec![128, 68, 128]);
        let higher_axis = scheduled
            .clone()
            .with_other_axis_physical_blocks(2, None, None, device)
            .unwrap();
        let higher_components = higher_axis
            .forced_rader_three_upload_mapped_components()
            .unwrap()
            .expect("higher-axis mixed DD three-upload should expose mapped components");
        assert!(matches!(
            &higher_components[0],
            DoubleDoubleForcedRaderThreeUploadComponentIr::Stockham { upload_id: 2, ir }
                if ir.axis_batch_block.is_some_and(|block| block.transforms_on_x)
        ));
        assert!(matches!(
            &higher_components[1],
            DoubleDoubleForcedRaderThreeUploadComponentIr::Recursive {
                upload_id: 1,
                ir: DoubleDoubleRecursiveFftNodeIr::CooleyTukey(cooley),
            } if cooley.pack_right.axis_batch_block.is_some_and(|block| block.transforms_on_x)
                && cooley.twiddle_transpose.axis_batch_block
                    == cooley.pack_right.axis_batch_block
                && cooley.scatter_output.axis_batch_block
                    == cooley.pack_right.axis_batch_block
        ));
        assert!(matches!(
            &higher_components[2],
            DoubleDoubleForcedRaderThreeUploadComponentIr::Stockham { upload_id: 0, ir }
                if ir.axis_batch_block.is_some_and(|block| block.transforms_on_x)
        ));
        higher_axis.validate().unwrap();
        let higher_program = crate::ProgramIr::double_double_recursive(&higher_axis).unwrap();
        higher_program.validate().unwrap();
        let higher_shaders = crate::backend::vulkan::VulkanGlslBackend
            .lower_double_double_recursive(&higher_axis)
            .unwrap();
        assert_eq!(higher_shaders.len(), higher_program.passes.len());
        for shader in &higher_shaders {
            assert_eq!(shader.compile_spirv().unwrap().words[0], 0x0723_0203);
        }
        let DoubleDoubleRecursiveFftNodeIr::CooleyTukey(root) = &scheduled.root else {
            panic!("DD promoted three-upload root should remain Cooley-Tukey");
        };
        let DoubleDoubleRecursiveFftNodeIr::CooleyTukey(upper) = &root.right else {
            panic!("DD promoted three-upload upper branch should remain Cooley-Tukey");
        };
        let skipped_parent_passes = [
            root.pack_right.name.clone(),
            root.twiddle_transpose.name.clone(),
            root.scatter_output.name.clone(),
            upper.pack_right.name.clone(),
            upper.twiddle_transpose.name.clone(),
            upper.scatter_output.name.clone(),
        ];
        let components = scheduled
            .forced_rader_three_upload_mapped_components()
            .unwrap()
            .expect("mixed DD three-upload should expose mapped components");
        assert_eq!(components.len(), 3);
        assert_eq!(
            components
                .iter()
                .map(DoubleDoubleForcedRaderThreeUploadComponentIr::upload_id)
                .collect::<Vec<_>>(),
            vec![2, 1, 0]
        );
        assert_eq!(
            components
                .iter()
                .map(DoubleDoubleForcedRaderThreeUploadComponentIr::logical_len)
                .collect::<Vec<_>>(),
            vec![128, 68, 128]
        );
        assert!(matches!(
            &components[0],
            DoubleDoubleForcedRaderThreeUploadComponentIr::Stockham { upload_id: 2, ir }
                if ir.sequence_len == 128 && ir.axis_batch_block.is_some()
        ));
        let DoubleDoubleForcedRaderThreeUploadComponentIr::Recursive {
            upload_id: 1,
            ir: DoubleDoubleRecursiveFftNodeIr::CooleyTukey(middle),
        } = &components[1]
        else {
            panic!("middle 68-point upload should remain mapped Cooley-Tukey");
        };
        let mapping = ThreeUploadFourStepMapping {
            logical_len: length,
            axis_split: [128, 68, 128],
            outer_batch_count: 1,
        };
        assert_eq!(
            middle.scatter_output.output_modifier,
            DoubleDoubleCooleyTukeyOutputModifier::FourStepThreeUpload1(mapping)
        );
        assert!(matches!(
            &components[2],
            DoubleDoubleForcedRaderThreeUploadComponentIr::Stockham { upload_id: 0, ir }
                if ir.sequence_len == 128 && ir.axis_batch_block.is_some()
        ));
        let middle_scatter_name = middle.scatter_output.name.clone();
        let program = crate::ProgramIr::double_double_recursive(&scheduled).unwrap();
        assert!(program.name.contains("forced_rader_three_upload"));
        assert!(
            skipped_parent_passes
                .iter()
                .all(|name| { program.passes.iter().all(|pass| &pass.name != name) })
        );
        let high_pass = program
            .passes
            .iter()
            .find(|pass| pass.name.ends_with("forced_rader_three_upload_2"))
            .expect("mapped high Stockham upload pass");
        assert!(high_pass.bindings.iter().any(|binding| {
            binding.binding == 3
                && binding.role == crate::BufferRole::Auxiliary
                && program.resources[binding.resource.0].elements == length
        }));
        let middle_pass = program
            .passes
            .iter()
            .find(|pass| pass.name == middle_scatter_name)
            .expect("mapped middle Cooley scatter pass");
        assert!(middle_pass.bindings.iter().any(|binding| {
            binding.binding == 2
                && binding.role == crate::BufferRole::TwiddleLookupTable
                && program.resources[binding.resource.0].elements == 128 * 68
        }));
        let low_pass = program
            .passes
            .iter()
            .find(|pass| pass.name.ends_with("forced_rader_three_upload_0"))
            .expect("mapped low Stockham upload pass");
        assert!(low_pass.bindings.iter().all(|binding| binding.binding != 3));

        let shaders = crate::backend::vulkan::VulkanGlslBackend
            .lower_double_double_recursive(&scheduled)
            .unwrap();
        assert_eq!(shaders.len(), program.passes.len());
        assert!(shaders.iter().all(|shader| shader.compile_spirv().is_ok()));
        let middle_shader = shaders
            .iter()
            .find(|shader| shader.glsl.contains(&middle_scatter_name))
            .expect("mapped middle Cooley scatter shader");
        assert!(middle_shader.glsl.contains("VKFFT_DD_FOUR_STEP_AB"));
        assert!(middle_shader.glsl.contains("vkfft_four_step_group"));
    }

    #[test]
    fn fft_rader_three_upload_middle_mapping_uses_ab_phase() {
        let mapping = ThreeUploadFourStepMapping {
            logical_len: 2 * 31 * 31,
            axis_split: [2, 31, 31],
            outer_batch_count: 1,
        };
        let plan = FftPlan::build(
            FftConfig::new(vec![31])
                .with_batch_count(62)
                .with_precision(Precision::DoubleDouble),
        )
        .unwrap();
        let mapped = crate::DoubleDoubleFftRaderIr::build(&plan, Direction::Forward)
            .unwrap()
            .with_four_step_io_mapping(
                StockhamIoMapping::FourStepThreeUpload1(mapping),
                PrecisionStorage::DoubleDouble,
                PrecisionStorage::DoubleDouble,
            )
            .unwrap();
        assert_eq!(mapped.prime, 31);
        assert_eq!(mapped.batch_count, 62);
        assert_eq!(
            mapped.io_mapping,
            StockhamIoMapping::FourStepThreeUpload1(mapping)
        );

        let program = crate::ProgramIr::double_double_fft_rader(&mapped).unwrap();
        let scatter = program
            .passes
            .iter()
            .find(|pass| pass.name.ends_with("_scatter"))
            .expect("mapped p31 FFT-Rader middle scatter pass");
        assert!(scatter.bindings.iter().any(|binding| {
            binding.binding == 3
                && binding.role == crate::BufferRole::TwiddleLookupTable
                && program.resources[binding.resource.0].elements == 2 * 31
        }));
        let shaders = crate::backend::vulkan::VulkanGlslBackend
            .lower_double_double_fft_rader(&mapped)
            .unwrap();
        assert_eq!(shaders.len(), program.passes.len());
        assert!(shaders.iter().all(|shader| shader.compile_spirv().is_ok()));
        let scatter_shader = shaders
            .iter()
            .find(|shader| shader.glsl.contains("DD FFT-Rader natural-order scatter"))
            .expect("mapped p31 FFT-Rader middle scatter shader");
        assert!(scatter_shader.glsl.contains("VKFFT_DD_FOUR_STEP_PHASE_N"));
        assert!(scatter_shader.glsl.contains("vkfft_four_step_phase_lane"));
        assert!(scatter_shader.glsl.contains("vkfft_four_step_roots"));
    }

    #[test]
    fn forced_rader_three_upload_maps_fft_rader_high_component() {
        let length = 2usize * 31 * 31;
        let device = DeviceProfile {
            shared_memory_bytes: 1024,
            shared_memory_pow2_bytes: 1024,
            max_threads_per_block: 32,
            max_workgroup_size: [1024, 1024, 64],
            supports_f64: true,
            ..DeviceProfile::generic(crate::Backend::Vulkan, crate::GpuVendor::Nvidia)
        };
        for precision in [Precision::DoubleDouble, Precision::DoubleDoubleF64Storage] {
            let plan =
                FftPlan::build(FftConfig::new(vec![length]).with_precision(precision)).unwrap();
            let scheduled =
                DoubleDoubleRecursiveFftIr::build_for_device(&plan, Direction::Forward, device)
                    .unwrap();
            let upload = scheduled
                .rader_forced_upload_schedule
                .as_ref()
                .expect("DD p31 composite should require three forced-Rader uploads");
            assert_eq!(upload.upload_count, 3);
            assert_eq!(upload.axis_split, vec![2, 31, 31]);
            let components = scheduled
                .forced_rader_three_upload_mapped_components()
                .unwrap()
                .expect("DD p31 high upload should materialize mapped three-upload components");
            let DoubleDoubleForcedRaderThreeUploadComponentIr::Recursive {
                upload_id: 2,
                ir: DoubleDoubleRecursiveFftNodeIr::FftRader(high),
            } = &components[0]
            else {
                panic!("31-point high upload should stay FFT Rader");
            };
            let mapping = ThreeUploadFourStepMapping {
                logical_len: length,
                axis_split: [2, 31, 31],
                outer_batch_count: 1,
            };
            assert_eq!(
                high.io_mapping,
                StockhamIoMapping::FourStepThreeUpload2(mapping)
            );
            assert_eq!(high.input_storage, scheduled.external_storage);
            assert_eq!(high.output_storage, PrecisionStorage::DoubleDouble);
            assert!(matches!(
                &components[1],
                DoubleDoubleForcedRaderThreeUploadComponentIr::Recursive {
                    upload_id: 1,
                    ir: DoubleDoubleRecursiveFftNodeIr::FftRader(middle),
                } if middle.prime == 31
                    && matches!(middle.io_mapping, StockhamIoMapping::FourStepThreeUpload1(_))
            ));
            assert!(matches!(
                &components[2],
                DoubleDoubleForcedRaderThreeUploadComponentIr::Stockham {
                    upload_id: 0,
                    ir: low,
                } if low.sequence_len == 2
            ));

            let program = crate::ProgramIr::double_double_recursive(&scheduled).unwrap();
            assert_eq!(
                program.input_resource().unwrap().scalar,
                match precision {
                    Precision::DoubleDouble => crate::ScalarType::DoubleDouble,
                    Precision::DoubleDoubleF64Storage => crate::ScalarType::F64,
                    _ => unreachable!(),
                }
            );
            let scatter = program
                .passes
                .iter()
                .find(|pass| pass.name.contains("rader_fft_31") && pass.name.ends_with("_scatter"))
                .expect("mapped p31 FFT-Rader scatter pass");
            assert!(scatter.bindings.iter().any(|binding| {
                binding.binding == 3
                    && binding.role == crate::BufferRole::TwiddleLookupTable
                    && program.resources[binding.resource.0].elements == length
            }));
            let shaders = crate::backend::vulkan::VulkanGlslBackend
                .lower_double_double_recursive(&scheduled)
                .unwrap();
            assert_eq!(shaders.len(), program.passes.len());
            assert!(shaders.iter().all(|shader| shader.compile_spirv().is_ok()));
            let scatter_shader = shaders
                .iter()
                .find(|shader| shader.glsl.contains("DD FFT-Rader natural-order scatter"))
                .expect("mapped p31 FFT-Rader scatter shader");
            assert!(scatter_shader.glsl.contains("vkfft_four_step_phase_lane"));
            assert!(scatter_shader.glsl.contains("VKFFT_DD_FOUR_STEP_PHASE_N"));
            assert!(scatter_shader.glsl.contains("vkfft_four_step_roots"));
        }
    }

    #[test]
    fn forced_rader_three_upload_n3196_maps_direct_rader_middle_component() {
        let length = 4usize * 17 * 47;
        let device = DeviceProfile {
            shared_memory_bytes: 3 * 1024,
            shared_memory_pow2_bytes: 3 * 1024,
            max_threads_per_block: 128,
            max_workgroup_size: [1024, 1024, 64],
            supports_f64: true,
            ..DeviceProfile::generic(crate::Backend::Vulkan, crate::GpuVendor::Nvidia)
        };
        for precision in [Precision::DoubleDouble, Precision::DoubleDoubleF64Storage] {
            let plan =
                FftPlan::build(FftConfig::new(vec![length]).with_precision(precision)).unwrap();
            let scheduled =
                DoubleDoubleRecursiveFftIr::build_for_device(&plan, Direction::Forward, device)
                    .unwrap();
            let upload = scheduled
                .rader_forced_upload_schedule
                .as_ref()
                .expect("DD p17/p47 axis should require three forced-Rader uploads");
            assert_eq!(upload.upload_count, 3);
            assert_eq!(upload.axis_split, vec![4, 47, 17]);
            let components = scheduled
                .forced_rader_three_upload_mapped_components()
                .unwrap()
                .expect("DD p17/p47 axis should materialize mapped components");
            let mapping = ThreeUploadFourStepMapping {
                logical_len: length,
                axis_split: [4, 47, 17],
                outer_batch_count: 1,
            };
            let DoubleDoubleForcedRaderThreeUploadComponentIr::Recursive {
                upload_id: 2,
                ir: DoubleDoubleRecursiveFftNodeIr::FftRader(high),
            } = &components[0]
            else {
                panic!("high 17-point upload should stay FFT Rader");
            };
            assert_eq!(high.prime, 17);
            assert_eq!(
                high.io_mapping,
                StockhamIoMapping::FourStepThreeUpload2(mapping)
            );
            assert_eq!(high.input_storage, scheduled.external_storage);
            assert_eq!(high.output_storage, PrecisionStorage::DoubleDouble);
            let high_block = high
                .caller_axis_batch_block
                .expect("DD p17 upload2 should receive a pass-local caller block");
            assert_eq!(high_block.threads_per_transform, 2);
            assert_eq!(high_block.grouped_batch, 4);
            assert!(high_block.transforms_on_x);
            assert!(!high_block.axis_swapped);
            assert_eq!([high_block.local_size_x, high_block.local_size_y], [4, 2]);
            let DoubleDoubleForcedRaderThreeUploadComponentIr::Recursive {
                upload_id: 1,
                ir: DoubleDoubleRecursiveFftNodeIr::DirectRader(middle),
            } = &components[1]
            else {
                panic!("middle 47-point upload should stay direct Rader");
            };
            assert_eq!(
                middle.io_mapping,
                StockhamIoMapping::FourStepThreeUpload1(mapping)
            );
            assert_eq!(middle.input_storage, PrecisionStorage::DoubleDouble);
            assert_eq!(middle.output_storage, PrecisionStorage::DoubleDouble);
            let middle_block = middle
                .axis_batch_block
                .expect("DD p47 upload1 should receive a pass-local direct-Rader block");
            assert_eq!(middle_block.threads_per_transform, 24);
            assert_eq!(middle_block.grouped_batch, 1);
            assert!(middle_block.transforms_on_x);
            assert!(!middle_block.axis_swapped);
            assert_eq!(
                [middle_block.local_size_x, middle_block.local_size_y],
                [1, 24]
            );

            let program = crate::ProgramIr::double_double_recursive(&scheduled).unwrap();
            let direct_passes = program
                .passes
                .iter()
                .filter(|pass| pass.name.contains("rader_direct_47"))
                .collect::<Vec<_>>();
            assert_eq!(direct_passes.len(), 1);
            assert!(direct_passes[0].bindings.iter().any(|binding| {
                binding.binding == 3
                    && binding.role == crate::BufferRole::TwiddleLookupTable
                    && program.resources[binding.resource.0].elements == 4 * 47
            }));
            let shaders = crate::backend::vulkan::VulkanGlslBackend
                .lower_double_double_recursive(&scheduled)
                .unwrap();
            assert_eq!(shaders.len(), program.passes.len());
            assert!(shaders.iter().all(|shader| shader.compile_spirv().is_ok()));
            let direct_shaders = shaders
                .iter()
                .filter(|shader| shader.glsl.contains("DoubleDoubleDirectRaderIr"))
                .collect::<Vec<_>>();
            assert_eq!(direct_shaders.len(), 1);
            assert!(
                direct_shaders[0]
                    .glsl
                    .contains("vkfft_four_step_phase_lane")
            );
            assert!(
                direct_shaders[0]
                    .glsl
                    .contains("VKFFT_DD_FOUR_STEP_PHASE_N")
            );
            assert!(direct_shaders[0].glsl.contains("vkfft_four_step_roots"));
        }
    }

    #[test]
    fn forced_rader_three_upload_maps_direct_rader_middle_component() {
        let length = 17usize * 47 * 128;
        let device = DeviceProfile {
            shared_memory_bytes: 8 * 1024,
            shared_memory_pow2_bytes: 8 * 1024,
            max_threads_per_block: 1024,
            max_workgroup_size: [1024, 1024, 64],
            supports_f64: true,
            ..DeviceProfile::generic(crate::Backend::Vulkan, crate::GpuVendor::Nvidia)
        };
        for precision in [Precision::DoubleDouble, Precision::DoubleDoubleF64Storage] {
            let plan =
                FftPlan::build(FftConfig::new(vec![length]).with_precision(precision)).unwrap();
            let scheduled =
                DoubleDoubleRecursiveFftIr::build_for_device(&plan, Direction::Forward, device)
                    .unwrap();
            let upload = scheduled
                .rader_forced_upload_schedule
                .as_ref()
                .expect("DD p47 composite should require three forced-Rader uploads");
            assert_eq!(upload.upload_count, 3);
            assert_eq!(upload.axis_split, vec![64, 47, 34]);
            let components = scheduled
                .forced_rader_three_upload_mapped_components()
                .unwrap()
                .expect("DD p47 middle upload should materialize mapped three-upload components");
            let DoubleDoubleForcedRaderThreeUploadComponentIr::Recursive {
                upload_id: 1,
                ir: DoubleDoubleRecursiveFftNodeIr::DirectRader(middle),
            } = &components[1]
            else {
                panic!("47-point middle upload should stay direct Rader");
            };
            let mapping = ThreeUploadFourStepMapping {
                logical_len: length,
                axis_split: [64, 47, 34],
                outer_batch_count: 1,
            };
            assert_eq!(
                middle.io_mapping,
                StockhamIoMapping::FourStepThreeUpload1(mapping)
            );
            assert_eq!(middle.input_storage, PrecisionStorage::DoubleDouble);
            assert_eq!(middle.output_storage, PrecisionStorage::DoubleDouble);
            let middle_block = middle
                .axis_batch_block
                .expect("DD p47 upload1 should receive a pass-local direct-Rader block");
            assert_eq!(middle_block.threads_per_transform, 24);
            assert_eq!(middle_block.grouped_batch, 4);
            assert!(middle_block.transforms_on_x);
            assert!(!middle_block.axis_swapped);
            assert_eq!(
                [middle_block.local_size_x, middle_block.local_size_y],
                [4, 24]
            );

            let program = crate::ProgramIr::double_double_direct_rader(middle).unwrap();
            let direct = program
                .passes
                .iter()
                .find(|pass| pass.name.contains("rader_direct_47"))
                .expect("mapped p47 direct-Rader middle pass");
            assert!(direct.bindings.iter().any(|binding| {
                binding.binding == 3
                    && binding.role == crate::BufferRole::TwiddleLookupTable
                    && program.resources[binding.resource.0].elements == 64 * 47
            }));
            let direct_shader = crate::backend::vulkan::VulkanGlslBackend
                .lower_double_double_direct_rader(middle)
                .unwrap();
            assert_eq!(direct_shader.compile_spirv().unwrap().words[0], 0x0723_0203);
            assert!(direct_shader.glsl.contains("vkfft_four_step_phase_lane"));
            assert!(direct_shader.glsl.contains("VKFFT_DD_FOUR_STEP_PHASE_N"));
            assert!(direct_shader.glsl.contains("vkfft_four_step_roots"));
        }
    }

    #[test]
    fn forced_rader_two_upload_maps_direct_rader_high_component() {
        let length = 11usize * 17 * 47;
        let device = DeviceProfile {
            shared_memory_bytes: 8 * 1024,
            shared_memory_pow2_bytes: 8 * 1024,
            max_threads_per_block: 1024,
            max_workgroup_size: [1024, 1024, 64],
            supports_f64: true,
            ..DeviceProfile::generic(crate::Backend::Vulkan, crate::GpuVendor::Nvidia)
        };
        let plan =
            FftPlan::build(FftConfig::new(vec![length]).with_precision(Precision::DoubleDouble))
                .unwrap();
        let scheduled =
            DoubleDoubleRecursiveFftIr::build_for_device(&plan, Direction::Forward, device)
                .unwrap();
        let upload = scheduled
            .rader_forced_upload_schedule
            .as_ref()
            .expect("DD p47 composite should retain forced-Rader upload metadata");
        assert_eq!(upload.upload_count, 2);
        assert_eq!(upload.axis_split, vec![187, 47]);
        let DoubleDoubleRecursiveFftNodeIr::CooleyTukey(root) = &scheduled.root else {
            panic!("DD p47 forced upload should keep a Cooley-Tukey root");
        };
        assert_eq!(root.left.logical_len(), 187);
        assert!(matches!(
            root.left,
            DoubleDoubleRecursiveFftNodeIr::CooleyTukey(_)
        ));
        assert!(matches!(
            root.right,
            DoubleDoubleRecursiveFftNodeIr::DirectRader(ref direct) if direct.prime == 47
        ));
        let DoubleDoubleRecursiveFftNodeIr::DirectRader(high) = &root.right else {
            unreachable!("asserted direct-Rader high leaf");
        };
        let high_block = high
            .axis_batch_block
            .expect("DD p47 two-upload high leaf should receive automatic caller grouping");
        assert_eq!(high_block.threads_per_transform, 24);
        assert_eq!(high_block.grouped_batch, 4);
        assert!(high_block.transforms_on_x);
        assert!(!high_block.axis_swapped);
        assert_eq!([high_block.local_size_x, high_block.local_size_y], [4, 24]);
        let mapped_high = scheduled
            .forced_rader_two_upload_mapped_high_component()
            .unwrap()
            .expect("direct-Rader high upload should own the Four-step boundary");
        let DoubleDoubleRecursiveFftNodeIr::DirectRader(mapped_high) = mapped_high else {
            panic!("mapped high upload should remain direct Rader");
        };
        let mapping = FourStepMapping {
            logical_len: length,
            left_len: 187,
            right_len: 47,
            outer_batch_count: 1,
        };
        assert_eq!(
            mapped_high.io_mapping,
            StockhamIoMapping::FourStepRight(mapping)
        );
        assert_eq!(mapped_high.input_storage, PrecisionStorage::DoubleDouble);
        assert_eq!(mapped_high.output_storage, PrecisionStorage::DoubleDouble);
        assert!(
            scheduled
                .forced_rader_two_upload_mapped_low_stockham()
                .unwrap()
                .is_none()
        );
        let mapped_low = scheduled
            .forced_rader_two_upload_mapped_low_component()
            .unwrap()
            .expect("recursive 187-point low upload should own final Four-step scatter");
        let DoubleDoubleRecursiveFftNodeIr::CooleyTukey(mapped_low) = mapped_low else {
            panic!("mapped low upload should remain Cooley-Tukey");
        };
        assert_eq!(mapped_low.logical_len, 187);
        assert_eq!(
            mapped_low.scatter_output.output_storage,
            PrecisionStorage::DoubleDouble
        );
        assert_eq!(
            mapped_low.scatter_output.output_modifier,
            DoubleDoubleCooleyTukeyOutputModifier::FourStepLeft(mapping)
        );
        let mapped_low_scatter_name = mapped_low.scatter_output.name.clone();

        let program = crate::ProgramIr::double_double_recursive(&scheduled).unwrap();
        assert!(program.name.contains("mapped_rader_four_step"));
        assert!(
            program
                .passes
                .iter()
                .all(|pass| pass.name != root.pack_right.name)
        );
        assert!(
            program
                .passes
                .iter()
                .all(|pass| pass.name != root.twiddle_transpose.name)
        );
        assert!(
            program
                .passes
                .iter()
                .all(|pass| pass.name != root.scatter_output.name)
        );
        assert!(
            program
                .resources
                .iter()
                .all(|resource| resource.name != "double_double_rader_four_step_left_output")
        );
        assert!(
            program
                .passes
                .iter()
                .any(|pass| pass.name == mapped_low_scatter_name)
        );
        let direct_pass = program
            .passes
            .iter()
            .find(|pass| pass.name.contains("rader_direct_47"))
            .expect("mapped direct-Rader pass");
        assert!(direct_pass.bindings.iter().any(|binding| {
            binding.binding == 3
                && binding.role == crate::BufferRole::TwiddleLookupTable
                && program.resources[binding.resource.0].elements == length
        }));
        let shaders = crate::backend::vulkan::VulkanGlslBackend
            .lower_double_double_recursive(&scheduled)
            .unwrap();
        assert_eq!(shaders.len(), program.passes.len());
        assert!(shaders.iter().all(|shader| shader.compile_spirv().is_ok()));
        let direct_shader = shaders
            .iter()
            .find(|shader| shader.glsl.contains("DoubleDoubleDirectRaderIr"))
            .expect("mapped direct-Rader shader");
        assert!(direct_shader.glsl.contains("vkfft_four_step_roots"));
        assert!(direct_shader.glsl.contains("vkfft_output_stride"));
        let low_scatter_shader = shaders
            .iter()
            .find(|shader| shader.glsl.contains("vkfft_four_step_k2"))
            .expect("mapped recursive low scatter shader");
        assert!(
            low_scatter_shader
                .glsl
                .contains("VKFFT_DD_FOUR_STEP_RIGHT * component_index")
        );
    }

    #[test]
    fn forced_rader_two_upload_maps_fft_rader_high_component() {
        let length = 11usize * 17 * 31;
        let device = DeviceProfile {
            shared_memory_bytes: 24 * 1024,
            shared_memory_pow2_bytes: 24 * 1024,
            max_threads_per_block: 128,
            max_workgroup_size: [1024, 1024, 64],
            supports_f64: true,
            ..DeviceProfile::generic(crate::Backend::Vulkan, crate::GpuVendor::Nvidia)
        };
        let plan =
            FftPlan::build(FftConfig::new(vec![length]).with_precision(Precision::DoubleDouble))
                .unwrap();
        let scheduled =
            DoubleDoubleRecursiveFftIr::build_for_device(&plan, Direction::Forward, device)
                .unwrap();
        let upload = scheduled
            .rader_forced_upload_schedule
            .as_ref()
            .expect("DD p31 composite should retain forced-Rader upload metadata");
        assert_eq!(upload.upload_count, 2);
        assert_eq!(upload.axis_split, vec![187, 31]);
        let DoubleDoubleRecursiveFftNodeIr::CooleyTukey(root) = &scheduled.root else {
            panic!("DD p31 forced upload should keep a Cooley-Tukey root");
        };
        assert_eq!(root.left.logical_len(), 187);
        assert!(matches!(
            root.right,
            DoubleDoubleRecursiveFftNodeIr::FftRader(ref fft) if fft.prime == 31
        ));
        let mapped_high = scheduled
            .forced_rader_two_upload_mapped_high_component()
            .unwrap()
            .expect("FFT-Rader high upload should own the Four-step boundary");
        let DoubleDoubleRecursiveFftNodeIr::FftRader(mapped_high) = mapped_high else {
            panic!("mapped high upload should remain FFT Rader");
        };
        let mapping = FourStepMapping {
            logical_len: length,
            left_len: 187,
            right_len: 31,
            outer_batch_count: 1,
        };
        assert_eq!(
            mapped_high.io_mapping,
            StockhamIoMapping::FourStepRight(mapping)
        );
        assert_eq!(mapped_high.input_storage, PrecisionStorage::DoubleDouble);
        assert_eq!(mapped_high.output_storage, PrecisionStorage::DoubleDouble);
        let mapped_low = scheduled
            .forced_rader_two_upload_mapped_low_component()
            .unwrap()
            .expect("recursive p31 low upload should own final Four-step scatter");
        let DoubleDoubleRecursiveFftNodeIr::CooleyTukey(mapped_low) = mapped_low else {
            panic!("mapped p31 low upload should remain Cooley-Tukey");
        };
        assert_eq!(
            mapped_low.scatter_output.output_modifier,
            DoubleDoubleCooleyTukeyOutputModifier::FourStepLeft(mapping)
        );
        let mapped_low_scatter_name = mapped_low.scatter_output.name.clone();

        let program = crate::ProgramIr::double_double_recursive(&scheduled).unwrap();
        assert!(
            program
                .passes
                .iter()
                .all(|pass| pass.name != root.pack_right.name)
        );
        assert!(
            program
                .passes
                .iter()
                .all(|pass| pass.name != root.twiddle_transpose.name)
        );
        assert!(
            program
                .passes
                .iter()
                .all(|pass| pass.name != root.scatter_output.name)
        );
        assert!(
            program
                .resources
                .iter()
                .all(|resource| resource.name != "double_double_rader_four_step_left_output")
        );
        assert!(
            program
                .passes
                .iter()
                .any(|pass| pass.name == mapped_low_scatter_name)
        );
        let fft_scatter = program
            .passes
            .iter()
            .find(|pass| pass.name.contains("rader_fft_31") && pass.name.ends_with("_scatter"))
            .expect("mapped FFT-Rader scatter pass");
        assert!(fft_scatter.bindings.iter().any(|binding| {
            binding.binding == 3
                && binding.role == crate::BufferRole::TwiddleLookupTable
                && program.resources[binding.resource.0].elements == length
        }));
        let shaders = crate::backend::vulkan::VulkanGlslBackend
            .lower_double_double_recursive(&scheduled)
            .unwrap();
        assert_eq!(shaders.len(), program.passes.len());
        assert!(shaders.iter().all(|shader| shader.compile_spirv().is_ok()));
        let fft_scatter_shader = shaders
            .iter()
            .find(|shader| shader.glsl.contains("DD FFT-Rader natural-order scatter"))
            .expect("mapped FFT-Rader scatter shader");
        assert!(fft_scatter_shader.glsl.contains("vkfft_four_step_roots"));
        assert!(fft_scatter_shader.glsl.contains("vkfft_output_stride"));
        let low_scatter_shader = shaders
            .iter()
            .find(|shader| shader.glsl.contains("vkfft_four_step_k2"))
            .expect("mapped p31 recursive low scatter shader");
        assert!(
            low_scatter_shader
                .glsl
                .contains("VKFFT_DD_FOUR_STEP_RIGHT * component_index")
        );
    }

    #[test]
    fn forced_rader_two_upload_fft_high_preserves_f64_to_dd_boundary() {
        let length = 11usize * 17 * 31;
        let device = DeviceProfile {
            shared_memory_bytes: 24 * 1024,
            shared_memory_pow2_bytes: 24 * 1024,
            max_threads_per_block: 128,
            max_workgroup_size: [1024, 1024, 64],
            supports_f64: true,
            ..DeviceProfile::generic(crate::Backend::Vulkan, crate::GpuVendor::Nvidia)
        };
        let plan = FftPlan::build(
            FftConfig::new(vec![length]).with_precision(Precision::DoubleDoubleF64Storage),
        )
        .unwrap();
        let scheduled =
            DoubleDoubleRecursiveFftIr::build_for_device(&plan, Direction::Forward, device)
                .unwrap();
        let mapped_high = scheduled
            .forced_rader_two_upload_mapped_high_component()
            .unwrap()
            .expect("DD/F64 FFT-Rader high upload should be mapped");
        let DoubleDoubleRecursiveFftNodeIr::FftRader(mapped_high) = mapped_high else {
            panic!("DD/F64 mapped high upload should remain FFT Rader");
        };
        assert_eq!(mapped_high.input_storage, PrecisionStorage::F64);
        assert_eq!(mapped_high.output_storage, PrecisionStorage::DoubleDouble);
        assert!(matches!(
            mapped_high.io_mapping,
            StockhamIoMapping::FourStepRight(_)
        ));
        let mapped_low = scheduled
            .forced_rader_two_upload_mapped_low_component()
            .unwrap()
            .expect("DD/F64 FFT-Rader recursive low upload should be mapped");
        let DoubleDoubleRecursiveFftNodeIr::CooleyTukey(mapped_low) = mapped_low else {
            panic!("DD/F64 mapped FFT-Rader low upload should remain Cooley-Tukey");
        };
        assert_eq!(
            mapped_low.scatter_output.output_storage,
            PrecisionStorage::F64
        );
        assert!(matches!(
            mapped_low.scatter_output.output_modifier,
            DoubleDoubleCooleyTukeyOutputModifier::FourStepLeft(_)
        ));
        let program = crate::ProgramIr::double_double_recursive(&scheduled).unwrap();
        let shaders = crate::backend::vulkan::VulkanGlslBackend
            .lower_double_double_recursive(&scheduled)
            .unwrap();
        assert_eq!(shaders.len(), program.passes.len());
        let gather = shaders
            .iter()
            .find(|shader| shader.glsl.contains("DD FFT-Rader generator gather"))
            .expect("DD/F64 mapped FFT-Rader gather shader");
        let scatter = shaders
            .iter()
            .find(|shader| shader.glsl.contains("DD FFT-Rader natural-order scatter"))
            .expect("DD/F64 mapped FFT-Rader scatter shader");
        assert!(gather.glsl.contains("VkFftInput { dvec2 data[]; }"));
        assert!(scatter.glsl.contains("VkFftInput { dvec2 data[]; }"));
        assert!(scatter.glsl.contains("VkFftOutput { dvec4 data[]; }"));
        assert!(scatter.glsl.contains("vkfft_four_step_roots"));
        let low_scatter = shaders
            .iter()
            .find(|shader| shader.glsl.contains("vkfft_four_step_k2"))
            .expect("DD/F64 FFT-Rader mapped low scatter shader");
        assert!(low_scatter.glsl.contains("VkFftOutput { dvec2 data[]; }"));
        assert_eq!(low_scatter.compile_spirv().unwrap().words[0], 0x0723_0203);
        assert_eq!(gather.compile_spirv().unwrap().words[0], 0x0723_0203);
        assert_eq!(scatter.compile_spirv().unwrap().words[0], 0x0723_0203);
    }

    #[test]
    fn forced_rader_two_upload_direct_high_preserves_f64_to_dd_boundary() {
        let length = 11usize * 17 * 47;
        let device = DeviceProfile {
            shared_memory_bytes: 8 * 1024,
            shared_memory_pow2_bytes: 8 * 1024,
            max_threads_per_block: 1024,
            max_workgroup_size: [1024, 1024, 64],
            supports_f64: true,
            ..DeviceProfile::generic(crate::Backend::Vulkan, crate::GpuVendor::Nvidia)
        };
        let plan = FftPlan::build(
            FftConfig::new(vec![length]).with_precision(Precision::DoubleDoubleF64Storage),
        )
        .unwrap();
        let scheduled =
            DoubleDoubleRecursiveFftIr::build_for_device(&plan, Direction::Forward, device)
                .unwrap();
        let mapped_high = scheduled
            .forced_rader_two_upload_mapped_high_component()
            .unwrap()
            .expect("DD/F64 direct-Rader high upload should be mapped");
        let DoubleDoubleRecursiveFftNodeIr::DirectRader(mapped_high) = mapped_high else {
            panic!("DD/F64 mapped high upload should remain direct Rader");
        };
        assert_eq!(mapped_high.input_storage, PrecisionStorage::F64);
        assert_eq!(mapped_high.output_storage, PrecisionStorage::DoubleDouble);
        assert!(matches!(
            mapped_high.io_mapping,
            StockhamIoMapping::FourStepRight(_)
        ));
        let mapped_low = scheduled
            .forced_rader_two_upload_mapped_low_component()
            .unwrap()
            .expect("DD/F64 direct-Rader recursive low upload should be mapped");
        let DoubleDoubleRecursiveFftNodeIr::CooleyTukey(mapped_low) = mapped_low else {
            panic!("DD/F64 mapped direct-Rader low upload should remain Cooley-Tukey");
        };
        assert_eq!(
            mapped_low.scatter_output.output_storage,
            PrecisionStorage::F64
        );
        assert!(matches!(
            mapped_low.scatter_output.output_modifier,
            DoubleDoubleCooleyTukeyOutputModifier::FourStepLeft(_)
        ));
        let program = crate::ProgramIr::double_double_recursive(&scheduled).unwrap();
        let shaders = crate::backend::vulkan::VulkanGlslBackend
            .lower_double_double_recursive(&scheduled)
            .unwrap();
        assert_eq!(shaders.len(), program.passes.len());
        let direct_shader = shaders
            .iter()
            .find(|shader| shader.glsl.contains("DoubleDoubleDirectRaderIr"))
            .expect("DD/F64 mapped direct-Rader shader");
        assert!(direct_shader.glsl.contains("VkFftInput { dvec2 data[]; }"));
        assert!(direct_shader.glsl.contains("VkFftOutput { dvec4 data[]; }"));
        let low_scatter = shaders
            .iter()
            .find(|shader| shader.glsl.contains("vkfft_four_step_k2"))
            .expect("DD/F64 direct-Rader mapped low scatter shader");
        assert!(low_scatter.glsl.contains("VkFftOutput { dvec2 data[]; }"));
        assert_eq!(low_scatter.compile_spirv().unwrap().words[0], 0x0723_0203);
        assert_eq!(direct_shader.compile_spirv().unwrap().words[0], 0x0723_0203);
    }

    #[test]
    fn forced_rader_dd_f64_storage_uses_same_root_cut() {
        let length = 17usize * 300;
        let device = DeviceProfile {
            shared_memory_bytes: 48 * 1024,
            shared_memory_pow2_bytes: 32 * 1024,
            max_threads_per_block: 128,
            max_workgroup_size: [1024, 1024, 64],
            supports_f64: true,
            ..DeviceProfile::generic(crate::Backend::Vulkan, crate::GpuVendor::Nvidia)
        };
        let f64_storage_plan = FftPlan::build(
            FftConfig::new(vec![length]).with_precision(Precision::DoubleDoubleF64Storage),
        )
        .unwrap();
        let f64_storage = DoubleDoubleRecursiveFftIr::build_for_device(
            &f64_storage_plan,
            Direction::Forward,
            device,
        )
        .unwrap();
        assert_eq!(f64_storage.external_storage, PrecisionStorage::F64);
        assert_eq!(
            f64_storage
                .rader_forced_upload_schedule
                .as_ref()
                .expect("DD/F64 p17 axis should retain the same Quad forced-Rader split")
                .axis_split,
            vec![68, 75]
        );
        assert!(rader_upload_split_matches_root(
            &f64_storage.root,
            &[68, 75]
        ));
        f64_storage.validate().unwrap();
        let (mapped_high, _) = f64_storage
            .forced_rader_two_upload_mapped_high_stockham()
            .unwrap()
            .expect("DD/F64 Stockham high upload should own its Four-step input boundary");
        assert_eq!(mapped_high.sequence_len, 75);
        let mapped_low = f64_storage
            .forced_rader_two_upload_mapped_low_component()
            .unwrap()
            .expect("DD/F64 recursive low upload should own its Four-step output boundary");
        let DoubleDoubleRecursiveFftNodeIr::CooleyTukey(mapped_low) = mapped_low else {
            panic!("DD/F64 mapped low upload should remain Cooley-Tukey");
        };
        assert!(matches!(
            mapped_low.scatter_output.output_modifier,
            DoubleDoubleCooleyTukeyOutputModifier::FourStepLeft(_)
        ));
        assert_eq!(
            mapped_low.scatter_output.output_storage,
            PrecisionStorage::F64
        );
        let program = crate::ProgramIr::double_double_recursive(&f64_storage).unwrap();
        assert_eq!(
            program.input_resource().unwrap().scalar,
            crate::ScalarType::F64
        );
        assert_eq!(
            program.output_resource().unwrap().scalar,
            crate::ScalarType::F64
        );
        let shaders = crate::backend::vulkan::VulkanGlslBackend
            .lower_double_double_recursive(&f64_storage)
            .unwrap();
        assert_eq!(shaders.len(), program.passes.len());
        assert!(shaders.iter().all(|shader| shader.compile_spirv().is_ok()));
        let final_shader = shaders.last().expect("DD/F64 mapped low output shader");
        assert!(final_shader.glsl.contains("VkFftOutput { dvec2 data[]; }"));
    }

    #[test]
    fn device_aware_n8192_uses_quad_upload_split_and_high_level_route() {
        let length = 8_192usize;
        let device = DeviceProfile {
            shared_memory_bytes: 48 * 1024,
            shared_memory_pow2_bytes: 32 * 1024,
            max_threads_per_block: 1024,
            max_workgroup_size: [1024, 1024, 64],
            supports_f64: true,
            ..DeviceProfile::generic(crate::Backend::Vulkan, crate::GpuVendor::Nvidia)
        };
        let config = FftConfig::new(vec![length]).with_precision(Precision::DoubleDouble);
        let plan = FftPlan::build_for_device(config.clone(), device).unwrap();

        let portable = DoubleDoubleRecursiveFftIr::build(&plan, Direction::Forward).unwrap();
        assert!(portable.stockham_upload_schedule.is_none());
        let mut portable_leaves = Vec::new();
        collect_stockham_leaf_lengths(&portable.root, &mut portable_leaves).unwrap();
        assert_eq!(portable_leaves, vec![4_096, 2]);

        let scheduled =
            DoubleDoubleRecursiveFftIr::build_for_device(&plan, Direction::Forward, device)
                .unwrap();
        let upload = scheduled
            .stockham_upload_schedule
            .as_ref()
            .expect("device-aware DD N8192 should retain the Quad upload schedule");
        assert_eq!(upload.upload_count, 2);
        assert_eq!(upload.axis_split, vec![128, 64]);
        let mut scheduled_leaves = Vec::new();
        collect_stockham_leaf_lengths(&scheduled.root, &mut scheduled_leaves).unwrap();
        assert_eq!(scheduled_leaves, upload.axis_split);
        let DoubleDoubleRecursiveFftNodeIr::CooleyTukey(root) = &scheduled.root else {
            panic!("scheduled DD N8192 should remain a Cooley-Tukey root");
        };
        for leaf in [&root.left, &root.right] {
            let DoubleDoubleRecursiveFftNodeIr::Stockham(stockham) = leaf else {
                panic!("two-upload DD N8192 should have Stockham upload leaves");
            };
            let block = stockham
                .axis_batch_block
                .expect("scheduled DD upload leaf should own a physical Quad block");
            assert!(block.grouped_batch <= stockham.grouped_batch);
            assert!(block.threads_per_transform > 0);
        }
        scheduled.validate().unwrap();

        let program = crate::ProgramIr::double_double_recursive(&scheduled).unwrap();
        let shaders = crate::backend::vulkan::VulkanGlslBackend
            .lower_double_double_recursive(&scheduled)
            .unwrap();
        assert_eq!(program.passes.len(), 2);
        assert_eq!(shaders.len(), program.passes.len());
        for (index, shader) in shaders.iter().enumerate() {
            shader.compile_spirv().unwrap_or_else(|error| {
                panic!("DD N8192 Four-step upload {index} failed SPIR-V compilation: {error}")
            });
            assert!(shader.glsl.contains("fused DD Four-step upload"));
        }
        for expected in [128usize, 64usize] {
            assert!(shaders.iter().any(|shader| {
                shader.sequence_len == expected
                    && shader.workgroup_size.x > 1
                    && shader.glsl.contains("vkfft_fft_lane")
            }));
        }

        let high_level = crate::TransformIr::build(config, Direction::Forward, device).unwrap();
        let crate::TransformIr::Complex1dDoubleDouble(crate::DoubleDoubleOneDimIr::Recursive(ir)) =
            high_level
        else {
            panic!("high-level DD N8192 should route through recursive upload scheduling");
        };
        assert_eq!(
            ir.stockham_upload_schedule
                .as_ref()
                .expect("high-level DD N8192 should retain upload metadata")
                .axis_split,
            vec![128, 64]
        );
    }

    #[test]
    fn constrained_n8192_materializes_three_upload_four_step_and_spirv() {
        let length = 8_192usize;
        let device = DeviceProfile {
            shared_memory_bytes: 1024,
            shared_memory_pow2_bytes: 1024,
            max_threads_per_block: 1024,
            max_workgroup_size: [1024, 1024, 64],
            supports_f64: true,
            ..DeviceProfile::generic(crate::Backend::Vulkan, crate::GpuVendor::Nvidia)
        };
        let config = FftConfig::new(vec![length]).with_precision(Precision::DoubleDouble);
        let plan = FftPlan::build_for_device(config, device).unwrap();
        let ir = DoubleDoubleRecursiveFftIr::build_for_device(&plan, Direction::Forward, device)
            .unwrap();
        let schedule = ir
            .stockham_upload_schedule
            .as_ref()
            .expect("constrained DD N8192 should retain a three-upload schedule");
        assert_eq!(schedule.upload_count, 3);
        assert_eq!(schedule.axis_split, vec![32, 16, 16]);
        assert!(ir.two_upload_four_step_plan.is_none());
        let four_step = ir
            .three_upload_four_step_plan
            .expect("constrained DD N8192 should materialize three-upload Four-step metadata");
        assert_eq!(four_step.axis_split, [32, 16, 16]);
        ir.validate().unwrap();

        let program = crate::ProgramIr::double_double_recursive(&ir).unwrap();
        assert_eq!(program.passes.len(), 3);
        assert!(program.passes[0].name.ends_with("four_step_upload_2"));
        assert!(program.passes[1].name.ends_with("four_step_upload_1"));
        assert!(program.passes[2].name.ends_with("four_step_upload_0"));
        assert!(
            !program
                .passes
                .iter()
                .any(|pass| pass.name.contains("recursive_pack")
                    || pass.name.contains("recursive_twiddle")
                    || pass.name.contains("recursive_scatter"))
        );

        let shaders = crate::backend::vulkan::VulkanGlslBackend
            .lower_double_double_recursive(&ir)
            .unwrap();
        assert_eq!(shaders.len(), 3);
        assert_eq!(
            shaders
                .iter()
                .map(|shader| shader.sequence_len)
                .collect::<Vec<_>>(),
            vec![16, 16, 32]
        );
        assert!(shaders[0].glsl.contains("VKFFT_DD_FOUR_STEP_AB = 512u"));
        assert!(
            shaders[0]
                .glsl
                .contains("uint n12 = vkfft_batch % VKFFT_DD_FOUR_STEP_AB")
        );
        assert!(
            shaders[0]
                .glsl
                .contains("uint twiddle = (n12 * k3) % VKFFT_DD_FOUR_STEP_N")
        );
        assert!(
            shaders[1]
                .glsl
                .contains("uint twiddle = (n1 * k2) % VKFFT_DD_FOUR_STEP_AB")
        );
        assert!(
            shaders[2]
                .glsl
                .contains("VKFFT_DD_FOUR_STEP_C * VKFFT_DD_FOUR_STEP_B * i")
        );
        for (index, shader) in shaders.iter().enumerate() {
            shader.compile_spirv().unwrap_or_else(|error| {
                panic!("DD constrained N8192 three-upload {index} failed SPIR-V: {error}")
            });
        }
    }

    #[test]
    fn device_aware_n823543_fuses_343_point_three_upload_component() {
        let length = 823_543usize;
        let device = DeviceProfile {
            shared_memory_bytes: 16 * 1024,
            shared_memory_pow2_bytes: 16 * 1024,
            max_threads_per_block: 1024,
            max_workgroup_size: [1024, 1024, 64],
            supports_f64: true,
            ..DeviceProfile::generic(crate::Backend::Vulkan, crate::GpuVendor::Nvidia)
        };
        for precision in [Precision::DoubleDouble, Precision::DoubleDoubleF64Storage] {
            let plan = FftPlan::build_for_device(
                FftConfig::new(vec![length]).with_precision(precision),
                device,
            )
            .unwrap();
            let ir =
                DoubleDoubleRecursiveFftIr::build_for_device(&plan, Direction::Forward, device)
                    .unwrap();
            let schedule = ir
                .stockham_upload_schedule
                .as_ref()
                .expect("DD N823543 should retain its Quad upload schedule");
            assert_eq!(schedule.upload_count, 3);
            assert_eq!(schedule.axis_split, vec![343, 49, 49]);
            assert!(ir.two_upload_four_step_plan.is_none());
            let four_step = ir
                .three_upload_four_step_plan
                .expect("DD N823543 should fuse its 343x49x49 three-upload root");
            assert_eq!(four_step.axis_split, [343, 49, 49]);
            assert!(
                [
                    four_step.upload0_axis_block,
                    four_step.upload1_axis_block,
                    four_step.upload2_axis_block,
                ]
                .iter()
                .all(|block| block.grouped_batch > 0 && block.threads_per_transform > 0)
            );

            let program = crate::ProgramIr::double_double_recursive(&ir).unwrap();
            assert_eq!(program.passes.len(), 3);
            assert!(program.passes[0].name.ends_with("four_step_upload_2"));
            assert!(program.passes[1].name.ends_with("four_step_upload_1"));
            assert!(program.passes[2].name.ends_with("four_step_upload_0"));
            assert!(program.resources.iter().any(|resource| {
                resource.kind == crate::ProgramResourceKind::LookupTable
                    && resource.scalar == crate::ScalarType::DoubleDouble
                    && resource.elements == length
            }));

            let shaders = crate::backend::vulkan::VulkanGlslBackend
                .lower_double_double_recursive(&ir)
                .unwrap();
            assert_eq!(shaders.len(), 3);
            assert_eq!(
                shaders
                    .iter()
                    .map(|shader| shader.sequence_len)
                    .collect::<Vec<_>>(),
                vec![49, 49, 343]
            );
            assert!(shaders.iter().all(|shader| {
                shader.glsl.contains("fused DD Four-step upload")
                    && shader.required_shared_memory_bytes <= device.shared_memory_bytes
            }));
            if precision == Precision::DoubleDouble {
                let wide = shaders
                    .iter()
                    .find(|shader| shader.sequence_len == 343)
                    .expect("DD N823543 should emit the widened 343-point upload shader");
                assert_eq!(wide.compile_spirv().unwrap().words[0], 0x0723_0203);
            } else {
                assert!(shaders[0].glsl.contains("VkFftInput { dvec2 data[]; }"));
                assert!(shaders[2].glsl.contains("VkFftOutput { dvec2 data[]; }"));
            }
        }
    }

    #[test]
    fn device_aware_n9765625_fuses_625_point_three_upload_component() {
        let length = 9_765_625usize;
        let device = DeviceProfile {
            shared_memory_bytes: 48 * 1024,
            shared_memory_pow2_bytes: 32 * 1024,
            max_threads_per_block: 1024,
            max_workgroup_size: [1024, 1024, 64],
            supports_f64: true,
            ..DeviceProfile::generic(crate::Backend::Vulkan, crate::GpuVendor::Nvidia)
        };

        for precision in [Precision::DoubleDouble, Precision::DoubleDoubleF64Storage] {
            let plan = FftPlan::build_for_device(
                FftConfig::new(vec![length]).with_precision(precision),
                device,
            )
            .unwrap();
            let ir =
                DoubleDoubleRecursiveFftIr::build_for_device(&plan, Direction::Forward, device)
                    .unwrap();
            let schedule = ir
                .stockham_upload_schedule
                .as_ref()
                .expect("DD N9765625 should retain its Quad upload schedule");
            assert_eq!(schedule.upload_count, 3);
            assert_eq!(schedule.axis_split, vec![625, 125, 125]);
            assert!(ir.two_upload_four_step_plan.is_none());
            let four_step = ir
                .three_upload_four_step_plan
                .expect("DD N9765625 should fuse its 625x125x125 three-upload root");
            assert_eq!(four_step.axis_split, [625, 125, 125]);
            assert_eq!(four_step.upload0_axis_block.grouped_batch, 2);
            assert_eq!(625usize * 2 * 32, 40_000);
            assert!(
                [
                    four_step.upload0_axis_block,
                    four_step.upload1_axis_block,
                    four_step.upload2_axis_block,
                ]
                .iter()
                .all(|block| {
                    block.grouped_batch > 0
                        && block.threads_per_transform > 0
                        && block.local_size_x <= device.max_workgroup_size[0]
                        && block.local_size_y <= device.max_workgroup_size[1]
                })
            );

            // Lower directly instead of materializing ProgramIr: the latter owns full-period
            // true-DD root resources, which would make this 9.7M-point structural gate consume
            // hundreds of MiB merely to prove the widened 625-point component kernel.
            let shaders = crate::backend::vulkan::VulkanGlslBackend
                .lower_double_double_recursive(&ir)
                .unwrap();
            assert_eq!(shaders.len(), 3);
            assert_eq!(
                shaders
                    .iter()
                    .map(|shader| shader.sequence_len)
                    .collect::<Vec<_>>(),
                vec![125, 125, 625]
            );
            assert!(shaders.iter().all(|shader| {
                shader.glsl.contains("fused DD Four-step upload")
                    && shader.required_shared_memory_bytes <= device.shared_memory_bytes
            }));
            let wide = shaders
                .iter()
                .find(|shader| shader.sequence_len == 625)
                .expect("DD N9765625 should emit the widened 625-point upload shader");
            assert_eq!(wide.required_shared_memory_bytes, 40_000);
            if precision == Precision::DoubleDouble {
                assert_eq!(wide.compile_spirv().unwrap().words[0], 0x0723_0203);
            } else {
                assert!(shaders[0].glsl.contains("VkFftInput { dvec2 data[]; }"));
                assert!(shaders[2].glsl.contains("VkFftOutput { dvec2 data[]; }"));
            }
        }
    }

    #[test]
    fn device_aware_n16384_fuses_wide_two_upload_stockham_components() {
        let length = 16_384usize;
        let device = DeviceProfile {
            shared_memory_bytes: 48 * 1024,
            shared_memory_pow2_bytes: 32 * 1024,
            max_threads_per_block: 1024,
            max_workgroup_size: [1024, 1024, 64],
            supports_f64: true,
            ..DeviceProfile::generic(crate::Backend::Vulkan, crate::GpuVendor::Nvidia)
        };
        let plan = FftPlan::build_for_device(
            FftConfig::new(vec![length]).with_precision(Precision::DoubleDouble),
            device,
        )
        .unwrap();
        let ir = DoubleDoubleRecursiveFftIr::build_for_device(&plan, Direction::Forward, device)
            .unwrap();
        let schedule = ir
            .stockham_upload_schedule
            .as_ref()
            .expect("DD N16384 should retain its Quad upload schedule");
        assert_eq!(schedule.upload_count, 2);
        assert_eq!(schedule.axis_split, vec![256, 64]);
        let four_step = ir
            .two_upload_four_step_plan
            .expect("DD N16384 should fuse its 256x64 two-upload root");
        assert_eq!([four_step.left_len, four_step.right_len], [256, 64]);

        let program = crate::ProgramIr::double_double_recursive(&ir).unwrap();
        assert_eq!(program.passes.len(), 2);
        let shaders = crate::backend::vulkan::VulkanGlslBackend
            .lower_double_double_recursive(&ir)
            .unwrap();
        assert_eq!(shaders.len(), 2);
        assert_eq!(
            shaders
                .iter()
                .map(|shader| shader.sequence_len)
                .collect::<Vec<_>>(),
            vec![64, 256]
        );
        for shader in &shaders {
            assert!(shader.glsl.contains("fused DD Four-step upload"));
            assert!(shader.required_shared_memory_bytes <= device.shared_memory_bytes);
            assert_eq!(shader.compile_spirv().unwrap().words[0], 0x0723_0203);
        }
    }

    #[test]
    fn device_aware_n32768_fuses_512_point_two_upload_component() {
        let length = 32_768usize;
        let device = DeviceProfile {
            shared_memory_bytes: 32 * 1024,
            shared_memory_pow2_bytes: 32 * 1024,
            max_threads_per_block: 1024,
            max_workgroup_size: [1024, 1024, 64],
            supports_f64: true,
            ..DeviceProfile::generic(crate::Backend::Vulkan, crate::GpuVendor::Nvidia)
        };
        let plan = FftPlan::build_for_device(
            FftConfig::new(vec![length]).with_precision(Precision::DoubleDouble),
            device,
        )
        .unwrap();
        let ir = DoubleDoubleRecursiveFftIr::build_for_device(&plan, Direction::Forward, device)
            .unwrap();
        let schedule = ir
            .stockham_upload_schedule
            .as_ref()
            .expect("DD N32768 should retain its Quad upload schedule");
        assert_eq!(schedule.upload_count, 2);
        assert_eq!(schedule.axis_split, vec![512, 64]);
        let four_step = ir
            .two_upload_four_step_plan
            .expect("DD N32768 should fuse its 512x64 two-upload root");
        assert_eq!([four_step.left_len, four_step.right_len], [512, 64]);
        assert_eq!(
            [
                four_step.left_axis_block.local_size_x,
                four_step.left_axis_block.local_size_y,
            ],
            [64, 2]
        );
        assert_eq!(
            [
                four_step.right_axis_block.local_size_x,
                four_step.right_axis_block.local_size_y,
            ],
            [16, 8]
        );

        let program = crate::ProgramIr::double_double_recursive(&ir).unwrap();
        assert_eq!(program.passes.len(), 2);
        let shaders = crate::backend::vulkan::VulkanGlslBackend
            .lower_double_double_recursive(&ir)
            .unwrap();
        assert_eq!(shaders.len(), 2);
        assert_eq!(
            shaders
                .iter()
                .map(|shader| shader.sequence_len)
                .collect::<Vec<_>>(),
            vec![64, 512]
        );
        assert!(
            shaders
                .iter()
                .all(|shader| shader.required_shared_memory_bytes == 32 * 1024)
        );
        for shader in &shaders {
            assert!(shader.glsl.contains("fused DD Four-step upload"));
            assert_eq!(shader.compile_spirv().unwrap().words[0], 0x0723_0203);
        }
    }

    #[test]
    fn device_aware_n524288_fuses_1024_point_two_upload_component() {
        let length = 524_288usize;
        let device = DeviceProfile {
            shared_memory_bytes: 32 * 1024,
            shared_memory_pow2_bytes: 32 * 1024,
            max_threads_per_block: 1024,
            max_workgroup_size: [1024, 1024, 64],
            supports_f64: true,
            ..DeviceProfile::generic(crate::Backend::Vulkan, crate::GpuVendor::Nvidia)
        };
        for precision in [Precision::DoubleDouble, Precision::DoubleDoubleF64Storage] {
            let plan = FftPlan::build_for_device(
                FftConfig::new(vec![length]).with_precision(precision),
                device,
            )
            .unwrap();
            let ir =
                DoubleDoubleRecursiveFftIr::build_for_device(&plan, Direction::Forward, device)
                    .unwrap();
            let schedule = ir
                .stockham_upload_schedule
                .as_ref()
                .expect("DD N524288 should retain its Quad upload schedule");
            assert_eq!(schedule.upload_count, 2);
            assert_eq!(schedule.axis_split, vec![1024, 512]);
            let four_step = ir
                .two_upload_four_step_plan
                .expect("DD N524288 should fuse its 1024x512 two-upload root");
            assert_eq!([four_step.left_len, four_step.right_len], [1024, 512]);
            assert!(
                four_step.left_axis_block.grouped_batch > 0
                    && four_step.left_axis_block.threads_per_transform > 0
                    && four_step.right_axis_block.grouped_batch > 0
                    && four_step.right_axis_block.threads_per_transform > 0
            );

            let program = crate::ProgramIr::double_double_recursive(&ir).unwrap();
            assert_eq!(program.passes.len(), 2);
            assert!(program.resources.iter().any(|resource| {
                resource.kind == crate::ProgramResourceKind::LookupTable
                    && resource.scalar == crate::ScalarType::DoubleDouble
                    && resource.elements == length
            }));
            let shaders = crate::backend::vulkan::VulkanGlslBackend
                .lower_double_double_recursive(&ir)
                .unwrap();
            assert_eq!(shaders.len(), 2);
            assert_eq!(
                shaders
                    .iter()
                    .map(|shader| shader.sequence_len)
                    .collect::<Vec<_>>(),
                vec![512, 1024]
            );
            assert!(
                shaders
                    .iter()
                    .all(|shader| shader.required_shared_memory_bytes <= device.shared_memory_bytes)
            );
            for shader in &shaders {
                assert!(shader.glsl.contains("fused DD Four-step upload"));
            }
            if precision == Precision::DoubleDouble {
                let wide = shaders
                    .iter()
                    .find(|shader| shader.sequence_len == 1024)
                    .expect("DD N524288 should emit the widened 1024-point upload shader");
                assert_eq!(wide.compile_spirv().unwrap().words[0], 0x0723_0203);
            } else {
                assert!(shaders[0].glsl.contains("VkFftInput { dvec2 data[]; }"));
                assert!(shaders[1].glsl.contains("VkFftOutput { dvec2 data[]; }"));
            }
        }
    }

    #[test]
    fn device_aware_n2097152_fuses_2048_point_two_upload_component() {
        let length = 2_097_152usize;
        let device = DeviceProfile {
            shared_memory_bytes: 64 * 1024,
            shared_memory_pow2_bytes: 64 * 1024,
            max_threads_per_block: 1024,
            max_workgroup_size: [1024, 1024, 64],
            supports_f64: true,
            ..DeviceProfile::generic(crate::Backend::Vulkan, crate::GpuVendor::Nvidia)
        };
        for precision in [Precision::DoubleDouble, Precision::DoubleDoubleF64Storage] {
            let plan = FftPlan::build_for_device(
                FftConfig::new(vec![length]).with_precision(precision),
                device,
            )
            .unwrap();
            let ir =
                DoubleDoubleRecursiveFftIr::build_for_device(&plan, Direction::Forward, device)
                    .unwrap();
            let schedule = ir
                .stockham_upload_schedule
                .as_ref()
                .expect("DD N2097152 should retain its 64KiB Quad upload schedule");
            assert_eq!(schedule.upload_count, 2);
            assert_eq!(schedule.axis_split, vec![2048, 1024]);
            let four_step = ir
                .two_upload_four_step_plan
                .expect("DD N2097152 should fuse its 2048x1024 two-upload root");
            assert_eq!([four_step.left_len, four_step.right_len], [2048, 1024]);
            assert_eq!(four_step.left_axis_block.grouped_batch, 1);
            assert_eq!(four_step.left_axis_block.threads_per_transform, 256);
            assert_eq!(2048usize * 32, device.shared_memory_bytes);

            let program = crate::ProgramIr::double_double_recursive(&ir).unwrap();
            assert_eq!(program.passes.len(), 2);
            assert!(program.resources.iter().any(|resource| {
                resource.kind == crate::ProgramResourceKind::LookupTable
                    && resource.scalar == crate::ScalarType::DoubleDouble
                    && resource.elements == length
            }));
            let shaders = crate::backend::vulkan::VulkanGlslBackend
                .lower_double_double_recursive(&ir)
                .unwrap();
            assert_eq!(shaders.len(), 2);
            assert_eq!(
                shaders
                    .iter()
                    .map(|shader| shader.sequence_len)
                    .collect::<Vec<_>>(),
                vec![1024, 2048]
            );
            let wide = shaders
                .iter()
                .find(|shader| shader.sequence_len == 2048)
                .expect("DD N2097152 should emit a fused 2048-point upload shader");
            assert_eq!(wide.required_shared_memory_bytes, 64 * 1024);
            assert!(wide.glsl.contains("fused DD Four-step upload"));
            if precision == Precision::DoubleDouble {
                assert_eq!(wide.compile_spirv().unwrap().words[0], 0x0723_0203);
            } else {
                assert!(shaders[0].glsl.contains("VkFftInput { dvec2 data[]; }"));
                assert!(shaders[1].glsl.contains("VkFftOutput { dvec2 data[]; }"));
            }
        }
    }

    #[test]
    fn device_aware_n823543_96k_fuses_2401_point_two_upload_component() {
        let length = 823_543usize;
        let device = DeviceProfile {
            shared_memory_bytes: 96 * 1024,
            shared_memory_pow2_bytes: 64 * 1024,
            max_threads_per_block: 1024,
            max_workgroup_size: [1024, 1024, 64],
            supports_f64: true,
            ..DeviceProfile::generic(crate::Backend::Vulkan, crate::GpuVendor::Nvidia)
        };

        for precision in [Precision::DoubleDouble, Precision::DoubleDoubleF64Storage] {
            let plan = FftPlan::build_for_device(
                FftConfig::new(vec![length]).with_precision(precision),
                device,
            )
            .unwrap();
            let ir =
                DoubleDoubleRecursiveFftIr::build_for_device(&plan, Direction::Forward, device)
                    .unwrap();
            let schedule = ir
                .stockham_upload_schedule
                .as_ref()
                .expect("DD N823543 should retain its 96KiB Quad upload schedule");
            assert_eq!(schedule.upload_count, 2);
            assert_eq!(schedule.axis_split, vec![2401, 343]);
            assert!(ir.three_upload_four_step_plan.is_none());
            let four_step = ir
                .two_upload_four_step_plan
                .expect("DD N823543 should fuse its 2401x343 two-upload root on 96KiB");
            assert_eq!([four_step.left_len, four_step.right_len], [2401, 343]);
            assert_eq!(four_step.left_axis_block.grouped_batch, 1);
            assert_eq!(2401usize * 32, 76_832);

            // Lower directly rather than cloning the full-period parent roots into ProgramIr.
            // This gate is about the planner-reachable 2401-point component on a larger-memory
            // device profile; the current 48KiB NVIDIA runtime is intentionally not claimed.
            let shaders = crate::backend::vulkan::VulkanGlslBackend
                .lower_double_double_recursive(&ir)
                .unwrap();
            assert_eq!(shaders.len(), 2);
            assert_eq!(
                shaders
                    .iter()
                    .map(|shader| shader.sequence_len)
                    .collect::<Vec<_>>(),
                vec![343, 2401]
            );
            let wide = shaders
                .iter()
                .find(|shader| shader.sequence_len == 2401)
                .expect("DD N823543 should emit a fused 2401-point upload shader");
            assert_eq!(wide.required_shared_memory_bytes, 76_832);
            assert!(wide.required_shared_memory_bytes <= device.shared_memory_bytes);
            assert!(wide.glsl.contains("fused DD Four-step upload"));
            if precision == Precision::DoubleDouble {
                assert_eq!(wide.compile_spirv().unwrap().words[0], 0x0723_0203);
            } else {
                assert!(shaders[0].glsl.contains("VkFftInput { dvec2 data[]; }"));
                assert!(shaders[1].glsl.contains("VkFftOutput { dvec2 data[]; }"));
            }
        }
    }

    #[test]
    fn device_aware_n1953125_128k_fuses_3125_point_two_upload_component() {
        let length = 1_953_125usize;
        let device = DeviceProfile {
            shared_memory_bytes: 128 * 1024,
            shared_memory_pow2_bytes: 128 * 1024,
            max_threads_per_block: 1024,
            max_workgroup_size: [1024, 1024, 64],
            supports_f64: true,
            ..DeviceProfile::generic(crate::Backend::Vulkan, crate::GpuVendor::Nvidia)
        };

        for precision in [Precision::DoubleDouble, Precision::DoubleDoubleF64Storage] {
            let plan = FftPlan::build_for_device(
                FftConfig::new(vec![length]).with_precision(precision),
                device,
            )
            .unwrap();
            let ir =
                DoubleDoubleRecursiveFftIr::build_for_device(&plan, Direction::Forward, device)
                    .unwrap();
            let schedule = ir
                .stockham_upload_schedule
                .as_ref()
                .expect("DD N1953125 should retain its 128KiB Quad upload schedule");
            assert_eq!(schedule.upload_count, 2);
            assert_eq!(schedule.axis_split, vec![3125, 625]);
            assert!(ir.three_upload_four_step_plan.is_none());
            let four_step = ir
                .two_upload_four_step_plan
                .expect("DD N1953125 should fuse its 3125x625 two-upload root on 128KiB");
            assert_eq!([four_step.left_len, four_step.right_len], [3125, 625]);
            assert_eq!(four_step.left_axis_block.grouped_batch, 1);
            assert!(
                four_step.left_axis_block.threads_per_transform <= device.max_threads_per_block
            );
            assert!(
                four_step.right_axis_block.threads_per_transform <= device.max_threads_per_block
            );

            // Keep this builder gate structural. Component-local lowering below owns the 3125-point
            // SPIR-V proof without materializing the 1.95M-point parent root resources.
            let shaders = crate::backend::vulkan::VulkanGlslBackend
                .lower_double_double_recursive(&ir)
                .unwrap();
            assert_eq!(shaders.len(), 2);
            assert_eq!(
                shaders
                    .iter()
                    .map(|shader| shader.sequence_len)
                    .collect::<Vec<_>>(),
                vec![625, 3125]
            );
            let wide = shaders
                .iter()
                .find(|shader| shader.sequence_len == 3125)
                .expect("DD N1953125 should emit a fused 3125-point upload shader");
            assert_eq!(wide.required_shared_memory_bytes, 100_000);
            assert!(wide.glsl.contains("fused DD Four-step upload"));
            if precision == Precision::DoubleDoubleF64Storage {
                assert!(shaders[0].glsl.contains("VkFftInput { dvec2 data[]; }"));
                assert!(shaders[1].glsl.contains("VkFftOutput { dvec2 data[]; }"));
            }
        }
    }

    #[test]
    fn device_aware_n1679616_fuses_1296_point_two_upload_components() {
        let length = 1_679_616usize;
        let device = DeviceProfile {
            shared_memory_bytes: 48 * 1024,
            shared_memory_pow2_bytes: 32 * 1024,
            max_threads_per_block: 1024,
            max_workgroup_size: [1024, 1024, 64],
            supports_f64: true,
            ..DeviceProfile::generic(crate::Backend::Vulkan, crate::GpuVendor::Nvidia)
        };

        let plan = FftPlan::build_for_device(
            FftConfig::new(vec![length]).with_precision(Precision::DoubleDouble),
            device,
        )
        .unwrap();
        let ir = DoubleDoubleRecursiveFftIr::build_for_device(&plan, Direction::Forward, device)
            .unwrap();
        let schedule = ir
            .stockham_upload_schedule
            .as_ref()
            .expect("DD N1679616 should retain its Quad upload schedule");
        assert_eq!(schedule.upload_count, 2);
        assert_eq!(schedule.axis_split, vec![1296, 1296]);
        let four_step = ir
            .two_upload_four_step_plan
            .expect("DD N1679616 should fuse its 1296x1296 two-upload root");
        assert_eq!([four_step.left_len, four_step.right_len], [1296, 1296]);
        assert_eq!(four_step.left_axis_block.grouped_batch, 1);
        assert_eq!(four_step.right_axis_block.grouped_batch, 1);
        assert!(
            1296usize * 32 <= device.shared_memory_bytes
                && four_step.left_axis_block.threads_per_transform <= device.max_threads_per_block
                && four_step.right_axis_block.threads_per_transform <= device.max_threads_per_block
        );

        let program = crate::ProgramIr::double_double_recursive(&ir).unwrap();
        assert_eq!(program.passes.len(), 2);
        assert!(program.resources.iter().any(|resource| {
            resource.kind == crate::ProgramResourceKind::LookupTable
                && resource.scalar == crate::ScalarType::DoubleDouble
                && resource.elements == length
        }));
        let shaders = crate::backend::vulkan::VulkanGlslBackend
            .lower_double_double_recursive(&ir)
            .unwrap();
        assert_eq!(shaders.len(), 2);
        assert_eq!(
            shaders
                .iter()
                .map(|shader| shader.sequence_len)
                .collect::<Vec<_>>(),
            vec![1296, 1296]
        );
        assert!(shaders.iter().all(|shader| {
            shader.required_shared_memory_bytes == 1296 * 32
                && shader.glsl.contains("fused DD Four-step upload")
        }));
        assert_eq!(shaders[0].compile_spirv().unwrap().words[0], 0x0723_0203);

        let f64_plan = FftPlan::build_for_device(
            FftConfig::new(vec![length]).with_precision(Precision::DoubleDoubleF64Storage),
            device,
        )
        .unwrap();
        let f64_ir =
            DoubleDoubleRecursiveFftIr::build_for_device(&f64_plan, Direction::Forward, device)
                .unwrap();
        assert_eq!(
            f64_ir
                .stockham_upload_schedule
                .as_ref()
                .expect("DD/F64 N1679616 upload schedule")
                .axis_split,
            vec![1296, 1296]
        );
        assert!(f64_ir.two_upload_four_step_plan.is_some());
        let f64_shaders = crate::backend::vulkan::VulkanGlslBackend
            .lower_double_double_recursive(&f64_ir)
            .unwrap();
        assert!(f64_shaders[0].glsl.contains("VkFftInput { dvec2 data[]; }"));
        assert!(
            f64_shaders[1]
                .glsl
                .contains("VkFftOutput { dvec2 data[]; }")
        );

        let padded_plan = FftPlan::build_for_device(
            FftConfig::new(vec![length])
                .with_precision(Precision::DoubleDouble)
                .with_zero_padding(0, 10_000, 20_000)
                .unwrap(),
            device,
        )
        .unwrap();
        let padded =
            DoubleDoubleRecursiveFftIr::build_for_device(&padded_plan, Direction::Forward, device)
                .unwrap();
        assert_eq!(
            padded
                .stockham_upload_schedule
                .as_ref()
                .expect("padded DD N1679616 upload schedule")
                .axis_split,
            vec![1296, 1296]
        );
        assert!(padded.two_upload_four_step_plan.is_some());
        assert!(padded.zero_pad_pass.is_some());
        let padded_program = crate::ProgramIr::double_double_recursive(&padded).unwrap();
        assert_eq!(padded_program.passes.len(), 3);
        assert!(
            padded_program.passes[0]
                .name
                .contains("zero_pad_storage_forward_input")
        );
        let padded_shaders = crate::backend::vulkan::VulkanGlslBackend
            .lower_double_double_recursive(&padded)
            .unwrap();
        assert_eq!(padded_shaders.len(), 3);
        assert!(padded_shaders[0].glsl.contains("VKFFT_ZERO_PAD_LEFT"));
        assert!(padded_shaders[1..].iter().all(|shader| {
            shader.sequence_len == 1296 && shader.glsl.contains("fused DD Four-step upload")
        }));
        assert_eq!(
            padded_shaders[0].compile_spirv().unwrap().words[0],
            0x0723_0203
        );
    }

    #[test]
    fn repeated_rader_289_spatial_zero_padding_wraps_only_recursive_root() {
        let length = 17usize * 17;
        let batch_count = 3usize;
        let left = 97usize;
        let right = 151usize;
        let base = FftConfig::new(vec![length])
            .with_batch_count(batch_count)
            .with_precision(Precision::DoubleDouble)
            .with_grouped_batch(0, 2)
            .unwrap();
        let padded_plan =
            FftPlan::build(base.clone().with_zero_padding(0, left, right).unwrap()).unwrap();
        let forward = DoubleDoubleRecursiveFftIr::build(&padded_plan, Direction::Forward).unwrap();
        let pass = forward
            .zero_pad_pass
            .as_ref()
            .expect("DD recursive root should retain one external zero-pad pass");
        assert_eq!(pass.range.left, left);
        assert_eq!(pass.range.right, right);
        assert_eq!(pass.grouped_batch, 2);
        assert!(matches!(
            forward.root,
            DoubleDoubleRecursiveFftNodeIr::CooleyTukey(_)
        ));

        let input = sample(length, batch_count);
        let mut manual = input.clone();
        for batch in 0..batch_count {
            let base = batch * length;
            manual[base + left..base + right].fill(ComplexDoubleDouble::default());
        }
        let baseline = DoubleDoubleRecursiveFftIr::build(
            &FftPlan::build(base.clone()).unwrap(),
            Direction::Forward,
        )
        .unwrap();
        let expected = execute_double_double_recursive_ir(&baseline, &manual).unwrap();
        let actual = execute_double_double_recursive_ir(&forward, &input).unwrap();
        assert_eq!(actual, expected);

        let inverse_base = base.clone().with_inverse_normalization(true);
        let inverse = DoubleDoubleRecursiveFftIr::build(
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
        let baseline_inverse = DoubleDoubleRecursiveFftIr::build(
            &FftPlan::build(inverse_base).unwrap(),
            Direction::Inverse,
        )
        .unwrap();
        let mut inverse_expected =
            execute_double_double_recursive_ir(&baseline_inverse, &actual).unwrap();
        for batch in 0..batch_count {
            let base = batch * length;
            inverse_expected[base + left..base + right].fill(ComplexDoubleDouble::default());
        }
        let inverse_actual = execute_double_double_recursive_ir(&inverse, &actual).unwrap();
        assert_eq!(inverse_actual, inverse_expected);

        let baseline_program = crate::ProgramIr::double_double_recursive(&baseline).unwrap();
        let program = crate::ProgramIr::double_double_recursive(&forward).unwrap();
        assert_eq!(program.passes.len(), baseline_program.passes.len() + 1);
        assert!(
            program.passes[0]
                .name
                .contains("zero_pad_storage_forward_input")
        );
        let shaders = crate::backend::vulkan::VulkanGlslBackend
            .lower_double_double_recursive(&forward)
            .unwrap();
        assert_eq!(shaders.len(), program.passes.len());
        assert!(shaders[0].glsl.contains("VKFFT_ZERO_PAD_LEFT"));
        for shader in shaders {
            assert_eq!(shader.compile_spirv().unwrap().words[0], 0x0723_0203);
        }

        let f64_base = FftConfig::new(vec![length])
            .with_batch_count(batch_count)
            .with_precision(Precision::DoubleDoubleF64Storage)
            .with_grouped_batch(0, 2)
            .unwrap();
        let f64_forward = DoubleDoubleRecursiveFftIr::build(
            &FftPlan::build(f64_base.clone().with_zero_padding(0, left, right).unwrap()).unwrap(),
            Direction::Forward,
        )
        .unwrap();
        let f64_baseline = DoubleDoubleRecursiveFftIr::build(
            &FftPlan::build(f64_base).unwrap(),
            Direction::Forward,
        )
        .unwrap();
        let f64_input = input
            .iter()
            .copied()
            .map(ComplexDoubleDouble::to_complex64)
            .collect::<Vec<_>>();
        let mut f64_manual = f64_input.clone();
        for batch in 0..batch_count {
            let base = batch * length;
            f64_manual[base + left..base + right].fill(Complex64::default());
        }
        let f64_expected =
            execute_double_double_recursive_ir_f64_storage(&f64_baseline, &f64_manual).unwrap();
        let f64_actual =
            execute_double_double_recursive_ir_f64_storage(&f64_forward, &f64_input).unwrap();
        assert_eq!(f64_actual, f64_expected);
        let f64_program = crate::ProgramIr::double_double_recursive(&f64_forward).unwrap();
        assert!(f64_program.resources.iter().any(|resource| {
            resource.kind == crate::ProgramResourceKind::Scratch
                && resource.scalar == ScalarType::F64
                && resource.name.contains("zero_padded_input")
        }));
        let f64_shaders = crate::backend::vulkan::VulkanGlslBackend
            .lower_double_double_recursive(&f64_forward)
            .unwrap();
        assert_eq!(f64_shaders.len(), f64_program.passes.len());
        for shader in f64_shaders {
            assert_eq!(shader.compile_spirv().unwrap().words[0], 0x0723_0203);
        }
    }

    #[test]
    fn repeated_rader_289_composes_and_round_trips_in_full_dd() {
        let length = 17usize * 17;
        let forward_plan =
            FftPlan::build(FftConfig::new(vec![length]).with_precision(Precision::DoubleDouble))
                .unwrap();
        let inverse_plan = FftPlan::build(
            FftConfig::new(vec![length])
                .with_precision(Precision::DoubleDouble)
                .with_inverse_normalization(true),
        )
        .unwrap();
        let forward = DoubleDoubleRecursiveFftIr::build(&forward_plan, Direction::Forward).unwrap();
        let inverse = DoubleDoubleRecursiveFftIr::build(&inverse_plan, Direction::Inverse).unwrap();
        assert!(matches!(
            forward.root,
            DoubleDoubleRecursiveFftNodeIr::CooleyTukey(_)
        ));
        let input = sample(length, 1);
        let actual = execute_double_double_recursive_ir(&forward, &input).unwrap();
        let expected = dft(&input, Direction::Forward, false).unwrap();
        assert!(max_error(&actual, &expected) < 4.0e-24);
        let restored = execute_double_double_recursive_ir(&inverse, &actual).unwrap();
        assert!(max_error(&restored, &input) < 4.0e-22);
    }

    #[test]
    fn multi_prime_323_composes_and_f64_storage_round_trips() {
        let length = 17usize * 19;
        let batch_count = 2usize;
        let forward_plan = FftPlan::build(
            FftConfig::new(vec![length])
                .with_batch_count(batch_count)
                .with_precision(Precision::DoubleDoubleF64Storage),
        )
        .unwrap();
        let inverse_plan = FftPlan::build(
            FftConfig::new(vec![length])
                .with_batch_count(batch_count)
                .with_precision(Precision::DoubleDoubleF64Storage)
                .with_inverse_normalization(true),
        )
        .unwrap();
        let forward = DoubleDoubleRecursiveFftIr::build(&forward_plan, Direction::Forward).unwrap();
        let inverse = DoubleDoubleRecursiveFftIr::build(&inverse_plan, Direction::Inverse).unwrap();
        assert_eq!(forward.external_storage, PrecisionStorage::F64);
        let input = sample(length, batch_count)
            .into_iter()
            .map(ComplexDoubleDouble::to_complex64)
            .collect::<Vec<_>>();
        let spectrum = execute_double_double_recursive_ir_f64_storage(&forward, &input).unwrap();
        let restored = execute_double_double_recursive_ir_f64_storage(&inverse, &spectrum).unwrap();
        let error = restored
            .iter()
            .zip(&input)
            .map(|(actual, expected)| (*actual - *expected).norm_sqr().sqrt())
            .fold(0.0, f64::max);
        assert!(
            error < 5.0e-12,
            "DD/F64 recursive round-trip error {error:e}"
        );
    }

    #[test]
    fn grouped_tail_ownership_propagates_through_recursive_children() {
        let batch_count = 7usize;
        let grouped_batch = 3usize;
        for length in [17usize * 17, 17usize * 19] {
            let config = FftConfig::new(vec![length])
                .with_batch_count(batch_count)
                .with_precision(Precision::DoubleDouble)
                .with_grouped_batch(0, grouped_batch)
                .unwrap();
            let plan = FftPlan::build(config).unwrap();
            let ir = DoubleDoubleRecursiveFftIr::build(&plan, Direction::Forward).unwrap();
            assert_eq!(ir.grouped_batch, grouped_batch);
            let DoubleDoubleRecursiveFftNodeIr::CooleyTukey(root) = &ir.root else {
                panic!("grouped recursive DD root must be Cooley-Tukey");
            };
            assert_eq!(root.grouped_batch, grouped_batch);
            assert_eq!(root.pack_right.dispatch.x, 3);
            assert_eq!(root.twiddle_transpose.dispatch.x, 3);
            assert_eq!(root.scatter_output.dispatch.x, 3);
            assert_eq!(root.right.batch_count(), batch_count * root.left_len);
            assert_eq!(root.right.grouped_batch(), grouped_batch * root.left_len);
            assert_eq!(root.left.batch_count(), batch_count * root.right_len);
            assert_eq!(root.left.grouped_batch(), grouped_batch * root.right_len);
            assert_eq!(root.right.batch_group_count(), 3);
            assert_eq!(root.left.batch_group_count(), 3);

            let program = crate::ProgramIr::double_double_recursive(&ir).unwrap();
            assert!(program.passes.iter().all(|pass| pass.dispatch.x == 3));
            let shaders = crate::backend::vulkan::VulkanGlslBackend
                .lower_double_double_recursive(&ir)
                .unwrap();
            assert_eq!(shaders.len(), program.passes.len());
            assert!(shaders.iter().all(|shader| shader.dispatch.x == 3));
            assert!(shaders.iter().any(|shader| {
                shader
                    .glsl
                    .contains("groupedBatch ownership: 3 transforms/workgroup")
            }));
        }
    }

    #[test]
    fn recursive_program_spirv_and_native_sources_cover_dd_and_f64_boundaries() {
        for (length, batch_count) in [(17usize * 17, 1usize), (17usize * 19, 2usize)] {
            for precision in [Precision::DoubleDouble, Precision::DoubleDoubleF64Storage] {
                let plan = FftPlan::build(
                    FftConfig::new(vec![length])
                        .with_batch_count(batch_count)
                        .with_precision(precision),
                )
                .unwrap();
                let ir = DoubleDoubleRecursiveFftIr::build(&plan, Direction::Forward).unwrap();
                let program = crate::ProgramIr::double_double_recursive(&ir).unwrap();
                let shaders = crate::backend::vulkan::VulkanGlslBackend
                    .lower_double_double_recursive(&ir)
                    .unwrap();
                assert_eq!(shaders.len(), program.passes.len());
                assert!(
                    program
                        .passes
                        .first()
                        .unwrap()
                        .name
                        .contains("dd_recursive_pack")
                );
                assert!(
                    program
                        .passes
                        .last()
                        .unwrap()
                        .name
                        .contains("dd_recursive_scatter")
                );
                assert!(program.resources.iter().any(|resource| matches!(
                    &resource.initialization,
                    crate::ProgramResourceInitialization::ComplexDoubleDouble(values)
                        if values.len() == length
                )));
                let twiddle_pass = program
                    .passes
                    .iter()
                    .position(|pass| pass.name.contains("dd_recursive_twiddle"))
                    .unwrap();
                assert_eq!(program.passes[twiddle_pass].bindings.len(), 3);
                assert_eq!(
                    program.passes[twiddle_pass].bindings[2].role,
                    crate::BufferRole::TwiddleLookupTable
                );
                assert!(shaders[twiddle_pass].glsl.contains("vkfft_dd_cmul"));
                assert!(shaders[twiddle_pass].glsl.contains("vkfft_twiddles.data"));
                for (shader, pass) in shaders.iter().zip(&program.passes) {
                    assert_eq!(shader.scalar, crate::ScalarType::DoubleDouble);
                    assert_eq!(shader.descriptors.len(), pass.bindings.len());
                    assert_eq!(shader.compile_spirv().unwrap().words[0], 0x0723_0203);
                }
                match precision {
                    Precision::DoubleDouble => {
                        assert_eq!(
                            program.input_resource().unwrap().scalar,
                            crate::ScalarType::DoubleDouble
                        );
                        assert_eq!(
                            program.output_resource().unwrap().scalar,
                            crate::ScalarType::DoubleDouble
                        );
                        assert!(
                            shaders
                                .first()
                                .unwrap()
                                .glsl
                                .contains("VkFftInput { dvec4 data[]; }")
                        );
                        assert!(
                            shaders
                                .last()
                                .unwrap()
                                .glsl
                                .contains("VkFftOutput { dvec4 data[]; }")
                        );
                    }
                    Precision::DoubleDoubleF64Storage => {
                        assert_eq!(
                            program.input_resource().unwrap().scalar,
                            crate::ScalarType::F64
                        );
                        assert_eq!(
                            program.output_resource().unwrap().scalar,
                            crate::ScalarType::F64
                        );
                        assert!(
                            shaders
                                .first()
                                .unwrap()
                                .glsl
                                .contains("VkFftInput { dvec2 data[]; }")
                        );
                        assert!(
                            shaders
                                .last()
                                .unwrap()
                                .glsl
                                .contains("VkFftOutput { dvec2 data[]; }")
                        );
                    }
                    _ => unreachable!(),
                }
                for backend in [crate::Backend::Cuda, crate::Backend::OpenCl] {
                    let native = crate::backend::native::NativeSourceBackend::new(backend)
                        .lower_double_double_recursive(&ir)
                        .unwrap();
                    assert_eq!(native.shaders.len(), native.program.passes.len());
                    assert!(native.shaders.iter().all(|shader| {
                        shader.source.contains("VkFFT_main")
                            && !shader.source.contains("layout(set")
                            && !shader.source.contains("vkfft_twiddles.data")
                    }));
                }
            }
        }
    }
}
