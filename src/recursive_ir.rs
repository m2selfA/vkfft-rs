//! Recursive backend-neutral Cooley-Tukey decomposition for composite Rader axes.
//!
//! Planner Rader metadata can contain a smooth Stockham part, multiple distinct
//! Rader primes, and repeated prime multiplicities. This module expands that axis
//! into a binary Cooley-Tukey tree whose leaves reuse the already validated
//! Stockham, direct-Rader, and FFT-convolution-Rader IRs.

use core::f64::consts::TAU;

use crate::complex::Complex64;
use crate::config::{
    DeviceProfile, Direction, FftConfig, PlannerTuning, Precision, TransformKind, ZeroPaddingRange,
    upstream_coalesced_memory_bytes_for_precision, upstream_effective_rader_tuning,
};
use crate::error::{Result, VkFftError};
use crate::kernel_ir::{
    CooleyLeftStockhamMapping, CooleyRightInputMapping, DispatchGeometry, FourStepMapping,
    KernelIr, KernelOperation, RaderGeneratorMapping, RaderScatterMapping,
    RealEvenInversePreprocessMapping, RealEvenPackMapping, RealEvenPostprocessMapping,
    RealEvenUnpackMapping, ScalarType, StockhamIoMapping, ThreeUploadFourStepMapping,
    WorkgroupSize, executable_register_schedule_layout, execute_stockham_ir,
    plan_gpu_stockham_shared_memory_layout,
};
use crate::planner::{
    AxisAlgorithm, C2cDeviceAxisClass, FftPlan, RaderMode, RaderPrimePlan, prime_factorization,
};
use crate::rader_ir::{
    RaderDirectIr, RaderFftPipelineIr, execute_rader_direct_ir, execute_rader_fft_ir,
};
use crate::scheduler::{
    FourStepAxisBlockRequest, OtherAxisFourStepPhysicalContext, RaderFftTransposeSchedule,
    RaderUploadSchedule, RadixRegisterSchedule, StockhamAxisBlockSchedule,
    StockhamUploadAxisContext, StockhamUploadSchedule, has_specialized_gpu_scheduler_policy,
    plan_gpu_axis0_composite_direct_rader_threads_for_prime_multiplicities,
    plan_gpu_axis0_composite_direct_rader_threads_for_prime_multiplicities_with_max_batch_coalesced,
    plan_gpu_axis0_direct_rader_four_step_default_block_from_shape_for_precision,
    plan_gpu_axis0_direct_rader_four_step_grouped_block_from_shape_for_precision,
    plan_gpu_axis0_fft_rader_batch_block_with_grouped_batch_for_precision,
    plan_gpu_axis0_four_step_default_block_from_shape_for_precision,
    plan_gpu_axis0_four_step_grouped_block_from_shape,
    plan_gpu_axis0_four_step_grouped_block_from_shape_for_precision,
    plan_gpu_axis0_four_step_upload_block_with_grouped_batch_for_precision,
    plan_gpu_axis0_mixed_direct_multi_fft_rader_threads,
    plan_gpu_axis0_mixed_direct_multi_fft_rader_threads_with_pass_context,
    plan_gpu_axis0_mixed_direct_multi_fft_rader_threads_with_tuning,
    plan_gpu_axis0_multi_fft_rader_threads_for_prime_multiplicities,
    plan_gpu_axis0_multi_fft_rader_threads_for_prime_multiplicities_with_tuning,
    plan_gpu_axis0_multi_fft_rader_threads_with_pass_context_and_tuning,
    plan_gpu_axis0_single_upload_block_with_grouped_batch, plan_gpu_force_rader_two_upload,
    plan_gpu_other_axis_composite_rader_batch_block_for_precision,
    plan_gpu_other_axis_direct_rader_batch_block_for_precision,
    plan_gpu_other_axis_fft_rader_batch_block_for_precision,
    plan_gpu_other_axis_four_step_block_from_shape_for_precision,
    plan_gpu_other_axis_four_step_upload_block_with_grouped_batch_for_precision,
    plan_gpu_other_axis_single_upload_block_with_grouped_batch_for_precision,
    plan_gpu_power_of_two_radix_registers, plan_gpu_rader_upload_split_with_axis_context,
    plan_gpu_small_mixed_radix_registers, plan_gpu_smooth_stockham_uploads_for_batches,
    plan_gpu_smooth_stockham_uploads_for_batches_with_axis_context,
};
use crate::zero_pad_ir::{ZeroPadPassIr, execute_zero_pad_pass};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CooleyTukeyPassOperation {
    PackRightInput,
    TwiddleTranspose,
    ScatterOutput,
}

/// Optional transform applied while the root Cooley-Tukey pack pass reads its
/// external input. These modifiers deliberately live on the reshape boundary
/// rather than on child FFTs, so recursive convolution trees can reuse the same
/// generator-order and LUT fusion semantics as a single Stockham root.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CooleyTukeyInputModifier {
    #[default]
    None,
    RaderGeneratorReverse(RaderGeneratorMapping),
    MultiplyLookupTable,
    /// Read this recursive upload directly from natural full-axis input using the
    /// same two-upload Four-step lane mapping as `StockhamIoMapping::FourStepRight`.
    FourStepRight(FourStepMapping),
    /// Read upload 2 of a three-upload Four-step plan directly from natural input.
    FourStepThreeUpload2(ThreeUploadFourStepMapping),
    /// Pack a full real external input directly into the recursive half-size FFT root.
    RealEvenPack(RealEvenPackMapping),
    /// Reconstruct the packed inverse half-size spectrum directly at the recursive root.
    RealEvenInversePreprocess(RealEvenInversePreprocessMapping),
}

/// Optional transform applied by the root Cooley-Tukey scatter pass when writing
/// the external output of a recursive convolution tree.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CooleyTukeyOutputModifier {
    #[default]
    None,
    RaderScatter(RaderScatterMapping),
    /// Write the high/right upload with the two-upload Four-step twiddle+transpose.
    FourStepRight(FourStepMapping),
    /// Scatter the low/left upload directly to natural full-axis frequency order.
    FourStepLeft(FourStepMapping),
    /// Write upload 2 with the full-period three-upload Four-step twiddle+transpose.
    FourStepThreeUpload2(ThreeUploadFourStepMapping),
    /// Write upload 1 with the A*B-period twiddle+transpose.
    FourStepThreeUpload1(ThreeUploadFourStepMapping),
    /// Scatter upload 0 directly to natural full-axis frequency order.
    FourStepThreeUpload0(ThreeUploadFourStepMapping),
    /// Reconstruct the compact Hermitian R2C spectrum while scattering the completed
    /// recursive half-size FFT. The scatter reads the already-materialized `Z[k]` and
    /// `Z[M-k]`, so no same-dispatch auxiliary buffer or workgroup barrier is required.
    RealEvenPostprocess(RealEvenPostprocessMapping),
    /// Scatter each inverse half-size complex result directly to even/odd real slots.
    RealEvenUnpack(RealEvenUnpackMapping),
}

#[derive(Debug, Clone, Copy)]
struct CooleyTukeyPassShape {
    scalar: ScalarType,
    direction: Direction,
    logical_len: usize,
    left_len: usize,
    right_len: usize,
    batch_count: usize,
    device: DeviceProfile,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CooleyTukeyPassIr {
    pub name: String,
    pub scalar: ScalarType,
    /// Storage scalar of binding 0. It normally equals `scalar`; a narrower value is
    /// only valid on the caller-visible root `PackRightInput` boundary.
    pub input_storage_scalar: ScalarType,
    /// Storage scalar of binding 1. It normally equals `scalar`; a narrower value is
    /// only valid on the caller-visible root `ScatterOutput` boundary.
    pub output_storage_scalar: ScalarType,
    /// Storage scalar for the original prime input consumed by fused Rader scatter.
    /// This is independent from output storage for one-sided zero-padding boundaries.
    pub auxiliary_storage_scalar: ScalarType,
    pub direction: Direction,
    pub logical_len: usize,
    pub left_len: usize,
    pub right_len: usize,
    pub batch_count: usize,
    pub workgroup_size: WorkgroupSize,
    pub dispatch: DispatchGeometry,
    /// Optional grouped parent-transform ownership for a Four-step component root.
    /// Child FFTs keep their independent batch schedule; this only groups the
    /// reshape/twiddle boundary passes over the same batch-major scratch layout.
    pub axis_batch_block: Option<StockhamAxisBlockSchedule>,
    pub operation: CooleyTukeyPassOperation,
    pub input_modifier: CooleyTukeyInputModifier,
    pub output_modifier: CooleyTukeyOutputModifier,
}

impl CooleyTukeyPassIr {
    fn new(
        name: String,
        shape: CooleyTukeyPassShape,
        operation: CooleyTukeyPassOperation,
    ) -> Result<Self> {
        if shape.device.max_threads_per_block == 0 {
            return Err(VkFftError::ResourceLimitExceeded {
                resource: "threads per workgroup",
                required: 1,
                available: 0,
            });
        }
        let local_size = shape
            .logical_len
            .min(shape.device.max_threads_per_block)
            .max(1);
        let pass = Self {
            name,
            scalar: shape.scalar,
            input_storage_scalar: shape.scalar,
            output_storage_scalar: shape.scalar,
            auxiliary_storage_scalar: shape.scalar,
            direction: shape.direction,
            logical_len: shape.logical_len,
            left_len: shape.left_len,
            right_len: shape.right_len,
            batch_count: shape.batch_count,
            workgroup_size: WorkgroupSize {
                x: u32::try_from(local_size).map_err(|_| VkFftError::ValueOutOfRange {
                    field: "recursive Cooley-Tukey workgroup size",
                })?,
                y: 1,
                z: 1,
            },
            dispatch: DispatchGeometry {
                x: u32::try_from(shape.batch_count).map_err(|_| VkFftError::ValueOutOfRange {
                    field: "recursive Cooley-Tukey dispatch workgroup count",
                })?,
                y: 1,
                z: 1,
            },
            axis_batch_block: None,
            operation,
            input_modifier: CooleyTukeyInputModifier::None,
            output_modifier: CooleyTukeyOutputModifier::None,
        };
        pass.validate()?;
        Ok(pass)
    }

    fn with_axis0_batch_block(
        mut self,
        block: StockhamAxisBlockSchedule,
        device: DeviceProfile,
    ) -> Result<Self> {
        block.validate(self.batch_count, device)?;
        self.workgroup_size = WorkgroupSize {
            x: u32::try_from(block.local_size_x).map_err(|_| VkFftError::ValueOutOfRange {
                field: "recursive Cooley-Tukey grouped workgroup x size",
            })?,
            y: u32::try_from(block.local_size_y).map_err(|_| VkFftError::ValueOutOfRange {
                field: "recursive Cooley-Tukey grouped workgroup y size",
            })?,
            z: 1,
        };
        self.dispatch = DispatchGeometry {
            x: u32::try_from(self.batch_count.div_ceil(block.grouped_batch)).map_err(|_| {
                VkFftError::ValueOutOfRange {
                    field: "recursive Cooley-Tukey grouped dispatch count",
                }
            })?,
            y: 1,
            z: 1,
        };
        self.axis_batch_block = Some(block);
        self.validate()?;
        Ok(self)
    }

    pub fn validate(&self) -> Result<()> {
        if self.left_len < 2
            || self.right_len < 2
            || self.batch_count == 0
            || self.left_len.checked_mul(self.right_len) != Some(self.logical_len)
        {
            return Err(VkFftError::InvalidKernelIr(
                "recursive Cooley-Tukey pass dimensions are inconsistent",
            ));
        }
        if self.workgroup_size.x == 0
            || self.workgroup_size.y == 0
            || self.workgroup_size.z == 0
            || self.dispatch.y != 1
            || self.dispatch.z != 1
        {
            return Err(VkFftError::InvalidKernelIr(
                "recursive Cooley-Tukey pass launch metadata is inconsistent",
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
            {
                return Err(VkFftError::InvalidKernelIr(
                    "recursive Cooley-Tukey grouped launch metadata is inconsistent",
                ));
            }
        } else if self.dispatch.x as usize != self.batch_count || self.workgroup_size.y != 1 {
            return Err(VkFftError::InvalidKernelIr(
                "recursive Cooley-Tukey ungrouped launch metadata is inconsistent",
            ));
        }
        if self.scalar == ScalarType::F16 {
            return Err(VkFftError::InvalidKernelIr(
                "binary16 cannot be used as Cooley-Tukey compute scalar",
            ));
        }
        if self.input_storage_scalar != self.scalar
            && (!supported_external_storage_pair(self.scalar, self.input_storage_scalar)
                || self.operation != CooleyTukeyPassOperation::PackRightInput
                || !matches!(
                    self.input_modifier,
                    CooleyTukeyInputModifier::None
                        | CooleyTukeyInputModifier::RaderGeneratorReverse(_)
                        | CooleyTukeyInputModifier::FourStepRight(_)
                        | CooleyTukeyInputModifier::FourStepThreeUpload2(_)
                ))
        {
            return Err(VkFftError::InvalidKernelIr(
                "narrow Cooley-Tukey input storage requires a caller-visible pack boundary",
            ));
        }
        if self.output_storage_scalar != self.scalar
            && (!supported_external_storage_pair(self.scalar, self.output_storage_scalar)
                || self.operation != CooleyTukeyPassOperation::ScatterOutput
                || !matches!(
                    self.output_modifier,
                    CooleyTukeyOutputModifier::None
                        | CooleyTukeyOutputModifier::RaderScatter(_)
                        | CooleyTukeyOutputModifier::FourStepRight(_)
                        | CooleyTukeyOutputModifier::FourStepLeft(_)
                        | CooleyTukeyOutputModifier::FourStepThreeUpload2(_)
                        | CooleyTukeyOutputModifier::FourStepThreeUpload1(_)
                        | CooleyTukeyOutputModifier::FourStepThreeUpload0(_)
                ))
        {
            return Err(VkFftError::InvalidKernelIr(
                "narrow Cooley-Tukey output storage requires a caller-visible scatter boundary",
            ));
        }
        if self.auxiliary_storage_scalar != self.scalar
            && (!supported_external_storage_pair(self.scalar, self.auxiliary_storage_scalar)
                || self.operation != CooleyTukeyPassOperation::ScatterOutput
                || !matches!(
                    self.output_modifier,
                    CooleyTukeyOutputModifier::RaderScatter(_)
                ))
        {
            return Err(VkFftError::InvalidKernelIr(
                "narrow Cooley-Tukey auxiliary storage requires a fused Rader scatter boundary",
            ));
        }
        if !matches!(
            self.output_modifier,
            CooleyTukeyOutputModifier::RaderScatter(_)
        ) && self.auxiliary_storage_scalar != self.scalar
        {
            return Err(VkFftError::InvalidKernelIr(
                "non-Rader Cooley-Tukey pass cannot carry auxiliary storage",
            ));
        }
        match self.input_modifier {
            CooleyTukeyInputModifier::None => {}
            CooleyTukeyInputModifier::RaderGeneratorReverse(mapping) => {
                if self.operation != CooleyTukeyPassOperation::PackRightInput {
                    return Err(VkFftError::InvalidKernelIr(
                        "Rader generator input fusion requires a Cooley-Tukey pack pass",
                    ));
                }
                mapping.validate(self.logical_len)?;
            }
            CooleyTukeyInputModifier::MultiplyLookupTable => {
                if self.operation != CooleyTukeyPassOperation::PackRightInput {
                    return Err(VkFftError::InvalidKernelIr(
                        "lookup-table input fusion requires a Cooley-Tukey pack pass",
                    ));
                }
            }
            CooleyTukeyInputModifier::FourStepRight(mapping) => {
                if self.operation != CooleyTukeyPassOperation::PackRightInput {
                    return Err(VkFftError::InvalidKernelIr(
                        "two-upload Four-step input mapping requires a Cooley-Tukey pack pass",
                    ));
                }
                validate_cooley_four_step_mapping(self, mapping, 1)?;
            }
            CooleyTukeyInputModifier::FourStepThreeUpload2(mapping) => {
                if self.operation != CooleyTukeyPassOperation::PackRightInput {
                    return Err(VkFftError::InvalidKernelIr(
                        "three-upload Four-step input mapping requires a Cooley-Tukey pack pass",
                    ));
                }
                validate_cooley_three_upload_mapping(self, mapping, 2)?;
            }
            CooleyTukeyInputModifier::RealEvenPack(mapping) => {
                if self.operation != CooleyTukeyPassOperation::PackRightInput {
                    return Err(VkFftError::InvalidKernelIr(
                        "even-real pack fusion requires a Cooley-Tukey pack pass",
                    ));
                }
                mapping.validate(self.logical_len)?;
            }
            CooleyTukeyInputModifier::RealEvenInversePreprocess(mapping) => {
                if self.operation != CooleyTukeyPassOperation::PackRightInput {
                    return Err(VkFftError::InvalidKernelIr(
                        "even-real inverse preprocess fusion requires a Cooley-Tukey pack pass",
                    ));
                }
                mapping.validate(self.logical_len)?;
            }
        }
        match self.output_modifier {
            CooleyTukeyOutputModifier::None => {}
            CooleyTukeyOutputModifier::RaderScatter(mapping) => {
                if self.operation != CooleyTukeyPassOperation::ScatterOutput {
                    return Err(VkFftError::InvalidKernelIr(
                        "Rader scatter fusion requires a Cooley-Tukey scatter pass",
                    ));
                }
                mapping.validate(self.logical_len)?;
            }
            CooleyTukeyOutputModifier::FourStepRight(mapping) => {
                if self.operation != CooleyTukeyPassOperation::ScatterOutput {
                    return Err(VkFftError::InvalidKernelIr(
                        "two-upload Four-step output mapping requires a Cooley-Tukey scatter pass",
                    ));
                }
                validate_cooley_four_step_mapping(self, mapping, 1)?;
            }
            CooleyTukeyOutputModifier::FourStepLeft(mapping) => {
                if self.operation != CooleyTukeyPassOperation::ScatterOutput {
                    return Err(VkFftError::InvalidKernelIr(
                        "two-upload Four-step output mapping requires a Cooley-Tukey scatter pass",
                    ));
                }
                validate_cooley_four_step_mapping(self, mapping, 0)?;
            }
            CooleyTukeyOutputModifier::FourStepThreeUpload2(mapping) => {
                validate_cooley_three_upload_mapping(self, mapping, 2)?;
            }
            CooleyTukeyOutputModifier::FourStepThreeUpload1(mapping) => {
                validate_cooley_three_upload_mapping(self, mapping, 1)?;
            }
            CooleyTukeyOutputModifier::FourStepThreeUpload0(mapping) => {
                validate_cooley_three_upload_mapping(self, mapping, 0)?;
            }
            CooleyTukeyOutputModifier::RealEvenPostprocess(mapping) => {
                if self.operation != CooleyTukeyPassOperation::ScatterOutput
                    || self.direction != Direction::Forward
                {
                    return Err(VkFftError::InvalidKernelIr(
                        "even-real postprocess fusion requires a forward Cooley-Tukey scatter pass",
                    ));
                }
                mapping.validate(self.logical_len)?;
            }
            CooleyTukeyOutputModifier::RealEvenUnpack(mapping) => {
                if self.operation != CooleyTukeyPassOperation::ScatterOutput {
                    return Err(VkFftError::InvalidKernelIr(
                        "even-real unpack fusion requires a Cooley-Tukey scatter pass",
                    ));
                }
                mapping.validate(self.logical_len)?;
            }
        }
        Ok(())
    }

    pub(crate) fn with_external_input_storage(mut self, storage: ScalarType) -> Result<Self> {
        if storage == self.scalar {
            return Ok(self);
        }
        if self.operation != CooleyTukeyPassOperation::PackRightInput
            || !matches!(
                self.input_modifier,
                CooleyTukeyInputModifier::None
                    | CooleyTukeyInputModifier::RaderGeneratorReverse(_)
                    | CooleyTukeyInputModifier::FourStepRight(_)
                    | CooleyTukeyInputModifier::FourStepThreeUpload2(_)
            )
            || !supported_external_storage_pair(self.scalar, storage)
        {
            return Err(VkFftError::UnsupportedPrecision {
                backend: "recursive Cooley-Tukey input boundary",
                precision: "mixed storage on unsupported/non-pack boundary",
            });
        }
        self.input_storage_scalar = storage;
        self.validate()?;
        Ok(self)
    }

    pub(crate) fn with_external_output_storage(mut self, storage: ScalarType) -> Result<Self> {
        if storage == self.scalar {
            return Ok(self);
        }
        if self.operation != CooleyTukeyPassOperation::ScatterOutput
            || !matches!(
                self.output_modifier,
                CooleyTukeyOutputModifier::None
                    | CooleyTukeyOutputModifier::RaderScatter(_)
                    | CooleyTukeyOutputModifier::FourStepRight(_)
                    | CooleyTukeyOutputModifier::FourStepLeft(_)
                    | CooleyTukeyOutputModifier::FourStepThreeUpload2(_)
                    | CooleyTukeyOutputModifier::FourStepThreeUpload1(_)
                    | CooleyTukeyOutputModifier::FourStepThreeUpload0(_)
            )
            || !supported_external_storage_pair(self.scalar, storage)
        {
            return Err(VkFftError::UnsupportedPrecision {
                backend: "recursive Cooley-Tukey output boundary",
                precision: "mixed storage on unsupported/non-scatter boundary",
            });
        }
        self.output_storage_scalar = storage;
        self.validate()?;
        Ok(self)
    }

    pub(crate) fn with_rader_auxiliary_storage(mut self, storage: ScalarType) -> Result<Self> {
        if storage != self.scalar && !supported_external_storage_pair(self.scalar, storage) {
            return Err(VkFftError::UnsupportedPrecision {
                backend: "recursive Cooley-Tukey Rader auxiliary boundary",
                precision: "unsupported compute/storage scalar pair",
            });
        }
        if self.operation != CooleyTukeyPassOperation::ScatterOutput
            || !matches!(
                self.output_modifier,
                CooleyTukeyOutputModifier::RaderScatter(_)
            )
        {
            return Err(VkFftError::UnsupportedPrecision {
                backend: "recursive Cooley-Tukey Rader auxiliary boundary",
                precision: "auxiliary storage requires fused Rader scatter",
            });
        }
        self.auxiliary_storage_scalar = storage;
        self.validate()?;
        Ok(self)
    }

    pub(crate) fn with_rader_generator_input(
        mut self,
        mapping: RaderGeneratorMapping,
    ) -> Result<Self> {
        if self.input_modifier != CooleyTukeyInputModifier::None {
            return Err(VkFftError::UnsupportedKernelPath(
                "recursive Cooley-Tukey input already carries a modifier",
            ));
        }
        self.input_modifier = CooleyTukeyInputModifier::RaderGeneratorReverse(mapping);
        self.name.push_str("_rader_generator_input");
        self.validate()?;
        Ok(self)
    }

    pub(crate) fn with_lookup_table_input_multiply(mut self) -> Result<Self> {
        if self.input_modifier != CooleyTukeyInputModifier::None {
            return Err(VkFftError::UnsupportedKernelPath(
                "recursive Cooley-Tukey input already carries a modifier",
            ));
        }
        self.input_modifier = CooleyTukeyInputModifier::MultiplyLookupTable;
        self.name.push_str("_input_mul_lut");
        self.validate()?;
        Ok(self)
    }

    pub(crate) fn with_real_even_pack_input(
        mut self,
        mapping: RealEvenPackMapping,
    ) -> Result<Self> {
        if self.input_modifier != CooleyTukeyInputModifier::None {
            return Err(VkFftError::UnsupportedKernelPath(
                "recursive Cooley-Tukey input already carries a modifier",
            ));
        }
        self.input_modifier = CooleyTukeyInputModifier::RealEvenPack(mapping);
        self.name.push_str("_real_even_pack");
        self.validate()?;
        Ok(self)
    }

    pub(crate) fn with_real_even_inverse_preprocess_input(
        mut self,
        mapping: RealEvenInversePreprocessMapping,
    ) -> Result<Self> {
        if self.input_modifier != CooleyTukeyInputModifier::None {
            return Err(VkFftError::UnsupportedKernelPath(
                "recursive Cooley-Tukey input already carries a modifier",
            ));
        }
        self.input_modifier = CooleyTukeyInputModifier::RealEvenInversePreprocess(mapping);
        self.name.push_str("_real_even_inverse_preprocess");
        self.validate()?;
        Ok(self)
    }

    pub(crate) fn with_real_even_postprocess_output(
        mut self,
        mapping: RealEvenPostprocessMapping,
    ) -> Result<Self> {
        if self.output_modifier != CooleyTukeyOutputModifier::None {
            return Err(VkFftError::UnsupportedKernelPath(
                "recursive Cooley-Tukey output already carries a modifier",
            ));
        }
        if self.direction != Direction::Forward {
            return Err(VkFftError::UnsupportedKernelPath(
                "recursive even-real postprocess fusion requires a forward transform",
            ));
        }
        self.output_modifier = CooleyTukeyOutputModifier::RealEvenPostprocess(mapping);
        self.name.push_str("_real_even_postprocess");
        self.validate()?;
        Ok(self)
    }

    pub(crate) fn with_real_even_unpack_output(
        mut self,
        mapping: RealEvenUnpackMapping,
    ) -> Result<Self> {
        if self.output_modifier != CooleyTukeyOutputModifier::None {
            return Err(VkFftError::UnsupportedKernelPath(
                "recursive Cooley-Tukey output already carries a modifier",
            ));
        }
        self.output_modifier = CooleyTukeyOutputModifier::RealEvenUnpack(mapping);
        self.name.push_str("_real_even_unpack");
        self.validate()?;
        Ok(self)
    }

    pub(crate) fn with_rader_scatter_output(
        mut self,
        mapping: RaderScatterMapping,
    ) -> Result<Self> {
        if self.output_modifier != CooleyTukeyOutputModifier::None {
            return Err(VkFftError::UnsupportedKernelPath(
                "recursive Cooley-Tukey output already carries a modifier",
            ));
        }
        self.output_modifier = CooleyTukeyOutputModifier::RaderScatter(mapping);
        self.name.push_str("_rader_scatter");
        self.validate()?;
        Ok(self)
    }

    fn with_four_step_input_modifier(mut self, modifier: CooleyTukeyInputModifier) -> Result<Self> {
        if self.operation != CooleyTukeyPassOperation::PackRightInput
            || self.input_modifier != CooleyTukeyInputModifier::None
            || !matches!(
                modifier,
                CooleyTukeyInputModifier::FourStepRight(_)
                    | CooleyTukeyInputModifier::FourStepThreeUpload2(_)
            )
        {
            return Err(VkFftError::UnsupportedKernelPath(
                "recursive Four-step input mapping requires an unmodified root pack",
            ));
        }
        self.input_modifier = modifier;
        self.name.push_str("_four_step_input");
        self.validate()?;
        Ok(self)
    }

    fn with_four_step_output_modifier(
        mut self,
        modifier: CooleyTukeyOutputModifier,
    ) -> Result<Self> {
        if self.operation != CooleyTukeyPassOperation::ScatterOutput
            || self.output_modifier != CooleyTukeyOutputModifier::None
            || !matches!(
                modifier,
                CooleyTukeyOutputModifier::FourStepRight(_)
                    | CooleyTukeyOutputModifier::FourStepLeft(_)
                    | CooleyTukeyOutputModifier::FourStepThreeUpload2(_)
                    | CooleyTukeyOutputModifier::FourStepThreeUpload1(_)
                    | CooleyTukeyOutputModifier::FourStepThreeUpload0(_)
            )
        {
            return Err(VkFftError::UnsupportedKernelPath(
                "recursive Four-step output mapping requires an unmodified root scatter",
            ));
        }
        self.output_modifier = modifier;
        self.name.push_str("_four_step_output");
        self.validate()?;
        Ok(self)
    }
}

fn validate_cooley_four_step_mapping(
    pass: &CooleyTukeyPassIr,
    mapping: FourStepMapping,
    upload_id: usize,
) -> Result<()> {
    mapping.validate()?;
    let (expected_len, other) = match upload_id {
        1 => (mapping.right_len, mapping.left_len),
        0 => (mapping.left_len, mapping.right_len),
        _ => {
            return Err(VkFftError::InvalidKernelIr(
                "recursive two-upload Four-step boundary upload id is invalid",
            ));
        }
    };
    let expected_batches =
        mapping
            .outer_batch_count
            .checked_mul(other)
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "recursive Four-step boundary batch count",
            })?;
    if pass.logical_len != expected_len || pass.batch_count != expected_batches {
        return Err(VkFftError::InvalidKernelIr(
            "recursive two-upload Four-step boundary geometry is inconsistent",
        ));
    }
    Ok(())
}

fn validate_cooley_three_upload_mapping(
    pass: &CooleyTukeyPassIr,
    mapping: ThreeUploadFourStepMapping,
    upload_id: usize,
) -> Result<()> {
    mapping.validate()?;
    let [a, b, c] = mapping.axis_split;
    let (expected_len, factor0, factor1) = match upload_id {
        2 => (c, a, b),
        1 => (b, c, a),
        0 => (a, c, b),
        _ => {
            return Err(VkFftError::InvalidKernelIr(
                "recursive three-upload Four-step boundary upload id is invalid",
            ));
        }
    };
    let expected_batches = mapping
        .outer_batch_count
        .checked_mul(factor0)
        .and_then(|value| value.checked_mul(factor1))
        .ok_or(VkFftError::ArithmeticOverflow {
            operation: "recursive three-upload Four-step boundary batch count",
        })?;
    if pass.logical_len != expected_len || pass.batch_count != expected_batches {
        return Err(VkFftError::InvalidKernelIr(
            "recursive three-upload Four-step boundary geometry is inconsistent",
        ));
    }
    Ok(())
}

fn supported_external_storage_pair(compute: ScalarType, storage: ScalarType) -> bool {
    matches!(
        (compute, storage),
        (ScalarType::F32, ScalarType::F16) | (ScalarType::F64, ScalarType::F32)
    )
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FourStepInputLayout {
    NaturalStrided,
    TransposedContiguous,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FourStepOutputLayout {
    TwiddleTransposed,
    NaturalFrequency,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FourStepTwiddlePlacement {
    None,
    Write,
}

/// One scheduler upload in upstream execution order. VkFFT executes axis-0 uploads
/// from the highest upload id down to zero; `stage_start_size` is the product of
/// all lower-id `axisSplit` factors and controls the Four-step twiddle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FourStepUploadIr {
    pub axis_upload_id: usize,
    pub fft_len: usize,
    pub stage_start_size: usize,
    pub transform_count: usize,
    pub axis_block: Option<StockhamAxisBlockSchedule>,
    pub input_layout: FourStepInputLayout,
    pub output_layout: FourStepOutputLayout,
    pub twiddle: FourStepTwiddlePlacement,
}

/// Typed executable slice of VkFFT's `reorderFourStep` orchestration. Uploads are
/// stored in actual axis-0 launch order (highest upload id down to zero). The
/// covered forms are exactly two and exactly three Stockham uploads; the generic
/// recursive Cooley-Tukey tree remains the fallback for other shapes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FourStepPlanIr {
    pub logical_len: usize,
    pub batch_count: usize,
    pub reorder_four_step: bool,
    pub uploads: Vec<FourStepUploadIr>,
}

impl FourStepPlanIr {
    pub fn validate(&self) -> Result<()> {
        if self.logical_len == 0 || self.batch_count == 0 || !self.reorder_four_step {
            return Err(VkFftError::InvalidKernelIr(
                "Four-step plan metadata is incomplete",
            ));
        }
        match self.uploads.as_slice() {
            [first, second] => self.validate_two_upload(*first, *second),
            [first, second, third] => self.validate_three_upload(*first, *second, *third),
            _ => Err(VkFftError::InvalidKernelIr(
                "Four-step plan currently requires exactly two or three uploads",
            )),
        }
    }

    fn validate_two_upload(&self, first: FourStepUploadIr, second: FourStepUploadIr) -> Result<()> {
        let first_batches =
            self.batch_count
                .checked_mul(second.fft_len)
                .ok_or(VkFftError::ArithmeticOverflow {
                    operation: "Four-step first upload transform count",
                })?;
        let second_batches =
            self.batch_count
                .checked_mul(first.fft_len)
                .ok_or(VkFftError::ArithmeticOverflow {
                    operation: "Four-step second upload transform count",
                })?;
        if first.axis_upload_id != 1
            || second.axis_upload_id != 0
            || first.stage_start_size != second.fft_len
            || second.stage_start_size != 1
            || first.fft_len.checked_mul(second.fft_len) != Some(self.logical_len)
            || first.input_layout != FourStepInputLayout::NaturalStrided
            || first.output_layout != FourStepOutputLayout::TwiddleTransposed
            || first.twiddle != FourStepTwiddlePlacement::Write
            || second.input_layout != FourStepInputLayout::TransposedContiguous
            || second.output_layout != FourStepOutputLayout::NaturalFrequency
            || second.twiddle != FourStepTwiddlePlacement::None
            || first.transform_count != first_batches
            || second.transform_count != second_batches
        {
            return Err(VkFftError::InvalidKernelIr(
                "two-upload Four-step upload metadata is inconsistent",
            ));
        }
        Ok(())
    }

    fn validate_three_upload(
        &self,
        first: FourStepUploadIr,
        second: FourStepUploadIr,
        third: FourStepUploadIr,
    ) -> Result<()> {
        let ab =
            third
                .fft_len
                .checked_mul(second.fft_len)
                .ok_or(VkFftError::ArithmeticOverflow {
                    operation: "three-upload Four-step A*B product",
                })?;
        let product = ab
            .checked_mul(first.fft_len)
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "three-upload Four-step factor product",
            })?;
        let first_batches =
            self.batch_count
                .checked_mul(ab)
                .ok_or(VkFftError::ArithmeticOverflow {
                    operation: "three-upload Four-step upload-2 transform count",
                })?;
        let second_batches = self
            .batch_count
            .checked_mul(first.fft_len)
            .and_then(|value| value.checked_mul(third.fft_len))
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "three-upload Four-step upload-1 transform count",
            })?;
        let third_batches = self
            .batch_count
            .checked_mul(first.fft_len)
            .and_then(|value| value.checked_mul(second.fft_len))
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "three-upload Four-step upload-0 transform count",
            })?;
        if first.axis_upload_id != 2
            || second.axis_upload_id != 1
            || third.axis_upload_id != 0
            || first.stage_start_size != ab
            || second.stage_start_size != third.fft_len
            || third.stage_start_size != 1
            || product != self.logical_len
            || first.input_layout != FourStepInputLayout::NaturalStrided
            || first.output_layout != FourStepOutputLayout::TwiddleTransposed
            || first.twiddle != FourStepTwiddlePlacement::Write
            || second.input_layout != FourStepInputLayout::TransposedContiguous
            || second.output_layout != FourStepOutputLayout::TwiddleTransposed
            || second.twiddle != FourStepTwiddlePlacement::Write
            || third.input_layout != FourStepInputLayout::TransposedContiguous
            || third.output_layout != FourStepOutputLayout::NaturalFrequency
            || third.twiddle != FourStepTwiddlePlacement::None
            || first.transform_count != first_batches
            || second.transform_count != second_batches
            || third.transform_count != third_batches
        {
            return Err(VkFftError::InvalidKernelIr(
                "three-upload Four-step upload metadata is inconsistent",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct RecursiveRaderScheduleSummary {
    pub fft_rader_nodes: usize,
    pub scheduled_fft_rader_nodes: usize,
    pub transposed_fft_rader_nodes: usize,
    pub max_threads_per_workgroup: usize,
    pub max_container_fft_num: usize,
}

impl RecursiveRaderScheduleSummary {
    fn merge(&mut self, other: Self) {
        self.fft_rader_nodes += other.fft_rader_nodes;
        self.scheduled_fft_rader_nodes += other.scheduled_fft_rader_nodes;
        self.transposed_fft_rader_nodes += other.transposed_fft_rader_nodes;
        self.max_threads_per_workgroup = self
            .max_threads_per_workgroup
            .max(other.max_threads_per_workgroup);
        self.max_container_fft_num = self.max_container_fft_num.max(other.max_container_fft_num);
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum RecursiveFftNodeIr {
    Stockham(Box<KernelIr>),
    DirectRader(Box<RaderDirectIr>),
    FftRader(Box<RaderFftPipelineIr>),
    CooleyTukey(Box<RecursiveCooleyTukeyIr>),
}

impl RecursiveFftNodeIr {
    pub fn logical_len(&self) -> usize {
        match self {
            Self::Stockham(ir) => ir.sequence_len,
            Self::DirectRader(ir) => ir.prime,
            Self::FftRader(ir) => ir.prime,
            Self::CooleyTukey(ir) => ir.logical_len,
        }
    }

    pub fn batch_count(&self) -> usize {
        match self {
            Self::Stockham(ir) => ir.batch_count,
            Self::DirectRader(ir) => ir.batch_count,
            Self::FftRader(ir) => ir.batch_count,
            Self::CooleyTukey(ir) => ir.batch_count,
        }
    }

    pub fn scalar(&self) -> ScalarType {
        match self {
            Self::Stockham(ir) => ir.scalar,
            Self::DirectRader(ir) => ir.scalar,
            Self::FftRader(ir) => ir.scalar,
            Self::CooleyTukey(ir) => ir.scalar,
        }
    }

    pub fn direction(&self) -> Direction {
        match self {
            Self::Stockham(ir) => ir.direction,
            Self::DirectRader(ir) => ir.direction,
            Self::FftRader(ir) => ir.direction,
            Self::CooleyTukey(ir) => ir.direction,
        }
    }

    pub fn rader_schedule_summary(&self) -> RecursiveRaderScheduleSummary {
        let mut summary = RecursiveRaderScheduleSummary::default();
        match self {
            Self::Stockham(_) | Self::DirectRader(_) => {}
            Self::FftRader(ir) => {
                summary.fft_rader_nodes = 1;
                if let Some(schedule) = ir.internal_register_schedule.as_ref() {
                    summary.scheduled_fft_rader_nodes = 1;
                    summary.transposed_fft_rader_nodes =
                        usize::from(schedule.rader_transpose.is_some());
                    summary.max_threads_per_workgroup = schedule.execution_threads_per_workgroup;
                    summary.max_container_fft_num = schedule.container_fft_num;
                }
                if let Some(forward) = ir.forward_recursive() {
                    summary.merge(forward.root.rader_schedule_summary());
                }
            }
            Self::CooleyTukey(ir) => {
                summary.merge(ir.left.rader_schedule_summary());
                summary.merge(ir.right.rader_schedule_summary());
            }
        }
        summary
    }

    fn collect_rader_prime_multiplicities(
        &self,
        direct: &mut Vec<(usize, usize)>,
        fft: &mut Vec<(usize, usize)>,
    ) {
        fn add_prime(primes: &mut Vec<(usize, usize)>, prime: usize) {
            if let Some((_, multiplicity)) = primes.iter_mut().find(|(value, _)| *value == prime) {
                *multiplicity += 1;
            } else {
                primes.push((prime, 1));
            }
        }
        match self {
            Self::Stockham(_) => {}
            Self::DirectRader(ir) => add_prime(direct, ir.prime),
            Self::FftRader(ir) => add_prime(fft, ir.prime),
            Self::CooleyTukey(ir) => {
                ir.left.collect_rader_prime_multiplicities(direct, fft);
                ir.right.collect_rader_prime_multiplicities(direct, fft);
            }
        }
    }

    fn contains_rader(&self) -> bool {
        match self {
            Self::Stockham(_) => false,
            Self::DirectRader(_) | Self::FftRader(_) => true,
            Self::CooleyTukey(ir) => ir.left.contains_rader() || ir.right.contains_rader(),
        }
    }

    fn collect_fused_fft_rader_static_resource_reports(
        &self,
        reports: &mut Vec<FusedFftRaderStaticResourceReport>,
    ) -> Result<()> {
        let Self::CooleyTukey(parent) = self else {
            return Ok(());
        };
        if let Some(fused) = parent.fused_small_fft_rader_stockham()?
            && let Some(report) = fused.static_resource_report()?
        {
            reports.push(report);
            return Ok(());
        }
        parent
            .left
            .collect_fused_fft_rader_static_resource_reports(reports)?;
        parent
            .right
            .collect_fused_fft_rader_static_resource_reports(reports)
    }

    pub fn validate(&self) -> Result<()> {
        match self {
            Self::Stockham(ir) => ir.validate(),
            Self::DirectRader(ir) => ir.validate(),
            Self::FftRader(ir) => ir.validate(),
            Self::CooleyTukey(ir) => ir.validate(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct FusedSmallDirectRaderStockhamIr<'a> {
    pub parent: &'a RecursiveCooleyTukeyIr,
    pub stockham: &'a KernelIr,
    pub direct: &'a RaderDirectIr,
    pub axis_block: StockhamAxisBlockSchedule,
}

impl FusedSmallDirectRaderStockhamIr<'_> {
    pub(crate) fn name(self) -> String {
        format!(
            "vkfft_composite_direct_rader_stockham_{}_{}x{}_{:?}",
            self.parent.logical_len,
            self.parent.left_len,
            self.parent.right_len,
            self.parent.direction
        )
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct FusedSmallFftRaderStockhamIr<'a> {
    pub parent: &'a RecursiveCooleyTukeyIr,
    pub stockham: &'a KernelIr,
    pub rader: &'a RaderFftPipelineIr,
    pub forward: &'a KernelIr,
    pub inverse: &'a KernelIr,
    pub axis_block: StockhamAxisBlockSchedule,
}

impl FusedSmallFftRaderStockhamIr<'_> {
    pub(crate) fn name(self) -> String {
        format!(
            "vkfft_composite_fft_rader_stockham_{}_{}x{}_{:?}",
            self.parent.logical_len,
            self.parent.left_len,
            self.parent.right_len,
            self.parent.direction
        )
    }

    fn register_resident_convolution_candidate(self) -> Result<bool> {
        let forward_stages = self.forward.register_stockham_stages()?.unwrap_or_default();
        let inverse_stages = self.inverse.register_stockham_stages()?.unwrap_or_default();
        if forward_stages.len() != 1 || inverse_stages.len() != 1 {
            return Ok(false);
        }
        let forward = forward_stages[0];
        let inverse = inverse_stages[0];
        Ok(forward.radix == self.rader.convolution_len
            && inverse.radix == self.rader.convolution_len
            && forward.stage_size == 1
            && inverse.stage_size == 1
            && forward.virtual_thread_count == 1
            && inverse.virtual_thread_count == 1
            && self.forward.twiddle_lut_len().is_none()
            && self.inverse.twiddle_lut_len().is_none()
            && self.axis_block.threads_per_transform >= self.parent.left_len
            && self
                .forward
                .register_stage_boundaries()?
                .unwrap_or_default()
                .is_empty()
            && self
                .inverse
                .register_stage_boundaries()?
                .unwrap_or_default()
                .is_empty())
    }

    fn two_shared_required_shared_memory_bytes(self) -> Result<usize> {
        self.parent
            .logical_len
            .checked_mul(2)
            .and_then(|value| value.checked_mul(self.axis_block.grouped_batch))
            .and_then(|value| value.checked_mul(self.parent.scalar.complex_bytes()))
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "fused FFT-Rader two-shared memory size",
            })
    }

    fn reduced_shared_residency_required(self) -> Result<bool> {
        Ok(self.two_shared_required_shared_memory_bytes()? > self.rader.device_shared_memory_bytes)
    }

    pub(crate) fn register_resident_convolution(self) -> Result<bool> {
        Ok(self.register_resident_convolution_candidate()?
            && self.reduced_shared_residency_required()?)
    }

    fn single_shared_register_convolution_candidate(self) -> Result<bool> {
        if self.register_resident_convolution_candidate()? {
            return Ok(false);
        }
        let forward_stages = self.forward.register_stockham_stages()?.unwrap_or_default();
        let inverse_stages = self.inverse.register_stockham_stages()?.unwrap_or_default();
        if forward_stages.len() < 2 || forward_stages != inverse_stages {
            return Ok(false);
        }
        if forward_stages.iter().any(|stage| {
            stage.register_boost != 1
                || stage.is_register_boost_stage
                || stage.input != stage.output
        }) {
            return Ok(false);
        }
        let forward_boundaries = self
            .forward
            .register_stage_boundaries()?
            .unwrap_or_default();
        let inverse_boundaries = self
            .inverse
            .register_stage_boundaries()?
            .unwrap_or_default();
        if forward_boundaries.len() + 1 != forward_stages.len()
            || inverse_boundaries.len() + 1 != inverse_stages.len()
            || forward_boundaries.iter().any(|boundary| {
                boundary.residency
                    != crate::kernel_ir::RegisterStageBoundaryResidency::SharedExchangeRequired
            })
            || inverse_boundaries.iter().any(|boundary| {
                boundary.residency
                    != crate::kernel_ir::RegisterStageBoundaryResidency::SharedExchangeRequired
            })
        {
            return Ok(false);
        }
        let max_virtual_threads = forward_stages
            .iter()
            .map(|stage| stage.virtual_thread_count)
            .max()
            .unwrap_or(0);
        let required_parent_lanes = self
            .parent
            .left_len
            .checked_mul(max_virtual_threads)
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "single-shared fused FFT-Rader parent register lanes",
            })?;
        Ok(required_parent_lanes <= self.axis_block.threads_per_transform)
    }

    pub(crate) fn single_shared_register_convolution(self) -> Result<bool> {
        Ok(self.single_shared_register_convolution_candidate()?
            && self.reduced_shared_residency_required()?)
    }

    fn max_logical_register_complex_values_per_invocation(self) -> Result<Option<usize>> {
        if self.register_resident_convolution()? {
            return Ok(Some(self.rader.convolution_len));
        }
        if !self.single_shared_register_convolution()? {
            return Ok(None);
        }
        let forward = self.forward.register_stockham_stages()?.unwrap_or_default();
        let inverse = self.inverse.register_stockham_stages()?.unwrap_or_default();
        Ok(forward
            .iter()
            .chain(&inverse)
            .map(|stage| stage.registers_per_thread)
            .max())
    }

    fn static_resource_report(self) -> Result<Option<FusedFftRaderStaticResourceReport>> {
        let Some(max_logical_register_complex_values_per_invocation) =
            self.max_logical_register_complex_values_per_invocation()?
        else {
            return Ok(None);
        };
        Ok(Some(FusedFftRaderStaticResourceReport {
            pass_name: self.name(),
            required_shared_memory_bytes: self.required_shared_memory_bytes()?,
            workgroup_size: self.parent.pack_right.workgroup_size,
            uniform_barrier_count: self.uniform_barrier_count()?,
            max_logical_register_complex_values_per_invocation,
        }))
    }

    pub(crate) fn shared_stripe_count(self) -> Result<usize> {
        Ok(
            if self.register_resident_convolution()? || self.single_shared_register_convolution()? {
                1
            } else {
                2
            },
        )
    }

    pub(crate) fn uniform_barrier_count(self) -> Result<usize> {
        if self.register_resident_convolution()? {
            return Ok(1);
        }
        let forward_stages = self.forward.register_stockham_stages()?.unwrap_or_default();
        let inverse_stages = self.inverse.register_stockham_stages()?.unwrap_or_default();
        if self.single_shared_register_convolution()? {
            return forward_stages
                .len()
                .checked_sub(1)
                .and_then(|forward_shared_inputs| {
                    forward_shared_inputs
                        .checked_mul(2)
                        .and_then(|value| value.checked_add(1))
                })
                .and_then(|value| {
                    inverse_stages
                        .len()
                        .checked_mul(2)
                        .and_then(|inverse| value.checked_add(inverse))
                })
                .ok_or(VkFftError::ArithmeticOverflow {
                    operation: "single-shared fused FFT-Rader barrier count",
                });
        }
        Ok(forward_stages.len() + inverse_stages.len() + 3)
    }

    pub(crate) fn required_shared_memory_bytes(self) -> Result<usize> {
        let elements_per_transform = if self.single_shared_register_convolution()? {
            self.parent
                .left_len
                .checked_mul(self.rader.convolution_len)
                .ok_or(VkFftError::ArithmeticOverflow {
                    operation: "single-shared fused FFT-Rader convolution elements",
                })?
        } else {
            self.parent
                .logical_len
                .checked_mul(self.shared_stripe_count()?)
                .ok_or(VkFftError::ArithmeticOverflow {
                    operation: "fused FFT-Rader shared stripe elements",
                })?
        };
        elements_per_transform
            .checked_mul(self.axis_block.grouped_batch)
            .and_then(|value| value.checked_mul(self.parent.scalar.complex_bytes()))
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "fused FFT-Rader composite shared-memory size",
            })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct RecursiveCooleyTukeyIr {
    pub logical_len: usize,
    pub left_len: usize,
    pub right_len: usize,
    pub batch_count: usize,
    pub direction: Direction,
    pub scalar: ScalarType,
    pub pack_right: CooleyTukeyPassIr,
    pub right: RecursiveFftNodeIr,
    pub twiddle_transpose: CooleyTukeyPassIr,
    pub left: RecursiveFftNodeIr,
    pub scatter_output: CooleyTukeyPassIr,
}

impl RecursiveCooleyTukeyIr {
    pub(crate) fn fused_small_direct_rader_stockham(
        &self,
    ) -> Result<Option<FusedSmallDirectRaderStockhamIr<'_>>> {
        self.validate()?;
        if !matches!(self.left_len, 2 | 3)
            || self.logical_len > 128
            || self.pack_right.input_storage_scalar != self.scalar
            || self.pack_right.output_storage_scalar != self.scalar
            || self.scatter_output.input_storage_scalar != self.scalar
            || self.scatter_output.output_storage_scalar != self.scalar
            || self.pack_right.input_modifier != CooleyTukeyInputModifier::None
            || !matches!(
                self.scatter_output.output_modifier,
                CooleyTukeyOutputModifier::None | CooleyTukeyOutputModifier::FourStepLeft(_)
            )
        {
            return Ok(None);
        }
        let Some(axis_block) = self.pack_right.axis_batch_block else {
            return Ok(None);
        };
        if self.twiddle_transpose.axis_batch_block != Some(axis_block)
            || self.scatter_output.axis_batch_block != Some(axis_block)
        {
            return Ok(None);
        }
        let RecursiveFftNodeIr::Stockham(stockham) = &self.left else {
            return Ok(None);
        };
        let RecursiveFftNodeIr::DirectRader(direct) = &self.right else {
            return Ok(None);
        };
        if stockham.sequence_len != self.left_len
            || stockham.scalar != self.scalar
            || stockham.io_mapping != StockhamIoMapping::Contiguous
            || stockham.input_modifier != crate::kernel_ir::StockhamInputModifier::None
            || stockham.output_modifier != crate::kernel_ir::StockhamOutputModifier::None
            || direct.prime != self.right_len
            || direct.scalar != self.scalar
            || direct.input_storage_scalar != self.scalar
            || direct.output_storage_scalar != self.scalar
            || direct.io_mapping != StockhamIoMapping::Contiguous
        {
            return Ok(None);
        }
        Ok(Some(FusedSmallDirectRaderStockhamIr {
            parent: self,
            stockham,
            direct,
            axis_block,
        }))
    }

    /// Materialize the strict small `Stockham x FFT-Rader` parent as one workgroup-owned
    /// candidate. The fused backend must still execute the real `(p - 1)` forward FFT,
    /// kernel-spectrum multiply, inverse FFT, Rader reconstruction, parent twiddle, and
    /// left Stockham transform; this eligibility check never substitutes Direct-Rader.
    pub(crate) fn fused_small_fft_rader_stockham(
        &self,
    ) -> Result<Option<FusedSmallFftRaderStockhamIr<'_>>> {
        self.validate()?;
        if self.scalar != ScalarType::F32
            || !matches!(self.left_len, 2 | 3)
            || !matches!(self.right_len, 17 | 29)
            || self.logical_len > 128
            || self.pack_right.input_storage_scalar != self.scalar
            || self.pack_right.output_storage_scalar != self.scalar
            || self.scatter_output.input_storage_scalar != self.scalar
            || self.scatter_output.output_storage_scalar != self.scalar
            || self.pack_right.input_modifier != CooleyTukeyInputModifier::None
            || !matches!(
                self.scatter_output.output_modifier,
                CooleyTukeyOutputModifier::None | CooleyTukeyOutputModifier::FourStepLeft(_)
            )
        {
            return Ok(None);
        }
        let Some(axis_block) = self.pack_right.axis_batch_block else {
            return Ok(None);
        };
        if self.twiddle_transpose.axis_batch_block != Some(axis_block)
            || self.scatter_output.axis_batch_block != Some(axis_block)
        {
            return Ok(None);
        }
        let RecursiveFftNodeIr::Stockham(stockham) = &self.left else {
            return Ok(None);
        };
        let RecursiveFftNodeIr::FftRader(rader) = &self.right else {
            return Ok(None);
        };
        if stockham.sequence_len != self.left_len
            || stockham.scalar != self.scalar
            || stockham.io_mapping != StockhamIoMapping::Contiguous
            || stockham.input_modifier != crate::kernel_ir::StockhamInputModifier::None
            || stockham.output_modifier != crate::kernel_ir::StockhamOutputModifier::None
            || rader.prime != self.right_len
            || rader.scalar != self.scalar
            || rader.input_storage_scalar != self.scalar
            || rader.output_storage_scalar != self.scalar
            || rader.io_mapping != StockhamIoMapping::Contiguous
            || rader.input_strategy
                != crate::rader_ir::RaderFftInputStrategy::GeneratorOrderStockham
        {
            return Ok(None);
        }
        let Some(forward_ir) = rader.forward_recursive() else {
            return Ok(None);
        };
        let Some(inverse_ir) = rader.inverse_recursive() else {
            return Ok(None);
        };
        let RecursiveFftNodeIr::Stockham(forward) = &forward_ir.root else {
            return Ok(None);
        };
        let RecursiveFftNodeIr::Stockham(inverse) = &inverse_ir.root else {
            return Ok(None);
        };
        if forward.sequence_len + 1 != rader.prime
            || inverse.sequence_len != forward.sequence_len
            || forward.scalar != self.scalar
            || inverse.scalar != self.scalar
            || forward.input_modifier != crate::kernel_ir::StockhamInputModifier::None
            || forward.output_modifier != crate::kernel_ir::StockhamOutputModifier::None
            || inverse.io_mapping != StockhamIoMapping::Contiguous
            || inverse.input_modifier != crate::kernel_ir::StockhamInputModifier::None
            || inverse.output_modifier != crate::kernel_ir::StockhamOutputModifier::None
            || forward.twiddle_source != inverse.twiddle_source
            || forward.twiddle_lut_len() != inverse.twiddle_lut_len()
            || forward
                .operations
                .iter()
                .find_map(|operation| match operation {
                    KernelOperation::StoreSharedToGlobal { normalize, .. } => Some(*normalize),
                    _ => None,
                })
                != Some(false)
            || inverse
                .operations
                .iter()
                .find_map(|operation| match operation {
                    KernelOperation::StoreSharedToGlobal { normalize, .. } => Some(*normalize),
                    _ => None,
                })
                != Some(true)
        {
            return Ok(None);
        }
        let caller = CooleyRightInputMapping {
            parent_logical_len: self.logical_len,
            parent_left_len: self.left_len,
            parent_right_len: self.right_len,
            parent_batch_count: self.batch_count,
        };
        caller.validate(rader.prime, rader.batch_count)?;
        let mapped = match rader
            .as_ref()
            .clone()
            .with_cooley_right_input_mapping(caller)
        {
            Ok(mapped) => mapped,
            Err(VkFftError::UnsupportedKernelPath(_))
            | Err(VkFftError::ResourceLimitExceeded { .. }) => return Ok(None),
            Err(error) => return Err(error),
        };
        if mapped.fused_inverse_rader_kernel()?.is_none() {
            return Ok(None);
        }
        let fused = FusedSmallFftRaderStockhamIr {
            parent: self,
            stockham,
            rader,
            forward,
            inverse,
            axis_block,
        };
        if fused.required_shared_memory_bytes()? > rader.device_shared_memory_bytes {
            return Ok(None);
        }
        Ok(Some(fused))
    }

    /// Fuse only the Cooley parent boundary around a small left Stockham child when
    /// the right child is FFT-Rader. The right FFT-Rader pipeline stays intact; its
    /// output is consumed directly by the left Stockham loads with the parent twiddle,
    /// and the left Stockham stores directly through the parent natural/Four-step
    /// scatter. Unsupported shapes keep the ordinary multi-pass Cooley lowering.
    pub(crate) fn fused_fft_rader_left_stockham(&self) -> Result<Option<KernelIr>> {
        self.validate()?;
        if !matches!(self.left_len, 2 | 3)
            || self.logical_len > 128
            || self.scatter_output.input_storage_scalar != self.scalar
            || self.scatter_output.output_storage_scalar != self.scalar
            || !matches!(
                self.scatter_output.output_modifier,
                CooleyTukeyOutputModifier::None | CooleyTukeyOutputModifier::FourStepLeft(_)
            )
        {
            return Ok(None);
        }
        let RecursiveFftNodeIr::Stockham(stockham) = &self.left else {
            return Ok(None);
        };
        let RecursiveFftNodeIr::FftRader(rader) = &self.right else {
            return Ok(None);
        };
        if stockham.sequence_len != self.left_len
            || stockham.scalar != self.scalar
            || stockham.io_mapping != StockhamIoMapping::Contiguous
            || stockham.input_modifier != crate::kernel_ir::StockhamInputModifier::None
            || stockham.output_modifier != crate::kernel_ir::StockhamOutputModifier::None
            || rader.prime != self.right_len
            || rader.scalar != self.scalar
        {
            return Ok(None);
        }
        let outer_four_step = match self.scatter_output.output_modifier {
            CooleyTukeyOutputModifier::None => None,
            CooleyTukeyOutputModifier::FourStepLeft(mapping) => Some(mapping),
            _ => unreachable!("validated small FFT-Rader Cooley output modifier"),
        };
        let mapping = CooleyLeftStockhamMapping {
            parent_logical_len: self.logical_len,
            parent_left_len: self.left_len,
            parent_right_len: self.right_len,
            parent_batch_count: self.batch_count,
            direction: self.direction,
            outer_four_step,
        };
        let mapped = stockham
            .as_ref()
            .clone()
            .with_stockham_io_mapping(StockhamIoMapping::CooleyLeft(mapping))?;
        mapped.validate()?;
        Ok(Some(mapped))
    }

    /// Fuse the remaining Cooley pack boundary into the right FFT-Rader generator
    /// loads when the caller is same-scalar and unmodified. The right FFT-Rader
    /// pipeline itself remains unchanged; its fused inverse scatter receives the
    /// same Cooley-right mapping for x0/DC auxiliary reads. Unsupported resource or
    /// storage shapes deliberately fall back to the left-boundary-only path.
    pub(crate) fn fused_fft_rader_cooley_boundaries(
        &self,
    ) -> Result<Option<(RaderFftPipelineIr, KernelIr)>> {
        let Some(mapped_left) = self.fused_fft_rader_left_stockham()? else {
            return Ok(None);
        };
        if self.pack_right.input_modifier != CooleyTukeyInputModifier::None
            || self.pack_right.input_storage_scalar != self.scalar
            || self.pack_right.output_storage_scalar != self.scalar
        {
            return Ok(None);
        }
        let RecursiveFftNodeIr::FftRader(rader) = &self.right else {
            return Ok(None);
        };
        if rader.input_storage_scalar != self.scalar || rader.output_storage_scalar != self.scalar {
            return Ok(None);
        }
        let caller = CooleyRightInputMapping {
            parent_logical_len: self.logical_len,
            parent_left_len: self.left_len,
            parent_right_len: self.right_len,
            parent_batch_count: self.batch_count,
        };
        let mapped_right = match rader
            .as_ref()
            .clone()
            .with_cooley_right_input_mapping(caller)
        {
            Ok(mapped) => mapped,
            Err(VkFftError::UnsupportedKernelPath(_))
            | Err(VkFftError::ResourceLimitExceeded { .. }) => return Ok(None),
            Err(error) => return Err(error),
        };
        mapped_right.validate()?;
        Ok(Some((mapped_right, mapped_left)))
    }

    pub fn validate(&self) -> Result<()> {
        if self.left_len < 2
            || self.right_len < 2
            || self.batch_count == 0
            || self.left_len.checked_mul(self.right_len) != Some(self.logical_len)
        {
            return Err(VkFftError::InvalidKernelIr(
                "recursive Cooley-Tukey node dimensions are inconsistent",
            ));
        }
        self.pack_right.validate()?;
        self.right.validate()?;
        self.twiddle_transpose.validate()?;
        self.left.validate()?;
        self.scatter_output.validate()?;
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
                || pass.scalar != self.scalar
            {
                return Err(VkFftError::InvalidKernelIr(
                    "recursive Cooley-Tukey outer pass metadata is inconsistent",
                ));
            }
        }
        let right_batch_count =
            self.batch_count
                .checked_mul(self.left_len)
                .ok_or(VkFftError::ArithmeticOverflow {
                    operation: "recursive Cooley-Tukey right-child batch count",
                })?;
        let left_batch_count =
            self.batch_count
                .checked_mul(self.right_len)
                .ok_or(VkFftError::ArithmeticOverflow {
                    operation: "recursive Cooley-Tukey left-child batch count",
                })?;
        if self.pack_right.operation != CooleyTukeyPassOperation::PackRightInput
            || self.twiddle_transpose.operation != CooleyTukeyPassOperation::TwiddleTranspose
            || self.scatter_output.operation != CooleyTukeyPassOperation::ScatterOutput
            || self.right.logical_len() != self.right_len
            || self.right.batch_count() != right_batch_count
            || self.left.logical_len() != self.left_len
            || self.left.batch_count() != left_batch_count
            || self.right.direction() != self.direction
            || self.left.direction() != self.direction
            || self.right.scalar() != self.scalar
            || self.left.scalar() != self.scalar
        {
            return Err(VkFftError::InvalidKernelIr(
                "recursive Cooley-Tukey child metadata is inconsistent",
            ));
        }
        Ok(())
    }
}

/// Static compiler-input resource metrics for one recognized monolithic small
/// `Stockham x FFT-Rader` pass. These values describe typed IR/codegen ownership;
/// they are not hardware register allocation or achieved occupancy measurements.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FusedFftRaderStaticResourceReport {
    pub pass_name: String,
    pub required_shared_memory_bytes: usize,
    pub workgroup_size: WorkgroupSize,
    pub uniform_barrier_count: usize,
    pub max_logical_register_complex_values_per_invocation: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RecursiveFftIr {
    pub logical_len: usize,
    pub batch_count: usize,
    pub direction: Direction,
    pub scalar: ScalarType,
    /// Scalar used by caller-visible buffers. Four-step roots keep their recursive
    /// tree in compute precision and apply this only to the first/last physical upload.
    pub external_scalar: ScalarType,
    /// Precision contract used by the device scheduler. Mixed-storage internal children
    /// keep compute-only buffers while retaining the caller storage policy here.
    pub scheduler_precision: Precision,
    /// Effective Rader tuning used to materialize scheduler-facing parent blocks.
    /// Higher-axis retagging must reuse the same policy instead of reconstructing a
    /// device default that can disagree with caller-provided prime thresholds.
    pub(crate) scheduler_tuning: PlannerTuning,
    /// Physical device profile used by device-specific scheduler materialization.
    pub device: DeviceProfile,
    /// Optional fixed upstream `configuration.groupedBatch[0]` override.
    pub axis0_grouped_batch_override: Option<usize>,
    /// Upstream-derived one/two/three-upload decomposition when the current
    /// executable Stockham leaves can follow it exactly. Other paths retain the
    /// correctness-first balanced recursive decomposition.
    pub stockham_upload_schedule: Option<StockhamUploadSchedule>,
    /// Mirrors upstream `forceRaderTwoUpload`: when true, the axis cannot remain a
    /// single scheduler upload because an FFT-Rader prime has too many containers.
    pub rader_forced_two_upload: bool,
    /// Exact fixed-upstream non-power-of-two two/three-upload geometry when the
    /// forced Rader branch finds a legal divisor split.
    pub rader_forced_upload_schedule: Option<RaderUploadSchedule>,
    pub four_step_plan: Option<FourStepPlanIr>,
    /// Optional upstream-style spatial zero-padding boundary for top-level 1D C2C.
    /// Internal recursive FFTs created for Rader/Bluestein keep this `None`.
    pub zero_pad_pass: Option<ZeroPadPassIr>,
    pub root: RecursiveFftNodeIr,
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

impl RecursiveFftIr {
    /// Return typed static resource metrics for monolithic small FFT-Rader passes
    /// that are materialized directly by this recursive transform. This intentionally
    /// reports only the fused passes whose register/shared ownership is proven in IR;
    /// all other passes remain absent instead of being estimated from generated source.
    pub fn fused_fft_rader_static_resource_reports(
        &self,
    ) -> Result<Vec<FusedFftRaderStaticResourceReport>> {
        self.validate()?;
        let mut reports = Vec::new();
        if let Some(uploads) = self.four_step_rader_upload_nodes()? {
            for upload in &uploads {
                upload.collect_fused_fft_rader_static_resource_reports(&mut reports)?;
            }
        } else {
            self.root
                .collect_fused_fft_rader_static_resource_reports(&mut reports)?;
        }
        Ok(reports)
    }

    pub fn build(plan: &FftPlan, direction: Direction, device: DeviceProfile) -> Result<Self> {
        Self::build_with_internal_compute_storage(plan, direction, device, false, false)
    }

    pub(crate) fn build_for_convolution(
        plan: &FftPlan,
        direction: Direction,
        device: DeviceProfile,
    ) -> Result<Self> {
        Self::build_with_internal_compute_storage(plan, direction, device, false, true)
    }

    pub(crate) fn build_for_convolution_internal_compute_storage(
        plan: &FftPlan,
        direction: Direction,
        device: DeviceProfile,
    ) -> Result<Self> {
        Self::build_with_internal_compute_storage(plan, direction, device, true, true)
    }

    pub(crate) fn build_internal_compute_storage(
        plan: &FftPlan,
        direction: Direction,
        device: DeviceProfile,
    ) -> Result<Self> {
        Self::build_with_internal_compute_storage(plan, direction, device, true, false)
    }

    fn build_with_internal_compute_storage(
        plan: &FftPlan,
        direction: Direction,
        device: DeviceProfile,
        internal_compute_storage: bool,
        perform_convolution: bool,
    ) -> Result<Self> {
        if plan.config.dimensions.len() != 1 {
            return Err(VkFftError::UnsupportedKernelPath(
                "recursive FFT IR currently supports one-dimensional plans only",
            ));
        }
        if plan.config.transform != TransformKind::ComplexToComplex {
            return Err(VkFftError::UnsupportedKernelPath(
                "recursive FFT IR currently supports C2C transforms only",
            ));
        }
        if plan.config.kernel_convolution && direction != Direction::Forward {
            return Err(VkFftError::UnsupportedKernelPath(
                "kernelConvolution is a forward-only kernel-spectrum preparation transform",
            ));
        }
        let execution_batch_count = plan.config.kernel_preparation_system_count()?;
        // Preserve the caller-visible precision for upstream scheduler policy. Mixed-storage
        // formats still build compute leaves in F32/F64, but F16 storage must retain its
        // doubled coalescing width through upload selection and Direct-Rader thread caps.
        let scheduler_precision = plan.config.precision;
        let (scalar, leaf_precision, caller_external_storage_scalar) = match plan.config.precision {
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
                    backend: "recursive FFT IR",
                    precision: precision_name(other),
                });
            }
        };
        let effective_rader_tuning =
            upstream_effective_rader_tuning(plan.config.tuning, device, scheduler_precision);
        let external_storage_scalar = if internal_compute_storage {
            scalar
        } else {
            caller_external_storage_scalar
        };
        let axis = plan.axes.first().ok_or(VkFftError::InvalidKernelIr(
            "missing recursive FFT axis plan",
        ))?;
        let axis0_grouped_batch_override = plan.config.grouped_batch_for_axis(0);
        let stockham_axis_context = StockhamUploadAxisContext {
            strided_axis: matches!(
                plan.c2c_device_axis_class_override,
                Some(C2cDeviceAxisClass::Strided)
            ),
            bandwidth_boost: plan.config.bandwidth_boost,
            use_bluestein_fft: plan.c2c_device_use_bluestein_fft_override,
            // Pinned `kernelConvolution` leaves performConvolution=0 but forces the same
            // reorder/register-capacity policy. This internal bit describes capacity only;
            // application midpoint semantics remain owned by ConvolutionIr.
            perform_convolution: perform_convolution || plan.config.kernel_convolution,
        };
        let zero_pad_pass = plan
            .config
            .zero_padding_for_axis(0)
            .map(|range| {
                ZeroPadPassIr::build_with_domain(
                    axis.effective_fft_len,
                    execution_batch_count,
                    plan.config.precision,
                    direction,
                    range,
                    plan.config.zero_padding_domain,
                    device,
                )
            })
            .transpose()?;
        let stockham_leaf_limit = stockham_leaf_element_limit(device, scalar)?;
        let mut factors = Vec::new();
        let mut stockham_upload_schedule = None;
        let mut rader_forced_two_upload = false;
        let mut rader_forced_upload_schedule = None;
        let mut component_factor_counts = None;
        match &axis.algorithm {
            AxisAlgorithm::Stockham { radix } => {
                if axis.effective_fft_len == 1 {
                    factors.push(FactorSpec {
                        len: 1,
                        kind: FactorKind::Stockham,
                    });
                } else if let Some((scheduled_factors, schedule)) = gpu_smooth_upload_factors(
                    axis.effective_fft_len,
                    execution_batch_count,
                    scheduler_precision,
                    device,
                    stockham_leaf_limit,
                    stockham_axis_context,
                )? {
                    let upload_count = schedule.upload_count;
                    factors.extend(scheduled_factors);
                    component_factor_counts = Some(vec![1; upload_count]);
                    stockham_upload_schedule = Some(schedule);
                } else {
                    push_stockham_chunks(
                        &mut factors,
                        &radix.prime_factors,
                        stockham_leaf_limit,
                        scalar,
                        device,
                    )?;
                }
            }
            AxisAlgorithm::Rader { stockham, primes } => {
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
                rader_forced_two_upload = plan_gpu_force_rader_two_upload(
                    axis.effective_fft_len,
                    &fft_rader_primes,
                    device,
                )?;
                if let Some(schedule) = plan_gpu_rader_upload_split_with_axis_context(
                    axis.effective_fft_len,
                    &fft_rader_primes,
                    &direct_rader_primes,
                    scheduler_precision,
                    device,
                    stockham_axis_context,
                )? {
                    let (scheduled_factors, scheduled_component_factor_counts) =
                        rader_factor_specs_for_upload_split(
                            &stockham.prime_factors,
                            primes,
                            &schedule.axis_split,
                            stockham_leaf_limit,
                            scalar,
                            device,
                        )?;
                    factors = scheduled_factors;
                    component_factor_counts = Some(scheduled_component_factor_counts);
                    rader_forced_upload_schedule = Some(schedule);
                } else {
                    push_stockham_chunks(
                        &mut factors,
                        &stockham.prime_factors,
                        stockham_leaf_limit,
                        scalar,
                        device,
                    )?;
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
                    "recursive Rader/Stockham IR does not replace Bluestein axes",
                ));
            }
        }
        if axis0_grouped_batch_override.is_some()
            && matches!(&axis.algorithm, AxisAlgorithm::Stockham { .. })
        {
            let schedule =
                stockham_upload_schedule
                    .as_ref()
                    .ok_or(VkFftError::UnsupportedKernelPath(
                        "groupedBatch override requires the covered Stockham upload scheduler",
                    ))?;
            if schedule.register_boost != 1 || !matches!(schedule.upload_count, 1..=3) {
                return Err(VkFftError::UnsupportedKernelPath(
                    "groupedBatch override currently requires boost-1 one/two/three-upload Stockham",
                ));
            }
        }
        if factors.is_empty() {
            return Err(VkFftError::InvalidKernelIr(
                "recursive FFT factor list is empty",
            ));
        }
        let product = checked_factor_product(&factors)?;
        if product != axis.effective_fft_len {
            return Err(VkFftError::InvalidKernelIr(
                "recursive FFT factors do not cover the planned axis length",
            ));
        }
        // Standalone prime Rader keeps its existing leaf-owned groupedBatch path.
        // Composite/forced-Rader axes own grouping at the mapped Four-step component
        // boundary instead, so do not push the axis override into nested leaves.
        let rader_grouped_batch_override = if factors.len() == 1
            && plan.config.grouped_batch_for_axis(0).is_some()
            && matches!(&axis.algorithm, AxisAlgorithm::Rader { .. })
        {
            plan.config.grouped_batch_for_axis(0)
        } else {
            None
        };
        let rader_zero_padding =
            if factors.len() == 1 && matches!(&axis.algorithm, AxisAlgorithm::Rader { .. }) {
                plan.config.zero_padding_for_axis(0)
            } else {
                None
            };
        if component_factor_counts.is_none() && rader_forced_two_upload && factors.len() > 1 {
            component_factor_counts = factors
                .iter()
                .position(|factor| {
                    matches!(
                        factor.kind,
                        FactorKind::Rader(RaderMode::FftConvolution { .. })
                    )
                })
                .map(|index| index.max(1).min(factors.len() - 1))
                .map(|split| vec![split, factors.len() - split]);
        }
        let mut root = build_node(
            &factors,
            execution_batch_count,
            1,
            component_factor_counts.as_deref(),
            rader_grouped_batch_override,
            rader_zero_padding,
            direction,
            leaf_precision,
            plan.config.normalize_inverse,
            effective_rader_tuning,
            device,
        )?;
        // Mixed-storage recursive leaves execute in their compute precision, but
        // standalone axis-0 FFT-Rader physical batching is still scored from the
        // caller-visible scheduler precision. Retag the leaf-owned caller block
        // after compute materialization so F16 keeps upstream's doubled coalescing
        // width without teaching the arithmetic child about F16 storage.
        if scheduler_precision != leaf_precision
            && let RecursiveFftNodeIr::FftRader(rader) = &mut root
            && let Some(schedule) = rader.internal_register_schedule.as_ref()
        {
            let block = plan_gpu_axis0_fft_rader_batch_block_with_grouped_batch_for_precision(
                rader.prime,
                rader.batch_count,
                schedule,
                scheduler_precision,
                scalar.complex_bytes(),
                zero_pad_pass.is_some(),
                rader_grouped_batch_override,
                device,
            )?;
            if rader_grouped_batch_override.is_some() && block.is_none() {
                return Err(VkFftError::UnsupportedKernelPath(
                    "groupedBatch override is not executable for the scheduler-precision FFT-Rader axis",
                ));
            }
            if let Some(block) = block {
                **rader = rader
                    .as_ref()
                    .clone()
                    .with_standalone_axis_batch_block(block, device)?;
            }
        }
        let four_step_context = FourStepBuildContext {
            precision: plan.config.precision,
            complex_bytes: scalar.complex_bytes(),
            perform_zero_padding: zero_pad_pass.is_some(),
            grouped_batch_override: axis0_grouped_batch_override,
            tuning: effective_rader_tuning,
            device,
        };
        let four_step_plan = if let Some(schedule) = &stockham_upload_schedule {
            match schedule.upload_count {
                2 => build_two_upload_four_step_plan(
                    schedule,
                    &root,
                    axis.effective_fft_len,
                    execution_batch_count,
                    four_step_context,
                )?,
                3 => build_three_upload_four_step_plan(
                    schedule,
                    &root,
                    axis.effective_fft_len,
                    execution_batch_count,
                    four_step_context,
                )?,
                _ => None,
            }
        } else if let Some(schedule) = &rader_forced_upload_schedule {
            build_forced_rader_four_step_plan(
                schedule,
                &root,
                axis.effective_fft_len,
                execution_batch_count,
                four_step_context,
            )?
        } else {
            None
        };
        // The recursive Cooley boundary is an implementation detail, but its physical
        // lane count can still mirror the fixed-upstream single-kernel scheduler. Carry
        // exact type-1 coupling for all-direct/one-FFT mixed axes and the joint smooth
        // multi-type0 FFT-Rader slice; the GLSL/native lowerings stride
        // `slot += local_size`, so fewer lanes still cover the full logical N.
        let exact_rader_parent_threads = if four_step_plan.is_none() && factors.len() > 1 {
            if let AxisAlgorithm::Rader { primes, .. } = &axis.algorithm {
                if !primes.is_empty()
                    && primes
                        .iter()
                        .all(|prime| matches!(prime.mode, RaderMode::DirectMultiplication))
                {
                    let direct_prime_multiplicities = primes
                        .iter()
                        .map(|prime| (prime.prime, prime.multiplicity))
                        .collect::<Vec<_>>();
                    plan_gpu_axis0_composite_direct_rader_threads_for_prime_multiplicities(
                        axis.effective_fft_len,
                        &direct_prime_multiplicities,
                        execution_batch_count,
                        device,
                    )?
                } else {
                    let direct_prime_multiplicities = primes
                        .iter()
                        .filter_map(|prime| {
                            matches!(prime.mode, RaderMode::DirectMultiplication)
                                .then_some((prime.prime, prime.multiplicity))
                        })
                        .collect::<Vec<_>>();
                    let fft_prime_multiplicities = primes
                        .iter()
                        .filter_map(|prime| {
                            matches!(prime.mode, RaderMode::FftConvolution { .. })
                                .then_some((prime.prime, prime.multiplicity))
                        })
                        .collect::<Vec<_>>();
                    if direct_prime_multiplicities.is_empty()
                        && !fft_prime_multiplicities.is_empty()
                    {
                        plan_gpu_axis0_multi_fft_rader_threads_for_prime_multiplicities_with_tuning(
                            axis.effective_fft_len,
                            &fft_prime_multiplicities,
                            execution_batch_count,
                            effective_rader_tuning,
                            device,
                        )?
                    } else if !direct_prime_multiplicities.is_empty()
                        && !fft_prime_multiplicities.is_empty()
                    {
                        plan_gpu_axis0_mixed_direct_multi_fft_rader_threads_with_tuning(
                            axis.effective_fft_len,
                            &direct_prime_multiplicities,
                            &fft_prime_multiplicities,
                            execution_batch_count,
                            effective_rader_tuning,
                            device,
                        )?
                    } else {
                        None
                    }
                }
            } else {
                None
            }
        } else {
            None
        };
        if let Some(threads_per_transform) = exact_rader_parent_threads {
            let RecursiveFftNodeIr::CooleyTukey(cooley) = &mut root else {
                return Err(VkFftError::InvalidKernelIr(
                    "composite Rader thread coupling requires a Cooley-Tukey root",
                ));
            };
            let block = StockhamAxisBlockSchedule {
                threads_per_transform,
                grouped_batch: 1,
                transforms_on_x: false,
                axis_swapped: false,
                local_size_x: threads_per_transform,
                local_size_y: 1,
            };
            block.validate(cooley.batch_count, device)?;
            cooley.pack_right = cooley
                .pack_right
                .clone()
                .with_axis0_batch_block(block, device)?;
            cooley.twiddle_transpose = cooley
                .twiddle_transpose
                .clone()
                .with_axis0_batch_block(block, device)?;
            cooley.scatter_output = cooley
                .scatter_output
                .clone()
                .with_axis0_batch_block(block, device)?;
            cooley.validate()?;
        }
        if let Some(grouped_batch_override) = axis0_grouped_batch_override
            && factors.len() > 1
            && matches!(&axis.algorithm, AxisAlgorithm::Rader { .. })
            && four_step_plan.is_none()
        {
            let RecursiveFftNodeIr::CooleyTukey(cooley) = &mut root else {
                return Err(VkFftError::InvalidKernelIr(
                    "composite Rader groupedBatch requires a Cooley-Tukey root",
                ));
            };
            let block = plan_gpu_axis0_four_step_grouped_block_from_shape(
                1,
                cooley.logical_len,
                cooley
                    .pack_right
                    .axis_batch_block
                    .map_or(cooley.pack_right.workgroup_size.x as usize, |block| {
                        block.threads_per_transform
                    }),
                FourStepAxisBlockRequest {
                    axis_upload_id: 0,
                    stage_start_size: 1,
                    transform_count: cooley.batch_count,
                    outer_batch_count: cooley.batch_count,
                    perform_zero_padding: zero_pad_pass.is_some(),
                    grouped_batch_override: Some(grouped_batch_override),
                },
                scalar.complex_bytes(),
                device,
            )?
            .ok_or(VkFftError::UnsupportedKernelPath(
                "groupedBatch override is not executable for the composite Rader root boundary",
            ))?;
            cooley.pack_right = cooley
                .pack_right
                .clone()
                .with_axis0_batch_block(block, device)?;
            cooley.twiddle_transpose = cooley
                .twiddle_transpose
                .clone()
                .with_axis0_batch_block(block, device)?;
            cooley.scatter_output = cooley
                .scatter_output
                .clone()
                .with_axis0_batch_block(block, device)?;
            cooley.validate()?;
        }
        if external_storage_scalar != scalar && four_step_plan.is_none() {
            let input_boundary_owned_by_pad = zero_pad_pass
                .as_ref()
                .is_some_and(|pass| pass.operation.is_input_boundary());
            let output_boundary_owned_by_pad = zero_pad_pass
                .as_ref()
                .is_some_and(|pass| pass.operation.is_output_boundary());
            match &mut root {
                RecursiveFftNodeIr::Stockham(kernel) => {
                    let mut updated = kernel.as_ref().clone();
                    if !input_boundary_owned_by_pad {
                        updated =
                            updated.with_external_input_storage_scalar(external_storage_scalar)?;
                    }
                    if !output_boundary_owned_by_pad {
                        updated =
                            updated.with_external_output_storage_scalar(external_storage_scalar)?;
                    }
                    **kernel = updated;
                }
                RecursiveFftNodeIr::CooleyTukey(cooley) => {
                    if !input_boundary_owned_by_pad {
                        cooley.pack_right = cooley
                            .pack_right
                            .clone()
                            .with_external_input_storage(external_storage_scalar)?;
                    }
                    if !output_boundary_owned_by_pad {
                        cooley.scatter_output = cooley
                            .scatter_output
                            .clone()
                            .with_external_output_storage(external_storage_scalar)?;
                    }
                    cooley.validate()?;
                }
                RecursiveFftNodeIr::DirectRader(rader) => {
                    let mut updated = rader.as_ref().clone();
                    if !input_boundary_owned_by_pad {
                        updated =
                            updated.with_external_input_storage_scalar(external_storage_scalar)?;
                    }
                    if !output_boundary_owned_by_pad {
                        updated =
                            updated.with_external_output_storage_scalar(external_storage_scalar)?;
                    }
                    **rader = updated;
                }
                RecursiveFftNodeIr::FftRader(rader) => {
                    let mut updated = rader.as_ref().clone();
                    if !input_boundary_owned_by_pad {
                        updated =
                            updated.with_external_input_storage_scalar(external_storage_scalar)?;
                    }
                    if !output_boundary_owned_by_pad {
                        updated =
                            updated.with_external_output_storage_scalar(external_storage_scalar)?;
                    }
                    **rader = updated;
                }
            }
        }
        let ir = Self {
            logical_len: axis.effective_fft_len,
            batch_count: execution_batch_count,
            direction,
            scalar,
            external_scalar: external_storage_scalar,
            scheduler_precision,
            scheduler_tuning: effective_rader_tuning,
            device,
            stockham_upload_schedule,
            axis0_grouped_batch_override,
            rader_forced_two_upload,
            rader_forced_upload_schedule,
            four_step_plan,
            zero_pad_pass,
            root,
        };
        ir.validate()?;
        Ok(ir)
    }

    pub fn external_storage_scalar(&self) -> ScalarType {
        self.external_scalar
    }

    /// Enable the covered `VkFFTSplitAxisBlock` batch grouping only for a
    /// top-level, single-upload Stockham root. Spatial zero padding keeps grouping
    /// but suppresses upstream's bank-conflict axis swap; internal recursive/Rader/
    /// real-transform children keep their established one-dimensional workgroups.
    pub(crate) fn with_axis0_single_upload_block(self, device: DeviceProfile) -> Result<Self> {
        let perform_zero_padding = self.zero_pad_pass.is_some();
        self.with_axis0_single_upload_block_for_context(device, perform_zero_padding)
    }

    pub(crate) fn with_axis0_single_upload_block_for_context(
        mut self,
        device: DeviceProfile,
        perform_zero_padding: bool,
    ) -> Result<Self> {
        let Some(upload) = self.stockham_upload_schedule.as_ref() else {
            return Ok(self);
        };
        if upload.upload_count != 1 {
            return Ok(self);
        }
        let block = plan_gpu_axis0_single_upload_block_with_grouped_batch(
            upload,
            self.scalar.complex_bytes(),
            perform_zero_padding,
            self.axis0_grouped_batch_override,
            device,
        )?;
        let Some(block) = block else {
            if self.axis0_grouped_batch_override.is_some() {
                return Err(VkFftError::UnsupportedKernelPath(
                    "groupedBatch override produced an unsupported partial or resource-limited Stockham workgroup",
                ));
            }
            return Ok(self);
        };
        if block.grouped_batch <= 1 {
            return Ok(self);
        }
        let RecursiveFftNodeIr::Stockham(root) = &mut self.root else {
            return Ok(self);
        };
        let schedule = upload.radix_schedules[0].clone();
        let grouped = (**root)
            .clone()
            .with_axis0_batch_block(schedule, block, device)?;
        **root = grouped;
        self.validate()?;
        Ok(self)
    }

    /// Apply the covered `axis_id >= 1` Stockham block to a packed multidimensional
    /// child. Single-upload roots replace their kernel block directly; two/three-upload
    /// Four-step roots retag every upload's typed axis block. The child buffer is
    /// contiguous after the ND pack pass, but physical ownership intentionally follows
    /// upstream strided axes: independent sequences on X, FFT threads on Y, capped by
    /// the fastest physical tensor dimension rather than the flattened line count.
    pub(crate) fn with_other_axis_single_upload_block(
        self,
        fastest_axis_len: usize,
        device: DeviceProfile,
    ) -> Result<Self> {
        self.with_other_axis_single_upload_block_with_grouped_batch(
            fastest_axis_len,
            None,
            None,
            device,
        )
    }

    /// User-grouped higher-axis form of the fixed `VkFFTSplitAxisBlock` branch.
    /// The logical 1D child may still carry the user's groupedBatch metadata, but
    /// this replaces an axis-0-shaped Stockham root with the physical strided-axis
    /// X/Y tile selected from the parent tensor shape.
    pub(crate) fn with_other_axis_single_upload_block_with_grouped_batch(
        mut self,
        fastest_axis_len: usize,
        grouped_batch_override: Option<usize>,
        axis1_grouped_batch_override: Option<usize>,
        device: DeviceProfile,
    ) -> Result<Self> {
        let scheduler_precision = self.scheduler_precision;
        let scheduler_tuning = self.scheduler_tuning;
        match &mut self.root {
            RecursiveFftNodeIr::DirectRader(direct) => {
                let block = plan_gpu_other_axis_direct_rader_batch_block_for_precision(
                    direct.prime,
                    direct.batch_count,
                    fastest_axis_len,
                    scheduler_precision,
                    self.scalar.complex_bytes(),
                    grouped_batch_override,
                    axis1_grouped_batch_override,
                    device,
                )?;
                if grouped_batch_override.is_some() && block.is_none() {
                    return Err(VkFftError::UnsupportedKernelPath(
                        "grouped higher-axis direct-Rader workgroup is not executable on this device",
                    ));
                }
                if let Some(block) = block {
                    **direct = direct
                        .as_ref()
                        .clone()
                        .with_axis0_batch_block(block, device)?;
                    self.validate()?;
                }
                return Ok(self);
            }
            RecursiveFftNodeIr::FftRader(rader) => {
                let Some(schedule) = rader.internal_register_schedule.as_ref() else {
                    if grouped_batch_override.is_some() {
                        return Err(VkFftError::UnsupportedKernelPath(
                            "grouped higher-axis FFT-Rader requires a physical register schedule",
                        ));
                    }
                    return Ok(self);
                };
                let block = plan_gpu_other_axis_fft_rader_batch_block_for_precision(
                    rader.prime,
                    rader.batch_count,
                    fastest_axis_len,
                    schedule,
                    scheduler_precision,
                    self.scalar.complex_bytes(),
                    grouped_batch_override,
                    axis1_grouped_batch_override,
                    device,
                )?;
                if grouped_batch_override.is_some() && block.is_none() {
                    return Err(VkFftError::UnsupportedKernelPath(
                        "grouped higher-axis FFT-Rader workgroup is not executable on this device",
                    ));
                }
                if let Some(block) = block {
                    **rader = rader
                        .as_ref()
                        .clone()
                        .with_standalone_axis_batch_block(block, device)?;
                    self.validate()?;
                }
                return Ok(self);
            }
            RecursiveFftNodeIr::Stockham(_) | RecursiveFftNodeIr::CooleyTukey(_) => {}
        }

        if let (Some(schedule), Some(four_step)) = (
            self.rader_forced_upload_schedule.as_ref(),
            self.four_step_plan.as_ref(),
        ) {
            let mut retagged = Vec::with_capacity(four_step.uploads.len());
            for upload_ir in &four_step.uploads {
                let component = forced_rader_component_for_upload(
                    &self.root,
                    schedule,
                    upload_ir.axis_upload_id,
                )
                .ok_or(VkFftError::InvalidKernelIr(
                    "forced-Rader Four-step upload does not map to its recursive component",
                ))?;
                let max_batch_coalesced =
                    (upstream_coalesced_memory_bytes_for_precision(device, scheduler_precision)
                        / self.scalar.complex_bytes())
                    .max(1);
                let exact_threads = exact_rader_component_parent_threads_with_max_batch_coalesced(
                    component,
                    device,
                    max_batch_coalesced,
                    false,
                    scheduler_tuning,
                )?;
                let threads_per_transform = if let Some(threads) = exact_threads {
                    threads
                } else if let Some(threads) = higher_axis_forced_rader_component_threads(
                    component,
                    max_batch_coalesced,
                    device,
                )? {
                    threads
                } else if let Some(block) = upload_ir.axis_block {
                    block.threads_per_transform
                } else {
                    forced_rader_component_threads(component, device)?.ok_or(
                        VkFftError::UnsupportedKernelPath(
                            "higher-axis forced-Rader upload has no executable component thread shape",
                        ),
                    )?
                };
                let (available_shared_memory_bytes, has_direct_rader) =
                    forced_rader_component_physical_context(
                        component,
                        self.scalar.complex_bytes(),
                        device,
                    )?;
                let block = plan_gpu_other_axis_four_step_block_from_shape_for_precision(
                    schedule.upload_count,
                    upload_ir.axis_upload_id,
                    upload_ir.fft_len,
                    upload_ir.transform_count,
                    fastest_axis_len,
                    threads_per_transform,
                    scheduler_precision,
                    self.scalar.complex_bytes(),
                    grouped_batch_override,
                    axis1_grouped_batch_override,
                    OtherAxisFourStepPhysicalContext {
                        has_direct_rader,
                        available_shared_memory_bytes,
                    },
                    device,
                )?;
                if grouped_batch_override.is_some() && block.is_none() {
                    return Err(VkFftError::UnsupportedKernelPath(
                        "grouped higher-axis forced-Rader upload workgroup is not executable on this device",
                    ));
                }
                retagged.push(block);
            }
            if let Some(four_step) = self.four_step_plan.as_mut() {
                for (upload_ir, block) in four_step.uploads.iter_mut().zip(retagged) {
                    upload_ir.axis_block = block;
                }
            }
            if grouped_batch_override.is_some() {
                self.validate()?;
                if self.four_step_rader_upload_nodes()?.is_none() {
                    return Err(VkFftError::InvalidKernelIr(
                        "grouped higher-axis forced-Rader uploads cannot be materialized",
                    ));
                }
                // ProgramIr executes the materialized Four-step upload nodes above and bypasses
                // `self.root`. Do not apply the single-upload composite-Rader parent retag below
                // to this shadow recursive root; that can reject a valid grouped multi-upload
                // schedule even though every physical upload block is executable.
                return Ok(self);
            }
        }

        if self.root.contains_rader()
            && let RecursiveFftNodeIr::CooleyTukey(cooley) = &mut self.root
        {
            let threads_per_transform = cooley
                .pack_right
                .axis_batch_block
                .map_or(cooley.pack_right.workgroup_size.x as usize, |block| {
                    block.threads_per_transform
                });
            let block = plan_gpu_other_axis_composite_rader_batch_block_for_precision(
                cooley.logical_len,
                cooley.batch_count,
                fastest_axis_len,
                scheduler_precision,
                self.scalar.complex_bytes(),
                threads_per_transform,
                grouped_batch_override,
                axis1_grouped_batch_override,
                device,
            )?;
            if grouped_batch_override.is_some() && block.is_none() {
                return Err(VkFftError::UnsupportedKernelPath(
                    "grouped higher-axis composite-Rader parent workgroup is not executable on this device",
                ));
            }
            if let Some(block) = block {
                cooley.pack_right = cooley
                    .pack_right
                    .clone()
                    .with_axis0_batch_block(block, device)?;
                cooley.twiddle_transpose = cooley
                    .twiddle_transpose
                    .clone()
                    .with_axis0_batch_block(block, device)?;
                cooley.scatter_output = cooley
                    .scatter_output
                    .clone()
                    .with_axis0_batch_block(block, device)?;
                cooley.validate()?;
                self.validate()?;
            }
            return Ok(self);
        }

        let Some(upload) = self.stockham_upload_schedule.clone() else {
            return Ok(self);
        };
        if upload.upload_count == 1 {
            let block = plan_gpu_other_axis_single_upload_block_with_grouped_batch_for_precision(
                &upload,
                fastest_axis_len,
                scheduler_precision,
                self.scalar.complex_bytes(),
                grouped_batch_override,
                axis1_grouped_batch_override,
                device,
            )?;
            let Some(block) = block else {
                return Ok(self);
            };
            let RecursiveFftNodeIr::Stockham(root) = &mut self.root else {
                return Ok(self);
            };
            let schedule = upload.radix_schedules[0].clone();
            **root = root
                .as_ref()
                .clone()
                .with_axis0_batch_block(schedule, block, device)?;
            self.validate()?;
            return Ok(self);
        }
        if !matches!(upload.upload_count, 2 | 3) {
            return Ok(self);
        }
        let scheduler_precision = self.scheduler_precision;
        let Some(four_step) = self.four_step_plan.as_mut() else {
            return Ok(self);
        };
        for upload_ir in &mut four_step.uploads {
            let block =
                plan_gpu_other_axis_four_step_upload_block_with_grouped_batch_for_precision(
                    &upload,
                    upload_ir.axis_upload_id,
                    upload_ir.transform_count,
                    fastest_axis_len,
                    scheduler_precision,
                    self.scalar.complex_bytes(),
                    grouped_batch_override,
                    axis1_grouped_batch_override,
                    device,
                )?;
            if grouped_batch_override.is_some() && block.is_none() {
                return Err(VkFftError::UnsupportedKernelPath(
                    "grouped higher-axis multi-upload Stockham workgroup is not executable on this device",
                ));
            }
            upload_ir.axis_block = block;
        }
        self.validate()?;
        Ok(self)
    }

    /// Recursive counterpart of upstream `VkFFTGetRaderFFTThreadsNum`: collect the
    /// maximum physical FFT-Rader workgroup demand across nested/sub-Rader leaves.
    /// Direct-multiplication Rader nodes are intentionally excluded.
    pub fn rader_schedule_summary(&self) -> RecursiveRaderScheduleSummary {
        self.root.rader_schedule_summary()
    }

    /// Apply an explicitly planned register schedule to a single Stockham root.
    /// This is used by FFT-convolution Rader containers, whose `(p - 1)` FFT has
    /// different upstream register planning from an ordinary standalone Stockham
    /// transform. Generic multi-upload/Four-step metadata is cleared because this
    /// schedule becomes the authoritative execution plan for the root.
    /// Apply an explicit register schedule plus physical workgroup grouping to a
    /// single Stockham root. FFT-Rader uses this for `containerFFTNum > 1` while
    /// preserving the same logical batch and descriptor ABI.
    pub(crate) fn with_stockham_register_schedule_grouping(
        mut self,
        schedule: RadixRegisterSchedule,
        transforms_per_workgroup: usize,
        device: DeviceProfile,
    ) -> Result<Self> {
        if self.logical_len != schedule.fft_len || self.batch_count != schedule.rhs_transform_count
        {
            return Err(VkFftError::InvalidKernelIr(
                "Rader container register schedule does not match the recursive FFT",
            ));
        }
        let RecursiveFftNodeIr::Stockham(kernel) = &mut self.root else {
            return Err(VkFftError::UnsupportedKernelPath(
                "Rader container register scheduling currently requires a single Stockham root",
            ));
        };
        let replacement = (**kernel).clone().with_register_schedule_grouping(
            schedule,
            transforms_per_workgroup,
            device,
        )?;
        **kernel = replacement;
        self.stockham_upload_schedule = None;
        self.four_step_plan = None;
        self.validate()?;
        Ok(self)
    }

    /// Apply the first executable `raderTranspose` stage-lane schedule to a single
    /// Stockham root while retaining the recursive FFT descriptor ABI.
    pub(crate) fn with_stockham_rader_transpose_schedule(
        mut self,
        schedule: RadixRegisterSchedule,
        transpose: RaderFftTransposeSchedule,
        device: DeviceProfile,
    ) -> Result<Self> {
        if self.logical_len != schedule.fft_len || self.batch_count != schedule.rhs_transform_count
        {
            return Err(VkFftError::InvalidKernelIr(
                "Rader transpose schedule does not match the recursive FFT",
            ));
        }
        let RecursiveFftNodeIr::Stockham(kernel) = &mut self.root else {
            return Err(VkFftError::UnsupportedKernelPath(
                "Rader transpose scheduling currently requires a single Stockham root",
            ));
        };
        let replacement = (**kernel)
            .clone()
            .with_rader_transpose_register_schedule_grouping(schedule, transpose, device)?;
        **kernel = replacement;
        self.stockham_upload_schedule = None;
        self.four_step_plan = None;
        self.validate()?;
        Ok(self)
    }

    /// Materialize the physical Stockham kernels for the covered `reorderFourStep`
    /// paths. The returned order is VkFFT's axis-0 launch order: highest upload id
    /// first, upload 0 last.
    pub fn four_step_stockham_upload_kernels(&self) -> Result<Option<Vec<KernelIr>>> {
        let Some(plan) = &self.four_step_plan else {
            return Ok(None);
        };
        if self.stockham_upload_schedule.is_none() {
            return Ok(None);
        }
        plan.validate()?;
        match plan.uploads.len() {
            2 => self.two_upload_four_step_kernels(),
            3 => self.three_upload_four_step_kernels(),
            _ => Err(VkFftError::InvalidKernelIr(
                "unsupported Four-step upload count",
            )),
        }
    }

    pub(crate) fn four_step_rader_upload_nodes(&self) -> Result<Option<Vec<RecursiveFftNodeIr>>> {
        let Some(plan) = &self.four_step_plan else {
            return Ok(None);
        };
        let Some(schedule) = &self.rader_forced_upload_schedule else {
            return Ok(None);
        };
        if self.stockham_upload_schedule.is_some() {
            return Ok(None);
        }
        plan.validate()?;
        schedule.validate()?;
        match schedule.axis_split.as_slice() {
            [a, b] => {
                let RecursiveFftNodeIr::CooleyTukey(root) = &self.root else {
                    return Ok(None);
                };
                if root.left_len != *a || root.right_len != *b {
                    return Ok(None);
                }
                let mapping = FourStepMapping {
                    logical_len: self.logical_len,
                    left_len: *a,
                    right_len: *b,
                    outer_batch_count: self.batch_count,
                };
                mapping.validate()?;
                let Some(high) = map_two_upload_component(
                    &root.right,
                    mapping,
                    1,
                    plan.uploads[0].axis_block,
                    self.device,
                )?
                else {
                    return Ok(None);
                };
                let Some(low) = map_two_upload_component(
                    &root.left,
                    mapping,
                    0,
                    plan.uploads[1].axis_block,
                    self.device,
                )?
                else {
                    return Ok(None);
                };
                let mut uploads = vec![high, low];
                self.retag_rader_four_step_external_boundaries(&mut uploads)?;
                Ok(Some(uploads))
            }
            [a, b, c] => {
                let RecursiveFftNodeIr::CooleyTukey(root) = &self.root else {
                    return Ok(None);
                };
                let RecursiveFftNodeIr::CooleyTukey(upper) = &root.right else {
                    return Ok(None);
                };
                if root.left_len != *a
                    || root.right_len != b.checked_mul(*c).unwrap_or(0)
                    || upper.left_len != *b
                    || upper.right_len != *c
                {
                    return Ok(None);
                }
                let mapping = ThreeUploadFourStepMapping {
                    logical_len: self.logical_len,
                    axis_split: [*a, *b, *c],
                    outer_batch_count: self.batch_count,
                };
                mapping.validate()?;
                let Some(high) = map_three_upload_component(
                    &upper.right,
                    mapping,
                    2,
                    plan.uploads[0].axis_block,
                    self.device,
                )?
                else {
                    return Ok(None);
                };
                let Some(middle) = map_three_upload_component(
                    &upper.left,
                    mapping,
                    1,
                    plan.uploads[1].axis_block,
                    self.device,
                )?
                else {
                    return Ok(None);
                };
                let Some(low) = map_three_upload_component(
                    &root.left,
                    mapping,
                    0,
                    plan.uploads[2].axis_block,
                    self.device,
                )?
                else {
                    return Ok(None);
                };
                let mut uploads = vec![high, middle, low];
                self.retag_rader_four_step_external_boundaries(&mut uploads)?;
                Ok(Some(uploads))
            }
            _ => Ok(None),
        }
    }

    fn retag_rader_four_step_external_boundaries(
        &self,
        uploads: &mut [RecursiveFftNodeIr],
    ) -> Result<()> {
        if self.external_scalar == self.scalar {
            return Ok(());
        }
        let zero_padded = self.zero_pad_pass.is_some();
        if !zero_padded || self.direction == Direction::Inverse {
            let first = uploads.first_mut().ok_or(VkFftError::InvalidKernelIr(
                "Rader Four-step upload list is empty",
            ))?;
            retag_four_step_component_input(first, self.external_scalar)?;
        }
        if !zero_padded || self.direction == Direction::Forward {
            let last = uploads.last_mut().ok_or(VkFftError::InvalidKernelIr(
                "Rader Four-step upload list is empty",
            ))?;
            retag_four_step_component_output(last, self.external_scalar)?;
        }
        Ok(())
    }

    fn apply_four_step_axis_block(
        &self,
        kernel: KernelIr,
        upload: FourStepUploadIr,
    ) -> Result<KernelIr> {
        let Some(block) = upload.axis_block else {
            return Ok(kernel);
        };
        let schedule = self
            .stockham_upload_schedule
            .as_ref()
            .ok_or(VkFftError::InvalidKernelIr(
                "Four-step axis block requires scheduler metadata",
            ))?
            .radix_schedules
            .get(upload.axis_upload_id)
            .cloned()
            .ok_or(VkFftError::InvalidKernelIr(
                "Four-step axis block upload id is outside scheduler metadata",
            ))?;
        kernel.with_axis0_batch_block(schedule, block, self.device)
    }

    fn two_upload_four_step_kernels(&self) -> Result<Option<Vec<KernelIr>>> {
        let plan = self
            .four_step_plan
            .as_ref()
            .ok_or(VkFftError::InvalidKernelIr(
                "two-upload Four-step kernels require plan metadata",
            ))?;
        let [first_upload, second_upload] = plan.uploads.as_slice() else {
            return Err(VkFftError::InvalidKernelIr(
                "two-upload Four-step plan requires exactly two upload records",
            ));
        };
        let RecursiveFftNodeIr::CooleyTukey(node) = &self.root else {
            return Err(VkFftError::InvalidKernelIr(
                "two-upload Four-step plan requires a Cooley-Tukey root",
            ));
        };
        let RecursiveFftNodeIr::Stockham(right) = &node.right else {
            return Err(VkFftError::InvalidKernelIr(
                "two-upload Four-step upload 1 requires a Stockham right child",
            ));
        };
        let RecursiveFftNodeIr::Stockham(left) = &node.left else {
            return Err(VkFftError::InvalidKernelIr(
                "two-upload Four-step upload 0 requires a Stockham left child",
            ));
        };
        let mapping = FourStepMapping {
            logical_len: self.logical_len,
            left_len: node.left_len,
            right_len: node.right_len,
            outer_batch_count: self.batch_count,
        };
        mapping.validate()?;
        let mut first = self
            .apply_four_step_axis_block((**right).clone(), *first_upload)?
            .with_stockham_io_mapping(StockhamIoMapping::FourStepRight(mapping))?;
        let mut second = self
            .apply_four_step_axis_block((**left).clone(), *second_upload)?
            .with_stockham_io_mapping(StockhamIoMapping::FourStepLeft(mapping))?;
        if self.external_scalar != self.scalar {
            let input_boundary_owned_by_pad = self
                .zero_pad_pass
                .as_ref()
                .is_some_and(|pass| pass.operation.is_input_boundary());
            let output_boundary_owned_by_pad = self
                .zero_pad_pass
                .as_ref()
                .is_some_and(|pass| pass.operation.is_output_boundary());
            if !input_boundary_owned_by_pad {
                first = first.with_external_input_storage_scalar(self.external_scalar)?;
            }
            if !output_boundary_owned_by_pad {
                second = second.with_external_output_storage_scalar(self.external_scalar)?;
            }
        }
        first.name = format!("{}_four_step_upload_1", first.name);
        second.name = format!("{}_four_step_upload_0", second.name);
        Ok(Some(vec![first, second]))
    }

    fn three_upload_four_step_kernels(&self) -> Result<Option<Vec<KernelIr>>> {
        let plan = self
            .four_step_plan
            .as_ref()
            .ok_or(VkFftError::InvalidKernelIr(
                "three-upload Four-step kernels require plan metadata",
            ))?;
        let [upload2_meta, upload1_meta, upload0_meta] = plan.uploads.as_slice() else {
            return Err(VkFftError::InvalidKernelIr(
                "three-upload Four-step plan requires exactly three upload records",
            ));
        };
        let schedule =
            self.stockham_upload_schedule
                .as_ref()
                .ok_or(VkFftError::InvalidKernelIr(
                    "three-upload Four-step plan requires scheduler metadata",
                ))?;
        let [a, b, c]: [usize; 3] = schedule.axis_split.as_slice().try_into().map_err(|_| {
            VkFftError::InvalidKernelIr("three-upload Four-step split is incomplete")
        })?;
        let RecursiveFftNodeIr::CooleyTukey(root) = &self.root else {
            return Err(VkFftError::InvalidKernelIr(
                "three-upload Four-step plan requires a Cooley-Tukey root",
            ));
        };
        let RecursiveFftNodeIr::Stockham(low) = &root.left else {
            return Err(VkFftError::InvalidKernelIr(
                "three-upload Four-step upload 0 requires the low Stockham leaf",
            ));
        };
        let RecursiveFftNodeIr::CooleyTukey(upper) = &root.right else {
            return Err(VkFftError::InvalidKernelIr(
                "three-upload Four-step upper factors require a Cooley-Tukey node",
            ));
        };
        let RecursiveFftNodeIr::Stockham(middle) = &upper.left else {
            return Err(VkFftError::InvalidKernelIr(
                "three-upload Four-step upload 1 requires the middle Stockham leaf",
            ));
        };
        let RecursiveFftNodeIr::Stockham(high) = &upper.right else {
            return Err(VkFftError::InvalidKernelIr(
                "three-upload Four-step upload 2 requires the high Stockham leaf",
            ));
        };
        if low.sequence_len != a || middle.sequence_len != b || high.sequence_len != c {
            return Err(VkFftError::InvalidKernelIr(
                "three-upload Four-step Stockham leaves do not match axisSplit",
            ));
        }
        let mapping = ThreeUploadFourStepMapping {
            logical_len: self.logical_len,
            axis_split: [a, b, c],
            outer_batch_count: self.batch_count,
        };
        mapping.validate()?;
        let mut upload2 = self
            .apply_four_step_axis_block((**high).clone(), *upload2_meta)?
            .with_stockham_io_mapping(StockhamIoMapping::FourStepThreeUpload2(mapping))?;
        let mut upload1 = self
            .apply_four_step_axis_block((**middle).clone(), *upload1_meta)?
            .with_stockham_io_mapping(StockhamIoMapping::FourStepThreeUpload1(mapping))?;
        let mut upload0 = self
            .apply_four_step_axis_block((**low).clone(), *upload0_meta)?
            .with_stockham_io_mapping(StockhamIoMapping::FourStepThreeUpload0(mapping))?;
        if self.external_scalar != self.scalar {
            let input_boundary_owned_by_pad = self
                .zero_pad_pass
                .as_ref()
                .is_some_and(|pass| pass.operation.is_input_boundary());
            let output_boundary_owned_by_pad = self
                .zero_pad_pass
                .as_ref()
                .is_some_and(|pass| pass.operation.is_output_boundary());
            if !input_boundary_owned_by_pad {
                upload2 = upload2.with_external_input_storage_scalar(self.external_scalar)?;
            }
            if !output_boundary_owned_by_pad {
                upload0 = upload0.with_external_output_storage_scalar(self.external_scalar)?;
            }
        }
        upload2.name = format!("{}_four_step_upload_2", upload2.name);
        upload1.name = format!("{}_four_step_upload_1", upload1.name);
        upload0.name = format!("{}_four_step_upload_0", upload0.name);
        Ok(Some(vec![upload2, upload1, upload0]))
    }

    pub fn validate(&self) -> Result<()> {
        if self.logical_len == 0 || self.batch_count == 0 {
            return Err(VkFftError::InvalidKernelIr(
                "recursive FFT dimensions must be non-zero",
            ));
        }
        if self.external_scalar != self.scalar
            && !matches!(
                (self.scalar, self.external_scalar),
                (ScalarType::F32, ScalarType::F16) | (ScalarType::F64, ScalarType::F32)
            )
        {
            return Err(VkFftError::InvalidKernelIr(
                "recursive FFT external storage scalar is incompatible with compute precision",
            ));
        }
        if self.scheduler_precision.compute_complex_bytes() != self.scalar.complex_bytes() {
            return Err(VkFftError::InvalidKernelIr(
                "recursive FFT scheduler precision is incompatible with compute scalar",
            ));
        }
        self.root.validate()?;
        if let Some(pass) = &self.zero_pad_pass {
            pass.validate()?;
            let expected_storage = if pass.operation.is_input_boundary() {
                (self.external_scalar, self.scalar)
            } else {
                (self.scalar, self.external_scalar)
            };
            if pass.logical_len != self.logical_len
                || pass.batch_count != self.batch_count
                || pass.direction != self.direction
                || pass.scalar != self.scalar
                || (pass.input_storage_scalar, pass.output_storage_scalar) != expected_storage
            {
                return Err(VkFftError::InvalidKernelIr(
                    "recursive FFT zero-padding boundary does not match root metadata",
                ));
            }
        }
        if let Some(schedule) = &self.stockham_upload_schedule {
            schedule.validate()?;
            if schedule.sequence_len != self.logical_len || schedule.batch_count != self.batch_count
            {
                return Err(VkFftError::InvalidKernelIr(
                    "recursive FFT upload schedule does not match root dimensions",
                ));
            }
            let mut leaf_lengths = Vec::new();
            collect_stockham_leaf_lengths(&self.root, &mut leaf_lengths)?;
            if leaf_lengths != schedule.axis_split {
                return Err(VkFftError::InvalidKernelIr(
                    "recursive FFT Stockham leaves do not match upload schedule",
                ));
            }
        }
        if let Some(schedule) = &self.rader_forced_upload_schedule {
            schedule.validate()?;
            // A Rader multi-upload schedule may arise either from ordinary shared-capacity
            // pressure or from the later forceRaderTwoUpload promotion. The boolean records
            // only the latter cause; schedule presence records the final multi-upload fact.
            if schedule.sequence_len != self.logical_len
                || !rader_upload_split_matches_root(&self.root, &schedule.axis_split)
            {
                return Err(VkFftError::InvalidKernelIr(
                    "Rader forced upload schedule does not match recursive tree geometry",
                ));
            }
        }
        if let Some(four_step) = &self.four_step_plan {
            four_step.validate()?;
            for upload in &four_step.uploads {
                if let Some(block) = upload.axis_block {
                    block.validate(upload.transform_count, self.device)?;
                }
            }
            let matches_schedule = if let Some(schedule) = &self.stockham_upload_schedule {
                match schedule.axis_split.as_slice() {
                    [a, b]
                        if schedule.upload_count == 2
                            && four_step.uploads.len() == 2
                            && four_step.uploads[0].fft_len == *b
                            && four_step.uploads[1].fft_len == *a =>
                    {
                        root_is_two_stockham_uploads(&self.root, *a, *b)
                    }
                    [a, b, c]
                        if schedule.upload_count == 3
                            && four_step.uploads.len() == 3
                            && four_step.uploads[0].fft_len == *c
                            && four_step.uploads[1].fft_len == *b
                            && four_step.uploads[2].fft_len == *a =>
                    {
                        root_is_three_stockham_uploads(&self.root, *a, *b, *c)
                    }
                    _ => false,
                }
            } else if let Some(schedule) = &self.rader_forced_upload_schedule {
                let split_matches = match schedule.axis_split.as_slice() {
                    [a, b]
                        if schedule.upload_count == 2
                            && four_step.uploads.len() == 2
                            && four_step.uploads[0].fft_len == *b
                            && four_step.uploads[1].fft_len == *a =>
                    {
                        true
                    }
                    [a, b, c]
                        if schedule.upload_count == 3
                            && four_step.uploads.len() == 3
                            && four_step.uploads[0].fft_len == *c
                            && four_step.uploads[1].fft_len == *b
                            && four_step.uploads[2].fft_len == *a =>
                    {
                        true
                    }
                    _ => false,
                };
                let grouped_blocks_match = if self.axis0_grouped_batch_override.is_some() {
                    four_step
                        .uploads
                        .iter()
                        .all(|upload| upload.axis_block.is_some())
                } else {
                    // Exact pass-local scheduling may install an automatic physical
                    // block even without a user groupedBatch override. Its transform
                    // grouping is scheduler-owned and may exceed one; only logical
                    // ownership remains tied to the public groupedBatch surface.
                    four_step.uploads.iter().all(|upload| {
                        upload.axis_block.is_none_or(|block| {
                            block.validate(upload.transform_count, self.device).is_ok()
                        })
                    })
                };
                split_matches
                    && grouped_blocks_match
                    && rader_upload_split_matches_root(&self.root, &schedule.axis_split)
            } else {
                false
            };
            if four_step.logical_len != self.logical_len
                || four_step.batch_count != self.batch_count
                || !matches_schedule
            {
                return Err(VkFftError::InvalidKernelIr(
                    "Four-step plan does not match the scheduled recursive root",
                ));
            }
        }
        if self.axis0_grouped_batch_override.is_some()
            && self.four_step_plan.is_none()
            && let RecursiveFftNodeIr::CooleyTukey(root) = &self.root
        {
            let Some(block) = root.pack_right.axis_batch_block else {
                return Err(VkFftError::InvalidKernelIr(
                    "grouped composite recursive root is missing parent batch ownership",
                ));
            };
            if root.twiddle_transpose.axis_batch_block != Some(block)
                || root.scatter_output.axis_batch_block != Some(block)
            {
                return Err(VkFftError::InvalidKernelIr(
                    "grouped composite recursive root passes disagree on parent batch ownership",
                ));
            }
            block.validate(self.batch_count, self.device)?;
        }
        if self.root.logical_len() != self.logical_len
            || self.root.batch_count() != self.batch_count
            || self.root.direction() != self.direction
            || self.root.scalar() != self.scalar
        {
            return Err(VkFftError::InvalidKernelIr(
                "recursive FFT root metadata is inconsistent",
            ));
        }
        Ok(())
    }
}

fn retag_four_step_component_input(
    node: &mut RecursiveFftNodeIr,
    storage: ScalarType,
) -> Result<()> {
    match node {
        RecursiveFftNodeIr::Stockham(kernel) => {
            **kernel = kernel
                .as_ref()
                .clone()
                .with_external_input_storage_scalar(storage)?;
        }
        RecursiveFftNodeIr::CooleyTukey(cooley) => {
            cooley.pack_right = cooley
                .pack_right
                .clone()
                .with_external_input_storage(storage)?;
            cooley.validate()?;
        }
        RecursiveFftNodeIr::DirectRader(direct) => {
            **direct = direct
                .as_ref()
                .clone()
                .with_external_input_storage_scalar(storage)?;
        }
        RecursiveFftNodeIr::FftRader(rader) => {
            **rader = rader
                .as_ref()
                .clone()
                .with_external_input_storage_scalar(storage)?;
        }
    }
    node.validate()
}

fn retag_four_step_component_output(
    node: &mut RecursiveFftNodeIr,
    storage: ScalarType,
) -> Result<()> {
    match node {
        RecursiveFftNodeIr::Stockham(kernel) => {
            **kernel = kernel
                .as_ref()
                .clone()
                .with_external_output_storage_scalar(storage)?;
        }
        RecursiveFftNodeIr::CooleyTukey(cooley) => {
            cooley.scatter_output = cooley
                .scatter_output
                .clone()
                .with_external_output_storage(storage)?;
            cooley.validate()?;
        }
        RecursiveFftNodeIr::DirectRader(direct) => {
            **direct = direct
                .as_ref()
                .clone()
                .with_external_output_storage_scalar(storage)?;
        }
        RecursiveFftNodeIr::FftRader(rader) => {
            **rader = rader
                .as_ref()
                .clone()
                .with_external_output_storage_scalar(storage)?;
        }
    }
    node.validate()
}

fn forced_rader_stockham_schedule_for_axis_block(
    kernel: &KernelIr,
    block: StockhamAxisBlockSchedule,
    device: DeviceProfile,
) -> Result<RadixRegisterSchedule> {
    let mut schedule = if kernel.sequence_len.is_power_of_two() {
        plan_gpu_power_of_two_radix_registers(kernel.sequence_len, kernel.batch_count, 1)?
    } else {
        plan_gpu_small_mixed_radix_registers(kernel.sequence_len, kernel.batch_count)?
    };
    schedule.validate()?;
    let executable = executable_register_schedule_layout(
        &schedule,
        kernel.sequence_len,
        device.max_threads_per_block,
    )?
    .ok_or(VkFftError::UnsupportedKernelPath(
        "forced-Rader outer-upload Stockham register schedule is not executable",
    ))?;
    if executable.threads > block.threads_per_transform {
        let scale = executable
            .threads
            .div_ceil(block.threads_per_transform)
            .max(1);
        for value in schedule
            .registers_per_thread_per_radix
            .iter_mut()
            .filter(|value| **value > 0)
        {
            *value = value
                .checked_mul(scale)
                .ok_or(VkFftError::ArithmeticOverflow {
                    operation: "forced-Rader outer-upload register-table materialization",
                })?;
        }
        schedule.registers_per_thread = schedule.registers_per_thread.checked_mul(scale).ok_or(
            VkFftError::ArithmeticOverflow {
                operation: "forced-Rader outer-upload maximum-register materialization",
            },
        )?;
        schedule.min_registers_per_thread = schedule
            .min_registers_per_thread
            .checked_mul(scale)
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "forced-Rader outer-upload minimum-register materialization",
            })?;
        schedule.is_good_sequence = !(schedule.registers_per_thread > 16
            || schedule.registers_per_thread >= 2 * schedule.min_registers_per_thread);
        schedule.validate()?;
        let scaled = executable_register_schedule_layout(
            &schedule,
            kernel.sequence_len,
            device.max_threads_per_block,
        )?
        .ok_or(VkFftError::UnsupportedKernelPath(
            "scaled forced-Rader outer-upload Stockham register schedule is not executable",
        ))?;
        if scaled.threads > block.threads_per_transform {
            return Err(VkFftError::UnsupportedKernelPath(
                "scaled forced-Rader outer-upload register schedule still exceeds the physical lane budget",
            ));
        }
    }
    Ok(schedule)
}

fn apply_forced_rader_component_axis_block(
    node: &mut RecursiveFftNodeIr,
    block: StockhamAxisBlockSchedule,
    device: DeviceProfile,
) -> Result<()> {
    match node {
        RecursiveFftNodeIr::Stockham(kernel) => {
            // `VkFFTScheduler` rescales the outer upload's register table before
            // `VkFFTSplitAxisBlock`. A recursive leaf hint can therefore require more
            // lanes than the final upload block (for example Intel N512 at batch5/group3).
            // Rebuild and materialize the outer boost-1 schedule against the already
            // selected physical lane budget instead of rejecting that valid block.
            let schedule = forced_rader_stockham_schedule_for_axis_block(kernel, block, device)?;
            **kernel = kernel
                .as_ref()
                .clone()
                .with_axis0_batch_block(schedule, block, device)?;
        }
        RecursiveFftNodeIr::CooleyTukey(cooley) => {
            cooley.pack_right = cooley
                .pack_right
                .clone()
                .with_axis0_batch_block(block, device)?;
            cooley.twiddle_transpose = cooley
                .twiddle_transpose
                .clone()
                .with_axis0_batch_block(block, device)?;
            cooley.scatter_output = cooley
                .scatter_output
                .clone()
                .with_axis0_batch_block(block, device)?;
            cooley.validate()?;
        }
        RecursiveFftNodeIr::DirectRader(direct) => {
            **direct = direct
                .as_ref()
                .clone()
                .with_axis0_batch_block(block, device)?;
        }
        RecursiveFftNodeIr::FftRader(rader) => {
            **rader = rader
                .as_ref()
                .clone()
                .with_caller_axis0_batch_block(block, device)?;
        }
    }
    node.validate()
}

fn map_two_upload_component(
    node: &RecursiveFftNodeIr,
    mapping: FourStepMapping,
    upload_id: usize,
    axis_block: Option<StockhamAxisBlockSchedule>,
    device: DeviceProfile,
) -> Result<Option<RecursiveFftNodeIr>> {
    let mut mapped = node.clone();
    let physical_block = match axis_block {
        Some(block) => Some(block),
        None => forced_rader_component_exact_parent_block(&mapped, device)?,
    };
    if let Some(block) = physical_block {
        apply_forced_rader_component_axis_block(&mut mapped, block, device)?;
    }
    match &mut mapped {
        RecursiveFftNodeIr::Stockham(kernel) => {
            let io_mapping = match upload_id {
                1 => StockhamIoMapping::FourStepRight(mapping),
                0 => StockhamIoMapping::FourStepLeft(mapping),
                _ => {
                    return Err(VkFftError::InvalidKernelIr(
                        "two-upload Four-step component id is invalid",
                    ));
                }
            };
            **kernel = kernel
                .as_ref()
                .clone()
                .with_stockham_io_mapping(io_mapping)?;
            kernel
                .name
                .push_str(&format!("_four_step_upload_{upload_id}"));
        }
        RecursiveFftNodeIr::CooleyTukey(cooley) => {
            match upload_id {
                1 => {
                    cooley.pack_right = cooley.pack_right.clone().with_four_step_input_modifier(
                        CooleyTukeyInputModifier::FourStepRight(mapping),
                    )?;
                    cooley.scatter_output = cooley
                        .scatter_output
                        .clone()
                        .with_four_step_output_modifier(
                            CooleyTukeyOutputModifier::FourStepRight(mapping),
                        )?;
                }
                0 => {
                    cooley.scatter_output = cooley
                        .scatter_output
                        .clone()
                        .with_four_step_output_modifier(CooleyTukeyOutputModifier::FourStepLeft(
                            mapping,
                        ))?;
                }
                _ => {
                    return Err(VkFftError::InvalidKernelIr(
                        "two-upload Four-step component id is invalid",
                    ));
                }
            }
            cooley.validate()?;
        }
        RecursiveFftNodeIr::DirectRader(direct) => {
            let io_mapping = match upload_id {
                1 => StockhamIoMapping::FourStepRight(mapping),
                0 => StockhamIoMapping::FourStepLeft(mapping),
                _ => {
                    return Err(VkFftError::InvalidKernelIr(
                        "two-upload Four-step direct-Rader component id is invalid",
                    ));
                }
            };
            **direct = direct
                .as_ref()
                .clone()
                .with_stockham_io_mapping(io_mapping)?;
            direct
                .name
                .push_str(&format!("_four_step_upload_{upload_id}"));
        }
        RecursiveFftNodeIr::FftRader(rader) => {
            let io_mapping = match upload_id {
                1 => StockhamIoMapping::FourStepRight(mapping),
                0 => StockhamIoMapping::FourStepLeft(mapping),
                _ => {
                    return Err(VkFftError::InvalidKernelIr(
                        "two-upload Four-step FFT-Rader component id is invalid",
                    ));
                }
            };
            **rader = rader
                .as_ref()
                .clone()
                .with_stockham_io_mapping(io_mapping)?;
        }
    }
    mapped.validate()?;
    Ok(Some(mapped))
}

fn map_three_upload_component(
    node: &RecursiveFftNodeIr,
    mapping: ThreeUploadFourStepMapping,
    upload_id: usize,
    axis_block: Option<StockhamAxisBlockSchedule>,
    device: DeviceProfile,
) -> Result<Option<RecursiveFftNodeIr>> {
    let mut mapped = node.clone();
    let physical_block = match axis_block {
        Some(block) => Some(block),
        None => forced_rader_component_exact_parent_block(&mapped, device)?,
    };
    if let Some(block) = physical_block {
        apply_forced_rader_component_axis_block(&mut mapped, block, device)?;
    }
    match &mut mapped {
        RecursiveFftNodeIr::Stockham(kernel) => {
            let io_mapping = match upload_id {
                2 => StockhamIoMapping::FourStepThreeUpload2(mapping),
                1 => StockhamIoMapping::FourStepThreeUpload1(mapping),
                0 => StockhamIoMapping::FourStepThreeUpload0(mapping),
                _ => {
                    return Err(VkFftError::InvalidKernelIr(
                        "three-upload Four-step component id is invalid",
                    ));
                }
            };
            **kernel = kernel
                .as_ref()
                .clone()
                .with_stockham_io_mapping(io_mapping)?;
            kernel
                .name
                .push_str(&format!("_four_step_upload_{upload_id}"));
        }
        RecursiveFftNodeIr::CooleyTukey(cooley) => {
            match upload_id {
                2 => {
                    cooley.pack_right = cooley.pack_right.clone().with_four_step_input_modifier(
                        CooleyTukeyInputModifier::FourStepThreeUpload2(mapping),
                    )?;
                    cooley.scatter_output = cooley
                        .scatter_output
                        .clone()
                        .with_four_step_output_modifier(
                            CooleyTukeyOutputModifier::FourStepThreeUpload2(mapping),
                        )?;
                }
                1 => {
                    cooley.scatter_output = cooley
                        .scatter_output
                        .clone()
                        .with_four_step_output_modifier(
                            CooleyTukeyOutputModifier::FourStepThreeUpload1(mapping),
                        )?;
                }
                0 => {
                    cooley.scatter_output = cooley
                        .scatter_output
                        .clone()
                        .with_four_step_output_modifier(
                            CooleyTukeyOutputModifier::FourStepThreeUpload0(mapping),
                        )?;
                }
                _ => {
                    return Err(VkFftError::InvalidKernelIr(
                        "three-upload Four-step component id is invalid",
                    ));
                }
            }
            cooley.validate()?;
        }
        RecursiveFftNodeIr::DirectRader(direct) => {
            let io_mapping = match upload_id {
                2 => StockhamIoMapping::FourStepThreeUpload2(mapping),
                1 => StockhamIoMapping::FourStepThreeUpload1(mapping),
                0 => StockhamIoMapping::FourStepThreeUpload0(mapping),
                _ => {
                    return Err(VkFftError::InvalidKernelIr(
                        "three-upload Four-step direct-Rader component id is invalid",
                    ));
                }
            };
            **direct = direct
                .as_ref()
                .clone()
                .with_stockham_io_mapping(io_mapping)?;
            direct
                .name
                .push_str(&format!("_four_step_upload_{upload_id}"));
        }
        RecursiveFftNodeIr::FftRader(rader) => {
            let io_mapping = match upload_id {
                2 => StockhamIoMapping::FourStepThreeUpload2(mapping),
                1 => StockhamIoMapping::FourStepThreeUpload1(mapping),
                0 => StockhamIoMapping::FourStepThreeUpload0(mapping),
                _ => {
                    return Err(VkFftError::InvalidKernelIr(
                        "three-upload Four-step FFT-Rader component id is invalid",
                    ));
                }
            };
            **rader = rader
                .as_ref()
                .clone()
                .with_stockham_io_mapping(io_mapping)?;
        }
    }
    mapped.validate()?;
    Ok(Some(mapped))
}

fn root_is_two_stockham_uploads(
    root: &RecursiveFftNodeIr,
    left_len: usize,
    right_len: usize,
) -> bool {
    let RecursiveFftNodeIr::CooleyTukey(node) = root else {
        return false;
    };
    node.left_len == left_len
        && node.right_len == right_len
        && matches!(&node.left, RecursiveFftNodeIr::Stockham(kernel) if kernel.sequence_len == left_len)
        && matches!(&node.right, RecursiveFftNodeIr::Stockham(kernel) if kernel.sequence_len == right_len)
}

fn root_is_three_stockham_uploads(
    root: &RecursiveFftNodeIr,
    low_len: usize,
    middle_len: usize,
    high_len: usize,
) -> bool {
    let RecursiveFftNodeIr::CooleyTukey(root) = root else {
        return false;
    };
    let RecursiveFftNodeIr::CooleyTukey(upper) = &root.right else {
        return false;
    };
    root.left_len == low_len
        && root.right_len == middle_len * high_len
        && upper.left_len == middle_len
        && upper.right_len == high_len
        && matches!(&root.left, RecursiveFftNodeIr::Stockham(kernel) if kernel.sequence_len == low_len)
        && matches!(&upper.left, RecursiveFftNodeIr::Stockham(kernel) if kernel.sequence_len == middle_len)
        && matches!(&upper.right, RecursiveFftNodeIr::Stockham(kernel) if kernel.sequence_len == high_len)
}

#[derive(Debug, Clone, Copy)]
struct FourStepBuildContext {
    precision: Precision,
    complex_bytes: usize,
    perform_zero_padding: bool,
    grouped_batch_override: Option<usize>,
    tuning: PlannerTuning,
    device: DeviceProfile,
}

fn exact_rader_component_parent_threads_with_max_batch_coalesced(
    node: &RecursiveFftNodeIr,
    device: DeviceProfile,
    max_batch_coalesced: usize,
    shrink_first_upload_coalescing: bool,
    tuning: PlannerTuning,
) -> Result<Option<usize>> {
    if max_batch_coalesced == 0 {
        return Ok(None);
    }
    if let RecursiveFftNodeIr::FftRader(rader) = node {
        return plan_gpu_axis0_multi_fft_rader_threads_with_pass_context_and_tuning(
            rader.prime,
            &[(rader.prime, 1)],
            node.batch_count(),
            max_batch_coalesced,
            shrink_first_upload_coalescing,
            tuning,
            device,
        );
    }
    if !matches!(node, RecursiveFftNodeIr::CooleyTukey(_)) {
        return Ok(None);
    }
    let mut direct = Vec::new();
    let mut fft = Vec::new();
    node.collect_rader_prime_multiplicities(&mut direct, &mut fft);
    if direct.is_empty() && fft.is_empty() {
        return Ok(None);
    }
    if !direct.is_empty() && fft.is_empty() {
        plan_gpu_axis0_composite_direct_rader_threads_for_prime_multiplicities_with_max_batch_coalesced(
            node.logical_len(),
            &direct,
            node.batch_count(),
            max_batch_coalesced,
            device,
        )
    } else if direct.is_empty() && !fft.is_empty() {
        plan_gpu_axis0_multi_fft_rader_threads_with_pass_context_and_tuning(
            node.logical_len(),
            &fft,
            node.batch_count(),
            max_batch_coalesced,
            shrink_first_upload_coalescing,
            tuning,
            device,
        )
    } else {
        plan_gpu_axis0_mixed_direct_multi_fft_rader_threads_with_pass_context(
            node.logical_len(),
            &direct,
            &fft,
            node.batch_count(),
            max_batch_coalesced,
            shrink_first_upload_coalescing,
            tuning,
            device,
        )
    }
}

fn exact_rader_component_parent_threads(
    node: &RecursiveFftNodeIr,
    device: DeviceProfile,
) -> Result<Option<usize>> {
    if !matches!(node, RecursiveFftNodeIr::CooleyTukey(_)) {
        return Ok(None);
    }
    let mut direct = Vec::new();
    let mut fft = Vec::new();
    node.collect_rader_prime_multiplicities(&mut direct, &mut fft);
    if direct.is_empty() && fft.is_empty() {
        return Ok(None);
    }
    let sequence_len = node.logical_len();
    let batch_count = node.batch_count();
    if !direct.is_empty() && fft.is_empty() {
        plan_gpu_axis0_composite_direct_rader_threads_for_prime_multiplicities(
            sequence_len,
            &direct,
            batch_count,
            device,
        )
    } else if direct.is_empty() && !fft.is_empty() {
        plan_gpu_axis0_multi_fft_rader_threads_for_prime_multiplicities(
            sequence_len,
            &fft,
            batch_count,
            device,
        )
    } else if !direct.is_empty() && !fft.is_empty() {
        plan_gpu_axis0_mixed_direct_multi_fft_rader_threads(
            sequence_len,
            &direct,
            &fft,
            batch_count,
            device,
        )
    } else {
        Ok(None)
    }
}

fn forced_rader_component_exact_parent_block(
    node: &RecursiveFftNodeIr,
    device: DeviceProfile,
) -> Result<Option<StockhamAxisBlockSchedule>> {
    let Some(threads_per_transform) = exact_rader_component_parent_threads(node, device)? else {
        return Ok(None);
    };
    let block = StockhamAxisBlockSchedule {
        threads_per_transform,
        grouped_batch: 1,
        transforms_on_x: false,
        axis_swapped: false,
        local_size_x: threads_per_transform,
        local_size_y: 1,
    };
    block.validate(node.batch_count(), device)?;
    Ok(Some(block))
}

fn forced_rader_component_threads(
    node: &RecursiveFftNodeIr,
    device: DeviceProfile,
) -> Result<Option<usize>> {
    match node {
        RecursiveFftNodeIr::Stockham(kernel) => {
            Ok(Some(kernel.workgroup_grouping.threads_per_transform.max(1)))
        }
        RecursiveFftNodeIr::CooleyTukey(cooley) => {
            if let Some(threads) = exact_rader_component_parent_threads(node, device)? {
                Ok(Some(threads))
            } else {
                Ok((cooley.pack_right.axis_batch_block.is_none())
                    .then_some(cooley.pack_right.workgroup_size.x as usize))
            }
        }
        RecursiveFftNodeIr::DirectRader(direct) => Ok(Some(
            direct
                .axis_batch_block
                .map_or(direct.prime.div_ceil(2), |block| {
                    block.threads_per_transform
                }),
        )),
        RecursiveFftNodeIr::FftRader(rader) => {
            Ok(Some(rader.caller_axis_batch_block.map_or(
                rader.scatter.workgroup_size.x as usize,
                |block| block.threads_per_transform,
            )))
        }
    }
}

fn higher_axis_forced_rader_component_threads(
    node: &RecursiveFftNodeIr,
    max_batch_coalesced: usize,
    device: DeviceProfile,
) -> Result<Option<usize>> {
    if matches!(
        node,
        RecursiveFftNodeIr::DirectRader(_) | RecursiveFftNodeIr::FftRader(_)
    ) {
        return forced_rader_component_threads(node, device);
    }
    if matches!(node, RecursiveFftNodeIr::CooleyTukey(_)) {
        let mut direct = Vec::new();
        let mut fft = Vec::new();
        node.collect_rader_prime_multiplicities(&mut direct, &mut fft);
        if !direct.is_empty() || !fft.is_empty() {
            return forced_rader_component_threads(node, device);
        }
    }
    // The recursive Stockham/Cooley execution node may have been materialized under a
    // different local leaf context. VkFFTSplitAxisBlock instead uses the outer upload's
    // own register table with registerBoost4Step=1, so rebuild that schedule from the
    // upload logical length and RHS count rather than inheriting `scheduler_hint`.
    let fft_len = node.logical_len();
    let rhs_transform_count = node.batch_count();
    let planned = if fft_len.is_power_of_two() {
        plan_gpu_power_of_two_radix_registers(fft_len, rhs_transform_count, 1)
    } else {
        plan_gpu_small_mixed_radix_registers(fft_len, rhs_transform_count)
    };
    let schedule = match planned {
        Ok(schedule) => Some(schedule),
        Err(VkFftError::UnsupportedKernelPath(_)) => None,
        Err(error) => return Err(error),
    };
    if let Some(schedule) = schedule {
        schedule.validate()?;
        if schedule.fft_len == node.logical_len()
            && schedule.min_registers_per_thread > 0
            && schedule.register_boost > 0
        {
            if max_batch_coalesced == 0 {
                return Ok(None);
            }
            let denominator = schedule
                .min_registers_per_thread
                .checked_mul(schedule.register_boost)
                .and_then(|value| value.checked_mul(device.max_threads_per_block))
                .ok_or(VkFftError::ArithmeticOverflow {
                    operation: "higher-axis forced-Rader upload register scaling denominator",
                })?;
            let coalesced_elements = node.logical_len().checked_mul(max_batch_coalesced).ok_or(
                VkFftError::ArithmeticOverflow {
                    operation: "higher-axis forced-Rader upload coalesced element pressure",
                },
            )?;
            let scale_registers_num = if coalesced_elements
                > schedule
                    .min_registers_per_thread
                    .checked_mul(schedule.register_boost)
                    .and_then(|value| value.checked_mul(device.max_threads_per_block))
                    .ok_or(VkFftError::ArithmeticOverflow {
                        operation: "higher-axis forced-Rader upload pressure threshold",
                    })? {
                coalesced_elements.div_ceil(denominator).max(1)
            } else {
                1
            };
            let effective_min_registers = schedule
                .min_registers_per_thread
                .checked_mul(scale_registers_num)
                .ok_or(VkFftError::ArithmeticOverflow {
                    operation: "higher-axis forced-Rader upload scaled minimum registers",
                })?;
            return Ok(Some(
                node.logical_len()
                    .div_ceil(effective_min_registers)
                    .checked_div(schedule.register_boost)
                    .unwrap_or(0)
                    .max(1),
            ));
        }
    }
    forced_rader_component_threads(node, device)
}

fn forced_rader_component_for_upload<'a>(
    root: &'a RecursiveFftNodeIr,
    schedule: &RaderUploadSchedule,
    axis_upload_id: usize,
) -> Option<&'a RecursiveFftNodeIr> {
    match schedule.axis_split.as_slice() {
        [a, b] if schedule.upload_count == 2 => {
            let RecursiveFftNodeIr::CooleyTukey(cooley) = root else {
                return None;
            };
            match axis_upload_id {
                0 if cooley.left.logical_len() == *a => Some(&cooley.left),
                1 if cooley.right.logical_len() == *b => Some(&cooley.right),
                _ => None,
            }
        }
        [a, b, c] if schedule.upload_count == 3 => {
            let RecursiveFftNodeIr::CooleyTukey(cooley) = root else {
                return None;
            };
            let RecursiveFftNodeIr::CooleyTukey(upper) = &cooley.right else {
                return None;
            };
            match axis_upload_id {
                0 if cooley.left.logical_len() == *a => Some(&cooley.left),
                1 if upper.left.logical_len() == *b => Some(&upper.left),
                2 if upper.right.logical_len() == *c => Some(&upper.right),
                _ => None,
            }
        }
        _ => None,
    }
}

fn forced_rader_component_physical_context(
    node: &RecursiveFftNodeIr,
    complex_bytes: usize,
    device: DeviceProfile,
) -> Result<(usize, bool)> {
    let mut direct = Vec::new();
    let mut fft = Vec::new();
    node.collect_rader_prime_multiplicities(&mut direct, &mut fft);
    let reserve = direct
        .iter()
        .map(|(prime, _)| prime.saturating_sub(1))
        .max()
        .unwrap_or(0)
        .checked_mul(complex_bytes)
        .ok_or(VkFftError::ArithmeticOverflow {
            operation: "higher-axis forced-Rader direct-prime shared reserve",
        })?;
    Ok((
        device.shared_memory_bytes.saturating_sub(reserve),
        !direct.is_empty(),
    ))
}

fn forced_rader_component_axis_block(
    node: &RecursiveFftNodeIr,
    schedule: &RaderUploadSchedule,
    axis_upload_id: usize,
    stage_start_size: usize,
    transform_count: usize,
    batch_count: usize,
    context: FourStepBuildContext,
) -> Result<Option<StockhamAxisBlockSchedule>> {
    let max_batch_coalesced =
        (upstream_coalesced_memory_bytes_for_precision(context.device, context.precision)
            / context.complex_bytes)
            .max(1);
    let exact_coalesced_threads = exact_rader_component_parent_threads_with_max_batch_coalesced(
        node,
        context.device,
        max_batch_coalesced,
        axis_upload_id == 0,
        context.tuning,
    )?;
    if context.grouped_batch_override.is_none() {
        let threads_per_transform = if let Some(threads) = exact_coalesced_threads {
            threads
        } else if matches!(
            node,
            RecursiveFftNodeIr::Stockham(_)
                | RecursiveFftNodeIr::DirectRader(_)
                | RecursiveFftNodeIr::FftRader(_)
        ) {
            let Some(threads) = forced_rader_component_threads(node, context.device)? else {
                return Ok(None);
            };
            threads
        } else {
            return Ok(None);
        };
        let request = FourStepAxisBlockRequest {
            axis_upload_id,
            stage_start_size,
            transform_count,
            outer_batch_count: batch_count,
            perform_zero_padding: context.perform_zero_padding,
            grouped_batch_override: None,
        };
        let mut direct_rader_primes = Vec::new();
        let mut fft_rader_primes = Vec::new();
        node.collect_rader_prime_multiplicities(&mut direct_rader_primes, &mut fft_rader_primes);
        let max_direct_rader_prime = direct_rader_primes.iter().map(|(prime, _)| *prime).max();
        let block = if let Some(direct_rader_prime) = max_direct_rader_prime {
            plan_gpu_axis0_direct_rader_four_step_default_block_from_shape_for_precision(
                schedule.upload_count,
                node.logical_len(),
                threads_per_transform,
                direct_rader_prime,
                request,
                context.precision,
                context.complex_bytes,
                context.device,
            )?
        } else {
            plan_gpu_axis0_four_step_default_block_from_shape_for_precision(
                schedule.upload_count,
                node.logical_len(),
                threads_per_transform,
                request,
                context.precision,
                context.complex_bytes,
                context.device,
            )?
        };
        if let Some(block) = block {
            return Ok(Some(block));
        }
        let transforms_on_x = axis_upload_id > 0;
        let (local_size_x, local_size_y) = if transforms_on_x {
            (1, threads_per_transform)
        } else {
            (threads_per_transform, 1)
        };
        let block = StockhamAxisBlockSchedule {
            threads_per_transform,
            grouped_batch: 1,
            transforms_on_x,
            axis_swapped: false,
            local_size_x,
            local_size_y,
        };
        block.validate(node.batch_count(), context.device)?;
        return Ok(Some(block));
    }
    let threads_per_transform = if let Some(threads) = exact_coalesced_threads {
        threads
    } else {
        forced_rader_component_threads(node, context.device)?.ok_or(
            VkFftError::UnsupportedKernelPath(
                "grouped forced-Rader Four-step requires an executable upload component",
            ),
        )?
    };
    let request = FourStepAxisBlockRequest {
        axis_upload_id,
        stage_start_size,
        transform_count,
        outer_batch_count: batch_count,
        perform_zero_padding: context.perform_zero_padding,
        grouped_batch_override: context.grouped_batch_override,
    };
    let mut direct_rader_primes = Vec::new();
    let mut fft_rader_primes = Vec::new();
    node.collect_rader_prime_multiplicities(&mut direct_rader_primes, &mut fft_rader_primes);
    let block = if let Some(direct_rader_prime) =
        direct_rader_primes.iter().map(|(prime, _)| *prime).max()
    {
        plan_gpu_axis0_direct_rader_four_step_grouped_block_from_shape_for_precision(
            schedule.upload_count,
            node.logical_len(),
            threads_per_transform,
            direct_rader_prime,
            request,
            context.precision,
            context.complex_bytes,
            context.device,
        )?
    } else {
        plan_gpu_axis0_four_step_grouped_block_from_shape_for_precision(
            schedule.upload_count,
            node.logical_len(),
            threads_per_transform,
            request,
            context.precision,
            context.complex_bytes,
            context.device,
        )?
    };
    if block.is_none() {
        return Err(VkFftError::UnsupportedKernelPath(
            "grouped forced-Rader Four-step component cannot satisfy the requested axis block",
        ));
    }
    Ok(block)
}

fn build_forced_rader_four_step_plan(
    schedule: &RaderUploadSchedule,
    root: &RecursiveFftNodeIr,
    logical_len: usize,
    batch_count: usize,
    context: FourStepBuildContext,
) -> Result<Option<FourStepPlanIr>> {
    schedule.validate()?;
    if schedule.sequence_len != logical_len
        || !rader_upload_split_matches_root(root, &schedule.axis_split)
    {
        return Ok(None);
    }
    let uploads = match schedule.axis_split.as_slice() {
        [a, b] if schedule.upload_count == 2 => {
            let first_transform_count =
                batch_count
                    .checked_mul(*a)
                    .ok_or(VkFftError::ArithmeticOverflow {
                        operation: "forced Rader Four-step right upload transform count",
                    })?;
            let second_transform_count =
                batch_count
                    .checked_mul(*b)
                    .ok_or(VkFftError::ArithmeticOverflow {
                        operation: "forced Rader Four-step left upload transform count",
                    })?;
            let RecursiveFftNodeIr::CooleyTukey(cooley) = root else {
                return Ok(None);
            };
            let first_axis_block = forced_rader_component_axis_block(
                &cooley.right,
                schedule,
                1,
                *a,
                first_transform_count,
                batch_count,
                context,
            )?;
            let second_axis_block = forced_rader_component_axis_block(
                &cooley.left,
                schedule,
                0,
                1,
                second_transform_count,
                batch_count,
                context,
            )?;
            vec![
                FourStepUploadIr {
                    axis_upload_id: 1,
                    fft_len: *b,
                    stage_start_size: *a,
                    transform_count: first_transform_count,
                    axis_block: first_axis_block,
                    input_layout: FourStepInputLayout::NaturalStrided,
                    output_layout: FourStepOutputLayout::TwiddleTransposed,
                    twiddle: FourStepTwiddlePlacement::Write,
                },
                FourStepUploadIr {
                    axis_upload_id: 0,
                    fft_len: *a,
                    stage_start_size: 1,
                    transform_count: second_transform_count,
                    axis_block: second_axis_block,
                    input_layout: FourStepInputLayout::TransposedContiguous,
                    output_layout: FourStepOutputLayout::NaturalFrequency,
                    twiddle: FourStepTwiddlePlacement::None,
                },
            ]
        }
        [a, b, c] if schedule.upload_count == 3 => {
            let ab = a.checked_mul(*b).ok_or(VkFftError::ArithmeticOverflow {
                operation: "forced Rader three-upload A*B product",
            })?;
            let upload2_transforms =
                batch_count
                    .checked_mul(ab)
                    .ok_or(VkFftError::ArithmeticOverflow {
                        operation: "forced Rader three-upload upload-2 transform count",
                    })?;
            let upload1_transforms = batch_count
                .checked_mul(*c)
                .and_then(|value| value.checked_mul(*a))
                .ok_or(VkFftError::ArithmeticOverflow {
                    operation: "forced Rader three-upload upload-1 transform count",
                })?;
            let upload0_transforms = batch_count
                .checked_mul(*c)
                .and_then(|value| value.checked_mul(*b))
                .ok_or(VkFftError::ArithmeticOverflow {
                    operation: "forced Rader three-upload upload-0 transform count",
                })?;
            let RecursiveFftNodeIr::CooleyTukey(cooley) = root else {
                return Ok(None);
            };
            let RecursiveFftNodeIr::CooleyTukey(upper) = &cooley.right else {
                return Ok(None);
            };
            let upload2_axis_block = forced_rader_component_axis_block(
                &upper.right,
                schedule,
                2,
                ab,
                upload2_transforms,
                batch_count,
                context,
            )?;
            let upload1_axis_block = forced_rader_component_axis_block(
                &upper.left,
                schedule,
                1,
                *a,
                upload1_transforms,
                batch_count,
                context,
            )?;
            let upload0_axis_block = forced_rader_component_axis_block(
                &cooley.left,
                schedule,
                0,
                1,
                upload0_transforms,
                batch_count,
                context,
            )?;
            vec![
                FourStepUploadIr {
                    axis_upload_id: 2,
                    fft_len: *c,
                    stage_start_size: ab,
                    transform_count: upload2_transforms,
                    axis_block: upload2_axis_block,
                    input_layout: FourStepInputLayout::NaturalStrided,
                    output_layout: FourStepOutputLayout::TwiddleTransposed,
                    twiddle: FourStepTwiddlePlacement::Write,
                },
                FourStepUploadIr {
                    axis_upload_id: 1,
                    fft_len: *b,
                    stage_start_size: *a,
                    transform_count: upload1_transforms,
                    axis_block: upload1_axis_block,
                    input_layout: FourStepInputLayout::TransposedContiguous,
                    output_layout: FourStepOutputLayout::TwiddleTransposed,
                    twiddle: FourStepTwiddlePlacement::Write,
                },
                FourStepUploadIr {
                    axis_upload_id: 0,
                    fft_len: *a,
                    stage_start_size: 1,
                    transform_count: upload0_transforms,
                    axis_block: upload0_axis_block,
                    input_layout: FourStepInputLayout::TransposedContiguous,
                    output_layout: FourStepOutputLayout::NaturalFrequency,
                    twiddle: FourStepTwiddlePlacement::None,
                },
            ]
        }
        _ => return Ok(None),
    };
    let plan = FourStepPlanIr {
        logical_len,
        batch_count,
        reorder_four_step: true,
        uploads,
    };
    plan.validate()?;
    Ok(Some(plan))
}

fn build_two_upload_four_step_plan(
    schedule: &StockhamUploadSchedule,
    root: &RecursiveFftNodeIr,
    logical_len: usize,
    batch_count: usize,
    context: FourStepBuildContext,
) -> Result<Option<FourStepPlanIr>> {
    let FourStepBuildContext {
        precision,
        complex_bytes,
        perform_zero_padding,
        grouped_batch_override,
        tuning: _,
        device,
    } = context;
    if schedule.upload_count != 2 || schedule.axis_split.len() != 2 {
        return Ok(None);
    }
    let left_len = schedule.axis_split[0];
    let right_len = schedule.axis_split[1];
    if !root_is_two_stockham_uploads(root, left_len, right_len) {
        return Ok(None);
    }
    let first_transform_count =
        batch_count
            .checked_mul(left_len)
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "Four-step right upload transform count",
            })?;
    let second_transform_count =
        batch_count
            .checked_mul(right_len)
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "Four-step left upload transform count",
            })?;
    let first_axis_block = plan_gpu_axis0_four_step_upload_block_with_grouped_batch_for_precision(
        schedule,
        FourStepAxisBlockRequest {
            axis_upload_id: 1,
            stage_start_size: left_len,
            transform_count: first_transform_count,
            outer_batch_count: batch_count,
            perform_zero_padding,
            grouped_batch_override,
        },
        precision,
        complex_bytes,
        device,
    )?;
    let second_axis_block = plan_gpu_axis0_four_step_upload_block_with_grouped_batch_for_precision(
        schedule,
        FourStepAxisBlockRequest {
            axis_upload_id: 0,
            stage_start_size: 1,
            transform_count: second_transform_count,
            outer_batch_count: batch_count,
            perform_zero_padding,
            grouped_batch_override,
        },
        precision,
        complex_bytes,
        device,
    )?;
    if grouped_batch_override.is_some()
        && (first_axis_block.is_none() || second_axis_block.is_none())
    {
        return Err(VkFftError::UnsupportedKernelPath(
            "groupedBatch override requires fully executable grouped two-upload Four-step workgroups",
        ));
    }
    let plan = FourStepPlanIr {
        logical_len,
        batch_count,
        reorder_four_step: true,
        uploads: vec![
            FourStepUploadIr {
                axis_upload_id: 1,
                fft_len: right_len,
                stage_start_size: left_len,
                transform_count: first_transform_count,
                axis_block: first_axis_block,
                input_layout: FourStepInputLayout::NaturalStrided,
                output_layout: FourStepOutputLayout::TwiddleTransposed,
                twiddle: FourStepTwiddlePlacement::Write,
            },
            FourStepUploadIr {
                axis_upload_id: 0,
                fft_len: left_len,
                stage_start_size: 1,
                transform_count: second_transform_count,
                axis_block: second_axis_block,
                input_layout: FourStepInputLayout::TransposedContiguous,
                output_layout: FourStepOutputLayout::NaturalFrequency,
                twiddle: FourStepTwiddlePlacement::None,
            },
        ],
    };
    plan.validate()?;
    Ok(Some(plan))
}

fn build_three_upload_four_step_plan(
    schedule: &StockhamUploadSchedule,
    root: &RecursiveFftNodeIr,
    logical_len: usize,
    batch_count: usize,
    context: FourStepBuildContext,
) -> Result<Option<FourStepPlanIr>> {
    let FourStepBuildContext {
        precision,
        complex_bytes,
        perform_zero_padding,
        grouped_batch_override,
        tuning: _,
        device,
    } = context;
    let [a, b, c]: [usize; 3] = match schedule.axis_split.as_slice().try_into() {
        Ok(split) if schedule.upload_count == 3 => split,
        _ => return Ok(None),
    };
    if !root_is_three_stockham_uploads(root, a, b, c) {
        return Ok(None);
    }
    let ab = a.checked_mul(b).ok_or(VkFftError::ArithmeticOverflow {
        operation: "three-upload Four-step stageStartSize",
    })?;
    let upload2_transforms = batch_count
        .checked_mul(ab)
        .ok_or(VkFftError::ArithmeticOverflow {
            operation: "three-upload Four-step upload-2 transform count",
        })?;
    let upload1_transforms = batch_count
        .checked_mul(c)
        .and_then(|value| value.checked_mul(a))
        .ok_or(VkFftError::ArithmeticOverflow {
            operation: "three-upload Four-step upload-1 transform count",
        })?;
    let upload0_transforms = batch_count
        .checked_mul(c)
        .and_then(|value| value.checked_mul(b))
        .ok_or(VkFftError::ArithmeticOverflow {
            operation: "three-upload Four-step upload-0 transform count",
        })?;
    let upload2_axis_block =
        plan_gpu_axis0_four_step_upload_block_with_grouped_batch_for_precision(
            schedule,
            FourStepAxisBlockRequest {
                axis_upload_id: 2,
                stage_start_size: ab,
                transform_count: upload2_transforms,
                outer_batch_count: batch_count,
                perform_zero_padding,
                grouped_batch_override,
            },
            precision,
            complex_bytes,
            device,
        )?;
    let upload1_axis_block =
        plan_gpu_axis0_four_step_upload_block_with_grouped_batch_for_precision(
            schedule,
            FourStepAxisBlockRequest {
                axis_upload_id: 1,
                stage_start_size: a,
                transform_count: upload1_transforms,
                outer_batch_count: batch_count,
                perform_zero_padding,
                grouped_batch_override,
            },
            precision,
            complex_bytes,
            device,
        )?;
    let upload0_axis_block =
        plan_gpu_axis0_four_step_upload_block_with_grouped_batch_for_precision(
            schedule,
            FourStepAxisBlockRequest {
                axis_upload_id: 0,
                stage_start_size: 1,
                transform_count: upload0_transforms,
                outer_batch_count: batch_count,
                perform_zero_padding,
                grouped_batch_override,
            },
            precision,
            complex_bytes,
            device,
        )?;
    if grouped_batch_override.is_some()
        && (upload2_axis_block.is_none()
            || upload1_axis_block.is_none()
            || upload0_axis_block.is_none())
    {
        return Err(VkFftError::UnsupportedKernelPath(
            "groupedBatch override requires fully executable grouped three-upload Four-step workgroups",
        ));
    }
    let plan = FourStepPlanIr {
        logical_len,
        batch_count,
        reorder_four_step: true,
        uploads: vec![
            FourStepUploadIr {
                axis_upload_id: 2,
                fft_len: c,
                stage_start_size: ab,
                transform_count: upload2_transforms,
                axis_block: upload2_axis_block,
                input_layout: FourStepInputLayout::NaturalStrided,
                output_layout: FourStepOutputLayout::TwiddleTransposed,
                twiddle: FourStepTwiddlePlacement::Write,
            },
            FourStepUploadIr {
                axis_upload_id: 1,
                fft_len: b,
                stage_start_size: a,
                transform_count: upload1_transforms,
                axis_block: upload1_axis_block,
                input_layout: FourStepInputLayout::TransposedContiguous,
                output_layout: FourStepOutputLayout::TwiddleTransposed,
                twiddle: FourStepTwiddlePlacement::Write,
            },
            FourStepUploadIr {
                axis_upload_id: 0,
                fft_len: a,
                stage_start_size: 1,
                transform_count: upload0_transforms,
                axis_block: upload0_axis_block,
                input_layout: FourStepInputLayout::TransposedContiguous,
                output_layout: FourStepOutputLayout::NaturalFrequency,
                twiddle: FourStepTwiddlePlacement::None,
            },
        ],
    };
    plan.validate()?;
    Ok(Some(plan))
}

fn gpu_smooth_upload_factors(
    sequence_len: usize,
    batch_count: usize,
    precision: Precision,
    device: DeviceProfile,
    leaf_limit: usize,
    axis_context: StockhamUploadAxisContext,
) -> Result<Option<(Vec<FactorSpec>, StockhamUploadSchedule)>> {
    if !has_specialized_gpu_scheduler_policy(device) || sequence_len <= 1 {
        return Ok(None);
    }
    let Ok(schedule) = plan_gpu_smooth_stockham_uploads_for_batches_with_axis_context(
        sequence_len,
        batch_count,
        precision,
        device,
        axis_context,
    ) else {
        return Ok(None);
    };
    for (&factor, radix_schedule) in schedule.axis_split.iter().zip(&schedule.radix_schedules) {
        if !scheduled_stockham_leaf_fits(
            factor,
            radix_schedule,
            leaf_limit,
            scalar_from_precision(precision)?,
            device,
        )? {
            return Ok(None);
        }
    }
    let factors = schedule
        .axis_split
        .iter()
        .copied()
        .map(|len| FactorSpec {
            len,
            kind: FactorKind::Stockham,
        })
        .collect::<Vec<_>>();
    Ok(Some((factors, schedule)))
}

fn scheduled_stockham_leaf_fits(
    factor: usize,
    schedule: &RadixRegisterSchedule,
    ping_pong_leaf_limit: usize,
    scalar: ScalarType,
    device: DeviceProfile,
) -> Result<bool> {
    if factor <= ping_pong_leaf_limit {
        return Ok(true);
    }
    let Some(layout) =
        executable_register_schedule_layout(schedule, factor, device.max_threads_per_block)?
    else {
        return Ok(false);
    };
    let shared_layout =
        plan_gpu_stockham_shared_memory_layout(factor, layout.shared_elements, scalar, device)?;
    let required_shared = shared_layout
        .allocated_elements
        .checked_mul(scalar.complex_bytes())
        .ok_or(VkFftError::ArithmeticOverflow {
            operation: "scheduler Stockham register-shared leaf bytes",
        })?;
    Ok(required_shared <= device.shared_memory_bytes)
}

fn scalar_from_precision(precision: Precision) -> Result<ScalarType> {
    match precision {
        Precision::F32 | Precision::F16StorageF32Compute => Ok(ScalarType::F32),
        Precision::F64 | Precision::F64ComputeF32Storage => Ok(ScalarType::F64),
        other => Err(VkFftError::UnsupportedPrecision {
            backend: "recursive scheduler leaf fit",
            precision: precision_name(other),
        }),
    }
}

fn collect_stockham_leaf_lengths(node: &RecursiveFftNodeIr, output: &mut Vec<usize>) -> Result<()> {
    match node {
        RecursiveFftNodeIr::Stockham(ir) => output.push(ir.sequence_len),
        RecursiveFftNodeIr::CooleyTukey(ir) => {
            collect_stockham_leaf_lengths(&ir.left, output)?;
            collect_stockham_leaf_lengths(&ir.right, output)?;
        }
        RecursiveFftNodeIr::DirectRader(_) | RecursiveFftNodeIr::FftRader(_) => {
            return Err(VkFftError::InvalidKernelIr(
                "Stockham upload schedule cannot contain Rader leaves",
            ));
        }
    }
    Ok(())
}

fn preserve_factor_rader_mode(plan: &mut FftPlan, factor: &FactorSpec) -> Result<()> {
    let FactorKind::Rader(expected_mode) = &factor.kind else {
        return Ok(());
    };
    let axis = plan.axes.first_mut().ok_or(VkFftError::InvalidKernelIr(
        "recursive Rader leaf plan is missing axis metadata",
    ))?;
    let AxisAlgorithm::Rader { stockham, primes } = &mut axis.algorithm else {
        return Err(VkFftError::InvalidKernelIr(
            "recursive Rader leaf plan lost its Rader algorithm",
        ));
    };
    if !stockham.prime_factors.is_empty()
        || !stockham.merged_radices.is_empty()
        || primes.len() != 1
        || primes[0].prime != factor.len
        || primes[0].multiplicity != 1
    {
        return Err(VkFftError::InvalidKernelIr(
            "recursive Rader leaf plan is not a standalone prime",
        ));
    }
    primes[0].mode = expected_mode.clone();
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn build_node(
    factors: &[FactorSpec],
    batch_count: usize,
    container_fft_num: usize,
    component_factor_counts: Option<&[usize]>,
    grouped_batch_override: Option<usize>,
    zero_padding: Option<ZeroPaddingRange>,
    direction: Direction,
    precision: Precision,
    normalize_inverse: bool,
    tuning: crate::config::PlannerTuning,
    device: DeviceProfile,
) -> Result<RecursiveFftNodeIr> {
    if factors.len() == 1 {
        let factor = factors[0].clone();
        let mut config = FftConfig::new(vec![factor.len])
            .with_batch_count(batch_count)
            .with_precision(precision)
            .with_inverse_normalization(normalize_inverse)
            .with_tuning(tuning);
        if let Some(grouped_batch) = grouped_batch_override {
            config = config.with_grouped_batch(0, grouped_batch)?;
        }
        if let Some(range) = zero_padding {
            config = config.with_zero_padding(0, range.left, range.right)?;
        }
        let mut plan = FftPlan::build(config)?;
        preserve_factor_rader_mode(&mut plan, &factor)?;
        return match factor.kind {
            FactorKind::Stockham => Ok(RecursiveFftNodeIr::Stockham(Box::new(
                KernelIr::stockham_1d(&plan, direction, device)?,
            ))),
            FactorKind::Rader(RaderMode::DirectMultiplication) => {
                Ok(RecursiveFftNodeIr::DirectRader(Box::new(
                    RaderDirectIr::build(&plan, direction, device)?,
                )))
            }
            FactorKind::Rader(RaderMode::FftConvolution { .. }) => {
                let pipeline = if container_fft_num > 1 {
                    let outer_fft_len = factor.len.checked_mul(container_fft_num).ok_or(
                        VkFftError::ArithmeticOverflow {
                            operation: "recursive nested-Rader outer/container product",
                        },
                    )?;
                    RaderFftPipelineIr::build_with_container_context(
                        &plan,
                        direction,
                        device,
                        outer_fft_len,
                        container_fft_num,
                    )?
                } else {
                    RaderFftPipelineIr::build(&plan, direction, device)?
                };
                Ok(RecursiveFftNodeIr::FftRader(Box::new(pipeline)))
            }
        };
    }

    // VkFFT's multi-upload/Four-step scheduler prefers approximately balanced
    // decompositions (sqrt/cuberoot neighborhoods) rather than a linear chain of
    // tiny factors. Keep the correctness-first recursive IR backend-neutral, but
    // choose the contiguous factor cut that minimizes the larger child product.
    // This reduces tree depth/temporary reshapes while preserving factor order.
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
                "forced recursive FFT component factor counts are inconsistent",
            ));
        }
        None => balanced_factor_split_index(factors)?,
    };
    let left_len = checked_factor_product(&factors[..split])?;
    let right_len = checked_factor_product(&factors[split..])?;
    let logical_len = left_len
        .checked_mul(right_len)
        .ok_or(VkFftError::ArithmeticOverflow {
            operation: "recursive Cooley-Tukey node length",
        })?;
    let right_batch = batch_count
        .checked_mul(left_len)
        .ok_or(VkFftError::ArithmeticOverflow {
            operation: "recursive right-child batch count",
        })?;
    let left_batch = batch_count
        .checked_mul(right_len)
        .ok_or(VkFftError::ArithmeticOverflow {
            operation: "recursive left-child batch count",
        })?;
    let right_container_fft_num =
        container_fft_num
            .checked_mul(left_len)
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "recursive right-child Rader container count",
            })?;
    let left_container_fft_num =
        container_fft_num
            .checked_mul(right_len)
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "recursive left-child Rader container count",
            })?;
    let right = build_node(
        &factors[split..],
        right_batch,
        right_container_fft_num,
        component_factor_counts.and_then(|counts| (counts.len() > 2).then_some(&counts[1..])),
        None,
        None,
        direction,
        precision,
        normalize_inverse,
        tuning,
        device,
    )?;
    let left = build_node(
        &factors[..split],
        left_batch,
        left_container_fft_num,
        None,
        None,
        None,
        direction,
        precision,
        normalize_inverse,
        tuning,
        device,
    )?;
    let direction_name = match direction {
        Direction::Forward => "forward",
        Direction::Inverse => "inverse",
    };
    let shape = CooleyTukeyPassShape {
        scalar: left.scalar(),
        direction,
        logical_len,
        left_len,
        right_len,
        batch_count,
        device,
    };
    let node = RecursiveCooleyTukeyIr {
        logical_len,
        left_len,
        right_len,
        batch_count,
        direction,
        scalar: left.scalar(),
        pack_right: CooleyTukeyPassIr::new(
            format!("vkfft_recursive_pack_{logical_len}_{direction_name}"),
            shape,
            CooleyTukeyPassOperation::PackRightInput,
        )?,
        right,
        twiddle_transpose: CooleyTukeyPassIr::new(
            format!("vkfft_recursive_twiddle_{logical_len}_{direction_name}"),
            shape,
            CooleyTukeyPassOperation::TwiddleTranspose,
        )?,
        left,
        scatter_output: CooleyTukeyPassIr::new(
            format!("vkfft_recursive_scatter_{logical_len}_{direction_name}"),
            shape,
            CooleyTukeyPassOperation::ScatterOutput,
        )?,
    };
    node.validate()?;
    Ok(RecursiveFftNodeIr::CooleyTukey(Box::new(node)))
}

fn balanced_factor_split_index(factors: &[FactorSpec]) -> Result<usize> {
    if factors.len() < 2 {
        return Err(VkFftError::InvalidKernelIr(
            "balanced recursive split requires at least two factors",
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
                operation: "balanced recursive FFT left factor product",
            })?;
        let right = total / left;
        let larger_child = left.max(right);
        let difference = left.abs_diff(right);
        if larger_child < best_larger_child
            || (larger_child == best_larger_child && difference < best_difference)
        {
            best_larger_child = larger_child;
            best_difference = difference;
            best_index = index + 1;
        }
    }
    Ok(best_index)
}

fn stockham_leaf_element_limit(device: DeviceProfile, scalar: ScalarType) -> Result<usize> {
    let bytes_per_element =
        scalar
            .complex_bytes()
            .checked_mul(2)
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "recursive Stockham leaf shared-memory bytes per element",
            })?;
    let limit = device.shared_memory_bytes / bytes_per_element;
    if limit == 0 {
        return Err(VkFftError::ResourceLimitExceeded {
            resource: "shared memory for recursive Stockham leaf",
            required: bytes_per_element,
            available: device.shared_memory_bytes,
        });
    }
    Ok(limit)
}

fn rader_factor_specs_for_upload_split(
    stockham_prime_factors: &[usize],
    rader_primes: &[RaderPrimePlan],
    axis_split: &[usize],
    leaf_limit: usize,
    scalar: ScalarType,
    device: DeviceProfile,
) -> Result<(Vec<FactorSpec>, Vec<usize>)> {
    if !matches!(axis_split.len(), 2 | 3) || axis_split.contains(&0) {
        return Err(VkFftError::InvalidKernelIr(
            "Rader forced upload split must contain two or three non-zero factors",
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
                "Rader forced upload split requires non-trivial factors",
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
                    push_stockham_chunks(
                        &mut output,
                        &pending_stockham,
                        leaf_limit,
                        scalar,
                        device,
                    )?;
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
                    "upstream Rader upload split contains a factor absent from the planner axis",
                ));
            }
        }
        if !pending_stockham.is_empty() {
            push_stockham_chunks(&mut output, &pending_stockham, leaf_limit, scalar, device)?;
        }
        if checked_factor_product(&output[component_start..])? != component {
            return Err(VkFftError::InvalidKernelIr(
                "Rader upload factor bin does not match its scheduled axis split",
            ));
        }
        let factor_count = output.len() - component_start;
        if factor_count == 0 {
            return Err(VkFftError::InvalidKernelIr(
                "Rader upload factor bin is empty",
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
            "Rader upload factor bins did not consume the complete planner factorization",
        ));
    }
    if component_factor_counts.iter().sum::<usize>() != output.len() {
        return Err(VkFftError::InvalidKernelIr(
            "Rader upload factor bins do not cover the recursive factor list",
        ));
    }
    Ok((output, component_factor_counts))
}

fn rader_upload_split_matches_root(node: &RecursiveFftNodeIr, axis_split: &[usize]) -> bool {
    if axis_split.len() < 2 {
        return false;
    }
    let Some(right_len) = axis_split[1..]
        .iter()
        .try_fold(1usize, |product, factor| product.checked_mul(*factor))
    else {
        return false;
    };
    let RecursiveFftNodeIr::CooleyTukey(root) = node else {
        return false;
    };
    if root.left_len != axis_split[0] || root.right_len != right_len {
        return false;
    }
    axis_split.len() == 2 || rader_upload_split_matches_root(&root.right, &axis_split[1..])
}

fn push_stockham_chunks(
    output: &mut Vec<FactorSpec>,
    prime_factors: &[usize],
    leaf_limit: usize,
    scalar: ScalarType,
    device: DeviceProfile,
) -> Result<()> {
    if prime_factors.is_empty() {
        return Ok(());
    }

    let mut current = 1usize;
    for &factor in prime_factors {
        if !stockham_chunk_fits(factor, leaf_limit, scalar, device)? {
            return Err(VkFftError::ResourceLimitExceeded {
                resource: "recursive Stockham leaf elements",
                required: factor,
                available: leaf_limit,
            });
        }
        let combined = current
            .checked_mul(factor)
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "recursive Stockham chunk product",
            })?;
        if current > 1 && !stockham_chunk_fits(combined, leaf_limit, scalar, device)? {
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

fn stockham_chunk_fits(
    factor: usize,
    ping_pong_leaf_limit: usize,
    scalar: ScalarType,
    device: DeviceProfile,
) -> Result<bool> {
    if factor <= ping_pong_leaf_limit {
        return Ok(true);
    }
    if !has_specialized_gpu_scheduler_policy(device) || factor.is_power_of_two() {
        return Ok(false);
    }
    let precision = match scalar {
        ScalarType::F16 => {
            return Err(VkFftError::InvalidKernelIr(
                "binary16 cannot be used as a recursive FFT compute scalar",
            ));
        }
        ScalarType::F32 => Precision::F32,
        ScalarType::F64 => Precision::F64,
        ScalarType::DoubleDouble => {
            return Err(VkFftError::UnsupportedPrecision {
                backend: "recursive FFT scheduler",
                precision: "double-double recursive scheduling is not implemented",
            });
        }
    };
    let upload = match plan_gpu_smooth_stockham_uploads_for_batches(factor, 1, precision, device) {
        Ok(schedule) => schedule,
        Err(VkFftError::UnsupportedKernelPath(_)) => return Ok(false),
        Err(error) => return Err(error),
    };
    if upload.upload_count != 1 || upload.axis_split.as_slice() != [factor] {
        return Ok(false);
    }
    let schedule = upload
        .radix_schedules
        .first()
        .ok_or(VkFftError::InvalidKernelIr(
            "single-upload smooth scheduler omitted its radix schedule",
        ))?;
    scheduled_stockham_leaf_fits(factor, schedule, ping_pong_leaf_limit, scalar, device)
}

fn checked_factor_product(factors: &[FactorSpec]) -> Result<usize> {
    factors.iter().try_fold(1usize, |acc, factor| {
        acc.checked_mul(factor.len)
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "recursive FFT factor product",
            })
    })
}

pub fn execute_recursive_fft_ir(
    ir: &RecursiveFftIr,
    input: &[Complex64],
) -> Result<Vec<Complex64>> {
    ir.validate()?;
    let prepared;
    let effective_input = if let Some(pass) = &ir.zero_pad_pass
        && pass.operation.is_input_boundary()
    {
        prepared = execute_zero_pad_pass(pass, input)?;
        prepared.as_slice()
    } else {
        input
    };
    let output = execute_recursive_fft_ir_with_resources(ir, effective_input, None, None)?;
    if let Some(pass) = &ir.zero_pad_pass
        && pass.operation.is_output_boundary()
    {
        return execute_zero_pad_pass(pass, &output);
    }
    Ok(output)
}

/// Execute a recursive FFT whose root reshape may consume an immutable lookup table
/// and/or an auxiliary external input. This is the recursive counterpart of
/// `execute_stockham_ir_with_resources` and is intentionally limited to root-boundary
/// modifiers; child FFT nodes keep their ordinary contiguous contracts.
pub(crate) fn execute_recursive_fft_ir_with_resources(
    ir: &RecursiveFftIr,
    input: &[Complex64],
    lookup: Option<&[Complex64]>,
    auxiliary: Option<&[Complex64]>,
) -> Result<Vec<Complex64>> {
    ir.validate()?;
    let expected = match &ir.root {
        RecursiveFftNodeIr::CooleyTukey(root) => match root.pack_right.input_modifier {
            CooleyTukeyInputModifier::RaderGeneratorReverse(mapping) => mapping
                .prime
                .checked_mul(ir.batch_count)
                .ok_or(VkFftError::ArithmeticOverflow {
                    operation: "recursive Rader generator input element count",
                })?,
            CooleyTukeyInputModifier::RealEvenPack(mapping) => mapping
                .full_len
                .checked_mul(ir.batch_count)
                .ok_or(VkFftError::ArithmeticOverflow {
                    operation: "recursive even-real pack input element count",
                })?,
            CooleyTukeyInputModifier::RealEvenInversePreprocess(mapping) => mapping
                .compact_len()
                .checked_mul(ir.batch_count)
                .ok_or(VkFftError::ArithmeticOverflow {
                    operation: "recursive even-real inverse input element count",
                })?,
            CooleyTukeyInputModifier::None
            | CooleyTukeyInputModifier::MultiplyLookupTable
            | CooleyTukeyInputModifier::FourStepRight(_)
            | CooleyTukeyInputModifier::FourStepThreeUpload2(_) => ir
                .logical_len
                .checked_mul(ir.batch_count)
                .ok_or(VkFftError::ArithmeticOverflow {
                    operation: "recursive FFT input element count",
                })?,
        },
        _ => ir
            .logical_len
            .checked_mul(ir.batch_count)
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "recursive FFT input element count",
            })?,
    };
    if input.len() != expected {
        return Err(VkFftError::InputLengthMismatch {
            expected,
            actual: input.len(),
        });
    }
    if let Some(uploads) = ir.four_step_stockham_upload_kernels()? {
        if lookup.is_some() || auxiliary.is_some() {
            return Err(VkFftError::UnsupportedKernelPath(
                "Four-step recursive execution does not accept root boundary resources",
            ));
        }
        let mut values = input.to_vec();
        for upload in uploads {
            values = execute_stockham_ir(&upload, &values)?;
        }
        Ok(values)
    } else if let Some(uploads) = ir.four_step_rader_upload_nodes()? {
        if lookup.is_some() || auxiliary.is_some() {
            return Err(VkFftError::UnsupportedKernelPath(
                "Rader Four-step recursive execution does not accept root boundary resources",
            ));
        }
        let mut values = input.to_vec();
        for upload in uploads {
            values = execute_node(&upload, &values)?;
        }
        Ok(values)
    } else {
        match &ir.root {
            RecursiveFftNodeIr::CooleyTukey(root) => {
                execute_cooley_tukey_with_resources(root, input, lookup, auxiliary)
            }
            _ if lookup.is_none() && auxiliary.is_none() => execute_node(&ir.root, input),
            _ => Err(VkFftError::UnsupportedKernelPath(
                "recursive boundary resources require a Cooley-Tukey root",
            )),
        }
    }
}

fn execute_node(node: &RecursiveFftNodeIr, input: &[Complex64]) -> Result<Vec<Complex64>> {
    node.validate()?;
    let expected = node.logical_len().checked_mul(node.batch_count()).ok_or(
        VkFftError::ArithmeticOverflow {
            operation: "recursive FFT node input element count",
        },
    )?;
    if input.len() != expected {
        return Err(VkFftError::InputLengthMismatch {
            expected,
            actual: input.len(),
        });
    }
    match node {
        RecursiveFftNodeIr::Stockham(ir) => execute_stockham_ir(ir, input),
        RecursiveFftNodeIr::DirectRader(ir) => execute_rader_direct_ir(ir, input),
        RecursiveFftNodeIr::FftRader(ir) => execute_rader_fft_ir(ir, input),
        RecursiveFftNodeIr::CooleyTukey(ir) => execute_cooley_tukey(ir, input),
    }
}

fn execute_cooley_tukey(
    ir: &RecursiveCooleyTukeyIr,
    input: &[Complex64],
) -> Result<Vec<Complex64>> {
    execute_cooley_tukey_with_resources(ir, input, None, None)
}

fn execute_cooley_tukey_with_resources(
    ir: &RecursiveCooleyTukeyIr,
    input: &[Complex64],
    lookup: Option<&[Complex64]>,
    auxiliary: Option<&[Complex64]>,
) -> Result<Vec<Complex64>> {
    let total = ir.logical_len * ir.batch_count;
    let generator_permutation = match ir.pack_right.input_modifier {
        CooleyTukeyInputModifier::RaderGeneratorReverse(mapping) => {
            Some(mapping.permutation(ir.logical_len)?)
        }
        _ => None,
    };
    let lookup = match ir.pack_right.input_modifier {
        CooleyTukeyInputModifier::MultiplyLookupTable => {
            let lookup = lookup.ok_or(VkFftError::InvalidKernelIr(
                "recursive Cooley-Tukey LUT input fusion requires a lookup table",
            ))?;
            if lookup.len() != ir.logical_len {
                return Err(VkFftError::InputLengthMismatch {
                    expected: ir.logical_len,
                    actual: lookup.len(),
                });
            }
            Some(lookup)
        }
        _ => None,
    };
    let mut right_input = vec![Complex64::new(0.0, 0.0); total];
    for batch in 0..ir.batch_count {
        for n1 in 0..ir.left_len {
            let transform = batch * ir.left_len + n1;
            let destination_base = transform * ir.right_len;
            for n2 in 0..ir.right_len {
                let logical_source = n1 + ir.left_len * n2;
                let value = match ir.pack_right.input_modifier {
                    CooleyTukeyInputModifier::None => {
                        input[batch * ir.logical_len + logical_source]
                    }
                    CooleyTukeyInputModifier::RaderGeneratorReverse(mapping) => {
                        let permutation =
                            generator_permutation
                                .as_ref()
                                .ok_or(VkFftError::InvalidKernelIr(
                                    "recursive Rader generator permutation is missing",
                                ))?;
                        let exponent = (ir.logical_len - logical_source) % ir.logical_len;
                        input[batch * mapping.prime + permutation[exponent]]
                    }
                    CooleyTukeyInputModifier::MultiplyLookupTable => {
                        input[batch * ir.logical_len + logical_source]
                            * lookup.ok_or(VkFftError::InvalidKernelIr(
                                "recursive Cooley-Tukey lookup table is missing",
                            ))?[logical_source]
                    }
                    CooleyTukeyInputModifier::FourStepRight(mapping) => {
                        let outer_batch = batch / mapping.left_len;
                        let n1 = batch % mapping.left_len;
                        input[outer_batch * mapping.logical_len
                            + n1
                            + mapping.left_len * logical_source]
                    }
                    CooleyTukeyInputModifier::FourStepThreeUpload2(mapping) => {
                        let [a, b, _c] = mapping.axis_split;
                        let ab = a * b;
                        let outer_batch = batch / ab;
                        let n12 = batch % ab;
                        input[outer_batch * mapping.logical_len + n12 + ab * logical_source]
                    }
                    CooleyTukeyInputModifier::RealEvenPack(mapping) => {
                        let real_base = batch * mapping.full_len;
                        Complex64::new(
                            input[real_base + 2 * logical_source].re,
                            input[real_base + 2 * logical_source + 1].re,
                        )
                    }
                    CooleyTukeyInputModifier::RealEvenInversePreprocess(mapping) => {
                        let compact_base = batch * mapping.compact_len();
                        let x = input[compact_base + logical_source];
                        let mirrored = input[compact_base + ir.logical_len - logical_source].conj();
                        let w_conj =
                            Complex64::exp_i(TAU * logical_source as f64 / mapping.full_len as f64);
                        let rotated = w_conj * (x - mirrored);
                        let i_rotated = Complex64::new(-rotated.im, rotated.re);
                        (x + mirrored + i_rotated).scale(if mapping.normalize { 0.5 } else { 1.0 })
                    }
                };
                right_input[destination_base + n2] = value;
            }
        }
    }
    let right_output = execute_node(&ir.right, &right_input)?;

    let sign = ir.direction.exponent_sign();
    let mut left_input = vec![Complex64::new(0.0, 0.0); total];
    for batch in 0..ir.batch_count {
        for k2 in 0..ir.right_len {
            let destination_base = (batch * ir.right_len + k2) * ir.left_len;
            for n1 in 0..ir.left_len {
                let source = (batch * ir.left_len + n1) * ir.right_len + k2;
                let angle = sign * TAU * (n1 * k2) as f64 / ir.logical_len as f64;
                left_input[destination_base + n1] = right_output[source] * Complex64::exp_i(angle);
            }
        }
    }
    let left_output = execute_node(&ir.left, &left_input)?;
    let logical_value = |batch: usize, local_index: usize| {
        let k2 = local_index % ir.right_len;
        let k1 = local_index / ir.right_len;
        left_output[(batch * ir.right_len + k2) * ir.left_len + k1]
    };

    match ir.scatter_output.output_modifier {
        CooleyTukeyOutputModifier::None => {
            let mut output = vec![Complex64::new(0.0, 0.0); total];
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
        CooleyTukeyOutputModifier::FourStepRight(mapping) => {
            let mut output = vec![Complex64::new(0.0, 0.0); total];
            let sign = ir.direction.exponent_sign();
            for batch in 0..ir.batch_count {
                let outer_batch = batch / mapping.left_len;
                let n1 = batch % mapping.left_len;
                for local_index in 0..ir.logical_len {
                    let output_index =
                        (outer_batch * mapping.right_len + local_index) * mapping.left_len + n1;
                    let angle = sign * TAU * (n1 * local_index) as f64 / mapping.logical_len as f64;
                    output[output_index] =
                        logical_value(batch, local_index) * Complex64::exp_i(angle);
                }
            }
            Ok(output)
        }
        CooleyTukeyOutputModifier::FourStepLeft(mapping) => {
            let mut output = vec![Complex64::new(0.0, 0.0); total];
            for batch in 0..ir.batch_count {
                let outer_batch = batch / mapping.right_len;
                let k2 = batch % mapping.right_len;
                for local_index in 0..ir.logical_len {
                    let output_index =
                        outer_batch * mapping.logical_len + k2 + mapping.right_len * local_index;
                    output[output_index] = logical_value(batch, local_index);
                }
            }
            Ok(output)
        }
        CooleyTukeyOutputModifier::FourStepThreeUpload2(mapping) => {
            let [a, b, c] = mapping.axis_split;
            let ab = a * b;
            let sign = ir.direction.exponent_sign();
            let mut output = vec![Complex64::new(0.0, 0.0); total];
            for batch in 0..ir.batch_count {
                let outer_batch = batch / ab;
                let n12 = batch % ab;
                let n1 = n12 % a;
                let n2 = n12 / a;
                for k3 in 0..ir.logical_len {
                    let output_index = (((outer_batch * c + k3) * a + n1) * b) + n2;
                    let angle = sign * TAU * (n12 * k3) as f64 / mapping.logical_len as f64;
                    output[output_index] = logical_value(batch, k3) * Complex64::exp_i(angle);
                }
            }
            Ok(output)
        }
        CooleyTukeyOutputModifier::FourStepThreeUpload1(mapping) => {
            let [a, b, _c] = mapping.axis_split;
            let sign = ir.direction.exponent_sign();
            let mut output = vec![Complex64::new(0.0, 0.0); total];
            for batch in 0..ir.batch_count {
                let group = batch / a;
                let n1 = batch % a;
                for k2 in 0..ir.logical_len {
                    let output_index = (group * b + k2) * a + n1;
                    let angle = sign * TAU * (n1 * k2) as f64 / (a * b) as f64;
                    output[output_index] = logical_value(batch, k2) * Complex64::exp_i(angle);
                }
            }
            Ok(output)
        }
        CooleyTukeyOutputModifier::FourStepThreeUpload0(mapping) => {
            let [_a, b, c] = mapping.axis_split;
            let mut output = vec![Complex64::new(0.0, 0.0); total];
            for batch in 0..ir.batch_count {
                let group = batch / b;
                let k2 = batch % b;
                let outer_batch = group / c;
                let k3 = group % c;
                for k1 in 0..ir.logical_len {
                    let output_index = outer_batch * mapping.logical_len + k3 + c * k2 + c * b * k1;
                    output[output_index] = logical_value(batch, k1);
                }
            }
            Ok(output)
        }
        CooleyTukeyOutputModifier::RealEvenPostprocess(mapping) => {
            let compact_len =
                ir.logical_len
                    .checked_add(1)
                    .ok_or(VkFftError::ArithmeticOverflow {
                        operation: "recursive even-real compact output length",
                    })?;
            let expected_output =
                compact_len
                    .checked_mul(ir.batch_count)
                    .ok_or(VkFftError::ArithmeticOverflow {
                        operation: "recursive even-real postprocess output element count",
                    })?;
            let mut output = vec![Complex64::new(0.0, 0.0); expected_output];
            let logical_value = |batch: usize, k: usize| {
                let k2 = k % ir.right_len;
                let k1 = k / ir.right_len;
                left_output[(batch * ir.right_len + k2) * ir.left_len + k1]
            };
            for batch in 0..ir.batch_count {
                let output_base = batch * compact_len;
                let z0 = logical_value(batch, 0);
                output[output_base] = Complex64::new(z0.re + z0.im, 0.0);
                output[output_base + ir.logical_len] = Complex64::new(z0.re - z0.im, 0.0);
                for k in 1..ir.logical_len {
                    let a = logical_value(batch, k);
                    let b = logical_value(batch, ir.logical_len - k).conj();
                    let w = Complex64::exp_i(-TAU * k as f64 / mapping.full_len as f64);
                    let rotated = w * (a - b);
                    output[output_base + k] = Complex64::new(
                        0.5 * (a.re + b.re + rotated.im),
                        0.5 * (a.im + b.im - rotated.re),
                    );
                }
            }
            Ok(output)
        }
        CooleyTukeyOutputModifier::RealEvenUnpack(mapping) => {
            let expected_output = mapping.full_len.checked_mul(ir.batch_count).ok_or(
                VkFftError::ArithmeticOverflow {
                    operation: "recursive even-real unpack output element count",
                },
            )?;
            let mut output = vec![Complex64::new(0.0, 0.0); expected_output];
            for batch in 0..ir.batch_count {
                let real_base = batch * mapping.full_len;
                for k2 in 0..ir.right_len {
                    let source_base = (batch * ir.right_len + k2) * ir.left_len;
                    for k1 in 0..ir.left_len {
                        let logical_output = k2 + ir.right_len * k1;
                        let value = left_output[source_base + k1];
                        output[real_base + 2 * logical_output] = Complex64::new(value.re, 0.0);
                        output[real_base + 2 * logical_output + 1] = Complex64::new(value.im, 0.0);
                    }
                }
            }
            Ok(output)
        }
        CooleyTukeyOutputModifier::RaderScatter(mapping) => {
            let auxiliary = auxiliary.ok_or(VkFftError::InvalidKernelIr(
                "recursive Cooley-Tukey Rader scatter requires auxiliary prime input",
            ))?;
            let expected_aux = mapping.prime.checked_mul(ir.batch_count).ok_or(
                VkFftError::ArithmeticOverflow {
                    operation: "recursive Rader scatter auxiliary element count",
                },
            )?;
            if auxiliary.len() != expected_aux {
                return Err(VkFftError::InputLengthMismatch {
                    expected: expected_aux,
                    actual: auxiliary.len(),
                });
            }
            let permutation = mapping.permutation(ir.logical_len)?;
            let mut output = vec![Complex64::new(0.0, 0.0); expected_aux];
            let scale = if mapping.normalize_prime {
                1.0 / mapping.prime as f64
            } else {
                1.0
            };
            for batch in 0..ir.batch_count {
                let prime_base = batch * mapping.prime;
                let dc = auxiliary[prime_base..prime_base + mapping.prime]
                    .iter()
                    .copied()
                    .fold(Complex64::new(0.0, 0.0), |sum, value| sum + value);
                output[prime_base] = dc.scale(scale);
                let x0 = auxiliary[prime_base];
                for k2 in 0..ir.right_len {
                    let source_base = (batch * ir.right_len + k2) * ir.left_len;
                    for k1 in 0..ir.left_len {
                        let logical_output = k2 + ir.right_len * k1;
                        output[prime_base + permutation[logical_output]] =
                            (x0 + left_output[source_base + k1]).scale(scale);
                    }
                }
            }
            Ok(output)
        }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{Backend, GpuVendor};
    use crate::reference::dft;

    fn device() -> DeviceProfile {
        DeviceProfile {
            shared_memory_bytes: 128 * 1024,
            shared_memory_pow2_bytes: 128 * 1024,
            ..DeviceProfile::generic(Backend::Vulkan, GpuVendor::Nvidia)
        }
    }

    fn sample(length: usize) -> Vec<Complex64> {
        (0..length)
            .map(|index| {
                let x = index as f64;
                Complex64::new(
                    (0.041 * x).sin() + x * 0.0007,
                    (0.029 * x).cos() - x * 0.0011,
                )
            })
            .collect()
    }

    fn max_error(lhs: &[Complex64], rhs: &[Complex64]) -> f64 {
        lhs.iter()
            .zip(rhs)
            .map(|(lhs, rhs)| (*lhs - *rhs).norm_sqr().sqrt())
            .fold(0.0, f64::max)
    }

    fn small_shared_memory_device() -> DeviceProfile {
        DeviceProfile {
            shared_memory_bytes: 1024,
            shared_memory_pow2_bytes: 1024,
            ..DeviceProfile::generic(Backend::Vulkan, GpuVendor::Nvidia)
        }
    }

    fn upstream_scheduler_device() -> DeviceProfile {
        DeviceProfile {
            shared_memory_bytes: 48 * 1024,
            shared_memory_pow2_bytes: 48 * 1024,
            max_threads_per_block: 1024,
            ..DeviceProfile::generic(Backend::Vulkan, GpuVendor::Nvidia)
        }
    }

    fn register_leaf_device() -> DeviceProfile {
        DeviceProfile {
            shared_memory_bytes: 32 * 1024,
            shared_memory_pow2_bytes: 32 * 1024,
            max_threads_per_block: 1024,
            ..DeviceProfile::generic(Backend::Vulkan, GpuVendor::Nvidia)
        }
    }

    fn tiny_shared_memory_device() -> DeviceProfile {
        DeviceProfile {
            // F32 Stockham uses two complex shared buffers: 4 elements * 2 * 8 B.
            shared_memory_bytes: 64,
            shared_memory_pow2_bytes: 64,
            ..DeviceProfile::generic(Backend::Vulkan, GpuVendor::Nvidia)
        }
    }

    fn assert_stockham_leaves_fit(node: &RecursiveFftNodeIr, max_len: usize) {
        match node {
            RecursiveFftNodeIr::Stockham(ir) => assert!(ir.sequence_len <= max_len),
            RecursiveFftNodeIr::DirectRader(_) | RecursiveFftNodeIr::FftRader(_) => {}
            RecursiveFftNodeIr::CooleyTukey(ir) => {
                assert_stockham_leaves_fit(&ir.left, max_len);
                assert_stockham_leaves_fit(&ir.right, max_len);
            }
        }
    }

    #[test]
    fn amd_vulkan_large_stockham_consumes_three_upload_policy() {
        let profile = DeviceProfile {
            shared_memory_bytes: 64 * 1024,
            shared_memory_pow2_bytes: 64 * 1024,
            max_threads_per_block: 1024,
            ..DeviceProfile::generic(Backend::Vulkan, GpuVendor::Amd)
        };
        let length = 1_048_576usize;
        let plan = FftPlan::build(FftConfig::new(vec![length])).unwrap();
        let ir = RecursiveFftIr::build(&plan, Direction::Forward, profile).unwrap();
        let upload = ir
            .stockham_upload_schedule
            .as_ref()
            .expect("AMD Vulkan should use the policy-driven upload scheduler");
        assert_eq!(upload.upload_count, 3);
        assert_eq!(upload.axis_split.iter().product::<usize>(), length);
        let four_step = ir
            .four_step_plan
            .as_ref()
            .expect("three scheduler uploads should materialize Four-step IR");
        assert_eq!(four_step.uploads.len(), 3);
        assert_eq!(four_step.logical_len, length);
    }

    #[test]
    fn nested_sub_rader_inherits_power_of_two_container_context() {
        let plan = FftPlan::build(
            FftConfig::new(vec![107])
                .with_tuning(crate::PlannerTuning::portable().with_recursive_fft_rader(true)),
        )
        .unwrap();
        let ir = RecursiveFftIr::build(&plan, Direction::Forward, device()).unwrap();
        let RecursiveFftNodeIr::FftRader(outer) = &ir.root else {
            panic!("107 should materialize as an outer FFT-Rader leaf");
        };
        assert_eq!(outer.prime, 107);
        assert_eq!(
            outer.input_strategy,
            crate::RaderFftInputStrategy::GeneratorOrderRecursive
        );
        let RecursiveFftNodeIr::CooleyTukey(convolution) = &outer.forward_recursive().unwrap().root
        else {
            panic!("107-point Rader convolution should split 106 as 2 x 53");
        };
        assert_eq!((convolution.left_len, convolution.right_len), (2, 53));
        let RecursiveFftNodeIr::FftRader(sub_rader) = &convolution.right else {
            panic!("the 53-point child should be a nested FFT-Rader leaf");
        };
        let schedule = sub_rader
            .internal_register_schedule
            .as_ref()
            .expect("nested p53 should inherit the two-container register schedule");
        assert_eq!(schedule.prime, 53);
        assert_eq!(schedule.outer_fft_len, 106);
        assert_eq!(schedule.container_fft_num, 2);
        assert_eq!(schedule.execution_container_fft_num, 2);
        assert_eq!(schedule.internal_fft.stage_radices, vec![13, 4]);
        for fft in [
            sub_rader.forward_recursive().unwrap(),
            sub_rader.inverse_recursive().unwrap(),
        ] {
            let RecursiveFftNodeIr::Stockham(kernel) = &fft.root else {
                panic!("p53 nested convolution should retain a Stockham root");
            };
            assert_eq!(kernel.workgroup_grouping.transforms_per_workgroup, 2);
        }

        let input = sample(107);
        let actual = execute_recursive_fft_ir(&ir, &input).unwrap();
        let expected = dft(&input, Direction::Forward, false);
        assert!(max_error(&actual, &expected) < 2.0e-8 * 107.0);
    }

    #[test]
    fn nested_direct_rader_subcontainer_composite_uses_recursive_parent_floor() {
        let length = 2usize * 283;
        let batch_count = 2usize;
        let tuning = crate::PlannerTuning::portable().with_recursive_fft_rader(true);
        let plan = FftPlan::build(
            FftConfig::new(vec![length])
                .with_batch_count(batch_count)
                .with_tuning(tuning),
        )
        .unwrap();
        let AxisAlgorithm::Rader { primes, .. } = &plan.axes[0].algorithm else {
            panic!("N566 should retain p283 recursive Rader metadata");
        };
        assert_eq!(primes.len(), 1);
        assert_eq!(primes[0].prime, 283);
        assert!(matches!(primes[0].mode, RaderMode::FftConvolution { .. }));

        let ir = RecursiveFftIr::build(&plan, Direction::Forward, device()).unwrap();
        assert!(ir.four_step_plan.is_none());
        assert!(ir.rader_forced_upload_schedule.is_none());
        let RecursiveFftNodeIr::CooleyTukey(root) = &ir.root else {
            panic!("N566 should keep smooth-2 x p283 Cooley root");
        };
        let block = root.pack_right.axis_batch_block.unwrap();
        // The generic test profile is capped at 256 threads, so the same recursive
        // p283/p47 state enters scaleRegistersNum=2 and reaches the exact 142-lane
        // narrow-device parent. A 1024-thread profile stays at 283 lanes.
        assert_eq!(block.threads_per_transform, 142);
        assert_eq!(block.grouped_batch, 1);
        assert_eq!([block.local_size_x, block.local_size_y], [142, 1]);
        assert_eq!(root.twiddle_transpose.axis_batch_block, Some(block));
        assert_eq!(root.scatter_output.axis_batch_block, Some(block));

        let outer = match (&root.left, &root.right) {
            (RecursiveFftNodeIr::FftRader(rader), _) if rader.prime == 283 => rader,
            (_, RecursiveFftNodeIr::FftRader(rader)) if rader.prime == 283 => rader,
            _ => panic!("N566 should contain outer p283 FFT-Rader child"),
        };
        fn contains_direct_prime(node: &RecursiveFftNodeIr, prime: usize) -> bool {
            match node {
                RecursiveFftNodeIr::DirectRader(direct) => direct.prime == prime,
                RecursiveFftNodeIr::CooleyTukey(cooley) => {
                    contains_direct_prime(&cooley.left, prime)
                        || contains_direct_prime(&cooley.right, prime)
                }
                RecursiveFftNodeIr::Stockham(_) | RecursiveFftNodeIr::FftRader(_) => false,
            }
        }
        assert!(contains_direct_prime(
            &outer.forward_recursive().unwrap().root,
            47
        ));
        assert!(contains_direct_prime(
            &outer.inverse_recursive().unwrap().root,
            47
        ));

        let shaders = crate::backend::vulkan::VulkanGlslBackend
            .lower_recursive_fft(&ir)
            .unwrap();
        for shader in shaders {
            shader.compile_spirv().unwrap();
        }
        let mut impulse = vec![Complex64::new(0.0, 0.0); length * batch_count];
        for batch in 0..batch_count {
            impulse[batch * length + 1] = Complex64::new(1.0, 0.0);
        }
        let actual = execute_recursive_fft_ir(&ir, &impulse).unwrap();
        let error = actual
            .iter()
            .enumerate()
            .map(|(index, value)| {
                let k = index % length;
                let angle = -TAU * k as f64 / length as f64;
                (*value - Complex64::exp_i(angle)).norm_sqr().sqrt()
            })
            .fold(0.0f64, f64::max);
        assert!(error <= 2.0e-8 * length as f64);
        ir.validate().unwrap();
    }

    #[test]
    fn nested_rader_parent_uses_actual_planner_tuning() {
        let length = 2usize * 283;
        let batch_count = 2usize;
        let mut tuning = crate::PlannerTuning::portable().with_recursive_fft_rader(true);
        tuning.max_rader_direct_prime = 47;
        tuning.validate().unwrap();
        let plan = FftPlan::build(
            FftConfig::new(vec![length])
                .with_batch_count(batch_count)
                .with_tuning(tuning),
        )
        .unwrap();
        let ir = RecursiveFftIr::build(&plan, Direction::Forward, device()).unwrap();
        let RecursiveFftNodeIr::CooleyTukey(root) = &ir.root else {
            panic!("custom N566 should keep smooth-2 x p283 Cooley root");
        };
        let block = root.pack_right.axis_batch_block.unwrap();
        assert_eq!(block.threads_per_transform, 57);
        assert_eq!(block.grouped_batch, 1);
        assert_eq!([block.local_size_x, block.local_size_y], [57, 1]);
        assert_eq!(root.twiddle_transpose.axis_batch_block, Some(block));
        assert_eq!(root.scatter_output.axis_batch_block, Some(block));

        let outer = match (&root.left, &root.right) {
            (RecursiveFftNodeIr::FftRader(rader), _) if rader.prime == 283 => rader,
            (_, RecursiveFftNodeIr::FftRader(rader)) if rader.prime == 283 => rader,
            _ => panic!("custom N566 should contain outer p283 FFT-Rader child"),
        };
        fn contains_fft_prime(node: &RecursiveFftNodeIr, prime: usize) -> bool {
            match node {
                RecursiveFftNodeIr::FftRader(rader) => rader.prime == prime,
                RecursiveFftNodeIr::CooleyTukey(cooley) => {
                    contains_fft_prime(&cooley.left, prime)
                        || contains_fft_prime(&cooley.right, prime)
                }
                RecursiveFftNodeIr::Stockham(_) | RecursiveFftNodeIr::DirectRader(_) => false,
            }
        }
        fn contains_direct_prime(node: &RecursiveFftNodeIr, prime: usize) -> bool {
            match node {
                RecursiveFftNodeIr::DirectRader(direct) => direct.prime == prime,
                RecursiveFftNodeIr::CooleyTukey(cooley) => {
                    contains_direct_prime(&cooley.left, prime)
                        || contains_direct_prime(&cooley.right, prime)
                }
                RecursiveFftNodeIr::Stockham(_) | RecursiveFftNodeIr::FftRader(_) => false,
            }
        }
        let convolution_root = &outer.forward_recursive().unwrap().root;
        assert!(contains_fft_prime(convolution_root, 47));
        assert!(!contains_direct_prime(convolution_root, 47));

        let shaders = crate::backend::vulkan::VulkanGlslBackend
            .lower_recursive_fft(&ir)
            .unwrap();
        for shader in shaders {
            shader.compile_spirv().unwrap();
        }
        let mut impulse = vec![Complex64::new(0.0, 0.0); length * batch_count];
        for batch in 0..batch_count {
            impulse[batch * length + 1] = Complex64::new(1.0, 0.0);
        }
        let actual = execute_recursive_fft_ir(&ir, &impulse).unwrap();
        let error = actual
            .iter()
            .enumerate()
            .map(|(index, value)| {
                let k = index % length;
                let angle = -TAU * k as f64 / length as f64;
                (*value - Complex64::exp_i(angle)).norm_sqr().sqrt()
            })
            .fold(0.0f64, f64::max);
        assert!(error <= 2.0e-8 * length as f64);
        ir.validate().unwrap();
    }

    #[test]
    fn nested_rader_parent_uses_device_capped_direct_range() {
        let length = 2usize * 283;
        let batch_count = 2usize;
        let tuning = crate::PlannerTuning::portable().with_recursive_fft_rader(true);
        let plan = FftPlan::build(
            FftConfig::new(vec![length])
                .with_batch_count(batch_count)
                .with_tuning(tuning),
        )
        .unwrap();
        let mut constrained = device();
        constrained.max_threads_per_block = 64;
        constrained.max_workgroup_size[0] = 64;
        let effective =
            crate::config::upstream_effective_rader_tuning(tuning, constrained, Precision::F32);
        assert_eq!(effective.max_rader_direct_prime, 31);
        assert_eq!(
            crate::config::upstream_effective_rader_tuning(tuning, constrained, Precision::F64,)
                .max_rader_direct_prime,
            63
        );

        let ir = RecursiveFftIr::build(&plan, Direction::Forward, constrained).unwrap();
        let RecursiveFftNodeIr::CooleyTukey(root) = &ir.root else {
            panic!("device-capped N566 should keep smooth-2 x p283 Cooley root");
        };
        let block = root.pack_right.axis_batch_block.unwrap();
        assert_eq!(block.threads_per_transform, 19);
        assert_eq!(block.grouped_batch, 1);
        assert_eq!([block.local_size_x, block.local_size_y], [19, 1]);
        assert_eq!(root.twiddle_transpose.axis_batch_block, Some(block));
        assert_eq!(root.scatter_output.axis_batch_block, Some(block));

        let outer = match (&root.left, &root.right) {
            (RecursiveFftNodeIr::FftRader(rader), _) if rader.prime == 283 => rader,
            (_, RecursiveFftNodeIr::FftRader(rader)) if rader.prime == 283 => rader,
            _ => panic!("device-capped N566 should contain outer p283 FFT-Rader child"),
        };
        fn contains_fft_prime(node: &RecursiveFftNodeIr, prime: usize) -> bool {
            match node {
                RecursiveFftNodeIr::FftRader(rader) => rader.prime == prime,
                RecursiveFftNodeIr::CooleyTukey(cooley) => {
                    contains_fft_prime(&cooley.left, prime)
                        || contains_fft_prime(&cooley.right, prime)
                }
                RecursiveFftNodeIr::Stockham(_) | RecursiveFftNodeIr::DirectRader(_) => false,
            }
        }
        fn contains_direct_prime(node: &RecursiveFftNodeIr, prime: usize) -> bool {
            match node {
                RecursiveFftNodeIr::DirectRader(direct) => direct.prime == prime,
                RecursiveFftNodeIr::CooleyTukey(cooley) => {
                    contains_direct_prime(&cooley.left, prime)
                        || contains_direct_prime(&cooley.right, prime)
                }
                RecursiveFftNodeIr::Stockham(_) | RecursiveFftNodeIr::FftRader(_) => false,
            }
        }
        let convolution_root = &outer.forward_recursive().unwrap().root;
        assert!(contains_fft_prime(convolution_root, 47));
        assert!(!contains_direct_prime(convolution_root, 47));

        let shaders = crate::backend::vulkan::VulkanGlslBackend
            .lower_recursive_fft(&ir)
            .unwrap();
        for shader in shaders {
            shader.compile_spirv().unwrap();
        }
        let mut impulse = vec![Complex64::new(0.0, 0.0); length * batch_count];
        for batch in 0..batch_count {
            impulse[batch * length + 1] = Complex64::new(1.0, 0.0);
        }
        let actual = execute_recursive_fft_ir(&ir, &impulse).unwrap();
        let error = actual
            .iter()
            .enumerate()
            .map(|(index, value)| {
                let k = index % length;
                let angle = -TAU * k as f64 / length as f64;
                (*value - Complex64::exp_i(angle)).norm_sqr().sqrt()
            })
            .fold(0.0f64, f64::max);
        assert!(error <= 2.0e-8 * length as f64);
        ir.validate().unwrap();
    }

    #[test]
    fn nested_sub_rader_composite_parent_uses_recursive_type0_floor() {
        let length = 2usize * 107;
        let batch_count = 2usize;
        let plan = FftPlan::build(
            FftConfig::new(vec![length])
                .with_batch_count(batch_count)
                .with_tuning(crate::PlannerTuning::portable().with_recursive_fft_rader(true)),
        )
        .unwrap();
        let AxisAlgorithm::Rader { primes, .. } = &plan.axes[0].algorithm else {
            panic!("N214 should retain p107 recursive Rader metadata");
        };
        assert_eq!(primes.len(), 1);
        assert_eq!(primes[0].prime, 107);
        assert!(matches!(primes[0].mode, RaderMode::FftConvolution { .. }));

        let ir = RecursiveFftIr::build(&plan, Direction::Forward, device()).unwrap();
        assert!(ir.four_step_plan.is_none());
        assert!(ir.rader_forced_upload_schedule.is_none());
        let RecursiveFftNodeIr::CooleyTukey(root) = &ir.root else {
            panic!("N214 should keep a smooth-2 x p107 Cooley root");
        };
        let block = root.pack_right.axis_batch_block.unwrap();
        assert_eq!(block.threads_per_transform, 18);
        assert_eq!(block.grouped_batch, 1);
        assert_eq!([block.local_size_x, block.local_size_y], [18, 1]);
        assert_eq!(root.twiddle_transpose.axis_batch_block, Some(block));
        assert_eq!(root.scatter_output.axis_batch_block, Some(block));

        let outer = match (&root.left, &root.right) {
            (RecursiveFftNodeIr::FftRader(rader), _) if rader.prime == 107 => rader,
            (_, RecursiveFftNodeIr::FftRader(rader)) if rader.prime == 107 => rader,
            _ => panic!("N214 should contain an outer p107 FFT-Rader leaf"),
        };
        let RecursiveFftNodeIr::CooleyTukey(convolution) = &outer.forward_recursive().unwrap().root
        else {
            panic!("p107 convolution should remain 106=2x53 recursive Cooley");
        };
        assert!(matches!(
            convolution.right,
            RecursiveFftNodeIr::FftRader(ref sub) if sub.prime == 53
        ));

        let shaders = crate::backend::vulkan::VulkanGlslBackend
            .lower_recursive_fft(&ir)
            .unwrap();
        for shader in shaders {
            shader.compile_spirv().unwrap();
        }
        let mut impulse = vec![Complex64::new(0.0, 0.0); length * batch_count];
        for batch in 0..batch_count {
            impulse[batch * length + 1] = Complex64::new(1.0, 0.0);
        }
        let actual = execute_recursive_fft_ir(&ir, &impulse).unwrap();
        let error = actual
            .iter()
            .enumerate()
            .map(|(index, value)| {
                let k = index % length;
                let angle = -TAU * k as f64 / length as f64;
                (*value - Complex64::exp_i(angle)).norm_sqr().sqrt()
            })
            .fold(0.0f64, f64::max);
        assert!(error <= 2.0e-8 * length as f64);
        ir.validate().unwrap();
    }

    #[test]
    fn nested_sub_rader_and_sibling_type0_composite_uses_joint_floor() {
        let length = 19usize * 107;
        let batch_count = 2usize;
        let plan = FftPlan::build(
            FftConfig::new(vec![length])
                .with_batch_count(batch_count)
                .with_tuning(crate::PlannerTuning::portable().with_recursive_fft_rader(true)),
        )
        .unwrap();
        let AxisAlgorithm::Rader { primes, .. } = &plan.axes[0].algorithm else {
            panic!("N2033 should retain p19+p107 recursive Rader metadata");
        };
        assert_eq!(primes.len(), 2);
        assert_eq!((primes[0].prime, primes[1].prime), (19, 107));
        assert!(
            primes
                .iter()
                .all(|prime| matches!(prime.mode, RaderMode::FftConvolution { .. }))
        );

        let mut profile = device();
        profile.max_threads_per_block = 1024;
        profile.max_workgroup_size = [1024, 1024, 64];
        let ir = RecursiveFftIr::build(&plan, Direction::Forward, profile).unwrap();
        assert!(ir.four_step_plan.is_none());
        assert!(ir.rader_forced_upload_schedule.is_none());
        let RecursiveFftNodeIr::CooleyTukey(root) = &ir.root else {
            panic!("N2033 should keep a p19 x p107 Cooley root");
        };
        let block = root.pack_right.axis_batch_block.unwrap();
        assert_eq!(block.threads_per_transform, 214);
        assert_eq!(block.grouped_batch, 1);
        assert_eq!([block.local_size_x, block.local_size_y], [214, 1]);
        assert_eq!(root.twiddle_transpose.axis_batch_block, Some(block));
        assert_eq!(root.scatter_output.axis_batch_block, Some(block));
        assert!(matches!(
            (&root.left, &root.right),
            (RecursiveFftNodeIr::FftRader(left), RecursiveFftNodeIr::FftRader(right))
                if [left.prime, right.prime] == [19, 107]
                    || [left.prime, right.prime] == [107, 19]
        ));
        let outer = match (&root.left, &root.right) {
            (RecursiveFftNodeIr::FftRader(rader), _) if rader.prime == 107 => rader,
            (_, RecursiveFftNodeIr::FftRader(rader)) if rader.prime == 107 => rader,
            _ => unreachable!("asserted p107 sibling"),
        };
        let RecursiveFftNodeIr::CooleyTukey(convolution) = &outer.forward_recursive().unwrap().root
        else {
            panic!("p107 sibling convolution should remain 106=2x53 recursive Cooley");
        };
        assert!(matches!(
            convolution.right,
            RecursiveFftNodeIr::FftRader(ref sub) if sub.prime == 53
        ));

        let shaders = crate::backend::vulkan::VulkanGlslBackend
            .lower_recursive_fft(&ir)
            .unwrap();
        for shader in shaders {
            shader.compile_spirv().unwrap();
        }
        let mut impulse = vec![Complex64::new(0.0, 0.0); length * batch_count];
        for batch in 0..batch_count {
            impulse[batch * length + 1] = Complex64::new(1.0, 0.0);
        }
        let actual = execute_recursive_fft_ir(&ir, &impulse).unwrap();
        let error = actual
            .iter()
            .enumerate()
            .map(|(index, value)| {
                let k = index % length;
                let angle = -TAU * k as f64 / length as f64;
                (*value - Complex64::exp_i(angle)).norm_sqr().sqrt()
            })
            .fold(0.0f64, f64::max);
        assert!(error <= 2.0e-8 * length as f64);
        ir.validate().unwrap();
    }

    #[test]
    fn forced_rader_standalone_p103_upload_reoptimizes_cross_bluestein_caller() {
        let length = 64usize * 103;
        let mut tuning = crate::PlannerTuning::portable();
        tuning.min_rader_direct_prime = 29;
        tuning.min_rader_fft_prime = 29;
        tuning.validate().unwrap();
        let plan = FftPlan::build(FftConfig::new(vec![length]).with_tuning(tuning)).unwrap();
        let AxisAlgorithm::Rader { primes, .. } = &plan.axes[0].algorithm else {
            panic!("N6592 should retain p103 FFT-Rader metadata");
        };
        assert_eq!(primes.len(), 1);
        assert_eq!(primes[0].prime, 103);
        assert!(matches!(primes[0].mode, RaderMode::FftConvolution { .. }));

        let mut profile = device();
        profile.shared_memory_bytes = 32 * 1024;
        profile.shared_memory_pow2_bytes = 32 * 1024;
        profile.max_threads_per_block = 1024;
        profile.max_workgroup_size = [1024, 1024, 64];
        profile.coalesced_memory_bytes = 32;
        let ir = RecursiveFftIr::build(&plan, Direction::Forward, profile).unwrap();
        let schedule = ir
            .rader_forced_upload_schedule
            .as_ref()
            .expect("N6592 should use the pinned two-upload Rader schedule");
        assert_eq!(schedule.axis_split, vec![64, 103]);
        assert_eq!(schedule.upload_count, 2);

        let four_step = ir
            .four_step_plan
            .as_ref()
            .expect("N6592 should materialize Four-step metadata");
        let high = four_step
            .uploads
            .iter()
            .find(|upload| upload.axis_upload_id == 1)
            .unwrap();
        assert_eq!((high.fft_len, high.transform_count), (103, 64));
        let high_block = high.axis_block.unwrap();
        assert_eq!(high_block.threads_per_transform, 35);
        assert_eq!(high_block.grouped_batch, 16);
        assert!(high_block.transforms_on_x);
        assert!(!high_block.axis_swapped);
        assert_eq!([high_block.local_size_x, high_block.local_size_y], [16, 35]);
        let low = four_step
            .uploads
            .iter()
            .find(|upload| upload.axis_upload_id == 0)
            .unwrap();
        assert_eq!((low.fft_len, low.transform_count), (64, 103));
        let low_block = low.axis_block.unwrap();
        assert_eq!(low_block.threads_per_transform, 8);
        assert_eq!(low_block.grouped_batch, 16);
        assert!(low_block.transforms_on_x);
        assert!(low_block.axis_swapped);
        assert_eq!([low_block.local_size_x, low_block.local_size_y], [16, 8]);

        let uploads = ir.four_step_rader_upload_nodes().unwrap().unwrap();
        let RecursiveFftNodeIr::FftRader(rader) = &uploads[0] else {
            panic!("N6592 upload1 should remain the standalone p103 FFT-Rader component");
        };
        assert_eq!(rader.prime, 103);
        assert_eq!(rader.caller_axis_batch_block, Some(high_block));
        assert!(matches!(
            rader.forward_fft.as_ref(),
            crate::OneDimFftIr::Bluestein(_)
        ));
        assert!(matches!(
            rader.inverse_fft.as_ref(),
            crate::OneDimFftIr::Bluestein(_)
        ));
        let RecursiveFftNodeIr::Stockham(low_node) = &uploads[1] else {
            panic!("N6592 upload0 should remain the smooth N64 Stockham component");
        };
        assert_eq!(low_node.sequence_len, 64);
        assert_eq!(low_node.workgroup_size.x as usize, low_block.local_size_x);
        assert_eq!(low_node.workgroup_size.y as usize, low_block.local_size_y);

        let program = crate::ProgramIr::recursive_fft(&ir).unwrap();
        program.validate().unwrap();
        let shaders = crate::backend::vulkan::VulkanGlslBackend
            .lower_recursive_fft(&ir)
            .unwrap();
        assert_eq!(shaders.len(), program.passes.len());
        for shader in shaders {
            assert_eq!(shader.compile_spirv().unwrap().words[0], 0x0723_0203);
        }

        let mut impulse = vec![Complex64::new(0.0, 0.0); length];
        impulse[1] = Complex64::new(1.0, 0.0);
        let actual = execute_recursive_fft_ir(&ir, &impulse).unwrap();
        let error = actual
            .iter()
            .enumerate()
            .map(|(k, value)| {
                let angle = -TAU * k as f64 / length as f64;
                (*value - Complex64::exp_i(angle)).norm_sqr().sqrt()
            })
            .fold(0.0f64, f64::max);
        assert!(
            error <= 2.0e-8 * length as f64,
            "N6592 cross-Bluestein Four-step impulse error {error:e}"
        );
        ir.validate().unwrap();
    }

    #[test]
    fn multi_type0_parent_crosses_into_bluestein_with_upstream_lane_floor() {
        let length = 31usize * 103;
        let mut tuning = crate::PlannerTuning::portable();
        tuning.min_rader_direct_prime = 29;
        tuning.min_rader_fft_prime = 29;
        tuning.validate().unwrap();
        let plan = FftPlan::build(FftConfig::new(vec![length]).with_tuning(tuning)).unwrap();
        let AxisAlgorithm::Rader { primes, .. } = &plan.axes[0].algorithm else {
            panic!("N3193 should retain p31+p103 FFT-Rader metadata");
        };
        assert_eq!(primes.len(), 2);
        assert_eq!((primes[0].prime, primes[1].prime), (31, 103));
        assert!(
            primes
                .iter()
                .all(|prime| matches!(prime.mode, RaderMode::FftConvolution { .. }))
        );

        let mut profile = device();
        profile.shared_memory_bytes = 256 * 1024;
        profile.shared_memory_pow2_bytes = 256 * 1024;
        profile.max_threads_per_block = 1024;
        profile.max_workgroup_size = [1024, 1024, 64];
        let ir = RecursiveFftIr::build(&plan, Direction::Forward, profile).unwrap();
        assert!(ir.four_step_plan.is_none());
        assert!(ir.rader_forced_upload_schedule.is_none());
        let RecursiveFftNodeIr::CooleyTukey(root) = &ir.root else {
            panic!("N3193 should keep a p31 x p103 Cooley root");
        };
        assert_eq!((root.left_len, root.right_len), (31, 103));
        let block = root
            .pack_right
            .axis_batch_block
            .expect("N3193 parent should carry the pinned one-upload block");
        assert_eq!(block.threads_per_transform, 639);
        assert_eq!([block.local_size_x, block.local_size_y], [639, 1]);
        assert_eq!(root.twiddle_transpose.axis_batch_block, Some(block));
        assert_eq!(root.scatter_output.axis_batch_block, Some(block));

        let mut saw_p31 = false;
        let mut saw_p103_bluestein = false;
        for child in [&root.left, &root.right] {
            let RecursiveFftNodeIr::FftRader(rader) = child else {
                panic!("N3193 children should both be FFT-Rader leaves");
            };
            match rader.prime {
                31 => {
                    saw_p31 = true;
                    assert!(matches!(
                        rader.forward_fft.as_ref(),
                        crate::OneDimFftIr::Recursive(_)
                    ));
                }
                103 => {
                    saw_p103_bluestein = true;
                    assert!(matches!(
                        rader.forward_fft.as_ref(),
                        crate::OneDimFftIr::Bluestein(_)
                    ));
                    assert!(matches!(
                        rader.inverse_fft.as_ref(),
                        crate::OneDimFftIr::Bluestein(_)
                    ));
                }
                prime => panic!("unexpected N3193 Rader child p{prime}"),
            }
        }
        assert!(saw_p31 && saw_p103_bluestein);

        let program = crate::ProgramIr::recursive_fft(&ir).unwrap();
        program.validate().unwrap();
        let shaders = crate::backend::vulkan::VulkanGlslBackend
            .lower_recursive_fft(&ir)
            .unwrap();
        assert!(!shaders.is_empty());
        for shader in shaders {
            assert_eq!(shader.compile_spirv().unwrap().words[0], 0x0723_0203);
        }

        let mut impulse = vec![Complex64::new(0.0, 0.0); length];
        impulse[1] = Complex64::new(1.0, 0.0);
        let actual = execute_recursive_fft_ir(&ir, &impulse).unwrap();
        let error = actual
            .iter()
            .enumerate()
            .map(|(k, value)| {
                let angle = -TAU * k as f64 / length as f64;
                (*value - Complex64::exp_i(angle)).norm_sqr().sqrt()
            })
            .fold(0.0f64, f64::max);
        assert!(
            error <= 2.0e-8 * length as f64,
            "N3193 impulse error {error:e}"
        );
        ir.validate().unwrap();
    }

    #[test]
    fn nested_sub_rader_container_context_is_policy_driven_across_gpu_backends() {
        let plan = FftPlan::build(
            FftConfig::new(vec![107])
                .with_tuning(crate::PlannerTuning::portable().with_recursive_fft_rader(true)),
        )
        .unwrap();
        for (backend, vendor) in [
            (Backend::Vulkan, GpuVendor::Nvidia),
            (Backend::Vulkan, GpuVendor::Amd),
            (Backend::Vulkan, GpuVendor::Intel),
            (Backend::OpenCl, GpuVendor::Nvidia),
            (Backend::OpenCl, GpuVendor::Amd),
            (Backend::OpenCl, GpuVendor::Intel),
            (Backend::Cuda, GpuVendor::Nvidia),
            (Backend::Hip, GpuVendor::Amd),
            (Backend::LevelZero, GpuVendor::Intel),
            (Backend::Metal, GpuVendor::Apple),
        ] {
            let profile = DeviceProfile {
                shared_memory_bytes: 64 * 1024,
                shared_memory_pow2_bytes: 64 * 1024,
                max_threads_per_block: 1024,
                ..DeviceProfile::generic(backend, vendor)
            };
            let ir =
                RecursiveFftIr::build(&plan, Direction::Forward, profile).unwrap_or_else(|error| {
                    panic!("{backend:?}/{vendor:?} nested Rader planning failed: {error}")
                });
            let RecursiveFftNodeIr::FftRader(outer) = &ir.root else {
                panic!("{backend:?}/{vendor:?}: expected outer p107 FFT-Rader");
            };
            let RecursiveFftNodeIr::CooleyTukey(convolution) =
                &outer.forward_recursive().unwrap().root
            else {
                panic!("{backend:?}/{vendor:?}: expected 106=2x53 convolution tree");
            };
            let RecursiveFftNodeIr::FftRader(sub_rader) = &convolution.right else {
                panic!("{backend:?}/{vendor:?}: expected nested p53 FFT-Rader");
            };
            let schedule = sub_rader
                .internal_register_schedule
                .as_ref()
                .unwrap_or_else(|| {
                    panic!("{backend:?}/{vendor:?}: nested p53 lost its register schedule")
                });
            assert_eq!(schedule.outer_fft_len, 106, "{backend:?}/{vendor:?}");
            assert_eq!(schedule.container_fft_num, 2, "{backend:?}/{vendor:?}");
            assert_eq!(
                schedule.execution_container_fft_num, 2,
                "{backend:?}/{vendor:?}"
            );
            assert_eq!(
                schedule.internal_fft.stage_radices,
                vec![13, 4],
                "{backend:?}/{vendor:?}"
            );
            assert!(
                schedule.upstream_grouping_is_executable(),
                "{backend:?}/{vendor:?}"
            );
        }
    }

    #[test]
    fn recursive_rader_summary_tracks_nested_thread_and_container_pressure() {
        let plan = FftPlan::build(
            FftConfig::new(vec![107])
                .with_tuning(crate::PlannerTuning::portable().with_recursive_fft_rader(true)),
        )
        .unwrap();
        let ir = RecursiveFftIr::build(&plan, Direction::Forward, device()).unwrap();
        let summary = ir.rader_schedule_summary();
        assert_eq!(summary.fft_rader_nodes, 2);
        assert_eq!(summary.scheduled_fft_rader_nodes, 1);
        assert_eq!(summary.transposed_fft_rader_nodes, 0);
        assert_eq!(summary.max_container_fft_num, 2);
        assert!(summary.max_threads_per_workgroup > 0);

        let transposed_plan = FftPlan::build(FftConfig::new(vec![152])).unwrap();
        let transposed =
            RecursiveFftIr::build(&transposed_plan, Direction::Forward, device()).unwrap();
        let summary = transposed.rader_schedule_summary();
        assert_eq!(summary.fft_rader_nodes, 1);
        assert_eq!(summary.scheduled_fft_rader_nodes, 1);
        assert_eq!(summary.transposed_fft_rader_nodes, 1);
        assert_eq!(summary.max_container_fft_num, 8);
    }

    #[test]
    fn recursive_rader_inherits_arbitrary_smooth_container_context() {
        for (length, prime, containers, expected_radices) in [
            (57usize, 19usize, 3usize, vec![6usize, 3usize]),
            (174usize, 29usize, 6usize, vec![7usize, 4usize]),
        ] {
            let plan = FftPlan::build(FftConfig::new(vec![length])).unwrap();
            let ir = RecursiveFftIr::build(&plan, Direction::Forward, device()).unwrap();
            let RecursiveFftNodeIr::CooleyTukey(root) = &ir.root else {
                panic!("{length} should split into Stockham x Rader factors");
            };
            let rader = match (&root.left, &root.right) {
                (RecursiveFftNodeIr::FftRader(rader), _) if rader.prime == prime => rader,
                (_, RecursiveFftNodeIr::FftRader(rader)) if rader.prime == prime => rader,
                _ => panic!("{length} should contain a p{prime} FFT-Rader leaf"),
            };
            let schedule = rader
                .internal_register_schedule
                .as_ref()
                .expect("recursive Rader leaf should retain the smooth container schedule");
            assert_eq!(schedule.container_fft_num, containers);
            assert_eq!(schedule.execution_container_fft_num, containers);
            assert_eq!(schedule.outer_fft_len, length);
            assert_eq!(schedule.internal_fft.stage_radices, expected_radices);

            let input = sample(length);
            let actual = execute_recursive_fft_ir(&ir, &input).unwrap();
            let expected = dft(&input, Direction::Forward, false);
            assert!(max_error(&actual, &expected) < 2.0e-8 * length as f64);
        }
    }

    #[test]
    fn direct_rader_normal_capacity_materializes_two_upload_four_step() {
        let length = 3usize * 29 * 47;
        let profile = register_leaf_device();
        let mut tuning = crate::PlannerTuning::portable();
        tuning.min_rader_fft_prime = 53;
        tuning.validate().unwrap();
        let plan = FftPlan::build(FftConfig::new(vec![length]).with_tuning(tuning)).unwrap();
        let AxisAlgorithm::Rader { primes, .. } = &plan.axes[0].algorithm else {
            panic!("N4089 should retain direct-Rader metadata");
        };
        assert_eq!(primes.len(), 2);
        assert_eq!(primes[0].prime, 29);
        assert_eq!(primes[1].prime, 47);
        assert!(
            primes
                .iter()
                .all(|prime| matches!(prime.mode, RaderMode::DirectMultiplication))
        );

        let ir = RecursiveFftIr::build(&plan, Direction::Forward, profile).unwrap();
        assert!(!ir.rader_forced_two_upload);
        let schedule = ir
            .rader_forced_upload_schedule
            .as_ref()
            .expect("N4089 normal capacity must materialize Rader multi-upload metadata");
        assert_eq!(schedule.axis_split, vec![87, 47]);
        assert_eq!(schedule.upload_count, 2);
        assert!(ir.four_step_plan.is_some());
        let uploads = ir.four_step_rader_upload_nodes().unwrap().unwrap();
        assert!(matches!(
            uploads[0],
            RecursiveFftNodeIr::DirectRader(ref rader) if rader.prime == 47
        ));

        let program = crate::ProgramIr::recursive_fft(&ir).unwrap();
        program.validate().unwrap();
        let shaders = crate::backend::vulkan::VulkanGlslBackend
            .lower_recursive_fft(&ir)
            .unwrap();
        assert_eq!(shaders.len(), program.passes.len());
        for shader in shaders {
            shader.compile_spirv().unwrap();
        }
        let input = sample(length);
        let actual = execute_recursive_fft_ir(&ir, &input).unwrap();
        let expected = crate::reference::fft(&input, Direction::Forward, false).unwrap();
        assert!(max_error(&actual, &expected) <= 6.0e-8 * length.ilog2() as f64);
        ir.validate().unwrap();
    }

    #[test]
    fn small_composite_direct_rader_stockham_fuses_to_one_program_pass() {
        let length = 2usize * 47;
        let batch_count = 5usize;
        let grouped_batch = 3usize;
        let config = FftConfig::new(vec![length])
            .with_batch_count(batch_count)
            .with_grouped_batch(0, grouped_batch)
            .unwrap();
        let plan = FftPlan::build(config).unwrap();
        let ir = RecursiveFftIr::build(&plan, Direction::Forward, device()).unwrap();
        let RecursiveFftNodeIr::CooleyTukey(root) = &ir.root else {
            panic!("N94 should keep a Cooley root");
        };
        assert_eq!((root.left_len, root.right_len), (2, 47));
        assert!(root.fused_small_direct_rader_stockham().unwrap().is_some());
        let block = root.pack_right.axis_batch_block.unwrap();
        assert_eq!(block.threads_per_transform, 48);
        assert_eq!(block.grouped_batch, grouped_batch);
        assert_eq!([block.local_size_x, block.local_size_y], [3, 48]);

        let program = crate::ProgramIr::recursive_fft(&ir).unwrap();
        program.validate().unwrap();
        assert_eq!(program.passes.len(), 1);
        assert!(
            program.passes[0]
                .name
                .contains("composite_direct_rader_stockham_94_2x47")
        );

        let shaders = crate::backend::vulkan::VulkanGlslBackend
            .lower_recursive_fft(&ir)
            .unwrap();
        assert_eq!(shaders.len(), 1);
        assert_eq!(shaders[0].sequence_len, length);
        assert_eq!(shaders[0].dispatch, root.pack_right.dispatch);
        assert_eq!(shaders[0].workgroup_size, root.pack_right.workgroup_size);
        assert!(
            shaders[0]
                .glsl
                .contains("fused small Stockham x Direct-Rader Cooley IR")
        );
        assert_eq!(shaders[0].compile_spirv().unwrap().words[0], 0x0723_0203);
        ir.validate().unwrap();
    }

    #[test]
    fn small_fft_rader_cooley_fuses_left_twiddle_and_scatter_boundary() {
        let length = 3usize * 17;
        let mut tuning = crate::PlannerTuning::portable();
        tuning.min_rader_direct_prime = 29;
        tuning.min_rader_fft_prime = 17;
        tuning.validate().unwrap();
        let plan = FftPlan::build(FftConfig::new(vec![length]).with_tuning(tuning)).unwrap();
        let ir = RecursiveFftIr::build(&plan, Direction::Forward, device()).unwrap();
        let RecursiveFftNodeIr::CooleyTukey(root) = &ir.root else {
            panic!("N51 should keep a 3 x p17 Cooley root");
        };
        assert_eq!((root.left_len, root.right_len), (3, 17));
        assert!(matches!(
            root.left,
            RecursiveFftNodeIr::Stockham(ref kernel) if kernel.sequence_len == 3
        ));
        assert!(matches!(
            root.right,
            RecursiveFftNodeIr::FftRader(ref rader) if rader.prime == 17
        ));
        let mapped_left = root
            .fused_fft_rader_left_stockham()
            .unwrap()
            .expect("N51 should fuse the Cooley-left Stockham boundary");
        let StockhamIoMapping::CooleyLeft(mapping) = mapped_left.io_mapping else {
            panic!("N51 mapped left kernel should carry CooleyLeft I/O");
        };
        assert_eq!(mapping.parent_logical_len, length);
        assert_eq!((mapping.parent_left_len, mapping.parent_right_len), (3, 17));
        assert!(mapping.outer_four_step.is_none());
        let (mapped_right, mapped_left_full) = root
            .fused_fft_rader_cooley_boundaries()
            .unwrap()
            .expect("N51 should also fuse the Cooley pack into p17 generator loads");
        assert_eq!(mapped_left_full, mapped_left);
        let forward = mapped_right.forward_recursive().unwrap();
        let RecursiveFftNodeIr::Stockham(forward_root) = &forward.root else {
            panic!("N51 mapped p17 forward convolution should remain Stockham");
        };
        let StockhamIoMapping::RaderGeneratorCooleyRight(generator_right) = forward_root.io_mapping
        else {
            panic!("N51 p17 forward Stockham should read the Cooley-right source directly");
        };
        assert_eq!(generator_right.caller.parent_logical_len, length);
        assert_eq!(generator_right.caller.parent_left_len, 3);
        let inverse = mapped_right
            .fused_inverse_rader_kernel()
            .unwrap()
            .expect("N51 p17 should retain its fused inverse Rader kernel");
        let crate::kernel_ir::StockhamOutputModifier::RaderScatter(scatter) =
            inverse.output_modifier
        else {
            panic!("N51 p17 fused inverse should retain Rader scatter");
        };
        assert_eq!(scatter.auxiliary_input, Some(generator_right.caller));
        let fused = root.fused_small_fft_rader_stockham().unwrap().expect(
            "N51 should fuse the complete p17 FFT-Rader component into one parent workgroup",
        );
        assert_eq!(fused.rader.prime, 17);
        assert_eq!(fused.forward.sequence_len, 16);
        assert_eq!(fused.forward.twiddle_lut_len(), None);
        for kernel in [fused.forward, fused.inverse] {
            let stages = kernel.register_stockham_stages().unwrap().unwrap();
            assert_eq!(stages.len(), 1);
            assert_eq!(stages[0].radix, 16);
            assert_eq!(stages[0].virtual_thread_count, 1);
            assert!(
                kernel
                    .register_stage_boundaries()
                    .unwrap()
                    .unwrap()
                    .is_empty()
            );
        }
        assert!(!fused.register_resident_convolution().unwrap());
        assert!(!fused.single_shared_register_convolution().unwrap());
        assert_eq!(fused.shared_stripe_count().unwrap(), 2);
        assert_eq!(fused.uniform_barrier_count().unwrap(), 5);
        assert_eq!(fused.required_shared_memory_bytes().unwrap(), 816);
        assert!(
            fused.required_shared_memory_bytes().unwrap() <= fused.rader.device_shared_memory_bytes
        );
        assert!(
            ir.fused_fft_rader_static_resource_reports()
                .unwrap()
                .is_empty()
        );

        let mut constrained = root.clone();
        let RecursiveFftNodeIr::FftRader(constrained_rader) = &mut constrained.right else {
            unreachable!("validated N51 root has a p17 FFT-Rader child");
        };
        constrained_rader.device_shared_memory_bytes = 512;
        let constrained_fused = constrained
            .fused_small_fft_rader_stockham()
            .unwrap()
            .expect("N51 resident FFT-Rader should fit when two shared stripes do not");
        assert!(constrained_fused.register_resident_convolution().unwrap());
        assert_eq!(constrained_fused.shared_stripe_count().unwrap(), 1);
        assert_eq!(constrained_fused.uniform_barrier_count().unwrap(), 1);
        assert_eq!(
            constrained_fused.required_shared_memory_bytes().unwrap(),
            408
        );
        let report = constrained_fused.static_resource_report().unwrap().unwrap();
        assert_eq!(report.required_shared_memory_bytes, 408);
        assert_eq!(report.uniform_barrier_count, 1);
        assert_eq!(
            report.max_logical_register_complex_values_per_invocation,
            16
        );
        let RecursiveFftNodeIr::FftRader(constrained_rader) = &mut constrained.right else {
            unreachable!("validated N51 root has a p17 FFT-Rader child");
        };
        constrained_rader.device_shared_memory_bytes = 400;
        assert!(
            constrained
                .fused_small_fft_rader_stockham()
                .unwrap()
                .is_none()
        );

        let program = crate::ProgramIr::recursive_fft(&ir).unwrap();
        program.validate().unwrap();
        assert_eq!(program.passes.len(), 1);
        assert_eq!(program.passes[0].name, fused.name());
        assert_eq!(program.passes[0].bindings.len(), 3);
        assert_eq!(
            program.passes[0].bindings[2].role,
            crate::kernel_ir::BufferRole::LookupTable
        );
        assert!(
            !program
                .passes
                .iter()
                .any(|pass| pass.name.contains("recursive_twiddle_51"))
        );
        assert!(
            !program
                .passes
                .iter()
                .any(|pass| pass.name.contains("recursive_scatter_51"))
        );
        assert!(
            !program
                .passes
                .iter()
                .any(|pass| pass.name.contains("recursive_pack_51"))
        );
        let shaders = crate::backend::vulkan::VulkanGlslBackend
            .lower_recursive_fft(&ir)
            .unwrap();
        assert_eq!(shaders.len(), program.passes.len());
        assert_eq!(shaders[0].sequence_len, length);
        assert_eq!(
            shaders[0].required_shared_memory_bytes,
            fused.required_shared_memory_bytes().unwrap()
        );
        assert!(
            shaders[0]
                .glsl
                .contains("fused small Stockham x FFT-Rader Cooley IR")
        );
        assert!(!shaders[0].glsl.contains("vkfft_rader_forward_resident"));
        assert!(!shaders[0].glsl.contains("vkfft_rader_inverse_resident"));
        assert!(shaders[0].glsl.contains("shared vec2 vkfft_fft_a"));
        assert!(shaders[0].glsl.contains("shared vec2 vkfft_fft_b"));
        assert_eq!(shaders[0].glsl.matches("barrier();").count(), 5);
        assert!(!shaders[0].glsl.contains("VKFFT_TWIDDLE_LUT_N"));
        assert_eq!(shaders[0].compile_spirv().unwrap().words[0], 0x0723_0203);

        let input = sample(length);
        let actual = execute_recursive_fft_ir(&ir, &input).unwrap();
        let expected = dft(&input, Direction::Forward, false);
        assert!(max_error(&actual, &expected) < 2.0e-8 * length as f64);
        ir.validate().unwrap();
    }

    #[test]
    fn force_rader_two_upload_controls_the_root_recursive_cut() {
        let length = 17usize * 8192;
        let profile = register_leaf_device();
        let plan = FftPlan::build(FftConfig::new(vec![length])).unwrap();
        let ir = RecursiveFftIr::build(&plan, Direction::Forward, profile).unwrap();
        assert!(ir.rader_forced_two_upload);
        let schedule = ir
            .rader_forced_upload_schedule
            .as_ref()
            .expect("covered forced-Rader branch should retain exact upstream geometry");
        assert_eq!(schedule.upload_count, 2);
        assert_eq!(schedule.axis_split, vec![512, 272]);
        let RecursiveFftNodeIr::CooleyTukey(root) = &ir.root else {
            panic!("forced Rader two-upload policy should create a Cooley-Tukey root");
        };
        assert_eq!((root.left_len, root.right_len), (512, 272));
        assert!(matches!(
            root.left,
            RecursiveFftNodeIr::Stockham(ref kernel) if kernel.sequence_len == 512
        ));
        let RecursiveFftNodeIr::CooleyTukey(right) = &root.right else {
            panic!("272-point second upload should compose 16 x p17");
        };
        assert_eq!((right.left_len, right.right_len), (16, 17));
        assert!(matches!(
            right.left,
            RecursiveFftNodeIr::Stockham(ref kernel) if kernel.sequence_len == 16
        ));
        let RecursiveFftNodeIr::FftRader(rader) = &right.right else {
            panic!("272-point second upload should preserve the p17 FFT-Rader leaf");
        };
        assert_eq!(rader.batch_count, 8192);
        assert!(
            rader.internal_register_schedule.is_none(),
            "8192 p17 containers should exceed the covered monolithic register schedule envelope"
        );
        ir.validate().unwrap();

        let mut narrow_profile = profile;
        narrow_profile.max_threads_per_block = 64;
        narrow_profile.max_workgroup_size[0] = 64;
        let narrow = RecursiveFftIr::build(&plan, Direction::Forward, narrow_profile).unwrap();
        assert_eq!(
            narrow
                .rader_forced_upload_schedule
                .as_ref()
                .expect("64-thread p17-container path must remain forced")
                .axis_split,
            vec![512, 272]
        );
        let narrow_uploads = narrow.four_step_rader_upload_nodes().unwrap().unwrap();
        let RecursiveFftNodeIr::CooleyTukey(narrow_high) = &narrow_uploads[0] else {
            panic!("64-thread 272-point upload1 should remain 16 x p17 Cooley");
        };
        let narrow_block = narrow_high.pack_right.axis_batch_block.unwrap();
        assert_eq!(narrow_block.threads_per_transform, 16);
        assert_eq!(narrow_block.grouped_batch, 4);
        assert!(narrow_block.transforms_on_x);
        assert_eq!(
            [narrow_block.local_size_x, narrow_block.local_size_y],
            [4, 16]
        );
        assert_eq!(
            narrow_high.twiddle_transpose.axis_batch_block,
            Some(narrow_block)
        );
        assert_eq!(
            narrow_high.scatter_output.axis_batch_block,
            Some(narrow_block)
        );
        // The same physical N272 component demonstrates fixed-upstream's k==0
        // special case: scheduler-time coalescing is four transforms, but upload0
        // shrinks that coalescing before scaleRegistersNum while upload1 keeps it.
        let RecursiveFftNodeIr::CooleyTukey(narrow_root) = &narrow.root else {
            panic!("64-thread forced N139264 root should remain Cooley");
        };
        let narrow_schedule = narrow.rader_forced_upload_schedule.as_ref().unwrap();
        let context = FourStepBuildContext {
            precision: Precision::F32,
            complex_bytes: narrow.scalar.complex_bytes(),
            perform_zero_padding: false,
            grouped_batch_override: None,
            tuning: crate::PlannerTuning::portable(),
            device: narrow_profile,
        };
        let upload0_floor = forced_rader_component_axis_block(
            &narrow_root.right,
            narrow_schedule,
            0,
            1,
            narrow_root.right.batch_count(),
            narrow.batch_count,
            context,
        )
        .unwrap()
        .unwrap();
        assert_eq!(upload0_floor.threads_per_transform, 17);
        assert_eq!(upload0_floor.grouped_batch, 2);
        assert!(upload0_floor.transforms_on_x);
        assert!(upload0_floor.axis_swapped);
        assert_eq!(
            [upload0_floor.local_size_x, upload0_floor.local_size_y],
            [2, 17]
        );
        let upload1_floor = forced_rader_component_axis_block(
            &narrow_root.right,
            narrow_schedule,
            1,
            512,
            narrow_root.right.batch_count(),
            narrow.batch_count,
            context,
        )
        .unwrap()
        .unwrap();
        assert_eq!(upload1_floor.threads_per_transform, 16);
        assert_eq!(upload1_floor.grouped_batch, 4);
        assert!(upload1_floor.transforms_on_x);
        assert!(!upload1_floor.axis_swapped);
        assert_eq!(
            [upload1_floor.local_size_x, upload1_floor.local_size_y],
            [4, 16]
        );
        let narrow_shaders = crate::backend::vulkan::VulkanGlslBackend
            .lower_recursive_fft(&narrow)
            .unwrap();
        for shader in narrow_shaders {
            shader.compile_spirv().unwrap();
        }
        narrow.validate().unwrap();

        let small = FftPlan::build(FftConfig::new(vec![17usize * 512])).unwrap();
        let small_ir = RecursiveFftIr::build(&small, Direction::Forward, profile).unwrap();
        assert!(!small_ir.rader_forced_two_upload);
        let small_schedule = small_ir
            .rader_forced_upload_schedule
            .as_ref()
            .expect("N8704 exceeds normal 32 KiB capacity even below the force threshold");
        assert_eq!(small_schedule.axis_split, vec![128, 68]);
        assert!(small_ir.four_step_plan.is_some());

        let mut thread_limited = profile;
        thread_limited.max_threads_per_block = 128;
        let medium_length = 17usize * 300;
        let medium_plan = FftPlan::build(FftConfig::new(vec![medium_length])).unwrap();
        let medium =
            RecursiveFftIr::build(&medium_plan, Direction::Forward, thread_limited).unwrap();
        assert_eq!(
            medium
                .rader_forced_upload_schedule
                .as_ref()
                .unwrap()
                .axis_split,
            vec![68, 75]
        );
        assert!(medium.four_step_plan.is_some());
        let medium_uploads = medium
            .four_step_rader_upload_nodes()
            .unwrap()
            .expect("forced two-upload Rader plan must materialize generic Four-step upload nodes");
        assert_eq!(medium_uploads.len(), 2);
        assert!(matches!(medium_uploads[0], RecursiveFftNodeIr::Stockham(_)));
        let RecursiveFftNodeIr::CooleyTukey(low) = &medium_uploads[1] else {
            panic!("68-point forced-Rader low upload should remain a Cooley-Tukey component");
        };
        assert!(matches!(
            low.scatter_output.output_modifier,
            CooleyTukeyOutputModifier::FourStepLeft(_)
        ));
        let low_block = low
            .pack_right
            .axis_batch_block
            .expect("68-point p17 FFT-Rader component should use pass-local parent scoring");
        assert_eq!(low_block.threads_per_transform, 5);
        assert_eq!(low.twiddle_transpose.axis_batch_block, Some(low_block));
        assert_eq!(low.scatter_output.axis_batch_block, Some(low_block));
        let input = sample(medium_length);
        let actual = execute_recursive_fft_ir(&medium, &input).unwrap();
        let expected = crate::reference::fft(&input, Direction::Forward, false).unwrap();
        assert!(max_error(&actual, &expected) <= 4.0e-8 * medium_length.ilog2() as f64);

        let three_length = 17usize * 65_536;
        let three_plan = FftPlan::build(FftConfig::new(vec![three_length])).unwrap();
        let three = RecursiveFftIr::build(&three_plan, Direction::Forward, profile).unwrap();
        let schedule = three
            .rader_forced_upload_schedule
            .as_ref()
            .expect("large forced Rader geometry should promote beyond two uploads");
        assert_eq!(schedule.upload_count, 3);
        assert_eq!(schedule.axis_split, vec![128, 68, 128]);
        assert!(rader_upload_split_matches_root(
            &three.root,
            &schedule.axis_split
        ));
        assert!(three.four_step_plan.is_some());
        let three_uploads = three.four_step_rader_upload_nodes().unwrap().expect(
            "forced three-upload Rader plan must materialize generic Four-step upload nodes",
        );
        assert_eq!(three_uploads.len(), 3);
        assert!(matches!(three_uploads[0], RecursiveFftNodeIr::Stockham(_)));
        let RecursiveFftNodeIr::CooleyTukey(middle) = &three_uploads[1] else {
            panic!("68-point forced-Rader middle upload should remain a Cooley-Tukey component");
        };
        assert!(matches!(
            middle.scatter_output.output_modifier,
            CooleyTukeyOutputModifier::FourStepThreeUpload1(_)
        ));
        assert!(matches!(three_uploads[2], RecursiveFftNodeIr::Stockham(_)));
        three.validate().unwrap();

        let mut intel = DeviceProfile::generic(Backend::Vulkan, GpuVendor::Intel);
        intel.shared_memory_bytes = 32 * 1024;
        intel.shared_memory_pow2_bytes = 32 * 1024;
        intel.max_threads_per_block = 1024;
        intel.max_workgroup_size = [1024, 1024, 64];
        let bandwidth_length = 17usize * 16_384;
        let strided_config =
            FftConfig::new(vec![bandwidth_length]).resolve_tuning_for_device(intel);
        let strided_plan =
            FftPlan::build_c2c_child_for_device(strided_config, intel, C2cDeviceAxisClass::Strided)
                .unwrap();
        let strided = RecursiveFftIr::build(&strided_plan, Direction::Forward, intel).unwrap();
        let strided_schedule = strided
            .rader_forced_upload_schedule
            .as_ref()
            .expect("Intel strided composite p17 must retain forced-Rader geometry");
        assert_eq!(strided_schedule.upload_count, 3);
        assert_eq!(strided_schedule.axis_split, vec![64, 68, 64]);
        assert!(rader_upload_split_matches_root(
            &strided.root,
            &strided_schedule.axis_split
        ));

        let boosted_config = FftConfig::new(vec![bandwidth_length])
            .with_bandwidth_boost(2)
            .resolve_tuning_for_device(intel);
        let boosted_plan =
            FftPlan::build_c2c_child_for_device(boosted_config, intel, C2cDeviceAxisClass::Strided)
                .unwrap();
        let boosted = RecursiveFftIr::build(&boosted_plan, Direction::Forward, intel).unwrap();
        let boosted_schedule = boosted
            .rader_forced_upload_schedule
            .as_ref()
            .expect("B=2 Intel composite p17 must retain forced-Rader scheduling");
        assert_eq!(boosted_schedule.upload_count, 2);
        assert_eq!(boosted_schedule.axis_split, vec![544, 512]);
        assert!(rader_upload_split_matches_root(
            &boosted.root,
            &boosted_schedule.axis_split
        ));
        let uploads = boosted
            .four_step_rader_upload_nodes()
            .unwrap()
            .expect("B=2 forced-Rader schedule must materialize two Four-step uploads");
        assert_eq!(uploads.len(), 2);
        let program = crate::ProgramIr::recursive_fft(&boosted).unwrap();
        program.validate().unwrap();
        let shaders = crate::backend::vulkan::VulkanGlslBackend
            .lower_recursive_fft(&boosted)
            .unwrap();
        assert_eq!(shaders.len(), program.passes.len());
        for shader in shaders {
            assert_eq!(shader.compile_spirv().unwrap().words[0], 0x0723_0203);
        }
    }

    #[test]
    fn forced_rader_stockham_components_use_upstream_four_step_grouping() {
        let cases = [
            (
                8_704usize,
                32 * 1024usize,
                32 * 1024usize,
                1024usize,
                0usize,
                StockhamAxisBlockSchedule {
                    threads_per_transform: 16,
                    grouped_batch: 16,
                    transforms_on_x: true,
                    axis_swapped: true,
                    local_size_x: 16,
                    local_size_y: 16,
                },
            ),
            (
                5_100,
                48 * 1024,
                32 * 1024,
                128,
                1,
                StockhamAxisBlockSchedule {
                    threads_per_transform: 5,
                    grouped_batch: 16,
                    transforms_on_x: true,
                    axis_swapped: false,
                    local_size_x: 16,
                    local_size_y: 5,
                },
            ),
            (
                1_922,
                1024,
                1024,
                32,
                0,
                StockhamAxisBlockSchedule {
                    threads_per_transform: 1,
                    grouped_batch: 16,
                    transforms_on_x: true,
                    axis_swapped: true,
                    local_size_x: 16,
                    local_size_y: 1,
                },
            ),
            (
                33_728,
                2 * 1024,
                2 * 1024,
                64,
                0,
                StockhamAxisBlockSchedule {
                    threads_per_transform: 4,
                    grouped_batch: 8,
                    transforms_on_x: false,
                    axis_swapped: false,
                    local_size_x: 4,
                    local_size_y: 8,
                },
            ),
            (
                4_352,
                48 * 1024,
                32 * 1024,
                128,
                0,
                StockhamAxisBlockSchedule {
                    threads_per_transform: 8,
                    grouped_batch: 16,
                    transforms_on_x: true,
                    axis_swapped: true,
                    local_size_x: 16,
                    local_size_y: 8,
                },
            ),
        ];

        for (length, shared, shared_pow2, thread_cap, upload_id, expected) in cases {
            let mut profile = DeviceProfile::generic(Backend::Vulkan, GpuVendor::Nvidia);
            profile.shared_memory_bytes = shared;
            profile.shared_memory_pow2_bytes = shared_pow2;
            profile.max_threads_per_block = thread_cap;
            let plan = FftPlan::build(FftConfig::new(vec![length])).unwrap();
            let ir = RecursiveFftIr::build(&plan, Direction::Forward, profile).unwrap();
            let four_step = ir
                .four_step_plan
                .as_ref()
                .expect("ordinary Rader upload must materialize Four-step metadata");
            let upload = four_step
                .uploads
                .iter()
                .find(|upload| upload.axis_upload_id == upload_id)
                .expect("requested Stockham upload must exist");
            assert_eq!(upload.axis_block, Some(expected), "length {length}");

            let mapped = ir
                .four_step_rader_upload_nodes()
                .unwrap()
                .expect("ordinary Rader upload must map executable Four-step nodes");
            let (_, mapped_node) = four_step
                .uploads
                .iter()
                .zip(mapped.iter())
                .find(|(upload, _)| upload.axis_upload_id == upload_id)
                .expect("mapped Stockham upload must exist");
            let RecursiveFftNodeIr::Stockham(kernel) = mapped_node else {
                panic!("length {length} upload {upload_id} should remain Stockham");
            };
            assert_eq!(
                kernel.workgroup_grouping.transforms_per_workgroup,
                expected.grouped_batch
            );
            assert_eq!(
                kernel.workgroup_grouping.threads_per_transform,
                expected.threads_per_transform
            );
            assert_eq!(
                kernel.workgroup_grouping.axis_layout,
                if expected.transforms_on_x {
                    crate::kernel_ir::StockhamWorkgroupAxisLayout::TransformsXThreadsY
                } else {
                    crate::kernel_ir::StockhamWorkgroupAxisLayout::ThreadsXTransformsY
                }
            );
            assert_eq!(kernel.workgroup_size.x as usize, expected.local_size_x);
            assert_eq!(kernel.workgroup_size.y as usize, expected.local_size_y);
            ir.validate().unwrap();

            if matches!(length, 5_100 | 33_728) {
                for shader in crate::backend::vulkan::VulkanGlslBackend
                    .lower_recursive_fft(&ir)
                    .unwrap()
                {
                    assert_eq!(shader.compile_spirv().unwrap().words[0], 0x0723_0203);
                }
            }
        }
    }

    #[test]
    fn intel_forced_rader_direct_component_keeps_upstream_group_cap() {
        let length = 4_089usize;
        let mut profile = DeviceProfile::generic(Backend::OpenCl, GpuVendor::Intel);
        profile.shared_memory_bytes = 32 * 1024;
        profile.shared_memory_pow2_bytes = 32 * 1024;
        profile.max_threads_per_block = 256;
        profile.max_workgroup_size = [256, 256, 64];
        profile.coalesced_memory_bytes = 64;

        let plan = FftPlan::build(FftConfig::new(vec![length])).unwrap();
        let ir = RecursiveFftIr::build(&plan, Direction::Forward, profile).unwrap();
        let schedule = ir
            .rader_forced_upload_schedule
            .as_ref()
            .expect("Intel N4089 must retain the upstream two-upload Rader split");
        assert_eq!(schedule.upload_count, 2);
        assert_eq!(schedule.axis_split, vec![87, 47]);

        let four_step = ir
            .four_step_plan
            .as_ref()
            .expect("Intel N4089 must materialize Four-step metadata");
        let high = four_step
            .uploads
            .iter()
            .find(|upload| upload.axis_upload_id == 1)
            .unwrap();
        let high_block = high.axis_block.unwrap();
        assert_eq!((high.fft_len, high.transform_count), (47, 87));
        assert_eq!(high_block.threads_per_transform, 24);
        assert_eq!(high_block.grouped_batch, 8);
        assert!(high_block.transforms_on_x);
        assert!(!high_block.axis_swapped);
        assert_eq!([high_block.local_size_x, high_block.local_size_y], [8, 24]);

        let low = four_step
            .uploads
            .iter()
            .find(|upload| upload.axis_upload_id == 0)
            .unwrap();
        let low_block = low.axis_block.unwrap();
        assert_eq!((low.fft_len, low.transform_count), (87, 47));
        assert_eq!(low_block.threads_per_transform, 15);
        assert_eq!(low_block.grouped_batch, 16);
        assert!(!low_block.transforms_on_x);
        assert!(!low_block.axis_swapped);
        assert_eq!([low_block.local_size_x, low_block.local_size_y], [15, 16]);

        let uploads = ir.four_step_rader_upload_nodes().unwrap().unwrap();
        let RecursiveFftNodeIr::DirectRader(high_node) = &uploads[0] else {
            panic!("Intel N4089 upload1 must remain p47 Direct-Rader");
        };
        assert_eq!(high_node.prime, 47);
        assert_eq!(high_node.axis_batch_block, Some(high_block));
        let RecursiveFftNodeIr::CooleyTukey(low_node) = &uploads[1] else {
            panic!("Intel N4089 upload0 must remain the 87-point Cooley component");
        };
        assert_eq!(low_node.logical_len, 87);
        assert_eq!(low_node.pack_right.axis_batch_block, Some(low_block));
        let mapped_left = low_node
            .fused_fft_rader_left_stockham()
            .unwrap()
            .expect("Intel N4089 upload0 should fuse its Cooley left boundary");
        let StockhamIoMapping::CooleyLeft(cooley_left) = mapped_left.io_mapping else {
            panic!("Intel N4089 mapped left child should carry CooleyLeft I/O");
        };
        assert_eq!(cooley_left.parent_logical_len, 87);
        assert_eq!(
            (cooley_left.parent_left_len, cooley_left.parent_right_len),
            (3, 29)
        );
        assert_eq!(cooley_left.parent_batch_count, 47);
        assert_eq!(cooley_left.outer_four_step.unwrap().logical_len, length);
        let (mapped_right, mapped_left_full) = low_node
            .fused_fft_rader_cooley_boundaries()
            .unwrap()
            .expect("Intel N4089 u0 should also fuse its pack into p29 generator loads");
        assert_eq!(mapped_left_full, mapped_left);
        let forward = mapped_right.forward_recursive().unwrap();
        let RecursiveFftNodeIr::Stockham(forward_root) = &forward.root else {
            panic!("Intel N4089 p29 forward convolution should remain Stockham");
        };
        let StockhamIoMapping::RaderGeneratorCooleyRight(generator_right) = forward_root.io_mapping
        else {
            panic!("Intel N4089 p29 forward Stockham should read the Cooley-right source directly");
        };
        assert_eq!(generator_right.caller.parent_logical_len, 87);
        assert_eq!(generator_right.caller.parent_left_len, 3);
        assert_eq!(generator_right.caller.parent_batch_count, 47);
        let inverse = mapped_right
            .fused_inverse_rader_kernel()
            .unwrap()
            .expect("Intel N4089 p29 should retain its fused inverse Rader kernel");
        let crate::kernel_ir::StockhamOutputModifier::RaderScatter(scatter) =
            inverse.output_modifier
        else {
            panic!("Intel N4089 p29 fused inverse should retain Rader scatter");
        };
        assert_eq!(scatter.auxiliary_input, Some(generator_right.caller));
        let fused = low_node
            .fused_small_fft_rader_stockham()
            .unwrap()
            .expect("Intel N4089 u0 should fuse the complete p29 FFT-Rader component into one parent workgroup");
        assert_eq!(fused.rader.prime, 29);
        assert_eq!(fused.forward.sequence_len, 28);
        assert_eq!(fused.forward.twiddle_lut_len(), Some(28));
        for kernel in [fused.forward, fused.inverse] {
            let stages = kernel.register_stockham_stages().unwrap().unwrap();
            assert_eq!(
                stages.iter().map(|stage| stage.radix).collect::<Vec<_>>(),
                vec![14, 2]
            );
            assert!(stages.iter().all(|stage| stage.virtual_thread_count == 2));
            let boundaries = kernel.register_stage_boundaries().unwrap().unwrap();
            assert_eq!(boundaries.len(), 1);
            assert_eq!(
                boundaries[0].residency,
                crate::RegisterStageBoundaryResidency::SharedExchangeRequired
            );
        }
        assert!(!fused.register_resident_convolution().unwrap());
        assert!(!fused.single_shared_register_convolution().unwrap());
        assert_eq!(fused.shared_stripe_count().unwrap(), 2);
        assert_eq!(fused.uniform_barrier_count().unwrap(), 7);
        assert_eq!(fused.required_shared_memory_bytes().unwrap(), 22_272);
        assert!(fused.required_shared_memory_bytes().unwrap() <= 32 * 1024);
        assert!(
            ir.fused_fft_rader_static_resource_reports()
                .unwrap()
                .is_empty()
        );
        let mut constrained = low_node.clone();
        let RecursiveFftNodeIr::FftRader(constrained_rader) = &mut constrained.right else {
            unreachable!("validated N4089 low upload has a p29 FFT-Rader child");
        };
        constrained_rader.device_shared_memory_bytes = 16 * 1024;
        let constrained_fused = constrained
            .fused_small_fft_rader_stockham()
            .unwrap()
            .expect("N87 single-shared FFT-Rader should fit a retained 16 KiB budget");
        assert!(
            constrained_fused
                .single_shared_register_convolution()
                .unwrap()
        );
        assert_eq!(constrained_fused.shared_stripe_count().unwrap(), 1);
        assert_eq!(constrained_fused.uniform_barrier_count().unwrap(), 7);
        assert_eq!(
            constrained_fused.required_shared_memory_bytes().unwrap(),
            10_752
        );
        let constrained_report = constrained_fused.static_resource_report().unwrap().unwrap();
        assert_eq!(constrained_report.required_shared_memory_bytes, 10_752);
        assert_eq!(constrained_report.uniform_barrier_count, 7);
        assert_eq!(
            constrained_report.max_logical_register_complex_values_per_invocation,
            14
        );
        let RecursiveFftNodeIr::FftRader(constrained_rader) = &mut constrained.right else {
            unreachable!("validated N4089 low upload has a p29 FFT-Rader child");
        };
        constrained_rader.device_shared_memory_bytes = 8 * 1024;
        assert!(
            constrained
                .fused_small_fft_rader_stockham()
                .unwrap()
                .is_none(),
            "N87 single-shared FFT-Rader must fail soft below its exact 10,752-byte footprint"
        );

        let program = crate::ProgramIr::recursive_fft(&ir).unwrap();
        program.validate().unwrap();
        assert_eq!(program.passes.len(), 2);
        let fused_pass = program
            .passes
            .iter()
            .find(|pass| pass.name == fused.name())
            .expect("Intel N4089 should contain one fused N87 upload pass");
        assert_eq!(fused_pass.bindings.len(), 4);
        assert_eq!(
            fused_pass.bindings[2].role,
            crate::kernel_ir::BufferRole::LookupTable
        );
        assert_eq!(
            fused_pass.bindings[3].role,
            crate::kernel_ir::BufferRole::TwiddleLookupTable
        );
        assert!(
            !program
                .passes
                .iter()
                .any(|pass| pass.name == "vkfft_recursive_twiddle_87_forward")
        );
        assert!(
            !program
                .passes
                .iter()
                .any(|pass| pass.name.contains("recursive_scatter_87"))
        );
        assert!(
            !program
                .passes
                .iter()
                .any(|pass| pass.name.contains("recursive_pack_87"))
        );
        let shaders = crate::backend::vulkan::VulkanGlslBackend
            .lower_recursive_fft(&ir)
            .unwrap();
        assert_eq!(shaders.len(), program.passes.len());
        let fused_shader = shaders
            .iter()
            .find(|shader| shader.sequence_len == 87)
            .expect("Intel N4089 should lower one fused N87 upload shader");
        assert_eq!(fused_shader.dispatch, low_node.pack_right.dispatch);
        assert_eq!(
            fused_shader.workgroup_size,
            low_node.pack_right.workgroup_size
        );
        assert_eq!(fused_shader.required_shared_memory_bytes, 22_272);
        assert!(
            fused_shader
                .glsl
                .contains("fused small Stockham x FFT-Rader Cooley IR")
        );
        assert!(
            fused_shader
                .glsl
                .contains("VKFFT_TWIDDLE_LUT_N = VKFFT_COUNT")
        );
        assert!(fused_shader.glsl.contains("vkfft_twiddle_lut.data"));
        assert!(
            !fused_shader
                .glsl
                .contains("single-shared register Stockham")
        );
        assert!(fused_shader.glsl.contains("fused forward Stockham stage"));
        assert!(fused_shader.glsl.contains("fused inverse Stockham stage"));
        assert!(fused_shader.glsl.contains("shared vec2 vkfft_fft_a[1392]"));
        assert!(fused_shader.glsl.contains("shared vec2 vkfft_fft_b[1392]"));
        assert_eq!(fused_shader.glsl.matches("barrier();").count(), 7);
        for shader in shaders {
            assert_eq!(shader.compile_spirv().unwrap().words[0], 0x0723_0203);
        }
        for backend in [crate::Backend::Cuda, crate::Backend::OpenCl] {
            let native = crate::backend::native::NativeSourceBackend::new(backend)
                .lower_recursive_fft(&ir)
                .unwrap();
            assert_eq!(native.len(), 2);
            let fused_native = native
                .iter()
                .find(|shader| shader.sequence_len == 87)
                .expect("Intel N4089 fused N87 upload should translate to native source");
            assert_eq!(fused_native.bindings.len(), 4);
            assert_eq!(
                fused_native.bindings[2].role,
                crate::kernel_ir::BufferRole::LookupTable
            );
            assert_eq!(
                fused_native.bindings[3].role,
                crate::kernel_ir::BufferRole::TwiddleLookupTable
            );
            assert!(fused_native.source.contains("twiddle_lut"));
            assert!(!fused_native.source.contains("vkfft_twiddle_lut.data"));
            fused_native.validate().unwrap();
        }

        let mut impulse = vec![Complex64::new(0.0, 0.0); length];
        impulse[1] = Complex64::new(1.0, 0.0);
        let actual = execute_recursive_fft_ir(&ir, &impulse).unwrap();
        let error = actual
            .iter()
            .enumerate()
            .map(|(k, value)| {
                let angle = -TAU * k as f64 / length as f64;
                (*value - Complex64::exp_i(angle)).norm_sqr().sqrt()
            })
            .fold(0.0f64, f64::max);
        assert!(
            error <= 2.0e-8 * length as f64,
            "Intel N4089 Direct-Rader Four-step impulse error {error:e}"
        );
        ir.validate().unwrap();
    }

    #[test]
    fn forced_rader_three_upload_maps_fft_rader_component() {
        let length = 17usize * 31 * 64;
        let profile = DeviceProfile {
            shared_memory_bytes: 2 * 1024,
            shared_memory_pow2_bytes: 2 * 1024,
            max_threads_per_block: 1024,
            ..DeviceProfile::generic(Backend::Vulkan, GpuVendor::Nvidia)
        };
        let plan = FftPlan::build(FftConfig::new(vec![length])).unwrap();
        let ir = RecursiveFftIr::build(&plan, Direction::Forward, profile).unwrap();
        let schedule = ir
            .rader_forced_upload_schedule
            .as_ref()
            .expect("constrained p31 axis should require forced Rader uploads");
        assert_eq!(schedule.axis_split, vec![32, 34, 31]);
        assert_eq!(schedule.upload_count, 3);

        let uploads = ir.four_step_rader_upload_nodes().unwrap().unwrap();
        let RecursiveFftNodeIr::FftRader(high) = &uploads[0] else {
            panic!("31-point high upload should stay FFT Rader");
        };
        assert!(matches!(
            high.io_mapping,
            StockhamIoMapping::FourStepThreeUpload2(_)
        ));
        assert_eq!(
            high.input_strategy,
            crate::RaderFftInputStrategy::GeneratorOrderStockham
        );
        let forward = high
            .forward_recursive()
            .expect("mapped p31 FFT Rader must retain its forward convolution child");
        let RecursiveFftNodeIr::Stockham(forward_kernel) = &forward.root else {
            panic!("mapped p31 FFT Rader should fuse generator loads into Stockham");
        };
        let StockhamIoMapping::RaderGeneratorFourStep(composed) = forward_kernel.io_mapping else {
            panic!("mapped p31 FFT Rader should compose generator and Four-step input mapping");
        };
        assert_eq!(composed.rader.prime, 31);
        assert!(matches!(
            composed.caller,
            crate::kernel_ir::RaderFourStepInputMapping::ThreeUpload2(_)
        ));
        assert!(!high.has_fused_recursive_inverse());
        assert!(high.fused_inverse_rader_kernel().unwrap().is_none());
        let high_block = high
            .caller_axis_batch_block
            .expect("p31 forced upload2 should receive a pass-local caller block");
        assert_eq!(high_block.threads_per_transform, 7);
        assert_eq!(high_block.grouped_batch, 8);
        assert!(high_block.transforms_on_x);
        assert!(!high_block.axis_swapped);
        assert_eq!([high_block.local_size_x, high_block.local_size_y], [8, 7]);
        assert_eq!(high.gather.axis_batch_block, Some(high_block));
        assert_eq!(high.scatter.axis_batch_block, Some(high_block));

        let gather_name = high.gather.name.clone();
        let program = crate::ProgramIr::recursive_fft(&ir).unwrap();
        assert!(
            program.passes.iter().all(|pass| pass.name != gather_name),
            "mapped p31 upload should not materialize an explicit Rader gather pass"
        );
        program.validate().unwrap();
        let shaders = crate::backend::vulkan::VulkanGlslBackend
            .lower_recursive_fft(&ir)
            .unwrap();
        assert_eq!(shaders.len(), program.passes.len());
        for shader in shaders {
            assert_eq!(shader.compile_spirv().unwrap().words[0], 0x0723_0203);
        }

        let input = sample(length);
        let actual = execute_recursive_fft_ir(&ir, &input).unwrap();
        let expected = crate::reference::fft(&input, Direction::Forward, false).unwrap();
        assert!(max_error(&actual, &expected) <= 8.0e-8 * length.ilog2() as f64);
        ir.validate().unwrap();
    }

    #[test]
    fn forced_rader_n1922_fuses_both_p31_generator_loads() {
        let length = 1_922usize;
        let profile = DeviceProfile {
            shared_memory_bytes: 1024,
            shared_memory_pow2_bytes: 1024,
            max_threads_per_block: 32,
            max_workgroup_size: [32, 32, 64],
            ..DeviceProfile::generic(Backend::Vulkan, GpuVendor::Nvidia)
        };
        let plan = FftPlan::build(FftConfig::new(vec![length])).unwrap();
        let ir = RecursiveFftIr::build(&plan, Direction::Forward, profile).unwrap();
        assert_eq!(
            ir.rader_forced_upload_schedule.as_ref().unwrap().axis_split,
            vec![2, 31, 31]
        );
        let uploads = ir.four_step_rader_upload_nodes().unwrap().unwrap();
        assert_eq!(uploads.len(), 3);

        let mut gather_names = Vec::new();
        for (upload_index, expected_caller) in [(0usize, 2usize), (1usize, 1usize)] {
            let RecursiveFftNodeIr::FftRader(rader) = &uploads[upload_index] else {
                panic!("N1922 upload {upload_index} should remain p31 FFT Rader");
            };
            assert_eq!(rader.prime, 31);
            assert_eq!(
                rader.input_strategy,
                crate::RaderFftInputStrategy::GeneratorOrderStockham
            );
            let forward = rader.forward_recursive().unwrap();
            let RecursiveFftNodeIr::Stockham(kernel) = &forward.root else {
                panic!("N1922 p31 forward convolution should remain Stockham");
            };
            let StockhamIoMapping::RaderGeneratorFourStep(composed) = kernel.io_mapping else {
                panic!("N1922 p31 forward convolution should compose its caller mapping");
            };
            assert_eq!(composed.rader.prime, 31);
            match expected_caller {
                2 => assert!(matches!(
                    composed.caller,
                    crate::kernel_ir::RaderFourStepInputMapping::ThreeUpload2(_)
                )),
                1 => assert!(matches!(
                    composed.caller,
                    crate::kernel_ir::RaderFourStepInputMapping::ThreeUpload1(_)
                )),
                _ => unreachable!(),
            }
            gather_names.push(rader.gather.name.clone());
        }

        let program = crate::ProgramIr::recursive_fft(&ir).unwrap();
        for gather_name in &gather_names {
            assert!(
                program.passes.iter().all(|pass| &pass.name != gather_name),
                "N1922 mapped p31 upload should not materialize gather pass {gather_name}"
            );
        }
        assert_eq!(program.passes.len(), 9);
        program.validate().unwrap();
        let shaders = crate::backend::vulkan::VulkanGlslBackend
            .lower_recursive_fft(&ir)
            .unwrap();
        assert_eq!(shaders.len(), program.passes.len());
        for shader in shaders {
            assert_eq!(shader.compile_spirv().unwrap().words[0], 0x0723_0203);
        }

        let mut impulse = vec![Complex64::new(0.0, 0.0); length];
        impulse[1] = Complex64::new(1.0, 0.0);
        let actual = execute_recursive_fft_ir(&ir, &impulse).unwrap();
        let error = actual
            .iter()
            .enumerate()
            .map(|(k, value)| {
                let angle = -TAU * k as f64 / length as f64;
                (*value - Complex64::exp_i(angle)).norm_sqr().sqrt()
            })
            .fold(0.0f64, f64::max);
        assert!(
            error <= 2.0e-8 * length as f64,
            "N1922 dual p31 generator/Four-step fusion impulse error {error:e}"
        );
        ir.validate().unwrap();
    }

    #[test]
    fn forced_rader_upload0_shrinks_coalescing_before_global_register_scale() {
        let length = (16usize * 17) * 256;
        let mut profile = register_leaf_device();
        profile.max_threads_per_block = 64;
        profile.max_workgroup_size[0] = 64;
        let plan = FftPlan::build(FftConfig::new(vec![length])).unwrap();
        let ir = RecursiveFftIr::build(&plan, Direction::Forward, profile).unwrap();
        let schedule = ir
            .rader_forced_upload_schedule
            .as_ref()
            .expect("N69632 should force p17 into two scheduler uploads");
        assert_eq!(schedule.upload_count, 2);
        assert_eq!(schedule.axis_split, vec![272, 256]);
        let uploads = ir.four_step_rader_upload_nodes().unwrap().unwrap();
        assert_eq!(uploads.len(), 2);
        assert!(matches!(uploads[0], RecursiveFftNodeIr::Stockham(_)));
        let RecursiveFftNodeIr::CooleyTukey(low) = &uploads[1] else {
            panic!("N69632 upload0 should remain 16 x p17 Cooley");
        };
        assert_eq!(low.logical_len, 272);
        assert!(matches!(low.right, RecursiveFftNodeIr::FftRader(_)));
        let block = low.pack_right.axis_batch_block.unwrap();
        assert_eq!(block.threads_per_transform, 17);
        assert_eq!(block.grouped_batch, 2);
        assert!(block.transforms_on_x);
        assert!(block.axis_swapped);
        assert_eq!([block.local_size_x, block.local_size_y], [2, 17]);
        assert_eq!(low.twiddle_transpose.axis_batch_block, Some(block));
        assert_eq!(low.scatter_output.axis_batch_block, Some(block));
        assert!(matches!(
            low.scatter_output.output_modifier,
            CooleyTukeyOutputModifier::FourStepLeft(_)
        ));
        let shaders = crate::backend::vulkan::VulkanGlslBackend
            .lower_recursive_fft(&ir)
            .unwrap();
        for shader in shaders {
            shader.compile_spirv().unwrap();
        }
        ir.validate().unwrap();
    }

    #[test]
    fn forced_rader_component_recomputes_pass_local_direct_parent_threads() {
        let length = 17usize * 31 * 64;
        let profile = DeviceProfile {
            shared_memory_bytes: 2 * 1024,
            shared_memory_pow2_bytes: 2 * 1024,
            max_threads_per_block: 1024,
            max_workgroup_size: [1024, 1024, 64],
            ..DeviceProfile::generic(Backend::Vulkan, GpuVendor::Nvidia)
        };
        let tuning = || {
            let mut tuning = crate::PlannerTuning::portable();
            tuning.min_rader_fft_prime = 19;
            tuning.validate().unwrap();
            tuning
        };

        let plan = FftPlan::build(FftConfig::new(vec![length]).with_tuning(tuning())).unwrap();
        let ir = RecursiveFftIr::build(&plan, Direction::Forward, profile).unwrap();
        let schedule = ir
            .rader_forced_upload_schedule
            .as_ref()
            .expect("p31 must force the constrained mixed-Rader axis into three uploads");
        assert_eq!(schedule.axis_split, vec![32, 34, 31]);
        let uploads = ir.four_step_rader_upload_nodes().unwrap().unwrap();
        assert_eq!(uploads.len(), 3);
        let RecursiveFftNodeIr::CooleyTukey(middle) = &uploads[1] else {
            panic!("34-point upload1 should remain a 2 x p17 Cooley component");
        };
        assert_eq!(middle.logical_len, 34);
        assert!(matches!(middle.right, RecursiveFftNodeIr::DirectRader(_)));
        let block = middle
            .pack_right
            .axis_batch_block
            .expect("forced upload1 must consume pass-local type-1 parent scoring");
        assert_eq!(block.threads_per_transform, 18);
        assert_eq!(block.grouped_batch, 4);
        assert!(block.transforms_on_x);
        assert!(!block.axis_swapped);
        assert_eq!([block.local_size_x, block.local_size_y], [4, 18]);
        assert_eq!(middle.twiddle_transpose.axis_batch_block, Some(block));
        assert_eq!(middle.scatter_output.axis_batch_block, Some(block));
        assert!(matches!(
            middle.scatter_output.output_modifier,
            CooleyTukeyOutputModifier::FourStepThreeUpload1(_)
        ));
        let shaders = crate::backend::vulkan::VulkanGlslBackend
            .lower_recursive_fft(&ir)
            .unwrap();
        for shader in shaders {
            shader.compile_spirv().unwrap();
        }
        ir.validate().unwrap();

        let mut narrow_profile = profile;
        narrow_profile.max_threads_per_block = 64;
        narrow_profile.max_workgroup_size[0] = 64;
        let narrow = RecursiveFftIr::build(&plan, Direction::Forward, narrow_profile).unwrap();
        assert_eq!(
            narrow
                .rader_forced_upload_schedule
                .as_ref()
                .expect("64-thread p31 path must retain the same forced split")
                .axis_split,
            vec![32, 34, 31]
        );
        let narrow_uploads = narrow.four_step_rader_upload_nodes().unwrap().unwrap();
        let RecursiveFftNodeIr::CooleyTukey(narrow_middle) = &narrow_uploads[1] else {
            panic!("64-thread 34-point upload1 should remain Cooley");
        };
        let narrow_block = narrow_middle.pack_right.axis_batch_block.unwrap();
        assert_eq!(narrow_block.threads_per_transform, 9);
        assert_eq!(narrow_block.grouped_batch, 4);
        assert!(narrow_block.transforms_on_x);
        assert!(!narrow_block.axis_swapped);
        assert_eq!(
            [narrow_block.local_size_x, narrow_block.local_size_y],
            [4, 9]
        );
        assert_eq!(
            narrow_middle.twiddle_transpose.axis_batch_block,
            Some(narrow_block)
        );
        assert_eq!(
            narrow_middle.scatter_output.axis_batch_block,
            Some(narrow_block)
        );
        let narrow_shaders = crate::backend::vulkan::VulkanGlslBackend
            .lower_recursive_fft(&narrow)
            .unwrap();
        for shader in narrow_shaders {
            shader.compile_spirv().unwrap();
        }
        narrow.validate().unwrap();

        let grouped_batch = 3usize;
        let grouped_plan = FftPlan::build(
            FftConfig::new(vec![length])
                .with_batch_count(5)
                .with_grouped_batch(0, grouped_batch)
                .unwrap()
                .with_tuning(tuning()),
        )
        .unwrap();
        let grouped = RecursiveFftIr::build(&grouped_plan, Direction::Forward, profile).unwrap();
        assert_eq!(
            grouped
                .rader_forced_upload_schedule
                .as_ref()
                .unwrap()
                .axis_split,
            vec![32, 34, 31]
        );
        let grouped_uploads = grouped.four_step_rader_upload_nodes().unwrap().unwrap();
        let RecursiveFftNodeIr::CooleyTukey(grouped_middle) = &grouped_uploads[1] else {
            panic!("grouped 34-point upload1 should remain Cooley");
        };
        let grouped_block = grouped_middle.pack_right.axis_batch_block.unwrap();
        assert_eq!(grouped_block.threads_per_transform, 18);
        assert!(grouped_block.grouped_batch >= grouped_batch);
        assert_eq!(
            grouped_middle.twiddle_transpose.axis_batch_block,
            Some(grouped_block)
        );
        assert_eq!(
            grouped_middle.scatter_output.axis_batch_block,
            Some(grouped_block)
        );
        grouped.validate().unwrap();
    }

    #[test]
    fn grouped_fft_rader_forced_upload_owns_mixed_padded_caller_boundary() {
        let length = 17usize * 31 * 64;
        let batch_count = 5usize;
        let grouped_batch = 3usize;
        let profile = DeviceProfile {
            shared_memory_bytes: 4 * 1024,
            shared_memory_pow2_bytes: 4 * 1024,
            max_threads_per_block: 1024,
            max_workgroup_size: [1024, 1024, 64],
            ..DeviceProfile::generic(Backend::Vulkan, GpuVendor::Nvidia)
        };

        for (direction, padded, expected_input_storage) in [
            (Direction::Forward, false, ScalarType::F16),
            (Direction::Forward, true, ScalarType::F32),
            (Direction::Inverse, true, ScalarType::F16),
        ] {
            let mut config = FftConfig::new(vec![length])
                .with_batch_count(batch_count)
                .with_grouped_batch(0, grouped_batch)
                .unwrap()
                .with_precision(Precision::F16StorageF32Compute)
                .with_inverse_normalization(direction == Direction::Inverse);
            if padded {
                config = config.with_zero_padding(0, 1000, 1200).unwrap();
            }
            let plan = FftPlan::build(config).unwrap();
            let ir = RecursiveFftIr::build(&plan, direction, profile).unwrap();
            assert_eq!(
                ir.rader_forced_upload_schedule.as_ref().unwrap().axis_split,
                vec![32, 34, 31]
            );
            let uploads = ir.four_step_rader_upload_nodes().unwrap().unwrap();
            let RecursiveFftNodeIr::FftRader(high) = &uploads[0] else {
                panic!("31-point high upload should stay FFT Rader");
            };
            assert!(matches!(
                high.io_mapping,
                StockhamIoMapping::FourStepThreeUpload2(_)
            ));
            assert_eq!(
                high.input_strategy,
                crate::RaderFftInputStrategy::GeneratorOrderStockham
            );
            let forward = high
                .forward_recursive()
                .expect("mapped p31 FFT Rader must retain its forward convolution child");
            let RecursiveFftNodeIr::Stockham(forward_kernel) = &forward.root else {
                panic!("mapped p31 FFT Rader should fuse generator loads into Stockham");
            };
            let StockhamIoMapping::RaderGeneratorFourStep(composed) = forward_kernel.io_mapping
            else {
                panic!("mapped p31 FFT Rader should compose generator and Four-step input mapping");
            };
            assert!(matches!(
                composed.caller,
                crate::kernel_ir::RaderFourStepInputMapping::ThreeUpload2(_)
            ));
            assert_eq!(forward_kernel.bindings[0].scalar, expected_input_storage);
            assert_eq!(high.input_storage_scalar, expected_input_storage);
            assert_eq!(high.output_storage_scalar, ScalarType::F32);
            assert_eq!(
                high.scatter.auxiliary_storage_scalar,
                expected_input_storage
            );
            let block = high
                .caller_axis_batch_block
                .expect("mapped FFT Rader caller should own grouped tail");
            assert!(block.grouped_batch >= grouped_batch);
            assert_eq!(high.gather.axis_batch_block, Some(block));
            assert_eq!(high.scatter.axis_batch_block, Some(block));
            assert_eq!(
                high.gather.dispatch.x as usize,
                high.batch_count.div_ceil(block.grouped_batch)
            );
            assert_eq!(high.scatter.dispatch, high.gather.dispatch);
            let gather_name = high.gather.name.clone();
            let program = crate::ProgramIr::recursive_fft(&ir).unwrap();
            assert!(
                program.passes.iter().all(|pass| pass.name != gather_name),
                "mapped mixed-storage p31 upload should not materialize an explicit gather"
            );
            program.validate().unwrap();
            ir.validate().unwrap();
        }
    }

    #[test]
    fn forced_rader_three_upload_maps_direct_rader_component() {
        let length = 17usize * 47 * 128;
        let profile = DeviceProfile {
            shared_memory_bytes: 4 * 1024,
            shared_memory_pow2_bytes: 4 * 1024,
            max_threads_per_block: 1024,
            ..DeviceProfile::generic(Backend::Vulkan, GpuVendor::Nvidia)
        };
        let plan = FftPlan::build(FftConfig::new(vec![length])).unwrap();
        let ir = RecursiveFftIr::build(&plan, Direction::Forward, profile).unwrap();
        let schedule = ir
            .rader_forced_upload_schedule
            .as_ref()
            .expect("constrained p17/p47 axis should require forced Rader uploads");
        assert_eq!(schedule.axis_split, vec![64, 47, 34]);
        assert_eq!(schedule.upload_count, 3);

        let uploads = ir
            .four_step_rader_upload_nodes()
            .unwrap()
            .expect("direct-Rader upload bin should materialize in the Four-step graph");
        assert_eq!(uploads.len(), 3);
        assert_eq!(uploads[0].logical_len(), 34);
        let RecursiveFftNodeIr::DirectRader(middle) = &uploads[1] else {
            panic!("47-point middle upload should be one direct-Rader component");
        };
        assert_eq!(middle.prime, 47);
        assert!(matches!(
            middle.io_mapping,
            StockhamIoMapping::FourStepThreeUpload1(mapping)
                if mapping.logical_len == length && mapping.axis_split == [64, 47, 34]
        ));
        let middle_block = middle
            .axis_batch_block
            .expect("p47 forced upload1 should receive a pass-local direct-Rader block");
        assert_eq!(middle_block.threads_per_transform, 24);
        assert_eq!(middle_block.grouped_batch, 8);
        assert!(middle_block.transforms_on_x);
        assert!(!middle_block.axis_swapped);
        assert_eq!(
            [middle_block.local_size_x, middle_block.local_size_y],
            [8, 24]
        );
        assert_eq!(uploads[2].logical_len(), 64);

        let input = sample(length);
        let actual = execute_recursive_fft_ir(&ir, &input).unwrap();
        let expected = crate::reference::fft(&input, Direction::Forward, false).unwrap();
        assert!(max_error(&actual, &expected) <= 6.0e-8 * length.ilog2() as f64);
        ir.validate().unwrap();
    }

    #[test]
    fn forced_rader_two_upload_direct_rader_owns_caller_input_boundary() {
        let length = 11usize * 17 * 47;
        let profile = DeviceProfile {
            shared_memory_bytes: 8 * 1024,
            shared_memory_pow2_bytes: 8 * 1024,
            max_threads_per_block: 1024,
            ..DeviceProfile::generic(Backend::Vulkan, GpuVendor::Nvidia)
        };

        let plan = FftPlan::build(FftConfig::new(vec![length])).unwrap();
        let ir = RecursiveFftIr::build(&plan, Direction::Forward, profile).unwrap();
        assert_eq!(
            ir.rader_forced_upload_schedule.as_ref().unwrap().axis_split,
            vec![187, 47]
        );
        let uploads = ir.four_step_rader_upload_nodes().unwrap().unwrap();
        let RecursiveFftNodeIr::DirectRader(high) = &uploads[0] else {
            panic!("47-point high upload should be direct Rader");
        };
        assert!(matches!(
            high.io_mapping,
            StockhamIoMapping::FourStepRight(mapping)
                if mapping.logical_len == length
                    && mapping.left_len == 187
                    && mapping.right_len == 47
        ));
        let high_block = high
            .axis_batch_block
            .expect("p47 two-upload high leaf should receive automatic caller grouping");
        assert_eq!(high_block.threads_per_transform, 24);
        assert_eq!(high_block.grouped_batch, 16);
        assert!(high_block.transforms_on_x);
        assert!(!high_block.axis_swapped);
        assert_eq!([high_block.local_size_x, high_block.local_size_y], [16, 24]);
        let input = sample(length);
        let actual = execute_recursive_fft_ir(&ir, &input).unwrap();
        let expected = crate::reference::fft(&input, Direction::Forward, false).unwrap();
        assert!(max_error(&actual, &expected) <= 6.0e-8 * length.ilog2() as f64);

        let batch_count = 5usize;
        for (direction, padded, expected_input_storage) in [
            (Direction::Forward, false, ScalarType::F16),
            (Direction::Forward, true, ScalarType::F32),
            (Direction::Inverse, true, ScalarType::F16),
        ] {
            let mut config = FftConfig::new(vec![length])
                .with_batch_count(batch_count)
                .with_grouped_batch(0, 3)
                .unwrap()
                .with_precision(Precision::F16StorageF32Compute)
                .with_inverse_normalization(direction == Direction::Inverse);
            if padded {
                config = config.with_zero_padding(0, 1000, 1200).unwrap();
            }
            let plan = FftPlan::build(config).unwrap();
            let grouped = RecursiveFftIr::build(&plan, direction, profile).unwrap();
            assert_eq!(
                grouped
                    .rader_forced_upload_schedule
                    .as_ref()
                    .unwrap()
                    .axis_split,
                vec![47, 17, 11]
            );
            let uploads = grouped.four_step_rader_upload_nodes().unwrap().unwrap();
            assert_eq!(uploads.len(), 3);
            let RecursiveFftNodeIr::Stockham(high) = &uploads[0] else {
                panic!("grouped mixed 11-point high upload should stay Stockham");
            };
            assert_eq!(high.bindings[0].scalar, expected_input_storage);
            assert!(matches!(
                high.io_mapping,
                StockhamIoMapping::FourStepThreeUpload2(_)
            ));
            let RecursiveFftNodeIr::DirectRader(low) = &uploads[2] else {
                panic!("grouped mixed 47-point low upload should stay direct Rader");
            };
            assert_eq!(low.prime, 47);
            assert_eq!(low.input_storage_scalar, ScalarType::F32);
            assert_eq!(
                low.output_storage_scalar,
                if direction == Direction::Inverse {
                    ScalarType::F32
                } else {
                    ScalarType::F16
                }
            );
            assert!(low.axis_batch_block.is_some());
            assert!(matches!(
                low.io_mapping,
                StockhamIoMapping::FourStepThreeUpload0(_)
            ));
            assert_eq!(
                low.dispatch.x as usize,
                low.batch_count
                    .div_ceil(low.axis_batch_block.unwrap().grouped_batch)
            );
            grouped.validate().unwrap();
        }
    }

    #[test]
    fn forced_rader_four_step_mixed_storage_and_padding_keep_directional_boundaries() {
        let length = 17usize * 300;
        let range = crate::ZeroPaddingRange {
            left: 211,
            right: 317,
        };
        let mut profile = register_leaf_device();
        profile.max_threads_per_block = 128;

        let input_scalar = |node: &RecursiveFftNodeIr| match node {
            RecursiveFftNodeIr::Stockham(kernel) => {
                kernel
                    .bindings
                    .iter()
                    .find(|binding| binding.role == crate::BufferRole::Input)
                    .unwrap()
                    .scalar
            }
            RecursiveFftNodeIr::CooleyTukey(cooley) => cooley.pack_right.input_storage_scalar,
            _ => panic!("forced Rader Four-step boundary unexpectedly became a Rader leaf"),
        };
        let output_scalar = |node: &RecursiveFftNodeIr| match node {
            RecursiveFftNodeIr::Stockham(kernel) => {
                kernel
                    .bindings
                    .iter()
                    .find(|binding| binding.role == crate::BufferRole::Output)
                    .unwrap()
                    .scalar
            }
            RecursiveFftNodeIr::CooleyTukey(cooley) => cooley.scatter_output.output_storage_scalar,
            _ => panic!("forced Rader Four-step boundary unexpectedly became a Rader leaf"),
        };

        for direction in [Direction::Forward, Direction::Inverse] {
            let plan = FftPlan::build(
                FftConfig::new(vec![length])
                    .with_precision(Precision::F16StorageF32Compute)
                    .with_inverse_normalization(direction == Direction::Inverse),
            )
            .unwrap();
            let ir = RecursiveFftIr::build(&plan, direction, profile).unwrap();
            let uploads = ir
                .four_step_rader_upload_nodes()
                .unwrap()
                .expect("mixed-storage forced Rader plan should retain Four-step upload nodes");
            assert_eq!(uploads.len(), 2);
            assert_eq!(input_scalar(&uploads[0]), ScalarType::F16);
            assert_eq!(output_scalar(&uploads[0]), ScalarType::F32);
            assert_eq!(input_scalar(&uploads[1]), ScalarType::F32);
            assert_eq!(output_scalar(&uploads[1]), ScalarType::F16);
            ir.validate().unwrap();
        }

        for direction in [Direction::Forward, Direction::Inverse] {
            let plan = FftPlan::build(
                FftConfig::new(vec![length])
                    .with_precision(Precision::F16StorageF32Compute)
                    .with_inverse_normalization(direction == Direction::Inverse)
                    .with_zero_padding(0, range.left, range.right)
                    .unwrap(),
            )
            .unwrap();
            let ir = RecursiveFftIr::build(&plan, direction, profile).unwrap();
            let uploads = ir
                .four_step_rader_upload_nodes()
                .unwrap()
                .expect("zero-padded forced Rader plan should retain Four-step upload nodes");
            assert_eq!(uploads.len(), 2);
            match direction {
                Direction::Forward => {
                    assert_eq!(input_scalar(&uploads[0]), ScalarType::F32);
                    assert_eq!(output_scalar(&uploads[1]), ScalarType::F16);
                }
                Direction::Inverse => {
                    assert_eq!(input_scalar(&uploads[0]), ScalarType::F16);
                    assert_eq!(output_scalar(&uploads[1]), ScalarType::F32);
                }
            }
            assert_eq!(output_scalar(&uploads[0]), ScalarType::F32);
            assert_eq!(input_scalar(&uploads[1]), ScalarType::F32);

            let input = sample(length);
            let actual = execute_recursive_fft_ir(&ir, &input).unwrap();
            let expected = match direction {
                Direction::Forward => {
                    let mut manual = input.clone();
                    manual[range.left..range.right].fill(Complex64::new(0.0, 0.0));
                    crate::reference::fft(&manual, Direction::Forward, false).unwrap()
                }
                Direction::Inverse => {
                    let mut manual =
                        crate::reference::fft(&input, Direction::Inverse, true).unwrap();
                    manual[range.left..range.right].fill(Complex64::new(0.0, 0.0));
                    manual
                }
            };
            assert!(max_error(&actual, &expected) <= 5.0e-8 * length.ilog2() as f64);
            ir.validate().unwrap();
        }
    }

    #[test]
    fn grouped_forced_rader_four_step_owns_component_boundaries_and_tail() {
        let length = 17usize * 8192;
        let batch_count = 5usize;
        let grouped_batch = 3usize;
        let config = FftConfig::new(vec![length])
            .with_batch_count(batch_count)
            .with_grouped_batch(0, grouped_batch)
            .unwrap();
        let plan = FftPlan::build(config).unwrap();
        let mut profile = register_leaf_device();
        profile.max_workgroup_size = [1024, 1024, 64];
        let ir = RecursiveFftIr::build(&plan, Direction::Forward, profile).unwrap();
        assert_eq!(ir.axis0_grouped_batch_override, Some(grouped_batch));
        let schedule = ir
            .rader_forced_upload_schedule
            .as_ref()
            .expect("grouped composite p17 containers should keep forced-Rader geometry");
        assert_eq!(schedule.axis_split, vec![512, 272]);
        let four_step = ir
            .four_step_plan
            .as_ref()
            .expect("grouped forced-Rader axis should materialize Four-step ownership");
        assert_eq!(four_step.uploads.len(), 2);
        assert!(
            four_step
                .uploads
                .iter()
                .all(|upload| upload.axis_block.is_some())
        );

        let uploads = ir
            .four_step_rader_upload_nodes()
            .unwrap()
            .expect("grouped forced-Rader Four-step must materialize mapped upload nodes");
        assert_eq!(uploads.len(), 2);
        let RecursiveFftNodeIr::CooleyTukey(high) = &uploads[0] else {
            panic!("272-point grouped upload should remain a Cooley component");
        };
        let high_block = high
            .pack_right
            .axis_batch_block
            .expect("high Cooley boundary should own grouped parent transforms");
        assert_eq!(high.twiddle_transpose.axis_batch_block, Some(high_block));
        assert_eq!(high.scatter_output.axis_batch_block, Some(high_block));
        assert_eq!(
            high.pack_right.dispatch.x as usize,
            high.batch_count.div_ceil(high_block.grouped_batch)
        );
        let RecursiveFftNodeIr::Stockham(low) = &uploads[1] else {
            panic!("512-point grouped upload should remain Stockham");
        };
        let low_meta = four_step.uploads[1].axis_block.unwrap();
        assert_eq!(
            low.workgroup_grouping.transforms_per_workgroup,
            low_meta.grouped_batch
        );
        assert_eq!(
            low.dispatch.x as usize,
            low.batch_count.div_ceil(low_meta.grouped_batch)
        );
        assert!(low.batch_count % low_meta.grouped_batch != 0);
        ir.validate().unwrap();
    }

    #[test]
    fn grouped_bandwidth_boost_forced_rader_rebuilds_component_blocks_from_new_split() {
        let length = 17usize * 16_384;
        let batch_count = 5usize;
        let grouped_batch = 3usize;
        let mut profile = DeviceProfile::generic(Backend::Vulkan, GpuVendor::Intel);
        profile.shared_memory_bytes = 32 * 1024;
        profile.shared_memory_pow2_bytes = 32 * 1024;
        profile.max_threads_per_block = 1024;
        profile.max_workgroup_size = [1024, 1024, 64];
        let config = FftConfig::new(vec![length])
            .with_batch_count(batch_count)
            .with_bandwidth_boost(2)
            .with_grouped_batch(0, grouped_batch)
            .unwrap()
            .resolve_tuning_for_device(profile);
        let plan =
            FftPlan::build_c2c_child_for_device(config, profile, C2cDeviceAxisClass::Strided)
                .unwrap();
        let ir = RecursiveFftIr::build(&plan, Direction::Forward, profile).unwrap();
        assert_eq!(ir.axis0_grouped_batch_override, Some(grouped_batch));
        let schedule = ir
            .rader_forced_upload_schedule
            .as_ref()
            .expect("grouped B=2 composite p17 must retain forced-Rader scheduling");
        assert_eq!(schedule.upload_count, 2);
        assert_eq!(schedule.axis_split, vec![544, 512]);
        let four_step = ir
            .four_step_plan
            .as_ref()
            .expect("grouped B=2 forced-Rader axis must materialize Four-step blocks");
        assert_eq!(four_step.uploads.len(), 2);
        assert!(
            four_step
                .uploads
                .iter()
                .all(|upload| upload.axis_block.is_some())
        );
        assert_eq!(four_step.uploads[0].fft_len, 512);
        assert_eq!(four_step.uploads[0].stage_start_size, 544);
        assert_eq!(four_step.uploads[0].transform_count, batch_count * 544);
        assert_eq!(four_step.uploads[1].fft_len, 544);
        assert_eq!(four_step.uploads[1].transform_count, batch_count * 512);
        assert!(
            four_step
                .uploads
                .iter()
                .all(|upload| !upload.transform_count.is_multiple_of(grouped_batch))
        );

        let uploads = ir
            .four_step_rader_upload_nodes()
            .unwrap()
            .expect("grouped B=2 forced-Rader plan must materialize mapped components");
        assert_eq!(uploads.len(), 2);
        let RecursiveFftNodeIr::Stockham(high) = &uploads[0] else {
            panic!("512-point B=2 high upload should remain Stockham");
        };
        let high_block = four_step.uploads[0].axis_block.unwrap();
        assert_eq!(
            high.workgroup_grouping.transforms_per_workgroup,
            high_block.grouped_batch
        );
        assert_eq!(high_block.grouped_batch, 8);
        assert_ne!(high_block.grouped_batch, grouped_batch);
        assert_eq!(
            high.dispatch.x as usize,
            high.batch_count.div_ceil(high_block.grouped_batch)
        );
        let RecursiveFftNodeIr::CooleyTukey(low) = &uploads[1] else {
            panic!("544-point B=2 low upload should remain 32 x p17 Cooley");
        };
        let low_block = low
            .pack_right
            .axis_batch_block
            .expect("544-point low component must own its rebuilt grouped boundary");
        assert_eq!(low.twiddle_transpose.axis_batch_block, Some(low_block));
        assert_eq!(low.scatter_output.axis_batch_block, Some(low_block));
        assert_eq!(
            low.pack_right.dispatch.x as usize,
            low.batch_count.div_ceil(low_block.grouped_batch)
        );
        assert_eq!(low_block.grouped_batch, grouped_batch);

        let program = crate::ProgramIr::recursive_fft(&ir).unwrap();
        program.validate().unwrap();
        let shaders = crate::backend::vulkan::VulkanGlslBackend
            .lower_recursive_fft(&ir)
            .unwrap();
        assert_eq!(shaders.len(), program.passes.len());
        for shader in shaders {
            assert_eq!(shader.compile_spirv().unwrap().words[0], 0x0723_0203);
        }
    }

    #[test]
    fn grouped_forced_rader_three_upload_owns_all_component_boundaries() {
        let length = 17usize * 65_536;
        let batch_count = 5usize;
        let grouped_batch = 3usize;
        let plan = FftPlan::build(
            FftConfig::new(vec![length])
                .with_batch_count(batch_count)
                .with_grouped_batch(0, grouped_batch)
                .unwrap(),
        )
        .unwrap();
        let mut profile = register_leaf_device();
        profile.max_workgroup_size = [1024, 1024, 64];
        let ir = RecursiveFftIr::build(&plan, Direction::Forward, profile).unwrap();
        let schedule = ir.rader_forced_upload_schedule.as_ref().unwrap();
        assert_eq!(schedule.axis_split, vec![128, 68, 128]);
        let four_step = ir.four_step_plan.as_ref().unwrap();
        assert_eq!(four_step.uploads.len(), 3);
        assert!(
            four_step
                .uploads
                .iter()
                .all(|upload| upload.axis_block.is_some())
        );
        assert_eq!(
            four_step.uploads[2].axis_block.unwrap().grouped_batch,
            grouped_batch
        );
        assert!(
            !four_step.uploads[2]
                .transform_count
                .is_multiple_of(grouped_batch)
        );

        let uploads = ir.four_step_rader_upload_nodes().unwrap().unwrap();
        assert_eq!(uploads.len(), 3);
        let RecursiveFftNodeIr::Stockham(high) = &uploads[0] else {
            panic!("128-point high upload should remain Stockham");
        };
        assert_eq!(
            high.workgroup_grouping.transforms_per_workgroup,
            four_step.uploads[0].axis_block.unwrap().grouped_batch
        );
        let RecursiveFftNodeIr::CooleyTukey(middle) = &uploads[1] else {
            panic!("68-point middle upload should remain Cooley");
        };
        let middle_block = middle.pack_right.axis_batch_block.unwrap();
        assert_eq!(
            middle.twiddle_transpose.axis_batch_block,
            Some(middle_block)
        );
        assert_eq!(middle.scatter_output.axis_batch_block, Some(middle_block));
        assert_eq!(
            middle.pack_right.dispatch.x as usize,
            middle.batch_count.div_ceil(middle_block.grouped_batch)
        );
        let RecursiveFftNodeIr::Stockham(low) = &uploads[2] else {
            panic!("128-point low upload should remain Stockham");
        };
        let low_block = four_step.uploads[2].axis_block.unwrap();
        assert_eq!(
            low.workgroup_grouping.transforms_per_workgroup,
            low_block.grouped_batch
        );
        assert_eq!(
            low.dispatch.x as usize,
            low.batch_count.div_ceil(low_block.grouped_batch)
        );
        ir.validate().unwrap();
    }

    #[test]
    fn f16_higher_axis_single_upload_preserves_scheduler_precision() {
        for (vendor, expected_boost, expected_threads) in [
            (GpuVendor::Nvidia, 1usize, 64usize),
            (GpuVendor::Amd, 1usize, 64usize),
            (GpuVendor::Intel, 2usize, 32usize),
        ] {
            let mut profile = DeviceProfile::generic(Backend::Vulkan, vendor);
            profile.shared_memory_bytes = 32 * 1024;
            profile.shared_memory_pow2_bytes = 32 * 1024;
            profile.max_threads_per_block = 1024;
            profile.max_workgroup_size = [1024, 1024, 64];
            let plan = FftPlan::build_c2c_child_for_device(
                FftConfig::new(vec![512])
                    .with_batch_count(64)
                    .with_precision(Precision::F16StorageF32Compute),
                profile,
                C2cDeviceAxisClass::Strided,
            )
            .unwrap();
            let ir = RecursiveFftIr::build(&plan, Direction::Forward, profile)
                .unwrap()
                .with_other_axis_single_upload_block(64, profile)
                .unwrap();
            let schedule = ir.stockham_upload_schedule.as_ref().unwrap();
            assert_eq!(schedule.upload_count, 1, "{vendor:?}");
            assert_eq!(schedule.register_boost, expected_boost, "{vendor:?}");
            let RecursiveFftNodeIr::Stockham(root) = &ir.root else {
                panic!("{vendor:?} F16 N512 higher axis should remain Stockham");
            };
            assert_eq!(
                root.workgroup_grouping.transforms_per_workgroup, 8,
                "{vendor:?}"
            );
            assert_eq!(
                root.workgroup_grouping.threads_per_transform, expected_threads,
                "{vendor:?}"
            );
            assert_eq!(
                [root.workgroup_size.x, root.workgroup_size.y],
                [8, expected_threads as u32],
                "{vendor:?}"
            );
            assert_eq!(
                root.execution_layout,
                if expected_boost == 1 {
                    crate::StockhamExecutionLayout::RegisterSingleShared
                } else {
                    crate::StockhamExecutionLayout::RegisterBoostSingleShared
                },
                "{vendor:?}"
            );
            assert_eq!(root.dispatch.x, 8, "{vendor:?}");
            let shaders = crate::backend::vulkan::VulkanGlslBackend
                .lower_recursive_fft(&ir)
                .unwrap();
            assert_eq!(shaders.len(), 1, "{vendor:?}");
            if vendor == GpuVendor::Intel {
                let glsl = &shaders[0].glsl;
                assert!(glsl.contains("uint vkfft_container = gl_LocalInvocationID.x;"));
                assert!(glsl.contains("uint vkfft_lid = gl_LocalInvocationID.y;"));
                assert!(glsl.contains(
                    "uint vkfft_shared_base = vkfft_container * VKFFT_SHARED_ELEMENTS_PER_TRANSFORM;"
                ));
                assert!(glsl.contains("vkfft_smem_a[vkfft_shared_base + ("));
                assert!(glsl.contains("if (vkfft_batch_active)"));
            }
            if vendor == GpuVendor::Intel {
                let mut impulse = vec![Complex64::new(0.0, 0.0); 512 * 64];
                for batch in 0..64 {
                    impulse[batch * 512 + 1] = Complex64::new(1.0, 0.0);
                }
                let actual = execute_recursive_fft_ir(&ir, &impulse).unwrap();
                let error = actual
                    .iter()
                    .enumerate()
                    .map(|(index, value)| {
                        let k = index % 512;
                        let angle = -TAU * k as f64 / 512.0;
                        (*value - Complex64::exp_i(angle)).norm_sqr().sqrt()
                    })
                    .fold(0.0f64, f64::max);
                assert!(error <= 2.0e-8 * 512.0, "Intel F16 N512 error={error}");
            }
            assert_eq!(shaders[0].compile_spirv().unwrap().words[0], 0x0723_0203);
            ir.validate().unwrap();
        }
    }

    #[test]
    fn f16_axis0_fft_rader_grouped_batch_preserves_scheduler_precision() {
        let mut profile = DeviceProfile::generic(Backend::Vulkan, GpuVendor::Nvidia);
        profile.shared_memory_bytes = 32 * 1024;
        profile.shared_memory_pow2_bytes = 32 * 1024;
        profile.max_threads_per_block = 1024;
        profile.max_workgroup_size = [1024, 1024, 64];
        let config = FftConfig::new(vec![769])
            .with_batch_count(32)
            .with_precision(Precision::F16StorageF32Compute)
            .with_grouped_batch(0, 8)
            .unwrap();
        let plan = FftPlan::build_for_device(config, profile).unwrap();
        let ir = RecursiveFftIr::build(&plan, Direction::Forward, profile).unwrap();
        assert_eq!(ir.external_storage_scalar(), ScalarType::F16);
        let RecursiveFftNodeIr::FftRader(rader) = &ir.root else {
            panic!("F16 p769 grouped axis should remain standalone FFT-Rader");
        };
        let block = rader
            .axis_batch_block
            .expect("F16 p769 grouped axis should retain the upstream caller block");
        assert_eq!(block.threads_per_transform, 52);
        assert_eq!(block.grouped_batch, 4);
        assert_eq!([block.local_size_x, block.local_size_y], [52, 4]);
        assert!(!block.transforms_on_x);
        assert!(!block.axis_swapped);

        let shaders = crate::backend::vulkan::VulkanGlslBackend
            .lower_recursive_fft(&ir)
            .unwrap();
        assert!(!shaders.is_empty());
        for shader in &shaders {
            assert_eq!(shader.compile_spirv().unwrap().words[0], 0x0723_0203);
        }

        let mut impulse = vec![Complex64::new(0.0, 0.0); 769 * 32];
        for batch in 0..32 {
            impulse[batch * 769 + 1] = Complex64::new(1.0, 0.0);
        }
        let actual = execute_recursive_fft_ir(&ir, &impulse).unwrap();
        let error = actual
            .iter()
            .enumerate()
            .map(|(index, value)| {
                let k = index % 769;
                let angle = -TAU * k as f64 / 769.0;
                (*value - Complex64::exp_i(angle)).norm_sqr().sqrt()
            })
            .fold(0.0f64, f64::max);
        assert!(error <= 2.0e-8 * 769.0, "F16 p769 grouped error={error}");
        ir.validate().unwrap();
    }

    #[test]
    fn upstream_power_of_two_upload_schedule_drives_recursive_factors() {
        for (length, expected_split) in [
            (4096usize, vec![4096usize]),
            (1_048_576usize, vec![1024usize, 1024]),
            (8_388_608usize, vec![256usize, 128, 256]),
        ] {
            let plan = FftPlan::build(FftConfig::new(vec![length])).unwrap();
            let ir = RecursiveFftIr::build(&plan, Direction::Forward, upstream_scheduler_device())
                .unwrap();
            let schedule = ir
                .stockham_upload_schedule
                .as_ref()
                .expect("eligible power-of-two tree should retain the upstream upload schedule");
            assert_eq!(schedule.axis_split, expected_split);
            assert_eq!(schedule.upload_count, expected_split.len());
            let mut leaves = Vec::new();
            collect_stockham_leaf_lengths(&ir.root, &mut leaves).unwrap();
            assert_eq!(leaves, expected_split);
            ir.validate().unwrap();
        }
    }

    #[test]
    fn amd_32k_eight_meg_stockham_preserves_three_upload_tree() {
        let profile = DeviceProfile {
            shared_memory_bytes: 32 * 1024,
            shared_memory_pow2_bytes: 32 * 1024,
            max_threads_per_block: 1024,
            max_workgroup_size: [1024, 1024, 64],
            ..DeviceProfile::generic(Backend::Vulkan, GpuVendor::Amd)
        };
        let length = 8_388_608usize;
        let plan = FftPlan::build(FftConfig::new(vec![length])).unwrap();
        let ir = RecursiveFftIr::build(&plan, Direction::Forward, profile).unwrap();
        let schedule = ir
            .stockham_upload_schedule
            .as_ref()
            .expect("AMD 32 KiB N8388608 should retain the upstream Stockham upload schedule");
        assert_eq!(schedule.upload_count, 3);
        assert_eq!(schedule.axis_split, vec![64, 512, 256]);
        assert!(root_is_three_stockham_uploads(&ir.root, 64, 512, 256));
        let four_step = ir
            .four_step_plan
            .as_ref()
            .expect("AMD 32 KiB N8388608 should materialize the three-upload Four-step plan");
        assert_eq!(four_step.uploads.len(), 3);
        for (upload_id, fft_len, transform_count, threads, grouped, local, swapped) in [
            (
                0usize,
                64usize,
                131_072usize,
                8usize,
                32usize,
                [32usize, 8usize],
                true,
            ),
            (1, 512, 16_384, 64, 8, [8, 64], false),
            (2, 256, 32_768, 32, 16, [16, 32], false),
        ] {
            let upload = four_step
                .uploads
                .iter()
                .find(|upload| upload.axis_upload_id == upload_id)
                .unwrap();
            assert_eq!(upload.fft_len, fft_len);
            assert_eq!(upload.transform_count, transform_count);
            let block = upload.axis_block.unwrap();
            assert_eq!(block.threads_per_transform, threads);
            assert_eq!(block.grouped_batch, grouped);
            assert!(block.transforms_on_x);
            assert_eq!([block.local_size_x, block.local_size_y], local);
            assert_eq!(block.axis_swapped, swapped);
        }
        ir.validate().unwrap();
    }

    #[test]
    fn extracted_strided_axis_preserves_upload_context_into_recursive_materialization() {
        let mut profile = upstream_scheduler_device();
        profile.shared_memory_bytes = 48 * 1024;
        profile.shared_memory_pow2_bytes = 32 * 1024;
        profile.max_threads_per_block = 1024;
        profile.max_workgroup_size = [1024, 1024, 64];
        let config = FftConfig::new(vec![8_192usize]);

        let contiguous_plan = FftPlan::build_c2c_child_for_device(
            config.clone(),
            profile,
            C2cDeviceAxisClass::Contiguous,
        )
        .unwrap();
        let contiguous =
            RecursiveFftIr::build(&contiguous_plan, Direction::Forward, profile).unwrap();
        let contiguous_schedule = contiguous
            .stockham_upload_schedule
            .as_ref()
            .expect("contiguous N8192 should retain scheduler metadata");
        assert_eq!(contiguous_schedule.upload_count, 1);
        assert_eq!(contiguous_schedule.axis_split, vec![8_192]);
        assert!(contiguous.four_step_plan.is_none());

        let strided_plan =
            FftPlan::build_c2c_child_for_device(config, profile, C2cDeviceAxisClass::Strided)
                .unwrap();
        let strided = RecursiveFftIr::build(&strided_plan, Direction::Forward, profile).unwrap();
        let strided_schedule = strided
            .stockham_upload_schedule
            .as_ref()
            .expect("strided N8192 should retain scheduler metadata");
        assert_eq!(strided_schedule.upload_count, 2);
        assert_eq!(strided_schedule.axis_split, vec![128, 64]);
        assert!(strided.four_step_plan.is_some());
        let shaders = crate::backend::vulkan::VulkanGlslBackend
            .lower_recursive_fft(&strided)
            .unwrap();
        assert_eq!(shaders.len(), 2);
        for shader in shaders {
            assert_eq!(shader.compile_spirv().unwrap().words[0], 0x0723_0203);
        }
        contiguous.validate().unwrap();
        strided.validate().unwrap();
    }

    #[test]
    fn strided_bandwidth_boost_propagates_from_config_into_recursive_split() {
        let mut profile = upstream_scheduler_device();
        profile.shared_memory_bytes = 48 * 1024;
        profile.shared_memory_pow2_bytes = 32 * 1024;
        profile.max_threads_per_block = 1024;
        profile.max_workgroup_size = [1024, 1024, 64];
        let length = 2_097_152usize;

        let baseline_plan = FftPlan::build_c2c_child_for_device(
            FftConfig::new(vec![length]),
            profile,
            C2cDeviceAxisClass::Strided,
        )
        .unwrap();
        let baseline = RecursiveFftIr::build(&baseline_plan, Direction::Forward, profile).unwrap();
        let baseline_schedule = baseline
            .stockham_upload_schedule
            .as_ref()
            .expect("baseline strided N2097152 should retain scheduler metadata");
        assert_eq!(baseline_schedule.upload_count, 3);
        assert_eq!(baseline_schedule.axis_split, vec![128, 128, 128]);

        let boosted_plan = FftPlan::build_c2c_child_for_device(
            FftConfig::new(vec![length]).with_bandwidth_boost(2),
            profile,
            C2cDeviceAxisClass::Strided,
        )
        .unwrap();
        let boosted = RecursiveFftIr::build(&boosted_plan, Direction::Forward, profile).unwrap();
        let boosted_schedule = boosted
            .stockham_upload_schedule
            .as_ref()
            .expect("boosted strided N2097152 should retain scheduler metadata");
        assert_eq!(boosted_schedule.upload_count, 2);
        assert_eq!(boosted_schedule.axis_split, vec![2_048, 1_024]);
        let four_step = boosted
            .four_step_plan
            .as_ref()
            .expect("boosted two-upload split should materialize Four-step fusion");
        assert_eq!(four_step.uploads.len(), 2);
        let shaders = crate::backend::vulkan::VulkanGlslBackend
            .lower_recursive_fft(&boosted)
            .unwrap();
        assert_eq!(shaders.len(), 2);
        for shader in shaders {
            assert_eq!(shader.compile_spirv().unwrap().words[0], 0x0723_0203);
        }
        baseline.validate().unwrap();
        boosted.validate().unwrap();
    }

    #[test]
    fn non_nvidia_large_pow2_schedule_materializes_three_upload_four_step() {
        let length = 524_288usize;
        for vendor in [GpuVendor::Amd, GpuVendor::Intel] {
            let profile = DeviceProfile {
                shared_memory_bytes: 48 * 1024,
                shared_memory_pow2_bytes: 32 * 1024,
                max_threads_per_block: 1024,
                max_workgroup_size: [1024, 1024, 64],
                ..DeviceProfile::generic(Backend::Vulkan, vendor)
            };
            let plan = FftPlan::build(FftConfig::new(vec![length])).unwrap();
            let ir = RecursiveFftIr::build(&plan, Direction::Forward, profile).unwrap();
            let schedule = ir
                .stockham_upload_schedule
                .as_ref()
                .expect("non-NVIDIA large pow2 should retain upload metadata");
            assert_eq!(schedule.upload_count, 3, "vendor={vendor:?}");
            assert_eq!(schedule.axis_split, vec![64, 128, 64], "vendor={vendor:?}");
            let four_step = ir
                .four_step_plan
                .as_ref()
                .expect("non-NVIDIA large pow2 should materialize Four-step fusion");
            assert_eq!(
                four_step
                    .uploads
                    .iter()
                    .map(|upload| upload.axis_upload_id)
                    .collect::<Vec<_>>(),
                vec![2, 1, 0],
                "vendor={vendor:?}"
            );
            assert_eq!(
                four_step
                    .uploads
                    .iter()
                    .map(|upload| upload.fft_len)
                    .collect::<Vec<_>>(),
                vec![64, 128, 64],
                "vendor={vendor:?}"
            );
            assert_eq!(
                four_step
                    .uploads
                    .iter()
                    .map(|upload| upload.stage_start_size)
                    .collect::<Vec<_>>(),
                vec![8_192, 64, 1],
                "vendor={vendor:?}"
            );
            let expected_blocks = match vendor {
                GpuVendor::Amd => vec![[64usize, 8usize], [48, 16], [64, 8]],
                GpuVendor::Intel => vec![[96usize, 8usize], [32, 16], [32, 8]],
                _ => unreachable!(),
            };
            assert_eq!(
                four_step
                    .uploads
                    .iter()
                    .map(|upload| {
                        let block = upload.axis_block.expect("cross-vendor Four-step block");
                        assert!(block.transforms_on_x);
                        [block.local_size_x, block.local_size_y]
                    })
                    .collect::<Vec<_>>(),
                expected_blocks,
                "vendor={vendor:?}"
            );
            let kernels = ir
                .four_step_stockham_upload_kernels()
                .unwrap()
                .expect("cross-vendor three-upload kernels should materialize");
            assert_eq!(kernels.len(), 3);
            let shaders = crate::backend::vulkan::VulkanGlslBackend
                .lower_recursive_fft(&ir)
                .unwrap();
            assert_eq!(shaders.len(), 3);
            for shader in shaders {
                assert_eq!(shader.compile_spirv().unwrap().words[0], 0x0723_0203);
            }
            ir.validate().unwrap();
        }
    }

    #[test]
    fn hip_large_pow2_policy_materializes_pinned_three_upload_blocks() {
        let length = 2_097_152usize;
        let profile = DeviceProfile {
            shared_memory_bytes: 48 * 1024,
            shared_memory_pow2_bytes: 32 * 1024,
            max_threads_per_block: 1024,
            max_workgroup_size: [1024, 1024, 64],
            ..DeviceProfile::generic(Backend::Hip, GpuVendor::Amd)
        };
        let plan = FftPlan::build(FftConfig::new(vec![length])).unwrap();
        let ir = RecursiveFftIr::build(&plan, Direction::Forward, profile).unwrap();
        let schedule = ir
            .stockham_upload_schedule
            .as_ref()
            .expect("HIP N2097152 should retain Stockham upload metadata");
        assert_eq!(schedule.register_boost, 1);
        assert_eq!(schedule.upload_count, 3);
        assert_eq!(schedule.axis_split, vec![64, 512, 64]);

        let four_step = ir
            .four_step_plan
            .as_ref()
            .expect("HIP N2097152 should materialize three-upload Four-step");
        assert_eq!(four_step.uploads.len(), 3);
        assert_eq!(
            four_step
                .uploads
                .iter()
                .map(|upload| upload.axis_upload_id)
                .collect::<Vec<_>>(),
            vec![2, 1, 0]
        );
        for (upload_id, fft_len, transform_count, grouped, local, swapped) in [
            (
                0usize,
                64usize,
                32_768usize,
                64usize,
                [64usize, 8usize],
                true,
            ),
            (1, 512, 4_096, 8, [8, 64], false),
            (2, 64, 32_768, 64, [64, 8], false),
        ] {
            let upload = four_step
                .uploads
                .iter()
                .find(|upload| upload.axis_upload_id == upload_id)
                .expect("HIP pinned upload id must materialize");
            assert_eq!(upload.fft_len, fft_len);
            assert_eq!(upload.transform_count, transform_count);
            let block = upload
                .axis_block
                .expect("HIP pinned upload must retain AxisBlock");
            assert_eq!(block.grouped_batch, grouped);
            assert_eq!([block.local_size_x, block.local_size_y], local);
            assert!(block.transforms_on_x);
            assert_eq!(block.axis_swapped, swapped);
        }

        let kernels = ir
            .four_step_stockham_upload_kernels()
            .unwrap()
            .expect("HIP pinned Stockham uploads should materialize kernels");
        assert_eq!(kernels.len(), 3);
        let shaders = crate::backend::vulkan::VulkanGlslBackend
            .lower_recursive_fft(&ir)
            .unwrap();
        assert_eq!(shaders.len(), 3);
        for shader in shaders {
            assert_eq!(shader.compile_spirv().unwrap().words[0], 0x0723_0203);
        }
        ir.validate().unwrap();
    }

    #[test]
    fn two_upload_scheduler_materializes_upstream_four_step_order_and_mappings() {
        let length = 1_048_576usize;
        let plan = FftPlan::build(FftConfig::new(vec![length])).unwrap();
        let ir =
            RecursiveFftIr::build(&plan, Direction::Forward, upstream_scheduler_device()).unwrap();
        let four_step = ir
            .four_step_plan
            .as_ref()
            .expect("two-upload NVIDIA schedule should select Four-step fusion");
        assert_eq!(four_step.logical_len, length);
        assert_eq!(
            four_step
                .uploads
                .iter()
                .map(|upload| upload.axis_upload_id)
                .collect::<Vec<_>>(),
            vec![1, 0]
        );
        assert_eq!(four_step.uploads[0].fft_len, 1024);
        assert_eq!(four_step.uploads[0].stage_start_size, 1024);
        assert_eq!(
            four_step.uploads[0].twiddle,
            FourStepTwiddlePlacement::Write
        );
        assert_eq!(four_step.uploads[1].fft_len, 1024);
        assert_eq!(four_step.uploads[1].stage_start_size, 1);
        assert_eq!(four_step.uploads[1].twiddle, FourStepTwiddlePlacement::None);
        assert_eq!(
            four_step
                .uploads
                .iter()
                .map(|upload| upload.axis_block.unwrap().grouped_batch)
                .collect::<Vec<_>>(),
            vec![4, 4]
        );

        let uploads = ir.four_step_stockham_upload_kernels().unwrap().unwrap();
        assert_eq!(uploads.len(), 2);
        assert_eq!(
            uploads
                .iter()
                .map(|kernel| [kernel.workgroup_size.x, kernel.workgroup_size.y])
                .collect::<Vec<_>>(),
            vec![[4, 128], [4, 128]]
        );
        assert_eq!(
            uploads
                .iter()
                .map(|kernel| kernel.dispatch.x)
                .collect::<Vec<_>>(),
            vec![256, 256]
        );
        assert!(uploads[0].name.ends_with("four_step_upload_1"));
        assert!(uploads[1].name.ends_with("four_step_upload_0"));
        assert!(matches!(
            uploads[0].io_mapping,
            StockhamIoMapping::FourStepRight(_)
        ));
        assert!(matches!(
            uploads[1].io_mapping,
            StockhamIoMapping::FourStepLeft(_)
        ));
    }

    #[test]
    fn three_upload_scheduler_materializes_upstream_four_step_order_and_mappings() {
        let length = 8_388_608usize;
        let plan = FftPlan::build(FftConfig::new(vec![length])).unwrap();
        let ir =
            RecursiveFftIr::build(&plan, Direction::Forward, upstream_scheduler_device()).unwrap();
        let schedule = ir.stockham_upload_schedule.as_ref().unwrap();
        assert_eq!(schedule.upload_count, 3);
        assert_eq!(schedule.axis_split, vec![256, 128, 256]);

        let four_step = ir
            .four_step_plan
            .as_ref()
            .expect("three-upload NVIDIA schedule should select Four-step fusion");
        assert_eq!(
            four_step
                .uploads
                .iter()
                .map(|upload| upload.axis_upload_id)
                .collect::<Vec<_>>(),
            vec![2, 1, 0]
        );
        assert_eq!(
            four_step
                .uploads
                .iter()
                .map(|upload| upload.fft_len)
                .collect::<Vec<_>>(),
            vec![256, 128, 256]
        );
        assert_eq!(
            four_step
                .uploads
                .iter()
                .map(|upload| upload.stage_start_size)
                .collect::<Vec<_>>(),
            vec![32_768, 256, 1]
        );
        assert_eq!(
            four_step
                .uploads
                .iter()
                .map(|upload| upload.twiddle)
                .collect::<Vec<_>>(),
            vec![
                FourStepTwiddlePlacement::Write,
                FourStepTwiddlePlacement::Write,
                FourStepTwiddlePlacement::None,
            ]
        );
        assert_eq!(
            four_step
                .uploads
                .iter()
                .map(|upload| upload.transform_count)
                .collect::<Vec<_>>(),
            vec![32_768, 65_536, 32_768]
        );
        assert_eq!(
            four_step
                .uploads
                .iter()
                .map(|upload| upload.axis_block.unwrap().grouped_batch)
                .collect::<Vec<_>>(),
            vec![16, 16, 16]
        );

        let uploads = ir.four_step_stockham_upload_kernels().unwrap().unwrap();
        assert_eq!(uploads.len(), 3);
        assert_eq!(
            uploads
                .iter()
                .map(|kernel| [kernel.workgroup_size.x, kernel.workgroup_size.y])
                .collect::<Vec<_>>(),
            vec![[16, 32], [16, 16], [16, 32]]
        );
        assert_eq!(uploads[0].dispatch.x, 2048);
        assert_eq!(uploads[1].dispatch.x, 4096);
        assert_eq!(uploads[2].dispatch.x, 2048);
        assert!(uploads[0].name.ends_with("four_step_upload_2"));
        assert!(uploads[1].name.ends_with("four_step_upload_1"));
        assert!(uploads[2].name.ends_with("four_step_upload_0"));
        assert!(matches!(
            uploads[0].io_mapping,
            StockhamIoMapping::FourStepThreeUpload2(_)
        ));
        assert!(matches!(
            uploads[1].io_mapping,
            StockhamIoMapping::FourStepThreeUpload1(_)
        ));
        assert!(matches!(
            uploads[2].io_mapping,
            StockhamIoMapping::FourStepThreeUpload0(_)
        ));
        ir.validate().unwrap();
    }

    #[test]
    fn extracted_strided_pow2_child_materializes_two_upload_four_step() {
        let profile = DeviceProfile {
            shared_memory_bytes: 48 * 1024,
            shared_memory_pow2_bytes: 32 * 1024,
            max_threads_per_block: 1024,
            max_workgroup_size: [1024, 1024, 64],
            ..DeviceProfile::generic(Backend::Vulkan, GpuVendor::Nvidia)
        };
        let plan = FftPlan::build_c2c_child_for_device(
            FftConfig::new(vec![8192]),
            profile,
            C2cDeviceAxisClass::Strided,
        )
        .unwrap();
        assert_eq!(
            plan.c2c_device_axis_class_override,
            Some(C2cDeviceAxisClass::Strided)
        );

        let ir = RecursiveFftIr::build(&plan, Direction::Forward, profile).unwrap();
        let schedule = ir
            .stockham_upload_schedule
            .as_ref()
            .expect("strided N8192 child should retain its two-upload scheduler metadata");
        assert_eq!(schedule.upload_count, 2);
        assert_eq!(schedule.axis_split, vec![128, 64]);
        let four_step = ir
            .four_step_plan
            .as_ref()
            .expect("strided two-upload child should materialize Four-step IR");
        assert_eq!(four_step.uploads.len(), 2);
        let kernels = ir.four_step_stockham_upload_kernels().unwrap().unwrap();
        assert_eq!(
            kernels
                .iter()
                .map(|kernel| kernel.sequence_len)
                .collect::<Vec<_>>(),
            vec![64, 128]
        );
        let shaders = crate::backend::vulkan::VulkanGlslBackend
            .lower_recursive_fft(&ir)
            .unwrap();
        assert_eq!(shaders.len(), 2);
        for shader in shaders {
            assert_eq!(shader.compile_spirv().unwrap().words[0], 0x0723_0203);
        }
        ir.validate().unwrap();
    }

    #[test]
    fn register_boost_leaf_keeps_16384_as_one_scheduler_upload() {
        let length = 16_384usize;
        let device = DeviceProfile {
            shared_memory_bytes: 32 * 1024,
            shared_memory_pow2_bytes: 32 * 1024,
            max_threads_per_block: 1024,
            ..DeviceProfile::generic(Backend::Vulkan, GpuVendor::Nvidia)
        };
        let plan = FftPlan::build(FftConfig::new(vec![length])).unwrap();
        let ir = RecursiveFftIr::build(&plan, Direction::Forward, device).unwrap();
        let schedule = ir.stockham_upload_schedule.as_ref().unwrap();
        assert_eq!(schedule.upload_count, 1);
        assert_eq!(schedule.axis_split, vec![length]);
        assert_eq!(schedule.register_boost, 4);
        let RecursiveFftNodeIr::Stockham(kernel) = &ir.root else {
            panic!("registerBoost=4 schedule should remain a single Stockham leaf");
        };
        assert_eq!(
            kernel.execution_layout,
            crate::StockhamExecutionLayout::RegisterBoostSingleShared
        );
        assert_eq!(kernel.shared_memory.elements_per_buffer, 4096);
        assert_eq!(kernel.required_shared_memory_bytes().unwrap(), 32 * 1024);
        ir.validate().unwrap();
    }

    #[test]
    fn multi_upload_stockham_tree_prefers_balanced_factor_products() {
        let length = 256usize;
        let plan = FftPlan::build(FftConfig::new(vec![length])).unwrap();
        let ir =
            RecursiveFftIr::build(&plan, Direction::Forward, tiny_shared_memory_device()).unwrap();
        let RecursiveFftNodeIr::CooleyTukey(root) = &ir.root else {
            panic!("256-point tiny-memory plan should split into multiple uploads");
        };
        // The leaf limit is four elements, so 2^8 becomes four 4-point leaves.
        // A linear factor chain would split 4 x 64; the balanced scheduler chooses
        // 16 x 16, matching the square-root tendency of upstream Four-step planning.
        assert_eq!((root.left_len, root.right_len), (16, 16));
        assert_stockham_leaves_fit(&ir.root, 4);

        let input = sample(length);
        let actual = execute_recursive_fft_ir(&ir, &input).unwrap();
        let expected = dft(&input, Direction::Forward, false);
        assert!(max_error(&actual, &expected) < 2.0e-8 * length as f64);
    }

    #[test]
    fn large_non_power_of_two_smooth_axis_uses_upload_scheduler_and_four_step() {
        let profile = register_leaf_device();
        for length in [3072usize, 3840, 4095] {
            let plan = FftPlan::build(FftConfig::new(vec![length])).unwrap();
            let ir = RecursiveFftIr::build(&plan, Direction::Forward, profile).unwrap();
            let schedule = ir
                .stockham_upload_schedule
                .as_ref()
                .expect("smooth NVIDIA path should retain one-upload scheduler metadata");
            assert_eq!(schedule.upload_count, 1);
            assert_eq!(schedule.axis_split, vec![length]);
            let RecursiveFftNodeIr::Stockham(kernel) = &ir.root else {
                panic!("N={length} should fit one non-power-of-two register Stockham leaf");
            };
            assert_eq!(kernel.sequence_len, length);
            assert!(kernel.scheduler_hint.is_some());
            assert_eq!(
                kernel.execution_layout,
                crate::StockhamExecutionLayout::RegisterSingleShared
            );
            assert!(kernel.required_shared_memory_bytes().unwrap() <= profile.shared_memory_bytes);

            let input = sample(length);
            let actual = execute_recursive_fft_ir(&ir, &input).unwrap();
            let expected = crate::reference::fft(&input, Direction::Forward, false).unwrap();
            assert!(
                max_error(&actual, &expected) <= 3.0e-8 * length.ilog2().max(1) as f64,
                "single-leaf register Stockham mismatch for N={length}"
            );
        }

        let length = 6144usize;
        for direction in [Direction::Forward, Direction::Inverse] {
            let plan = FftPlan::build(
                FftConfig::new(vec![length])
                    .with_inverse_normalization(direction == Direction::Inverse),
            )
            .unwrap();
            let ir = RecursiveFftIr::build(&plan, direction, profile).unwrap();
            let schedule = ir.stockham_upload_schedule.as_ref().unwrap();
            assert_eq!(schedule.upload_count, 2);
            assert_eq!(schedule.axis_split, vec![96, 64]);
            assert!(ir.four_step_plan.is_some());
            let mut leaves = Vec::new();
            collect_stockham_leaf_lengths(&ir.root, &mut leaves).unwrap();
            assert_eq!(leaves, vec![96, 64]);
            let uploads = ir.four_step_stockham_upload_kernels().unwrap().unwrap();
            assert_eq!(uploads.len(), 2);
            assert_eq!(uploads[0].sequence_len, 64);
            assert_eq!(uploads[1].sequence_len, 96);

            let input = sample(length);
            let actual = execute_recursive_fft_ir(&ir, &input).unwrap();
            let expected =
                crate::reference::fft(&input, direction, direction == Direction::Inverse).unwrap();
            assert!(
                max_error(&actual, &expected) <= 4.0e-8 * length.ilog2().max(1) as f64,
                "two-upload smooth Stockham mismatch for N={length} direction {direction:?}"
            );
        }
    }

    #[test]
    fn smooth_register_leaf_fit_is_policy_driven_across_gpu_backends() {
        let length = 3840usize;
        let plan = FftPlan::build(FftConfig::new(vec![length])).unwrap();
        let input = sample(length);
        let expected = crate::reference::fft(&input, Direction::Forward, false).unwrap();
        for (backend, vendor) in [
            (Backend::Vulkan, GpuVendor::Amd),
            (Backend::Vulkan, GpuVendor::Intel),
            (Backend::OpenCl, GpuVendor::Nvidia),
            (Backend::Cuda, GpuVendor::Nvidia),
            (Backend::Hip, GpuVendor::Amd),
            (Backend::LevelZero, GpuVendor::Intel),
            (Backend::Metal, GpuVendor::Apple),
        ] {
            let profile = DeviceProfile {
                shared_memory_bytes: 32 * 1024,
                shared_memory_pow2_bytes: 32 * 1024,
                max_threads_per_block: 1024,
                ..DeviceProfile::generic(backend, vendor)
            };
            let ir = RecursiveFftIr::build(&plan, Direction::Forward, profile).unwrap();
            let schedule = ir
                .stockham_upload_schedule
                .as_ref()
                .expect("known GPU policy should retain smooth upload metadata");
            assert_eq!(schedule.upload_count, 1, "{backend:?}/{vendor:?}");
            assert_eq!(schedule.axis_split, vec![length], "{backend:?}/{vendor:?}");
            let RecursiveFftNodeIr::Stockham(kernel) = &ir.root else {
                panic!("{backend:?}/{vendor:?} should keep N={length} as one register leaf");
            };
            assert!(kernel.scheduler_hint.is_some(), "{backend:?}/{vendor:?}");
            assert!(
                kernel.required_shared_memory_bytes().unwrap() <= profile.shared_memory_bytes,
                "{backend:?}/{vendor:?}"
            );
            let actual = execute_recursive_fft_ir(&ir, &input).unwrap();
            assert!(
                max_error(&actual, &expected) <= 3.0e-8 * length.ilog2().max(1) as f64,
                "cross-backend scheduler leaf mismatch for {backend:?}/{vendor:?}"
            );
        }
    }

    #[test]
    fn policy_driven_power_of_two_four_step_materializes_across_backends() {
        for (backend, vendor) in [
            (Backend::Vulkan, GpuVendor::Nvidia),
            (Backend::Vulkan, GpuVendor::Amd),
            (Backend::Vulkan, GpuVendor::Intel),
            (Backend::OpenCl, GpuVendor::Nvidia),
            (Backend::OpenCl, GpuVendor::Amd),
            (Backend::OpenCl, GpuVendor::Intel),
            (Backend::Cuda, GpuVendor::Nvidia),
            (Backend::Hip, GpuVendor::Amd),
            (Backend::LevelZero, GpuVendor::Intel),
            (Backend::Metal, GpuVendor::Apple),
        ] {
            let profile = DeviceProfile {
                shared_memory_bytes: 48 * 1024,
                shared_memory_pow2_bytes: 32 * 1024,
                max_threads_per_block: 1024,
                ..DeviceProfile::generic(backend, vendor)
            };
            let mut found_two = None;
            let mut found_three = None;
            for exponent in 16u32..=26 {
                let length = 1usize << exponent;
                let schedule = crate::scheduler::plan_gpu_power_of_two_stockham_uploads(
                    length,
                    Precision::F32,
                    profile,
                )
                .unwrap();
                if schedule.upload_count == 2 && found_two.is_none() {
                    found_two = Some(length);
                }
                if schedule.upload_count == 3 && found_three.is_none() {
                    found_three = Some(length);
                }
                if found_two.is_some() && found_three.is_some() {
                    break;
                }
            }
            for (upload_count, length) in [
                (
                    2usize,
                    found_two.expect("known GPU policy should expose a two-upload range"),
                ),
                (
                    3usize,
                    found_three.expect("known GPU policy should expose a three-upload range"),
                ),
            ] {
                let plan = FftPlan::build(FftConfig::new(vec![length])).unwrap();
                let ir = RecursiveFftIr::build(&plan, Direction::Forward, profile).unwrap();
                let schedule = ir.stockham_upload_schedule.as_ref().unwrap();
                assert_eq!(
                    schedule.upload_count, upload_count,
                    "{backend:?}/{vendor:?}"
                );
                let four_step = ir
                    .four_step_plan
                    .as_ref()
                    .expect("two/three scheduler uploads should materialize Four-step IR");
                assert_eq!(
                    four_step.uploads.len(),
                    upload_count,
                    "{backend:?}/{vendor:?}"
                );
                assert_eq!(four_step.logical_len, length);
                let kernels = ir
                    .four_step_stockham_upload_kernels()
                    .unwrap()
                    .expect("Four-step plan should materialize physical upload kernels");
                assert_eq!(kernels.len(), upload_count);
                assert_eq!(
                    kernels
                        .iter()
                        .map(|kernel| kernel.sequence_len)
                        .product::<usize>(),
                    length,
                    "{backend:?}/{vendor:?}"
                );
            }
        }
    }

    #[test]
    fn policy_driven_smooth_non_power_of_two_four_step_materializes_across_backends() {
        let candidates = [
            6_144usize, 12_288, 24_576, 49_152, 98_304, 196_608, 393_216, 786_432, 1_572_864,
            3_145_728, 4_718_592, 6_291_456, 12_582_912,
        ];
        for (backend, vendor) in [
            (Backend::Vulkan, GpuVendor::Nvidia),
            (Backend::Vulkan, GpuVendor::Amd),
            (Backend::Vulkan, GpuVendor::Intel),
            (Backend::OpenCl, GpuVendor::Nvidia),
            (Backend::OpenCl, GpuVendor::Amd),
            (Backend::OpenCl, GpuVendor::Intel),
            (Backend::Cuda, GpuVendor::Nvidia),
            (Backend::Hip, GpuVendor::Amd),
            (Backend::LevelZero, GpuVendor::Intel),
            (Backend::Metal, GpuVendor::Apple),
        ] {
            let profile = DeviceProfile {
                shared_memory_bytes: 48 * 1024,
                shared_memory_pow2_bytes: 32 * 1024,
                max_threads_per_block: 1024,
                ..DeviceProfile::generic(backend, vendor)
            };
            let mut found = [None, None];
            for length in candidates {
                let Ok(schedule) = plan_gpu_smooth_stockham_uploads_for_batches(
                    length,
                    1,
                    Precision::F32,
                    profile,
                ) else {
                    continue;
                };
                if schedule.upload_count == 2 && found[0].is_none() {
                    found[0] = Some(length);
                }
                if schedule.upload_count == 3 && found[1].is_none() {
                    found[1] = Some(length);
                }
            }
            for (slot, upload_count) in [2usize, 3].into_iter().enumerate() {
                let length = found[slot].unwrap_or_else(|| {
                    panic!(
                        "{backend:?}/{vendor:?} should expose a smooth non-power-of-two {upload_count}-upload range"
                    )
                });
                let plan = FftPlan::build(FftConfig::new(vec![length])).unwrap();
                let ir = RecursiveFftIr::build(&plan, Direction::Forward, profile).unwrap();
                let schedule = ir.stockham_upload_schedule.as_ref().unwrap();
                assert_eq!(
                    schedule.upload_count, upload_count,
                    "{backend:?}/{vendor:?}"
                );
                assert_eq!(schedule.axis_split.iter().product::<usize>(), length);
                let four_step = ir
                    .four_step_plan
                    .as_ref()
                    .expect("smooth two/three upload schedule should materialize Four-step IR");
                assert_eq!(four_step.uploads.len(), upload_count);
                let kernels = ir
                    .four_step_stockham_upload_kernels()
                    .unwrap()
                    .expect("smooth Four-step plan should materialize upload kernels");
                assert_eq!(kernels.len(), upload_count);
                assert_eq!(
                    kernels
                        .iter()
                        .map(|kernel| kernel.sequence_len)
                        .product::<usize>(),
                    length
                );
            }
        }
    }

    #[test]
    fn large_non_power_of_two_three_upload_materializes_four_step() {
        let length = 1_572_864usize;
        let plan = FftPlan::build(FftConfig::new(vec![length])).unwrap();
        let ir = RecursiveFftIr::build(&plan, Direction::Forward, register_leaf_device()).unwrap();
        let schedule = ir.stockham_upload_schedule.as_ref().unwrap();
        assert_eq!(schedule.upload_count, 3);
        assert_eq!(schedule.axis_split, vec![128, 96, 128]);
        let four_step = ir.four_step_plan.as_ref().unwrap();
        assert_eq!(
            four_step
                .uploads
                .iter()
                .map(|upload| upload.axis_upload_id)
                .collect::<Vec<_>>(),
            vec![2, 1, 0]
        );
        let kernels = ir.four_step_stockham_upload_kernels().unwrap().unwrap();
        assert_eq!(
            kernels
                .iter()
                .map(|kernel| kernel.sequence_len)
                .collect::<Vec<_>>(),
            vec![128, 96, 128]
        );
        assert!(kernels.iter().all(|kernel| kernel.scheduler_hint.is_some()));
    }

    #[test]
    fn higher_axis_three_upload_retags_all_four_step_components() {
        let profile = register_leaf_device();
        let length = 1_572_864usize;
        let plan = FftPlan::build(
            FftConfig::new(vec![length])
                .with_batch_count(8)
                .with_grouped_batch(0, 3)
                .unwrap(),
        )
        .unwrap();
        let ir = RecursiveFftIr::build(&plan, Direction::Forward, profile)
            .unwrap()
            .with_other_axis_single_upload_block_with_grouped_batch(8, Some(3), Some(3), profile)
            .unwrap();
        let schedule = ir.stockham_upload_schedule.as_ref().unwrap();
        assert_eq!(schedule.upload_count, 3);
        assert_eq!(schedule.axis_split, vec![128, 96, 128]);
        let four_step = ir.four_step_plan.as_ref().unwrap();
        assert_eq!(
            four_step
                .uploads
                .iter()
                .map(|upload| upload.axis_block.unwrap().grouped_batch)
                .collect::<Vec<_>>(),
            vec![3, 3, 3]
        );
        assert!(four_step.uploads.iter().all(|upload| {
            upload.axis_block.is_some_and(|block| {
                block.transforms_on_x && !block.axis_swapped && block.local_size_x == 3
            })
        }));

        let kernels = ir.four_step_stockham_upload_kernels().unwrap().unwrap();
        assert_eq!(
            kernels
                .iter()
                .map(|kernel| [kernel.workgroup_size.x, kernel.workgroup_size.y])
                .collect::<Vec<_>>(),
            vec![[3, 16], [3, 8], [3, 16]]
        );
        assert_eq!(
            kernels
                .iter()
                .map(|kernel| kernel.dispatch.x)
                .collect::<Vec<_>>(),
            vec![32_768, 43_691, 32_768]
        );
        let shaders = crate::backend::vulkan::VulkanGlslBackend
            .lower_recursive_fft(&ir)
            .unwrap();
        assert_eq!(shaders.len(), 3);
        for shader in shaders {
            assert_eq!(shader.compile_spirv().unwrap().words[0], 0x0723_0203);
        }
        ir.validate().unwrap();
    }

    #[test]
    fn amd_f16_large_higher_axis_preserves_scheduler_precision_in_four_step_blocks() {
        let profile = DeviceProfile {
            shared_memory_bytes: 48 * 1024,
            shared_memory_pow2_bytes: 32 * 1024,
            max_threads_per_block: 1024,
            max_workgroup_size: [1024, 1024, 64],
            coalesced_memory_bytes: 64,
            supports_f64: true,
            ..DeviceProfile::generic(Backend::Vulkan, GpuVendor::Amd)
        };
        let plan = FftPlan::build(
            FftConfig::new(vec![524_288])
                .with_batch_count(64)
                .with_precision(Precision::F16StorageF32Compute),
        )
        .unwrap();
        let ir = RecursiveFftIr::build(&plan, Direction::Forward, profile)
            .unwrap()
            .with_other_axis_single_upload_block(64, profile)
            .unwrap();

        assert_eq!(ir.scalar, ScalarType::F32);
        assert_eq!(ir.external_storage_scalar(), ScalarType::F16);
        let schedule = ir.stockham_upload_schedule.as_ref().unwrap();
        assert_eq!(schedule.axis_split, vec![64, 128, 64]);
        let four_step = ir.four_step_plan.as_ref().unwrap();
        for (axis_upload_id, expected_grouped, expected_local) in [
            (0usize, 32usize, [32usize, 8usize]),
            (1, 32, [32, 16]),
            (2, 64, [64, 8]),
        ] {
            let upload = four_step
                .uploads
                .iter()
                .find(|upload| upload.axis_upload_id == axis_upload_id)
                .unwrap();
            let block = upload.axis_block.unwrap();
            assert_eq!(block.grouped_batch, expected_grouped);
            assert_eq!([block.local_size_x, block.local_size_y], expected_local);
            assert!(block.transforms_on_x);
            assert!(!block.axis_swapped);
        }
        ir.validate().unwrap();
    }

    #[test]
    fn mixed_storage_recursive_cooley_root_converts_only_external_boundaries() {
        let mut profile = DeviceProfile {
            shared_memory_bytes: 64,
            shared_memory_pow2_bytes: 64,
            ..DeviceProfile::generic(Backend::Vulkan, GpuVendor::Other(0xD00D))
        };
        profile.supports_f64 = true;

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
            let plan = FftPlan::build(FftConfig::new(vec![256]).with_precision(precision)).unwrap();
            let ir = RecursiveFftIr::build(&plan, Direction::Forward, profile).unwrap();
            assert!(ir.four_step_plan.is_none());
            assert_eq!(ir.scalar, compute);
            assert_eq!(ir.external_storage_scalar(), storage);
            let RecursiveFftNodeIr::CooleyTukey(root) = &ir.root else {
                panic!("portable 64-byte profile should force a Cooley-Tukey root");
            };
            assert_eq!(root.pack_right.scalar, compute);
            assert_eq!(root.pack_right.input_storage_scalar, storage);
            assert_eq!(root.pack_right.output_storage_scalar, compute);
            assert_eq!(root.twiddle_transpose.input_storage_scalar, compute);
            assert_eq!(root.twiddle_transpose.output_storage_scalar, compute);
            assert_eq!(root.scatter_output.input_storage_scalar, compute);
            assert_eq!(root.scatter_output.output_storage_scalar, storage);

            let program = crate::ProgramIr::recursive_fft(&ir).unwrap();
            assert_eq!(program.scalar, compute);
            assert_eq!(program.input_resource().unwrap().scalar, storage);
            assert_eq!(program.output_resource().unwrap().scalar, storage);
            assert!(program.resources.iter().all(|resource| {
                matches!(
                    resource.kind,
                    crate::ProgramResourceKind::Input | crate::ProgramResourceKind::Output
                ) || resource.scalar == compute
            }));

            let shaders = crate::backend::vulkan::VulkanGlslBackend
                .lower_recursive_fft(&ir)
                .unwrap();
            assert_eq!(shaders.len(), program.passes.len());
            for shader in &shaders {
                shader.compile_spirv().unwrap();
            }
            let first = &shaders[0].glsl;
            let last = &shaders[shaders.len() - 1].glsl;
            match precision {
                Precision::F16StorageF32Compute => {
                    assert!(first.contains("readonly buffer VkFftInput { uint data[]; }"));
                    assert!(first.contains("unpackHalf2x16(vkfft_input.data[base + source])"));
                    assert!(last.contains("buffer VkFftOutput { uint data[]; }"));
                    assert!(last.contains("packHalf2x16(vkfft_input.data[base + source])"));
                }
                Precision::F64ComputeF32Storage => {
                    assert!(first.contains("readonly buffer VkFftInput { vec2 data[]; }"));
                    assert!(first.contains("dvec2((vkfft_input.data[base + source]).x"));
                    assert!(last.contains("buffer VkFftOutput { vec2 data[]; }"));
                    assert!(last.contains("= vec2((vkfft_input.data[base + source]).x"));
                }
                _ => unreachable!(),
            }

            let one_dim = crate::OneDimFftIr::Recursive(Box::new(ir.clone()));
            let transform = crate::TransformIr::Complex1d(one_dim);
            let native = crate::backend::NativeSourceBackend::new(Backend::Cuda)
                .lower_transform(&transform)
                .unwrap();
            assert_eq!(native.program.input_resource().unwrap().scalar, storage);
            assert_eq!(native.program.output_resource().unwrap().scalar, storage);
            assert!(native.program.resources.iter().all(|resource| {
                matches!(
                    resource.kind,
                    crate::ProgramResourceKind::Input | crate::ProgramResourceKind::Output
                ) || resource.scalar == compute
            }));
        }
    }

    #[test]
    fn zero_padded_mixed_storage_owns_only_the_directional_root_boundary() {
        let mut profile = DeviceProfile {
            shared_memory_bytes: 64,
            shared_memory_pow2_bytes: 64,
            ..DeviceProfile::generic(Backend::Vulkan, GpuVendor::Other(0xD00D))
        };
        profile.supports_f64 = true;

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
                let config = FftConfig::new(vec![256])
                    .with_precision(precision)
                    .with_zero_padding(0, 128, 256)
                    .unwrap();
                let plan = FftPlan::build(config).unwrap();
                let ir = RecursiveFftIr::build(&plan, direction, profile).unwrap();
                assert_eq!(ir.scalar, compute);
                assert_eq!(ir.external_storage_scalar(), storage);
                let zero = ir.zero_pad_pass.as_ref().unwrap();
                match direction {
                    Direction::Forward => {
                        assert_eq!(zero.input_storage_scalar, storage);
                        assert_eq!(zero.output_storage_scalar, compute);
                    }
                    Direction::Inverse => {
                        assert_eq!(zero.input_storage_scalar, compute);
                        assert_eq!(zero.output_storage_scalar, storage);
                    }
                }
                let RecursiveFftNodeIr::CooleyTukey(root) = &ir.root else {
                    panic!("portable zero-padded N=256 should force a Cooley-Tukey root");
                };
                match direction {
                    Direction::Forward => {
                        assert_eq!(root.pack_right.input_storage_scalar, compute);
                        assert_eq!(root.scatter_output.output_storage_scalar, storage);
                    }
                    Direction::Inverse => {
                        assert_eq!(root.pack_right.input_storage_scalar, storage);
                        assert_eq!(root.scatter_output.output_storage_scalar, compute);
                    }
                }
            }
        }
    }

    #[test]
    fn zero_padded_mixed_storage_rader_roots_own_only_the_directional_boundary() {
        let mut profile = DeviceProfile::generic(Backend::Vulkan, GpuVendor::Nvidia);
        profile.shared_memory_bytes = 128 * 1024;
        profile.shared_memory_pow2_bytes = 128 * 1024;
        profile.supports_f64 = true;

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
                for length in [47usize, 19usize] {
                    let config = FftConfig::new(vec![length])
                        .with_precision(precision)
                        .with_zero_padding(0, length / 2, length)
                        .unwrap();
                    let plan = FftPlan::build(config).unwrap();
                    let ir = RecursiveFftIr::build(&plan, direction, profile).unwrap();
                    let zero = ir.zero_pad_pass.as_ref().unwrap();
                    let expected_zero = match direction {
                        Direction::Forward => (storage, compute),
                        Direction::Inverse => (compute, storage),
                    };
                    assert_eq!(
                        (zero.input_storage_scalar, zero.output_storage_scalar),
                        expected_zero
                    );

                    match (&ir.root, length) {
                        (RecursiveFftNodeIr::DirectRader(rader), 47) => {
                            let expected_root = match direction {
                                Direction::Forward => (compute, storage),
                                Direction::Inverse => (storage, compute),
                            };
                            assert_eq!(
                                (rader.input_storage_scalar, rader.output_storage_scalar),
                                expected_root
                            );
                        }
                        (RecursiveFftNodeIr::FftRader(rader), 19) => {
                            let expected_root = match direction {
                                Direction::Forward => (compute, storage),
                                Direction::Inverse => (storage, compute),
                            };
                            assert_eq!(
                                (rader.input_storage_scalar, rader.output_storage_scalar),
                                expected_root
                            );
                            let fused = rader
                                .fused_inverse_rader_kernel()
                                .unwrap()
                                .expect("p19 should keep fused inverse Rader scatter");
                            assert_eq!(fused.bindings[1].scalar, expected_root.1);
                            let auxiliary = fused
                                .bindings
                                .iter()
                                .find(|binding| binding.role == crate::BufferRole::Auxiliary)
                                .expect("fused Rader scatter must keep original-prime auxiliary");
                            assert_eq!(auxiliary.scalar, expected_root.0);
                        }
                        _ => panic!("unexpected zero-padded Rader root for N={length}"),
                    }
                }
            }
        }
    }

    #[test]
    fn zero_padded_recursive_p4001_rader_tracks_input_output_and_auxiliary_storage_independently() {
        let mut profile = DeviceProfile::generic(Backend::Vulkan, GpuVendor::Nvidia);
        profile.shared_memory_bytes = 16 * 128;
        profile.shared_memory_pow2_bytes = profile.shared_memory_bytes;
        profile.supports_f64 = true;
        let length = 4001usize;

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
                let config = FftConfig::new(vec![length])
                    .with_precision(precision)
                    .with_zero_padding(0, length / 2, length)
                    .unwrap();
                let plan = FftPlan::build(config).unwrap();
                let one_dim = crate::OneDimFftIr::build(&plan, direction, profile).unwrap();
                let crate::OneDimFftIr::Recursive(ir) = &one_dim else {
                    panic!("p4001 should remain recursive FFT-Rader");
                };
                let RecursiveFftNodeIr::FftRader(rader) = &ir.root else {
                    panic!("p4001 should keep FFT-Rader root");
                };
                assert_eq!(
                    rader.input_strategy,
                    crate::RaderFftInputStrategy::GeneratorOrderRecursive
                );
                let expected_root = match direction {
                    Direction::Forward => (compute, storage),
                    Direction::Inverse => (storage, compute),
                };
                assert_eq!(
                    (rader.input_storage_scalar, rader.output_storage_scalar),
                    expected_root
                );
                let RecursiveFftNodeIr::CooleyTukey(forward) =
                    &rader.forward_recursive().unwrap().root
                else {
                    panic!("p4001 generator input should be recursive");
                };
                let RecursiveFftNodeIr::CooleyTukey(inverse) =
                    &rader.inverse_recursive().unwrap().root
                else {
                    panic!("p4001 fused scatter should be recursive");
                };
                assert_eq!(forward.pack_right.input_storage_scalar, expected_root.0);
                assert_eq!(
                    inverse.scatter_output.output_storage_scalar,
                    expected_root.1
                );
                assert_eq!(
                    inverse.scatter_output.auxiliary_storage_scalar,
                    expected_root.0
                );
                let zero = ir.zero_pad_pass.as_ref().unwrap();
                let expected_zero = match direction {
                    Direction::Forward => (storage, compute),
                    Direction::Inverse => (compute, storage),
                };
                assert_eq!(
                    (zero.input_storage_scalar, zero.output_storage_scalar),
                    expected_zero
                );

                let program = crate::ProgramIr::one_dim_fft(&one_dim).unwrap();
                assert_eq!(program.input_resource().unwrap().scalar, storage);
                assert_eq!(program.output_resource().unwrap().scalar, storage);
                assert!(program.resources.iter().all(|resource| {
                    matches!(
                        resource.kind,
                        crate::ProgramResourceKind::Input | crate::ProgramResourceKind::Output
                    ) || resource.scalar == compute
                }));
                let shaders = crate::backend::vulkan::VulkanGlslBackend
                    .lower_one_dim_fft(&one_dim)
                    .unwrap();
                assert_eq!(shaders.len(), program.passes.len());
                for shader in shaders {
                    assert_eq!(shader.compile_spirv().unwrap().words[0], 0x0723_0203);
                }
            }
        }
    }

    #[test]
    fn four_step_mixed_storage_keeps_only_first_and_last_upload_external() {
        let mut profile = register_leaf_device();
        profile.supports_f64 = true;

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
            let length = 6144usize;
            let plan =
                FftPlan::build(FftConfig::new(vec![length]).with_precision(precision)).unwrap();
            let ir = RecursiveFftIr::build(&plan, Direction::Forward, profile).unwrap();
            assert_eq!(ir.scalar, compute);
            assert_eq!(ir.external_storage_scalar(), storage);
            assert_eq!(
                ir.stockham_upload_schedule.as_ref().unwrap().upload_count,
                2
            );
            assert!(ir.four_step_plan.is_some());
            let kernels = ir.four_step_stockham_upload_kernels().unwrap().unwrap();
            assert_eq!(kernels.len(), 2);
            assert_eq!(kernels[0].bindings[0].scalar, storage);
            assert_eq!(kernels[0].bindings[1].scalar, compute);
            assert_eq!(kernels[1].bindings[0].scalar, compute);
            assert_eq!(kernels[1].bindings[1].scalar, storage);
            assert!(
                kernels[0]
                    .bindings
                    .iter()
                    .skip(1)
                    .all(|binding| binding.scalar == compute)
            );
            assert!(
                kernels[1]
                    .bindings
                    .iter()
                    .skip(2)
                    .all(|binding| binding.scalar == compute)
            );

            let program = crate::ProgramIr::recursive_fft(&ir).unwrap();
            assert_eq!(program.input_resource().unwrap().scalar, storage);
            assert_eq!(program.output_resource().unwrap().scalar, storage);
            assert!(program.resources.iter().all(|resource| {
                matches!(
                    resource.kind,
                    crate::ProgramResourceKind::Input | crate::ProgramResourceKind::Output
                ) || resource.scalar == compute
            }));
            let shaders = crate::backend::vulkan::VulkanGlslBackend
                .lower_recursive_fft(&ir)
                .unwrap();
            assert_eq!(shaders.len(), 2);
            for shader in &shaders {
                shader.compile_spirv().unwrap();
            }
        }

        let length = 1_572_864usize;
        let plan = FftPlan::build(
            FftConfig::new(vec![length]).with_precision(Precision::F16StorageF32Compute),
        )
        .unwrap();
        let ir = RecursiveFftIr::build(&plan, Direction::Forward, profile).unwrap();
        assert_eq!(ir.external_storage_scalar(), ScalarType::F16);
        assert_eq!(
            ir.stockham_upload_schedule.as_ref().unwrap().upload_count,
            3
        );
        let kernels = ir.four_step_stockham_upload_kernels().unwrap().unwrap();
        assert_eq!(kernels.len(), 3);
        assert_eq!(kernels[0].bindings[0].scalar, ScalarType::F16);
        assert_eq!(kernels[0].bindings[1].scalar, ScalarType::F32);
        assert_eq!(kernels[1].bindings[0].scalar, ScalarType::F32);
        assert_eq!(kernels[1].bindings[1].scalar, ScalarType::F32);
        assert_eq!(kernels[2].bindings[0].scalar, ScalarType::F32);
        assert_eq!(kernels[2].bindings[1].scalar, ScalarType::F16);
        let program = crate::ProgramIr::recursive_fft(&ir).unwrap();
        let scratch_count = program
            .resources
            .iter()
            .filter(|resource| resource.kind == crate::ProgramResourceKind::Scratch)
            .count();
        assert_eq!(scratch_count, 2);
        assert!(program.resources.iter().all(|resource| {
            matches!(
                resource.kind,
                crate::ProgramResourceKind::Input | crate::ProgramResourceKind::Output
            ) || resource.scalar == ScalarType::F32
        }));
        let shaders = crate::backend::vulkan::VulkanGlslBackend
            .lower_recursive_fft(&ir)
            .unwrap();
        assert_eq!(shaders.len(), 3);
        for shader in &shaders {
            shader.compile_spirv().unwrap();
        }
    }

    #[test]
    fn large_stockham_axis_splits_to_fit_shared_memory() {
        let length = 4096usize;
        for direction in [Direction::Forward, Direction::Inverse] {
            let config = FftConfig::new(vec![length])
                .with_inverse_normalization(direction == Direction::Inverse);
            let plan = FftPlan::build(config).unwrap();
            let ir = RecursiveFftIr::build(&plan, direction, small_shared_memory_device()).unwrap();
            assert!(matches!(ir.root, RecursiveFftNodeIr::CooleyTukey(_)));
            assert_stockham_leaves_fit(&ir.root, 64);
            let input = sample(length);
            let actual = execute_recursive_fft_ir(&ir, &input).unwrap();
            let expected =
                crate::reference::fft(&input, direction, direction == Direction::Inverse).unwrap();
            assert!(
                max_error(&actual, &expected) < 2.0e-8 * length as f64,
                "large split Stockham mismatch for direction {direction:?}"
            );
        }
    }

    #[test]
    fn grouped_composite_rader_root_owns_parent_batch_boundary() {
        let length = 34usize;
        let batch_count = 5usize;
        let grouped_batch = 3usize;
        let mut profile = device();
        profile.max_workgroup_size = [1024, 1024, 64];

        for direction in [Direction::Forward, Direction::Inverse] {
            let config = FftConfig::new(vec![length])
                .with_batch_count(batch_count)
                .with_grouped_batch(0, grouped_batch)
                .unwrap()
                .with_inverse_normalization(direction == Direction::Inverse);
            let plan = FftPlan::build(config).unwrap();
            let ir = RecursiveFftIr::build(&plan, direction, profile).unwrap();
            assert!(ir.four_step_plan.is_none());
            assert!(ir.rader_forced_upload_schedule.is_none());
            let RecursiveFftNodeIr::CooleyTukey(root) = &ir.root else {
                panic!("N=34 grouped composite Rader should keep a Cooley root");
            };
            let block = root
                .pack_right
                .axis_batch_block
                .expect("composite root should own grouped parent transforms");
            assert_eq!(block.grouped_batch, grouped_batch);
            assert_eq!(root.twiddle_transpose.axis_batch_block, Some(block));
            assert_eq!(root.scatter_output.axis_batch_block, Some(block));
            assert_eq!(root.pack_right.dispatch.x, 2);
            assert_eq!(root.twiddle_transpose.dispatch.x, 2);
            assert_eq!(root.scatter_output.dispatch.x, 2);

            let input = sample(length * batch_count);
            let actual = execute_recursive_fft_ir(&ir, &input).unwrap();
            let mut expected = Vec::with_capacity(actual.len());
            for batch in 0..batch_count {
                let start = batch * length;
                expected.extend(dft(
                    &input[start..start + length],
                    direction,
                    direction == Direction::Inverse,
                ));
            }
            assert!(max_error(&actual, &expected) <= 2.0e-8 * length as f64);
            ir.validate().unwrap();
        }

        for direction in [Direction::Forward, Direction::Inverse] {
            let config = FftConfig::new(vec![length])
                .with_batch_count(batch_count)
                .with_grouped_batch(0, grouped_batch)
                .unwrap()
                .with_precision(Precision::F16StorageF32Compute)
                .with_inverse_normalization(direction == Direction::Inverse)
                .with_zero_padding(0, 7, 12)
                .unwrap();
            let plan = FftPlan::build(config).unwrap();
            let ir = RecursiveFftIr::build(&plan, direction, profile).unwrap();
            let RecursiveFftNodeIr::CooleyTukey(root) = &ir.root else {
                panic!("mixed-storage padded N=34 should keep a Cooley root");
            };
            let block = root.pack_right.axis_batch_block.unwrap();
            assert_eq!(block.grouped_batch, grouped_batch);
            assert_eq!(root.twiddle_transpose.axis_batch_block, Some(block));
            assert_eq!(root.scatter_output.axis_batch_block, Some(block));
            match direction {
                Direction::Forward => {
                    assert_eq!(root.pack_right.input_storage_scalar, ScalarType::F32);
                    assert_eq!(root.scatter_output.output_storage_scalar, ScalarType::F16);
                }
                Direction::Inverse => {
                    assert_eq!(root.pack_right.input_storage_scalar, ScalarType::F16);
                    assert_eq!(root.scatter_output.output_storage_scalar, ScalarType::F32);
                }
            }
            ir.validate().unwrap();
        }
    }

    #[test]
    fn composite_direct_rader_parent_uses_upstream_type1_thread_floor() {
        let length = 2usize * 47;
        let batch_count = 5usize;
        let mut profile = device();
        profile.max_workgroup_size = [1024, 1024, 64];

        for (grouped_batch, expected_group) in [(None, 1usize), (Some(3usize), 3usize)] {
            let mut config = FftConfig::new(vec![length]).with_batch_count(batch_count);
            if let Some(grouped_batch) = grouped_batch {
                config = config.with_grouped_batch(0, grouped_batch).unwrap();
            }
            let plan = FftPlan::build(config).unwrap();
            let ir = RecursiveFftIr::build(&plan, Direction::Forward, profile).unwrap();
            assert!(ir.four_step_plan.is_none());
            assert!(ir.rader_forced_upload_schedule.is_none());
            let RecursiveFftNodeIr::CooleyTukey(root) = &ir.root else {
                panic!("N=94 composite direct-Rader should keep a Cooley root");
            };
            assert_eq!((root.left_len, root.right_len), (2, 47));
            assert!(matches!(root.right, RecursiveFftNodeIr::DirectRader(_)));
            let block = root
                .pack_right
                .axis_batch_block
                .expect("composite direct-Rader root should retain physical parent lanes");
            assert_eq!(block.threads_per_transform, 48);
            assert_eq!(block.grouped_batch, expected_group);
            assert_eq!(root.twiddle_transpose.axis_batch_block, Some(block));
            assert_eq!(root.scatter_output.axis_batch_block, Some(block));
            assert_eq!(
                root.pack_right.dispatch.x as usize,
                batch_count.div_ceil(expected_group)
            );

            let shaders = crate::backend::vulkan::VulkanGlslBackend
                .lower_recursive_fft(&ir)
                .unwrap();
            for shader in &shaders {
                shader.compile_spirv().unwrap();
            }

            let input = sample(length * batch_count);
            let actual = execute_recursive_fft_ir(&ir, &input).unwrap();
            let mut expected = Vec::with_capacity(actual.len());
            for batch in 0..batch_count {
                let start = batch * length;
                expected.extend(dft(
                    &input[start..start + length],
                    Direction::Forward,
                    false,
                ));
            }
            assert!(max_error(&actual, &expected) <= 2.0e-8 * length as f64);
            ir.validate().unwrap();
        }
    }

    #[test]
    fn multiple_fft_rader_type0_parent_uses_joint_upstream_optimizer_floor() {
        let mut profile = device();
        profile.max_threads_per_block = 256;
        profile.max_workgroup_size = [256, 256, 64];
        let length = 19usize * 29;
        let batch_count = 2usize;
        let plan =
            FftPlan::build(FftConfig::new(vec![length]).with_batch_count(batch_count)).unwrap();
        let AxisAlgorithm::Rader { primes, .. } = &plan.axes[0].algorithm else {
            panic!("N551 should retain multi FFT-Rader metadata");
        };
        assert_eq!(primes.len(), 2);
        assert_eq!(primes[0].prime, 19);
        assert_eq!(primes[1].prime, 29);
        assert!(
            primes
                .iter()
                .all(|prime| matches!(prime.mode, RaderMode::FftConvolution { .. }))
        );

        let ir = RecursiveFftIr::build(&plan, Direction::Forward, profile).unwrap();
        assert!(ir.four_step_plan.is_none());
        assert!(ir.rader_forced_upload_schedule.is_none());
        let RecursiveFftNodeIr::CooleyTukey(root) = &ir.root else {
            panic!("N551 multi FFT-Rader should keep a Cooley root");
        };
        assert_eq!((root.left_len, root.right_len), (19, 29));
        assert!(matches!(root.left, RecursiveFftNodeIr::FftRader(_)));
        assert!(matches!(root.right, RecursiveFftNodeIr::FftRader(_)));
        let block = root.pack_right.axis_batch_block.unwrap();
        assert_eq!(block.threads_per_transform, 92);
        assert_eq!(block.grouped_batch, 1);
        assert_eq!([block.local_size_x, block.local_size_y], [92, 1]);
        assert_eq!(root.twiddle_transpose.axis_batch_block, Some(block));
        assert_eq!(root.scatter_output.axis_batch_block, Some(block));

        let shaders = crate::backend::vulkan::VulkanGlslBackend
            .lower_recursive_fft(&ir)
            .unwrap();
        for shader in &shaders {
            shader.compile_spirv().unwrap();
        }
        let mut impulse = vec![Complex64::new(0.0, 0.0); length * batch_count];
        for batch in 0..batch_count {
            impulse[batch * length + 1] = Complex64::new(1.0, 0.0);
        }
        let actual = execute_recursive_fft_ir(&ir, &impulse).unwrap();
        let error = actual
            .iter()
            .enumerate()
            .map(|(index, value)| {
                let k = index % length;
                let angle = -std::f64::consts::TAU * k as f64 / length as f64;
                (*value - Complex64::exp_i(angle)).norm_sqr().sqrt()
            })
            .fold(0.0f64, f64::max);
        assert!(error <= 2.0e-8 * length as f64);
        ir.validate().unwrap();
    }

    #[test]
    fn mixed_direct_multi_fft_rader_parent_uses_joint_type0_then_type1_floor() {
        let mut profile = device();
        profile.max_threads_per_block = 1024;
        profile.max_workgroup_size = [1024, 1024, 64];
        let length = 17usize * 19 * 23;
        let batch_count = 2usize;
        let mut tuning = crate::PlannerTuning::portable();
        tuning.min_rader_direct_prime = 17;
        tuning.min_rader_fft_prime = 19;
        tuning.validate().unwrap();
        let plan = FftPlan::build(
            FftConfig::new(vec![length])
                .with_batch_count(batch_count)
                .with_tuning(tuning),
        )
        .unwrap();
        let AxisAlgorithm::Rader { primes, .. } = &plan.axes[0].algorithm else {
            panic!("N7429 should retain mixed multi-Rader metadata");
        };
        assert_eq!(primes.len(), 3);
        assert_eq!(primes[0].prime, 17);
        assert!(matches!(primes[0].mode, RaderMode::DirectMultiplication));
        assert_eq!(primes[1].prime, 19);
        assert!(matches!(primes[1].mode, RaderMode::FftConvolution { .. }));
        assert_eq!(primes[2].prime, 23);
        assert!(matches!(primes[2].mode, RaderMode::FftConvolution { .. }));

        let ir = RecursiveFftIr::build(&plan, Direction::Forward, profile).unwrap();
        assert!(ir.four_step_plan.is_none());
        assert!(ir.rader_forced_upload_schedule.is_none());
        let RecursiveFftNodeIr::CooleyTukey(root) = &ir.root else {
            panic!("N7429 mixed multi-Rader should keep a Cooley root");
        };
        let block = root.pack_right.axis_batch_block.unwrap();
        assert_eq!(block.threads_per_transform, 990);
        assert_eq!(block.grouped_batch, 1);
        assert_eq!([block.local_size_x, block.local_size_y], [990, 1]);
        assert_eq!(root.twiddle_transpose.axis_batch_block, Some(block));
        assert_eq!(root.scatter_output.axis_batch_block, Some(block));

        fn count_rader_kinds(node: &RecursiveFftNodeIr) -> (usize, usize) {
            match node {
                RecursiveFftNodeIr::DirectRader(_) => (1, 0),
                RecursiveFftNodeIr::FftRader(_) => (0, 1),
                RecursiveFftNodeIr::CooleyTukey(root) => {
                    let left = count_rader_kinds(&root.left);
                    let right = count_rader_kinds(&root.right);
                    (left.0 + right.0, left.1 + right.1)
                }
                RecursiveFftNodeIr::Stockham(_) => (0, 0),
            }
        }
        assert_eq!(count_rader_kinds(&ir.root), (1, 2));

        let shaders = crate::backend::vulkan::VulkanGlslBackend
            .lower_recursive_fft(&ir)
            .unwrap();
        for shader in &shaders {
            shader.compile_spirv().unwrap();
        }
        let mut impulse = vec![Complex64::new(0.0, 0.0); length * batch_count];
        for batch in 0..batch_count {
            impulse[batch * length + 1] = Complex64::new(1.0, 0.0);
        }
        let actual = execute_recursive_fft_ir(&ir, &impulse).unwrap();
        let error = actual
            .iter()
            .enumerate()
            .map(|(index, value)| {
                let k = index % length;
                let angle = -std::f64::consts::TAU * k as f64 / length as f64;
                (*value - Complex64::exp_i(angle)).norm_sqr().sqrt()
            })
            .fold(0.0f64, f64::max);
        assert!(error <= 2.0e-8 * length as f64);
        ir.validate().unwrap();
    }

    #[test]
    fn mixed_direct_multi_fft_rader_global_register_scaling_reaches_recursive_parent() {
        let mut profile = upstream_scheduler_device();
        // Isolate the one-upload parent register scaler from normal capacity splitting.
        profile.shared_memory_bytes = 128 * 1024;
        profile.shared_memory_pow2_bytes = 128 * 1024;
        profile.max_threads_per_block = 1024;
        profile.max_workgroup_size = [1024, 1024, 64];
        let length = 17usize * 19 * 29;
        let mut tuning = crate::PlannerTuning::portable();
        tuning.min_rader_direct_prime = 17;
        tuning.min_rader_fft_prime = 19;
        tuning.validate().unwrap();
        let plan = FftPlan::build(FftConfig::new(vec![length]).with_tuning(tuning)).unwrap();
        let AxisAlgorithm::Rader { primes, .. } = &plan.axes[0].algorithm else {
            panic!("N9367 should retain mixed multi-Rader metadata");
        };
        assert_eq!(primes.len(), 3);
        assert_eq!(primes[0].prime, 17);
        assert!(matches!(primes[0].mode, RaderMode::DirectMultiplication));
        assert_eq!(primes[1].prime, 19);
        assert!(matches!(primes[1].mode, RaderMode::FftConvolution { .. }));
        assert_eq!(primes[2].prime, 29);
        assert!(matches!(primes[2].mode, RaderMode::FftConvolution { .. }));

        let ir = RecursiveFftIr::build(&plan, Direction::Forward, profile).unwrap();
        assert!(!ir.rader_forced_two_upload);
        assert!(ir.rader_forced_upload_schedule.is_none());
        assert!(ir.four_step_plan.is_none());
        let RecursiveFftNodeIr::CooleyTukey(root) = &ir.root else {
            panic!("N9367 1024-thread mixed multi-Rader should keep a Cooley root");
        };
        let block = root
            .pack_right
            .axis_batch_block
            .expect("N9367 one-upload parent must consume global scaleRegistersNum");
        assert_eq!(block.threads_per_transform, 999);
        assert_eq!(block.grouped_batch, 1);
        assert_eq!([block.local_size_x, block.local_size_y], [999, 1]);
        assert_eq!(root.twiddle_transpose.axis_batch_block, Some(block));
        assert_eq!(root.scatter_output.axis_batch_block, Some(block));

        let shaders = crate::backend::vulkan::VulkanGlslBackend
            .lower_recursive_fft(&ir)
            .unwrap();
        for shader in &shaders {
            shader.compile_spirv().unwrap();
        }
        let mut impulse = vec![Complex64::new(0.0, 0.0); length];
        impulse[1] = Complex64::new(1.0, 0.0);
        let actual = execute_recursive_fft_ir(&ir, &impulse).unwrap();
        let error = actual
            .iter()
            .enumerate()
            .map(|(k, value)| {
                let angle = -std::f64::consts::TAU * k as f64 / length as f64;
                (*value - Complex64::exp_i(angle)).norm_sqr().sqrt()
            })
            .fold(0.0f64, f64::max);
        assert!(error <= 2.0e-8 * length as f64);
        ir.validate().unwrap();
    }

    #[test]
    fn pure_multi_fft_rader_global_register_scaling_reaches_recursive_parent() {
        let mut profile = device();
        profile.max_threads_per_block = 1024;
        profile.max_workgroup_size = [1024, 1024, 64];
        let length = 19usize * 19 * 23;
        let mut tuning = crate::PlannerTuning::portable();
        tuning.min_rader_fft_prime = 19;
        tuning.validate().unwrap();
        let plan = FftPlan::build(FftConfig::new(vec![length]).with_tuning(tuning)).unwrap();
        let AxisAlgorithm::Rader { primes, .. } = &plan.axes[0].algorithm else {
            panic!("N8303 should retain pure repeated FFT-Rader metadata");
        };
        assert_eq!(primes.len(), 2);
        assert_eq!(primes[0].prime, 19);
        assert_eq!(primes[0].multiplicity, 2);
        assert!(matches!(primes[0].mode, RaderMode::FftConvolution { .. }));
        assert_eq!(primes[1].prime, 23);
        assert_eq!(primes[1].multiplicity, 1);
        assert!(matches!(primes[1].mode, RaderMode::FftConvolution { .. }));

        let ir = RecursiveFftIr::build(&plan, Direction::Forward, profile).unwrap();
        assert!(ir.four_step_plan.is_none());
        assert!(ir.rader_forced_upload_schedule.is_none());
        let RecursiveFftNodeIr::CooleyTukey(root) = &ir.root else {
            panic!("N8303 pure repeated FFT-Rader should keep a Cooley root");
        };
        let block = root.pack_right.axis_batch_block.unwrap();
        assert_eq!(block.threads_per_transform, 437);
        assert_eq!(block.grouped_batch, 1);
        assert_eq!([block.local_size_x, block.local_size_y], [437, 1]);
        assert_eq!(root.twiddle_transpose.axis_batch_block, Some(block));
        assert_eq!(root.scatter_output.axis_batch_block, Some(block));

        let shaders = crate::backend::vulkan::VulkanGlslBackend
            .lower_recursive_fft(&ir)
            .unwrap();
        for shader in &shaders {
            shader.compile_spirv().unwrap();
        }
        let mut impulse = vec![Complex64::new(0.0, 0.0); length];
        impulse[1] = Complex64::new(1.0, 0.0);
        let actual = execute_recursive_fft_ir(&ir, &impulse).unwrap();
        let error = actual
            .iter()
            .enumerate()
            .map(|(k, value)| {
                let angle = -std::f64::consts::TAU * k as f64 / length as f64;
                (*value - Complex64::exp_i(angle)).norm_sqr().sqrt()
            })
            .fold(0.0f64, f64::max);
        assert!(error <= 2.0e-8 * length as f64);
        ir.validate().unwrap();
    }

    #[test]
    fn mixed_direct_fft_rader_parent_uses_type0_optimized_type1_floor() {
        let mut profile = device();
        profile.max_threads_per_block = 256;
        profile.max_workgroup_size = [256, 256, 64];
        let length = 17usize * 31;
        let batch_count = 2usize;
        let mut tuning = crate::PlannerTuning::portable();
        tuning.min_rader_fft_prime = 19;
        tuning.validate().unwrap();
        let plan = FftPlan::build(
            FftConfig::new(vec![length])
                .with_batch_count(batch_count)
                .with_tuning(tuning),
        )
        .unwrap();
        let AxisAlgorithm::Rader { primes, .. } = &plan.axes[0].algorithm else {
            panic!("N527 should retain mixed Rader metadata");
        };
        assert_eq!(primes.len(), 2);
        assert_eq!(primes[0].prime, 17);
        assert!(matches!(primes[0].mode, RaderMode::DirectMultiplication));
        assert_eq!(primes[1].prime, 31);
        assert!(matches!(primes[1].mode, RaderMode::FftConvolution { .. }));

        let ir = RecursiveFftIr::build(&plan, Direction::Forward, profile).unwrap();
        assert!(ir.four_step_plan.is_none());
        assert!(ir.rader_forced_upload_schedule.is_none());
        let RecursiveFftNodeIr::CooleyTukey(root) = &ir.root else {
            panic!("N527 mixed Rader should keep a Cooley root");
        };
        assert_eq!((root.left_len, root.right_len), (17, 31));
        assert!(matches!(root.left, RecursiveFftNodeIr::DirectRader(_)));
        assert!(matches!(root.right, RecursiveFftNodeIr::FftRader(_)));
        let block = root.pack_right.axis_batch_block.unwrap();
        assert_eq!(block.threads_per_transform, 108);
        assert_eq!(block.grouped_batch, 1);
        assert_eq!([block.local_size_x, block.local_size_y], [108, 1]);
        assert_eq!(root.twiddle_transpose.axis_batch_block, Some(block));
        assert_eq!(root.scatter_output.axis_batch_block, Some(block));

        let shaders = crate::backend::vulkan::VulkanGlslBackend
            .lower_recursive_fft(&ir)
            .unwrap();
        for shader in &shaders {
            shader.compile_spirv().unwrap();
        }
        let mut impulse = vec![Complex64::new(0.0, 0.0); length * batch_count];
        for batch in 0..batch_count {
            impulse[batch * length + 1] = Complex64::new(1.0, 0.0);
        }
        let actual = execute_recursive_fft_ir(&ir, &impulse).unwrap();
        let error = actual
            .iter()
            .enumerate()
            .map(|(index, value)| {
                let k = index % length;
                let angle = -std::f64::consts::TAU * k as f64 / length as f64;
                (*value - Complex64::exp_i(angle)).norm_sqr().sqrt()
            })
            .fold(0.0f64, f64::max);
        assert!(error <= 2.0e-8 * length as f64);
        ir.validate().unwrap();
    }

    #[test]
    fn mixed_rader_parent_keeps_ordinary_outer_register_floor_with_smooth_radix() {
        let mut profile = device();
        profile.max_threads_per_block = 1024;
        profile.max_workgroup_size = [1024, 1024, 64];
        let length = 15usize * 17 * 31;
        let mut tuning = crate::PlannerTuning::portable();
        tuning.min_rader_fft_prime = 19;
        tuning.validate().unwrap();
        let plan = FftPlan::build(FftConfig::new(vec![length]).with_tuning(tuning)).unwrap();
        let AxisAlgorithm::Rader { primes, .. } = &plan.axes[0].algorithm else {
            panic!("N7905 should retain mixed Rader metadata");
        };
        assert!(primes.iter().any(|prime| {
            prime.prime == 17 && matches!(prime.mode, RaderMode::DirectMultiplication)
        }));
        assert!(primes.iter().any(|prime| {
            prime.prime == 31 && matches!(prime.mode, RaderMode::FftConvolution { .. })
        }));

        let ir = RecursiveFftIr::build(&plan, Direction::Forward, profile).unwrap();
        assert!(ir.four_step_plan.is_none());
        assert!(ir.rader_forced_upload_schedule.is_none());
        let RecursiveFftNodeIr::CooleyTukey(root) = &ir.root else {
            panic!("N7905 mixed Rader should keep a Cooley root");
        };
        let block = root.pack_right.axis_batch_block.unwrap();
        assert_eq!(block.threads_per_transform, 990);
        assert_eq!(block.grouped_batch, 1);
        assert_eq!([block.local_size_x, block.local_size_y], [990, 1]);
        assert_eq!(root.twiddle_transpose.axis_batch_block, Some(block));
        assert_eq!(root.scatter_output.axis_batch_block, Some(block));

        let shaders = crate::backend::vulkan::VulkanGlslBackend
            .lower_recursive_fft(&ir)
            .unwrap();
        for shader in &shaders {
            shader.compile_spirv().unwrap();
        }
        let mut impulse = vec![Complex64::new(0.0, 0.0); length];
        impulse[1] = Complex64::new(1.0, 0.0);
        let actual = execute_recursive_fft_ir(&ir, &impulse).unwrap();
        let error = actual
            .iter()
            .enumerate()
            .map(|(k, value)| {
                let angle = -std::f64::consts::TAU * k as f64 / length as f64;
                (*value - Complex64::exp_i(angle)).norm_sqr().sqrt()
            })
            .fold(0.0f64, f64::max);
        assert!(error <= 2.0e-8 * length as f64);
        ir.validate().unwrap();
    }

    #[test]
    fn repeated_direct_rader_parent_uses_single_upstream_container_multiplier() {
        let mut profile = device();
        profile.max_workgroup_size = [256, 256, 64];
        profile.max_threads_per_block = 256;
        let length = 17usize * 17;
        let batch_count = 2usize;
        let mut tuning = crate::PlannerTuning::portable();
        tuning.min_rader_fft_prime = 29;
        tuning.validate().unwrap();
        let plan = FftPlan::build(
            FftConfig::new(vec![length])
                .with_batch_count(batch_count)
                .with_tuning(tuning),
        )
        .unwrap();
        let AxisAlgorithm::Rader { primes, .. } = &plan.axes[0].algorithm else {
            panic!("N289 should retain Rader metadata");
        };
        assert_eq!(primes.len(), 1);
        assert_eq!(primes[0].prime, 17);
        assert_eq!(primes[0].multiplicity, 2);
        assert!(matches!(primes[0].mode, RaderMode::DirectMultiplication));

        let ir = RecursiveFftIr::build(&plan, Direction::Forward, profile).unwrap();
        assert!(ir.four_step_plan.is_none());
        assert!(ir.rader_forced_upload_schedule.is_none());
        let RecursiveFftNodeIr::CooleyTukey(root) = &ir.root else {
            panic!("N289 repeated direct-Rader should keep a Cooley root");
        };
        assert_eq!((root.left_len, root.right_len), (17, 17));
        assert!(matches!(root.left, RecursiveFftNodeIr::DirectRader(_)));
        assert!(matches!(root.right, RecursiveFftNodeIr::DirectRader(_)));
        let block = root.pack_right.axis_batch_block.unwrap();
        assert_eq!(block.threads_per_transform, 153);
        assert_eq!(block.grouped_batch, 1);
        assert_eq!([block.local_size_x, block.local_size_y], [153, 1]);
        assert_eq!(root.twiddle_transpose.axis_batch_block, Some(block));
        assert_eq!(root.scatter_output.axis_batch_block, Some(block));

        let shaders = crate::backend::vulkan::VulkanGlslBackend
            .lower_recursive_fft(&ir)
            .unwrap();
        for shader in &shaders {
            shader.compile_spirv().unwrap();
        }
        let mut impulse = vec![Complex64::new(0.0, 0.0); length * batch_count];
        for batch in 0..batch_count {
            impulse[batch * length + 1] = Complex64::new(1.0, 0.0);
        }
        let actual = execute_recursive_fft_ir(&ir, &impulse).unwrap();
        let error = actual
            .iter()
            .enumerate()
            .map(|(index, value)| {
                let k = index % length;
                let angle = -std::f64::consts::TAU * k as f64 / length as f64;
                (*value - Complex64::exp_i(angle)).norm_sqr().sqrt()
            })
            .fold(0.0f64, f64::max);
        assert!(error <= 2.0e-8 * length as f64);
        ir.validate().unwrap();
    }

    #[test]
    fn multi_direct_rader_parent_shares_upstream_register_scale() {
        let mut profile = device();
        profile.max_workgroup_size = [1024, 1024, 64];
        let mut tuning = crate::PlannerTuning::portable();
        tuning.min_rader_fft_prime = 89;
        tuning.validate().unwrap();
        let length = 47usize * 53;
        let plan = FftPlan::build(
            FftConfig::new(vec![length])
                .with_batch_count(2)
                .with_tuning(tuning),
        )
        .unwrap();
        let ir = RecursiveFftIr::build(&plan, Direction::Forward, profile).unwrap();
        assert!(ir.four_step_plan.is_none());
        let RecursiveFftNodeIr::CooleyTukey(root) = &ir.root else {
            panic!("p47*p53 all-direct Rader axis should keep a Cooley root");
        };
        let block = root.pack_right.axis_batch_block.unwrap();
        assert_eq!(block.threads_per_transform, 216);
        assert_eq!(block.grouped_batch, 1);
        assert_eq!(root.twiddle_transpose.axis_batch_block, Some(block));
        assert_eq!(root.scatter_output.axis_batch_block, Some(block));
        assert!(matches!(root.left, RecursiveFftNodeIr::DirectRader(_)));
        assert!(matches!(root.right, RecursiveFftNodeIr::DirectRader(_)));

        let shaders = crate::backend::vulkan::VulkanGlslBackend
            .lower_recursive_fft(&ir)
            .unwrap();
        for shader in &shaders {
            shader.compile_spirv().unwrap();
        }
        let mut impulse = vec![Complex64::new(0.0, 0.0); length * 2];
        impulse[1] = Complex64::new(1.0, 0.0);
        impulse[length + 1] = Complex64::new(1.0, 0.0);
        let actual = execute_recursive_fft_ir(&ir, &impulse).unwrap();
        let error = actual
            .iter()
            .enumerate()
            .map(|(index, value)| {
                let k = index % length;
                let angle = -std::f64::consts::TAU * k as f64 / length as f64;
                (*value - Complex64::exp_i(angle)).norm_sqr().sqrt()
            })
            .fold(0.0f64, f64::max);
        assert!(error <= 2.0e-8 * length as f64);
        ir.validate().unwrap();
    }

    #[test]
    fn amd_f16_n8789_fft_rader_four_step_uses_storage_coalescing() {
        let length = 11usize * 17 * 47;
        let profile = DeviceProfile {
            shared_memory_bytes: 8 * 1024,
            shared_memory_pow2_bytes: 8 * 1024,
            max_threads_per_block: 1024,
            max_workgroup_size: [1024, 1024, 64],
            ..DeviceProfile::generic(Backend::Vulkan, GpuVendor::Amd)
        };
        let plan = FftPlan::build(
            FftConfig::new(vec![length]).with_precision(Precision::F16StorageF32Compute),
        )
        .unwrap();
        let ir = RecursiveFftIr::build(&plan, Direction::Forward, profile).unwrap();
        let schedule = ir.rader_forced_upload_schedule.as_ref().unwrap();
        assert_eq!(schedule.axis_split, vec![47, 17, 11]);
        assert!(ir.rader_forced_two_upload);

        let RecursiveFftNodeIr::CooleyTukey(root) = &ir.root else {
            panic!("AMD F16 N8789 should keep a Cooley root");
        };
        let RecursiveFftNodeIr::CooleyTukey(upper) = &root.right else {
            panic!("AMD F16 N8789 should keep a nested Cooley upper branch");
        };
        assert!(matches!(
            upper.left,
            RecursiveFftNodeIr::FftRader(ref rader) if rader.prime == 17
        ));
        let block = ir
            .four_step_plan
            .as_ref()
            .unwrap()
            .uploads
            .iter()
            .find(|upload| upload.axis_upload_id == 1)
            .unwrap()
            .axis_block
            .unwrap();
        assert_eq!(block.threads_per_transform, 2);
        assert_eq!(block.grouped_batch, 32);
        assert_eq!([block.local_size_x, block.local_size_y], [32, 2]);
        assert!(block.transforms_on_x);
        assert!(!block.axis_swapped);
    }

    #[test]
    fn repeated_and_multi_prime_rader_trees_match_dft() {
        for length in [289usize, 323, 578] {
            for direction in [Direction::Forward, Direction::Inverse] {
                let config = FftConfig::new(vec![length])
                    .with_inverse_normalization(direction == Direction::Inverse);
                let plan = FftPlan::build(config).unwrap();
                let ir = RecursiveFftIr::build(&plan, direction, device()).unwrap();
                assert_eq!(ir.logical_len, length);
                assert!(matches!(ir.root, RecursiveFftNodeIr::CooleyTukey(_)));
                let input = sample(length);
                let actual = execute_recursive_fft_ir(&ir, &input).unwrap();
                let expected = dft(&input, direction, direction == Direction::Inverse);
                assert!(
                    max_error(&actual, &expected) < 2.0e-8 * length as f64,
                    "recursive transform mismatch for length {length} direction {direction:?}"
                );
            }
        }
    }
}
