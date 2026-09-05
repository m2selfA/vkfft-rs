//! Correctness-first real-to-real DCT/DST execution IR.
//!
//! VkFFT compares its DCT implementation directly with FFTW's REDFT family.
//! This module adopts the same unnormalized REDFT/RODFT definitions for forward
//! execution. Inverse execution swaps type II/III and, when requested, applies
//! the matching FFTW inverse-pair scale. The first Vulkan slice evaluates the
//! definition directly; later pre/post-processing can replace it without changing
//! this public numerical contract.

use core::f64::consts::PI;

use crate::complex::Complex64;
use crate::config::{
    DctType, DeviceProfile, Direction, DstType, FftConfig, Precision, TransformKind,
    ZeroPaddingRange,
};
use crate::error::{Result, VkFftError};
use crate::kernel_ir::{DispatchGeometry, ScalarType, WorkgroupSize};
use crate::nd_ir::{
    NdExternalTensorLayout, NdFormattedCopyOperation, NdFormattedCopyPassIr,
    pack_logical_tensor_batches, unpack_logical_tensor_batches,
};
use crate::one_dim_ir::{OneDimFftIr, execute_one_dim_fft_ir};
use crate::planner::{C2cDeviceAxisClass, FftPlan};
use crate::scheduler::{StockhamAxisBlockSchedule, plan_gpu_other_axis_elementwise_wrapper_block};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum R2rTransform {
    Dct(DctType),
    Dst(DstType),
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum R2rFftPassOperation {
    DctIEvenExtension,
    DctIExtract { scale: f64 },
    DstIOddExtension,
    DstIExtract { scale: f64 },
    DctIvEvenPack,
    DctIvEvenExtract { scale: f64 },
    DstIvEvenPack,
    DstIvEvenExtract { scale: f64 },
    DctIvOddPhasePack,
    DctIvOddPhaseExtract { scale: f64 },
    DstIvOddPhasePack,
    DstIvOddPhaseExtract { scale: f64 },
    DctIiReorder,
    DctIiExtract { scale: f64 },
    DctIiiSpectrum,
    DctIiiUnreorder { scale: f64 },
    DstIiReorder,
    DstIiExtract { scale: f64 },
    DstIiiSpectrum,
    DstIiiUnreorder { scale: f64 },
}

#[derive(Debug, Clone, PartialEq)]
pub struct R2rFftPassIr {
    pub name: String,
    pub scalar: ScalarType,
    pub input_storage_scalar: ScalarType,
    pub output_storage_scalar: ScalarType,
    pub length: usize,
    pub fft_len: usize,
    pub batch_count: usize,
    pub grouped_batch: usize,
    pub axis_batch_block: Option<StockhamAxisBlockSchedule>,
    /// Spatial zero-padding interval fused into this external scalar boundary.
    /// Only the forward preprocess or inverse postprocess carries it.
    pub zero_padding: Option<ZeroPaddingRange>,
    pub operation: R2rFftPassOperation,
    pub workgroup_size: WorkgroupSize,
    pub dispatch: DispatchGeometry,
}

fn r2r_fft_pass_expected_len(length: usize, operation: R2rFftPassOperation) -> Result<usize> {
    match operation {
        R2rFftPassOperation::DctIEvenExtension | R2rFftPassOperation::DctIExtract { .. } => length
            .checked_sub(1)
            .and_then(|value| value.checked_mul(2))
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "DCT-I FFT reduction length",
            }),
        R2rFftPassOperation::DstIOddExtension | R2rFftPassOperation::DstIExtract { .. } => length
            .checked_add(1)
            .and_then(|value| value.checked_mul(2))
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "DST-I FFT reduction length",
            }),
        R2rFftPassOperation::DctIvEvenPack
        | R2rFftPassOperation::DctIvEvenExtract { .. }
        | R2rFftPassOperation::DstIvEvenPack
        | R2rFftPassOperation::DstIvEvenExtract { .. } => {
            if length == 0 || !length.is_multiple_of(2) {
                return Err(VkFftError::InvalidKernelIr(
                    "DCT/DST-IV half-size FFT reduction requires an even length",
                ));
            }
            Ok(length / 2)
        }
        R2rFftPassOperation::DctIvOddPhasePack
        | R2rFftPassOperation::DctIvOddPhaseExtract { .. }
        | R2rFftPassOperation::DstIvOddPhasePack
        | R2rFftPassOperation::DstIvOddPhaseExtract { .. } => {
            length.checked_mul(2).ok_or(VkFftError::ArithmeticOverflow {
                operation: "odd DCT/DST-IV FFT reduction length",
            })
        }
        R2rFftPassOperation::DctIiReorder
        | R2rFftPassOperation::DctIiExtract { .. }
        | R2rFftPassOperation::DctIiiSpectrum
        | R2rFftPassOperation::DctIiiUnreorder { .. }
        | R2rFftPassOperation::DstIiReorder
        | R2rFftPassOperation::DstIiExtract { .. }
        | R2rFftPassOperation::DstIiiSpectrum
        | R2rFftPassOperation::DstIiiUnreorder { .. } => Ok(length),
    }
}

fn r2r_fft_preprocess_operation(operation: R2rFftPassOperation) -> bool {
    matches!(
        operation,
        R2rFftPassOperation::DctIEvenExtension
            | R2rFftPassOperation::DstIOddExtension
            | R2rFftPassOperation::DctIvEvenPack
            | R2rFftPassOperation::DstIvEvenPack
            | R2rFftPassOperation::DctIvOddPhasePack
            | R2rFftPassOperation::DstIvOddPhasePack
            | R2rFftPassOperation::DctIiReorder
            | R2rFftPassOperation::DctIiiSpectrum
            | R2rFftPassOperation::DstIiReorder
            | R2rFftPassOperation::DstIiiSpectrum
    )
}

fn r2r_fft_postprocess_operation(operation: R2rFftPassOperation) -> bool {
    matches!(
        operation,
        R2rFftPassOperation::DctIExtract { .. }
            | R2rFftPassOperation::DstIExtract { .. }
            | R2rFftPassOperation::DctIvEvenExtract { .. }
            | R2rFftPassOperation::DstIvEvenExtract { .. }
            | R2rFftPassOperation::DctIvOddPhaseExtract { .. }
            | R2rFftPassOperation::DstIvOddPhaseExtract { .. }
            | R2rFftPassOperation::DctIiExtract { .. }
            | R2rFftPassOperation::DctIiiUnreorder { .. }
            | R2rFftPassOperation::DstIiExtract { .. }
            | R2rFftPassOperation::DstIiiUnreorder { .. }
    )
}

fn r2r_fft_operation_scale(operation: R2rFftPassOperation) -> f64 {
    match operation {
        R2rFftPassOperation::DctIEvenExtension
        | R2rFftPassOperation::DstIOddExtension
        | R2rFftPassOperation::DctIvEvenPack
        | R2rFftPassOperation::DstIvEvenPack
        | R2rFftPassOperation::DctIvOddPhasePack
        | R2rFftPassOperation::DstIvOddPhasePack
        | R2rFftPassOperation::DctIiReorder
        | R2rFftPassOperation::DctIiiSpectrum
        | R2rFftPassOperation::DstIiReorder
        | R2rFftPassOperation::DstIiiSpectrum => 1.0,
        R2rFftPassOperation::DctIExtract { scale }
        | R2rFftPassOperation::DstIExtract { scale }
        | R2rFftPassOperation::DctIvEvenExtract { scale }
        | R2rFftPassOperation::DstIvEvenExtract { scale }
        | R2rFftPassOperation::DctIvOddPhaseExtract { scale }
        | R2rFftPassOperation::DstIvOddPhaseExtract { scale }
        | R2rFftPassOperation::DctIiExtract { scale }
        | R2rFftPassOperation::DctIiiUnreorder { scale }
        | R2rFftPassOperation::DstIiExtract { scale }
        | R2rFftPassOperation::DstIiiUnreorder { scale } => scale,
    }
}

impl R2rFftPassIr {
    fn new(
        name: String,
        scalar: ScalarType,
        length: usize,
        fft_len: usize,
        batch_count: usize,
        grouped_batch: usize,
        operation: R2rFftPassOperation,
        device: DeviceProfile,
    ) -> Result<Self> {
        if grouped_batch == 0 {
            return Err(VkFftError::InvalidKernelIr(
                "R2R FFT groupedBatch must be non-zero",
            ));
        }
        let local_size = fft_len.min(device.max_threads_per_block).max(1);
        let pass = Self {
            name,
            scalar,
            input_storage_scalar: scalar,
            output_storage_scalar: scalar,
            length,
            fft_len,
            batch_count,
            grouped_batch,
            axis_batch_block: None,
            zero_padding: None,
            operation,
            workgroup_size: WorkgroupSize {
                x: u32::try_from(local_size).map_err(|_| VkFftError::ValueOutOfRange {
                    field: "R2R FFT reduction workgroup size",
                })?,
                y: 1,
                z: 1,
            },
            dispatch: DispatchGeometry {
                x: u32::try_from(batch_count.div_ceil(grouped_batch)).map_err(|_| {
                    VkFftError::ValueOutOfRange {
                        field: "R2R FFT grouped dispatch count",
                    }
                })?,
                y: 1,
                z: 1,
            },
        };
        pass.validate()?;
        Ok(pass)
    }

    fn with_external_input_storage(mut self, storage: ScalarType) -> Result<Self> {
        if storage != self.scalar
            && (!supported_r2r_storage_pair(self.scalar, storage)
                || !r2r_fft_preprocess_operation(self.operation))
        {
            return Err(VkFftError::UnsupportedPrecision {
                backend: "R2R FFT preprocess",
                precision: "mixed storage on unsupported preprocess boundary",
            });
        }
        self.input_storage_scalar = storage;
        self.validate()?;
        Ok(self)
    }

    fn with_external_output_storage(mut self, storage: ScalarType) -> Result<Self> {
        if storage != self.scalar
            && (!supported_r2r_storage_pair(self.scalar, storage)
                || !r2r_fft_postprocess_operation(self.operation))
        {
            return Err(VkFftError::UnsupportedPrecision {
                backend: "R2R FFT postprocess",
                precision: "mixed storage on unsupported postprocess boundary",
            });
        }
        self.output_storage_scalar = storage;
        self.validate()?;
        Ok(self)
    }

    fn with_zero_padding(mut self, range: ZeroPaddingRange) -> Result<Self> {
        if range.left > range.right || range.right > self.length {
            return Err(VkFftError::InvalidZeroPaddingRange {
                axis: 0,
                left: range.left,
                right: range.right,
                length: self.length,
            });
        }
        self.zero_padding = Some(range);
        self.validate()?;
        Ok(self)
    }

    fn with_axis_batch_block(
        mut self,
        block: StockhamAxisBlockSchedule,
        device: DeviceProfile,
    ) -> Result<Self> {
        block.validate(self.batch_count, device)?;
        self.axis_batch_block = Some(block);
        self.workgroup_size = WorkgroupSize {
            x: u32::try_from(block.local_size_x).map_err(|_| VkFftError::ValueOutOfRange {
                field: "R2R wrapper local_size_x",
            })?,
            y: u32::try_from(block.local_size_y).map_err(|_| VkFftError::ValueOutOfRange {
                field: "R2R wrapper local_size_y",
            })?,
            z: 1,
        };
        self.dispatch.x =
            u32::try_from(self.batch_count.div_ceil(block.grouped_batch)).map_err(|_| {
                VkFftError::ValueOutOfRange {
                    field: "R2R wrapper physical dispatch count",
                }
            })?;
        self.validate()?;
        Ok(self)
    }

    pub fn validate(&self) -> Result<()> {
        let expected_fft_len = r2r_fft_pass_expected_len(self.length, self.operation)?;
        let scale = r2r_fft_operation_scale(self.operation);
        let input_storage_ok = self.input_storage_scalar == self.scalar
            || (supported_r2r_storage_pair(self.scalar, self.input_storage_scalar)
                && r2r_fft_preprocess_operation(self.operation));
        let output_storage_ok = self.output_storage_scalar == self.scalar
            || (supported_r2r_storage_pair(self.scalar, self.output_storage_scalar)
                && r2r_fft_postprocess_operation(self.operation));
        if self.scalar == ScalarType::F16
            || !input_storage_ok
            || !output_storage_ok
            || self.length == 0
            || self.batch_count == 0
            || self.grouped_batch == 0
            || self.fft_len != expected_fft_len
            || self.workgroup_size.x == 0
            || self.workgroup_size.y == 0
            || self.workgroup_size.z != 1
            || self.dispatch.y != 1
            || self.dispatch.z != 1
            || !scale.is_finite()
            || scale <= 0.0
            || self
                .zero_padding
                .is_some_and(|range| range.left > range.right || range.right > self.length)
        {
            return Err(VkFftError::InvalidKernelIr(
                "R2R FFT reduction pass metadata is inconsistent",
            ));
        }
        if let Some(block) = self.axis_batch_block {
            let expected = if block.transforms_on_x {
                [block.grouped_batch, block.threads_per_transform]
            } else {
                [block.threads_per_transform, block.grouped_batch]
            };
            if [
                self.workgroup_size.x as usize,
                self.workgroup_size.y as usize,
            ] != expected
                || self.dispatch.x as usize != self.batch_count.div_ceil(block.grouped_batch)
            {
                return Err(VkFftError::InvalidKernelIr(
                    "R2R FFT wrapper physical block is inconsistent",
                ));
            }
        } else if self.workgroup_size.y != 1
            || self.dispatch.x as usize != self.batch_count.div_ceil(self.grouped_batch)
        {
            return Err(VkFftError::InvalidKernelIr(
                "R2R FFT logical groupedBatch dispatch is inconsistent",
            ));
        }
        Ok(())
    }
}

fn r2r_fft_reduction_expected_len(transform: R2rTransform, length: usize) -> Result<usize> {
    match transform {
        R2rTransform::Dct(DctType::I) => length
            .checked_sub(1)
            .and_then(|value| value.checked_mul(2))
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "DCT-I FFT reduction length",
            }),
        R2rTransform::Dst(DstType::I) => length
            .checked_add(1)
            .and_then(|value| value.checked_mul(2))
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "DST-I FFT reduction length",
            }),
        R2rTransform::Dct(DctType::IV) | R2rTransform::Dst(DstType::IV)
            if length.is_multiple_of(2) && length > 0 =>
        {
            Ok(length / 2)
        }
        R2rTransform::Dct(DctType::IV) | R2rTransform::Dst(DstType::IV) if length > 0 => {
            length.checked_mul(2).ok_or(VkFftError::ArithmeticOverflow {
                operation: "odd DCT/DST-IV FFT reduction length",
            })
        }
        R2rTransform::Dct(DctType::II | DctType::III)
        | R2rTransform::Dst(DstType::II | DstType::III) => Ok(length),
        _ => Err(VkFftError::InvalidKernelIr(
            "R2R transform does not use the ordinary FFT reduction",
        )),
    }
}

fn plan_axis0_r2r_wrapper_block(
    element_len: usize,
    batch_count: usize,
    grouped_batch: usize,
    device: DeviceProfile,
) -> Result<Option<StockhamAxisBlockSchedule>> {
    if element_len == 0
        || grouped_batch == 0
        || grouped_batch > batch_count
        || grouped_batch > device.max_workgroup_size[1]
        || grouped_batch > device.max_threads_per_block
    {
        return Ok(None);
    }
    let max_x_by_threads = device.max_threads_per_block / grouped_batch;
    let threads_per_transform = element_len
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
pub struct R2rFftReductionIr {
    pub effective_transform: R2rTransform,
    pub length: usize,
    pub fft_len: usize,
    pub batch_count: usize,
    pub grouped_batch: usize,
    pub scalar: ScalarType,
    pub external_scalar: ScalarType,
    pub fft: OneDimFftIr,
    pub preprocess: R2rFftPassIr,
    pub postprocess: R2rFftPassIr,
}

impl R2rFftReductionIr {
    fn apply_axis0_wrapper_block(&mut self, device: DeviceProfile) -> Result<()> {
        if self.grouped_batch > 1
            && let Some(block) = plan_axis0_r2r_wrapper_block(
                self.fft_len,
                self.batch_count,
                self.grouped_batch,
                device,
            )?
        {
            self.preprocess = self
                .preprocess
                .clone()
                .with_axis_batch_block(block, device)?;
            self.postprocess = self
                .postprocess
                .clone()
                .with_axis_batch_block(block, device)?;
        }
        self.validate()
    }

    fn apply_other_axis_wrapper_block(
        &mut self,
        fastest_axis_len: usize,
        grouped_batch_override: Option<usize>,
        axis1_grouped_batch_override: Option<usize>,
        device: DeviceProfile,
    ) -> Result<()> {
        if let Some(block) = plan_gpu_other_axis_elementwise_wrapper_block(
            self.fft_len,
            self.batch_count,
            fastest_axis_len,
            self.scalar.complex_bytes(),
            grouped_batch_override,
            axis1_grouped_batch_override,
            device,
        )? {
            self.preprocess = self
                .preprocess
                .clone()
                .with_axis_batch_block(block, device)?;
            self.postprocess = self
                .postprocess
                .clone()
                .with_axis_batch_block(block, device)?;
        }
        self.validate()
    }

    pub fn validate(&self) -> Result<()> {
        self.preprocess.validate()?;
        self.postprocess.validate()?;
        if self.preprocess.axis_batch_block != self.postprocess.axis_batch_block {
            return Err(VkFftError::InvalidKernelIr(
                "R2R FFT preprocess/postprocess physical ownership differs",
            ));
        }
        let expected_direction = match self.preprocess.operation {
            R2rFftPassOperation::DctIiiSpectrum
            | R2rFftPassOperation::DstIiiSpectrum
            | R2rFftPassOperation::DctIvEvenPack
            | R2rFftPassOperation::DstIvEvenPack => Direction::Inverse,
            R2rFftPassOperation::DctIEvenExtension
            | R2rFftPassOperation::DstIOddExtension
            | R2rFftPassOperation::DctIiReorder
            | R2rFftPassOperation::DstIiReorder
            | R2rFftPassOperation::DctIvOddPhasePack
            | R2rFftPassOperation::DstIvOddPhasePack => Direction::Forward,
            _ => {
                return Err(VkFftError::InvalidKernelIr(
                    "R2R FFT reduction preprocess operation cannot determine FFT direction",
                ));
            }
        };
        if self.fft_len != r2r_fft_reduction_expected_len(self.effective_transform, self.length)?
            || self.fft.logical_len() != self.fft_len
            || self.fft.batch_count() != self.batch_count
            || self.fft.scalar() != self.scalar
            || self.fft.external_storage_scalar() != self.scalar
            || self.fft.direction() != expected_direction
            || self.preprocess.scalar != self.scalar
            || self.postprocess.scalar != self.scalar
            || self.preprocess.length != self.length
            || self.postprocess.length != self.length
            || self.preprocess.fft_len != self.fft_len
            || self.postprocess.fft_len != self.fft_len
            || self.preprocess.batch_count != self.batch_count
            || self.postprocess.batch_count != self.batch_count
            || self.grouped_batch == 0
            || self.preprocess.grouped_batch != self.grouped_batch
            || self.postprocess.grouped_batch != self.grouped_batch
            || (self.external_scalar != self.scalar
                && !supported_r2r_storage_pair(self.scalar, self.external_scalar))
            || self.preprocess.input_storage_scalar != self.external_scalar
            || self.preprocess.output_storage_scalar != self.scalar
            || self.postprocess.input_storage_scalar != self.scalar
            || self.postprocess.output_storage_scalar != self.external_scalar
        {
            return Err(VkFftError::InvalidKernelIr(
                "R2R FFT reduction metadata does not match its complex FFT",
            ));
        }
        match (
            self.effective_transform,
            self.preprocess.operation,
            self.postprocess.operation,
        ) {
            (
                R2rTransform::Dct(DctType::I),
                R2rFftPassOperation::DctIEvenExtension,
                R2rFftPassOperation::DctIExtract { .. },
            )
            | (
                R2rTransform::Dst(DstType::I),
                R2rFftPassOperation::DstIOddExtension,
                R2rFftPassOperation::DstIExtract { .. },
            )
            | (
                R2rTransform::Dst(DstType::II),
                R2rFftPassOperation::DstIiReorder,
                R2rFftPassOperation::DstIiExtract { .. },
            )
            | (
                R2rTransform::Dst(DstType::III),
                R2rFftPassOperation::DstIiiSpectrum,
                R2rFftPassOperation::DstIiiUnreorder { .. },
            )
            | (
                R2rTransform::Dct(DctType::IV),
                R2rFftPassOperation::DctIvEvenPack,
                R2rFftPassOperation::DctIvEvenExtract { .. },
            )
            | (
                R2rTransform::Dst(DstType::IV),
                R2rFftPassOperation::DstIvEvenPack,
                R2rFftPassOperation::DstIvEvenExtract { .. },
            )
            | (
                R2rTransform::Dct(DctType::IV),
                R2rFftPassOperation::DctIvOddPhasePack,
                R2rFftPassOperation::DctIvOddPhaseExtract { .. },
            )
            | (
                R2rTransform::Dst(DstType::IV),
                R2rFftPassOperation::DstIvOddPhasePack,
                R2rFftPassOperation::DstIvOddPhaseExtract { .. },
            )
            | (
                R2rTransform::Dct(DctType::II),
                R2rFftPassOperation::DctIiReorder,
                R2rFftPassOperation::DctIiExtract { .. },
            )
            | (
                R2rTransform::Dct(DctType::III),
                R2rFftPassOperation::DctIiiSpectrum,
                R2rFftPassOperation::DctIiiUnreorder { .. },
            ) => Ok(()),
            _ => Err(VkFftError::InvalidKernelIr(
                "R2R FFT reduction pass operations do not match the transform",
            )),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct R2rIr {
    pub transform: R2rTransform,
    pub effective_transform: R2rTransform,
    pub direction: Direction,
    pub length: usize,
    pub batch_count: usize,
    pub grouped_batch: usize,
    pub axis_batch_block: Option<StockhamAxisBlockSchedule>,
    pub scalar: ScalarType,
    pub external_scalar: ScalarType,
    pub normalize: bool,
    pub normalization_scale: f64,
    /// Upstream spatial zero-padding range on the external scalar boundary.
    pub zero_padding: Option<ZeroPaddingRange>,
    pub workgroup_size: WorkgroupSize,
    pub dispatch: DispatchGeometry,
    pub fft_reduction: Option<Box<R2rFftReductionIr>>,
}

impl R2rIr {
    pub fn build(plan: &FftPlan, direction: Direction, device: DeviceProfile) -> Result<Self> {
        Self::build_with_axis_class(plan, direction, device, C2cDeviceAxisClass::Contiguous)
    }

    pub(crate) fn build_with_axis_class(
        plan: &FftPlan,
        direction: Direction,
        device: DeviceProfile,
        axis_class: C2cDeviceAxisClass,
    ) -> Result<Self> {
        if plan.config.dimensions.len() != 1 {
            return Err(VkFftError::UnsupportedKernelPath(
                "DCT/DST IR currently supports one-dimensional transforms only",
            ));
        }
        let zero_padding = plan.config.zero_padding_for_axis(0);
        let transform = match plan.config.transform {
            TransformKind::Dct(kind) => R2rTransform::Dct(kind),
            TransformKind::Dst(kind) => R2rTransform::Dst(kind),
            _ => {
                return Err(VkFftError::UnsupportedKernelPath(
                    "DCT/DST IR requires a DCT or DST planner configuration",
                ));
            }
        };
        let (scalar, external_scalar, compute_precision) = match plan.config.precision {
            Precision::F16StorageF32Compute => (ScalarType::F32, ScalarType::F16, Precision::F32),
            Precision::F32 => (ScalarType::F32, ScalarType::F32, Precision::F32),
            Precision::F64 if device.supports_f64 => {
                (ScalarType::F64, ScalarType::F64, Precision::F64)
            }
            Precision::F64ComputeF32Storage if device.supports_f64 => {
                (ScalarType::F64, ScalarType::F32, Precision::F64)
            }
            other => {
                return Err(VkFftError::UnsupportedPrecision {
                    backend: "DCT/DST IR",
                    precision: precision_name(other),
                });
            }
        };
        let length = plan.config.dimensions[0];
        let grouped_batch_override = plan.config.grouped_batch_for_axis(0);
        let grouped_batch = grouped_batch_override.unwrap_or(1);
        if matches!(transform, R2rTransform::Dct(DctType::I)) && length < 2 {
            return Err(VkFftError::InvalidTransformLength {
                axis: 0,
                transform: "DCT-I",
                length,
            });
        }
        if device.max_threads_per_block == 0 {
            return Err(VkFftError::ResourceLimitExceeded {
                resource: "threads per workgroup",
                required: 1,
                available: 0,
            });
        }
        let effective_transform = if direction == Direction::Inverse {
            inverse_partner(transform)
        } else {
            transform
        };
        let normalize = direction == Direction::Inverse && plan.config.normalize_inverse;
        let normalization_scale = if normalize {
            inverse_pair_scale(transform, length)?
        } else {
            1.0
        };
        let mut fft_reduction = match effective_transform {
            R2rTransform::Dct(DctType::I | DctType::II | DctType::III)
            | R2rTransform::Dst(DstType::I | DstType::II | DstType::III) => {
                let fft_len = r2r_fft_reduction_expected_len(effective_transform, length)?;
                let mut internal_config = FftConfig::new(vec![fft_len])
                    .with_batch_count(plan.config.batch_count)
                    .with_precision(compute_precision)
                    .with_inverse_normalization(false)
                    .with_tuning(plan.config.tuning)
                    .with_bandwidth_boost(plan.config.bandwidth_boost);
                if let Some(grouped_batch) = grouped_batch_override {
                    internal_config = internal_config.with_grouped_batch(0, grouped_batch)?;
                }
                let internal_plan =
                    FftPlan::build_c2c_child_for_device(internal_config, device, axis_class)?;
                let fft_direction = match effective_transform {
                    R2rTransform::Dct(DctType::I)
                    | R2rTransform::Dst(DstType::I)
                    | R2rTransform::Dct(DctType::II)
                    | R2rTransform::Dst(DstType::II) => Direction::Forward,
                    R2rTransform::Dct(DctType::III) | R2rTransform::Dst(DstType::III) => {
                        Direction::Inverse
                    }
                    _ => unreachable!(),
                };
                let fft = OneDimFftIr::build(&internal_plan, fft_direction, device)?;
                let fft = if grouped_batch_override.is_some() {
                    fft.with_axis0_single_upload_block(device)?
                } else {
                    fft
                };
                let (preprocess_operation, postprocess_operation, label) = match effective_transform
                {
                    R2rTransform::Dct(DctType::I) => (
                        R2rFftPassOperation::DctIEvenExtension,
                        R2rFftPassOperation::DctIExtract {
                            scale: normalization_scale,
                        },
                        "dct1",
                    ),
                    R2rTransform::Dst(DstType::I) => (
                        R2rFftPassOperation::DstIOddExtension,
                        R2rFftPassOperation::DstIExtract {
                            scale: normalization_scale,
                        },
                        "dst1",
                    ),
                    R2rTransform::Dct(DctType::II) => (
                        R2rFftPassOperation::DctIiReorder,
                        R2rFftPassOperation::DctIiExtract {
                            scale: normalization_scale,
                        },
                        "dct2",
                    ),
                    R2rTransform::Dct(DctType::III) => (
                        R2rFftPassOperation::DctIiiSpectrum,
                        R2rFftPassOperation::DctIiiUnreorder {
                            scale: normalization_scale,
                        },
                        "dct3",
                    ),
                    R2rTransform::Dst(DstType::II) => (
                        R2rFftPassOperation::DstIiReorder,
                        R2rFftPassOperation::DstIiExtract {
                            scale: normalization_scale,
                        },
                        "dst2",
                    ),
                    R2rTransform::Dst(DstType::III) => (
                        R2rFftPassOperation::DstIiiSpectrum,
                        R2rFftPassOperation::DstIiiUnreorder {
                            scale: normalization_scale,
                        },
                        "dst3",
                    ),
                    _ => unreachable!(),
                };
                let preprocess = R2rFftPassIr::new(
                    format!("vkfft_r2r_fft_{label}_pre_{length}"),
                    scalar,
                    length,
                    fft_len,
                    plan.config.batch_count,
                    grouped_batch,
                    preprocess_operation,
                    device,
                )?
                .with_external_input_storage(external_scalar)?;
                let postprocess = R2rFftPassIr::new(
                    format!("vkfft_r2r_fft_{label}_post_{length}"),
                    scalar,
                    length,
                    fft_len,
                    plan.config.batch_count,
                    grouped_batch,
                    postprocess_operation,
                    device,
                )?
                .with_external_output_storage(external_scalar)?;
                let reduction = R2rFftReductionIr {
                    effective_transform,
                    length,
                    fft_len,
                    batch_count: plan.config.batch_count,
                    grouped_batch,
                    scalar,
                    external_scalar,
                    fft,
                    preprocess,
                    postprocess,
                };
                reduction.validate()?;
                Some(Box::new(reduction))
            }
            R2rTransform::Dct(DctType::IV) | R2rTransform::Dst(DstType::IV)
                if length.is_multiple_of(2) =>
            {
                let fft_len = length / 2;
                let mut internal_config = FftConfig::new(vec![fft_len])
                    .with_batch_count(plan.config.batch_count)
                    .with_precision(compute_precision)
                    .with_inverse_normalization(false)
                    .with_tuning(plan.config.tuning)
                    .with_bandwidth_boost(plan.config.bandwidth_boost);
                if let Some(grouped_batch) = grouped_batch_override {
                    internal_config = internal_config.with_grouped_batch(0, grouped_batch)?;
                }
                let internal_plan =
                    FftPlan::build_c2c_child_for_device(internal_config, device, axis_class)?;
                let fft = OneDimFftIr::build(&internal_plan, Direction::Inverse, device)?;
                let fft = if grouped_batch_override.is_some() {
                    fft.with_axis0_single_upload_block(device)?
                } else {
                    fft
                };
                let (preprocess_operation, postprocess_operation, label) = match effective_transform
                {
                    R2rTransform::Dct(DctType::IV) => (
                        R2rFftPassOperation::DctIvEvenPack,
                        R2rFftPassOperation::DctIvEvenExtract {
                            scale: normalization_scale,
                        },
                        "dct4_even",
                    ),
                    R2rTransform::Dst(DstType::IV) => (
                        R2rFftPassOperation::DstIvEvenPack,
                        R2rFftPassOperation::DstIvEvenExtract {
                            scale: normalization_scale,
                        },
                        "dst4_even",
                    ),
                    _ => unreachable!(),
                };
                let preprocess = R2rFftPassIr::new(
                    format!("vkfft_r2r_fft_{label}_pre_{length}"),
                    scalar,
                    length,
                    fft_len,
                    plan.config.batch_count,
                    grouped_batch,
                    preprocess_operation,
                    device,
                )?
                .with_external_input_storage(external_scalar)?;
                let postprocess = R2rFftPassIr::new(
                    format!("vkfft_r2r_fft_{label}_post_{length}"),
                    scalar,
                    length,
                    fft_len,
                    plan.config.batch_count,
                    grouped_batch,
                    postprocess_operation,
                    device,
                )?
                .with_external_output_storage(external_scalar)?;
                let reduction = R2rFftReductionIr {
                    effective_transform,
                    length,
                    fft_len,
                    batch_count: plan.config.batch_count,
                    grouped_batch,
                    scalar,
                    external_scalar,
                    fft,
                    preprocess,
                    postprocess,
                };
                reduction.validate()?;
                Some(Box::new(reduction))
            }
            R2rTransform::Dct(DctType::IV) | R2rTransform::Dst(DstType::IV) => {
                let fft_len = length
                    .checked_mul(2)
                    .ok_or(VkFftError::ArithmeticOverflow {
                        operation: "odd DCT/DST-IV FFT reduction length",
                    })?;
                let mut internal_config = FftConfig::new(vec![fft_len])
                    .with_batch_count(plan.config.batch_count)
                    .with_precision(compute_precision)
                    .with_inverse_normalization(false)
                    .with_tuning(plan.config.tuning)
                    .with_bandwidth_boost(plan.config.bandwidth_boost);
                if let Some(grouped_batch) = grouped_batch_override {
                    internal_config = internal_config.with_grouped_batch(0, grouped_batch)?;
                }
                let internal_plan =
                    FftPlan::build_c2c_child_for_device(internal_config, device, axis_class)?;
                let fft = OneDimFftIr::build(&internal_plan, Direction::Forward, device)?;
                let fft = if grouped_batch_override.is_some() {
                    fft.with_axis0_single_upload_block(device)?
                } else {
                    fft
                };
                let (preprocess_operation, postprocess_operation, label) = match effective_transform
                {
                    R2rTransform::Dct(DctType::IV) => (
                        R2rFftPassOperation::DctIvOddPhasePack,
                        R2rFftPassOperation::DctIvOddPhaseExtract {
                            scale: normalization_scale,
                        },
                        "dct4_odd",
                    ),
                    R2rTransform::Dst(DstType::IV) => (
                        R2rFftPassOperation::DstIvOddPhasePack,
                        R2rFftPassOperation::DstIvOddPhaseExtract {
                            scale: normalization_scale,
                        },
                        "dst4_odd",
                    ),
                    _ => unreachable!(),
                };
                let preprocess = R2rFftPassIr::new(
                    format!("vkfft_r2r_fft_{label}_pre_{length}"),
                    scalar,
                    length,
                    fft_len,
                    plan.config.batch_count,
                    grouped_batch,
                    preprocess_operation,
                    device,
                )?
                .with_external_input_storage(external_scalar)?;
                let postprocess = R2rFftPassIr::new(
                    format!("vkfft_r2r_fft_{label}_post_{length}"),
                    scalar,
                    length,
                    fft_len,
                    plan.config.batch_count,
                    grouped_batch,
                    postprocess_operation,
                    device,
                )?
                .with_external_output_storage(external_scalar)?;
                let reduction = R2rFftReductionIr {
                    effective_transform,
                    length,
                    fft_len,
                    batch_count: plan.config.batch_count,
                    grouped_batch,
                    scalar,
                    external_scalar,
                    fft,
                    preprocess,
                    postprocess,
                };
                reduction.validate()?;
                Some(Box::new(reduction))
            }
        };
        if let (Some(range), Some(reduction)) = (zero_padding, fft_reduction.as_mut()) {
            match direction {
                Direction::Forward => {
                    reduction.preprocess = reduction.preprocess.clone().with_zero_padding(range)?;
                }
                Direction::Inverse => {
                    reduction.postprocess =
                        reduction.postprocess.clone().with_zero_padding(range)?;
                }
            }
            reduction.validate()?;
        }
        if grouped_batch_override.is_some()
            && let Some(reduction) = fft_reduction.as_mut()
        {
            reduction.apply_axis0_wrapper_block(device)?;
        }
        let local_size = length.min(device.max_threads_per_block).max(1);
        let mut ir = Self {
            transform,
            effective_transform,
            direction,
            length,
            batch_count: plan.config.batch_count,
            grouped_batch,
            axis_batch_block: None,
            scalar,
            external_scalar,
            normalize,
            normalization_scale,
            zero_padding,
            workgroup_size: WorkgroupSize {
                x: u32::try_from(local_size).map_err(|_| VkFftError::ValueOutOfRange {
                    field: "DCT/DST workgroup size",
                })?,
                y: 1,
                z: 1,
            },
            dispatch: DispatchGeometry {
                x: u32::try_from(plan.config.batch_count.div_ceil(grouped_batch)).map_err(
                    |_| VkFftError::ValueOutOfRange {
                        field: "DCT/DST grouped dispatch count",
                    },
                )?,
                y: 1,
                z: 1,
            },
            fft_reduction,
        };
        if ir.fft_reduction.is_none()
            && grouped_batch_override.is_some()
            && let Some(block) =
                plan_axis0_r2r_wrapper_block(length, ir.batch_count, grouped_batch, device)?
        {
            ir.apply_direct_axis_batch_block(block, device)?;
        }
        ir.validate()?;
        Ok(ir)
    }

    fn apply_direct_axis_batch_block(
        &mut self,
        block: StockhamAxisBlockSchedule,
        device: DeviceProfile,
    ) -> Result<()> {
        if self.fft_reduction.is_some() {
            return Err(VkFftError::InvalidKernelIr(
                "R2R direct physical block cannot be attached to an FFT-backed reduction",
            ));
        }
        block.validate(self.batch_count, device)?;
        self.axis_batch_block = Some(block);
        self.workgroup_size = WorkgroupSize {
            x: u32::try_from(block.local_size_x).map_err(|_| VkFftError::ValueOutOfRange {
                field: "R2R direct local_size_x",
            })?,
            y: u32::try_from(block.local_size_y).map_err(|_| VkFftError::ValueOutOfRange {
                field: "R2R direct local_size_y",
            })?,
            z: 1,
        };
        self.dispatch.x =
            u32::try_from(self.batch_count.div_ceil(block.grouped_batch)).map_err(|_| {
                VkFftError::ValueOutOfRange {
                    field: "R2R direct physical dispatch count",
                }
            })?;
        self.validate()
    }

    pub(crate) fn with_other_axis_single_upload_block(
        mut self,
        fastest_axis_len: usize,
        device: DeviceProfile,
    ) -> Result<Self> {
        if let Some(reduction) = self.fft_reduction.as_mut() {
            reduction.fft = reduction
                .fft
                .clone()
                .with_other_axis_single_upload_block(fastest_axis_len, device)?;
            reduction.apply_other_axis_wrapper_block(fastest_axis_len, None, None, device)?;
        } else if let Some(block) = plan_gpu_other_axis_elementwise_wrapper_block(
            self.length,
            self.batch_count,
            fastest_axis_len,
            self.scalar.complex_bytes(),
            None,
            None,
            device,
        )? {
            self.apply_direct_axis_batch_block(block, device)?;
        }
        self.validate()?;
        Ok(self)
    }

    pub(crate) fn with_other_axis_single_upload_block_with_grouped_batch(
        mut self,
        fastest_axis_len: usize,
        grouped_batch_override: Option<usize>,
        axis1_grouped_batch_override: Option<usize>,
        device: DeviceProfile,
    ) -> Result<Self> {
        if let Some(reduction) = self.fft_reduction.as_mut() {
            reduction.fft = reduction
                .fft
                .clone()
                .with_other_axis_single_upload_block_with_grouped_batch(
                    fastest_axis_len,
                    grouped_batch_override,
                    axis1_grouped_batch_override,
                    device,
                )?;
            reduction.apply_other_axis_wrapper_block(
                fastest_axis_len,
                grouped_batch_override,
                axis1_grouped_batch_override,
                device,
            )?;
        } else if let Some(block) = plan_gpu_other_axis_elementwise_wrapper_block(
            self.length,
            self.batch_count,
            fastest_axis_len,
            self.scalar.complex_bytes(),
            grouped_batch_override,
            axis1_grouped_batch_override,
            device,
        )? {
            self.apply_direct_axis_batch_block(block, device)?;
        }
        self.validate()?;
        Ok(self)
    }

    pub fn validate(&self) -> Result<()> {
        if self.length == 0
            || self.batch_count == 0
            || self.grouped_batch == 0
            || self.workgroup_size.x == 0
            || self.workgroup_size.y == 0
            || self.workgroup_size.z != 1
            || self.dispatch.y != 1
            || self.dispatch.z != 1
            || !self.normalization_scale.is_finite()
            || self.normalization_scale <= 0.0
        {
            return Err(VkFftError::InvalidKernelIr(
                "DCT/DST launch or normalization metadata is inconsistent",
            ));
        }
        if let Some(block) = self.axis_batch_block {
            let expected = if block.transforms_on_x {
                [block.grouped_batch, block.threads_per_transform]
            } else {
                [block.threads_per_transform, block.grouped_batch]
            };
            if self.fft_reduction.is_some()
                || [
                    self.workgroup_size.x as usize,
                    self.workgroup_size.y as usize,
                ] != expected
                || self.dispatch.x as usize != self.batch_count.div_ceil(block.grouped_batch)
            {
                return Err(VkFftError::InvalidKernelIr(
                    "direct R2R physical ownership is inconsistent",
                ));
            }
        } else if self.workgroup_size.y != 1
            || self.dispatch.x as usize != self.batch_count.div_ceil(self.grouped_batch)
        {
            return Err(VkFftError::InvalidKernelIr(
                "R2R logical groupedBatch launch is inconsistent",
            ));
        }
        if self
            .zero_padding
            .is_some_and(|range| range.left > range.right || range.right > self.length)
        {
            return Err(VkFftError::InvalidKernelIr(
                "DCT/DST zero-padding interval is outside the logical transform",
            ));
        }
        if matches!(self.transform, R2rTransform::Dct(DctType::I)) && self.length < 2 {
            return Err(VkFftError::InvalidKernelIr(
                "DCT-I requires at least two logical elements",
            ));
        }
        if self.scalar == ScalarType::F16
            || (self.external_scalar != self.scalar
                && !supported_r2r_storage_pair(self.scalar, self.external_scalar))
        {
            return Err(VkFftError::InvalidKernelIr(
                "DCT/DST storage precision is incompatible with compute precision",
            ));
        }
        let expected_effective = if self.direction == Direction::Inverse {
            inverse_partner(self.transform)
        } else {
            self.transform
        };
        if self.effective_transform != expected_effective {
            return Err(VkFftError::InvalidKernelIr(
                "DCT/DST effective transform does not match direction",
            ));
        }
        let expects_reduction = matches!(
            self.effective_transform,
            R2rTransform::Dct(_) | R2rTransform::Dst(_)
        );
        match (expects_reduction, self.fft_reduction.as_deref()) {
            (true, Some(reduction)) => {
                reduction.validate()?;
                let expected_pre_zero = if self.direction == Direction::Forward {
                    self.zero_padding
                } else {
                    None
                };
                let expected_post_zero = if self.direction == Direction::Inverse {
                    self.zero_padding
                } else {
                    None
                };
                if reduction.external_scalar != self.external_scalar
                    || reduction.grouped_batch != self.grouped_batch
                    || reduction.preprocess.zero_padding != expected_pre_zero
                    || reduction.postprocess.zero_padding != expected_post_zero
                {
                    return Err(VkFftError::InvalidKernelIr(
                        "R2R reduction storage scalar does not match its wrapper",
                    ));
                }
            }
            (true, None) => {
                return Err(VkFftError::InvalidKernelIr(
                    "FFT-backed R2R IR is missing its reduction",
                ));
            }
            (false, Some(_)) => {
                return Err(VkFftError::InvalidKernelIr(
                    "R2R transform unexpectedly carries an FFT reduction",
                ));
            }
            (false, None) => {}
        }
        Ok(())
    }
}

pub fn execute_r2r_ir(ir: &R2rIr, input: &[f64]) -> Result<Vec<f64>> {
    ir.validate()?;
    let expected = ir
        .length
        .checked_mul(ir.batch_count)
        .ok_or(VkFftError::ArithmeticOverflow {
            operation: "DCT/DST input element count",
        })?;
    if input.len() != expected {
        return Err(VkFftError::InputLengthMismatch {
            expected,
            actual: input.len(),
        });
    }
    let masked_input = if ir.direction == Direction::Forward {
        ir.zero_padding.map(|range| {
            let mut values = input.to_vec();
            for batch in 0..ir.batch_count {
                let base = batch * ir.length;
                values[base + range.left..base + range.right].fill(0.0);
            }
            values
        })
    } else {
        None
    };
    let transform_input = masked_input.as_deref().unwrap_or(input);
    let mut output = if let Some(reduction) = ir.fft_reduction.as_deref() {
        execute_r2r_fft_reduction(reduction, transform_input)?
    } else {
        let mut output = vec![0.0; expected];
        for batch in 0..ir.batch_count {
            let base = batch * ir.length;
            for k in 0..ir.length {
                output[base + k] = evaluate_bin(
                    ir.effective_transform,
                    &transform_input[base..base + ir.length],
                    k,
                ) * ir.normalization_scale;
            }
        }
        output
    };
    if ir.direction == Direction::Inverse
        && let Some(range) = ir.zero_padding
    {
        for batch in 0..ir.batch_count {
            let base = batch * ir.length;
            output[base + range.left..base + range.right].fill(0.0);
        }
    }
    Ok(output)
}

fn execute_r2r_fft_reduction(reduction: &R2rFftReductionIr, input: &[f64]) -> Result<Vec<f64>> {
    reduction.validate()?;
    let n = reduction.length;
    let fft_n = reduction.fft_len;
    let mut fft_input = vec![Complex64::default(); fft_n * reduction.batch_count];
    match reduction.preprocess.operation {
        R2rFftPassOperation::DctIEvenExtension => {
            for batch in 0..reduction.batch_count {
                let source_base = batch * n;
                let fft_base = batch * fft_n;
                for j in 0..n {
                    fft_input[fft_base + j] = Complex64::new(input[source_base + j], 0.0);
                }
                for j in 1..n - 1 {
                    fft_input[fft_base + fft_n - j] = Complex64::new(input[source_base + j], 0.0);
                }
            }
        }
        R2rFftPassOperation::DstIOddExtension => {
            for batch in 0..reduction.batch_count {
                let source_base = batch * n;
                let fft_base = batch * fft_n;
                for j in 0..n {
                    let value = input[source_base + j];
                    let forward = j + 1;
                    fft_input[fft_base + forward] = Complex64::new(value, 0.0);
                    fft_input[fft_base + fft_n - forward] = Complex64::new(-value, 0.0);
                }
            }
        }
        R2rFftPassOperation::DctIvEvenPack | R2rFftPassOperation::DstIvEvenPack => {
            let is_dst = matches!(
                reduction.preprocess.operation,
                R2rFftPassOperation::DstIvEvenPack
            );
            for batch in 0..reduction.batch_count {
                let source_base = batch * n;
                let fft_base = batch * fft_n;
                for j in 0..fft_n {
                    let a = input[source_base + 2 * j];
                    let b = input[source_base + n - 1 - 2 * j];
                    let base = Complex64::new(a, if is_dst { b } else { -b });
                    let phase = Complex64::exp_i(PI * j as f64 / n as f64);
                    fft_input[fft_base + j] = base * phase;
                }
            }
        }
        R2rFftPassOperation::DctIvOddPhasePack | R2rFftPassOperation::DstIvOddPhasePack => {
            for batch in 0..reduction.batch_count {
                let source_base = batch * n;
                let fft_base = batch * fft_n;
                for j in 0..n {
                    let phase = Complex64::exp_i(-PI * j as f64 / (2 * n) as f64);
                    fft_input[fft_base + j] = Complex64::new(input[source_base + j], 0.0) * phase;
                }
            }
        }
        R2rFftPassOperation::DctIiReorder | R2rFftPassOperation::DstIiReorder => {
            let is_dst = matches!(
                reduction.preprocess.operation,
                R2rFftPassOperation::DstIiReorder
            );
            let split = n.div_ceil(2);
            for batch in 0..reduction.batch_count {
                let source_base = batch * n;
                let fft_base = batch * fft_n;
                for j in 0..n {
                    let source = if j < split { 2 * j } else { 2 * (n - j) - 1 };
                    let mut value = input[source_base + source];
                    if is_dst && !source.is_multiple_of(2) {
                        value = -value;
                    }
                    fft_input[fft_base + j] = Complex64::new(value, 0.0);
                }
            }
        }
        R2rFftPassOperation::DctIiiSpectrum | R2rFftPassOperation::DstIiiSpectrum => {
            let is_dst = matches!(
                reduction.preprocess.operation,
                R2rFftPassOperation::DstIiiSpectrum
            );
            for batch in 0..reduction.batch_count {
                let source_base = batch * n;
                let fft_base = batch * fft_n;
                for k in 0..n {
                    let (a, b) = if is_dst {
                        (
                            input[source_base + n - 1 - k],
                            if k == 0 {
                                0.0
                            } else {
                                input[source_base + k - 1]
                            },
                        )
                    } else {
                        (
                            input[source_base + k],
                            if k == 0 {
                                0.0
                            } else {
                                input[source_base + n - k]
                            },
                        )
                    };
                    let phase = Complex64::exp_i(PI * k as f64 / (2 * n) as f64);
                    fft_input[fft_base + k] = Complex64::new(a, -b) * phase;
                }
            }
        }
        _ => {
            return Err(VkFftError::InvalidKernelIr(
                "R2R FFT reduction has an invalid preprocess operation",
            ));
        }
    }
    let transformed = execute_one_dim_fft_ir(&reduction.fft, &fft_input)?;
    let mut output = vec![0.0; n * reduction.batch_count];
    match reduction.postprocess.operation {
        R2rFftPassOperation::DctIExtract { scale } => {
            for batch in 0..reduction.batch_count {
                let fft_base = batch * fft_n;
                let output_base = batch * n;
                for k in 0..n {
                    output[output_base + k] = transformed[fft_base + k].re * scale;
                }
            }
        }
        R2rFftPassOperation::DstIExtract { scale } => {
            for batch in 0..reduction.batch_count {
                let fft_base = batch * fft_n;
                let output_base = batch * n;
                for k in 0..n {
                    output[output_base + k] = -transformed[fft_base + k + 1].im * scale;
                }
            }
        }
        R2rFftPassOperation::DctIvEvenExtract { scale }
        | R2rFftPassOperation::DstIvEvenExtract { scale } => {
            let is_dst = matches!(
                reduction.postprocess.operation,
                R2rFftPassOperation::DstIvEvenExtract { .. }
            );
            for batch in 0..reduction.batch_count {
                let fft_base = batch * fft_n;
                let output_base = batch * n;
                for k in 0..n {
                    let value = if k.is_multiple_of(2) {
                        transformed[fft_base + k / 2]
                    } else {
                        transformed[fft_base + fft_n - k.div_ceil(2)].conj()
                    };
                    let phase = Complex64::exp_i(PI * (2 * k + 1) as f64 / (4 * n) as f64);
                    let rotated = value * phase;
                    let component = if is_dst { rotated.im } else { rotated.re };
                    output[output_base + k] = 2.0 * component * scale;
                }
            }
        }
        R2rFftPassOperation::DctIvOddPhaseExtract { scale }
        | R2rFftPassOperation::DstIvOddPhaseExtract { scale } => {
            let is_dst = matches!(
                reduction.postprocess.operation,
                R2rFftPassOperation::DstIvOddPhaseExtract { .. }
            );
            for batch in 0..reduction.batch_count {
                let fft_base = batch * fft_n;
                let output_base = batch * n;
                for k in 0..n {
                    let phase = Complex64::exp_i(-PI * (2 * k + 1) as f64 / (4 * n) as f64);
                    let rotated = transformed[fft_base + k] * phase;
                    let component = if is_dst { -rotated.im } else { rotated.re };
                    output[output_base + k] = 2.0 * component * scale;
                }
            }
        }
        R2rFftPassOperation::DctIiExtract { scale }
        | R2rFftPassOperation::DstIiExtract { scale } => {
            let is_dst = matches!(
                reduction.postprocess.operation,
                R2rFftPassOperation::DstIiExtract { .. }
            );
            for batch in 0..reduction.batch_count {
                let fft_base = batch * fft_n;
                let output_base = batch * n;
                for k in 0..n {
                    let q = if is_dst { n - 1 - k } else { k };
                    let phase = Complex64::exp_i(-PI * q as f64 / (2 * n) as f64);
                    let rotated = transformed[fft_base + q] * phase;
                    output[output_base + k] = 2.0 * rotated.re * scale;
                }
            }
        }
        R2rFftPassOperation::DctIiiUnreorder { scale }
        | R2rFftPassOperation::DstIiiUnreorder { scale } => {
            let is_dst = matches!(
                reduction.postprocess.operation,
                R2rFftPassOperation::DstIiiUnreorder { .. }
            );
            for batch in 0..reduction.batch_count {
                let fft_base = batch * fft_n;
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
                    output[output_base + k] = value * scale;
                }
            }
        }
        _ => {
            return Err(VkFftError::InvalidKernelIr(
                "R2R FFT reduction has an invalid postprocess operation",
            ));
        }
    }
    Ok(output)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum R2rNdPassOperation {
    PackAxis,
    ScatterAxis,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct R2rNdZeroPadding {
    pub dimensions: Vec<usize>,
    pub ranges: Vec<Option<ZeroPaddingRange>>,
    pub tensor_len: usize,
}

impl R2rNdZeroPadding {
    fn new(
        dimensions: &[usize],
        ranges: &[Option<ZeroPaddingRange>],
        tensor_len: usize,
    ) -> Result<Self> {
        if dimensions.is_empty()
            || dimensions.len() != ranges.len()
            || dimensions.contains(&0)
            || checked_product(dimensions, "multidimensional DCT/DST zero-pad tensor size")?
                != tensor_len
        {
            return Err(VkFftError::InvalidKernelIr(
                "multidimensional DCT/DST zero-padding shape is inconsistent",
            ));
        }
        for (axis, (length, range)) in dimensions.iter().zip(ranges).enumerate() {
            if let Some(range) = range
                && (range.left > range.right || range.right > *length)
            {
                return Err(VkFftError::InvalidZeroPaddingRange {
                    axis,
                    left: range.left,
                    right: range.right,
                    length: *length,
                });
            }
        }
        Ok(Self {
            dimensions: dimensions.to_vec(),
            ranges: ranges.to_vec(),
            tensor_len,
        })
    }

    pub fn contains_linear_index(&self, index: usize) -> bool {
        let mut local = index % self.tensor_len;
        for axis in (0..self.dimensions.len()).rev() {
            let length = self.dimensions[axis];
            let coordinate = local % length;
            local /= length;
            if self.ranges[axis].is_some_and(|range| range.contains(coordinate)) {
                return true;
            }
        }
        false
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct R2rNdPassIr {
    pub name: String,
    pub scalar: ScalarType,
    pub input_storage_scalar: ScalarType,
    pub output_storage_scalar: ScalarType,
    pub tensor_len: usize,
    pub axis: usize,
    pub axis_len: usize,
    pub inner_stride: usize,
    pub line_count: usize,
    pub batch_count: usize,
    pub grouped_batch: Option<usize>,
    /// Present only on the first forward pack or final inverse scatter.
    pub zero_padding: Option<R2rNdZeroPadding>,
    pub workgroup_size: WorkgroupSize,
    pub dispatch: DispatchGeometry,
    pub operation: R2rNdPassOperation,
}

impl R2rNdPassIr {
    #[allow(clippy::too_many_arguments)]
    fn new(
        name: String,
        scalar: ScalarType,
        tensor_len: usize,
        axis: usize,
        axis_len: usize,
        inner_stride: usize,
        line_count: usize,
        batch_count: usize,
        operation: R2rNdPassOperation,
        device: DeviceProfile,
    ) -> Result<Self> {
        if device.max_threads_per_block == 0 {
            return Err(VkFftError::ResourceLimitExceeded {
                resource: "threads per workgroup",
                required: 1,
                available: 0,
            });
        }
        let transform_count =
            line_count
                .checked_mul(batch_count)
                .ok_or(VkFftError::ArithmeticOverflow {
                    operation: "multidimensional DCT/DST transform count",
                })?;
        let ir = Self {
            name,
            scalar,
            input_storage_scalar: scalar,
            output_storage_scalar: scalar,
            tensor_len,
            axis,
            axis_len,
            inner_stride,
            line_count,
            batch_count,
            grouped_batch: None,
            zero_padding: None,
            workgroup_size: WorkgroupSize {
                x: u32::try_from(axis_len.min(device.max_threads_per_block).max(1)).map_err(
                    |_| VkFftError::ValueOutOfRange {
                        field: "multidimensional DCT/DST pack workgroup size",
                    },
                )?,
                y: 1,
                z: 1,
            },
            dispatch: DispatchGeometry {
                x: u32::try_from(transform_count).map_err(|_| VkFftError::ValueOutOfRange {
                    field: "multidimensional DCT/DST pack dispatch count",
                })?,
                y: 1,
                z: 1,
            },
            operation,
        };
        ir.validate()?;
        Ok(ir)
    }

    fn with_grouped_batch(mut self, grouped_batch: usize) -> Result<Self> {
        if grouped_batch == 0 {
            return Err(VkFftError::InvalidKernelIr(
                "multidimensional DCT/DST groupedBatch must be non-zero",
            ));
        }
        self.grouped_batch = Some(grouped_batch);
        self.dispatch.x =
            u32::try_from(self.batch_count.div_ceil(grouped_batch)).map_err(|_| {
                VkFftError::ValueOutOfRange {
                    field: "multidimensional DCT/DST grouped dispatch count",
                }
            })?;
        self.validate()?;
        Ok(self)
    }

    fn with_zero_padding(mut self, zero_padding: R2rNdZeroPadding) -> Result<Self> {
        if zero_padding.tensor_len != self.tensor_len {
            return Err(VkFftError::InvalidKernelIr(
                "multidimensional DCT/DST boundary zero-padding tensor size differs",
            ));
        }
        self.zero_padding = Some(zero_padding);
        self.validate()?;
        Ok(self)
    }

    fn with_external_input_storage(mut self, storage: ScalarType) -> Result<Self> {
        if storage != self.scalar
            && (!supported_r2r_storage_pair(self.scalar, storage)
                || self.operation != R2rNdPassOperation::PackAxis)
        {
            return Err(VkFftError::UnsupportedPrecision {
                backend: "multidimensional DCT/DST pack boundary",
                precision: "unsupported input storage scalar",
            });
        }
        self.input_storage_scalar = storage;
        self.validate()?;
        Ok(self)
    }

    fn with_external_output_storage(mut self, storage: ScalarType) -> Result<Self> {
        if storage != self.scalar
            && (!supported_r2r_storage_pair(self.scalar, storage)
                || self.operation != R2rNdPassOperation::ScatterAxis)
        {
            return Err(VkFftError::UnsupportedPrecision {
                backend: "multidimensional DCT/DST scatter boundary",
                precision: "unsupported output storage scalar",
            });
        }
        self.output_storage_scalar = storage;
        self.validate()?;
        Ok(self)
    }

    pub fn transform_count(&self) -> Result<usize> {
        self.line_count
            .checked_mul(self.batch_count)
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "multidimensional DCT/DST pass transform count",
            })
    }

    pub fn validate(&self) -> Result<()> {
        let axis_span =
            self.axis_len
                .checked_mul(self.inner_stride)
                .ok_or(VkFftError::ArithmeticOverflow {
                    operation: "multidimensional DCT/DST axis span",
                })?;
        let input_storage_ok = self.input_storage_scalar == self.scalar
            || (self.operation == R2rNdPassOperation::PackAxis
                && supported_r2r_storage_pair(self.scalar, self.input_storage_scalar));
        let output_storage_ok = self.output_storage_scalar == self.scalar
            || (self.operation == R2rNdPassOperation::ScatterAxis
                && supported_r2r_storage_pair(self.scalar, self.output_storage_scalar));
        if let Some(zero_padding) = &self.zero_padding {
            R2rNdZeroPadding::new(
                &zero_padding.dimensions,
                &zero_padding.ranges,
                zero_padding.tensor_len,
            )?;
            if zero_padding.tensor_len != self.tensor_len {
                return Err(VkFftError::InvalidKernelIr(
                    "multidimensional DCT/DST boundary zero-padding metadata is inconsistent",
                ));
            }
        }
        if self.scalar == ScalarType::F16
            || !input_storage_ok
            || !output_storage_ok
            || self.tensor_len == 0
            || self.axis_len == 0
            || self.inner_stride == 0
            || self.line_count == 0
            || self.batch_count == 0
            || self.grouped_batch == Some(0)
            || !self.tensor_len.is_multiple_of(self.axis_len)
            || self.line_count != self.tensor_len / self.axis_len
            || axis_span > self.tensor_len
            || self.workgroup_size.x == 0
            || self.dispatch.x as usize
                != if let Some(grouped_batch) = self.grouped_batch {
                    self.batch_count.div_ceil(grouped_batch)
                } else {
                    self.transform_count()?
                }
        {
            return Err(VkFftError::InvalidKernelIr(
                "multidimensional DCT/DST pass metadata is inconsistent",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct NdR2rAxisIr {
    pub axis: usize,
    pub pack: R2rNdPassIr,
    pub transform: R2rIr,
    pub scatter: R2rNdPassIr,
}

#[derive(Debug, Clone, PartialEq)]
pub struct NdR2rIr {
    pub dimensions: Vec<usize>,
    pub tensor_len: usize,
    pub batch_count: usize,
    pub direction: Direction,
    pub scalar: ScalarType,
    pub external_scalar: ScalarType,
    pub input_external_layout: NdExternalTensorLayout,
    pub output_external_layout: NdExternalTensorLayout,
    pub input_formatted_copy: Option<NdFormattedCopyPassIr>,
    pub output_formatted_copy: Option<NdFormattedCopyPassIr>,
    /// Tensor-level spatial zero-padding, applied once at the external boundary.
    pub zero_padding: Option<R2rNdZeroPadding>,
    pub omitted_axes: Vec<bool>,
    pub axes: Vec<NdR2rAxisIr>,
}

impl NdR2rIr {
    pub fn build(plan: &FftPlan, direction: Direction, device: DeviceProfile) -> Result<Self> {
        if !matches!(
            plan.config.transform,
            TransformKind::Dct(_) | TransformKind::Dst(_)
        ) {
            return Err(VkFftError::UnsupportedKernelPath(
                "multidimensional R2R IR requires a DCT or DST transform",
            ));
        }
        let tensor_len = checked_product(
            &plan.config.dimensions,
            "multidimensional DCT/DST tensor size",
        )?;
        let omitted_axes = (0..plan.config.dimensions.len())
            .map(|axis| plan.config.axis_is_omitted(axis))
            .collect::<Vec<_>>();
        let zero_padding = if plan.config.zero_padding.iter().any(Option::is_some) {
            Some(R2rNdZeroPadding::new(
                &plan.config.dimensions,
                &plan.config.zero_padding,
                tensor_len,
            )?)
        } else {
            None
        };
        let (scalar, external_scalar, child_precision) = match plan.config.precision {
            Precision::F16StorageF32Compute => (ScalarType::F32, ScalarType::F16, Precision::F32),
            Precision::F32 => (ScalarType::F32, ScalarType::F32, Precision::F32),
            Precision::F64 if device.supports_f64 => {
                (ScalarType::F64, ScalarType::F64, Precision::F64)
            }
            Precision::F64ComputeF32Storage if device.supports_f64 => {
                (ScalarType::F64, ScalarType::F32, Precision::F64)
            }
            other => {
                return Err(VkFftError::UnsupportedPrecision {
                    backend: "multidimensional DCT/DST IR",
                    precision: precision_name(other),
                });
            }
        };
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
        let input_formatted_copy = (!input_external_layout.is_tightly_packed()?)
            .then(|| {
                NdFormattedCopyPassIr::new(
                    "vkfft_nd_r2r_gather_formatted_input".to_owned(),
                    scalar,
                    external_scalar,
                    plan.config.batch_count,
                    input_external_layout.clone(),
                    NdFormattedCopyOperation::GatherExternalToDense,
                    device,
                )
            })
            .transpose()?;
        let output_formatted_copy = (!output_external_layout.is_tightly_packed()?)
            .then(|| {
                NdFormattedCopyPassIr::new(
                    "vkfft_nd_r2r_scatter_formatted_output".to_owned(),
                    scalar,
                    external_scalar,
                    plan.config.batch_count,
                    output_external_layout.clone(),
                    NdFormattedCopyOperation::ScatterDenseToExternal,
                    device,
                )
            })
            .transpose()?;
        let mut axes = Vec::with_capacity(plan.config.dimensions.len());
        let fastest_axis = plan.config.dimensions.len() - 1;
        let upstream_axis1_grouped_batch = plan
            .config
            .grouped_batch_for_axis(fastest_axis.saturating_sub(1));
        for axis in (0..plan.config.dimensions.len()).rev() {
            if omitted_axes[axis] {
                continue;
            }
            let axis_len = plan.config.dimensions[axis];
            let inner_stride = checked_product(
                &plan.config.dimensions[axis + 1..],
                "multidimensional DCT/DST inner stride",
            )?;
            let line_count = tensor_len / axis_len;
            let grouped_batch_override = plan.config.grouped_batch_for_axis(axis);
            let transform_batches = plan.config.batch_count.checked_mul(line_count).ok_or(
                VkFftError::ArithmeticOverflow {
                    operation: "multidimensional DCT/DST axis batch count",
                },
            )?;
            let mut axis_config = FftConfig::new(vec![axis_len])
                .with_batch_count(transform_batches)
                .with_precision(child_precision)
                .with_transform(plan.config.transform)
                .with_inverse_normalization(plan.config.normalize_inverse)
                .with_tuning(plan.config.tuning)
                .with_bandwidth_boost(plan.config.bandwidth_boost);
            if let Some(grouped_batch) = grouped_batch_override {
                axis_config = axis_config.with_grouped_batch(0, grouped_batch)?;
            }
            let axis_plan = FftPlan::build(axis_config)?;
            let axis_class = if inner_stride == 1 {
                C2cDeviceAxisClass::Contiguous
            } else {
                C2cDeviceAxisClass::Strided
            };
            let transform =
                R2rIr::build_with_axis_class(&axis_plan, direction, device, axis_class)?;
            let transform = if axis != fastest_axis {
                if grouped_batch_override.is_some() {
                    transform.with_other_axis_single_upload_block_with_grouped_batch(
                        plan.config.dimensions[fastest_axis],
                        grouped_batch_override,
                        upstream_axis1_grouped_batch,
                        device,
                    )?
                } else {
                    transform.with_other_axis_single_upload_block(
                        plan.config.dimensions[fastest_axis],
                        device,
                    )?
                }
            } else {
                transform
            };
            let label = match direction {
                Direction::Forward => "forward",
                Direction::Inverse => "inverse",
            };
            let mut pack = R2rNdPassIr::new(
                format!("vkfft_nd_r2r_pack_axis_{axis}_{label}"),
                transform.scalar,
                tensor_len,
                axis,
                axis_len,
                inner_stride,
                line_count,
                plan.config.batch_count,
                R2rNdPassOperation::PackAxis,
                device,
            )?;
            let mut scatter = R2rNdPassIr::new(
                format!("vkfft_nd_r2r_scatter_axis_{axis}_{label}"),
                transform.scalar,
                tensor_len,
                axis,
                axis_len,
                inner_stride,
                line_count,
                plan.config.batch_count,
                R2rNdPassOperation::ScatterAxis,
                device,
            )?;
            if let Some(grouped_batch) = grouped_batch_override {
                pack = pack.with_grouped_batch(grouped_batch)?;
                scatter = scatter.with_grouped_batch(grouped_batch)?;
            }
            axes.push(NdR2rAxisIr {
                axis,
                pack,
                transform,
                scatter,
            });
        }
        if axes.is_empty() {
            return Err(VkFftError::InvalidKernelIr(
                "multidimensional DCT/DST has no axes",
            ));
        }
        let last_index = axes.len() - 1;
        if let Some(zero_padding) = zero_padding.clone() {
            match direction {
                Direction::Forward => {
                    axes[0].pack = axes[0].pack.clone().with_zero_padding(zero_padding)?;
                }
                Direction::Inverse => {
                    axes[last_index].scatter = axes[last_index]
                        .scatter
                        .clone()
                        .with_zero_padding(zero_padding)?;
                }
            }
        }
        if external_scalar != scalar {
            if input_formatted_copy.is_none() {
                axes[0].pack = axes[0]
                    .pack
                    .clone()
                    .with_external_input_storage(external_scalar)?;
            }
            if output_formatted_copy.is_none() {
                axes[last_index].scatter = axes[last_index]
                    .scatter
                    .clone()
                    .with_external_output_storage(external_scalar)?;
            }
        }
        let ir = Self {
            dimensions: plan.config.dimensions.clone(),
            tensor_len,
            batch_count: plan.config.batch_count,
            direction,
            scalar,
            external_scalar,
            input_external_layout,
            output_external_layout,
            input_formatted_copy,
            output_formatted_copy,
            zero_padding,
            omitted_axes,
            axes,
        };
        ir.validate()?;
        Ok(ir)
    }

    pub fn validate(&self) -> Result<()> {
        if self.omitted_axes.len() != self.dimensions.len()
            || self.axes.len() != self.omitted_axes.iter().filter(|&&omit| !omit).count()
            || checked_product(&self.dimensions, "multidimensional DCT/DST validation size")?
                != self.tensor_len
        {
            return Err(VkFftError::InvalidKernelIr(
                "multidimensional DCT/DST tensor metadata is inconsistent",
            ));
        }
        if self.external_scalar != self.scalar
            && !supported_r2r_storage_pair(self.scalar, self.external_scalar)
        {
            return Err(VkFftError::InvalidKernelIr(
                "multidimensional DCT/DST storage scalar is incompatible with compute precision",
            ));
        }
        self.input_external_layout.validate(self.tensor_len)?;
        self.output_external_layout.validate(self.tensor_len)?;
        if self.input_external_layout.dimensions != self.dimensions
            || self.output_external_layout.dimensions != self.dimensions
        {
            return Err(VkFftError::InvalidKernelIr(
                "multidimensional DCT/DST formatted tensor dimensions are inconsistent",
            ));
        }
        let input_needs_copy = !self.input_external_layout.is_tightly_packed()?;
        let output_needs_copy = !self.output_external_layout.is_tightly_packed()?;
        if self.input_formatted_copy.is_some() != input_needs_copy
            || self.output_formatted_copy.is_some() != output_needs_copy
        {
            return Err(VkFftError::InvalidKernelIr(
                "multidimensional DCT/DST formatted copy ownership is inconsistent",
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
                    "multidimensional DCT/DST formatted input copy metadata is inconsistent",
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
                    "multidimensional DCT/DST formatted output copy metadata is inconsistent",
                ));
            }
        }
        if let Some(zero_padding) = &self.zero_padding {
            R2rNdZeroPadding::new(
                &zero_padding.dimensions,
                &zero_padding.ranges,
                zero_padding.tensor_len,
            )?;
            if zero_padding.dimensions != self.dimensions
                || zero_padding.tensor_len != self.tensor_len
            {
                return Err(VkFftError::InvalidKernelIr(
                    "multidimensional DCT/DST zero-padding metadata differs from the tensor",
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
                "multidimensional DCT/DST omitted-axis execution order is inconsistent",
            ));
        }
        for (axis_index, axis) in self.axes.iter().enumerate() {
            axis.pack.validate()?;
            axis.transform.validate()?;
            axis.scatter.validate()?;
            if self.omitted_axes[axis.axis] {
                return Err(VkFftError::InvalidKernelIr(
                    "multidimensional DCT/DST materialized an omitted axis",
                ));
            }
            let axis_len = self.dimensions[axis.axis];
            let expected_batches = self
                .batch_count
                .checked_mul(self.tensor_len / axis_len)
                .ok_or(VkFftError::ArithmeticOverflow {
                    operation: "multidimensional DCT/DST validation batch count",
                })?;
            let expected_pack_zero = if self.direction == Direction::Forward && axis_index == 0 {
                self.zero_padding.as_ref()
            } else {
                None
            };
            let expected_scatter_zero =
                if self.direction == Direction::Inverse && axis_index + 1 == self.axes.len() {
                    self.zero_padding.as_ref()
                } else {
                    None
                };
            if axis.transform.length != axis_len
                || axis.transform.batch_count != expected_batches
                || axis.transform.direction != self.direction
                || axis.transform.scalar != self.scalar
                || axis.transform.external_scalar != self.scalar
                || axis.pack.grouped_batch != axis.scatter.grouped_batch
                || axis
                    .pack
                    .grouped_batch
                    .is_some_and(|grouped_batch| axis.transform.grouped_batch != grouped_batch)
                || axis.pack.operation != R2rNdPassOperation::PackAxis
                || axis.scatter.operation != R2rNdPassOperation::ScatterAxis
                || axis.pack.zero_padding.as_ref() != expected_pack_zero
                || axis.scatter.zero_padding.as_ref() != expected_scatter_zero
                || axis.pack.input_storage_scalar
                    != if axis_index == 0 && self.input_formatted_copy.is_none() {
                        self.external_scalar
                    } else {
                        self.scalar
                    }
                || axis.pack.output_storage_scalar != self.scalar
                || axis.scatter.input_storage_scalar != self.scalar
                || axis.scatter.output_storage_scalar
                    != if axis_index + 1 == self.axes.len() && self.output_formatted_copy.is_none()
                    {
                        self.external_scalar
                    } else {
                        self.scalar
                    }
            {
                return Err(VkFftError::InvalidKernelIr(
                    "multidimensional DCT/DST axis metadata is inconsistent",
                ));
            }
        }
        Ok(())
    }
}

impl NdR2rIr {
    pub(crate) fn pack_formatted_input<T: Copy + Default>(&self, input: &[T]) -> Result<Vec<T>> {
        self.validate()?;
        let expected = self.tensor_len.checked_mul(self.batch_count).ok_or(
            VkFftError::ArithmeticOverflow {
                operation: "formatted ND R2R logical input element count",
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
}

pub fn execute_nd_r2r_ir(ir: &NdR2rIr, input: &[f64]) -> Result<Vec<f64>> {
    ir.validate()?;
    let expected =
        ir.tensor_len
            .checked_mul(ir.batch_count)
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "multidimensional DCT/DST input element count",
            })?;
    if input.len() != expected {
        return Err(VkFftError::InputLengthMismatch {
            expected,
            actual: input.len(),
        });
    }
    let mut current = if ir.input_formatted_copy.is_some() {
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
    if ir.direction == Direction::Forward
        && let Some(zero_padding) = &ir.zero_padding
    {
        for (index, value) in current.iter_mut().enumerate() {
            if zero_padding.contains_linear_index(index) {
                *value = 0.0;
            }
        }
    }
    for axis in &ir.axes {
        let pass = &axis.pack;
        let transform_count = pass.transform_count()?;
        let mut packed = vec![0.0; expected];
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
        let transformed = execute_r2r_ir(&axis.transform, &packed)?;
        let mut next = vec![0.0; expected];
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
    if ir.direction == Direction::Inverse
        && let Some(zero_padding) = &ir.zero_padding
    {
        for (index, value) in current.iter_mut().enumerate() {
            if zero_padding.contains_linear_index(index) {
                *value = 0.0;
            }
        }
    }
    if ir.output_formatted_copy.is_some() {
        let physical = pack_logical_tensor_batches(
            &current,
            &ir.output_external_layout.dimensions,
            &ir.output_external_layout.axis_strides,
            ir.output_external_layout.batch_stride,
        )?;
        ir.unpack_formatted_output(&physical)
    } else {
        Ok(current)
    }
}

fn checked_product(values: &[usize], operation: &'static str) -> Result<usize> {
    values.iter().try_fold(1usize, |product, value| {
        product
            .checked_mul(*value)
            .ok_or(VkFftError::ArithmeticOverflow { operation })
    })
}

pub(crate) fn inverse_partner(transform: R2rTransform) -> R2rTransform {
    match transform {
        R2rTransform::Dct(DctType::II) => R2rTransform::Dct(DctType::III),
        R2rTransform::Dct(DctType::III) => R2rTransform::Dct(DctType::II),
        R2rTransform::Dst(DstType::II) => R2rTransform::Dst(DstType::III),
        R2rTransform::Dst(DstType::III) => R2rTransform::Dst(DstType::II),
        other => other,
    }
}

pub(crate) fn inverse_pair_scale(transform: R2rTransform, length: usize) -> Result<f64> {
    let denominator = match transform {
        R2rTransform::Dct(DctType::I) => length
            .checked_sub(1)
            .and_then(|value| value.checked_mul(2))
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "DCT-I inverse normalization denominator",
            })?,
        R2rTransform::Dst(DstType::I) => length
            .checked_add(1)
            .and_then(|value| value.checked_mul(2))
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "DST-I inverse normalization denominator",
            })?,
        R2rTransform::Dct(DctType::II | DctType::III | DctType::IV)
        | R2rTransform::Dst(DstType::II | DstType::III | DstType::IV) => length
            .checked_mul(2)
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "DCT/DST inverse normalization denominator",
            })?,
    };
    Ok(1.0 / denominator as f64)
}

fn evaluate_bin(transform: R2rTransform, input: &[f64], k: usize) -> f64 {
    let n = input.len();
    match transform {
        // FFTW REDFT00.
        R2rTransform::Dct(DctType::I) => {
            let mut sum = input[0]
                + if k.is_multiple_of(2) {
                    input[n - 1]
                } else {
                    -input[n - 1]
                };
            for (j, value) in input.iter().enumerate().take(n - 1).skip(1) {
                sum += 2.0 * *value * (PI * j as f64 * k as f64 / (n - 1) as f64).cos();
            }
            sum
        }
        // FFTW REDFT10.
        R2rTransform::Dct(DctType::II) => input
            .iter()
            .enumerate()
            .map(|(j, value)| 2.0 * *value * (PI * (j as f64 + 0.5) * k as f64 / n as f64).cos())
            .sum(),
        // FFTW REDFT01.
        R2rTransform::Dct(DctType::III) => {
            input[0]
                + input
                    .iter()
                    .enumerate()
                    .skip(1)
                    .map(|(j, value)| {
                        2.0 * *value * (PI * j as f64 * (k as f64 + 0.5) / n as f64).cos()
                    })
                    .sum::<f64>()
        }
        // FFTW REDFT11.
        R2rTransform::Dct(DctType::IV) => input
            .iter()
            .enumerate()
            .map(|(j, value)| {
                2.0 * *value * (PI * (j as f64 + 0.5) * (k as f64 + 0.5) / n as f64).cos()
            })
            .sum(),
        // FFTW RODFT00.
        R2rTransform::Dst(DstType::I) => input
            .iter()
            .enumerate()
            .map(|(j, value)| {
                2.0 * *value * (PI * (j + 1) as f64 * (k + 1) as f64 / (n + 1) as f64).sin()
            })
            .sum(),
        // FFTW RODFT10.
        R2rTransform::Dst(DstType::II) => input
            .iter()
            .enumerate()
            .map(|(j, value)| {
                2.0 * *value * (PI * (j as f64 + 0.5) * (k + 1) as f64 / n as f64).sin()
            })
            .sum(),
        // FFTW RODFT01.
        R2rTransform::Dst(DstType::III) => {
            let tail = if k.is_multiple_of(2) {
                input[n - 1]
            } else {
                -input[n - 1]
            };
            tail + input
                .iter()
                .enumerate()
                .take(n - 1)
                .map(|(j, value)| {
                    2.0 * *value * (PI * (j + 1) as f64 * (k as f64 + 0.5) / n as f64).sin()
                })
                .sum::<f64>()
        }
        // FFTW RODFT11.
        R2rTransform::Dst(DstType::IV) => input
            .iter()
            .enumerate()
            .map(|(j, value)| {
                2.0 * *value * (PI * (j as f64 + 0.5) * (k as f64 + 0.5) / n as f64).sin()
            })
            .sum(),
    }
}

fn supported_r2r_storage_pair(compute: ScalarType, storage: ScalarType) -> bool {
    compute == storage
        || matches!(
            (compute, storage),
            (ScalarType::F32, ScalarType::F16) | (ScalarType::F64, ScalarType::F32)
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
    use crate::config::{Backend, FftConfig, GpuVendor};

    fn device() -> DeviceProfile {
        DeviceProfile {
            shared_memory_bytes: 128 * 1024,
            shared_memory_pow2_bytes: 128 * 1024,
            ..DeviceProfile::generic(Backend::Vulkan, GpuVendor::Nvidia)
        }
    }

    fn all_transforms() -> [TransformKind; 8] {
        [
            TransformKind::Dct(DctType::I),
            TransformKind::Dct(DctType::II),
            TransformKind::Dct(DctType::III),
            TransformKind::Dct(DctType::IV),
            TransformKind::Dst(DstType::I),
            TransformKind::Dst(DstType::II),
            TransformKind::Dst(DstType::III),
            TransformKind::Dst(DstType::IV),
        ]
    }

    #[test]
    fn omit_dimension_executes_only_selected_nd_r2r_axis() {
        let dimensions = vec![3usize, 4];
        let input = (0..12)
            .map(|index| {
                let x = index as f64;
                (0.17 * x).cos() - 0.03 * x
            })
            .collect::<Vec<_>>();
        for omitted_axis in [0usize, 1] {
            let plan = FftPlan::build(
                FftConfig::new(dimensions.clone())
                    .with_transform(TransformKind::Dct(DctType::II))
                    .with_omit_dimension(omitted_axis, true)
                    .unwrap(),
            )
            .unwrap();
            let ir = NdR2rIr::build(&plan, Direction::Forward, device()).unwrap();
            let active_axis = 1 - omitted_axis;
            assert_eq!(ir.omitted_axes, vec![omitted_axis == 0, omitted_axis == 1]);
            assert_eq!(ir.axes.len(), 1);
            assert_eq!(ir.axes[0].axis, active_axis);
            assert_eq!(
                ir.axes[0].pack.inner_stride,
                if active_axis == 0 { 4 } else { 1 }
            );

            let mut packed = Vec::with_capacity(input.len());
            if active_axis == 1 {
                packed.extend_from_slice(&input);
            } else {
                for column in 0..4 {
                    for row in 0..3 {
                        packed.push(input[row * 4 + column]);
                    }
                }
            }
            let transformed = execute_r2r_ir(&ir.axes[0].transform, &packed).unwrap();
            let mut expected = vec![0.0; input.len()];
            if active_axis == 1 {
                expected.copy_from_slice(&transformed);
            } else {
                for column in 0..4 {
                    for row in 0..3 {
                        expected[row * 4 + column] = transformed[column * 3 + row];
                    }
                }
            }
            let actual = execute_nd_r2r_ir(&ir, &input).unwrap();
            let max_error = actual
                .iter()
                .zip(&expected)
                .map(|(actual, expected)| (actual - expected).abs())
                .fold(0.0, f64::max);
            assert!(
                max_error <= 2.0e-10,
                "omit axis {omitted_axis} ND-R2R mismatch {max_error:e}"
            );
        }
    }

    #[test]
    fn formatted_nd_r2r_strides_wrap_dense_cpu_math_in_2d_and_3d() {
        let run_case = |dimensions: Vec<usize>,
                        input_strides: &[(usize, usize)],
                        output_strides: &[(usize, usize)],
                        expected_input_batch: usize,
                        expected_output_batch: usize| {
            let transform = TransformKind::Dct(DctType::II);
            let mut formatted_config = FftConfig::new(dimensions.clone())
                .with_batch_count(2)
                .with_transform(transform);
            for &(axis, stride) in input_strides {
                formatted_config = formatted_config
                    .with_input_buffer_axis_stride(axis, stride)
                    .unwrap();
            }
            for &(axis, stride) in output_strides {
                formatted_config = formatted_config
                    .with_output_buffer_axis_stride(axis, stride)
                    .unwrap();
            }
            let dense_config = FftConfig::new(dimensions.clone())
                .with_batch_count(2)
                .with_transform(transform);
            let formatted_plan = FftPlan::build(formatted_config.clone()).unwrap();
            let dense_plan = FftPlan::build(dense_config.clone()).unwrap();
            let formatted = NdR2rIr::build(&formatted_plan, Direction::Forward, device()).unwrap();
            let dense = NdR2rIr::build(&dense_plan, Direction::Forward, device()).unwrap();
            assert_eq!(
                formatted.input_external_layout.batch_stride,
                expected_input_batch
            );
            assert_eq!(
                formatted.output_external_layout.batch_stride,
                expected_output_batch
            );
            assert!(formatted.input_formatted_copy.is_some());
            assert!(formatted.output_formatted_copy.is_some());

            let tensor_len = dimensions.iter().product::<usize>();
            let input = (0..2 * tensor_len)
                .map(|index| {
                    let x = index as f64;
                    (0.13 * x).sin() + 0.07 * (0.19 * x).cos() - 0.003 * x
                })
                .collect::<Vec<_>>();
            let physical_input = formatted.pack_formatted_input(&input).unwrap();
            assert_eq!(physical_input.len(), 2 * expected_input_batch);
            let actual = execute_nd_r2r_ir(&formatted, &input).unwrap();
            let expected = execute_nd_r2r_ir(&dense, &input).unwrap();
            let max_error = actual
                .iter()
                .zip(&expected)
                .map(|(actual, expected)| (actual - expected).abs())
                .fold(0.0, f64::max);
            assert!(
                max_error <= 2.0e-10,
                "formatted ND-R2R forward mismatch {max_error:e}"
            );

            let formatted_inverse_plan =
                FftPlan::build(formatted_config.with_inverse_normalization(true)).unwrap();
            let dense_inverse_plan =
                FftPlan::build(dense_config.with_inverse_normalization(true)).unwrap();
            let formatted_inverse =
                NdR2rIr::build(&formatted_inverse_plan, Direction::Inverse, device()).unwrap();
            let dense_inverse =
                NdR2rIr::build(&dense_inverse_plan, Direction::Inverse, device()).unwrap();
            let actual_inverse = execute_nd_r2r_ir(&formatted_inverse, &actual).unwrap();
            let expected_inverse = execute_nd_r2r_ir(&dense_inverse, &actual).unwrap();
            let max_inverse_error = actual_inverse
                .iter()
                .zip(&expected_inverse)
                .map(|(actual, expected)| (actual - expected).abs())
                .fold(0.0, f64::max);
            assert!(
                max_inverse_error <= 2.0e-10,
                "formatted ND-R2R inverse mismatch {max_inverse_error:e}"
            );
        };

        run_case(vec![3, 4], &[(0, 7)], &[(0, 9)], 21, 27);
        run_case(
            vec![2, 3, 4],
            &[(1, 6), (0, 20)],
            &[(1, 7), (0, 24)],
            40,
            48,
        );
    }

    #[test]
    fn dct_fft_children_preserve_contiguous_and_strided_device_scoring() {
        let profile = DeviceProfile::generic(Backend::Vulkan, GpuVendor::Nvidia);
        let transform = TransformKind::Dct(DctType::II);

        // DCT-II keeps an N-point C2C child, matching the pinned upstream
        // scheduler. N=1153 fits the contiguous prime budget, so the
        // one-dimensional transform retains Rader.
        let one_dim_plan =
            FftPlan::build(FftConfig::new(vec![1153]).with_transform(transform)).unwrap();
        let one_dim = R2rIr::build(&one_dim_plan, Direction::Forward, profile).unwrap();
        let reduction = one_dim
            .fft_reduction
            .as_deref()
            .expect("DCT-II should use the FFT reduction");
        assert!(matches!(reduction.fft, OneDimFftIr::Recursive(_)));

        // The same logical DCT axis extracted from the outer dimension of a
        // row-major ND tensor is strided. Its N1153 child sees the 1024
        // coalesced prime limit, excluding p1153 and selecting Bluestein.
        let nd_plan =
            FftPlan::build(FftConfig::new(vec![1153usize, 4]).with_transform(transform)).unwrap();
        let nd = NdR2rIr::build(&nd_plan, Direction::Forward, profile).unwrap();
        let outer = nd
            .axes
            .iter()
            .find(|axis| axis.axis == 0)
            .expect("ND DCT-II outer axis");
        let outer_reduction = outer
            .transform
            .fft_reduction
            .as_deref()
            .expect("ND DCT-II should use the FFT reduction");
        assert!(matches!(outer_reduction.fft, OneDimFftIr::Bluestein(_)));
    }

    #[test]
    fn nd_r2r_bandwidth_boost_reaches_outer_fft_reduction() {
        let mut profile = device();
        profile.shared_memory_bytes = 48 * 1024;
        profile.shared_memory_pow2_bytes = 32 * 1024;
        profile.max_threads_per_block = 1024;
        profile.max_workgroup_size = [1024, 1024, 64];
        // DCT-I reduces through a 2*(N-1) C2C child. N=1,048,577 therefore
        // exercises the same 2,097,152-point strided scheduler shape as ND C2C.
        let transform = TransformKind::Dct(DctType::I);
        let dimensions = vec![1_048_577usize, 2usize];

        let baseline_plan =
            FftPlan::build(FftConfig::new(dimensions.clone()).with_transform(transform)).unwrap();
        let baseline = NdR2rIr::build(&baseline_plan, Direction::Forward, profile).unwrap();
        let baseline_outer = baseline.axes.iter().find(|axis| axis.axis == 0).unwrap();
        let baseline_reduction = baseline_outer.transform.fft_reduction.as_deref().unwrap();
        let OneDimFftIr::Recursive(baseline_recursive) = &baseline_reduction.fft else {
            panic!("baseline ND DCT-I outer reduction should remain recursive Stockham");
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
                .with_transform(transform)
                .with_bandwidth_boost(2),
        )
        .unwrap();
        let boosted = NdR2rIr::build(&boosted_plan, Direction::Forward, profile).unwrap();
        let boosted_outer = boosted.axes.iter().find(|axis| axis.axis == 0).unwrap();
        let boosted_reduction = boosted_outer.transform.fft_reduction.as_deref().unwrap();
        let OneDimFftIr::Recursive(boosted_recursive) = &boosted_reduction.fft else {
            panic!("boosted ND DCT-I outer reduction should remain recursive Stockham");
        };
        let schedule = boosted_recursive.stockham_upload_schedule.as_ref().unwrap();
        assert_eq!(schedule.upload_count, 2);
        assert_eq!(schedule.axis_split, vec![2_048, 1_024]);
        assert!(boosted_recursive.four_step_plan.is_some());
        boosted.validate().unwrap();
    }

    #[test]
    fn nd_r2r_higher_axis_auto_groups_bluestein_reduction_without_logical_group_override() {
        let mut profile = device();
        profile.shared_memory_bytes = 48 * 1024;
        profile.shared_memory_pow2_bytes = 32 * 1024;
        profile.max_threads_per_block = 1024;
        profile.max_workgroup_size = [1024, 1024, 64];
        let mut tuning = crate::PlannerTuning::portable();
        tuning.max_rader_fft_prime = 100;

        let plan = FftPlan::build(
            FftConfig::new(vec![103usize, 8])
                .with_transform(TransformKind::Dct(DctType::II))
                .with_tuning(tuning),
        )
        .unwrap();
        let ir = NdR2rIr::build(&plan, Direction::Forward, profile).unwrap();
        let outer = ir.axes.iter().find(|axis| axis.axis == 0).unwrap();
        assert_eq!(outer.transform.grouped_batch, 1);
        let reduction = outer
            .transform
            .fft_reduction
            .as_deref()
            .expect("outer DCT-II should use an FFT reduction");
        assert_eq!(reduction.batch_count, 8);
        assert_eq!(reduction.grouped_batch, 1);
        let r2r_wrapper = reduction
            .preprocess
            .axis_batch_block
            .expect("higher-axis R2R preprocess should be physically grouped");
        assert_eq!(r2r_wrapper.grouped_batch, 4);
        assert_eq!(r2r_wrapper.threads_per_transform, 103);
        assert_eq!(
            [r2r_wrapper.local_size_x, r2r_wrapper.local_size_y],
            [4, 103]
        );
        assert!(r2r_wrapper.transforms_on_x);
        assert_eq!(reduction.preprocess.dispatch.x, 2);
        assert_eq!(reduction.postprocess.axis_batch_block, Some(r2r_wrapper));
        let OneDimFftIr::Bluestein(pipeline) = &reduction.fft else {
            panic!("forced 103-point strided DCT-II reduction should use Bluestein");
        };
        assert_eq!(pipeline.grouped_batch, 1);
        let wrapper = pipeline
            .preprocess
            .axis_batch_block
            .expect("higher-axis R2R Bluestein wrapper should be physically grouped");
        assert_eq!(wrapper.grouped_batch, 4);
        assert_eq!(wrapper.threads_per_transform, 128);
        assert_eq!([wrapper.local_size_x, wrapper.local_size_y], [4, 128]);
        assert!(wrapper.transforms_on_x);
        assert_eq!(pipeline.preprocess.dispatch.x, 2);
        assert_eq!(pipeline.multiply.axis_batch_block, Some(wrapper));
        assert_eq!(pipeline.postprocess.axis_batch_block, Some(wrapper));

        let shaders = crate::backend::vulkan::VulkanGlslBackend
            .lower_nd_r2r(&ir)
            .unwrap();
        assert!(shaders.iter().any(|shader| {
            shader.workgroup_size.x == 4
                && shader.workgroup_size.y == 128
                && shader.glsl.contains("vkfft_transform_slot")
        }));
        assert!(shaders.iter().any(|shader| {
            shader.workgroup_size.x == 4
                && shader.workgroup_size.y == 103
                && shader.dispatch.x == 2
                && shader.glsl.contains("vkfft_transform_slot")
                && shader
                    .glsl
                    .contains("uint vkfft_lid = gl_LocalInvocationID.y")
        }));
        for shader in shaders {
            assert_eq!(shader.compile_spirv().unwrap().words[0], 0x0723_0203);
        }
        ir.validate().unwrap();
    }

    #[test]
    fn formatted_nd_r2r_strides_compose_with_spatial_zero_padding() {
        let dimensions = vec![3usize, 4usize];
        let tensor_len = dimensions.iter().product::<usize>();
        let batch_count = 2usize;
        let transform = TransformKind::Dct(DctType::II);
        let input = (0..tensor_len * batch_count)
            .map(|index| {
                let x = index as f64;
                (0.13 * x).sin() + 0.23 * (0.079 * x).cos() + 0.001 * x
            })
            .collect::<Vec<_>>();
        let mut manual_zero = input.clone();
        let zero = R2rNdZeroPadding::new(
            &dimensions,
            &[
                Some(ZeroPaddingRange::new(1, 3)),
                Some(ZeroPaddingRange::new(2, 4)),
            ],
            tensor_len,
        )
        .unwrap();
        for batch in 0..batch_count {
            let base = batch * tensor_len;
            for linear in 0..tensor_len {
                if zero.contains_linear_index(linear) {
                    manual_zero[base + linear] = 0.0;
                }
            }
        }

        let formatted_config = FftConfig::new(dimensions.clone())
            .with_batch_count(batch_count)
            .with_transform(transform)
            .with_input_buffer_axis_stride(0, 7)
            .unwrap()
            .with_output_buffer_axis_stride(0, 9)
            .unwrap()
            .with_zero_padding(0, 1, 3)
            .unwrap()
            .with_zero_padding(1, 2, 4)
            .unwrap();
        let forward = NdR2rIr::build(
            &FftPlan::build(formatted_config.clone()).unwrap(),
            Direction::Forward,
            device(),
        )
        .unwrap();
        assert!(forward.input_formatted_copy.is_some());
        assert!(forward.output_formatted_copy.is_some());
        assert_eq!(
            forward.axes[0].pack.zero_padding.as_ref(),
            forward.zero_padding.as_ref()
        );
        assert_eq!(forward.axes[0].pack.input_storage_scalar, forward.scalar);
        assert_eq!(forward.axes[0].pack.output_storage_scalar, forward.scalar);
        let baseline = NdR2rIr::build(
            &FftPlan::build(
                FftConfig::new(dimensions.clone())
                    .with_batch_count(batch_count)
                    .with_transform(transform),
            )
            .unwrap(),
            Direction::Forward,
            device(),
        )
        .unwrap();
        let actual = execute_nd_r2r_ir(&forward, &input).unwrap();
        let expected = execute_nd_r2r_ir(&baseline, &manual_zero).unwrap();
        let max_forward_error = actual
            .iter()
            .zip(&expected)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0, f64::max);
        assert!(max_forward_error <= 5.0e-10 * tensor_len as f64);

        let inverse = NdR2rIr::build(
            &FftPlan::build(formatted_config.with_inverse_normalization(true)).unwrap(),
            Direction::Inverse,
            device(),
        )
        .unwrap();
        let last = inverse.axes.len() - 1;
        assert_eq!(
            inverse.axes[last].scatter.zero_padding.as_ref(),
            inverse.zero_padding.as_ref()
        );
        assert_eq!(
            inverse.axes[last].scatter.input_storage_scalar,
            inverse.scalar
        );
        assert_eq!(
            inverse.axes[last].scatter.output_storage_scalar,
            inverse.scalar
        );
        let baseline_inverse = NdR2rIr::build(
            &FftPlan::build(
                FftConfig::new(dimensions.clone())
                    .with_batch_count(batch_count)
                    .with_transform(transform)
                    .with_inverse_normalization(true),
            )
            .unwrap(),
            Direction::Inverse,
            device(),
        )
        .unwrap();
        let mut expected_inverse = execute_nd_r2r_ir(&baseline_inverse, &actual).unwrap();
        for batch in 0..batch_count {
            let base = batch * tensor_len;
            for linear in 0..tensor_len {
                if zero.contains_linear_index(linear) {
                    expected_inverse[base + linear] = 0.0;
                }
            }
        }
        let actual_inverse = execute_nd_r2r_ir(&inverse, &actual).unwrap();
        let max_inverse_error = actual_inverse
            .iter()
            .zip(&expected_inverse)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0, f64::max);
        assert!(max_inverse_error <= 5.0e-10 * tensor_len as f64);

        let program = crate::ProgramIr::nd_r2r(&forward).unwrap();
        assert_eq!(
            program.passes.first().unwrap().name,
            "vkfft_nd_r2r_gather_formatted_input"
        );
        assert_eq!(
            program.passes.last().unwrap().name,
            "vkfft_nd_r2r_scatter_formatted_output"
        );
        let shaders = crate::backend::vulkan::VulkanGlslBackend
            .lower_nd_r2r(&forward)
            .unwrap();
        assert_eq!(shaders.len(), program.passes.len());
        assert!(
            shaders[1].glsl.contains("uint coord_0")
                && shaders[1]
                    .glsl
                    .contains("vkfft_r2r_nd_boundary_value(natural")
        );
        for shader in shaders {
            assert_eq!(shader.compile_spirv().unwrap().words[0], 0x0723_0203);
        }

        let mut mixed_device = device();
        mixed_device.supports_f64 = true;
        let mixed_plan = FftPlan::build(
            FftConfig::new(dimensions)
                .with_batch_count(batch_count)
                .with_transform(transform)
                .with_precision(Precision::F64ComputeF32Storage)
                .with_input_buffer_axis_stride(0, 7)
                .unwrap()
                .with_output_buffer_axis_stride(0, 9)
                .unwrap()
                .with_zero_padding(0, 1, 3)
                .unwrap()
                .with_zero_padding(1, 2, 4)
                .unwrap(),
        )
        .unwrap();
        let mixed = NdR2rIr::build(&mixed_plan, Direction::Forward, mixed_device).unwrap();
        assert_eq!(
            mixed
                .input_formatted_copy
                .as_ref()
                .unwrap()
                .input_storage_scalar,
            ScalarType::F32
        );
        assert_eq!(
            mixed
                .input_formatted_copy
                .as_ref()
                .unwrap()
                .output_storage_scalar,
            ScalarType::F64
        );
        assert_eq!(mixed.axes[0].pack.input_storage_scalar, ScalarType::F64);
        let mixed_shaders = crate::backend::vulkan::VulkanGlslBackend
            .lower_nd_r2r(&mixed)
            .unwrap();
        assert!(
            mixed_shaders[0]
                .glsl
                .contains("dvec2((vkfft_input.data[external]).x")
        );
        for shader in mixed_shaders {
            assert_eq!(shader.compile_spirv().unwrap().words[0], 0x0723_0203);
        }
    }

    #[test]
    fn nd_spatial_zero_padding_is_tensor_boundary_only_and_directional() {
        let dimensions = vec![3usize, 4usize];
        let tensor_len = dimensions.iter().product::<usize>();
        let batch_count = 2usize;
        let transform = TransformKind::Dct(DctType::II);
        let base_config = FftConfig::new(dimensions.clone())
            .with_batch_count(batch_count)
            .with_transform(transform);
        let padded_config = base_config
            .clone()
            .with_zero_padding(0, 1, 3)
            .unwrap()
            .with_zero_padding(1, 2, 4)
            .unwrap();
        let padded_plan = FftPlan::build(padded_config.clone()).unwrap();
        let padded = NdR2rIr::build(&padded_plan, Direction::Forward, device()).unwrap();
        let zero_padding = padded.zero_padding.as_ref().expect("ND zero-padding");
        assert_eq!(zero_padding.dimensions, dimensions);
        assert_eq!(
            zero_padding.ranges,
            vec![
                Some(ZeroPaddingRange::new(1, 3)),
                Some(ZeroPaddingRange::new(2, 4)),
            ]
        );
        assert_eq!(
            padded.axes[0].pack.zero_padding.as_ref(),
            Some(zero_padding)
        );
        assert!(padded.axes[0].scatter.zero_padding.is_none());
        for axis in &padded.axes {
            assert!(axis.transform.zero_padding.is_none());
        }
        for axis in padded.axes.iter().skip(1) {
            assert!(axis.pack.zero_padding.is_none());
            assert!(axis.scatter.zero_padding.is_none());
        }

        let input = (0..tensor_len * batch_count)
            .map(|index| {
                let x = index as f64;
                (0.13 * x).sin() + 0.23 * (0.079 * x).cos() + 0.001 * x
            })
            .collect::<Vec<_>>();
        let mut manual_zero = input.clone();
        for batch in 0..batch_count {
            let base = batch * tensor_len;
            for linear in 0..tensor_len {
                if zero_padding.contains_linear_index(linear) {
                    manual_zero[base + linear] = 0.0;
                }
            }
        }
        let baseline = NdR2rIr::build(
            &FftPlan::build(base_config.clone()).unwrap(),
            Direction::Forward,
            device(),
        )
        .unwrap();
        let expected = execute_nd_r2r_ir(&baseline, &manual_zero).unwrap();
        let actual = execute_nd_r2r_ir(&padded, &input).unwrap();
        let forward_error = actual
            .iter()
            .zip(&expected)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0, f64::max);
        assert!(
            forward_error < 5.0e-10 * tensor_len as f64,
            "{forward_error}"
        );

        let padded_program = crate::ProgramIr::nd_r2r(&padded).unwrap();
        let baseline_program = crate::ProgramIr::nd_r2r(&baseline).unwrap();
        assert_eq!(padded_program.passes.len(), baseline_program.passes.len());
        padded_program.validate().unwrap();
        let shaders = crate::backend::vulkan::VulkanGlslBackend
            .lower_nd_r2r(&padded)
            .unwrap();
        assert_eq!(
            shaders
                .iter()
                .filter(|shader| shader.glsl.contains("uint coord_0"))
                .count(),
            1
        );
        assert!(shaders.iter().any(|shader| {
            shader.glsl.contains("uint coord_0")
                && shader.glsl.contains("uint coord_1")
                && shader.glsl.contains("vkfft_r2r_nd_boundary_value(natural")
        }));
        for shader in shaders {
            assert_eq!(shader.compile_spirv().unwrap().words[0], 0x0723_0203);
        }

        let inverse_base = base_config.clone().with_inverse_normalization(true);
        let inverse_config = inverse_base
            .clone()
            .with_zero_padding(0, 1, 3)
            .unwrap()
            .with_zero_padding(1, 2, 4)
            .unwrap();
        let inverse = NdR2rIr::build(
            &FftPlan::build(inverse_config).unwrap(),
            Direction::Inverse,
            device(),
        )
        .unwrap();
        let inverse_zero = inverse.zero_padding.as_ref().unwrap();
        let last = inverse.axes.len() - 1;
        assert_eq!(
            inverse.axes[last].scatter.zero_padding.as_ref(),
            Some(inverse_zero)
        );
        assert!(inverse.axes[0].pack.zero_padding.is_none());
        for axis in &inverse.axes {
            assert!(axis.transform.zero_padding.is_none());
        }
        let baseline_inverse = NdR2rIr::build(
            &FftPlan::build(inverse_base).unwrap(),
            Direction::Inverse,
            device(),
        )
        .unwrap();
        let mut inverse_expected = execute_nd_r2r_ir(&baseline_inverse, &actual).unwrap();
        for batch in 0..batch_count {
            let base = batch * tensor_len;
            for linear in 0..tensor_len {
                if inverse_zero.contains_linear_index(linear) {
                    inverse_expected[base + linear] = 0.0;
                }
            }
        }
        let inverse_actual = execute_nd_r2r_ir(&inverse, &actual).unwrap();
        let inverse_error = inverse_actual
            .iter()
            .zip(&inverse_expected)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0, f64::max);
        assert!(
            inverse_error < 5.0e-10 * tensor_len as f64,
            "{inverse_error}"
        );
        for batch in 0..batch_count {
            let base = batch * tensor_len;
            for linear in 0..tensor_len {
                if inverse_zero.contains_linear_index(linear) {
                    assert_eq!(inverse_actual[base + linear], 0.0);
                }
            }
        }
        let inverse_shaders = crate::backend::vulkan::VulkanGlslBackend
            .lower_nd_r2r(&inverse)
            .unwrap();
        assert_eq!(
            inverse_shaders
                .iter()
                .filter(|shader| shader.glsl.contains("uint coord_0"))
                .count(),
            1
        );
        for shader in inverse_shaders {
            assert_eq!(shader.compile_spirv().unwrap().words[0], 0x0723_0203);
        }
    }

    #[test]
    fn spatial_zero_padding_is_directional_across_r2r_fft_reductions() {
        let batch_count = 5usize;
        let grouped_batch = 3usize;
        for (transform, length, expect_reduction) in [
            (TransformKind::Dct(DctType::IV), 9usize, true),
            (TransformKind::Dct(DctType::II), 16usize, true),
        ] {
            let left = 3usize;
            let right = 6usize;
            let base_config = FftConfig::new(vec![length])
                .with_batch_count(batch_count)
                .with_transform(transform)
                .with_grouped_batch(0, grouped_batch)
                .unwrap();
            let padded_config = base_config
                .clone()
                .with_zero_padding(0, left, right)
                .unwrap();
            let padded_plan = FftPlan::build(padded_config.clone()).unwrap();
            let padded = R2rIr::build(&padded_plan, Direction::Forward, device()).unwrap();
            assert_eq!(
                padded.zero_padding,
                Some(ZeroPaddingRange::new(left, right))
            );
            assert_eq!(padded.fft_reduction.is_some(), expect_reduction);
            if let Some(reduction) = padded.fft_reduction.as_deref() {
                assert_eq!(reduction.preprocess.zero_padding, padded.zero_padding);
                assert_eq!(reduction.postprocess.zero_padding, None);
            }

            let input = (0..length * batch_count)
                .map(|index| {
                    let x = index as f64;
                    (0.17 * x).sin() + 0.31 * (0.071 * x).cos() + 0.002 * x
                })
                .collect::<Vec<_>>();
            let mut manual_zero = input.clone();
            for batch in 0..batch_count {
                let base = batch * length;
                manual_zero[base + left..base + right].fill(0.0);
            }
            let baseline_plan = FftPlan::build(base_config.clone()).unwrap();
            let baseline = R2rIr::build(&baseline_plan, Direction::Forward, device()).unwrap();
            let expected = execute_r2r_ir(&baseline, &manual_zero).unwrap();
            let actual = execute_r2r_ir(&padded, &input).unwrap();
            let forward_error = actual
                .iter()
                .zip(&expected)
                .map(|(actual, expected)| (actual - expected).abs())
                .fold(0.0, f64::max);
            assert!(
                forward_error < 3.0e-10 * length as f64,
                "forward zero-pad {transform:?} N={length}: {forward_error}"
            );

            let program = crate::ProgramIr::r2r(&padded).unwrap();
            program.validate().unwrap();
            let shaders = crate::backend::vulkan::VulkanGlslBackend
                .lower_r2r_program(&padded)
                .unwrap();
            assert_eq!(shaders.len(), program.passes.len());
            assert!(shaders.iter().any(|shader| {
                shader.glsl.contains("VKFFT_ZERO_PAD_LEFT")
                    && shader.glsl.contains("VKFFT_ZERO_PAD_RIGHT")
            }));
            for shader in shaders {
                assert_eq!(shader.compile_spirv().unwrap().words[0], 0x0723_0203);
            }

            let inverse_base = base_config.clone().with_inverse_normalization(true);
            let inverse_padded = inverse_base
                .clone()
                .with_zero_padding(0, left, right)
                .unwrap();
            let inverse_plan = FftPlan::build(inverse_padded).unwrap();
            let inverse = R2rIr::build(&inverse_plan, Direction::Inverse, device()).unwrap();
            if let Some(reduction) = inverse.fft_reduction.as_deref() {
                assert_eq!(reduction.preprocess.zero_padding, None);
                assert_eq!(reduction.postprocess.zero_padding, inverse.zero_padding);
            }
            let baseline_inverse = R2rIr::build(
                &FftPlan::build(inverse_base).unwrap(),
                Direction::Inverse,
                device(),
            )
            .unwrap();
            let mut inverse_expected = execute_r2r_ir(&baseline_inverse, &actual).unwrap();
            for batch in 0..batch_count {
                let base = batch * length;
                inverse_expected[base + left..base + right].fill(0.0);
            }
            let inverse_actual = execute_r2r_ir(&inverse, &actual).unwrap();
            let inverse_error = inverse_actual
                .iter()
                .zip(&inverse_expected)
                .map(|(actual, expected)| (actual - expected).abs())
                .fold(0.0, f64::max);
            assert!(
                inverse_error < 3.0e-10 * length as f64,
                "inverse zero-pad {transform:?} N={length}: {inverse_error}"
            );
            for batch in 0..batch_count {
                let base = batch * length;
                assert!(
                    inverse_actual[base + left..base + right]
                        .iter()
                        .all(|value| *value == 0.0)
                );
            }
            let inverse_shaders = crate::backend::vulkan::VulkanGlslBackend
                .lower_r2r_program(&inverse)
                .unwrap();
            assert!(inverse_shaders.iter().any(|shader| {
                shader.glsl.contains("VKFFT_ZERO_PAD_LEFT")
                    && shader.glsl.contains("vkfft_r2r_store")
            }));
            for shader in inverse_shaders {
                assert_eq!(shader.compile_spirv().unwrap().words[0], 0x0723_0203);
            }
        }
    }

    #[test]
    fn dct_i_and_dst_i_use_upstream_fft_lengths_and_match_direct_definitions() {
        let length = 9usize;
        let input = (0..length)
            .map(|index| {
                let x = index as f64;
                (0.17 * x).sin() + 0.29 * (0.071 * x).cos() + x * 0.003
            })
            .collect::<Vec<_>>();

        for (transform, expected_fft_len) in [
            (TransformKind::Dct(DctType::I), 2 * length - 2),
            (TransformKind::Dst(DstType::I), 2 * length + 2),
        ] {
            let forward_plan =
                FftPlan::build(FftConfig::new(vec![length]).with_transform(transform)).unwrap();
            let forward = R2rIr::build(&forward_plan, Direction::Forward, device()).unwrap();
            let reduction = forward.fft_reduction.as_deref().unwrap();
            assert_eq!(reduction.fft_len, expected_fft_len);
            assert_eq!(reduction.fft.logical_len(), expected_fft_len);
            assert_eq!(reduction.fft.direction(), Direction::Forward);

            let actual = execute_r2r_ir(&forward, &input).unwrap();
            let effective = match transform {
                TransformKind::Dct(kind) => R2rTransform::Dct(kind),
                TransformKind::Dst(kind) => R2rTransform::Dst(kind),
                _ => unreachable!(),
            };
            let expected = (0..length)
                .map(|k| evaluate_bin(effective, &input, k))
                .collect::<Vec<_>>();
            let forward_error = actual
                .iter()
                .zip(&expected)
                .map(|(actual, expected)| (actual - expected).abs())
                .fold(0.0, f64::max);
            assert!(
                forward_error < 2.0e-10 * length as f64,
                "{transform:?}: {forward_error}"
            );

            let inverse_plan = FftPlan::build(
                FftConfig::new(vec![length])
                    .with_transform(transform)
                    .with_inverse_normalization(true),
            )
            .unwrap();
            let inverse = R2rIr::build(&inverse_plan, Direction::Inverse, device()).unwrap();
            assert_eq!(
                inverse.fft_reduction.as_deref().unwrap().fft.direction(),
                Direction::Forward
            );
            let restored = execute_r2r_ir(&inverse, &actual).unwrap();
            let inverse_error = restored
                .iter()
                .zip(&input)
                .map(|(actual, expected)| (actual - expected).abs())
                .fold(0.0, f64::max);
            assert!(
                inverse_error < 2.0e-10 * length as f64,
                "{transform:?}: {inverse_error}"
            );
        }
    }

    #[test]
    fn dct_iv_dst_iv_use_even_half_size_and_odd_two_n_fft_reductions() {
        for length in [2usize, 4, 6, 8, 10, 12] {
            let input = (0..length)
                .map(|index| {
                    let x = index as f64;
                    (0.17 * x).sin() + 0.29 * (0.071 * x).cos() + x * 0.003
                })
                .collect::<Vec<_>>();
            for transform in [
                TransformKind::Dct(DctType::IV),
                TransformKind::Dst(DstType::IV),
            ] {
                let forward_plan =
                    FftPlan::build(FftConfig::new(vec![length]).with_transform(transform)).unwrap();
                let forward = R2rIr::build(&forward_plan, Direction::Forward, device()).unwrap();
                let reduction = forward.fft_reduction.as_deref().unwrap();
                assert_eq!(reduction.fft_len, length / 2);
                assert_eq!(reduction.fft.logical_len(), length / 2);
                assert_eq!(reduction.fft.direction(), Direction::Inverse);

                let actual = execute_r2r_ir(&forward, &input).unwrap();
                let effective = match transform {
                    TransformKind::Dct(kind) => R2rTransform::Dct(kind),
                    TransformKind::Dst(kind) => R2rTransform::Dst(kind),
                    _ => unreachable!(),
                };
                let expected = (0..length)
                    .map(|k| evaluate_bin(effective, &input, k))
                    .collect::<Vec<_>>();
                let forward_error = actual
                    .iter()
                    .zip(&expected)
                    .map(|(actual, expected)| (actual - expected).abs())
                    .fold(0.0, f64::max);
                assert!(
                    forward_error < 2.0e-10 * length as f64,
                    "N={length} {transform:?}: {forward_error}"
                );

                let inverse_plan = FftPlan::build(
                    FftConfig::new(vec![length])
                        .with_transform(transform)
                        .with_inverse_normalization(true),
                )
                .unwrap();
                let inverse = R2rIr::build(&inverse_plan, Direction::Inverse, device()).unwrap();
                assert_eq!(
                    inverse.fft_reduction.as_deref().unwrap().fft.direction(),
                    Direction::Inverse
                );
                let restored = execute_r2r_ir(&inverse, &actual).unwrap();
                let inverse_error = restored
                    .iter()
                    .zip(&input)
                    .map(|(actual, expected)| (actual - expected).abs())
                    .fold(0.0, f64::max);
                assert!(
                    inverse_error < 2.0e-10 * length as f64,
                    "inverse N={length} {transform:?}: {inverse_error}"
                );
            }
        }

        let length = 9usize;
        let input = (0..length)
            .map(|index| {
                let x = index as f64;
                (0.17 * x).sin() + 0.29 * (0.071 * x).cos() + x * 0.003
            })
            .collect::<Vec<_>>();
        for transform in [
            TransformKind::Dct(DctType::IV),
            TransformKind::Dst(DstType::IV),
        ] {
            let odd = R2rIr::build(
                &FftPlan::build(FftConfig::new(vec![length]).with_transform(transform)).unwrap(),
                Direction::Forward,
                device(),
            )
            .unwrap();
            let reduction = odd
                .fft_reduction
                .as_deref()
                .expect("odd DCT/DST-IV must use the generic 2N FFT reduction");
            assert_eq!(reduction.fft_len, 2 * length);
            assert_eq!(reduction.fft.logical_len(), 2 * length);
            assert_eq!(reduction.fft.direction(), Direction::Forward);
            match transform {
                TransformKind::Dct(DctType::IV) => {
                    assert_eq!(
                        reduction.preprocess.operation,
                        R2rFftPassOperation::DctIvOddPhasePack
                    );
                    assert!(matches!(
                        reduction.postprocess.operation,
                        R2rFftPassOperation::DctIvOddPhaseExtract { .. }
                    ));
                }
                TransformKind::Dst(DstType::IV) => {
                    assert_eq!(
                        reduction.preprocess.operation,
                        R2rFftPassOperation::DstIvOddPhasePack
                    );
                    assert!(matches!(
                        reduction.postprocess.operation,
                        R2rFftPassOperation::DstIvOddPhaseExtract { .. }
                    ));
                }
                _ => unreachable!(),
            }
            let actual = execute_r2r_ir(&odd, &input).unwrap();
            let effective = match transform {
                TransformKind::Dct(kind) => R2rTransform::Dct(kind),
                TransformKind::Dst(kind) => R2rTransform::Dst(kind),
                _ => unreachable!(),
            };
            let expected = (0..length)
                .map(|k| evaluate_bin(effective, &input, k))
                .collect::<Vec<_>>();
            let error = actual
                .iter()
                .zip(&expected)
                .map(|(actual, expected)| (actual - expected).abs())
                .fold(0.0, f64::max);
            assert!(
                error < 2.0e-10 * length as f64,
                "odd {transform:?}: {error}"
            );

            let program = crate::ProgramIr::r2r(&odd).unwrap();
            let shaders = crate::backend::vulkan::VulkanGlslBackend
                .lower_r2r_program(&odd)
                .unwrap();
            assert_eq!(shaders.len(), program.passes.len());
            for shader in shaders {
                assert_eq!(shader.compile_spirv().unwrap().words[0], 0x0723_0203);
            }

            let inverse = R2rIr::build(
                &FftPlan::build(
                    FftConfig::new(vec![length])
                        .with_transform(transform)
                        .with_inverse_normalization(true),
                )
                .unwrap(),
                Direction::Inverse,
                device(),
            )
            .unwrap();
            assert_eq!(
                inverse.fft_reduction.as_deref().unwrap().fft.direction(),
                Direction::Forward
            );
            let restored = execute_r2r_ir(&inverse, &actual).unwrap();
            let inverse_error = restored
                .iter()
                .zip(&input)
                .map(|(actual, expected)| (actual - expected).abs())
                .fold(0.0, f64::max);
            assert!(
                inverse_error < 2.0e-10 * length as f64,
                "odd inverse {transform:?}: {inverse_error}"
            );
        }
    }

    #[test]
    fn dct_dst_ii_iii_use_upstream_n_point_fft_and_match_direct_definitions() {
        for length in [3usize, 4, 7, 8, 9] {
            let input = (0..length)
                .map(|index| {
                    let x = index as f64;
                    (0.17 * x).sin() + 0.29 * (0.071 * x).cos() + x * 0.003
                })
                .collect::<Vec<_>>();
            for transform in [
                TransformKind::Dct(DctType::II),
                TransformKind::Dct(DctType::III),
                TransformKind::Dst(DstType::II),
                TransformKind::Dst(DstType::III),
            ] {
                let plan =
                    FftPlan::build(FftConfig::new(vec![length]).with_transform(transform)).unwrap();
                let ir = R2rIr::build(&plan, Direction::Forward, device()).unwrap();
                let reduction = ir.fft_reduction.as_deref().unwrap();
                assert_eq!(reduction.fft_len, length);
                let expected_direction = match transform {
                    TransformKind::Dct(DctType::II) | TransformKind::Dst(DstType::II) => {
                        Direction::Forward
                    }
                    TransformKind::Dct(DctType::III) | TransformKind::Dst(DstType::III) => {
                        Direction::Inverse
                    }
                    _ => unreachable!(),
                };
                assert_eq!(reduction.fft.direction(), expected_direction);

                let actual = execute_r2r_ir(&ir, &input).unwrap();
                let effective = match transform {
                    TransformKind::Dct(kind) => R2rTransform::Dct(kind),
                    TransformKind::Dst(kind) => R2rTransform::Dst(kind),
                    _ => unreachable!(),
                };
                let expected = (0..length)
                    .map(|k| evaluate_bin(effective, &input, k))
                    .collect::<Vec<_>>();
                let error = actual
                    .iter()
                    .zip(&expected)
                    .map(|(actual, expected)| (actual - expected).abs())
                    .fold(0.0, f64::max);
                assert!(
                    error < 3.0e-10 * length as f64,
                    "N={length} {transform:?}: {error}"
                );
            }
        }
    }

    #[test]
    fn all_dct_dst_types_round_trip_with_inverse_normalization() {
        let length = 9usize;
        let batch_count = 2usize;
        let input = (0..length * batch_count)
            .map(|index| {
                let x = index as f64;
                (0.19 * x).sin() + 0.37 * (0.071 * x).cos() + x * 0.001
            })
            .collect::<Vec<_>>();
        for transform in all_transforms() {
            let forward_plan = FftPlan::build(
                FftConfig::new(vec![length])
                    .with_batch_count(batch_count)
                    .with_transform(transform),
            )
            .unwrap();
            let forward = R2rIr::build(&forward_plan, Direction::Forward, device()).unwrap();
            let spectrum = execute_r2r_ir(&forward, &input).unwrap();
            assert!(spectrum.iter().all(|value| value.is_finite()));

            let inverse_plan = FftPlan::build(
                FftConfig::new(vec![length])
                    .with_batch_count(batch_count)
                    .with_transform(transform)
                    .with_inverse_normalization(true),
            )
            .unwrap();
            let inverse = R2rIr::build(&inverse_plan, Direction::Inverse, device()).unwrap();
            let restored = execute_r2r_ir(&inverse, &spectrum).unwrap();
            let max_error = restored
                .iter()
                .zip(&input)
                .map(|(actual, expected)| (actual - expected).abs())
                .fold(0.0, f64::max);
            assert!(
                max_error < 3.0e-10 * length as f64,
                "round-trip mismatch for {transform:?}: {max_error}"
            );
        }
    }

    #[test]
    fn grouped_one_dimensional_r2r_covers_fft_and_direct_paths() {
        let batch_count = 7usize;
        let grouped_batch = 3usize;
        for length in [9usize] {
            let input = (0..length * batch_count)
                .map(|index| {
                    let x = index as f64;
                    (0.19 * x).sin() + 0.37 * (0.071 * x).cos() + x * 0.001
                })
                .collect::<Vec<_>>();
            for transform in all_transforms() {
                let forward_plan = FftPlan::build(
                    FftConfig::new(vec![length])
                        .with_batch_count(batch_count)
                        .with_grouped_batch(0, grouped_batch)
                        .unwrap()
                        .with_transform(transform),
                )
                .unwrap();
                let forward = R2rIr::build(&forward_plan, Direction::Forward, device()).unwrap();
                assert_eq!(forward.grouped_batch, grouped_batch);
                assert_eq!(forward.dispatch.x, 3);
                if let Some(reduction) = forward.fft_reduction.as_deref() {
                    assert_eq!(reduction.grouped_batch, grouped_batch);
                    assert_eq!(reduction.preprocess.grouped_batch, grouped_batch);
                    assert_eq!(reduction.postprocess.grouped_batch, grouped_batch);
                    assert_eq!(reduction.preprocess.dispatch.x, 3);
                    assert_eq!(reduction.postprocess.dispatch.x, 3);
                    let program = crate::ProgramIr::r2r(&forward).unwrap();
                    assert!(program.passes.iter().all(|pass| pass.dispatch.x == 3));
                } else {
                    assert!(matches!(
                        transform,
                        TransformKind::Dct(DctType::IV) | TransformKind::Dst(DstType::IV)
                    ));
                    assert_eq!(
                        crate::ProgramIr::r2r(&forward).unwrap().passes[0]
                            .dispatch
                            .x,
                        3
                    );
                }
                let spectrum = execute_r2r_ir(&forward, &input).unwrap();

                let inverse_plan = FftPlan::build(
                    FftConfig::new(vec![length])
                        .with_batch_count(batch_count)
                        .with_grouped_batch(0, grouped_batch)
                        .unwrap()
                        .with_transform(transform)
                        .with_inverse_normalization(true),
                )
                .unwrap();
                let inverse = R2rIr::build(&inverse_plan, Direction::Inverse, device()).unwrap();
                assert_eq!(inverse.grouped_batch, grouped_batch);
                let restored = execute_r2r_ir(&inverse, &spectrum).unwrap();
                let max_error = restored
                    .iter()
                    .zip(&input)
                    .map(|(actual, expected)| (actual - expected).abs())
                    .fold(0.0, f64::max);
                assert!(
                    max_error < 3.0e-10 * length as f64,
                    "grouped round-trip mismatch for {transform:?}: {max_error}"
                );
            }
        }

        for transform in [
            TransformKind::Dct(DctType::IV),
            TransformKind::Dst(DstType::IV),
        ] {
            let plan = FftPlan::build(
                FftConfig::new(vec![8])
                    .with_batch_count(batch_count)
                    .with_grouped_batch(0, grouped_batch)
                    .unwrap()
                    .with_transform(transform),
            )
            .unwrap();
            let ir = R2rIr::build(&plan, Direction::Forward, device()).unwrap();
            let reduction = ir.fft_reduction.as_deref().unwrap();
            assert_eq!(reduction.fft_len, 4);
            assert_eq!(reduction.grouped_batch, grouped_batch);
            assert_eq!(reduction.preprocess.dispatch.x, 3);
            assert_eq!(reduction.postprocess.dispatch.x, 3);
            assert!(
                crate::ProgramIr::r2r(&ir)
                    .unwrap()
                    .passes
                    .iter()
                    .all(|pass| pass.dispatch.x == 3)
            );
        }
    }

    #[test]
    fn grouped_multidimensional_r2r_owns_fft_and_direct_axis_children() {
        let batch_count = 7usize;
        let grouped_batch = 3usize;
        for (dimensions, transform, expect_fft_reduction) in [
            (vec![3usize, 4], TransformKind::Dct(DctType::II), true),
            (vec![3usize, 5], TransformKind::Dst(DstType::IV), true),
        ] {
            let tensor_len = dimensions.iter().product::<usize>();
            let input = (0..tensor_len * batch_count)
                .map(|index| {
                    let x = index as f64;
                    (0.13 * x).sin() + 0.21 * (0.059 * x).cos() + x * 0.001
                })
                .collect::<Vec<_>>();
            let grouped_config = FftConfig::new(dimensions.clone())
                .with_batch_count(batch_count)
                .with_precision(Precision::F16StorageF32Compute)
                .with_transform(transform)
                .with_grouped_batch(0, grouped_batch)
                .unwrap()
                .with_grouped_batch(1, grouped_batch)
                .unwrap();
            let forward_plan = FftPlan::build(grouped_config.clone()).unwrap();
            let forward = NdR2rIr::build(&forward_plan, Direction::Forward, device()).unwrap();
            assert_eq!(forward.axes.len(), dimensions.len());
            for axis in &forward.axes {
                assert_eq!(axis.pack.grouped_batch, Some(grouped_batch));
                assert_eq!(axis.scatter.grouped_batch, Some(grouped_batch));
                assert_eq!(axis.pack.dispatch.x, 3);
                assert_eq!(axis.scatter.dispatch.x, 3);
                assert_eq!(axis.transform.grouped_batch, grouped_batch);
                assert_eq!(axis.transform.fft_reduction.is_some(), expect_fft_reduction);
            }
            let program = crate::ProgramIr::nd_r2r(&forward).unwrap();
            assert!(program.passes.iter().any(|pass| pass.dispatch.x > 3));
            let spectrum = execute_nd_r2r_ir(&forward, &input).unwrap();

            let inverse_plan =
                FftPlan::build(grouped_config.with_inverse_normalization(true)).unwrap();
            let inverse = NdR2rIr::build(&inverse_plan, Direction::Inverse, device()).unwrap();
            assert!(
                crate::ProgramIr::nd_r2r(&inverse)
                    .unwrap()
                    .passes
                    .iter()
                    .any(|pass| pass.dispatch.x > 3)
            );
            let restored = execute_nd_r2r_ir(&inverse, &spectrum).unwrap();
            let error = restored
                .iter()
                .zip(&input)
                .map(|(actual, expected)| (actual - expected).abs())
                .fold(0.0, f64::max);
            assert!(
                error < 8.0e-10 * tensor_len as f64,
                "grouped {transform:?} {dimensions:?}: {error}"
            );
        }
    }

    #[test]
    fn grouped_multidimensional_r2r_strided_fft_uses_upstream_xy_tile() {
        let grouped_batch = 3usize;
        let plan = FftPlan::build(
            FftConfig::new(vec![64usize, 4])
                .with_batch_count(5)
                .with_transform(TransformKind::Dct(DctType::II))
                .with_grouped_batch(0, grouped_batch)
                .unwrap()
                .with_grouped_batch(1, grouped_batch)
                .unwrap(),
        )
        .unwrap();
        let ir = NdR2rIr::build(&plan, Direction::Forward, device()).unwrap();
        let outer = ir.axes.iter().find(|axis| axis.axis == 0).unwrap();
        assert_eq!(outer.transform.grouped_batch, grouped_batch);
        let reduction = outer.transform.fft_reduction.as_deref().unwrap();
        let OneDimFftIr::Recursive(recursive) = &reduction.fft else {
            panic!("N=64 DCT-II reduction should use recursive Stockham");
        };
        let crate::recursive_ir::RecursiveFftNodeIr::Stockham(kernel) = &recursive.root else {
            panic!("N=64 DCT-II reduction should have a Stockham root");
        };
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
    fn multidimensional_dct_dst_round_trip() {
        let dimensions = vec![3usize, 4];
        let tensor_len = 12usize;
        let input = (0..tensor_len)
            .map(|index| {
                let x = index as f64;
                (0.13 * x).sin() + 0.21 * (0.059 * x).cos() + x * 0.001
            })
            .collect::<Vec<_>>();
        for transform in all_transforms() {
            let forward_plan =
                FftPlan::build(FftConfig::new(dimensions.clone()).with_transform(transform))
                    .unwrap();
            let forward = NdR2rIr::build(&forward_plan, Direction::Forward, device()).unwrap();
            let spectrum = execute_nd_r2r_ir(&forward, &input).unwrap();
            let inverse_plan = FftPlan::build(
                FftConfig::new(dimensions.clone())
                    .with_transform(transform)
                    .with_inverse_normalization(true),
            )
            .unwrap();
            let inverse = NdR2rIr::build(&inverse_plan, Direction::Inverse, device()).unwrap();
            let restored = execute_nd_r2r_ir(&inverse, &spectrum).unwrap();
            let error = restored
                .iter()
                .zip(&input)
                .map(|(actual, expected)| (actual - expected).abs())
                .fold(0.0, f64::max);
            assert!(
                error < 5.0e-10 * tensor_len as f64,
                "{transform:?}: {error}"
            );
        }
    }

    #[test]
    fn dct_ii_matches_fftw_redft10_definition() {
        let input = [1.0, -0.25, 0.5, 2.0];
        let plan = FftPlan::build(
            FftConfig::new(vec![input.len()]).with_transform(TransformKind::Dct(DctType::II)),
        )
        .unwrap();
        let ir = R2rIr::build(&plan, Direction::Forward, device()).unwrap();
        let actual = execute_r2r_ir(&ir, &input).unwrap();
        let expected = (0..input.len())
            .map(|k| {
                input
                    .iter()
                    .enumerate()
                    .map(|(j, value)| {
                        2.0 * value * (PI * (j as f64 + 0.5) * k as f64 / input.len() as f64).cos()
                    })
                    .sum::<f64>()
            })
            .collect::<Vec<_>>();
        assert!(
            actual
                .iter()
                .zip(expected)
                .all(|(actual, expected)| (actual - expected).abs() < 1.0e-12)
        );
    }
}
