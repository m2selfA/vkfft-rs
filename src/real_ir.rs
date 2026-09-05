//! Backend-neutral 1D real/complex transform composition.
//!
//! Odd lengths use the correctness-first full complex FFT fallback. Even lengths
//! use the standard half-size real decomposition: pack even/odd samples into one
//! complex sequence, transform N/2 points, then reconstruct the Hermitian spectrum;
//! C2R applies the inverse algebra before unpacking the N/2 complex samples.

use core::f64::consts::TAU;

use crate::bluestein_ir::{BluesteinPipelineIr, execute_bluestein_ir};
use crate::complex::Complex64;
use crate::config::{
    DeviceProfile, Direction, FftConfig, GpuVendor, PlannerTuning, Precision, TransformKind,
    has_fixed_upstream_gpu_scheduler_profile,
};
use crate::error::{Result, VkFftError};
use crate::kernel_ir::{
    DispatchGeometry, KernelIr, RealEvenInversePreprocessMapping, RealEvenPackMapping,
    RealEvenPostprocessMapping, RealEvenUnpackMapping, ScalarType, StockhamIoMapping,
    WorkgroupSize, execute_stockham_ir,
};
use crate::one_dim_ir::{OneDimFftIr, execute_one_dim_fft_ir};
use crate::planner::{AxisAlgorithm, C2cDeviceAxisClass, FftPlan, RaderMode};
use crate::recursive_ir::{
    RecursiveFftIr, RecursiveFftNodeIr, execute_recursive_fft_ir_with_resources,
};
use crate::scheduler::{plan_gpu_force_rader_two_upload, upstream_bluestein_auto_padding};
use crate::zero_pad_ir::ZeroPadPassIr;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RealFftKind {
    RealToComplex,
    ComplexToReal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RealFftAlgorithm {
    FullComplex,
    EvenHalfSize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RealFftShapePolicy {
    PortableHalfSizeAllEven,
    FixedUpstreamDevice,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RealPassOperation {
    PromoteFullComplex,
    CompactHermitian,
    ExpandHermitian,
    PackEvenOdd,
    PostprocessEvenHalf,
    PreprocessEvenHalf { normalize: bool },
    UnpackEvenOdd,
    FinalizeFullComplexReal,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RealPassIr {
    pub name: String,
    pub scalar: ScalarType,
    pub input_storage_scalar: ScalarType,
    pub output_storage_scalar: ScalarType,
    pub length: usize,
    pub half_spectrum_len: usize,
    pub batch_count: usize,
    pub grouped_batch: usize,
    pub workgroup_size: WorkgroupSize,
    pub dispatch: DispatchGeometry,
    pub operation: RealPassOperation,
}

impl RealPassIr {
    fn new(
        name: String,
        scalar: ScalarType,
        length: usize,
        batch_count: usize,
        grouped_batch: usize,
        operation: RealPassOperation,
        device: DeviceProfile,
    ) -> Result<Self> {
        if device.max_threads_per_block == 0 {
            return Err(VkFftError::ResourceLimitExceeded {
                resource: "threads per workgroup",
                required: 1,
                available: 0,
            });
        }
        if grouped_batch == 0 {
            return Err(VkFftError::InvalidKernelIr(
                "real FFT groupedBatch must be non-zero",
            ));
        }
        let half_spectrum_len = length / 2 + 1;
        let logical_work = match operation {
            RealPassOperation::CompactHermitian | RealPassOperation::PostprocessEvenHalf => {
                half_spectrum_len
            }
            RealPassOperation::PromoteFullComplex
            | RealPassOperation::ExpandHermitian
            | RealPassOperation::UnpackEvenOdd
            | RealPassOperation::FinalizeFullComplexReal => length,
            RealPassOperation::PackEvenOdd | RealPassOperation::PreprocessEvenHalf { .. } => {
                length / 2
            }
        };
        let local_size = logical_work.min(device.max_threads_per_block).max(1);
        let pass = Self {
            name,
            scalar,
            input_storage_scalar: scalar,
            output_storage_scalar: scalar,
            length,
            half_spectrum_len,
            batch_count,
            grouped_batch,
            workgroup_size: WorkgroupSize {
                x: u32::try_from(local_size).map_err(|_| VkFftError::ValueOutOfRange {
                    field: "real FFT workgroup size",
                })?,
                y: 1,
                z: 1,
            },
            dispatch: DispatchGeometry {
                x: u32::try_from(batch_count.div_ceil(grouped_batch)).map_err(|_| {
                    VkFftError::ValueOutOfRange {
                        field: "real FFT grouped dispatch count",
                    }
                })?,
                y: 1,
                z: 1,
            },
            operation,
        };
        pass.validate()?;
        Ok(pass)
    }

    pub const fn input_len(&self) -> usize {
        match self.operation {
            RealPassOperation::PromoteFullComplex
            | RealPassOperation::CompactHermitian
            | RealPassOperation::PackEvenOdd
            | RealPassOperation::FinalizeFullComplexReal => self.length,
            RealPassOperation::ExpandHermitian | RealPassOperation::PreprocessEvenHalf { .. } => {
                self.half_spectrum_len
            }
            RealPassOperation::PostprocessEvenHalf | RealPassOperation::UnpackEvenOdd => {
                self.length / 2
            }
        }
    }

    pub const fn output_len(&self) -> usize {
        match self.operation {
            RealPassOperation::CompactHermitian | RealPassOperation::PostprocessEvenHalf => {
                self.half_spectrum_len
            }
            RealPassOperation::PromoteFullComplex
            | RealPassOperation::ExpandHermitian
            | RealPassOperation::UnpackEvenOdd
            | RealPassOperation::FinalizeFullComplexReal => self.length,
            RealPassOperation::PackEvenOdd | RealPassOperation::PreprocessEvenHalf { .. } => {
                self.length / 2
            }
        }
    }

    pub(crate) fn with_external_input_storage(mut self, storage: ScalarType) -> Result<Self> {
        if storage != self.scalar
            && (!supported_real_storage_pair(self.scalar, storage)
                || !matches!(
                    self.operation,
                    RealPassOperation::PromoteFullComplex
                        | RealPassOperation::ExpandHermitian
                        | RealPassOperation::PackEvenOdd
                        | RealPassOperation::PreprocessEvenHalf { .. }
                ))
        {
            return Err(VkFftError::UnsupportedPrecision {
                backend: "real FFT input boundary",
                precision: "mixed storage on unsupported real input pass",
            });
        }
        self.input_storage_scalar = storage;
        self.validate()?;
        Ok(self)
    }

    pub(crate) fn with_external_output_storage(mut self, storage: ScalarType) -> Result<Self> {
        if storage != self.scalar
            && (!supported_real_storage_pair(self.scalar, storage)
                || !matches!(
                    self.operation,
                    RealPassOperation::CompactHermitian
                        | RealPassOperation::PostprocessEvenHalf
                        | RealPassOperation::UnpackEvenOdd
                        | RealPassOperation::FinalizeFullComplexReal
                ))
        {
            return Err(VkFftError::UnsupportedPrecision {
                backend: "real FFT output boundary",
                precision: "mixed storage on unsupported real output pass",
            });
        }
        self.output_storage_scalar = storage;
        self.validate()?;
        Ok(self)
    }

    pub fn validate(&self) -> Result<()> {
        let requires_even_length = matches!(
            self.operation,
            RealPassOperation::PackEvenOdd
                | RealPassOperation::PostprocessEvenHalf
                | RealPassOperation::PreprocessEvenHalf { .. }
                | RealPassOperation::UnpackEvenOdd
        );
        let narrow_input_ok = self.input_storage_scalar == self.scalar
            || (supported_real_storage_pair(self.scalar, self.input_storage_scalar)
                && matches!(
                    self.operation,
                    RealPassOperation::PromoteFullComplex
                        | RealPassOperation::ExpandHermitian
                        | RealPassOperation::PackEvenOdd
                        | RealPassOperation::PreprocessEvenHalf { .. }
                ));
        let narrow_output_ok = self.output_storage_scalar == self.scalar
            || (supported_real_storage_pair(self.scalar, self.output_storage_scalar)
                && matches!(
                    self.operation,
                    RealPassOperation::CompactHermitian
                        | RealPassOperation::PostprocessEvenHalf
                        | RealPassOperation::UnpackEvenOdd
                        | RealPassOperation::FinalizeFullComplexReal
                ));
        if self.scalar == ScalarType::F16
            || !narrow_input_ok
            || !narrow_output_ok
            || self.length == 0
            || self.half_spectrum_len != self.length / 2 + 1
            || self.batch_count == 0
            || self.grouped_batch == 0
            || (requires_even_length && (self.length < 2 || !self.length.is_multiple_of(2)))
            || self.workgroup_size.x == 0
            || self.workgroup_size.y != 1
            || self.workgroup_size.z != 1
            || self.dispatch.x as usize != self.batch_count.div_ceil(self.grouped_batch)
            || self.dispatch.y != 1
            || self.dispatch.z != 1
        {
            return Err(VkFftError::InvalidKernelIr(
                "real FFT pass metadata is inconsistent",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct RealFftIr {
    pub kind: RealFftKind,
    pub algorithm: RealFftAlgorithm,
    pub length: usize,
    pub half_spectrum_len: usize,
    pub batch_count: usize,
    pub grouped_batch: usize,
    pub scalar: ScalarType,
    pub external_scalar: ScalarType,
    pub input_storage_scalar: ScalarType,
    pub output_storage_scalar: ScalarType,
    pub transform: OneDimFftIr,
    pub preprocess: Option<RealPassIr>,
    pub postprocess: Option<RealPassIr>,
    pub zero_pad_pass: Option<ZeroPadPassIr>,
}

impl RealFftIr {
    pub fn build(plan: &FftPlan, device: DeviceProfile) -> Result<Self> {
        Self::build_with_shape_policy(plan, device, RealFftShapePolicy::PortableHalfSizeAllEven)
    }

    pub(crate) fn build_for_device_plan(plan: &FftPlan, device: DeviceProfile) -> Result<Self> {
        Self::build_with_shape_policy(plan, device, RealFftShapePolicy::FixedUpstreamDevice)
    }

    pub(crate) fn build_with_shape_policy(
        plan: &FftPlan,
        device: DeviceProfile,
        shape_policy: RealFftShapePolicy,
    ) -> Result<Self> {
        if plan.config.dimensions.len() != 1 {
            return Err(VkFftError::UnsupportedKernelPath(
                "real FFT IR currently supports one-dimensional transforms only",
            ));
        }
        let (kind, direction) = match plan.config.transform {
            TransformKind::RealToComplex => (RealFftKind::RealToComplex, Direction::Forward),
            TransformKind::ComplexToReal => (RealFftKind::ComplexToReal, Direction::Inverse),
            _ => {
                return Err(VkFftError::UnsupportedKernelPath(
                    "real FFT IR requires an R2C or C2R planner configuration",
                ));
            }
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
                    backend: "real FFT IR",
                    precision: precision_name(other),
                });
            }
        };
        let execution_batch_count = plan.config.kernel_preparation_system_count()?;
        let length = plan.axes[0].logical_len;
        let grouped_batch_override = plan.config.grouped_batch_for_axis(0);
        let grouped_batch = grouped_batch_override.unwrap_or(1);
        if plan.axes[0].effective_fft_len != length {
            return Err(VkFftError::InvalidKernelIr(
                "real FFT planner changed the effective C2C transform length",
            ));
        }
        let algorithm = real_fft_algorithm_for_shape_policy(
            length,
            execution_batch_count,
            plan.config.precision,
            plan.config.tuning,
            device,
            shape_policy,
        )?;
        let transform_len = match algorithm {
            RealFftAlgorithm::FullComplex => length,
            RealFftAlgorithm::EvenHalfSize => length / 2,
        };
        let mut internal_config = FftConfig::new(vec![transform_len])
            .with_batch_count(execution_batch_count)
            .with_precision(plan.config.precision)
            .with_kernel_convolution(plan.config.kernel_convolution)
            .with_inverse_normalization(
                direction == Direction::Inverse && plan.config.normalize_inverse,
            )
            .with_tuning(plan.config.tuning)
            .with_bandwidth_boost(plan.config.bandwidth_boost);
        if let Some(grouped_batch) = grouped_batch_override {
            internal_config = internal_config.with_grouped_batch(0, grouped_batch)?;
        }
        let internal_plan = FftPlan::build_c2c_child_for_device(
            internal_config,
            device,
            C2cDeviceAxisClass::Contiguous,
        )?;
        let transform =
            OneDimFftIr::build_internal_compute_storage(&internal_plan, direction, device)?;
        let transform = if grouped_batch_override.is_some() {
            transform.with_axis0_single_upload_block(device)?
        } else {
            transform
        };
        let direction_name = match kind {
            RealFftKind::RealToComplex => "r2c",
            RealFftKind::ComplexToReal => "c2r",
        };
        let pass = |suffix: &str, operation| {
            RealPassIr::new(
                format!("vkfft_real_{direction_name}_{suffix}_{length}"),
                scalar,
                length,
                execution_batch_count,
                grouped_batch,
                operation,
                device,
            )
        };
        let (preprocess, postprocess) = match (kind, algorithm) {
            (RealFftKind::RealToComplex, RealFftAlgorithm::FullComplex) => (
                (external_scalar != scalar)
                    .then(|| pass("promote_full", RealPassOperation::PromoteFullComplex))
                    .transpose()?,
                Some(pass("compact", RealPassOperation::CompactHermitian)?),
            ),
            (RealFftKind::ComplexToReal, RealFftAlgorithm::FullComplex) => (
                Some(pass("expand", RealPassOperation::ExpandHermitian)?),
                (external_scalar != scalar)
                    .then(|| {
                        pass(
                            "finalize_full_real",
                            RealPassOperation::FinalizeFullComplexReal,
                        )
                    })
                    .transpose()?,
            ),
            (RealFftKind::RealToComplex, RealFftAlgorithm::EvenHalfSize) => (
                Some(pass("pack_even_odd", RealPassOperation::PackEvenOdd)?),
                Some(pass(
                    "postprocess_even_half",
                    RealPassOperation::PostprocessEvenHalf,
                )?),
            ),
            (RealFftKind::ComplexToReal, RealFftAlgorithm::EvenHalfSize) => (
                Some(pass(
                    "preprocess_even_half",
                    RealPassOperation::PreprocessEvenHalf {
                        normalize: plan.config.normalize_inverse,
                    },
                )?),
                Some(pass("unpack_even_odd", RealPassOperation::UnpackEvenOdd)?),
            ),
        };
        let zero_pad_pass = plan
            .config
            .zero_padding_for_axis(0)
            .map(|range| {
                let pass = ZeroPadPassIr::build(
                    length,
                    execution_batch_count,
                    plan.config.precision,
                    direction,
                    range,
                    device,
                )?;
                if let Some(grouped_batch) = grouped_batch_override {
                    pass.with_grouped_batch(grouped_batch)
                } else {
                    Ok(pass)
                }
            })
            .transpose()?;
        let (mut preprocess, mut postprocess) = (preprocess, postprocess);
        if external_scalar != scalar {
            match kind {
                RealFftKind::RealToComplex => {
                    if zero_pad_pass.is_none() {
                        let boundary = preprocess.as_mut().ok_or(VkFftError::InvalidKernelIr(
                            "mixed-storage R2C requires an explicit input boundary",
                        ))?;
                        *boundary = boundary
                            .clone()
                            .with_external_input_storage(external_scalar)?;
                    }
                    let boundary = postprocess.as_mut().ok_or(VkFftError::InvalidKernelIr(
                        "mixed-storage R2C requires an explicit output boundary",
                    ))?;
                    *boundary = boundary
                        .clone()
                        .with_external_output_storage(external_scalar)?;
                }
                RealFftKind::ComplexToReal => {
                    let boundary = preprocess.as_mut().ok_or(VkFftError::InvalidKernelIr(
                        "mixed-storage C2R requires an explicit input boundary",
                    ))?;
                    *boundary = boundary
                        .clone()
                        .with_external_input_storage(external_scalar)?;
                    if zero_pad_pass.is_none() {
                        let boundary = postprocess.as_mut().ok_or(VkFftError::InvalidKernelIr(
                            "mixed-storage C2R requires an explicit output boundary",
                        ))?;
                        *boundary = boundary
                            .clone()
                            .with_external_output_storage(external_scalar)?;
                    }
                }
            }
        }
        let ir = Self {
            kind,
            algorithm,
            length,
            half_spectrum_len: length / 2 + 1,
            batch_count: execution_batch_count,
            grouped_batch,
            scalar,
            external_scalar,
            input_storage_scalar: external_scalar,
            output_storage_scalar: external_scalar,
            transform,
            preprocess,
            postprocess,
            zero_pad_pass,
        };
        ir.validate()?;
        Ok(ir)
    }

    pub fn transform_len(&self) -> usize {
        self.transform.logical_len()
    }

    pub const fn external_storage_scalar(&self) -> ScalarType {
        self.external_scalar
    }

    pub const fn input_storage_scalar(&self) -> ScalarType {
        self.input_storage_scalar
    }

    pub const fn output_storage_scalar(&self) -> ScalarType {
        self.output_storage_scalar
    }

    pub(crate) fn with_input_storage_scalar(mut self, storage: ScalarType) -> Result<Self> {
        if storage != self.scalar && !supported_real_storage_pair(self.scalar, storage) {
            return Err(VkFftError::UnsupportedPrecision {
                backend: "real FFT composed input boundary",
                precision: "unsupported mixed input storage scalar",
            });
        }
        if self.zero_pad_pass.is_some() {
            return Err(VkFftError::UnsupportedKernelPath(
                "retagging a real FFT input boundary with a local zero-pad pass is unsupported",
            ));
        }
        let boundary = self.preprocess.as_mut().ok_or(VkFftError::InvalidKernelIr(
            "real FFT input storage retag requires an explicit preprocess boundary",
        ))?;
        *boundary = boundary.clone().with_external_input_storage(storage)?;
        self.input_storage_scalar = storage;
        self.validate()?;
        Ok(self)
    }

    pub(crate) fn with_output_storage_scalar(mut self, storage: ScalarType) -> Result<Self> {
        if storage != self.scalar && !supported_real_storage_pair(self.scalar, storage) {
            return Err(VkFftError::UnsupportedPrecision {
                backend: "real FFT composed output boundary",
                precision: "unsupported mixed output storage scalar",
            });
        }
        if self.zero_pad_pass.is_some() {
            return Err(VkFftError::UnsupportedKernelPath(
                "retagging a real FFT output boundary with a local zero-pad pass is unsupported",
            ));
        }
        let boundary = self
            .postprocess
            .as_mut()
            .ok_or(VkFftError::InvalidKernelIr(
                "real FFT output storage retag requires an explicit postprocess boundary",
            ))?;
        *boundary = boundary.clone().with_external_output_storage(storage)?;
        self.output_storage_scalar = storage;
        self.validate()?;
        Ok(self)
    }

    /// Clone the half-size Stockham root with the even-real preprocess folded into
    /// stage-0 global loads. Rader/Bluestein/recursive internal transforms retain the
    /// explicit typed preprocess pass so no unsupported storage semantics are guessed.
    pub(crate) fn fused_even_input_stockham_kernel(&self) -> Result<Option<KernelIr>> {
        if self.input_storage_scalar != self.scalar
            || self.output_storage_scalar != self.scalar
            || self.algorithm != RealFftAlgorithm::EvenHalfSize
        {
            return Ok(None);
        }
        let OneDimFftIr::Recursive(recursive) = &self.transform else {
            return Ok(None);
        };
        let RecursiveFftNodeIr::Stockham(kernel) = &recursive.root else {
            return Ok(None);
        };
        let mapping = match self.kind {
            RealFftKind::RealToComplex => {
                if !matches!(
                    self.preprocess.as_ref().map(|pass| pass.operation),
                    Some(RealPassOperation::PackEvenOdd)
                ) {
                    return Ok(None);
                }
                StockhamIoMapping::RealEvenPack(RealEvenPackMapping {
                    full_len: self.length,
                })
            }
            RealFftKind::ComplexToReal => {
                let Some(RealPassOperation::PreprocessEvenHalf { normalize }) =
                    self.preprocess.as_ref().map(|pass| pass.operation)
                else {
                    return Ok(None);
                };
                StockhamIoMapping::RealEvenInversePreprocess(RealEvenInversePreprocessMapping {
                    full_len: self.length,
                    normalize,
                })
            }
        };
        let mut fused = kernel.as_ref().clone().with_stockham_io_mapping(mapping)?;
        fused = match self.kind {
            RealFftKind::RealToComplex => fused.with_real_even_postprocess_output(self.length)?,
            RealFftKind::ComplexToReal => fused.with_real_even_unpack_output(self.length)?,
        };
        Ok(Some(fused))
    }

    /// Clone a recursive Cooley-Tukey half-size root with the invocation-local
    /// even-real boundary transforms folded into its external pack/scatter passes.
    /// Four-step roots are intentionally excluded because their physical upload
    /// mappings bypass these generic Cooley-Tukey boundary passes.
    pub(crate) fn fused_even_recursive_ir(&self) -> Result<Option<RecursiveFftIr>> {
        if self.input_storage_scalar != self.scalar
            || self.output_storage_scalar != self.scalar
            || self.algorithm != RealFftAlgorithm::EvenHalfSize
        {
            return Ok(None);
        }
        let OneDimFftIr::Recursive(recursive) = &self.transform else {
            return Ok(None);
        };
        if recursive.four_step_plan.is_some() {
            return Ok(None);
        }
        let RecursiveFftNodeIr::CooleyTukey(_) = &recursive.root else {
            return Ok(None);
        };
        let mut fused = recursive.as_ref().clone();
        let RecursiveFftNodeIr::CooleyTukey(root) = &mut fused.root else {
            unreachable!();
        };
        match self.kind {
            RealFftKind::RealToComplex => {
                if !matches!(
                    self.preprocess.as_ref().map(|pass| pass.operation),
                    Some(RealPassOperation::PackEvenOdd)
                ) {
                    return Ok(None);
                }
                root.pack_right =
                    root.pack_right
                        .clone()
                        .with_real_even_pack_input(RealEvenPackMapping {
                            full_len: self.length,
                        })?;
                root.scatter_output = root
                    .scatter_output
                    .clone()
                    .with_real_even_postprocess_output(RealEvenPostprocessMapping {
                        full_len: self.length,
                    })?;
            }
            RealFftKind::ComplexToReal => {
                let Some(RealPassOperation::PreprocessEvenHalf { normalize }) =
                    self.preprocess.as_ref().map(|pass| pass.operation)
                else {
                    return Ok(None);
                };
                root.pack_right = root
                    .pack_right
                    .clone()
                    .with_real_even_inverse_preprocess_input(RealEvenInversePreprocessMapping {
                        full_len: self.length,
                        normalize,
                    })?;
                root.scatter_output = root.scatter_output.clone().with_real_even_unpack_output(
                    RealEvenUnpackMapping {
                        full_len: self.length,
                    },
                )?;
            }
        }
        fused.validate()?;
        Ok(Some(fused))
    }

    /// Clone a half-size Bluestein child with the even-real boundary algebra folded
    /// into its existing outer chirp preprocess/postprocess dispatches. Both boundary
    /// passes execute after their required mirrored values are globally materialized,
    /// so this fusion does not add cross-workgroup synchronization.
    pub(crate) fn fused_even_bluestein_ir(&self) -> Result<Option<BluesteinPipelineIr>> {
        if self.input_storage_scalar != self.scalar
            || self.output_storage_scalar != self.scalar
            || self.algorithm != RealFftAlgorithm::EvenHalfSize
        {
            return Ok(None);
        }
        let OneDimFftIr::Bluestein(pipeline) = &self.transform else {
            return Ok(None);
        };
        if pipeline.zero_pad_pass.is_some() {
            return Ok(None);
        }
        let mut fused = pipeline.as_ref().clone();
        match self.kind {
            RealFftKind::RealToComplex => {
                if !matches!(
                    self.preprocess.as_ref().map(|pass| pass.operation),
                    Some(RealPassOperation::PackEvenOdd)
                ) || !matches!(
                    self.postprocess.as_ref().map(|pass| pass.operation),
                    Some(RealPassOperation::PostprocessEvenHalf)
                ) {
                    return Ok(None);
                }
                fused.preprocess =
                    fused
                        .preprocess
                        .clone()
                        .with_real_even_pack_input(RealEvenPackMapping {
                            full_len: self.length,
                        })?;
                fused.postprocess = fused
                    .postprocess
                    .clone()
                    .with_real_even_postprocess_output(RealEvenPostprocessMapping {
                        full_len: self.length,
                    })?;
            }
            RealFftKind::ComplexToReal => {
                let Some(RealPassOperation::PreprocessEvenHalf { normalize }) =
                    self.preprocess.as_ref().map(|pass| pass.operation)
                else {
                    return Ok(None);
                };
                if !matches!(
                    self.postprocess.as_ref().map(|pass| pass.operation),
                    Some(RealPassOperation::UnpackEvenOdd)
                ) {
                    return Ok(None);
                }
                fused.preprocess = fused
                    .preprocess
                    .clone()
                    .with_real_even_inverse_preprocess_input(RealEvenInversePreprocessMapping {
                        full_len: self.length,
                        normalize,
                    })?;
                fused.postprocess = fused.postprocess.clone().with_real_even_unpack_output(
                    RealEvenUnpackMapping {
                        full_len: self.length,
                    },
                )?;
            }
        }
        fused.validate()?;
        Ok(Some(fused))
    }

    pub fn validate(&self) -> Result<()> {
        self.transform.validate()?;
        if let Some(pass) = &self.preprocess {
            pass.validate()?;
        }
        if let Some(pass) = &self.postprocess {
            pass.validate()?;
        }
        if let Some(pass) = &self.zero_pad_pass {
            pass.validate()?;
            let (expected_operation, expected_storage) = match self.kind {
                RealFftKind::RealToComplex => (
                    crate::ZeroPadPassOperation::PrepareForwardInput,
                    (self.input_storage_scalar, self.scalar),
                ),
                RealFftKind::ComplexToReal => (
                    crate::ZeroPadPassOperation::FinalizeInverseOutput,
                    (self.scalar, self.output_storage_scalar),
                ),
            };
            if pass.logical_len != self.length
                || pass.batch_count != self.batch_count
                || pass.grouped_batch != self.grouped_batch
                || pass.scalar != self.scalar
                || pass.operation != expected_operation
                || (pass.input_storage_scalar, pass.output_storage_scalar) != expected_storage
            {
                return Err(VkFftError::InvalidKernelIr(
                    "real FFT zero-padding boundary metadata is inconsistent",
                ));
            }
        }
        let expected_direction = match self.kind {
            RealFftKind::RealToComplex => Direction::Forward,
            RealFftKind::ComplexToReal => Direction::Inverse,
        };
        let expected_transform_len = match self.algorithm {
            RealFftAlgorithm::FullComplex => self.length,
            RealFftAlgorithm::EvenHalfSize => self.length / 2,
        };
        let actual_passes = (
            self.preprocess.as_ref().map(|pass| pass.operation),
            self.postprocess.as_ref().map(|pass| pass.operation),
        );
        let pass_shape_matches = match (self.kind, self.algorithm) {
            (RealFftKind::RealToComplex, RealFftAlgorithm::FullComplex) => {
                matches!(
                    actual_passes,
                    (
                        None | Some(RealPassOperation::PromoteFullComplex),
                        Some(RealPassOperation::CompactHermitian)
                    )
                ) && (self.input_storage_scalar == self.scalar
                    || matches!(actual_passes.0, Some(RealPassOperation::PromoteFullComplex)))
            }
            (RealFftKind::ComplexToReal, RealFftAlgorithm::FullComplex) => {
                matches!(
                    actual_passes,
                    (
                        Some(RealPassOperation::ExpandHermitian),
                        None | Some(RealPassOperation::FinalizeFullComplexReal)
                    )
                ) && (self.output_storage_scalar == self.scalar
                    || matches!(
                        actual_passes.1,
                        Some(RealPassOperation::FinalizeFullComplexReal)
                    ))
            }
            (RealFftKind::RealToComplex, RealFftAlgorithm::EvenHalfSize) => matches!(
                actual_passes,
                (
                    Some(RealPassOperation::PackEvenOdd),
                    Some(RealPassOperation::PostprocessEvenHalf)
                )
            ),
            (RealFftKind::ComplexToReal, RealFftAlgorithm::EvenHalfSize) => matches!(
                actual_passes,
                (
                    Some(RealPassOperation::PreprocessEvenHalf { .. }),
                    Some(RealPassOperation::UnpackEvenOdd)
                )
            ),
        };
        let pass_metadata_matches =
            self.preprocess
                .iter()
                .chain(self.postprocess.iter())
                .all(|pass| {
                    pass.length == self.length
                        && pass.half_spectrum_len == self.half_spectrum_len
                        && pass.batch_count == self.batch_count
                        && pass.grouped_batch == self.grouped_batch
                        && pass.scalar == self.scalar
                });
        let boundary_storage_matches = match self.kind {
            RealFftKind::RealToComplex => {
                let input_ok = if self.zero_pad_pass.is_some() {
                    self.preprocess
                        .as_ref()
                        .is_none_or(|pass| pass.input_storage_scalar == self.scalar)
                } else {
                    self.preprocess
                        .as_ref()
                        .is_some_and(|pass| pass.input_storage_scalar == self.input_storage_scalar)
                        || (self.preprocess.is_none() && self.input_storage_scalar == self.scalar)
                };
                let output_ok = self
                    .postprocess
                    .as_ref()
                    .is_some_and(|pass| pass.output_storage_scalar == self.output_storage_scalar);
                input_ok && output_ok
            }
            RealFftKind::ComplexToReal => {
                let input_ok = self
                    .preprocess
                    .as_ref()
                    .is_some_and(|pass| pass.input_storage_scalar == self.input_storage_scalar);
                let output_ok = if self.zero_pad_pass.is_some() {
                    self.postprocess
                        .as_ref()
                        .is_none_or(|pass| pass.output_storage_scalar == self.scalar)
                } else {
                    self.postprocess.as_ref().is_some_and(|pass| {
                        pass.output_storage_scalar == self.output_storage_scalar
                    }) || (self.postprocess.is_none() && self.output_storage_scalar == self.scalar)
                };
                input_ok && output_ok
            }
        };
        if self.length == 0
            || (self.algorithm == RealFftAlgorithm::EvenHalfSize
                && (self.length < 2 || !self.length.is_multiple_of(2)))
            || self.half_spectrum_len != self.length / 2 + 1
            || self.batch_count == 0
            || self.grouped_batch == 0
            || self.transform.logical_len() != expected_transform_len
            || self.transform.batch_count() != self.batch_count
            || self.transform.direction() != expected_direction
            || self.transform.scalar() != self.scalar
            || self.transform.external_storage_scalar() != self.scalar
            || (self.external_scalar != self.scalar
                && !supported_real_storage_pair(self.scalar, self.external_scalar))
            || (self.input_storage_scalar != self.scalar
                && !supported_real_storage_pair(self.scalar, self.input_storage_scalar))
            || (self.output_storage_scalar != self.scalar
                && !supported_real_storage_pair(self.scalar, self.output_storage_scalar))
            || !pass_shape_matches
            || !pass_metadata_matches
            || !boundary_storage_matches
        {
            return Err(VkFftError::InvalidKernelIr(
                "real FFT composition metadata is inconsistent",
            ));
        }
        Ok(())
    }
}

pub fn execute_r2c_ir(ir: &RealFftIr, input: &[f64]) -> Result<Vec<Complex64>> {
    ir.validate()?;
    if ir.kind != RealFftKind::RealToComplex {
        return Err(VkFftError::UnsupportedKernelPath(
            "R2C executor requires a RealToComplex IR",
        ));
    }
    let expected = ir
        .length
        .checked_mul(ir.batch_count)
        .ok_or(VkFftError::ArithmeticOverflow {
            operation: "R2C input element count",
        })?;
    if input.len() != expected {
        return Err(VkFftError::InputLengthMismatch {
            expected,
            actual: input.len(),
        });
    }
    let mut padded_storage = None;
    if let Some(pass) = &ir.zero_pad_pass {
        let mut padded = input.to_vec();
        for batch in 0..ir.batch_count {
            let base = batch * ir.length;
            for index in pass.range.left..pass.range.right {
                padded[base + index] = 0.0;
            }
        }
        padded_storage = Some(padded);
    }
    let input = padded_storage.as_deref().unwrap_or(input);
    match ir.algorithm {
        RealFftAlgorithm::FullComplex => {
            let complex = input
                .iter()
                .map(|value| Complex64::new(*value, 0.0))
                .collect::<Vec<_>>();
            let full = execute_one_dim_fft_ir(&ir.transform, &complex)?;
            let mut output = Vec::with_capacity(ir.half_spectrum_len * ir.batch_count);
            for batch in 0..ir.batch_count {
                let base = batch * ir.length;
                output.extend_from_slice(&full[base..base + ir.half_spectrum_len]);
            }
            Ok(output)
        }
        RealFftAlgorithm::EvenHalfSize => execute_r2c_even_half(ir, input),
    }
}

fn execute_r2c_even_half(ir: &RealFftIr, input: &[f64]) -> Result<Vec<Complex64>> {
    let half = ir.length / 2;
    let transformed = if let Some(fused) = ir.fused_even_input_stockham_kernel()? {
        let external = input
            .iter()
            .map(|value| Complex64::new(*value, 0.0))
            .collect::<Vec<_>>();
        let fused_output = execute_stockham_ir(&fused, &external)?;
        if matches!(
            fused.output_modifier,
            crate::StockhamOutputModifier::RealEvenPostprocess(_)
        ) {
            return Ok(fused_output);
        }
        fused_output
    } else if let Some(fused) = ir.fused_even_recursive_ir()? {
        let external = input
            .iter()
            .map(|value| Complex64::new(*value, 0.0))
            .collect::<Vec<_>>();
        let fused_output = execute_recursive_fft_ir_with_resources(&fused, &external, None, None)?;
        if matches!(
            &fused.root,
            RecursiveFftNodeIr::CooleyTukey(root)
                if matches!(
                    root.scatter_output.output_modifier,
                    crate::recursive_ir::CooleyTukeyOutputModifier::RealEvenPostprocess(_)
                )
        ) {
            return Ok(fused_output);
        }
        fused_output
    } else if let Some(fused) = ir.fused_even_bluestein_ir()? {
        let external = input
            .iter()
            .map(|value| Complex64::new(*value, 0.0))
            .collect::<Vec<_>>();
        return execute_bluestein_ir(&fused, &external);
    } else {
        let mut packed = vec![Complex64::default(); half * ir.batch_count];
        for batch in 0..ir.batch_count {
            let input_base = batch * ir.length;
            let packed_base = batch * half;
            for n in 0..half {
                packed[packed_base + n] =
                    Complex64::new(input[input_base + 2 * n], input[input_base + 2 * n + 1]);
            }
        }
        execute_one_dim_fft_ir(&ir.transform, &packed)?
    };
    let mut output = vec![Complex64::default(); ir.half_spectrum_len * ir.batch_count];
    for batch in 0..ir.batch_count {
        let transformed_base = batch * half;
        let output_base = batch * ir.half_spectrum_len;
        for k in 0..=half {
            let a = transformed[transformed_base + (k % half)];
            let b = transformed[transformed_base + ((half - k) % half)].conj();
            let w = Complex64::exp_i(-TAU * k as f64 / ir.length as f64);
            let rotated = w * (a - b);
            let minus_i_rotated = Complex64::new(rotated.im, -rotated.re);
            output[output_base + k] = (a + b + minus_i_rotated).scale(0.5);
        }
    }
    Ok(output)
}

pub fn execute_c2r_ir(ir: &RealFftIr, input: &[Complex64]) -> Result<Vec<f64>> {
    ir.validate()?;
    if ir.kind != RealFftKind::ComplexToReal {
        return Err(VkFftError::UnsupportedKernelPath(
            "C2R executor requires a ComplexToReal IR",
        ));
    }
    let expected =
        ir.half_spectrum_len
            .checked_mul(ir.batch_count)
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "C2R input element count",
            })?;
    if input.len() != expected {
        return Err(VkFftError::InputLengthMismatch {
            expected,
            actual: input.len(),
        });
    }
    let mut output = match ir.algorithm {
        RealFftAlgorithm::FullComplex => {
            let mut full = vec![Complex64::new(0.0, 0.0); ir.length * ir.batch_count];
            for batch in 0..ir.batch_count {
                let half_base = batch * ir.half_spectrum_len;
                let full_base = batch * ir.length;
                for k in 0..ir.length {
                    full[full_base + k] = if k < ir.half_spectrum_len {
                        input[half_base + k]
                    } else {
                        input[half_base + ir.length - k].conj()
                    };
                }
            }
            let real = execute_one_dim_fft_ir(&ir.transform, &full)?;
            real.into_iter().map(|value| value.re).collect()
        }
        RealFftAlgorithm::EvenHalfSize => execute_c2r_even_half(ir, input)?,
    };
    if let Some(pass) = &ir.zero_pad_pass {
        for batch in 0..ir.batch_count {
            let base = batch * ir.length;
            for index in pass.range.left..pass.range.right {
                output[base + index] = 0.0;
            }
        }
    }
    Ok(output)
}

fn execute_c2r_even_half(ir: &RealFftIr, input: &[Complex64]) -> Result<Vec<f64>> {
    let half = ir.length / 2;
    if let Some(fused) = ir.fused_even_input_stockham_kernel()? {
        let unpacked = execute_stockham_ir(&fused, input)?;
        return Ok(unpacked.into_iter().map(|value| value.re).collect());
    }
    if let Some(fused) = ir.fused_even_recursive_ir()? {
        let unpacked = execute_recursive_fft_ir_with_resources(&fused, input, None, None)?;
        return Ok(unpacked.into_iter().map(|value| value.re).collect());
    }
    if let Some(fused) = ir.fused_even_bluestein_ir()? {
        let unpacked = execute_bluestein_ir(&fused, input)?;
        return Ok(unpacked.into_iter().map(|value| value.re).collect());
    }
    let packed = {
        let normalize = matches!(
            ir.preprocess.as_ref().map(|pass| pass.operation),
            Some(RealPassOperation::PreprocessEvenHalf { normalize: true })
        );
        let reconstruction_scale = if normalize { 0.5 } else { 1.0 };
        let mut packed_spectrum = vec![Complex64::default(); half * ir.batch_count];
        for batch in 0..ir.batch_count {
            let input_base = batch * ir.half_spectrum_len;
            let packed_base = batch * half;
            for k in 0..half {
                let x = input[input_base + k];
                let mirrored = input[input_base + (half - k)].conj();
                let w_conj = Complex64::exp_i(TAU * k as f64 / ir.length as f64);
                let rotated = w_conj * (x - mirrored);
                let i_rotated = Complex64::new(-rotated.im, rotated.re);
                packed_spectrum[packed_base + k] =
                    (x + mirrored + i_rotated).scale(reconstruction_scale);
            }
        }
        execute_one_dim_fft_ir(&ir.transform, &packed_spectrum)?
    };
    let mut output = vec![0.0; ir.length * ir.batch_count];
    for batch in 0..ir.batch_count {
        let packed_base = batch * half;
        let output_base = batch * ir.length;
        for n in 0..half {
            let value = packed[packed_base + n];
            output[output_base + 2 * n] = value.re;
            output[output_base + 2 * n + 1] = value.im;
        }
    }
    Ok(output)
}

fn supported_real_storage_pair(compute: ScalarType, storage: ScalarType) -> bool {
    compute == storage
        || matches!(
            (compute, storage),
            (ScalarType::F32, ScalarType::F16) | (ScalarType::F64, ScalarType::F32)
        )
}

pub(crate) fn real_fft_algorithm_for_shape_policy(
    length: usize,
    batch_count: usize,
    compute_precision: Precision,
    tuning: PlannerTuning,
    device: DeviceProfile,
    shape_policy: RealFftShapePolicy,
) -> Result<RealFftAlgorithm> {
    if length < 2 || !length.is_multiple_of(2) {
        return Ok(RealFftAlgorithm::FullComplex);
    }
    if shape_policy == RealFftShapePolicy::PortableHalfSizeAllEven
        || !has_fixed_upstream_gpu_scheduler_profile(device)
    {
        return Ok(RealFftAlgorithm::EvenHalfSize);
    }
    // Fixed upstream leaves Apple real transforms on the callback/full-size path
    // instead of enabling bigSequenceEvenR2C.
    if device.vendor == GpuVendor::Apple {
        return Ok(RealFftAlgorithm::FullComplex);
    }

    let complex_bytes = compute_precision.compute_complex_bytes();
    let used_shared_memory = if length.is_power_of_two() {
        device.shared_memory_pow2_bytes
    } else {
        device.shared_memory_bytes
    };
    let max_single_size_non_strided = used_shared_memory / complex_bytes;
    if length > max_single_size_non_strided {
        return Ok(RealFftAlgorithm::EvenHalfSize);
    }

    // Upstream re-checks bigSequenceEvenR2C after its initial Stockham/Rader
    // factor scan. Reuse the same device-scored C2C classifier here rather than
    // maintaining a second prime-selection implementation. The Rust-only
    // recursive/sub-Rader extension is disabled for this shape decision because
    // fixed upstream would leave such a sequence for the Bluestein phase instead.
    let mut upstream_tuning = tuning;
    upstream_tuning.allow_recursive_fft_rader = false;
    let scoring_plan = FftPlan::build_c2c_child_for_device(
        FftConfig::new(vec![length])
            .with_precision(compute_precision)
            .with_tuning(upstream_tuning),
        device,
        C2cDeviceAxisClass::Contiguous,
    )?;
    let primes = match &scoring_plan.axes[0].algorithm {
        AxisAlgorithm::Stockham { .. } => return Ok(RealFftAlgorithm::FullComplex),
        AxisAlgorithm::Bluestein { .. } => {
            // After Bluestein auto-padding, fixed upstream performs a third R2C
            // check against the non-strided shared-memory capacity. This uses
            // the fixed auto-padding table first and then the exact ordinary
            // register-goodness search rather than the crate's execution padding.
            let padded =
                upstream_bluestein_auto_padding(length, batch_count, compute_precision, device)?;
            return Ok(if padded > max_single_size_non_strided {
                RealFftAlgorithm::EvenHalfSize
            } else {
                RealFftAlgorithm::FullComplex
            });
        }
        AxisAlgorithm::Rader { primes, .. } => primes,
    };

    let largest_direct_prime = primes
        .iter()
        .filter(|prime| matches!(prime.mode, RaderMode::DirectMultiplication))
        .map(|prime| prime.prime)
        .max();
    let reserved_shared_memory = largest_direct_prime
        .and_then(|prime| (prime - 1).checked_mul(complex_bytes))
        .map_or(used_shared_memory, |reservation| {
            used_shared_memory.saturating_sub(reservation)
        });
    let reserved_limit = reserved_shared_memory / complex_bytes;

    let fft_rader_primes = primes
        .iter()
        .filter(|prime| matches!(prime.mode, RaderMode::FftConvolution { .. }))
        .map(|prime| prime.prime)
        .collect::<Vec<_>>();
    let force_two_upload = plan_gpu_force_rader_two_upload(length, &fft_rader_primes, device)?;

    if length > reserved_limit || force_two_upload {
        Ok(RealFftAlgorithm::EvenHalfSize)
    } else {
        Ok(RealFftAlgorithm::FullComplex)
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

    fn max_complex_error(lhs: &[Complex64], rhs: &[Complex64]) -> f64 {
        lhs.iter()
            .zip(rhs)
            .map(|(lhs, rhs)| (*lhs - *rhs).norm_sqr().sqrt())
            .fold(0.0, f64::max)
    }

    #[test]
    fn real_internal_c2c_child_preserves_device_scored_contiguous_context() {
        let base = DeviceProfile::generic(Backend::Vulkan, GpuVendor::Nvidia);

        // Even N=2306 uses the existing half-size real reduction with a p1153
        // complex child. p1153 exceeds the 1024 strided prime limit but fits the
        // 4096-complex contiguous 32 KiB budget, so the real axis must keep its
        // internal child in the contiguous Rader family.
        let plan =
            FftPlan::build(FftConfig::new(vec![2306]).with_transform(TransformKind::RealToComplex))
                .unwrap();
        let ir = RealFftIr::build(&plan, base).unwrap();
        assert_eq!(ir.transform_len(), 1153);
        assert!(matches!(ir.transform, OneDimFftIr::Recursive(_)));

        // The direct-Rader physical cap is also applied inside the real child.
        // N=166 -> p83: on a 128-thread F32 profile the cap is 63, so p83 falls
        // to Bluestein; F64 uses a 16-byte compute complex and retains p83 direct.
        let constrained = DeviceProfile {
            max_threads_per_block: 128,
            max_workgroup_size: [128, 128, 64],
            ..base
        };
        let f32_plan =
            FftPlan::build(FftConfig::new(vec![166]).with_transform(TransformKind::RealToComplex))
                .unwrap();
        let f32_ir = RealFftIr::build(&f32_plan, constrained).unwrap();
        assert!(matches!(f32_ir.transform, OneDimFftIr::Bluestein(_)));

        let f64_plan = FftPlan::build(
            FftConfig::new(vec![166])
                .with_transform(TransformKind::RealToComplex)
                .with_precision(Precision::F64),
        )
        .unwrap();
        let f64_ir = RealFftIr::build(&f64_plan, constrained).unwrap();
        let OneDimFftIr::Recursive(recursive) = &f64_ir.transform else {
            panic!("F64 p83 real child should remain recursive direct-Rader");
        };
        assert!(matches!(
            recursive.root,
            crate::recursive_ir::RecursiveFftNodeIr::DirectRader(_)
        ));
    }

    #[test]
    fn fixed_device_f16_full_complex_real_child_keeps_bluestein_selection() {
        for shared in [4 * 1024usize, 8 * 1024] {
            let device = DeviceProfile {
                shared_memory_bytes: shared,
                shared_memory_pow2_bytes: shared,
                max_threads_per_block: 128,
                max_workgroup_size: [128, 128, 64],
                ..DeviceProfile::generic(Backend::Vulkan, GpuVendor::Nvidia)
            };
            let plan = FftPlan::build_for_device(
                FftConfig::new(vec![94])
                    .with_precision(Precision::F16StorageF32Compute)
                    .with_transform(TransformKind::RealToComplex),
                device,
            )
            .unwrap();
            let ir = RealFftIr::build_for_device_plan(&plan, device).unwrap();
            assert_eq!(ir.algorithm, RealFftAlgorithm::FullComplex);
            assert_eq!(ir.scalar, ScalarType::F32);
            assert_eq!(ir.external_scalar, ScalarType::F16);
            let OneDimFftIr::Bluestein(bluestein) = &ir.transform else {
                panic!("F16 N94/128-thread real child should use upstream Bluestein");
            };
            assert_eq!(bluestein.logical_len, 94);
            assert_eq!(bluestein.convolution_len, 256);
            assert_eq!(bluestein.scalar, ScalarType::F32);
            assert_eq!(bluestein.external_scalar, ScalarType::F32);
            assert_eq!(bluestein.forward_fft.scalar, ScalarType::F32);
            assert_eq!(bluestein.forward_fft.external_scalar, ScalarType::F32);
            assert_eq!(bluestein.inverse_fft.scalar, ScalarType::F32);
            assert_eq!(bluestein.inverse_fft.external_scalar, ScalarType::F32);
        }
    }

    #[test]
    fn r2c_and_c2r_match_full_dft_and_round_trip() {
        for length in [2usize, 4, 8, 30, 34, 60, 94, 206, 17, 103] {
            let batch_count = 2usize;
            let mut tuning = crate::PlannerTuning::portable();
            if length == 103 || length == 206 {
                tuning.max_rader_fft_prime = 100;
            }
            let input = (0..length * batch_count)
                .map(|index| {
                    let x = index as f64;
                    (0.17 * x).sin() + 0.3 * (0.07 * x).cos() + x * 0.001
                })
                .collect::<Vec<_>>();
            let forward_plan = FftPlan::build(
                FftConfig::new(vec![length])
                    .with_batch_count(batch_count)
                    .with_transform(TransformKind::RealToComplex)
                    .with_tuning(tuning),
            )
            .unwrap();
            let forward = RealFftIr::build(&forward_plan, device()).unwrap();
            if length.is_multiple_of(2) {
                assert_eq!(forward.algorithm, RealFftAlgorithm::EvenHalfSize);
                assert_eq!(forward.transform_len(), length / 2);
                assert!(matches!(
                    forward.preprocess.as_ref().map(|pass| pass.operation),
                    Some(RealPassOperation::PackEvenOdd)
                ));
                assert!(matches!(
                    forward.postprocess.as_ref().map(|pass| pass.operation),
                    Some(RealPassOperation::PostprocessEvenHalf)
                ));
            } else {
                assert_eq!(forward.algorithm, RealFftAlgorithm::FullComplex);
                assert_eq!(forward.transform_len(), length);
            }
            if length == 103 || length == 206 {
                assert!(matches!(
                    forward.transform,
                    crate::OneDimFftIr::Bluestein(_)
                ));
            }
            if length == 206 {
                let fused = forward.fused_even_bluestein_ir().unwrap().unwrap();
                assert!(matches!(
                    fused.preprocess.input_modifier,
                    crate::bluestein_ir::BluesteinInputModifier::RealEvenPack(_)
                ));
                assert!(matches!(
                    fused.postprocess.output_modifier,
                    crate::bluestein_ir::BluesteinOutputModifier::RealEvenPostprocess(_)
                ));
            }
            let spectrum = execute_r2c_ir(&forward, &input).unwrap();
            for batch in 0..batch_count {
                let start = batch * length;
                let complex = input[start..start + length]
                    .iter()
                    .map(|value| Complex64::new(*value, 0.0))
                    .collect::<Vec<_>>();
                let expected = dft(&complex, Direction::Forward, false);
                let half_start = batch * forward.half_spectrum_len;
                assert!(
                    max_complex_error(
                        &spectrum[half_start..half_start + forward.half_spectrum_len],
                        &expected[..forward.half_spectrum_len],
                    ) < 2.0e-9 * length as f64
                );
            }

            let inverse_plan = FftPlan::build(
                FftConfig::new(vec![length])
                    .with_batch_count(batch_count)
                    .with_transform(TransformKind::ComplexToReal)
                    .with_inverse_normalization(true)
                    .with_tuning(tuning),
            )
            .unwrap();
            let inverse = RealFftIr::build(&inverse_plan, device()).unwrap();
            if length == 206 {
                let fused = inverse.fused_even_bluestein_ir().unwrap().unwrap();
                assert!(matches!(
                    fused.preprocess.input_modifier,
                    crate::bluestein_ir::BluesteinInputModifier::RealEvenInversePreprocess(
                        RealEvenInversePreprocessMapping {
                            full_len: 206,
                            normalize: true
                        }
                    )
                ));
                assert!(matches!(
                    fused.postprocess.output_modifier,
                    crate::bluestein_ir::BluesteinOutputModifier::RealEvenUnpack(_)
                ));
            }
            let restored = execute_c2r_ir(&inverse, &spectrum).unwrap();
            let max_error = restored
                .iter()
                .zip(&input)
                .map(|(actual, expected)| (actual - expected).abs())
                .fold(0.0, f64::max);
            assert!(max_error < 2.0e-9 * length as f64);
        }
    }

    #[test]
    fn grouped_even_real_stockham_fusion_keeps_xy_block_and_tail_safe_epilogue() {
        let length = 64usize;
        let batch_count = 7usize;
        let grouped_batch = 3usize;
        let input = (0..length * batch_count)
            .map(|index| {
                let x = index as f64;
                (0.071 * x).sin() + 0.25 * (0.019 * x).cos() + x * 0.0002
            })
            .collect::<Vec<_>>();

        let forward_plan = FftPlan::build(
            FftConfig::new(vec![length])
                .with_batch_count(batch_count)
                .with_grouped_batch(0, grouped_batch)
                .unwrap()
                .with_transform(TransformKind::RealToComplex),
        )
        .unwrap();
        let forward = RealFftIr::build(&forward_plan, device()).unwrap();
        let fused = forward
            .fused_even_input_stockham_kernel()
            .unwrap()
            .expect("grouped N64 R2C should fuse around its N32 Stockham child");
        assert_eq!(
            fused.workgroup_grouping.transforms_per_workgroup,
            grouped_batch
        );
        assert_eq!(fused.workgroup_grouping.threads_per_transform, 4);
        assert_eq!([fused.workgroup_size.x, fused.workgroup_size.y], [3, 4]);
        assert_eq!(fused.dispatch.x, 3);
        assert!(matches!(
            fused.io_mapping,
            StockhamIoMapping::RealEvenPack(_)
        ));
        assert!(matches!(
            fused.output_modifier,
            crate::StockhamOutputModifier::RealEvenPostprocess(_)
        ));
        let program = crate::ProgramIr::real_fft(&forward).unwrap();
        assert_eq!(program.passes.len(), 1);
        let shaders = crate::backend::vulkan::VulkanGlslBackend
            .lower_real_fft(&forward)
            .unwrap();
        assert_eq!(shaders.len(), 1);
        assert_eq!(
            [shaders[0].workgroup_size.x, shaders[0].workgroup_size.y],
            [3, 4]
        );
        assert_eq!(shaders[0].dispatch.x, 3);
        assert!(shaders[0].glsl.contains("vkfft_batch_active"));
        assert!(
            shaders[0]
                .glsl
                .contains("for (uint vkfft_real_k = vkfft_lid")
        );
        assert!(shaders[0].glsl.contains("VKFFT_THREADS_PER_TRANSFORM"));
        assert_eq!(shaders[0].compile_spirv().unwrap().words[0], 0x0723_0203);

        let spectrum = execute_r2c_ir(&forward, &input).unwrap();
        let inverse_plan = FftPlan::build(
            FftConfig::new(vec![length])
                .with_batch_count(batch_count)
                .with_grouped_batch(0, grouped_batch)
                .unwrap()
                .with_transform(TransformKind::ComplexToReal)
                .with_inverse_normalization(true),
        )
        .unwrap();
        let inverse = RealFftIr::build(&inverse_plan, device()).unwrap();
        let fused_inverse = inverse
            .fused_even_input_stockham_kernel()
            .unwrap()
            .expect("grouped N64 C2R should fuse around its N32 Stockham child");
        assert_eq!(
            fused_inverse.workgroup_grouping.transforms_per_workgroup,
            grouped_batch
        );
        assert_eq!(
            [
                fused_inverse.workgroup_size.x,
                fused_inverse.workgroup_size.y,
            ],
            [3, 4]
        );
        assert_eq!(fused_inverse.dispatch.x, 3);
        assert!(matches!(
            fused_inverse.io_mapping,
            StockhamIoMapping::RealEvenInversePreprocess(_)
        ));
        assert!(matches!(
            fused_inverse.output_modifier,
            crate::StockhamOutputModifier::RealEvenUnpack(_)
        ));
        let restored = execute_c2r_ir(&inverse, &spectrum).unwrap();
        let error = restored
            .iter()
            .zip(&input)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0, f64::max);
        assert!(
            error < 2.0e-9 * length as f64,
            "grouped real fusion error {error}"
        );
        forward.validate().unwrap();
        inverse.validate().unwrap();
    }

    #[test]
    fn grouped_even_real_recursive_fusion_keeps_parent_boundary_ownership() {
        let length = 68usize;
        let batch_count = 5usize;
        let grouped_batch = 3usize;
        let input = (0..length * batch_count)
            .map(|index| {
                let x = index as f64;
                (0.031 * x).sin() + 0.17 * (0.013 * x).cos() + x * 0.0001
            })
            .collect::<Vec<_>>();

        let forward_plan = FftPlan::build(
            FftConfig::new(vec![length])
                .with_batch_count(batch_count)
                .with_grouped_batch(0, grouped_batch)
                .unwrap()
                .with_transform(TransformKind::RealToComplex),
        )
        .unwrap();
        let forward = RealFftIr::build(&forward_plan, device()).unwrap();
        assert!(
            forward
                .fused_even_input_stockham_kernel()
                .unwrap()
                .is_none()
        );
        let fused = forward
            .fused_even_recursive_ir()
            .unwrap()
            .expect("grouped N68 R2C should fuse into its N34 recursive root");
        let RecursiveFftNodeIr::CooleyTukey(root) = &fused.root else {
            panic!("grouped N68 R2C half child should keep a Cooley root");
        };
        let block = root
            .pack_right
            .axis_batch_block
            .expect("grouped recursive real root should retain parent batch ownership");
        assert_eq!(block.grouped_batch, grouped_batch);
        assert_eq!(root.pack_right.dispatch.x, 2);
        assert_eq!(root.twiddle_transpose.axis_batch_block, Some(block));
        assert_eq!(root.scatter_output.axis_batch_block, Some(block));
        assert!(matches!(
            root.pack_right.input_modifier,
            crate::recursive_ir::CooleyTukeyInputModifier::RealEvenPack(_)
        ));
        assert!(matches!(
            root.scatter_output.output_modifier,
            crate::recursive_ir::CooleyTukeyOutputModifier::RealEvenPostprocess(_)
        ));
        let program = crate::ProgramIr::real_fft(&forward).unwrap();
        assert!(program.passes.iter().all(|pass| {
            !pass.name.contains("pack_even_odd") && !pass.name.contains("postprocess_even_half")
        }));
        let shaders = crate::backend::vulkan::VulkanGlslBackend
            .lower_real_fft(&forward)
            .unwrap();
        assert_eq!(shaders.len(), program.passes.len());
        assert!(shaders.iter().any(|shader| {
            shader.dispatch.x == 2
                && shader.glsl.contains("VKFFT_GROUPED_BATCH = 3u")
                && shader.glsl.contains("VKFFT_REAL_FULL_N")
        }));
        for shader in shaders {
            assert_eq!(shader.compile_spirv().unwrap().words[0], 0x0723_0203);
        }

        let spectrum = execute_r2c_ir(&forward, &input).unwrap();
        let inverse_plan = FftPlan::build(
            FftConfig::new(vec![length])
                .with_batch_count(batch_count)
                .with_grouped_batch(0, grouped_batch)
                .unwrap()
                .with_transform(TransformKind::ComplexToReal)
                .with_inverse_normalization(true),
        )
        .unwrap();
        let inverse = RealFftIr::build(&inverse_plan, device()).unwrap();
        let fused_inverse = inverse
            .fused_even_recursive_ir()
            .unwrap()
            .expect("grouped N68 C2R should fuse into its N34 recursive root");
        let RecursiveFftNodeIr::CooleyTukey(root) = &fused_inverse.root else {
            panic!("grouped N68 C2R half child should keep a Cooley root");
        };
        assert_eq!(
            root.pack_right.axis_batch_block.unwrap().grouped_batch,
            grouped_batch
        );
        assert!(matches!(
            root.pack_right.input_modifier,
            crate::recursive_ir::CooleyTukeyInputModifier::RealEvenInversePreprocess(_)
        ));
        assert!(matches!(
            root.scatter_output.output_modifier,
            crate::recursive_ir::CooleyTukeyOutputModifier::RealEvenUnpack(_)
        ));
        let restored = execute_c2r_ir(&inverse, &spectrum).unwrap();
        let error = restored
            .iter()
            .zip(&input)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0, f64::max);
        assert!(
            error < 3.0e-8 * length as f64,
            "grouped recursive real fusion error {error}"
        );
        forward.validate().unwrap();
        inverse.validate().unwrap();
    }

    #[test]
    fn grouped_even_real_bluestein_fusion_owns_axis0_wrapper_block() {
        let length = 206usize;
        let batch_count = 5usize;
        let grouped_batch = 3usize;
        let mut profile = device();
        profile.max_threads_per_block = 1024;
        profile.max_workgroup_size = [1024, 1024, 64];
        let mut tuning = crate::PlannerTuning::portable();
        tuning.max_rader_fft_prime = 100;
        let input = (0..length * batch_count)
            .map(|index| {
                let x = index as f64;
                (0.027 * x).sin() + 0.13 * (0.017 * x).cos() + x * 0.0001
            })
            .collect::<Vec<_>>();

        let forward_plan = FftPlan::build(
            FftConfig::new(vec![length])
                .with_batch_count(batch_count)
                .with_grouped_batch(0, grouped_batch)
                .unwrap()
                .with_transform(TransformKind::RealToComplex)
                .with_tuning(tuning),
        )
        .unwrap();
        let forward = RealFftIr::build(&forward_plan, profile).unwrap();
        let OneDimFftIr::Bluestein(base_pipeline) = &forward.transform else {
            panic!("forced N206 half-size child should use p103 Bluestein");
        };
        let base_block = base_pipeline
            .preprocess
            .axis_batch_block
            .expect("explicit G3 Bluestein should own an axis0 wrapper block");
        assert_eq!(base_block.grouped_batch, grouped_batch);
        assert_eq!(base_block.threads_per_transform, 128);
        assert!(!base_block.transforms_on_x);
        assert_eq!([base_block.local_size_x, base_block.local_size_y], [128, 3]);
        assert_eq!(base_pipeline.preprocess.dispatch.x, 2);
        assert_eq!(base_pipeline.multiply.axis_batch_block, Some(base_block));
        assert_eq!(base_pipeline.postprocess.axis_batch_block, Some(base_block));

        let fused = forward
            .fused_even_bluestein_ir()
            .unwrap()
            .expect("grouped N206 R2C should fuse around p103 Bluestein");
        assert_eq!(fused.preprocess.axis_batch_block, Some(base_block));
        assert_eq!(fused.postprocess.axis_batch_block, Some(base_block));
        assert!(matches!(
            fused.preprocess.input_modifier,
            crate::bluestein_ir::BluesteinInputModifier::RealEvenPack(_)
        ));
        assert!(matches!(
            fused.postprocess.output_modifier,
            crate::bluestein_ir::BluesteinOutputModifier::RealEvenPostprocess(_)
        ));
        let program = crate::ProgramIr::real_fft(&forward).unwrap();
        assert!(program.passes.iter().all(|pass| {
            !pass.name.contains("pack_even_odd") && !pass.name.contains("postprocess_even_half")
        }));
        let shaders = crate::backend::vulkan::VulkanGlslBackend
            .lower_real_fft(&forward)
            .unwrap();
        assert_eq!(shaders.len(), program.passes.len());
        assert!(shaders.iter().any(|shader| {
            shader.workgroup_size.x == 128
                && shader.workgroup_size.y == 3
                && shader.dispatch.x == 2
                && shader.glsl.contains("vkfft_transform_slot")
                && shader.glsl.contains("VKFFT_REAL_FULL_N")
        }));
        for shader in shaders {
            assert_eq!(shader.compile_spirv().unwrap().words[0], 0x0723_0203);
        }

        let spectrum = execute_r2c_ir(&forward, &input).unwrap();
        let inverse_plan = FftPlan::build(
            FftConfig::new(vec![length])
                .with_batch_count(batch_count)
                .with_grouped_batch(0, grouped_batch)
                .unwrap()
                .with_transform(TransformKind::ComplexToReal)
                .with_inverse_normalization(true)
                .with_tuning(tuning),
        )
        .unwrap();
        let inverse = RealFftIr::build(&inverse_plan, profile).unwrap();
        let fused_inverse = inverse
            .fused_even_bluestein_ir()
            .unwrap()
            .expect("grouped N206 C2R should fuse around p103 Bluestein");
        assert_eq!(fused_inverse.preprocess.axis_batch_block, Some(base_block));
        assert!(matches!(
            fused_inverse.preprocess.input_modifier,
            crate::bluestein_ir::BluesteinInputModifier::RealEvenInversePreprocess(_)
        ));
        assert!(matches!(
            fused_inverse.postprocess.output_modifier,
            crate::bluestein_ir::BluesteinOutputModifier::RealEvenUnpack(_)
        ));
        let restored = execute_c2r_ir(&inverse, &spectrum).unwrap();
        let error = restored
            .iter()
            .zip(&input)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0, f64::max);
        assert!(
            error < 5.0e-8 * length as f64,
            "grouped Bluestein real fusion error {error}"
        );
        forward.validate().unwrap();
        inverse.validate().unwrap();
    }

    #[test]
    fn grouped_real_odd_even_owns_explicit_tail_boundaries() {
        let batch_count = 7usize;
        let grouped_batch = 3usize;
        for length in [15usize, 16] {
            let input = (0..length * batch_count)
                .map(|index| {
                    let x = index as f64;
                    (0.071 * x).sin() + 0.25 * (0.019 * x).cos()
                })
                .collect::<Vec<_>>();
            let forward_plan = FftPlan::build(
                FftConfig::new(vec![length])
                    .with_batch_count(batch_count)
                    .with_grouped_batch(0, grouped_batch)
                    .unwrap()
                    .with_transform(TransformKind::RealToComplex)
                    .with_precision(Precision::F16StorageF32Compute)
                    .with_zero_padding(0, 3, 7)
                    .unwrap(),
            )
            .unwrap();
            let forward = RealFftIr::build(&forward_plan, device()).unwrap();
            assert_eq!(forward.grouped_batch, grouped_batch);
            assert!(
                forward
                    .fused_even_input_stockham_kernel()
                    .unwrap()
                    .is_none()
            );
            assert!(forward.fused_even_recursive_ir().unwrap().is_none());
            assert!(forward.fused_even_bluestein_ir().unwrap().is_none());
            for pass in forward.preprocess.iter().chain(forward.postprocess.iter()) {
                assert_eq!(pass.grouped_batch, grouped_batch);
                assert_eq!(pass.dispatch.x, 3);
            }
            let zero = forward.zero_pad_pass.as_ref().unwrap();
            assert_eq!(zero.grouped_batch, grouped_batch);
            assert_eq!(zero.dispatch.x, 3);
            let actual = execute_r2c_ir(&forward, &input).unwrap();
            let mut manual = input.clone();
            for batch in 0..batch_count {
                let base = batch * length;
                manual[base + 3..base + 7].fill(0.0);
            }
            let baseline_plan = FftPlan::build(
                FftConfig::new(vec![length])
                    .with_batch_count(batch_count)
                    .with_transform(TransformKind::RealToComplex),
            )
            .unwrap();
            let baseline = RealFftIr::build(&baseline_plan, device()).unwrap();
            let expected = execute_r2c_ir(&baseline, &manual).unwrap();
            assert!(max_complex_error(&actual, &expected) < 1.0e-9 * length as f64);

            let inverse_plan = FftPlan::build(
                FftConfig::new(vec![length])
                    .with_batch_count(batch_count)
                    .with_grouped_batch(0, grouped_batch)
                    .unwrap()
                    .with_transform(TransformKind::ComplexToReal)
                    .with_precision(Precision::F16StorageF32Compute)
                    .with_inverse_normalization(true)
                    .with_zero_padding(0, 3, 7)
                    .unwrap(),
            )
            .unwrap();
            let inverse = RealFftIr::build(&inverse_plan, device()).unwrap();
            assert_eq!(inverse.grouped_batch, grouped_batch);
            for pass in inverse.preprocess.iter().chain(inverse.postprocess.iter()) {
                assert_eq!(pass.grouped_batch, grouped_batch);
                assert_eq!(pass.dispatch.x, 3);
            }
            assert_eq!(inverse.zero_pad_pass.as_ref().unwrap().dispatch.x, 3);
            let restored = execute_c2r_ir(&inverse, &actual).unwrap();
            for batch in 0..batch_count {
                let base = batch * length;
                assert!(
                    restored[base + 3..base + 7]
                        .iter()
                        .all(|value| *value == 0.0)
                );
                for index in [0usize, 1, length - 1] {
                    assert!((restored[base + index] - manual[base + index]).abs() < 2.0e-8);
                }
            }
            forward.validate().unwrap();
            inverse.validate().unwrap();
        }
    }

    #[test]
    fn real_spatial_zero_padding_matches_manual_zero_and_finalizes_inverse() {
        let length = 34usize;
        let batch_count = 2usize;
        let left = 7usize;
        let right = 13usize;
        let mut input = (0..length * batch_count)
            .map(|index| {
                let x = index as f64;
                (0.17 * x).sin() + 0.23 * (0.041 * x).cos()
            })
            .collect::<Vec<_>>();
        for batch in 0..batch_count {
            let base = batch * length;
            for index in left..right {
                input[base + index] = 1.0e6 + (base + index) as f64;
            }
        }
        let mut manual = input.clone();
        for batch in 0..batch_count {
            manual[batch * length + left..batch * length + right].fill(0.0);
        }

        let padded_plan = FftPlan::build(
            FftConfig::new(vec![length])
                .with_batch_count(batch_count)
                .with_transform(TransformKind::RealToComplex)
                .with_zero_padding(0, left, right)
                .unwrap(),
        )
        .unwrap();
        let padded = RealFftIr::build(&padded_plan, device()).unwrap();
        assert!(padded.zero_pad_pass.is_some());
        let actual = execute_r2c_ir(&padded, &input).unwrap();

        let baseline_plan = FftPlan::build(
            FftConfig::new(vec![length])
                .with_batch_count(batch_count)
                .with_transform(TransformKind::RealToComplex),
        )
        .unwrap();
        let baseline = RealFftIr::build(&baseline_plan, device()).unwrap();
        let expected = execute_r2c_ir(&baseline, &manual).unwrap();
        assert!(max_complex_error(&actual, &expected) < 1.0e-10 * length as f64);

        let inverse_plan = FftPlan::build(
            FftConfig::new(vec![length])
                .with_batch_count(batch_count)
                .with_transform(TransformKind::ComplexToReal)
                .with_inverse_normalization(true)
                .with_zero_padding(0, left, right)
                .unwrap(),
        )
        .unwrap();
        let inverse = RealFftIr::build(&inverse_plan, device()).unwrap();
        let restored = execute_c2r_ir(&inverse, &actual).unwrap();
        for batch in 0..batch_count {
            let base = batch * length;
            for index in 0..length {
                let expected = if (left..right).contains(&index) {
                    0.0
                } else {
                    manual[base + index]
                };
                assert!((restored[base + index] - expected).abs() < 1.0e-8 * length as f64);
            }
        }
    }

    #[test]
    fn bluestein_even_real_fusion_propagates_recursive_convolution() {
        let length = 4106usize;
        let mut tuning = crate::PlannerTuning::portable();
        tuning.max_rader_fft_prime = 100;
        let mut constrained = device();
        constrained.shared_memory_bytes = 32 * 1024;
        constrained.shared_memory_pow2_bytes = 32 * 1024;
        let forward_plan = FftPlan::build(
            FftConfig::new(vec![length])
                .with_transform(TransformKind::RealToComplex)
                .with_tuning(tuning),
        )
        .unwrap();
        let forward = RealFftIr::build(&forward_plan, constrained).unwrap();
        let fused = forward.fused_even_bluestein_ir().unwrap().unwrap();
        assert_eq!(fused.logical_len, 2053);
        assert_eq!(fused.convolution_len, 4368);
        assert!(matches!(
            fused.forward_fft.root,
            RecursiveFftNodeIr::CooleyTukey(_)
        ));
        assert!(matches!(
            fused.inverse_fft.root,
            RecursiveFftNodeIr::CooleyTukey(_)
        ));

        let mut input = vec![0.0f64; length];
        input[0] = 1.0;
        let spectrum = execute_r2c_ir(&forward, &input).unwrap();
        assert_eq!(spectrum.len(), length / 2 + 1);
        assert!(spectrum.iter().all(|value| {
            (value.re - 1.0).abs() <= 2.0e-9 * length as f64
                && value.im.abs() <= 2.0e-9 * length as f64
        }));

        let inverse_plan = FftPlan::build(
            FftConfig::new(vec![length])
                .with_transform(TransformKind::ComplexToReal)
                .with_inverse_normalization(true)
                .with_tuning(tuning),
        )
        .unwrap();
        let inverse = RealFftIr::build(&inverse_plan, constrained).unwrap();
        let inverse_fused = inverse.fused_even_bluestein_ir().unwrap().unwrap();
        assert_eq!(inverse_fused.convolution_len, 4368);
        let restored = execute_c2r_ir(&inverse, &spectrum).unwrap();
        let max_error = restored
            .iter()
            .zip(&input)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0, f64::max);
        assert!(max_error <= 3.0e-9 * length as f64);
    }

    #[test]
    fn recursive_even_real_boundaries_fuse_and_match_fft_oracle() {
        let length = 578usize;
        let batch_count = 2usize;
        let input = (0..length * batch_count)
            .map(|index| {
                let x = index as f64;
                (0.13 * x).sin() + 0.21 * (0.037 * x).cos() + x * 0.0003
            })
            .collect::<Vec<_>>();
        let forward_plan = FftPlan::build(
            FftConfig::new(vec![length])
                .with_batch_count(batch_count)
                .with_transform(TransformKind::RealToComplex),
        )
        .unwrap();
        let forward = RealFftIr::build(&forward_plan, device()).unwrap();
        let fused_forward = forward
            .fused_even_recursive_ir()
            .unwrap()
            .expect("578-point R2C should expose a recursive 289-point root");
        let RecursiveFftNodeIr::CooleyTukey(root) = &fused_forward.root else {
            panic!("expected a Cooley-Tukey half-size root");
        };
        assert!(matches!(
            root.pack_right.input_modifier,
            crate::recursive_ir::CooleyTukeyInputModifier::RealEvenPack(_)
        ));
        assert!(matches!(
            root.scatter_output.output_modifier,
            crate::recursive_ir::CooleyTukeyOutputModifier::RealEvenPostprocess(_)
        ));
        let spectrum = execute_r2c_ir(&forward, &input).unwrap();
        for batch in 0..batch_count {
            let base = batch * length;
            let complex = input[base..base + length]
                .iter()
                .map(|value| Complex64::new(*value, 0.0))
                .collect::<Vec<_>>();
            let expected = crate::reference::fft(&complex, Direction::Forward, false).unwrap();
            let half_base = batch * forward.half_spectrum_len;
            assert!(
                max_complex_error(
                    &spectrum[half_base..half_base + forward.half_spectrum_len],
                    &expected[..forward.half_spectrum_len],
                ) < 8.0e-9 * length as f64
            );
        }

        let inverse_plan = FftPlan::build(
            FftConfig::new(vec![length])
                .with_batch_count(batch_count)
                .with_transform(TransformKind::ComplexToReal)
                .with_inverse_normalization(true),
        )
        .unwrap();
        let inverse = RealFftIr::build(&inverse_plan, device()).unwrap();
        let fused_inverse = inverse
            .fused_even_recursive_ir()
            .unwrap()
            .expect("578-point C2R should expose a recursive 289-point root");
        let RecursiveFftNodeIr::CooleyTukey(root) = &fused_inverse.root else {
            panic!("expected a Cooley-Tukey half-size root");
        };
        assert!(matches!(
            root.pack_right.input_modifier,
            crate::recursive_ir::CooleyTukeyInputModifier::RealEvenInversePreprocess(_)
        ));
        assert!(matches!(
            root.scatter_output.output_modifier,
            crate::recursive_ir::CooleyTukeyOutputModifier::RealEvenUnpack(_)
        ));
        let restored = execute_c2r_ir(&inverse, &spectrum).unwrap();
        let max_error = restored
            .iter()
            .zip(&input)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0, f64::max);
        assert!(max_error < 1.0e-8 * length as f64, "max error {max_error}");
    }

    #[test]
    fn even_half_size_c2r_preserves_unnormalized_inverse_scale() {
        let length = 30usize;
        let batch_count = 2usize;
        let input = (0..length * batch_count)
            .map(|index| {
                let x = index as f64;
                (0.11 * x).sin() - 0.2 * (0.037 * x).cos() + x * 0.0004
            })
            .collect::<Vec<_>>();
        let forward_plan = FftPlan::build(
            FftConfig::new(vec![length])
                .with_batch_count(batch_count)
                .with_transform(TransformKind::RealToComplex),
        )
        .unwrap();
        let forward = RealFftIr::build(&forward_plan, device()).unwrap();
        let spectrum = execute_r2c_ir(&forward, &input).unwrap();

        let inverse_plan = FftPlan::build(
            FftConfig::new(vec![length])
                .with_batch_count(batch_count)
                .with_transform(TransformKind::ComplexToReal)
                .with_inverse_normalization(false),
        )
        .unwrap();
        let inverse = RealFftIr::build(&inverse_plan, device()).unwrap();
        assert!(matches!(
            inverse.preprocess.as_ref().map(|pass| pass.operation),
            Some(RealPassOperation::PreprocessEvenHalf { normalize: false })
        ));
        let restored = execute_c2r_ir(&inverse, &spectrum).unwrap();
        let max_error = restored
            .iter()
            .zip(&input)
            .map(|(actual, expected)| (actual - expected * length as f64).abs())
            .fold(0.0, f64::max);
        assert!(max_error < 3.0e-9 * length as f64);
    }
}
