use crate::error::{Result, VkFftError};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    Forward,
    Inverse,
}

impl Direction {
    pub(crate) const fn exponent_sign(self) -> f64 {
        match self {
            Self::Forward => -1.0,
            Self::Inverse => 1.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DctType {
    I,
    II,
    III,
    IV,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DstType {
    I,
    II,
    III,
    IV,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransformKind {
    ComplexToComplex,
    RealToComplex,
    ComplexToReal,
    Dct(DctType),
    Dst(DstType),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrecisionStorage {
    F16,
    F32,
    F64,
    DoubleDouble,
}

impl PrecisionStorage {
    pub const fn scalar_bytes(self) -> usize {
        match self {
            Self::F16 => 2,
            Self::F32 => 4,
            Self::F64 => 8,
            Self::DoubleDouble => 16,
        }
    }

    pub const fn complex_bytes(self) -> usize {
        self.scalar_bytes() * 2
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrecisionCompute {
    F32,
    F64,
    DoubleDouble,
}

impl PrecisionCompute {
    pub const fn scalar_bytes(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F64 => 8,
            Self::DoubleDouble => 16,
        }
    }

    pub const fn complex_bytes(self) -> usize {
        self.scalar_bytes() * 2
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrecisionLayout {
    pub storage: PrecisionStorage,
    pub compute: PrecisionCompute,
}

impl PrecisionLayout {
    pub const fn storage_complex_bytes(self) -> usize {
        self.storage.complex_bytes()
    }

    pub const fn compute_complex_bytes(self) -> usize {
        self.compute.complex_bytes()
    }

    pub const fn has_mixed_storage_compute(self) -> bool {
        self.storage_complex_bytes() != self.compute_complex_bytes()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Precision {
    F16StorageF32Compute,
    F32,
    F64,
    F64ComputeF32Storage,
    DoubleDouble,
    DoubleDoubleF64Storage,
}

impl Precision {
    pub const fn layout(self) -> PrecisionLayout {
        match self {
            Self::F16StorageF32Compute => PrecisionLayout {
                storage: PrecisionStorage::F16,
                compute: PrecisionCompute::F32,
            },
            Self::F32 => PrecisionLayout {
                storage: PrecisionStorage::F32,
                compute: PrecisionCompute::F32,
            },
            Self::F64 => PrecisionLayout {
                storage: PrecisionStorage::F64,
                compute: PrecisionCompute::F64,
            },
            Self::F64ComputeF32Storage => PrecisionLayout {
                storage: PrecisionStorage::F32,
                compute: PrecisionCompute::F64,
            },
            Self::DoubleDouble => PrecisionLayout {
                storage: PrecisionStorage::DoubleDouble,
                compute: PrecisionCompute::DoubleDouble,
            },
            Self::DoubleDoubleF64Storage => PrecisionLayout {
                storage: PrecisionStorage::F64,
                compute: PrecisionCompute::DoubleDouble,
            },
        }
    }

    /// Bytes used by one complex value in registers/shared memory. Existing scheduler
    /// capacity decisions historically called this `complex_bytes`; keep that contract
    /// explicit while exposing storage width separately for mixed-precision execution.
    pub const fn compute_complex_bytes(self) -> usize {
        self.layout().compute_complex_bytes()
    }

    pub const fn storage_complex_bytes(self) -> usize {
        self.layout().storage_complex_bytes()
    }

    pub const fn complex_bytes(self) -> usize {
        self.compute_complex_bytes()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    Vulkan,
    Cuda,
    Hip,
    OpenCl,
    LevelZero,
    Metal,
    CpuReference,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuVendor {
    Nvidia,
    Amd,
    Intel,
    Apple,
    Other(u32),
}

/// Compute-subgroup capabilities used to gate warp/subgroup-local kernel paths.
/// `unavailable()` is deliberately conservative for generic/non-Vulkan profiles.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SubgroupProfile {
    pub size: usize,
    /// Minimum/maximum subgroup sizes exposed by subgroup-size control. When that
    /// extension is unavailable both equal `size`.
    pub min_size: usize,
    pub max_size: usize,
    /// Whether compute stages may request an exact subgroup size at pipeline creation.
    pub required_size_compute_supported: bool,
    pub compute_supported: bool,
    pub basic_supported: bool,
    pub shuffle_supported: bool,
    pub shuffle_relative_supported: bool,
    pub compute_full_subgroups: bool,
}

impl SubgroupProfile {
    pub const fn unavailable() -> Self {
        Self {
            size: 0,
            min_size: 0,
            max_size: 0,
            required_size_compute_supported: false,
            compute_supported: false,
            basic_supported: false,
            shuffle_supported: false,
            shuffle_relative_supported: false,
            compute_full_subgroups: false,
        }
    }

    pub const fn supports_shuffle_compute(self) -> bool {
        self.size > 0 && self.compute_supported && self.basic_supported && self.shuffle_supported
    }

    /// Whether a compute pipeline can safely request full subgroups and then use
    /// subgroup shuffle operations with a stable subgroup width. A variable-width
    /// implementation is accepted only when compute stages can request the exact
    /// width recorded in `size`; otherwise the reported default must be the sole
    /// supported subgroup width.
    pub const fn supports_full_subgroup_shuffle_compute(self) -> bool {
        self.supports_shuffle_compute()
            && self.compute_full_subgroups
            && (self.required_size_compute_supported
                || (self.size > 0 && self.min_size == self.size && self.max_size == self.size))
    }

    /// Exact subgroup size to request for a proven full-subgroup path when Vulkan
    /// exposes `requiredSubgroupSizeStages` for compute. `None` keeps the device's
    /// default subgroup size while still allowing `REQUIRE_FULL_SUBGROUPS`.
    pub const fn required_compute_subgroup_size(self) -> Option<u32> {
        if self.supports_full_subgroup_shuffle_compute()
            && self.required_size_compute_supported
            && self.size >= self.min_size
            && self.size <= self.max_size
            && self.size.is_power_of_two()
            && self.size <= u32::MAX as usize
        {
            Some(self.size as u32)
        } else {
            None
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceProfile {
    pub backend: Backend,
    pub vendor: GpuVendor,
    pub shared_memory_bytes: usize,
    pub shared_memory_pow2_bytes: usize,
    /// Maximum total invocations/work-items in one compute workgroup/block.
    pub max_threads_per_block: usize,
    /// Per-dimension workgroup/block limits. VkFFT's axis block splitter clamps
    /// localSizeX/localSizeY independently before applying the total-thread limit.
    pub max_workgroup_size: [usize; 3],
    pub coalesced_memory_bytes: usize,
    pub shared_banks: usize,
    pub supports_f64: bool,
    pub subgroup: SubgroupProfile,
}

impl DeviceProfile {
    pub const fn generic(backend: Backend, vendor: GpuVendor) -> Self {
        Self {
            backend,
            vendor,
            shared_memory_bytes: 32 * 1024,
            shared_memory_pow2_bytes: 32 * 1024,
            max_threads_per_block: 256,
            max_workgroup_size: [256, 256, 256],
            coalesced_memory_bytes: 32,
            shared_banks: 32,
            supports_f64: !matches!(backend, Backend::Metal),
            subgroup: SubgroupProfile::unavailable(),
        }
    }
}

/// Whether this backend/vendor pair has a fixed-upstream GPU scheduler profile.
/// Unknown combinations keep the portable/fail-soft IR path instead of borrowing
/// another vendor's discrete resource heuristics.
pub(crate) const fn has_fixed_upstream_gpu_scheduler_profile(profile: DeviceProfile) -> bool {
    matches!(
        (profile.backend, profile.vendor),
        (
            Backend::Vulkan | Backend::OpenCl,
            GpuVendor::Nvidia | GpuVendor::Amd | GpuVendor::Intel
        ) | (Backend::Cuda, GpuVendor::Nvidia)
            | (Backend::Hip, GpuVendor::Amd)
            | (Backend::LevelZero, GpuVendor::Intel)
            | (Backend::Metal, GpuVendor::Apple)
    )
}

/// Fixed-commit `configuration.coalescedMemory` initialization used by the
/// scheduler. Keep this separate from the physical/profile field: runtime
/// profiles may report conservative transport metadata, while VkFFT's planner
/// uses backend/vendor initialization defaults for its discrete scoring.
pub(crate) const fn upstream_coalesced_memory_bytes(profile: DeviceProfile) -> usize {
    match profile.backend {
        Backend::Vulkan | Backend::OpenCl => match profile.vendor {
            GpuVendor::Nvidia | GpuVendor::Amd => 32,
            GpuVendor::Intel | GpuVendor::Apple | GpuVendor::Other(_) => 64,
        },
        Backend::Cuda | Backend::Hip => 32,
        Backend::LevelZero | Backend::Metal => 64,
        Backend::CpuReference => profile.coalesced_memory_bytes,
    }
}

/// Precision-adjusted fixed-upstream coalescing width. VkFFT doubles
/// `configuration.coalescedMemory` in half-storage mode while keeping F32 compute
/// `complexSize`, so F16-storage scheduling must not reuse the ordinary F32 width.
pub(crate) const fn upstream_coalesced_memory_bytes_for_precision(
    profile: DeviceProfile,
    precision: Precision,
) -> usize {
    let base = upstream_coalesced_memory_bytes(profile);
    if matches!(precision, Precision::F16StorageF32Compute) {
        base * 2
    } else {
        base
    }
}

/// Fixed-upstream `VkFFTScheduler` cap applied to `fixMaxRaderPrimeMult` after
/// the FFT-Rader safe-prime scan. `complexSize` follows compute precision even
/// when external storage is narrower.
pub(crate) fn upstream_direct_rader_thread_cap(
    profile: DeviceProfile,
    precision: Precision,
) -> usize {
    let complex_size = precision.compute_complex_bytes();
    let values_per_coalesced =
        (upstream_coalesced_memory_bytes_for_precision(profile, precision) / complex_size).max(1);
    (profile.max_threads_per_block / values_per_coalesced)
        .saturating_mul(2)
        .saturating_sub(1)
}

/// Apply the fixed-upstream device thread/coalescing cap to the Direct-Rader
/// exclusive upper bound before constructing or scoring nested Rader trees.
pub(crate) fn upstream_effective_rader_tuning(
    mut tuning: PlannerTuning,
    profile: DeviceProfile,
    precision: Precision,
) -> PlannerTuning {
    tuning.max_rader_direct_prime = tuning
        .max_rader_direct_prime
        .min(upstream_direct_rader_thread_cap(profile, precision))
        .max(tuning.min_rader_direct_prime);
    tuning
}

pub const PLANNER_TUNING_PROFILE_VERSION: u32 = 1;
const PLANNER_TUNING_PROFILE_MAGIC: &[u8; 8] = b"VKFTRSTN";
const PLANNER_TUNING_PROFILE_COMMIT_BYTES: usize = 40;
const PLANNER_TUNING_PROFILE_BYTES: usize = 8 + 4 + PLANNER_TUNING_PROFILE_COMMIT_BYTES + 4 * 8 + 1;

/// Host-side algorithm thresholds corresponding to VkFFT's Rader/Bluestein planner knobs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlannerTuning {
    /// First prime eligible for direct-multiplication Rader handling.
    pub min_rader_direct_prime: usize,
    /// Exclusive upper bound for direct-multiplication Rader handling.
    pub max_rader_direct_prime: usize,
    /// First prime eligible for FFT-convolution Rader handling.
    pub min_rader_fft_prime: usize,
    /// Exclusive upper bound for FFT-convolution Rader handling.
    pub max_rader_fft_prime: usize,
    /// Opt-in extension beyond upstream's default safe-prime gate. When enabled,
    /// FFT-Rader may recursively decompose residual factors of `(p - 1)` into
    /// smaller Stockham/Rader nodes. VkFFT 1.3.4 keeps this disabled by default.
    pub allow_recursive_fft_rader: bool,
}

impl PlannerTuning {
    /// Portable defaults following the current VkFFT scheduler's normal-precision ranges.
    pub const fn portable() -> Self {
        Self {
            min_rader_direct_prime: 17,
            max_rader_direct_prime: 89,
            min_rader_fft_prime: 17,
            max_rader_fft_prime: 16_384,
            allow_recursive_fft_rader: false,
        }
    }

    /// Apply the vendor/precision-specific Rader thresholds used by the upstream scheduler.
    pub const fn for_device(profile: DeviceProfile, precision: Precision) -> Self {
        let mut tuning = Self::portable();

        if matches!(
            precision,
            Precision::DoubleDouble | Precision::DoubleDoubleF64Storage
        ) {
            tuning.min_rader_direct_prime = 11;
            tuning.max_rader_direct_prime = 29;
        } else {
            tuning.max_rader_direct_prime = match profile.vendor {
                GpuVendor::Nvidia | GpuVendor::Amd => 89,
                GpuVendor::Intel | GpuVendor::Apple | GpuVendor::Other(_) => 17,
            };
        }

        if matches!(profile.vendor, GpuVendor::Amd) {
            tuning.min_rader_fft_prime = match precision {
                Precision::DoubleDouble | Precision::DoubleDoubleF64Storage => 19,
                Precision::F64 | Precision::F64ComputeF32Storage => 29,
                _ => 17,
            };
        }

        tuning
    }

    pub const fn with_recursive_fft_rader(mut self, enabled: bool) -> Self {
        self.allow_recursive_fft_rader = enabled;
        self
    }

    /// Deterministic, versioned planner-tuning profile bytes. Unlike Rust's `Hash`,
    /// this representation is stable across processes and includes the fixed upstream
    /// commit so cached/tuned decisions are never silently reused against another port baseline.
    pub fn to_profile_bytes(self) -> Vec<u8> {
        let commit = crate::UPSTREAM_VKFFT_COMMIT.as_bytes();
        debug_assert_eq!(commit.len(), PLANNER_TUNING_PROFILE_COMMIT_BYTES);
        let mut bytes = Vec::with_capacity(PLANNER_TUNING_PROFILE_BYTES);
        bytes.extend_from_slice(PLANNER_TUNING_PROFILE_MAGIC);
        bytes.extend_from_slice(&PLANNER_TUNING_PROFILE_VERSION.to_le_bytes());
        bytes.extend_from_slice(commit);
        for value in [
            self.min_rader_direct_prime,
            self.max_rader_direct_prime,
            self.min_rader_fft_prime,
            self.max_rader_fft_prime,
        ] {
            bytes.extend_from_slice(&(value as u64).to_le_bytes());
        }
        bytes.push(u8::from(self.allow_recursive_fft_rader));
        bytes
    }

    /// Decode a planner profile only when its schema and upstream baseline match this crate.
    pub fn from_profile_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() != PLANNER_TUNING_PROFILE_BYTES {
            return Err(VkFftError::InvalidPlannerTuning(
                "planner tuning profile has an invalid byte length",
            ));
        }
        if &bytes[..8] != PLANNER_TUNING_PROFILE_MAGIC {
            return Err(VkFftError::InvalidPlannerTuning(
                "planner tuning profile magic does not match",
            ));
        }
        let version = u32::from_le_bytes(bytes[8..12].try_into().map_err(|_| {
            VkFftError::InvalidPlannerTuning("planner tuning profile version is malformed")
        })?);
        if version != PLANNER_TUNING_PROFILE_VERSION {
            return Err(VkFftError::InvalidPlannerTuning(
                "planner tuning profile schema version is unsupported",
            ));
        }
        let commit_start = 12;
        let commit_end = commit_start + PLANNER_TUNING_PROFILE_COMMIT_BYTES;
        if &bytes[commit_start..commit_end] != crate::UPSTREAM_VKFFT_COMMIT.as_bytes() {
            return Err(VkFftError::InvalidPlannerTuning(
                "planner tuning profile upstream commit does not match",
            ));
        }
        let mut offset = commit_end;
        let mut next_usize = || -> Result<usize> {
            let end = offset + 8;
            let raw = u64::from_le_bytes(bytes[offset..end].try_into().map_err(|_| {
                VkFftError::InvalidPlannerTuning("planner tuning profile integer is malformed")
            })?);
            offset = end;
            usize::try_from(raw).map_err(|_| {
                VkFftError::InvalidPlannerTuning(
                    "planner tuning profile integer does not fit this platform",
                )
            })
        };
        let tuning = Self {
            min_rader_direct_prime: next_usize()?,
            max_rader_direct_prime: next_usize()?,
            min_rader_fft_prime: next_usize()?,
            max_rader_fft_prime: next_usize()?,
            allow_recursive_fft_rader: match bytes[offset] {
                0 => false,
                1 => true,
                _ => {
                    return Err(VkFftError::InvalidPlannerTuning(
                        "planner tuning profile boolean is malformed",
                    ));
                }
            },
        };
        tuning.validate()?;
        Ok(tuning)
    }

    /// Stable FNV-1a fingerprint of the canonical profile bytes for cache/report keys.
    pub fn profile_fingerprint(self) -> u64 {
        let mut hash = 0xcbf2_9ce4_8422_2325u64;
        for byte in self.to_profile_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
        hash
    }

    pub fn validate(self) -> Result<()> {
        if self.min_rader_direct_prime < 2 || self.min_rader_fft_prime < 2 {
            return Err(VkFftError::InvalidPlannerTuning(
                "minimum Rader prime must be at least 2",
            ));
        }
        // `VkFFTRaderContainer::loc_multipliers` has 33 entries and upstream
        // indexes every radix `j < fixMinRaderPrimeMult` directly. Threshold 33
        // is therefore the largest upstream-safe exclusive bound (max index 32).
        if self.min_rader_direct_prime > 33 {
            return Err(VkFftError::InvalidPlannerTuning(
                "direct Rader minimum must not exceed upstream loc_multipliers capacity (33)",
            ));
        }
        if self.max_rader_direct_prime < self.min_rader_direct_prime {
            return Err(VkFftError::InvalidPlannerTuning(
                "direct Rader maximum must not be smaller than its minimum",
            ));
        }
        if self.max_rader_fft_prime < self.min_rader_fft_prime {
            return Err(VkFftError::InvalidPlannerTuning(
                "FFT Rader maximum must not be smaller than its minimum",
            ));
        }
        Ok(())
    }
}

impl Default for PlannerTuning {
    fn default() -> Self {
        Self::portable()
    }
}

/// Select which transform-domain boundary owns `performZeropadding`.
///
/// Fixed upstream uses spatial padding on the forward input / inverse output and
/// frequency padding on the forward output / inverse input. The FFT extent itself is
/// unchanged in both modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ZeroPaddingDomain {
    #[default]
    Spatial,
    Frequency,
}

/// Upstream-compatible logical zero interval for one FFT axis. The physical
/// transform size is unchanged; values with indices in `[left, right)` are treated
/// as zero at the external boundary selected by [`ZeroPaddingDomain`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ZeroPaddingRange {
    pub left: usize,
    pub right: usize,
}

impl ZeroPaddingRange {
    pub const fn new(left: usize, right: usize) -> Self {
        Self { left, right }
    }

    pub const fn contains(self, index: usize) -> bool {
        index >= self.left && index < self.right
    }

    fn validate(self, axis: usize, length: usize) -> Result<()> {
        if self.left > self.right || self.right > length {
            return Err(VkFftError::InvalidZeroPaddingRange {
                axis,
                left: self.left,
                right: self.right,
                length,
            });
        }
        Ok(())
    }
}

/// Upstream `configuration.conjugateConvolution` application policy.
///
/// The pinned 1.3.4 codegen contains an executable branch only for value `1`
/// (`Sequence`). Value `2` is documented as kernel conjugation but has no matching
/// codegen branch in the pinned source, so Rust exposes the value for ABI clarity
/// while validation deliberately keeps it fail-closed until upstream behavior is pinned.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ConvolutionConjugation {
    #[default]
    None,
    Sequence,
    Kernel,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FftConfig {
    pub dimensions: Vec<usize>,
    pub batch_count: usize,
    pub transform: TransformKind,
    pub precision: Precision,
    pub normalize_inverse: bool,
    /// Upstream `configuration.performConvolution`. Rust exposes one-dimensional
    /// application pipelines through `ConvolutionIr` and the first correctness-first
    /// multidimensional C2C surface through `NdConvolutionIr`; ordinary `FftPlan`
    /// construction deliberately rejects this flag so convolution cannot silently
    /// degrade into a plain FFT.
    pub perform_convolution: bool,
    /// Upstream `configuration.kernelConvolution`. This is a forward FFT used to prepare
    /// an application kernel spectrum under the same scheduler-capacity knobs as
    /// `performConvolution`, but it does not insert an application multiply or inverse.
    pub kernel_convolution: bool,
    /// Upstream `configuration.conjugateConvolution`.
    pub convolution_conjugation: ConvolutionConjugation,
    /// Upstream `configuration.crossPowerSpectrumNormalization`.
    pub cross_power_spectrum_normalization: bool,
    /// Upstream `configuration.coordinateFeatures`. The first Rust matrix tranche
    /// consumes this only through `matrixConvolution`, which overrides it to M.
    pub coordinate_features: usize,
    /// Upstream `configuration.matrixConvolution`; internal VkFFT default is 1.
    pub matrix_convolution: usize,
    /// Upstream `configuration.symmetricKernel`.
    pub symmetric_convolution_kernel: bool,
    /// Upstream `configuration.numberKernels`. Pinned VkFFT defines this as a
    /// one-input fan-out: its batched-convolution sample forces `numberBatches=1`,
    /// while `convolutionStep` uses kernel `batchID` as the batch/output stride.
    pub convolution_kernel_count: usize,
    pub tuning: PlannerTuning,
    /// When true, the high-level `TransformIr::build` path replaces `tuning` with
    /// fixed-upstream vendor/precision defaults derived from its `DeviceProfile`.
    /// Low-level `FftPlan::build` has no device context and therefore keeps `tuning`
    /// unchanged. `with_tuning` disables this flag so explicit user overrides win.
    pub use_device_tuning: bool,
    pub zero_padding: Vec<Option<ZeroPaddingRange>>,
    /// Explicit upstream `configuration.omitDimension[axis]` flags. Supported ND
    /// transform families also treat eligible unit-length axes as omitted, matching fixed
    /// upstream without changing the crate's established one-dimensional N=1 identity path.
    pub omit_dimension: Vec<bool>,
    /// Upstream `configuration.frequencyZeroPadding`. `Spatial` preserves the
    /// historical forward-input / inverse-output zero-padding boundary.
    pub zero_padding_domain: ZeroPaddingDomain,
    /// Optional upstream `configuration.groupedBatch[axis]` override. `None`
    /// preserves automatic VkFFT workgroup grouping for that axis.
    pub grouped_batch: Vec<Option<usize>>,
    /// Upstream `configuration.performBandwidthBoost`. Zero preserves the fixed
    /// default/automatic behavior; positive values relax the allowed coalesced
    /// bandwidth for strided axes when that reduces the required upload count.
    pub bandwidth_boost: usize,
    /// Optional physical stored-element stride for each Rust logical tensor axis.
    /// Dimensions are stored slowest-to-fastest, so for `[planes, rows, cols]` the
    /// dense default is `[rows * cols, cols, 1]`. This is the Rust-facing reversal
    /// of upstream cumulative `inputBufferStride[]`; the fastest axis must remain 1.
    pub input_buffer_axis_strides: Vec<Option<usize>>,
    /// Output counterpart of [`FftConfig::input_buffer_axis_strides`].
    pub output_buffer_axis_strides: Vec<Option<usize>>,
    /// First formatted-buffer stride slice: physical element distance between adjacent
    /// input batches, corresponding to upstream `inputBufferStride[FFTdim-1]`.
    /// `None` keeps the dense/default buffer layout.
    pub input_buffer_batch_stride: Option<usize>,
    /// Physical element distance between adjacent formatted output batches, corresponding
    /// to upstream `outputBufferStride[FFTdim-1]`.
    pub output_buffer_batch_stride: Option<usize>,
}

impl FftConfig {
    pub fn new(dimensions: impl Into<Vec<usize>>) -> Self {
        let dimensions = dimensions.into();
        let zero_padding = vec![None; dimensions.len()];
        let omit_dimension = vec![false; dimensions.len()];
        let grouped_batch = vec![None; dimensions.len()];
        let input_buffer_axis_strides = vec![None; dimensions.len()];
        let output_buffer_axis_strides = vec![None; dimensions.len()];
        Self {
            dimensions,
            batch_count: 1,
            transform: TransformKind::ComplexToComplex,
            precision: Precision::F32,
            normalize_inverse: false,
            perform_convolution: false,
            kernel_convolution: false,
            convolution_conjugation: ConvolutionConjugation::None,
            cross_power_spectrum_normalization: false,
            coordinate_features: 1,
            matrix_convolution: 1,
            symmetric_convolution_kernel: false,
            convolution_kernel_count: 1,
            tuning: PlannerTuning::default(),
            use_device_tuning: true,
            zero_padding,
            omit_dimension,
            zero_padding_domain: ZeroPaddingDomain::Spatial,
            grouped_batch,
            bandwidth_boost: 0,
            input_buffer_axis_strides,
            output_buffer_axis_strides,
            input_buffer_batch_stride: None,
            output_buffer_batch_stride: None,
        }
    }

    pub fn with_transform(mut self, transform: TransformKind) -> Self {
        self.transform = transform;
        self
    }

    pub fn with_precision(mut self, precision: Precision) -> Self {
        self.precision = precision;
        self
    }

    pub fn with_batch_count(mut self, batch_count: usize) -> Self {
        self.batch_count = batch_count;
        self
    }

    pub fn with_inverse_normalization(mut self, normalize: bool) -> Self {
        self.normalize_inverse = normalize;
        self
    }

    /// Enable or disable upstream-style application convolution. The current P0
    /// contract is one-dimensional ordinary F32/F64 C2C with one pre-transformed
    /// frequency-domain kernel; build it through [`crate::ConvolutionIr`].
    pub fn with_convolution(mut self, enabled: bool) -> Self {
        self.perform_convolution = enabled;
        self
    }

    /// Mark this FFT as upstream-style convolution-kernel preparation. The current Rust
    /// contract is dense one-dimensional C2C/R2C with independent batch/coordinate
    /// systems; matrix, multi-kernel, formatted, padded, and mixed-storage preparation
    /// remain separately gated.
    pub fn with_kernel_convolution(mut self, enabled: bool) -> Self {
        self.kernel_convolution = enabled;
        self
    }

    /// Select upstream convolution conjugation semantics. `Sequence` is the pinned
    /// executable mode; `Kernel` remains fail-closed because the pinned source only
    /// documents value 2 and does not emit a corresponding arithmetic branch.
    pub fn with_convolution_conjugation(mut self, conjugation: ConvolutionConjugation) -> Self {
        self.convolution_conjugation = conjugation;
        self
    }

    /// Normalize each frequency-domain convolution product using upstream's
    /// `norm` then `rsqrt` sequence (`z * rsqrt(re² + im²)`).
    pub fn with_cross_power_spectrum_normalization(mut self, enabled: bool) -> Self {
        self.cross_power_spectrum_normalization = enabled;
        self
    }

    /// Set upstream `coordinateFeatures`. Matrix convolution overrides this value to
    /// the matrix size during execution, matching pinned VkFFT initialization.
    pub fn with_coordinate_features(mut self, coordinate_features: usize) -> Self {
        self.coordinate_features = coordinate_features;
        self
    }

    /// Select upstream matrix-vector convolution. Values 2 and 3 represent 2x2 and
    /// 3x3 respectively; setting a matrix mode also mirrors upstream's effective
    /// coordinate count by setting `coordinate_features` to the matrix size.
    pub fn with_matrix_convolution(mut self, matrix_size: usize) -> Self {
        self.matrix_convolution = matrix_size;
        if matrix_size > 1 {
            self.coordinate_features = matrix_size;
        }
        self
    }

    /// Select packed upper-triangular kernel storage for matrix convolution.
    pub fn with_symmetric_convolution_kernel(mut self, symmetric: bool) -> Self {
        self.symmetric_convolution_kernel = symmetric;
        self
    }

    /// Set upstream `numberKernels`. The executable contract is one input fanned out
    /// to multiple kernel-indexed outputs; it is not a `numberBatches * numberKernels`
    /// Cartesian product.
    pub fn with_convolution_kernel_count(mut self, kernel_count: usize) -> Self {
        self.convolution_kernel_count = kernel_count;
        self
    }

    pub(crate) fn convolution_coordinate_count(&self) -> usize {
        if self.matrix_convolution > 1 {
            self.matrix_convolution
        } else {
            self.coordinate_features
        }
    }

    /// Number of independent transforms owned by a kernel-preparation plan. Pinned
    /// `VkFFT_RunApp` dispatches `coordinateFeatures * numberBatches` for forward kernel
    /// preparation; ordinary transforms keep their public batch count unchanged.
    pub(crate) fn kernel_preparation_system_count(&self) -> Result<usize> {
        if !self.kernel_convolution {
            return Ok(self.batch_count);
        }
        self.batch_count
            .checked_mul(self.coordinate_features)
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "kernelConvolution batch x coordinate system count",
            })
    }

    pub fn with_tuning(mut self, tuning: PlannerTuning) -> Self {
        self.tuning = tuning;
        self.use_device_tuning = false;
        self
    }

    /// Restore fixed-upstream device/precision planner defaults for the high-level
    /// Application -> Plan -> Code path after an earlier explicit tuning override.
    pub fn with_device_tuning(mut self) -> Self {
        self.tuning = PlannerTuning::portable();
        self.use_device_tuning = true;
        self
    }

    pub(crate) fn resolve_tuning_for_device(mut self, profile: DeviceProfile) -> Self {
        if self.use_device_tuning {
            self.tuning = PlannerTuning::for_device(profile, self.precision);
        }
        self
    }

    /// Mirror upstream `configuration.omitDimension[axis]` for supported multidimensional
    /// C2C, real, and R2R transforms. Tensor extent/storage layout is unchanged; the selected
    /// axis simply has no transform passes. R2C/C2R cannot omit the contiguous real axis.
    pub fn with_omit_dimension(mut self, axis: usize, omit: bool) -> Result<Self> {
        let slot = self
            .omit_dimension
            .get_mut(axis)
            .ok_or(VkFftError::ValueOutOfRange {
                field: "omitDimension axis",
            })?;
        *slot = omit;
        Ok(self)
    }

    pub fn axis_is_omitted(&self, axis: usize) -> bool {
        if self.omit_dimension.get(axis).copied().unwrap_or(false) {
            return true;
        }
        if self.dimensions.len() <= 1 || self.dimensions.get(axis).copied() != Some(1) {
            return false;
        }
        let fastest_axis = self.dimensions.len() - 1;
        !matches!(
            self.transform,
            TransformKind::RealToComplex | TransformKind::ComplexToReal
        ) || axis != fastest_axis
    }

    /// Override upstream `configuration.groupedBatch[axis]`. A value of zero clears
    /// the override and restores automatic scheduler grouping.
    pub fn with_grouped_batch(mut self, axis: usize, grouped_batch: usize) -> Result<Self> {
        let slot = self
            .grouped_batch
            .get_mut(axis)
            .ok_or(VkFftError::ValueOutOfRange {
                field: "groupedBatch axis",
            })?;
        *slot = (grouped_batch != 0).then_some(grouped_batch);
        Ok(self)
    }

    pub fn grouped_batch_for_axis(&self, axis: usize) -> Option<usize> {
        self.grouped_batch.get(axis).copied().flatten()
    }

    /// Override upstream `configuration.performBandwidthBoost`. The value is only
    /// consumed by strided device-scored axes; zero restores the default behavior.
    pub fn with_bandwidth_boost(mut self, bandwidth_boost: usize) -> Self {
        self.bandwidth_boost = bandwidth_boost;
        self
    }

    /// Set the formatted input stride for one Rust logical axis in stored elements.
    /// Zero clears the override. The fastest logical axis is physically contiguous
    /// in upstream VkFFT and therefore may only resolve to stride 1.
    pub fn with_input_buffer_axis_stride(mut self, axis: usize, stride: usize) -> Result<Self> {
        let slot =
            self.input_buffer_axis_strides
                .get_mut(axis)
                .ok_or(VkFftError::ValueOutOfRange {
                    field: "formatted input axis stride",
                })?;
        *slot = (stride != 0).then_some(stride);
        Ok(self)
    }

    /// Set the formatted output stride for one Rust logical axis in stored elements.
    /// Zero clears the override.
    pub fn with_output_buffer_axis_stride(mut self, axis: usize, stride: usize) -> Result<Self> {
        let slot =
            self.output_buffer_axis_strides
                .get_mut(axis)
                .ok_or(VkFftError::ValueOutOfRange {
                    field: "formatted output axis stride",
                })?;
        *slot = (stride != 0).then_some(stride);
        Ok(self)
    }

    pub fn input_buffer_axis_stride_for_axis(&self, axis: usize) -> Option<usize> {
        self.input_buffer_axis_strides.get(axis).copied().flatten()
    }

    pub fn output_buffer_axis_stride_for_axis(&self, axis: usize) -> Option<usize> {
        self.output_buffer_axis_strides.get(axis).copied().flatten()
    }

    pub(crate) fn resolved_input_buffer_axis_strides(&self) -> Result<Vec<usize>> {
        self.resolved_input_buffer_axis_strides_for(&self.dimensions)
    }

    pub(crate) fn resolved_output_buffer_axis_strides(&self) -> Result<Vec<usize>> {
        self.resolved_output_buffer_axis_strides_for(&self.dimensions)
    }

    pub(crate) fn resolved_input_buffer_axis_strides_for(
        &self,
        dimensions: &[usize],
    ) -> Result<Vec<usize>> {
        resolve_formatted_axis_strides(
            dimensions,
            &self.input_buffer_axis_strides,
            "formatted input axis stride",
        )
    }

    pub(crate) fn resolved_output_buffer_axis_strides_for(
        &self,
        dimensions: &[usize],
    ) -> Result<Vec<usize>> {
        resolve_formatted_axis_strides(
            dimensions,
            &self.output_buffer_axis_strides,
            "formatted output axis stride",
        )
    }

    pub(crate) fn resolved_input_buffer_batch_stride(&self) -> Result<usize> {
        self.resolved_input_buffer_batch_stride_for(&self.dimensions)
    }

    pub(crate) fn resolved_output_buffer_batch_stride(&self) -> Result<usize> {
        self.resolved_output_buffer_batch_stride_for(&self.dimensions)
    }

    pub(crate) fn resolved_input_buffer_batch_stride_for(
        &self,
        dimensions: &[usize],
    ) -> Result<usize> {
        resolve_formatted_batch_stride(
            dimensions,
            &self.resolved_input_buffer_axis_strides_for(dimensions)?,
            self.input_buffer_batch_stride,
            "formatted input batch stride",
        )
    }

    pub(crate) fn resolved_output_buffer_batch_stride_for(
        &self,
        dimensions: &[usize],
    ) -> Result<usize> {
        resolve_formatted_batch_stride(
            dimensions,
            &self.resolved_output_buffer_axis_strides_for(dimensions)?,
            self.output_buffer_batch_stride,
            "formatted output batch stride",
        )
    }

    /// Set the formatted input batch stride in stored elements. This is the last
    /// cumulative input stride in upstream VkFFT. Zero clears the override.
    pub fn with_input_buffer_batch_stride(mut self, stride: usize) -> Self {
        self.input_buffer_batch_stride = (stride != 0).then_some(stride);
        self
    }

    /// Set the formatted output batch stride in stored elements. This is the last
    /// cumulative output stride in upstream VkFFT. Zero clears the override.
    pub fn with_output_buffer_batch_stride(mut self, stride: usize) -> Self {
        self.output_buffer_batch_stride = (stride != 0).then_some(stride);
        self
    }

    /// Set the upstream-style logical zero interval for one axis. The interval stays
    /// inside the existing FFT length and is applied in the currently selected
    /// [`ZeroPaddingDomain`] (spatial by default); it does not resize storage.
    pub fn with_zero_padding(mut self, axis: usize, left: usize, right: usize) -> Result<Self> {
        let length =
            self.dimensions
                .get(axis)
                .copied()
                .ok_or(VkFftError::InvalidZeroPaddingRange {
                    axis,
                    left,
                    right,
                    length: 0,
                })?;
        let range = ZeroPaddingRange::new(left, right);
        range.validate(axis, length)?;
        self.zero_padding[axis] = (left != right).then_some(range);
        Ok(self)
    }

    pub fn zero_padding_for_axis(&self, axis: usize) -> Option<ZeroPaddingRange> {
        self.zero_padding.get(axis).copied().flatten()
    }

    /// Select the domain whose external boundary owns zero padding. This mirrors
    /// upstream `configuration.frequencyZeroPadding`; frequency-domain padding is
    /// currently implemented for C2C transforms and fails closed for real/R2R paths.
    pub fn with_zero_padding_domain(mut self, domain: ZeroPaddingDomain) -> Self {
        self.zero_padding_domain = domain;
        self
    }

    pub fn validate(&self) -> Result<()> {
        if self.dimensions.is_empty() {
            return Err(VkFftError::EmptyDimensions);
        }
        if self.batch_count == 0 {
            return Err(VkFftError::ZeroBatchCount);
        }
        if self.zero_padding.len() != self.dimensions.len() {
            return Err(VkFftError::InvalidKernelIr(
                "zero-padding metadata must match the FFT dimensionality",
            ));
        }
        if self.omit_dimension.len() != self.dimensions.len() {
            return Err(VkFftError::InvalidKernelIr(
                "omitDimension metadata must match the FFT dimensionality",
            ));
        }
        if self.omit_dimension.iter().any(|&omit| omit) {
            if self.dimensions.len() == 1 {
                return Err(VkFftError::UnsupportedKernelPath(
                    "omitDimension currently supports multidimensional transforms only",
                ));
            }
            if matches!(
                self.transform,
                TransformKind::RealToComplex | TransformKind::ComplexToReal
            ) && self.omit_dimension[self.dimensions.len() - 1]
            {
                return Err(VkFftError::UnsupportedKernelPath(
                    "omitDimension cannot disable the contiguous R2C/C2R axis",
                ));
            }
        }
        if self.dimensions.len() > 1
            && !matches!(
                self.transform,
                TransformKind::RealToComplex | TransformKind::ComplexToReal
            )
            && (0..self.dimensions.len()).all(|axis| self.axis_is_omitted(axis))
        {
            return Err(VkFftError::UnsupportedKernelPath(
                "multidimensional transform requires at least one active axis",
            ));
        }
        if self.input_buffer_axis_strides.len() != self.dimensions.len()
            || self.output_buffer_axis_strides.len() != self.dimensions.len()
        {
            return Err(VkFftError::InvalidKernelIr(
                "formatted axis-stride metadata must match the FFT dimensionality",
            ));
        }
        let has_formatted_strides = self.input_buffer_batch_stride.is_some()
            || self.output_buffer_batch_stride.is_some()
            || self.input_buffer_axis_strides.iter().any(Option::is_some)
            || self.output_buffer_axis_strides.iter().any(Option::is_some);
        if self.coordinate_features == 0 {
            return Err(VkFftError::ValueOutOfRange {
                field: "coordinateFeatures",
            });
        }
        if self.convolution_kernel_count == 0 {
            return Err(VkFftError::ValueOutOfRange {
                field: "numberKernels",
            });
        }
        if self.kernel_convolution {
            if self.perform_convolution {
                return Err(VkFftError::UnsupportedKernelPath(
                    "kernelConvolution prepares a kernel spectrum and must not be combined with performConvolution",
                ));
            }
            let _ = self.kernel_preparation_system_count()?;
            let dense_one_dimensional =
                self.dimensions.len() == 1 && self.zero_padding.iter().all(Option::is_none);
            let sample52_unpadded_nd = self.transform == TransformKind::RealToComplex
                && self.dimensions.len() == 2
                && self.batch_count == 2
                && self.coordinate_features == 2
                && self.zero_padding.iter().all(Option::is_none);
            let sample51_padded_nd = self.transform == TransformKind::RealToComplex
                && self.dimensions.len() == 3
                && self.batch_count == 1
                && self.coordinate_features == 9
                && self.zero_padding.iter().all(Option::is_some)
                && self.zero_padding_domain == ZeroPaddingDomain::Spatial;
            if !matches!(
                self.transform,
                TransformKind::ComplexToComplex | TransformKind::RealToComplex
            ) || (!dense_one_dimensional && !sample52_unpadded_nd && !sample51_padded_nd)
                || self.matrix_convolution != 1
                || self.symmetric_convolution_kernel
                || self.convolution_kernel_count != 1
                || !matches!(self.precision, Precision::F32 | Precision::F64)
                || has_formatted_strides
                || self.convolution_conjugation != ConvolutionConjugation::None
                || self.cross_power_spectrum_normalization
            {
                return Err(VkFftError::UnsupportedKernelPath(
                    "kernelConvolution currently requires dense 1D C2C/R2C independent batch/coordinate K1 preparation, pinned sample_52-style 2D R2C B2/C2 preparation, or pinned sample_51-style batch1 3D R2C C9 Spatial-padding preparation in ordinary F32/F64",
                ));
            }
        }
        let has_extended_convolution_layout = self.coordinate_features != 1
            || self.matrix_convolution != 1
            || self.symmetric_convolution_kernel
            || self.convolution_kernel_count != 1;
        if !self.perform_convolution && !self.kernel_convolution && has_extended_convolution_layout
        {
            return Err(VkFftError::UnsupportedKernelPath(
                "coordinate/matrix/multi-kernel layout currently requires performConvolution or the pinned independent-coordinate kernelConvolution preparation contract",
            ));
        }
        if !self.perform_convolution
            && (self.convolution_conjugation != ConvolutionConjugation::None
                || self.cross_power_spectrum_normalization)
        {
            return Err(VkFftError::UnsupportedKernelPath(
                "convolution conjugation/cross-power semantics require performConvolution",
            ));
        }
        if self.perform_convolution {
            let real_application =
                self.transform == TransformKind::RealToComplex && self.dimensions.len() > 1;
            if self.transform != TransformKind::ComplexToComplex && !real_application {
                return Err(VkFftError::UnsupportedKernelPath(
                    "performConvolution supports C2C plus multidimensional R2C application transforms",
                ));
            }
            if self.dimensions.len() > 1 {
                let scalar_k1_layout = !has_extended_convolution_layout;
                let scalar_fanout_layout = self.matrix_convolution == 1
                    && self.coordinate_features == 1
                    && self.convolution_kernel_count > 1;
                if real_application {
                    let independent_coordinate_layout = self.matrix_convolution == 1;
                    let matrix_batch1_layout = self.matrix_convolution > 1
                        && self.coordinate_features == self.matrix_convolution;
                    if self.batch_count != 1
                        || (!independent_coordinate_layout && !matrix_batch1_layout)
                    {
                        return Err(VkFftError::UnsupportedKernelPath(
                            "multidimensional real performConvolution supports batch1 independent-coordinate or matrix K=1/K>1 layouts",
                        ));
                    }
                } else {
                    let matrix_k1_batch1_layout = self.matrix_convolution > 1
                        && self.batch_count == 1
                        && self.convolution_kernel_count == 1
                        && self.coordinate_features == self.matrix_convolution;
                    if !scalar_k1_layout && !scalar_fanout_layout && !matrix_k1_batch1_layout {
                        return Err(VkFftError::UnsupportedKernelPath(
                            "multidimensional performConvolution supports scalar K=1/K>1 layouts or a batch1/K1 matrix layout",
                        ));
                    }
                }
            }
            if self.convolution_kernel_count > 1 && self.batch_count != 1 {
                return Err(VkFftError::UnsupportedKernelPath(
                    "pinned VkFFT numberKernels is a one-input fan-out, not a numberBatches*numberKernels Cartesian product: sample_52 fixes numberBatches=1 and convolutionStep uses kernel batchID for the batch/output stride",
                ));
            }
            if !matches!(self.matrix_convolution, 1..=3) {
                return Err(VkFftError::UnsupportedKernelPath(
                    "matrixConvolution currently accepts pinned VkFFT sizes 1, 2, or 3",
                ));
            }
            if self.matrix_convolution == 1 && self.coordinate_features != 1 && !real_application {
                return Err(VkFftError::UnsupportedKernelPath(
                    "coordinateFeatures > 1 without matrixConvolution is only materialized for multidimensional real application convolution",
                ));
            }
            if self.symmetric_convolution_kernel {
                match self.matrix_convolution {
                    2 => {}
                    3 => {
                        return Err(VkFftError::UnsupportedKernelPath(
                            "pinned VkFFT 1.3.4 3x3 symmetric-kernel code aliases yz and zz while the API guide documents six planes; keep this path fail-closed",
                        ));
                    }
                    _ => {
                        return Err(VkFftError::UnsupportedKernelPath(
                            "symmetricKernel requires matrixConvolution > 1",
                        ));
                    }
                }
            }
            if self.convolution_conjugation == ConvolutionConjugation::Kernel {
                return Err(VkFftError::UnsupportedKernelPath(
                    "pinned VkFFT documents conjugateConvolution=2 but has no executable kernel-conjugation codegen branch",
                ));
            }
            let mixed_storage = matches!(
                self.precision,
                Precision::F16StorageF32Compute | Precision::F64ComputeF32Storage
            );
            if !matches!(
                self.precision,
                Precision::F16StorageF32Compute
                    | Precision::F32
                    | Precision::F64
                    | Precision::F64ComputeF32Storage
            ) {
                return Err(VkFftError::UnsupportedKernelPath(
                    "performConvolution currently requires ordinary or mixed F16/F32/F64 storage",
                ));
            }
            let mixed_c2c_scalar_k1 = self.transform == TransformKind::ComplexToComplex
                && self.dimensions.len() == 1
                && self.batch_count == 1
                && self.coordinate_features == 1
                && self.matrix_convolution == 1
                && !self.symmetric_convolution_kernel
                && self.convolution_kernel_count == 1
                && !has_formatted_strides
                && self.zero_padding.iter().all(Option::is_none)
                && self.convolution_conjugation == ConvolutionConjugation::None
                && !self.cross_power_spectrum_normalization;
            let mixed_scalar_layout = self.coordinate_features == 1
                && self.matrix_convolution == 1
                && (self.convolution_kernel_count == 1 || !has_formatted_strides);
            let mixed_independent_fanout_layout = self.coordinate_features > 1
                && self.matrix_convolution == 1
                && self.convolution_kernel_count > 1
                && !has_formatted_strides;
            let mixed_matrix_layout = self.matrix_convolution == 3
                && self.coordinate_features == 3
                && !has_formatted_strides;
            if mixed_storage
                && !(mixed_c2c_scalar_k1
                    || (real_application
                        && self.batch_count == 1
                        && (mixed_scalar_layout
                            || mixed_independent_fanout_layout
                            || mixed_matrix_layout)))
            {
                return Err(VkFftError::UnsupportedKernelPath(
                    "mixed-storage performConvolution currently requires dense default-policy 1D C2C batch1 scalar K1, batch1 scalar Real ownership, unformatted independent-coordinate K>1 Real fan-out, or unformatted 3x3 Real matrix ownership",
                ));
            }
            if has_formatted_strides {
                let real_formatted_k1 = real_application
                    && self.batch_count == 1
                    && self.coordinate_features == 1
                    && self.matrix_convolution == 1
                    && self.convolution_kernel_count == 1;
                if !real_formatted_k1 {
                    return Err(VkFftError::UnsupportedKernelPath(
                        "formatted performConvolution currently requires batch1 scalar ND Real K1",
                    ));
                }
            }
            if self.zero_padding.iter().any(Option::is_some) {
                let real_zero_padding = real_application
                    && self.batch_count == 1
                    && !self.symmetric_convolution_kernel
                    && (self.matrix_convolution == 1
                        || (self.matrix_convolution > 1
                            && self.coordinate_features == self.matrix_convolution));
                if !real_zero_padding {
                    return Err(VkFftError::UnsupportedKernelPath(
                        "performConvolution zero padding requires a pinned batch1 multidimensional real scalar/independent-coordinate/nonsymmetric-matrix ownership",
                    ));
                }
            }
            if self.omit_dimension.iter().any(|&omit| omit) {
                return Err(VkFftError::UnsupportedKernelPath(
                    "omitDimension does not compose with performConvolution",
                ));
            }
        }
        if has_formatted_strides {
            let multidimensional = self.dimensions.len() >= 2;
            let double_double = matches!(
                self.precision,
                Precision::DoubleDouble | Precision::DoubleDoubleF64Storage
            );
            let ordinary_supported = multidimensional
                && !double_double
                && matches!(
                    self.transform,
                    TransformKind::ComplexToComplex
                        | TransformKind::RealToComplex
                        | TransformKind::ComplexToReal
                        | TransformKind::Dct(_)
                        | TransformKind::Dst(_)
                );
            let double_double_supported = multidimensional
                && double_double
                && matches!(
                    self.transform,
                    TransformKind::ComplexToComplex
                        | TransformKind::RealToComplex
                        | TransformKind::ComplexToReal
                        | TransformKind::Dct(_)
                        | TransformKind::Dst(_)
                );
            if !ordinary_supported && !double_double_supported {
                return Err(VkFftError::UnsupportedKernelPath(
                    "formatted strides currently support ordinary and double-double multidimensional C2C, real, and R2R transforms",
                ));
            }
            if self.dimensions.len() > 4 {
                return Err(VkFftError::UnsupportedKernelPath(
                    "formatted custom strides follow upstream VKFFT_MAX_FFT_DIMENSIONS=4",
                ));
            }
            let mut compact_dimensions = self.dimensions.clone();
            if matches!(
                self.transform,
                TransformKind::RealToComplex | TransformKind::ComplexToReal
            ) {
                let fastest = compact_dimensions.len() - 1;
                compact_dimensions[fastest] = compact_dimensions[fastest] / 2 + 1;
            }
            let (input_dimensions, output_dimensions) =
                if self.perform_convolution && self.transform == TransformKind::RealToComplex {
                    (&self.dimensions, &self.dimensions)
                } else {
                    match self.transform {
                        TransformKind::RealToComplex => (&self.dimensions, &compact_dimensions),
                        TransformKind::ComplexToReal => (&compact_dimensions, &self.dimensions),
                        _ => (&self.dimensions, &self.dimensions),
                    }
                };
            let input_axis_strides =
                self.resolved_input_buffer_axis_strides_for(input_dimensions)?;
            let output_axis_strides =
                self.resolved_output_buffer_axis_strides_for(output_dimensions)?;
            let _ = resolve_formatted_batch_stride(
                input_dimensions,
                &input_axis_strides,
                self.input_buffer_batch_stride,
                "formatted input batch stride",
            )?;
            let _ = resolve_formatted_batch_stride(
                output_dimensions,
                &output_axis_strides,
                self.output_buffer_batch_stride,
                "formatted output batch stride",
            )?;
        }
        if self.grouped_batch.len() != self.dimensions.len() {
            return Err(VkFftError::InvalidKernelIr(
                "groupedBatch metadata must match the FFT dimensionality",
            ));
        }
        if self.grouped_batch.iter().any(Option::is_some) {
            let one_dimensional_c2c =
                self.dimensions.len() == 1 && self.transform == TransformKind::ComplexToComplex;
            let double_double_precision = matches!(
                self.precision,
                Precision::DoubleDouble | Precision::DoubleDoubleF64Storage
            );
            let multidimensional_c2c =
                self.dimensions.len() > 1 && self.transform == TransformKind::ComplexToComplex;
            let one_dimensional_real = self.dimensions.len() == 1
                && matches!(
                    self.transform,
                    TransformKind::RealToComplex | TransformKind::ComplexToReal
                );
            let one_dimensional_r2r = self.dimensions.len() == 1
                && matches!(
                    self.transform,
                    TransformKind::Dct(_) | TransformKind::Dst(_)
                );
            let multidimensional_real = self.dimensions.len() > 1
                && matches!(
                    self.transform,
                    TransformKind::RealToComplex | TransformKind::ComplexToReal
                );
            let multidimensional_r2r = self.dimensions.len() > 1
                && matches!(
                    self.transform,
                    TransformKind::Dct(_) | TransformKind::Dst(_)
                );
            let double_double_real = matches!(
                self.transform,
                TransformKind::RealToComplex | TransformKind::ComplexToReal
            ) && double_double_precision;
            let double_double_r2r = matches!(
                self.transform,
                TransformKind::Dct(_) | TransformKind::Dst(_)
            ) && double_double_precision;
            if !one_dimensional_c2c
                && !multidimensional_c2c
                && !one_dimensional_real
                && !one_dimensional_r2r
                && !multidimensional_real
                && !multidimensional_r2r
                && !double_double_real
                && !double_double_r2r
            {
                return Err(VkFftError::UnsupportedKernelPath(
                    "groupedBatch override currently supports C2C, real, and R2R transforms in one or multiple dimensions",
                ));
            }
        }
        if self.zero_padding_domain == ZeroPaddingDomain::Frequency
            && self.zero_padding.iter().any(Option::is_some)
            && self.transform != TransformKind::ComplexToComplex
        {
            return Err(VkFftError::UnsupportedKernelPath(
                "frequency-domain zero padding currently supports C2C transforms only",
            ));
        }
        for (axis, &length) in self.dimensions.iter().enumerate() {
            if length == 0 {
                return Err(VkFftError::ZeroLength { axis });
            }
            if matches!(self.transform, TransformKind::Dct(DctType::I))
                && !self.axis_is_omitted(axis)
                && length < 2
            {
                return Err(VkFftError::InvalidTransformLength {
                    axis,
                    transform: "DCT-I",
                    length,
                });
            }
            if let Some(range) = self.zero_padding[axis] {
                range.validate(axis, length)?;
            }
        }
        self.tuning.validate()
    }
}

fn resolve_formatted_axis_strides(
    dimensions: &[usize],
    overrides: &[Option<usize>],
    field: &'static str,
) -> Result<Vec<usize>> {
    if dimensions.is_empty() {
        return Err(VkFftError::EmptyDimensions);
    }
    if overrides.len() != dimensions.len() {
        return Err(VkFftError::InvalidKernelIr(
            "formatted axis-stride metadata must match the FFT dimensionality",
        ));
    }
    let rank = dimensions.len();
    let fastest = rank - 1;
    if let Some(stride) = overrides[fastest]
        && stride != 1
    {
        return Err(VkFftError::InvalidKernelIr(
            "formatted fastest-axis stride must remain one stored element",
        ));
    }
    let mut strides = vec![1usize; rank];
    for axis in (0..fastest).rev() {
        let minimum = strides[axis + 1]
            .checked_mul(dimensions[axis + 1])
            .ok_or(VkFftError::ArithmeticOverflow { operation: field })?;
        let stride = overrides[axis].unwrap_or(minimum);
        if stride < minimum {
            return Err(VkFftError::InvalidKernelIr(
                "formatted axis stride overlaps the next faster logical axis",
            ));
        }
        strides[axis] = stride;
    }
    Ok(strides)
}

fn resolve_formatted_batch_stride(
    dimensions: &[usize],
    axis_strides: &[usize],
    override_stride: Option<usize>,
    field: &'static str,
) -> Result<usize> {
    if dimensions.is_empty() || axis_strides.len() != dimensions.len() {
        return Err(VkFftError::InvalidKernelIr(
            "formatted batch-stride metadata must match the FFT dimensionality",
        ));
    }
    let minimum = axis_strides[0]
        .checked_mul(dimensions[0])
        .ok_or(VkFftError::ArithmeticOverflow { operation: field })?;
    let stride = override_stride.unwrap_or(minimum);
    if stride < minimum {
        return Err(VkFftError::InvalidKernelIr(
            "formatted batch stride overlaps the logical tensor",
        ));
    }
    Ok(stride)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_zero_dimension() {
        let err = FftConfig::new(vec![16, 0]).validate().unwrap_err();
        assert_eq!(err, VkFftError::ZeroLength { axis: 1 });
    }

    #[test]
    fn precision_layout_separates_storage_from_compute_width() {
        let f16 = Precision::F16StorageF32Compute.layout();
        assert_eq!(f16.storage, PrecisionStorage::F16);
        assert_eq!(f16.compute, PrecisionCompute::F32);
        assert_eq!(f16.storage_complex_bytes(), 4);
        assert_eq!(f16.compute_complex_bytes(), 8);
        assert!(f16.has_mixed_storage_compute());

        let mixed_f64 = Precision::F64ComputeF32Storage.layout();
        assert_eq!(mixed_f64.storage, PrecisionStorage::F32);
        assert_eq!(mixed_f64.compute, PrecisionCompute::F64);
        assert_eq!(mixed_f64.storage_complex_bytes(), 8);
        assert_eq!(mixed_f64.compute_complex_bytes(), 16);
        assert_eq!(Precision::F64ComputeF32Storage.complex_bytes(), 16);

        let dd_f64 = Precision::DoubleDoubleF64Storage.layout();
        assert_eq!(dd_f64.storage, PrecisionStorage::F64);
        assert_eq!(dd_f64.compute, PrecisionCompute::DoubleDouble);
        assert_eq!(dd_f64.storage_complex_bytes(), 16);
        assert_eq!(dd_f64.compute_complex_bytes(), 32);
        assert!(dd_f64.has_mixed_storage_compute());
    }

    #[test]
    fn grouped_batch_override_is_axis_typed_and_zero_clears_it() {
        let config = FftConfig::new(vec![64]).with_grouped_batch(0, 8).unwrap();
        assert_eq!(config.grouped_batch_for_axis(0), Some(8));
        let config = config.with_grouped_batch(0, 0).unwrap();
        assert_eq!(config.grouped_batch_for_axis(0), None);
        assert!(config.validate().is_ok());

        assert!(matches!(
            FftConfig::new(vec![64]).with_grouped_batch(1, 8),
            Err(VkFftError::ValueOutOfRange {
                field: "groupedBatch axis"
            })
        ));
        let nd_c2c = FftConfig::new(vec![8, 8])
            .with_grouped_batch(0, 4)
            .unwrap()
            .with_grouped_batch(1, 3)
            .unwrap();
        assert_eq!(nd_c2c.grouped_batch_for_axis(0), Some(4));
        assert_eq!(nd_c2c.grouped_batch_for_axis(1), Some(3));
        assert!(nd_c2c.validate().is_ok());
        let dd_nd = FftConfig::new(vec![8, 8])
            .with_batch_count(7)
            .with_precision(Precision::DoubleDouble)
            .with_grouped_batch(0, 3)
            .unwrap()
            .with_grouped_batch(1, 5)
            .unwrap();
        assert_eq!(dd_nd.grouped_batch_for_axis(0), Some(3));
        assert_eq!(dd_nd.grouped_batch_for_axis(1), Some(5));
        assert!(dd_nd.validate().is_ok());

        for transform in [
            TransformKind::RealToComplex,
            TransformKind::ComplexToReal,
            TransformKind::Dct(DctType::II),
            TransformKind::Dst(DstType::IV),
        ] {
            let grouped = FftConfig::new(vec![64])
                .with_transform(transform)
                .with_grouped_batch(0, 4)
                .unwrap();
            assert_eq!(grouped.grouped_batch_for_axis(0), Some(4));
            assert!(grouped.validate().is_ok());
        }
        for transform in [
            TransformKind::RealToComplex,
            TransformKind::ComplexToReal,
            TransformKind::Dct(DctType::II),
            TransformKind::Dst(DstType::IV),
        ] {
            let grouped = FftConfig::new(vec![8, 8])
                .with_transform(transform)
                .with_grouped_batch(0, 4)
                .unwrap()
                .with_grouped_batch(1, 3)
                .unwrap();
            assert_eq!(grouped.grouped_batch_for_axis(0), Some(4));
            assert_eq!(grouped.grouped_batch_for_axis(1), Some(3));
            assert!(grouped.validate().is_ok());
        }
    }

    #[test]
    fn omit_dimension_supports_nd_c2c_real_r2r_with_upstream_real_axis_guard() {
        let config = FftConfig::new(vec![8, 1, 4])
            .with_omit_dimension(0, true)
            .unwrap();
        assert!(config.axis_is_omitted(0));
        assert!(config.axis_is_omitted(1));
        assert!(!config.axis_is_omitted(2));
        assert!(config.validate().is_ok());

        let err = FftConfig::new(vec![16])
            .with_omit_dimension(0, true)
            .unwrap()
            .validate()
            .unwrap_err();
        assert_eq!(
            err,
            VkFftError::UnsupportedKernelPath(
                "omitDimension currently supports multidimensional transforms only"
            )
        );

        let real_outer = FftConfig::new(vec![1, 8])
            .with_transform(TransformKind::RealToComplex)
            .with_omit_dimension(0, true)
            .unwrap();
        assert!(real_outer.axis_is_omitted(0));
        assert!(!real_outer.axis_is_omitted(1));
        assert!(real_outer.validate().is_ok());
        let err = FftConfig::new(vec![8, 4])
            .with_transform(TransformKind::RealToComplex)
            .with_omit_dimension(1, true)
            .unwrap()
            .validate()
            .unwrap_err();
        assert_eq!(
            err,
            VkFftError::UnsupportedKernelPath(
                "omitDimension cannot disable the contiguous R2C/C2R axis"
            )
        );

        let r2r = FftConfig::new(vec![1, 4]).with_transform(TransformKind::Dct(DctType::I));
        assert!(r2r.axis_is_omitted(0));
        assert!(r2r.validate().is_ok());

        let err = FftConfig::new(vec![1, 1]).validate().unwrap_err();
        assert_eq!(
            err,
            VkFftError::UnsupportedKernelPath(
                "multidimensional transform requires at least one active axis"
            )
        );
    }

    #[test]
    fn formatted_batch_stride_supports_ordinary_and_double_double_nd_transforms() {
        let config = FftConfig::new(vec![3, 4])
            .with_batch_count(2)
            .with_input_buffer_batch_stride(17)
            .with_output_buffer_batch_stride(19);
        assert_eq!(config.input_buffer_batch_stride, Some(17));
        assert_eq!(config.output_buffer_batch_stride, Some(19));
        assert!(config.validate().is_ok());

        let r2r = FftConfig::new(vec![3, 4])
            .with_transform(TransformKind::Dct(DctType::II))
            .with_input_buffer_batch_stride(17);
        assert!(r2r.validate().is_ok());

        for precision in [Precision::DoubleDouble, Precision::DoubleDoubleF64Storage] {
            for transform in [
                TransformKind::ComplexToComplex,
                TransformKind::RealToComplex,
                TransformKind::ComplexToReal,
                TransformKind::Dct(DctType::II),
            ] {
                let dd = FftConfig::new(vec![3, 4])
                    .with_precision(precision)
                    .with_transform(transform)
                    .with_input_buffer_batch_stride(17)
                    .with_output_buffer_batch_stride(19);
                assert!(dd.validate().is_ok(), "{precision:?} {transform:?}");
            }
        }

        let cleared = config
            .clone()
            .with_input_buffer_batch_stride(0)
            .with_output_buffer_batch_stride(0);
        assert_eq!(cleared.input_buffer_batch_stride, None);
        assert_eq!(cleared.output_buffer_batch_stride, None);
        assert!(cleared.validate().is_ok());

        let too_small = FftConfig::new(vec![3, 4])
            .with_input_buffer_batch_stride(11)
            .validate()
            .unwrap_err();
        assert_eq!(
            too_small,
            VkFftError::InvalidKernelIr("formatted batch stride overlaps the logical tensor")
        );

        let unsupported = FftConfig::new(vec![12]).with_input_buffer_batch_stride(17);
        assert_eq!(
            unsupported.validate().unwrap_err(),
            VkFftError::UnsupportedKernelPath(
                "formatted strides currently support ordinary and double-double multidimensional C2C, real, and R2R transforms"
            )
        );

        let padded = FftConfig::new(vec![3, 4])
            .with_input_buffer_batch_stride(17)
            .with_zero_padding(1, 1, 2)
            .unwrap();
        assert!(padded.validate().is_ok());
    }

    #[test]
    fn formatted_tensor_strides_resolve_row_plane_and_batch_pitches() {
        let two_d = FftConfig::new(vec![3, 4])
            .with_batch_count(2)
            .with_input_buffer_axis_stride(0, 7)
            .unwrap()
            .with_output_buffer_axis_stride(0, 9)
            .unwrap();
        assert_eq!(
            two_d.resolved_input_buffer_axis_strides().unwrap(),
            vec![7, 1]
        );
        assert_eq!(
            two_d.resolved_output_buffer_axis_strides().unwrap(),
            vec![9, 1]
        );
        assert_eq!(two_d.resolved_input_buffer_batch_stride().unwrap(), 21);
        assert_eq!(two_d.resolved_output_buffer_batch_stride().unwrap(), 27);
        assert!(two_d.validate().is_ok());

        let three_d = FftConfig::new(vec![2, 3, 4])
            .with_input_buffer_axis_stride(1, 6)
            .unwrap()
            .with_input_buffer_axis_stride(0, 20)
            .unwrap();
        assert_eq!(
            three_d.resolved_input_buffer_axis_strides().unwrap(),
            vec![20, 6, 1]
        );
        assert_eq!(three_d.resolved_input_buffer_batch_stride().unwrap(), 40);
        assert!(three_d.validate().is_ok());

        let derived_plane = FftConfig::new(vec![2, 3, 4])
            .with_input_buffer_axis_stride(1, 6)
            .unwrap();
        assert_eq!(
            derived_plane.resolved_input_buffer_axis_strides().unwrap(),
            vec![18, 6, 1]
        );
        assert_eq!(
            derived_plane.resolved_input_buffer_batch_stride().unwrap(),
            36
        );

        let cleared = two_d
            .clone()
            .with_input_buffer_axis_stride(0, 0)
            .unwrap()
            .with_output_buffer_axis_stride(0, 0)
            .unwrap();
        assert_eq!(cleared.input_buffer_axis_stride_for_axis(0), None);
        assert_eq!(cleared.output_buffer_axis_stride_for_axis(0), None);
        assert_eq!(
            cleared.resolved_input_buffer_axis_strides().unwrap(),
            vec![4, 1]
        );
        assert!(cleared.validate().is_ok());
    }

    #[test]
    fn formatted_real_strides_resolve_full_and_compact_tensor_shapes() {
        let r2c = FftConfig::new(vec![3, 8])
            .with_batch_count(2)
            .with_transform(TransformKind::RealToComplex)
            .with_input_buffer_axis_stride(0, 11)
            .unwrap()
            .with_output_buffer_axis_stride(0, 7)
            .unwrap();
        let compact = vec![3, 5];
        assert_eq!(
            r2c.resolved_input_buffer_axis_strides_for(&r2c.dimensions)
                .unwrap(),
            vec![11, 1]
        );
        assert_eq!(
            r2c.resolved_output_buffer_axis_strides_for(&compact)
                .unwrap(),
            vec![7, 1]
        );
        assert_eq!(
            r2c.resolved_input_buffer_batch_stride_for(&r2c.dimensions)
                .unwrap(),
            33
        );
        assert_eq!(
            r2c.resolved_output_buffer_batch_stride_for(&compact)
                .unwrap(),
            21
        );
        assert!(r2c.validate().is_ok());

        let c2r = FftConfig::new(vec![3, 8])
            .with_batch_count(2)
            .with_transform(TransformKind::ComplexToReal)
            .with_input_buffer_axis_stride(0, 7)
            .unwrap()
            .with_output_buffer_axis_stride(0, 11)
            .unwrap();
        assert_eq!(
            c2r.resolved_input_buffer_axis_strides_for(&compact)
                .unwrap(),
            vec![7, 1]
        );
        assert_eq!(
            c2r.resolved_output_buffer_axis_strides_for(&c2r.dimensions)
                .unwrap(),
            vec![11, 1]
        );
        assert_eq!(
            c2r.resolved_input_buffer_batch_stride_for(&compact)
                .unwrap(),
            21
        );
        assert_eq!(
            c2r.resolved_output_buffer_batch_stride_for(&c2r.dimensions)
                .unwrap(),
            33
        );
        assert!(c2r.validate().is_ok());

        let padded_real = r2c.with_zero_padding(1, 2, 4).unwrap();
        assert!(padded_real.validate().is_ok());
    }

    #[test]
    fn formatted_tensor_strides_fail_closed_on_overlap_and_unsupported_surfaces() {
        for config in [
            FftConfig::new(vec![3, 4])
                .with_input_buffer_axis_stride(1, 2)
                .unwrap(),
            FftConfig::new(vec![3, 4])
                .with_input_buffer_axis_stride(0, 3)
                .unwrap(),
            FftConfig::new(vec![2, 3, 4])
                .with_input_buffer_axis_stride(1, 6)
                .unwrap()
                .with_input_buffer_axis_stride(0, 17)
                .unwrap(),
        ] {
            assert!(matches!(
                config.validate(),
                Err(VkFftError::InvalidKernelIr(_))
            ));
        }

        let short_batch = FftConfig::new(vec![3, 4])
            .with_input_buffer_axis_stride(0, 7)
            .unwrap()
            .with_input_buffer_batch_stride(20);
        assert_eq!(
            short_batch.validate().unwrap_err(),
            VkFftError::InvalidKernelIr("formatted batch stride overlaps the logical tensor")
        );

        let padded = FftConfig::new(vec![3, 4])
            .with_input_buffer_axis_stride(0, 7)
            .unwrap()
            .with_zero_padding(1, 1, 2)
            .unwrap();
        assert!(padded.validate().is_ok());

        let dd_padded = FftConfig::new(vec![3, 4])
            .with_precision(Precision::DoubleDouble)
            .with_input_buffer_axis_stride(0, 7)
            .unwrap()
            .with_output_buffer_axis_stride(0, 9)
            .unwrap()
            .with_zero_padding(1, 1, 3)
            .unwrap();
        assert!(dd_padded.validate().is_ok());

        let dd_r2r_padded = FftConfig::new(vec![3, 4])
            .with_precision(Precision::DoubleDoubleF64Storage)
            .with_transform(TransformKind::Dct(DctType::II))
            .with_input_buffer_axis_stride(0, 7)
            .unwrap()
            .with_output_buffer_axis_stride(0, 9)
            .unwrap()
            .with_zero_padding(1, 1, 3)
            .unwrap();
        assert!(dd_r2r_padded.validate().is_ok());

        let dd_real_padded = FftConfig::new(vec![3, 8])
            .with_precision(Precision::DoubleDouble)
            .with_transform(TransformKind::RealToComplex)
            .with_input_buffer_axis_stride(0, 11)
            .unwrap()
            .with_output_buffer_axis_stride(0, 7)
            .unwrap()
            .with_zero_padding(1, 2, 4)
            .unwrap();
        assert!(dd_real_padded.validate().is_ok());

        let rank_five = FftConfig::new(vec![2, 2, 2, 2, 2])
            .with_input_buffer_axis_stride(3, 3)
            .unwrap();
        assert_eq!(
            rank_five.validate().unwrap_err(),
            VkFftError::UnsupportedKernelPath(
                "formatted custom strides follow upstream VKFFT_MAX_FFT_DIMENSIONS=4"
            )
        );
    }

    #[test]
    fn multidimensional_convolution_admits_scalar_fanout_and_batch1_matrix_surfaces() {
        let scalar = FftConfig::new(vec![3, 4])
            .with_batch_count(2)
            .with_precision(Precision::F32)
            .with_convolution(true);
        assert!(scalar.validate().is_ok());
        for supported in [
            scalar
                .clone()
                .with_convolution_conjugation(ConvolutionConjugation::Sequence),
            scalar.clone().with_cross_power_spectrum_normalization(true),
            scalar
                .clone()
                .with_convolution_conjugation(ConvolutionConjugation::Sequence)
                .with_cross_power_spectrum_normalization(true),
        ] {
            assert!(supported.validate().is_ok());
        }

        for fanout in [
            FftConfig::new(vec![3, 4])
                .with_precision(Precision::F32)
                .with_convolution(true)
                .with_convolution_kernel_count(3),
            FftConfig::new(vec![3, 4])
                .with_precision(Precision::F32)
                .with_convolution(true)
                .with_convolution_kernel_count(3)
                .with_convolution_conjugation(ConvolutionConjugation::Sequence)
                .with_cross_power_spectrum_normalization(true),
        ] {
            assert!(fanout.validate().is_ok());
        }

        for matrix in [
            FftConfig::new(vec![3, 4])
                .with_precision(Precision::F32)
                .with_convolution(true)
                .with_matrix_convolution(2),
            FftConfig::new(vec![3, 4])
                .with_precision(Precision::F32)
                .with_convolution(true)
                .with_matrix_convolution(2)
                .with_symmetric_convolution_kernel(true),
            FftConfig::new(vec![3, 4])
                .with_precision(Precision::F32)
                .with_convolution(true)
                .with_matrix_convolution(3)
                .with_convolution_conjugation(ConvolutionConjugation::Sequence)
                .with_cross_power_spectrum_normalization(true),
        ] {
            assert!(matrix.validate().is_ok());
        }

        assert!(matches!(
            scalar.clone().with_convolution_kernel_count(2).validate(),
            Err(VkFftError::UnsupportedKernelPath(message))
                if message.contains("one-input fan-out")
                    && message.contains("numberBatches*numberKernels")
        ));
        for unsupported in [
            scalar.clone().with_matrix_convolution(2),
            FftConfig::new(vec![3, 4])
                .with_convolution(true)
                .with_matrix_convolution(2)
                .with_convolution_kernel_count(2),
        ] {
            assert!(matches!(
                unsupported.validate(),
                Err(VkFftError::UnsupportedKernelPath(
                    "multidimensional performConvolution supports scalar K=1/K>1 layouts or a batch1/K1 matrix layout"
                ))
            ));
        }
        assert!(matches!(
            scalar
                .clone()
                .with_convolution_conjugation(ConvolutionConjugation::Kernel)
                .validate(),
            Err(VkFftError::UnsupportedKernelPath(message))
                if message.contains("no executable kernel-conjugation codegen branch")
        ));
        for real in [
            FftConfig::new(vec![3, 4])
                .with_transform(TransformKind::RealToComplex)
                .with_precision(Precision::F32)
                .with_convolution(true),
            FftConfig::new(vec![3, 4])
                .with_transform(TransformKind::RealToComplex)
                .with_precision(Precision::F32)
                .with_convolution(true)
                .with_convolution_kernel_count(3),
            FftConfig::new(vec![3, 4])
                .with_transform(TransformKind::RealToComplex)
                .with_precision(Precision::F32)
                .with_convolution(true)
                .with_coordinate_features(2)
                .with_convolution_kernel_count(2),
            FftConfig::new(vec![3, 4])
                .with_transform(TransformKind::RealToComplex)
                .with_precision(Precision::F32)
                .with_convolution(true)
                .with_matrix_convolution(3),
            FftConfig::new(vec![3, 4])
                .with_transform(TransformKind::RealToComplex)
                .with_precision(Precision::F32)
                .with_convolution(true)
                .with_matrix_convolution(3)
                .with_convolution_kernel_count(2),
            FftConfig::new(vec![3, 4])
                .with_transform(TransformKind::RealToComplex)
                .with_precision(Precision::F32)
                .with_convolution(true)
                .with_matrix_convolution(3)
                .with_zero_padding(0, 2, 3)
                .unwrap()
                .with_zero_padding(1, 2, 4)
                .unwrap(),
            FftConfig::new(vec![3, 4])
                .with_transform(TransformKind::RealToComplex)
                .with_precision(Precision::F32)
                .with_convolution(true)
                .with_convolution_conjugation(ConvolutionConjugation::Sequence)
                .with_cross_power_spectrum_normalization(true),
            FftConfig::new(vec![3, 4])
                .with_transform(TransformKind::RealToComplex)
                .with_precision(Precision::F32)
                .with_convolution(true)
                .with_convolution_kernel_count(3)
                .with_zero_padding(0, 2, 3)
                .unwrap(),
            FftConfig::new(vec![3, 4])
                .with_transform(TransformKind::RealToComplex)
                .with_precision(Precision::F32)
                .with_convolution(true)
                .with_coordinate_features(2)
                .with_convolution_kernel_count(2)
                .with_convolution_conjugation(ConvolutionConjugation::Sequence)
                .with_cross_power_spectrum_normalization(true)
                .with_zero_padding(0, 2, 3)
                .unwrap(),
            FftConfig::new(vec![3, 4])
                .with_transform(TransformKind::RealToComplex)
                .with_precision(Precision::F32)
                .with_convolution(true)
                .with_matrix_convolution(3)
                .with_convolution_kernel_count(2)
                .with_convolution_conjugation(ConvolutionConjugation::Sequence)
                .with_cross_power_spectrum_normalization(true)
                .with_zero_padding(0, 2, 3)
                .unwrap()
                .with_zero_padding(1, 2, 4)
                .unwrap(),
        ] {
            assert!(real.validate().is_ok());
        }
        let real_batch2 = FftConfig::new(vec![3, 4])
            .with_transform(TransformKind::RealToComplex)
            .with_batch_count(2)
            .with_convolution(true);
        assert!(matches!(
            real_batch2.validate(),
            Err(VkFftError::UnsupportedKernelPath(
                "multidimensional real performConvolution supports batch1 independent-coordinate or matrix K=1/K>1 layouts"
            ))
        ));
        assert!(matches!(
            FftConfig::new(vec![3, 4])
                .with_transform(TransformKind::RealToComplex)
                .with_convolution(true)
                .with_matrix_convolution(2)
                .with_symmetric_convolution_kernel(true)
                .with_zero_padding(0, 2, 3)
                .unwrap()
                .validate(),
            Err(VkFftError::UnsupportedKernelPath(
                "performConvolution zero padding requires a pinned batch1 multidimensional real scalar/independent-coordinate/nonsymmetric-matrix ownership"
            ))
        ));
        assert!(matches!(
            FftConfig::new(vec![3, 4])
                .with_transform(TransformKind::RealToComplex)
                .with_convolution(true)
                .with_convolution_conjugation(ConvolutionConjugation::Kernel)
                .validate(),
            Err(VkFftError::UnsupportedKernelPath(message))
                if message.contains("no executable kernel-conjugation codegen branch")
        ));
        assert!(matches!(
            FftConfig::new(vec![3, 4])
                .with_transform(TransformKind::ComplexToReal)
                .with_convolution(true)
                .validate(),
            Err(VkFftError::UnsupportedKernelPath(
                "performConvolution supports C2C plus multidimensional R2C application transforms"
            ))
        ));
        assert!(matches!(
            scalar
                .with_input_buffer_axis_stride(0, 7)
                .unwrap()
                .validate(),
            Err(VkFftError::UnsupportedKernelPath(
                "formatted performConvolution currently requires batch1 scalar ND Real K1"
            ))
        ));
        let formatted_real = FftConfig::new(vec![3, 4])
            .with_transform(TransformKind::RealToComplex)
            .with_convolution(true)
            .with_input_buffer_axis_stride(0, 7)
            .unwrap()
            .with_output_buffer_axis_stride(0, 9)
            .unwrap();
        assert!(formatted_real.validate().is_ok());
        assert!(
            FftConfig::new(vec![3, 4])
                .with_transform(TransformKind::RealToComplex)
                .with_convolution(true)
                .with_output_buffer_axis_stride(0, 3)
                .unwrap()
                .validate()
                .is_err()
        );
        assert!(
            formatted_real
                .clone()
                .with_zero_padding(0, 2, 3)
                .unwrap()
                .validate()
                .is_ok()
        );
        for unsupported in [
            formatted_real.clone().with_convolution_kernel_count(2),
            formatted_real.clone().with_coordinate_features(2),
            formatted_real.clone().with_matrix_convolution(2),
        ] {
            assert!(matches!(
                unsupported.validate(),
                Err(VkFftError::UnsupportedKernelPath(
                    "formatted performConvolution currently requires batch1 scalar ND Real K1"
                ))
            ));
        }
    }

    #[test]
    fn kernel_convolution_first_contract_is_forward_kernel_preparation_only() {
        for precision in [Precision::F32, Precision::F64] {
            let base = FftConfig::new(vec![16])
                .with_precision(precision)
                .with_kernel_convolution(true);
            assert!(base.validate().is_ok(), "{precision:?}");
            assert!(
                base.clone()
                    .with_transform(TransformKind::RealToComplex)
                    .validate()
                    .is_ok(),
                "{precision:?} R2C kernel preparation should be admitted"
            );
            let independent = base.clone().with_batch_count(2).with_coordinate_features(2);
            assert!(
                independent.validate().is_ok(),
                "{precision:?} independent C2C kernel preparation should be admitted"
            );
            assert!(
                independent
                    .clone()
                    .with_transform(TransformKind::RealToComplex)
                    .validate()
                    .is_ok(),
                "{precision:?} independent R2C kernel preparation should be admitted"
            );
            assert_eq!(independent.kernel_preparation_system_count().unwrap(), 4);
            let sample52_nd = FftConfig::new(vec![4, 8])
                .with_precision(precision)
                .with_transform(TransformKind::RealToComplex)
                .with_kernel_convolution(true)
                .with_batch_count(2)
                .with_coordinate_features(2);
            assert!(
                sample52_nd.validate().is_ok(),
                "{precision:?} sample_52-style 2D B2/C2 preparation should be admitted"
            );
            assert_eq!(sample52_nd.kernel_preparation_system_count().unwrap(), 4);
            for unsupported_sample52 in [
                sample52_nd.clone().with_batch_count(1),
                sample52_nd.clone().with_coordinate_features(3),
                sample52_nd.clone().with_zero_padding(0, 2, 4).unwrap(),
                sample52_nd
                    .clone()
                    .with_transform(TransformKind::ComplexToComplex),
            ] {
                assert!(matches!(
                    unsupported_sample52.validate().unwrap_err(),
                    VkFftError::UnsupportedKernelPath(_)
                ));
            }
            let sample51 = FftConfig::new(vec![4, 4, 8])
                .with_precision(precision)
                .with_transform(TransformKind::RealToComplex)
                .with_kernel_convolution(true)
                .with_coordinate_features(9)
                .with_zero_padding(0, 2, 4)
                .unwrap()
                .with_zero_padding(1, 2, 4)
                .unwrap()
                .with_zero_padding(2, 4, 8)
                .unwrap();
            assert!(
                sample51.validate().is_ok(),
                "{precision:?} sample_51-style C9 padded ND preparation should be admitted"
            );
            assert_eq!(sample51.kernel_preparation_system_count().unwrap(), 9);
            for unsupported_sample51 in [
                sample51.clone().with_batch_count(2),
                FftConfig::new(vec![4, 4, 8])
                    .with_precision(precision)
                    .with_transform(TransformKind::RealToComplex)
                    .with_kernel_convolution(true)
                    .with_coordinate_features(8)
                    .with_zero_padding(0, 2, 4)
                    .unwrap()
                    .with_zero_padding(1, 2, 4)
                    .unwrap()
                    .with_zero_padding(2, 4, 8)
                    .unwrap(),
                FftConfig::new(vec![4, 4, 8])
                    .with_precision(precision)
                    .with_transform(TransformKind::RealToComplex)
                    .with_kernel_convolution(true)
                    .with_coordinate_features(9)
                    .with_zero_padding(0, 2, 4)
                    .unwrap()
                    .with_zero_padding(1, 2, 4)
                    .unwrap(),
                sample51
                    .clone()
                    .with_transform(TransformKind::ComplexToComplex),
            ] {
                assert!(matches!(
                    unsupported_sample51.validate().unwrap_err(),
                    VkFftError::UnsupportedKernelPath(_)
                ));
            }
            for unsupported in [
                base.clone().with_convolution(true),
                base.clone().with_convolution_kernel_count(2),
                base.clone().with_transform(TransformKind::ComplexToReal),
                base.clone().with_precision(Precision::F16StorageF32Compute),
                base.clone().with_zero_padding(0, 4, 8).unwrap(),
                base.clone().with_input_buffer_batch_stride(20),
                base.clone().with_matrix_convolution(2),
            ] {
                assert!(matches!(
                    unsupported.validate().unwrap_err(),
                    VkFftError::UnsupportedKernelPath(_)
                ));
            }
        }
    }

    #[test]
    fn one_dimensional_c2c_convolution_mixed_storage_keeps_first_contract_narrow() {
        for precision in [
            Precision::F16StorageF32Compute,
            Precision::F64ComputeF32Storage,
        ] {
            let base = FftConfig::new(vec![16])
                .with_precision(precision)
                .with_convolution(true);
            assert!(base.validate().is_ok(), "{precision:?}");
            for unsupported in [
                base.clone().with_batch_count(2),
                base.clone().with_convolution_kernel_count(2),
                base.clone()
                    .with_convolution_conjugation(ConvolutionConjugation::Sequence),
                base.clone().with_cross_power_spectrum_normalization(true),
                base.clone().with_zero_padding(0, 4, 8).unwrap(),
                base.clone().with_input_buffer_batch_stride(20),
            ] {
                assert_eq!(
                    unsupported.validate().unwrap_err(),
                    VkFftError::UnsupportedKernelPath(
                        "mixed-storage performConvolution currently requires dense default-policy 1D C2C batch1 scalar K1, batch1 scalar Real ownership, unformatted independent-coordinate K>1 Real fan-out, or unformatted 3x3 Real matrix ownership"
                    ),
                    "{precision:?} unexpectedly escaped the mixed-storage 1D C2C boundary"
                );
            }
        }
    }

    #[test]
    fn multidimensional_real_convolution_mixed_storage_keeps_each_boundary_composition_explicit() {
        for precision in [
            Precision::F16StorageF32Compute,
            Precision::F64ComputeF32Storage,
        ] {
            let base = FftConfig::new(vec![3, 4])
                .with_transform(TransformKind::RealToComplex)
                .with_precision(precision)
                .with_convolution(true);
            assert!(base.validate().is_ok(), "{precision:?}");
            let padded = base.clone().with_zero_padding(0, 2, 3).unwrap();
            assert!(padded.validate().is_ok(), "{precision:?} spatial padding");
            let formatted = base.clone().with_input_buffer_axis_stride(0, 7).unwrap();
            assert!(
                formatted.validate().is_ok(),
                "{precision:?} formatted input"
            );
            let formatted = formatted.with_output_buffer_axis_stride(0, 9).unwrap();
            assert!(
                formatted.validate().is_ok(),
                "{precision:?} formatted input/output"
            );
            let formatted_padded = formatted.clone().with_zero_padding(0, 2, 3).unwrap();
            assert!(
                formatted_padded.validate().is_ok(),
                "{precision:?} formatted spatial padding"
            );
            let fanout = base.clone().with_convolution_kernel_count(3);
            assert!(
                fanout.validate().is_ok(),
                "{precision:?} dense scalar K3 fan-out"
            );
            let fanout_padded = fanout.clone().with_zero_padding(0, 2, 3).unwrap();
            assert!(
                fanout_padded.validate().is_ok(),
                "{precision:?} scalar K3 spatial padding"
            );
            let independent = base
                .clone()
                .with_coordinate_features(2)
                .with_convolution_kernel_count(2);
            assert!(
                independent.validate().is_ok(),
                "{precision:?} independent-coordinate C2/K2 fan-out"
            );
            let independent_padded = independent.clone().with_zero_padding(0, 2, 3).unwrap();
            assert!(
                independent_padded.validate().is_ok(),
                "{precision:?} independent-coordinate C2/K2 spatial padding"
            );
            let matrix_k1 = base.clone().with_matrix_convolution(3);
            assert!(
                matrix_k1.validate().is_ok(),
                "{precision:?} dense 3x3 matrix K1"
            );
            let matrix_k1_padded = matrix_k1.clone().with_zero_padding(0, 2, 3).unwrap();
            assert!(
                matrix_k1_padded.validate().is_ok(),
                "{precision:?} 3x3 matrix K1 spatial padding"
            );
            let matrix_k2 = matrix_k1.clone().with_convolution_kernel_count(2);
            assert!(
                matrix_k2.validate().is_ok(),
                "{precision:?} dense 3x3 matrix K2 fan-out"
            );
            let matrix_k2_padded = matrix_k2.clone().with_zero_padding(0, 2, 3).unwrap();
            assert!(
                matrix_k2_padded.validate().is_ok(),
                "{precision:?} 3x3 matrix K2 spatial padding"
            );

            for unsupported in [
                base.clone().with_coordinate_features(2),
                base.clone().with_matrix_convolution(2),
                padded.clone().with_coordinate_features(2),
                padded.clone().with_matrix_convolution(2),
                formatted.clone().with_convolution_kernel_count(2),
                formatted.clone().with_coordinate_features(2),
                formatted.clone().with_matrix_convolution(2),
                formatted_padded.clone().with_convolution_kernel_count(2),
                formatted_padded.clone().with_coordinate_features(2),
                formatted_padded.clone().with_matrix_convolution(2),
                independent
                    .clone()
                    .with_input_buffer_axis_stride(0, 7)
                    .unwrap(),
                independent.clone().with_matrix_convolution(2),
                matrix_k1
                    .clone()
                    .with_input_buffer_axis_stride(0, 7)
                    .unwrap(),
                matrix_k2
                    .clone()
                    .with_input_buffer_axis_stride(0, 7)
                    .unwrap(),
            ] {
                assert_eq!(
                    unsupported.validate().unwrap_err(),
                    VkFftError::UnsupportedKernelPath(
                        "mixed-storage performConvolution currently requires dense default-policy 1D C2C batch1 scalar K1, batch1 scalar Real ownership, unformatted independent-coordinate K>1 Real fan-out, or unformatted 3x3 Real matrix ownership"
                    ),
                    "{precision:?} unexpectedly escaped the mixed-storage scalar/independent admission"
                );
            }
        }
    }

    #[test]
    fn frequency_zero_padding_is_c2c_only_and_preserves_default_spatial_domain() {
        let spatial = FftConfig::new(vec![16]).with_zero_padding(0, 4, 8).unwrap();
        assert_eq!(spatial.zero_padding_domain, ZeroPaddingDomain::Spatial);
        assert!(spatial.validate().is_ok());

        let frequency = FftConfig::new(vec![8, 16])
            .with_zero_padding(1, 4, 8)
            .unwrap()
            .with_zero_padding_domain(ZeroPaddingDomain::Frequency);
        assert_eq!(frequency.zero_padding_domain, ZeroPaddingDomain::Frequency);
        assert!(frequency.validate().is_ok());

        for transform in [
            TransformKind::RealToComplex,
            TransformKind::ComplexToReal,
            TransformKind::Dct(DctType::II),
            TransformKind::Dst(DstType::IV),
        ] {
            let err = FftConfig::new(vec![16])
                .with_transform(transform)
                .with_zero_padding(0, 4, 8)
                .unwrap()
                .with_zero_padding_domain(ZeroPaddingDomain::Frequency)
                .validate()
                .unwrap_err();
            assert_eq!(
                err,
                VkFftError::UnsupportedKernelPath(
                    "frequency-domain zero padding currently supports C2C transforms only"
                )
            );
        }
    }

    #[test]
    fn planner_tuning_profile_round_trips_and_fingerprint_is_stable() {
        let tuning = PlannerTuning::portable().with_recursive_fft_rader(true);
        let bytes = tuning.to_profile_bytes();
        assert_eq!(bytes.len(), PLANNER_TUNING_PROFILE_BYTES);
        assert_eq!(PlannerTuning::from_profile_bytes(&bytes).unwrap(), tuning);
        assert_eq!(tuning.profile_fingerprint(), tuning.profile_fingerprint());

        let mut changed = tuning;
        changed.max_rader_fft_prime -= 1;
        assert_ne!(changed.profile_fingerprint(), tuning.profile_fingerprint());

        let mut bad_version = bytes.clone();
        bad_version[8..12].copy_from_slice(&(PLANNER_TUNING_PROFILE_VERSION + 1).to_le_bytes());
        assert!(matches!(
            PlannerTuning::from_profile_bytes(&bad_version),
            Err(VkFftError::InvalidPlannerTuning(_))
        ));

        let mut bad_commit = bytes;
        bad_commit[12] ^= 0x01;
        assert!(matches!(
            PlannerTuning::from_profile_bytes(&bad_commit),
            Err(VkFftError::InvalidPlannerTuning(_))
        ));
    }

    #[test]
    fn subgroup_fast_path_requires_stable_or_enforceable_width() {
        let base = SubgroupProfile {
            size: 32,
            min_size: 32,
            max_size: 32,
            required_size_compute_supported: false,
            compute_supported: true,
            basic_supported: true,
            shuffle_supported: true,
            shuffle_relative_supported: false,
            compute_full_subgroups: true,
        };
        assert!(base.supports_full_subgroup_shuffle_compute());
        assert_eq!(base.required_compute_subgroup_size(), None);

        let variable_uncontrolled = SubgroupProfile {
            min_size: 16,
            ..base
        };
        assert!(!variable_uncontrolled.supports_full_subgroup_shuffle_compute());
        assert_eq!(variable_uncontrolled.required_compute_subgroup_size(), None);

        let variable_controlled = SubgroupProfile {
            required_size_compute_supported: true,
            ..variable_uncontrolled
        };
        assert!(variable_controlled.supports_full_subgroup_shuffle_compute());
        assert_eq!(
            variable_controlled.required_compute_subgroup_size(),
            Some(32)
        );
    }

    #[test]
    fn device_rader_thresholds_match_fixed_upstream_initialization() {
        let cases = [
            (
                GpuVendor::Nvidia,
                Precision::F32,
                (17usize, 89usize, 17usize, 16_384usize),
            ),
            (
                GpuVendor::Amd,
                Precision::F64,
                (17usize, 89usize, 29usize, 16_384usize),
            ),
            (
                GpuVendor::Amd,
                Precision::DoubleDouble,
                (11usize, 29usize, 19usize, 16_384usize),
            ),
            (
                GpuVendor::Nvidia,
                Precision::DoubleDouble,
                (11usize, 29usize, 17usize, 16_384usize),
            ),
            (
                GpuVendor::Intel,
                Precision::F32,
                (17usize, 17usize, 17usize, 16_384usize),
            ),
            (
                GpuVendor::Apple,
                Precision::F64,
                (17usize, 17usize, 17usize, 16_384usize),
            ),
            (
                GpuVendor::Other(0x1234),
                Precision::F32,
                (17usize, 17usize, 17usize, 16_384usize),
            ),
        ];
        for (vendor, precision, expected) in cases {
            let profile = DeviceProfile::generic(Backend::Vulkan, vendor);
            let tuning = PlannerTuning::for_device(profile, precision);
            assert_eq!(
                (
                    tuning.min_rader_direct_prime,
                    tuning.max_rader_direct_prime,
                    tuning.min_rader_fft_prime,
                    tuning.max_rader_fft_prime,
                ),
                expected,
                "vendor={vendor:?}, precision={precision:?}"
            );
        }
    }

    #[test]
    fn device_rader_thread_coalescing_cap_matches_fixed_upstream_scheduler() {
        let constrained = DeviceProfile {
            max_threads_per_block: 128,
            max_workgroup_size: [128, 128, 64],
            ..DeviceProfile::generic(Backend::Vulkan, GpuVendor::Nvidia)
        };
        assert_eq!(
            upstream_direct_rader_thread_cap(constrained, Precision::F32),
            63
        );
        assert_eq!(
            upstream_direct_rader_thread_cap(constrained, Precision::F16StorageF32Compute),
            31
        );
        assert_eq!(
            upstream_direct_rader_thread_cap(constrained, Precision::F64),
            127
        );
        assert_eq!(
            upstream_direct_rader_thread_cap(constrained, Precision::F64ComputeF32Storage),
            127
        );
        // The initialization default stays independent of the scheduler cap.
        assert_eq!(
            PlannerTuning::for_device(constrained, Precision::F32).max_rader_direct_prime,
            89
        );

        let very_small = DeviceProfile {
            max_threads_per_block: 32,
            max_workgroup_size: [32, 32, 32],
            ..constrained
        };
        assert_eq!(
            upstream_direct_rader_thread_cap(very_small, Precision::F32),
            15
        );

        assert!(has_fixed_upstream_gpu_scheduler_profile(
            DeviceProfile::generic(Backend::Vulkan, GpuVendor::Nvidia)
        ));
        assert!(has_fixed_upstream_gpu_scheduler_profile(
            DeviceProfile::generic(Backend::Metal, GpuVendor::Apple)
        ));
        assert!(!has_fixed_upstream_gpu_scheduler_profile(
            DeviceProfile::generic(Backend::Vulkan, GpuVendor::Apple)
        ));
        assert!(!has_fixed_upstream_gpu_scheduler_profile(
            DeviceProfile::generic(Backend::Vulkan, GpuVendor::Other(0x1234))
        ));

        assert_eq!(
            upstream_coalesced_memory_bytes(DeviceProfile::generic(
                Backend::Vulkan,
                GpuVendor::Nvidia
            )),
            32
        );
        assert_eq!(
            upstream_coalesced_memory_bytes_for_precision(
                DeviceProfile::generic(Backend::Vulkan, GpuVendor::Nvidia),
                Precision::F16StorageF32Compute,
            ),
            64
        );
        assert_eq!(
            upstream_coalesced_memory_bytes(DeviceProfile::generic(
                Backend::Vulkan,
                GpuVendor::Intel
            )),
            64
        );
        assert_eq!(
            upstream_coalesced_memory_bytes(DeviceProfile::generic(
                Backend::LevelZero,
                GpuVendor::Intel
            )),
            64
        );
        assert_eq!(
            upstream_coalesced_memory_bytes(DeviceProfile::generic(Backend::Hip, GpuVendor::Amd)),
            32
        );
    }

    #[test]
    fn explicit_tuning_disables_high_level_device_default_resolution() {
        let explicit = PlannerTuning::portable().with_recursive_fft_rader(true);
        let config = FftConfig::new(vec![47]).with_tuning(explicit);
        assert!(!config.use_device_tuning);
        let resolved = config
            .resolve_tuning_for_device(DeviceProfile::generic(Backend::Vulkan, GpuVendor::Intel));
        assert_eq!(resolved.tuning, explicit);

        let automatic = FftConfig::new(vec![47])
            .resolve_tuning_for_device(DeviceProfile::generic(Backend::Vulkan, GpuVendor::Intel));
        assert!(automatic.use_device_tuning);
        assert_eq!(automatic.tuning.max_rader_direct_prime, 17);

        let reenabled = FftConfig::new(vec![47])
            .with_tuning(explicit)
            .with_device_tuning()
            .resolve_tuning_for_device(DeviceProfile::generic(Backend::Vulkan, GpuVendor::Intel));
        assert!(reenabled.use_device_tuning);
        assert_eq!(reenabled.tuning.max_rader_direct_prime, 17);
    }
}
