//! Backend-neutral compute-kernel IR.
//!
//! VkFFT's source generator is layered: Level-2 code describes the complete
//! transform, Level-1 routines implement FFT/reorder/pre/post operations, and
//! Level-0 routines lower math and memory operations to a concrete API. This
//! module starts the Rust equivalent with a deliberately small but executable
//! IR for one-upload Stockham C2C transforms.

use core::f64::consts::{FRAC_1_SQRT_2, TAU};

use crate::complex::Complex64;
use crate::config::{
    Backend, DeviceProfile, Direction, GpuVendor, Precision, SubgroupProfile, TransformKind,
};
use crate::error::{Result, VkFftError};
use crate::planner::{AxisAlgorithm, FftPlan, RadixPlan};
use crate::scheduler::{
    RaderFftTransposeSchedule, RadixRegisterSchedule, StockhamAxisBlockSchedule,
    StockhamTwiddleSource, has_specialized_gpu_scheduler_policy,
    plan_gpu_smooth_stockham_uploads_for_batches, plan_gpu_stockham_twiddle_source,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalarType {
    /// IEEE-754 binary16 storage. Executable kernels never use this as their
    /// arithmetic scalar; it is reserved for typed external/resource bindings.
    F16,
    F32,
    F64,
    /// One double-double scalar stored as two IEEE-754 binary64 words. A complex
    /// compute value therefore occupies 32 bytes. GPU lowering is capability-gated.
    DoubleDouble,
}

impl ScalarType {
    pub const fn bytes(self) -> usize {
        match self {
            Self::F16 => 2,
            Self::F32 => 4,
            Self::F64 => 8,
            Self::DoubleDouble => 16,
        }
    }

    pub const fn complex_bytes(self) -> usize {
        self.bytes() * 2
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::F16 => "f16-storage",
            Self::F32 => "f32",
            Self::F64 => "f64",
            Self::DoubleDouble => "double-double",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BufferRole {
    Input,
    Output,
    LookupTable,
    TwiddleLookupTable,
    Auxiliary,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BufferAccess {
    ReadOnly,
    WriteOnly,
    ReadWrite,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BufferBinding {
    pub set: u32,
    pub binding: u32,
    pub role: BufferRole,
    pub access: BufferAccess,
    pub scalar: ScalarType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WorkgroupSize {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DispatchGeometry {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SharedBuffer {
    A,
    B,
}

impl SharedBuffer {
    pub const fn alternate(self) -> Self {
        match self {
            Self::A => Self::B,
            Self::B => Self::A,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SharedMemoryPlan {
    /// Physical complex elements allocated per shared buffer, including padding.
    pub elements_per_buffer: usize,
    pub buffers: usize,
    pub scalar: ScalarType,
}

impl SharedMemoryPlan {
    pub fn required_bytes(self) -> Result<usize> {
        self.elements_per_buffer
            .checked_mul(self.buffers)
            .and_then(|value| value.checked_mul(self.scalar.complex_bytes()))
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "kernel shared-memory size",
            })
    }
}

/// Upstream-shaped shared-memory stride metadata for the covered single-upload
/// NVIDIA/Vulkan Stockham register paths.
///
/// VkFFT treats one complex value as spanning two shared-memory banks, so the
/// first-stage bank-conflict remap works in `shared_banks / 2` complex-element
/// chunks and inserts one padding element per chunk. `allocated_elements` is the
/// maximum of that padded stride and the read/write-conflict stride. When the
/// padded candidate would exceed the device budget, upstream collapses both
/// strides back to `logical_elements`; this plan records that fallback exactly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StockhamSharedMemoryLayout {
    pub logical_elements: usize,
    pub allocated_elements: usize,
    pub shared_banks: usize,
    pub bank_span_elements: usize,
    pub first_stage_stride: usize,
    pub read_write_stride: usize,
}

impl StockhamSharedMemoryLayout {
    pub const fn first_stage_padding_enabled(self) -> bool {
        self.bank_span_elements > 0 && self.first_stage_stride != self.logical_elements
    }

    pub fn stage_uses_first_stage_padding(
        self,
        sequence_len: usize,
        stage_size: usize,
        radix: usize,
    ) -> bool {
        self.first_stage_padding_enabled()
            && stage_size <= self.bank_span_elements
            && sequence_len > self.bank_span_elements
            && sequence_len.is_power_of_two()
            && stage_size
                .checked_mul(radix)
                .is_some_and(|covered| covered != sequence_len)
    }

    pub fn physical_index(self, logical_index: usize, padded: bool) -> Result<usize> {
        if logical_index >= self.logical_elements {
            return Err(VkFftError::InvalidKernelIr(
                "Stockham shared-memory logical index is out of range",
            ));
        }
        if !padded || !self.first_stage_padding_enabled() {
            return Ok(logical_index);
        }
        let span = self.bank_span_elements;
        let block = logical_index / span;
        let lane = logical_index % span;
        let mapped = block
            .checked_mul(span + 1)
            .and_then(|base| base.checked_add(lane))
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "Stockham shared-memory bank-conflict index",
            })?;
        if mapped >= self.allocated_elements {
            return Err(VkFftError::InvalidKernelIr(
                "Stockham padded shared-memory index exceeds allocation",
            ));
        }
        Ok(mapped)
    }
}

/// Geometry shared by the two scheduler-selected uploads in a fused Four-step FFT.
/// `left_len * right_len == logical_len`; `outer_batch_count` counts independent
/// full-length transforms rather than the child transforms launched by each upload.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FourStepMapping {
    pub logical_len: usize,
    pub left_len: usize,
    pub right_len: usize,
    pub outer_batch_count: usize,
}

impl FourStepMapping {
    pub fn validate(self) -> Result<()> {
        if self.left_len < 2
            || self.right_len < 2
            || self.outer_batch_count == 0
            || self.left_len.checked_mul(self.right_len) != Some(self.logical_len)
        {
            return Err(VkFftError::InvalidKernelIr(
                "Four-step Stockham mapping dimensions are inconsistent",
            ));
        }
        let _ = self.logical_len.checked_mul(self.outer_batch_count).ok_or(
            VkFftError::ArithmeticOverflow {
                operation: "Four-step Stockham mapped element count",
            },
        )?;
        Ok(())
    }
}

/// Two-upload high-axis mapping used by `performConvolution` when upstream disables
/// `reorderFourStep`. The global gather/scatter geometry is identical to
/// [`StockhamIoMapping::FourStepRight`], but the cross-upload twiddle is consumed on
/// the stage-0 read boundary instead of the final store. This is the inverse leftover
/// contract after upload 0 has executed the embedded convolution step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FourStepPreTwiddleMapping {
    pub four_step: FourStepMapping,
    pub direction: Direction,
}

impl FourStepPreTwiddleMapping {
    pub fn validate(self) -> Result<()> {
        self.four_step.validate()
    }
}

/// Geometry for the covered three-upload Four-step schedule. `axis_split` keeps
/// VkFFT's native low-to-high upload-id order `[A, B, C]`, while execution runs
/// the corresponding FFTs in reverse order `C -> B -> A`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ThreeUploadFourStepMapping {
    pub logical_len: usize,
    pub axis_split: [usize; 3],
    pub outer_batch_count: usize,
}

impl ThreeUploadFourStepMapping {
    pub fn validate(self) -> Result<()> {
        let [a, b, c] = self.axis_split;
        let product = a
            .checked_mul(b)
            .and_then(|value| value.checked_mul(c))
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "three-upload Four-step factor product",
            })?;
        if a < 2 || b < 2 || c < 2 || self.outer_batch_count == 0 || product != self.logical_len {
            return Err(VkFftError::InvalidKernelIr(
                "three-upload Four-step Stockham mapping dimensions are inconsistent",
            ));
        }
        let _ = self.logical_len.checked_mul(self.outer_batch_count).ok_or(
            VkFftError::ArithmeticOverflow {
                operation: "three-upload Four-step mapped element count",
            },
        )?;
        Ok(())
    }
}

/// Inverse leftover mapping for three-upload `performConvolution` when upstream runs
/// `reorderFourStep == 0`. Upload 0 has already executed the embedded convolution step,
/// so upload 1 or 2 consumes its cross-upload twiddle on stage-0 reads and stores an
/// untwiddled result toward the next higher upload (or final natural spatial order).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ThreeUploadPreTwiddleMapping {
    pub three_upload: ThreeUploadFourStepMapping,
    pub axis_upload_id: usize,
    pub direction: Direction,
}

impl ThreeUploadPreTwiddleMapping {
    pub fn validate(self) -> Result<()> {
        self.three_upload.validate()?;
        if !matches!(self.axis_upload_id, 1 | 2) {
            return Err(VkFftError::InvalidKernelIr(
                "three-upload pre-twiddle mapping requires upload id 1 or 2",
            ));
        }
        Ok(())
    }
}

/// Primitive-root input geometry for a fused FFT-Rader forward Stockham kernel.
/// The Stockham transform still has `prime - 1` logical values, but stage 0 reads
/// them directly from the original prime-length input in reversed generator order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RaderGeneratorMapping {
    pub prime: usize,
    pub generator: usize,
}

impl RaderGeneratorMapping {
    pub fn validate(self, sequence_len: usize) -> Result<()> {
        if sequence_len < 2
            || sequence_len.checked_add(1) != Some(self.prime)
            || self.generator == 0
            || self.generator >= self.prime
        {
            return Err(VkFftError::InvalidKernelIr(
                "Rader generator Stockham mapping dimensions are inconsistent",
            ));
        }
        let mut residue = 1usize;
        for exponent in 1..=sequence_len {
            residue = Self::mul_mod(residue, self.generator, self.prime);
            if exponent < sequence_len && residue == 1 {
                return Err(VkFftError::InvalidKernelIr(
                    "Rader generator Stockham mapping does not span prime-1 residues",
                ));
            }
        }
        if residue != 1 {
            return Err(VkFftError::InvalidKernelIr(
                "Rader generator Stockham mapping does not close after prime-1 powers",
            ));
        }
        Ok(())
    }

    fn mul_mod(lhs: usize, rhs: usize, modulus: usize) -> usize {
        ((lhs as u128 * rhs as u128) % modulus as u128) as usize
    }

    fn residue_at(self, mut exponent: usize) -> usize {
        let mut value = 1usize;
        let mut base = self.generator;
        while exponent != 0 {
            if exponent & 1 != 0 {
                value = Self::mul_mod(value, base, self.prime);
            }
            exponent >>= 1;
            if exponent != 0 {
                base = Self::mul_mod(base, base, self.prime);
            }
        }
        value
    }

    pub(crate) fn permutation(self, sequence_len: usize) -> Result<Vec<usize>> {
        self.validate(sequence_len)?;
        Ok((0..sequence_len)
            .map(|exponent| self.residue_at(exponent))
            .collect())
    }
}

/// Outer caller geometry composed with FFT-Rader generator-order stage-0 loads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RaderFourStepInputMapping {
    TwoUploadRight(FourStepMapping),
    TwoUploadLeft(FourStepMapping),
    ThreeUpload2(ThreeUploadFourStepMapping),
    ThreeUpload1(ThreeUploadFourStepMapping),
    ThreeUpload0(ThreeUploadFourStepMapping),
}

impl RaderFourStepInputMapping {
    pub(crate) fn from_stockham(mapping: StockhamIoMapping) -> Option<Self> {
        match mapping {
            StockhamIoMapping::FourStepRight(mapping) => Some(Self::TwoUploadRight(mapping)),
            StockhamIoMapping::FourStepLeft(mapping) => Some(Self::TwoUploadLeft(mapping)),
            StockhamIoMapping::FourStepThreeUpload2(mapping) => Some(Self::ThreeUpload2(mapping)),
            StockhamIoMapping::FourStepThreeUpload1(mapping) => Some(Self::ThreeUpload1(mapping)),
            StockhamIoMapping::FourStepThreeUpload0(mapping) => Some(Self::ThreeUpload0(mapping)),
            _ => None,
        }
    }

    fn validate_prime_batches(self, prime: usize, batch_count: usize) -> Result<()> {
        let (component_len, expected_batches) = match self {
            Self::TwoUploadRight(mapping) => {
                mapping.validate()?;
                (
                    mapping.right_len,
                    mapping.outer_batch_count.checked_mul(mapping.left_len),
                )
            }
            Self::TwoUploadLeft(mapping) => {
                mapping.validate()?;
                (
                    mapping.left_len,
                    mapping.outer_batch_count.checked_mul(mapping.right_len),
                )
            }
            Self::ThreeUpload2(mapping) => {
                mapping.validate()?;
                let [a, b, c] = mapping.axis_split;
                (
                    c,
                    mapping
                        .outer_batch_count
                        .checked_mul(a)
                        .and_then(|value| value.checked_mul(b)),
                )
            }
            Self::ThreeUpload1(mapping) => {
                mapping.validate()?;
                let [a, b, c] = mapping.axis_split;
                (
                    b,
                    mapping
                        .outer_batch_count
                        .checked_mul(c)
                        .and_then(|value| value.checked_mul(a)),
                )
            }
            Self::ThreeUpload0(mapping) => {
                mapping.validate()?;
                let [a, b, c] = mapping.axis_split;
                (
                    a,
                    mapping
                        .outer_batch_count
                        .checked_mul(c)
                        .and_then(|value| value.checked_mul(b)),
                )
            }
        };
        let expected_batches = expected_batches.ok_or(VkFftError::ArithmeticOverflow {
            operation: "Rader/Four-step composed batch count",
        })?;
        if component_len != prime || expected_batches != batch_count {
            return Err(VkFftError::InvalidKernelIr(
                "Rader/Four-step composed caller geometry is inconsistent",
            ));
        }
        Ok(())
    }

    pub(crate) const fn reads_external_input(self) -> bool {
        matches!(self, Self::TwoUploadRight(_) | Self::ThreeUpload2(_))
    }

    fn input_index(self, batch: usize, prime_local_index: usize) -> usize {
        match self {
            Self::TwoUploadRight(mapping) => {
                let outer_batch = batch / mapping.left_len;
                let n1 = batch % mapping.left_len;
                outer_batch * mapping.logical_len + n1 + mapping.left_len * prime_local_index
            }
            Self::TwoUploadLeft(mapping) => batch * mapping.left_len + prime_local_index,
            Self::ThreeUpload2(mapping) => {
                let [a, b, _c] = mapping.axis_split;
                let ab = a * b;
                let outer_batch = batch / ab;
                let n12 = batch % ab;
                outer_batch * mapping.logical_len + n12 + ab * prime_local_index
            }
            Self::ThreeUpload1(mapping) => {
                let [_a, b, _c] = mapping.axis_split;
                batch * b + prime_local_index
            }
            Self::ThreeUpload0(mapping) => {
                let [a, _b, _c] = mapping.axis_split;
                batch * a + prime_local_index
            }
        }
    }
}

/// Source layout of a prime-length right child inside a Cooley-Tukey parent.
/// Right-child transform `batch = parent_batch * A + n1` reads prime-local
/// element `n2` from natural parent index `parent_batch * N + n1 + A * n2`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CooleyRightInputMapping {
    pub parent_logical_len: usize,
    pub parent_left_len: usize,
    pub parent_right_len: usize,
    pub parent_batch_count: usize,
}

impl CooleyRightInputMapping {
    pub(crate) fn validate(self, prime: usize, batch_count: usize) -> Result<()> {
        if self.parent_left_len < 2
            || self.parent_right_len < 2
            || self.parent_batch_count == 0
            || self.parent_right_len != prime
            || self.parent_left_len.checked_mul(self.parent_right_len)
                != Some(self.parent_logical_len)
            || self.parent_batch_count.checked_mul(self.parent_left_len) != Some(batch_count)
        {
            return Err(VkFftError::InvalidKernelIr(
                "Cooley-right prime input mapping does not match the parent geometry",
            ));
        }
        Ok(())
    }

    fn input_index(self, batch: usize, prime_local_index: usize) -> usize {
        let parent_batch = batch / self.parent_left_len;
        let n1 = batch % self.parent_left_len;
        parent_batch * self.parent_logical_len + n1 + self.parent_left_len * prime_local_index
    }

    fn input_elements(self) -> Result<usize> {
        self.parent_logical_len
            .checked_mul(self.parent_batch_count)
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "Cooley-right prime input element count",
            })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RaderGeneratorCooleyRightMapping {
    pub rader: RaderGeneratorMapping,
    pub caller: CooleyRightInputMapping,
}

impl RaderGeneratorCooleyRightMapping {
    pub(crate) fn validate(self, sequence_len: usize, batch_count: usize) -> Result<()> {
        self.rader.validate(sequence_len)?;
        self.caller.validate(self.rader.prime, batch_count)
    }

    fn input_index(self, sequence_len: usize, batch: usize, local_index: usize) -> usize {
        let exponent = (sequence_len - local_index) % sequence_len;
        let prime_local_index = self.rader.residue_at(exponent);
        self.caller.input_index(batch, prime_local_index)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RaderGeneratorFourStepMapping {
    pub rader: RaderGeneratorMapping,
    pub caller: RaderFourStepInputMapping,
}

impl RaderGeneratorFourStepMapping {
    pub(crate) fn validate(self, sequence_len: usize, batch_count: usize) -> Result<()> {
        self.rader.validate(sequence_len)?;
        self.caller
            .validate_prime_batches(self.rader.prime, batch_count)
    }

    fn input_index(self, sequence_len: usize, batch: usize, local_index: usize) -> usize {
        let exponent = (sequence_len - local_index) % sequence_len;
        let prime_local_index = self.rader.residue_at(exponent);
        self.caller.input_index(batch, prime_local_index)
    }
}

/// Fuse a Cooley-Tukey parent's twiddle-transpose input boundary and final output
/// scatter into its small left Stockham child. The optional outer Four-step mapping
/// represents the parent's caller-visible `FourStepLeft` boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CooleyLeftStockhamMapping {
    pub parent_logical_len: usize,
    pub parent_left_len: usize,
    pub parent_right_len: usize,
    pub parent_batch_count: usize,
    pub direction: Direction,
    pub outer_four_step: Option<FourStepMapping>,
}

impl CooleyLeftStockhamMapping {
    pub(crate) fn validate(self, sequence_len: usize, batch_count: usize) -> Result<()> {
        if self.parent_left_len < 2
            || self.parent_right_len < 2
            || self.parent_batch_count == 0
            || self.parent_left_len.checked_mul(self.parent_right_len)
                != Some(self.parent_logical_len)
            || sequence_len != self.parent_left_len
            || self.parent_batch_count.checked_mul(self.parent_right_len) != Some(batch_count)
        {
            return Err(VkFftError::InvalidKernelIr(
                "Cooley-left Stockham mapping does not match the parent transform geometry",
            ));
        }
        if let Some(outer) = self.outer_four_step {
            outer.validate()?;
            if outer.left_len != self.parent_logical_len
                || outer.outer_batch_count.checked_mul(outer.right_len)
                    != Some(self.parent_batch_count)
            {
                return Err(VkFftError::InvalidKernelIr(
                    "Cooley-left Stockham mapping does not match the outer Four-step boundary",
                ));
            }
        }
        Ok(())
    }

    fn input_index(self, batch: usize, local_index: usize) -> usize {
        let parent_batch = batch / self.parent_right_len;
        let k2 = batch % self.parent_right_len;
        (parent_batch * self.parent_left_len + local_index) * self.parent_right_len + k2
    }

    fn map_output(self, batch: usize, local_index: usize) -> usize {
        let parent_batch = batch / self.parent_right_len;
        let k2 = batch % self.parent_right_len;
        let parent_local_index = k2 + self.parent_right_len * local_index;
        match self.outer_four_step {
            None => parent_batch * self.parent_logical_len + parent_local_index,
            Some(outer) => {
                let outer_batch = parent_batch / outer.right_len;
                let outer_k2 = parent_batch % outer.right_len;
                outer_batch * outer.logical_len + outer_k2 + outer.right_len * parent_local_index
            }
        }
    }
}

/// Optional pointwise transform applied to each logical Stockham input element before
/// stage-0 twiddles/butterflies. The first fused use is FFT-Rader's kernel-spectrum
/// multiply immediately before the inverse convolution FFT.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum StockhamInputModifier {
    #[default]
    None,
    MultiplyLookupTable,
}

/// Natural-order prime-output reconstruction fused into the final inverse
/// convolution Stockham store of an FFT-Rader pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RaderScatterMapping {
    pub prime: usize,
    pub generator: usize,
    /// Apply the outer transform's normalized-inverse `1 / prime` scale after
    /// x0/DC reconstruction. The internal `(prime - 1)` inverse FFT has already
    /// applied its own convolution normalization.
    pub normalize_prime: bool,
    /// Optional caller layout for the original prime input used by x0/DC
    /// reconstruction. `None` keeps the ordinary contiguous prime-batch layout.
    pub auxiliary_input: Option<CooleyRightInputMapping>,
}

impl RaderScatterMapping {
    pub fn validate(self, sequence_len: usize) -> Result<()> {
        RaderGeneratorMapping {
            prime: self.prime,
            generator: self.generator,
        }
        .validate(sequence_len)
    }

    pub(crate) fn permutation(self, sequence_len: usize) -> Result<Vec<usize>> {
        RaderGeneratorMapping {
            prime: self.prime,
            generator: self.generator,
        }
        .permutation(sequence_len)
    }

    fn validate_batches(self, sequence_len: usize, batch_count: usize) -> Result<()> {
        self.validate(sequence_len)?;
        if let Some(mapping) = self.auxiliary_input {
            mapping.validate(self.prime, batch_count)?;
        }
        Ok(())
    }

    fn auxiliary_input_index(self, batch: usize, prime_local_index: usize) -> usize {
        self.auxiliary_input
            .map_or(batch * self.prime + prime_local_index, |mapping| {
                mapping.input_index(batch, prime_local_index)
            })
    }
}

/// Final-store mapping for the even-length C2R half-size decomposition. Every
/// inverse half-size result `z[n]` owns exactly the two real samples
/// `x[2n] = re(z[n])`, `x[2n+1] = im(z[n])`, so this fusion is invocation-local.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RealEvenUnpackMapping {
    pub full_len: usize,
}

impl RealEvenUnpackMapping {
    pub(crate) fn validate(self, sequence_len: usize) -> Result<()> {
        if self.full_len < 2
            || !self.full_len.is_multiple_of(2)
            || sequence_len.checked_mul(2) != Some(self.full_len)
        {
            return Err(VkFftError::InvalidKernelIr(
                "even-real unpack mapping does not match the half-size Stockham length",
            ));
        }
        Ok(())
    }
}

/// Final-store mapping for the even-length R2C half-size decomposition. The
/// half-size Stockham result is first staged in a per-transform auxiliary buffer;
/// after a workgroup-visible storage barrier the same dispatch reconstructs the
/// compact Hermitian spectrum from `Z[k]` and `conj(Z[M-k])`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RealEvenPostprocessMapping {
    pub full_len: usize,
}

impl RealEvenPostprocessMapping {
    pub(crate) fn validate(self, sequence_len: usize) -> Result<()> {
        if self.full_len < 2
            || !self.full_len.is_multiple_of(2)
            || sequence_len.checked_mul(2) != Some(self.full_len)
        {
            return Err(VkFftError::InvalidKernelIr(
                "even-real postprocess mapping does not match the half-size Stockham length",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum StockhamOutputModifier {
    #[default]
    None,
    RaderScatter(RaderScatterMapping),
    RealEvenPostprocess(RealEvenPostprocessMapping),
    RealEvenUnpack(RealEvenUnpackMapping),
}

impl StockhamOutputModifier {
    fn validate_kernel(self, sequence_len: usize, batch_count: usize) -> Result<()> {
        match self {
            Self::None => Ok(()),
            Self::RaderScatter(mapping) => mapping.validate_batches(sequence_len, batch_count),
            Self::RealEvenPostprocess(mapping) => mapping.validate(sequence_len),
            Self::RealEvenUnpack(mapping) => mapping.validate(sequence_len),
        }
    }

    fn output_elements(self, sequence_len: usize, batch_count: usize) -> Result<usize> {
        let stride = match self {
            Self::None => sequence_len,
            Self::RaderScatter(mapping) => mapping.prime,
            Self::RealEvenPostprocess(_) => sequence_len + 1,
            Self::RealEvenUnpack(mapping) => mapping.full_len,
        };
        stride
            .checked_mul(batch_count)
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "Stockham mapped output element count",
            })
    }

    const fn extra_binding_count(self) -> usize {
        match self {
            Self::None | Self::RealEvenUnpack(_) => 0,
            Self::RaderScatter(_) | Self::RealEvenPostprocess(_) => 1,
        }
    }
}

/// Stage-0 input geometry for the even-length R2C half-size decomposition. One
/// logical N/2 complex input `z[n]` is packed directly from real external slots
/// `x[2n] + i*x[2n+1]`, so the standalone PackEvenOdd pass can disappear.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RealEvenPackMapping {
    pub full_len: usize,
}

impl RealEvenPackMapping {
    pub(crate) fn validate(self, sequence_len: usize) -> Result<()> {
        if self.full_len < 2
            || !self.full_len.is_multiple_of(2)
            || sequence_len.checked_mul(2) != Some(self.full_len)
        {
            return Err(VkFftError::InvalidKernelIr(
                "even-real pack mapping does not match the half-size Stockham length",
            ));
        }
        Ok(())
    }
}

/// Stage-0 input geometry for the even-length C2R half-size decomposition. The
/// compact Hermitian input has N/2+1 values; each logical inverse-FFT slot reconstructs
/// the packed spectrum value `Z[k]` using `X[k]`, `conj(X[N/2-k])`, and `W_N^-k`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RealEvenInversePreprocessMapping {
    pub full_len: usize,
    pub normalize: bool,
}

impl RealEvenInversePreprocessMapping {
    pub(crate) fn validate(self, sequence_len: usize) -> Result<()> {
        if self.full_len < 2
            || !self.full_len.is_multiple_of(2)
            || sequence_len.checked_mul(2) != Some(self.full_len)
        {
            return Err(VkFftError::InvalidKernelIr(
                "even-real inverse preprocess mapping does not match the half-size Stockham length",
            ));
        }
        Ok(())
    }

    pub(crate) const fn compact_len(self) -> usize {
        self.full_len / 2 + 1
    }
}

/// External storage mapping for a Stockham upload.
///
/// The two non-contiguous variants are the exact two-upload Cooley-Tukey layouts
/// used by the covered NVIDIA/Vulkan Four-step path. `FourStepRight` gathers the
/// first executed (higher upload-id) B-point FFTs directly from natural input and
/// fuses the Four-step twiddle plus transpose into their stores. `FourStepLeft`
/// consumes those contiguous A-point batches and scatters them directly to natural
/// frequency order. Internal Stockham stage indexing is unchanged.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum StockhamIoMapping {
    #[default]
    Contiguous,
    /// Fuse FFT-Rader's `GatherReverse` into stage-0 global loads. Logical slot `s`
    /// reads `g^((N-s) mod N) mod p` from a prime-length external input, where
    /// `N = p - 1`; Stockham output remains contiguous convolution order.
    RaderGeneratorReverse(RaderGeneratorMapping),
    /// Compose FFT-Rader generator-order stage-0 loads with an outer Four-step
    /// caller mapping. The convolution transform remains `prime - 1` points, but
    /// each logical slot first resolves its primitive-root residue and then reads
    /// that prime-local position from the full Four-step input.
    RaderGeneratorFourStep(RaderGeneratorFourStepMapping),
    /// Compose FFT-Rader generator-order loads with the natural input layout of a
    /// prime-length right child inside a Cooley-Tukey parent.
    RaderGeneratorCooleyRight(RaderGeneratorCooleyRightMapping),
    /// Consume a Cooley parent's right-child output directly, apply its twiddle
    /// during stage-0 loads, then store the small left FFT directly through the
    /// parent's natural/Four-step output mapping.
    CooleyLeft(CooleyLeftStockhamMapping),
    /// Fuse R2C `PackEvenOdd` into stage-0 global loads.
    RealEvenPack(RealEvenPackMapping),
    /// Fuse C2R `PreprocessEvenHalf` into stage-0 global loads.
    RealEvenInversePreprocess(RealEvenInversePreprocessMapping),
    FourStepRight(FourStepMapping),
    FourStepRightPreTwiddle(FourStepPreTwiddleMapping),
    FourStepLeft(FourStepMapping),
    /// Three-upload `axis_upload_id == 2`: gather C-point transforms from natural
    /// `n = n12 + A*B*n3`, apply `W_N^(n12*k3)`, and transpose for the B upload.
    FourStepThreeUpload2(ThreeUploadFourStepMapping),
    /// Three-upload `axis_upload_id == 1`: consume contiguous B-point transforms,
    /// apply `W_(A*B)^(n1*k2)`, and transpose for the final A upload.
    FourStepThreeUpload1(ThreeUploadFourStepMapping),
    /// Three-upload inverse leftover for `performConvolution`: upload 1 or 2 keeps
    /// its physical FFT geometry while moving the Four-step twiddle to stage-0 reads.
    FourStepThreeUploadPreTwiddle(ThreeUploadPreTwiddleMapping),
    /// Three-upload `axis_upload_id == 0`: consume contiguous A-point transforms
    /// and scatter directly to natural frequency order.
    FourStepThreeUpload0(ThreeUploadFourStepMapping),
}

impl StockhamIoMapping {
    pub(crate) fn validate_kernel(self, sequence_len: usize, batch_count: usize) -> Result<()> {
        match self {
            Self::Contiguous => Ok(()),
            Self::RaderGeneratorReverse(mapping) => mapping.validate(sequence_len),
            Self::RaderGeneratorFourStep(mapping) => mapping.validate(sequence_len, batch_count),
            Self::RaderGeneratorCooleyRight(mapping) => mapping.validate(sequence_len, batch_count),
            Self::CooleyLeft(mapping) => mapping.validate(sequence_len, batch_count),
            Self::RealEvenPack(mapping) => mapping.validate(sequence_len),
            Self::RealEvenInversePreprocess(mapping) => mapping.validate(sequence_len),
            Self::FourStepRight(mapping) => {
                mapping.validate()?;
                let expected_batches = mapping
                    .outer_batch_count
                    .checked_mul(mapping.left_len)
                    .ok_or(VkFftError::ArithmeticOverflow {
                        operation: "Four-step right-upload batch count",
                    })?;
                if sequence_len != mapping.right_len || batch_count != expected_batches {
                    return Err(VkFftError::InvalidKernelIr(
                        "Four-step right-upload Stockham geometry is inconsistent",
                    ));
                }
                Ok(())
            }
            Self::FourStepRightPreTwiddle(mapping) => {
                mapping.validate()?;
                let mapping = mapping.four_step;
                let expected_batches = mapping
                    .outer_batch_count
                    .checked_mul(mapping.left_len)
                    .ok_or(VkFftError::ArithmeticOverflow {
                        operation: "Four-step pre-twiddle right-upload batch count",
                    })?;
                if sequence_len != mapping.right_len || batch_count != expected_batches {
                    return Err(VkFftError::InvalidKernelIr(
                        "Four-step pre-twiddle right-upload Stockham geometry is inconsistent",
                    ));
                }
                Ok(())
            }
            Self::FourStepLeft(mapping) => {
                mapping.validate()?;
                let expected_batches = mapping
                    .outer_batch_count
                    .checked_mul(mapping.right_len)
                    .ok_or(VkFftError::ArithmeticOverflow {
                        operation: "Four-step left-upload batch count",
                    })?;
                if sequence_len != mapping.left_len || batch_count != expected_batches {
                    return Err(VkFftError::InvalidKernelIr(
                        "Four-step left-upload Stockham geometry is inconsistent",
                    ));
                }
                Ok(())
            }
            Self::FourStepThreeUpload2(mapping) => {
                mapping.validate()?;
                let [a, b, c] = mapping.axis_split;
                let expected_batches = mapping
                    .outer_batch_count
                    .checked_mul(a)
                    .and_then(|value| value.checked_mul(b))
                    .ok_or(VkFftError::ArithmeticOverflow {
                        operation: "three-upload Four-step upload-2 batch count",
                    })?;
                if sequence_len != c || batch_count != expected_batches {
                    return Err(VkFftError::InvalidKernelIr(
                        "three-upload Four-step upload-2 geometry is inconsistent",
                    ));
                }
                Ok(())
            }
            Self::FourStepThreeUpload1(mapping) => {
                mapping.validate()?;
                let [a, b, c] = mapping.axis_split;
                let expected_batches = mapping
                    .outer_batch_count
                    .checked_mul(c)
                    .and_then(|value| value.checked_mul(a))
                    .ok_or(VkFftError::ArithmeticOverflow {
                        operation: "three-upload Four-step upload-1 batch count",
                    })?;
                if sequence_len != b || batch_count != expected_batches {
                    return Err(VkFftError::InvalidKernelIr(
                        "three-upload Four-step upload-1 geometry is inconsistent",
                    ));
                }
                Ok(())
            }
            Self::FourStepThreeUploadPreTwiddle(mapping) => {
                mapping.validate()?;
                let [a, b, c] = mapping.three_upload.axis_split;
                let expected_batches = match mapping.axis_upload_id {
                    1 => mapping
                        .three_upload
                        .outer_batch_count
                        .checked_mul(c)
                        .and_then(|value| value.checked_mul(a))
                        .ok_or(VkFftError::ArithmeticOverflow {
                            operation: "three-upload pre-twiddle upload-1 batch count",
                        })?,
                    2 => mapping
                        .three_upload
                        .outer_batch_count
                        .checked_mul(a)
                        .and_then(|value| value.checked_mul(b))
                        .ok_or(VkFftError::ArithmeticOverflow {
                            operation: "three-upload pre-twiddle upload-2 batch count",
                        })?,
                    _ => unreachable!("validated pre-twiddle upload id"),
                };
                let expected_len = if mapping.axis_upload_id == 1 { b } else { c };
                if sequence_len != expected_len || batch_count != expected_batches {
                    return Err(VkFftError::InvalidKernelIr(
                        "three-upload pre-twiddle Stockham geometry is inconsistent",
                    ));
                }
                Ok(())
            }
            Self::FourStepThreeUpload0(mapping) => {
                mapping.validate()?;
                let [a, b, c] = mapping.axis_split;
                let expected_batches = mapping
                    .outer_batch_count
                    .checked_mul(c)
                    .and_then(|value| value.checked_mul(b))
                    .ok_or(VkFftError::ArithmeticOverflow {
                        operation: "three-upload Four-step upload-0 batch count",
                    })?;
                if sequence_len != a || batch_count != expected_batches {
                    return Err(VkFftError::InvalidKernelIr(
                        "three-upload Four-step upload-0 geometry is inconsistent",
                    ));
                }
                Ok(())
            }
        }
    }

    fn input_elements(self, sequence_len: usize, batch_count: usize) -> Result<usize> {
        let stride = match self {
            Self::RaderGeneratorReverse(mapping) => mapping.prime,
            Self::RaderGeneratorFourStep(mapping) => mapping.rader.prime,
            Self::RaderGeneratorCooleyRight(mapping) => return mapping.caller.input_elements(),
            Self::RealEvenPack(mapping) => mapping.full_len,
            Self::RealEvenInversePreprocess(mapping) => mapping.compact_len(),
            _ => sequence_len,
        };
        stride
            .checked_mul(batch_count)
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "Stockham mapped input element count",
            })
    }

    fn supports_grouped_register_input(self) -> bool {
        matches!(
            self,
            Self::Contiguous
                | Self::RaderGeneratorReverse(_)
                | Self::RaderGeneratorFourStep(_)
                | Self::RaderGeneratorCooleyRight(_)
                | Self::CooleyLeft(_)
                | Self::RealEvenPack(_)
                | Self::RealEvenInversePreprocess(_)
                | Self::FourStepRight(_)
                | Self::FourStepRightPreTwiddle(_)
                | Self::FourStepLeft(_)
                | Self::FourStepThreeUpload2(_)
                | Self::FourStepThreeUpload1(_)
                | Self::FourStepThreeUploadPreTwiddle(_)
                | Self::FourStepThreeUpload0(_)
        )
    }

    pub(crate) fn input_index(
        self,
        sequence_len: usize,
        batch: usize,
        local_index: usize,
    ) -> usize {
        match self {
            Self::Contiguous
            | Self::FourStepLeft(_)
            | Self::FourStepThreeUpload1(_)
            | Self::FourStepThreeUpload0(_) => batch * sequence_len + local_index,
            Self::RealEvenPack(_) | Self::RealEvenInversePreprocess(_) => {
                unreachable!("even-real Stockham mappings load multiple external values")
            }
            Self::RaderGeneratorReverse(mapping) => {
                let exponent = (sequence_len - local_index) % sequence_len;
                batch * mapping.prime + mapping.residue_at(exponent)
            }
            Self::RaderGeneratorFourStep(mapping) => {
                mapping.input_index(sequence_len, batch, local_index)
            }
            Self::RaderGeneratorCooleyRight(mapping) => {
                mapping.input_index(sequence_len, batch, local_index)
            }
            Self::CooleyLeft(mapping) => mapping.input_index(batch, local_index),
            Self::FourStepRight(mapping) => {
                let outer_batch = batch / mapping.left_len;
                let n1 = batch % mapping.left_len;
                outer_batch * mapping.logical_len + n1 + mapping.left_len * local_index
            }
            Self::FourStepRightPreTwiddle(mapping) => {
                let mapping = mapping.four_step;
                let outer_batch = batch / mapping.left_len;
                let n1 = batch % mapping.left_len;
                outer_batch * mapping.logical_len + n1 + mapping.left_len * local_index
            }
            Self::FourStepThreeUploadPreTwiddle(mapping) => {
                let geometry = mapping.three_upload;
                let [a, b, _c] = geometry.axis_split;
                match mapping.axis_upload_id {
                    1 => {
                        let group = batch / a;
                        let n1 = batch % a;
                        group * a * b + n1 + a * local_index
                    }
                    2 => {
                        let ab = a * b;
                        let outer_batch = batch / ab;
                        let n12 = batch % ab;
                        outer_batch * geometry.logical_len + n12 + ab * local_index
                    }
                    _ => unreachable!("validated pre-twiddle upload id"),
                }
            }
            Self::FourStepThreeUpload2(mapping) => {
                let [a, b, _c] = mapping.axis_split;
                let ab = a * b;
                let outer_batch = batch / ab;
                let n12 = batch % ab;
                outer_batch * mapping.logical_len + n12 + ab * local_index
            }
        }
    }

    fn input_value(
        self,
        sequence_len: usize,
        batch: usize,
        local_index: usize,
        input: &[Complex64],
    ) -> Complex64 {
        match self {
            Self::FourStepRightPreTwiddle(mapping) => {
                let geometry = mapping.four_step;
                let source = input[self.input_index(sequence_len, batch, local_index)];
                let n1 = batch % geometry.left_len;
                let angle = mapping.direction.exponent_sign() * TAU * (n1 * local_index) as f64
                    / geometry.logical_len as f64;
                source * Complex64::exp_i(angle)
            }
            Self::FourStepThreeUploadPreTwiddle(mapping) => {
                let geometry = mapping.three_upload;
                let [a, b, _c] = geometry.axis_split;
                let source = input[self.input_index(sequence_len, batch, local_index)];
                let (exponent, denominator) = match mapping.axis_upload_id {
                    1 => ((batch % a) * local_index, a * b),
                    2 => ((batch % (a * b)) * local_index, geometry.logical_len),
                    _ => unreachable!("validated pre-twiddle upload id"),
                };
                let angle =
                    mapping.direction.exponent_sign() * TAU * exponent as f64 / denominator as f64;
                source * Complex64::exp_i(angle)
            }
            Self::CooleyLeft(mapping) => {
                let source = input[mapping.input_index(batch, local_index)];
                let k2 = batch % mapping.parent_right_len;
                let angle = mapping.direction.exponent_sign() * TAU * (local_index * k2) as f64
                    / mapping.parent_logical_len as f64;
                source * Complex64::exp_i(angle)
            }
            Self::RealEvenPack(mapping) => {
                let base = batch * mapping.full_len;
                Complex64::new(
                    input[base + 2 * local_index].re,
                    input[base + 2 * local_index + 1].re,
                )
            }
            Self::RealEvenInversePreprocess(mapping) => {
                let base = batch * mapping.compact_len();
                let x = input[base + local_index];
                let mirrored = input[base + sequence_len - local_index].conj();
                let w_conj = Complex64::exp_i(TAU * local_index as f64 / mapping.full_len as f64);
                let rotated = w_conj * (x - mirrored);
                let i_rotated = Complex64::new(-rotated.im, rotated.re);
                let scale = if mapping.normalize { 0.5 } else { 1.0 };
                (x + mirrored + i_rotated).scale(scale)
            }
            _ => input[self.input_index(sequence_len, batch, local_index)],
        }
    }

    pub(crate) fn map_output(
        self,
        sequence_len: usize,
        batch: usize,
        local_index: usize,
        value: Complex64,
        sign: f64,
    ) -> (usize, Complex64) {
        match self {
            Self::Contiguous
            | Self::RaderGeneratorReverse(_)
            | Self::RaderGeneratorFourStep(_)
            | Self::RaderGeneratorCooleyRight(_)
            | Self::RealEvenPack(_)
            | Self::RealEvenInversePreprocess(_) => (batch * sequence_len + local_index, value),
            Self::CooleyLeft(mapping) => (mapping.map_output(batch, local_index), value),
            Self::FourStepRight(mapping) => {
                let outer_batch = batch / mapping.left_len;
                let n1 = batch % mapping.left_len;
                let output_index =
                    (outer_batch * mapping.right_len + local_index) * mapping.left_len + n1;
                let angle = sign * TAU * (n1 * local_index) as f64 / mapping.logical_len as f64;
                (output_index, value * Complex64::exp_i(angle))
            }
            Self::FourStepRightPreTwiddle(mapping) => {
                let mapping = mapping.four_step;
                let outer_batch = batch / mapping.left_len;
                let n1 = batch % mapping.left_len;
                let output_index =
                    (outer_batch * mapping.right_len + local_index) * mapping.left_len + n1;
                (output_index, value)
            }
            Self::FourStepThreeUploadPreTwiddle(_) => {
                let output_index = self.input_index(sequence_len, batch, local_index);
                (output_index, value)
            }
            Self::FourStepLeft(mapping) => {
                let outer_batch = batch / mapping.right_len;
                let k2 = batch % mapping.right_len;
                let output_index =
                    outer_batch * mapping.logical_len + k2 + mapping.right_len * local_index;
                (output_index, value)
            }
            Self::FourStepThreeUpload2(mapping) => {
                let [a, b, c] = mapping.axis_split;
                let ab = a * b;
                let outer_batch = batch / ab;
                let n12 = batch % ab;
                let n1 = n12 % a;
                let n2 = n12 / a;
                let k3 = local_index;
                let output_index = (((outer_batch * c + k3) * a + n1) * b) + n2;
                let angle = sign * TAU * (n12 * k3) as f64 / mapping.logical_len as f64;
                (output_index, value * Complex64::exp_i(angle))
            }
            Self::FourStepThreeUpload1(mapping) => {
                let [a, b, _c] = mapping.axis_split;
                let group = batch / a;
                let n1 = batch % a;
                let k2 = local_index;
                let output_index = (group * b + k2) * a + n1;
                let angle = sign * TAU * (n1 * k2) as f64 / (a * b) as f64;
                (output_index, value * Complex64::exp_i(angle))
            }
            Self::FourStepThreeUpload0(mapping) => {
                let [_a, b, c] = mapping.axis_split;
                let group = batch / b;
                let k2 = batch % b;
                let outer_batch = group / c;
                let k3 = group % c;
                let k1 = local_index;
                let output_index = outer_batch * mapping.logical_len + k3 + c * k2 + c * b * k1;
                (output_index, value)
            }
        }
    }
}

/// Physical grouping of independent contiguous Stockham transforms inside one
/// workgroup. Upstream FFT-Rader uses this to execute multiple Rader containers in
/// one local group. The arithmetic remains per-transform: `threads_per_transform`
/// consecutive local invocations own one transform, while shared-memory stripes are
/// disjoint between transforms.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StockhamWorkgroupAxisLayout {
    /// Existing flattened layout: transforms are consecutive stripes in local X.
    LinearX,
    /// Upstream axisBlock before the bank-conflict swap: X owns FFT threads, Y owns batches.
    ThreadsXTransformsY,
    /// Upstream `axisSwapped=1`: X owns batches, Y owns FFT threads.
    TransformsXThreadsY,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StockhamWorkgroupGrouping {
    pub transforms_per_workgroup: usize,
    pub threads_per_transform: usize,
    pub axis_layout: StockhamWorkgroupAxisLayout,
}

impl StockhamWorkgroupGrouping {
    pub const fn single(threads_per_transform: usize) -> Self {
        Self {
            transforms_per_workgroup: 1,
            threads_per_transform,
            axis_layout: StockhamWorkgroupAxisLayout::LinearX,
        }
    }

    pub const fn is_grouped(self) -> bool {
        self.transforms_per_workgroup > 1
    }

    fn validate(
        self,
        batch_count: usize,
        workgroup_size: WorkgroupSize,
        dispatch: DispatchGeometry,
    ) -> Result<()> {
        if self.transforms_per_workgroup == 0 || self.threads_per_transform == 0 {
            return Err(VkFftError::InvalidKernelIr(
                "Stockham workgroup grouping must contain non-zero transforms and threads",
            ));
        }
        let expected_threads = self
            .transforms_per_workgroup
            .checked_mul(self.threads_per_transform)
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "Stockham grouped workgroup thread count",
            })?;
        let (expected_x, expected_y) = match self.axis_layout {
            StockhamWorkgroupAxisLayout::LinearX => (expected_threads, 1),
            StockhamWorkgroupAxisLayout::ThreadsXTransformsY => {
                (self.threads_per_transform, self.transforms_per_workgroup)
            }
            StockhamWorkgroupAxisLayout::TransformsXThreadsY => {
                (self.transforms_per_workgroup, self.threads_per_transform)
            }
        };
        let expected_dispatch = batch_count.div_ceil(self.transforms_per_workgroup);
        if workgroup_size.x as usize != expected_x
            || workgroup_size.y as usize != expected_y
            || workgroup_size.z != 1
            || dispatch.x as usize != expected_dispatch
            || dispatch.y != 1
            || dispatch.z != 1
        {
            return Err(VkFftError::InvalidKernelIr(
                "Stockham grouped workgroup geometry is inconsistent",
            ));
        }
        Ok(())
    }
}

/// Physical shared-memory exchange selected for the executable Stockham path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StockhamExecutionLayout {
    /// Portable correctness fallback: every stage alternates between two full buffers.
    SharedPingPong,
    /// Upstream-shaped register path for `registerBoost == 1`: stage 0 loads global
    /// data directly into registers, intermediate stages reuse one shared buffer,
    /// and the final stage stores registers directly to global memory.
    RegisterSingleShared,
    /// Upstream-shaped boosted register path: one physical invocation owns
    /// `registerBoost * registers_per_thread` values, inter-stage shuffles reuse
    /// only `N / registerBoost` shared complex values, and the extracted final
    /// boost radix stays register-resident.
    RegisterBoostSingleShared,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StockhamStage {
    pub index: usize,
    pub radix: usize,
    /// Product of all preceding radices. This matches VkFFT's `stageSize`.
    pub stage_size: usize,
    pub butterflies: usize,
    pub input: SharedBuffer,
    pub output: SharedBuffer,
}

/// Executable register-staged view of one scheduler-selected Stockham radix.
///
/// A virtual thread owns `registers_per_thread` complex values for this stage,
/// grouped into `butterflies_per_virtual_thread` independent radix butterflies.
/// The Vulkan lowering can either use the portable two-buffer exchange or the
/// upstream-shaped single-shared-buffer exchange selected by `execution_layout`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RegisterStockhamStage {
    pub index: usize,
    pub radix: usize,
    pub stage_size: usize,
    pub butterflies: usize,
    pub register_boost: usize,
    pub registers_per_thread: usize,
    pub logical_storage_per_thread: usize,
    pub butterflies_per_virtual_thread: usize,
    pub virtual_thread_count: usize,
    pub is_register_boost_stage: bool,
    pub input: SharedBuffer,
    pub output: SharedBuffer,
}

/// Whether an adjacent register-scheduled Stockham boundary can remain inside
/// the same physical invocation instead of round-tripping through shared memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegisterStageBoundaryResidency {
    /// Every target register already owns the exact source value in the same slot.
    RegisterResidentIdentity,
    /// Every target register is sourced from another register owned by the same
    /// invocation; `target_to_source_registers` records the exact permutation.
    RegisterResidentPermutation,
    /// At least one target value belongs to a different physical invocation, so
    /// the boundary still requires the shared-memory exchange and synchronization.
    SharedExchangeRequired,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RegisterStageBoundary {
    pub source_stage: usize,
    pub target_stage: usize,
    pub residency: RegisterStageBoundaryResidency,
    /// For resident boundaries, maps each target-stage register slot to the source
    /// stage register slot holding the same logical Stockham value. `None` means a
    /// shared exchange is required.
    pub target_to_source_registers: Option<Vec<usize>>,
}

/// Source location for a subgroup-local cross-stage register transfer. The source
/// lane is relative to the subgroup containing the target invocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RegisterSubgroupSource {
    pub lane: usize,
    pub register: usize,
}

/// Lane-numbering assumption used by a conditional subgroup-boundary proof. Vulkan
/// subgroup lane numbering is not generally defined by local-invocation numbering,
/// so codegen must not consume this model unless a separate backend guarantee proves
/// that mapping for the dispatched pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegisterSubgroupLaneModel {
    ContiguousLocalInvocation,
    /// Physical invocation ownership follows the stage-local `raderTranspose`
    /// mapping rather than contiguous transform stripes.
    RaderTransposePhysicalInvocation,
}

/// Conditional proof that one adjacent register-stage boundary can be satisfied
/// inside a fixed-size lane group under `lane_model`. The Vulkan backend consumes
/// this only after capability checks establish a one-full-subgroup launch. Grouped
/// Rader kernels additionally expand the per-transform proof across contiguous
/// container lane stripes and reject the proof if any source would cross a stripe.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RegisterSubgroupBoundary {
    pub source_stage: usize,
    pub target_stage: usize,
    pub subgroup_size: usize,
    /// Number of subgroup lanes represented by the reusable ownership proof.
    /// Single-transform kernels may use fewer lanes than `subgroup_size` and pad the
    /// rest; grouped kernels use one complete subgroup pattern that may be repeated
    /// across multiple full subgroups in the same workgroup.
    pub active_lanes: usize,
    /// Number of independent Stockham transforms represented by this subgroup proof.
    pub transforms_per_subgroup: usize,
    /// Contiguous local-invocation lanes owned by each transform in the subgroup.
    pub lanes_per_transform: usize,
    pub lane_model: RegisterSubgroupLaneModel,
    /// Indexed as `[target_register][target_subgroup_lane]`.
    pub target_to_source: Vec<Vec<RegisterSubgroupSource>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KernelOperation {
    LoadGlobalToShared {
        target: SharedBuffer,
    },
    StockhamStage(StockhamStage),
    StoreSharedToGlobal {
        source: SharedBuffer,
        normalize: bool,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KernelIr {
    pub name: String,
    pub scalar: ScalarType,
    pub direction: Direction,
    pub sequence_len: usize,
    pub batch_count: usize,
    pub workgroup_grouping: StockhamWorkgroupGrouping,
    pub workgroup_size: WorkgroupSize,
    pub dispatch: DispatchGeometry,
    pub bindings: Vec<BufferBinding>,
    pub shared_memory: SharedMemoryPlan,
    pub stockham_shared_layout: Option<StockhamSharedMemoryLayout>,
    pub io_mapping: StockhamIoMapping,
    pub input_modifier: StockhamInputModifier,
    pub output_modifier: StockhamOutputModifier,
    pub twiddle_source: StockhamTwiddleSource,
    pub execution_layout: StockhamExecutionLayout,
    /// Device subgroup capabilities captured when this kernel was planned. Backend
    /// lowering remains conservative when this profile is unavailable.
    pub subgroup: SubgroupProfile,
    /// Upstream-derived register/radix scheduler metadata. On the covered NVIDIA
    /// Vulkan power-of-two path this now drives register-staged Stockham arithmetic;
    /// `operations` remains the portable shared-ping-pong fallback.
    pub scheduler_hint: Option<RadixRegisterSchedule>,
    /// Optional stage-local lane transpose used by the executable FFT-Rader
    /// multi-container slice. The logical FFT remains contiguous; only physical
    /// invocation/container ownership and the intermediate shared layout change.
    pub rader_transpose: Option<RaderFftTransposeSchedule>,
    pub operations: Vec<KernelOperation>,
}

impl KernelIr {
    /// Build one workgroup per batched 1D C2C Stockham transform. The portable
    /// fallback uses two full shared-memory ping-pong buffers. Covered NVIDIA/Vulkan
    /// power-of-two schedules with `registerBoost == 1` instead use one reusable
    /// shared buffer between register-resident stages, matching the upstream memory
    /// shape closely enough to execute transforms that do not fit the old fallback.
    pub fn stockham_1d(
        plan: &FftPlan,
        direction: Direction,
        device: DeviceProfile,
    ) -> Result<Self> {
        if plan.config.dimensions.len() != 1 {
            return Err(VkFftError::UnsupportedKernelPath(
                "initial Stockham kernel IR supports one-dimensional plans only",
            ));
        }
        if plan.config.transform != TransformKind::ComplexToComplex {
            return Err(VkFftError::UnsupportedKernelPath(
                "initial Stockham kernel IR supports C2C transforms only",
            ));
        }

        let scalar = match plan.config.precision {
            Precision::F32 => ScalarType::F32,
            Precision::F64 if device.supports_f64 => ScalarType::F64,
            other => {
                return Err(VkFftError::UnsupportedPrecision {
                    backend: "kernel IR baseline",
                    precision: precision_name(other),
                });
            }
        };

        let axis = plan
            .axes
            .first()
            .ok_or(VkFftError::InvalidKernelIr("missing axis plan"))?;
        let AxisAlgorithm::Stockham { radix } = &axis.algorithm else {
            return Err(VkFftError::UnsupportedKernelPath(
                "Rader and Bluestein GPU kernels are not lowered yet",
            ));
        };

        let sequence_len = axis.effective_fft_len;
        if sequence_len > u32::MAX as usize {
            return Err(VkFftError::ValueOutOfRange {
                field: "FFT sequence length",
            });
        }
        if plan.config.batch_count > u32::MAX as usize {
            return Err(VkFftError::ValueOutOfRange {
                field: "FFT batch count",
            });
        }
        if device.max_threads_per_block == 0 {
            return Err(VkFftError::ResourceLimitExceeded {
                resource: "threads per workgroup",
                required: 1,
                available: 0,
            });
        }

        let dispatch_x =
            u32::try_from(plan.config.batch_count).map_err(|_| VkFftError::ValueOutOfRange {
                field: "dispatch workgroup count",
            })?;

        let schedule = stockham_schedule(radix, sequence_len)?;
        let mut operations = Vec::with_capacity(schedule.len() + 2);
        let mut source = SharedBuffer::A;
        operations.push(KernelOperation::LoadGlobalToShared { target: source });

        let mut stage_size = 1usize;
        for (index, radix) in schedule.into_iter().enumerate() {
            if radix < 2 || sequence_len % radix != 0 {
                return Err(VkFftError::InvalidKernelIr(
                    "Stockham stage radix does not divide sequence length",
                ));
            }
            let output = source.alternate();
            operations.push(KernelOperation::StockhamStage(StockhamStage {
                index,
                radix,
                stage_size,
                butterflies: sequence_len / radix,
                input: source,
                output,
            }));
            stage_size = stage_size
                .checked_mul(radix)
                .ok_or(VkFftError::ArithmeticOverflow {
                    operation: "Stockham stage size",
                })?;
            source = output;
        }

        if stage_size != sequence_len {
            return Err(VkFftError::InvalidKernelIr(
                "Stockham radix schedule does not cover the sequence",
            ));
        }

        operations.push(KernelOperation::StoreSharedToGlobal {
            source,
            normalize: direction == Direction::Inverse && plan.config.normalize_inverse,
        });

        let direction_name = match direction {
            Direction::Forward => "forward",
            Direction::Inverse => "inverse",
        };
        let scheduler_candidate = if has_specialized_gpu_scheduler_policy(device) {
            match plan_gpu_smooth_stockham_uploads_for_batches(
                sequence_len,
                plan.config.batch_count,
                plan.config.precision,
                device,
            ) {
                Ok(upload_schedule)
                    if upload_schedule.upload_count == 1
                        && upload_schedule.axis_split.as_slice() == [sequence_len] =>
                {
                    upload_schedule.radix_schedules.into_iter().next()
                }
                Ok(_) | Err(VkFftError::UnsupportedKernelPath(_)) => None,
                Err(error) => return Err(error),
            }
        } else {
            None
        };

        let executable_schedule = scheduler_candidate
            .as_ref()
            .map(|schedule| {
                executable_register_schedule_layout(
                    schedule,
                    sequence_len,
                    device.max_threads_per_block,
                )
            })
            .transpose()?
            .flatten();
        let scheduler_hint = if executable_schedule.is_some() {
            scheduler_candidate
        } else {
            None
        };
        let execution_layout = executable_schedule
            .map(|plan| plan.layout)
            .unwrap_or(StockhamExecutionLayout::SharedPingPong);
        let stockham_shared_layout = executable_schedule
            .map(|plan| {
                plan_gpu_stockham_shared_memory_layout(
                    sequence_len,
                    plan.shared_elements,
                    scalar,
                    device,
                )
            })
            .transpose()?;
        let shared_memory = SharedMemoryPlan {
            elements_per_buffer: stockham_shared_layout
                .map(|layout| layout.allocated_elements)
                .unwrap_or(sequence_len),
            buffers: match execution_layout {
                StockhamExecutionLayout::SharedPingPong => 2,
                StockhamExecutionLayout::RegisterSingleShared
                | StockhamExecutionLayout::RegisterBoostSingleShared => 1,
            },
            scalar,
        };
        let required_shared = shared_memory.required_bytes()?;
        if required_shared > device.shared_memory_bytes {
            return Err(VkFftError::ResourceLimitExceeded {
                resource: "shared memory",
                required: required_shared,
                available: device.shared_memory_bytes,
            });
        }

        let local_size = executable_schedule
            .map(|plan| plan.threads)
            .unwrap_or(sequence_len.min(device.max_threads_per_block))
            .max(1);
        let workgroup_x = u32::try_from(local_size).map_err(|_| VkFftError::ValueOutOfRange {
            field: "workgroup size",
        })?;

        let twiddle_source = plan_gpu_stockham_twiddle_source(plan.config.precision, device);
        let mut bindings = vec![
            BufferBinding {
                set: 0,
                binding: 0,
                role: BufferRole::Input,
                access: BufferAccess::ReadOnly,
                scalar,
            },
            BufferBinding {
                set: 0,
                binding: 1,
                role: BufferRole::Output,
                access: BufferAccess::WriteOnly,
                scalar,
            },
        ];
        if twiddle_source == StockhamTwiddleSource::LookupTable {
            bindings.push(BufferBinding {
                set: 0,
                binding: 2,
                role: BufferRole::TwiddleLookupTable,
                access: BufferAccess::ReadOnly,
                scalar,
            });
        }

        let kernel = Self {
            name: format!("vkfft_stockham_{sequence_len}_{direction_name}"),
            scalar,
            direction,
            sequence_len,
            batch_count: plan.config.batch_count,
            workgroup_grouping: StockhamWorkgroupGrouping::single(local_size),
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
            bindings,
            shared_memory,
            stockham_shared_layout,
            io_mapping: StockhamIoMapping::Contiguous,
            input_modifier: StockhamInputModifier::None,
            output_modifier: StockhamOutputModifier::None,
            twiddle_source,
            execution_layout,
            subgroup: device.subgroup,
            scheduler_hint,
            rader_transpose: None,
            operations,
        };
        kernel.validate()?;
        Ok(kernel)
    }

    /// Use a narrower scalar only for a caller-visible input boundary. Arithmetic,
    /// registers, shared memory and immutable tables retain `self.scalar`.
    pub(crate) fn with_external_input_storage_scalar(
        mut self,
        storage: ScalarType,
    ) -> Result<Self> {
        if storage == self.scalar {
            return Ok(self);
        }
        let supported_pair = matches!(
            (self.scalar, storage),
            (ScalarType::F64, ScalarType::F32) | (ScalarType::F32, ScalarType::F16)
        );
        let supported_mapping = self.input_modifier == StockhamInputModifier::None
            && (matches!(
                self.io_mapping,
                StockhamIoMapping::Contiguous
                    | StockhamIoMapping::RaderGeneratorReverse(_)
                    | StockhamIoMapping::FourStepRight(_)
                    | StockhamIoMapping::FourStepThreeUpload2(_)
            ) || matches!(
                self.io_mapping,
                StockhamIoMapping::RaderGeneratorFourStep(mapping)
                    if mapping.caller.reads_external_input()
            ));
        if !supported_pair || !supported_mapping {
            return Err(VkFftError::UnsupportedPrecision {
                backend: "Stockham external input-storage boundary",
                precision: "mixed compute/storage scalar combination",
            });
        }
        self.bindings[0].scalar = storage;
        self.validate()?;
        Ok(self)
    }

    /// Use a narrower scalar only for a caller-visible output boundary. Rader scatter
    /// auxiliary storage is configured independently because zero-padding can place the
    /// original prime input on a compute-precision scratch boundary while output narrows.
    pub(crate) fn with_external_output_storage_scalar(
        mut self,
        storage: ScalarType,
    ) -> Result<Self> {
        if storage == self.scalar {
            return Ok(self);
        }
        let supported_pair = matches!(
            (self.scalar, storage),
            (ScalarType::F64, ScalarType::F32) | (ScalarType::F32, ScalarType::F16)
        );
        let supported_mapping = match self.output_modifier {
            StockhamOutputModifier::None => matches!(
                self.io_mapping,
                StockhamIoMapping::Contiguous
                    | StockhamIoMapping::FourStepLeft(_)
                    | StockhamIoMapping::FourStepThreeUpload0(_)
            ),
            StockhamOutputModifier::RaderScatter(_) => {
                self.io_mapping == StockhamIoMapping::Contiguous
            }
            _ => false,
        };
        if !supported_pair || !supported_mapping {
            return Err(VkFftError::UnsupportedPrecision {
                backend: "Stockham external output-storage boundary",
                precision: "mixed compute/storage scalar combination",
            });
        }
        self.bindings[1].scalar = storage;
        self.validate()?;
        Ok(self)
    }

    /// Configure the original prime-input resource consumed by a fused Rader scatter.
    /// This is intentionally independent from output storage: without zero padding both
    /// are caller storage, while forward zero padding feeds compute scratch to `aux` and
    /// inverse zero padding can keep `aux` narrow even when the final output is compute.
    pub(crate) fn with_rader_auxiliary_storage_scalar(
        mut self,
        storage: ScalarType,
    ) -> Result<Self> {
        if !matches!(
            self.output_modifier,
            StockhamOutputModifier::RaderScatter(_)
        ) || self.io_mapping != StockhamIoMapping::Contiguous
        {
            return Err(VkFftError::UnsupportedPrecision {
                backend: "Stockham Rader auxiliary-storage boundary",
                precision: "Rader auxiliary storage requires a contiguous fused scatter",
            });
        }
        if storage != self.scalar
            && !matches!(
                (self.scalar, storage),
                (ScalarType::F64, ScalarType::F32) | (ScalarType::F32, ScalarType::F16)
            )
        {
            return Err(VkFftError::UnsupportedPrecision {
                backend: "Stockham Rader auxiliary-storage boundary",
                precision: "mixed compute/storage scalar combination",
            });
        }
        let auxiliary = self
            .bindings
            .iter_mut()
            .find(|binding| binding.role == BufferRole::Auxiliary)
            .ok_or(VkFftError::InvalidKernelIr(
                "Rader scatter output is missing its auxiliary prime-input binding",
            ))?;
        auxiliary.scalar = storage;
        self.validate()?;
        Ok(self)
    }

    /// Replace only the external read/write mapping while preserving the validated
    /// Stockham arithmetic and scheduler/resource plan. This is used by the
    /// two-upload Four-step composition; standalone kernels remain contiguous.
    pub fn with_stockham_io_mapping(mut self, io_mapping: StockhamIoMapping) -> Result<Self> {
        self.io_mapping = io_mapping;
        self.validate()?;
        Ok(self)
    }

    /// Retarget the single terminal Stockham store normalization without changing
    /// arithmetic stages or external layout. `performConvolution` uses this when the
    /// embedded upload-0 inverse owns the full `1/N` scale and later inverse uploads
    /// must therefore remain unnormalized.
    pub(crate) fn with_store_normalization(mut self, normalize: bool) -> Result<Self> {
        let mut stores = 0usize;
        for operation in &mut self.operations {
            if let KernelOperation::StoreSharedToGlobal {
                normalize: store_normalize,
                ..
            } = operation
            {
                *store_normalize = normalize;
                stores += 1;
            }
        }
        if stores != 1 {
            return Err(VkFftError::InvalidKernelIr(
                "Stockham normalization retag requires exactly one terminal store",
            ));
        }
        self.validate()?;
        Ok(self)
    }

    /// Fuse a pointwise complex multiply from binding 2 into every logical stage-0
    /// input. This is intentionally restricted to contiguous input so the LUT index is
    /// exactly the convolution-frequency index independent of batch/grouping layout.
    pub(crate) fn with_lookup_table_input_multiply(mut self) -> Result<Self> {
        if self.input_modifier != StockhamInputModifier::None
            || self.io_mapping != StockhamIoMapping::Contiguous
        {
            return Err(VkFftError::UnsupportedKernelPath(
                "Stockham lookup-table input fusion currently requires an unmodified contiguous input",
            ));
        }
        let twiddle = self
            .bindings
            .last()
            .is_some_and(|binding| binding.role == BufferRole::TwiddleLookupTable)
            .then(|| {
                self.bindings
                    .pop()
                    .expect("checked trailing twiddle binding")
            });
        let binding =
            u32::try_from(self.bindings.len()).map_err(|_| VkFftError::ArithmeticOverflow {
                operation: "Stockham input LUT binding index",
            })?;
        self.input_modifier = StockhamInputModifier::MultiplyLookupTable;
        self.name.push_str("_input_mul_lut");
        self.bindings.push(BufferBinding {
            set: 0,
            binding,
            role: BufferRole::LookupTable,
            access: BufferAccess::ReadOnly,
            scalar: self.scalar,
        });
        if let Some(mut twiddle) = twiddle {
            twiddle.binding =
                u32::try_from(self.bindings.len()).map_err(|_| VkFftError::ArithmeticOverflow {
                    operation: "Stockham twiddle LUT binding index",
                })?;
            self.bindings.push(twiddle);
        }
        self.validate()?;
        Ok(self)
    }

    /// Fuse FFT-Rader's final scatter/DC/x0 reconstruction into Stockham stores.
    /// Binding 0/1 remain input/output; the original prime input is appended as a
    /// read-only auxiliary binding after any stage-0 LUT binding.
    pub(crate) fn with_rader_scatter_output(
        mut self,
        mapping: RaderScatterMapping,
    ) -> Result<Self> {
        if self.output_modifier != StockhamOutputModifier::None
            || self.io_mapping != StockhamIoMapping::Contiguous
        {
            return Err(VkFftError::UnsupportedKernelPath(
                "Rader scatter fusion currently requires an unmodified contiguous Stockham output",
            ));
        }
        mapping.validate(self.sequence_len)?;
        let twiddle = self
            .bindings
            .last()
            .is_some_and(|binding| binding.role == BufferRole::TwiddleLookupTable)
            .then(|| {
                self.bindings
                    .pop()
                    .expect("checked trailing twiddle binding")
            });
        let binding =
            u32::try_from(self.bindings.len()).map_err(|_| VkFftError::ArithmeticOverflow {
                operation: "Rader scatter auxiliary binding index",
            })?;
        self.output_modifier = StockhamOutputModifier::RaderScatter(mapping);
        self.name.push_str("_rader_scatter");
        self.bindings.push(BufferBinding {
            set: 0,
            binding,
            role: BufferRole::Auxiliary,
            access: BufferAccess::ReadOnly,
            scalar: self.scalar,
        });
        if let Some(mut twiddle) = twiddle {
            twiddle.binding =
                u32::try_from(self.bindings.len()).map_err(|_| VkFftError::ArithmeticOverflow {
                    operation: "Stockham twiddle LUT binding index",
                })?;
            self.bindings.push(twiddle);
        }
        self.validate()?;
        Ok(self)
    }

    /// Fuse R2C's mirrored-bin postprocess into the same Stockham dispatch. Binding
    /// 0/1 remain external input/output; a trailing read-write auxiliary buffer holds
    /// the complete half-size FFT until every invocation reaches the storage barrier.
    pub(crate) fn with_real_even_postprocess_output(mut self, full_len: usize) -> Result<Self> {
        if self.output_modifier != StockhamOutputModifier::None {
            return Err(VkFftError::UnsupportedKernelPath(
                "even-real postprocess fusion requires an unmodified Stockham output",
            ));
        }
        let StockhamIoMapping::RealEvenPack(input_mapping) = self.io_mapping else {
            return Err(VkFftError::UnsupportedKernelPath(
                "even-real postprocess fusion requires the matching packed-real input mapping",
            ));
        };
        let mapping = RealEvenPostprocessMapping { full_len };
        mapping.validate(self.sequence_len)?;
        if input_mapping.full_len != full_len || self.direction != Direction::Forward {
            return Err(VkFftError::UnsupportedKernelPath(
                "even-real postprocess fusion requires a matching forward R2C mapping",
            ));
        }
        let twiddle = self
            .bindings
            .last()
            .is_some_and(|binding| binding.role == BufferRole::TwiddleLookupTable)
            .then(|| {
                self.bindings
                    .pop()
                    .expect("checked trailing twiddle binding")
            });
        let binding =
            u32::try_from(self.bindings.len()).map_err(|_| VkFftError::ArithmeticOverflow {
                operation: "even-real postprocess scratch binding index",
            })?;
        self.output_modifier = StockhamOutputModifier::RealEvenPostprocess(mapping);
        self.name.push_str("_real_even_postprocess");
        self.bindings.push(BufferBinding {
            set: 0,
            binding,
            role: BufferRole::Auxiliary,
            access: BufferAccess::ReadWrite,
            scalar: self.scalar,
        });
        if let Some(mut twiddle) = twiddle {
            twiddle.binding =
                u32::try_from(self.bindings.len()).map_err(|_| VkFftError::ArithmeticOverflow {
                    operation: "Stockham twiddle LUT binding index",
                })?;
            self.bindings.push(twiddle);
        }
        self.validate()?;
        Ok(self)
    }

    /// Fuse the invocation-local C2R `UnpackEvenOdd` final pass into Stockham stores.
    /// Unlike Rader scatter this mapping needs no extra descriptor binding.
    pub(crate) fn with_real_even_unpack_output(mut self, full_len: usize) -> Result<Self> {
        if self.output_modifier != StockhamOutputModifier::None {
            return Err(VkFftError::UnsupportedKernelPath(
                "even-real unpack fusion requires an unmodified Stockham output",
            ));
        }
        let mapping = RealEvenUnpackMapping { full_len };
        mapping.validate(self.sequence_len)?;
        self.output_modifier = StockhamOutputModifier::RealEvenUnpack(mapping);
        self.name.push_str("_real_even_unpack");
        self.validate()?;
        Ok(self)
    }

    /// Apply an explicitly ported register schedule while optionally grouping
    /// multiple independent contiguous transforms into one physical workgroup.
    /// FFT-convolution Rader uses this for the upstream
    /// `VkFFTGetRegistersPerThreadOptimizeShared` container schedule and the first
    /// executable `containerFFTNum = 2` slice.
    pub(crate) fn with_register_schedule_grouping(
        mut self,
        schedule: RadixRegisterSchedule,
        transforms_per_workgroup: usize,
        device: DeviceProfile,
    ) -> Result<Self> {
        schedule.validate()?;
        if schedule.fft_len != self.sequence_len
            || schedule.rhs_transform_count != self.batch_count
            || device.max_threads_per_block == 0
            || transforms_per_workgroup == 0
            || !self.batch_count.is_multiple_of(transforms_per_workgroup)
        {
            return Err(VkFftError::InvalidKernelIr(
                "explicit Stockham register schedule/grouping does not match the kernel/device",
            ));
        }
        let executable = executable_register_schedule_layout(
            &schedule,
            self.sequence_len,
            device.max_threads_per_block,
        )?
        .ok_or(VkFftError::UnsupportedKernelPath(
            "explicit Stockham register schedule is not executable by the current register layout",
        ))?;
        if transforms_per_workgroup > 1
            && (self.io_mapping != StockhamIoMapping::Contiguous
                || executable.layout != StockhamExecutionLayout::RegisterSingleShared
                || schedule.register_boost != 1)
        {
            return Err(VkFftError::UnsupportedKernelPath(
                "grouped Stockham workgroups currently require contiguous boost-1 register execution",
            ));
        }
        let grouped_threads = executable
            .threads
            .checked_mul(transforms_per_workgroup)
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "grouped register-scheduled workgroup size",
            })?;
        if grouped_threads > device.max_threads_per_block {
            return Err(VkFftError::ResourceLimitExceeded {
                resource: "threads per grouped Stockham workgroup",
                required: grouped_threads,
                available: device.max_threads_per_block,
            });
        }

        let mut shared_layout = plan_gpu_stockham_shared_memory_layout(
            self.sequence_len,
            executable.shared_elements,
            self.scalar,
            device,
        )?;
        let grouped_shared_bytes = |layout: StockhamSharedMemoryLayout| -> Result<usize> {
            layout
                .allocated_elements
                .checked_mul(transforms_per_workgroup)
                .and_then(|value| value.checked_mul(self.scalar.complex_bytes()))
                .ok_or(VkFftError::ArithmeticOverflow {
                    operation: "grouped Stockham shared-memory size",
                })
        };
        // The ordinary helper evaluates padding for one transform. A grouped Rader
        // workgroup must fit all stripes together; if only the padding overflows,
        // reproduce VkFFT's physical-memory fallback by collapsing each stripe to
        // its unpadded logical extent.
        if grouped_shared_bytes(shared_layout)? > device.shared_memory_bytes
            && shared_layout.allocated_elements != shared_layout.logical_elements
        {
            shared_layout.allocated_elements = shared_layout.logical_elements;
            shared_layout.first_stage_stride = shared_layout.logical_elements;
            shared_layout.read_write_stride = shared_layout.logical_elements;
        }
        let grouped_shared_elements = shared_layout
            .allocated_elements
            .checked_mul(transforms_per_workgroup)
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "grouped Stockham shared-memory elements",
            })?;
        let shared_memory = SharedMemoryPlan {
            elements_per_buffer: grouped_shared_elements,
            buffers: 1,
            scalar: self.scalar,
        };
        let required_shared = shared_memory.required_bytes()?;
        if required_shared > device.shared_memory_bytes {
            return Err(VkFftError::ResourceLimitExceeded {
                resource: "shared memory",
                required: required_shared,
                available: device.shared_memory_bytes,
            });
        }
        let dispatch_x = self.batch_count / transforms_per_workgroup;
        self.execution_layout = executable.layout;
        self.stockham_shared_layout = Some(shared_layout);
        self.shared_memory = shared_memory;
        self.workgroup_grouping = StockhamWorkgroupGrouping {
            transforms_per_workgroup,
            threads_per_transform: executable.threads,
            axis_layout: StockhamWorkgroupAxisLayout::LinearX,
        };
        self.workgroup_size = WorkgroupSize {
            x: u32::try_from(grouped_threads).map_err(|_| VkFftError::ValueOutOfRange {
                field: "explicit register-scheduled workgroup size",
            })?,
            y: 1,
            z: 1,
        };
        self.dispatch = DispatchGeometry {
            x: u32::try_from(dispatch_x).map_err(|_| VkFftError::ValueOutOfRange {
                field: "grouped register-scheduled dispatch count",
            })?,
            y: 1,
            z: 1,
        };
        self.subgroup = device.subgroup;
        self.scheduler_hint = Some(schedule);
        self.validate()?;
        Ok(self)
    }

    /// Apply upstream `raderTranspose` lane geometry to a grouped boost-1 Stockham
    /// Apply the exact axis-0/upload-0 two-dimensional batch block selected by the
    /// covered `VkFFTSplitAxisBlock` slice. Unlike FFT-Rader grouping, independent
    /// transforms occupy a true local X/Y rectangle and may use upstream `axisSwapped`.
    pub(crate) fn with_axis0_batch_block(
        self,
        schedule: RadixRegisterSchedule,
        block: StockhamAxisBlockSchedule,
        device: DeviceProfile,
    ) -> Result<Self> {
        if self.io_mapping != StockhamIoMapping::Contiguous
            || self.input_modifier != StockhamInputModifier::None
            || self.output_modifier != StockhamOutputModifier::None
        {
            return Err(VkFftError::UnsupportedKernelPath(
                "Stockham axis-block grouping currently requires an unmodified contiguous root",
            ));
        }
        self.with_axis_batch_block_preserving_io(schedule, block, device)
    }

    /// Recompute only physical Stockham X/Y ownership while preserving already
    /// validated caller mappings/modifiers. FFT-Rader uses this after generator-order
    /// gather/LUT/scatter fusion has been installed; arithmetic, descriptors, and IO
    /// semantics remain untouched while the parent ND axis changes physical layout.
    pub(crate) fn with_axis_batch_block_preserving_io(
        mut self,
        schedule: RadixRegisterSchedule,
        block: StockhamAxisBlockSchedule,
        device: DeviceProfile,
    ) -> Result<Self> {
        schedule.validate()?;
        block.validate(self.batch_count, device)?;
        if schedule.fft_len != self.sequence_len || schedule.rhs_transform_count != self.batch_count
        {
            return Err(VkFftError::UnsupportedKernelPath(
                "Stockham axis-block grouping requires matching scheduler metadata",
            ));
        }
        let executable = executable_register_schedule_layout(
            &schedule,
            self.sequence_len,
            device.max_threads_per_block,
        )?
        .ok_or(VkFftError::UnsupportedKernelPath(
            "Stockham axis-block grouping requires an executable register schedule",
        ))?;
        if !matches!(
            executable.layout,
            StockhamExecutionLayout::RegisterSingleShared
                | StockhamExecutionLayout::RegisterBoostSingleShared
        ) || executable.threads > block.threads_per_transform
        {
            return Err(VkFftError::UnsupportedKernelPath(
                "Stockham axis block cannot provide the physical lanes required by the executable register layout",
            ));
        }

        let mut shared_layout = plan_gpu_stockham_shared_memory_layout(
            self.sequence_len,
            executable.shared_elements,
            self.scalar,
            device,
        )?;
        let grouped_shared_bytes = |layout: StockhamSharedMemoryLayout| -> Result<usize> {
            layout
                .allocated_elements
                .checked_mul(block.grouped_batch)
                .and_then(|value| value.checked_mul(self.scalar.complex_bytes()))
                .ok_or(VkFftError::ArithmeticOverflow {
                    operation: "axis-0 grouped Stockham shared-memory size",
                })
        };
        if grouped_shared_bytes(shared_layout)? > device.shared_memory_bytes
            && shared_layout.allocated_elements != shared_layout.logical_elements
        {
            shared_layout.allocated_elements = shared_layout.logical_elements;
            shared_layout.first_stage_stride = shared_layout.logical_elements;
            shared_layout.read_write_stride = shared_layout.logical_elements;
        }
        let grouped_shared_elements = shared_layout
            .allocated_elements
            .checked_mul(block.grouped_batch)
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "axis-0 grouped Stockham shared-memory elements",
            })?;
        let shared_memory = SharedMemoryPlan {
            elements_per_buffer: grouped_shared_elements,
            buffers: 1,
            scalar: self.scalar,
        };
        let required_shared = shared_memory.required_bytes()?;
        if required_shared > device.shared_memory_bytes {
            return Err(VkFftError::ResourceLimitExceeded {
                resource: "axis-0 grouped Stockham shared memory",
                required: required_shared,
                available: device.shared_memory_bytes,
            });
        }

        self.execution_layout = executable.layout;
        self.stockham_shared_layout = Some(shared_layout);
        self.shared_memory = shared_memory;
        self.workgroup_grouping = StockhamWorkgroupGrouping {
            transforms_per_workgroup: block.grouped_batch,
            threads_per_transform: block.threads_per_transform,
            axis_layout: if block.transforms_on_x {
                StockhamWorkgroupAxisLayout::TransformsXThreadsY
            } else {
                StockhamWorkgroupAxisLayout::ThreadsXTransformsY
            },
        };
        self.workgroup_size = WorkgroupSize {
            x: u32::try_from(block.local_size_x).map_err(|_| VkFftError::ValueOutOfRange {
                field: "axis-0 Stockham local size X",
            })?,
            y: u32::try_from(block.local_size_y).map_err(|_| VkFftError::ValueOutOfRange {
                field: "axis-0 Stockham local size Y",
            })?,
            z: 1,
        };
        self.dispatch = DispatchGeometry {
            x: u32::try_from(self.batch_count.div_ceil(block.grouped_batch)).map_err(|_| {
                VkFftError::ValueOutOfRange {
                    field: "axis-0 grouped Stockham dispatch count",
                }
            })?,
            y: 1,
            z: 1,
        };
        self.subgroup = device.subgroup;
        self.scheduler_hint = Some(schedule);
        self.validate()?;
        Ok(self)
    }

    /// Apply upstream `raderTranspose` lane geometry to a grouped boost-1 Stockham
    /// kernel. Stage 0 is container-major and later stages are container-interleaved;
    /// stage-specific logical group sizes come from the optimized register schedule.
    pub(crate) fn with_rader_transpose_register_schedule_grouping(
        self,
        schedule: RadixRegisterSchedule,
        transpose: RaderFftTransposeSchedule,
        device: DeviceProfile,
    ) -> Result<Self> {
        transpose.validate()?;
        if transpose.container_fft_dim != self.sequence_len
            || schedule.fft_len != self.sequence_len
            || schedule.register_boost != 1
            || schedule.register_boost_stage_radix.is_some()
            || schedule.stage_radices.len() != transpose.stages.len()
            || schedule
                .stage_radices
                .iter()
                .zip(&transpose.stages)
                .enumerate()
                .any(|(index, (radix, lane))| {
                    let registers = schedule.registers_per_thread_per_radix[*radix];
                    lane.stage_index != index
                        || lane.radix != *radix
                        || lane.logical_storage_per_thread != registers
                        || registers == 0
                        || lane.sub_logical_group_size != self.sequence_len.div_ceil(registers)
                })
        {
            return Err(VkFftError::UnsupportedKernelPath(
                "raderTranspose schedule does not match the boost-1 Stockham stage geometry",
            ));
        }
        let mut kernel =
            self.with_register_schedule_grouping(schedule, transpose.container_fft_num, device)?;
        if kernel.workgroup_size.x as usize != transpose.workgroup_threads {
            return Err(VkFftError::InvalidKernelIr(
                "raderTranspose workgroup geometry does not match the grouped Stockham kernel",
            ));
        }
        // The interleaved layout stores logical element i of every container next to
        // each other. It therefore does not use the ordinary per-container bank-pad
        // stride; reserve exactly N*C complex values.
        let mut shared_layout =
            kernel
                .stockham_shared_layout
                .ok_or(VkFftError::InvalidKernelIr(
                    "raderTranspose kernel is missing shared layout metadata",
                ))?;
        shared_layout.allocated_elements = kernel.sequence_len;
        shared_layout.first_stage_stride = kernel.sequence_len;
        shared_layout.read_write_stride = kernel.sequence_len;
        kernel.stockham_shared_layout = Some(shared_layout);
        kernel.shared_memory.elements_per_buffer = kernel
            .sequence_len
            .checked_mul(transpose.container_fft_num)
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "raderTranspose interleaved shared-memory elements",
            })?;
        if kernel.shared_memory.required_bytes()? > device.shared_memory_bytes {
            return Err(VkFftError::ResourceLimitExceeded {
                resource: "raderTranspose interleaved shared memory",
                required: kernel.shared_memory.required_bytes()?,
                available: device.shared_memory_bytes,
            });
        }
        kernel.rader_transpose = Some(transpose);
        kernel.validate()?;
        Ok(kernel)
    }

    /// Expand the optional VkFFT register/radix schedule into executable Stockham
    /// stage metadata. `None` means the portable planner schedule remains active.
    pub fn register_stockham_stages(&self) -> Result<Option<Vec<RegisterStockhamStage>>> {
        let Some(schedule) = &self.scheduler_hint else {
            return Ok(None);
        };
        schedule.validate()?;
        if schedule.fft_len != self.sequence_len || schedule.rhs_transform_count != self.batch_count
        {
            return Err(VkFftError::InvalidKernelIr(
                "Stockham scheduler hint does not match kernel dimensions",
            ));
        }

        let mut stages = Vec::with_capacity(schedule.stage_radices.len());
        let mut stage_size = 1usize;
        let mut source = SharedBuffer::A;
        let register_boost = schedule.register_boost;
        for (index, &radix) in schedule.stage_radices.iter().enumerate() {
            let registers_per_thread = schedule.registers_per_thread_per_radix[radix];
            if registers_per_thread < radix || !registers_per_thread.is_multiple_of(radix) {
                return Err(VkFftError::InvalidKernelIr(
                    "register-scheduled Stockham stage cannot pack whole radix butterflies",
                ));
            }
            let butterflies = self.sequence_len / radix;
            let logical_storage_per_thread = registers_per_thread
                .checked_mul(register_boost)
                .ok_or(VkFftError::ArithmeticOverflow {
                    operation: "register-scheduled Stockham logical storage",
                })?;
            if register_boost > 1 && !self.sequence_len.is_multiple_of(logical_storage_per_thread) {
                return Err(VkFftError::InvalidKernelIr(
                    "boosted register-scheduled Stockham logical storage must divide the sequence",
                ));
            }
            let butterflies_per_virtual_thread = logical_storage_per_thread / radix;
            if butterflies_per_virtual_thread == 0 {
                return Err(VkFftError::InvalidKernelIr(
                    "register-scheduled Stockham virtual-thread packing is inconsistent",
                ));
            }
            let virtual_thread_count = butterflies.div_ceil(butterflies_per_virtual_thread);
            let output = match self.execution_layout {
                StockhamExecutionLayout::SharedPingPong => source.alternate(),
                StockhamExecutionLayout::RegisterSingleShared
                | StockhamExecutionLayout::RegisterBoostSingleShared => SharedBuffer::A,
            };
            let input = match self.execution_layout {
                StockhamExecutionLayout::SharedPingPong => source,
                StockhamExecutionLayout::RegisterSingleShared
                | StockhamExecutionLayout::RegisterBoostSingleShared => SharedBuffer::A,
            };
            stages.push(RegisterStockhamStage {
                index,
                radix,
                stage_size,
                butterflies,
                register_boost,
                registers_per_thread,
                logical_storage_per_thread,
                butterflies_per_virtual_thread,
                virtual_thread_count,
                is_register_boost_stage: register_boost > 1
                    && index + 1 == schedule.stage_radices.len()
                    && schedule.register_boost_stage_radix == Some(radix),
                input,
                output,
            });
            stage_size = stage_size
                .checked_mul(radix)
                .ok_or(VkFftError::ArithmeticOverflow {
                    operation: "register-scheduled Stockham stage size",
                })?;
            source = output;
        }
        if stage_size != self.sequence_len {
            return Err(VkFftError::InvalidKernelIr(
                "register-scheduled Stockham stages do not cover the sequence",
            ));
        }
        Ok(Some(stages))
    }

    /// Analyze adjacent scheduler stages for safe cross-stage register residency.
    /// A resident boundary is accepted only when every target slot can be matched
    /// bijectively to a source slot owned by the same physical invocation, and the
    /// same slot permutation is valid for every invocation in the workgroup.
    pub fn register_stage_boundaries(&self) -> Result<Option<Vec<RegisterStageBoundary>>> {
        let Some(stages) = self.register_stockham_stages()? else {
            return Ok(None);
        };
        let mut boundaries = Vec::with_capacity(stages.len().saturating_sub(1));
        for pair in stages.windows(2) {
            let source = pair[0];
            let target = pair[1];
            let permutation =
                if self.execution_layout == StockhamExecutionLayout::RegisterSingleShared {
                    register_stage_boundary_permutation(source, target)?
                } else {
                    None
                };
            let residency = match permutation.as_deref() {
                Some(mapping)
                    if mapping
                        .iter()
                        .enumerate()
                        .all(|(target, &source)| target == source) =>
                {
                    RegisterStageBoundaryResidency::RegisterResidentIdentity
                }
                Some(_) => RegisterStageBoundaryResidency::RegisterResidentPermutation,
                None => RegisterStageBoundaryResidency::SharedExchangeRequired,
            };
            boundaries.push(RegisterStageBoundary {
                source_stage: source.index,
                target_stage: target.index,
                residency,
                target_to_source_registers: permutation,
            });
        }
        Ok(Some(boundaries))
    }

    /// Prove which adjacent boost-1 register stages would be lane-local if local
    /// invocation IDs mapped contiguously onto fixed-size subgroup lanes. A proof is
    /// returned only when every logical value stays in the modeled lane group and the
    /// same lane/register mapping repeats for every group. Vulkan does not generally
    /// promise this lane-numbering relation, so this remains conditional metadata and
    /// lowering keeps the shared-memory path until the backend establishes it.
    pub fn register_subgroup_boundaries(
        &self,
        subgroup_size: usize,
    ) -> Result<Option<Vec<Option<RegisterSubgroupBoundary>>>> {
        if subgroup_size == 0 {
            return Err(VkFftError::InvalidKernelIr(
                "register subgroup proof requires a non-zero subgroup size",
            ));
        }
        let Some(stages) = self.register_stockham_stages()? else {
            return Ok(None);
        };
        let mut boundaries = Vec::with_capacity(stages.len().saturating_sub(1));
        for pair in stages.windows(2) {
            let proof = if let Some(transpose) = self.rader_transpose.as_ref() {
                register_rader_transpose_subgroup_shuffle(
                    pair[0],
                    pair[1],
                    transpose,
                    subgroup_size,
                )?
            } else if self.workgroup_grouping.is_grouped()
                && self.workgroup_grouping.axis_layout != StockhamWorkgroupAxisLayout::LinearX
            {
                // The current subgroup ownership proof models consecutive local-X lanes.
                // True X/Y axis-block grouping stays on the shared exchange path.
                None
            } else {
                let proof =
                    register_stage_boundary_subgroup_shuffle(pair[0], pair[1], subgroup_size)?;
                if self.workgroup_grouping.is_grouped() {
                    proof
                        .map(|proof| {
                            expand_grouped_subgroup_boundary(
                                proof,
                                self.workgroup_grouping,
                                subgroup_size,
                            )
                        })
                        .transpose()?
                        .flatten()
                } else {
                    proof
                }
            };
            boundaries.push(proof);
        }
        Ok(Some(boundaries))
    }

    pub fn validate(&self) -> Result<()> {
        if self.scalar == ScalarType::F16 {
            return Err(VkFftError::InvalidKernelIr(
                "binary16 is a storage-only scalar; kernel arithmetic must use F32 or F64",
            ));
        }
        if self.sequence_len == 0 {
            return Err(VkFftError::InvalidKernelIr(
                "kernel sequence length must be non-zero",
            ));
        }
        if self.batch_count == 0 {
            return Err(VkFftError::InvalidKernelIr(
                "kernel batch count must be non-zero",
            ));
        }
        if self.workgroup_size.x == 0 || self.workgroup_size.y == 0 || self.workgroup_size.z == 0 {
            return Err(VkFftError::InvalidKernelIr(
                "workgroup dimensions must be non-zero",
            ));
        }
        self.workgroup_grouping
            .validate(self.batch_count, self.workgroup_size, self.dispatch)?;
        if let Some(transpose) = &self.rader_transpose {
            transpose.validate()?;
            let layout = self
                .stockham_shared_layout
                .ok_or(VkFftError::InvalidKernelIr(
                    "raderTranspose kernel requires shared-memory layout metadata",
                ))?;
            if self.execution_layout != StockhamExecutionLayout::RegisterSingleShared
                || !self.io_mapping.supports_grouped_register_input()
                || transpose.container_fft_dim != self.sequence_len
                || transpose.container_fft_num != self.workgroup_grouping.transforms_per_workgroup
                || transpose.workgroup_threads != self.workgroup_size.x as usize
                || layout.logical_elements != self.sequence_len
                || layout.allocated_elements != self.sequence_len
                || layout.first_stage_stride != self.sequence_len
                || layout.read_write_stride != self.sequence_len
                || self.shared_memory.elements_per_buffer
                    != self
                        .sequence_len
                        .checked_mul(transpose.container_fft_num)
                        .ok_or(VkFftError::ArithmeticOverflow {
                            operation: "raderTranspose validation shared-memory elements",
                        })?
            {
                return Err(VkFftError::InvalidKernelIr(
                    "raderTranspose metadata does not match the Stockham execution layout",
                ));
            }
            let stages = self
                .register_stockham_stages()?
                .ok_or(VkFftError::InvalidKernelIr(
                    "raderTranspose kernel requires register Stockham stages",
                ))?;
            if stages.len() != transpose.stages.len()
                || stages.iter().zip(&transpose.stages).any(|(stage, lane)| {
                    stage.index != lane.stage_index
                        || stage.radix != lane.radix
                        || stage.virtual_thread_count != lane.sub_logical_group_size
                })
            {
                return Err(VkFftError::InvalidKernelIr(
                    "raderTranspose stage ownership does not match Stockham stages",
                ));
            }
        }
        if self.workgroup_grouping.is_grouped()
            && (!self.io_mapping.supports_grouped_register_input()
                || !matches!(
                    self.execution_layout,
                    StockhamExecutionLayout::RegisterSingleShared
                        | StockhamExecutionLayout::RegisterBoostSingleShared
                ))
        {
            return Err(VkFftError::InvalidKernelIr(
                "grouped Stockham workgroups require a batch-addressable contiguous, Four-step, or fused-Rader register-shared mapping",
            ));
        }
        self.io_mapping
            .validate_kernel(self.sequence_len, self.batch_count)?;
        match (self.io_mapping, self.direction) {
            (StockhamIoMapping::CooleyLeft(mapping), direction)
                if mapping.direction == direction
                    && self.input_modifier == StockhamInputModifier::None
                    && self.output_modifier == StockhamOutputModifier::None => {}
            (StockhamIoMapping::CooleyLeft(_), _) => {
                return Err(VkFftError::InvalidKernelIr(
                    "Cooley-left Stockham mapping requires matching direction and unmodified Stockham I/O",
                ));
            }
            (StockhamIoMapping::RealEvenPack(_), Direction::Forward)
            | (StockhamIoMapping::RealEvenInversePreprocess(_), Direction::Inverse) => {}
            (StockhamIoMapping::RealEvenPack(_), Direction::Inverse) => {
                return Err(VkFftError::InvalidKernelIr(
                    "even-real pack mapping requires a forward Stockham kernel",
                ));
            }
            (StockhamIoMapping::RealEvenInversePreprocess(_), Direction::Forward) => {
                return Err(VkFftError::InvalidKernelIr(
                    "even-real inverse preprocess mapping requires an inverse Stockham kernel",
                ));
            }
            _ => {}
        }
        match self.execution_layout {
            StockhamExecutionLayout::SharedPingPong => {
                if self.workgroup_grouping.is_grouped()
                    || self.shared_memory.elements_per_buffer < self.sequence_len
                    || self.shared_memory.buffers < 2
                    || self.stockham_shared_layout.is_some()
                {
                    return Err(VkFftError::InvalidKernelIr(
                        "Stockham ping-pong layout requires two full unpadded shared-memory buffers",
                    ));
                }
            }
            StockhamExecutionLayout::RegisterSingleShared => {
                let Some(layout) = self.stockham_shared_layout else {
                    return Err(VkFftError::InvalidKernelIr(
                        "register-scheduled Stockham layout requires shared-memory stride metadata",
                    ));
                };
                let expected_shared_elements = layout
                    .allocated_elements
                    .checked_mul(self.workgroup_grouping.transforms_per_workgroup)
                    .ok_or(VkFftError::ArithmeticOverflow {
                        operation: "grouped Stockham validation shared-memory elements",
                    })?;
                if self.shared_memory.buffers != 1
                    || self
                        .scheduler_hint
                        .as_ref()
                        .is_none_or(|schedule| schedule.register_boost != 1)
                    || layout.logical_elements != self.sequence_len
                    || expected_shared_elements != self.shared_memory.elements_per_buffer
                {
                    return Err(VkFftError::InvalidKernelIr(
                        "register-scheduled Stockham layout has inconsistent shared-memory metadata",
                    ));
                }
            }
            StockhamExecutionLayout::RegisterBoostSingleShared => {
                let Some(schedule) = &self.scheduler_hint else {
                    return Err(VkFftError::InvalidKernelIr(
                        "boosted register Stockham layout requires scheduler metadata",
                    ));
                };
                let Some(layout) = self.stockham_shared_layout else {
                    return Err(VkFftError::InvalidKernelIr(
                        "boosted register Stockham layout requires shared-memory stride metadata",
                    ));
                };
                let expected_shared_elements = layout
                    .allocated_elements
                    .checked_mul(self.workgroup_grouping.transforms_per_workgroup)
                    .ok_or(VkFftError::ArithmeticOverflow {
                        operation: "grouped boosted Stockham validation shared-memory elements",
                    })?;
                if schedule.register_boost <= 1
                    || self.shared_memory.buffers != 1
                    || layout.logical_elements != self.sequence_len / schedule.register_boost
                    || expected_shared_elements != self.shared_memory.elements_per_buffer
                {
                    return Err(VkFftError::InvalidKernelIr(
                        "boosted register Stockham layout has inconsistent shared-memory metadata",
                    ));
                }
            }
        }
        self.output_modifier
            .validate_kernel(self.sequence_len, self.batch_count)?;
        if let StockhamOutputModifier::RealEvenUnpack(output_mapping) = self.output_modifier {
            let StockhamIoMapping::RealEvenInversePreprocess(input_mapping) = self.io_mapping
            else {
                return Err(VkFftError::InvalidKernelIr(
                    "even-real unpack output requires the matching inverse preprocess input mapping",
                ));
            };
            if self.direction != Direction::Inverse
                || input_mapping.full_len != output_mapping.full_len
            {
                return Err(VkFftError::InvalidKernelIr(
                    "even-real unpack input/output mappings disagree on inverse direction or full length",
                ));
            }
        }
        if let StockhamOutputModifier::RealEvenPostprocess(output_mapping) = self.output_modifier {
            let StockhamIoMapping::RealEvenPack(input_mapping) = self.io_mapping else {
                return Err(VkFftError::InvalidKernelIr(
                    "even-real postprocess output requires the matching packed-real input mapping",
                ));
            };
            if self.direction != Direction::Forward
                || input_mapping.full_len != output_mapping.full_len
            {
                return Err(VkFftError::InvalidKernelIr(
                    "even-real postprocess input/output mappings disagree on forward direction or full length",
                ));
            }
        }
        let input_binding_count = match self.input_modifier {
            StockhamInputModifier::None => 2,
            StockhamInputModifier::MultiplyLookupTable => 3,
        };
        let twiddle_binding_count =
            usize::from(self.twiddle_source == StockhamTwiddleSource::LookupTable);
        let expected_binding_count = input_binding_count
            + self.output_modifier.extra_binding_count()
            + twiddle_binding_count;
        let input_storage = self.bindings.first().map(|binding| binding.scalar);
        let output_storage = self.bindings.get(1).map(|binding| binding.scalar);
        let supported_storage_pair = |storage: Option<ScalarType>| {
            matches!(
                (self.scalar, storage),
                (ScalarType::F64, Some(ScalarType::F32)) | (ScalarType::F32, Some(ScalarType::F16))
            )
        };
        let mixed_input_storage = supported_storage_pair(input_storage)
            && self.input_modifier == StockhamInputModifier::None
            && (matches!(
                self.io_mapping,
                StockhamIoMapping::Contiguous
                    | StockhamIoMapping::RaderGeneratorReverse(_)
                    | StockhamIoMapping::FourStepRight(_)
                    | StockhamIoMapping::FourStepThreeUpload2(_)
            ) || matches!(
                self.io_mapping,
                StockhamIoMapping::RaderGeneratorFourStep(mapping)
                    if mapping.caller.reads_external_input()
            ));
        let mixed_output_storage = supported_storage_pair(output_storage)
            && match self.output_modifier {
                StockhamOutputModifier::None => matches!(
                    self.io_mapping,
                    StockhamIoMapping::Contiguous
                        | StockhamIoMapping::FourStepLeft(_)
                        | StockhamIoMapping::FourStepThreeUpload0(_)
                ),
                StockhamOutputModifier::RaderScatter(_) => {
                    self.io_mapping == StockhamIoMapping::Contiguous
                }
                _ => false,
            };
        if self.bindings.len() != expected_binding_count
            || self.bindings[0].role != BufferRole::Input
            || self.bindings[0].access != BufferAccess::ReadOnly
            || (self.bindings[0].scalar != self.scalar && !mixed_input_storage)
            || self.bindings[1].role != BufferRole::Output
            || self.bindings[1].access != BufferAccess::WriteOnly
            || (self.bindings[1].scalar != self.scalar && !mixed_output_storage)
        {
            return Err(VkFftError::InvalidKernelIr(
                "Stockham input/output storage bindings do not match the input modifier",
            ));
        }
        if self.input_modifier == StockhamInputModifier::MultiplyLookupTable
            && (self.io_mapping != StockhamIoMapping::Contiguous
                || self.bindings[2].binding != 2
                || self.bindings[2].role != BufferRole::LookupTable
                || self.bindings[2].access != BufferAccess::ReadOnly
                || self.bindings[2].scalar != self.scalar)
        {
            return Err(VkFftError::InvalidKernelIr(
                "Stockham lookup-table input modifier requires contiguous input and a read-only binding-2 LUT",
            ));
        }
        if let StockhamOutputModifier::RaderScatter(_) = self.output_modifier {
            let auxiliary = &self.bindings[input_binding_count];
            let auxiliary_storage_matches = auxiliary.scalar == self.scalar
                || matches!(
                    (self.scalar, auxiliary.scalar),
                    (ScalarType::F64, ScalarType::F32) | (ScalarType::F32, ScalarType::F16)
                );
            if self.io_mapping != StockhamIoMapping::Contiguous
                || auxiliary.binding as usize != input_binding_count
                || auxiliary.role != BufferRole::Auxiliary
                || auxiliary.access != BufferAccess::ReadOnly
                || !auxiliary_storage_matches
            {
                return Err(VkFftError::InvalidKernelIr(
                    "Rader scatter output modifier requires contiguous output and a trailing read-only auxiliary binding",
                ));
            }
        }
        if let StockhamOutputModifier::RealEvenPostprocess(_) = self.output_modifier {
            let auxiliary = &self.bindings[input_binding_count];
            if auxiliary.binding as usize != input_binding_count
                || auxiliary.role != BufferRole::Auxiliary
                || auxiliary.access != BufferAccess::ReadWrite
                || auxiliary.scalar != self.scalar
            {
                return Err(VkFftError::InvalidKernelIr(
                    "even-real postprocess output requires a trailing read-write auxiliary scratch binding",
                ));
            }
        }
        if self.twiddle_source == StockhamTwiddleSource::LookupTable {
            let twiddle_index = expected_binding_count - 1;
            let twiddle = &self.bindings[twiddle_index];
            if twiddle.binding as usize != twiddle_index
                || twiddle.role != BufferRole::TwiddleLookupTable
                || twiddle.access != BufferAccess::ReadOnly
                || twiddle.scalar != self.scalar
            {
                return Err(VkFftError::InvalidKernelIr(
                    "Stockham twiddle LUT must be the trailing read-only twiddle binding",
                ));
            }
        }

        if self.scheduler_hint.is_some() {
            let stages = self
                .register_stockham_stages()?
                .ok_or(VkFftError::InvalidKernelIr(
                    "missing register Stockham stages for scheduler metadata",
                ))?;
            match self.execution_layout {
                StockhamExecutionLayout::RegisterSingleShared => {
                    if stages.iter().any(|stage| {
                        stage.virtual_thread_count == 0
                            || stage.virtual_thread_count
                                > self.workgroup_grouping.threads_per_transform
                            || stage.input != SharedBuffer::A
                            || stage.output != SharedBuffer::A
                    }) {
                        return Err(VkFftError::InvalidKernelIr(
                            "single-shared Stockham stage exceeds its physical workgroup",
                        ));
                    }
                }
                StockhamExecutionLayout::RegisterBoostSingleShared => {
                    if stages.iter().any(|stage| {
                        stage.virtual_thread_count != self.workgroup_grouping.threads_per_transform
                            || stage.input != SharedBuffer::A
                            || stage.output != SharedBuffer::A
                    }) {
                        return Err(VkFftError::InvalidKernelIr(
                            "boosted single-shared Stockham requires one virtual thread per invocation",
                        ));
                    }
                }
                StockhamExecutionLayout::SharedPingPong => {}
            }
        }

        let mut expected_stage_size = 1usize;
        let mut expected_source = SharedBuffer::A;
        let mut saw_load = false;
        let mut saw_store = false;
        for operation in &self.operations {
            match operation {
                KernelOperation::LoadGlobalToShared { target } => {
                    if saw_load || *target != SharedBuffer::A {
                        return Err(VkFftError::InvalidKernelIr(
                            "unexpected Stockham load operation",
                        ));
                    }
                    saw_load = true;
                }
                KernelOperation::StockhamStage(stage) => {
                    if !saw_load || saw_store {
                        return Err(VkFftError::InvalidKernelIr(
                            "Stockham stage appears outside load/store boundaries",
                        ));
                    }
                    if stage.stage_size != expected_stage_size
                        || stage.input != expected_source
                        || stage.output != expected_source.alternate()
                        || stage.butterflies != self.sequence_len / stage.radix
                    {
                        return Err(VkFftError::InvalidKernelIr(
                            "inconsistent Stockham stage metadata",
                        ));
                    }
                    expected_stage_size = expected_stage_size.checked_mul(stage.radix).ok_or(
                        VkFftError::ArithmeticOverflow {
                            operation: "kernel IR validation stage size",
                        },
                    )?;
                    expected_source = stage.output;
                }
                KernelOperation::StoreSharedToGlobal { source, .. } => {
                    if saw_store || *source != expected_source {
                        return Err(VkFftError::InvalidKernelIr(
                            "unexpected Stockham store operation",
                        ));
                    }
                    saw_store = true;
                }
            }
        }

        if !saw_load || !saw_store || expected_stage_size != self.sequence_len {
            return Err(VkFftError::InvalidKernelIr(
                "incomplete Stockham operation sequence",
            ));
        }
        Ok(())
    }

    /// Period of the immutable Stockham unit-root table used by this kernel.
    /// Four-step upload kernels share the full logical transform period so both
    /// stage twiddles and cross-upload reorder twiddles use one descriptor.
    pub fn twiddle_lut_len(&self) -> Option<usize> {
        if self.twiddle_source != StockhamTwiddleSource::LookupTable {
            return None;
        }
        Some(match self.io_mapping {
            StockhamIoMapping::FourStepRight(mapping)
            | StockhamIoMapping::FourStepLeft(mapping) => mapping.logical_len,
            StockhamIoMapping::FourStepRightPreTwiddle(mapping) => mapping.four_step.logical_len,
            StockhamIoMapping::FourStepThreeUpload2(mapping)
            | StockhamIoMapping::FourStepThreeUpload1(mapping)
            | StockhamIoMapping::FourStepThreeUpload0(mapping) => mapping.logical_len,
            StockhamIoMapping::FourStepThreeUploadPreTwiddle(mapping) => {
                mapping.three_upload.logical_len
            }
            StockhamIoMapping::Contiguous
            | StockhamIoMapping::RaderGeneratorReverse(_)
            | StockhamIoMapping::RaderGeneratorFourStep(_)
            | StockhamIoMapping::RaderGeneratorCooleyRight(_)
            | StockhamIoMapping::CooleyLeft(_)
            | StockhamIoMapping::RealEvenPack(_)
            | StockhamIoMapping::RealEvenInversePreprocess(_) => self.sequence_len,
        })
    }

    pub fn required_shared_memory_bytes(&self) -> Result<usize> {
        self.shared_memory.required_bytes()
    }
}

fn stockham_schedule(radix: &RadixPlan, sequence_len: usize) -> Result<Vec<usize>> {
    if sequence_len == 1 {
        return Ok(Vec::new());
    }

    let merged_product = checked_product(&radix.merged_radices)?;
    if !radix.merged_radices.is_empty() && merged_product == sequence_len {
        return Ok(radix.merged_radices.clone());
    }

    let prime_product = checked_product(&radix.prime_factors)?;
    if prime_product == sequence_len {
        return Ok(radix.prime_factors.clone());
    }

    Err(VkFftError::InvalidKernelIr(
        "planner did not provide a complete Stockham radix schedule",
    ))
}

fn checked_product(values: &[usize]) -> Result<usize> {
    values.iter().try_fold(1usize, |acc, &value| {
        acc.checked_mul(value)
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "kernel radix product",
            })
    })
}

fn register_stage_slot_is_active(
    stage: RegisterStockhamStage,
    virtual_thread: usize,
    slot: usize,
) -> Result<bool> {
    let group = slot / stage.radix;
    if group >= stage.butterflies_per_virtual_thread {
        return Err(VkFftError::InvalidKernelIr(
            "register stage slot exceeds its butterfly packing",
        ));
    }
    let butterfly = virtual_thread
        .checked_mul(stage.butterflies_per_virtual_thread)
        .and_then(|value| value.checked_add(group))
        .ok_or(VkFftError::ArithmeticOverflow {
            operation: "register stage active butterfly",
        })?;
    Ok(butterfly < stage.butterflies)
}

fn register_stage_input_index(
    stage: RegisterStockhamStage,
    virtual_thread: usize,
    slot: usize,
) -> Result<usize> {
    let group = slot / stage.radix;
    let lane = slot % stage.radix;
    if group >= stage.butterflies_per_virtual_thread {
        return Err(VkFftError::InvalidKernelIr(
            "register stage slot exceeds its butterfly packing",
        ));
    }
    let butterfly = virtual_thread
        .checked_mul(stage.butterflies_per_virtual_thread)
        .and_then(|value| value.checked_add(group))
        .ok_or(VkFftError::ArithmeticOverflow {
            operation: "register stage input butterfly",
        })?;
    lane.checked_mul(stage.butterflies)
        .and_then(|lane_offset| butterfly.checked_add(lane_offset))
        .ok_or(VkFftError::ArithmeticOverflow {
            operation: "register stage input index",
        })
}

fn register_stage_output_index(
    stage: RegisterStockhamStage,
    virtual_thread: usize,
    slot: usize,
) -> Result<usize> {
    let group = slot / stage.radix;
    let lane = slot % stage.radix;
    if group >= stage.butterflies_per_virtual_thread {
        return Err(VkFftError::InvalidKernelIr(
            "register stage slot exceeds its butterfly packing",
        ));
    }
    let butterfly = virtual_thread
        .checked_mul(stage.butterflies_per_virtual_thread)
        .and_then(|value| value.checked_add(group))
        .ok_or(VkFftError::ArithmeticOverflow {
            operation: "register stage output butterfly",
        })?;
    let stage_invocation = butterfly % stage.stage_size;
    butterfly
        .checked_sub(stage_invocation)
        .and_then(|value| value.checked_mul(stage.radix))
        .and_then(|value| value.checked_add(stage_invocation))
        .and_then(|value| {
            lane.checked_mul(stage.stage_size)
                .and_then(|lane_offset| value.checked_add(lane_offset))
        })
        .ok_or(VkFftError::ArithmeticOverflow {
            operation: "register stage output index",
        })
}

fn register_stage_boundary_permutation(
    source: RegisterStockhamStage,
    target: RegisterStockhamStage,
) -> Result<Option<Vec<usize>>> {
    if source.register_boost != 1
        || target.register_boost != 1
        || source.registers_per_thread != target.registers_per_thread
        || source.virtual_thread_count != target.virtual_thread_count
        || source.registers_per_thread == 0
        || source
            .virtual_thread_count
            .checked_mul(source.logical_storage_per_thread)
            != Some(source.butterflies * source.radix)
        || target
            .virtual_thread_count
            .checked_mul(target.logical_storage_per_thread)
            != Some(target.butterflies * target.radix)
    {
        return Ok(None);
    }

    let registers = source.registers_per_thread;
    let mut expected_permutation: Option<Vec<usize>> = None;
    for virtual_thread in 0..source.virtual_thread_count {
        let mut source_slots_by_index = Vec::with_capacity(registers);
        for source_slot in 0..registers {
            let logical_index = register_stage_output_index(source, virtual_thread, source_slot)?;
            if source_slots_by_index
                .iter()
                .any(|(existing, _)| *existing == logical_index)
            {
                return Ok(None);
            }
            source_slots_by_index.push((logical_index, source_slot));
        }

        let mut permutation = vec![usize::MAX; registers];
        let mut used_source_slots = vec![false; registers];
        for (target_slot, destination) in permutation.iter_mut().enumerate() {
            let logical_index = register_stage_input_index(target, virtual_thread, target_slot)?;
            let Some((_, source_slot)) = source_slots_by_index
                .iter()
                .find(|(source_index, _)| *source_index == logical_index)
            else {
                return Ok(None);
            };
            let source_slot = *source_slot;
            if used_source_slots[source_slot] {
                return Ok(None);
            }
            used_source_slots[source_slot] = true;
            *destination = source_slot;
        }
        if used_source_slots.iter().any(|used| !used) {
            return Ok(None);
        }
        if expected_permutation
            .as_ref()
            .is_some_and(|expected| *expected != permutation)
        {
            return Ok(None);
        }
        expected_permutation = Some(permutation);
    }
    Ok(expected_permutation)
}

fn register_stage_boundary_subgroup_shuffle(
    source: RegisterStockhamStage,
    target: RegisterStockhamStage,
    subgroup_size: usize,
) -> Result<Option<RegisterSubgroupBoundary>> {
    if subgroup_size < 2
        || source.register_boost != 1
        || target.register_boost != 1
        || source.registers_per_thread != target.registers_per_thread
        || source.virtual_thread_count != target.virtual_thread_count
        || source.registers_per_thread == 0
        || source
            .virtual_thread_count
            .checked_mul(source.logical_storage_per_thread)
            != Some(source.butterflies * source.radix)
        || target
            .virtual_thread_count
            .checked_mul(target.logical_storage_per_thread)
            != Some(target.butterflies * target.radix)
    {
        return Ok(None);
    }

    let registers = source.registers_per_thread;
    let threads = source.virtual_thread_count;
    if threads == 0 || (threads > subgroup_size && !threads.is_multiple_of(subgroup_size)) {
        return Ok(None);
    }
    let active_lanes = threads.min(subgroup_size);
    let logical_values = threads
        .checked_mul(registers)
        .ok_or(VkFftError::ArithmeticOverflow {
            operation: "register subgroup logical value count",
        })?;
    let mut owners = vec![None; logical_values];
    for source_thread in 0..threads {
        for source_register in 0..registers {
            let logical_index =
                register_stage_output_index(source, source_thread, source_register)?;
            if logical_index >= logical_values || owners[logical_index].is_some() {
                return Ok(None);
            }
            owners[logical_index] = Some((source_thread, source_register));
        }
    }
    if owners.iter().any(Option::is_none) {
        return Ok(None);
    }

    let subgroup_count = threads.div_ceil(subgroup_size);
    let mut expected: Option<Vec<Vec<RegisterSubgroupSource>>> = None;
    for subgroup in 0..subgroup_count {
        let base = subgroup
            .checked_mul(subgroup_size)
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "register subgroup base invocation",
            })?;
        let lanes = (threads - base).min(subgroup_size);
        if lanes != active_lanes {
            return Ok(None);
        }
        let mut mapping = (0..registers)
            .map(|_| Vec::with_capacity(lanes))
            .collect::<Vec<_>>();
        for target_lane in 0..lanes {
            let target_thread = base + target_lane;
            for (target_register, slot_mapping) in mapping.iter_mut().enumerate() {
                let logical_index =
                    register_stage_input_index(target, target_thread, target_register)?;
                let Some((source_thread, source_register)) =
                    owners.get(logical_index).and_then(|owner| *owner)
                else {
                    return Ok(None);
                };
                if source_thread / subgroup_size != subgroup {
                    return Ok(None);
                }
                slot_mapping.push(RegisterSubgroupSource {
                    lane: source_thread % subgroup_size,
                    register: source_register,
                });
            }
        }
        if expected.as_ref().is_some_and(|value| *value != mapping) {
            return Ok(None);
        }
        expected = Some(mapping);
    }

    Ok(expected.map(|target_to_source| RegisterSubgroupBoundary {
        source_stage: source.index,
        target_stage: target.index,
        subgroup_size,
        active_lanes,
        transforms_per_subgroup: 1,
        lanes_per_transform: threads,
        target_to_source,
        lane_model: RegisterSubgroupLaneModel::ContiguousLocalInvocation,
    }))
}

fn register_rader_transpose_subgroup_shuffle(
    source: RegisterStockhamStage,
    target: RegisterStockhamStage,
    transpose: &RaderFftTransposeSchedule,
    subgroup_size: usize,
) -> Result<Option<RegisterSubgroupBoundary>> {
    transpose.validate()?;
    if subgroup_size < 2
        || source.register_boost != 1
        || target.register_boost != 1
        || source.registers_per_thread == 0
        || target.registers_per_thread == 0
        || source.index + 1 != target.index
        || target.index >= transpose.stages.len()
    {
        return Ok(None);
    }
    let source_lane = &transpose.stages[source.index];
    let target_lane = &transpose.stages[target.index];
    if source_lane.radix != source.radix
        || target_lane.radix != target.radix
        || source_lane.sub_logical_group_size != source.virtual_thread_count
        || target_lane.sub_logical_group_size != target.virtual_thread_count
    {
        return Ok(None);
    }

    let logical_workgroup_size = transpose.workgroup_threads;
    if logical_workgroup_size == 0
        || (logical_workgroup_size > subgroup_size
            && !logical_workgroup_size.is_multiple_of(subgroup_size))
    {
        return Ok(None);
    }
    let emitted_workgroup_size = logical_workgroup_size.max(subgroup_size);
    let subgroup_count = emitted_workgroup_size / subgroup_size;
    let active_lanes = logical_workgroup_size.min(subgroup_size);

    let mut logical_owners = vec![None; transpose.container_fft_dim];
    for source_thread in 0..source.virtual_thread_count {
        for source_register in 0..source.registers_per_thread {
            if !register_stage_slot_is_active(source, source_thread, source_register)? {
                continue;
            }
            let logical_index =
                register_stage_output_index(source, source_thread, source_register)?;
            if logical_index >= transpose.container_fft_dim
                || logical_owners[logical_index].is_some()
            {
                return Ok(None);
            }
            logical_owners[logical_index] = Some((source_thread, source_register));
        }
    }
    if logical_owners.iter().any(Option::is_none) {
        return Ok(None);
    }

    let mut expected: Option<Vec<Vec<RegisterSubgroupSource>>> = None;
    for subgroup in 0..subgroup_count {
        let subgroup_base =
            subgroup
                .checked_mul(subgroup_size)
                .ok_or(VkFftError::ArithmeticOverflow {
                    operation: "Rader transpose subgroup base",
                })?;
        let lanes = if logical_workgroup_size <= subgroup_size {
            logical_workgroup_size
        } else {
            subgroup_size
        };
        let mut mapping = (0..target.registers_per_thread)
            .map(|_| Vec::with_capacity(lanes))
            .collect::<Vec<_>>();
        for target_local_lane in 0..lanes {
            let target_physical = subgroup_base.checked_add(target_local_lane).ok_or(
                VkFftError::ArithmeticOverflow {
                    operation: "Rader transpose target physical lane",
                },
            )?;
            let target_coordinates = transpose.lane_coordinates(target.index, target_physical)?;
            for (target_register, slot_mapping) in mapping.iter_mut().enumerate() {
                let Some((target_thread, container)) = target_coordinates else {
                    slot_mapping.push(RegisterSubgroupSource {
                        lane: target_local_lane,
                        register: 0,
                    });
                    continue;
                };
                if !register_stage_slot_is_active(target, target_thread, target_register)? {
                    slot_mapping.push(RegisterSubgroupSource {
                        lane: target_local_lane,
                        register: 0,
                    });
                    continue;
                }
                let logical_index =
                    register_stage_input_index(target, target_thread, target_register)?;
                let Some((source_thread, source_register)) =
                    logical_owners.get(logical_index).and_then(|owner| *owner)
                else {
                    return Ok(None);
                };
                let source_physical =
                    transpose.physical_invocation(source.index, source_thread, container)?;
                if source_physical / subgroup_size != subgroup {
                    return Ok(None);
                }
                slot_mapping.push(RegisterSubgroupSource {
                    lane: source_physical % subgroup_size,
                    register: source_register,
                });
            }
        }
        if expected.as_ref().is_some_and(|value| *value != mapping) {
            return Ok(None);
        }
        expected = Some(mapping);
    }

    Ok(expected.map(|target_to_source| RegisterSubgroupBoundary {
        source_stage: source.index,
        target_stage: target.index,
        subgroup_size,
        active_lanes,
        transforms_per_subgroup: transpose.container_fft_num,
        lanes_per_transform: source.virtual_thread_count.max(target.virtual_thread_count),
        target_to_source,
        lane_model: RegisterSubgroupLaneModel::RaderTransposePhysicalInvocation,
    }))
}

fn expand_grouped_subgroup_boundary(
    mut proof: RegisterSubgroupBoundary,
    grouping: StockhamWorkgroupGrouping,
    subgroup_size: usize,
) -> Result<Option<RegisterSubgroupBoundary>> {
    if !grouping.is_grouped() {
        return Ok(Some(proof));
    }
    let physical_lanes = grouping
        .transforms_per_workgroup
        .checked_mul(grouping.threads_per_transform)
        .ok_or(VkFftError::ArithmeticOverflow {
            operation: "grouped subgroup physical lane count",
        })?;
    // Reuse one exact subgroup-local ownership pattern across as many full subgroups
    // as the physical workgroup contains. Every logical transform must occupy a whole
    // contiguous lane stripe inside one subgroup; no transform may straddle a subgroup.
    if proof.transforms_per_subgroup != 1
        || proof.active_lanes != grouping.threads_per_transform
        || proof.lanes_per_transform != grouping.threads_per_transform
        || grouping.threads_per_transform == 0
        || !subgroup_size.is_multiple_of(grouping.threads_per_transform)
        || physical_lanes < subgroup_size
        || !physical_lanes.is_multiple_of(subgroup_size)
    {
        return Ok(None);
    }
    let transforms_per_subgroup = subgroup_size / grouping.threads_per_transform;
    if transforms_per_subgroup == 0
        || !grouping
            .transforms_per_workgroup
            .is_multiple_of(transforms_per_subgroup)
    {
        return Ok(None);
    }

    let mut expanded = Vec::with_capacity(proof.target_to_source.len());
    for local_slot in &proof.target_to_source {
        if local_slot.len() != grouping.threads_per_transform {
            return Ok(None);
        }
        let mut slot = Vec::with_capacity(subgroup_size);
        for container in 0..transforms_per_subgroup {
            let base = container
                .checked_mul(grouping.threads_per_transform)
                .ok_or(VkFftError::ArithmeticOverflow {
                    operation: "grouped subgroup container lane base",
                })?;
            for owner in local_slot {
                // The per-transform proof must never point outside its own lane stripe.
                // Only after this check is the same mapping replicated into each
                // container stripe of one subgroup. Every additional subgroup reuses
                // this exact local-lane mapping through SubgroupId/InvocationId.
                if owner.lane >= grouping.threads_per_transform {
                    return Ok(None);
                }
                slot.push(RegisterSubgroupSource {
                    lane: base + owner.lane,
                    register: owner.register,
                });
            }
        }
        expanded.push(slot);
    }

    proof.active_lanes = subgroup_size;
    proof.transforms_per_subgroup = transforms_per_subgroup;
    proof.lanes_per_transform = grouping.threads_per_transform;
    proof.target_to_source = expanded;
    Ok(Some(proof))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ExecutableRegisterScheduleLayout {
    pub layout: StockhamExecutionLayout,
    pub threads: usize,
    pub shared_elements: usize,
}

pub(crate) fn executable_register_schedule_layout(
    schedule: &RadixRegisterSchedule,
    sequence_len: usize,
    max_threads_per_block: usize,
) -> Result<Option<ExecutableRegisterScheduleLayout>> {
    schedule.validate()?;
    if schedule.fft_len != sequence_len || schedule.stage_radices.is_empty() {
        return Ok(None);
    }

    let boost = schedule.register_boost;
    if boost > 1
        && (schedule.register_boost_stage_radix != Some(boost)
            || schedule.stage_radices.last().copied() != Some(boost)
            || schedule.registers_per_thread != schedule.min_registers_per_thread)
    {
        return Ok(None);
    }

    let base_registers = schedule.registers_per_thread;
    let mut required_threads = 0usize;

    for &radix in &schedule.stage_radices {
        let registers = schedule.registers_per_thread_per_radix[radix];
        if registers < radix || !registers.is_multiple_of(radix) {
            return Ok(None);
        }
        if boost > 1 && registers != base_registers {
            return Ok(None);
        }
        let logical_storage =
            registers
                .checked_mul(boost)
                .ok_or(VkFftError::ArithmeticOverflow {
                    operation: "register-scheduled Stockham stage storage",
                })?;
        if logical_storage == 0 {
            return Ok(None);
        }
        let stage_threads = if boost > 1 {
            if !sequence_len.is_multiple_of(logical_storage) {
                return Ok(None);
            }
            sequence_len / logical_storage
        } else {
            sequence_len.div_ceil(logical_storage)
        };
        if boost > 1 {
            if required_threads == 0 {
                required_threads = stage_threads;
            } else if stage_threads != required_threads {
                return Ok(None);
            }
        } else {
            // Boost-1 kernels can safely keep a physical workgroup sized for the
            // most demanding stage. Stages with fewer virtual threads leave the
            // extra invocations idle; Vulkan lowering keeps all barriers outside
            // the active-lane loops so those idle invocations still synchronize.
            required_threads = required_threads.max(stage_threads);
        }
    }
    if required_threads == 0 || required_threads > max_threads_per_block {
        return Ok(None);
    }

    if boost > 1 {
        let prefix_product = schedule.stage_radices[..schedule.stage_radices.len() - 1]
            .iter()
            .try_fold(1usize, |product, &radix| {
                product
                    .checked_mul(radix)
                    .ok_or(VkFftError::ArithmeticOverflow {
                        operation: "register-boost Stockham prefix product",
                    })
            })?;
        if prefix_product != sequence_len / boost {
            return Ok(None);
        }
    }

    Ok(Some(ExecutableRegisterScheduleLayout {
        layout: if boost == 1 {
            StockhamExecutionLayout::RegisterSingleShared
        } else {
            StockhamExecutionLayout::RegisterBoostSingleShared
        },
        threads: required_threads,
        shared_elements: sequence_len / boost,
    }))
}

/// Port the single-C2C `vkFFT_SharedMemory.h` stride calculation used by the
/// covered NVIDIA/Vulkan register Stockham kernels. The current IR executes one
/// transform per workgroup (`localSize[1] == 1`), so the upstream read/write
/// conflict stride simplifies to `logical + shared_banks/2`.
pub(crate) fn plan_gpu_stockham_shared_memory_layout(
    sequence_len: usize,
    logical_elements: usize,
    scalar: ScalarType,
    device: DeviceProfile,
) -> Result<StockhamSharedMemoryLayout> {
    if device.backend == Backend::CpuReference {
        return Err(VkFftError::UnsupportedKernelPath(
            "Stockham bank-conflict layout requires a GPU DeviceProfile",
        ));
    }
    if logical_elements == 0 || logical_elements > sequence_len {
        return Err(VkFftError::InvalidKernelIr(
            "Stockham shared-memory logical size is inconsistent",
        ));
    }

    let shared_banks = device.shared_banks;
    let bank_span_elements = shared_banks / 2;
    if bank_span_elements == 0 {
        return Ok(StockhamSharedMemoryLayout {
            logical_elements,
            allocated_elements: logical_elements,
            shared_banks,
            bank_span_elements,
            first_stage_stride: logical_elements,
            read_write_stride: logical_elements,
        });
    }

    let first_stage_stride = if sequence_len > bank_span_elements && sequence_len.is_power_of_two()
    {
        logical_elements.checked_mul(bank_span_elements + 1).ok_or(
            VkFftError::ArithmeticOverflow {
                operation: "Stockham first-stage bank-conflict stride",
            },
        )? / bank_span_elements
    } else {
        logical_elements
    };
    let read_write_stride =
        logical_elements
            .checked_add(bank_span_elements)
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "Stockham read/write bank-conflict stride",
            })?;
    let padded_elements = first_stage_stride.max(read_write_stride);
    let padded_bytes = padded_elements.checked_mul(scalar.complex_bytes()).ok_or(
        VkFftError::ArithmeticOverflow {
            operation: "Stockham padded shared-memory bytes",
        },
    )?;

    // Exact upstream fallback: if maxSharedStride does not fit sharedMemSize,
    // collapse both conflict-avoidance strides to the logical unpadded size.
    let (allocated_elements, first_stage_stride, read_write_stride) =
        if padded_bytes > device.shared_memory_bytes {
            (logical_elements, logical_elements, logical_elements)
        } else {
            (padded_elements, first_stage_stride, read_write_stride)
        };

    Ok(StockhamSharedMemoryLayout {
        logical_elements,
        allocated_elements,
        shared_banks,
        bank_span_elements,
        first_stage_stride,
        read_write_stride,
    })
}

/// Compatibility wrapper for the original NVIDIA/Vulkan helper.
#[allow(dead_code)]
pub(crate) fn plan_nvidia_vulkan_stockham_shared_memory_layout(
    sequence_len: usize,
    logical_elements: usize,
    scalar: ScalarType,
    device: DeviceProfile,
) -> Result<StockhamSharedMemoryLayout> {
    if device.backend != Backend::Vulkan || device.vendor != GpuVendor::Nvidia {
        return Err(VkFftError::UnsupportedKernelPath(
            "Stockham bank-conflict compatibility helper requires NVIDIA Vulkan",
        ));
    }
    plan_gpu_stockham_shared_memory_layout(sequence_len, logical_elements, scalar, device)
}

fn register_index(
    raw_index: usize,
    logical_registers_per_thread: usize,
    physical_registers_per_boost_lane: usize,
) -> usize {
    (raw_index / logical_registers_per_thread) * physical_registers_per_boost_lane
        + raw_index % logical_registers_per_thread
}

fn should_skip_boost_exchange(
    sequence_len: usize,
    stage: &RegisterStockhamStage,
    next: &RegisterStockhamStage,
) -> bool {
    next.is_register_boost_stage
        && next.radix == stage.register_boost
        && stage.stage_size * stage.radix == sequence_len / next.radix
}

fn execute_register_boost_stockham_batch(
    kernel: &KernelIr,
    stages: &[RegisterStockhamStage],
    input: &[Complex64],
    sign: f64,
) -> Result<Vec<Complex64>> {
    let schedule = kernel
        .scheduler_hint
        .as_ref()
        .ok_or(VkFftError::InvalidKernelIr(
            "boosted Stockham CPU execution requires scheduler metadata",
        ))?;
    let boost = schedule.register_boost;
    let registers_per_lane = schedule.registers_per_thread;
    let physical_registers =
        registers_per_lane
            .checked_mul(boost)
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "boosted Stockham CPU register file",
            })?;
    let threads = kernel.workgroup_grouping.threads_per_transform;
    if input.len() != kernel.sequence_len
        || threads
            .checked_mul(physical_registers)
            .is_none_or(|value| value != kernel.sequence_len)
    {
        return Err(VkFftError::InvalidKernelIr(
            "boosted Stockham CPU register geometry is inconsistent",
        ));
    }
    let shared_layout = kernel
        .stockham_shared_layout
        .ok_or(VkFftError::InvalidKernelIr(
            "boosted Stockham CPU execution requires shared-memory stride metadata",
        ))?;

    let zero = Complex64::new(0.0, 0.0);
    let mut registers = vec![vec![zero; physical_registers]; threads];
    for (thread, thread_registers) in registers.iter_mut().enumerate() {
        for (slot, value) in thread_registers.iter_mut().enumerate() {
            *value = input[thread + slot * threads];
        }
    }

    for (stage_index, stage) in stages.iter().enumerate() {
        let radix = stage.radix;
        let groups_per_boost = stage.registers_per_thread / radix;
        let logical_group_size = stage.virtual_thread_count;
        let denominator = (stage.stage_size * radix) as f64;
        for (thread, thread_registers) in registers.iter_mut().enumerate() {
            for boost_lane in 0..boost {
                for group in 0..groups_per_boost {
                    let butterfly =
                        thread + (group + boost_lane * groups_per_boost) * logical_group_size;
                    let stage_invocation = butterfly % stage.stage_size;
                    let mut values = vec![zero; radix];
                    let mut indices = vec![0usize; radix];
                    for lane in 0..radix {
                        let raw_index = group
                            + boost_lane * groups_per_boost
                            + lane * stage.logical_storage_per_thread / radix;
                        let index = register_index(
                            raw_index,
                            stage.registers_per_thread,
                            registers_per_lane,
                        );
                        indices[lane] = index;
                        let twiddle_angle =
                            sign * TAU * (stage_invocation * lane) as f64 / denominator;
                        values[lane] = thread_registers[index] * Complex64::exp_i(twiddle_angle);
                    }
                    let results = scheduled_radix_dft(&values, sign).unwrap_or_else(|| {
                        (0..radix)
                            .map(|output_lane| {
                                values
                                    .iter()
                                    .enumerate()
                                    .fold(zero, |sum, (input_lane, &value)| {
                                        let angle = sign * TAU * (output_lane * input_lane) as f64
                                            / radix as f64;
                                        sum + value * Complex64::exp_i(angle)
                                    })
                            })
                            .collect::<Vec<_>>()
                    });
                    for (lane, value) in results.into_iter().enumerate() {
                        thread_registers[indices[lane]] = value;
                    }
                }
            }
        }

        let Some(next) = stages.get(stage_index + 1) else {
            break;
        };
        if should_skip_boost_exchange(kernel.sequence_len, stage, next) {
            // VkFFT's special pre-boost shuffle avoids shared memory, but it is
            // not a no-op: the current radix lanes are transposed inside each
            // register-boost block so the extracted final boost radix sees
            // contiguous groups. This is the `else // registerBoost` branch in
            // appendRadixShuffleNonStrided.
            let mut reordered = vec![vec![zero; physical_registers]; threads];
            for (thread, thread_registers) in registers.iter().enumerate() {
                for boost_lane in 0..boost {
                    for group in 0..groups_per_boost {
                        for lane in 0..radix {
                            let raw_index = group
                                + boost_lane * groups_per_boost
                                + lane * stage.logical_storage_per_thread / radix;
                            let source_index = register_index(
                                raw_index,
                                stage.registers_per_thread,
                                registers_per_lane,
                            );
                            let destination_index =
                                group + lane * groups_per_boost + boost_lane * registers_per_lane;
                            reordered[thread][destination_index] = thread_registers[source_index];
                        }
                    }
                }
            }
            registers = reordered;
            continue;
        }

        let mut next_registers = vec![vec![zero; physical_registers]; threads];
        let use_bank_padding = shared_layout.stage_uses_first_stage_padding(
            kernel.sequence_len,
            stage.stage_size,
            stage.radix,
        );
        for boost_lane in 0..boost {
            let mut shared = vec![zero; kernel.shared_memory.elements_per_buffer];
            for (thread, thread_registers) in registers.iter().enumerate() {
                for group in 0..groups_per_boost {
                    let thread_base = thread + group * logical_group_size;
                    let stage_invocation = thread_base % stage.stage_size;
                    let block_invocation = thread_base - stage_invocation;
                    let inout = stage_invocation + block_invocation * radix;
                    for lane in 0..radix {
                        let raw_index = group
                            + boost_lane * groups_per_boost
                            + lane * stage.logical_storage_per_thread / radix;
                        let index = register_index(
                            raw_index,
                            stage.registers_per_thread,
                            registers_per_lane,
                        );
                        let logical_shared_index = inout + lane * stage.stage_size;
                        let shared_index =
                            shared_layout.physical_index(logical_shared_index, use_bank_padding)?;
                        shared[shared_index] = thread_registers[index];
                    }
                }
            }

            let next_groups = next.registers_per_thread / next.radix;
            for (thread, thread_registers) in next_registers.iter_mut().enumerate() {
                for group in 0..next_groups {
                    for lane in 0..next.radix {
                        let transfer_index = group * next.radix + lane;
                        let logical_shared_index =
                            thread + transfer_index * next.virtual_thread_count;
                        let shared_index =
                            shared_layout.physical_index(logical_shared_index, use_bank_padding)?;
                        // Upstream reads the next-stage shared stripe into
                        // tempID[t + k * registers_per_thread] and only then
                        // copies tempID back to regIDs. The computed next-stage
                        // logical `id` is descriptive here, not the tempID
                        // destination index.
                        let destination_index = transfer_index + boost_lane * registers_per_lane;
                        thread_registers[destination_index] = shared[shared_index];
                    }
                }
            }
        }
        registers = next_registers;
    }

    let mut output = vec![zero; kernel.sequence_len];
    for (thread, thread_registers) in registers.iter().enumerate() {
        for (slot, &value) in thread_registers.iter().enumerate() {
            output[thread + slot * threads] = value;
        }
    }
    Ok(output)
}

fn execute_register_single_shared_stockham_batch(
    kernel: &KernelIr,
    stages: &[RegisterStockhamStage],
    input: &[Complex64],
    sign: f64,
) -> Result<Vec<Complex64>> {
    let boundaries = kernel
        .register_stage_boundaries()?
        .ok_or(VkFftError::InvalidKernelIr(
            "register-single-shared execution requires stage-boundary metadata",
        ))?;
    let zero = Complex64::new(0.0, 0.0);
    let mut shared = input.to_vec();
    let mut resident_registers: Option<Vec<Vec<Complex64>>> = None;

    for (stage_index, &stage) in stages.iter().enumerate() {
        let incoming_boundary = stage_index
            .checked_sub(1)
            .and_then(|index| boundaries.get(index));
        let outgoing_boundary = boundaries.get(stage_index);
        let incoming_resident = incoming_boundary.is_some_and(|boundary| {
            boundary.residency != RegisterStageBoundaryResidency::SharedExchangeRequired
        });
        let outgoing_resident = outgoing_boundary.is_some_and(|boundary| {
            boundary.residency != RegisterStageBoundaryResidency::SharedExchangeRequired
        });
        let mut registers =
            vec![vec![zero; stage.registers_per_thread]; stage.virtual_thread_count];

        for virtual_thread in 0..stage.virtual_thread_count {
            for slot in 0..stage.registers_per_thread {
                if !register_stage_slot_is_active(stage, virtual_thread, slot)? {
                    continue;
                }
                let mut value = if incoming_resident {
                    let previous =
                        resident_registers
                            .as_ref()
                            .ok_or(VkFftError::InvalidKernelIr(
                                "resident register boundary is missing source registers",
                            ))?;
                    let permutation = incoming_boundary
                        .and_then(|boundary| boundary.target_to_source_registers.as_ref())
                        .ok_or(VkFftError::InvalidKernelIr(
                            "resident register boundary is missing its permutation",
                        ))?;
                    previous[virtual_thread][permutation[slot]]
                } else {
                    shared[register_stage_input_index(stage, virtual_thread, slot)?]
                };

                let group = slot / stage.radix;
                let lane = slot % stage.radix;
                let butterfly = virtual_thread
                    .checked_mul(stage.butterflies_per_virtual_thread)
                    .and_then(|value| value.checked_add(group))
                    .ok_or(VkFftError::ArithmeticOverflow {
                        operation: "register-single-shared butterfly",
                    })?;
                let stage_invocation = butterfly % stage.stage_size;
                let denominator = stage.stage_size.checked_mul(stage.radix).ok_or(
                    VkFftError::ArithmeticOverflow {
                        operation: "register-single-shared twiddle denominator",
                    },
                )?;
                let angle = sign * TAU * (stage_invocation * lane) as f64 / denominator as f64;
                value *= Complex64::exp_i(angle);
                registers[virtual_thread][slot] = value;
            }
        }

        for (virtual_thread, thread_registers) in registers.iter_mut().enumerate() {
            for group in 0..stage.butterflies_per_virtual_thread {
                let butterfly = virtual_thread
                    .checked_mul(stage.butterflies_per_virtual_thread)
                    .and_then(|value| value.checked_add(group))
                    .ok_or(VkFftError::ArithmeticOverflow {
                        operation: "register-single-shared local DFT butterfly",
                    })?;
                if butterfly >= stage.butterflies {
                    continue;
                }
                let start = group * stage.radix;
                let end = start + stage.radix;
                let values = &thread_registers[start..end];
                let results = scheduled_radix_dft(values, sign).unwrap_or_else(|| {
                    (0..stage.radix)
                        .map(|output_lane| {
                            values
                                .iter()
                                .enumerate()
                                .fold(zero, |sum, (input_lane, &value)| {
                                    let angle = sign * TAU * (output_lane * input_lane) as f64
                                        / stage.radix as f64;
                                    sum + value * Complex64::exp_i(angle)
                                })
                        })
                        .collect::<Vec<_>>()
                });
                thread_registers[start..end].copy_from_slice(&results);
            }
        }

        if stage_index + 1 == stages.len() {
            let mut output = vec![zero; kernel.sequence_len];
            for (virtual_thread, thread_registers) in registers.iter().enumerate() {
                for (slot, &value) in thread_registers.iter().enumerate() {
                    if !register_stage_slot_is_active(stage, virtual_thread, slot)? {
                        continue;
                    }
                    output[register_stage_output_index(stage, virtual_thread, slot)?] = value;
                }
            }
            return Ok(output);
        }

        if outgoing_resident {
            resident_registers = Some(registers);
        } else {
            for (virtual_thread, thread_registers) in registers.iter().enumerate() {
                for (slot, &value) in thread_registers.iter().enumerate() {
                    if !register_stage_slot_is_active(stage, virtual_thread, slot)? {
                        continue;
                    }
                    shared[register_stage_output_index(stage, virtual_thread, slot)?] = value;
                }
            }
            resident_registers = None;
        }
    }

    Err(VkFftError::InvalidKernelIr(
        "register-single-shared execution contains no final stage",
    ))
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

fn multiply_i_signed(value: Complex64, sign: f64) -> Complex64 {
    Complex64::new(-sign * value.im, sign * value.re)
}

fn radix3_dft(values: [Complex64; 3], sign: f64) -> [Complex64; 3] {
    const SIN_2PI_OVER_3: f64 = 0.866_025_403_784_438_6;
    let pair_sum = values[1] + values[2];
    let pair_diff = values[1] - values[2];
    let base = values[0] + pair_sum.scale(-0.5);
    let rotation = multiply_i_signed(pair_diff.scale(SIN_2PI_OVER_3), sign);
    [values[0] + pair_sum, base + rotation, base - rotation]
}

fn radix5_dft(values: [Complex64; 5], sign: f64) -> [Complex64; 5] {
    const C1: f64 = 0.309_016_994_374_947_45;
    const C2: f64 = -0.809_016_994_374_947_5;
    const S1: f64 = 0.951_056_516_295_153_5;
    const S2: f64 = 0.587_785_252_292_473_1;

    let p1 = values[1] + values[4];
    let p2 = values[2] + values[3];
    let q1 = values[1] - values[4];
    let q2 = values[2] - values[3];
    let base1 = values[0] + p1.scale(C1) + p2.scale(C2);
    let base2 = values[0] + p1.scale(C2) + p2.scale(C1);
    let rotation1 = multiply_i_signed(q1.scale(S1) + q2.scale(S2), sign);
    let rotation2 = multiply_i_signed(q1.scale(S2) - q2.scale(S1), sign);
    [
        values[0] + p1 + p2,
        base1 + rotation1,
        base2 + rotation2,
        base2 - rotation2,
        base1 - rotation1,
    ]
}

fn radix7_dft(values: [Complex64; 7], sign: f64) -> [Complex64; 7] {
    const C1: f64 = 0.623_489_801_858_733_5;
    const C2: f64 = -0.222_520_933_956_314_4;
    const C3: f64 = -0.900_968_867_902_419_1;
    const S1: f64 = 0.781_831_482_468_029_8;
    const S2: f64 = 0.974_927_912_181_823_6;
    const S3: f64 = 0.433_883_739_117_558_1;

    let p1 = values[1] + values[6];
    let p2 = values[2] + values[5];
    let p3 = values[3] + values[4];
    let q1 = values[1] - values[6];
    let q2 = values[2] - values[5];
    let q3 = values[3] - values[4];

    let base1 = values[0] + p1.scale(C1) + p2.scale(C2) + p3.scale(C3);
    let base2 = values[0] + p1.scale(C2) + p2.scale(C3) + p3.scale(C1);
    let base3 = values[0] + p1.scale(C3) + p2.scale(C1) + p3.scale(C2);
    let rotation1 = multiply_i_signed(q1.scale(S1) + q2.scale(S2) + q3.scale(S3), sign);
    let rotation2 = multiply_i_signed(q1.scale(S2) - q2.scale(S3) - q3.scale(S1), sign);
    let rotation3 = multiply_i_signed(q1.scale(S3) - q2.scale(S1) + q3.scale(S2), sign);
    [
        values[0] + p1 + p2 + p3,
        base1 + rotation1,
        base2 + rotation2,
        base3 + rotation3,
        base3 - rotation3,
        base2 - rotation2,
        base1 - rotation1,
    ]
}

fn upstream_prime_radix_dft<const N: usize>(
    values: [Complex64; N],
    sign: f64,
    input_permutation: &[usize],
    output_permutation: &[usize],
    cosine_cycle: &[f64],
    forward_sine_cycle: &[f64],
) -> [Complex64; N] {
    let pairs = (N - 1) / 2;
    debug_assert_eq!(input_permutation.len(), N);
    debug_assert_eq!(output_permutation.len(), N);
    debug_assert_eq!(cosine_cycle.len(), pairs);
    debug_assert_eq!(forward_sine_cycle.len(), pairs);
    let zero = Complex64::new(0.0, 0.0);
    let mut permuted = [zero; N];
    for (target, &source) in input_permutation.iter().enumerate() {
        permuted[target] = values[source];
    }

    let mut positive = [zero; N];
    let mut conjugate = [zero; N];
    let mut registers = [zero; N];
    registers[0] = permuted[0];
    for i in 0..pairs {
        let lhs = permuted[i + 1];
        let rhs = permuted[i + 1 + pairs];
        positive[i] = Complex64::new(lhs.re + rhs.re, lhs.im - rhs.im);
        conjugate[i] = Complex64::new(lhs.re - rhs.re, lhs.im + rhs.im);
        registers[0].re += positive[i].re;
        registers[0].im += conjugate[i].im;
    }

    let mut accumulator = [zero; N];
    let mut cross = [zero; N];
    for value in accumulator.iter_mut().take(pairs) {
        *value = permuted[0];
    }
    for i in 0..pairs {
        for j in 0..pairs {
            let id = ((2 * pairs - i) + j) % (2 * pairs);
            let cycle_index = id % pairs;
            let cosine = cosine_cycle[cycle_index];
            let mut sine = forward_sine_cycle[cycle_index] * -sign;
            if id >= pairs {
                sine = -sine;
            }
            accumulator[j].re += positive[i].re * cosine;
            accumulator[j].im += conjugate[i].im * cosine;
            cross[j].re += positive[i].im * sine;
            cross[j].im += conjugate[i].re * sine;
        }
    }

    for j in 0..pairs {
        registers[j + 1] = Complex64::new(
            accumulator[j].re - cross[j].re,
            accumulator[j].im + cross[j].im,
        );
        registers[j + 1 + pairs] = Complex64::new(
            accumulator[j].re + cross[j].re,
            accumulator[j].im - cross[j].im,
        );
    }

    let mut output = [zero; N];
    for (target, &source) in output_permutation.iter().enumerate() {
        output[target] = registers[source];
    }
    output
}

fn radix11_dft(values: [Complex64; 11], sign: f64) -> [Complex64; 11] {
    const INPUT_PERMUTATION: [usize; 11] = [0, 1, 2, 4, 8, 5, 10, 9, 7, 3, 6];
    const OUTPUT_PERMUTATION: [usize; 11] = [0, 1, 10, 3, 9, 7, 2, 4, 8, 5, 6];
    const COSINE_CYCLE: [f64; 5] = [
        0.841_253_532_831_181_2,
        -0.959_492_973_614_497_4,
        -0.142_314_838_273_285_14,
        -0.654_860_733_945_285_1,
        0.415_415_013_001_886_44,
    ];
    const FORWARD_SINE_CYCLE: [f64; 5] = [
        -0.540_640_817_455_597_6,
        0.281_732_556_841_429_7,
        -0.989_821_441_880_932_7,
        0.755_749_574_354_258_3,
        0.909_631_995_354_518_4,
    ];
    upstream_prime_radix_dft(
        values,
        sign,
        &INPUT_PERMUTATION,
        &OUTPUT_PERMUTATION,
        &COSINE_CYCLE,
        &FORWARD_SINE_CYCLE,
    )
}

fn radix13_dft(values: [Complex64; 13], sign: f64) -> [Complex64; 13] {
    const INPUT_PERMUTATION: [usize; 13] = [0, 1, 2, 4, 8, 3, 6, 12, 11, 9, 5, 10, 7];
    const OUTPUT_PERMUTATION: [usize; 13] = [0, 1, 12, 9, 11, 4, 8, 2, 10, 5, 3, 6, 7];
    const COSINE_CYCLE: [f64; 6] = [
        0.885_456_025_653_209_9,
        -0.970_941_817_426_052,
        0.120_536_680_255_323_05,
        -0.748_510_748_171_101_1,
        -0.354_604_887_042_535_6,
        0.568_064_746_731_155_8,
    ];
    const FORWARD_SINE_CYCLE: [f64; 6] = [
        -0.464_723_172_043_768_54,
        0.239_315_664_287_557_77,
        0.992_708_874_098_054,
        -0.663_122_658_240_795_2,
        0.935_016_242_685_414_8,
        0.822_983_865_893_656_4,
    ];
    upstream_prime_radix_dft(
        values,
        sign,
        &INPUT_PERMUTATION,
        &OUTPUT_PERMUTATION,
        &COSINE_CYCLE,
        &FORWARD_SINE_CYCLE,
    )
}

fn radix4_dft(values: [Complex64; 4], sign: f64) -> [Complex64; 4] {
    let sum02 = values[0] + values[2];
    let diff02 = values[0] - values[2];
    let sum13 = values[1] + values[3];
    let diff13 = values[1] - values[3];
    let rotated = Complex64::new(-sign * diff13.im, sign * diff13.re);
    [
        sum02 + sum13,
        diff02 + rotated,
        sum02 - sum13,
        diff02 - rotated,
    ]
}

fn radix8_dft(values: [Complex64; 8], sign: f64) -> [Complex64; 8] {
    let even = radix4_dft([values[0], values[2], values[4], values[6]], sign);
    let odd = radix4_dft([values[1], values[3], values[5], values[7]], sign);
    let twiddles = [
        Complex64::new(1.0, 0.0),
        Complex64::new(FRAC_1_SQRT_2, sign * FRAC_1_SQRT_2),
        Complex64::new(0.0, sign),
        Complex64::new(-FRAC_1_SQRT_2, sign * FRAC_1_SQRT_2),
    ];
    let mut output = [Complex64::new(0.0, 0.0); 8];
    for lane in 0..4 {
        let rotated = odd[lane] * twiddles[lane];
        output[lane] = even[lane] + rotated;
        output[lane + 4] = even[lane] - rotated;
    }
    output
}

fn radix16_dft(values: [Complex64; 16], sign: f64) -> [Complex64; 16] {
    const COS_PI_OVER_8: f64 = 0.923_879_532_511_286_7;
    const SIN_PI_OVER_8: f64 = 0.382_683_432_365_089_8;

    let even = radix8_dft(
        [
            values[0], values[2], values[4], values[6], values[8], values[10], values[12],
            values[14],
        ],
        sign,
    );
    let odd = radix8_dft(
        [
            values[1], values[3], values[5], values[7], values[9], values[11], values[13],
            values[15],
        ],
        sign,
    );
    let twiddles = [
        Complex64::new(1.0, 0.0),
        Complex64::new(COS_PI_OVER_8, sign * SIN_PI_OVER_8),
        Complex64::new(FRAC_1_SQRT_2, sign * FRAC_1_SQRT_2),
        Complex64::new(SIN_PI_OVER_8, sign * COS_PI_OVER_8),
        Complex64::new(0.0, sign),
        Complex64::new(-SIN_PI_OVER_8, sign * COS_PI_OVER_8),
        Complex64::new(-FRAC_1_SQRT_2, sign * FRAC_1_SQRT_2),
        Complex64::new(-COS_PI_OVER_8, sign * SIN_PI_OVER_8),
    ];
    let mut output = [Complex64::new(0.0, 0.0); 16];
    for lane in 0..8 {
        let rotated = odd[lane] * twiddles[lane];
        output[lane] = even[lane] + rotated;
        output[lane + 8] = even[lane] - rotated;
    }
    output
}

fn radix32_dft(values: [Complex64; 32], sign: f64) -> [Complex64; 32] {
    let even = radix16_dft(
        [
            values[0], values[2], values[4], values[6], values[8], values[10], values[12],
            values[14], values[16], values[18], values[20], values[22], values[24], values[26],
            values[28], values[30],
        ],
        sign,
    );
    let odd = radix16_dft(
        [
            values[1], values[3], values[5], values[7], values[9], values[11], values[13],
            values[15], values[17], values[19], values[21], values[23], values[25], values[27],
            values[29], values[31],
        ],
        sign,
    );
    let twiddles: [Complex64; 16] =
        core::array::from_fn(|lane| Complex64::exp_i(sign * TAU * lane as f64 / 32.0));
    let mut output = [Complex64::new(0.0, 0.0); 32];
    for lane in 0..16 {
        let rotated = odd[lane] * twiddles[lane];
        output[lane] = even[lane] + rotated;
        output[lane + 16] = even[lane] - rotated;
    }
    output
}

fn scheduled_radix_dft(values: &[Complex64], sign: f64) -> Option<Vec<Complex64>> {
    match values.len() {
        2 => Some(vec![values[0] + values[1], values[0] - values[1]]),
        3 => Some(radix3_dft([values[0], values[1], values[2]], sign).to_vec()),
        4 => Some(radix4_dft([values[0], values[1], values[2], values[3]], sign).to_vec()),
        5 => Some(
            radix5_dft(
                [values[0], values[1], values[2], values[3], values[4]],
                sign,
            )
            .to_vec(),
        ),
        6 => composite_radix_dft(values, 2, 3, sign),
        7 => Some(
            radix7_dft(
                [
                    values[0], values[1], values[2], values[3], values[4], values[5], values[6],
                ],
                sign,
            )
            .to_vec(),
        ),
        11 => Some(
            radix11_dft(
                [
                    values[0], values[1], values[2], values[3], values[4], values[5], values[6],
                    values[7], values[8], values[9], values[10],
                ],
                sign,
            )
            .to_vec(),
        ),
        12 => composite_radix_dft(values, 3, 4, sign),
        13 => Some(
            radix13_dft(
                [
                    values[0], values[1], values[2], values[3], values[4], values[5], values[6],
                    values[7], values[8], values[9], values[10], values[11], values[12],
                ],
                sign,
            )
            .to_vec(),
        ),
        14 => composite_radix_dft(values, 2, 7, sign),
        15 => composite_radix_dft(values, 3, 5, sign),
        8 => Some(
            radix8_dft(
                [
                    values[0], values[1], values[2], values[3], values[4], values[5], values[6],
                    values[7],
                ],
                sign,
            )
            .to_vec(),
        ),
        9 => composite_radix_dft(values, 3, 3, sign),
        10 => composite_radix_dft(values, 2, 5, sign),
        16 => Some(
            radix16_dft(
                [
                    values[0], values[1], values[2], values[3], values[4], values[5], values[6],
                    values[7], values[8], values[9], values[10], values[11], values[12],
                    values[13], values[14], values[15],
                ],
                sign,
            )
            .to_vec(),
        ),
        32 => Some(
            radix32_dft(
                [
                    values[0], values[1], values[2], values[3], values[4], values[5], values[6],
                    values[7], values[8], values[9], values[10], values[11], values[12],
                    values[13], values[14], values[15], values[16], values[17], values[18],
                    values[19], values[20], values[21], values[22], values[23], values[24],
                    values[25], values[26], values[27], values[28], values[29], values[30],
                    values[31],
                ],
                sign,
            )
            .to_vec(),
        ),
        _ => None,
    }
}

fn composite_radix_dft(
    values: &[Complex64],
    outer_radix: usize,
    inner_radix: usize,
    sign: f64,
) -> Option<Vec<Complex64>> {
    let radix = outer_radix.checked_mul(inner_radix)?;
    if values.len() != radix {
        return None;
    }

    // Local Cooley-Tukey decomposition with n = n1 + outer_radix*n2 and
    // k = k2 + inner_radix*k1. The child transforms are existing fixed kernels,
    // removing the generic O(radix^2) local DFT for VkFFT's merged composite
    // scheduler radices 6/9/10/12/14/15.
    let mut first_pass = vec![Complex64::new(0.0, 0.0); radix];
    for n1 in 0..outer_radix {
        let inner_input = (0..inner_radix)
            .map(|n2| values[n1 + outer_radix * n2])
            .collect::<Vec<_>>();
        let inner_output = scheduled_radix_dft(&inner_input, sign)?;
        for k2 in 0..inner_radix {
            let angle = sign * core::f64::consts::TAU * (n1 * k2) as f64 / radix as f64;
            first_pass[n1 * inner_radix + k2] = inner_output[k2] * Complex64::exp_i(angle);
        }
    }

    let mut output = vec![Complex64::new(0.0, 0.0); radix];
    for k2 in 0..inner_radix {
        let outer_input = (0..outer_radix)
            .map(|n1| first_pass[n1 * inner_radix + k2])
            .collect::<Vec<_>>();
        let outer_output = scheduled_radix_dft(&outer_input, sign)?;
        for k1 in 0..outer_radix {
            output[k2 + inner_radix * k1] = outer_output[k1];
        }
    }
    Some(output)
}

/// Execute the typed Stockham IR on the CPU. This is not the general CPU
/// reference FFT; it is an interpreter for the exact stage/indexing semantics
/// emitted to GPU backends and is therefore useful for validating codegen.
pub fn execute_stockham_ir(kernel: &KernelIr, input: &[Complex64]) -> Result<Vec<Complex64>> {
    execute_stockham_ir_with_resources(kernel, input, None, None)
}

pub(crate) fn execute_stockham_ir_with_lookup(
    kernel: &KernelIr,
    input: &[Complex64],
    lookup: Option<&[Complex64]>,
) -> Result<Vec<Complex64>> {
    execute_stockham_ir_with_resources(kernel, input, lookup, None)
}

pub(crate) fn execute_stockham_ir_with_resources(
    kernel: &KernelIr,
    input: &[Complex64],
    lookup: Option<&[Complex64]>,
    auxiliary: Option<&[Complex64]>,
) -> Result<Vec<Complex64>> {
    kernel.validate()?;
    let expected_input = kernel
        .io_mapping
        .input_elements(kernel.sequence_len, kernel.batch_count)?;
    let expected_output = kernel
        .output_modifier
        .output_elements(kernel.sequence_len, kernel.batch_count)?;
    if input.len() != expected_input {
        return Err(VkFftError::InputLengthMismatch {
            expected: expected_input,
            actual: input.len(),
        });
    }
    let lookup = match kernel.input_modifier {
        StockhamInputModifier::None => None,
        StockhamInputModifier::MultiplyLookupTable => {
            let lookup = lookup.ok_or(VkFftError::InvalidKernelIr(
                "Stockham lookup-table input modifier requires CPU lookup data",
            ))?;
            if lookup.len() != kernel.sequence_len {
                return Err(VkFftError::InputLengthMismatch {
                    expected: kernel.sequence_len,
                    actual: lookup.len(),
                });
            }
            Some(lookup)
        }
    };
    let rader_scatter = match kernel.output_modifier {
        StockhamOutputModifier::None
        | StockhamOutputModifier::RealEvenPostprocess(_)
        | StockhamOutputModifier::RealEvenUnpack(_) => None,
        StockhamOutputModifier::RaderScatter(mapping) => {
            let auxiliary = auxiliary.ok_or(VkFftError::InvalidKernelIr(
                "Rader scatter Stockham output requires CPU auxiliary prime input",
            ))?;
            if auxiliary.len() != expected_output {
                return Err(VkFftError::InputLengthMismatch {
                    expected: expected_output,
                    actual: auxiliary.len(),
                });
            }
            Some((
                mapping,
                mapping.permutation(kernel.sequence_len)?,
                auxiliary,
            ))
        }
    };

    let mut output = vec![Complex64::new(0.0, 0.0); expected_output];
    let sign = match kernel.direction {
        Direction::Forward => -1.0,
        Direction::Inverse => 1.0,
    };

    let normalize = kernel
        .operations
        .iter()
        .find_map(|operation| match operation {
            KernelOperation::StoreSharedToGlobal { normalize, .. } => Some(*normalize),
            _ => None,
        });
    let normalize = normalize.ok_or(VkFftError::InvalidKernelIr(
        "missing Stockham store operation",
    ))?;

    let register_stages = kernel.register_stockham_stages()?;
    let fallback_stages = kernel
        .operations
        .iter()
        .filter_map(|operation| match operation {
            KernelOperation::StockhamStage(stage) => Some(*stage),
            _ => None,
        })
        .collect::<Vec<_>>();
    let math_stages = if let Some(stages) = &register_stages {
        stages
            .iter()
            .map(|stage| StockhamStage {
                index: stage.index,
                radix: stage.radix,
                stage_size: stage.stage_size,
                butterflies: stage.butterflies,
                input: stage.input,
                output: stage.output,
            })
            .collect::<Vec<_>>()
    } else {
        fallback_stages
    };

    for batch in 0..kernel.batch_count {
        let mut source = (0..kernel.sequence_len)
            .map(|local_index| {
                let mut value =
                    kernel
                        .io_mapping
                        .input_value(kernel.sequence_len, batch, local_index, input);
                if let Some(lookup) = lookup {
                    value *= lookup[local_index];
                }
                value
            })
            .collect::<Vec<_>>();

        if kernel.execution_layout == StockhamExecutionLayout::RegisterBoostSingleShared {
            let stages = register_stages.as_ref().ok_or(VkFftError::InvalidKernelIr(
                "boosted Stockham execution requires register stages",
            ))?;
            source = execute_register_boost_stockham_batch(kernel, stages, &source, sign)?;
        } else if kernel.execution_layout == StockhamExecutionLayout::RegisterSingleShared {
            let stages = register_stages.as_ref().ok_or(VkFftError::InvalidKernelIr(
                "single-shared Stockham execution requires register stages",
            ))?;
            source = execute_register_single_shared_stockham_batch(kernel, stages, &source, sign)?;
        } else {
            for stage in &math_stages {
                let radix = stage.radix;
                let butterflies = stage.butterflies;
                let mut destination = vec![Complex64::new(0.0, 0.0); kernel.sequence_len];
                let denominator = (stage.stage_size * radix) as f64;

                for butterfly in 0..butterflies {
                    let stage_invocation = butterfly % stage.stage_size;
                    let mut values = vec![Complex64::new(0.0, 0.0); radix];
                    for (lane, value) in values.iter_mut().enumerate() {
                        let input_index = butterfly + lane * butterflies;
                        let twiddle_angle =
                            sign * TAU * (stage_invocation * lane) as f64 / denominator;
                        *value = source[input_index] * Complex64::exp_i(twiddle_angle);
                    }

                    let results = if register_stages.is_some() {
                        scheduled_radix_dft(&values, sign)
                    } else {
                        None
                    }
                    .unwrap_or_else(|| {
                        (0..radix)
                            .map(|output_lane| {
                                values.iter().enumerate().fold(
                                    Complex64::new(0.0, 0.0),
                                    |sum, (input_lane, &value)| {
                                        let angle = sign * TAU * (output_lane * input_lane) as f64
                                            / radix as f64;
                                        sum + value * Complex64::exp_i(angle)
                                    },
                                )
                            })
                            .collect::<Vec<_>>()
                    });
                    for (output_lane, sum) in results.into_iter().enumerate() {
                        let output_index = stage_invocation
                            + (butterfly - stage_invocation) * radix
                            + output_lane * stage.stage_size;
                        destination[output_index] = sum;
                    }
                }
                source = destination;
            }
        }

        if normalize {
            let scale = 1.0 / kernel.sequence_len as f64;
            for value in &mut source {
                *value = value.scale(scale);
            }
        }
        if let StockhamOutputModifier::RealEvenPostprocess(mapping) = kernel.output_modifier {
            let output_base = batch * (kernel.sequence_len + 1);
            for k in 0..=kernel.sequence_len {
                let a = source[k % kernel.sequence_len];
                let b = source[(kernel.sequence_len - k) % kernel.sequence_len].conj();
                let w = Complex64::exp_i(-TAU * k as f64 / mapping.full_len as f64);
                let rotated = w * (a - b);
                let minus_i_rotated = Complex64::new(rotated.im, -rotated.re);
                output[output_base + k] = (a + b + minus_i_rotated).scale(0.5);
            }
        } else if let StockhamOutputModifier::RealEvenUnpack(mapping) = kernel.output_modifier {
            let output_base = batch * mapping.full_len;
            for (local_index, value) in source.into_iter().enumerate() {
                output[output_base + 2 * local_index] = Complex64::new(value.re, 0.0);
                output[output_base + 2 * local_index + 1] = Complex64::new(value.im, 0.0);
            }
        } else if let Some((mapping, permutation, auxiliary)) = &rader_scatter {
            let prime_base = batch * mapping.prime;
            let scale = if mapping.normalize_prime {
                1.0 / mapping.prime as f64
            } else {
                1.0
            };
            let dc = (0..mapping.prime)
                .map(|prime_local_index| {
                    auxiliary[mapping.auxiliary_input_index(batch, prime_local_index)]
                })
                .fold(Complex64::new(0.0, 0.0), |sum, value| sum + value);
            output[prime_base] = dc.scale(scale);
            let x0 = auxiliary[mapping.auxiliary_input_index(batch, 0)];
            for (local_index, value) in source.into_iter().enumerate() {
                output[prime_base + permutation[local_index]] = (x0 + value).scale(scale);
            }
        } else {
            for (local_index, value) in source.into_iter().enumerate() {
                let (output_index, value) = kernel.io_mapping.map_output(
                    kernel.sequence_len,
                    batch,
                    local_index,
                    value,
                    sign,
                );
                output[output_index] = value;
            }
        }
    }

    Ok(output)
}

#[cfg(test)]
fn synthetic_register_kernel(
    length: usize,
    stage_radices: Vec<usize>,
    registers_per_thread: usize,
    direction: Direction,
    precision: Precision,
    normalize_inverse: bool,
    device: DeviceProfile,
) -> Result<KernelIr> {
    let plan = FftPlan::build(
        crate::FftConfig::new(vec![length])
            .with_precision(precision)
            .with_inverse_normalization(normalize_inverse),
    )?;
    let mut kernel = KernelIr::stockham_1d(&plan, direction, device)?;
    let mut registers = [0usize; crate::scheduler::VKFFT_RADIX_TABLE_LEN];
    let mut multipliers = [0usize; crate::scheduler::VKFFT_RADIX_TABLE_LEN];
    for &radix in &stage_radices {
        registers[radix] = registers_per_thread;
        multipliers[radix] += 1;
    }
    let max_non_power_of_two_radix = stage_radices
        .iter()
        .copied()
        .filter(|radix| !radix.is_power_of_two())
        .max()
        .unwrap_or(1);
    let schedule = RadixRegisterSchedule {
        fft_len: length,
        rhs_transform_count: 1,
        register_boost: 1,
        registers_per_thread_per_radix: registers,
        stage_radix_multipliers: multipliers,
        stage_radices,
        register_boost_stage_radix: None,
        registers_per_thread,
        min_registers_per_thread: registers_per_thread,
        is_good_sequence: true,
        max_non_power_of_two_radix,
        required_local_registers: 1,
    };
    schedule.validate()?;
    let shared_layout =
        plan_gpu_stockham_shared_memory_layout(length, length, kernel.scalar, device)?;
    kernel.scheduler_hint = Some(schedule);
    kernel.execution_layout = StockhamExecutionLayout::RegisterSingleShared;
    kernel.stockham_shared_layout = Some(shared_layout);
    kernel.shared_memory = SharedMemoryPlan {
        elements_per_buffer: shared_layout.allocated_elements,
        buffers: 1,
        scalar: kernel.scalar,
    };
    kernel.workgroup_size = WorkgroupSize { x: 1, y: 1, z: 1 };
    kernel.workgroup_grouping = StockhamWorkgroupGrouping::single(1);
    kernel.validate()?;
    Ok(kernel)
}

#[cfg(test)]
pub(crate) fn synthetic_odd_radix_register_kernel(
    radix: usize,
    direction: Direction,
    precision: Precision,
    normalize_inverse: bool,
    device: DeviceProfile,
) -> Result<KernelIr> {
    if !matches!(radix, 3 | 5 | 7 | 11 | 13) {
        return Err(VkFftError::InvalidKernelIr(
            "synthetic odd-radix kernel only covers radix 3/5/7/11/13",
        ));
    }
    synthetic_register_kernel(
        radix,
        vec![radix],
        radix,
        direction,
        precision,
        normalize_inverse,
        device,
    )
}

#[cfg(test)]
pub(crate) fn synthetic_composite_radix_register_kernel(
    radix: usize,
    direction: Direction,
    precision: Precision,
    normalize_inverse: bool,
    device: DeviceProfile,
) -> Result<KernelIr> {
    if !matches!(radix, 6 | 9 | 10 | 12 | 14 | 15) {
        return Err(VkFftError::InvalidKernelIr(
            "synthetic composite-radix kernel only covers radix 6/9/10/12/14/15",
        ));
    }
    synthetic_register_kernel(
        radix,
        vec![radix],
        radix,
        direction,
        precision,
        normalize_inverse,
        device,
    )
}

#[cfg(test)]
pub(crate) fn synthetic_radix16_register_kernel(
    direction: Direction,
    precision: Precision,
    normalize_inverse: bool,
    device: DeviceProfile,
) -> Result<KernelIr> {
    synthetic_register_kernel(
        16,
        vec![16],
        16,
        direction,
        precision,
        normalize_inverse,
        device,
    )
}

#[cfg(test)]
pub(crate) fn synthetic_radix32_register_kernel(
    direction: Direction,
    precision: Precision,
    normalize_inverse: bool,
    device: DeviceProfile,
) -> Result<KernelIr> {
    synthetic_register_kernel(
        32,
        vec![32],
        32,
        direction,
        precision,
        normalize_inverse,
        device,
    )
}

#[cfg(test)]
pub(crate) fn synthetic_boost_radix32_register_kernel(
    direction: Direction,
    precision: Precision,
    normalize_inverse: bool,
    device: DeviceProfile,
) -> Result<KernelIr> {
    let length = 128usize;
    let plan = FftPlan::build(
        crate::FftConfig::new(vec![length])
            .with_precision(precision)
            .with_inverse_normalization(normalize_inverse),
    )?;
    let mut kernel = KernelIr::stockham_1d(&plan, direction, device)?;
    let mut registers = [0usize; crate::scheduler::VKFFT_RADIX_TABLE_LEN];
    registers[32] = 32;
    registers[4] = 32;
    let mut multipliers = [0usize; crate::scheduler::VKFFT_RADIX_TABLE_LEN];
    multipliers[32] = 1;
    let schedule = RadixRegisterSchedule {
        fft_len: length,
        rhs_transform_count: 1,
        register_boost: 4,
        registers_per_thread_per_radix: registers,
        stage_radix_multipliers: multipliers,
        stage_radices: vec![32, 4],
        register_boost_stage_radix: Some(4),
        registers_per_thread: 32,
        min_registers_per_thread: 32,
        is_good_sequence: true,
        max_non_power_of_two_radix: 1,
        required_local_registers: 1,
    };
    schedule.validate()?;
    let shared_layout = plan_gpu_stockham_shared_memory_layout(length, 32, kernel.scalar, device)?;
    kernel.scheduler_hint = Some(schedule);
    kernel.execution_layout = StockhamExecutionLayout::RegisterBoostSingleShared;
    kernel.stockham_shared_layout = Some(shared_layout);
    kernel.shared_memory = SharedMemoryPlan {
        elements_per_buffer: shared_layout.allocated_elements,
        buffers: 1,
        scalar: kernel.scalar,
    };
    kernel.workgroup_size = WorkgroupSize { x: 1, y: 1, z: 1 };
    kernel.workgroup_grouping = StockhamWorkgroupGrouping::single(1);
    kernel.validate()?;
    Ok(kernel)
}

#[cfg(test)]
pub(crate) fn synthetic_local_permutation_register_kernel(
    direction: Direction,
    precision: Precision,
    normalize_inverse: bool,
    device: DeviceProfile,
) -> Result<KernelIr> {
    synthetic_register_kernel(
        16,
        vec![4, 4],
        16,
        direction,
        precision,
        normalize_inverse,
        device,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{Backend, GpuVendor};
    use crate::reference::{dft, fft};

    fn device() -> DeviceProfile {
        DeviceProfile {
            shared_memory_bytes: 128 * 1024,
            shared_memory_pow2_bytes: 128 * 1024,
            ..DeviceProfile::generic(Backend::Vulkan, GpuVendor::Nvidia)
        }
    }

    fn sample(length: usize, batch: usize) -> Vec<Complex64> {
        (0..length * batch)
            .map(|index| {
                let x = index as f64;
                Complex64::new((0.17 * x).sin() + x * 0.003, (0.11 * x).cos() - x * 0.004)
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
    fn amd_vulkan_stockham_consumes_vendor_policy_and_f64_lut() {
        let amd_48k = DeviceProfile {
            shared_memory_bytes: 48 * 1024,
            shared_memory_pow2_bytes: 32 * 1024,
            max_threads_per_block: 1024,
            ..DeviceProfile::generic(Backend::Vulkan, GpuVendor::Amd)
        };
        let plan = FftPlan::build(crate::FftConfig::new(vec![16_384])).unwrap();
        let kernel = KernelIr::stockham_1d(&plan, Direction::Forward, amd_48k).unwrap();
        assert_eq!(kernel.scheduler_hint.as_ref().unwrap().register_boost, 4);
        assert_eq!(
            kernel.execution_layout,
            StockhamExecutionLayout::RegisterBoostSingleShared
        );
        assert!(kernel.required_shared_memory_bytes().unwrap() <= amd_48k.shared_memory_bytes);

        let amd_64k = DeviceProfile {
            shared_memory_bytes: 64 * 1024,
            shared_memory_pow2_bytes: 64 * 1024,
            supports_f64: true,
            ..amd_48k
        };
        let kernel = KernelIr::stockham_1d(&plan, Direction::Forward, amd_64k).unwrap();
        assert_eq!(kernel.scheduler_hint.as_ref().unwrap().register_boost, 2);

        let f64_plan =
            FftPlan::build(crate::FftConfig::new(vec![256]).with_precision(Precision::F64))
                .unwrap();
        let f64 = KernelIr::stockham_1d(&f64_plan, Direction::Forward, amd_64k).unwrap();
        assert_eq!(f64.twiddle_source, StockhamTwiddleSource::LookupTable);
        assert!(
            f64.bindings
                .iter()
                .any(|binding| binding.role == BufferRole::TwiddleLookupTable)
        );
        let input = sample(256, 1);
        let actual = execute_stockham_ir(&f64, &input).unwrap();
        let expected = fft(&input, Direction::Forward, false).unwrap();
        assert!(max_error(&actual, &expected) <= 3.0e-9 * 256.0);
    }

    #[test]
    fn unknown_vulkan_vendor_keeps_portable_stockham_fallback() {
        let device = DeviceProfile {
            shared_memory_bytes: 128 * 1024,
            shared_memory_pow2_bytes: 128 * 1024,
            max_threads_per_block: 1024,
            ..DeviceProfile::generic(Backend::Vulkan, GpuVendor::Other(0x1234))
        };
        let plan = FftPlan::build(crate::FftConfig::new(vec![60])).unwrap();
        let kernel = KernelIr::stockham_1d(&plan, Direction::Forward, device).unwrap();
        assert!(kernel.scheduler_hint.is_none());
        assert_eq!(
            kernel.execution_layout,
            StockhamExecutionLayout::SharedPingPong
        );
        assert_eq!(kernel.twiddle_source, StockhamTwiddleSource::OnTheFly);
    }

    #[test]
    fn specialized_odd_radix_register_stages_match_dft_and_round_trip() {
        let profile = device();
        for radix in [3usize, 5, 7, 11, 13] {
            let input = sample(radix, 1);
            let forward = synthetic_odd_radix_register_kernel(
                radix,
                Direction::Forward,
                Precision::F32,
                false,
                profile,
            )
            .unwrap();
            let stages = forward.register_stockham_stages().unwrap().unwrap();
            assert_eq!(stages.len(), 1);
            assert_eq!(stages[0].radix, radix);
            let actual = execute_stockham_ir(&forward, &input).unwrap();
            let expected = dft(&input, Direction::Forward, false);
            assert!(
                max_error(&actual, &expected) < 3.0e-10,
                "radix {radix} fixed butterfly mismatch"
            );

            let inverse = synthetic_odd_radix_register_kernel(
                radix,
                Direction::Inverse,
                Precision::F32,
                true,
                profile,
            )
            .unwrap();
            let restored = execute_stockham_ir(&inverse, &actual).unwrap();
            assert!(
                max_error(&restored, &input) < 5.0e-10,
                "radix {radix} fixed butterfly round-trip mismatch"
            );
        }
    }

    #[test]
    fn specialized_composite_radix_register_stages_match_dft_and_round_trip() {
        let profile = device();
        for radix in [6usize, 9, 10, 12, 14, 15] {
            let input = sample(radix, 1);
            let forward = synthetic_composite_radix_register_kernel(
                radix,
                Direction::Forward,
                Precision::F32,
                false,
                profile,
            )
            .unwrap();
            let stages = forward.register_stockham_stages().unwrap().unwrap();
            assert_eq!(stages.len(), 1);
            assert_eq!(stages[0].radix, radix);
            let actual = execute_stockham_ir(&forward, &input).unwrap();
            let expected = dft(&input, Direction::Forward, false);
            assert!(
                max_error(&actual, &expected) < 5.0e-10,
                "composite radix {radix} fixed Cooley-Tukey mismatch"
            );

            let inverse = synthetic_composite_radix_register_kernel(
                radix,
                Direction::Inverse,
                Precision::F32,
                true,
                profile,
            )
            .unwrap();
            let restored = execute_stockham_ir(&inverse, &actual).unwrap();
            assert!(
                max_error(&restored, &input) < 8.0e-10,
                "composite radix {radix} fixed Cooley-Tukey round-trip mismatch"
            );
        }
    }

    #[test]
    fn specialized_radix16_register_stage_matches_dft_and_round_trips() {
        let mut profile = device();
        profile.max_threads_per_block = 1024;
        let input = sample(16, 1);
        let forward =
            synthetic_radix16_register_kernel(Direction::Forward, Precision::F32, false, profile)
                .unwrap();
        let stages = forward.register_stockham_stages().unwrap().unwrap();
        assert_eq!(stages.len(), 1);
        assert_eq!(stages[0].radix, 16);
        assert_eq!(stages[0].registers_per_thread, 16);
        let actual = execute_stockham_ir(&forward, &input).unwrap();
        let expected = dft(&input, Direction::Forward, false);
        assert!(max_error(&actual, &expected) < 2.0e-10);

        let inverse =
            synthetic_radix16_register_kernel(Direction::Inverse, Precision::F32, true, profile)
                .unwrap();
        let restored = execute_stockham_ir(&inverse, &actual).unwrap();
        assert!(max_error(&restored, &input) < 3.0e-10);
    }

    #[test]
    fn specialized_radix32_register_stage_matches_dft_and_round_trips() {
        let mut profile = device();
        profile.max_threads_per_block = 1024;
        let input = sample(32, 1);
        let forward =
            synthetic_radix32_register_kernel(Direction::Forward, Precision::F32, false, profile)
                .unwrap();
        let stages = forward.register_stockham_stages().unwrap().unwrap();
        assert_eq!(stages.len(), 1);
        assert_eq!(stages[0].radix, 32);
        assert_eq!(stages[0].registers_per_thread, 32);
        let actual = execute_stockham_ir(&forward, &input).unwrap();
        let expected = dft(&input, Direction::Forward, false);
        assert!(max_error(&actual, &expected) < 5.0e-10);

        let inverse =
            synthetic_radix32_register_kernel(Direction::Inverse, Precision::F32, true, profile)
                .unwrap();
        let restored = execute_stockham_ir(&inverse, &actual).unwrap();
        assert!(max_error(&restored, &input) < 8.0e-10);
    }

    #[test]
    fn boosted_radix32_stage_matches_dft_and_round_trips() {
        let profile = device();
        let input = sample(128, 1);
        let forward = synthetic_boost_radix32_register_kernel(
            Direction::Forward,
            Precision::F32,
            false,
            profile,
        )
        .unwrap();
        assert_eq!(
            forward.execution_layout,
            StockhamExecutionLayout::RegisterBoostSingleShared
        );
        let stages = forward.register_stockham_stages().unwrap().unwrap();
        assert_eq!(
            stages.iter().map(|stage| stage.radix).collect::<Vec<_>>(),
            vec![32, 4]
        );
        let actual = execute_stockham_ir(&forward, &input).unwrap();
        let expected = dft(&input, Direction::Forward, false);
        assert!(max_error(&actual, &expected) < 3.0e-9);

        let inverse = synthetic_boost_radix32_register_kernel(
            Direction::Inverse,
            Precision::F32,
            true,
            profile,
        )
        .unwrap();
        let restored = execute_stockham_ir(&inverse, &actual).unwrap();
        assert!(max_error(&restored, &input) < 5.0e-9);
    }

    #[test]
    fn real_scheduler_register_boost_two_and_four_shapes_execute() {
        let cases = [
            (Precision::F32, 16usize, 16usize, 4096usize, 2usize, false),
            (Precision::F32, 16, 16, 8192, 4, false),
            (Precision::F32, 48, 32, 8192, 2, true),
            (Precision::F32, 48, 32, 16384, 4, true),
            (Precision::F64, 16, 16, 2048, 2, false),
            (Precision::F64, 16, 16, 4096, 4, false),
            (Precision::F64, 48, 32, 4096, 2, true),
            (Precision::F64, 48, 32, 8192, 4, true),
        ];
        for (precision, shared_kib, pow2_kib, length, boost, padded) in cases {
            let mut profile = device();
            profile.shared_memory_bytes = shared_kib * 1024;
            profile.shared_memory_pow2_bytes = pow2_kib * 1024;
            profile.max_threads_per_block = 1024;
            profile.supports_f64 = true;
            let plan =
                FftPlan::build(crate::FftConfig::new(vec![length]).with_precision(precision))
                    .unwrap();
            let forward = KernelIr::stockham_1d(&plan, Direction::Forward, profile).unwrap();
            assert_eq!(
                forward.execution_layout,
                StockhamExecutionLayout::RegisterBoostSingleShared,
                "precision={precision:?} shared={shared_kib} KiB length={length}"
            );
            let schedule = forward.scheduler_hint.as_ref().unwrap();
            assert_eq!(schedule.register_boost, boost);
            assert_eq!(schedule.stage_radices.last().copied(), Some(boost));
            let shared = forward.stockham_shared_layout.unwrap();
            assert_eq!(shared.logical_elements, length / boost);
            assert_eq!(shared.allocated_elements > shared.logical_elements, padded);

            let input = sample(length, 1);
            let actual = execute_stockham_ir(&forward, &input).unwrap();
            let expected = fft(&input, Direction::Forward, false).unwrap();
            assert!(
                max_error(&actual, &expected) <= 3.0e-9 * length.ilog2() as f64,
                "boost-{boost} forward mismatch for {precision:?} length {length}"
            );

            let inverse_plan = FftPlan::build(
                crate::FftConfig::new(vec![length])
                    .with_precision(precision)
                    .with_inverse_normalization(true),
            )
            .unwrap();
            let inverse =
                KernelIr::stockham_1d(&inverse_plan, Direction::Inverse, profile).unwrap();
            let restored = execute_stockham_ir(&inverse, &actual).unwrap();
            assert!(
                max_error(&restored, &input) <= 8.0e-9 * length.ilog2() as f64,
                "boost-{boost} inverse mismatch for {precision:?} length {length}"
            );
        }
    }

    #[test]
    fn local_register_permutation_boundary_is_proven_and_executes_without_shared_semantics() {
        let profile = device();
        let input = sample(16, 1);
        let forward = synthetic_local_permutation_register_kernel(
            Direction::Forward,
            Precision::F32,
            false,
            profile,
        )
        .unwrap();
        let boundaries = forward.register_stage_boundaries().unwrap().unwrap();
        assert_eq!(boundaries.len(), 1);
        assert_eq!(
            boundaries[0].residency,
            RegisterStageBoundaryResidency::RegisterResidentPermutation
        );
        assert_eq!(
            boundaries[0].target_to_source_registers.as_deref(),
            Some(&[0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15][..])
        );
        let actual = execute_stockham_ir(&forward, &input).unwrap();
        let expected = dft(&input, Direction::Forward, false);
        assert!(max_error(&actual, &expected) < 3.0e-10);

        let inverse = synthetic_local_permutation_register_kernel(
            Direction::Inverse,
            Precision::F32,
            true,
            profile,
        )
        .unwrap();
        let restored = execute_stockham_ir(&inverse, &actual).unwrap();
        assert!(max_error(&restored, &input) < 5.0e-10);
    }

    #[test]
    fn register_boundary_metadata_keeps_current_1024_shared_exchanges() {
        let plan = FftPlan::build(crate::FftConfig::new(vec![1024])).unwrap();
        let kernel = KernelIr::stockham_1d(&plan, Direction::Forward, device()).unwrap();
        let boundaries = kernel.register_stage_boundaries().unwrap().unwrap();
        assert_eq!(boundaries.len(), 3);
        assert!(boundaries.iter().all(|boundary| {
            boundary.residency == RegisterStageBoundaryResidency::SharedExchangeRequired
        }));
        assert_eq!(boundaries[0].source_stage, 0);
        assert_eq!(boundaries[0].target_stage, 1);
    }

    #[test]
    fn real_scheduler_subgroup_proof_finds_small_warp_local_boundaries_only() {
        let subgroup_size = 32usize;
        let plan64 = FftPlan::build(crate::FftConfig::new(vec![64])).unwrap();
        let kernel64 = KernelIr::stockham_1d(&plan64, Direction::Forward, device()).unwrap();
        let stages64 = kernel64.register_stockham_stages().unwrap().unwrap();
        assert!(stages64.len() >= 2);
        assert!(kernel64.workgroup_size.x as usize <= subgroup_size);
        let invocation64 = kernel64.register_stage_boundaries().unwrap().unwrap();
        assert!(invocation64.iter().all(|boundary| {
            boundary.residency == RegisterStageBoundaryResidency::SharedExchangeRequired
        }));
        let subgroup64 = kernel64
            .register_subgroup_boundaries(subgroup_size)
            .unwrap()
            .unwrap();
        assert_eq!(subgroup64.len(), stages64.len() - 1);
        assert!(subgroup64.iter().all(Option::is_some));
        for proof in subgroup64.into_iter().flatten() {
            assert_eq!(proof.subgroup_size, subgroup_size);
            assert_eq!(proof.active_lanes, kernel64.workgroup_size.x as usize);
            assert_eq!(
                proof.lane_model,
                RegisterSubgroupLaneModel::ContiguousLocalInvocation
            );
            for slot in proof.target_to_source {
                assert_eq!(slot.len(), proof.active_lanes);
                assert!(slot.iter().all(|source| {
                    source.lane < proof.active_lanes
                        && source.register < stages64[proof.source_stage].registers_per_thread
                }));
            }
        }

        for (length, expected_proofs) in [
            (16usize, 1usize),
            (32, 1),
            (64, 1),
            (128, 2),
            (256, 2),
            (512, 0),
            (1024, 0),
            (2048, 0),
            (4096, 0),
        ] {
            let plan = FftPlan::build(crate::FftConfig::new(vec![length])).unwrap();
            let kernel = KernelIr::stockham_1d(&plan, Direction::Forward, device()).unwrap();
            let proofs = kernel
                .register_subgroup_boundaries(subgroup_size)
                .unwrap()
                .unwrap_or_default();
            assert_eq!(
                proofs.iter().filter(|proof| proof.is_some()).count(),
                expected_proofs,
                "unexpected modeled subgroup-local boundary count for N={length}"
            );
        }

        let plan1024 = FftPlan::build(crate::FftConfig::new(vec![1024])).unwrap();
        let kernel1024 = KernelIr::stockham_1d(&plan1024, Direction::Forward, device()).unwrap();
        assert!(kernel1024.workgroup_size.x as usize > subgroup_size);
        let subgroup1024 = kernel1024
            .register_subgroup_boundaries(subgroup_size)
            .unwrap()
            .unwrap();
        assert!(subgroup1024.iter().all(Option::is_none));
    }

    #[test]
    fn stockham_ir_matches_dft_for_mixed_radices() {
        for length in [2usize, 3, 4, 6, 8, 12, 15, 16, 30, 32, 60, 77] {
            let plan = FftPlan::build(crate::FftConfig::new(vec![length])).unwrap();
            let kernel = KernelIr::stockham_1d(&plan, Direction::Forward, device()).unwrap();
            let input = sample(length, 1);
            let actual = execute_stockham_ir(&kernel, &input).unwrap();
            let expected = dft(&input, Direction::Forward, false);
            assert!(
                max_error(&actual, &expected) <= 2.0e-10 * length as f64,
                "length {length} did not match DFT"
            );
        }
    }

    #[test]
    fn all_small_stockham_lengths_match_dft() {
        for length in 1usize..=128 {
            let plan = FftPlan::build(crate::FftConfig::new(vec![length])).unwrap();
            if !matches!(plan.axes[0].algorithm, AxisAlgorithm::Stockham { .. }) {
                continue;
            }
            let kernel = KernelIr::stockham_1d(&plan, Direction::Forward, device()).unwrap();
            let input = sample(length, 1);
            let actual = execute_stockham_ir(&kernel, &input).unwrap();
            let expected = dft(&input, Direction::Forward, false);
            assert!(
                max_error(&actual, &expected) <= 3.0e-10 * length as f64,
                "length {length} did not match DFT"
            );
        }
    }

    #[test]
    fn batched_inverse_normalization_round_trips() {
        let length = 30usize;
        let batch_count = 3usize;
        let forward_plan =
            FftPlan::build(crate::FftConfig::new(vec![length]).with_batch_count(batch_count))
                .unwrap();
        let inverse_plan = FftPlan::build(
            crate::FftConfig::new(vec![length])
                .with_batch_count(batch_count)
                .with_inverse_normalization(true),
        )
        .unwrap();
        let forward = KernelIr::stockham_1d(&forward_plan, Direction::Forward, device()).unwrap();
        let inverse = KernelIr::stockham_1d(&inverse_plan, Direction::Inverse, device()).unwrap();
        let input = sample(length, batch_count);
        let spectrum = execute_stockham_ir(&forward, &input).unwrap();
        let restored = execute_stockham_ir(&inverse, &spectrum).unwrap();
        assert!(max_error(&restored, &input) < 5.0e-10 * length as f64);
    }

    #[test]
    fn four_step_mapped_stockham_uploads_match_full_fft() {
        let left_len = 8usize;
        let right_len = 8usize;
        let logical_len = left_len * right_len;
        let batch_count = 2usize;
        let mapping = FourStepMapping {
            logical_len,
            left_len,
            right_len,
            outer_batch_count: batch_count,
        };

        let build_uploads = |direction: Direction, normalize: bool| {
            let right_plan = FftPlan::build(
                crate::FftConfig::new(vec![right_len])
                    .with_batch_count(batch_count * left_len)
                    .with_inverse_normalization(normalize),
            )
            .unwrap();
            let right = KernelIr::stockham_1d(&right_plan, direction, device())
                .unwrap()
                .with_stockham_io_mapping(StockhamIoMapping::FourStepRight(mapping))
                .unwrap();
            let left_plan = FftPlan::build(
                crate::FftConfig::new(vec![left_len])
                    .with_batch_count(batch_count * right_len)
                    .with_inverse_normalization(normalize),
            )
            .unwrap();
            let left = KernelIr::stockham_1d(&left_plan, direction, device())
                .unwrap()
                .with_stockham_io_mapping(StockhamIoMapping::FourStepLeft(mapping))
                .unwrap();
            (right, left)
        };

        let input = sample(logical_len, batch_count);
        let (right, left) = build_uploads(Direction::Forward, false);
        let intermediate = execute_stockham_ir(&right, &input).unwrap();
        let spectrum = execute_stockham_ir(&left, &intermediate).unwrap();
        for batch in 0..batch_count {
            let start = batch * logical_len;
            let expected = fft(
                &input[start..start + logical_len],
                Direction::Forward,
                false,
            )
            .unwrap();
            assert!(max_error(&spectrum[start..start + logical_len], &expected) < 2.0e-9);
        }

        let (right_inverse, left_inverse) = build_uploads(Direction::Inverse, true);
        let intermediate = execute_stockham_ir(&right_inverse, &spectrum).unwrap();
        let restored = execute_stockham_ir(&left_inverse, &intermediate).unwrap();
        assert!(max_error(&restored, &input) < 2.0e-9);
    }

    #[test]
    fn cooley_left_mapping_matches_explicit_twiddle_left_fft_and_scatter() {
        let parent_left_len = 3usize;
        let parent_right_len = 5usize;
        let parent_logical_len = parent_left_len * parent_right_len;
        let parent_batch_count = 2usize;
        let child_batch_count = parent_batch_count * parent_right_len;
        let right_output = sample(parent_logical_len, parent_batch_count);
        let outer = FourStepMapping {
            logical_len: parent_logical_len * parent_batch_count,
            left_len: parent_logical_len,
            right_len: parent_batch_count,
            outer_batch_count: 1,
        };

        for outer_four_step in [None, Some(outer)] {
            let plan = FftPlan::build(
                crate::FftConfig::new(vec![parent_left_len]).with_batch_count(child_batch_count),
            )
            .unwrap();
            let mapping = CooleyLeftStockhamMapping {
                parent_logical_len,
                parent_left_len,
                parent_right_len,
                parent_batch_count,
                direction: Direction::Forward,
                outer_four_step,
            };
            let kernel = KernelIr::stockham_1d(&plan, Direction::Forward, device())
                .unwrap()
                .with_stockham_io_mapping(StockhamIoMapping::CooleyLeft(mapping))
                .unwrap();
            let actual = execute_stockham_ir(&kernel, &right_output).unwrap();
            let mut expected = vec![Complex64::new(0.0, 0.0); right_output.len()];
            for parent_batch in 0..parent_batch_count {
                for k2 in 0..parent_right_len {
                    let mut left_input = Vec::with_capacity(parent_left_len);
                    for n1 in 0..parent_left_len {
                        let source_index =
                            (parent_batch * parent_left_len + n1) * parent_right_len + k2;
                        let angle = -TAU * (n1 * k2) as f64 / parent_logical_len as f64;
                        left_input.push(right_output[source_index] * Complex64::exp_i(angle));
                    }
                    let left_output = fft(&left_input, Direction::Forward, false).unwrap();
                    for (k1, value) in left_output.into_iter().enumerate() {
                        let parent_local_index = k2 + parent_right_len * k1;
                        let output_index = match outer_four_step {
                            None => parent_batch * parent_logical_len + parent_local_index,
                            Some(outer) => {
                                let outer_batch = parent_batch / outer.right_len;
                                let outer_k2 = parent_batch % outer.right_len;
                                outer_batch * outer.logical_len
                                    + outer_k2
                                    + outer.right_len * parent_local_index
                            }
                        };
                        expected[output_index] = value;
                    }
                }
            }
            assert!(max_error(&actual, &expected) < 2.0e-11);
        }
    }

    #[test]
    fn three_upload_four_step_mappings_match_full_fft() {
        let axis_split = [4usize, 2, 4];
        let [a, b, c] = axis_split;
        let logical_len = a * b * c;
        let batch_count = 2usize;
        let mapping = ThreeUploadFourStepMapping {
            logical_len,
            axis_split,
            outer_batch_count: batch_count,
        };
        mapping.validate().unwrap();

        let build_uploads = |direction: Direction, normalize: bool| {
            let upload2_plan = FftPlan::build(
                crate::FftConfig::new(vec![c])
                    .with_batch_count(batch_count * a * b)
                    .with_inverse_normalization(normalize),
            )
            .unwrap();
            let upload2 = KernelIr::stockham_1d(&upload2_plan, direction, device())
                .unwrap()
                .with_stockham_io_mapping(StockhamIoMapping::FourStepThreeUpload2(mapping))
                .unwrap();
            let upload1_plan = FftPlan::build(
                crate::FftConfig::new(vec![b])
                    .with_batch_count(batch_count * c * a)
                    .with_inverse_normalization(normalize),
            )
            .unwrap();
            let upload1 = KernelIr::stockham_1d(&upload1_plan, direction, device())
                .unwrap()
                .with_stockham_io_mapping(StockhamIoMapping::FourStepThreeUpload1(mapping))
                .unwrap();
            let upload0_plan = FftPlan::build(
                crate::FftConfig::new(vec![a])
                    .with_batch_count(batch_count * c * b)
                    .with_inverse_normalization(normalize),
            )
            .unwrap();
            let upload0 = KernelIr::stockham_1d(&upload0_plan, direction, device())
                .unwrap()
                .with_stockham_io_mapping(StockhamIoMapping::FourStepThreeUpload0(mapping))
                .unwrap();
            [upload2, upload1, upload0]
        };

        let input = sample(logical_len, batch_count);
        let forward = build_uploads(Direction::Forward, false);
        let stage2 = execute_stockham_ir(&forward[0], &input).unwrap();
        let stage1 = execute_stockham_ir(&forward[1], &stage2).unwrap();
        let spectrum = execute_stockham_ir(&forward[2], &stage1).unwrap();
        for batch in 0..batch_count {
            let start = batch * logical_len;
            let expected = fft(
                &input[start..start + logical_len],
                Direction::Forward,
                false,
            )
            .unwrap();
            assert!(max_error(&spectrum[start..start + logical_len], &expected) < 3.0e-9);
        }

        let inverse = build_uploads(Direction::Inverse, true);
        let stage2 = execute_stockham_ir(&inverse[0], &spectrum).unwrap();
        let stage1 = execute_stockham_ir(&inverse[1], &stage2).unwrap();
        let restored = execute_stockham_ir(&inverse[2], &stage1).unwrap();
        assert!(max_error(&restored, &input) < 4.0e-9);
    }

    #[test]
    fn three_upload_perform_convolution_pre_twiddle_factorization_matches_full_fft() {
        let axis_split = [4usize, 2, 4];
        let [a, b, c] = axis_split;
        let logical_len = a * b * c;
        let batch_count = 2usize;
        let mapping = ThreeUploadFourStepMapping {
            logical_len,
            axis_split,
            outer_batch_count: batch_count,
        };
        mapping.validate().unwrap();

        let build_uploads = |direction: Direction| {
            let upload2_plan = FftPlan::build(
                crate::FftConfig::new(vec![c]).with_batch_count(batch_count * a * b),
            )
            .unwrap();
            let upload2 = KernelIr::stockham_1d(&upload2_plan, direction, device())
                .unwrap()
                .with_stockham_io_mapping(StockhamIoMapping::FourStepThreeUpload2(mapping))
                .unwrap();
            let upload1_plan = FftPlan::build(
                crate::FftConfig::new(vec![b]).with_batch_count(batch_count * c * a),
            )
            .unwrap();
            let upload1 = KernelIr::stockham_1d(&upload1_plan, direction, device())
                .unwrap()
                .with_stockham_io_mapping(StockhamIoMapping::FourStepThreeUpload1(mapping))
                .unwrap();
            let upload0_plan = FftPlan::build(
                crate::FftConfig::new(vec![a]).with_batch_count(batch_count * c * b),
            )
            .unwrap();
            let upload0 = KernelIr::stockham_1d(&upload0_plan, direction, device())
                .unwrap()
                .with_stockham_io_mapping(StockhamIoMapping::FourStepThreeUpload0(mapping))
                .unwrap();
            [upload2, upload1, upload0]
        };

        let input = sample(logical_len, batch_count);
        let kernel_spectrum = (0..logical_len)
            .map(|index| {
                let x = index as f64;
                Complex64::new(0.75 + 0.03 * (0.17 * x).cos(), 0.04 * (0.11 * x).sin())
            })
            .collect::<Vec<_>>();
        let forward = build_uploads(Direction::Forward);
        let stage2 = execute_stockham_ir(&forward[0], &input).unwrap();
        let stage1 = execute_stockham_ir(&forward[1], &stage2).unwrap();
        let full_spectrum = execute_stockham_ir(&forward[2], &stage1).unwrap();

        let mut expected = vec![Complex64::default(); input.len()];
        for outer in 0..batch_count {
            let base = outer * logical_len;
            let multiplied = (0..logical_len)
                .map(|bin| full_spectrum[base + bin] * kernel_spectrum[bin])
                .collect::<Vec<_>>();
            let spatial = fft(&multiplied, Direction::Inverse, true).unwrap();
            expected[base..base + logical_len].copy_from_slice(&spatial);
        }

        // Emulate the embedded upload-0 convolutionStep. Forward upload 0 has already
        // produced natural-frequency bins, so each (k3,k2) A-point line is multiplied
        // in natural order, inverse transformed, normalized by the complete N, and
        // written back to the layout consumed by inverse upload 1.
        let mut special_u0_output = vec![Complex64::default(); input.len()];
        for outer in 0..batch_count {
            let base = outer * logical_len;
            for k3 in 0..c {
                for k2 in 0..b {
                    let local_spectrum = (0..a)
                        .map(|k1| {
                            let full_bin = k3 + c * k2 + c * b * k1;
                            full_spectrum[base + full_bin] * kernel_spectrum[full_bin]
                        })
                        .collect::<Vec<_>>();
                    let local_spatial = fft(&local_spectrum, Direction::Inverse, false).unwrap();
                    for (n1, value) in local_spatial.into_iter().enumerate() {
                        let layout_index = base + a * b * k3 + n1 + a * k2;
                        special_u0_output[layout_index] = value.scale(1.0 / logical_len as f64);
                    }
                }
            }
        }

        let inverse = build_uploads(Direction::Inverse);
        let inverse_u1 = inverse[1]
            .clone()
            .with_stockham_io_mapping(StockhamIoMapping::FourStepThreeUploadPreTwiddle(
                ThreeUploadPreTwiddleMapping {
                    three_upload: mapping,
                    axis_upload_id: 1,
                    direction: Direction::Inverse,
                },
            ))
            .unwrap()
            .with_store_normalization(false)
            .unwrap();
        let inverse_u2 = inverse[0]
            .clone()
            .with_stockham_io_mapping(StockhamIoMapping::FourStepThreeUploadPreTwiddle(
                ThreeUploadPreTwiddleMapping {
                    three_upload: mapping,
                    axis_upload_id: 2,
                    direction: Direction::Inverse,
                },
            ))
            .unwrap()
            .with_store_normalization(false)
            .unwrap();
        let stage1_inverse = execute_stockham_ir(&inverse_u1, &special_u0_output).unwrap();
        let candidate = execute_stockham_ir(&inverse_u2, &stage1_inverse).unwrap();
        assert!(max_error(&candidate, &expected) < 5.0e-9);
    }

    #[test]
    fn rejects_non_stockham_axis_for_now() {
        let plan = FftPlan::build(crate::FftConfig::new(vec![17])).unwrap();
        let error = KernelIr::stockham_1d(&plan, Direction::Forward, device()).unwrap_err();
        assert!(matches!(error, VkFftError::UnsupportedKernelPath(_)));
    }

    #[test]
    fn shared_memory_limit_is_checked() {
        let plan = FftPlan::build(crate::FftConfig::new(vec![4096])).unwrap();
        let error = KernelIr::stockham_1d(
            &plan,
            Direction::Forward,
            DeviceProfile::generic(Backend::Vulkan, GpuVendor::Nvidia),
        )
        .unwrap_err();
        assert!(matches!(error, VkFftError::ResourceLimitExceeded { .. }));
    }

    #[test]
    fn nvidia_power_of_two_kernel_carries_register_scheduler_hint() {
        let batch_count = 2usize;
        let plan = FftPlan::build(crate::FftConfig::new(vec![1024]).with_batch_count(batch_count))
            .unwrap();
        let kernel = KernelIr::stockham_1d(&plan, Direction::Forward, device()).unwrap();
        let hint = kernel.scheduler_hint.as_ref().unwrap();
        assert_eq!(hint.fft_len, 1024);
        assert_eq!(hint.rhs_transform_count, batch_count);
        assert_eq!(hint.register_boost, 1);
        assert_eq!(hint.stage_radices, vec![8, 8, 8, 2]);
        assert_eq!(
            kernel.execution_layout,
            StockhamExecutionLayout::RegisterSingleShared
        );
        assert_eq!(kernel.shared_memory.buffers, 1);
        let shared_layout = kernel.stockham_shared_layout.unwrap();
        assert_eq!(shared_layout.logical_elements, 1024);
        assert_eq!(shared_layout.bank_span_elements, 16);
        assert_eq!(shared_layout.first_stage_stride, 1088);
        assert_eq!(shared_layout.read_write_stride, 1040);
        assert_eq!(shared_layout.allocated_elements, 1088);
        assert!(shared_layout.first_stage_padding_enabled());
        assert_eq!(shared_layout.physical_index(15, true).unwrap(), 15);
        assert_eq!(shared_layout.physical_index(16, true).unwrap(), 17);
        assert_eq!(shared_layout.physical_index(1023, true).unwrap(), 1086);
        assert_eq!(kernel.required_shared_memory_bytes().unwrap(), 1088 * 8);
    }

    #[test]
    fn single_shared_register_layout_unlocks_4096_point_stockham() {
        let length = 4096usize;
        let mut profile = DeviceProfile::generic(Backend::Vulkan, GpuVendor::Nvidia);
        profile.max_threads_per_block = 1024;
        let plan = FftPlan::build(crate::FftConfig::new(vec![length])).unwrap();
        let kernel = KernelIr::stockham_1d(&plan, Direction::Forward, profile).unwrap();
        assert_eq!(
            kernel.execution_layout,
            StockhamExecutionLayout::RegisterSingleShared
        );
        assert_eq!(kernel.shared_memory.buffers, 1);
        assert_eq!(kernel.shared_memory.elements_per_buffer, length);
        assert_eq!(kernel.required_shared_memory_bytes().unwrap(), 32 * 1024);
        let shared_layout = kernel.stockham_shared_layout.unwrap();
        assert_eq!(shared_layout.logical_elements, length);
        assert_eq!(shared_layout.allocated_elements, length);
        assert_eq!(shared_layout.first_stage_stride, length);
        assert_eq!(shared_layout.read_write_stride, length);
        assert!(!shared_layout.first_stage_padding_enabled());
        assert_eq!(kernel.workgroup_size.x, 512);
        let stages = kernel.register_stockham_stages().unwrap().unwrap();
        assert!(stages.iter().all(|stage| {
            stage.input == SharedBuffer::A
                && stage.output == SharedBuffer::A
                && stage.virtual_thread_count == 512
        }));

        let input = sample(length, 1);
        let actual = execute_stockham_ir(&kernel, &input).unwrap();
        let expected = fft(&input, Direction::Forward, false).unwrap();
        assert!(max_error(&actual, &expected) <= 3.0e-9 * length as f64);
    }

    #[test]
    fn register_boost_four_executes_16384_in_32k_shared_memory() {
        let length = 16_384usize;
        let mut profile = DeviceProfile::generic(Backend::Vulkan, GpuVendor::Nvidia);
        profile.shared_memory_bytes = 32 * 1024;
        profile.shared_memory_pow2_bytes = 32 * 1024;
        profile.max_threads_per_block = 1024;

        let plan = FftPlan::build(crate::FftConfig::new(vec![length])).unwrap();
        let kernel = KernelIr::stockham_1d(&plan, Direction::Forward, profile).unwrap();
        let hint = kernel.scheduler_hint.as_ref().unwrap();
        assert_eq!(hint.register_boost, 4);
        assert_eq!(hint.stage_radices, vec![8, 8, 8, 8, 4]);
        assert_eq!(
            kernel.execution_layout,
            StockhamExecutionLayout::RegisterBoostSingleShared
        );
        assert_eq!(kernel.shared_memory.buffers, 1);
        assert_eq!(kernel.shared_memory.elements_per_buffer, 4096);
        assert_eq!(kernel.required_shared_memory_bytes().unwrap(), 32 * 1024);
        let shared_layout = kernel.stockham_shared_layout.unwrap();
        assert_eq!(shared_layout.logical_elements, 4096);
        assert_eq!(shared_layout.allocated_elements, 4096);
        assert_eq!(shared_layout.first_stage_stride, 4096);
        assert_eq!(shared_layout.read_write_stride, 4096);
        assert!(!shared_layout.first_stage_padding_enabled());
        assert_eq!(kernel.workgroup_size.x, 512);
        let stages = kernel.register_stockham_stages().unwrap().unwrap();
        assert!(stages.iter().all(|stage| stage.register_boost == 4));
        assert_eq!(stages.last().unwrap().radix, 4);
        assert!(stages.last().unwrap().is_register_boost_stage);
        assert!(should_skip_boost_exchange(length, &stages[3], &stages[4]));

        let input = sample(length, 1);
        let actual = execute_stockham_ir(&kernel, &input).unwrap();
        let expected = fft(&input, Direction::Forward, false).unwrap();
        assert!(
            max_error(&actual, &expected) <= 8.0e-9 * length as f64,
            "boost-4 forward path did not match reference FFT"
        );

        let inverse_plan =
            FftPlan::build(crate::FftConfig::new(vec![length]).with_inverse_normalization(true))
                .unwrap();
        let inverse = KernelIr::stockham_1d(&inverse_plan, Direction::Inverse, profile).unwrap();
        let restored = execute_stockham_ir(&inverse, &actual).unwrap();
        assert!(max_error(&restored, &input) <= 1.6e-8 * length as f64);
    }

    #[test]
    fn register_boost_bank_padding_uses_upstream_strides_when_budget_allows() {
        let length = 16_384usize;
        let mut profile = DeviceProfile::generic(Backend::Vulkan, GpuVendor::Nvidia);
        profile.shared_memory_bytes = 48 * 1024;
        profile.shared_memory_pow2_bytes = 32 * 1024;
        profile.max_threads_per_block = 1024;

        let plan = FftPlan::build(crate::FftConfig::new(vec![length])).unwrap();
        let kernel = KernelIr::stockham_1d(&plan, Direction::Forward, profile).unwrap();
        assert_eq!(
            kernel.execution_layout,
            StockhamExecutionLayout::RegisterBoostSingleShared
        );
        let shared_layout = kernel.stockham_shared_layout.unwrap();
        assert_eq!(shared_layout.logical_elements, 4096);
        assert_eq!(shared_layout.shared_banks, 32);
        assert_eq!(shared_layout.bank_span_elements, 16);
        assert_eq!(shared_layout.first_stage_stride, 4352);
        assert_eq!(shared_layout.read_write_stride, 4112);
        assert_eq!(shared_layout.allocated_elements, 4352);
        assert!(shared_layout.first_stage_padding_enabled());
        assert_eq!(shared_layout.physical_index(16, true).unwrap(), 17);
        assert_eq!(shared_layout.physical_index(4095, true).unwrap(), 4350);
        assert_eq!(kernel.required_shared_memory_bytes().unwrap(), 34 * 1024);

        let stages = kernel.register_stockham_stages().unwrap().unwrap();
        assert!(shared_layout.stage_uses_first_stage_padding(
            length,
            stages[0].stage_size,
            stages[0].radix
        ));
        assert!(shared_layout.stage_uses_first_stage_padding(
            length,
            stages[1].stage_size,
            stages[1].radix
        ));
        assert!(!shared_layout.stage_uses_first_stage_padding(
            length,
            stages[2].stage_size,
            stages[2].radix
        ));

        let input = sample(length, 1);
        let actual = execute_stockham_ir(&kernel, &input).unwrap();
        let expected = fft(&input, Direction::Forward, false).unwrap();
        assert!(max_error(&actual, &expected) <= 8.0e-9 * length as f64);
    }

    #[test]
    fn register_scheduled_specialized_radices_match_reference_fft() {
        let length = 1024usize;
        let batch_count = 2usize;
        let plan =
            FftPlan::build(crate::FftConfig::new(vec![length]).with_batch_count(batch_count))
                .unwrap();
        let kernel = KernelIr::stockham_1d(&plan, Direction::Forward, device()).unwrap();
        assert!(kernel.scheduler_hint.is_some());
        let input = sample(length, batch_count);
        let actual = execute_stockham_ir(&kernel, &input).unwrap();
        for batch in 0..batch_count {
            let start = batch * length;
            let expected = fft(&input[start..start + length], Direction::Forward, false).unwrap();
            assert!(
                max_error(&actual[start..start + length], &expected) <= 2.0e-10 * length as f64,
                "register-scheduled forward batch {batch} did not match reference FFT"
            );
        }

        let inverse_plan = FftPlan::build(
            crate::FftConfig::new(vec![length])
                .with_batch_count(batch_count)
                .with_inverse_normalization(true),
        )
        .unwrap();
        let inverse = KernelIr::stockham_1d(&inverse_plan, Direction::Inverse, device()).unwrap();
        let restored = execute_stockham_ir(&inverse, &actual).unwrap();
        assert!(max_error(&restored, &input) <= 4.0e-10 * length as f64);
    }

    #[test]
    fn register_stockham_stages_expand_scheduler_virtual_threads() {
        let batch_count = 2usize;
        let plan = FftPlan::build(crate::FftConfig::new(vec![1024]).with_batch_count(batch_count))
            .unwrap();
        let kernel = KernelIr::stockham_1d(&plan, Direction::Forward, device()).unwrap();
        let stages = kernel.register_stockham_stages().unwrap().unwrap();
        assert_eq!(kernel.workgroup_size.x, 128);
        assert_eq!(
            stages.iter().map(|stage| stage.radix).collect::<Vec<_>>(),
            vec![8, 8, 8, 2]
        );
        assert_eq!(
            stages
                .iter()
                .map(|stage| stage.stage_size)
                .collect::<Vec<_>>(),
            vec![1, 8, 64, 512]
        );
        assert_eq!(
            stages
                .iter()
                .map(|stage| stage.registers_per_thread)
                .collect::<Vec<_>>(),
            vec![8, 8, 8, 8]
        );
        assert_eq!(
            stages
                .iter()
                .map(|stage| stage.butterflies_per_virtual_thread)
                .collect::<Vec<_>>(),
            vec![1, 1, 1, 4]
        );
        assert!(stages.iter().all(|stage| stage.virtual_thread_count == 128));
        assert_eq!(stages.last().unwrap().output, SharedBuffer::A);
    }

    #[test]
    fn ordinary_non_power_of_two_stockham_scheduler_covers_exact_small_mixed_leaves() {
        for length in [
            3usize, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 18, 20, 21, 22, 24, 25, 26, 27, 28, 30, 33,
            35, 36, 39, 42, 45, 49, 50, 54, 55, 60, 65, 66, 70, 72, 77, 78, 91, 105, 110, 121, 130,
            143, 154, 165, 169, 182, 195, 210, 231, 273, 286, 330, 390, 429, 462, 546, 770, 858,
        ] {
            let plan = FftPlan::build(crate::FftConfig::new(vec![length])).unwrap();
            let kernel = KernelIr::stockham_1d(&plan, Direction::Forward, device()).unwrap();
            let schedule = kernel.scheduler_hint.as_ref().expect(
                "covered NVIDIA small mixed-radix Stockham should use the register scheduler",
            );
            assert_eq!(schedule.fft_len, length);
            assert_eq!(
                kernel.execution_layout,
                StockhamExecutionLayout::RegisterSingleShared
            );
            let input = sample(length, 1);
            let actual = execute_stockham_ir(&kernel, &input).unwrap();
            let expected = dft(&input, Direction::Forward, false);
            assert!(max_error(&actual, &expected) < 1.0e-9 * length as f64);
        }
    }

    #[test]
    fn all_non_power_of_two_smooth_stockham_lengths_up_to_512_use_register_scheduler() {
        let profile = device();
        let mut covered = 0usize;
        for length in 3usize..=512 {
            if length.is_power_of_two() {
                continue;
            }
            let mut remaining = length;
            for prime in [2usize, 3, 5, 7, 11, 13] {
                while remaining.is_multiple_of(prime) {
                    remaining /= prime;
                }
            }
            if remaining != 1 {
                continue;
            }
            covered += 1;
            let plan = FftPlan::build(crate::FftConfig::new(vec![length])).unwrap();
            let kernel = KernelIr::stockham_1d(&plan, Direction::Forward, profile).unwrap();
            let schedule = kernel.scheduler_hint.as_ref().unwrap_or_else(|| {
                panic!("smooth non-power-of-two N={length} lost its register schedule")
            });
            assert_eq!(schedule.stage_radices.iter().product::<usize>(), length);
            assert_eq!(
                kernel.execution_layout,
                StockhamExecutionLayout::RegisterSingleShared,
                "N={length}"
            );
            let input = sample(length, 1);
            let actual = execute_stockham_ir(&kernel, &input).unwrap();
            let expected = fft(&input, Direction::Forward, false).unwrap();
            assert!(
                max_error(&actual, &expected) <= 2.0e-8 * length.ilog2().max(1) as f64,
                "register-scheduled smooth Stockham mismatch for N={length}"
            );
        }
        assert!(
            covered > 100,
            "expected broad smooth-length scheduler coverage"
        );
    }
}
