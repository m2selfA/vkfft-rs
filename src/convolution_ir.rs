//! Application-level convolution IR matching VkFFT's `performConvolution` semantics.
//!
//! The application surface keeps the frequency-domain multiply typed even when scalar
//! Stockham paths later fuse it into one or more dispatches. Callers supply immutable
//! frequency-space kernel data matching VkFFT's `kernel` buffer; supported single-kernel
//! matrix layouts expand coordinate planes while retaining the same forward/multiply/inverse
//! semantics.

use crate::complex::Complex64;
use crate::config::{
    ConvolutionConjugation, DeviceProfile, Direction, FftConfig, Precision, TransformKind,
    ZeroPaddingRange,
};
use crate::error::{Result, VkFftError};
use crate::kernel_ir::{
    DispatchGeometry, FourStepMapping, FourStepPreTwiddleMapping, KernelIr, KernelOperation,
    ScalarType, StockhamExecutionLayout, StockhamInputModifier, StockhamIoMapping,
    StockhamOutputModifier, StockhamStage, StockhamWorkgroupAxisLayout, ThreeUploadFourStepMapping,
    ThreeUploadPreTwiddleMapping, WorkgroupSize, execute_stockham_ir_with_lookup,
};
use crate::nd_ir::{
    NdExternalTensorLayout, NdFftIr, NdFormattedCopyOperation, NdFormattedCopyPassIr,
    execute_nd_fft_ir,
};
use crate::nd_real_ir::{NdRealFftIr, execute_nd_c2r_ir, execute_nd_r2c_ir};
use crate::one_dim_ir::{OneDimFftIr, execute_one_dim_fft_ir};
use crate::planner::{AxisAlgorithm, FftPlan};
use crate::rader_ir::{
    RaderDirectIr, RaderFftInputStrategy, RaderFftPassOperation, RaderFftPipelineIr,
};
use crate::real_ir::RealFftKind;
use crate::recursive_ir::RecursiveFftNodeIr;
use crate::scheduler::StockhamTwiddleSource;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ConvolutionMultiplyPolicy {
    pub conjugation: ConvolutionConjugation,
    pub cross_power_spectrum_normalization: bool,
}

impl ConvolutionMultiplyPolicy {
    fn from_config(config: &FftConfig) -> Self {
        Self {
            conjugation: config.convolution_conjugation,
            cross_power_spectrum_normalization: config.cross_power_spectrum_normalization,
        }
    }

    pub fn validate(self) -> Result<()> {
        if self.conjugation == ConvolutionConjugation::Kernel {
            return Err(VkFftError::UnsupportedKernelPath(
                "kernel-conjugated convolution is not executable in the pinned VkFFT codegen",
            ));
        }
        Ok(())
    }

    pub fn prepare_sequence(self, sequence: Complex64) -> Complex64 {
        match self.conjugation {
            ConvolutionConjugation::None => sequence,
            ConvolutionConjugation::Sequence => sequence.conj(),
            ConvolutionConjugation::Kernel => sequence,
        }
    }

    pub fn normalize_product(self, product: Complex64) -> Complex64 {
        if self.cross_power_spectrum_normalization {
            // Exact pinned-upstream order: PfNorm(re²+im²) -> PfRsqrt -> complex*scalar.
            // There is intentionally no zero guard/epsilon in VkFFT 1.3.4. Matrix
            // convolution applies this only after completing each row sum.
            product.scale(1.0 / product.norm_sqr().sqrt())
        } else {
            product
        }
    }

    pub fn apply(self, sequence: Complex64, kernel: Complex64) -> Complex64 {
        self.normalize_product(self.prepare_sequence(sequence) * kernel)
    }

    pub const fn is_default(self) -> bool {
        matches!(self.conjugation, ConvolutionConjugation::None)
            && !self.cross_power_spectrum_normalization
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConvolutionMatrixLayout {
    pub matrix_size: usize,
    pub symmetric_kernel: bool,
}

impl ConvolutionMatrixLayout {
    fn from_config(config: &FftConfig) -> Result<Option<Self>> {
        if config.matrix_convolution <= 1 {
            return Ok(None);
        }
        let layout = Self {
            matrix_size: config.matrix_convolution,
            symmetric_kernel: config.symmetric_convolution_kernel,
        };
        layout.validate()?;
        Ok(Some(layout))
    }

    pub fn validate(self) -> Result<()> {
        if !matches!(self.matrix_size, 2 | 3) {
            return Err(VkFftError::InvalidKernelIr(
                "matrix convolution IR requires a 2x2 or 3x3 matrix",
            ));
        }
        if self.symmetric_kernel && self.matrix_size == 3 {
            return Err(VkFftError::UnsupportedKernelPath(
                "pinned VkFFT 1.3.4 3x3 symmetric-kernel code aliases yz and zz while the API guide documents six planes",
            ));
        }
        Ok(())
    }

    pub fn kernel_plane_count(self) -> usize {
        if self.symmetric_kernel {
            self.matrix_size * (self.matrix_size + 1) / 2
        } else {
            self.matrix_size * self.matrix_size
        }
    }

    pub fn kernel_plane_index(
        self,
        output_coordinate: usize,
        input_coordinate: usize,
    ) -> Result<usize> {
        self.validate()?;
        if output_coordinate >= self.matrix_size || input_coordinate >= self.matrix_size {
            return Err(VkFftError::ValueOutOfRange {
                field: "matrix convolution coordinate",
            });
        }
        if !self.symmetric_kernel {
            return Ok(output_coordinate * self.matrix_size + input_coordinate);
        }
        let (j, l) = (output_coordinate, input_coordinate);
        Ok(if l < j {
            l * self.matrix_size - l * l + j
        } else {
            j * self.matrix_size - j * j + l
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ConvolutionMultiplyIr {
    pub name: String,
    pub scalar: ScalarType,
    pub sequence_len: usize,
    pub batch_count: usize,
    pub coordinate_count: usize,
    pub kernel_count: usize,
    pub matrix_layout: Option<ConvolutionMatrixLayout>,
    pub independent_coordinates: bool,
    pub workgroup_size: WorkgroupSize,
    pub dispatch: DispatchGeometry,
    pub policy: ConvolutionMultiplyPolicy,
    pub kernel_spectrum: Vec<Complex64>,
}

impl ConvolutionMultiplyIr {
    fn new(
        sequence_len: usize,
        batch_count: usize,
        coordinate_count: usize,
        kernel_count: usize,
        matrix_layout: Option<ConvolutionMatrixLayout>,
        scalar: ScalarType,
        policy: ConvolutionMultiplyPolicy,
        kernel_spectrum: Vec<Complex64>,
        device: DeviceProfile,
    ) -> Result<Self> {
        Self::new_with_coordinate_mode(
            sequence_len,
            batch_count,
            coordinate_count,
            kernel_count,
            matrix_layout,
            false,
            scalar,
            policy,
            kernel_spectrum,
            device,
        )
    }

    fn new_independent_coordinates(
        sequence_len: usize,
        batch_count: usize,
        coordinate_count: usize,
        kernel_count: usize,
        scalar: ScalarType,
        policy: ConvolutionMultiplyPolicy,
        kernel_spectrum: Vec<Complex64>,
        device: DeviceProfile,
    ) -> Result<Self> {
        Self::new_with_coordinate_mode(
            sequence_len,
            batch_count,
            coordinate_count,
            kernel_count,
            None,
            true,
            scalar,
            policy,
            kernel_spectrum,
            device,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn new_with_coordinate_mode(
        sequence_len: usize,
        batch_count: usize,
        coordinate_count: usize,
        kernel_count: usize,
        matrix_layout: Option<ConvolutionMatrixLayout>,
        independent_coordinates: bool,
        scalar: ScalarType,
        policy: ConvolutionMultiplyPolicy,
        kernel_spectrum: Vec<Complex64>,
        device: DeviceProfile,
    ) -> Result<Self> {
        if sequence_len == 0 || batch_count == 0 || coordinate_count == 0 || kernel_count == 0 {
            return Err(VkFftError::InvalidKernelIr(
                "convolution multiply dimensions must be non-zero",
            ));
        }
        let local_size = sequence_len
            .min(device.max_threads_per_block)
            .min(device.max_workgroup_size[0])
            .max(1);
        let pass = Self {
            name: if independent_coordinates {
                format!("vkfft_convolution_multiply_independent_{sequence_len}")
            } else {
                format!("vkfft_convolution_multiply_{sequence_len}")
            },
            scalar,
            sequence_len,
            batch_count,
            coordinate_count,
            kernel_count,
            matrix_layout,
            independent_coordinates,
            workgroup_size: WorkgroupSize {
                x: u32::try_from(local_size).map_err(|_| VkFftError::ValueOutOfRange {
                    field: "convolution multiply workgroup size",
                })?,
                y: 1,
                z: 1,
            },
            dispatch: DispatchGeometry {
                x: u32::try_from(if kernel_count > 1 {
                    kernel_count
                } else {
                    batch_count
                })
                .map_err(|_| VkFftError::ValueOutOfRange {
                    field: "convolution multiply output dispatch",
                })?,
                y: 1,
                z: 1,
            },
            policy,
            kernel_spectrum,
        };
        pass.validate()?;
        Ok(pass)
    }

    pub fn validate(&self) -> Result<()> {
        self.policy.validate()?;
        if let Some(matrix) = self.matrix_layout {
            if self.independent_coordinates {
                return Err(VkFftError::InvalidKernelIr(
                    "matrix and independent-coordinate convolution modes are mutually exclusive",
                ));
            }
            matrix.validate()?;
            if self.coordinate_count != matrix.matrix_size {
                return Err(VkFftError::InvalidKernelIr(
                    "matrix convolution coordinate count must equal its matrix size",
                ));
            }
        } else if self.independent_coordinates {
            if self.coordinate_count <= 1 || self.batch_count != 1 {
                return Err(VkFftError::InvalidKernelIr(
                    "independent-coordinate convolution requires batch1 and more than one coordinate plane",
                ));
            }
        } else if self.coordinate_count != 1 {
            return Err(VkFftError::InvalidKernelIr(
                "scalar convolution multiply requires one coordinate plane",
            ));
        }
        if self.sequence_len == 0
            || self.batch_count == 0
            || self.coordinate_count == 0
            || self.kernel_count == 0
            || self.workgroup_size.x == 0
            || self.workgroup_size.y != 1
            || self.workgroup_size.z != 1
            || self.dispatch.x as usize
                != if self.kernel_count > 1 {
                    self.kernel_count
                } else {
                    self.batch_count
                }
            || self.dispatch.y != 1
            || self.dispatch.z != 1
        {
            return Err(VkFftError::InvalidKernelIr(
                "convolution multiply dispatch metadata is inconsistent",
            ));
        }
        if !matches!(self.scalar, ScalarType::F32 | ScalarType::F64) {
            return Err(VkFftError::InvalidKernelIr(
                "initial convolution multiply requires F32 or F64 compute",
            ));
        }
        let kernel_planes = if let Some(matrix) = self.matrix_layout {
            matrix.kernel_plane_count()
        } else if self.independent_coordinates {
            self.coordinate_count
        } else {
            1
        };
        let expected_kernel_len = self
            .sequence_len
            .checked_mul(kernel_planes)
            .and_then(|value| value.checked_mul(self.kernel_count))
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "convolution kernel spectrum element count",
            })?;
        if self.kernel_spectrum.len() != expected_kernel_len {
            return Err(VkFftError::InvalidKernelIr(
                "convolution kernel spectrum length must match its scalar/independent/matrix plane layout",
            ));
        }
        if self
            .kernel_spectrum
            .iter()
            .any(|value| !value.re.is_finite() || !value.im.is_finite())
        {
            return Err(VkFftError::InvalidKernelIr(
                "convolution kernel spectrum contains non-finite values",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConvolutionStockhamStepIr {
    pub name: String,
    pub sequence_len: usize,
    pub batch_count: usize,
    pub scalar: ScalarType,
    pub policy: ConvolutionMultiplyPolicy,
    pub forward: KernelIr,
    pub inverse: KernelIr,
    pub workgroup_size: WorkgroupSize,
    pub dispatch: DispatchGeometry,
    pub required_shared_memory_bytes: usize,
}

impl ConvolutionStockhamStepIr {
    fn try_new(
        forward: &KernelIr,
        inverse: &KernelIr,
        policy: ConvolutionMultiplyPolicy,
        device: DeviceProfile,
    ) -> Result<Option<Self>> {
        forward.validate()?;
        inverse.validate()?;
        policy.validate()?;
        // Start with the ordinary F32/no-LUT surface. F64 and any mapped/modified
        // boundary keep the already-proven two-dispatch fallback until their exact
        // convolutionStep resource/binding contract is pinned separately.
        if forward.scalar != ScalarType::F32
            || inverse.scalar != ScalarType::F32
            || forward.direction != Direction::Forward
            || inverse.direction != Direction::Inverse
            || forward.sequence_len != inverse.sequence_len
            || forward.batch_count != inverse.batch_count
            || forward.io_mapping != StockhamIoMapping::Contiguous
            || inverse.io_mapping != StockhamIoMapping::Contiguous
            || forward.input_modifier != StockhamInputModifier::None
            || inverse.input_modifier != StockhamInputModifier::None
            || forward.output_modifier != StockhamOutputModifier::None
            || inverse.output_modifier != StockhamOutputModifier::None
            || forward.twiddle_source != StockhamTwiddleSource::OnTheFly
            || inverse.twiddle_source != StockhamTwiddleSource::OnTheFly
            || forward.workgroup_grouping != inverse.workgroup_grouping
            || forward.workgroup_size != inverse.workgroup_size
            || forward.dispatch != inverse.dispatch
        {
            return Ok(None);
        }
        let forward_normalize = stockham_store_normalize(forward)?;
        let inverse_normalize = stockham_store_normalize(inverse)?;
        if forward_normalize || !inverse_normalize {
            return Ok(None);
        }
        let required_shared_memory_bytes = forward
            .sequence_len
            .checked_mul(forward.workgroup_grouping.transforms_per_workgroup)
            .and_then(|value| value.checked_mul(forward.scalar.complex_bytes()))
            .and_then(|value| value.checked_mul(2))
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "performConvolution fused Stockham shared memory",
            })?;
        let available_shared = if forward.sequence_len.is_power_of_two() {
            device.shared_memory_pow2_bytes
        } else {
            device.shared_memory_bytes
        };
        if required_shared_memory_bytes > available_shared {
            return Ok(None);
        }
        let step = Self {
            name: format!("vkfft_convolution_step_stockham_{}", forward.sequence_len),
            sequence_len: forward.sequence_len,
            batch_count: forward.batch_count,
            scalar: forward.scalar,
            policy,
            forward: forward.clone(),
            inverse: inverse.clone(),
            workgroup_size: forward.workgroup_size,
            dispatch: forward.dispatch,
            required_shared_memory_bytes,
        };
        step.validate()?;
        Ok(Some(step))
    }

    pub fn validate(&self) -> Result<()> {
        self.forward.validate()?;
        self.inverse.validate()?;
        self.policy.validate()?;
        if self.sequence_len == 0
            || self.batch_count == 0
            || self.scalar != ScalarType::F32
            || self.forward.direction != Direction::Forward
            || self.inverse.direction != Direction::Inverse
            || self.forward.sequence_len != self.sequence_len
            || self.inverse.sequence_len != self.sequence_len
            || self.forward.batch_count != self.batch_count
            || self.inverse.batch_count != self.batch_count
            || self.forward.scalar != self.scalar
            || self.inverse.scalar != self.scalar
            || self.forward.io_mapping != StockhamIoMapping::Contiguous
            || self.inverse.io_mapping != StockhamIoMapping::Contiguous
            || self.forward.input_modifier != StockhamInputModifier::None
            || self.inverse.input_modifier != StockhamInputModifier::None
            || self.forward.output_modifier != StockhamOutputModifier::None
            || self.inverse.output_modifier != StockhamOutputModifier::None
            || self.forward.twiddle_source != StockhamTwiddleSource::OnTheFly
            || self.inverse.twiddle_source != StockhamTwiddleSource::OnTheFly
            || self.forward.workgroup_grouping != self.inverse.workgroup_grouping
            || self.forward.workgroup_size != self.workgroup_size
            || self.inverse.workgroup_size != self.workgroup_size
            || self.forward.dispatch != self.dispatch
            || self.inverse.dispatch != self.dispatch
            || stockham_store_normalize(&self.forward)?
            || !stockham_store_normalize(&self.inverse)?
            || self.required_shared_memory_bytes == 0
        {
            return Err(VkFftError::InvalidKernelIr(
                "fused performConvolution Stockham step metadata is inconsistent",
            ));
        }
        Ok(())
    }
}

fn stockham_store_normalize(kernel: &KernelIr) -> Result<bool> {
    kernel
        .operations
        .iter()
        .find_map(|operation| match operation {
            KernelOperation::StoreSharedToGlobal { normalize, .. } => Some(*normalize),
            _ => None,
        })
        .ok_or(VkFftError::InvalidKernelIr(
            "performConvolution Stockham child is missing its store operation",
        ))
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConvolutionMatrixStockhamStepIr {
    pub name: String,
    pub sequence_len: usize,
    pub coordinate_count: usize,
    pub kernel_count: usize,
    pub matrix_layout: ConvolutionMatrixLayout,
    pub scalar: ScalarType,
    pub policy: ConvolutionMultiplyPolicy,
    pub forward: KernelIr,
    pub inverse: KernelIr,
    pub workgroup_size: WorkgroupSize,
    pub dispatch: DispatchGeometry,
    pub required_shared_memory_bytes: usize,
}

fn stockham_stage_chain(kernel: &KernelIr) -> Vec<StockhamStage> {
    kernel
        .operations
        .iter()
        .filter_map(|operation| match operation {
            KernelOperation::StockhamStage(stage) => Some(*stage),
            _ => None,
        })
        .collect()
}

impl ConvolutionMatrixStockhamStepIr {
    fn try_new(
        forward: &KernelIr,
        inverse: &KernelIr,
        matrix_layout: ConvolutionMatrixLayout,
        kernel_count: usize,
        policy: ConvolutionMultiplyPolicy,
        device: DeviceProfile,
    ) -> Result<Option<Self>> {
        forward.validate()?;
        inverse.validate()?;
        matrix_layout.validate()?;
        policy.validate()?;
        let coordinate_count = matrix_layout.matrix_size;
        let inverse_batch_count =
            coordinate_count
                .checked_mul(kernel_count)
                .ok_or(VkFftError::ArithmeticOverflow {
                    operation: "matrix performConvolution fused inverse batch count",
                })?;
        let forward_stages = stockham_stage_chain(forward);
        let inverse_stages = stockham_stage_chain(inverse);
        let threads_per_transform = forward.workgroup_grouping.threads_per_transform;
        if kernel_count == 0
            || !matches!(forward.scalar, ScalarType::F32 | ScalarType::F64)
            || inverse.scalar != forward.scalar
            || forward.direction != Direction::Forward
            || inverse.direction != Direction::Inverse
            || forward.sequence_len != inverse.sequence_len
            || forward.batch_count != coordinate_count
            || inverse.batch_count != inverse_batch_count
            || forward.io_mapping != StockhamIoMapping::Contiguous
            || inverse.io_mapping != StockhamIoMapping::Contiguous
            || forward.input_modifier != StockhamInputModifier::None
            || inverse.input_modifier != StockhamInputModifier::None
            || forward.output_modifier != StockhamOutputModifier::None
            || inverse.output_modifier != StockhamOutputModifier::None
            || forward.twiddle_source != inverse.twiddle_source
            || forward.rader_transpose.is_some()
            || inverse.rader_transpose.is_some()
            || forward_stages.is_empty()
            || forward_stages != inverse_stages
            || threads_per_transform == 0
            || inverse.workgroup_grouping.threads_per_transform != threads_per_transform
            || threads_per_transform > device.max_threads_per_block
            || threads_per_transform > device.max_workgroup_size[0]
            || stockham_store_normalize(forward)?
            || !stockham_store_normalize(inverse)?
        {
            return Ok(None);
        }
        let shared_elements = coordinate_count
            .checked_add(2)
            .and_then(|value| value.checked_mul(forward.sequence_len))
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "matrix performConvolution fused shared element count",
            })?;
        let required_shared_memory_bytes = shared_elements
            .checked_mul(forward.scalar.complex_bytes())
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "matrix performConvolution fused shared byte count",
            })?;
        let available_shared = if forward.sequence_len.is_power_of_two() {
            device.shared_memory_pow2_bytes
        } else {
            device.shared_memory_bytes
        };
        if required_shared_memory_bytes > available_shared {
            return Ok(None);
        }
        let step = Self {
            name: format!(
                "vkfft_convolution_step_matrix_stockham_{}_m{}_k{}",
                forward.sequence_len, coordinate_count, kernel_count
            ),
            sequence_len: forward.sequence_len,
            coordinate_count,
            kernel_count,
            matrix_layout,
            scalar: forward.scalar,
            policy,
            forward: forward.clone(),
            inverse: inverse.clone(),
            workgroup_size: WorkgroupSize {
                x: threads_per_transform as u32,
                y: 1,
                z: 1,
            },
            dispatch: DispatchGeometry { x: 1, y: 1, z: 1 },
            required_shared_memory_bytes,
        };
        step.validate()?;
        Ok(Some(step))
    }

    pub fn requires_twiddle_lut(&self) -> Result<bool> {
        if self.forward.twiddle_source != self.inverse.twiddle_source {
            return Err(VkFftError::InvalidKernelIr(
                "fused matrix Stockham children disagree on twiddle source",
            ));
        }
        Ok(self.forward.twiddle_source == StockhamTwiddleSource::LookupTable)
    }

    pub fn validate(&self) -> Result<()> {
        self.forward.validate()?;
        self.inverse.validate()?;
        self.matrix_layout.validate()?;
        self.policy.validate()?;
        let inverse_batch_count = self.coordinate_count.checked_mul(self.kernel_count).ok_or(
            VkFftError::ArithmeticOverflow {
                operation: "matrix performConvolution fused validation inverse batch count",
            },
        )?;
        let expected_shared = self
            .coordinate_count
            .checked_add(2)
            .and_then(|value| value.checked_mul(self.sequence_len))
            .and_then(|value| value.checked_mul(self.scalar.complex_bytes()))
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "matrix performConvolution fused validation shared byte count",
            })?;
        let forward_stages = stockham_stage_chain(&self.forward);
        if self.sequence_len == 0
            || self.coordinate_count != self.matrix_layout.matrix_size
            || self.kernel_count == 0
            || !matches!(self.scalar, ScalarType::F32 | ScalarType::F64)
            || self.forward.direction != Direction::Forward
            || self.inverse.direction != Direction::Inverse
            || self.forward.sequence_len != self.sequence_len
            || self.inverse.sequence_len != self.sequence_len
            || self.forward.batch_count != self.coordinate_count
            || self.inverse.batch_count != inverse_batch_count
            || self.forward.scalar != self.scalar
            || self.inverse.scalar != self.scalar
            || self.forward.io_mapping != StockhamIoMapping::Contiguous
            || self.inverse.io_mapping != StockhamIoMapping::Contiguous
            || self.forward.input_modifier != StockhamInputModifier::None
            || self.inverse.input_modifier != StockhamInputModifier::None
            || self.forward.output_modifier != StockhamOutputModifier::None
            || self.inverse.output_modifier != StockhamOutputModifier::None
            || self.forward.twiddle_source != self.inverse.twiddle_source
            || self.forward.rader_transpose.is_some()
            || self.inverse.rader_transpose.is_some()
            || forward_stages.is_empty()
            || forward_stages != stockham_stage_chain(&self.inverse)
            || self.forward.workgroup_grouping.threads_per_transform as u32 != self.workgroup_size.x
            || self.inverse.workgroup_grouping.threads_per_transform as u32 != self.workgroup_size.x
            || self.workgroup_size.y != 1
            || self.workgroup_size.z != 1
            || self.dispatch != (DispatchGeometry { x: 1, y: 1, z: 1 })
            || stockham_store_normalize(&self.forward)?
            || !stockham_store_normalize(&self.inverse)?
            || self.required_shared_memory_bytes != expected_shared
        {
            return Err(VkFftError::InvalidKernelIr(
                "fused matrix performConvolution Stockham metadata is inconsistent",
            ));
        }
        let _ = self.requires_twiddle_lut()?;
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConvolutionMultiKernelStockhamStepIr {
    pub name: String,
    pub sequence_len: usize,
    pub kernel_count: usize,
    pub scalar: ScalarType,
    pub policy: ConvolutionMultiplyPolicy,
    pub forward: KernelIr,
    pub inverse: KernelIr,
    pub workgroup_size: WorkgroupSize,
    pub dispatch: DispatchGeometry,
    pub required_shared_memory_bytes: usize,
}

impl ConvolutionMultiKernelStockhamStepIr {
    fn try_new(
        forward: &KernelIr,
        inverse: &KernelIr,
        kernel_count: usize,
        policy: ConvolutionMultiplyPolicy,
        device: DeviceProfile,
    ) -> Result<Option<Self>> {
        forward.validate()?;
        inverse.validate()?;
        policy.validate()?;
        let forward_stages = stockham_stage_chain(forward);
        let inverse_stages = stockham_stage_chain(inverse);
        let threads_per_transform = forward.workgroup_grouping.threads_per_transform;
        if kernel_count <= 1
            || !matches!(forward.scalar, ScalarType::F32 | ScalarType::F64)
            || inverse.scalar != forward.scalar
            || forward.direction != Direction::Forward
            || inverse.direction != Direction::Inverse
            || forward.sequence_len != inverse.sequence_len
            || forward.batch_count != 1
            || inverse.batch_count != kernel_count
            || forward.io_mapping != StockhamIoMapping::Contiguous
            || inverse.io_mapping != StockhamIoMapping::Contiguous
            || forward.input_modifier != StockhamInputModifier::None
            || inverse.input_modifier != StockhamInputModifier::None
            || forward.output_modifier != StockhamOutputModifier::None
            || inverse.output_modifier != StockhamOutputModifier::None
            || forward.twiddle_source != inverse.twiddle_source
            || forward.rader_transpose.is_some()
            || inverse.rader_transpose.is_some()
            || forward_stages.is_empty()
            || forward_stages != inverse_stages
            || threads_per_transform == 0
            || inverse.workgroup_grouping.threads_per_transform != threads_per_transform
            || threads_per_transform > device.max_threads_per_block
            || threads_per_transform > device.max_workgroup_size[0]
            || stockham_store_normalize(forward)?
            || !stockham_store_normalize(inverse)?
        {
            return Ok(None);
        }
        let required_shared_memory_bytes = forward
            .sequence_len
            .checked_mul(3)
            .and_then(|value| value.checked_mul(forward.scalar.complex_bytes()))
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "multi-kernel performConvolution fused shared byte count",
            })?;
        let available_shared = if forward.sequence_len.is_power_of_two() {
            device.shared_memory_pow2_bytes
        } else {
            device.shared_memory_bytes
        };
        if required_shared_memory_bytes > available_shared {
            return Ok(None);
        }
        let step = Self {
            name: format!(
                "vkfft_convolution_step_multi_kernel_stockham_{}_k{}",
                forward.sequence_len, kernel_count
            ),
            sequence_len: forward.sequence_len,
            kernel_count,
            scalar: forward.scalar,
            policy,
            forward: forward.clone(),
            inverse: inverse.clone(),
            workgroup_size: WorkgroupSize {
                x: threads_per_transform as u32,
                y: 1,
                z: 1,
            },
            dispatch: DispatchGeometry { x: 1, y: 1, z: 1 },
            required_shared_memory_bytes,
        };
        step.validate()?;
        Ok(Some(step))
    }

    pub fn requires_twiddle_lut(&self) -> Result<bool> {
        if self.forward.twiddle_source != self.inverse.twiddle_source {
            return Err(VkFftError::InvalidKernelIr(
                "fused multi-kernel Stockham children disagree on twiddle source",
            ));
        }
        Ok(self.forward.twiddle_source == StockhamTwiddleSource::LookupTable)
    }

    pub fn validate(&self) -> Result<()> {
        self.forward.validate()?;
        self.inverse.validate()?;
        self.policy.validate()?;
        let expected_shared = self
            .sequence_len
            .checked_mul(3)
            .and_then(|value| value.checked_mul(self.scalar.complex_bytes()))
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "multi-kernel performConvolution fused validation shared byte count",
            })?;
        let forward_stages = stockham_stage_chain(&self.forward);
        if self.sequence_len == 0
            || self.kernel_count <= 1
            || !matches!(self.scalar, ScalarType::F32 | ScalarType::F64)
            || self.forward.direction != Direction::Forward
            || self.inverse.direction != Direction::Inverse
            || self.forward.sequence_len != self.sequence_len
            || self.inverse.sequence_len != self.sequence_len
            || self.forward.batch_count != 1
            || self.inverse.batch_count != self.kernel_count
            || self.forward.scalar != self.scalar
            || self.inverse.scalar != self.scalar
            || self.forward.io_mapping != StockhamIoMapping::Contiguous
            || self.inverse.io_mapping != StockhamIoMapping::Contiguous
            || self.forward.input_modifier != StockhamInputModifier::None
            || self.inverse.input_modifier != StockhamInputModifier::None
            || self.forward.output_modifier != StockhamOutputModifier::None
            || self.inverse.output_modifier != StockhamOutputModifier::None
            || self.forward.twiddle_source != self.inverse.twiddle_source
            || self.forward.rader_transpose.is_some()
            || self.inverse.rader_transpose.is_some()
            || forward_stages.is_empty()
            || forward_stages != stockham_stage_chain(&self.inverse)
            || self.forward.workgroup_grouping.threads_per_transform as u32 != self.workgroup_size.x
            || self.inverse.workgroup_grouping.threads_per_transform as u32 != self.workgroup_size.x
            || self.workgroup_size.y != 1
            || self.workgroup_size.z != 1
            || self.dispatch != (DispatchGeometry { x: 1, y: 1, z: 1 })
            || stockham_store_normalize(&self.forward)?
            || !stockham_store_normalize(&self.inverse)?
            || self.required_shared_memory_bytes != expected_shared
        {
            return Err(VkFftError::InvalidKernelIr(
                "fused multi-kernel performConvolution Stockham metadata is inconsistent",
            ));
        }
        let _ = self.requires_twiddle_lut()?;
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConvolutionTwoUploadStockhamIr {
    pub name: String,
    pub sequence_len: usize,
    pub batch_count: usize,
    pub scalar: ScalarType,
    pub policy: ConvolutionMultiplyPolicy,
    pub mapping: FourStepMapping,
    pub forward_high: KernelIr,
    pub forward_low: KernelIr,
    pub inverse_low: KernelIr,
    pub inverse_high: KernelIr,
    pub special_workgroup_size: WorkgroupSize,
    pub special_dispatch: DispatchGeometry,
    pub required_shared_memory_bytes: usize,
}

impl ConvolutionTwoUploadStockhamIr {
    fn try_new(
        forward: &crate::recursive_ir::RecursiveFftIr,
        inverse: &crate::recursive_ir::RecursiveFftIr,
        policy: ConvolutionMultiplyPolicy,
        device: DeviceProfile,
    ) -> Result<Option<Self>> {
        policy.validate()?;
        let Some(forward_uploads) = forward.four_step_stockham_upload_kernels()? else {
            return Ok(None);
        };
        let Some(inverse_uploads) = inverse.four_step_stockham_upload_kernels()? else {
            return Ok(None);
        };
        let [forward_high, forward_low] = forward_uploads.as_slice() else {
            return Ok(None);
        };
        let [inverse_high, inverse_low] = inverse_uploads.as_slice() else {
            return Ok(None);
        };
        let scalar = forward_low.scalar;
        let twiddle_source = forward_low.twiddle_source;
        if !matches!(scalar, ScalarType::F32 | ScalarType::F64)
            || forward_high.scalar != scalar
            || inverse_low.scalar != scalar
            || inverse_high.scalar != scalar
            || forward_high.twiddle_source != twiddle_source
            || inverse_low.twiddle_source != twiddle_source
            || inverse_high.twiddle_source != twiddle_source
            || forward_low.workgroup_grouping != inverse_low.workgroup_grouping
            || forward_low.workgroup_size != inverse_low.workgroup_size
            || forward_low.dispatch != inverse_low.dispatch
            || forward_low.input_modifier != StockhamInputModifier::None
            || inverse_low.input_modifier != StockhamInputModifier::None
            || forward_low.output_modifier != StockhamOutputModifier::None
            || inverse_low.output_modifier != StockhamOutputModifier::None
            || stockham_store_normalize(forward_low)?
            || !stockham_store_normalize(inverse_low)?
        {
            return Ok(None);
        }
        let StockhamIoMapping::FourStepLeft(mapping) = forward_low.io_mapping else {
            return Ok(None);
        };
        if inverse_low.io_mapping != StockhamIoMapping::FourStepLeft(mapping)
            || forward_high.io_mapping != StockhamIoMapping::FourStepRight(mapping)
            || inverse_high.io_mapping != StockhamIoMapping::FourStepRight(mapping)
            || mapping.logical_len != forward.logical_len
            || mapping.logical_len != inverse.logical_len
            || mapping.outer_batch_count != forward.batch_count
            || mapping.outer_batch_count != inverse.batch_count
        {
            return Ok(None);
        }
        let inverse_high = inverse_high
            .clone()
            .with_stockham_io_mapping(StockhamIoMapping::FourStepRightPreTwiddle(
                FourStepPreTwiddleMapping {
                    four_step: mapping,
                    direction: Direction::Inverse,
                },
            ))?
            .with_store_normalization(false)?;
        let required_shared_memory_bytes = forward_low
            .sequence_len
            .checked_mul(forward_low.workgroup_grouping.transforms_per_workgroup)
            .and_then(|value| value.checked_mul(forward_low.scalar.complex_bytes()))
            .and_then(|value| value.checked_mul(2))
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "two-upload performConvolution special shared memory",
            })?;
        let available_shared = if forward_low.sequence_len.is_power_of_two() {
            device.shared_memory_pow2_bytes
        } else {
            device.shared_memory_bytes
        };
        if required_shared_memory_bytes > available_shared {
            return Ok(None);
        }
        let ir = Self {
            name: format!("vkfft_convolution_two_upload_{}", mapping.logical_len),
            sequence_len: mapping.logical_len,
            batch_count: mapping.outer_batch_count,
            scalar,
            policy,
            mapping,
            forward_high: forward_high.clone(),
            forward_low: forward_low.clone(),
            inverse_low: inverse_low.clone(),
            inverse_high,
            special_workgroup_size: forward_low.workgroup_size,
            special_dispatch: forward_low.dispatch,
            required_shared_memory_bytes,
        };
        ir.validate()?;
        Ok(Some(ir))
    }

    pub fn special_twiddle_lut_len(&self) -> Result<Option<usize>> {
        let forward = self.forward_low.twiddle_lut_len();
        let inverse = self.inverse_low.twiddle_lut_len();
        if forward != inverse
            || self.forward_high.twiddle_lut_len() != forward
            || self.inverse_high.twiddle_lut_len() != forward
        {
            return Err(VkFftError::InvalidKernelIr(
                "two-upload performConvolution Stockham twiddle LUT periods are inconsistent",
            ));
        }
        Ok(forward)
    }

    pub fn validate(&self) -> Result<()> {
        self.policy.validate()?;
        self.mapping.validate()?;
        self.forward_high.validate()?;
        self.forward_low.validate()?;
        self.inverse_low.validate()?;
        self.inverse_high.validate()?;
        let _ = self.special_twiddle_lut_len()?;
        let high_batches = self.batch_count.checked_mul(self.mapping.left_len).ok_or(
            VkFftError::ArithmeticOverflow {
                operation: "two-upload convolution high batch count",
            },
        )?;
        let low_batches = self.batch_count.checked_mul(self.mapping.right_len).ok_or(
            VkFftError::ArithmeticOverflow {
                operation: "two-upload convolution low batch count",
            },
        )?;
        let twiddle_source = self.forward_low.twiddle_source;
        if self.sequence_len != self.mapping.logical_len
            || !matches!(self.scalar, ScalarType::F32 | ScalarType::F64)
            || self.forward_high.scalar != self.scalar
            || self.forward_low.scalar != self.scalar
            || self.inverse_low.scalar != self.scalar
            || self.inverse_high.scalar != self.scalar
            || self.forward_high.twiddle_source != twiddle_source
            || self.inverse_low.twiddle_source != twiddle_source
            || self.inverse_high.twiddle_source != twiddle_source
            || self.forward_high.direction != Direction::Forward
            || self.forward_low.direction != Direction::Forward
            || self.inverse_low.direction != Direction::Inverse
            || self.inverse_high.direction != Direction::Inverse
            || self.forward_high.sequence_len != self.mapping.right_len
            || self.inverse_high.sequence_len != self.mapping.right_len
            || self.forward_low.sequence_len != self.mapping.left_len
            || self.inverse_low.sequence_len != self.mapping.left_len
            || self.forward_high.batch_count != high_batches
            || self.inverse_high.batch_count != high_batches
            || self.forward_low.batch_count != low_batches
            || self.inverse_low.batch_count != low_batches
            || self.forward_high.io_mapping != StockhamIoMapping::FourStepRight(self.mapping)
            || self.forward_low.io_mapping != StockhamIoMapping::FourStepLeft(self.mapping)
            || self.inverse_low.io_mapping != StockhamIoMapping::FourStepLeft(self.mapping)
            || self.inverse_high.io_mapping
                != StockhamIoMapping::FourStepRightPreTwiddle(FourStepPreTwiddleMapping {
                    four_step: self.mapping,
                    direction: Direction::Inverse,
                })
            || stockham_store_normalize(&self.forward_high)?
            || stockham_store_normalize(&self.forward_low)?
            || !stockham_store_normalize(&self.inverse_low)?
            || stockham_store_normalize(&self.inverse_high)?
            || self.forward_low.workgroup_grouping != self.inverse_low.workgroup_grouping
            || self.forward_low.workgroup_size != self.special_workgroup_size
            || self.inverse_low.workgroup_size != self.special_workgroup_size
            || self.forward_low.dispatch != self.special_dispatch
            || self.inverse_low.dispatch != self.special_dispatch
            || self.required_shared_memory_bytes == 0
        {
            return Err(VkFftError::InvalidKernelIr(
                "two-upload performConvolution Stockham metadata is inconsistent",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConvolutionTwoUploadMultiKernelStockhamIr {
    pub name: String,
    pub sequence_len: usize,
    pub kernel_count: usize,
    pub scalar: ScalarType,
    pub policy: ConvolutionMultiplyPolicy,
    pub forward_mapping: FourStepMapping,
    pub inverse_mapping: FourStepMapping,
    pub forward_high: KernelIr,
    pub forward_low: KernelIr,
    pub inverse_low: KernelIr,
    pub inverse_high: KernelIr,
    pub special_workgroup_size: WorkgroupSize,
    pub special_dispatch: DispatchGeometry,
    pub required_shared_memory_bytes: usize,
}

impl ConvolutionTwoUploadMultiKernelStockhamIr {
    fn try_new(
        forward: &crate::recursive_ir::RecursiveFftIr,
        inverse: &crate::recursive_ir::RecursiveFftIr,
        kernel_count: usize,
        policy: ConvolutionMultiplyPolicy,
        device: DeviceProfile,
    ) -> Result<Option<Self>> {
        policy.validate()?;
        if kernel_count <= 1 || forward.batch_count != 1 || inverse.batch_count != kernel_count {
            return Ok(None);
        }
        let Some(forward_uploads) = forward.four_step_stockham_upload_kernels()? else {
            return Ok(None);
        };
        let Some(inverse_uploads) = inverse.four_step_stockham_upload_kernels()? else {
            return Ok(None);
        };
        let [forward_high, forward_low] = forward_uploads.as_slice() else {
            return Ok(None);
        };
        let [inverse_high, inverse_low] = inverse_uploads.as_slice() else {
            return Ok(None);
        };
        let StockhamIoMapping::FourStepLeft(forward_mapping) = forward_low.io_mapping else {
            return Ok(None);
        };
        let StockhamIoMapping::FourStepLeft(inverse_mapping) = inverse_low.io_mapping else {
            return Ok(None);
        };
        let scalar = forward_low.scalar;
        let twiddle_source = forward_low.twiddle_source;
        if !matches!(scalar, ScalarType::F32 | ScalarType::F64)
            || forward_high.scalar != scalar
            || inverse_low.scalar != scalar
            || inverse_high.scalar != scalar
            || forward_high.twiddle_source != twiddle_source
            || inverse_low.twiddle_source != twiddle_source
            || inverse_high.twiddle_source != twiddle_source
            || forward_low.workgroup_grouping != inverse_low.workgroup_grouping
            || forward_low.workgroup_size != inverse_low.workgroup_size
            || forward_low.input_modifier != StockhamInputModifier::None
            || inverse_low.input_modifier != StockhamInputModifier::None
            || forward_low.output_modifier != StockhamOutputModifier::None
            || inverse_low.output_modifier != StockhamOutputModifier::None
            || stockham_store_normalize(forward_low)?
            || !stockham_store_normalize(inverse_low)?
            || forward_mapping.logical_len != inverse_mapping.logical_len
            || forward_mapping.left_len != inverse_mapping.left_len
            || forward_mapping.right_len != inverse_mapping.right_len
            || forward_mapping.outer_batch_count != 1
            || inverse_mapping.outer_batch_count != kernel_count
            || forward_high.io_mapping != StockhamIoMapping::FourStepRight(forward_mapping)
            || inverse_high.io_mapping != StockhamIoMapping::FourStepRight(inverse_mapping)
        {
            return Ok(None);
        }
        let inverse_high = inverse_high
            .clone()
            .with_stockham_io_mapping(StockhamIoMapping::FourStepRightPreTwiddle(
                FourStepPreTwiddleMapping {
                    four_step: inverse_mapping,
                    direction: Direction::Inverse,
                },
            ))?
            .with_store_normalization(false)?;
        let required_shared_memory_bytes = forward_low
            .sequence_len
            .checked_mul(forward_low.workgroup_grouping.transforms_per_workgroup)
            .and_then(|value| value.checked_mul(forward_low.scalar.complex_bytes()))
            .and_then(|value| value.checked_mul(2))
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "two-upload multi-kernel performConvolution special shared memory",
            })?;
        let available_shared = if forward_low.sequence_len.is_power_of_two() {
            device.shared_memory_pow2_bytes
        } else {
            device.shared_memory_bytes
        };
        if required_shared_memory_bytes > available_shared {
            return Ok(None);
        }
        let ir = Self {
            name: format!(
                "vkfft_convolution_two_upload_multi_kernel_{}_k{}",
                forward_mapping.logical_len, kernel_count
            ),
            sequence_len: forward_mapping.logical_len,
            kernel_count,
            scalar,
            policy,
            forward_mapping,
            inverse_mapping,
            forward_high: forward_high.clone(),
            forward_low: forward_low.clone(),
            inverse_low: inverse_low.clone(),
            inverse_high,
            special_workgroup_size: forward_low.workgroup_size,
            special_dispatch: forward_low.dispatch,
            required_shared_memory_bytes,
        };
        ir.validate()?;
        Ok(Some(ir))
    }

    pub fn special_twiddle_lut_len(&self) -> Result<Option<usize>> {
        let forward = self.forward_low.twiddle_lut_len();
        let inverse = self.inverse_low.twiddle_lut_len();
        if forward != inverse
            || self.forward_high.twiddle_lut_len() != forward
            || self.inverse_high.twiddle_lut_len() != forward
        {
            return Err(VkFftError::InvalidKernelIr(
                "two-upload multi-kernel performConvolution Stockham twiddle LUT periods are inconsistent",
            ));
        }
        Ok(forward)
    }

    pub fn validate(&self) -> Result<()> {
        self.policy.validate()?;
        self.forward_mapping.validate()?;
        self.inverse_mapping.validate()?;
        self.forward_high.validate()?;
        self.forward_low.validate()?;
        self.inverse_low.validate()?;
        self.inverse_high.validate()?;
        let _ = self.special_twiddle_lut_len()?;
        let forward_high_batches = self.forward_mapping.left_len;
        let forward_low_batches = self.forward_mapping.right_len;
        let inverse_high_batches = self
            .kernel_count
            .checked_mul(self.inverse_mapping.left_len)
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "two-upload multi-kernel inverse high batch count",
            })?;
        let inverse_low_batches = self
            .kernel_count
            .checked_mul(self.inverse_mapping.right_len)
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "two-upload multi-kernel inverse low batch count",
            })?;
        let expected_shared = self
            .forward_low
            .sequence_len
            .checked_mul(self.forward_low.workgroup_grouping.transforms_per_workgroup)
            .and_then(|value| value.checked_mul(self.scalar.complex_bytes()))
            .and_then(|value| value.checked_mul(2))
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "two-upload multi-kernel validation shared memory",
            })?;
        let twiddle_source = self.forward_low.twiddle_source;
        if self.kernel_count <= 1
            || self.sequence_len != self.forward_mapping.logical_len
            || self.sequence_len != self.inverse_mapping.logical_len
            || self.forward_mapping.left_len != self.inverse_mapping.left_len
            || self.forward_mapping.right_len != self.inverse_mapping.right_len
            || self.forward_mapping.outer_batch_count != 1
            || self.inverse_mapping.outer_batch_count != self.kernel_count
            || !matches!(self.scalar, ScalarType::F32 | ScalarType::F64)
            || self.forward_high.scalar != self.scalar
            || self.forward_low.scalar != self.scalar
            || self.inverse_low.scalar != self.scalar
            || self.inverse_high.scalar != self.scalar
            || self.forward_high.twiddle_source != twiddle_source
            || self.inverse_low.twiddle_source != twiddle_source
            || self.inverse_high.twiddle_source != twiddle_source
            || self.forward_high.direction != Direction::Forward
            || self.forward_low.direction != Direction::Forward
            || self.inverse_low.direction != Direction::Inverse
            || self.inverse_high.direction != Direction::Inverse
            || self.forward_high.sequence_len != self.forward_mapping.right_len
            || self.inverse_high.sequence_len != self.inverse_mapping.right_len
            || self.forward_low.sequence_len != self.forward_mapping.left_len
            || self.inverse_low.sequence_len != self.inverse_mapping.left_len
            || self.forward_high.batch_count != forward_high_batches
            || self.forward_low.batch_count != forward_low_batches
            || self.inverse_high.batch_count != inverse_high_batches
            || self.inverse_low.batch_count != inverse_low_batches
            || self.forward_high.io_mapping
                != StockhamIoMapping::FourStepRight(self.forward_mapping)
            || self.forward_low.io_mapping != StockhamIoMapping::FourStepLeft(self.forward_mapping)
            || self.inverse_low.io_mapping != StockhamIoMapping::FourStepLeft(self.inverse_mapping)
            || self.inverse_high.io_mapping
                != StockhamIoMapping::FourStepRightPreTwiddle(FourStepPreTwiddleMapping {
                    four_step: self.inverse_mapping,
                    direction: Direction::Inverse,
                })
            || stockham_store_normalize(&self.forward_high)?
            || stockham_store_normalize(&self.forward_low)?
            || !stockham_store_normalize(&self.inverse_low)?
            || stockham_store_normalize(&self.inverse_high)?
            || self.forward_low.workgroup_grouping != self.inverse_low.workgroup_grouping
            || self.forward_low.workgroup_size != self.special_workgroup_size
            || self.inverse_low.workgroup_size != self.special_workgroup_size
            || self.forward_low.dispatch != self.special_dispatch
            || self.required_shared_memory_bytes != expected_shared
        {
            return Err(VkFftError::InvalidKernelIr(
                "two-upload multi-kernel performConvolution Stockham metadata is inconsistent",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConvolutionThreeUploadStockhamIr {
    pub name: String,
    pub sequence_len: usize,
    pub batch_count: usize,
    pub scalar: ScalarType,
    pub policy: ConvolutionMultiplyPolicy,
    pub mapping: ThreeUploadFourStepMapping,
    pub forward_high: KernelIr,
    pub forward_middle: KernelIr,
    pub forward_low: KernelIr,
    pub inverse_low: KernelIr,
    pub inverse_middle: KernelIr,
    pub inverse_high: KernelIr,
    pub special_workgroup_size: WorkgroupSize,
    pub special_dispatch: DispatchGeometry,
    pub required_shared_memory_bytes: usize,
}

impl ConvolutionThreeUploadStockhamIr {
    fn try_new(
        forward: &crate::recursive_ir::RecursiveFftIr,
        inverse: &crate::recursive_ir::RecursiveFftIr,
        policy: ConvolutionMultiplyPolicy,
        device: DeviceProfile,
    ) -> Result<Option<Self>> {
        policy.validate()?;
        let Some(forward_uploads) = forward.four_step_stockham_upload_kernels()? else {
            return Ok(None);
        };
        let Some(inverse_uploads) = inverse.four_step_stockham_upload_kernels()? else {
            return Ok(None);
        };
        let [forward_high, forward_middle, forward_low] = forward_uploads.as_slice() else {
            return Ok(None);
        };
        let [inverse_high, inverse_middle, inverse_low] = inverse_uploads.as_slice() else {
            return Ok(None);
        };
        let scalar = forward_low.scalar;
        let twiddle_source = forward_low.twiddle_source;
        if scalar != ScalarType::F32
            || [
                forward_high,
                forward_middle,
                inverse_high,
                inverse_middle,
                inverse_low,
            ]
            .iter()
            .any(|kernel| kernel.scalar != scalar)
            || [
                forward_high,
                forward_middle,
                forward_low,
                inverse_high,
                inverse_middle,
                inverse_low,
            ]
            .iter()
            .any(|kernel| {
                kernel.twiddle_source != twiddle_source
                    || kernel.input_modifier != StockhamInputModifier::None
                    || kernel.output_modifier != StockhamOutputModifier::None
            })
            || stockham_store_normalize(forward_high)?
            || stockham_store_normalize(forward_middle)?
            || stockham_store_normalize(forward_low)?
            || !stockham_store_normalize(inverse_low)?
        {
            return Ok(None);
        }
        let StockhamIoMapping::FourStepThreeUpload0(mapping) = forward_low.io_mapping else {
            return Ok(None);
        };
        if forward_high.io_mapping != StockhamIoMapping::FourStepThreeUpload2(mapping)
            || forward_middle.io_mapping != StockhamIoMapping::FourStepThreeUpload1(mapping)
            || inverse_high.io_mapping != StockhamIoMapping::FourStepThreeUpload2(mapping)
            || inverse_middle.io_mapping != StockhamIoMapping::FourStepThreeUpload1(mapping)
            || inverse_low.io_mapping != StockhamIoMapping::FourStepThreeUpload0(mapping)
            || mapping.logical_len != forward.logical_len
            || mapping.logical_len != inverse.logical_len
            || mapping.outer_batch_count != forward.batch_count
            || mapping.outer_batch_count != inverse.batch_count
            || forward_low.workgroup_grouping != inverse_low.workgroup_grouping
            || forward_low.workgroup_size != inverse_low.workgroup_size
            || forward_low.dispatch != inverse_low.dispatch
            || forward_low.stockham_shared_layout != inverse_low.stockham_shared_layout
        {
            return Ok(None);
        }
        let low_layout_supported = match forward_low.execution_layout {
            StockhamExecutionLayout::RegisterSingleShared => {
                inverse_low.execution_layout == StockhamExecutionLayout::RegisterSingleShared
                    && forward_low.shared_memory.buffers == 1
                    && inverse_low.shared_memory.buffers == 1
                    && forward_low.register_stockham_stages()?.is_some()
                    && inverse_low.register_stockham_stages()?.is_some()
            }
            StockhamExecutionLayout::SharedPingPong => {
                inverse_low.execution_layout == StockhamExecutionLayout::SharedPingPong
                    && forward_low.shared_memory.buffers == 2
                    && inverse_low.shared_memory.buffers == 2
            }
            _ => false,
        };
        if !low_layout_supported {
            return Ok(None);
        }
        let inverse_middle = inverse_middle
            .clone()
            .with_stockham_io_mapping(StockhamIoMapping::FourStepThreeUploadPreTwiddle(
                ThreeUploadPreTwiddleMapping {
                    three_upload: mapping,
                    axis_upload_id: 1,
                    direction: Direction::Inverse,
                },
            ))?
            .with_store_normalization(false)?;
        let inverse_high = inverse_high
            .clone()
            .with_stockham_io_mapping(StockhamIoMapping::FourStepThreeUploadPreTwiddle(
                ThreeUploadPreTwiddleMapping {
                    three_upload: mapping,
                    axis_upload_id: 2,
                    direction: Direction::Inverse,
                },
            ))?
            .with_store_normalization(false)?;
        let required_shared_memory_bytes = forward_low.required_shared_memory_bytes()?;
        // `shared_memory_pow2_bytes` participates in upload selection, but upstream
        // shader shared-stride fallback compares the concrete padded allocation against
        // the device's full `sharedMemSize`. N4096 therefore keeps its 4352-complex
        // bank-padded one-shared layout on a 48 KiB device even though the scheduling
        // threshold used 32 KiB for power-of-two factor selection.
        let available_shared = device.shared_memory_bytes;
        if required_shared_memory_bytes == 0 || required_shared_memory_bytes > available_shared {
            return Ok(None);
        }
        let ir = Self {
            name: format!("vkfft_convolution_three_upload_{}", mapping.logical_len),
            sequence_len: mapping.logical_len,
            batch_count: mapping.outer_batch_count,
            scalar,
            policy,
            mapping,
            forward_high: forward_high.clone(),
            forward_middle: forward_middle.clone(),
            forward_low: forward_low.clone(),
            inverse_low: inverse_low.clone(),
            inverse_middle,
            inverse_high,
            special_workgroup_size: forward_low.workgroup_size,
            special_dispatch: forward_low.dispatch,
            required_shared_memory_bytes,
        };
        ir.validate()?;
        Ok(Some(ir))
    }

    pub fn special_twiddle_lut_len(&self) -> Result<Option<usize>> {
        let period = self.forward_low.twiddle_lut_len();
        if self.forward_middle.twiddle_lut_len() != period
            || self.forward_high.twiddle_lut_len() != period
            || self.inverse_low.twiddle_lut_len() != period
            || self.inverse_middle.twiddle_lut_len() != period
            || self.inverse_high.twiddle_lut_len() != period
        {
            return Err(VkFftError::InvalidKernelIr(
                "three-upload performConvolution Stockham twiddle LUT periods are inconsistent",
            ));
        }
        Ok(period)
    }

    pub fn validate(&self) -> Result<()> {
        self.policy.validate()?;
        self.mapping.validate()?;
        self.forward_high.validate()?;
        self.forward_middle.validate()?;
        self.forward_low.validate()?;
        self.inverse_low.validate()?;
        self.inverse_middle.validate()?;
        self.inverse_high.validate()?;
        let _ = self.special_twiddle_lut_len()?;
        let low_layout_valid = match self.forward_low.execution_layout {
            StockhamExecutionLayout::RegisterSingleShared => {
                self.inverse_low.execution_layout == StockhamExecutionLayout::RegisterSingleShared
                    && self.forward_low.shared_memory.buffers == 1
                    && self.inverse_low.shared_memory.buffers == 1
                    && self.forward_low.register_stockham_stages()?.is_some()
                    && self.inverse_low.register_stockham_stages()?.is_some()
            }
            StockhamExecutionLayout::SharedPingPong => {
                self.inverse_low.execution_layout == StockhamExecutionLayout::SharedPingPong
                    && self.forward_low.shared_memory.buffers == 2
                    && self.inverse_low.shared_memory.buffers == 2
            }
            _ => false,
        };
        let [a, b, c] = self.mapping.axis_split;
        let high_batches = self
            .batch_count
            .checked_mul(a)
            .and_then(|value| value.checked_mul(b))
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "three-upload convolution high batch count",
            })?;
        let middle_batches = self
            .batch_count
            .checked_mul(c)
            .and_then(|value| value.checked_mul(a))
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "three-upload convolution middle batch count",
            })?;
        let low_batches = self
            .batch_count
            .checked_mul(c)
            .and_then(|value| value.checked_mul(b))
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "three-upload convolution low batch count",
            })?;
        if self.sequence_len != self.mapping.logical_len
            || self.scalar != ScalarType::F32
            || self.forward_high.direction != Direction::Forward
            || self.forward_middle.direction != Direction::Forward
            || self.forward_low.direction != Direction::Forward
            || self.inverse_low.direction != Direction::Inverse
            || self.inverse_middle.direction != Direction::Inverse
            || self.inverse_high.direction != Direction::Inverse
            || self.forward_high.sequence_len != c
            || self.inverse_high.sequence_len != c
            || self.forward_middle.sequence_len != b
            || self.inverse_middle.sequence_len != b
            || self.forward_low.sequence_len != a
            || self.inverse_low.sequence_len != a
            || self.forward_high.batch_count != high_batches
            || self.inverse_high.batch_count != high_batches
            || self.forward_middle.batch_count != middle_batches
            || self.inverse_middle.batch_count != middle_batches
            || self.forward_low.batch_count != low_batches
            || self.inverse_low.batch_count != low_batches
            || self.forward_high.io_mapping != StockhamIoMapping::FourStepThreeUpload2(self.mapping)
            || self.forward_middle.io_mapping
                != StockhamIoMapping::FourStepThreeUpload1(self.mapping)
            || self.forward_low.io_mapping != StockhamIoMapping::FourStepThreeUpload0(self.mapping)
            || self.inverse_low.io_mapping != StockhamIoMapping::FourStepThreeUpload0(self.mapping)
            || self.inverse_middle.io_mapping
                != StockhamIoMapping::FourStepThreeUploadPreTwiddle(ThreeUploadPreTwiddleMapping {
                    three_upload: self.mapping,
                    axis_upload_id: 1,
                    direction: Direction::Inverse,
                })
            || self.inverse_high.io_mapping
                != StockhamIoMapping::FourStepThreeUploadPreTwiddle(ThreeUploadPreTwiddleMapping {
                    three_upload: self.mapping,
                    axis_upload_id: 2,
                    direction: Direction::Inverse,
                })
            || stockham_store_normalize(&self.forward_high)?
            || stockham_store_normalize(&self.forward_middle)?
            || stockham_store_normalize(&self.forward_low)?
            || !stockham_store_normalize(&self.inverse_low)?
            || stockham_store_normalize(&self.inverse_middle)?
            || stockham_store_normalize(&self.inverse_high)?
            || !low_layout_valid
            || self.forward_high.twiddle_source != self.forward_low.twiddle_source
            || self.forward_middle.twiddle_source != self.forward_low.twiddle_source
            || self.inverse_low.twiddle_source != self.forward_low.twiddle_source
            || self.inverse_middle.twiddle_source != self.forward_low.twiddle_source
            || self.inverse_high.twiddle_source != self.forward_low.twiddle_source
            || self.forward_low.workgroup_grouping != self.inverse_low.workgroup_grouping
            || self.forward_low.workgroup_size != self.special_workgroup_size
            || self.inverse_low.workgroup_size != self.special_workgroup_size
            || self.forward_low.dispatch != self.special_dispatch
            || self.inverse_low.dispatch != self.special_dispatch
            || self.forward_low.stockham_shared_layout != self.inverse_low.stockham_shared_layout
            || self.required_shared_memory_bytes
                != self.forward_low.required_shared_memory_bytes()?
        {
            return Err(VkFftError::InvalidKernelIr(
                "three-upload performConvolution Stockham metadata is inconsistent",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConvolutionThreeUploadMultiKernelStockhamIr {
    pub name: String,
    pub sequence_len: usize,
    pub kernel_count: usize,
    pub scalar: ScalarType,
    pub policy: ConvolutionMultiplyPolicy,
    pub forward_mapping: ThreeUploadFourStepMapping,
    pub inverse_mapping: ThreeUploadFourStepMapping,
    pub forward_high: KernelIr,
    pub forward_middle: KernelIr,
    pub forward_low: KernelIr,
    pub inverse_low: KernelIr,
    pub inverse_middle: KernelIr,
    pub inverse_high: KernelIr,
    pub special_workgroup_size: WorkgroupSize,
    pub special_dispatch: DispatchGeometry,
    pub required_shared_memory_bytes: usize,
}

impl ConvolutionThreeUploadMultiKernelStockhamIr {
    fn try_new(
        forward: &crate::recursive_ir::RecursiveFftIr,
        inverse: &crate::recursive_ir::RecursiveFftIr,
        kernel_count: usize,
        policy: ConvolutionMultiplyPolicy,
        device: DeviceProfile,
    ) -> Result<Option<Self>> {
        policy.validate()?;
        if kernel_count <= 1 || forward.batch_count != 1 || inverse.batch_count != kernel_count {
            return Ok(None);
        }
        let Some(forward_uploads) = forward.four_step_stockham_upload_kernels()? else {
            return Ok(None);
        };
        let Some(inverse_uploads) = inverse.four_step_stockham_upload_kernels()? else {
            return Ok(None);
        };
        let [forward_high, forward_middle, forward_low] = forward_uploads.as_slice() else {
            return Ok(None);
        };
        let [inverse_high, inverse_middle, inverse_low] = inverse_uploads.as_slice() else {
            return Ok(None);
        };
        let StockhamIoMapping::FourStepThreeUpload0(forward_mapping) = forward_low.io_mapping
        else {
            return Ok(None);
        };
        let StockhamIoMapping::FourStepThreeUpload0(inverse_mapping) = inverse_low.io_mapping
        else {
            return Ok(None);
        };
        let scalar = forward_low.scalar;
        if scalar != ScalarType::F32
            || [
                forward_high,
                forward_middle,
                inverse_high,
                inverse_middle,
                inverse_low,
            ]
            .iter()
            .any(|kernel| kernel.scalar != scalar)
            || [
                forward_high,
                forward_middle,
                forward_low,
                inverse_high,
                inverse_middle,
                inverse_low,
            ]
            .iter()
            .any(|kernel| {
                kernel.twiddle_source != StockhamTwiddleSource::OnTheFly
                    || kernel.input_modifier != StockhamInputModifier::None
                    || kernel.output_modifier != StockhamOutputModifier::None
            })
            || stockham_store_normalize(forward_high)?
            || stockham_store_normalize(forward_middle)?
            || stockham_store_normalize(forward_low)?
            || !stockham_store_normalize(inverse_low)?
            || forward_mapping.logical_len != inverse_mapping.logical_len
            || forward_mapping.axis_split != inverse_mapping.axis_split
            || forward_mapping.outer_batch_count != 1
            || inverse_mapping.outer_batch_count != kernel_count
            || forward_high.io_mapping != StockhamIoMapping::FourStepThreeUpload2(forward_mapping)
            || forward_middle.io_mapping != StockhamIoMapping::FourStepThreeUpload1(forward_mapping)
            || inverse_high.io_mapping != StockhamIoMapping::FourStepThreeUpload2(inverse_mapping)
            || inverse_middle.io_mapping != StockhamIoMapping::FourStepThreeUpload1(inverse_mapping)
            || forward_low.execution_layout != StockhamExecutionLayout::RegisterSingleShared
            || inverse_low.execution_layout != StockhamExecutionLayout::RegisterSingleShared
            || forward_low.shared_memory.buffers != 1
            || inverse_low.shared_memory.buffers != 1
            || forward_low.workgroup_grouping != inverse_low.workgroup_grouping
            || forward_low.workgroup_size != inverse_low.workgroup_size
            || forward_low.stockham_shared_layout != inverse_low.stockham_shared_layout
            || forward_low.register_stockham_stages()?.is_none()
            || inverse_low.register_stockham_stages()?.is_none()
        {
            return Ok(None);
        }
        let inverse_middle = inverse_middle
            .clone()
            .with_stockham_io_mapping(StockhamIoMapping::FourStepThreeUploadPreTwiddle(
                ThreeUploadPreTwiddleMapping {
                    three_upload: inverse_mapping,
                    axis_upload_id: 1,
                    direction: Direction::Inverse,
                },
            ))?
            .with_store_normalization(false)?;
        let inverse_high = inverse_high
            .clone()
            .with_stockham_io_mapping(StockhamIoMapping::FourStepThreeUploadPreTwiddle(
                ThreeUploadPreTwiddleMapping {
                    three_upload: inverse_mapping,
                    axis_upload_id: 2,
                    direction: Direction::Inverse,
                },
            ))?
            .with_store_normalization(false)?;
        let required_shared_memory_bytes = forward_low.required_shared_memory_bytes()?;
        if required_shared_memory_bytes == 0
            || required_shared_memory_bytes > device.shared_memory_bytes
        {
            return Ok(None);
        }
        let ir = Self {
            name: format!(
                "vkfft_convolution_three_upload_multi_kernel_{}_k{}",
                forward_mapping.logical_len, kernel_count
            ),
            sequence_len: forward_mapping.logical_len,
            kernel_count,
            scalar,
            policy,
            forward_mapping,
            inverse_mapping,
            forward_high: forward_high.clone(),
            forward_middle: forward_middle.clone(),
            forward_low: forward_low.clone(),
            inverse_low: inverse_low.clone(),
            inverse_middle,
            inverse_high,
            special_workgroup_size: forward_low.workgroup_size,
            special_dispatch: forward_low.dispatch,
            required_shared_memory_bytes,
        };
        ir.validate()?;
        Ok(Some(ir))
    }

    pub fn validate(&self) -> Result<()> {
        self.policy.validate()?;
        self.forward_mapping.validate()?;
        self.inverse_mapping.validate()?;
        self.forward_high.validate()?;
        self.forward_middle.validate()?;
        self.forward_low.validate()?;
        self.inverse_low.validate()?;
        self.inverse_middle.validate()?;
        self.inverse_high.validate()?;
        let [a, b, c] = self.forward_mapping.axis_split;
        let forward_high_batches = a.checked_mul(b).ok_or(VkFftError::ArithmeticOverflow {
            operation: "three-upload multi-kernel forward high batch count",
        })?;
        let forward_middle_batches = c.checked_mul(a).ok_or(VkFftError::ArithmeticOverflow {
            operation: "three-upload multi-kernel forward middle batch count",
        })?;
        let forward_low_batches = c.checked_mul(b).ok_or(VkFftError::ArithmeticOverflow {
            operation: "three-upload multi-kernel forward low batch count",
        })?;
        let inverse_high_batches = self.kernel_count.checked_mul(forward_high_batches).ok_or(
            VkFftError::ArithmeticOverflow {
                operation: "three-upload multi-kernel inverse high batch count",
            },
        )?;
        let inverse_middle_batches = self
            .kernel_count
            .checked_mul(forward_middle_batches)
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "three-upload multi-kernel inverse middle batch count",
            })?;
        let inverse_low_batches = self.kernel_count.checked_mul(forward_low_batches).ok_or(
            VkFftError::ArithmeticOverflow {
                operation: "three-upload multi-kernel inverse low batch count",
            },
        )?;
        if self.kernel_count <= 1
            || self.sequence_len != self.forward_mapping.logical_len
            || self.sequence_len != self.inverse_mapping.logical_len
            || self.forward_mapping.axis_split != self.inverse_mapping.axis_split
            || self.forward_mapping.outer_batch_count != 1
            || self.inverse_mapping.outer_batch_count != self.kernel_count
            || self.scalar != ScalarType::F32
            || self.forward_high.sequence_len != c
            || self.inverse_high.sequence_len != c
            || self.forward_middle.sequence_len != b
            || self.inverse_middle.sequence_len != b
            || self.forward_low.sequence_len != a
            || self.inverse_low.sequence_len != a
            || self.forward_high.batch_count != forward_high_batches
            || self.forward_middle.batch_count != forward_middle_batches
            || self.forward_low.batch_count != forward_low_batches
            || self.inverse_high.batch_count != inverse_high_batches
            || self.inverse_middle.batch_count != inverse_middle_batches
            || self.inverse_low.batch_count != inverse_low_batches
            || self.forward_high.io_mapping
                != StockhamIoMapping::FourStepThreeUpload2(self.forward_mapping)
            || self.forward_middle.io_mapping
                != StockhamIoMapping::FourStepThreeUpload1(self.forward_mapping)
            || self.forward_low.io_mapping
                != StockhamIoMapping::FourStepThreeUpload0(self.forward_mapping)
            || self.inverse_low.io_mapping
                != StockhamIoMapping::FourStepThreeUpload0(self.inverse_mapping)
            || self.inverse_middle.io_mapping
                != StockhamIoMapping::FourStepThreeUploadPreTwiddle(ThreeUploadPreTwiddleMapping {
                    three_upload: self.inverse_mapping,
                    axis_upload_id: 1,
                    direction: Direction::Inverse,
                })
            || self.inverse_high.io_mapping
                != StockhamIoMapping::FourStepThreeUploadPreTwiddle(ThreeUploadPreTwiddleMapping {
                    three_upload: self.inverse_mapping,
                    axis_upload_id: 2,
                    direction: Direction::Inverse,
                })
            || stockham_store_normalize(&self.forward_high)?
            || stockham_store_normalize(&self.forward_middle)?
            || stockham_store_normalize(&self.forward_low)?
            || !stockham_store_normalize(&self.inverse_low)?
            || stockham_store_normalize(&self.inverse_middle)?
            || stockham_store_normalize(&self.inverse_high)?
            || self.forward_low.execution_layout != StockhamExecutionLayout::RegisterSingleShared
            || self.inverse_low.execution_layout != StockhamExecutionLayout::RegisterSingleShared
            || self.forward_low.shared_memory.buffers != 1
            || self.inverse_low.shared_memory.buffers != 1
            || self.forward_low.workgroup_grouping != self.inverse_low.workgroup_grouping
            || self.forward_low.workgroup_size != self.special_workgroup_size
            || self.inverse_low.workgroup_size != self.special_workgroup_size
            || self.forward_low.dispatch != self.special_dispatch
            || self.forward_low.stockham_shared_layout != self.inverse_low.stockham_shared_layout
            || self.required_shared_memory_bytes
                != self.forward_low.required_shared_memory_bytes()?
        {
            return Err(VkFftError::InvalidKernelIr(
                "three-upload multi-kernel performConvolution metadata is inconsistent",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ConvolutionDirectRaderStepIr {
    pub name: String,
    pub prime: usize,
    pub batch_count: usize,
    pub scalar: ScalarType,
    pub policy: ConvolutionMultiplyPolicy,
    pub forward: RaderDirectIr,
    pub inverse: RaderDirectIr,
    pub workgroup_size: WorkgroupSize,
    pub dispatch: DispatchGeometry,
    pub required_shared_memory_bytes: usize,
}

impl ConvolutionDirectRaderStepIr {
    fn try_new(
        forward: &RaderDirectIr,
        inverse: &RaderDirectIr,
        policy: ConvolutionMultiplyPolicy,
        device: DeviceProfile,
    ) -> Result<Option<Self>> {
        forward.validate()?;
        inverse.validate()?;
        policy.validate()?;
        // Pinned VkFFT puts application convolutionStep around a Direct-Rader radix in
        // one shader. Keep the first Rust slice deliberately narrow: one logical batch,
        // contiguous ordinary storage, one application kernel, and no grouped tail.
        if forward.batch_count != 1
            || inverse.batch_count != 1
            || forward.direction != Direction::Forward
            || inverse.direction != Direction::Inverse
            || forward.prime != inverse.prime
            || forward.scalar != inverse.scalar
            || forward.input_storage_scalar != forward.scalar
            || forward.output_storage_scalar != forward.scalar
            || inverse.input_storage_scalar != inverse.scalar
            || inverse.output_storage_scalar != inverse.scalar
            || forward.io_mapping != StockhamIoMapping::Contiguous
            || inverse.io_mapping != StockhamIoMapping::Contiguous
            || forward.normalize
            || !inverse.normalize
            || forward.workgroup_size != inverse.workgroup_size
            || forward.dispatch != inverse.dispatch
            || forward.axis_batch_block != inverse.axis_batch_block
            || forward
                .axis_batch_block
                .is_some_and(|block| block.grouped_batch != 1)
            || forward.table.permutation != inverse.table.permutation
        {
            return Ok(None);
        }
        let required_shared_memory_bytes = forward
            .prime
            .checked_mul(forward.scalar.complex_bytes())
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "performConvolution Direct-Rader shared spectrum",
            })?;
        if required_shared_memory_bytes > device.shared_memory_bytes {
            return Ok(None);
        }
        let step = Self {
            name: format!("vkfft_convolution_step_direct_rader_{}", forward.prime),
            prime: forward.prime,
            batch_count: 1,
            scalar: forward.scalar,
            policy,
            forward: forward.clone(),
            inverse: inverse.clone(),
            workgroup_size: forward.workgroup_size,
            dispatch: forward.dispatch,
            required_shared_memory_bytes,
        };
        step.validate()?;
        Ok(Some(step))
    }

    pub fn validate(&self) -> Result<()> {
        self.forward.validate()?;
        self.inverse.validate()?;
        self.policy.validate()?;
        if self.prime < 2
            || self.batch_count != 1
            || self.forward.prime != self.prime
            || self.inverse.prime != self.prime
            || self.forward.batch_count != 1
            || self.inverse.batch_count != 1
            || self.forward.direction != Direction::Forward
            || self.inverse.direction != Direction::Inverse
            || self.forward.scalar != self.scalar
            || self.inverse.scalar != self.scalar
            || self.forward.input_storage_scalar != self.scalar
            || self.forward.output_storage_scalar != self.scalar
            || self.inverse.input_storage_scalar != self.scalar
            || self.inverse.output_storage_scalar != self.scalar
            || self.forward.io_mapping != StockhamIoMapping::Contiguous
            || self.inverse.io_mapping != StockhamIoMapping::Contiguous
            || self.forward.normalize
            || !self.inverse.normalize
            || self.forward.workgroup_size != self.workgroup_size
            || self.inverse.workgroup_size != self.workgroup_size
            || self.forward.dispatch != self.dispatch
            || self.inverse.dispatch != self.dispatch
            || self.forward.axis_batch_block != self.inverse.axis_batch_block
            || self
                .forward
                .axis_batch_block
                .is_some_and(|block| block.grouped_batch != 1)
            || self.forward.table.permutation != self.inverse.table.permutation
            || self.required_shared_memory_bytes != self.prime * self.scalar.complex_bytes()
        {
            return Err(VkFftError::InvalidKernelIr(
                "fused performConvolution Direct-Rader metadata is inconsistent",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ConvolutionDirectRaderMultiKernelStepIr {
    pub name: String,
    pub prime: usize,
    pub kernel_count: usize,
    pub scalar: ScalarType,
    pub policy: ConvolutionMultiplyPolicy,
    /// The batch-1 physical forward transform owned by the fused launch.
    pub forward: RaderDirectIr,
    /// Semantic K-batch inverse metadata; its standalone launch geometry is not inherited.
    pub inverse: RaderDirectIr,
    pub workgroup_size: WorkgroupSize,
    pub dispatch: DispatchGeometry,
    pub required_shared_memory_bytes: usize,
}

impl ConvolutionDirectRaderMultiKernelStepIr {
    fn try_new(
        forward: &RaderDirectIr,
        inverse: &RaderDirectIr,
        kernel_count: usize,
        policy: ConvolutionMultiplyPolicy,
        device: DeviceProfile,
    ) -> Result<Option<Self>> {
        forward.validate()?;
        inverse.validate()?;
        policy.validate()?;
        if kernel_count <= 1
            || forward.batch_count != 1
            || inverse.batch_count != kernel_count
            || forward.direction != Direction::Forward
            || inverse.direction != Direction::Inverse
            || forward.prime != inverse.prime
            || forward.scalar != inverse.scalar
            || forward.input_storage_scalar != forward.scalar
            || forward.output_storage_scalar != forward.scalar
            || inverse.input_storage_scalar != inverse.scalar
            || inverse.output_storage_scalar != inverse.scalar
            || forward.io_mapping != StockhamIoMapping::Contiguous
            || inverse.io_mapping != StockhamIoMapping::Contiguous
            || forward.normalize
            || !inverse.normalize
            || forward
                .axis_batch_block
                .is_some_and(|block| block.grouped_batch != 1)
            || forward.table.permutation != inverse.table.permutation
        {
            return Ok(None);
        }
        let required_shared_memory_bytes = forward
            .prime
            .checked_mul(forward.scalar.complex_bytes())
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "multi-kernel performConvolution Direct-Rader shared spectrum",
            })?;
        if required_shared_memory_bytes > device.shared_memory_bytes {
            return Ok(None);
        }
        let step = Self {
            name: format!(
                "vkfft_convolution_step_direct_rader_{}_k{}",
                forward.prime, kernel_count
            ),
            prime: forward.prime,
            kernel_count,
            scalar: forward.scalar,
            policy,
            forward: forward.clone(),
            inverse: inverse.clone(),
            workgroup_size: forward.workgroup_size,
            dispatch: forward.dispatch,
            required_shared_memory_bytes,
        };
        step.validate()?;
        Ok(Some(step))
    }

    pub fn validate(&self) -> Result<()> {
        self.forward.validate()?;
        self.inverse.validate()?;
        self.policy.validate()?;
        let expected_shared = self.prime.checked_mul(self.scalar.complex_bytes()).ok_or(
            VkFftError::ArithmeticOverflow {
                operation: "multi-kernel Direct-Rader validation shared bytes",
            },
        )?;
        if self.prime < 2
            || self.kernel_count <= 1
            || self.forward.prime != self.prime
            || self.inverse.prime != self.prime
            || self.forward.batch_count != 1
            || self.inverse.batch_count != self.kernel_count
            || self.forward.direction != Direction::Forward
            || self.inverse.direction != Direction::Inverse
            || self.forward.scalar != self.scalar
            || self.inverse.scalar != self.scalar
            || self.forward.input_storage_scalar != self.scalar
            || self.forward.output_storage_scalar != self.scalar
            || self.inverse.input_storage_scalar != self.scalar
            || self.inverse.output_storage_scalar != self.scalar
            || self.forward.io_mapping != StockhamIoMapping::Contiguous
            || self.inverse.io_mapping != StockhamIoMapping::Contiguous
            || self.forward.normalize
            || !self.inverse.normalize
            || self.forward.workgroup_size != self.workgroup_size
            || self.forward.dispatch != self.dispatch
            || self
                .forward
                .axis_batch_block
                .is_some_and(|block| block.grouped_batch != 1)
            || self.forward.table.permutation != self.inverse.table.permutation
            || self.required_shared_memory_bytes != expected_shared
        {
            return Err(VkFftError::InvalidKernelIr(
                "fused multi-kernel performConvolution Direct-Rader metadata is inconsistent",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ConvolutionFftRaderStepIr {
    pub name: String,
    pub prime: usize,
    pub convolution_len: usize,
    pub batch_count: usize,
    pub scalar: ScalarType,
    pub policy: ConvolutionMultiplyPolicy,
    pub forward: RaderFftPipelineIr,
    pub inverse: RaderFftPipelineIr,
    pub workgroup_size: WorkgroupSize,
    pub dispatch: DispatchGeometry,
    pub required_shared_memory_bytes: usize,
}

fn fft_rader_stockham_pair(pipeline: &RaderFftPipelineIr) -> Option<(&KernelIr, &KernelIr)> {
    let forward = pipeline.forward_recursive()?;
    let inverse = pipeline.inverse_recursive()?;
    let RecursiveFftNodeIr::Stockham(forward) = &forward.root else {
        return None;
    };
    let RecursiveFftNodeIr::Stockham(inverse) = &inverse.root else {
        return None;
    };
    Some((forward.as_ref(), inverse.as_ref()))
}

fn fft_rader_scatter_normalize(pipeline: &RaderFftPipelineIr) -> Result<bool> {
    match pipeline.scatter.operation {
        RaderFftPassOperation::Scatter { normalize } => Ok(normalize),
        _ => Err(VkFftError::InvalidKernelIr(
            "FFT-Rader performConvolution step lost its scatter operation",
        )),
    }
}

impl ConvolutionFftRaderStepIr {
    fn try_new(
        forward: &RaderFftPipelineIr,
        inverse: &RaderFftPipelineIr,
        policy: ConvolutionMultiplyPolicy,
        device: DeviceProfile,
    ) -> Result<Option<Self>> {
        forward.validate()?;
        inverse.validate()?;
        policy.validate()?;
        let (forward_inner_forward, forward_inner_inverse) = match fft_rader_stockham_pair(forward)
        {
            Some(pair) => pair,
            None => return Ok(None),
        };
        let (inverse_inner_forward, inverse_inner_inverse) = match fft_rader_stockham_pair(inverse)
        {
            Some(pair) => pair,
            None => return Ok(None),
        };
        let Some(forward_schedule) = forward.internal_register_schedule.as_ref() else {
            return Ok(None);
        };
        let Some(inverse_schedule) = inverse.internal_register_schedule.as_ref() else {
            return Ok(None);
        };
        let kernels = [
            forward_inner_forward,
            forward_inner_inverse,
            inverse_inner_forward,
            inverse_inner_inverse,
        ];
        let reference = forward_inner_forward;
        let supported_kernel = |kernel: &KernelIr| -> Result<bool> {
            Ok(kernel.sequence_len == forward.convolution_len
                && kernel.batch_count == 1
                && kernel.scalar == forward.scalar
                && kernel.workgroup_size == reference.workgroup_size
                && kernel.dispatch == reference.dispatch
                && kernel.workgroup_grouping == reference.workgroup_grouping
                && kernel.workgroup_grouping.transforms_per_workgroup == 1
                && kernel.workgroup_grouping.axis_layout
                    == StockhamWorkgroupAxisLayout::ThreadsXTransformsY
                && kernel.workgroup_size.y == 1
                && kernel.workgroup_size.z == 1
                && kernel.dispatch.x == 1
                && kernel.dispatch.y == 1
                && kernel.dispatch.z == 1
                && kernel.rader_transpose.is_none()
                && kernel.register_stockham_stages()?.is_some())
        };
        for kernel in kernels {
            if !supported_kernel(kernel)? {
                return Ok(None);
            }
        }
        if forward.batch_count != 1
            || inverse.batch_count != 1
            || forward.direction != Direction::Forward
            || inverse.direction != Direction::Inverse
            || forward.prime != inverse.prime
            || forward.convolution_len != inverse.convolution_len
            || forward.scalar != inverse.scalar
            || forward.input_storage_scalar != forward.scalar
            || forward.output_storage_scalar != forward.scalar
            || inverse.input_storage_scalar != inverse.scalar
            || inverse.output_storage_scalar != inverse.scalar
            || forward.io_mapping != StockhamIoMapping::Contiguous
            || inverse.io_mapping != StockhamIoMapping::Contiguous
            || forward.input_strategy != RaderFftInputStrategy::GeneratorOrderStockham
            || inverse.input_strategy != RaderFftInputStrategy::GeneratorOrderStockham
            || fft_rader_scatter_normalize(forward)?
            || !fft_rader_scatter_normalize(inverse)?
            || forward.table.permutation != inverse.table.permutation
            || forward_schedule != inverse_schedule
            || forward_schedule.container_fft_num != 1
            || forward_schedule.execution_container_fft_num != 1
            || forward_schedule.rader_transpose.is_some()
            || stockham_store_normalize(forward_inner_forward)?
            || !stockham_store_normalize(forward_inner_inverse)?
            || stockham_store_normalize(inverse_inner_forward)?
            || !stockham_store_normalize(inverse_inner_inverse)?
            || kernels
                .iter()
                .any(|kernel| kernel.twiddle_source != reference.twiddle_source)
        {
            return Ok(None);
        }
        let shared_elements = forward
            .convolution_len
            .checked_mul(2)
            .and_then(|value| value.checked_add(forward.prime))
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "performConvolution FFT-Rader shared element count",
            })?;
        let required_shared_memory_bytes = shared_elements
            .checked_mul(forward.scalar.complex_bytes())
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "performConvolution FFT-Rader shared byte count",
            })?;
        if required_shared_memory_bytes > device.shared_memory_bytes {
            return Ok(None);
        }
        let step = Self {
            name: format!("vkfft_convolution_step_fft_rader_{}", forward.prime),
            prime: forward.prime,
            convolution_len: forward.convolution_len,
            batch_count: 1,
            scalar: forward.scalar,
            policy,
            forward: forward.clone(),
            inverse: inverse.clone(),
            workgroup_size: reference.workgroup_size,
            dispatch: reference.dispatch,
            required_shared_memory_bytes,
        };
        step.validate()?;
        Ok(Some(step))
    }

    pub(crate) fn internal_stockham_kernels(&self) -> Result<[(&KernelIr, &KernelIr); 2]> {
        let forward = fft_rader_stockham_pair(&self.forward).ok_or(VkFftError::InvalidKernelIr(
            "fused performConvolution FFT-Rader forward pipeline lost Stockham children",
        ))?;
        let inverse = fft_rader_stockham_pair(&self.inverse).ok_or(VkFftError::InvalidKernelIr(
            "fused performConvolution FFT-Rader inverse pipeline lost Stockham children",
        ))?;
        Ok([forward, inverse])
    }

    pub fn requires_twiddle_lut(&self) -> Result<bool> {
        let pairs = self.internal_stockham_kernels()?;
        let first = pairs[0].0.twiddle_source;
        if pairs
            .iter()
            .flat_map(|(forward, inverse)| [*forward, *inverse])
            .any(|kernel| kernel.twiddle_source != first)
        {
            return Err(VkFftError::InvalidKernelIr(
                "fused performConvolution FFT-Rader children disagree on Stockham twiddle source",
            ));
        }
        Ok(first == StockhamTwiddleSource::LookupTable)
    }

    pub fn validate(&self) -> Result<()> {
        self.forward.validate()?;
        self.inverse.validate()?;
        self.policy.validate()?;
        let pairs = self.internal_stockham_kernels()?;
        let kernels = [pairs[0].0, pairs[0].1, pairs[1].0, pairs[1].1];
        let reference = kernels[0];
        let forward_schedule =
            self.forward
                .internal_register_schedule
                .as_ref()
                .ok_or(VkFftError::InvalidKernelIr(
                    "fused performConvolution FFT-Rader is missing its forward register schedule",
                ))?;
        let inverse_schedule =
            self.inverse
                .internal_register_schedule
                .as_ref()
                .ok_or(VkFftError::InvalidKernelIr(
                    "fused performConvolution FFT-Rader is missing its inverse register schedule",
                ))?;
        let expected_shared = self
            .convolution_len
            .checked_mul(2)
            .and_then(|value| value.checked_add(self.prime))
            .and_then(|value| value.checked_mul(self.scalar.complex_bytes()))
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "fused performConvolution FFT-Rader validation shared bytes",
            })?;
        if self.prime < 3
            || self.convolution_len + 1 != self.prime
            || self.batch_count != 1
            || self.forward.prime != self.prime
            || self.inverse.prime != self.prime
            || self.forward.convolution_len != self.convolution_len
            || self.inverse.convolution_len != self.convolution_len
            || self.forward.batch_count != 1
            || self.inverse.batch_count != 1
            || self.forward.direction != Direction::Forward
            || self.inverse.direction != Direction::Inverse
            || self.forward.scalar != self.scalar
            || self.inverse.scalar != self.scalar
            || self.forward.input_storage_scalar != self.scalar
            || self.forward.output_storage_scalar != self.scalar
            || self.inverse.input_storage_scalar != self.scalar
            || self.inverse.output_storage_scalar != self.scalar
            || self.forward.io_mapping != StockhamIoMapping::Contiguous
            || self.inverse.io_mapping != StockhamIoMapping::Contiguous
            || self.forward.input_strategy != RaderFftInputStrategy::GeneratorOrderStockham
            || self.inverse.input_strategy != RaderFftInputStrategy::GeneratorOrderStockham
            || fft_rader_scatter_normalize(&self.forward)?
            || !fft_rader_scatter_normalize(&self.inverse)?
            || self.forward.table.permutation != self.inverse.table.permutation
            || forward_schedule != inverse_schedule
            || forward_schedule.container_fft_num != 1
            || forward_schedule.execution_container_fft_num != 1
            || forward_schedule.rader_transpose.is_some()
            || reference.workgroup_size != self.workgroup_size
            || reference.dispatch != self.dispatch
            || reference.workgroup_grouping.transforms_per_workgroup != 1
            || reference.workgroup_grouping.axis_layout
                != StockhamWorkgroupAxisLayout::ThreadsXTransformsY
            || kernels.iter().any(|kernel| {
                kernel.sequence_len != self.convolution_len
                    || kernel.batch_count != 1
                    || kernel.scalar != self.scalar
                    || kernel.workgroup_size != self.workgroup_size
                    || kernel.dispatch != self.dispatch
                    || kernel.workgroup_grouping != reference.workgroup_grouping
                    || kernel.rader_transpose.is_some()
                    || kernel.register_stockham_stages().ok().flatten().is_none()
            })
            || stockham_store_normalize(pairs[0].0)?
            || !stockham_store_normalize(pairs[0].1)?
            || stockham_store_normalize(pairs[1].0)?
            || !stockham_store_normalize(pairs[1].1)?
            || self.required_shared_memory_bytes != expected_shared
        {
            return Err(VkFftError::InvalidKernelIr(
                "fused performConvolution FFT-Rader metadata is inconsistent",
            ));
        }
        let _ = self.requires_twiddle_lut()?;
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ConvolutionFftRaderMultiKernelStepIr {
    pub name: String,
    pub prime: usize,
    pub convolution_len: usize,
    pub kernel_count: usize,
    pub scalar: ScalarType,
    pub policy: ConvolutionMultiplyPolicy,
    pub forward: RaderFftPipelineIr,
    pub inverse: RaderFftPipelineIr,
    pub workgroup_size: WorkgroupSize,
    pub dispatch: DispatchGeometry,
    pub required_shared_memory_bytes: usize,
}

impl ConvolutionFftRaderMultiKernelStepIr {
    fn try_new(
        forward: &RaderFftPipelineIr,
        inverse: &RaderFftPipelineIr,
        kernel_count: usize,
        policy: ConvolutionMultiplyPolicy,
        device: DeviceProfile,
    ) -> Result<Option<Self>> {
        forward.validate()?;
        inverse.validate()?;
        policy.validate()?;
        if kernel_count <= 1 || forward.batch_count != 1 || inverse.batch_count != kernel_count {
            return Ok(None);
        }
        let Some((inner_forward, inner_inverse)) = fft_rader_stockham_pair(forward) else {
            return Ok(None);
        };
        let Some(schedule) = forward.internal_register_schedule.as_ref() else {
            return Ok(None);
        };
        let supported_physical = |kernel: &KernelIr| -> Result<bool> {
            Ok(kernel.sequence_len == forward.convolution_len
                && kernel.batch_count == 1
                && kernel.scalar == forward.scalar
                && kernel.workgroup_size == inner_forward.workgroup_size
                && kernel.dispatch == inner_forward.dispatch
                && kernel.workgroup_grouping == inner_forward.workgroup_grouping
                && kernel.workgroup_grouping.transforms_per_workgroup == 1
                && kernel.workgroup_grouping.axis_layout
                    == StockhamWorkgroupAxisLayout::ThreadsXTransformsY
                && kernel.workgroup_size.y == 1
                && kernel.workgroup_size.z == 1
                && kernel.dispatch == (DispatchGeometry { x: 1, y: 1, z: 1 })
                && kernel.rader_transpose.is_none()
                && kernel.register_stockham_stages()?.is_some())
        };
        if !supported_physical(inner_forward)?
            || !supported_physical(inner_inverse)?
            || forward.direction != Direction::Forward
            || inverse.direction != Direction::Inverse
            || forward.prime != inverse.prime
            || forward.convolution_len != inverse.convolution_len
            || forward.scalar != inverse.scalar
            || forward.input_storage_scalar != forward.scalar
            || forward.output_storage_scalar != forward.scalar
            || inverse.input_storage_scalar != inverse.scalar
            || inverse.output_storage_scalar != inverse.scalar
            || forward.io_mapping != StockhamIoMapping::Contiguous
            || inverse.io_mapping != StockhamIoMapping::Contiguous
            || forward.input_strategy != RaderFftInputStrategy::GeneratorOrderStockham
            || inverse.input_strategy != RaderFftInputStrategy::GeneratorOrderStockham
            || fft_rader_scatter_normalize(forward)?
            || !fft_rader_scatter_normalize(inverse)?
            || forward.table.permutation != inverse.table.permutation
            || schedule.container_fft_num != 1
            || schedule.execution_container_fft_num != 1
            || schedule.rader_transpose.is_some()
            || stockham_store_normalize(inner_forward)?
            || !stockham_store_normalize(inner_inverse)?
            || inner_forward.twiddle_source != inner_inverse.twiddle_source
        {
            return Ok(None);
        }
        let shared_elements = forward
            .convolution_len
            .checked_mul(2)
            .and_then(|value| value.checked_add(forward.prime))
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "multi-kernel performConvolution FFT-Rader shared element count",
            })?;
        let required_shared_memory_bytes = shared_elements
            .checked_mul(forward.scalar.complex_bytes())
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "multi-kernel performConvolution FFT-Rader shared byte count",
            })?;
        if required_shared_memory_bytes > device.shared_memory_bytes {
            return Ok(None);
        }
        let step = Self {
            name: format!(
                "vkfft_convolution_step_fft_rader_multi_kernel_{}_k{}",
                forward.prime, kernel_count
            ),
            prime: forward.prime,
            convolution_len: forward.convolution_len,
            kernel_count,
            scalar: forward.scalar,
            policy,
            forward: forward.clone(),
            inverse: inverse.clone(),
            workgroup_size: inner_forward.workgroup_size,
            dispatch: inner_forward.dispatch,
            required_shared_memory_bytes,
        };
        step.validate()?;
        Ok(Some(step))
    }

    pub(crate) fn physical_stockham_kernels(&self) -> Result<(&KernelIr, &KernelIr)> {
        fft_rader_stockham_pair(&self.forward).ok_or(VkFftError::InvalidKernelIr(
            "fused multi-kernel FFT-Rader forward pipeline lost Stockham children",
        ))
    }

    pub fn requires_twiddle_lut(&self) -> Result<bool> {
        let (forward, inverse) = self.physical_stockham_kernels()?;
        if forward.twiddle_source != inverse.twiddle_source {
            return Err(VkFftError::InvalidKernelIr(
                "fused multi-kernel FFT-Rader physical children disagree on Stockham twiddle source",
            ));
        }
        Ok(forward.twiddle_source == StockhamTwiddleSource::LookupTable)
    }

    pub fn validate(&self) -> Result<()> {
        self.forward.validate()?;
        self.inverse.validate()?;
        self.policy.validate()?;
        let (inner_forward, inner_inverse) = self.physical_stockham_kernels()?;
        let schedule =
            self.forward
                .internal_register_schedule
                .as_ref()
                .ok_or(VkFftError::InvalidKernelIr(
                    "fused multi-kernel FFT-Rader is missing its physical register schedule",
                ))?;
        let expected_shared = self
            .convolution_len
            .checked_mul(2)
            .and_then(|value| value.checked_add(self.prime))
            .and_then(|value| value.checked_mul(self.scalar.complex_bytes()))
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "multi-kernel FFT-Rader validation shared bytes",
            })?;
        if self.prime < 3
            || self.convolution_len + 1 != self.prime
            || self.kernel_count <= 1
            || self.forward.prime != self.prime
            || self.inverse.prime != self.prime
            || self.forward.convolution_len != self.convolution_len
            || self.inverse.convolution_len != self.convolution_len
            || self.forward.batch_count != 1
            || self.inverse.batch_count != self.kernel_count
            || self.forward.direction != Direction::Forward
            || self.inverse.direction != Direction::Inverse
            || self.forward.scalar != self.scalar
            || self.inverse.scalar != self.scalar
            || self.forward.input_storage_scalar != self.scalar
            || self.forward.output_storage_scalar != self.scalar
            || self.inverse.input_storage_scalar != self.scalar
            || self.inverse.output_storage_scalar != self.scalar
            || self.forward.io_mapping != StockhamIoMapping::Contiguous
            || self.inverse.io_mapping != StockhamIoMapping::Contiguous
            || self.forward.input_strategy != RaderFftInputStrategy::GeneratorOrderStockham
            || self.inverse.input_strategy != RaderFftInputStrategy::GeneratorOrderStockham
            || fft_rader_scatter_normalize(&self.forward)?
            || !fft_rader_scatter_normalize(&self.inverse)?
            || self.forward.table.permutation != self.inverse.table.permutation
            || schedule.container_fft_num != 1
            || schedule.execution_container_fft_num != 1
            || schedule.rader_transpose.is_some()
            || inner_forward.sequence_len != self.convolution_len
            || inner_inverse.sequence_len != self.convolution_len
            || inner_forward.batch_count != 1
            || inner_inverse.batch_count != 1
            || inner_forward.scalar != self.scalar
            || inner_inverse.scalar != self.scalar
            || inner_forward.workgroup_size != self.workgroup_size
            || inner_inverse.workgroup_size != self.workgroup_size
            || inner_forward.dispatch != self.dispatch
            || inner_inverse.dispatch != self.dispatch
            || inner_forward.workgroup_grouping != inner_inverse.workgroup_grouping
            || inner_forward.workgroup_grouping.transforms_per_workgroup != 1
            || inner_forward.workgroup_grouping.axis_layout
                != StockhamWorkgroupAxisLayout::ThreadsXTransformsY
            || inner_forward.rader_transpose.is_some()
            || inner_inverse.rader_transpose.is_some()
            || inner_forward.register_stockham_stages()?.is_none()
            || inner_inverse.register_stockham_stages()?.is_none()
            || stockham_store_normalize(inner_forward)?
            || !stockham_store_normalize(inner_inverse)?
            || inner_forward.twiddle_source != inner_inverse.twiddle_source
            || self.required_shared_memory_bytes != expected_shared
        {
            return Err(VkFftError::InvalidKernelIr(
                "fused multi-kernel performConvolution FFT-Rader metadata is inconsistent",
            ));
        }
        let _ = self.requires_twiddle_lut()?;
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ConvolutionIr {
    pub sequence_len: usize,
    pub batch_count: usize,
    pub coordinate_count: usize,
    pub kernel_count: usize,
    pub matrix_layout: Option<ConvolutionMatrixLayout>,
    pub precision: Precision,
    pub scalar: ScalarType,
    pub external_scalar: ScalarType,
    pub input_storage_copy: Option<NdFormattedCopyPassIr>,
    pub output_storage_copy: Option<NdFormattedCopyPassIr>,
    pub forward_fft: OneDimFftIr,
    pub multiply: ConvolutionMultiplyIr,
    /// True upstream-style single-dispatch `convolutionStep`: forward Stockham stages,
    /// application kernel-spectrum multiply, then normalized inverse Stockham stages.
    /// The first tranche is deliberately restricted to ordinary contiguous F32.
    pub fused_stockham_step: Option<ConvolutionStockhamStepIr>,
    /// Upstream batched-kernel single-upload convolutionStep. One forward spectrum is
    /// preserved across K application kernel multiplies and normalized inverse outputs.
    pub fused_multi_kernel_stockham_step: Option<ConvolutionMultiKernelStockhamStepIr>,
    /// Upstream-style matrix/multi-kernel single-upload convolutionStep. One workgroup
    /// serializes coordinate forward spectra, kernel-set matrix rows, and normalized
    /// inverse coordinate transforms without exposing application scratch buffers.
    pub fused_matrix_stockham_step: Option<ConvolutionMatrixStockhamStepIr>,
    /// Pinned single-upload Direct-Rader application convolutionStep. The first
    /// materialized surface is one contiguous logical batch; forward spectrum,
    /// application multiply, and normalized inverse share one workgroup launch.
    pub fused_direct_rader_step: Option<ConvolutionDirectRaderStepIr>,
    /// Pinned Direct-Rader one-input/multi-kernel application. The physical batch-1
    /// forward caller is reused while K application/inverse iterations stay in one pass.
    pub fused_direct_rader_multi_kernel_step: Option<ConvolutionDirectRaderMultiKernelStepIr>,
    /// Pinned single-container FFT-Rader application convolutionStep. The first
    /// materialized surface keeps the complete outer Rader forward, application multiply,
    /// and normalized outer Rader inverse inside one batch-1 workgroup dispatch.
    pub fused_fft_rader_step: Option<ConvolutionFftRaderStepIr>,
    /// Pinned single-container FFT-Rader one-input/multi-kernel application. The K-way
    /// inverse loop reuses the batch1 physical container while the semantic inverse owns K batches.
    pub fused_fft_rader_multi_kernel_step: Option<ConvolutionFftRaderMultiKernelStepIr>,
    /// Upstream-equivalent two-upload application path: high forward upload, embedded
    /// low upload convolutionStep, then high inverse leftover with read-side twiddle.
    /// F32 uses on-the-fly roots; F64 reuses one immutable full-period unit-root LUT.
    pub fused_two_upload_stockham: Option<ConvolutionTwoUploadStockhamIr>,
    /// Two-upload one-input/multi-kernel application path: one high forward upload,
    /// upload-0 convolutionStep fan-out, then K-way high inverse leftover dispatch.
    pub fused_two_upload_multi_kernel_stockham: Option<ConvolutionTwoUploadMultiKernelStockhamIr>,
    /// Upstream three-upload application path: uploads 2 and 1 forward, one embedded
    /// upload-0 convolutionStep, then inverse uploads 1 and 2 with read-side twiddles.
    /// The first materialized witness is ordinary F32 N8388608 `[4096,32,64]`.
    pub fused_three_upload_stockham: Option<ConvolutionThreeUploadStockhamIr>,
    /// Three-upload one-input/multi-kernel path: uploads 2/1 forward once, upload0
    /// fans out K inverse-low results, then inverse uploads 1/2 own K outer batches.
    pub fused_three_upload_multi_kernel_stockham:
        Option<ConvolutionThreeUploadMultiKernelStockhamIr>,
    /// Older one-boundary fusion retained as a fail-soft fallback for F64 and other
    /// shapes that do not yet satisfy the exact single-dispatch contract.
    pub fused_inverse_stockham: Option<KernelIr>,
    pub inverse_fft: OneDimFftIr,
}

impl ConvolutionIr {
    /// Build `performConvolution` from pre-transformed frequency kernels. The current
    /// surface supports one-input/multiple-output `numberKernels` expansion for scalar
    /// and pinned matrix coordinate layouts materialized by [`ConvolutionMatrixLayout`].
    pub fn build_from_spectrum(
        config: FftConfig,
        kernel_spectrum: Vec<Complex64>,
        device: DeviceProfile,
    ) -> Result<Self> {
        config.validate()?;
        if !config.perform_convolution {
            return Err(VkFftError::UnsupportedKernelPath(
                "ConvolutionIr requires FftConfig::with_convolution(true)",
            ));
        }
        let sequence_len = config.dimensions[0];
        let policy = ConvolutionMultiplyPolicy::from_config(&config);
        policy.validate()?;
        let matrix_layout = ConvolutionMatrixLayout::from_config(&config)?;
        let coordinate_count = config.convolution_coordinate_count();
        let kernel_count = config.convolution_kernel_count;
        let output_batch_count = if kernel_count > 1 {
            kernel_count
        } else {
            config.batch_count
        };
        let forward_transform_batch_count = config
            .batch_count
            .checked_mul(coordinate_count)
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "performConvolution forward coordinate-expanded batch count",
            })?;
        let inverse_transform_batch_count = output_batch_count
            .checked_mul(coordinate_count)
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "performConvolution inverse output-expanded batch count",
            })?;
        let (scalar, external_scalar) = match config.precision {
            Precision::F16StorageF32Compute => (ScalarType::F32, ScalarType::F16),
            Precision::F32 => (ScalarType::F32, ScalarType::F32),
            Precision::F64 if device.supports_f64 => (ScalarType::F64, ScalarType::F64),
            Precision::F64ComputeF32Storage if device.supports_f64 => {
                (ScalarType::F64, ScalarType::F32)
            }
            Precision::F64 | Precision::F64ComputeF32Storage => {
                return Err(VkFftError::UnsupportedPrecision {
                    backend: "convolution IR",
                    precision: "f64 compute requires a device with f64 support",
                });
            }
            _ => {
                return Err(VkFftError::UnsupportedKernelPath(
                    "performConvolution currently requires ordinary or mixed F16/F32/F64 storage",
                ));
            }
        };
        let caller_mixed_storage = external_scalar != scalar;
        let caller_layout = NdExternalTensorLayout {
            dimensions: vec![sequence_len],
            axis_strides: vec![1],
            batch_stride: sequence_len,
        };
        let input_storage_copy = if caller_mixed_storage {
            Some(NdFormattedCopyPassIr::new(
                format!("vkfft_convolution_gather_{sequence_len}"),
                scalar,
                external_scalar,
                forward_transform_batch_count,
                caller_layout.clone(),
                NdFormattedCopyOperation::GatherExternalToDense,
                device,
            )?)
        } else {
            None
        };
        let output_storage_copy = if caller_mixed_storage {
            Some(NdFormattedCopyPassIr::new(
                format!("vkfft_convolution_scatter_{sequence_len}"),
                scalar,
                external_scalar,
                inverse_transform_batch_count,
                caller_layout,
                NdFormattedCopyOperation::ScatterDenseToExternal,
                device,
            )?)
        } else {
            None
        };

        // `performConvolution` is an application pipeline, not an attribute of its
        // child FFT plans. Clear the flag before ordinary planning so a plain plan can
        // never accidentally masquerade as a complete convolution.
        let mut forward_config = config.clone();
        forward_config.perform_convolution = false;
        forward_config.convolution_conjugation = ConvolutionConjugation::None;
        forward_config.cross_power_spectrum_normalization = false;
        forward_config.coordinate_features = 1;
        forward_config.matrix_convolution = 1;
        forward_config.symmetric_convolution_kernel = false;
        forward_config.convolution_kernel_count = 1;
        forward_config.batch_count = forward_transform_batch_count;
        forward_config.normalize_inverse = false;
        let mut inverse_config = forward_config.clone().with_inverse_normalization(true);
        inverse_config.batch_count = inverse_transform_batch_count;
        let forward_plan = FftPlan::build_for_device(forward_config, device)?;
        let inverse_plan = FftPlan::build_for_device(inverse_config, device)?;
        let stockham_convolution = matches!(
            &forward_plan.axes[0].algorithm,
            AxisAlgorithm::Stockham { .. }
        ) && matches!(
            &inverse_plan.axes[0].algorithm,
            AxisAlgorithm::Stockham { .. }
        );
        let forward_fft = if stockham_convolution {
            if caller_mixed_storage {
                OneDimFftIr::build_for_convolution_stockham_internal_compute_storage(
                    &forward_plan,
                    Direction::Forward,
                    device,
                )?
            } else {
                OneDimFftIr::build_for_convolution_stockham(
                    &forward_plan,
                    Direction::Forward,
                    device,
                )?
            }
        } else if caller_mixed_storage {
            OneDimFftIr::build_internal_compute_storage(&forward_plan, Direction::Forward, device)?
        } else {
            OneDimFftIr::build(&forward_plan, Direction::Forward, device)?
        }
        .with_axis0_single_upload_block(device)?;
        let inverse_fft = if stockham_convolution {
            if caller_mixed_storage {
                OneDimFftIr::build_for_convolution_stockham_internal_compute_storage(
                    &inverse_plan,
                    Direction::Inverse,
                    device,
                )?
            } else {
                OneDimFftIr::build_for_convolution_stockham(
                    &inverse_plan,
                    Direction::Inverse,
                    device,
                )?
            }
        } else if caller_mixed_storage {
            OneDimFftIr::build_internal_compute_storage(&inverse_plan, Direction::Inverse, device)?
        } else {
            OneDimFftIr::build(&inverse_plan, Direction::Inverse, device)?
        }
        .with_axis0_single_upload_block(device)?;
        let fused_stockham_step = if caller_mixed_storage
            || matrix_layout.is_some()
            || kernel_count > 1
        {
            None
        } else {
            match (&forward_fft, &inverse_fft) {
                (OneDimFftIr::Recursive(forward), OneDimFftIr::Recursive(inverse))
                    if forward.zero_pad_pass.is_none()
                        && inverse.zero_pad_pass.is_none()
                        && forward.four_step_stockham_upload_kernels()?.is_none()
                        && inverse.four_step_stockham_upload_kernels()?.is_none()
                        && forward.four_step_rader_upload_nodes()?.is_none()
                        && inverse.four_step_rader_upload_nodes()?.is_none() =>
                {
                    match (&forward.root, &inverse.root) {
                        (
                            RecursiveFftNodeIr::Stockham(forward),
                            RecursiveFftNodeIr::Stockham(inverse),
                        ) => ConvolutionStockhamStepIr::try_new(forward, inverse, policy, device)?,
                        _ => None,
                    }
                }
                _ => None,
            }
        };
        let fused_multi_kernel_stockham_step =
            if config.batch_count == 1 && matrix_layout.is_none() && kernel_count > 1 {
                match (&forward_fft, &inverse_fft) {
                    (OneDimFftIr::Recursive(forward), OneDimFftIr::Recursive(inverse))
                        if forward.zero_pad_pass.is_none()
                            && inverse.zero_pad_pass.is_none()
                            && forward.four_step_stockham_upload_kernels()?.is_none()
                            && inverse.four_step_stockham_upload_kernels()?.is_none()
                            && forward.four_step_rader_upload_nodes()?.is_none()
                            && inverse.four_step_rader_upload_nodes()?.is_none() =>
                    {
                        match (&forward.root, &inverse.root) {
                            (
                                RecursiveFftNodeIr::Stockham(forward),
                                RecursiveFftNodeIr::Stockham(inverse),
                            ) => ConvolutionMultiKernelStockhamStepIr::try_new(
                                forward,
                                inverse,
                                kernel_count,
                                policy,
                                device,
                            )?,
                            _ => None,
                        }
                    }
                    _ => None,
                }
            } else {
                None
            };
        let fused_matrix_stockham_step = if config.batch_count == 1 {
            if let Some(matrix_layout) = matrix_layout {
                match (&forward_fft, &inverse_fft) {
                    (OneDimFftIr::Recursive(forward), OneDimFftIr::Recursive(inverse))
                        if forward.zero_pad_pass.is_none()
                            && inverse.zero_pad_pass.is_none()
                            && forward.four_step_stockham_upload_kernels()?.is_none()
                            && inverse.four_step_stockham_upload_kernels()?.is_none()
                            && forward.four_step_rader_upload_nodes()?.is_none()
                            && inverse.four_step_rader_upload_nodes()?.is_none() =>
                    {
                        match (&forward.root, &inverse.root) {
                            (
                                RecursiveFftNodeIr::Stockham(forward),
                                RecursiveFftNodeIr::Stockham(inverse),
                            ) => ConvolutionMatrixStockhamStepIr::try_new(
                                forward,
                                inverse,
                                matrix_layout,
                                kernel_count,
                                policy,
                                device,
                            )?,
                            _ => None,
                        }
                    }
                    _ => None,
                }
            } else {
                None
            }
        } else {
            None
        };
        let fused_direct_rader_step = if !caller_mixed_storage
            && matrix_layout.is_none()
            && kernel_count == 1
            && fused_stockham_step.is_none()
        {
            match (&forward_fft, &inverse_fft) {
                (OneDimFftIr::Recursive(forward), OneDimFftIr::Recursive(inverse))
                    if forward.zero_pad_pass.is_none()
                        && inverse.zero_pad_pass.is_none()
                        && forward.four_step_stockham_upload_kernels()?.is_none()
                        && inverse.four_step_stockham_upload_kernels()?.is_none()
                        && forward.four_step_rader_upload_nodes()?.is_none()
                        && inverse.four_step_rader_upload_nodes()?.is_none() =>
                {
                    match (&forward.root, &inverse.root) {
                        (
                            RecursiveFftNodeIr::DirectRader(forward),
                            RecursiveFftNodeIr::DirectRader(inverse),
                        ) => {
                            ConvolutionDirectRaderStepIr::try_new(forward, inverse, policy, device)?
                        }
                        _ => None,
                    }
                }
                _ => None,
            }
        } else {
            None
        };
        let fused_direct_rader_multi_kernel_step = if config.batch_count == 1
            && matrix_layout.is_none()
            && kernel_count > 1
            && fused_multi_kernel_stockham_step.is_none()
        {
            match (&forward_fft, &inverse_fft) {
                (OneDimFftIr::Recursive(forward), OneDimFftIr::Recursive(inverse))
                    if forward.zero_pad_pass.is_none()
                        && inverse.zero_pad_pass.is_none()
                        && forward.four_step_stockham_upload_kernels()?.is_none()
                        && inverse.four_step_stockham_upload_kernels()?.is_none()
                        && forward.four_step_rader_upload_nodes()?.is_none()
                        && inverse.four_step_rader_upload_nodes()?.is_none() =>
                {
                    match (&forward.root, &inverse.root) {
                        (
                            RecursiveFftNodeIr::DirectRader(forward),
                            RecursiveFftNodeIr::DirectRader(inverse),
                        ) => ConvolutionDirectRaderMultiKernelStepIr::try_new(
                            forward,
                            inverse,
                            kernel_count,
                            policy,
                            device,
                        )?,
                        _ => None,
                    }
                }
                _ => None,
            }
        } else {
            None
        };
        let fused_fft_rader_step = if !caller_mixed_storage
            && matrix_layout.is_none()
            && kernel_count == 1
            && fused_stockham_step.is_none()
            && fused_direct_rader_step.is_none()
        {
            match (&forward_fft, &inverse_fft) {
                (OneDimFftIr::Recursive(forward), OneDimFftIr::Recursive(inverse))
                    if forward.zero_pad_pass.is_none()
                        && inverse.zero_pad_pass.is_none()
                        && forward.four_step_stockham_upload_kernels()?.is_none()
                        && inverse.four_step_stockham_upload_kernels()?.is_none()
                        && forward.four_step_rader_upload_nodes()?.is_none()
                        && inverse.four_step_rader_upload_nodes()?.is_none() =>
                {
                    match (&forward.root, &inverse.root) {
                        (
                            RecursiveFftNodeIr::FftRader(forward),
                            RecursiveFftNodeIr::FftRader(inverse),
                        ) => ConvolutionFftRaderStepIr::try_new(forward, inverse, policy, device)?,
                        _ => None,
                    }
                }
                _ => None,
            }
        } else {
            None
        };
        let fused_fft_rader_multi_kernel_step = if config.batch_count == 1
            && matrix_layout.is_none()
            && kernel_count > 1
            && fused_multi_kernel_stockham_step.is_none()
            && fused_direct_rader_multi_kernel_step.is_none()
        {
            match (&forward_fft, &inverse_fft) {
                (OneDimFftIr::Recursive(forward), OneDimFftIr::Recursive(inverse))
                    if forward.zero_pad_pass.is_none()
                        && inverse.zero_pad_pass.is_none()
                        && forward.four_step_stockham_upload_kernels()?.is_none()
                        && inverse.four_step_stockham_upload_kernels()?.is_none()
                        && forward.four_step_rader_upload_nodes()?.is_none()
                        && inverse.four_step_rader_upload_nodes()?.is_none() =>
                {
                    match (&forward.root, &inverse.root) {
                        (
                            RecursiveFftNodeIr::FftRader(forward),
                            RecursiveFftNodeIr::FftRader(inverse),
                        ) => ConvolutionFftRaderMultiKernelStepIr::try_new(
                            forward,
                            inverse,
                            kernel_count,
                            policy,
                            device,
                        )?,
                        _ => None,
                    }
                }
                _ => None,
            }
        } else {
            None
        };
        let fused_two_upload_multi_kernel_stockham = if config.batch_count == 1
            && matrix_layout.is_none()
            && kernel_count > 1
            && fused_multi_kernel_stockham_step.is_none()
            && fused_fft_rader_multi_kernel_step.is_none()
        {
            match (&forward_fft, &inverse_fft) {
                (OneDimFftIr::Recursive(forward), OneDimFftIr::Recursive(inverse))
                    if forward.zero_pad_pass.is_none()
                        && inverse.zero_pad_pass.is_none()
                        && forward.four_step_rader_upload_nodes()?.is_none()
                        && inverse.four_step_rader_upload_nodes()?.is_none() =>
                {
                    ConvolutionTwoUploadMultiKernelStockhamIr::try_new(
                        forward,
                        inverse,
                        kernel_count,
                        policy,
                        device,
                    )?
                }
                _ => None,
            }
        } else {
            None
        };
        let fused_two_upload_stockham = if !caller_mixed_storage
            && matrix_layout.is_none()
            && kernel_count == 1
            && fused_stockham_step.is_none()
            && fused_direct_rader_step.is_none()
            && fused_fft_rader_step.is_none()
        {
            match (&forward_fft, &inverse_fft) {
                (OneDimFftIr::Recursive(forward), OneDimFftIr::Recursive(inverse))
                    if forward.zero_pad_pass.is_none()
                        && inverse.zero_pad_pass.is_none()
                        && forward.four_step_rader_upload_nodes()?.is_none()
                        && inverse.four_step_rader_upload_nodes()?.is_none() =>
                {
                    ConvolutionTwoUploadStockhamIr::try_new(forward, inverse, policy, device)?
                }
                _ => None,
            }
        } else {
            None
        };
        let fused_three_upload_multi_kernel_stockham = if config.batch_count == 1
            && matrix_layout.is_none()
            && kernel_count > 1
            && fused_multi_kernel_stockham_step.is_none()
            && fused_fft_rader_multi_kernel_step.is_none()
            && fused_two_upload_multi_kernel_stockham.is_none()
        {
            match (&forward_fft, &inverse_fft) {
                (OneDimFftIr::Recursive(forward), OneDimFftIr::Recursive(inverse))
                    if forward.zero_pad_pass.is_none()
                        && inverse.zero_pad_pass.is_none()
                        && forward.four_step_rader_upload_nodes()?.is_none()
                        && inverse.four_step_rader_upload_nodes()?.is_none() =>
                {
                    ConvolutionThreeUploadMultiKernelStockhamIr::try_new(
                        forward,
                        inverse,
                        kernel_count,
                        policy,
                        device,
                    )?
                }
                _ => None,
            }
        } else {
            None
        };
        let fused_three_upload_stockham = if !caller_mixed_storage
            && matrix_layout.is_none()
            && kernel_count == 1
            && fused_stockham_step.is_none()
            && fused_two_upload_stockham.is_none()
            && fused_direct_rader_step.is_none()
            && fused_fft_rader_step.is_none()
        {
            match (&forward_fft, &inverse_fft) {
                (OneDimFftIr::Recursive(forward), OneDimFftIr::Recursive(inverse))
                    if forward.zero_pad_pass.is_none()
                        && inverse.zero_pad_pass.is_none()
                        && forward.four_step_rader_upload_nodes()?.is_none()
                        && inverse.four_step_rader_upload_nodes()?.is_none() =>
                {
                    ConvolutionThreeUploadStockhamIr::try_new(forward, inverse, policy, device)?
                }
                _ => None,
            }
        } else {
            None
        };
        let fused_inverse_stockham = if caller_mixed_storage
            || fused_stockham_step.is_some()
            || fused_multi_kernel_stockham_step.is_some()
            || fused_fft_rader_multi_kernel_step.is_some()
            || fused_two_upload_multi_kernel_stockham.is_some()
            || fused_three_upload_multi_kernel_stockham.is_some()
            || fused_direct_rader_step.is_some()
            || fused_fft_rader_step.is_some()
            || fused_two_upload_stockham.is_some()
            || fused_three_upload_stockham.is_some()
            || matrix_layout.is_some()
            || kernel_count > 1
            || !policy.is_default()
        {
            None
        } else {
            match &inverse_fft {
                OneDimFftIr::Recursive(recursive)
                    if recursive.zero_pad_pass.is_none()
                        && recursive.four_step_stockham_upload_kernels()?.is_none()
                        && recursive.four_step_rader_upload_nodes()?.is_none() =>
                {
                    match &recursive.root {
                        RecursiveFftNodeIr::Stockham(kernel) => {
                            match kernel.as_ref().clone().with_lookup_table_input_multiply() {
                                Ok(kernel) => Some(kernel),
                                Err(VkFftError::UnsupportedKernelPath(_))
                                | Err(VkFftError::ResourceLimitExceeded { .. }) => None,
                                Err(error) => return Err(error),
                            }
                        }
                        _ => None,
                    }
                }
                _ => None,
            }
        };
        let multiply = ConvolutionMultiplyIr::new(
            sequence_len,
            config.batch_count,
            coordinate_count,
            kernel_count,
            matrix_layout,
            scalar,
            policy,
            kernel_spectrum,
            device,
        )?;
        let ir = Self {
            sequence_len,
            batch_count: config.batch_count,
            coordinate_count,
            kernel_count,
            matrix_layout,
            precision: config.precision,
            scalar,
            external_scalar,
            input_storage_copy,
            output_storage_copy,
            forward_fft,
            multiply,
            fused_stockham_step,
            fused_multi_kernel_stockham_step,
            fused_matrix_stockham_step,
            fused_direct_rader_step,
            fused_direct_rader_multi_kernel_step,
            fused_fft_rader_step,
            fused_fft_rader_multi_kernel_step,
            fused_two_upload_stockham,
            fused_two_upload_multi_kernel_stockham,
            fused_three_upload_stockham,
            fused_three_upload_multi_kernel_stockham,
            fused_inverse_stockham,
            inverse_fft,
        };
        ir.validate()?;
        Ok(ir)
    }

    pub fn validate(&self) -> Result<()> {
        self.forward_fft.validate()?;
        self.inverse_fft.validate()?;
        self.multiply.validate()?;
        let output_batch_count = self.output_batch_count();
        let forward_transform_batch_count = self
            .batch_count
            .checked_mul(self.coordinate_count)
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "performConvolution validation forward coordinate-expanded batch count",
            })?;
        let inverse_transform_batch_count = output_batch_count
            .checked_mul(self.coordinate_count)
            .ok_or(VkFftError::ArithmeticOverflow {
            operation: "performConvolution validation inverse output-expanded batch count",
        })?;
        if self.kernel_count == 0 {
            return Err(VkFftError::InvalidKernelIr(
                "convolution kernel count must be non-zero",
            ));
        }
        if self.kernel_count > 1 && self.batch_count != 1 {
            return Err(VkFftError::InvalidKernelIr(
                "multi-kernel convolution requires one logical input batch",
            ));
        }
        if let Some(matrix) = self.matrix_layout {
            matrix.validate()?;
            if self.coordinate_count != matrix.matrix_size {
                return Err(VkFftError::InvalidKernelIr(
                    "matrix convolution coordinate ownership is inconsistent",
                ));
            }
        } else if self.coordinate_count != 1 {
            return Err(VkFftError::InvalidKernelIr(
                "scalar convolution requires one coordinate plane",
            ));
        }
        let fused_special_count = [
            self.fused_stockham_step.is_some(),
            self.fused_multi_kernel_stockham_step.is_some(),
            self.fused_matrix_stockham_step.is_some(),
            self.fused_direct_rader_step.is_some(),
            self.fused_direct_rader_multi_kernel_step.is_some(),
            self.fused_fft_rader_step.is_some(),
            self.fused_fft_rader_multi_kernel_step.is_some(),
            self.fused_two_upload_stockham.is_some(),
            self.fused_two_upload_multi_kernel_stockham.is_some(),
            self.fused_three_upload_stockham.is_some(),
            self.fused_three_upload_multi_kernel_stockham.is_some(),
            self.fused_inverse_stockham.is_some(),
        ]
        .into_iter()
        .filter(|active| *active)
        .count();
        let caller_mixed_storage = self.external_scalar != self.scalar;
        match (&self.input_storage_copy, &self.output_storage_copy) {
            (Some(input), Some(output)) if caller_mixed_storage => {
                input.validate()?;
                output.validate()?;
                if input.scalar != self.scalar
                    || output.scalar != self.scalar
                    || input.input_storage_scalar != self.external_scalar
                    || input.output_storage_scalar != self.scalar
                    || output.input_storage_scalar != self.scalar
                    || output.output_storage_scalar != self.external_scalar
                    || input.operation != NdFormattedCopyOperation::GatherExternalToDense
                    || output.operation != NdFormattedCopyOperation::ScatterDenseToExternal
                    || input.tensor_len != self.sequence_len
                    || output.tensor_len != self.sequence_len
                    || input.batch_count != forward_transform_batch_count
                    || output.batch_count != inverse_transform_batch_count
                {
                    return Err(VkFftError::InvalidKernelIr(
                        "mixed-storage convolution caller-copy ownership is inconsistent",
                    ));
                }
            }
            (None, None) if !caller_mixed_storage => {}
            _ => {
                return Err(VkFftError::InvalidKernelIr(
                    "convolution caller-copy ownership does not match external storage",
                ));
            }
        }
        if caller_mixed_storage && fused_special_count != 0 {
            return Err(VkFftError::InvalidKernelIr(
                "mixed-storage convolution must keep application fusion disabled",
            ));
        }
        if fused_special_count > 1 {
            return Err(VkFftError::InvalidKernelIr(
                "performConvolution fused ownership fields are mutually exclusive",
            ));
        }
        if let Some(step) = &self.fused_stockham_step {
            step.validate()?;
            if step.sequence_len != self.sequence_len
                || step.batch_count != self.batch_count
                || step.scalar != self.scalar
                || step.policy != self.multiply.policy
                || self.kernel_count != 1
                || self.fused_multi_kernel_stockham_step.is_some()
                || self.fused_matrix_stockham_step.is_some()
                || self.fused_direct_rader_step.is_some()
                || self.fused_direct_rader_multi_kernel_step.is_some()
                || self.fused_fft_rader_step.is_some()
                || self.fused_two_upload_stockham.is_some()
                || self.fused_two_upload_multi_kernel_stockham.is_some()
                || self.fused_three_upload_stockham.is_some()
                || self.fused_three_upload_multi_kernel_stockham.is_some()
                || self.fused_inverse_stockham.is_some()
                || self.fused_fft_rader_multi_kernel_step.is_some()
            {
                return Err(VkFftError::InvalidKernelIr(
                    "fused performConvolution Stockham step ownership is inconsistent",
                ));
            }
        }
        if let Some(step) = &self.fused_multi_kernel_stockham_step {
            step.validate()?;
            if step.sequence_len != self.sequence_len
                || self.batch_count != 1
                || self.coordinate_count != 1
                || self.matrix_layout.is_some()
                || step.kernel_count != self.kernel_count
                || step.scalar != self.scalar
                || step.policy != self.multiply.policy
                || self.fused_stockham_step.is_some()
                || self.fused_matrix_stockham_step.is_some()
                || self.fused_direct_rader_step.is_some()
                || self.fused_direct_rader_multi_kernel_step.is_some()
                || self.fused_fft_rader_step.is_some()
                || self.fused_two_upload_stockham.is_some()
                || self.fused_two_upload_multi_kernel_stockham.is_some()
                || self.fused_three_upload_stockham.is_some()
                || self.fused_three_upload_multi_kernel_stockham.is_some()
                || self.fused_inverse_stockham.is_some()
                || self.fused_fft_rader_multi_kernel_step.is_some()
            {
                return Err(VkFftError::InvalidKernelIr(
                    "fused multi-kernel performConvolution Stockham ownership is inconsistent",
                ));
            }
        }
        if let Some(step) = &self.fused_matrix_stockham_step {
            step.validate()?;
            if step.sequence_len != self.sequence_len
                || self.batch_count != 1
                || step.coordinate_count != self.coordinate_count
                || step.kernel_count != self.kernel_count
                || Some(step.matrix_layout) != self.matrix_layout
                || step.scalar != self.scalar
                || step.policy != self.multiply.policy
                || self.fused_stockham_step.is_some()
                || self.fused_multi_kernel_stockham_step.is_some()
                || self.fused_direct_rader_step.is_some()
                || self.fused_direct_rader_multi_kernel_step.is_some()
                || self.fused_fft_rader_step.is_some()
                || self.fused_two_upload_stockham.is_some()
                || self.fused_two_upload_multi_kernel_stockham.is_some()
                || self.fused_three_upload_stockham.is_some()
                || self.fused_three_upload_multi_kernel_stockham.is_some()
                || self.fused_inverse_stockham.is_some()
                || self.fused_fft_rader_multi_kernel_step.is_some()
            {
                return Err(VkFftError::InvalidKernelIr(
                    "fused matrix performConvolution Stockham ownership is inconsistent",
                ));
            }
        }
        if let Some(step) = &self.fused_direct_rader_step {
            step.validate()?;
            if step.prime != self.sequence_len
                || step.batch_count != self.batch_count
                || step.scalar != self.scalar
                || step.policy != self.multiply.policy
                || self.batch_count != 1
                || self.kernel_count != 1
                || self.matrix_layout.is_some()
                || self.coordinate_count != 1
                || self.fused_stockham_step.is_some()
                || self.fused_multi_kernel_stockham_step.is_some()
                || self.fused_matrix_stockham_step.is_some()
                || self.fused_direct_rader_multi_kernel_step.is_some()
                || self.fused_fft_rader_step.is_some()
                || self.fused_two_upload_stockham.is_some()
                || self.fused_two_upload_multi_kernel_stockham.is_some()
                || self.fused_three_upload_stockham.is_some()
                || self.fused_three_upload_multi_kernel_stockham.is_some()
                || self.fused_inverse_stockham.is_some()
                || self.fused_fft_rader_multi_kernel_step.is_some()
            {
                return Err(VkFftError::InvalidKernelIr(
                    "fused performConvolution Direct-Rader ownership is inconsistent",
                ));
            }
        }
        if let Some(step) = &self.fused_direct_rader_multi_kernel_step {
            step.validate()?;
            if step.prime != self.sequence_len
                || self.batch_count != 1
                || self.coordinate_count != 1
                || self.matrix_layout.is_some()
                || step.kernel_count != self.kernel_count
                || step.scalar != self.scalar
                || step.policy != self.multiply.policy
                || self.kernel_count <= 1
            {
                return Err(VkFftError::InvalidKernelIr(
                    "fused multi-kernel performConvolution Direct-Rader ownership is inconsistent",
                ));
            }
        }
        if let Some(step) = &self.fused_fft_rader_step {
            step.validate()?;
            if step.prime != self.sequence_len
                || step.batch_count != self.batch_count
                || step.scalar != self.scalar
                || step.policy != self.multiply.policy
                || self.batch_count != 1
                || self.kernel_count != 1
                || self.matrix_layout.is_some()
                || self.coordinate_count != 1
                || self.fused_stockham_step.is_some()
                || self.fused_multi_kernel_stockham_step.is_some()
                || self.fused_matrix_stockham_step.is_some()
                || self.fused_direct_rader_step.is_some()
                || self.fused_direct_rader_multi_kernel_step.is_some()
                || self.fused_two_upload_stockham.is_some()
                || self.fused_two_upload_multi_kernel_stockham.is_some()
                || self.fused_three_upload_stockham.is_some()
                || self.fused_three_upload_multi_kernel_stockham.is_some()
                || self.fused_inverse_stockham.is_some()
                || self.fused_fft_rader_multi_kernel_step.is_some()
            {
                return Err(VkFftError::InvalidKernelIr(
                    "fused performConvolution FFT-Rader ownership is inconsistent",
                ));
            }
        }
        if let Some(step) = &self.fused_fft_rader_multi_kernel_step {
            step.validate()?;
            if step.prime != self.sequence_len
                || self.batch_count != 1
                || self.coordinate_count != 1
                || self.matrix_layout.is_some()
                || step.kernel_count != self.kernel_count
                || step.scalar != self.scalar
                || step.policy != self.multiply.policy
                || self.fused_stockham_step.is_some()
                || self.fused_multi_kernel_stockham_step.is_some()
                || self.fused_matrix_stockham_step.is_some()
                || self.fused_direct_rader_step.is_some()
                || self.fused_direct_rader_multi_kernel_step.is_some()
                || self.fused_fft_rader_step.is_some()
                || self.fused_two_upload_stockham.is_some()
                || self.fused_two_upload_multi_kernel_stockham.is_some()
                || self.fused_three_upload_stockham.is_some()
                || self.fused_three_upload_multi_kernel_stockham.is_some()
                || self.fused_inverse_stockham.is_some()
            {
                return Err(VkFftError::InvalidKernelIr(
                    "fused multi-kernel performConvolution FFT-Rader ownership is inconsistent",
                ));
            }
        }
        if let Some(two_upload) = &self.fused_two_upload_multi_kernel_stockham {
            two_upload.validate()?;
            if two_upload.sequence_len != self.sequence_len
                || self.batch_count != 1
                || self.coordinate_count != 1
                || self.matrix_layout.is_some()
                || two_upload.kernel_count != self.kernel_count
                || two_upload.scalar != self.scalar
                || two_upload.policy != self.multiply.policy
                || self.fused_stockham_step.is_some()
                || self.fused_multi_kernel_stockham_step.is_some()
                || self.fused_matrix_stockham_step.is_some()
                || self.fused_direct_rader_step.is_some()
                || self.fused_direct_rader_multi_kernel_step.is_some()
                || self.fused_fft_rader_step.is_some()
                || self.fused_two_upload_stockham.is_some()
                || self.fused_three_upload_stockham.is_some()
                || self.fused_three_upload_multi_kernel_stockham.is_some()
                || self.fused_inverse_stockham.is_some()
                || self.fused_fft_rader_multi_kernel_step.is_some()
            {
                return Err(VkFftError::InvalidKernelIr(
                    "two-upload multi-kernel performConvolution ownership is inconsistent",
                ));
            }
        }
        if let Some(two_upload) = &self.fused_two_upload_stockham {
            two_upload.validate()?;
            if two_upload.sequence_len != self.sequence_len
                || two_upload.batch_count != self.batch_count
                || two_upload.scalar != self.scalar
                || two_upload.policy != self.multiply.policy
                || self.kernel_count != 1
                || self.fused_stockham_step.is_some()
                || self.fused_multi_kernel_stockham_step.is_some()
                || self.fused_matrix_stockham_step.is_some()
                || self.fused_direct_rader_step.is_some()
                || self.fused_direct_rader_multi_kernel_step.is_some()
                || self.fused_fft_rader_step.is_some()
                || self.fused_two_upload_multi_kernel_stockham.is_some()
                || self.fused_three_upload_stockham.is_some()
                || self.fused_three_upload_multi_kernel_stockham.is_some()
                || self.fused_inverse_stockham.is_some()
                || self.fused_fft_rader_multi_kernel_step.is_some()
            {
                return Err(VkFftError::InvalidKernelIr(
                    "two-upload performConvolution Stockham ownership is inconsistent",
                ));
            }
        }
        if let Some(three_upload) = &self.fused_three_upload_multi_kernel_stockham {
            three_upload.validate()?;
            if three_upload.sequence_len != self.sequence_len
                || self.batch_count != 1
                || self.coordinate_count != 1
                || self.matrix_layout.is_some()
                || three_upload.kernel_count != self.kernel_count
                || three_upload.scalar != self.scalar
                || three_upload.policy != self.multiply.policy
                || self.fused_stockham_step.is_some()
                || self.fused_multi_kernel_stockham_step.is_some()
                || self.fused_matrix_stockham_step.is_some()
                || self.fused_direct_rader_step.is_some()
                || self.fused_direct_rader_multi_kernel_step.is_some()
                || self.fused_fft_rader_step.is_some()
                || self.fused_two_upload_stockham.is_some()
                || self.fused_two_upload_multi_kernel_stockham.is_some()
                || self.fused_three_upload_stockham.is_some()
                || self.fused_inverse_stockham.is_some()
                || self.fused_fft_rader_multi_kernel_step.is_some()
            {
                return Err(VkFftError::InvalidKernelIr(
                    "three-upload multi-kernel performConvolution ownership is inconsistent",
                ));
            }
        }
        if let Some(three_upload) = &self.fused_three_upload_stockham {
            three_upload.validate()?;
            if three_upload.sequence_len != self.sequence_len
                || three_upload.batch_count != self.batch_count
                || three_upload.scalar != self.scalar
                || three_upload.policy != self.multiply.policy
                || self.kernel_count != 1
                || self.fused_stockham_step.is_some()
                || self.fused_multi_kernel_stockham_step.is_some()
                || self.fused_matrix_stockham_step.is_some()
                || self.fused_direct_rader_step.is_some()
                || self.fused_direct_rader_multi_kernel_step.is_some()
                || self.fused_fft_rader_step.is_some()
                || self.fused_two_upload_stockham.is_some()
                || self.fused_two_upload_multi_kernel_stockham.is_some()
                || self.fused_three_upload_multi_kernel_stockham.is_some()
                || self.fused_inverse_stockham.is_some()
                || self.fused_fft_rader_multi_kernel_step.is_some()
            {
                return Err(VkFftError::InvalidKernelIr(
                    "three-upload performConvolution Stockham ownership is inconsistent",
                ));
            }
        }
        if let Some(fused) = &self.fused_inverse_stockham {
            fused.validate()?;
            if fused.sequence_len != self.sequence_len
                || fused.batch_count != self.batch_count
                || fused.direction != Direction::Inverse
                || fused.scalar != self.scalar
                || fused.io_mapping != StockhamIoMapping::Contiguous
                || fused.input_modifier != StockhamInputModifier::MultiplyLookupTable
                || self.kernel_count != 1
                || !self.multiply.policy.is_default()
                || self.fused_stockham_step.is_some()
                || self.fused_multi_kernel_stockham_step.is_some()
                || self.fused_matrix_stockham_step.is_some()
                || self.fused_direct_rader_step.is_some()
                || self.fused_direct_rader_multi_kernel_step.is_some()
                || self.fused_fft_rader_step.is_some()
                || self.fused_two_upload_stockham.is_some()
                || self.fused_two_upload_multi_kernel_stockham.is_some()
                || self.fused_three_upload_stockham.is_some()
                || self.fused_three_upload_multi_kernel_stockham.is_some()
                || self.fused_fft_rader_multi_kernel_step.is_some()
            {
                return Err(VkFftError::InvalidKernelIr(
                    "fused performConvolution inverse Stockham metadata is inconsistent",
                ));
            }
        }
        if self.forward_fft.direction() != Direction::Forward
            || self.inverse_fft.direction() != Direction::Inverse
            || self.forward_fft.logical_len() != self.sequence_len
            || self.inverse_fft.logical_len() != self.sequence_len
            || self.forward_fft.batch_count() != forward_transform_batch_count
            || self.inverse_fft.batch_count() != inverse_transform_batch_count
            || self.forward_fft.scalar() != self.scalar
            || self.inverse_fft.scalar() != self.scalar
            || self.forward_fft.external_storage_scalar() != self.scalar
            || self.inverse_fft.external_storage_scalar() != self.scalar
            || self.multiply.sequence_len != self.sequence_len
            || self.multiply.batch_count != self.batch_count
            || self.multiply.coordinate_count != self.coordinate_count
            || self.multiply.kernel_count != self.kernel_count
            || self.multiply.matrix_layout != self.matrix_layout
            || self.multiply.scalar != self.scalar
        {
            return Err(VkFftError::InvalidKernelIr(
                "convolution forward/multiply/inverse metadata is inconsistent",
            ));
        }
        if !matches!(
            (self.precision, self.scalar, self.external_scalar),
            (
                Precision::F16StorageF32Compute,
                ScalarType::F32,
                ScalarType::F16
            ) | (Precision::F32, ScalarType::F32, ScalarType::F32)
                | (Precision::F64, ScalarType::F64, ScalarType::F64)
                | (
                    Precision::F64ComputeF32Storage,
                    ScalarType::F64,
                    ScalarType::F32
                )
        ) {
            return Err(VkFftError::InvalidKernelIr(
                "convolution precision does not match its compute scalar",
            ));
        }
        Ok(())
    }

    pub fn has_fused_stockham_step(&self) -> bool {
        self.fused_stockham_step.is_some()
    }

    pub fn has_fused_multi_kernel_stockham_step(&self) -> bool {
        self.fused_multi_kernel_stockham_step.is_some()
    }

    pub fn has_fused_matrix_stockham_step(&self) -> bool {
        self.fused_matrix_stockham_step.is_some()
    }

    pub fn has_fused_direct_rader_step(&self) -> bool {
        self.fused_direct_rader_step.is_some()
    }

    pub fn has_fused_direct_rader_multi_kernel_step(&self) -> bool {
        self.fused_direct_rader_multi_kernel_step.is_some()
    }

    pub fn has_fused_fft_rader_step(&self) -> bool {
        self.fused_fft_rader_step.is_some()
    }

    pub fn has_fused_fft_rader_multi_kernel_step(&self) -> bool {
        self.fused_fft_rader_multi_kernel_step.is_some()
    }

    pub fn has_fused_two_upload_stockham(&self) -> bool {
        self.fused_two_upload_stockham.is_some()
    }

    pub fn has_fused_two_upload_multi_kernel_stockham(&self) -> bool {
        self.fused_two_upload_multi_kernel_stockham.is_some()
    }

    pub fn has_fused_three_upload_stockham(&self) -> bool {
        self.fused_three_upload_stockham.is_some()
    }

    pub fn has_fused_three_upload_multi_kernel_stockham(&self) -> bool {
        self.fused_three_upload_multi_kernel_stockham.is_some()
    }

    pub fn has_fused_inverse_stockham(&self) -> bool {
        self.fused_inverse_stockham.is_some()
    }

    pub fn output_batch_count(&self) -> usize {
        if self.kernel_count > 1 {
            self.kernel_count
        } else {
            self.batch_count
        }
    }

    pub fn kernel_spectrum(&self) -> &[Complex64] {
        &self.multiply.kernel_spectrum
    }
}

/// Correctness-first multidimensional application convolution.
///
/// Pinned VkFFT marks only the final FFT dimension's first upload as `convolutionStep`;
/// that shader consumes the complete ND forward spectrum, applies the application
/// kernel once, and starts the inverse before RunApp walks the remaining dimensions
/// backwards. This IR preserves the same ownership explicitly as
/// `ND forward -> tensor-frequency multiply -> normalized ND inverse` and deliberately
/// leaves last-axis fusion to a later optimization tranche.
#[derive(Debug, Clone, PartialEq)]
pub struct NdConvolutionIr {
    pub dimensions: Vec<usize>,
    pub tensor_len: usize,
    pub batch_count: usize,
    pub coordinate_count: usize,
    pub kernel_count: usize,
    pub matrix_layout: Option<ConvolutionMatrixLayout>,
    pub precision: Precision,
    pub scalar: ScalarType,
    pub forward_fft: NdFftIr,
    pub multiply: ConvolutionMultiplyIr,
    pub inverse_fft: NdFftIr,
}

impl NdConvolutionIr {
    pub fn build_from_spectrum(
        config: FftConfig,
        kernel_spectrum: Vec<Complex64>,
        device: DeviceProfile,
    ) -> Result<Self> {
        config.validate()?;
        if !config.perform_convolution
            || config.transform != crate::TransformKind::ComplexToComplex
            || config.dimensions.len() < 2
        {
            return Err(VkFftError::UnsupportedKernelPath(
                "multidimensional performConvolution IR requires a multidimensional C2C convolution config",
            ));
        }
        let matrix_layout = ConvolutionMatrixLayout::from_config(&config)?;
        let coordinate_count = config.convolution_coordinate_count();
        let kernel_count = config.convolution_kernel_count;
        let scalar_ownership_ok = matrix_layout.is_none()
            && coordinate_count == 1
            && (kernel_count == 1 || config.batch_count == 1);
        let matrix_ownership_ok =
            matrix_layout.is_some() && config.batch_count == 1 && kernel_count == 1;
        if !scalar_ownership_ok && !matrix_ownership_ok {
            return Err(VkFftError::UnsupportedKernelPath(
                "multidimensional performConvolution IR supports scalar K=1/K>1 ownership or batch1/K1 matrix coordinates",
            ));
        }
        if config.input_buffer_batch_stride.is_some()
            || config.output_buffer_batch_stride.is_some()
            || config.input_buffer_axis_strides.iter().any(Option::is_some)
            || config
                .output_buffer_axis_strides
                .iter()
                .any(Option::is_some)
            || config.zero_padding.iter().any(Option::is_some)
            || config.omit_dimension.iter().any(|&omit| omit)
        {
            return Err(VkFftError::UnsupportedKernelPath(
                "multidimensional performConvolution currently requires dense unpadded active dimensions",
            ));
        }
        let scalar = match config.precision {
            Precision::F32 => ScalarType::F32,
            Precision::F64 if device.supports_f64 => ScalarType::F64,
            Precision::F64 => {
                return Err(VkFftError::UnsupportedPrecision {
                    backend: "multidimensional performConvolution IR",
                    precision: "F64 requires device fp64 support",
                });
            }
            _ => {
                return Err(VkFftError::UnsupportedKernelPath(
                    "multidimensional performConvolution currently requires ordinary F32 or F64 storage",
                ));
            }
        };
        let tensor_len = config
            .dimensions
            .iter()
            .try_fold(1usize, |product, &length| {
                product
                    .checked_mul(length)
                    .ok_or(VkFftError::ArithmeticOverflow {
                        operation: "multidimensional convolution tensor element count",
                    })
            })?;
        let output_batch_count = if kernel_count > 1 {
            kernel_count
        } else {
            config.batch_count
        };
        let forward_transform_batch_count = config
            .batch_count
            .checked_mul(coordinate_count)
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "multidimensional convolution forward coordinate-expanded batch count",
            })?;
        let inverse_transform_batch_count = output_batch_count
            .checked_mul(coordinate_count)
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "multidimensional convolution inverse output-expanded batch count",
            })?;
        let policy = ConvolutionMultiplyPolicy::from_config(&config);
        let mut forward_config = config.clone();
        forward_config.perform_convolution = false;
        forward_config.convolution_conjugation = ConvolutionConjugation::None;
        forward_config.cross_power_spectrum_normalization = false;
        forward_config.coordinate_features = 1;
        forward_config.matrix_convolution = 1;
        forward_config.symmetric_convolution_kernel = false;
        forward_config.convolution_kernel_count = 1;
        forward_config.batch_count = forward_transform_batch_count;
        forward_config.normalize_inverse = false;
        let mut inverse_config = forward_config.clone();
        inverse_config.batch_count = inverse_transform_batch_count;
        inverse_config.normalize_inverse = true;
        let forward_plan = FftPlan::build_for_device(forward_config, device)?;
        let forward_fft = NdFftIr::build(&forward_plan, Direction::Forward, device)?;
        let inverse_plan = FftPlan::build_for_device(inverse_config, device)?;
        let inverse_fft = NdFftIr::build(&inverse_plan, Direction::Inverse, device)?;
        let multiply = ConvolutionMultiplyIr::new(
            tensor_len,
            config.batch_count,
            coordinate_count,
            kernel_count,
            matrix_layout,
            scalar,
            policy,
            kernel_spectrum,
            device,
        )?;
        let ir = Self {
            dimensions: config.dimensions,
            tensor_len,
            batch_count: config.batch_count,
            coordinate_count,
            kernel_count,
            matrix_layout,
            precision: config.precision,
            scalar,
            forward_fft,
            multiply,
            inverse_fft,
        };
        ir.validate()?;
        Ok(ir)
    }

    pub fn validate(&self) -> Result<()> {
        self.forward_fft.validate()?;
        self.inverse_fft.validate()?;
        self.multiply.validate()?;
        let tensor_len =
            self.dimensions
                .iter()
                .try_fold(1usize, |product, &length| {
                    product.checked_mul(length).ok_or(VkFftError::ArithmeticOverflow {
                operation: "multidimensional convolution validation tensor element count",
            })
                })?;
        let output_batch_count = self.output_batch_count();
        let forward_transform_batch_count = self
            .batch_count
            .checked_mul(self.coordinate_count)
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "multidimensional convolution validation forward batch count",
            })?;
        let inverse_transform_batch_count = output_batch_count
            .checked_mul(self.coordinate_count)
            .ok_or(VkFftError::ArithmeticOverflow {
            operation: "multidimensional convolution validation inverse batch count",
        })?;
        let ownership_ok = match self.matrix_layout {
            Some(matrix) => {
                self.batch_count == 1
                    && self.kernel_count == 1
                    && self.coordinate_count == matrix.matrix_size
            }
            None => self.coordinate_count == 1 && (self.kernel_count == 1 || self.batch_count == 1),
        };
        if self.dimensions.len() < 2
            || tensor_len != self.tensor_len
            || self.batch_count == 0
            || self.coordinate_count == 0
            || self.kernel_count == 0
            || !ownership_ok
            || self.forward_fft.dimensions != self.dimensions
            || self.inverse_fft.dimensions != self.dimensions
            || self.forward_fft.tensor_len != self.tensor_len
            || self.inverse_fft.tensor_len != self.tensor_len
            || self.forward_fft.batch_count != forward_transform_batch_count
            || self.inverse_fft.batch_count != inverse_transform_batch_count
            || self.forward_fft.direction != Direction::Forward
            || self.inverse_fft.direction != Direction::Inverse
            || self.forward_fft.scalar != self.scalar
            || self.inverse_fft.scalar != self.scalar
            || self.forward_fft.external_scalar != self.scalar
            || self.inverse_fft.external_scalar != self.scalar
            || self.forward_fft.input_formatted_copy.is_some()
            || self.forward_fft.output_formatted_copy.is_some()
            || self.inverse_fft.input_formatted_copy.is_some()
            || self.inverse_fft.output_formatted_copy.is_some()
            || self.forward_fft.zero_pad_pass.is_some()
            || self.inverse_fft.zero_pad_pass.is_some()
            || self.forward_fft.omitted_axes.iter().any(|&omit| omit)
            || self.inverse_fft.omitted_axes.iter().any(|&omit| omit)
            || self.multiply.sequence_len != self.tensor_len
            || self.multiply.batch_count != self.batch_count
            || self.multiply.coordinate_count != self.coordinate_count
            || self.multiply.kernel_count != self.kernel_count
            || self.multiply.matrix_layout != self.matrix_layout
            || self.multiply.scalar != self.scalar
            || !matches!(
                (self.precision, self.scalar),
                (Precision::F32, ScalarType::F32) | (Precision::F64, ScalarType::F64)
            )
        {
            return Err(VkFftError::InvalidKernelIr(
                "multidimensional convolution forward/multiply/inverse ownership is inconsistent",
            ));
        }
        Ok(())
    }

    pub fn output_batch_count(&self) -> usize {
        if self.kernel_count > 1 {
            self.kernel_count
        } else {
            self.batch_count
        }
    }

    pub fn kernel_spectrum(&self) -> &[Complex64] {
        &self.multiply.kernel_spectrum
    }
}

/// Correctness-first multidimensional real application convolution.
///
/// The pinned sample_52 contract uses tightly packed full-real coordinate systems, one
/// complete multidimensional R2C per coordinate, an application multiply over the compact
/// Hermitian spectra, then normalized C2R outputs. `numberKernels > 1` is one-input-set
/// fan-out: the forward side owns C coordinate systems while the midpoint/inverse/output
/// side owns K*C systems in kernel-major, coordinate-inner order.
#[derive(Debug, Clone, PartialEq)]
pub struct NdRealConvolutionIr {
    pub dimensions: Vec<usize>,
    pub full_tensor_len: usize,
    pub compact_tensor_len: usize,
    pub batch_count: usize,
    pub coordinate_count: usize,
    pub kernel_count: usize,
    pub matrix_layout: Option<ConvolutionMatrixLayout>,
    pub zero_padding: Vec<Option<ZeroPaddingRange>>,
    pub precision: Precision,
    pub scalar: ScalarType,
    pub external_scalar: ScalarType,
    pub forward_r2c: NdRealFftIr,
    pub multiply: ConvolutionMultiplyIr,
    pub inverse_c2r: NdRealFftIr,
}

impl NdRealConvolutionIr {
    pub fn build_from_spectrum(
        config: FftConfig,
        kernel_spectrum: Vec<Complex64>,
        device: DeviceProfile,
    ) -> Result<Self> {
        config.validate()?;
        if !config.perform_convolution
            || config.transform != TransformKind::RealToComplex
            || config.dimensions.len() < 2
            || config.batch_count != 1
        {
            return Err(VkFftError::UnsupportedKernelPath(
                "multidimensional real performConvolution IR requires batch1 ND R2C application ownership",
            ));
        }
        let matrix_layout = ConvolutionMatrixLayout::from_config(&config)?;
        let coordinate_count = config.coordinate_features;
        let kernel_count = config.convolution_kernel_count;
        let independent_ownership_ok = matrix_layout.is_none();
        let matrix_ownership_ok = matrix_layout
            .map(|matrix| coordinate_count == matrix.matrix_size)
            .unwrap_or(false);
        if !independent_ownership_ok && !matrix_ownership_ok {
            return Err(VkFftError::UnsupportedKernelPath(
                "multidimensional real performConvolution IR supports batch1 independent-coordinate or matrix K=1/K>1 ownership",
            ));
        }
        let has_formatted_strides = config.input_buffer_batch_stride.is_some()
            || config.output_buffer_batch_stride.is_some()
            || config.input_buffer_axis_strides.iter().any(Option::is_some)
            || config
                .output_buffer_axis_strides
                .iter()
                .any(Option::is_some);
        if config.omit_dimension.iter().any(|&omit| omit) {
            return Err(VkFftError::UnsupportedKernelPath(
                "multidimensional real performConvolution currently requires active dimensions without omitDimension",
            ));
        }
        if has_formatted_strides
            && (coordinate_count != 1 || kernel_count != 1 || matrix_layout.is_some())
        {
            return Err(VkFftError::UnsupportedKernelPath(
                "multidimensional real formatted performConvolution currently requires scalar batch1 K1",
            ));
        }
        let has_zero_padding = config.zero_padding.iter().any(Option::is_some);
        let zero_padding_ownership_ok = matrix_layout
            .map(|matrix| coordinate_count == matrix.matrix_size && !matrix.symmetric_kernel)
            .unwrap_or(true);
        if has_zero_padding && !zero_padding_ownership_ok {
            return Err(VkFftError::UnsupportedKernelPath(
                "multidimensional real performConvolution zero padding requires batch1/K1 nonsymmetric matrix ownership",
            ));
        }
        let (scalar, external_scalar) = match config.precision {
            Precision::F16StorageF32Compute => (ScalarType::F32, ScalarType::F16),
            Precision::F32 => (ScalarType::F32, ScalarType::F32),
            Precision::F64 if device.supports_f64 => (ScalarType::F64, ScalarType::F64),
            Precision::F64ComputeF32Storage if device.supports_f64 => {
                (ScalarType::F64, ScalarType::F32)
            }
            Precision::F64 | Precision::F64ComputeF32Storage => {
                return Err(VkFftError::UnsupportedPrecision {
                    backend: "multidimensional real performConvolution IR",
                    precision: "F64 compute requires device fp64 support",
                });
            }
            _ => {
                return Err(VkFftError::UnsupportedKernelPath(
                    "multidimensional real performConvolution currently requires ordinary or mixed F16/F32/F64 storage",
                ));
            }
        };
        let policy = ConvolutionMultiplyPolicy::from_config(&config);
        policy.validate()?;
        let zero_padding = config.zero_padding.clone();
        let application_output_axis_strides = config.output_buffer_axis_strides.clone();
        let application_output_batch_stride = config.output_buffer_batch_stride;

        let mut forward_config = config.clone();
        forward_config.perform_convolution = false;
        forward_config.output_buffer_axis_strides = vec![None; forward_config.dimensions.len()];
        forward_config.output_buffer_batch_stride = None;
        forward_config.convolution_conjugation = ConvolutionConjugation::None;
        forward_config.cross_power_spectrum_normalization = false;
        forward_config.coordinate_features = 1;
        forward_config.matrix_convolution = 1;
        forward_config.symmetric_convolution_kernel = false;
        forward_config.convolution_kernel_count = 1;
        forward_config.batch_count = coordinate_count;
        forward_config.transform = TransformKind::RealToComplex;
        forward_config.normalize_inverse = false;
        let forward_plan = FftPlan::build_for_device(forward_config.clone(), device)?;
        let mut forward_r2c = NdRealFftIr::build_for_device_plan(&forward_plan, device)?;
        if external_scalar != scalar {
            forward_r2c = forward_r2c.with_internal_output_compute_storage()?;
        }

        let mut inverse_config = forward_config;
        inverse_config.transform = TransformKind::ComplexToReal;
        inverse_config.input_buffer_axis_strides = vec![None; inverse_config.dimensions.len()];
        inverse_config.input_buffer_batch_stride = None;
        inverse_config.output_buffer_axis_strides = application_output_axis_strides;
        inverse_config.output_buffer_batch_stride = application_output_batch_stride;
        inverse_config.batch_count =
            kernel_count
                .checked_mul(coordinate_count)
                .ok_or(VkFftError::ArithmeticOverflow {
                    operation: "multidimensional real convolution inverse system count",
                })?;
        inverse_config.normalize_inverse = true;
        let inverse_plan = FftPlan::build_for_device(inverse_config, device)?;
        let mut inverse_c2r = NdRealFftIr::build_for_device_plan(&inverse_plan, device)?;
        if external_scalar != scalar {
            inverse_c2r = inverse_c2r.with_internal_input_compute_storage()?;
        }

        let full_tensor_len = forward_r2c.full_tensor_len;
        let compact_tensor_len = forward_r2c.compact_tensor_len;
        let multiply = if let Some(matrix) = matrix_layout {
            ConvolutionMultiplyIr::new(
                compact_tensor_len,
                1,
                coordinate_count,
                kernel_count,
                Some(matrix),
                scalar,
                policy,
                kernel_spectrum,
                device,
            )?
        } else if coordinate_count > 1 {
            ConvolutionMultiplyIr::new_independent_coordinates(
                compact_tensor_len,
                1,
                coordinate_count,
                kernel_count,
                scalar,
                policy,
                kernel_spectrum,
                device,
            )?
        } else {
            ConvolutionMultiplyIr::new(
                compact_tensor_len,
                1,
                1,
                kernel_count,
                None,
                scalar,
                policy,
                kernel_spectrum,
                device,
            )?
        };
        let ir = Self {
            dimensions: config.dimensions,
            full_tensor_len,
            compact_tensor_len,
            batch_count: 1,
            coordinate_count,
            kernel_count,
            matrix_layout,
            zero_padding,
            precision: config.precision,
            scalar,
            external_scalar,
            forward_r2c,
            multiply,
            inverse_c2r,
        };
        ir.validate()?;
        Ok(ir)
    }

    pub fn validate(&self) -> Result<()> {
        self.forward_r2c.validate()?;
        self.inverse_c2r.validate()?;
        self.multiply.validate()?;
        let output_system_count = self.output_system_count()?;
        let ownership_ok = match self.matrix_layout {
            Some(matrix) => self.coordinate_count == matrix.matrix_size,
            None => true,
        };
        let has_zero_padding = self.zero_padding.iter().any(Option::is_some);
        let forward_zero_padding_ok = match (&self.forward_r2c.zero_pad_pass, has_zero_padding) {
            (Some(pass), true) => {
                pass.direction == Direction::Forward
                    && pass.dimensions == self.dimensions
                    && pass.batch_count == self.coordinate_count
                    && pass.ranges == self.zero_padding
            }
            (None, false) => true,
            _ => false,
        };
        let inverse_zero_padding_ok = match (&self.inverse_c2r.zero_pad_pass, has_zero_padding) {
            (Some(pass), true) => {
                pass.direction == Direction::Inverse
                    && pass.dimensions == self.dimensions
                    && pass.batch_count == output_system_count
                    && pass.ranges == self.zero_padding
            }
            (None, false) => true,
            _ => false,
        };
        let zero_padding_ownership_ok = if has_zero_padding {
            self.matrix_layout
                .map(|matrix| {
                    self.coordinate_count == matrix.matrix_size && !matrix.symmetric_kernel
                })
                .unwrap_or(true)
                && forward_zero_padding_ok
                && inverse_zero_padding_ok
        } else {
            forward_zero_padding_ok && inverse_zero_padding_ok
        };
        if self.dimensions.len() < 2
            || self.batch_count != 1
            || self.coordinate_count == 0
            || self.kernel_count == 0
            || self.zero_padding.len() != self.dimensions.len()
            || !ownership_ok
            || !zero_padding_ownership_ok
            || self.forward_r2c.kind != RealFftKind::RealToComplex
            || self.inverse_c2r.kind != RealFftKind::ComplexToReal
            || self.forward_r2c.dimensions != self.dimensions
            || self.inverse_c2r.dimensions != self.dimensions
            || self.forward_r2c.full_tensor_len != self.full_tensor_len
            || self.inverse_c2r.full_tensor_len != self.full_tensor_len
            || self.forward_r2c.compact_tensor_len != self.compact_tensor_len
            || self.inverse_c2r.compact_tensor_len != self.compact_tensor_len
            || self.forward_r2c.batch_count != self.coordinate_count
            || self.inverse_c2r.batch_count != output_system_count
            || self.forward_r2c.scalar != self.scalar
            || self.inverse_c2r.scalar != self.scalar
            || self.forward_r2c.external_scalar != self.external_scalar
            || self.inverse_c2r.external_scalar != self.external_scalar
            || self.forward_r2c.output_boundary_compute_storage
                != (self.external_scalar != self.scalar)
            || self.inverse_c2r.input_boundary_compute_storage
                != (self.external_scalar != self.scalar)
            || self.forward_r2c.input_boundary_compute_storage
            || self.inverse_c2r.output_boundary_compute_storage
            || self.forward_r2c.output_formatted_copy.is_some()
            || self.inverse_c2r.input_formatted_copy.is_some()
            || ((self.forward_r2c.input_formatted_copy.is_some()
                || self.inverse_c2r.output_formatted_copy.is_some())
                && (self.coordinate_count != 1
                    || self.kernel_count != 1
                    || self.matrix_layout.is_some()))
            || self.forward_r2c.omitted_axes.iter().any(|&omit| omit)
            || self.inverse_c2r.omitted_axes.iter().any(|&omit| omit)
            || self.multiply.sequence_len != self.compact_tensor_len
            || self.multiply.batch_count != 1
            || self.multiply.coordinate_count != self.coordinate_count
            || self.multiply.kernel_count != self.kernel_count
            || self.multiply.matrix_layout != self.matrix_layout
            || self.multiply.independent_coordinates
                != (self.matrix_layout.is_none() && self.coordinate_count > 1)
            || self.multiply.scalar != self.scalar
            || !matches!(
                (self.precision, self.scalar, self.external_scalar),
                (
                    Precision::F16StorageF32Compute,
                    ScalarType::F32,
                    ScalarType::F16
                ) | (Precision::F32, ScalarType::F32, ScalarType::F32)
                    | (Precision::F64, ScalarType::F64, ScalarType::F64)
                    | (
                        Precision::F64ComputeF32Storage,
                        ScalarType::F64,
                        ScalarType::F32
                    )
            )
        {
            return Err(VkFftError::InvalidKernelIr(
                "multidimensional real convolution R2C/multiply/C2R ownership is inconsistent",
            ));
        }
        Ok(())
    }

    pub fn output_batch_count(&self) -> usize {
        self.kernel_count
    }

    pub fn output_system_count(&self) -> Result<usize> {
        self.kernel_count
            .checked_mul(self.coordinate_count)
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "multidimensional real convolution output system count",
            })
    }

    pub fn kernel_spectrum(&self) -> &[Complex64] {
        &self.multiply.kernel_spectrum
    }
}

pub fn execute_nd_real_convolution_ir(ir: &NdRealConvolutionIr, input: &[f64]) -> Result<Vec<f64>> {
    ir.validate()?;
    let expected_input = ir.full_tensor_len.checked_mul(ir.coordinate_count).ok_or(
        VkFftError::ArithmeticOverflow {
            operation: "multidimensional real convolution input system size",
        },
    )?;
    if input.len() != expected_input {
        return Err(VkFftError::InputLengthMismatch {
            expected: expected_input,
            actual: input.len(),
        });
    }
    let spectrum = execute_nd_r2c_ir(&ir.forward_r2c, input)?;
    let output_system_count = ir.output_system_count()?;
    let output_compact_len = ir
        .compact_tensor_len
        .checked_mul(output_system_count)
        .ok_or(VkFftError::ArithmeticOverflow {
            operation: "multidimensional real convolution compact fanout size",
        })?;
    let mut multiplied = vec![Complex64::default(); output_compact_len];
    if let Some(matrix) = ir.matrix_layout {
        let kernel_planes = matrix.kernel_plane_count();
        for kernel_id in 0..ir.kernel_count {
            for index in 0..ir.compact_tensor_len {
                for output_coordinate in 0..matrix.matrix_size {
                    let mut sum = Complex64::default();
                    for input_coordinate in 0..matrix.matrix_size {
                        let input_offset = input_coordinate * ir.compact_tensor_len + index;
                        let kernel_plane =
                            matrix.kernel_plane_index(output_coordinate, input_coordinate)?;
                        let kernel_offset = (kernel_id * kernel_planes + kernel_plane)
                            * ir.compact_tensor_len
                            + index;
                        let sequence = ir.multiply.policy.prepare_sequence(spectrum[input_offset]);
                        sum += sequence * ir.multiply.kernel_spectrum[kernel_offset];
                    }
                    let output_offset = (kernel_id * matrix.matrix_size + output_coordinate)
                        * ir.compact_tensor_len
                        + index;
                    multiplied[output_offset] = ir.multiply.policy.normalize_product(sum);
                }
            }
        }
        debug_assert_eq!(
            ir.multiply.kernel_spectrum.len(),
            ir.kernel_count * kernel_planes * ir.compact_tensor_len
        );
    } else {
        for kernel_id in 0..ir.kernel_count {
            for coordinate in 0..ir.coordinate_count {
                let output_base =
                    (kernel_id * ir.coordinate_count + coordinate) * ir.compact_tensor_len;
                let input_base = coordinate * ir.compact_tensor_len;
                for index in 0..ir.compact_tensor_len {
                    multiplied[output_base + index] = ir.multiply.policy.apply(
                        spectrum[input_base + index],
                        ir.multiply.kernel_spectrum[output_base + index],
                    );
                }
            }
        }
    }
    execute_nd_c2r_ir(&ir.inverse_c2r, &multiplied)
}

pub fn execute_nd_convolution_ir(
    ir: &NdConvolutionIr,
    input: &[Complex64],
) -> Result<Vec<Complex64>> {
    ir.validate()?;
    let forward_transform_batch_count =
        ir.batch_count
            .checked_mul(ir.coordinate_count)
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "multidimensional convolution forward input batch count",
            })?;
    let inverse_transform_batch_count = ir
        .output_batch_count()
        .checked_mul(ir.coordinate_count)
        .ok_or(VkFftError::ArithmeticOverflow {
            operation: "multidimensional convolution inverse output batch count",
        })?;
    let expected_input = ir
        .tensor_len
        .checked_mul(forward_transform_batch_count)
        .ok_or(VkFftError::ArithmeticOverflow {
            operation: "multidimensional convolution input element count",
        })?;
    let expected_output = ir
        .tensor_len
        .checked_mul(inverse_transform_batch_count)
        .ok_or(VkFftError::ArithmeticOverflow {
            operation: "multidimensional convolution output element count",
        })?;
    if input.len() != expected_input {
        return Err(VkFftError::InputLengthMismatch {
            expected: expected_input,
            actual: input.len(),
        });
    }
    let spectrum = execute_nd_fft_ir(&ir.forward_fft, input)?;
    let multiplied = if let Some(matrix) = ir.matrix_layout {
        let mut multiplied = vec![Complex64::default(); expected_output];
        let kernel_planes = matrix.kernel_plane_count();
        for batch in 0..ir.batch_count {
            for index in 0..ir.tensor_len {
                for output_coordinate in 0..matrix.matrix_size {
                    let mut sum = Complex64::default();
                    for input_coordinate in 0..matrix.matrix_size {
                        let input_offset = ((batch * ir.coordinate_count + input_coordinate)
                            * ir.tensor_len)
                            + index;
                        let kernel_plane =
                            matrix.kernel_plane_index(output_coordinate, input_coordinate)?;
                        let kernel_offset = kernel_plane * ir.tensor_len + index;
                        let sequence = ir.multiply.policy.prepare_sequence(spectrum[input_offset]);
                        sum += sequence * ir.multiply.kernel_spectrum[kernel_offset];
                    }
                    let output_offset =
                        ((batch * ir.coordinate_count + output_coordinate) * ir.tensor_len) + index;
                    multiplied[output_offset] = ir.multiply.policy.normalize_product(sum);
                }
            }
        }
        debug_assert_eq!(
            ir.multiply.kernel_spectrum.len(),
            kernel_planes * ir.tensor_len
        );
        multiplied
    } else if ir.kernel_count > 1 {
        let mut multiplied = vec![Complex64::default(); expected_output];
        for kernel_id in 0..ir.kernel_count {
            let output_base = kernel_id * ir.tensor_len;
            let kernel_base = kernel_id * ir.tensor_len;
            for index in 0..ir.tensor_len {
                multiplied[output_base + index] = ir.multiply.policy.apply(
                    spectrum[index],
                    ir.multiply.kernel_spectrum[kernel_base + index],
                );
            }
        }
        multiplied
    } else {
        let mut multiplied = spectrum;
        for batch in 0..ir.batch_count {
            let base = batch * ir.tensor_len;
            for index in 0..ir.tensor_len {
                multiplied[base + index] = ir
                    .multiply
                    .policy
                    .apply(multiplied[base + index], ir.multiply.kernel_spectrum[index]);
            }
        }
        multiplied
    };
    execute_nd_fft_ir(&ir.inverse_fft, &multiplied)
}

pub fn execute_convolution_ir(ir: &ConvolutionIr, input: &[Complex64]) -> Result<Vec<Complex64>> {
    ir.validate()?;
    let expected = ir
        .sequence_len
        .checked_mul(ir.batch_count)
        .and_then(|value| value.checked_mul(ir.coordinate_count))
        .ok_or(VkFftError::ArithmeticOverflow {
            operation: "convolution input element count",
        })?;
    if input.len() != expected {
        return Err(VkFftError::InputLengthMismatch {
            expected,
            actual: input.len(),
        });
    }
    // CPU execution keeps the mathematical stages explicit even when the GPU IR owns
    // them in one dispatch; this remains an independent numeric oracle for the fused shader.
    let mut spectrum = execute_one_dim_fft_ir(&ir.forward_fft, input)?;
    if let Some(fused_inverse) = &ir.fused_inverse_stockham {
        return execute_stockham_ir_with_lookup(
            fused_inverse,
            &spectrum,
            Some(ir.kernel_spectrum()),
        );
    }
    if let Some(matrix) = ir.matrix_layout {
        let output_system_count = if ir.kernel_count > 1 {
            ir.kernel_count
        } else {
            ir.batch_count
        };
        let output_len = ir
            .sequence_len
            .checked_mul(ir.coordinate_count)
            .and_then(|value| value.checked_mul(output_system_count))
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "matrix multi-kernel convolution output spectrum element count",
            })?;
        let mut multiplied = vec![Complex64::default(); output_len];
        let kernel_planes = matrix.kernel_plane_count();
        for output_system in 0..output_system_count {
            let input_system = if ir.kernel_count > 1 {
                0
            } else {
                output_system
            };
            let kernel_id = if ir.kernel_count > 1 {
                output_system
            } else {
                0
            };
            let kernel_set_base = kernel_id * kernel_planes * ir.sequence_len;
            for index in 0..ir.sequence_len {
                for output_coordinate in 0..matrix.matrix_size {
                    let mut sum = Complex64::default();
                    for input_coordinate in 0..matrix.matrix_size {
                        let input_offset = ((input_system * ir.coordinate_count
                            + input_coordinate)
                            * ir.sequence_len)
                            + index;
                        let kernel_plane =
                            matrix.kernel_plane_index(output_coordinate, input_coordinate)?;
                        let kernel_offset =
                            kernel_set_base + kernel_plane * ir.sequence_len + index;
                        let sequence = ir.multiply.policy.prepare_sequence(spectrum[input_offset]);
                        sum += sequence * ir.multiply.kernel_spectrum[kernel_offset];
                    }
                    let output_offset = ((output_system * ir.coordinate_count + output_coordinate)
                        * ir.sequence_len)
                        + index;
                    multiplied[output_offset] = ir.multiply.policy.normalize_product(sum);
                }
            }
        }
        spectrum = multiplied;
    } else if ir.kernel_count > 1 {
        let mut multiplied = vec![
            Complex64::default();
            ir.sequence_len.checked_mul(ir.kernel_count).ok_or(
                VkFftError::ArithmeticOverflow {
                    operation: "multi-kernel convolution output spectrum element count",
                }
            )?
        ];
        for kernel_id in 0..ir.kernel_count {
            let kernel_base = kernel_id * ir.sequence_len;
            for index in 0..ir.sequence_len {
                multiplied[kernel_base + index] = ir.multiply.policy.apply(
                    spectrum[index],
                    ir.multiply.kernel_spectrum[kernel_base + index],
                );
            }
        }
        spectrum = multiplied;
    } else {
        for batch in 0..ir.batch_count {
            let base = batch * ir.sequence_len;
            for index in 0..ir.sequence_len {
                spectrum[base + index] = ir
                    .multiply
                    .policy
                    .apply(spectrum[base + index], ir.multiply.kernel_spectrum[index]);
            }
        }
    }
    // The inverse child is always normalized. This matches VkFFT's convolutionStep
    // scaling branch rather than inheriting the caller's standalone inverse flag.
    execute_one_dim_fft_ir(&ir.inverse_fft, &spectrum)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{Backend, GpuVendor};
    use crate::kernel_ir::execute_stockham_ir;
    use crate::program_ir::ProgramIr;
    use crate::reference::dft;

    fn device() -> DeviceProfile {
        let mut profile = DeviceProfile::generic(Backend::Vulkan, GpuVendor::Nvidia);
        profile.supports_f64 = true;
        profile
    }

    fn circular_convolution(input: &[Complex64], kernel: &[Complex64]) -> Vec<Complex64> {
        let n = input.len();
        (0..n)
            .map(|out| {
                (0..n).fold(Complex64::default(), |sum, index| {
                    sum + input[index] * kernel[(out + n - index) % n]
                })
            })
            .collect()
    }

    fn direct_dft_2d(input: &[Complex64], dimensions: [usize; 2]) -> Vec<Complex64> {
        let [rows, cols] = dimensions;
        let mut output = vec![Complex64::default(); rows * cols];
        for k0 in 0..rows {
            for k1 in 0..cols {
                let mut sum = Complex64::default();
                for n0 in 0..rows {
                    for n1 in 0..cols {
                        let phase = -std::f64::consts::TAU
                            * (k0 as f64 * n0 as f64 / rows as f64
                                + k1 as f64 * n1 as f64 / cols as f64);
                        sum += input[n0 * cols + n1] * Complex64::new(phase.cos(), phase.sin());
                    }
                }
                output[k0 * cols + k1] = sum;
            }
        }
        output
    }

    fn direct_circular_convolution_2d(
        input: &[Complex64],
        kernel: &[Complex64],
        dimensions: [usize; 2],
    ) -> Vec<Complex64> {
        let [rows, cols] = dimensions;
        let mut output = vec![Complex64::default(); rows * cols];
        for out0 in 0..rows {
            for out1 in 0..cols {
                let mut sum = Complex64::default();
                for n0 in 0..rows {
                    for n1 in 0..cols {
                        let k0 = (out0 + rows - n0) % rows;
                        let k1 = (out1 + cols - n1) % cols;
                        sum += input[n0 * cols + n1] * kernel[k0 * cols + k1];
                    }
                }
                output[out0 * cols + out1] = sum;
            }
        }
        output
    }

    #[test]
    fn multidimensional_convolution_matches_direct_two_dimensional_circular_oracle() {
        let dimensions = [3usize, 4usize];
        let tensor_len = dimensions.iter().product::<usize>();
        let batch_count = 2usize;
        let kernel = (0..tensor_len)
            .map(|index| {
                let x = index as f64;
                Complex64::new((0.29 * x).cos() + 0.013 * x, (0.17 * x).sin() - 0.009 * x)
            })
            .collect::<Vec<_>>();
        let kernel_spectrum = direct_dft_2d(&kernel, dimensions);
        let input = (0..tensor_len * batch_count)
            .map(|index| {
                let batch = index / tensor_len;
                let local = (index % tensor_len) as f64;
                Complex64::new(
                    0.31 * batch as f64 + (0.11 * local).sin() + 0.004 * local,
                    -0.23 * batch as f64 + (0.07 * local).cos() - 0.003 * local,
                )
            })
            .collect::<Vec<_>>();

        for precision in [Precision::F32, Precision::F64] {
            let ir = NdConvolutionIr::build_from_spectrum(
                FftConfig::new(dimensions.to_vec())
                    .with_batch_count(batch_count)
                    .with_precision(precision)
                    .with_convolution(true),
                kernel_spectrum.clone(),
                device(),
            )
            .unwrap();
            assert_eq!(ir.tensor_len, tensor_len);
            assert_eq!(ir.multiply.sequence_len, tensor_len);
            assert_eq!(ir.multiply.dispatch.x, batch_count as u32);
            let actual = execute_nd_convolution_ir(&ir, &input).unwrap();
            for batch in 0..batch_count {
                let base = batch * tensor_len;
                let expected = direct_circular_convolution_2d(
                    &input[base..base + tensor_len],
                    &kernel,
                    dimensions,
                );
                let max_error = actual[base..base + tensor_len]
                    .iter()
                    .zip(&expected)
                    .map(|(actual, expected)| (*actual - *expected).norm_sqr().sqrt())
                    .fold(0.0, f64::max);
                assert!(
                    max_error <= 8.0e-12,
                    "{precision:?} ND batch {batch} convolution error {max_error:e}"
                );
            }
        }
    }

    #[test]
    fn perform_convolution_matches_direct_circular_convolution() {
        let n = 16usize;
        let batch_count = 2usize;
        let kernel = (0..n)
            .map(|index| {
                let x = index as f64;
                Complex64::new((0.23 * x).cos() + 0.02 * x, (0.17 * x).sin() - 0.01 * x)
            })
            .collect::<Vec<_>>();
        let kernel_spectrum = dft(&kernel, Direction::Forward, false);
        let config = FftConfig::new(vec![n])
            .with_batch_count(batch_count)
            .with_precision(Precision::F32)
            .with_convolution(true);
        let ir = ConvolutionIr::build_from_spectrum(config, kernel_spectrum, device()).unwrap();
        assert!(ir.has_fused_stockham_step());
        assert!(!ir.has_fused_inverse_stockham());
        let input = (0..n * batch_count)
            .map(|index| {
                let x = index as f64;
                Complex64::new((0.11 * x).sin() + 0.003 * x, (0.07 * x).cos() - 0.004 * x)
            })
            .collect::<Vec<_>>();
        let actual = execute_convolution_ir(&ir, &input).unwrap();
        for batch in 0..batch_count {
            let base = batch * n;
            let expected = circular_convolution(&input[base..base + n], &kernel);
            let max_error = actual[base..base + n]
                .iter()
                .zip(&expected)
                .map(|(actual, expected)| (*actual - *expected).norm_sqr().sqrt())
                .fold(0.0, f64::max);
            assert!(
                max_error <= 3.0e-12,
                "batch {batch} convolution error {max_error:e}"
            );
        }
    }

    #[test]
    fn sequence_conjugation_and_cross_power_match_explicit_frequency_policy() {
        let n = 16usize;
        let input = (0..n)
            .map(|index| {
                let x = index as f64;
                Complex64::new(0.7 + (0.13 * x).sin(), -0.2 + (0.17 * x).cos())
            })
            .collect::<Vec<_>>();
        let kernel_spectrum = (0..n)
            .map(|index| {
                let x = index as f64;
                Complex64::new(0.8 + 0.03 * x, 0.25 + 0.02 * x)
            })
            .collect::<Vec<_>>();
        let policy = ConvolutionMultiplyPolicy {
            conjugation: ConvolutionConjugation::Sequence,
            cross_power_spectrum_normalization: true,
        };
        let config = FftConfig::new(vec![n])
            .with_convolution(true)
            .with_convolution_conjugation(ConvolutionConjugation::Sequence)
            .with_cross_power_spectrum_normalization(true);
        let ir =
            ConvolutionIr::build_from_spectrum(config, kernel_spectrum.clone(), device()).unwrap();
        assert_eq!(ir.multiply.policy, policy);
        assert_eq!(ir.fused_stockham_step.as_ref().unwrap().policy, policy);

        let actual = execute_convolution_ir(&ir, &input).unwrap();
        let mut spectrum = dft(&input, Direction::Forward, false);
        for (value, kernel) in spectrum.iter_mut().zip(&kernel_spectrum) {
            let sequence = value.conj();
            let product = sequence * *kernel;
            *value = product.scale(1.0 / product.norm_sqr().sqrt());
        }
        let expected = dft(&spectrum, Direction::Inverse, true);
        let error = actual
            .iter()
            .zip(&expected)
            .map(|(actual, expected)| (*actual - *expected).norm_sqr().sqrt())
            .fold(0.0f64, f64::max);
        assert!(
            error <= 4.0e-12,
            "conjugated cross-power CPU error {error:e}"
        );
    }

    #[test]
    fn kernel_conjugation_mode_remains_fail_closed_against_pinned_codegen() {
        let error = FftConfig::new(vec![16])
            .with_convolution(true)
            .with_convolution_conjugation(ConvolutionConjugation::Kernel)
            .validate()
            .unwrap_err();
        assert!(matches!(error, VkFftError::UnsupportedKernelPath(_)));
    }

    #[test]
    fn matrix_convolution_matches_direct_spatial_oracle_for_supported_layouts() {
        let n = 16usize;
        let batch_count = 2usize;
        for (matrix_size, symmetric) in [(2usize, false), (2, true), (3, false)] {
            let layout = ConvolutionMatrixLayout {
                matrix_size,
                symmetric_kernel: symmetric,
            };
            layout.validate().unwrap();
            let kernel_planes = layout.kernel_plane_count();
            let spatial_kernels = (0..kernel_planes)
                .map(|plane| {
                    (0..n)
                        .map(|index| {
                            let x = index as f64;
                            Complex64::new(
                                0.2 + 0.03 * plane as f64 + (0.11 * x).cos(),
                                -0.1 + 0.02 * plane as f64 + (0.07 * x).sin(),
                            )
                        })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            let kernel_spectrum = spatial_kernels
                .iter()
                .flat_map(|kernel| dft(kernel, Direction::Forward, false))
                .collect::<Vec<_>>();
            let input = (0..batch_count * matrix_size * n)
                .map(|index| {
                    let local = (index % n) as f64;
                    let coordinate = (index / n) % matrix_size;
                    let batch = index / (matrix_size * n);
                    Complex64::new(
                        0.4 * batch as f64 + 0.1 * coordinate as f64 + (0.13 * local).sin(),
                        -0.2 * batch as f64 + 0.07 * coordinate as f64 + (0.17 * local).cos(),
                    )
                })
                .collect::<Vec<_>>();
            let config = FftConfig::new(vec![n])
                .with_batch_count(batch_count)
                .with_convolution(true)
                .with_matrix_convolution(matrix_size)
                .with_symmetric_convolution_kernel(symmetric);
            let ir = ConvolutionIr::build_from_spectrum(config, kernel_spectrum, device()).unwrap();
            assert_eq!(ir.coordinate_count, matrix_size);
            assert_eq!(ir.matrix_layout, Some(layout));
            assert!(!ir.has_fused_stockham_step());
            assert!(!ir.has_fused_two_upload_stockham());
            assert!(!ir.has_fused_three_upload_stockham());
            assert!(!ir.has_fused_inverse_stockham());
            assert_eq!(ir.forward_fft.batch_count(), batch_count * matrix_size);

            let actual = execute_convolution_ir(&ir, &input).unwrap();
            let mut expected = vec![Complex64::default(); input.len()];
            for batch in 0..batch_count {
                for output_coordinate in 0..matrix_size {
                    let output_base = (batch * matrix_size + output_coordinate) * n;
                    for input_coordinate in 0..matrix_size {
                        let input_base = (batch * matrix_size + input_coordinate) * n;
                        let kernel_plane = layout
                            .kernel_plane_index(output_coordinate, input_coordinate)
                            .unwrap();
                        let term = circular_convolution(
                            &input[input_base..input_base + n],
                            &spatial_kernels[kernel_plane],
                        );
                        for index in 0..n {
                            expected[output_base + index] += term[index];
                        }
                    }
                }
            }
            let error = actual
                .iter()
                .zip(&expected)
                .map(|(actual, expected)| (*actual - *expected).norm_sqr().sqrt())
                .fold(0.0f64, f64::max);
            assert!(
                error <= 2.0e-11,
                "matrix {matrix_size} symmetric={symmetric} error {error:e}"
            );
        }
    }

    #[test]
    fn matrix_conjugated_cross_power_normalizes_after_each_row_sum() {
        let n = 16usize;
        let matrix_size = 2usize;
        let input = (0..matrix_size * n)
            .map(|index| {
                let coordinate = index / n;
                let x = (index % n) as f64;
                Complex64::new(
                    0.8 + 0.15 * coordinate as f64 + (0.09 * x).sin(),
                    -0.3 + 0.11 * coordinate as f64 + (0.14 * x).cos(),
                )
            })
            .collect::<Vec<_>>();
        let kernel_spectrum = (0..matrix_size * matrix_size * n)
            .map(|index| {
                let plane = index / n;
                let x = (index % n) as f64;
                Complex64::new(
                    0.6 + 0.08 * plane as f64 + 0.005 * x,
                    0.2 + 0.01 * plane as f64,
                )
            })
            .collect::<Vec<_>>();
        let ir = ConvolutionIr::build_from_spectrum(
            FftConfig::new(vec![n])
                .with_convolution(true)
                .with_matrix_convolution(matrix_size)
                .with_convolution_conjugation(ConvolutionConjugation::Sequence)
                .with_cross_power_spectrum_normalization(true),
            kernel_spectrum.clone(),
            device(),
        )
        .unwrap();
        let actual = execute_convolution_ir(&ir, &input).unwrap();

        let input_spectra = (0..matrix_size)
            .map(|coordinate| {
                dft(
                    &input[coordinate * n..(coordinate + 1) * n],
                    Direction::Forward,
                    false,
                )
            })
            .collect::<Vec<_>>();
        let mut expected = vec![Complex64::default(); matrix_size * n];
        for output_coordinate in 0..matrix_size {
            let mut output_spectrum = vec![Complex64::default(); n];
            for index in 0..n {
                let mut sum = Complex64::default();
                for (input_coordinate, input_spectrum) in input_spectra.iter().enumerate() {
                    let kernel_plane = output_coordinate * matrix_size + input_coordinate;
                    sum += input_spectrum[index].conj() * kernel_spectrum[kernel_plane * n + index];
                }
                output_spectrum[index] = sum.scale(1.0 / sum.norm_sqr().sqrt());
            }
            expected[output_coordinate * n..(output_coordinate + 1) * n].copy_from_slice(&dft(
                &output_spectrum,
                Direction::Inverse,
                true,
            ));
        }
        let error = actual
            .iter()
            .zip(&expected)
            .map(|(actual, expected)| (*actual - *expected).norm_sqr().sqrt())
            .fold(0.0f64, f64::max);
        assert!(
            error <= 4.0e-12,
            "matrix conjugated cross-power error {error:e}"
        );
    }

    #[test]
    fn multi_kernel_convolution_reuses_one_forward_input_and_expands_outputs() {
        let n = 16usize;
        let kernel_count = 3usize;
        let input = (0..n)
            .map(|index| {
                let x = index as f64;
                Complex64::new(0.3 + (0.13 * x).sin(), -0.4 + (0.17 * x).cos())
            })
            .collect::<Vec<_>>();
        let spatial_kernels = (0..kernel_count)
            .map(|kernel_id| {
                (0..n)
                    .map(|index| {
                        let x = index as f64;
                        Complex64::new(
                            0.2 + 0.07 * kernel_id as f64 + (0.11 * x).cos(),
                            -0.15 + 0.03 * kernel_id as f64 + (0.09 * x).sin(),
                        )
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let kernel_spectrum = spatial_kernels
            .iter()
            .flat_map(|kernel| dft(kernel, Direction::Forward, false))
            .collect::<Vec<_>>();
        let ir = ConvolutionIr::build_from_spectrum(
            FftConfig::new(vec![n])
                .with_convolution(true)
                .with_convolution_kernel_count(kernel_count),
            kernel_spectrum,
            device(),
        )
        .unwrap();
        assert_eq!(ir.kernel_count, kernel_count);
        assert_eq!(ir.batch_count, 1);
        assert_eq!(ir.output_batch_count(), kernel_count);
        assert_eq!(ir.forward_fft.batch_count(), 1);
        assert_eq!(ir.inverse_fft.batch_count(), kernel_count);
        assert!(!ir.has_fused_stockham_step());
        assert!(ir.has_fused_multi_kernel_stockham_step());
        assert!(!ir.has_fused_two_upload_stockham());
        assert!(!ir.has_fused_three_upload_stockham());
        assert!(!ir.has_fused_inverse_stockham());
        let step = ir.fused_multi_kernel_stockham_step.as_ref().unwrap();
        assert_eq!(
            step.required_shared_memory_bytes,
            3 * n * step.scalar.complex_bytes()
        );
        let program = ProgramIr::convolution(&ir).unwrap();
        assert_eq!(program.passes.len(), 1);
        assert_eq!(program.passes[0].name, step.name);

        let actual = execute_convolution_ir(&ir, &input).unwrap();
        let expected = spatial_kernels
            .iter()
            .flat_map(|kernel| circular_convolution(&input, kernel))
            .collect::<Vec<_>>();
        assert_eq!(actual.len(), kernel_count * n);
        let error = actual
            .iter()
            .zip(&expected)
            .map(|(actual, expected)| (*actual - *expected).norm_sqr().sqrt())
            .fold(0.0f64, f64::max);
        assert!(error <= 3.0e-12, "multi-kernel convolution error {error:e}");
    }

    #[test]
    fn multi_kernel_two_upload_stockham_uses_pinned_three_launch_graph() {
        let n = 8_192usize;
        let kernel_count = 3usize;
        let ir = ConvolutionIr::build_from_spectrum(
            FftConfig::new(vec![n])
                .with_convolution(true)
                .with_convolution_kernel_count(kernel_count),
            vec![Complex64::new(1.0, 0.0); kernel_count * n],
            device(),
        )
        .unwrap();
        assert_eq!(ir.batch_count, 1);
        assert_eq!(ir.kernel_count, kernel_count);
        assert!(!ir.has_fused_multi_kernel_stockham_step());
        assert!(ir.has_fused_two_upload_multi_kernel_stockham());
        assert!(!ir.has_fused_two_upload_stockham());
        let step = ir.fused_two_upload_multi_kernel_stockham.as_ref().unwrap();
        assert_eq!(
            (
                step.forward_mapping.left_len,
                step.forward_mapping.right_len,
                step.forward_mapping.outer_batch_count,
            ),
            (128, 64, 1)
        );
        assert_eq!(
            (
                step.inverse_mapping.left_len,
                step.inverse_mapping.right_len,
                step.inverse_mapping.outer_batch_count,
            ),
            (128, 64, kernel_count)
        );
        assert_eq!(
            step.special_workgroup_size,
            WorkgroupSize { x: 16, y: 16, z: 1 }
        );
        assert_eq!(step.special_dispatch, DispatchGeometry { x: 4, y: 1, z: 1 });
        assert_eq!(step.required_shared_memory_bytes, 32 * 1024);

        let program = ProgramIr::convolution(&ir).unwrap();
        assert_eq!(program.passes.len(), 3);
        assert_eq!(program.passes[0].name, step.forward_high.name);
        assert_eq!(program.passes[1].name, step.name);
        assert_eq!(program.passes[2].name, step.inverse_high.name);
        assert!(
            !program
                .passes
                .iter()
                .any(|pass| pass.name == ir.multiply.name)
        );
        let forward_mid = program
            .resources
            .iter()
            .find(|resource| {
                resource
                    .name
                    .starts_with("convolution_two_upload_multi_kernel_forward_mid")
            })
            .unwrap();
        let inverse_mid = program
            .resources
            .iter()
            .find(|resource| {
                resource
                    .name
                    .starts_with("convolution_two_upload_multi_kernel_inverse_mid")
            })
            .unwrap();
        assert_eq!(forward_mid.elements, n);
        assert_eq!(inverse_mid.elements, kernel_count * n);
    }

    #[test]
    fn matrix_multi_kernel_convolution_fans_out_one_vector_to_kernel_sets() {
        let n = 16usize;
        let matrix_size = 2usize;
        let kernel_count = 3usize;
        let layout = ConvolutionMatrixLayout {
            matrix_size,
            symmetric_kernel: true,
        };
        let kernel_planes = layout.kernel_plane_count();
        let input = (0..matrix_size * n)
            .map(|index| {
                let coordinate = index / n;
                let x = (index % n) as f64;
                Complex64::new(
                    0.55 + 0.13 * coordinate as f64 + (0.07 * x).sin(),
                    -0.24 + 0.09 * coordinate as f64 + (0.12 * x).cos(),
                )
            })
            .collect::<Vec<_>>();
        let kernel_spectrum = (0..kernel_count * kernel_planes * n)
            .map(|index| {
                let frequency = index % n;
                let plane = (index / n) % kernel_planes;
                let kernel_id = index / (kernel_planes * n);
                Complex64::new(
                    0.64 + 0.11 * kernel_id as f64 + 0.05 * plane as f64 + 0.003 * frequency as f64,
                    0.12 + 0.02 * kernel_id as f64 + 0.013 * plane as f64
                        - 0.001 * frequency as f64,
                )
            })
            .collect::<Vec<_>>();
        let ir = ConvolutionIr::build_from_spectrum(
            FftConfig::new(vec![n])
                .with_convolution(true)
                .with_matrix_convolution(matrix_size)
                .with_symmetric_convolution_kernel(true)
                .with_convolution_kernel_count(kernel_count)
                .with_convolution_conjugation(ConvolutionConjugation::Sequence)
                .with_cross_power_spectrum_normalization(true),
            kernel_spectrum.clone(),
            device(),
        )
        .unwrap();
        assert_eq!(ir.batch_count, 1);
        assert_eq!(ir.coordinate_count, matrix_size);
        assert_eq!(ir.kernel_count, kernel_count);
        assert_eq!(ir.forward_fft.batch_count(), matrix_size);
        assert_eq!(ir.inverse_fft.batch_count(), kernel_count * matrix_size);
        assert!(ir.has_fused_matrix_stockham_step());
        let step = ir.fused_matrix_stockham_step.as_ref().unwrap();
        assert_eq!(step.coordinate_count, matrix_size);
        assert_eq!(step.kernel_count, kernel_count);
        assert_eq!(
            step.required_shared_memory_bytes,
            (matrix_size + 2) * n * step.scalar.complex_bytes()
        );
        let program = ProgramIr::convolution(&ir).unwrap();
        assert_eq!(program.passes.len(), 1);
        assert_eq!(program.passes[0].name, step.name);
        assert_eq!(program.resources[0].elements, matrix_size * n);
        assert_eq!(
            program.resources[1].elements,
            kernel_count * matrix_size * n
        );
        assert_eq!(ir.multiply.dispatch.x, kernel_count as u32);

        let actual = execute_convolution_ir(&ir, &input).unwrap();
        let input_spectra = (0..matrix_size)
            .map(|coordinate| {
                dft(
                    &input[coordinate * n..(coordinate + 1) * n],
                    Direction::Forward,
                    false,
                )
            })
            .collect::<Vec<_>>();
        let mut expected = vec![Complex64::default(); kernel_count * matrix_size * n];
        for kernel_id in 0..kernel_count {
            let kernel_set_base = kernel_id * kernel_planes * n;
            for output_coordinate in 0..matrix_size {
                let mut output_spectrum = vec![Complex64::default(); n];
                for index in 0..n {
                    let mut sum = Complex64::default();
                    for (input_coordinate, input_spectrum) in input_spectra.iter().enumerate() {
                        let plane = layout
                            .kernel_plane_index(output_coordinate, input_coordinate)
                            .unwrap();
                        sum += input_spectrum[index].conj()
                            * kernel_spectrum[kernel_set_base + plane * n + index];
                    }
                    output_spectrum[index] = sum.scale(1.0 / sum.norm_sqr().sqrt());
                }
                let output_base = (kernel_id * matrix_size + output_coordinate) * n;
                expected[output_base..output_base + n].copy_from_slice(&dft(
                    &output_spectrum,
                    Direction::Inverse,
                    true,
                ));
            }
        }
        assert_eq!(actual.len(), expected.len());
        let error = actual
            .iter()
            .zip(&expected)
            .map(|(actual, expected)| (*actual - *expected).norm_sqr().sqrt())
            .fold(0.0f64, f64::max);
        assert!(error <= 5.0e-12, "matrix multi-kernel error {error:e}");
    }

    #[test]
    fn unsupported_matrix_application_edges_remain_fail_closed() {
        let symmetric_3x3 = FftConfig::new(vec![16])
            .with_convolution(true)
            .with_matrix_convolution(3)
            .with_symmetric_convolution_kernel(true)
            .validate()
            .unwrap_err();
        assert!(matches!(
            symmetric_3x3,
            VkFftError::UnsupportedKernelPath(_)
        ));

        let batched_multiple_kernels = FftConfig::new(vec![16])
            .with_batch_count(2)
            .with_convolution(true)
            .with_convolution_kernel_count(2)
            .validate()
            .unwrap_err();
        assert!(matches!(
            &batched_multiple_kernels,
            VkFftError::UnsupportedKernelPath(message)
                if message.contains("one-input fan-out")
                    && message.contains("numberBatches*numberKernels")
                    && message.contains("sample_52")
                    && message.contains("kernel batchID")
        ));

        FftConfig::new(vec![16])
            .with_convolution(true)
            .with_matrix_convolution(2)
            .with_convolution_kernel_count(2)
            .validate()
            .unwrap();

        let layout = ConvolutionMatrixLayout {
            matrix_size: 2,
            symmetric_kernel: true,
        };
        assert_eq!(layout.kernel_plane_count(), 3);
        assert_eq!(layout.kernel_plane_index(0, 0).unwrap(), 0);
        assert_eq!(layout.kernel_plane_index(0, 1).unwrap(), 1);
        assert_eq!(layout.kernel_plane_index(1, 0).unwrap(), 1);
        assert_eq!(layout.kernel_plane_index(1, 1).unwrap(), 2);
    }

    #[test]
    fn perform_convolution_n8388608_materializes_upstream_three_upload_schedule() {
        let n = 8_388_608usize;
        let mut profile = device();
        profile.shared_memory_bytes = 48 * 1024;
        profile.shared_memory_pow2_bytes = 32 * 1024;
        profile.max_threads_per_block = 1024;
        profile.max_workgroup_size = [1024, 1024, 64];
        let ir = ConvolutionIr::build_from_spectrum(
            FftConfig::new(vec![n])
                .with_precision(Precision::F32)
                .with_convolution(true),
            vec![Complex64::new(1.0, 0.0); n],
            profile,
        )
        .unwrap();
        assert!(!ir.has_fused_stockham_step());
        assert!(!ir.has_fused_two_upload_stockham());
        assert!(ir.has_fused_three_upload_stockham());
        assert!(!ir.has_fused_inverse_stockham());
        let three_upload = ir
            .fused_three_upload_stockham
            .as_ref()
            .expect("N8388608 convolution must materialize the three-upload path");
        assert_eq!(three_upload.mapping.axis_split, [4096, 32, 64]);
        assert_eq!(three_upload.scalar, ScalarType::F32);
        assert_eq!(three_upload.required_shared_memory_bytes, 34 * 1024);
        assert_eq!(three_upload.special_workgroup_size.x, 512);
        assert_eq!(three_upload.special_workgroup_size.y, 1);
        assert_eq!(
            three_upload.inverse_middle.io_mapping,
            StockhamIoMapping::FourStepThreeUploadPreTwiddle(ThreeUploadPreTwiddleMapping {
                three_upload: three_upload.mapping,
                axis_upload_id: 1,
                direction: Direction::Inverse,
            })
        );
        assert_eq!(
            three_upload.inverse_high.io_mapping,
            StockhamIoMapping::FourStepThreeUploadPreTwiddle(ThreeUploadPreTwiddleMapping {
                three_upload: three_upload.mapping,
                axis_upload_id: 2,
                direction: Direction::Inverse,
            })
        );
        assert!(!stockham_store_normalize(&three_upload.inverse_middle).unwrap());
        assert!(!stockham_store_normalize(&three_upload.inverse_high).unwrap());
        let program = ProgramIr::convolution(&ir).unwrap();
        assert_eq!(program.passes.len(), 5);
        assert_eq!(program.passes[2].name, three_upload.name);
        assert_eq!(program.passes[2].bindings.len(), 3);
        let scratch_count = program
            .resources
            .iter()
            .filter(|resource| resource.kind == crate::program_ir::ProgramResourceKind::Scratch)
            .count();
        assert_eq!(scratch_count, 2);
    }

    #[test]
    fn multi_kernel_three_upload_stockham_uses_pinned_five_launch_graph() {
        let n = 8_388_608usize;
        let kernel_count = 3usize;
        let mut profile = device();
        profile.shared_memory_bytes = 48 * 1024;
        profile.shared_memory_pow2_bytes = 32 * 1024;
        profile.max_threads_per_block = 1024;
        profile.max_workgroup_size = [1024, 1024, 64];
        let ir = ConvolutionIr::build_from_spectrum(
            FftConfig::new(vec![n])
                .with_precision(Precision::F32)
                .with_convolution(true)
                .with_convolution_kernel_count(kernel_count),
            vec![Complex64::new(1.0, 0.0); kernel_count * n],
            profile,
        )
        .unwrap();
        assert!(!ir.has_fused_multi_kernel_stockham_step());
        assert!(!ir.has_fused_two_upload_multi_kernel_stockham());
        assert!(ir.has_fused_three_upload_multi_kernel_stockham());
        assert!(!ir.has_fused_three_upload_stockham());
        let step = ir
            .fused_three_upload_multi_kernel_stockham
            .as_ref()
            .unwrap();
        assert_eq!(step.forward_mapping.axis_split, [4096, 32, 64]);
        assert_eq!(step.forward_mapping.outer_batch_count, 1);
        assert_eq!(step.inverse_mapping.axis_split, [4096, 32, 64]);
        assert_eq!(step.inverse_mapping.outer_batch_count, kernel_count);
        assert_eq!(step.forward_low.batch_count, 2_048);
        assert_eq!(step.inverse_low.batch_count, kernel_count * 2_048);
        assert_eq!(step.required_shared_memory_bytes, 34 * 1024);
        assert_eq!(
            step.special_workgroup_size,
            WorkgroupSize { x: 512, y: 1, z: 1 }
        );

        let program = ProgramIr::convolution(&ir).unwrap();
        assert_eq!(program.passes.len(), 5);
        assert_eq!(program.passes[2].name, step.name);
        assert!(
            !program
                .passes
                .iter()
                .any(|pass| pass.name == ir.multiply.name)
        );
        let scratch = program
            .resources
            .iter()
            .filter(|resource| resource.kind == crate::program_ir::ProgramResourceKind::Scratch)
            .collect::<Vec<_>>();
        assert_eq!(scratch.len(), 2);
        assert!(
            scratch
                .iter()
                .all(|resource| resource.elements == kernel_count * n)
        );
    }

    #[test]
    fn perform_convolution_n8192_materializes_upstream_two_upload_schedule() {
        let n = 8_192usize;
        let mut profile = device();
        profile.shared_memory_bytes = 32 * 1024;
        profile.shared_memory_pow2_bytes = 32 * 1024;
        profile.max_threads_per_block = 1024;
        profile.max_workgroup_size = [1024, 1024, 64];
        let config = FftConfig::new(vec![n])
            .with_precision(Precision::F32)
            .with_convolution(true);
        let kernel_spectrum = (0..n)
            .map(|index| {
                let x = index as f64;
                Complex64::new(0.8 + 0.00001 * x, 0.15 * (0.007 * x).sin())
            })
            .collect::<Vec<_>>();
        let ir =
            ConvolutionIr::build_from_spectrum(config, kernel_spectrum.clone(), profile).unwrap();
        for transform in [&ir.forward_fft, &ir.inverse_fft] {
            let OneDimFftIr::Recursive(recursive) = transform else {
                panic!("N8192 convolution must retain recursive Stockham");
            };
            let schedule = recursive
                .stockham_upload_schedule
                .as_ref()
                .expect("N8192 convolution lost Stockham upload schedule");
            assert_eq!(schedule.register_boost, 1);
            assert_eq!(schedule.upload_count, 2);
            assert_eq!(schedule.axis_split, vec![128, 64]);
            assert_eq!(
                recursive
                    .four_step_stockham_upload_kernels()
                    .unwrap()
                    .unwrap()
                    .len(),
                2
            );
        }
        assert!(!ir.has_fused_stockham_step());
        assert!(ir.has_fused_two_upload_stockham());
        assert!(!ir.has_fused_inverse_stockham());
        let program = ProgramIr::convolution(&ir).unwrap();
        assert_eq!(program.passes.len(), 3);
        let two_upload = ir
            .fused_two_upload_stockham
            .as_ref()
            .expect("N8192 convolution must materialize the two-upload fused path");
        assert_eq!(two_upload.mapping.left_len, 128);
        assert_eq!(two_upload.mapping.right_len, 64);
        assert_eq!(two_upload.required_shared_memory_bytes, 32 * 1024);
        assert_eq!(
            two_upload.inverse_high.io_mapping,
            StockhamIoMapping::FourStepRightPreTwiddle(FourStepPreTwiddleMapping {
                four_step: two_upload.mapping,
                direction: Direction::Inverse,
            })
        );
        assert!(!stockham_store_normalize(&two_upload.inverse_high).unwrap());

        let input = (0..n)
            .map(|index| {
                let x = index as f64;
                Complex64::new(
                    (0.013 * x).sin() + 1.0e-5 * x,
                    (0.019 * x).cos() - 8.0e-6 * x,
                )
            })
            .collect::<Vec<_>>();
        let reference = execute_convolution_ir(&ir, &input).unwrap();
        let OneDimFftIr::Recursive(forward) = &ir.forward_fft else {
            unreachable!()
        };
        let OneDimFftIr::Recursive(inverse) = &ir.inverse_fft else {
            unreachable!()
        };
        let forward_uploads = forward
            .four_step_stockham_upload_kernels()
            .unwrap()
            .unwrap();
        let inverse_uploads = inverse
            .four_step_stockham_upload_kernels()
            .unwrap()
            .unwrap();
        let [forward_u1, forward_u0] = forward_uploads.as_slice() else {
            panic!("N8192 forward convolution must have exactly two physical uploads")
        };
        let [_inverse_u1, inverse_u0] = inverse_uploads.as_slice() else {
            panic!("N8192 inverse convolution must have exactly two physical uploads")
        };
        let StockhamIoMapping::FourStepLeft(mapping) = forward_u0.io_mapping else {
            panic!("N8192 forward upload0 must own FourStepLeft mapping")
        };
        assert_eq!(
            inverse_u0.io_mapping,
            StockhamIoMapping::FourStepLeft(mapping)
        );
        let forward_mid = execute_stockham_ir(forward_u1, &input).unwrap();
        let full_spectrum = execute_stockham_ir(forward_u0, &forward_mid).unwrap();
        let mut special_u0_input = vec![Complex64::default(); n];
        for k2 in 0..mapping.right_len {
            for k1 in 0..mapping.left_len {
                let full_bin = k2 + mapping.right_len * k1;
                special_u0_input[k2 * mapping.left_len + k1] =
                    full_spectrum[full_bin] * kernel_spectrum[full_bin];
            }
        }
        let mut direct_inverse_u1_input = vec![Complex64::default(); n];
        for k2 in 0..mapping.right_len {
            let local_spectrum = (0..mapping.left_len)
                .map(|k1| special_u0_input[k2 * mapping.left_len + k1])
                .collect::<Vec<_>>();
            let local_spatial = dft(&local_spectrum, Direction::Inverse, true);
            for (n1, value) in local_spatial.into_iter().enumerate() {
                direct_inverse_u1_input[k2 * mapping.left_len + n1] = value;
            }
        }
        let mut direct_candidate = vec![Complex64::default(); n];
        for n1 in 0..mapping.left_len {
            let twiddled = (0..mapping.right_len)
                .map(|k2| {
                    let angle = std::f64::consts::TAU * (n1 * k2) as f64 / n as f64;
                    direct_inverse_u1_input[k2 * mapping.left_len + n1] * Complex64::exp_i(angle)
                })
                .collect::<Vec<_>>();
            let line = dft(&twiddled, Direction::Inverse, true);
            for (n2, value) in line.into_iter().enumerate() {
                direct_candidate[n1 + mapping.left_len * n2] = value;
            }
        }
        let direct_error = direct_candidate
            .iter()
            .zip(&reference)
            .map(|(actual, expected)| (*actual - *expected).norm_sqr().sqrt())
            .fold(0.0, f64::max);
        assert!(
            direct_error <= 3.0e-11,
            "N8192 direct factored convolution error {direct_error:e}"
        );
        let inverse_u0_scattered = execute_stockham_ir(inverse_u0, &special_u0_input).unwrap();
        let mut inverse_u1_input = vec![Complex64::default(); n];
        for k2 in 0..mapping.right_len {
            for n1 in 0..mapping.left_len {
                inverse_u1_input[k2 * mapping.left_len + n1] =
                    inverse_u0_scattered[k2 + mapping.right_len * n1];
            }
        }
        let u0_error = inverse_u1_input
            .iter()
            .zip(&direct_inverse_u1_input)
            .map(|(actual, expected)| (*actual - *expected).norm_sqr().sqrt())
            .fold(0.0, f64::max);
        assert!(
            u0_error <= 3.0e-11,
            "N8192 inverse-u0 logical error {u0_error:e}"
        );
        // The production special-u0 owns the complete 1/N convolution normalization.
        // The ordinary inverse-u0 oracle above applied 1/A, so divide its matrix output
        // by B before feeding the production inverse-high kernel, whose store normalization
        // is deliberately disabled.
        for value in &mut inverse_u1_input {
            *value = value.scale(1.0 / mapping.right_len as f64);
        }
        let candidate = execute_stockham_ir(&two_upload.inverse_high, &inverse_u1_input).unwrap();
        let max_error = candidate
            .iter()
            .zip(&reference)
            .map(|(actual, expected)| (*actual - *expected).norm_sqr().sqrt())
            .fold(0.0, f64::max);
        assert!(
            max_error <= 3.0e-11,
            "N8192 three-launch topology error {max_error:e}"
        );
    }

    #[test]
    fn perform_convolution_n5760_materializes_mixed_radix_two_upload_topology() {
        let n = 5_760usize;
        let mut profile = device();
        profile.shared_memory_bytes = 32 * 1024;
        profile.shared_memory_pow2_bytes = 32 * 1024;
        profile.max_threads_per_block = 1024;
        profile.max_workgroup_size = [1024, 1024, 64];
        let kernel_spectrum = (0..n)
            .map(|index| {
                let x = index as f64;
                Complex64::new(0.78 + 0.06 * (0.007 * x).cos(), 0.05 * (0.011 * x).sin())
            })
            .collect::<Vec<_>>();
        let ir = ConvolutionIr::build_from_spectrum(
            FftConfig::new(vec![n]).with_convolution(true),
            kernel_spectrum.clone(),
            profile,
        )
        .unwrap();
        assert!(ir.has_fused_two_upload_stockham());
        assert!(!ir.has_fused_inverse_stockham());
        let two_upload = ir
            .fused_two_upload_stockham
            .as_ref()
            .expect("N5760 convolution must materialize the mixed-radix two-upload path");
        assert_eq!(two_upload.scalar, ScalarType::F32);
        assert_eq!(two_upload.mapping.left_len, 80);
        assert_eq!(two_upload.mapping.right_len, 72);
        assert_eq!(two_upload.required_shared_memory_bytes, 20_480);
        assert_eq!(two_upload.special_twiddle_lut_len().unwrap(), None);
        assert_eq!(ProgramIr::convolution(&ir).unwrap().passes.len(), 3);

        let input = (0..n)
            .map(|index| {
                let x = index as f64;
                Complex64::new(
                    (0.014 * x).sin() + 1.2e-5 * x,
                    (0.021 * x).cos() - 9.0e-6 * x,
                )
            })
            .collect::<Vec<_>>();
        let reference = execute_convolution_ir(&ir, &input).unwrap();
        let OneDimFftIr::Recursive(forward) = &ir.forward_fft else {
            unreachable!()
        };
        let OneDimFftIr::Recursive(inverse) = &ir.inverse_fft else {
            unreachable!()
        };
        let forward_uploads = forward
            .four_step_stockham_upload_kernels()
            .unwrap()
            .unwrap();
        let inverse_uploads = inverse
            .four_step_stockham_upload_kernels()
            .unwrap()
            .unwrap();
        let [forward_u1, forward_u0] = forward_uploads.as_slice() else {
            panic!("N5760 forward convolution must have exactly two uploads")
        };
        let [_inverse_u1, inverse_u0] = inverse_uploads.as_slice() else {
            panic!("N5760 inverse convolution must have exactly two uploads")
        };
        let StockhamIoMapping::FourStepLeft(mapping) = forward_u0.io_mapping else {
            panic!("N5760 upload0 must own FourStepLeft mapping")
        };
        assert_eq!((mapping.left_len, mapping.right_len), (80, 72));
        let forward_mid = execute_stockham_ir(forward_u1, &input).unwrap();
        let full_spectrum = execute_stockham_ir(forward_u0, &forward_mid).unwrap();
        let mut special_u0_input = vec![Complex64::default(); n];
        for k2 in 0..mapping.right_len {
            for k1 in 0..mapping.left_len {
                let full_bin = k2 + mapping.right_len * k1;
                special_u0_input[k2 * mapping.left_len + k1] =
                    full_spectrum[full_bin] * kernel_spectrum[full_bin];
            }
        }
        let inverse_u0_scattered = execute_stockham_ir(inverse_u0, &special_u0_input).unwrap();
        let mut inverse_u1_input = vec![Complex64::default(); n];
        for k2 in 0..mapping.right_len {
            for n1 in 0..mapping.left_len {
                inverse_u1_input[k2 * mapping.left_len + n1] = inverse_u0_scattered
                    [k2 + mapping.right_len * n1]
                    .scale(1.0 / mapping.right_len as f64);
            }
        }
        let candidate = execute_stockham_ir(&two_upload.inverse_high, &inverse_u1_input).unwrap();
        let max_error = candidate
            .iter()
            .zip(&reference)
            .map(|(actual, expected)| (*actual - *expected).norm_sqr().sqrt())
            .fold(0.0, f64::max);
        assert!(
            max_error <= 3.0e-11,
            "N5760 mixed-radix three-launch topology error {max_error:e}"
        );
    }

    #[test]
    fn perform_convolution_f64_n4096_materializes_two_upload_lut_topology() {
        let n = 4_096usize;
        let mut profile = device();
        profile.shared_memory_bytes = 32 * 1024;
        profile.shared_memory_pow2_bytes = 32 * 1024;
        profile.max_threads_per_block = 1024;
        profile.max_workgroup_size = [1024, 1024, 64];
        let kernel_spectrum = (0..n)
            .map(|index| {
                let x = index as f64;
                Complex64::new(0.85 + 0.04 * (0.005 * x).cos(), 0.03 * (0.009 * x).sin())
            })
            .collect::<Vec<_>>();
        let ir = ConvolutionIr::build_from_spectrum(
            FftConfig::new(vec![n])
                .with_precision(Precision::F64)
                .with_convolution(true),
            kernel_spectrum.clone(),
            profile,
        )
        .unwrap();
        assert!(ir.has_fused_two_upload_stockham());
        assert!(!ir.has_fused_inverse_stockham());
        let two_upload = ir
            .fused_two_upload_stockham
            .as_ref()
            .expect("F64 N4096 convolution must materialize the two-upload path");
        assert_eq!(two_upload.scalar, ScalarType::F64);
        assert_eq!(two_upload.mapping.left_len, 64);
        assert_eq!(two_upload.mapping.right_len, 64);
        assert_eq!(two_upload.required_shared_memory_bytes, 16 * 1024);
        assert_eq!(two_upload.special_twiddle_lut_len().unwrap(), Some(n));
        assert_eq!(ProgramIr::convolution(&ir).unwrap().passes.len(), 3);

        let input = (0..n)
            .map(|index| {
                let x = index as f64;
                Complex64::new(
                    (0.017 * x).sin() + 1.0e-5 * x,
                    (0.023 * x).cos() - 7.0e-6 * x,
                )
            })
            .collect::<Vec<_>>();
        let reference = execute_convolution_ir(&ir, &input).unwrap();
        let OneDimFftIr::Recursive(forward) = &ir.forward_fft else {
            unreachable!()
        };
        let OneDimFftIr::Recursive(inverse) = &ir.inverse_fft else {
            unreachable!()
        };
        let forward_uploads = forward
            .four_step_stockham_upload_kernels()
            .unwrap()
            .unwrap();
        let inverse_uploads = inverse
            .four_step_stockham_upload_kernels()
            .unwrap()
            .unwrap();
        let [forward_u1, forward_u0] = forward_uploads.as_slice() else {
            panic!("F64 N4096 forward convolution must have exactly two uploads")
        };
        let [_inverse_u1, inverse_u0] = inverse_uploads.as_slice() else {
            panic!("F64 N4096 inverse convolution must have exactly two uploads")
        };
        let StockhamIoMapping::FourStepLeft(mapping) = forward_u0.io_mapping else {
            panic!("F64 N4096 upload0 must own FourStepLeft mapping")
        };
        assert_eq!((mapping.left_len, mapping.right_len), (64, 64));
        let forward_mid = execute_stockham_ir(forward_u1, &input).unwrap();
        let full_spectrum = execute_stockham_ir(forward_u0, &forward_mid).unwrap();
        let mut special_u0_input = vec![Complex64::default(); n];
        for k2 in 0..mapping.right_len {
            for k1 in 0..mapping.left_len {
                let full_bin = k2 + mapping.right_len * k1;
                special_u0_input[k2 * mapping.left_len + k1] =
                    full_spectrum[full_bin] * kernel_spectrum[full_bin];
            }
        }
        let inverse_u0_scattered = execute_stockham_ir(inverse_u0, &special_u0_input).unwrap();
        let mut inverse_u1_input = vec![Complex64::default(); n];
        for k2 in 0..mapping.right_len {
            for n1 in 0..mapping.left_len {
                inverse_u1_input[k2 * mapping.left_len + n1] = inverse_u0_scattered
                    [k2 + mapping.right_len * n1]
                    .scale(1.0 / mapping.right_len as f64);
            }
        }
        let candidate = execute_stockham_ir(&two_upload.inverse_high, &inverse_u1_input).unwrap();
        let max_error = candidate
            .iter()
            .zip(&reference)
            .map(|(actual, expected)| (*actual - *expected).norm_sqr().sqrt())
            .fold(0.0, f64::max);
        assert!(
            max_error <= 3.0e-11,
            "F64 N4096 three-launch topology error {max_error:e}"
        );
    }

    #[test]
    fn multi_kernel_direct_rader_uses_pinned_batch1_one_pass_graph() {
        let n = 47usize;
        let kernel_count = 3usize;
        let kernel_spectrum = vec![Complex64::new(0.8, 0.2); kernel_count * n];
        let ir = ConvolutionIr::build_from_spectrum(
            FftConfig::new(vec![n])
                .with_precision(Precision::F32)
                .with_convolution(true)
                .with_convolution_kernel_count(kernel_count)
                .with_convolution_conjugation(ConvolutionConjugation::Sequence)
                .with_cross_power_spectrum_normalization(true),
            kernel_spectrum,
            device(),
        )
        .unwrap();
        assert_eq!(ir.forward_fft.batch_count(), 1);
        assert_eq!(ir.inverse_fft.batch_count(), kernel_count);
        assert!(!ir.has_fused_direct_rader_step());
        assert!(ir.has_fused_direct_rader_multi_kernel_step());
        assert!(!ir.has_fused_fft_rader_multi_kernel_step());
        let step = ir.fused_direct_rader_multi_kernel_step.as_ref().unwrap();
        assert_eq!(step.prime, n);
        assert_eq!(step.kernel_count, kernel_count);
        assert_eq!(step.forward.batch_count, 1);
        assert_eq!(step.inverse.batch_count, kernel_count);
        assert_eq!(step.required_shared_memory_bytes, n * 8);
        assert_eq!(step.workgroup_size, WorkgroupSize { x: 24, y: 1, z: 1 });
        assert_eq!(step.dispatch, DispatchGeometry { x: 1, y: 1, z: 1 });

        let program = ProgramIr::convolution(&ir).unwrap();
        assert_eq!(program.passes.len(), 1);
        assert_eq!(program.passes[0].name, step.name);
        assert_eq!(program.resources[0].elements, n);
        assert_eq!(program.resources[1].elements, kernel_count * n);
        assert!(!program.resources.iter().any(|resource| {
            matches!(
                resource.kind,
                crate::program_ir::ProgramResourceKind::Scratch
            )
        }));
        let luts = program
            .resources
            .iter()
            .filter(|resource| {
                matches!(
                    resource.kind,
                    crate::program_ir::ProgramResourceKind::LookupTable
                )
            })
            .collect::<Vec<_>>();
        assert_eq!(luts.len(), 2);
        assert_eq!(luts[0].elements, kernel_count * n);
        assert_eq!(luts[1].elements, n - 1);
        assert!(
            !program
                .passes
                .iter()
                .any(|pass| pass.name == ir.multiply.name)
        );
    }

    #[test]
    fn multi_kernel_fft_rader_uses_pinned_single_container_one_pass_graph() {
        let n = 257usize;
        let kernel_count = 3usize;
        let kernel_spectrum = (0..kernel_count * n)
            .map(|index| {
                let kernel_id = index / n;
                let x = (index % n) as f64;
                Complex64::new(
                    0.82 + 0.07 * kernel_id as f64 + 0.025 * (0.031 * x).cos(),
                    0.11 + 0.018 * kernel_id as f64 + 0.013 * (0.043 * x).sin(),
                )
            })
            .collect::<Vec<_>>();
        let ir = ConvolutionIr::build_from_spectrum(
            FftConfig::new(vec![n])
                .with_precision(Precision::F32)
                .with_convolution(true)
                .with_convolution_kernel_count(kernel_count)
                .with_convolution_conjugation(ConvolutionConjugation::Sequence)
                .with_cross_power_spectrum_normalization(true),
            kernel_spectrum,
            device(),
        )
        .unwrap();
        assert_eq!(ir.forward_fft.batch_count(), 1);
        assert_eq!(ir.inverse_fft.batch_count(), kernel_count);
        assert!(!ir.has_fused_fft_rader_step());
        assert!(ir.has_fused_fft_rader_multi_kernel_step());
        let step = ir.fused_fft_rader_multi_kernel_step.as_ref().unwrap();
        assert_eq!(step.prime, n);
        assert_eq!(step.convolution_len, 256);
        assert_eq!(step.kernel_count, kernel_count);
        assert_eq!(step.required_shared_memory_bytes, 6_152);
        assert_eq!(step.workgroup_size, WorkgroupSize { x: 17, y: 1, z: 1 });
        assert_eq!(step.dispatch, DispatchGeometry { x: 1, y: 1, z: 1 });
        let (inner_forward, inner_inverse) = step.physical_stockham_kernels().unwrap();
        assert_eq!(inner_forward.batch_count, 1);
        assert_eq!(inner_inverse.batch_count, 1);
        assert_eq!(inner_forward.sequence_len, 256);
        assert_eq!(inner_inverse.sequence_len, 256);

        let program = ProgramIr::convolution(&ir).unwrap();
        assert_eq!(program.passes.len(), 1);
        assert_eq!(program.passes[0].name, step.name);
        assert_eq!(program.resources[0].elements, n);
        assert_eq!(program.resources[1].elements, kernel_count * n);
        assert!(
            !program
                .passes
                .iter()
                .any(|pass| pass.name == ir.multiply.name)
        );
    }

    #[test]
    fn plain_fft_plan_rejects_convolution_application_flag() {
        let config = FftConfig::new(vec![16]).with_convolution(true);
        assert!(matches!(
            FftPlan::build(config).unwrap_err(),
            VkFftError::UnsupportedKernelPath(
                "performConvolution is an application pipeline; build ConvolutionIr instead of a plain FftPlan"
            )
        ));
    }
}
