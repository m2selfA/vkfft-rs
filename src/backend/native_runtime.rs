//! Shared correctness-first runtime support for native GPU backends.
//!
//! Source backends remain usable without this module. Runtime features dynamically load
//! vendor/OS APIs so the crate never links VkFFT or requires an SDK at build time.

#![cfg_attr(
    all(
        feature = "metal-runtime",
        not(target_os = "macos"),
        not(any(
            feature = "cuda-runtime",
            feature = "hip-runtime",
            feature = "opencl-runtime",
            feature = "level-zero-runtime"
        ))
    ),
    allow(dead_code, unused_imports)
)]

use libloading::Library;

use crate::application::TransformIr;
use crate::backend::native::{NativeProgramSource, NativeSourceBackend};
use crate::binary16::{decode_complex16_native, encode_complex16_native};
use crate::complex::{Complex32, Complex64};
use crate::config::{Backend, DeviceProfile};
use crate::convolution_ir::{ConvolutionIr, NdConvolutionIr, NdRealConvolutionIr};
use crate::double_double::{ComplexDoubleDouble, DoubleDouble};
use crate::error::{Result, VkFftError};
use crate::kernel_ir::BufferAccess;
use crate::lut::stockham_root_table;
use crate::program_ir::{
    ExternalBufferLayout, ProgramAllocationId, ProgramElementShape, ProgramIr, ProgramMemoryPlan,
    ProgramResourceInitialization,
};
use crate::real_ir::RealFftKind;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeRuntimeAvailability {
    pub backend: Backend,
    pub loader_available: bool,
    pub compiler_available: bool,
    pub device_count: usize,
    pub detail: String,
}

impl NativeRuntimeAvailability {
    pub const fn available(&self) -> bool {
        self.loader_available && self.compiler_available && self.device_count > 0
    }
}

/// Backend-authoritative resource attributes for one compiled native kernel.
///
/// These values come from the runtime/compiler after the generated source has been
/// compiled for a concrete device. They are deliberately kept separate from typed-IR
/// static resource reports and are not an achieved-occupancy estimate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NativeCompiledResourceMetrics {
    Cuda {
        registers_per_thread: usize,
        static_shared_memory_bytes_per_block: usize,
        local_memory_bytes_per_thread: usize,
        max_threads_per_block: usize,
    },
    Hip {
        registers_per_thread: usize,
        static_shared_memory_bytes_per_block: usize,
        local_memory_bytes_per_thread: usize,
        max_threads_per_block: usize,
    },
    OpenCl {
        local_memory_bytes_per_workgroup: usize,
        private_memory_bytes_per_work_item: usize,
        max_workgroup_size: usize,
        preferred_workgroup_size_multiple: usize,
    },
    LevelZero {
        local_memory_bytes_per_workgroup: usize,
        private_memory_bytes_per_thread: usize,
        spill_memory_bytes: usize,
        required_group_size: [usize; 3],
        required_num_subgroups: usize,
        required_subgroup_size: usize,
        max_subgroup_size: usize,
        max_num_subgroups: usize,
    },
    Metal {
        static_threadgroup_memory_bytes: usize,
        max_threads_per_threadgroup: usize,
        thread_execution_width: usize,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeCompiledPassResourceReport {
    pub pass_name: String,
    pub metrics: NativeCompiledResourceMetrics,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NativeOccupancyLimitingResource {
    ThreadSubgroups,
    Registers,
    SharedMemory,
    ArchitecturalBlocks,
}

/// Coarse backend-derived residency/occupancy upper bound for one compiled pass.
///
/// This is intentionally not a profiler measurement. The CUDA implementation uses
/// compiler/driver register and shared-memory allocation plus device resident limits,
/// rounds workgroups to whole warps, and ignores finer allocation granularity,
/// scheduling, memory latency, and concurrent-kernel effects. Unsupported runtimes
/// return no reports rather than synthesizing hardware limits.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeTheoreticalOccupancyReport {
    pub pass_name: String,
    pub backend: Backend,
    pub compute_unit_count: usize,
    pub subgroup_size: usize,
    pub workgroup_threads: usize,
    pub subgroups_per_workgroup: usize,
    pub max_subgroups_per_compute_unit: usize,
    pub blocks_limited_by_thread_subgroups: usize,
    pub blocks_limited_by_registers: Option<usize>,
    pub blocks_limited_by_shared_memory: Option<usize>,
    pub architectural_blocks_per_compute_unit: Option<usize>,
    pub resident_blocks_per_compute_unit_upper_bound: usize,
    pub resident_subgroups_per_compute_unit_upper_bound: usize,
    pub occupancy_basis_points_upper_bound: usize,
    pub limiting_resources: Vec<NativeOccupancyLimitingResource>,
}

#[derive(Debug, Clone, Copy)]
pub enum NativeTransformInput32<'a> {
    Complex(&'a [Complex32]),
    Real(&'a [f32]),
}

#[derive(Debug, Clone, PartialEq)]
pub enum NativeTransformOutput32 {
    Complex(Vec<Complex32>),
    Real(Vec<f32>),
}

#[derive(Debug, Clone, Copy)]
pub enum NativeTransformInput64<'a> {
    Complex(&'a [Complex64]),
    Real(&'a [f64]),
}

#[derive(Debug, Clone, PartialEq)]
pub enum NativeTransformOutput64 {
    Complex(Vec<Complex64>),
    Real(Vec<f64>),
}

/// Unified execution contract for native (non-Vulkan) GPU runtimes. Backend-specific
/// contexts own their driver/compiler handles, while ProgramIr/resource semantics and
/// high-level TransformIr input/output conversion remain shared.
pub trait NativeRuntime {
    fn backend(&self) -> Backend;
    fn device_profile(&self) -> DeviceProfile;
    fn device_name(&self) -> &str;

    /// Query backend/compiler resource attributes for every compiled pass when the
    /// runtime exposes authoritative values. Unsupported backends return an empty list
    /// rather than estimating hardware allocation from typed IR.
    fn compiled_pass_resource_reports(
        &self,
        _source: &NativeProgramSource,
    ) -> Result<Vec<NativeCompiledPassResourceReport>> {
        Ok(Vec::new())
    }

    fn theoretical_occupancy_reports(
        &self,
        _source: &NativeProgramSource,
        _compiled_reports: &[NativeCompiledPassResourceReport],
    ) -> Result<Vec<NativeTheoreticalOccupancyReport>> {
        Ok(Vec::new())
    }

    fn execute_program_complex32(
        &self,
        source: &NativeProgramSource,
        input: &[Complex32],
    ) -> Result<Vec<Complex32>>;

    fn execute_program_complex64(
        &self,
        source: &NativeProgramSource,
        input: &[Complex64],
    ) -> Result<Vec<Complex64>>;

    fn execute_convolution_f32(
        &self,
        ir: &ConvolutionIr,
        input: &[Complex32],
    ) -> Result<Vec<Complex32>> {
        if ir.scalar != crate::ScalarType::F32 {
            return Err(VkFftError::UnsupportedPrecision {
                backend: "native performConvolution runtime",
                precision: "F32 entry point requires F32 convolution IR",
            });
        }
        let source = NativeSourceBackend::new(self.backend()).lower_convolution(ir)?;
        self.execute_program_complex32(&source, input)
    }

    fn execute_convolution_f64(
        &self,
        ir: &ConvolutionIr,
        input: &[Complex64],
    ) -> Result<Vec<Complex64>> {
        if ir.scalar != crate::ScalarType::F64 {
            return Err(VkFftError::UnsupportedPrecision {
                backend: "native performConvolution runtime",
                precision: "F64 entry point requires F64 convolution IR",
            });
        }
        let source = NativeSourceBackend::new(self.backend()).lower_convolution(ir)?;
        self.execute_program_complex64(&source, input)
    }

    fn execute_nd_convolution_f32(
        &self,
        ir: &NdConvolutionIr,
        input: &[Complex32],
    ) -> Result<Vec<Complex32>> {
        if ir.scalar != crate::ScalarType::F32 {
            return Err(VkFftError::UnsupportedPrecision {
                backend: "native multidimensional performConvolution runtime",
                precision: "F32 entry point requires F32 ND convolution IR",
            });
        }
        let source = NativeSourceBackend::new(self.backend()).lower_nd_convolution(ir)?;
        self.execute_program_complex32(&source, input)
    }

    fn execute_nd_convolution_f64(
        &self,
        ir: &NdConvolutionIr,
        input: &[Complex64],
    ) -> Result<Vec<Complex64>> {
        if ir.scalar != crate::ScalarType::F64 {
            return Err(VkFftError::UnsupportedPrecision {
                backend: "native multidimensional performConvolution runtime",
                precision: "F64 entry point requires F64 ND convolution IR",
            });
        }
        let source = NativeSourceBackend::new(self.backend()).lower_nd_convolution(ir)?;
        self.execute_program_complex64(&source, input)
    }

    fn execute_nd_real_convolution_f32(
        &self,
        ir: &NdRealConvolutionIr,
        input: &[f32],
    ) -> Result<Vec<f32>> {
        if ir.scalar != crate::ScalarType::F32 {
            return Err(VkFftError::UnsupportedPrecision {
                backend: "native multidimensional real performConvolution runtime",
                precision: "F32 entry point requires F32 ND real convolution IR",
            });
        }
        let expected_input = ir.full_tensor_len.checked_mul(ir.coordinate_count).ok_or(
            VkFftError::ArithmeticOverflow {
                operation: "native multidimensional real convolution input size",
            },
        )?;
        if input.len() != expected_input {
            return Err(VkFftError::InputLengthMismatch {
                expected: expected_input,
                actual: input.len(),
            });
        }
        let source = NativeSourceBackend::new(self.backend()).lower_nd_real_convolution(ir)?;
        let complex_input = input
            .iter()
            .map(|value| Complex32::new(*value, 0.0))
            .collect::<Vec<_>>();
        let physical_input = if ir.forward_r2c.input_formatted_copy.is_some() {
            ir.forward_r2c.pack_formatted_input(&complex_input)?
        } else {
            complex_input
        };
        let physical_output = self.execute_program_complex32(&source, &physical_input)?;
        let output = if ir.inverse_c2r.output_formatted_copy.is_some() {
            ir.inverse_c2r.unpack_formatted_output(&physical_output)?
        } else {
            physical_output
        };
        Ok(output.into_iter().map(|value| value.re).collect())
    }

    fn execute_nd_real_convolution_f64(
        &self,
        ir: &NdRealConvolutionIr,
        input: &[f64],
    ) -> Result<Vec<f64>> {
        if ir.scalar != crate::ScalarType::F64 {
            return Err(VkFftError::UnsupportedPrecision {
                backend: "native multidimensional real performConvolution runtime",
                precision: "F64 entry point requires F64 ND real convolution IR",
            });
        }
        let expected_input = ir.full_tensor_len.checked_mul(ir.coordinate_count).ok_or(
            VkFftError::ArithmeticOverflow {
                operation: "native multidimensional real convolution input size",
            },
        )?;
        if input.len() != expected_input {
            return Err(VkFftError::InputLengthMismatch {
                expected: expected_input,
                actual: input.len(),
            });
        }
        let source = NativeSourceBackend::new(self.backend()).lower_nd_real_convolution(ir)?;
        let complex_input = input
            .iter()
            .map(|value| Complex64::new(*value, 0.0))
            .collect::<Vec<_>>();
        let physical_input = if ir.forward_r2c.input_formatted_copy.is_some() {
            ir.forward_r2c.pack_formatted_input(&complex_input)?
        } else {
            complex_input
        };
        let physical_output = self.execute_program_complex64(&source, &physical_input)?;
        let output = if ir.inverse_c2r.output_formatted_copy.is_some() {
            ir.inverse_c2r.unpack_formatted_output(&physical_output)?
        } else {
            physical_output
        };
        Ok(output.into_iter().map(|value| value.re).collect())
    }

    fn execute_transform_f32(
        &self,
        ir: &TransformIr,
        input: NativeTransformInput32<'_>,
    ) -> Result<NativeTransformOutput32> {
        let source = NativeSourceBackend::new(self.backend()).lower_transform(ir)?;
        let complex_input = prepare_transform_input32(ir, input)?;
        let output = self.execute_program_complex32(&source, &complex_input)?;
        finish_transform_output32(ir, output)
    }

    fn execute_transform_f64(
        &self,
        ir: &TransformIr,
        input: NativeTransformInput64<'_>,
    ) -> Result<NativeTransformOutput64> {
        let source = NativeSourceBackend::new(self.backend()).lower_transform(ir)?;
        let complex_input = prepare_transform_input64(ir, input)?;
        let output = self.execute_program_complex64(&source, &complex_input)?;
        finish_transform_output64(ir, output)
    }
}

/// Result contract for an asynchronously submitted native F32 complex program.
pub trait NativeProgramTicket32 {
    fn wait(self) -> Result<Vec<Complex32>>;
}

/// Result contract for an asynchronously submitted native F64 complex program.
pub trait NativeProgramTicket64 {
    fn wait(self) -> Result<Vec<Complex64>>;
}

/// Backend-neutral high-level F32 transform ticket built on a native program ticket.
pub struct NativeAsyncTransformTicket32<T> {
    ir: TransformIr,
    ticket: T,
}

impl<T: NativeProgramTicket32> NativeAsyncTransformTicket32<T> {
    pub(crate) fn new(ir: TransformIr, ticket: T) -> Self {
        Self { ir, ticket }
    }

    pub fn wait(self) -> Result<NativeTransformOutput32> {
        finish_transform_output32(&self.ir, self.ticket.wait()?)
    }
}

/// Backend-neutral high-level F64 transform ticket built on a native program ticket.
pub struct NativeAsyncTransformTicket64<T> {
    ir: TransformIr,
    ticket: T,
}

impl<T: NativeProgramTicket64> NativeAsyncTransformTicket64<T> {
    pub(crate) fn new(ir: TransformIr, ticket: T) -> Self {
        Self { ir, ticket }
    }

    pub fn wait(self) -> Result<NativeTransformOutput64> {
        finish_transform_output64(&self.ir, self.ticket.wait()?)
    }
}

/// Unified multi-in-flight submission contract for native runtimes. Implementations
/// return backend-owned tickets whose resources remain alive until `wait` or drop.
pub trait NativeAsyncRuntime: NativeRuntime {
    type Ticket32<'a>: NativeProgramTicket32
    where
        Self: 'a;
    type Ticket64<'a>: NativeProgramTicket64
    where
        Self: 'a;

    fn submit_program_complex32_async<'a>(
        &'a self,
        source: &NativeProgramSource,
        input: &[Complex32],
    ) -> Result<Self::Ticket32<'a>>;

    fn submit_program_complex64_async<'a>(
        &'a self,
        source: &NativeProgramSource,
        input: &[Complex64],
    ) -> Result<Self::Ticket64<'a>>;

    fn submit_transform_f32_async<'a>(
        &'a self,
        ir: &TransformIr,
        input: NativeTransformInput32<'_>,
    ) -> Result<NativeAsyncTransformTicket32<Self::Ticket32<'a>>> {
        let source = NativeSourceBackend::new(self.backend()).lower_transform(ir)?;
        let complex_input = prepare_transform_input32(ir, input)?;
        Ok(NativeAsyncTransformTicket32::new(
            ir.clone(),
            self.submit_program_complex32_async(&source, &complex_input)?,
        ))
    }

    fn submit_transform_f64_async<'a>(
        &'a self,
        ir: &TransformIr,
        input: NativeTransformInput64<'_>,
    ) -> Result<NativeAsyncTransformTicket64<Self::Ticket64<'a>>> {
        let source = NativeSourceBackend::new(self.backend()).lower_transform(ir)?;
        let complex_input = prepare_transform_input64(ir, input)?;
        Ok(NativeAsyncTransformTicket64::new(
            ir.clone(),
            self.submit_program_complex64_async(&source, &complex_input)?,
        ))
    }
}

#[cfg(any(
    feature = "hip-runtime",
    feature = "level-zero-runtime",
    feature = "metal-runtime"
))]
macro_rules! impl_native_transform_convenience_facade {
    ($backend:path, $ticket32:ident, $ticket64:ident) => {
        pub fn submit_transform_complex32<'a>(
            &'a self,
            ir: &$crate::TransformIr,
            input: &[$crate::Complex32],
        ) -> $crate::Result<$ticket32<'a>> {
            let source = $crate::backend::native::NativeSourceBackend::new($backend)
                .lower_transform(ir)?;
            self.submit_program_complex32(&source, input)
        }

        pub fn execute_transform_complex32(
            &self,
            ir: &$crate::TransformIr,
            input: &[$crate::Complex32],
        ) -> $crate::Result<Vec<$crate::Complex32>> {
            $crate::backend::native_runtime::NativeProgramTicket32::wait(
                self.submit_transform_complex32(ir, input)?,
            )
        }

        pub fn submit_transform_complex64<'a>(
            &'a self,
            ir: &$crate::TransformIr,
            input: &[$crate::Complex64],
        ) -> $crate::Result<$ticket64<'a>> {
            let source = $crate::backend::native::NativeSourceBackend::new($backend)
                .lower_transform(ir)?;
            self.submit_program_complex64(&source, input)
        }

        pub fn execute_transform_complex64(
            &self,
            ir: &$crate::TransformIr,
            input: &[$crate::Complex64],
        ) -> $crate::Result<Vec<$crate::Complex64>> {
            $crate::backend::native_runtime::NativeProgramTicket64::wait(
                self.submit_transform_complex64(ir, input)?,
            )
        }

        pub fn submit_transform_f32<'a>(
            &'a self,
            ir: &$crate::TransformIr,
            input: $crate::backend::native_runtime::NativeTransformInput32<'_>,
        ) -> $crate::Result<
            $crate::backend::native_runtime::NativeAsyncTransformTicket32<$ticket32<'a>>,
        > {
            <Self as $crate::backend::native_runtime::NativeAsyncRuntime>::submit_transform_f32_async(
                self, ir, input,
            )
        }

        pub fn submit_transform_f64<'a>(
            &'a self,
            ir: &$crate::TransformIr,
            input: $crate::backend::native_runtime::NativeTransformInput64<'_>,
        ) -> $crate::Result<
            $crate::backend::native_runtime::NativeAsyncTransformTicket64<$ticket64<'a>>,
        > {
            <Self as $crate::backend::native_runtime::NativeAsyncRuntime>::submit_transform_f64_async(
                self, ir, input,
            )
        }
    };
}

#[cfg(any(
    feature = "hip-runtime",
    feature = "level-zero-runtime",
    feature = "metal-runtime"
))]
pub(crate) use impl_native_transform_convenience_facade;

#[cfg(any(feature = "hip-runtime", feature = "level-zero-runtime"))]
macro_rules! impl_native_double_double_runtime_facade {
    ($backend:path, $label:literal) => {
        fn ensure_double_double_support_inner(&self) -> $crate::Result<()> {
            if !self.profile.supports_f64 {
                return Err($crate::VkFftError::UnsupportedPrecision {
                    backend: concat!($label, " runtime"),
                    precision: "double-double",
                });
            }
            Ok(())
        }

        fn execute_double_double_prepared_inner<T>(
            &self,
            source: &$crate::backend::native::NativeProgramSource,
            prepared: $crate::backend::native_runtime::PreparedProgramStorage,
            decode: impl FnOnce(
                &$crate::backend::native_runtime::PreparedProgramStorage,
            ) -> $crate::Result<T>,
        ) -> $crate::Result<T> {
            let mut pending = self.submit_prepared(source, prepared)?;
            pending.finish()?;
            decode(&pending.prepared)
        }

        pub fn execute_program_double_double(
            &self,
            source: &$crate::backend::native::NativeProgramSource,
            input: &[$crate::ComplexDoubleDouble],
        ) -> $crate::Result<Vec<$crate::ComplexDoubleDouble>> {
            source.validate()?;
            if source.backend != $backend
                || source.program.scalar != $crate::ScalarType::DoubleDouble
            {
                return Err($crate::VkFftError::InvalidKernelIr(concat!(
                    $label,
                    " double-double execution requires a matching DD native program"
                )));
            }
            self.ensure_double_double_support_inner()?;
            let prepared = $crate::backend::native_runtime::prepare_program_double_double(
                $backend,
                &source.program,
                input,
            )?;
            self.execute_double_double_prepared_inner(source, prepared, |prepared| {
                prepared.output_double_double()
            })
        }

        pub fn execute_program_double_double_f64_storage(
            &self,
            source: &$crate::backend::native::NativeProgramSource,
            input: &[$crate::Complex64],
        ) -> $crate::Result<Vec<$crate::Complex64>> {
            source.validate()?;
            if source.backend != $backend
                || source.program.scalar != $crate::ScalarType::DoubleDouble
            {
                return Err($crate::VkFftError::InvalidKernelIr(concat!(
                    $label,
                    " DD/F64-storage execution requires a matching DD native program"
                )));
            }
            self.ensure_double_double_support_inner()?;
            let prepared = $crate::backend::native_runtime::prepare_program_complex64(
                $backend,
                &source.program,
                input,
            )?;
            self.execute_double_double_prepared_inner(source, prepared, |prepared| {
                prepared.output_complex64()
            })
        }

        pub fn execute_double_double_r2c(
            &self,
            ir: &$crate::DoubleDoubleRealFftIr,
            input: &[$crate::DoubleDouble],
        ) -> $crate::Result<Vec<$crate::ComplexDoubleDouble>> {
            if ir.kind != $crate::RealFftKind::RealToComplex
                || ir.external_storage != $crate::PrecisionStorage::DoubleDouble
            {
                return Err($crate::VkFftError::InvalidKernelIr(concat!(
                    $label,
                    " DD R2C runtime requires a full-DD R2C IR"
                )));
            }
            self.ensure_double_double_support_inner()?;
            let source = $crate::backend::native::NativeSourceBackend::new($backend)
                .lower_double_double_real(ir)?;
            let prepared = $crate::backend::native_runtime::prepare_program_double_double_scalar(
                $backend,
                &source.program,
                input,
            )?;
            self.execute_double_double_prepared_inner(&source, prepared, |prepared| {
                prepared.output_double_double()
            })
        }

        pub fn execute_double_double_r2c_f64_storage(
            &self,
            ir: &$crate::DoubleDoubleRealFftIr,
            input: &[f64],
        ) -> $crate::Result<Vec<$crate::Complex64>> {
            if ir.kind != $crate::RealFftKind::RealToComplex
                || ir.external_storage != $crate::PrecisionStorage::F64
            {
                return Err($crate::VkFftError::InvalidKernelIr(concat!(
                    $label,
                    " DD/F64 R2C runtime requires an F64-storage R2C IR"
                )));
            }
            self.ensure_double_double_support_inner()?;
            let source = $crate::backend::native::NativeSourceBackend::new($backend)
                .lower_double_double_real(ir)?;
            let prepared = $crate::backend::native_runtime::prepare_program_f64_scalar(
                $backend,
                &source.program,
                input,
            )?;
            self.execute_double_double_prepared_inner(&source, prepared, |prepared| {
                prepared.output_complex64()
            })
        }

        pub fn execute_double_double_c2r(
            &self,
            ir: &$crate::DoubleDoubleRealFftIr,
            input: &[$crate::ComplexDoubleDouble],
        ) -> $crate::Result<Vec<$crate::DoubleDouble>> {
            if ir.kind != $crate::RealFftKind::ComplexToReal
                || ir.external_storage != $crate::PrecisionStorage::DoubleDouble
            {
                return Err($crate::VkFftError::InvalidKernelIr(concat!(
                    $label,
                    " DD C2R runtime requires a full-DD C2R IR"
                )));
            }
            self.ensure_double_double_support_inner()?;
            let source = $crate::backend::native::NativeSourceBackend::new($backend)
                .lower_double_double_real(ir)?;
            let prepared = $crate::backend::native_runtime::prepare_program_double_double(
                $backend,
                &source.program,
                input,
            )?;
            self.execute_double_double_prepared_inner(&source, prepared, |prepared| {
                prepared.output_double_double_scalar()
            })
        }

        pub fn execute_double_double_c2r_f64_storage(
            &self,
            ir: &$crate::DoubleDoubleRealFftIr,
            input: &[$crate::Complex64],
        ) -> $crate::Result<Vec<f64>> {
            if ir.kind != $crate::RealFftKind::ComplexToReal
                || ir.external_storage != $crate::PrecisionStorage::F64
            {
                return Err($crate::VkFftError::InvalidKernelIr(concat!(
                    $label,
                    " DD/F64 C2R runtime requires an F64-storage C2R IR"
                )));
            }
            self.ensure_double_double_support_inner()?;
            let source = $crate::backend::native::NativeSourceBackend::new($backend)
                .lower_double_double_real(ir)?;
            let prepared = $crate::backend::native_runtime::prepare_program_complex64(
                $backend,
                &source.program,
                input,
            )?;
            self.execute_double_double_prepared_inner(&source, prepared, |prepared| {
                prepared.output_f64_scalar()
            })
        }

        pub fn execute_double_double_nd_r2c(
            &self,
            ir: &$crate::DoubleDoubleNdRealFftIr,
            input: &[$crate::DoubleDouble],
        ) -> $crate::Result<Vec<$crate::ComplexDoubleDouble>> {
            if ir.kind != $crate::RealFftKind::RealToComplex
                || ir.external_storage != $crate::PrecisionStorage::DoubleDouble
            {
                return Err($crate::VkFftError::InvalidKernelIr(concat!(
                    $label,
                    " DD ND R2C runtime requires a full-DD R2C IR"
                )));
            }
            self.ensure_double_double_support_inner()?;
            let source = $crate::backend::native::NativeSourceBackend::new($backend)
                .lower_double_double_nd_real(ir)?;
            let physical_input = ir.pack_formatted_input(input)?;
            let prepared = $crate::backend::native_runtime::prepare_program_double_double_scalar(
                $backend,
                &source.program,
                &physical_input,
            )?;
            let physical_output =
                self.execute_double_double_prepared_inner(&source, prepared, |prepared| {
                    prepared.output_double_double()
                })?;
            ir.unpack_formatted_output(&physical_output)
        }

        pub fn execute_double_double_nd_r2c_f64_storage(
            &self,
            ir: &$crate::DoubleDoubleNdRealFftIr,
            input: &[f64],
        ) -> $crate::Result<Vec<$crate::Complex64>> {
            if ir.kind != $crate::RealFftKind::RealToComplex
                || ir.external_storage != $crate::PrecisionStorage::F64
            {
                return Err($crate::VkFftError::InvalidKernelIr(concat!(
                    $label,
                    " DD/F64 ND R2C runtime requires an F64-storage R2C IR"
                )));
            }
            self.ensure_double_double_support_inner()?;
            let source = $crate::backend::native::NativeSourceBackend::new($backend)
                .lower_double_double_nd_real(ir)?;
            let physical_input = ir.pack_formatted_input(input)?;
            let prepared = $crate::backend::native_runtime::prepare_program_f64_scalar(
                $backend,
                &source.program,
                &physical_input,
            )?;
            let physical_output =
                self.execute_double_double_prepared_inner(&source, prepared, |prepared| {
                    prepared.output_complex64()
                })?;
            ir.unpack_formatted_output(&physical_output)
        }

        pub fn execute_double_double_nd_c2r(
            &self,
            ir: &$crate::DoubleDoubleNdRealFftIr,
            input: &[$crate::ComplexDoubleDouble],
        ) -> $crate::Result<Vec<$crate::DoubleDouble>> {
            if ir.kind != $crate::RealFftKind::ComplexToReal
                || ir.external_storage != $crate::PrecisionStorage::DoubleDouble
            {
                return Err($crate::VkFftError::InvalidKernelIr(concat!(
                    $label,
                    " DD ND C2R runtime requires a full-DD C2R IR"
                )));
            }
            self.ensure_double_double_support_inner()?;
            let source = $crate::backend::native::NativeSourceBackend::new($backend)
                .lower_double_double_nd_real(ir)?;
            let physical_input = ir.pack_formatted_input(input)?;
            let prepared = $crate::backend::native_runtime::prepare_program_double_double(
                $backend,
                &source.program,
                &physical_input,
            )?;
            let physical_output =
                self.execute_double_double_prepared_inner(&source, prepared, |prepared| {
                    prepared.output_double_double_scalar()
                })?;
            ir.unpack_formatted_output(&physical_output)
        }

        pub fn execute_double_double_nd_c2r_f64_storage(
            &self,
            ir: &$crate::DoubleDoubleNdRealFftIr,
            input: &[$crate::Complex64],
        ) -> $crate::Result<Vec<f64>> {
            if ir.kind != $crate::RealFftKind::ComplexToReal
                || ir.external_storage != $crate::PrecisionStorage::F64
            {
                return Err($crate::VkFftError::InvalidKernelIr(concat!(
                    $label,
                    " DD/F64 ND C2R runtime requires an F64-storage C2R IR"
                )));
            }
            self.ensure_double_double_support_inner()?;
            let source = $crate::backend::native::NativeSourceBackend::new($backend)
                .lower_double_double_nd_real(ir)?;
            let physical_input = ir.pack_formatted_input(input)?;
            let prepared = $crate::backend::native_runtime::prepare_program_complex64(
                $backend,
                &source.program,
                &physical_input,
            )?;
            let physical_output =
                self.execute_double_double_prepared_inner(&source, prepared, |prepared| {
                    prepared.output_f64_scalar()
                })?;
            ir.unpack_formatted_output(&physical_output)
        }

        pub fn execute_double_double_r2r(
            &self,
            ir: &$crate::DoubleDoubleR2rIr,
            input: &[$crate::DoubleDouble],
        ) -> $crate::Result<Vec<$crate::DoubleDouble>> {
            if ir.external_storage != $crate::PrecisionStorage::DoubleDouble {
                return Err($crate::VkFftError::UnsupportedPrecision {
                    backend: concat!($label, " double-double R2R runtime"),
                    precision: "IR uses F64 external storage",
                });
            }
            self.ensure_double_double_support_inner()?;
            let source = $crate::backend::native::NativeSourceBackend::new($backend)
                .lower_double_double_r2r(ir)?;
            let prepared = $crate::backend::native_runtime::prepare_program_double_double_scalar(
                $backend,
                &source.program,
                input,
            )?;
            self.execute_double_double_prepared_inner(&source, prepared, |prepared| {
                prepared.output_double_double_scalar()
            })
        }

        pub fn execute_double_double_r2r_f64_storage(
            &self,
            ir: &$crate::DoubleDoubleR2rIr,
            input: &[f64],
        ) -> $crate::Result<Vec<f64>> {
            if ir.external_storage != $crate::PrecisionStorage::F64 {
                return Err($crate::VkFftError::UnsupportedPrecision {
                    backend: concat!($label, " double-double R2R runtime"),
                    precision: "IR uses double-double external storage",
                });
            }
            self.ensure_double_double_support_inner()?;
            let source = $crate::backend::native::NativeSourceBackend::new($backend)
                .lower_double_double_r2r(ir)?;
            let prepared = $crate::backend::native_runtime::prepare_program_f64_scalar(
                $backend,
                &source.program,
                input,
            )?;
            self.execute_double_double_prepared_inner(&source, prepared, |prepared| {
                prepared.output_f64_scalar()
            })
        }

        pub fn execute_double_double_nd_r2r(
            &self,
            ir: &$crate::DoubleDoubleNdR2rIr,
            input: &[$crate::DoubleDouble],
        ) -> $crate::Result<Vec<$crate::DoubleDouble>> {
            if ir.external_storage != $crate::PrecisionStorage::DoubleDouble {
                return Err($crate::VkFftError::UnsupportedPrecision {
                    backend: concat!($label, " double-double ND R2R runtime"),
                    precision: "IR uses F64 external storage",
                });
            }
            self.ensure_double_double_support_inner()?;
            let source = $crate::backend::native::NativeSourceBackend::new($backend)
                .lower_double_double_nd_r2r(ir)?;
            let physical_input = ir.pack_formatted_input(input)?;
            let prepared = $crate::backend::native_runtime::prepare_program_double_double_scalar(
                $backend,
                &source.program,
                &physical_input,
            )?;
            let physical_output =
                self.execute_double_double_prepared_inner(&source, prepared, |prepared| {
                    prepared.output_double_double_scalar()
                })?;
            ir.unpack_formatted_output(&physical_output)
        }

        pub fn execute_double_double_nd_r2r_f64_storage(
            &self,
            ir: &$crate::DoubleDoubleNdR2rIr,
            input: &[f64],
        ) -> $crate::Result<Vec<f64>> {
            if ir.external_storage != $crate::PrecisionStorage::F64 {
                return Err($crate::VkFftError::UnsupportedPrecision {
                    backend: concat!($label, " double-double ND R2R runtime"),
                    precision: "IR uses double-double external storage",
                });
            }
            self.ensure_double_double_support_inner()?;
            let source = $crate::backend::native::NativeSourceBackend::new($backend)
                .lower_double_double_nd_r2r(ir)?;
            let physical_input = ir.pack_formatted_input(input)?;
            let prepared = $crate::backend::native_runtime::prepare_program_f64_scalar(
                $backend,
                &source.program,
                &physical_input,
            )?;
            let physical_output =
                self.execute_double_double_prepared_inner(&source, prepared, |prepared| {
                    prepared.output_f64_scalar()
                })?;
            ir.unpack_formatted_output(&physical_output)
        }

        pub fn execute_transform_double_double_r2r(
            &self,
            ir: &$crate::TransformIr,
            input: &[$crate::DoubleDouble],
        ) -> $crate::Result<Vec<$crate::DoubleDouble>> {
            match ir {
                $crate::TransformIr::RealToRealDoubleDouble(r2r) => {
                    self.execute_double_double_r2r(r2r, input)
                }
                $crate::TransformIr::RealToRealNdDoubleDouble(r2r) => {
                    self.execute_double_double_nd_r2r(r2r, input)
                }
                _ => Err($crate::VkFftError::UnsupportedKernelPath(concat!(
                    "high-level ",
                    $label,
                    " DD R2R execution requires a double-double DCT/DST transform"
                ))),
            }
        }

        pub fn execute_transform_double_double_r2r_f64_storage(
            &self,
            ir: &$crate::TransformIr,
            input: &[f64],
        ) -> $crate::Result<Vec<f64>> {
            match ir {
                $crate::TransformIr::RealToRealDoubleDouble(r2r) => {
                    self.execute_double_double_r2r_f64_storage(r2r, input)
                }
                $crate::TransformIr::RealToRealNdDoubleDouble(r2r) => {
                    self.execute_double_double_nd_r2r_f64_storage(r2r, input)
                }
                _ => Err($crate::VkFftError::UnsupportedKernelPath(concat!(
                    "high-level ",
                    $label,
                    " DD/F64 R2R execution requires a double-double DCT/DST transform"
                ))),
            }
        }

        pub fn execute_transform_double_double(
            &self,
            ir: &$crate::TransformIr,
            input: &[$crate::ComplexDoubleDouble],
        ) -> $crate::Result<Vec<$crate::ComplexDoubleDouble>> {
            let source =
                $crate::backend::native::NativeSourceBackend::new($backend).lower_transform(ir)?;
            if let $crate::TransformIr::ComplexNdDoubleDouble(nd) = ir {
                let physical_input = nd.pack_formatted_input(input)?;
                let physical_output =
                    self.execute_program_double_double(&source, &physical_input)?;
                nd.unpack_formatted_output(&physical_output)
            } else {
                self.execute_program_double_double(&source, input)
            }
        }

        pub fn execute_transform_double_double_f64_storage(
            &self,
            ir: &$crate::TransformIr,
            input: &[$crate::Complex64],
        ) -> $crate::Result<Vec<$crate::Complex64>> {
            let source =
                $crate::backend::native::NativeSourceBackend::new($backend).lower_transform(ir)?;
            if let $crate::TransformIr::ComplexNdDoubleDouble(nd) = ir {
                let physical_input = nd.pack_formatted_input(input)?;
                let physical_output =
                    self.execute_program_double_double_f64_storage(&source, &physical_input)?;
                nd.unpack_formatted_output(&physical_output)
            } else {
                self.execute_program_double_double_f64_storage(&source, input)
            }
        }
    };
}

#[cfg(any(feature = "hip-runtime", feature = "level-zero-runtime"))]
pub(crate) use impl_native_double_double_runtime_facade;

pub(crate) const fn backend_label(backend: Backend) -> &'static str {
    match backend {
        Backend::Vulkan => "Vulkan",
        Backend::Cuda => "CUDA",
        Backend::Hip => "HIP",
        Backend::OpenCl => "OpenCL",
        Backend::LevelZero => "Level Zero",
        Backend::Metal => "Metal",
        Backend::CpuReference => "CPU reference",
    }
}

pub(crate) fn unavailable(backend: Backend, message: impl Into<String>) -> VkFftError {
    VkFftError::NativeUnavailable {
        backend: backend_label(backend),
        message: message.into(),
    }
}

pub(crate) fn runtime_error(backend: Backend, message: impl Into<String>) -> VkFftError {
    VkFftError::NativeRuntime {
        backend: backend_label(backend),
        message: message.into(),
    }
}

pub(crate) fn load_first_library(
    backend: Backend,
    candidates: &[&str],
) -> Result<(Library, String)> {
    let mut errors = Vec::new();
    for candidate in candidates {
        // SAFETY: the returned Library owns the handle and every runtime keeps it alive
        // for at least as long as any symbols copied from it can be called.
        match unsafe { Library::new(candidate) } {
            Ok(library) => return Ok((library, (*candidate).to_owned())),
            Err(error) => errors.push(format!("{candidate}: {error}")),
        }
    }
    Err(unavailable(
        backend,
        format!(
            "none of the runtime libraries could be loaded ({})",
            errors.join("; ")
        ),
    ))
}

#[derive(Debug)]
pub(crate) struct PreparedProgramAllocation {
    pub byte_len: usize,
    pub host_bytes: Option<Vec<u8>>,
}

#[derive(Debug)]
pub(crate) struct PreparedProgramStorage {
    pub backend: Backend,
    pub memory_plan: ProgramMemoryPlan,
    pub allocations: Vec<PreparedProgramAllocation>,
    pub output_allocation: ProgramAllocationId,
    pub output_layout: ExternalBufferLayout,
}

impl PreparedProgramStorage {
    pub(crate) fn output_bytes(&self) -> Result<&[u8]> {
        self.allocations
            .get(self.output_allocation.0)
            .and_then(|allocation| allocation.host_bytes.as_deref())
            .ok_or_else(|| runtime_error(self.backend, "missing native output host bytes"))
    }

    #[cfg(any(
        test,
        feature = "hip-runtime",
        feature = "opencl-runtime",
        feature = "level-zero-runtime",
        all(target_os = "macos", feature = "metal-runtime")
    ))]
    pub(crate) fn output_bytes_mut(&mut self) -> Result<&mut [u8]> {
        let allocation = self
            .allocations
            .get_mut(self.output_allocation.0)
            .ok_or_else(|| runtime_error(self.backend, "missing native output allocation"))?;
        if allocation.host_bytes.is_none() {
            allocation.host_bytes = Some(vec![0u8; allocation.byte_len]);
        }
        Ok(allocation
            .host_bytes
            .as_deref_mut()
            .expect("output host bytes initialized above"))
    }
    pub fn output_complex32(&self) -> Result<Vec<Complex32>> {
        let allocation = self
            .memory_plan
            .allocations
            .get(self.output_allocation.0)
            .ok_or_else(|| {
                runtime_error(self.backend, "missing native output allocation metadata")
            })?;
        if !matches!(
            allocation.scalar,
            crate::ScalarType::F16 | crate::ScalarType::F32
        ) {
            return Err(VkFftError::UnsupportedPrecision {
                backend: "native runtime output",
                precision: "Complex32 readback requires F16 or F32 external storage",
            });
        }
        let bytes = self.output_bytes()?;
        match allocation.scalar {
            crate::ScalarType::F16 => decode_complex16_layout(bytes, self.output_layout),
            crate::ScalarType::F32 => decode_complex32_layout(bytes, self.output_layout),
            crate::ScalarType::F64 | crate::ScalarType::DoubleDouble => {
                unreachable!("validated Complex32 output scalar")
            }
        }
    }

    pub fn output_double_double(&self) -> Result<Vec<ComplexDoubleDouble>> {
        let allocation = self
            .memory_plan
            .allocations
            .get(self.output_allocation.0)
            .ok_or_else(|| {
                runtime_error(self.backend, "missing native DD output allocation metadata")
            })?;
        if allocation.scalar != crate::ScalarType::DoubleDouble {
            return Err(VkFftError::UnsupportedPrecision {
                backend: "native runtime output",
                precision: "double-double readback requested for non-DD storage",
            });
        }
        let bytes = self.output_bytes()?;
        let physical = crate::double_double::decode_complex_double_double(bytes)?;
        let mut output = Vec::with_capacity(self.output_layout.logical_elements()?);
        for batch in 0..self.output_layout.batch_count {
            let start = batch * self.output_layout.physical_stride;
            output.extend_from_slice(&physical[start..start + self.output_layout.logical_len]);
        }
        Ok(output)
    }

    pub fn output_complex64(&self) -> Result<Vec<Complex64>> {
        let allocation = self
            .memory_plan
            .allocations
            .get(self.output_allocation.0)
            .ok_or_else(|| {
                runtime_error(self.backend, "missing native output allocation metadata")
            })?;
        if !matches!(
            allocation.scalar,
            crate::ScalarType::F32 | crate::ScalarType::F64
        ) {
            return Err(VkFftError::UnsupportedPrecision {
                backend: "native runtime output",
                precision: "Complex64 readback requires F32 or F64 external storage",
            });
        }
        let bytes = self.output_bytes()?;
        match allocation.scalar {
            crate::ScalarType::F32 => Ok(decode_complex32_layout(bytes, self.output_layout)?
                .into_iter()
                .map(|value| Complex64::new(f64::from(value.re), f64::from(value.im)))
                .collect()),
            crate::ScalarType::F64 => decode_complex64_layout(bytes, self.output_layout),
            crate::ScalarType::F16 | crate::ScalarType::DoubleDouble => {
                unreachable!("validated Complex64 output scalar")
            }
        }
    }

    pub fn output_double_double_scalar(&self) -> Result<Vec<DoubleDouble>> {
        let allocation = self
            .memory_plan
            .allocations
            .get(self.output_allocation.0)
            .ok_or_else(|| {
                runtime_error(self.backend, "missing native scalar DD output metadata")
            })?;
        if allocation.scalar != crate::ScalarType::DoubleDouble
            || self.output_layout.element_shape != ProgramElementShape::Scalar
        {
            return Err(VkFftError::UnsupportedPrecision {
                backend: "native runtime output",
                precision: "scalar double-double readback requires scalar DD external storage",
            });
        }
        let bytes = self.output_bytes()?;
        decode_double_double_scalar_layout(bytes, self.output_layout)
    }

    pub fn output_f64_scalar(&self) -> Result<Vec<f64>> {
        let allocation = self
            .memory_plan
            .allocations
            .get(self.output_allocation.0)
            .ok_or_else(|| {
                runtime_error(self.backend, "missing native scalar F64 output metadata")
            })?;
        if allocation.scalar != crate::ScalarType::F64
            || self.output_layout.element_shape != ProgramElementShape::Scalar
        {
            return Err(VkFftError::UnsupportedPrecision {
                backend: "native runtime output",
                precision: "scalar F64 readback requires scalar F64 external storage",
            });
        }
        let bytes = self.output_bytes()?;
        decode_f64_scalar_layout(bytes, self.output_layout)
    }
}

#[cfg(any(feature = "cuda-runtime", feature = "opencl-runtime"))]
pub(crate) fn prepare_program_complex32_resident_input(
    backend: Backend,
    program: &ProgramIr,
) -> Result<PreparedProgramStorage> {
    prepare_program_complex32_impl(backend, program, None)
}

pub(crate) fn prepare_program_complex32(
    backend: Backend,
    program: &ProgramIr,
    input: &[Complex32],
) -> Result<PreparedProgramStorage> {
    prepare_program_complex32_impl(backend, program, Some(input))
}

fn prepare_program_complex32_impl(
    backend: Backend,
    program: &ProgramIr,
    input: Option<&[Complex32]>,
) -> Result<PreparedProgramStorage> {
    let input_resource = program.input_resource()?;
    let output_resource = program.output_resource()?;
    if input_resource.scalar != output_resource.scalar
        || !matches!(
            input_resource.scalar,
            crate::ScalarType::F16 | crate::ScalarType::F32
        )
    {
        return Err(VkFftError::UnsupportedPrecision {
            backend: "native runtime",
            precision: "Complex32 host execution requires matching F16/F32 external storage",
        });
    }
    let input_layout = input_resource
        .external_layout
        .ok_or(VkFftError::InvalidKernelIr(
            "native program input is missing its external layout",
        ))?;
    let expected = input_layout.logical_elements()?;
    if let Some(input) = input
        && input.len() != expected
    {
        return Err(VkFftError::InputLengthMismatch {
            expected,
            actual: input.len(),
        });
    }
    prepare_program_storage(backend, program, |resource| {
        match &resource.initialization {
            ProgramResourceInitialization::ExternalInput => {
                let Some(input) = input else {
                    return Ok(None);
                };
                let layout = resource.external_layout.ok_or(VkFftError::InvalidKernelIr(
                    "native program external input is missing its layout",
                ))?;
                let dense_f32 = resource.scalar == crate::ScalarType::F32
                    && layout.physical_stride == layout.logical_len
                    && resource.elements == input.len();
                if dense_f32 {
                    return Ok(Some(encode_complex32(input)));
                }
                let mut physical = vec![Complex32::default(); resource.elements];
                for batch in 0..layout.batch_count {
                    let source = batch * layout.logical_len;
                    let destination = batch * layout.physical_stride;
                    physical[destination..destination + layout.logical_len]
                        .copy_from_slice(&input[source..source + layout.logical_len]);
                }
                let bytes = match resource.scalar {
                    crate::ScalarType::F16 => encode_complex16_native(&physical),
                    crate::ScalarType::F32 => encode_complex32(&physical),
                    crate::ScalarType::F64 | crate::ScalarType::DoubleDouble => {
                        return Err(VkFftError::UnsupportedPrecision {
                            backend: "native runtime",
                            precision: "Complex32 input cannot initialize F64/double-double external storage",
                        });
                    }
                };
                Ok(Some(bytes))
            }
            ProgramResourceInitialization::Zeroed => Ok(None),
            ProgramResourceInitialization::Complex64(values) => {
                encode_lut_for_scalar(backend, &resource.name, resource.scalar, values).map(Some)
            }
            ProgramResourceInitialization::ComplexDoubleDouble(values) => Ok(Some(
                crate::double_double::encode_complex_double_double(values),
            )),
            ProgramResourceInitialization::StockhamUnitRoots { len } => {
                let values = stockham_root_table(*len)?;
                encode_lut_for_scalar(backend, &resource.name, resource.scalar, &values).map(Some)
            }
        }
    })
}

pub(crate) fn prepare_program_double_double_scalar(
    backend: Backend,
    program: &ProgramIr,
    input: &[DoubleDouble],
) -> Result<PreparedProgramStorage> {
    let input_resource = program.input_resource()?;
    if program.scalar != crate::ScalarType::DoubleDouble
        || input_resource.scalar != crate::ScalarType::DoubleDouble
        || input_resource.element_shape() != ProgramElementShape::Scalar
    {
        return Err(VkFftError::UnsupportedPrecision {
            backend: "native double-double runtime",
            precision: "scalar DD host input requires DD compute and scalar DD external input",
        });
    }
    let input_layout = input_resource
        .external_layout
        .ok_or(VkFftError::InvalidKernelIr(
            "native scalar DD program input is missing its external layout",
        ))?;
    let expected = input_layout.logical_elements()?;
    if input.len() != expected {
        return Err(VkFftError::InputLengthMismatch {
            expected,
            actual: input.len(),
        });
    }
    prepare_program_storage(backend, program, |resource| {
        match &resource.initialization {
            ProgramResourceInitialization::ExternalInput => {
                let layout = resource.external_layout.ok_or(VkFftError::InvalidKernelIr(
                    "native scalar DD program external input is missing its layout",
                ))?;
                if layout.element_shape != ProgramElementShape::Scalar {
                    return Err(VkFftError::InvalidKernelIr(
                        "native scalar DD input resource is not scalar-shaped",
                    ));
                }
                let mut physical = vec![DoubleDouble::ZERO; resource.elements];
                for batch in 0..layout.batch_count {
                    let source = batch * layout.logical_len;
                    let destination = batch * layout.physical_stride;
                    physical[destination..destination + layout.logical_len]
                        .copy_from_slice(&input[source..source + layout.logical_len]);
                }
                Ok(Some(encode_double_double_scalars(&physical)))
            }
            ProgramResourceInitialization::Zeroed => Ok(None),
            ProgramResourceInitialization::Complex64(values) => {
                encode_lut_for_scalar(backend, &resource.name, resource.scalar, values).map(Some)
            }
            ProgramResourceInitialization::ComplexDoubleDouble(values) => Ok(Some(
                crate::double_double::encode_complex_double_double(values),
            )),
            ProgramResourceInitialization::StockhamUnitRoots { len } => {
                let values = stockham_root_table(*len)?;
                encode_lut_for_scalar(backend, &resource.name, resource.scalar, &values).map(Some)
            }
        }
    })
}

pub(crate) fn prepare_program_f64_scalar(
    backend: Backend,
    program: &ProgramIr,
    input: &[f64],
) -> Result<PreparedProgramStorage> {
    let input_resource = program.input_resource()?;
    if program.scalar != crate::ScalarType::DoubleDouble
        || input_resource.scalar != crate::ScalarType::F64
        || input_resource.element_shape() != ProgramElementShape::Scalar
    {
        return Err(VkFftError::UnsupportedPrecision {
            backend: "native double-double runtime",
            precision: "scalar F64 host input requires DD compute and scalar F64 external input",
        });
    }
    let input_layout = input_resource
        .external_layout
        .ok_or(VkFftError::InvalidKernelIr(
            "native scalar F64 program input is missing its external layout",
        ))?;
    let expected = input_layout.logical_elements()?;
    if input.len() != expected {
        return Err(VkFftError::InputLengthMismatch {
            expected,
            actual: input.len(),
        });
    }
    prepare_program_storage(backend, program, |resource| {
        match &resource.initialization {
            ProgramResourceInitialization::ExternalInput => {
                let layout = resource.external_layout.ok_or(VkFftError::InvalidKernelIr(
                    "native scalar F64 program external input is missing its layout",
                ))?;
                if layout.element_shape != ProgramElementShape::Scalar {
                    return Err(VkFftError::InvalidKernelIr(
                        "native scalar F64 input resource is not scalar-shaped",
                    ));
                }
                let mut physical = vec![0.0f64; resource.elements];
                for batch in 0..layout.batch_count {
                    let source = batch * layout.logical_len;
                    let destination = batch * layout.physical_stride;
                    physical[destination..destination + layout.logical_len]
                        .copy_from_slice(&input[source..source + layout.logical_len]);
                }
                Ok(Some(encode_f64_scalars(&physical)))
            }
            ProgramResourceInitialization::Zeroed => Ok(None),
            ProgramResourceInitialization::Complex64(values) => {
                encode_lut_for_scalar(backend, &resource.name, resource.scalar, values).map(Some)
            }
            ProgramResourceInitialization::ComplexDoubleDouble(values) => Ok(Some(
                crate::double_double::encode_complex_double_double(values),
            )),
            ProgramResourceInitialization::StockhamUnitRoots { len } => {
                let values = stockham_root_table(*len)?;
                encode_lut_for_scalar(backend, &resource.name, resource.scalar, &values).map(Some)
            }
        }
    })
}

pub(crate) fn prepare_program_double_double(
    backend: Backend,
    program: &ProgramIr,
    input: &[ComplexDoubleDouble],
) -> Result<PreparedProgramStorage> {
    let input_resource = program.input_resource()?;
    let output_resource = program.output_resource()?;
    if program.scalar != crate::ScalarType::DoubleDouble
        || input_resource.scalar != crate::ScalarType::DoubleDouble
        || output_resource.scalar != crate::ScalarType::DoubleDouble
    {
        return Err(VkFftError::UnsupportedPrecision {
            backend: "native double-double runtime",
            precision: "full DD host execution requires DD compute and DD external storage",
        });
    }
    let input_layout = input_resource
        .external_layout
        .ok_or(VkFftError::InvalidKernelIr(
            "native DD program input is missing its external layout",
        ))?;
    let expected = input_layout.logical_elements()?;
    if input.len() != expected {
        return Err(VkFftError::InputLengthMismatch {
            expected,
            actual: input.len(),
        });
    }
    prepare_program_storage(backend, program, |resource| {
        match &resource.initialization {
            ProgramResourceInitialization::ExternalInput => {
                let layout = resource.external_layout.ok_or(VkFftError::InvalidKernelIr(
                    "native DD program external input is missing its layout",
                ))?;
                let mut physical = vec![ComplexDoubleDouble::default(); resource.elements];
                for batch in 0..layout.batch_count {
                    let source = batch * layout.logical_len;
                    let destination = batch * layout.physical_stride;
                    physical[destination..destination + layout.logical_len]
                        .copy_from_slice(&input[source..source + layout.logical_len]);
                }
                Ok(Some(crate::double_double::encode_complex_double_double(
                    &physical,
                )))
            }
            ProgramResourceInitialization::Zeroed => Ok(None),
            ProgramResourceInitialization::Complex64(values) => {
                encode_lut_for_scalar(backend, &resource.name, resource.scalar, values).map(Some)
            }
            ProgramResourceInitialization::ComplexDoubleDouble(values) => Ok(Some(
                crate::double_double::encode_complex_double_double(values),
            )),
            ProgramResourceInitialization::StockhamUnitRoots { len } => {
                let values = stockham_root_table(*len)?;
                encode_lut_for_scalar(backend, &resource.name, resource.scalar, &values).map(Some)
            }
        }
    })
}

pub(crate) fn prepare_program_complex64(
    backend: Backend,
    program: &ProgramIr,
    input: &[Complex64],
) -> Result<PreparedProgramStorage> {
    let input_resource = program.input_resource()?;
    let output_resource = program.output_resource()?;
    if input_resource.scalar != output_resource.scalar
        || !matches!(
            input_resource.scalar,
            crate::ScalarType::F32 | crate::ScalarType::F64
        )
    {
        return Err(VkFftError::UnsupportedPrecision {
            backend: "native runtime",
            precision: "Complex64 host execution requires matching F32/F64 external storage",
        });
    }
    let input_layout = input_resource
        .external_layout
        .ok_or(VkFftError::InvalidKernelIr(
            "native program input is missing its external layout",
        ))?;
    let expected = input_layout.logical_elements()?;
    if input.len() != expected {
        return Err(VkFftError::InputLengthMismatch {
            expected,
            actual: input.len(),
        });
    }
    prepare_program_storage(backend, program, |resource| {
        match &resource.initialization {
            ProgramResourceInitialization::ExternalInput => {
                let layout = resource.external_layout.ok_or(VkFftError::InvalidKernelIr(
                    "native program external input is missing its layout",
                ))?;
                let dense_f64 = resource.scalar == crate::ScalarType::F64
                    && layout.physical_stride == layout.logical_len
                    && resource.elements == input.len();
                if dense_f64 {
                    return Ok(Some(encode_complex64(input)));
                }
                let mut physical = vec![Complex64::default(); resource.elements];
                for batch in 0..layout.batch_count {
                    let source = batch * layout.logical_len;
                    let destination = batch * layout.physical_stride;
                    physical[destination..destination + layout.logical_len]
                        .copy_from_slice(&input[source..source + layout.logical_len]);
                }
                match resource.scalar {
                    crate::ScalarType::F32 => {
                        let converted = physical
                            .iter()
                            .map(|value| Complex32::new(value.re as f32, value.im as f32))
                            .collect::<Vec<_>>();
                        Ok(Some(encode_complex32(&converted)))
                    }
                    crate::ScalarType::F64 => Ok(Some(encode_complex64(&physical))),
                    crate::ScalarType::F16 | crate::ScalarType::DoubleDouble => {
                        Err(VkFftError::UnsupportedPrecision {
                            backend: "native runtime",
                            precision: "Complex64 input cannot initialize F16/double-double external storage",
                        })
                    }
                }
            }
            ProgramResourceInitialization::Zeroed => Ok(None),
            ProgramResourceInitialization::Complex64(values) => {
                encode_lut_for_scalar(backend, &resource.name, resource.scalar, values).map(Some)
            }
            ProgramResourceInitialization::ComplexDoubleDouble(values) => Ok(Some(
                crate::double_double::encode_complex_double_double(values),
            )),
            ProgramResourceInitialization::StockhamUnitRoots { len } => {
                let values = stockham_root_table(*len)?;
                encode_lut_for_scalar(backend, &resource.name, resource.scalar, &values).map(Some)
            }
        }
    })
}

fn encode_lut_for_scalar(
    backend: Backend,
    name: &str,
    scalar: crate::ScalarType,
    values: &[Complex64],
) -> Result<Vec<u8>> {
    match scalar {
        crate::ScalarType::F16 => Err(VkFftError::UnsupportedPrecision {
            backend: "native runtime LUT",
            precision: "binary16 LUT storage is not part of F16-storage/F32-compute",
        }),
        crate::ScalarType::DoubleDouble => Err(VkFftError::UnsupportedPrecision {
            backend: "native runtime LUT",
            precision: "Complex64 LUT initialization cannot populate double-double storage",
        }),
        crate::ScalarType::F64 => {
            if values
                .iter()
                .any(|value| !value.re.is_finite() || !value.im.is_finite())
            {
                return Err(runtime_error(
                    backend,
                    format!("program LUT `{name}` contains a non-finite F64 value"),
                ));
            }
            Ok(encode_complex64(values))
        }
        crate::ScalarType::F32 => {
            let mut converted = Vec::with_capacity(values.len());
            for value in values {
                let value = Complex32::new(value.re as f32, value.im as f32);
                if !value.re.is_finite() || !value.im.is_finite() {
                    return Err(runtime_error(
                        backend,
                        format!("program LUT `{name}` cannot be represented as F32"),
                    ));
                }
                converted.push(value);
            }
            Ok(encode_complex32(&converted))
        }
    }
}

fn prepare_program_storage<F>(
    backend: Backend,
    program: &ProgramIr,
    mut initialization: F,
) -> Result<PreparedProgramStorage>
where
    F: FnMut(&crate::program_ir::ProgramResource) -> Result<Option<Vec<u8>>>,
{
    program.validate()?;
    let memory_plan = program.memory_plan()?;
    let mut allocations = memory_plan
        .allocations
        .iter()
        .map(|allocation| {
            let element_bytes = allocation
                .resources
                .first()
                .map(|resource| program.resource(*resource))
                .transpose()?
                .map(|resource| resource.element_bytes())
                .unwrap_or_else(|| allocation.scalar.complex_bytes());
            let byte_len = allocation.elements.checked_mul(element_bytes).ok_or(
                VkFftError::ArithmeticOverflow {
                    operation: "native program allocation byte size",
                },
            )?;
            let requires_zero_initialization = allocation.resources.iter().try_fold(
                false,
                |requires_zero, resource_id| -> Result<bool> {
                    let resource = program.resource(*resource_id)?;
                    if !matches!(
                        resource.initialization,
                        ProgramResourceInitialization::Zeroed
                    ) {
                        return Ok(requires_zero);
                    }
                    let first_access = program.passes.iter().find_map(|pass| {
                        pass.bindings
                            .iter()
                            .find(|binding| binding.resource == *resource_id)
                            .map(|binding| binding.access)
                    });
                    Ok(requires_zero
                        || matches!(
                            first_access,
                            Some(BufferAccess::ReadOnly | BufferAccess::ReadWrite)
                        ))
                },
            )?;
            Ok(PreparedProgramAllocation {
                byte_len,
                host_bytes: requires_zero_initialization.then(|| vec![0u8; byte_len]),
            })
        })
        .collect::<Result<Vec<_>>>()?;

    for resource in &program.resources {
        if let Some(bytes) = initialization(resource)? {
            let allocation = memory_plan.allocation_for(resource.id)?;
            let destination =
                allocations
                    .get_mut(allocation.0)
                    .ok_or(VkFftError::InvalidKernelIr(
                        "native program resource maps to a missing allocation",
                    ))?;
            if bytes.len() > destination.byte_len {
                return Err(VkFftError::InvalidKernelIr(
                    "native program initialization exceeds its physical allocation",
                ));
            }
            if bytes.len() == destination.byte_len && destination.host_bytes.is_none() {
                destination.host_bytes = Some(bytes);
            } else {
                let host_bytes = destination
                    .host_bytes
                    .get_or_insert_with(|| vec![0u8; destination.byte_len]);
                host_bytes[..bytes.len()].copy_from_slice(&bytes);
            }
        }
    }

    let output_resource = program.output_resource()?;
    let output_allocation = memory_plan.allocation_for(output_resource.id)?;
    let output_layout = output_resource
        .external_layout
        .ok_or(VkFftError::InvalidKernelIr(
            "native program output is missing its external layout",
        ))?;
    Ok(PreparedProgramStorage {
        backend,
        memory_plan,
        allocations,
        output_allocation,
        output_layout,
    })
}

#[cfg(test)]
#[allow(clippy::items_after_test_module)] // Keep storage tests adjacent to the helper they validate.
mod prepared_storage_tests {
    use super::*;
    use crate::kernel_ir::{BufferRole, DispatchGeometry, ScalarType};
    use crate::program_ir::{
        ProgramPass, ProgramPassBinding, ProgramResource, ProgramResourceId, ProgramResourceKind,
    };

    fn probe_program(scratch_first_access: BufferAccess) -> ProgramIr {
        let layout = ExternalBufferLayout {
            logical_len: 4,
            physical_stride: 4,
            batch_count: 1,
            element_shape: ProgramElementShape::Complex,
        };
        ProgramIr {
            name: "prepared_storage_probe".to_owned(),
            scalar: ScalarType::F32,
            resources: vec![
                ProgramResource {
                    id: ProgramResourceId(0),
                    name: "input".to_owned(),
                    kind: ProgramResourceKind::Input,
                    scalar: ScalarType::F32,
                    elements: 4,
                    external_layout: Some(layout),
                    initialization: ProgramResourceInitialization::ExternalInput,
                },
                ProgramResource {
                    id: ProgramResourceId(1),
                    name: "output".to_owned(),
                    kind: ProgramResourceKind::Output,
                    scalar: ScalarType::F32,
                    elements: 4,
                    external_layout: Some(layout),
                    initialization: ProgramResourceInitialization::Zeroed,
                },
                ProgramResource {
                    id: ProgramResourceId(2),
                    name: "scratch".to_owned(),
                    kind: ProgramResourceKind::Scratch,
                    scalar: ScalarType::F32,
                    elements: 4,
                    external_layout: None,
                    initialization: ProgramResourceInitialization::Zeroed,
                },
            ],
            passes: vec![
                ProgramPass {
                    name: "probe_fill_scratch".to_owned(),
                    dispatch: DispatchGeometry { x: 1, y: 1, z: 1 },
                    bindings: vec![
                        ProgramPassBinding {
                            binding: 0,
                            resource: ProgramResourceId(0),
                            role: BufferRole::Input,
                            access: BufferAccess::ReadOnly,
                        },
                        ProgramPassBinding {
                            binding: 1,
                            resource: ProgramResourceId(2),
                            role: BufferRole::Output,
                            access: scratch_first_access,
                        },
                    ],
                },
                ProgramPass {
                    name: "probe_write_output".to_owned(),
                    dispatch: DispatchGeometry { x: 1, y: 1, z: 1 },
                    bindings: vec![
                        ProgramPassBinding {
                            binding: 0,
                            resource: ProgramResourceId(2),
                            role: BufferRole::Input,
                            access: BufferAccess::ReadOnly,
                        },
                        ProgramPassBinding {
                            binding: 1,
                            resource: ProgramResourceId(1),
                            role: BufferRole::Output,
                            access: BufferAccess::WriteOnly,
                        },
                    ],
                },
            ],
        }
    }

    #[test]
    fn first_write_zeroed_allocations_skip_host_materialization_until_readback() {
        let program = probe_program(BufferAccess::WriteOnly);
        let mut prepared = prepare_program_storage(Backend::Cuda, &program, |resource| {
            Ok(match resource.initialization {
                ProgramResourceInitialization::ExternalInput => Some(vec![0x5a; 32]),
                _ => None,
            })
        })
        .expect("prepare first-write probe");
        let input = prepared
            .memory_plan
            .allocation_for(ProgramResourceId(0))
            .expect("input allocation");
        let output = prepared.output_allocation;
        let scratch = prepared
            .memory_plan
            .allocation_for(ProgramResourceId(2))
            .expect("scratch allocation");
        assert!(prepared.allocations[input.0].host_bytes.is_some());
        assert!(prepared.allocations[output.0].host_bytes.is_none());
        assert!(prepared.allocations[scratch.0].host_bytes.is_none());
        assert_eq!(prepared.allocations[output.0].byte_len, 32);
        assert_eq!(
            prepared
                .output_bytes_mut()
                .expect("lazy output bytes")
                .len(),
            32
        );
        assert!(prepared.allocations[output.0].host_bytes.is_some());
    }

    #[cfg(any(feature = "cuda-runtime", feature = "opencl-runtime"))]
    #[test]
    fn resident_complex32_input_skips_host_materialization() {
        let program = probe_program(BufferAccess::WriteOnly);
        let prepared = prepare_program_complex32_resident_input(Backend::Cuda, &program)
            .expect("prepare resident-input probe");
        let input = prepared
            .memory_plan
            .allocation_for(ProgramResourceId(0))
            .expect("input allocation");
        let output = prepared.output_allocation;
        let scratch = prepared
            .memory_plan
            .allocation_for(ProgramResourceId(2))
            .expect("scratch allocation");

        assert_eq!(prepared.allocations[input.0].byte_len, 32);
        assert!(prepared.allocations[input.0].host_bytes.is_none());
        assert!(prepared.allocations[output.0].host_bytes.is_none());
        assert!(prepared.allocations[scratch.0].host_bytes.is_none());
    }

    #[test]
    fn first_readwrite_zeroed_allocation_retains_zero_initialization() {
        let program = probe_program(BufferAccess::ReadWrite);
        let prepared = prepare_program_storage(Backend::Cuda, &program, |resource| {
            Ok(match resource.initialization {
                ProgramResourceInitialization::ExternalInput => Some(vec![0x5a; 32]),
                _ => None,
            })
        })
        .expect("prepare read-write probe");
        let scratch = prepared
            .memory_plan
            .allocation_for(ProgramResourceId(2))
            .expect("scratch allocation");
        let bytes = prepared.allocations[scratch.0]
            .host_bytes
            .as_deref()
            .expect("read-write scratch requires zero initialization");
        assert_eq!(bytes.len(), 32);
        assert!(bytes.iter().all(|byte| *byte == 0));
    }

    #[test]
    fn bulk_complex_codecs_preserve_native_endian_bit_patterns() {
        let values32 = [
            Complex32::new(0.0, -0.0),
            Complex32::new(f32::from_bits(0x7fc1_2345), f32::INFINITY),
            Complex32::new(f32::NEG_INFINITY, f32::from_bits(0xffc5_4321)),
        ];
        let mut expected32 = Vec::new();
        for value in values32 {
            expected32.extend_from_slice(&value.re.to_ne_bytes());
            expected32.extend_from_slice(&value.im.to_ne_bytes());
        }
        let encoded32 = encode_complex32(&values32);
        assert_eq!(encoded32, expected32);
        let decoded32 = decode_complex32(&encoded32).expect("decode F32 bulk bytes");
        assert_eq!(decoded32.len(), values32.len());
        for (actual, expected) in decoded32.iter().zip(values32) {
            assert_eq!(actual.re.to_bits(), expected.re.to_bits());
            assert_eq!(actual.im.to_bits(), expected.im.to_bits());
        }

        let values64 = [
            Complex64::new(0.0, -0.0),
            Complex64::new(f64::from_bits(0x7ff8_1234_5678_9abc), f64::INFINITY),
            Complex64::new(f64::NEG_INFINITY, f64::from_bits(0xfff8_cba9_8765_4321)),
        ];
        let mut expected64 = Vec::new();
        for value in values64 {
            expected64.extend_from_slice(&value.re.to_ne_bytes());
            expected64.extend_from_slice(&value.im.to_ne_bytes());
        }
        let encoded64 = encode_complex64(&values64);
        assert_eq!(encoded64, expected64);
        let decoded64 = decode_complex64(&encoded64).expect("decode F64 bulk bytes");
        assert_eq!(decoded64.len(), values64.len());
        for (actual, expected) in decoded64.iter().zip(values64) {
            assert_eq!(actual.re.to_bits(), expected.re.to_bits());
            assert_eq!(actual.im.to_bits(), expected.im.to_bits());
        }
    }
}

pub(crate) fn prepare_transform_input32(
    ir: &TransformIr,
    input: NativeTransformInput32<'_>,
) -> Result<Vec<Complex32>> {
    match (ir, input) {
        (TransformIr::Complex1d(_), NativeTransformInput32::Complex(values)) => Ok(values.to_vec()),
        (TransformIr::ComplexNd(nd), NativeTransformInput32::Complex(values)) => {
            nd.pack_formatted_input(values)
        }
        (TransformIr::Real(real), NativeTransformInput32::Real(values))
            if real.kind == RealFftKind::RealToComplex =>
        {
            Ok(values
                .iter()
                .map(|value| Complex32::new(*value, 0.0))
                .collect())
        }
        (TransformIr::Real(real), NativeTransformInput32::Complex(values))
            if real.kind == RealFftKind::ComplexToReal =>
        {
            Ok(values.to_vec())
        }
        (TransformIr::RealNd(real), NativeTransformInput32::Real(values))
            if real.kind == RealFftKind::RealToComplex =>
        {
            let logical = values
                .iter()
                .map(|value| Complex32::new(*value, 0.0))
                .collect::<Vec<_>>();
            real.pack_formatted_input(&logical)
        }
        (TransformIr::RealNd(real), NativeTransformInput32::Complex(values))
            if real.kind == RealFftKind::ComplexToReal =>
        {
            real.pack_formatted_input(values)
        }
        (TransformIr::RealToRealNd(r2r), NativeTransformInput32::Real(values)) => {
            let logical = values
                .iter()
                .map(|value| Complex32::new(*value, 0.0))
                .collect::<Vec<_>>();
            r2r.pack_formatted_input(&logical)
        }
        (
            TransformIr::RealToRealDoubleDouble(_)
            | TransformIr::RealToRealNdDoubleDouble(_)
            | TransformIr::RealToReal(_),
            NativeTransformInput32::Real(values),
        ) => Ok(values
            .iter()
            .map(|value| Complex32::new(*value, 0.0))
            .collect()),
        _ => Err(VkFftError::InvalidKernelIr(
            "native transform input kind does not match TransformIr",
        )),
    }
}

pub(crate) fn prepare_transform_input64(
    ir: &TransformIr,
    input: NativeTransformInput64<'_>,
) -> Result<Vec<Complex64>> {
    match (ir, input) {
        (TransformIr::Complex1d(_), NativeTransformInput64::Complex(values)) => Ok(values.to_vec()),
        (TransformIr::ComplexNd(nd), NativeTransformInput64::Complex(values)) => {
            nd.pack_formatted_input(values)
        }
        (TransformIr::Real(real), NativeTransformInput64::Real(values))
            if real.kind == RealFftKind::RealToComplex =>
        {
            Ok(values
                .iter()
                .map(|value| Complex64::new(*value, 0.0))
                .collect())
        }
        (TransformIr::Real(real), NativeTransformInput64::Complex(values))
            if real.kind == RealFftKind::ComplexToReal =>
        {
            Ok(values.to_vec())
        }
        (TransformIr::RealNd(real), NativeTransformInput64::Real(values))
            if real.kind == RealFftKind::RealToComplex =>
        {
            let logical = values
                .iter()
                .map(|value| Complex64::new(*value, 0.0))
                .collect::<Vec<_>>();
            real.pack_formatted_input(&logical)
        }
        (TransformIr::RealNd(real), NativeTransformInput64::Complex(values))
            if real.kind == RealFftKind::ComplexToReal =>
        {
            real.pack_formatted_input(values)
        }
        (TransformIr::RealToRealNd(r2r), NativeTransformInput64::Real(values)) => {
            let logical = values
                .iter()
                .map(|value| Complex64::new(*value, 0.0))
                .collect::<Vec<_>>();
            r2r.pack_formatted_input(&logical)
        }
        (TransformIr::RealToReal(_), NativeTransformInput64::Real(values)) => Ok(values
            .iter()
            .map(|value| Complex64::new(*value, 0.0))
            .collect()),
        _ => Err(VkFftError::InvalidKernelIr(
            "native transform input kind does not match TransformIr",
        )),
    }
}

pub(crate) fn finish_transform_output32(
    ir: &TransformIr,
    output: Vec<Complex32>,
) -> Result<NativeTransformOutput32> {
    Ok(match ir {
        TransformIr::ComplexNd(nd) => {
            NativeTransformOutput32::Complex(nd.unpack_formatted_output(&output)?)
        }
        TransformIr::Complex1d(_)
        | TransformIr::Complex1dDoubleDouble(_)
        | TransformIr::ComplexNdDoubleDouble(_) => NativeTransformOutput32::Complex(output),
        TransformIr::Real(real) if real.kind == RealFftKind::RealToComplex => {
            NativeTransformOutput32::Complex(output)
        }
        TransformIr::RealDoubleDouble(real) if real.kind == RealFftKind::RealToComplex => {
            NativeTransformOutput32::Complex(output)
        }
        TransformIr::RealNdDoubleDouble(real) if real.kind == RealFftKind::RealToComplex => {
            NativeTransformOutput32::Complex(output)
        }
        TransformIr::RealNd(real) => match real.kind {
            RealFftKind::RealToComplex => {
                NativeTransformOutput32::Complex(real.unpack_formatted_output(&output)?)
            }
            RealFftKind::ComplexToReal => {
                let logical = real.unpack_formatted_output(&output)?;
                NativeTransformOutput32::Real(logical.into_iter().map(|value| value.re).collect())
            }
        },
        TransformIr::RealToRealNd(r2r) => {
            let logical = r2r.unpack_formatted_output(&output)?;
            NativeTransformOutput32::Real(logical.into_iter().map(|value| value.re).collect())
        }
        TransformIr::Real(_)
        | TransformIr::RealDoubleDouble(_)
        | TransformIr::RealNdDoubleDouble(_)
        | TransformIr::RealToRealDoubleDouble(_)
        | TransformIr::RealToRealNdDoubleDouble(_)
        | TransformIr::RealToReal(_) => {
            NativeTransformOutput32::Real(output.into_iter().map(|value| value.re).collect())
        }
    })
}

pub(crate) fn finish_transform_output64(
    ir: &TransformIr,
    output: Vec<Complex64>,
) -> Result<NativeTransformOutput64> {
    Ok(match ir {
        TransformIr::ComplexNd(nd) => {
            NativeTransformOutput64::Complex(nd.unpack_formatted_output(&output)?)
        }
        TransformIr::Complex1d(_)
        | TransformIr::Complex1dDoubleDouble(_)
        | TransformIr::ComplexNdDoubleDouble(_) => NativeTransformOutput64::Complex(output),
        TransformIr::Real(real) if real.kind == RealFftKind::RealToComplex => {
            NativeTransformOutput64::Complex(output)
        }
        TransformIr::RealDoubleDouble(real) if real.kind == RealFftKind::RealToComplex => {
            NativeTransformOutput64::Complex(output)
        }
        TransformIr::RealNdDoubleDouble(real) if real.kind == RealFftKind::RealToComplex => {
            NativeTransformOutput64::Complex(output)
        }
        TransformIr::RealNd(real) => match real.kind {
            RealFftKind::RealToComplex => {
                NativeTransformOutput64::Complex(real.unpack_formatted_output(&output)?)
            }
            RealFftKind::ComplexToReal => {
                let logical = real.unpack_formatted_output(&output)?;
                NativeTransformOutput64::Real(logical.into_iter().map(|value| value.re).collect())
            }
        },
        TransformIr::RealToRealNd(r2r) => {
            let logical = r2r.unpack_formatted_output(&output)?;
            NativeTransformOutput64::Real(logical.into_iter().map(|value| value.re).collect())
        }
        TransformIr::Real(_)
        | TransformIr::RealDoubleDouble(_)
        | TransformIr::RealNdDoubleDouble(_)
        | TransformIr::RealToRealDoubleDouble(_)
        | TransformIr::RealToRealNdDoubleDouble(_)
        | TransformIr::RealToReal(_) => {
            NativeTransformOutput64::Real(output.into_iter().map(|value| value.re).collect())
        }
    })
}

#[cfg(all(test, any(feature = "cuda-runtime", feature = "opencl-runtime")))]
fn assert_native_precision_case<R: NativeRuntime>(
    runtime: &R,
    family: crate::PrecisionTransformFamily,
    shape: &[usize],
    f32_metrics: crate::PrecisionMetrics,
    f64_metrics: crate::PrecisionMetrics,
) {
    let logical_len = shape.iter().product::<usize>() as f64;
    assert!(
        f32_metrics.is_finite(),
        "non-finite {:?} F32 {family:?} metrics: {f32_metrics:?}",
        runtime.backend()
    );
    assert!(
        f64_metrics.is_finite(),
        "non-finite {:?} F64 {family:?} metrics: {f64_metrics:?}",
        runtime.backend()
    );
    assert!(
        f32_metrics.max_difference <= 1.0e-2 * logical_len,
        "{:?} F32 {family:?} max difference too large for {shape:?}: {f32_metrics:?}",
        runtime.backend()
    );
    assert!(
        f32_metrics.avg_eps <= 2.0e-3,
        "{:?} F32 {family:?} average relative epsilon too large for {shape:?}: {f32_metrics:?}",
        runtime.backend()
    );
    assert!(
        f64_metrics.max_difference <= 5.0e-8 * logical_len,
        "{:?} F64 {family:?} max difference too large for {shape:?}: {f64_metrics:?}",
        runtime.backend()
    );
    assert!(
        f64_metrics.avg_eps <= 1.0e-8,
        "{:?} F64 {family:?} average relative epsilon too large for {shape:?}: {f64_metrics:?}",
        runtime.backend()
    );
    assert!(
        f64_metrics.avg_difference <= f32_metrics.avg_difference * 0.2 + f64::EPSILON,
        "{:?} F64 should materially improve {family:?} average error for {shape:?}: f32={f32_metrics:?}, f64={f64_metrics:?}",
        runtime.backend()
    );
    for (precision, metrics) in [
        (crate::Precision::F32, f32_metrics),
        (crate::Precision::F64, f64_metrics),
    ] {
        let report = crate::PrecisionCaseReport {
            backend: runtime.backend(),
            device: runtime.device_name().to_owned(),
            family,
            shape: shape.to_vec(),
            precision,
            metrics,
        };
        let json = report.to_json_line();
        assert!(json.starts_with("{\"schema_version\":1,"));
        assert!(json.contains(&format!("\"family\":\"{}\"", family.as_str())));
    }
}

#[cfg(all(test, any(feature = "cuda-runtime", feature = "opencl-runtime")))]
fn native_complex_precision_case<R: NativeRuntime>(
    runtime: &R,
    shape: &[usize],
    input32: &[Complex32],
) -> (crate::PrecisionMetrics, crate::PrecisionMetrics) {
    let profile = runtime.device_profile();
    let input64 = input32
        .iter()
        .map(|value| Complex64::new(value.re as f64, value.im as f64))
        .collect::<Vec<_>>();
    let ir32 = TransformIr::build(
        crate::FftConfig::new(shape.to_vec()),
        crate::Direction::Forward,
        profile,
    )
    .unwrap();
    let ir64 = TransformIr::build(
        crate::FftConfig::new(shape.to_vec()).with_precision(crate::Precision::F64),
        crate::Direction::Forward,
        profile,
    )
    .unwrap();
    let reference = ir64.execute_complex_reference(&input64).unwrap();
    let actual32 = match runtime
        .execute_transform_f32(&ir32, NativeTransformInput32::Complex(input32))
        .unwrap()
    {
        NativeTransformOutput32::Complex(values) => values,
        NativeTransformOutput32::Real(_) => {
            panic!("native C2C precision case returned real output")
        }
    };
    let actual64 = match runtime
        .execute_transform_f64(&ir64, NativeTransformInput64::Complex(&input64))
        .unwrap()
    {
        NativeTransformOutput64::Complex(values) => values,
        NativeTransformOutput64::Real(_) => {
            panic!("native F64 C2C precision case returned real output")
        }
    };
    let actual32 = actual32
        .iter()
        .map(|value| Complex64::new(value.re as f64, value.im as f64))
        .collect::<Vec<_>>();
    (
        crate::complex_precision_metrics(&actual32, &reference).unwrap(),
        crate::complex_precision_metrics(&actual64, &reference).unwrap(),
    )
}

#[cfg(all(test, any(feature = "cuda-runtime", feature = "opencl-runtime")))]
fn native_r2c_precision_case<R: NativeRuntime>(
    runtime: &R,
    shape: &[usize],
    input32: &[f32],
) -> (crate::PrecisionMetrics, crate::PrecisionMetrics) {
    let profile = runtime.device_profile();
    let input64 = input32
        .iter()
        .map(|value| *value as f64)
        .collect::<Vec<_>>();
    let ir32 = TransformIr::build(
        crate::FftConfig::new(shape.to_vec()).with_transform(crate::TransformKind::RealToComplex),
        crate::Direction::Forward,
        profile,
    )
    .unwrap();
    let ir64 = TransformIr::build(
        crate::FftConfig::new(shape.to_vec())
            .with_precision(crate::Precision::F64)
            .with_transform(crate::TransformKind::RealToComplex),
        crate::Direction::Forward,
        profile,
    )
    .unwrap();
    let reference = ir64.execute_r2c_reference(&input64).unwrap();
    let actual32 = match runtime
        .execute_transform_f32(&ir32, NativeTransformInput32::Real(input32))
        .unwrap()
    {
        NativeTransformOutput32::Complex(values) => values,
        NativeTransformOutput32::Real(_) => {
            panic!("native R2C precision case returned real output")
        }
    };
    let actual64 = match runtime
        .execute_transform_f64(&ir64, NativeTransformInput64::Real(&input64))
        .unwrap()
    {
        NativeTransformOutput64::Complex(values) => values,
        NativeTransformOutput64::Real(_) => {
            panic!("native F64 R2C precision case returned real output")
        }
    };
    let actual32 = actual32
        .iter()
        .map(|value| Complex64::new(value.re as f64, value.im as f64))
        .collect::<Vec<_>>();
    (
        crate::complex_precision_metrics(&actual32, &reference).unwrap(),
        crate::complex_precision_metrics(&actual64, &reference).unwrap(),
    )
}

#[cfg(all(test, any(feature = "cuda-runtime", feature = "opencl-runtime")))]
fn native_r2r_precision_case<R: NativeRuntime>(
    runtime: &R,
    shape: &[usize],
    input32: &[f32],
) -> (crate::PrecisionMetrics, crate::PrecisionMetrics) {
    let profile = runtime.device_profile();
    let input64 = input32
        .iter()
        .map(|value| *value as f64)
        .collect::<Vec<_>>();
    let transform = crate::TransformKind::Dct(crate::DctType::II);
    let ir32 = TransformIr::build(
        crate::FftConfig::new(shape.to_vec()).with_transform(transform),
        crate::Direction::Forward,
        profile,
    )
    .unwrap();
    let ir64 = TransformIr::build(
        crate::FftConfig::new(shape.to_vec())
            .with_precision(crate::Precision::F64)
            .with_transform(transform),
        crate::Direction::Forward,
        profile,
    )
    .unwrap();
    let reference = ir64.execute_r2r_reference(&input64).unwrap();
    let actual32 = match runtime
        .execute_transform_f32(&ir32, NativeTransformInput32::Real(input32))
        .unwrap()
    {
        NativeTransformOutput32::Real(values) => values,
        NativeTransformOutput32::Complex(_) => {
            panic!("native R2R precision case returned complex output")
        }
    };
    let actual64 = match runtime
        .execute_transform_f64(&ir64, NativeTransformInput64::Real(&input64))
        .unwrap()
    {
        NativeTransformOutput64::Real(values) => values,
        NativeTransformOutput64::Complex(_) => {
            panic!("native F64 R2R precision case returned complex output")
        }
    };
    let actual32 = actual32
        .iter()
        .map(|value| *value as f64)
        .collect::<Vec<_>>();
    (
        crate::real_precision_metrics(&actual32, &reference).unwrap(),
        crate::real_precision_metrics(&actual64, &reference).unwrap(),
    )
}

#[cfg(all(test, any(feature = "cuda-runtime", feature = "opencl-runtime")))]
pub(crate) fn assert_native_bluestein_even_real<R: NativeRuntime>(runtime: &R) {
    let length = 206usize;
    let input32 = (0..length)
        .map(|index| {
            let x = index as f32;
            (0.083 * x).sin() + 0.19 * (0.031 * x).cos() + x * 0.0002
        })
        .collect::<Vec<_>>();
    let input64 = input32
        .iter()
        .map(|value| *value as f64)
        .collect::<Vec<_>>();
    let mut tuning = crate::PlannerTuning::portable();
    tuning.max_rader_fft_prime = 100;
    let portable_real = |config: crate::FftConfig, device: DeviceProfile| {
        let plan = crate::FftPlan::build(config).unwrap();
        TransformIr::Real(crate::RealFftIr::build(&plan, device).unwrap())
    };

    let forward32 = portable_real(
        crate::FftConfig::new(vec![length])
            .with_transform(crate::TransformKind::RealToComplex)
            .with_tuning(tuning),
        runtime.device_profile(),
    );
    let TransformIr::Real(real32) = &forward32 else {
        panic!("native Bluestein even-real test expected R2C IR");
    };
    assert!(real32.fused_even_bluestein_ir().unwrap().is_some());
    let expected32 = forward32.execute_r2c_reference(&input64).unwrap();
    let spectrum32 = match runtime
        .execute_transform_f32(&forward32, NativeTransformInput32::Real(&input32))
        .unwrap()
    {
        NativeTransformOutput32::Complex(values) => values,
        NativeTransformOutput32::Real(_) => panic!("native Bluestein R2C returned real output"),
    };
    let forward_error32 = spectrum32
        .iter()
        .zip(&expected32)
        .map(|(actual, expected)| {
            let dr = (actual.re as f64 - expected.re).abs();
            let di = (actual.im as f64 - expected.im).abs();
            dr.max(di)
        })
        .fold(0.0, f64::max);
    assert!(
        forward_error32 <= 3.0e-3 * length as f64,
        "{:?} p103-Bluestein R2C F32 error {forward_error32:e}",
        runtime.backend()
    );

    let inverse32 = portable_real(
        crate::FftConfig::new(vec![length])
            .with_transform(crate::TransformKind::ComplexToReal)
            .with_inverse_normalization(true)
            .with_tuning(tuning),
        runtime.device_profile(),
    );
    let TransformIr::Real(real32) = &inverse32 else {
        panic!("native Bluestein even-real test expected C2R IR");
    };
    assert!(real32.fused_even_bluestein_ir().unwrap().is_some());
    let restored32 = match runtime
        .execute_transform_f32(&inverse32, NativeTransformInput32::Complex(&spectrum32))
        .unwrap()
    {
        NativeTransformOutput32::Real(values) => values,
        NativeTransformOutput32::Complex(_) => {
            panic!("native Bluestein C2R returned complex output")
        }
    };
    let round_trip32 = restored32
        .iter()
        .zip(&input32)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0, f32::max);
    let spectrum32_as64 = spectrum32
        .iter()
        .map(|value| Complex64::new(value.re as f64, value.im as f64))
        .collect::<Vec<_>>();
    let inverse_expected32 = inverse32.execute_c2r_reference(&spectrum32_as64).unwrap();
    let oracle_error32 = restored32
        .iter()
        .zip(&inverse_expected32)
        .map(|(actual, expected)| (*actual as f64 - expected).abs())
        .fold(0.0, f64::max);
    assert!(
        round_trip32 <= 3.5e-3 * length as f32 && oracle_error32 <= 3.5e-3 * length as f64,
        "{:?} p103-Bluestein C2R F32 errors: round_trip={round_trip32:e}, oracle={oracle_error32:e}",
        runtime.backend()
    );

    let deep_length = 4106usize;
    let deep_input32 = (0..deep_length)
        .map(|index| {
            let x = index as f32;
            (0.041 * x).sin() + 0.17 * (0.013 * x).cos() + x * 0.00007
        })
        .collect::<Vec<_>>();
    let deep_input64 = deep_input32
        .iter()
        .map(|value| *value as f64)
        .collect::<Vec<_>>();
    let mut constrained = runtime.device_profile();
    constrained.shared_memory_bytes = 32 * 1024;
    constrained.shared_memory_pow2_bytes = 32 * 1024;
    let deep_forward = portable_real(
        crate::FftConfig::new(vec![deep_length])
            .with_transform(crate::TransformKind::RealToComplex)
            .with_tuning(tuning),
        constrained,
    );
    let TransformIr::Real(deep_real) = &deep_forward else {
        panic!("native deep Bluestein even-real test expected R2C IR");
    };
    let deep_fused = deep_real.fused_even_bluestein_ir().unwrap().unwrap();
    assert_eq!(deep_fused.logical_len, 2053);
    assert_eq!(deep_fused.convolution_len, 4368);
    assert!(matches!(
        deep_fused.forward_fft.root,
        crate::RecursiveFftNodeIr::CooleyTukey(_)
    ));
    let deep_expected = deep_forward.execute_r2c_reference(&deep_input64).unwrap();
    let deep_spectrum = match runtime
        .execute_transform_f32(&deep_forward, NativeTransformInput32::Real(&deep_input32))
        .unwrap()
    {
        NativeTransformOutput32::Complex(values) => values,
        NativeTransformOutput32::Real(_) => {
            panic!("native deep Bluestein R2C returned real output")
        }
    };
    let deep_forward_error = deep_spectrum
        .iter()
        .zip(&deep_expected)
        .map(|(actual, expected)| {
            (actual.re as f64 - expected.re)
                .abs()
                .max((actual.im as f64 - expected.im).abs())
        })
        .fold(0.0, f64::max);
    assert!(
        deep_forward_error <= 3.0e-3 * deep_length as f64,
        "{:?} recursive p2053-Bluestein R2C F32 error {deep_forward_error:e}",
        runtime.backend()
    );
    let deep_inverse = portable_real(
        crate::FftConfig::new(vec![deep_length])
            .with_transform(crate::TransformKind::ComplexToReal)
            .with_inverse_normalization(true)
            .with_tuning(tuning),
        constrained,
    );
    let TransformIr::Real(deep_real) = &deep_inverse else {
        panic!("native deep Bluestein even-real test expected C2R IR");
    };
    let deep_inverse_fused = deep_real.fused_even_bluestein_ir().unwrap().unwrap();
    assert!(matches!(
        deep_inverse_fused.inverse_fft.root,
        crate::RecursiveFftNodeIr::CooleyTukey(_)
    ));
    let deep_restored = match runtime
        .execute_transform_f32(
            &deep_inverse,
            NativeTransformInput32::Complex(&deep_spectrum),
        )
        .unwrap()
    {
        NativeTransformOutput32::Real(values) => values,
        NativeTransformOutput32::Complex(_) => {
            panic!("native deep Bluestein C2R returned complex output")
        }
    };
    let deep_round_trip = deep_restored
        .iter()
        .zip(&deep_input32)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0, f32::max);
    assert!(
        deep_round_trip <= 3.5e-3 * deep_length as f32,
        "{:?} recursive p2053-Bluestein C2R F32 round-trip error {deep_round_trip:e}",
        runtime.backend()
    );

    if !runtime.device_profile().supports_f64 {
        return;
    }
    let forward64 = portable_real(
        crate::FftConfig::new(vec![length])
            .with_precision(crate::Precision::F64)
            .with_transform(crate::TransformKind::RealToComplex)
            .with_tuning(tuning),
        runtime.device_profile(),
    );
    let spectrum64 = match runtime
        .execute_transform_f64(&forward64, NativeTransformInput64::Real(&input64))
        .unwrap()
    {
        NativeTransformOutput64::Complex(values) => values,
        NativeTransformOutput64::Real(_) => panic!("native Bluestein F64 R2C returned real output"),
    };
    let expected64 = forward64.execute_r2c_reference(&input64).unwrap();
    let forward_error64 = spectrum64
        .iter()
        .zip(&expected64)
        .map(|(actual, expected)| {
            (actual.re - expected.re)
                .abs()
                .max((actual.im - expected.im).abs())
        })
        .fold(0.0, f64::max);
    assert!(
        forward_error64 <= 5.0e-10 * length as f64,
        "{:?} p103-Bluestein R2C F64 error {forward_error64:e}",
        runtime.backend()
    );

    let inverse64 = portable_real(
        crate::FftConfig::new(vec![length])
            .with_precision(crate::Precision::F64)
            .with_transform(crate::TransformKind::ComplexToReal)
            .with_inverse_normalization(true)
            .with_tuning(tuning),
        runtime.device_profile(),
    );
    let restored64 = match runtime
        .execute_transform_f64(&inverse64, NativeTransformInput64::Complex(&spectrum64))
        .unwrap()
    {
        NativeTransformOutput64::Real(values) => values,
        NativeTransformOutput64::Complex(_) => {
            panic!("native Bluestein F64 C2R returned complex output")
        }
    };
    let round_trip64 = restored64
        .iter()
        .zip(&input64)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0, f64::max);
    assert!(
        round_trip64 <= 7.0e-10 * length as f64,
        "{:?} p103-Bluestein C2R F64 round-trip error {round_trip64:e}",
        runtime.backend()
    );
}

#[cfg(all(test, any(feature = "cuda-runtime", feature = "opencl-runtime")))]
pub(crate) fn assert_native_precision_matrix<R: NativeRuntime>(runtime: &R) {
    let nd_shape = [17usize, 34usize];
    let tensor_len = nd_shape.iter().product::<usize>();
    let complex_input = (0..tensor_len)
        .map(|index| {
            let x = index as f32;
            Complex32::new(
                (0.031 * x).sin() + 0.00021 * x,
                (0.047 * x).cos() - 0.00013 * x,
            )
        })
        .collect::<Vec<_>>();
    let (f32_metrics, f64_metrics) =
        native_complex_precision_case(runtime, &nd_shape, &complex_input);
    assert_native_precision_case(
        runtime,
        crate::PrecisionTransformFamily::C2cNd,
        &nd_shape,
        f32_metrics,
        f64_metrics,
    );

    let real_1d_shape = [65usize];
    let real_1d = (0..real_1d_shape[0])
        .map(|index| {
            let x = index as f32;
            (0.083 * x).sin() + 0.23 * (0.029 * x).cos() + 0.0007 * x
        })
        .collect::<Vec<_>>();
    let (f32_metrics, f64_metrics) = native_r2c_precision_case(runtime, &real_1d_shape, &real_1d);
    assert_native_precision_case(
        runtime,
        crate::PrecisionTransformFamily::R2c1d,
        &real_1d_shape,
        f32_metrics,
        f64_metrics,
    );

    let real_nd = (0..tensor_len)
        .map(|index| {
            let x = index as f32;
            (0.057 * x).sin() + 0.19 * (0.017 * x).cos() - 0.00031 * x
        })
        .collect::<Vec<_>>();
    let (f32_metrics, f64_metrics) = native_r2c_precision_case(runtime, &nd_shape, &real_nd);
    assert_native_precision_case(
        runtime,
        crate::PrecisionTransformFamily::R2cNd,
        &nd_shape,
        f32_metrics,
        f64_metrics,
    );

    let r2r_1d_shape = [257usize];
    let r2r_1d = (0..r2r_1d_shape[0])
        .map(|index| {
            let x = index as f32;
            (0.041 * x).sin() + 0.13 * (0.023 * x).cos() + 0.00011 * x
        })
        .collect::<Vec<_>>();
    let (f32_metrics, f64_metrics) = native_r2r_precision_case(runtime, &r2r_1d_shape, &r2r_1d);
    assert_native_precision_case(
        runtime,
        crate::PrecisionTransformFamily::R2r1d,
        &r2r_1d_shape,
        f32_metrics,
        f64_metrics,
    );

    let r2r_nd = (0..tensor_len)
        .map(|index| {
            let x = index as f32;
            (0.037 * x).sin() + 0.17 * (0.019 * x).cos() + 0.00009 * x
        })
        .collect::<Vec<_>>();
    let (f32_metrics, f64_metrics) = native_r2r_precision_case(runtime, &nd_shape, &r2r_nd);
    assert_native_precision_case(
        runtime,
        crate::PrecisionTransformFamily::R2rNd,
        &nd_shape,
        f32_metrics,
        f64_metrics,
    );
}

pub(crate) fn encode_complex32(values: &[Complex32]) -> Vec<u8> {
    const { assert!(std::mem::size_of::<Complex32>() == 8) };
    let byte_len = std::mem::size_of_val(values);
    // SAFETY: `Complex32` is `#[repr(C)]` with exactly two `f32` fields and the static
    // size assertion excludes padding. Reading initialized values as bytes preserves the
    // same native-endian representation previously produced by `to_ne_bytes`.
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), byte_len).to_vec() }
}

pub(crate) fn encode_complex64(values: &[Complex64]) -> Vec<u8> {
    const { assert!(std::mem::size_of::<Complex64>() == 16) };
    let byte_len = std::mem::size_of_val(values);
    // SAFETY: `Complex64` is `#[repr(C)]` with exactly two `f64` fields and the static
    // size assertion excludes padding. Reading initialized values as bytes preserves the
    // same native-endian representation previously produced by `to_ne_bytes`.
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), byte_len).to_vec() }
}

fn decode_complex16_layout(bytes: &[u8], layout: ExternalBufferLayout) -> Result<Vec<Complex32>> {
    let physical = decode_complex16_native(bytes)?;
    let mut output = Vec::with_capacity(layout.logical_elements()?);
    for batch in 0..layout.batch_count {
        let start = batch * layout.physical_stride;
        output.extend_from_slice(&physical[start..start + layout.logical_len]);
    }
    Ok(output)
}

fn decode_complex32_layout(bytes: &[u8], layout: ExternalBufferLayout) -> Result<Vec<Complex32>> {
    if layout.physical_stride == layout.logical_len {
        return decode_complex32(bytes);
    }
    let physical = decode_complex32(bytes)?;
    let mut output = Vec::with_capacity(layout.logical_elements()?);
    for batch in 0..layout.batch_count {
        let start = batch * layout.physical_stride;
        output.extend_from_slice(&physical[start..start + layout.logical_len]);
    }
    Ok(output)
}

fn decode_complex64_layout(bytes: &[u8], layout: ExternalBufferLayout) -> Result<Vec<Complex64>> {
    if layout.physical_stride == layout.logical_len {
        return decode_complex64(bytes);
    }
    let physical = decode_complex64(bytes)?;
    let mut output = Vec::with_capacity(layout.logical_elements()?);
    for batch in 0..layout.batch_count {
        let start = batch * layout.physical_stride;
        output.extend_from_slice(&physical[start..start + layout.logical_len]);
    }
    Ok(output)
}

fn encode_double_double_scalars(values: &[DoubleDouble]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 16);
    for value in values {
        bytes.extend_from_slice(&value.hi.to_ne_bytes());
        bytes.extend_from_slice(&value.lo.to_ne_bytes());
    }
    bytes
}

fn encode_f64_scalars(values: &[f64]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 8);
    for value in values {
        bytes.extend_from_slice(&value.to_ne_bytes());
    }
    bytes
}

fn decode_double_double_scalar_layout(
    bytes: &[u8],
    layout: ExternalBufferLayout,
) -> Result<Vec<DoubleDouble>> {
    if bytes.len() < layout.physical_elements()?.saturating_mul(16) {
        return Err(VkFftError::InvalidKernelIr(
            "scalar DD native byte buffer is shorter than its physical layout",
        ));
    }
    let mut output = Vec::with_capacity(layout.logical_elements()?);
    for batch in 0..layout.batch_count {
        let base = batch * layout.physical_stride;
        for index in 0..layout.logical_len {
            let offset = (base + index) * 16;
            let hi = f64::from_ne_bytes(bytes[offset..offset + 8].try_into().expect("fixed chunk"));
            let lo = f64::from_ne_bytes(
                bytes[offset + 8..offset + 16]
                    .try_into()
                    .expect("fixed chunk"),
            );
            output.push(DoubleDouble::from_parts(hi, lo));
        }
    }
    Ok(output)
}

fn decode_f64_scalar_layout(bytes: &[u8], layout: ExternalBufferLayout) -> Result<Vec<f64>> {
    if bytes.len() < layout.physical_elements()?.saturating_mul(8) {
        return Err(VkFftError::InvalidKernelIr(
            "scalar F64 native byte buffer is shorter than its physical layout",
        ));
    }
    let mut output = Vec::with_capacity(layout.logical_elements()?);
    for batch in 0..layout.batch_count {
        let base = batch * layout.physical_stride;
        for index in 0..layout.logical_len {
            let offset = (base + index) * 8;
            output.push(f64::from_ne_bytes(
                bytes[offset..offset + 8].try_into().expect("fixed chunk"),
            ));
        }
    }
    Ok(output)
}

pub(crate) fn decode_complex32(bytes: &[u8]) -> Result<Vec<Complex32>> {
    const { assert!(std::mem::size_of::<Complex32>() == 8) };
    if !bytes.len().is_multiple_of(8) {
        return Err(VkFftError::InvalidKernelIr(
            "F32 native complex byte buffer must be a multiple of eight",
        ));
    }
    let count = bytes.len() / 8;
    let mut output = Vec::<Complex32>::with_capacity(count);
    // SAFETY: capacity reserves exactly `count * size_of::<Complex32>() == bytes.len()`
    // writable bytes; every possible `f32` bit pattern is valid, and the copy initializes
    // every byte before `set_len` exposes the values.
    unsafe {
        std::ptr::copy_nonoverlapping(
            bytes.as_ptr(),
            output.as_mut_ptr().cast::<u8>(),
            bytes.len(),
        );
        output.set_len(count);
    }
    Ok(output)
}

pub(crate) fn decode_complex64(bytes: &[u8]) -> Result<Vec<Complex64>> {
    const { assert!(std::mem::size_of::<Complex64>() == 16) };
    if !bytes.len().is_multiple_of(16) {
        return Err(VkFftError::InvalidKernelIr(
            "F64 native complex byte buffer must be a multiple of sixteen",
        ));
    }
    let count = bytes.len() / 16;
    let mut output = Vec::<Complex64>::with_capacity(count);
    // SAFETY: capacity reserves exactly `count * size_of::<Complex64>() == bytes.len()`
    // writable bytes; every possible `f64` bit pattern is valid, and the copy initializes
    // every byte before `set_len` exposes the values.
    unsafe {
        std::ptr::copy_nonoverlapping(
            bytes.as_ptr(),
            output.as_mut_ptr().cast::<u8>(),
            bytes.len(),
        );
        output.set_len(count);
    }
    Ok(output)
}
