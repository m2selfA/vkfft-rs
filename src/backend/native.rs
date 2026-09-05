//! Shared source-dialect lowering for CUDA, HIP, OpenCL, Level Zero, and Metal.
//!
//! The mature Vulkan GLSL emitter is used as a semantic frontend: it consumes the
//! same typed IR and already covers Stockham, Rader, Bluestein, recursive, real,
//! multidimensional, and R2R kernels. This module rewrites only the execution ABI
//! and language surface (builtins, buffers, workgroup memory, barriers and vector
//! construction), keeping FFT arithmetic identical across source backends.

use crate::application::TransformIr;
use crate::backend::KernelBackend;
use crate::backend::vulkan::{VulkanDescriptorBinding, VulkanGlslBackend, VulkanShaderSource};
use crate::config::Backend;
use crate::convolution_ir::{ConvolutionIr, NdConvolutionIr, NdRealConvolutionIr};
use crate::error::{Result, VkFftError};
use crate::kernel_ir::{
    BufferAccess, BufferBinding, BufferRole, DispatchGeometry, KernelIr, ScalarType, WorkgroupSize,
};
use crate::program_ir::{ProgramElementShape, ProgramIr, ProgramPass};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeShaderSource {
    pub backend: Backend,
    pub entry_point: &'static str,
    pub source: String,
    pub compiler_fallback_source: Option<String>,
    pub scalar: ScalarType,
    pub sequence_len: usize,
    pub batch_count: usize,
    pub workgroup_size: WorkgroupSize,
    pub dispatch: DispatchGeometry,
    pub bindings: Vec<BufferBinding>,
    pub required_shared_memory_bytes: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct NativeProgramSource {
    pub backend: Backend,
    pub program: ProgramIr,
    pub shaders: Vec<NativeShaderSource>,
}

impl NativeProgramSource {
    pub fn validate(&self) -> Result<()> {
        self.program.validate()?;
        if matches!(self.backend, Backend::Vulkan | Backend::CpuReference) {
            return Err(VkFftError::InvalidKernelIr(
                "native program source requires CUDA/HIP/OpenCL/Level Zero/Metal backend",
            ));
        }
        if self.shaders.len() != self.program.passes.len() {
            return Err(VkFftError::InvalidKernelIr(
                "native program shader count must match ProgramIr pass count",
            ));
        }
        for (pass, shader) in self.program.passes.iter().zip(&self.shaders) {
            validate_program_pass_shader(self.backend, &self.program, pass, shader)?;
        }
        Ok(())
    }
}

impl NativeShaderSource {
    pub fn validate(&self) -> Result<()> {
        if matches!(self.backend, Backend::Vulkan | Backend::CpuReference) {
            return Err(VkFftError::InvalidKernelIr(
                "native source shader requires CUDA/HIP/OpenCL/Level Zero/Metal backend",
            ));
        }
        if self.entry_point != "VkFFT_main"
            || self.workgroup_size.x == 0
            || self.workgroup_size.y == 0
            || self.workgroup_size.z == 0
            || self.dispatch.x == 0
            || self.dispatch.y == 0
            || self.dispatch.z == 0
        {
            return Err(VkFftError::InvalidKernelIr(
                "native source shader launch metadata is inconsistent",
            ));
        }
        if self.backend == Backend::Metal && self.scalar == ScalarType::F64 {
            return Err(VkFftError::UnsupportedPrecision {
                backend: "Metal source backend",
                precision: "f64",
            });
        }
        let mut seen = std::collections::BTreeSet::new();
        for binding in &self.bindings {
            let supported_mixed_pair = matches!(
                (self.scalar, binding.scalar),
                (ScalarType::F64, ScalarType::F32)
                    | (ScalarType::F32, ScalarType::F16)
                    | (ScalarType::DoubleDouble, ScalarType::F64)
            );
            let caller_visible_role =
                matches!(binding.role, BufferRole::Input | BufferRole::Output)
                    || (binding.role == BufferRole::Auxiliary
                        && binding.access == BufferAccess::ReadOnly
                        && self.source.contains("VKFFT_RADER_PERM"));
            let mixed_external = supported_mixed_pair && caller_visible_role;
            if binding.set != 0
                || (binding.scalar != self.scalar && !mixed_external)
                || !seen.insert(binding.binding)
            {
                return Err(VkFftError::InvalidKernelIr(
                    "native source bindings must use unique set-0 slots and compute-compatible storage scalars",
                ));
            }
        }
        for forbidden in [
            "#version",
            "layout(set",
            "layout(local_size",
            "gl_LocalInvocationID",
            "gl_WorkGroupID",
            "vkfft_input.data",
            "vkfft_output.data",
            "vkfft_lut.data",
            "vkfft_four_step_roots.data",
            "vkfft_twiddle_lut.data",
            "vkfft_twiddles.data",
            "vkfft_aux.data",
        ] {
            if self.source.contains(forbidden) {
                return Err(VkFftError::ShaderCompilation(format!(
                    "native source still contains Vulkan GLSL token `{forbidden}`"
                )));
            }
        }
        if !self.source.contains("VkFFT_main") {
            return Err(VkFftError::ShaderCompilation(
                "native source is missing VkFFT_main".to_owned(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NativeSourceBackend {
    backend: Backend,
}

impl NativeSourceBackend {
    pub const fn new(backend: Backend) -> Self {
        Self { backend }
    }

    pub const fn backend(self) -> Backend {
        self.backend
    }

    pub fn lower_kernel(&self, kernel: &KernelIr) -> Result<NativeShaderSource> {
        let source = VulkanGlslBackend.lower(kernel)?;
        self.translate(source)
    }

    pub fn lower_one_dim_fft(&self, ir: &crate::OneDimFftIr) -> Result<Vec<NativeShaderSource>> {
        let program = ProgramIr::one_dim_fft(ir)?;
        let shaders = VulkanGlslBackend.lower_one_dim_fft(ir)?;
        if shaders.len() != program.passes.len() {
            return Err(VkFftError::InvalidKernelIr(
                "one-dimensional native source/pass count mismatch",
            ));
        }
        shaders
            .into_iter()
            .zip(&program.passes)
            .map(|(shader, pass)| self.translate_program_pass(shader, &program, pass))
            .collect()
    }

    pub fn lower_recursive_fft(
        &self,
        ir: &crate::RecursiveFftIr,
    ) -> Result<Vec<NativeShaderSource>> {
        self.translate_many(VulkanGlslBackend.lower_recursive_fft(ir)?)
    }

    pub fn lower_nd_fft(&self, ir: &crate::NdFftIr) -> Result<Vec<NativeShaderSource>> {
        let program = ProgramIr::nd_fft(ir)?;
        let shaders = VulkanGlslBackend.lower_nd_fft(ir)?;
        if shaders.len() != program.passes.len() {
            return Err(VkFftError::InvalidKernelIr(
                "multidimensional native source/pass count mismatch",
            ));
        }
        shaders
            .into_iter()
            .zip(&program.passes)
            .map(|(shader, pass)| self.translate_program_pass(shader, &program, pass))
            .collect()
    }

    pub fn lower_nd_real_convolution(
        &self,
        ir: &NdRealConvolutionIr,
    ) -> Result<NativeProgramSource> {
        if !matches!(
            self.backend,
            Backend::Cuda | Backend::Hip | Backend::OpenCl | Backend::LevelZero
        ) {
            return Err(VkFftError::UnsupportedKernelPath(
                "multidimensional real performConvolution native lowering requires CUDA/HIP/OpenCL/Level Zero",
            ));
        }
        let program = ProgramIr::nd_real_convolution(ir)?;
        let shaders = VulkanGlslBackend.lower_nd_real_convolution(ir)?;
        if shaders.len() != program.passes.len() {
            return Err(VkFftError::InvalidKernelIr(
                "multidimensional real performConvolution native source/pass count mismatch",
            ));
        }
        let shaders = shaders
            .into_iter()
            .zip(&program.passes)
            .map(|(shader, pass)| self.translate_program_pass(shader, &program, pass))
            .collect::<Result<Vec<_>>>()?;
        let output = NativeProgramSource {
            backend: self.backend,
            program,
            shaders,
        };
        output.validate()?;
        Ok(output)
    }

    pub fn lower_real_fft(&self, ir: &crate::RealFftIr) -> Result<Vec<NativeShaderSource>> {
        let program = ProgramIr::real_fft(ir)?;
        let shaders = VulkanGlslBackend.lower_real_fft(ir)?;
        if shaders.len() != program.passes.len() {
            return Err(VkFftError::InvalidKernelIr(
                "real native source/pass count mismatch",
            ));
        }
        shaders
            .into_iter()
            .zip(&program.passes)
            .map(|(shader, pass)| self.translate_program_pass(shader, &program, pass))
            .collect()
    }

    pub fn lower_nd_real_fft(&self, ir: &crate::NdRealFftIr) -> Result<Vec<NativeShaderSource>> {
        let program = ProgramIr::nd_real_fft(ir)?;
        let shaders = VulkanGlslBackend.lower_nd_real_fft(ir)?;
        if shaders.len() != program.passes.len() {
            return Err(VkFftError::InvalidKernelIr(
                "multidimensional real native source/pass count mismatch",
            ));
        }
        shaders
            .into_iter()
            .zip(&program.passes)
            .map(|(shader, pass)| self.translate_program_pass(shader, &program, pass))
            .collect()
    }

    pub fn lower_r2r(&self, ir: &crate::R2rIr) -> Result<NativeShaderSource> {
        self.translate(VulkanGlslBackend.lower_r2r(ir)?)
    }

    pub fn lower_r2r_program(&self, ir: &crate::R2rIr) -> Result<Vec<NativeShaderSource>> {
        let program = ProgramIr::r2r(ir)?;
        let shaders = VulkanGlslBackend.lower_r2r_program(ir)?;
        if shaders.len() != program.passes.len() {
            return Err(VkFftError::InvalidKernelIr(
                "R2R native source/pass count mismatch",
            ));
        }
        shaders
            .into_iter()
            .zip(&program.passes)
            .map(|(shader, pass)| self.translate_program_pass(shader, &program, pass))
            .collect()
    }

    pub fn lower_nd_r2r(&self, ir: &crate::NdR2rIr) -> Result<Vec<NativeShaderSource>> {
        let program = ProgramIr::nd_r2r(ir)?;
        let shaders = VulkanGlslBackend.lower_nd_r2r(ir)?;
        if shaders.len() != program.passes.len() {
            return Err(VkFftError::InvalidKernelIr(
                "multidimensional R2R native source/pass count mismatch",
            ));
        }
        shaders
            .into_iter()
            .zip(&program.passes)
            .map(|(shader, pass)| self.translate_program_pass(shader, &program, pass))
            .collect()
    }

    pub fn lower_rader_direct(&self, ir: &crate::RaderDirectIr) -> Result<NativeShaderSource> {
        self.translate(VulkanGlslBackend.lower_rader_direct(ir)?)
    }

    pub fn lower_double_double_direct_rader(
        &self,
        ir: &crate::DoubleDoubleDirectRaderIr,
    ) -> Result<NativeProgramSource> {
        if !matches!(
            self.backend,
            Backend::Cuda | Backend::Hip | Backend::OpenCl | Backend::LevelZero
        ) {
            return Err(VkFftError::UnsupportedPrecision {
                backend: "native double-double direct Rader source lowering",
                precision: "DD direct-Rader source lowering requires CUDA/HIP/OpenCL/Level Zero",
            });
        }
        let program = ProgramIr::double_double_direct_rader(ir)?;
        let shaders = VulkanGlslBackend.lower_double_double_direct_rader_program(ir)?;
        if shaders.len() != program.passes.len() {
            return Err(VkFftError::InvalidKernelIr(
                "double-double direct-Rader native source/pass count mismatch",
            ));
        }
        let shaders = shaders
            .into_iter()
            .zip(&program.passes)
            .map(|(shader, pass)| self.translate_program_pass(shader, &program, pass))
            .collect::<Result<Vec<_>>>()?;
        let output = NativeProgramSource {
            backend: self.backend,
            program,
            shaders,
        };
        output.validate()?;
        Ok(output)
    }

    pub fn lower_double_double_fft_rader(
        &self,
        ir: &crate::DoubleDoubleFftRaderIr,
    ) -> Result<NativeProgramSource> {
        if !matches!(
            self.backend,
            Backend::Cuda | Backend::Hip | Backend::OpenCl | Backend::LevelZero
        ) {
            return Err(VkFftError::UnsupportedPrecision {
                backend: "native double-double FFT Rader source lowering",
                precision: "DD FFT-Rader source lowering requires CUDA/HIP/OpenCL/Level Zero",
            });
        }
        let program = ProgramIr::double_double_fft_rader(ir)?;
        let shaders = VulkanGlslBackend.lower_double_double_fft_rader(ir)?;
        if shaders.len() != program.passes.len() {
            return Err(VkFftError::InvalidKernelIr(
                "double-double FFT-Rader native source/pass count mismatch",
            ));
        }
        let shaders = shaders
            .into_iter()
            .zip(&program.passes)
            .map(|(shader, pass)| self.translate_program_pass(shader, &program, pass))
            .collect::<Result<Vec<_>>>()?;
        let output = NativeProgramSource {
            backend: self.backend,
            program,
            shaders,
        };
        output.validate()?;
        Ok(output)
    }

    pub fn lower_double_double_bluestein(
        &self,
        ir: &crate::DoubleDoubleBluesteinIr,
    ) -> Result<NativeProgramSource> {
        if !matches!(
            self.backend,
            Backend::Cuda | Backend::Hip | Backend::OpenCl | Backend::LevelZero
        ) {
            return Err(VkFftError::UnsupportedPrecision {
                backend: "native double-double Bluestein source lowering",
                precision: "DD Bluestein source lowering requires CUDA/HIP/OpenCL/Level Zero",
            });
        }
        let program = ProgramIr::double_double_bluestein(ir)?;
        let shaders = VulkanGlslBackend.lower_double_double_bluestein(ir)?;
        if shaders.len() != program.passes.len() {
            return Err(VkFftError::InvalidKernelIr(
                "double-double Bluestein native source/pass count mismatch",
            ));
        }
        let shaders = shaders
            .into_iter()
            .zip(&program.passes)
            .map(|(shader, pass)| self.translate_program_pass(shader, &program, pass))
            .collect::<Result<Vec<_>>>()?;
        let output = NativeProgramSource {
            backend: self.backend,
            program,
            shaders,
        };
        output.validate()?;
        Ok(output)
    }

    pub fn lower_double_double_nd(
        &self,
        ir: &crate::DoubleDoubleNdFftIr,
    ) -> Result<NativeProgramSource> {
        if !matches!(
            self.backend,
            Backend::Cuda | Backend::Hip | Backend::OpenCl | Backend::LevelZero
        ) {
            return Err(VkFftError::UnsupportedPrecision {
                backend: "native double-double ND source lowering",
                precision: "DD ND source lowering requires CUDA/HIP/OpenCL/Level Zero",
            });
        }
        let program = ProgramIr::double_double_nd(ir)?;
        let shaders = VulkanGlslBackend.lower_double_double_nd(ir)?;
        if shaders.len() != program.passes.len() {
            return Err(VkFftError::InvalidKernelIr(
                "double-double ND native source/pass count mismatch",
            ));
        }
        let shaders = shaders
            .into_iter()
            .zip(&program.passes)
            .map(|(shader, pass)| self.translate_program_pass(shader, &program, pass))
            .collect::<Result<Vec<_>>>()?;
        let output = NativeProgramSource {
            backend: self.backend,
            program,
            shaders,
        };
        output.validate()?;
        Ok(output)
    }

    pub fn lower_double_double_nd_real(
        &self,
        ir: &crate::DoubleDoubleNdRealFftIr,
    ) -> Result<NativeProgramSource> {
        if !matches!(
            self.backend,
            Backend::Cuda | Backend::Hip | Backend::OpenCl | Backend::LevelZero
        ) {
            return Err(VkFftError::UnsupportedPrecision {
                backend: "native double-double ND real source lowering",
                precision: "DD ND real source lowering requires CUDA/HIP/OpenCL/Level Zero",
            });
        }
        let program = ProgramIr::double_double_nd_real(ir)?;
        let shaders = VulkanGlslBackend.lower_double_double_nd_real(ir)?;
        if shaders.len() != program.passes.len() {
            return Err(VkFftError::InvalidKernelIr(
                "double-double ND real native source/pass count mismatch",
            ));
        }
        let shaders = shaders
            .into_iter()
            .zip(&program.passes)
            .map(|(shader, pass)| self.translate_program_pass(shader, &program, pass))
            .collect::<Result<Vec<_>>>()?;
        let output = NativeProgramSource {
            backend: self.backend,
            program,
            shaders,
        };
        output.validate()?;
        Ok(output)
    }

    pub fn lower_double_double_nd_r2r(
        &self,
        ir: &crate::DoubleDoubleNdR2rIr,
    ) -> Result<NativeProgramSource> {
        if !matches!(
            self.backend,
            Backend::Cuda | Backend::Hip | Backend::OpenCl | Backend::LevelZero
        ) {
            return Err(VkFftError::UnsupportedPrecision {
                backend: "native double-double ND R2R source lowering",
                precision: "DD ND R2R source lowering requires CUDA/HIP/OpenCL/Level Zero",
            });
        }
        let program = ProgramIr::double_double_nd_r2r(ir)?;
        let shaders = VulkanGlslBackend.lower_double_double_nd_r2r(ir)?;
        if shaders.len() != program.passes.len() {
            return Err(VkFftError::InvalidKernelIr(
                "double-double ND R2R native source/pass count mismatch",
            ));
        }
        let shaders = shaders
            .into_iter()
            .zip(&program.passes)
            .map(|(shader, pass)| self.translate_program_pass(shader, &program, pass))
            .collect::<Result<Vec<_>>>()?;
        let output = NativeProgramSource {
            backend: self.backend,
            program,
            shaders,
        };
        output.validate()?;
        Ok(output)
    }

    pub fn lower_double_double_real(
        &self,
        ir: &crate::DoubleDoubleRealFftIr,
    ) -> Result<NativeProgramSource> {
        if !matches!(
            self.backend,
            Backend::Cuda | Backend::Hip | Backend::OpenCl | Backend::LevelZero
        ) {
            return Err(VkFftError::UnsupportedPrecision {
                backend: "native double-double real source lowering",
                precision: "DD real source lowering requires CUDA/HIP/OpenCL/Level Zero",
            });
        }
        let program = ProgramIr::double_double_real(ir)?;
        let shaders = VulkanGlslBackend.lower_double_double_real(ir)?;
        if shaders.len() != program.passes.len() {
            return Err(VkFftError::InvalidKernelIr(
                "double-double real native source/pass count mismatch",
            ));
        }
        let shaders = shaders
            .into_iter()
            .zip(&program.passes)
            .map(|(shader, pass)| self.translate_program_pass(shader, &program, pass))
            .collect::<Result<Vec<_>>>()?;
        let output = NativeProgramSource {
            backend: self.backend,
            program,
            shaders,
        };
        output.validate()?;
        Ok(output)
    }

    pub fn lower_double_double_r2r(
        &self,
        ir: &crate::DoubleDoubleR2rIr,
    ) -> Result<NativeProgramSource> {
        if !matches!(
            self.backend,
            Backend::Cuda | Backend::Hip | Backend::OpenCl | Backend::LevelZero
        ) {
            return Err(VkFftError::UnsupportedPrecision {
                backend: "native double-double R2R source lowering",
                precision: "DD R2R source lowering requires CUDA/HIP/OpenCL/Level Zero",
            });
        }
        let program = ProgramIr::double_double_r2r(ir)?;
        let shaders = VulkanGlslBackend.lower_double_double_r2r(ir)?;
        if shaders.len() != program.passes.len() {
            return Err(VkFftError::InvalidKernelIr(
                "double-double R2R native source/pass count mismatch",
            ));
        }
        let shaders = shaders
            .into_iter()
            .zip(&program.passes)
            .map(|(shader, pass)| self.translate_program_pass(shader, &program, pass))
            .collect::<Result<Vec<_>>>()?;
        let output = NativeProgramSource {
            backend: self.backend,
            program,
            shaders,
        };
        output.validate()?;
        Ok(output)
    }

    pub fn lower_double_double_recursive(
        &self,
        ir: &crate::DoubleDoubleRecursiveFftIr,
    ) -> Result<NativeProgramSource> {
        if !matches!(
            self.backend,
            Backend::Cuda | Backend::Hip | Backend::OpenCl | Backend::LevelZero
        ) {
            return Err(VkFftError::UnsupportedPrecision {
                backend: "native double-double recursive source lowering",
                precision: "recursive double-double source lowering requires CUDA/HIP/OpenCL/Level Zero",
            });
        }
        let program = ProgramIr::double_double_recursive(ir)?;
        let shaders = VulkanGlslBackend.lower_double_double_recursive(ir)?;
        if shaders.len() != program.passes.len() {
            return Err(VkFftError::InvalidKernelIr(
                "double-double recursive native source/pass count mismatch",
            ));
        }
        let shaders = shaders
            .into_iter()
            .zip(&program.passes)
            .map(|(shader, pass)| self.translate_program_pass(shader, &program, pass))
            .collect::<Result<Vec<_>>>()?;
        let output = NativeProgramSource {
            backend: self.backend,
            program,
            shaders,
        };
        output.validate()?;
        Ok(output)
    }

    pub fn lower_double_double_stockham(
        &self,
        ir: &crate::DoubleDoubleStockhamIr,
    ) -> Result<NativeProgramSource> {
        if !matches!(
            self.backend,
            Backend::Cuda | Backend::Hip | Backend::OpenCl | Backend::LevelZero
        ) {
            return Err(VkFftError::UnsupportedPrecision {
                backend: "native double-double source lowering",
                precision: "DD source lowering requires CUDA/HIP/OpenCL/Level Zero",
            });
        }
        let program = ProgramIr::double_double_stockham(ir)?;
        let shaders = VulkanGlslBackend.lower_double_double_stockham_program(ir)?;
        if shaders.len() != program.passes.len() {
            return Err(VkFftError::InvalidKernelIr(
                "double-double Stockham native source/pass count mismatch",
            ));
        }
        let shaders = shaders
            .into_iter()
            .zip(&program.passes)
            .map(|(shader, pass)| self.translate_program_pass(shader, &program, pass))
            .collect::<Result<Vec<_>>>()?;
        let output = NativeProgramSource {
            backend: self.backend,
            program,
            shaders,
        };
        output.validate()?;
        Ok(output)
    }

    pub fn lower_convolution(&self, ir: &ConvolutionIr) -> Result<NativeProgramSource> {
        if !matches!(
            self.backend,
            Backend::Cuda | Backend::Hip | Backend::OpenCl | Backend::LevelZero
        ) {
            return Err(VkFftError::UnsupportedKernelPath(
                "performConvolution native lowering requires CUDA/HIP/OpenCL/Level Zero",
            ));
        }
        let program = ProgramIr::convolution(ir)?;
        let shaders = VulkanGlslBackend.lower_convolution(ir)?;
        if shaders.len() != program.passes.len() {
            return Err(VkFftError::InvalidKernelIr(
                "performConvolution native source/pass count mismatch",
            ));
        }
        let shaders = shaders
            .into_iter()
            .zip(&program.passes)
            .map(|(shader, pass)| self.translate_program_pass(shader, &program, pass))
            .collect::<Result<Vec<_>>>()?;
        let output = NativeProgramSource {
            backend: self.backend,
            program,
            shaders,
        };
        output.validate()?;
        Ok(output)
    }

    pub fn lower_nd_convolution(&self, ir: &NdConvolutionIr) -> Result<NativeProgramSource> {
        if !matches!(
            self.backend,
            Backend::Cuda | Backend::Hip | Backend::OpenCl | Backend::LevelZero
        ) {
            return Err(VkFftError::UnsupportedKernelPath(
                "multidimensional performConvolution native lowering requires CUDA/HIP/OpenCL/Level Zero",
            ));
        }
        let program = ProgramIr::nd_convolution(ir)?;
        let shaders = VulkanGlslBackend.lower_nd_convolution(ir)?;
        if shaders.len() != program.passes.len() {
            return Err(VkFftError::InvalidKernelIr(
                "multidimensional performConvolution native source/pass count mismatch",
            ));
        }
        let shaders = shaders
            .into_iter()
            .zip(&program.passes)
            .map(|(shader, pass)| self.translate_program_pass(shader, &program, pass))
            .collect::<Result<Vec<_>>>()?;
        let output = NativeProgramSource {
            backend: self.backend,
            program,
            shaders,
        };
        output.validate()?;
        Ok(output)
    }

    pub fn lower_transform(&self, ir: &TransformIr) -> Result<NativeProgramSource> {
        let (program, shaders) = match ir {
            TransformIr::Complex1d(ir) => {
                (ProgramIr::one_dim_fft(ir)?, self.lower_one_dim_fft(ir)?)
            }
            TransformIr::Complex1dDoubleDouble(ir) => match ir {
                crate::DoubleDoubleOneDimIr::Stockham(stockham) => {
                    return self.lower_double_double_stockham(stockham);
                }
                crate::DoubleDoubleOneDimIr::DirectRader(rader) => {
                    return self.lower_double_double_direct_rader(rader);
                }
                crate::DoubleDoubleOneDimIr::FftRader(rader) => {
                    return self.lower_double_double_fft_rader(rader);
                }
                crate::DoubleDoubleOneDimIr::Bluestein(bluestein) => {
                    return self.lower_double_double_bluestein(bluestein);
                }
                crate::DoubleDoubleOneDimIr::Recursive(recursive) => {
                    return self.lower_double_double_recursive(recursive);
                }
            },
            TransformIr::ComplexNdDoubleDouble(nd) => {
                return self.lower_double_double_nd(nd);
            }
            TransformIr::ComplexNd(ir) => (ProgramIr::nd_fft(ir)?, self.lower_nd_fft(ir)?),
            TransformIr::RealDoubleDouble(real) => {
                return self.lower_double_double_real(real);
            }
            TransformIr::RealNdDoubleDouble(real) => {
                return self.lower_double_double_nd_real(real);
            }
            TransformIr::Real(ir) => (ProgramIr::real_fft(ir)?, self.lower_real_fft(ir)?),
            TransformIr::RealNd(ir) => (ProgramIr::nd_real_fft(ir)?, self.lower_nd_real_fft(ir)?),
            TransformIr::RealToRealDoubleDouble(ir) => {
                return self.lower_double_double_r2r(ir);
            }
            TransformIr::RealToRealNdDoubleDouble(ir) => {
                return self.lower_double_double_nd_r2r(ir);
            }
            TransformIr::RealToReal(ir) => (ProgramIr::r2r(ir)?, self.lower_r2r_program(ir)?),
            TransformIr::RealToRealNd(ir) => (ProgramIr::nd_r2r(ir)?, self.lower_nd_r2r(ir)?),
        };
        let output = NativeProgramSource {
            backend: self.backend,
            program,
            shaders,
        };
        output.validate()?;
        Ok(output)
    }

    fn translate_program_pass(
        &self,
        shader: VulkanShaderSource,
        program: &ProgramIr,
        pass: &ProgramPass,
    ) -> Result<NativeShaderSource> {
        let mut bindings = Vec::with_capacity(pass.bindings.len());
        let mut binding_shapes = Vec::with_capacity(pass.bindings.len());
        for expected in &pass.bindings {
            let descriptor = shader
                .descriptors
                .iter()
                .find(|descriptor| descriptor.binding == expected.binding)
                .ok_or(VkFftError::InvalidKernelIr(
                    "native source is missing a ProgramIr descriptor",
                ))?;
            let resource = program.resource(expected.resource)?;
            bindings.push(BufferBinding {
                set: descriptor.set,
                binding: descriptor.binding,
                role: descriptor.role,
                access: descriptor.access,
                scalar: resource.scalar,
            });
            binding_shapes.push(resource.element_shape());
        }
        self.translate_with_bindings(shader, bindings, binding_shapes)
    }

    fn translate_many(&self, shaders: Vec<VulkanShaderSource>) -> Result<Vec<NativeShaderSource>> {
        shaders
            .into_iter()
            .map(|shader| self.translate(shader))
            .collect()
    }

    fn translate(&self, shader: VulkanShaderSource) -> Result<NativeShaderSource> {
        let bindings = shader
            .descriptors
            .iter()
            .map(|descriptor| descriptor_binding(*descriptor, shader.scalar))
            .collect::<Vec<_>>();
        let binding_shapes = vec![ProgramElementShape::Complex; bindings.len()];
        self.translate_with_bindings(shader, bindings, binding_shapes)
    }

    fn translate_with_bindings(
        &self,
        shader: VulkanShaderSource,
        bindings: Vec<BufferBinding>,
        binding_shapes: Vec<ProgramElementShape>,
    ) -> Result<NativeShaderSource> {
        if binding_shapes.len() != bindings.len() {
            return Err(VkFftError::InvalidKernelIr(
                "native binding element-shape metadata does not match binding count",
            ));
        }
        if matches!(self.backend, Backend::Vulkan | Backend::CpuReference) {
            return Err(VkFftError::UnsupportedKernelPath(
                "native source lowering requires a non-Vulkan GPU backend",
            ));
        }
        if shader.required_subgroup_size.is_some()
            && !matches!(self.backend, Backend::OpenCl | Backend::LevelZero)
        {
            return Err(VkFftError::UnsupportedKernelPath(
                "native source lowering does not translate Vulkan subgroup-marker kernels for this backend; build the IR with the target backend profile so shared/wave fallbacks are selected",
            ));
        }
        if self.backend == Backend::Metal
            && matches!(shader.scalar, ScalarType::F64 | ScalarType::DoubleDouble)
        {
            return Err(VkFftError::UnsupportedPrecision {
                backend: "Metal source backend",
                precision: if shader.scalar == ScalarType::DoubleDouble {
                    "double-double"
                } else {
                    "f64"
                },
            });
        }
        let compiler_fallback_source = if self.backend == Backend::Hip {
            Some(translate_hip_opencl_fallback(
                &shader.glsl,
                shader.scalar,
                shader.workgroup_size,
                &bindings,
                &binding_shapes,
            )?)
        } else {
            None
        };
        let source = translate_glsl_dialect_with_required_subgroup(
            self.backend,
            &shader.glsl,
            shader.scalar,
            shader.workgroup_size,
            &bindings,
            &binding_shapes,
            shader.required_subgroup_size,
        )?;
        let output = NativeShaderSource {
            backend: self.backend,
            entry_point: "VkFFT_main",
            source,
            compiler_fallback_source,
            scalar: shader.scalar,
            sequence_len: shader.sequence_len,
            batch_count: shader.batch_count,
            workgroup_size: shader.workgroup_size,
            dispatch: shader.dispatch,
            bindings,
            required_shared_memory_bytes: shader.required_shared_memory_bytes,
        };
        output.validate()?;
        Ok(output)
    }
}

impl KernelBackend for NativeSourceBackend {
    type Output = NativeShaderSource;

    fn lower(&self, kernel: &KernelIr) -> Result<Self::Output> {
        self.lower_kernel(kernel)
    }
}

fn validate_program_pass_shader(
    backend: Backend,
    program: &ProgramIr,
    pass: &ProgramPass,
    shader: &NativeShaderSource,
) -> Result<()> {
    shader.validate()?;
    let storage_only_f64_in_dd = program.scalar == ScalarType::DoubleDouble
        && shader.scalar == ScalarType::F64
        && pass.bindings.iter().all(|binding| {
            program
                .resource(binding.resource)
                .is_ok_and(|resource| resource.scalar == ScalarType::F64)
        });
    if shader.backend != backend
        || (shader.scalar != program.scalar && !storage_only_f64_in_dd)
        || shader.dispatch != pass.dispatch
    {
        return Err(VkFftError::InvalidKernelIr(
            "native shader metadata does not match its ProgramIr pass",
        ));
    }
    if shader.bindings.len() != pass.bindings.len() {
        return Err(VkFftError::InvalidKernelIr(
            "native shader binding count does not match its ProgramIr pass",
        ));
    }
    for expected in &pass.bindings {
        let actual = shader
            .bindings
            .iter()
            .find(|binding| binding.binding == expected.binding)
            .ok_or(VkFftError::InvalidKernelIr(
                "native shader is missing a ProgramIr binding",
            ))?;
        let resource = program.resource(expected.resource)?;
        if actual.set != 0
            || actual.role != expected.role
            || actual.access != expected.access
            || actual.scalar != resource.scalar
        {
            return Err(VkFftError::InvalidKernelIr(
                "native shader binding metadata does not match its ProgramIr pass",
            ));
        }
    }
    Ok(())
}

fn descriptor_binding(descriptor: VulkanDescriptorBinding, scalar: ScalarType) -> BufferBinding {
    BufferBinding {
        set: descriptor.set,
        binding: descriptor.binding,
        role: descriptor.role,
        access: descriptor.access,
        scalar,
    }
}

const SUBGROUP_WIDTH_MARKER_PREFIX: &str = "// vkfft-rs subgroup width: ";

fn declared_subgroup_width(glsl: &str) -> Result<Option<usize>> {
    let mut found = None;
    for line in glsl.lines() {
        let Some(raw) = line.trim().strip_prefix(SUBGROUP_WIDTH_MARKER_PREFIX) else {
            continue;
        };
        let width = raw.parse::<usize>().map_err(|_| {
            VkFftError::ShaderCompilation("invalid subgroup-width semantic marker".to_owned())
        })?;
        if width == 0 || !width.is_power_of_two() {
            return Err(VkFftError::ShaderCompilation(
                "subgroup-width semantic marker must be a non-zero power of two".to_owned(),
            ));
        }
        if found.is_some_and(|previous| previous != width) {
            return Err(VkFftError::ShaderCompilation(
                "native source contains conflicting subgroup-width semantic markers".to_owned(),
            ));
        }
        found = Some(width);
    }
    Ok(found)
}

fn native_subgroup_width(backend: Backend, declared: Option<usize>) -> Result<usize> {
    let width = declared.unwrap_or(match backend {
        Backend::Cuda => 32,
        Backend::Hip => 64,
        Backend::OpenCl => 1,
        Backend::LevelZero | Backend::Metal | Backend::Vulkan | Backend::CpuReference => 1,
    });
    match backend {
        Backend::Cuda if width != 32 => Err(VkFftError::UnsupportedKernelPath(
            "CUDA subgroup semantic marker must remain warp32",
        )),
        Backend::Hip if !matches!(width, 32 | 64) => Err(VkFftError::UnsupportedKernelPath(
            "HIP subgroup semantic marker must be wave32 or wave64",
        )),
        _ => Ok(width),
    }
}

#[cfg(test)]
fn translate_glsl_dialect(
    backend: Backend,
    glsl: &str,
    scalar: ScalarType,
    workgroup: WorkgroupSize,
    bindings: &[BufferBinding],
    binding_shapes: &[ProgramElementShape],
) -> Result<String> {
    translate_glsl_dialect_with_required_subgroup(
        backend,
        glsl,
        scalar,
        workgroup,
        bindings,
        binding_shapes,
        None,
    )
}

fn translate_glsl_dialect_with_required_subgroup(
    backend: Backend,
    glsl: &str,
    scalar: ScalarType,
    workgroup: WorkgroupSize,
    bindings: &[BufferBinding],
    binding_shapes: &[ProgramElementShape],
    required_subgroup_size: Option<u32>,
) -> Result<String> {
    translate_glsl_dialect_impl(
        backend,
        glsl,
        scalar,
        workgroup,
        bindings,
        binding_shapes,
        false,
        required_subgroup_size,
    )
}

fn translate_hip_opencl_fallback(
    glsl: &str,
    scalar: ScalarType,
    workgroup: WorkgroupSize,
    bindings: &[BufferBinding],
    binding_shapes: &[ProgramElementShape],
) -> Result<String> {
    translate_glsl_dialect_impl(
        Backend::OpenCl,
        glsl,
        scalar,
        workgroup,
        bindings,
        binding_shapes,
        true,
        None,
    )
}

fn translate_glsl_dialect_impl(
    backend: Backend,
    glsl: &str,
    scalar: ScalarType,
    workgroup: WorkgroupSize,
    bindings: &[BufferBinding],
    binding_shapes: &[ProgramElementShape],
    amd_opencl_hip_fallback: bool,
    required_subgroup_size: Option<u32>,
) -> Result<String> {
    let mut output = String::with_capacity(glsl.len() + 4096);
    output.push_str(&dialect_prelude(backend, scalar)?);
    if matches!(backend, Backend::OpenCl | Backend::LevelZero) && required_subgroup_size.is_some() {
        output.push_str("#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable\n");
    }
    output
        .push_str("// vkfft-rs native source translated from the typed Vulkan semantic frontend\n");
    let subgroup_semantics_backend = if amd_opencl_hip_fallback {
        Backend::Hip
    } else {
        backend
    };
    let subgroup_width =
        native_subgroup_width(subgroup_semantics_backend, declared_subgroup_width(glsl)?)?;

    let mut shared = Vec::<String>::new();
    let mut in_rader_initializer = false;
    let mut skipping_subgroup_marker_body = false;
    let buffer_replacements = glsl
        .lines()
        .filter_map(parse_glsl_buffer_binding_and_instance)
        .filter_map(|(binding_number, instance)| {
            bindings
                .iter()
                .copied()
                .find(|binding| binding.binding == binding_number)
                .map(|binding| {
                    (
                        format!("{instance}.data"),
                        binding_argument_name(bindings, binding),
                    )
                })
        })
        .collect::<Vec<_>>();
    for raw_line in glsl.lines() {
        let trimmed = raw_line.trim();
        if skipping_subgroup_marker_body {
            if trimmed == "}" {
                skipping_subgroup_marker_body = false;
            }
            continue;
        }
        if trimmed.starts_with("vec2 vkfft_subgroup_shuffle_marker(")
            || trimmed.starts_with("dvec2 vkfft_subgroup_shuffle_marker(")
        {
            let helper = if amd_opencl_hip_fallback {
                amd_opencl_native_subgroup_shuffle_helper(scalar, subgroup_width)
            } else {
                native_subgroup_shuffle_helper(backend, scalar, subgroup_width)
            }?;
            output.push_str(&helper);
            skipping_subgroup_marker_body = true;
            continue;
        }
        if trimmed.starts_with("uint vkfft_subgroup_invocation_id_marker(") {
            let helper = if amd_opencl_hip_fallback {
                amd_opencl_native_subgroup_invocation_helper(subgroup_width)
            } else {
                native_subgroup_invocation_helper(backend, subgroup_width)
            }?;
            output.push_str(&helper);
            skipping_subgroup_marker_body = !trimmed.ends_with('}');
            continue;
        }
        if trimmed.starts_with("uint vkfft_subgroup_id_marker(") {
            let helper = if amd_opencl_hip_fallback {
                amd_opencl_native_subgroup_id_helper(subgroup_width)
            } else {
                native_subgroup_id_helper(backend, subgroup_width)
            }?;
            output.push_str(&helper);
            skipping_subgroup_marker_body = !trimmed.ends_with('}');
            continue;
        }
        if trimmed.starts_with("#version")
            || trimmed.starts_with("#extension")
            || trimmed.starts_with("layout(local_size")
            || (trimmed.starts_with("layout(set") && trimmed.contains(" buffer "))
        {
            continue;
        }
        if trimmed.starts_with("shared ") {
            shared.push(translate_shared_decl(backend, trimmed)?);
            continue;
        }
        if trimmed.starts_with("const uint VKFFT_RADER_") && trimmed.contains("= uint[") {
            let declaration =
                trimmed
                    .split(" = uint[")
                    .next()
                    .ok_or(VkFftError::ShaderCompilation(
                        "failed to parse Rader permutation declaration".to_owned(),
                    ))?;
            output.push_str(rader_constant_prefix(backend));
            output.push_str(declaration.trim_start_matches("const "));
            output.push_str(" = {\n");
            in_rader_initializer = true;
            continue;
        }
        if matches!(backend, Backend::OpenCl | Backend::LevelZero)
            && raw_line == trimmed
            && trimmed.ends_with(';')
            && (trimmed.starts_with("const uint VKFFT_")
                || trimmed.starts_with("const int VKFFT_")
                || trimmed.starts_with("const float VKFFT_")
                || trimmed.starts_with("const double VKFFT_"))
        {
            output.push_str("__constant ");
            output.push_str(trimmed.trim_start_matches("const "));
            output.push('\n');
            continue;
        }
        if in_rader_initializer && trimmed == ");" {
            output.push_str("};\n\n");
            in_rader_initializer = false;
            continue;
        }
        if trimmed == "void main() {" {
            output.push_str(&kernel_signature(
                backend,
                workgroup,
                bindings,
                binding_shapes,
                required_subgroup_size,
            )?);
            output.push_str(" {\n");
            for declaration in &shared {
                output.push_str("    ");
                output.push_str(declaration);
                output.push('\n');
            }
            if !shared.is_empty() {
                output.push('\n');
            }
            continue;
        }

        let mut line = raw_line.to_owned();
        for (glsl_name, native_name) in &buffer_replacements {
            line = line.replace(glsl_name, native_name);
        }
        line = line.replace("vkfft_input.data", "inputs");
        line = line.replace("vkfft_output.data", "outputs");
        line = line.replace("vkfft_lut.data", "lookup_table");
        line = line.replace("vkfft_four_step_roots.data", "twiddle_lut");
        line = line.replace("vkfft_twiddle_lut.data", "twiddle_lut");
        line = line.replace("vkfft_twiddles.data", "twiddle_lut");
        line = line.replace("vkfft_aux.data", "auxiliary");
        line = line.replace("gl_LocalInvocationID.x", local_id_expr(backend));
        line = line.replace("gl_LocalInvocationID.y", local_id_y_expr(backend));
        line = line.replace("gl_WorkGroupID.x", group_id_expr(backend));
        line = line.replace("barrier();", barrier_expr(backend));
        if matches!(backend, Backend::Cuda | Backend::Hip) && is_device_helper_definition(&line) {
            let indent = line.len() - line.trim_start().len();
            line = format!(
                "{}__device__ __forceinline__ {}",
                " ".repeat(indent),
                line.trim_start()
            );
        }
        output.push_str(&line);
        output.push('\n');
    }
    if in_rader_initializer {
        return Err(VkFftError::ShaderCompilation(
            "unterminated Rader permutation initializer".to_owned(),
        ));
    }

    output = output.replace("unpackHalf2x16(", "vkfft_unpack_half2(");
    output = output.replace("packHalf2x16(", "vkfft_pack_half2(");
    // Canonical Vulkan convolution code uses GLSL vector `dot` + `inversesqrt`,
    // matching upstream's PfNorm/PfRsqrt pair. Native C-like backends do not all
    // expose GLSL's vector `dot`, while CUDA/HIP/OpenCL spell reciprocal sqrt `rsqrt`.
    // Keep the canonical semantic frontend unchanged and lower only the typed scalar
    // product and matrix-row norm expressions at the dialect boundary.
    output = output.replace(
        "dot(vkfft_convolution_product, vkfft_convolution_product)",
        "(vkfft_convolution_product.x * vkfft_convolution_product.x + vkfft_convolution_product.y * vkfft_convolution_product.y)",
    );
    output = output.replace(
        "dot(vkfft_matrix_sum, vkfft_matrix_sum)",
        "(vkfft_matrix_sum.x * vkfft_matrix_sum.x + vkfft_matrix_sum.y * vkfft_matrix_sum.y)",
    );
    for coordinate in 0..3 {
        let value = format!("vkfft_matrix_sum_{coordinate}");
        output = output.replace(
            &format!("dot({value}, {value})"),
            &format!("({value}.x * {value}.x + {value}.y * {value}.y)"),
        );
    }
    if matches!(
        backend,
        Backend::Cuda | Backend::Hip | Backend::OpenCl | Backend::LevelZero | Backend::Metal
    ) {
        output = output.replace("inversesqrt(", "rsqrt(");
    }

    if matches!(backend, Backend::Cuda | Backend::Hip) {
        output = output.replace("dvec4(", "vkfft_dv4(");
        output = output.replace("dvec2(", "vkfft_dv2(");
        output = output.replace("vec2(", "vkfft_v2(");
    } else if matches!(backend, Backend::OpenCl | Backend::LevelZero) {
        output = output.replace("as_float(", "VKFFT_BITCAST_F32(");
        output = output.replace("dvec4(", "(dvec4)(");
        output = output.replace("dvec2(", "(dvec2)(");
        output = output.replace("vec2(", "(vec2)(");
        output = output.replace("double(", "convert_double(");
        output = output.replace("float(", "convert_float(");
        output = output.replace("VKFFT_BITCAST_F32(", "as_float(");
        if scalar == ScalarType::F32 {
            output = suffix_opencl_f32_literals(&output);
        }
    }
    Ok(output)
}

/// GLSL decimal literals are F32 by default, while OpenCL C follows C and treats an
/// unsuffixed decimal literal as `double`. Preserve the typed F32 IR contract by adding
/// an `f` suffix to standalone decimal/exponent literals when lowering the F32 dialect.
fn suffix_opencl_f32_literals(source: &str) -> String {
    let bytes = source.as_bytes();
    let mut output = String::with_capacity(source.len() + source.len() / 32);
    let mut cursor = 0usize;
    let mut index = 0usize;
    while index < bytes.len() {
        if !bytes[index].is_ascii_digit()
            || (index > 0 && (bytes[index - 1].is_ascii_alphanumeric() || bytes[index - 1] == b'_'))
        {
            index += 1;
            continue;
        }
        let start = index;
        while index < bytes.len() && bytes[index].is_ascii_digit() {
            index += 1;
        }
        let mut floating = false;
        if index < bytes.len() && bytes[index] == b'.' {
            floating = true;
            index += 1;
            while index < bytes.len() && bytes[index].is_ascii_digit() {
                index += 1;
            }
        }
        if index < bytes.len() && matches!(bytes[index], b'e' | b'E') {
            floating = true;
            index += 1;
            if index < bytes.len() && matches!(bytes[index], b'+' | b'-') {
                index += 1;
            }
            while index < bytes.len() && bytes[index].is_ascii_digit() {
                index += 1;
            }
        }
        if !floating {
            continue;
        }
        let already_suffixed =
            index < bytes.len() && matches!(bytes[index], b'f' | b'F' | b'l' | b'L');
        output.push_str(&source[cursor..index]);
        if !already_suffixed {
            output.push('f');
        }
        cursor = index;
        if index == start {
            index += 1;
        }
    }
    output.push_str(&source[cursor..]);
    output
}

fn dialect_prelude(backend: Backend, scalar: ScalarType) -> Result<String> {
    match backend {
        Backend::Cuda | Backend::Hip => {
            let mut prelude = String::new();
            if backend == Backend::Hip {
                prelude.push_str("#include <hip/hip_runtime.h>\n#include <math.h>\n");
            }
            prelude.push_str(
                r#"using uint = unsigned int;
template <typename T> struct VkFFTComplex { T x; T y; };
using vec2 = VkFFTComplex<float>;
using dvec2 = VkFFTComplex<double>;
struct dvec4 { double x; double y; double z; double w; };
__host__ __device__ inline vec2 vkfft_v2(float v) { return vec2{v, v}; }
__host__ __device__ inline vec2 vkfft_v2(float x, float y) { return vec2{x, y}; }
__host__ __device__ inline dvec2 vkfft_dv2(double v) { return dvec2{v, v}; }
__host__ __device__ inline dvec2 vkfft_dv2(double x, double y) { return dvec2{x, y}; }
__host__ __device__ inline dvec4 vkfft_dv4(double v) { return dvec4{v, v, v, v}; }
__host__ __device__ inline dvec4 vkfft_dv4(double x, double y, double z, double w) { return dvec4{x, y, z, w}; }
template <typename T> __host__ __device__ inline VkFFTComplex<T>& operator+=(VkFFTComplex<T>& a, VkFFTComplex<T> b) { a.x += b.x; a.y += b.y; return a; }
template <typename T> __host__ __device__ inline VkFFTComplex<T>& operator-=(VkFFTComplex<T>& a, VkFFTComplex<T> b) { a.x -= b.x; a.y -= b.y; return a; }
template <typename T, typename U> __host__ __device__ inline VkFFTComplex<T>& operator*=(VkFFTComplex<T>& a, U b) { T s = static_cast<T>(b); a.x *= s; a.y *= s; return a; }
template <typename T> __host__ __device__ inline VkFFTComplex<T> operator+(VkFFTComplex<T> a, VkFFTComplex<T> b) { return VkFFTComplex<T>{a.x + b.x, a.y + b.y}; }
template <typename T> __host__ __device__ inline VkFFTComplex<T> operator-(VkFFTComplex<T> a, VkFFTComplex<T> b) { return VkFFTComplex<T>{a.x - b.x, a.y - b.y}; }
template <typename T> __host__ __device__ inline VkFFTComplex<T> operator-(VkFFTComplex<T> a) { return VkFFTComplex<T>{-a.x, -a.y}; }
template <typename T, typename U> __host__ __device__ inline VkFFTComplex<T> operator*(VkFFTComplex<T> a, U b) { T s = static_cast<T>(b); return VkFFTComplex<T>{a.x * s, a.y * s}; }
template <typename T, typename U> __host__ __device__ inline VkFFTComplex<T> operator*(U b, VkFFTComplex<T> a) { return a * b; }
template <typename T, typename U> __host__ __device__ inline VkFFTComplex<T> operator/(VkFFTComplex<T> a, U b) { T s = static_cast<T>(b); return VkFFTComplex<T>{a.x / s, a.y / s}; }
__device__ __forceinline__ float vkfft_half_to_float(unsigned short h) {
    uint sign = (uint)(h & 0x8000u) << 16;
    uint exponent = (h >> 10) & 0x1fu;
    uint mantissa = h & 0x03ffu;
    uint bits;
    if (exponent == 0u) {
        if (mantissa == 0u) bits = sign;
        else {
            int unbiased = -14;
            while ((mantissa & 0x0400u) == 0u) { mantissa <<= 1; --unbiased; }
            mantissa &= 0x03ffu;
            bits = sign | (uint)(unbiased + 127) << 23 | mantissa << 13;
        }
    } else if (exponent == 0x1fu) {
        bits = sign | 0x7f800000u | mantissa << 13;
    } else {
        bits = sign | (exponent - 15u + 127u) << 23 | mantissa << 13;
    }
    return __uint_as_float(bits);
}
__device__ __forceinline__ unsigned short vkfft_float_to_half(float value) {
    uint bits = __float_as_uint(value);
    uint sign = (bits >> 16) & 0x8000u;
    int exponent = (int)((bits >> 23) & 0xffu);
    uint mantissa = bits & 0x007fffffu;
    if (exponent == 0xff) {
        if (mantissa == 0u) return (unsigned short)(sign | 0x7c00u);
        uint payload = mantissa >> 13; if (payload == 0u) payload = 1u;
        return (unsigned short)(sign | 0x7c00u | payload);
    }
    int half_exponent = exponent - 127 + 15;
    if (half_exponent >= 0x1f) return (unsigned short)(sign | 0x7c00u);
    if (half_exponent <= 0) {
        if (half_exponent < -10) return (unsigned short)sign;
        mantissa |= 0x00800000u;
        uint shift = (uint)(14 - half_exponent);
        uint half_mantissa = mantissa >> shift;
        uint remainder = mantissa & ((1u << shift) - 1u);
        uint halfway = 1u << (shift - 1u);
        if (remainder > halfway || (remainder == halfway && (half_mantissa & 1u))) ++half_mantissa;
        return (unsigned short)(sign | half_mantissa);
    }
    uint half_mantissa = mantissa >> 13;
    uint remainder = mantissa & 0x1fffu;
    if (remainder > 0x1000u || (remainder == 0x1000u && (half_mantissa & 1u))) {
        ++half_mantissa;
        if (half_mantissa == 0x0400u) {
            half_mantissa = 0u; ++half_exponent;
            if (half_exponent >= 0x1f) return (unsigned short)(sign | 0x7c00u);
        }
    }
    return (unsigned short)(sign | (uint)half_exponent << 10 | half_mantissa);
}
__device__ __forceinline__ vec2 vkfft_unpack_half2(uint packed) { return vec2{vkfft_half_to_float((unsigned short)packed), vkfft_half_to_float((unsigned short)(packed >> 16))}; }
__device__ __forceinline__ uint vkfft_pack_half2(vec2 value) { return (uint)vkfft_float_to_half(value.x) | (uint)vkfft_float_to_half(value.y) << 16; }

"#,
            );
            Ok(prelude)
        }
        Backend::OpenCl | Backend::LevelZero => {
            let mut prelude = String::new();
            if matches!(scalar, ScalarType::F64 | ScalarType::DoubleDouble) {
                prelude.push_str("#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n");
                if scalar == ScalarType::DoubleDouble {
                    prelude.push_str("#pragma OPENCL FP_CONTRACT OFF\n");
                }
                prelude.push_str(
                    "typedef float2 vec2;\ntypedef double2 dvec2;\ntypedef double4 dvec4;\n\n",
                );
            } else {
                prelude.push_str(
                    "typedef float2 vec2;\ntypedef float2 dvec2;\ntypedef float4 dvec4;\n\n",
                );
            }
            prelude.push_str(r#"inline float vkfft_half_to_float(ushort h) {
    uint sign = ((uint)(h & 0x8000u)) << 16;
    uint exponent = ((uint)h >> 10) & 0x1fu;
    uint mantissa = (uint)h & 0x03ffu;
    uint bits;
    if (exponent == 0u) {
        if (mantissa == 0u) bits = sign;
        else {
            int unbiased = -14;
            while ((mantissa & 0x0400u) == 0u) { mantissa <<= 1; --unbiased; }
            mantissa &= 0x03ffu;
            bits = sign | (uint)(unbiased + 127) << 23 | mantissa << 13;
        }
    } else if (exponent == 0x1fu) bits = sign | 0x7f800000u | mantissa << 13;
    else bits = sign | (exponent - 15u + 127u) << 23 | mantissa << 13;
    return as_float(bits);
}
inline ushort vkfft_float_to_half(float value) {
    uint bits = as_uint(value);
    uint sign = (bits >> 16) & 0x8000u;
    int exponent = (int)((bits >> 23) & 0xffu);
    uint mantissa = bits & 0x007fffffu;
    if (exponent == 0xff) {
        if (mantissa == 0u) return (ushort)(sign | 0x7c00u);
        uint payload = mantissa >> 13; if (payload == 0u) payload = 1u;
        return (ushort)(sign | 0x7c00u | payload);
    }
    int half_exponent = exponent - 127 + 15;
    if (half_exponent >= 0x1f) return (ushort)(sign | 0x7c00u);
    if (half_exponent <= 0) {
        if (half_exponent < -10) return (ushort)sign;
        mantissa |= 0x00800000u;
        uint shift = (uint)(14 - half_exponent);
        uint half_mantissa = mantissa >> shift;
        uint remainder = mantissa & ((1u << shift) - 1u);
        uint halfway = 1u << (shift - 1u);
        if (remainder > halfway || (remainder == halfway && (half_mantissa & 1u))) ++half_mantissa;
        return (ushort)(sign | half_mantissa);
    }
    uint half_mantissa = mantissa >> 13;
    uint remainder = mantissa & 0x1fffu;
    if (remainder > 0x1000u || (remainder == 0x1000u && (half_mantissa & 1u))) {
        ++half_mantissa;
        if (half_mantissa == 0x0400u) {
            half_mantissa = 0u; ++half_exponent;
            if (half_exponent >= 0x1f) return (ushort)(sign | 0x7c00u);
        }
    }
    return (ushort)(sign | (uint)half_exponent << 10 | half_mantissa);
}
inline vec2 vkfft_unpack_half2(uint packed) { return (vec2)(vkfft_half_to_float((ushort)packed), vkfft_half_to_float((ushort)(packed >> 16))); }
inline uint vkfft_pack_half2(vec2 value) { return (uint)vkfft_float_to_half(value.x) | (uint)vkfft_float_to_half(value.y) << 16; }

"#);
            Ok(prelude)
        }
        Backend::Metal => {
            if scalar == ScalarType::F64 {
                return Err(VkFftError::UnsupportedPrecision {
                    backend: "Metal source backend",
                    precision: "f64",
                });
            }
            Ok(
                r#"#include <metal_stdlib>
using namespace metal;
using vec2 = float2;
inline float vkfft_half_to_float(ushort h) { return float(as_type<half>(h)); }
inline ushort vkfft_float_to_half(float value) { return as_type<ushort>(half(value)); }
inline vec2 vkfft_unpack_half2(uint packed) { return vec2(vkfft_half_to_float((ushort)packed), vkfft_half_to_float((ushort)(packed >> 16))); }
inline uint vkfft_pack_half2(vec2 value) { return (uint)vkfft_float_to_half(value.x) | (uint)vkfft_float_to_half(value.y) << 16; }

"#
                .to_owned(),
            )
        }
        Backend::Vulkan | Backend::CpuReference => Err(VkFftError::UnsupportedKernelPath(
            "native source prelude requires a non-Vulkan GPU backend",
        )),
    }
}

fn amd_opencl_native_subgroup_shuffle_helper(
    scalar: ScalarType,
    subgroup_width: usize,
) -> Result<String> {
    if !matches!(subgroup_width, 32 | 64) {
        return Err(VkFftError::UnsupportedKernelPath(
            "AMD OpenCL native HIP fallback requires wave32 or wave64",
        ));
    }
    let vector = match scalar {
        ScalarType::F16 => {
            return Err(VkFftError::UnsupportedPrecision {
                backend: "AMD OpenCL native HIP subgroup lowering",
                precision: "binary16 compute",
            });
        }
        ScalarType::F32 => "vec2",
        ScalarType::F64 => "dvec2",
        ScalarType::DoubleDouble => {
            return Err(VkFftError::UnsupportedPrecision {
                backend: "AMD OpenCL native HIP subgroup lowering",
                precision: "double-double compute lowering is not implemented",
            });
        }
    };
    Ok(format!(
        "#pragma OPENCL EXTENSION cl_khr_subgroups : enable\ninline {vector} vkfft_subgroup_shuffle_marker({vector} value, uint source_lane) {{\n    return ({vector})(sub_group_broadcast(value.x, source_lane), sub_group_broadcast(value.y, source_lane));\n}}\n"
    ))
}

fn amd_opencl_native_subgroup_invocation_helper(subgroup_width: usize) -> Result<String> {
    if !matches!(subgroup_width, 32 | 64) {
        return Err(VkFftError::UnsupportedKernelPath(
            "AMD OpenCL native HIP fallback requires wave32 or wave64",
        ));
    }
    Ok("#pragma OPENCL EXTENSION cl_khr_subgroups : enable\ninline uint vkfft_subgroup_invocation_id_marker(uint parser_anchor) { (void)parser_anchor; return get_sub_group_local_id(); }\n".to_owned())
}

fn amd_opencl_native_subgroup_id_helper(subgroup_width: usize) -> Result<String> {
    if !matches!(subgroup_width, 32 | 64) {
        return Err(VkFftError::UnsupportedKernelPath(
            "AMD OpenCL native HIP fallback requires wave32 or wave64",
        ));
    }
    Ok("#pragma OPENCL EXTENSION cl_khr_subgroups : enable\ninline uint vkfft_subgroup_id_marker(uint parser_anchor) { (void)parser_anchor; return get_sub_group_id(); }\n".to_owned())
}

fn native_subgroup_shuffle_helper(
    backend: Backend,
    scalar: ScalarType,
    subgroup_width: usize,
) -> Result<String> {
    let vector = match scalar {
        ScalarType::F16 => {
            return Err(VkFftError::UnsupportedPrecision {
                backend: "native subgroup lowering",
                precision: "binary16 compute",
            });
        }
        ScalarType::F32 => "vec2",
        ScalarType::F64 => "dvec2",
        ScalarType::DoubleDouble => {
            return Err(VkFftError::UnsupportedPrecision {
                backend: "native subgroup lowering",
                precision: "double-double compute lowering is not implemented",
            });
        }
    };
    let constructor = match scalar {
        ScalarType::F16 => unreachable!("binary16 compute rejected above"),
        ScalarType::F32 => "vkfft_v2",
        ScalarType::F64 => "vkfft_dv2",
        ScalarType::DoubleDouble => unreachable!("double-double compute rejected above"),
    };
    match backend {
        Backend::Cuda => {
            if subgroup_width != 32 {
                return Err(VkFftError::UnsupportedKernelPath(
                    "CUDA subgroup shuffle lowering requires warp32",
                ));
            }
            Ok(format!(
                "#if !defined(__CUDACC_VER_MAJOR__) || (__CUDACC_VER_MAJOR__ < 9)\n__device__ __forceinline__ {vector} vkfft_subgroup_shuffle_marker({vector} value, uint source_lane) {{\n    return {constructor}(__shfl(value.x, source_lane, {subgroup_width}), __shfl(value.y, source_lane, {subgroup_width}));\n}}\n#else\n__device__ __forceinline__ {vector} vkfft_subgroup_shuffle_marker({vector} value, uint source_lane) {{\n    unsigned mask = __activemask();\n    return {constructor}(__shfl_sync(mask, value.x, source_lane, {subgroup_width}), __shfl_sync(mask, value.y, source_lane, {subgroup_width}));\n}}\n#endif\n"
            ))
        }
        Backend::Hip => {
            if !matches!(subgroup_width, 32 | 64) {
                return Err(VkFftError::UnsupportedKernelPath(
                    "HIP subgroup shuffle lowering requires wave32 or wave64",
                ));
            }
            Ok(format!(
                "__device__ __forceinline__ {vector} vkfft_subgroup_shuffle_marker({vector} value, uint source_lane) {{\n    return {constructor}(__shfl(value.x, source_lane, {subgroup_width}), __shfl(value.y, source_lane, {subgroup_width}));\n}}\n"
            ))
        }
        Backend::OpenCl => Ok(format!(
            "#pragma OPENCL EXTENSION cl_khr_subgroups : enable\n#pragma OPENCL EXTENSION cl_khr_subgroup_shuffle : enable\ninline {vector} vkfft_subgroup_shuffle_marker({vector} value, uint source_lane) {{\n    return ({vector})(sub_group_shuffle(value.x, source_lane), sub_group_shuffle(value.y, source_lane));\n}}\n"
        )),
        Backend::LevelZero => Ok(format!(
            "#pragma OPENCL EXTENSION cl_intel_subgroups : enable\ninline {vector} vkfft_subgroup_shuffle_marker({vector} value, uint source_lane) {{\n    return ({vector})(intel_sub_group_shuffle(value.x, source_lane), intel_sub_group_shuffle(value.y, source_lane));\n}}\n"
        )),
        Backend::Metal => Err(VkFftError::UnsupportedKernelPath(
            "native subgroup intrinsic lowering is not yet proven for Metal; use the shared/workgroup fallback",
        )),
        Backend::Vulkan | Backend::CpuReference => Err(VkFftError::UnsupportedKernelPath(
            "native subgroup lowering requires a native GPU backend",
        )),
    }
}

fn native_subgroup_invocation_helper(backend: Backend, subgroup_width: usize) -> Result<String> {
    match backend {
        Backend::Cuda | Backend::Hip => {
            let mask = subgroup_width.checked_sub(1).ok_or(VkFftError::InvalidKernelIr(
                "native subgroup invocation lowering received zero subgroup width",
            ))?;
            Ok(format!(
                "__device__ __forceinline__ uint vkfft_subgroup_invocation_id_marker(uint parser_anchor) {{ return parser_anchor & {mask}u; }}\n"
            ))
        }
        Backend::OpenCl | Backend::LevelZero => Ok(
            "inline uint vkfft_subgroup_invocation_id_marker(uint parser_anchor) { (void)parser_anchor; return get_sub_group_local_id(); }\n".to_owned(),
        ),
        Backend::Metal => Err(VkFftError::UnsupportedKernelPath(
            "native subgroup id lowering is not yet proven for Metal",
        )),
        Backend::Vulkan | Backend::CpuReference => Err(VkFftError::UnsupportedKernelPath(
            "native subgroup lowering requires a native GPU backend",
        )),
    }
}

fn native_subgroup_id_helper(backend: Backend, subgroup_width: usize) -> Result<String> {
    match backend {
        Backend::Cuda | Backend::Hip => {
            if subgroup_width == 0 || !subgroup_width.is_power_of_two() {
                return Err(VkFftError::InvalidKernelIr(
                    "native subgroup id lowering requires a power-of-two subgroup width",
                ));
            }
            let shift = subgroup_width.trailing_zeros();
            Ok(format!(
                "__device__ __forceinline__ uint vkfft_subgroup_id_marker(uint parser_anchor) {{ return parser_anchor >> {shift}u; }}\n"
            ))
        }
        Backend::OpenCl | Backend::LevelZero => Ok(
            "inline uint vkfft_subgroup_id_marker(uint parser_anchor) { (void)parser_anchor; return get_sub_group_id(); }\n".to_owned(),
        ),
        Backend::Metal => Err(VkFftError::UnsupportedKernelPath(
            "native subgroup id lowering is not yet proven for Metal",
        )),
        Backend::Vulkan | Backend::CpuReference => Err(VkFftError::UnsupportedKernelPath(
            "native subgroup lowering requires a native GPU backend",
        )),
    }
}

fn kernel_signature(
    backend: Backend,
    workgroup: WorkgroupSize,
    bindings: &[BufferBinding],
    binding_shapes: &[ProgramElementShape],
    required_subgroup_size: Option<u32>,
) -> Result<String> {
    let mut arguments = Vec::new();
    match backend {
        Backend::Metal => {
            arguments.push(
                "uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]".to_owned(),
            );
            arguments.push(
                "uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]]"
                    .to_owned(),
            );
        }
        Backend::Cuda | Backend::Hip | Backend::OpenCl | Backend::LevelZero => {}
        Backend::Vulkan | Backend::CpuReference => {
            return Err(VkFftError::UnsupportedKernelPath(
                "native source signature requires a non-Vulkan GPU backend",
            ));
        }
    }
    if binding_shapes.len() != bindings.len() {
        return Err(VkFftError::InvalidKernelIr(
            "native signature element-shape metadata does not match binding count",
        ));
    }
    let mut ordered = bindings
        .iter()
        .copied()
        .zip(binding_shapes.iter().copied())
        .collect::<Vec<_>>();
    ordered.sort_by_key(|(binding, _)| binding.binding);
    for (binding, shape) in ordered {
        let name = binding_argument_name(bindings, binding);
        let read_only = binding.access == BufferAccess::ReadOnly;
        let vector = match (shape, binding.scalar) {
            (ProgramElementShape::Scalar, ScalarType::F16) => "ushort",
            (ProgramElementShape::Scalar, ScalarType::F32) => "float",
            (ProgramElementShape::Scalar, ScalarType::F64) => "double",
            (ProgramElementShape::Scalar, ScalarType::DoubleDouble) => "dvec2",
            (ProgramElementShape::Complex, ScalarType::F16) => "uint",
            (ProgramElementShape::Complex, ScalarType::F32) => "vec2",
            (ProgramElementShape::Complex, ScalarType::F64) => "dvec2",
            (ProgramElementShape::Complex, ScalarType::DoubleDouble) => "dvec4",
        };
        let argument = match backend {
            Backend::Cuda | Backend::Hip => {
                format!(
                    "{}{}* {}",
                    if read_only { "const " } else { "" },
                    vector,
                    name
                )
            }
            Backend::OpenCl | Backend::LevelZero => format!(
                "__global {}{}* {}",
                if read_only { "const " } else { "" },
                vector,
                name
            ),
            Backend::Metal => {
                let address_space = if read_only
                    && matches!(
                        binding.role,
                        BufferRole::LookupTable
                            | BufferRole::TwiddleLookupTable
                            | BufferRole::Auxiliary
                    ) {
                    "constant"
                } else {
                    "device"
                };
                format!(
                    "{address_space} {}{}* {} [[buffer({})]]",
                    if read_only { "const " } else { "" },
                    vector,
                    name,
                    binding.binding
                )
            }
            Backend::Vulkan | Backend::CpuReference => unreachable!(),
        };
        arguments.push(argument);
    }
    let joined = arguments.join(",\n    ");
    let signature = match backend {
        Backend::Cuda | Backend::Hip => {
            let threads = workgroup
                .x
                .checked_mul(workgroup.y)
                .and_then(|value| value.checked_mul(workgroup.z))
                .ok_or(VkFftError::ArithmeticOverflow {
                    operation: "native launch-bounds workgroup product",
                })?;
            format!(
                "extern \"C\" __global__ __launch_bounds__({threads}) void VkFFT_main(\n    {joined}\n)"
            )
        }
        Backend::OpenCl => {
            let required_subgroup = required_subgroup_size
                .map(|size| format!("__attribute__((intel_reqd_sub_group_size({size}))) "))
                .unwrap_or_default();
            format!(
                "{required_subgroup}__kernel __attribute__((reqd_work_group_size({}, {}, {}))) void VkFFT_main(\n    {joined}\n)",
                workgroup.x, workgroup.y, workgroup.z
            )
        }
        Backend::LevelZero => {
            let required_subgroup = required_subgroup_size
                .map(|size| format!("__attribute__((intel_reqd_sub_group_size({size}))) "))
                .unwrap_or_default();
            format!(
                "{required_subgroup}__kernel __attribute__((reqd_work_group_size({}, {}, {}))) void VkFFT_main(\n    {joined}\n)",
                workgroup.x, workgroup.y, workgroup.z
            )
        }
        Backend::Metal => format!("kernel void VkFFT_main(\n    {joined}\n)"),
        Backend::Vulkan | Backend::CpuReference => unreachable!(),
    };
    Ok(signature)
}

fn translate_shared_decl(backend: Backend, declaration: &str) -> Result<String> {
    let rest = declaration
        .strip_prefix("shared ")
        .ok_or(VkFftError::ShaderCompilation(
            "invalid Vulkan shared-memory declaration".to_owned(),
        ))?;
    let qualifier = match backend {
        Backend::Cuda | Backend::Hip => "__shared__ ",
        Backend::OpenCl | Backend::LevelZero => "__local ",
        Backend::Metal => "threadgroup ",
        Backend::Vulkan | Backend::CpuReference => {
            return Err(VkFftError::UnsupportedKernelPath(
                "native shared-memory translation requires a source backend",
            ));
        }
    };
    Ok(format!("{qualifier}{rest}"))
}

const fn local_id_expr(backend: Backend) -> &'static str {
    match backend {
        Backend::Cuda | Backend::Hip => "threadIdx.x",
        Backend::OpenCl | Backend::LevelZero => "get_local_id(0)",
        Backend::Metal => "thread_position_in_threadgroup.x",
        Backend::Vulkan => "gl_LocalInvocationID.x",
        Backend::CpuReference => "0u",
    }
}

const fn local_id_y_expr(backend: Backend) -> &'static str {
    match backend {
        Backend::Cuda | Backend::Hip => "threadIdx.y",
        Backend::OpenCl | Backend::LevelZero => "get_local_id(1)",
        Backend::Metal => "thread_position_in_threadgroup.y",
        Backend::Vulkan => "gl_LocalInvocationID.y",
        Backend::CpuReference => "0u",
    }
}

const fn group_id_expr(backend: Backend) -> &'static str {
    match backend {
        Backend::Cuda | Backend::Hip => "blockIdx.x",
        Backend::OpenCl | Backend::LevelZero => "get_group_id(0)",
        Backend::Metal => "threadgroup_position_in_grid.x",
        Backend::Vulkan => "gl_WorkGroupID.x",
        Backend::CpuReference => "0u",
    }
}

const fn barrier_expr(backend: Backend) -> &'static str {
    match backend {
        Backend::Cuda | Backend::Hip => "__syncthreads();",
        Backend::OpenCl | Backend::LevelZero => {
            "barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);"
        }
        Backend::Metal => {
            "threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);"
        }
        Backend::Vulkan => "barrier();",
        Backend::CpuReference => "",
    }
}

fn parse_glsl_buffer_binding_and_instance(line: &str) -> Option<(u32, String)> {
    let trimmed = line.trim();
    if !trimmed.starts_with("layout(set") || !trimmed.contains(" buffer ") {
        return None;
    }
    let binding_tail = trimmed.split_once("binding = ")?.1;
    let binding_digits = binding_tail
        .chars()
        .take_while(|ch| ch.is_ascii_digit())
        .collect::<String>();
    let binding = binding_digits.parse::<u32>().ok()?;
    let (_, instance_tail) = trimmed.rsplit_once("} ")?;
    let instance = instance_tail.trim_end_matches(';').trim();
    if instance.is_empty()
        || !instance
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || ch == '_')
    {
        return None;
    }
    Some((binding, instance.to_owned()))
}

fn binding_argument_name(bindings: &[BufferBinding], binding: BufferBinding) -> String {
    let base = role_argument_name(binding.role);
    if bindings
        .iter()
        .filter(|candidate| candidate.role == binding.role)
        .count()
        > 1
    {
        format!("{base}_{}", binding.binding)
    } else {
        base.to_owned()
    }
}

const fn role_argument_name(role: BufferRole) -> &'static str {
    match role {
        BufferRole::Input => "inputs",
        BufferRole::Output => "outputs",
        BufferRole::LookupTable => "lookup_table",
        BufferRole::TwiddleLookupTable => "twiddle_lut",
        BufferRole::Auxiliary => "auxiliary",
    }
}

const fn rader_constant_prefix(backend: Backend) -> &'static str {
    match backend {
        Backend::Cuda | Backend::Hip => "__device__ __constant__ ",
        Backend::OpenCl | Backend::LevelZero => "__constant ",
        Backend::Metal => "constant ",
        Backend::Vulkan | Backend::CpuReference => "const ",
    }
}

fn is_device_helper_definition(line: &str) -> bool {
    let trimmed = line.trim_start();
    (trimmed.starts_with("vec2 vkfft_")
        || trimmed.starts_with("dvec2 vkfft_")
        || trimmed.starts_with("dvec4 vkfft_")
        || trimmed.starts_with("uint vkfft_")
        || trimmed.starts_with("float vkfft_")
        || trimmed.starts_with("double vkfft_"))
        && trimmed.contains('(')
        && (trimmed.ends_with('{') || trimmed.contains(" { return "))
}

macro_rules! define_native_backend {
    ($name:ident, $backend:expr) => {
        #[derive(Debug, Clone, Copy, Default)]
        pub struct $name;

        impl crate::backend::KernelBackend for $name {
            type Output = crate::backend::native::NativeShaderSource;

            fn lower(&self, kernel: &crate::KernelIr) -> crate::Result<Self::Output> {
                crate::backend::native::NativeSourceBackend::new($backend).lower_kernel(kernel)
            }
        }

        impl $name {
            pub fn lower_one_dim_fft(
                &self,
                ir: &crate::OneDimFftIr,
            ) -> crate::Result<Vec<crate::backend::native::NativeShaderSource>> {
                crate::backend::native::NativeSourceBackend::new($backend).lower_one_dim_fft(ir)
            }

            pub fn lower_recursive_fft(
                &self,
                ir: &crate::RecursiveFftIr,
            ) -> crate::Result<Vec<crate::backend::native::NativeShaderSource>> {
                crate::backend::native::NativeSourceBackend::new($backend).lower_recursive_fft(ir)
            }

            pub fn lower_nd_fft(
                &self,
                ir: &crate::NdFftIr,
            ) -> crate::Result<Vec<crate::backend::native::NativeShaderSource>> {
                crate::backend::native::NativeSourceBackend::new($backend).lower_nd_fft(ir)
            }

            pub fn lower_real_fft(
                &self,
                ir: &crate::RealFftIr,
            ) -> crate::Result<Vec<crate::backend::native::NativeShaderSource>> {
                crate::backend::native::NativeSourceBackend::new($backend).lower_real_fft(ir)
            }

            pub fn lower_nd_real_fft(
                &self,
                ir: &crate::NdRealFftIr,
            ) -> crate::Result<Vec<crate::backend::native::NativeShaderSource>> {
                crate::backend::native::NativeSourceBackend::new($backend).lower_nd_real_fft(ir)
            }

            pub fn lower_r2r(
                &self,
                ir: &crate::R2rIr,
            ) -> crate::Result<crate::backend::native::NativeShaderSource> {
                crate::backend::native::NativeSourceBackend::new($backend).lower_r2r(ir)
            }

            pub fn lower_r2r_program(
                &self,
                ir: &crate::R2rIr,
            ) -> crate::Result<Vec<crate::backend::native::NativeShaderSource>> {
                crate::backend::native::NativeSourceBackend::new($backend).lower_r2r_program(ir)
            }

            pub fn lower_nd_r2r(
                &self,
                ir: &crate::NdR2rIr,
            ) -> crate::Result<Vec<crate::backend::native::NativeShaderSource>> {
                crate::backend::native::NativeSourceBackend::new($backend).lower_nd_r2r(ir)
            }

            pub fn lower_rader_direct(
                &self,
                ir: &crate::RaderDirectIr,
            ) -> crate::Result<crate::backend::native::NativeShaderSource> {
                crate::backend::native::NativeSourceBackend::new($backend).lower_rader_direct(ir)
            }

            pub fn lower_transform(
                &self,
                ir: &crate::TransformIr,
            ) -> crate::Result<crate::backend::native::NativeProgramSource> {
                crate::backend::native::NativeSourceBackend::new($backend).lower_transform(ir)
            }
        }
    };
}

pub(crate) use define_native_backend;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::KernelBackend;
    use crate::config::{
        DeviceProfile, Direction, FftConfig, GpuVendor, Precision, SubgroupProfile,
    };
    use crate::{FftPlan, KernelIr};

    fn assert_native_tokens(shader: &NativeShaderSource) {
        shader.validate().unwrap();
        assert!(!shader.source.contains("gl_LocalInvocationID"));
        assert!(!shader.source.contains("vkfft_input.data"));
    }

    #[test]
    fn native_multi_lut_bindings_keep_glsl_instances_unique() {
        let bindings = vec![
            BufferBinding {
                set: 0,
                binding: 2,
                role: BufferRole::LookupTable,
                access: BufferAccess::ReadOnly,
                scalar: ScalarType::DoubleDouble,
            },
            BufferBinding {
                set: 0,
                binding: 3,
                role: BufferRole::LookupTable,
                access: BufferAccess::ReadOnly,
                scalar: ScalarType::DoubleDouble,
            },
            BufferBinding {
                set: 0,
                binding: 4,
                role: BufferRole::TwiddleLookupTable,
                access: BufferAccess::ReadOnly,
                scalar: ScalarType::DoubleDouble,
            },
        ];
        assert_eq!(
            binding_argument_name(&bindings, bindings[0]),
            "lookup_table_2"
        );
        assert_eq!(
            binding_argument_name(&bindings, bindings[1]),
            "lookup_table_3"
        );
        assert_eq!(binding_argument_name(&bindings, bindings[2]), "twiddle_lut");
        assert_eq!(
            parse_glsl_buffer_binding_and_instance(
                "layout(set = 0, binding = 3, std430) readonly buffer Left { dvec4 data[]; } vkfft_left_lut;"
            ),
            Some((3, "vkfft_left_lut".to_owned()))
        );

        let signature = kernel_signature(
            Backend::Cuda,
            WorkgroupSize { x: 19, y: 1, z: 1 },
            &bindings,
            &vec![ProgramElementShape::Complex; bindings.len()],
            None,
        )
        .unwrap();
        assert!(signature.contains("lookup_table_2"));
        assert!(signature.contains("lookup_table_3"));
        assert_eq!(signature.matches("lookup_table_2").count(), 1);
        assert_eq!(signature.matches("lookup_table_3").count(), 1);
    }

    #[test]
    fn intel_opencl_dialects_emit_required_subgroup_width_on_native_kernel() {
        for backend in [Backend::OpenCl, Backend::LevelZero] {
            let mut profile = DeviceProfile::generic(backend, GpuVendor::Intel);
            profile.subgroup = SubgroupProfile {
                size: 32,
                min_size: 8,
                max_size: 32,
                required_size_compute_supported: true,
                compute_supported: true,
                basic_supported: true,
                shuffle_supported: true,
                shuffle_relative_supported: false,
                compute_full_subgroups: true,
            };
            let ir =
                crate::TransformIr::build(FftConfig::new(vec![152]), Direction::Forward, profile)
                    .unwrap();
            let source = NativeSourceBackend::new(backend)
                .lower_transform(&ir)
                .unwrap();
            let shuffle_intrinsic = if backend == Backend::OpenCl {
                "sub_group_shuffle("
            } else {
                "intel_sub_group_shuffle("
            };
            let subgroup_shader = source
                .shaders
                .iter()
                .find(|shader| shader.source.contains(shuffle_intrinsic))
                .unwrap_or_else(|| {
                    panic!("Intel required-subgroup N152 should use {backend:?} subgroup shuffle")
                });
            assert!(
                subgroup_shader
                    .source
                    .contains("#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable")
            );
            assert!(
                subgroup_shader
                    .source
                    .contains("__attribute__((intel_reqd_sub_group_size(32))) __kernel")
            );
        }
    }

    #[test]
    fn level_zero_power_of_two_stockham_emits_exact_subgroup_widths() {
        for width in [8usize, 16, 32] {
            let mut profile = DeviceProfile::generic(Backend::LevelZero, GpuVendor::Intel);
            profile.subgroup = SubgroupProfile {
                size: width,
                min_size: 8,
                max_size: 32,
                required_size_compute_supported: true,
                compute_supported: true,
                basic_supported: true,
                shuffle_supported: true,
                shuffle_relative_supported: false,
                compute_full_subgroups: true,
            };
            let length = width * 8;
            let ir = crate::TransformIr::build(
                FftConfig::new(vec![length]),
                Direction::Forward,
                profile,
            )
            .unwrap();
            let source = NativeSourceBackend::new(Backend::LevelZero)
                .lower_transform(&ir)
                .unwrap();
            let subgroup_shader = source
                .shaders
                .iter()
                .find(|shader| shader.source.contains("intel_sub_group_shuffle("))
                .unwrap_or_else(|| {
                    panic!("Level Zero exact-{width} N{length} should use subgroup shuffle")
                });
            assert_eq!(subgroup_shader.required_shared_memory_bytes, 0);
            assert!(
                subgroup_shader
                    .source
                    .contains("#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable")
            );
            assert!(subgroup_shader.source.contains(&format!(
                "__attribute__((intel_reqd_sub_group_size({width}))) __kernel"
            )));
        }
    }

    #[test]
    fn opencl_dialects_put_program_scope_vkfft_scalars_in_constant_address_space() {
        let glsl = r#"#version 450
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
const uint VKFFT_N = 34u;
const float VKFFT_SIGN = -1.0;
void main() {
    const uint VKFFT_LOCAL = 3u;
    uint sink = VKFFT_N + VKFFT_LOCAL;
}
"#;
        let workgroup = WorkgroupSize { x: 1, y: 1, z: 1 };

        for backend in [Backend::OpenCl, Backend::LevelZero] {
            let source =
                translate_glsl_dialect(backend, glsl, ScalarType::F32, workgroup, &[], &[])
                    .unwrap();
            assert!(source.contains("__constant uint VKFFT_N = 34u;"));
            assert!(source.contains("__constant float VKFFT_SIGN ="));
            assert!(source.contains("    const uint VKFFT_LOCAL = 3u;"));
            assert!(!source.contains("\nconst uint VKFFT_N = 34u;"));
        }

        for backend in [Backend::Cuda, Backend::Metal] {
            let source =
                translate_glsl_dialect(backend, glsl, ScalarType::F32, workgroup, &[], &[])
                    .unwrap();
            assert!(source.contains("\nconst uint VKFFT_N = 34u;"));
            assert!(source.contains("    const uint VKFFT_LOCAL = 3u;"));
            assert!(!source.contains("__constant uint VKFFT_N = 34u;"));
        }
    }

    #[test]
    fn five_native_dialects_lower_policy_driven_stockham() {
        for (backend, vendor) in [
            (Backend::Cuda, GpuVendor::Nvidia),
            (Backend::Hip, GpuVendor::Amd),
            (Backend::OpenCl, GpuVendor::Amd),
            (Backend::LevelZero, GpuVendor::Intel),
            (Backend::Metal, GpuVendor::Apple),
        ] {
            let device = DeviceProfile::generic(backend, vendor);
            let plan = FftPlan::build(FftConfig::new(vec![256])).unwrap();
            let kernel = KernelIr::stockham_1d(&plan, Direction::Forward, device).unwrap();
            let shader = NativeSourceBackend::new(backend).lower(&kernel).unwrap();
            assert_native_tokens(&shader);
            if backend == Backend::Hip {
                let fallback = shader
                    .compiler_fallback_source
                    .as_deref()
                    .expect("HIP lowering must retain an OpenCL compiler fallback source");
                assert!(fallback.contains("__kernel"));
                assert!(fallback.contains("get_local_id(0)"));
                assert!(fallback.contains("CLK_LOCAL_MEM_FENCE"));
                assert!(!fallback.contains("#include <hip/hip_runtime.h>"));
                assert!(!fallback.contains("threadIdx.x"));
            } else {
                assert!(shader.compiler_fallback_source.is_none());
            }
            match backend {
                Backend::Cuda | Backend::Hip => {
                    assert!(shader.source.contains("__global__"));
                    assert!(shader.source.contains("threadIdx.x"));
                    assert!(shader.source.contains("__syncthreads"));
                }
                Backend::OpenCl | Backend::LevelZero => {
                    assert!(shader.source.contains("__kernel"));
                    assert!(shader.source.contains("get_local_id(0)"));
                    assert!(shader.source.contains("CLK_LOCAL_MEM_FENCE"));
                    assert!(shader.source.contains("__constant float VKFFT_PI ="));
                    assert!(shader.source.contains("__constant float VKFFT_TAU ="));
                    assert!(!shader.source.contains("\nconst float VKFFT_PI ="));
                    assert!(!shader.source.contains("\nconst float VKFFT_TAU ="));
                }
                Backend::Metal => {
                    assert!(shader.source.contains("kernel void VkFFT_main"));
                    assert!(shader.source.contains("thread_position_in_threadgroup"));
                    assert!(shader.source.contains("threadgroup_barrier"));
                }
                _ => unreachable!(),
            }
        }
    }

    #[test]
    fn convolution_cross_power_builtins_lower_across_native_dialects() {
        let glsl = "vec2 vkfft_convolution_product = vec2(1.0, 2.0);\nfloat vkfft_convolution_norm = dot(vkfft_convolution_product, vkfft_convolution_product);\nvec2 vkfft_matrix_sum = vec2(3.0, 4.0);\nfloat vkfft_matrix_norm = dot(vkfft_matrix_sum, vkfft_matrix_sum);\nfloat vkfft_convolution_scale = inversesqrt(vkfft_convolution_norm);\n";
        let workgroup = WorkgroupSize { x: 1, y: 1, z: 1 };

        for backend in [
            Backend::Cuda,
            Backend::Hip,
            Backend::OpenCl,
            Backend::LevelZero,
            Backend::Metal,
        ] {
            let source =
                translate_glsl_dialect(backend, glsl, ScalarType::F32, workgroup, &[], &[])
                    .unwrap();
            assert!(!source.contains("dot(vkfft_convolution_product"));
            assert!(!source.contains("dot(vkfft_matrix_sum"));
            assert!(!source.contains("inversesqrt("));
            assert!(source.contains("vkfft_convolution_product.x * vkfft_convolution_product.x"));
            assert!(source.contains("vkfft_convolution_product.y * vkfft_convolution_product.y"));
            assert!(source.contains("vkfft_matrix_sum.x * vkfft_matrix_sum.x"));
            assert!(source.contains("vkfft_matrix_sum.y * vkfft_matrix_sum.y"));
            assert!(source.contains("rsqrt(vkfft_convolution_norm)"));
        }
    }

    #[test]
    fn cuda_real_n34_embeds_the_same_p17_native_rader_shaders_as_standalone() {
        let mut device = DeviceProfile::generic(Backend::Cuda, GpuVendor::Nvidia);
        device.shared_memory_bytes = 48 * 1024;
        device.shared_memory_pow2_bytes = 32 * 1024;
        device.max_threads_per_block = 1024;
        let lowering = NativeSourceBackend::new(Backend::Cuda);

        let p17_plan = FftPlan::build(FftConfig::new(vec![17])).unwrap();
        let p17 = crate::OneDimFftIr::build(&p17_plan, Direction::Forward, device).unwrap();
        let standalone_ir = crate::TransformIr::Complex1d(p17);
        let standalone = lowering.lower_transform(&standalone_ir).unwrap();
        assert_eq!(standalone.shaders.len(), 2);

        let real_plan = FftPlan::build(
            FftConfig::new(vec![34]).with_transform(crate::TransformKind::RealToComplex),
        )
        .unwrap();
        let real = crate::RealFftIr::build(&real_plan, device).unwrap();
        let real_ir = crate::TransformIr::Real(real);
        let embedded = lowering.lower_transform(&real_ir).unwrap();
        assert_eq!(embedded.shaders.len(), 4);
        assert_eq!(embedded.shaders[1].source, standalone.shaders[0].source);
        assert_eq!(embedded.shaders[2].source, standalone.shaders[1].source);
        assert_eq!(embedded.shaders[1].bindings, standalone.shaders[0].bindings);
        assert_eq!(embedded.shaders[2].bindings, standalone.shaders[1].bindings);
    }

    #[test]
    fn native_rader_and_high_level_paths_remove_vulkan_abi() {
        let cuda_device = DeviceProfile {
            shared_memory_bytes: 48 * 1024,
            shared_memory_pow2_bytes: 32 * 1024,
            max_threads_per_block: 1024,
            ..DeviceProfile::generic(Backend::Cuda, GpuVendor::Nvidia)
        };
        let plan = FftPlan::build(FftConfig::new(vec![257])).unwrap();
        let ir = crate::RaderFftPipelineIr::build(&plan, Direction::Forward, cuda_device).unwrap();
        let shaders = NativeSourceBackend::new(Backend::Cuda)
            .lower_one_dim_fft(&ir.forward_fft)
            .unwrap();
        assert!(!shaders.is_empty());
        for shader in shaders {
            assert_native_tokens(&shader);
        }

        let opencl_device = DeviceProfile::generic(Backend::OpenCl, GpuVendor::Amd);
        let plan = FftPlan::build(FftConfig::new(vec![103])).unwrap();
        let ir = crate::OneDimFftIr::build(&plan, Direction::Forward, opencl_device).unwrap();
        let shaders = NativeSourceBackend::new(Backend::OpenCl)
            .lower_one_dim_fft(&ir)
            .unwrap();
        assert!(shaders.len() >= 4);
        for shader in shaders {
            assert_native_tokens(&shader);
        }
    }

    #[test]
    fn hip_wave_width_changes_high_level_direct_rader_launch_geometry() {
        let mut tuning = crate::PlannerTuning::portable();
        tuning.min_rader_fft_prime = 89;
        tuning.validate().unwrap();

        for (wave_size, expected_group) in [(32usize, 4usize), (64usize, 3usize)] {
            let mut device = DeviceProfile::generic(Backend::Hip, GpuVendor::Amd);
            device.shared_memory_bytes = 32 * 1024;
            device.shared_memory_pow2_bytes = 32 * 1024;
            device.max_threads_per_block = 512;
            device.max_workgroup_size = [512, 512, 64];
            device.coalesced_memory_bytes = 32;
            device.subgroup = crate::SubgroupProfile {
                size: wave_size,
                min_size: wave_size,
                max_size: wave_size,
                required_size_compute_supported: false,
                compute_supported: true,
                basic_supported: true,
                shuffle_supported: true,
                shuffle_relative_supported: true,
                compute_full_subgroups: true,
            };

            let ir = TransformIr::build(
                FftConfig::new(vec![67])
                    .with_batch_count(32)
                    .with_tuning(tuning),
                Direction::Forward,
                device,
            )
            .unwrap();
            let TransformIr::Complex1d(crate::OneDimFftIr::Recursive(recursive)) = &ir else {
                panic!("HIP p67 wave-width witness must remain recursive Rader IR");
            };
            let crate::recursive_ir::RecursiveFftNodeIr::DirectRader(direct_ir) = &recursive.root
            else {
                panic!("HIP p67 wave-width witness must retain a Direct-Rader root");
            };
            let block = direct_ir
                .axis_batch_block
                .expect("HIP p67 Direct-Rader must retain physical batching");
            assert_eq!(block.threads_per_transform, 34);
            assert_eq!(block.grouped_batch, expected_group);
            assert_eq!(block.local_size_x, 34);
            assert_eq!(block.local_size_y, expected_group);
            assert!(!block.transforms_on_x);
            assert!(!block.axis_swapped);

            let source = NativeSourceBackend::new(Backend::Hip)
                .lower_transform(&ir)
                .unwrap();
            source.validate().unwrap();
            let shader = source
                .shaders
                .iter()
                .find(|shader| shader.sequence_len == 67 && shader.workgroup_size.x == 34)
                .expect("HIP p67 direct kernel must preserve the scheduler workgroup");
            assert_eq!(shader.batch_count, 32);
            assert_eq!(shader.workgroup_size.x, 34);
            assert_eq!(shader.workgroup_size.y, expected_group as u32);
            assert_eq!(shader.workgroup_size.z, 1);
            assert_native_tokens(shader);
        }
    }

    #[test]
    fn hip_f64_compute_f32_storage_keeps_distinct_four_step_threshold_end_to_end() {
        let mut device = DeviceProfile::generic(Backend::Hip, GpuVendor::Amd);
        device.shared_memory_bytes = 64 * 1024;
        device.shared_memory_pow2_bytes = 64 * 1024;
        device.max_threads_per_block = 1024;
        device.max_workgroup_size = [1024, 1024, 64];
        device.coalesced_memory_bytes = 32;
        device.supports_f64 = true;

        for (precision, expected_split, external_scalar) in [
            (
                Precision::F64,
                vec![512usize, 64usize, 32usize],
                ScalarType::F64,
            ),
            (
                Precision::F64ComputeF32Storage,
                vec![2_048usize, 512usize],
                ScalarType::F32,
            ),
        ] {
            let ir = TransformIr::build(
                FftConfig::new(vec![1_048_576]).with_precision(precision),
                Direction::Forward,
                device,
            )
            .unwrap();
            let TransformIr::Complex1d(crate::OneDimFftIr::Recursive(recursive)) = &ir else {
                panic!("HIP N1048576 threshold witness must remain recursive Stockham");
            };
            assert_eq!(recursive.scalar, ScalarType::F64);
            assert_eq!(recursive.external_scalar, external_scalar);
            assert_eq!(recursive.scheduler_precision, precision);
            let schedule = recursive
                .stockham_upload_schedule
                .as_ref()
                .expect("HIP N1048576 threshold witness must retain an upload schedule");
            assert_eq!(schedule.axis_split, expected_split);
            assert_eq!(schedule.upload_count, expected_split.len());
            let kernels = recursive
                .four_step_stockham_upload_kernels()
                .unwrap()
                .expect("HIP N1048576 threshold witness must materialize Four-step kernels");
            assert_eq!(kernels.len(), expected_split.len());

            let source = NativeSourceBackend::new(Backend::Hip)
                .lower_transform(&ir)
                .unwrap();
            source.validate().unwrap();
            assert_eq!(source.program.scalar, ScalarType::F64);
            assert_eq!(
                source.program.input_resource().unwrap().scalar,
                external_scalar
            );
            assert_eq!(
                source.program.output_resource().unwrap().scalar,
                external_scalar
            );
            assert_eq!(source.shaders.len(), expected_split.len());
            for shader in &source.shaders {
                assert_native_tokens(shader);
            }
        }
    }

    #[test]
    fn level_zero_f64_compute_f32_storage_keeps_distinct_four_step_threshold_end_to_end() {
        let mut device = DeviceProfile::generic(Backend::LevelZero, GpuVendor::Intel);
        device.shared_memory_bytes = 64 * 1024;
        device.shared_memory_pow2_bytes = 64 * 1024;
        device.max_threads_per_block = 1024;
        device.max_workgroup_size = [1024, 1024, 64];
        device.coalesced_memory_bytes = 64;
        device.supports_f64 = true;

        for (precision, expected_split, external_scalar) in [
            (
                Precision::F64,
                vec![512usize, 64usize, 8usize],
                ScalarType::F64,
            ),
            (
                Precision::F64ComputeF32Storage,
                vec![512usize, 512usize],
                ScalarType::F32,
            ),
        ] {
            let ir = TransformIr::build(
                FftConfig::new(vec![262_144]).with_precision(precision),
                Direction::Forward,
                device,
            )
            .unwrap();
            let TransformIr::Complex1d(crate::OneDimFftIr::Recursive(recursive)) = &ir else {
                panic!("Level Zero N262144 threshold witness must remain recursive Stockham");
            };
            assert_eq!(recursive.scalar, ScalarType::F64);
            assert_eq!(recursive.external_scalar, external_scalar);
            assert_eq!(recursive.scheduler_precision, precision);
            let schedule = recursive
                .stockham_upload_schedule
                .as_ref()
                .expect("Level Zero N262144 threshold witness must retain an upload schedule");
            assert_eq!(schedule.axis_split, expected_split);
            assert_eq!(schedule.upload_count, expected_split.len());
            let kernels = recursive
                .four_step_stockham_upload_kernels()
                .unwrap()
                .expect("Level Zero N262144 threshold witness must materialize Four-step kernels");
            assert_eq!(kernels.len(), expected_split.len());

            let source = NativeSourceBackend::new(Backend::LevelZero)
                .lower_transform(&ir)
                .unwrap();
            source.validate().unwrap();
            assert_eq!(source.program.scalar, ScalarType::F64);
            assert_eq!(
                source.program.input_resource().unwrap().scalar,
                external_scalar
            );
            assert_eq!(
                source.program.output_resource().unwrap().scalar,
                external_scalar
            );
            assert_eq!(source.shaders.len(), expected_split.len());
            for shader in &source.shaders {
                assert_native_tokens(shader);
            }
        }
    }

    #[test]
    fn opencl_f64_compute_f32_storage_keeps_distinct_four_step_threshold_end_to_end() {
        let mut device = DeviceProfile::generic(Backend::OpenCl, GpuVendor::Amd);
        device.shared_memory_bytes = 64 * 1024;
        device.shared_memory_pow2_bytes = 64 * 1024;
        device.max_threads_per_block = 1024;
        device.max_workgroup_size = [1024, 1024, 64];
        device.coalesced_memory_bytes = 32;
        device.supports_f64 = true;

        for (precision, expected_split, external_scalar) in [
            (
                Precision::F64,
                vec![512usize, 64usize, 8usize],
                ScalarType::F64,
            ),
            (
                Precision::F64ComputeF32Storage,
                vec![512usize, 512usize],
                ScalarType::F32,
            ),
        ] {
            let ir = TransformIr::build(
                FftConfig::new(vec![262_144]).with_precision(precision),
                Direction::Forward,
                device,
            )
            .unwrap();
            let TransformIr::Complex1d(crate::OneDimFftIr::Recursive(recursive)) = &ir else {
                panic!("OpenCL N262144 threshold witness must remain recursive Stockham");
            };
            assert_eq!(recursive.scalar, ScalarType::F64);
            assert_eq!(recursive.external_scalar, external_scalar);
            assert_eq!(recursive.scheduler_precision, precision);
            let schedule = recursive
                .stockham_upload_schedule
                .as_ref()
                .expect("OpenCL N262144 threshold witness must retain an upload schedule");
            assert_eq!(schedule.axis_split, expected_split);
            assert_eq!(schedule.upload_count, expected_split.len());
            let kernels = recursive
                .four_step_stockham_upload_kernels()
                .unwrap()
                .expect("OpenCL N262144 threshold witness must materialize Four-step kernels");
            assert_eq!(kernels.len(), expected_split.len());

            let source = NativeSourceBackend::new(Backend::OpenCl)
                .lower_transform(&ir)
                .unwrap();
            source.validate().unwrap();
            assert_eq!(source.program.scalar, ScalarType::F64);
            assert_eq!(
                source.program.input_resource().unwrap().scalar,
                external_scalar
            );
            assert_eq!(
                source.program.output_resource().unwrap().scalar,
                external_scalar
            );
            assert_eq!(source.shaders.len(), expected_split.len());
            for shader in &source.shaders {
                assert_native_tokens(shader);
            }
        }
    }

    #[test]
    fn all_native_dialects_cover_nd_real_and_r2r_transform_families() {
        for (backend, vendor) in [
            (Backend::Cuda, GpuVendor::Nvidia),
            (Backend::Hip, GpuVendor::Amd),
            (Backend::OpenCl, GpuVendor::Amd),
            (Backend::LevelZero, GpuVendor::Intel),
            (Backend::Metal, GpuVendor::Apple),
        ] {
            let device = DeviceProfile::generic(backend, vendor);
            let lowering = NativeSourceBackend::new(backend);

            let nd_plan = FftPlan::build(FftConfig::new(vec![3, 8])).unwrap();
            let nd = crate::NdFftIr::build(&nd_plan, Direction::Forward, device).unwrap();
            let nd_shaders = lowering.lower_nd_fft(&nd).unwrap();
            assert!(!nd_shaders.is_empty());
            for shader in &nd_shaders {
                assert_native_tokens(shader);
            }

            let real_plan = FftPlan::build(
                FftConfig::new(vec![16]).with_transform(crate::TransformKind::RealToComplex),
            )
            .unwrap();
            let real = crate::RealFftIr::build(&real_plan, device).unwrap();
            let real_shaders = lowering.lower_real_fft(&real).unwrap();
            assert!(!real_shaders.is_empty());
            for shader in &real_shaders {
                assert_native_tokens(shader);
            }

            let nd_real_plan = FftPlan::build(
                FftConfig::new(vec![3, 8]).with_transform(crate::TransformKind::RealToComplex),
            )
            .unwrap();
            let nd_real = crate::NdRealFftIr::build(&nd_real_plan, device).unwrap();
            let nd_real_shaders = lowering.lower_nd_real_fft(&nd_real).unwrap();
            assert!(!nd_real_shaders.is_empty());
            for shader in &nd_real_shaders {
                assert_native_tokens(shader);
            }

            let r2r_plan = FftPlan::build(
                FftConfig::new(vec![9])
                    .with_transform(crate::TransformKind::Dct(crate::DctType::II)),
            )
            .unwrap();
            let r2r = crate::R2rIr::build(&r2r_plan, Direction::Forward, device).unwrap();
            assert_native_tokens(&lowering.lower_r2r(&r2r).unwrap());

            let nd_r2r_plan = FftPlan::build(
                FftConfig::new(vec![3, 4])
                    .with_transform(crate::TransformKind::Dst(crate::DstType::III)),
            )
            .unwrap();
            let nd_r2r = crate::NdR2rIr::build(&nd_r2r_plan, Direction::Forward, device).unwrap();
            let nd_r2r_shaders = lowering.lower_nd_r2r(&nd_r2r).unwrap();
            assert!(!nd_r2r_shaders.is_empty());
            for shader in &nd_r2r_shaders {
                assert_native_tokens(shader);
            }
        }
    }

    #[test]
    fn cuda_warp32_and_hip_runtime_wave_intrinsics_require_typed_rader_transpose_proof() {
        for (backend, vendor, subgroup_size, intrinsic) in [
            (Backend::Cuda, GpuVendor::Nvidia, 32usize, "__shfl_sync"),
            (Backend::Hip, GpuVendor::Amd, 32usize, "__shfl("),
            (Backend::Hip, GpuVendor::Amd, 64usize, "__shfl("),
        ] {
            let mut device = DeviceProfile::generic(backend, vendor);
            device.shared_memory_bytes = 48 * 1024;
            device.shared_memory_pow2_bytes = 32 * 1024;
            device.max_threads_per_block = 1024;
            device.subgroup = crate::SubgroupProfile {
                size: subgroup_size,
                min_size: subgroup_size,
                max_size: subgroup_size,
                required_size_compute_supported: false,
                compute_supported: true,
                basic_supported: true,
                shuffle_supported: true,
                shuffle_relative_supported: true,
                compute_full_subgroups: true,
            };
            let plan = FftPlan::build(FftConfig::new(vec![152])).unwrap();
            let mixed =
                crate::MixedRaderStockhamIr::build(&plan, Direction::Forward, device).unwrap();
            let crate::MixedPrimeRaderIr::FftConvolution(rader) = &mixed.prime_stage else {
                panic!("152 should use p19 FFT-Rader");
            };
            let shaders = NativeSourceBackend::new(backend)
                .lower_one_dim_fft(&rader.forward_fft)
                .unwrap();
            let subgroup = shaders
                .iter()
                .find(|shader| shader.source.contains(intrinsic))
                .expect("typed p19 x8 transpose proof should select native subgroup exchange");
            assert_eq!(subgroup.required_shared_memory_bytes, 0);
            assert!(subgroup.source.contains("vkfft_subgroup_shuffle_marker"));
            if backend == Backend::Cuda {
                assert!(subgroup.source.contains("__CUDACC_VER_MAJOR__ < 9"));
                assert!(subgroup.source.contains("!defined(__CUDACC_VER_MAJOR__)"));
                assert!(subgroup.source.contains("__shfl(value.x"));
                assert!(subgroup.source.contains("__activemask()"));
                assert!(subgroup.source.contains("__shfl_sync"));
            }
            if backend == Backend::Hip {
                let fallback = subgroup
                    .compiler_fallback_source
                    .as_deref()
                    .expect("HIP subgroup lowering must retain an OpenCL compiler fallback source");
                assert!(fallback.contains("cl_khr_subgroups : enable"));
                assert!(fallback.contains("sub_group_broadcast("));
                assert!(fallback.contains("get_sub_group_local_id()"));
                assert!(fallback.contains("get_sub_group_id()"));
                assert!(!fallback.contains("__builtin_amdgcn_readlane"));
                assert!(!fallback.contains("cl_khr_subgroup_shuffle"));
            }
            assert!(subgroup.source.contains(&format!(", {subgroup_size})")));
            assert!(
                subgroup
                    .source
                    .contains(&format!("& {}u", subgroup_size - 1))
            );
            assert!(
                subgroup
                    .source
                    .contains(&format!(">> {}u", subgroup_size.trailing_zeros()))
            );
        }
    }

    #[test]
    fn opencl_and_level_zero_subgroup_intrinsics_lower_only_for_proven_synthetic_profiles() {
        for (backend, vendor) in [
            (Backend::OpenCl, GpuVendor::Nvidia),
            (Backend::LevelZero, GpuVendor::Intel),
        ] {
            let mut device = DeviceProfile::generic(backend, vendor);
            device.shared_memory_bytes = 48 * 1024;
            device.shared_memory_pow2_bytes = 32 * 1024;
            device.max_threads_per_block = 1024;
            device.subgroup = crate::SubgroupProfile {
                size: 32,
                min_size: 32,
                max_size: 32,
                required_size_compute_supported: false,
                compute_supported: true,
                basic_supported: true,
                shuffle_supported: true,
                shuffle_relative_supported: true,
                compute_full_subgroups: true,
            };
            let plan = FftPlan::build(FftConfig::new(vec![152])).unwrap();
            let mixed =
                crate::MixedRaderStockhamIr::build(&plan, Direction::Forward, device).unwrap();
            let crate::MixedPrimeRaderIr::FftConvolution(rader) = &mixed.prime_stage else {
                panic!("152 should use p19 FFT-Rader");
            };
            let shaders = NativeSourceBackend::new(backend)
                .lower_one_dim_fft(&rader.forward_fft)
                .unwrap();
            let (shuffle_intrinsic, required_extension) = match backend {
                Backend::OpenCl => ("sub_group_shuffle(", "cl_khr_subgroups : enable"),
                Backend::LevelZero => ("intel_sub_group_shuffle(", "cl_intel_subgroups : enable"),
                _ => unreachable!("test only covers OpenCL-C subgroup dialects"),
            };
            let subgroup = shaders
                .iter()
                .find(|shader| shader.source.contains(shuffle_intrinsic))
                .unwrap_or_else(|| {
                    panic!(
                        "proven synthetic {backend:?} profile should lower typed subgroup exchange"
                    )
                });
            assert_eq!(subgroup.required_shared_memory_bytes, 0);
            assert!(subgroup.source.contains(required_extension));
            if backend == Backend::OpenCl {
                assert!(subgroup.source.contains("cl_khr_subgroup_shuffle : enable"));
                assert!(!subgroup.source.contains("intel_sub_group_shuffle("));
            } else {
                assert!(!subgroup.source.contains("cl_khr_subgroup_shuffle : enable"));
                assert!(!subgroup.source.contains("return (vec2)(sub_group_shuffle("));
            }
            assert!(subgroup.source.contains("get_sub_group_local_id()"));
            assert!(subgroup.source.contains("get_sub_group_id()"));
        }
    }

    #[test]
    fn high_level_transform_lowering_matches_program_ir_on_all_native_backends() {
        for (backend, vendor) in [
            (Backend::Cuda, GpuVendor::Nvidia),
            (Backend::Hip, GpuVendor::Amd),
            (Backend::OpenCl, GpuVendor::Amd),
            (Backend::LevelZero, GpuVendor::Intel),
            (Backend::Metal, GpuVendor::Apple),
        ] {
            let device = DeviceProfile::generic(backend, vendor);
            let lowering = NativeSourceBackend::new(backend);
            for config in [
                FftConfig::new(vec![103]),
                FftConfig::new(vec![16]).with_transform(crate::TransformKind::RealToComplex),
                FftConfig::new(vec![9])
                    .with_transform(crate::TransformKind::Dct(crate::DctType::II)),
            ] {
                let ir = crate::TransformIr::build(config, Direction::Forward, device).unwrap();
                let program = lowering.lower_transform(&ir).unwrap();
                program.validate().unwrap();
                assert_eq!(program.backend, backend);
                assert_eq!(program.shaders.len(), program.program.passes.len());
                for shader in &program.shaders {
                    assert_native_tokens(shader);
                }
            }
        }
    }

    #[test]
    fn all_native_dialects_r2r_zero_padding_survives_native_translation() {
        for (backend, vendor) in [
            (Backend::Cuda, GpuVendor::Nvidia),
            (Backend::Hip, GpuVendor::Amd),
            (Backend::OpenCl, GpuVendor::Amd),
            (Backend::LevelZero, GpuVendor::Intel),
            (Backend::Metal, GpuVendor::Apple),
        ] {
            let device = DeviceProfile::generic(backend, vendor);
            let base = FftConfig::new(vec![9])
                .with_batch_count(7)
                .with_transform(crate::TransformKind::Dct(crate::DctType::II))
                .with_grouped_batch(0, 3)
                .unwrap();
            for direction in [Direction::Forward, Direction::Inverse] {
                let config = base
                    .clone()
                    .with_inverse_normalization(direction == Direction::Inverse)
                    .with_zero_padding(0, 3, 6)
                    .unwrap();
                let plan = FftPlan::build(config).unwrap();
                let ir = crate::R2rIr::build(&plan, direction, device).unwrap();
                let shaders = NativeSourceBackend::new(backend)
                    .lower_r2r_program(&ir)
                    .unwrap();
                assert!(!shaders.is_empty());
                for shader in &shaders {
                    assert_native_tokens(shader);
                }
                match direction {
                    Direction::Forward => assert!(shaders.iter().any(|shader| {
                        shader.source.contains("VKFFT_ZERO_PAD_LEFT")
                            || shader.source.contains("vkfft_r2r_boundary_load")
                    })),
                    Direction::Inverse => assert!(shaders.iter().any(|shader| {
                        shader.source.contains("VKFFT_ZERO_PAD_LEFT")
                            || shader.source.contains("k >= VKFFT_ZERO_PAD_LEFT")
                    })),
                }
            }
        }
    }

    #[test]
    fn hip_and_level_zero_double_double_algorithm_families_lower_and_validate() {
        let one_dim = |length: usize, tuning: crate::PlannerTuning| {
            let plan = FftPlan::build(
                FftConfig::new(vec![length])
                    .with_precision(Precision::DoubleDouble)
                    .with_tuning(tuning),
            )
            .unwrap();
            crate::DoubleDoubleOneDimIr::build(&plan, Direction::Forward).unwrap()
        };
        let stockham = one_dim(16, crate::PlannerTuning::portable());
        assert!(matches!(stockham, crate::DoubleDoubleOneDimIr::Stockham(_)));
        let direct = one_dim(47, crate::PlannerTuning::portable());
        assert!(matches!(
            direct,
            crate::DoubleDoubleOneDimIr::DirectRader(_)
        ));
        let fft_rader = one_dim(257, crate::PlannerTuning::portable());
        assert!(matches!(
            fft_rader,
            crate::DoubleDoubleOneDimIr::FftRader(_)
        ));
        let mut bluestein_tuning = crate::PlannerTuning::portable();
        bluestein_tuning.max_rader_fft_prime = 100;
        let bluestein = one_dim(103, bluestein_tuning);
        assert!(matches!(
            bluestein,
            crate::DoubleDoubleOneDimIr::Bluestein(_)
        ));

        for (backend, vendor) in [
            (Backend::Hip, GpuVendor::Amd),
            (Backend::LevelZero, GpuVendor::Intel),
        ] {
            let mut device = DeviceProfile::generic(backend, vendor);
            device.supports_f64 = true;
            let lowering = NativeSourceBackend::new(backend);
            let mut transforms = vec![
                crate::TransformIr::Complex1dDoubleDouble(stockham.clone()),
                crate::TransformIr::Complex1dDoubleDouble(direct.clone()),
                crate::TransformIr::Complex1dDoubleDouble(fft_rader.clone()),
                crate::TransformIr::Complex1dDoubleDouble(bluestein.clone()),
            ];
            for config in [
                FftConfig::new(vec![3, 17])
                    .with_precision(Precision::DoubleDouble)
                    .with_tuning(crate::PlannerTuning::portable()),
                FftConfig::new(vec![15])
                    .with_precision(Precision::DoubleDouble)
                    .with_transform(crate::TransformKind::RealToComplex),
                FftConfig::new(vec![15])
                    .with_precision(Precision::DoubleDouble)
                    .with_transform(crate::TransformKind::Dct(crate::DctType::II)),
            ] {
                transforms
                    .push(crate::TransformIr::build(config, Direction::Forward, device).unwrap());
            }

            for transform in transforms {
                let program = lowering.lower_transform(&transform).unwrap();
                program.validate().unwrap();
                assert_eq!(program.backend, backend);
                assert_eq!(program.program.scalar, ScalarType::DoubleDouble);
                assert_eq!(program.shaders.len(), program.program.passes.len());
                assert!(!program.shaders.is_empty());
                for shader in &program.shaders {
                    assert_native_tokens(shader);
                }
            }
        }
    }

    #[test]
    fn four_native_backends_double_double_r2r_zero_padding_survives_translation() {
        for (backend, vendor) in [
            (Backend::Cuda, GpuVendor::Nvidia),
            (Backend::Hip, GpuVendor::Amd),
            (Backend::OpenCl, GpuVendor::Amd),
            (Backend::LevelZero, GpuVendor::Intel),
        ] {
            for precision in [Precision::DoubleDouble, Precision::DoubleDoubleF64Storage] {
                let device = DeviceProfile::generic(backend, vendor);
                let base = FftConfig::new(vec![9])
                    .with_batch_count(7)
                    .with_precision(precision)
                    .with_transform(crate::TransformKind::Dct(crate::DctType::II))
                    .with_grouped_batch(0, 3)
                    .unwrap();
                let forward_plan =
                    FftPlan::build(base.clone().with_zero_padding(0, 3, 6).unwrap()).unwrap();
                let forward = crate::DoubleDoubleR2rIr::build_for_device(
                    &forward_plan,
                    Direction::Forward,
                    device,
                )
                .unwrap();
                let program = NativeSourceBackend::new(backend)
                    .lower_double_double_r2r(&forward)
                    .unwrap();
                program.validate().unwrap();
                assert_eq!(program.program.passes.len(), 1);
                assert_eq!(program.program.passes.len(), program.shaders.len());
                let fused = program.shaders.first().unwrap();
                assert_native_tokens(fused);
                assert!(fused.source.contains("logical_source"));
                assert!(fused.source.contains("logical_source >= 3u"));
                assert!(fused.source.contains("logical_source < 6u"));

                let inverse_plan = FftPlan::build(
                    base.with_inverse_normalization(true)
                        .with_zero_padding(0, 3, 6)
                        .unwrap(),
                )
                .unwrap();
                let inverse = crate::DoubleDoubleR2rIr::build_for_device(
                    &inverse_plan,
                    Direction::Inverse,
                    device,
                )
                .unwrap();
                let inverse_program = NativeSourceBackend::new(backend)
                    .lower_double_double_r2r(&inverse)
                    .unwrap();
                inverse_program.validate().unwrap();
                assert_eq!(inverse_program.program.passes.len(), 1);
                assert_eq!(
                    inverse_program.program.passes.len(),
                    inverse_program.shaders.len()
                );
                let fused = inverse_program.shaders.first().unwrap();
                assert_native_tokens(fused);
                assert!(fused.source.contains("i >= 3u"));
                assert!(fused.source.contains("i < 6u"));
                assert!(fused.source.contains("real_value"));
            }
        }
    }

    #[test]
    fn four_native_backends_double_double_sources_keep_four_word_abi_and_disable_contraction() {
        for backend in [
            Backend::Cuda,
            Backend::Hip,
            Backend::OpenCl,
            Backend::LevelZero,
        ] {
            for precision in [Precision::DoubleDouble, Precision::DoubleDoubleF64Storage] {
                let plan = FftPlan::build(
                    FftConfig::new(vec![15])
                        .with_batch_count(2)
                        .with_precision(precision),
                )
                .unwrap();
                let ir = crate::DoubleDoubleStockhamIr::build(&plan, Direction::Forward).unwrap();
                let program = NativeSourceBackend::new(backend)
                    .lower_double_double_stockham(&ir)
                    .unwrap();
                program.validate().unwrap();
                assert_eq!(program.program.scalar, ScalarType::DoubleDouble);
                assert_eq!(program.shaders.len(), 1);
                let shader = &program.shaders[0];
                assert_native_tokens(shader);
                assert!(shader.source.contains("vkfft_dd_two_prod"));
                assert!(shader.source.contains("vkfft_dd_cmul"));
                assert!(shader.source.contains("dvec4"));
                assert!(!shader.source.contains("vkfft_twiddles.data"));
                match backend {
                    Backend::Cuda | Backend::Hip => {
                        assert!(shader.source.contains("struct dvec4"));
                        assert!(shader.source.contains("const dvec4* twiddle_lut"));
                        if precision == Precision::DoubleDouble {
                            assert!(shader.source.contains("const dvec4* inputs"));
                            assert!(shader.source.contains("dvec4* outputs"));
                        } else {
                            assert!(shader.source.contains("const dvec2* inputs"));
                            assert!(shader.source.contains("dvec2* outputs"));
                        }
                    }
                    Backend::OpenCl | Backend::LevelZero => {
                        assert!(shader.source.contains("#pragma OPENCL FP_CONTRACT OFF"));
                        assert!(shader.source.contains("typedef double4 dvec4"));
                        assert!(shader.source.contains("__global const dvec4* twiddle_lut"));
                        if precision == Precision::DoubleDouble {
                            assert!(shader.source.contains("__global const dvec4* inputs"));
                            assert!(shader.source.contains("__global dvec4* outputs"));
                        } else {
                            assert!(shader.source.contains("__global const dvec2* inputs"));
                            assert!(shader.source.contains("__global dvec2* outputs"));
                        }
                    }
                    _ => unreachable!(),
                }
            }
        }
    }

    #[test]
    fn four_native_backends_double_double_real_use_scalar_and_complex_argument_shapes() {
        for (backend, vendor) in [
            (Backend::Cuda, GpuVendor::Nvidia),
            (Backend::Hip, GpuVendor::Amd),
            (Backend::OpenCl, GpuVendor::Nvidia),
            (Backend::LevelZero, GpuVendor::Intel),
        ] {
            let mut device = DeviceProfile::generic(backend, vendor);
            device.supports_f64 = true;
            for precision in [Precision::DoubleDouble, Precision::DoubleDoubleF64Storage] {
                for transform in [
                    crate::TransformKind::RealToComplex,
                    crate::TransformKind::ComplexToReal,
                ] {
                    let direction = if transform == crate::TransformKind::RealToComplex {
                        Direction::Forward
                    } else {
                        Direction::Inverse
                    };
                    let ir = crate::TransformIr::build(
                        FftConfig::new(vec![15])
                            .with_transform(transform)
                            .with_precision(precision)
                            .with_inverse_normalization(
                                transform == crate::TransformKind::ComplexToReal,
                            ),
                        direction,
                        device,
                    )
                    .unwrap();
                    let crate::TransformIr::RealDoubleDouble(real) = &ir else {
                        panic!("DD real transform must preserve RealDoubleDouble IR");
                    };
                    let native = NativeSourceBackend::new(backend)
                        .lower_transform(&ir)
                        .unwrap();
                    native.validate().unwrap();
                    assert_eq!(native.shaders.len(), native.program.passes.len());
                    let first = &native.shaders.first().unwrap().source;
                    let last = &native.shaders.last().unwrap().source;
                    let input_prefix = match backend {
                        Backend::Cuda | Backend::Hip => "const ",
                        Backend::OpenCl | Backend::LevelZero => "__global const ",
                        _ => unreachable!(),
                    };
                    let output_prefix = match backend {
                        Backend::Cuda | Backend::Hip => "",
                        Backend::OpenCl | Backend::LevelZero => "__global ",
                        _ => unreachable!(),
                    };
                    match (precision, transform) {
                        (Precision::DoubleDouble, crate::TransformKind::RealToComplex) => {
                            assert!(first.contains(&format!("{input_prefix}dvec2* inputs")));
                            assert!(last.contains(&format!("{output_prefix}dvec4* outputs")));
                        }
                        (
                            Precision::DoubleDoubleF64Storage,
                            crate::TransformKind::RealToComplex,
                        ) => {
                            assert!(first.contains(&format!("{input_prefix}double* inputs")));
                            assert!(last.contains(&format!("{output_prefix}dvec2* outputs")));
                        }
                        (Precision::DoubleDouble, crate::TransformKind::ComplexToReal) => {
                            assert!(first.contains(&format!("{input_prefix}dvec4* inputs")));
                            assert!(last.contains(&format!("{output_prefix}dvec2* outputs")));
                        }
                        (
                            Precision::DoubleDoubleF64Storage,
                            crate::TransformKind::ComplexToReal,
                        ) => {
                            assert!(first.contains(&format!("{input_prefix}dvec2* inputs")));
                            assert!(last.contains(&format!("{output_prefix}double* outputs")));
                        }
                        _ => unreachable!(),
                    }
                    assert_eq!(real.length, 15);
                }
            }
        }
    }

    #[test]
    fn metal_f16_storage_keeps_packed_native_facade_and_f32_compute() {
        let device = DeviceProfile::generic(Backend::Metal, GpuVendor::Apple);
        let lower = |config: FftConfig, direction| {
            let ir = crate::TransformIr::build(config, direction, device).unwrap();
            let source = NativeSourceBackend::new(Backend::Metal)
                .lower_transform(&ir)
                .unwrap();
            source.validate().unwrap();
            assert_eq!(source.program.scalar, ScalarType::F32);
            assert_eq!(
                source.program.input_resource().unwrap().scalar,
                ScalarType::F16
            );
            assert_eq!(
                source.program.output_resource().unwrap().scalar,
                ScalarType::F16
            );
            // Native transform facades always hand ProgramIr Complex32/Complex64 vectors;
            // Real/R2R caller values are packed as (re, 0) before device submission and
            // unpacked after readback. F16 storage therefore remains packed complex `uint`
            // at the GPU external boundary for every transform family.
            assert_eq!(
                source.program.input_resource().unwrap().element_shape(),
                crate::program_ir::ProgramElementShape::Complex
            );
            assert_eq!(
                source.program.output_resource().unwrap().element_shape(),
                crate::program_ir::ProgramElementShape::Complex
            );
            assert!(
                source
                    .shaders
                    .iter()
                    .all(|shader| shader.scalar == ScalarType::F32)
            );
            source
        };
        let msl = |source: &NativeProgramSource| {
            source
                .shaders
                .iter()
                .map(|shader| shader.source.as_str())
                .collect::<Vec<_>>()
                .join("\n")
        };

        let c2c = lower(
            FftConfig::new(vec![64]).with_precision(Precision::F16StorageF32Compute),
            Direction::Forward,
        );
        let c2c_msl = msl(&c2c);
        assert!(c2c_msl.contains("vkfft_unpack_half2"));
        assert!(c2c_msl.contains("vkfft_pack_half2"));
        assert!(c2c_msl.contains("device const uint* inputs [[buffer(0)]]"));
        assert!(c2c_msl.contains("device uint* outputs [[buffer(1)]]"));
        assert!(!c2c_msl.contains("device const vec2* inputs [[buffer(0)]]"));

        let r2c = lower(
            FftConfig::new(vec![16])
                .with_precision(Precision::F16StorageF32Compute)
                .with_transform(crate::TransformKind::RealToComplex),
            Direction::Forward,
        );
        let r2c_msl = msl(&r2c);
        assert!(r2c_msl.contains("device const uint*"));
        assert!(r2c_msl.contains("device uint*"));
        assert!(r2c_msl.contains("vkfft_unpack_half2"));
        assert!(r2c_msl.contains("vkfft_pack_half2"));

        let c2r = lower(
            FftConfig::new(vec![16])
                .with_precision(Precision::F16StorageF32Compute)
                .with_transform(crate::TransformKind::ComplexToReal),
            Direction::Inverse,
        );
        let c2r_msl = msl(&c2r);
        assert!(c2r_msl.contains("device const uint*"));
        assert!(c2r_msl.contains("device uint*"));
        assert!(c2r_msl.contains("vkfft_unpack_half2"));
        assert!(c2r_msl.contains("vkfft_pack_half2"));

        let dct = lower(
            FftConfig::new(vec![9])
                .with_precision(Precision::F16StorageF32Compute)
                .with_transform(crate::TransformKind::Dct(crate::DctType::II)),
            Direction::Forward,
        );
        let dct_msl = msl(&dct);
        assert!(dct_msl.contains("device const uint*"));
        assert!(dct_msl.contains("device uint*"));
        assert!(dct_msl.contains("vkfft_unpack_half2"));
        assert!(dct_msl.contains("vkfft_pack_half2"));
    }

    #[test]
    fn f64_native_sources_use_target_vector_types_and_metal_rejects_f64() {
        for (backend, vendor) in [
            (Backend::Cuda, GpuVendor::Nvidia),
            (Backend::Hip, GpuVendor::Amd),
            (Backend::OpenCl, GpuVendor::Amd),
            (Backend::LevelZero, GpuVendor::Intel),
        ] {
            let mut device = DeviceProfile::generic(backend, vendor);
            device.supports_f64 = true;
            let plan =
                FftPlan::build(FftConfig::new(vec![64]).with_precision(Precision::F64)).unwrap();
            let kernel = KernelIr::stockham_1d(&plan, Direction::Forward, device).unwrap();
            let shader = NativeSourceBackend::new(backend).lower(&kernel).unwrap();
            assert_native_tokens(&shader);
            assert!(shader.source.contains("dvec2"));
        }

        let plan = FftPlan::build(FftConfig::new(vec![64]).with_precision(Precision::F64)).unwrap();
        let device = DeviceProfile {
            supports_f64: true,
            ..DeviceProfile::generic(Backend::Metal, GpuVendor::Apple)
        };
        let kernel = KernelIr::stockham_1d(&plan, Direction::Forward, device).unwrap();
        let error = NativeSourceBackend::new(Backend::Metal)
            .lower(&kernel)
            .unwrap_err();
        assert!(matches!(error, VkFftError::UnsupportedPrecision { .. }));
    }
}
