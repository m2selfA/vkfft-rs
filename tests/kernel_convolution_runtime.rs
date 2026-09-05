#![cfg(any(
    feature = "cuda-runtime",
    feature = "opencl-runtime",
    feature = "vulkan-runtime"
))]

use std::f64::consts::TAU;

use vkfft_rs::{
    Backend, Complex32, Complex64, DeviceProfile, Direction, FftConfig, GpuVendor, OneDimFftIr,
    Precision, ProgramIr, TransformIr, TransformKind, VkFftError,
};

const SMALL_N: usize = 16;
const PINNED_N: usize = 8192;
const SAMPLE50_PLANES: usize = 9;
const SAMPLE52_BATCHES: usize = 2;
const SAMPLE52_COORDINATES: usize = 2;
const SAMPLE52_SYSTEMS: usize = SAMPLE52_BATCHES * SAMPLE52_COORDINATES;
const SAMPLE52_ND_DIMENSIONS: [usize; 2] = [4, 8];
const SAMPLE51_DIMENSIONS: [usize; 3] = [4, 4, 8];
const SAMPLE51_PLANES: usize = 9;

fn kernel_config(n: usize, precision: Precision) -> FftConfig {
    FftConfig::new(vec![n])
        .with_precision(precision)
        .with_kernel_convolution(true)
}

fn sample50_kernel_config(precision: Precision) -> FftConfig {
    kernel_config(SMALL_N, precision).with_coordinate_features(SAMPLE50_PLANES)
}

fn real_kernel_config(n: usize, precision: Precision) -> FftConfig {
    kernel_config(n, precision).with_transform(TransformKind::RealToComplex)
}

fn independent_kernel_config(n: usize, precision: Precision) -> FftConfig {
    kernel_config(n, precision)
        .with_batch_count(SAMPLE52_BATCHES)
        .with_coordinate_features(SAMPLE52_COORDINATES)
}

fn independent_real_kernel_config(n: usize, precision: Precision) -> FftConfig {
    independent_kernel_config(n, precision).with_transform(TransformKind::RealToComplex)
}

fn sample52_nd_kernel_config(precision: Precision) -> FftConfig {
    FftConfig::new(SAMPLE52_ND_DIMENSIONS.to_vec())
        .with_precision(precision)
        .with_transform(TransformKind::RealToComplex)
        .with_kernel_convolution(true)
        .with_batch_count(SAMPLE52_BATCHES)
        .with_coordinate_features(SAMPLE52_COORDINATES)
}

fn sample51_kernel_config(precision: Precision) -> FftConfig {
    FftConfig::new(SAMPLE51_DIMENSIONS.to_vec())
        .with_precision(precision)
        .with_transform(TransformKind::RealToComplex)
        .with_kernel_convolution(true)
        .with_coordinate_features(SAMPLE51_PLANES)
        .with_zero_padding(0, 2, 4)
        .unwrap()
        .with_zero_padding(1, 2, 4)
        .unwrap()
        .with_zero_padding(2, 4, 8)
        .unwrap()
}

fn nvidia_vulkan_32k() -> DeviceProfile {
    DeviceProfile {
        shared_memory_bytes: 32 * 1024,
        shared_memory_pow2_bytes: 32 * 1024,
        max_threads_per_block: 1024,
        max_workgroup_size: [1024, 1024, 64],
        coalesced_memory_bytes: 32,
        ..DeviceProfile::generic(Backend::Vulkan, GpuVendor::Nvidia)
    }
}

fn input64() -> Vec<Complex64> {
    (0..SMALL_N)
        .map(|i| {
            let x = i as f64;
            Complex64::new(
                0.25 + 0.07 * x + (0.31 * x).sin(),
                -0.18 + 0.03 * x + (0.23 * x).cos(),
            )
        })
        .collect()
}

fn input32() -> Vec<Complex32> {
    input64()
        .iter()
        .map(|v| Complex32::new(v.re as f32, v.im as f32))
        .collect()
}

fn sample50_input64() -> Vec<Complex64> {
    let mut input = vec![Complex64::new(0.0, 0.0); SAMPLE50_PLANES * SMALL_N];
    for plane in 0..SAMPLE50_PLANES {
        let s = (plane + 1) as f64;
        input[plane * SMALL_N] = Complex64::new(0.25 * s, -0.125 * s);
    }
    input
}

fn sample50_input32() -> Vec<Complex32> {
    sample50_input64()
        .into_iter()
        .map(|value| Complex32::new(value.re as f32, value.im as f32))
        .collect()
}

fn sample50_expected64() -> Vec<Complex64> {
    (0..SAMPLE50_PLANES)
        .flat_map(|plane| {
            let s = (plane + 1) as f64;
            std::iter::repeat_n(Complex64::new(0.25 * s, -0.125 * s), SMALL_N)
        })
        .collect()
}

fn real_input64() -> Vec<f64> {
    (0..SMALL_N)
        .map(|i| {
            let x = i as f64;
            0.4 + 0.05 * x + (0.37 * x).sin() - 0.2 * (0.11 * x).cos()
        })
        .collect()
}

fn real_input32() -> Vec<f32> {
    real_input64().iter().map(|v| *v as f32).collect()
}

fn sample52_real_input64() -> Vec<f64> {
    (0..SAMPLE52_SYSTEMS)
        .flat_map(|system| {
            (0..SMALL_N).map(move |i| {
                let x = i as f64;
                let s = (system + 1) as f64;
                0.13 * s + 0.017 * s * x + (0.19 * s * x).sin()
                    - 0.11 * (0.07 * (s + 1.0) * x).cos()
            })
        })
        .collect()
}

fn sample52_real_input32() -> Vec<f32> {
    sample52_real_input64()
        .into_iter()
        .map(|value| value as f32)
        .collect()
}

fn sample52_real_expected64() -> Vec<Complex64> {
    sample52_real_input64()
        .chunks_exact(SMALL_N)
        .flat_map(direct_real_dft64)
        .collect()
}

fn sample52_nd_full_tensor_len() -> usize {
    SAMPLE52_ND_DIMENSIONS.iter().product()
}

fn sample52_nd_compact_tensor_len() -> usize {
    SAMPLE52_ND_DIMENSIONS[0] * (SAMPLE52_ND_DIMENSIONS[1] / 2 + 1)
}

fn sample52_nd_real_input64() -> Vec<f64> {
    let tensor_len = sample52_nd_full_tensor_len();
    let mut input = vec![0.0; SAMPLE52_SYSTEMS * tensor_len];
    for system in 0..SAMPLE52_SYSTEMS {
        input[system * tensor_len] = (system + 1) as f64 * 0.25;
    }
    input
}

fn sample52_nd_real_input32() -> Vec<f32> {
    sample52_nd_real_input64()
        .into_iter()
        .map(|value| value as f32)
        .collect()
}

fn sample52_nd_expected64() -> Vec<Complex64> {
    let compact_len = sample52_nd_compact_tensor_len();
    (0..SAMPLE52_SYSTEMS)
        .flat_map(|system| {
            let amplitude = (system + 1) as f64 * 0.25;
            std::iter::repeat_n(Complex64::new(amplitude, 0.0), compact_len)
        })
        .collect()
}

fn sample51_full_tensor_len() -> usize {
    SAMPLE51_DIMENSIONS.iter().product()
}

fn sample51_compact_tensor_len() -> usize {
    SAMPLE51_DIMENSIONS[0] * SAMPLE51_DIMENSIONS[1] * (SAMPLE51_DIMENSIONS[2] / 2 + 1)
}

fn sample51_real_input64() -> Vec<f64> {
    let tensor_len = sample51_full_tensor_len();
    let mut input = vec![0.0; SAMPLE51_PLANES * tensor_len];
    for plane in 0..SAMPLE51_PLANES {
        let base = plane * tensor_len;
        for a in 0..SAMPLE51_DIMENSIONS[0] {
            for b in 0..SAMPLE51_DIMENSIONS[1] {
                for c in 0..SAMPLE51_DIMENSIONS[2] {
                    let local = (a * SAMPLE51_DIMENSIONS[1] + b) * SAMPLE51_DIMENSIONS[2] + c;
                    if a >= 2 || b >= 2 || c >= 4 {
                        input[base + local] = 1_000.0 + 100.0 * plane as f64 + local as f64;
                    }
                }
            }
        }
        input[base] = (plane + 1) as f64 * 0.375;
    }
    input
}

fn sample51_real_input32() -> Vec<f32> {
    sample51_real_input64()
        .into_iter()
        .map(|value| value as f32)
        .collect()
}

fn sample51_expected64() -> Vec<Complex64> {
    let compact_len = sample51_compact_tensor_len();
    (0..SAMPLE51_PLANES)
        .flat_map(|plane| {
            let amplitude = (plane + 1) as f64 * 0.375;
            std::iter::repeat_n(Complex64::new(amplitude, 0.0), compact_len)
        })
        .collect()
}

fn direct_real_dft64(input: &[f64]) -> Vec<Complex64> {
    let complex = input
        .iter()
        .map(|value| Complex64::new(*value, 0.0))
        .collect::<Vec<_>>();
    direct_dft64(&complex)[..input.len() / 2 + 1].to_vec()
}

fn direct_dft64(input: &[Complex64]) -> Vec<Complex64> {
    let n = input.len();
    (0..n)
        .map(|k| {
            input
                .iter()
                .enumerate()
                .fold(Complex64::new(0.0, 0.0), |sum, (j, value)| {
                    let phase = -TAU * (j * k) as f64 / n as f64;
                    let root = Complex64::new(phase.cos(), phase.sin());
                    sum + *value * root
                })
        })
        .collect()
}

fn assert_close64(actual: &[Complex64], expected: &[Complex64], tolerance: f64, label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label} length mismatch");
    let max_error = actual
        .iter()
        .zip(expected)
        .map(|(a, e)| ((a.re - e.re).powi(2) + (a.im - e.im).powi(2)).sqrt())
        .fold(0.0f64, f64::max);
    assert!(max_error <= tolerance, "{label} max error {max_error:e}");
}

fn assert_close32(actual: &[Complex32], expected: &[Complex64], tolerance: f64, label: &str) {
    let widened = actual
        .iter()
        .map(|v| Complex64::new(v.re as f64, v.im as f64))
        .collect::<Vec<_>>();
    assert_close64(&widened, expected, tolerance, label);
}

#[test]
fn kernel_convolution_n8192_uses_pinned_capacity_without_application_midpoint() {
    let profile = nvidia_vulkan_32k();
    let transform = TransformIr::build(
        kernel_config(PINNED_N, Precision::F32),
        Direction::Forward,
        profile,
    )
    .unwrap();
    let TransformIr::Complex1d(OneDimFftIr::Recursive(recursive)) = &transform else {
        panic!("kernelConvolution N8192 must materialize recursive two-upload Stockham IR");
    };
    let four_step = recursive
        .four_step_plan
        .as_ref()
        .expect("kernelConvolution N8192 must retain Four-step upload metadata");
    let mut uploads = four_step.uploads.iter().collect::<Vec<_>>();
    uploads.sort_by_key(|upload| upload.axis_upload_id);
    assert_eq!(
        uploads
            .iter()
            .map(|upload| upload.fft_len)
            .collect::<Vec<_>>(),
        vec![128, 64]
    );
    for (upload, expected) in uploads
        .iter()
        .zip([(16usize, 16usize, true), (16usize, 8usize, false)])
    {
        let block = upload
            .axis_block
            .as_ref()
            .expect("kernelConvolution upload must retain physical AxisBlock");
        assert_eq!(
            (block.local_size_x, block.local_size_y),
            (expected.0, expected.1)
        );
        assert_eq!(block.grouped_batch, 16);
        assert!(block.transforms_on_x);
        assert_eq!(block.axis_swapped, expected.2);
    }

    let program = ProgramIr::one_dim_fft(match &transform {
        TransformIr::Complex1d(ir) => ir,
        _ => unreachable!(),
    })
    .unwrap();
    assert_eq!(
        program.passes.len(),
        2,
        "kernel preparation is forward FFT only"
    );
    assert!(
        program
            .resources
            .iter()
            .all(|resource| !resource.name.contains("convolution_kernel")),
        "kernel preparation must not bind an application kernel spectrum"
    );
    assert!(
        program
            .passes
            .iter()
            .all(|pass| !pass.name.contains("convolution")),
        "kernel preparation must not materialize ConvolutionMultiplyIr"
    );

    assert!(matches!(
        TransformIr::build(
            kernel_config(PINNED_N, Precision::F32),
            Direction::Inverse,
            profile,
        )
        .unwrap_err(),
        VkFftError::UnsupportedKernelPath(_)
    ));
}

#[test]
fn kernel_convolution_r2c_n16384_propagates_capacity_to_half_size_child() {
    let profile = nvidia_vulkan_32k();
    let transform = TransformIr::build(
        real_kernel_config(16_384, Precision::F32),
        Direction::Forward,
        profile,
    )
    .unwrap();
    let TransformIr::Real(real) = &transform else {
        panic!("R2C kernelConvolution must materialize RealFftIr");
    };
    assert_eq!(real.length, 16_384);
    assert_eq!(real.transform.logical_len(), 8_192);
    let OneDimFftIr::Recursive(recursive) = &real.transform else {
        panic!("R2C kernelConvolution half-size child must be recursive Stockham");
    };
    let four_step = recursive
        .four_step_plan
        .as_ref()
        .expect("R2C kernelConvolution N8192 child must use two uploads");
    let mut uploads = four_step.uploads.iter().collect::<Vec<_>>();
    uploads.sort_by_key(|upload| upload.axis_upload_id);
    assert_eq!(
        uploads
            .iter()
            .map(|upload| upload.fft_len)
            .collect::<Vec<_>>(),
        vec![128, 64]
    );
    let program = ProgramIr::real_fft(real).unwrap();
    assert!(
        program
            .resources
            .iter()
            .all(|resource| !resource.name.contains("convolution_kernel"))
    );
    assert!(
        program
            .passes
            .iter()
            .all(|pass| !pass.name.contains("convolution"))
    );
}

#[test]
fn kernel_convolution_sample50_c9_c2c_keeps_nine_independent_planes() {
    let profile = nvidia_vulkan_32k();
    let transform = TransformIr::build(
        sample50_kernel_config(Precision::F32),
        Direction::Forward,
        profile,
    )
    .unwrap();
    let TransformIr::Complex1d(ir) = &transform else {
        panic!("sample_50 kernel preparation must materialize one-dimensional C2C IR");
    };
    assert_eq!(ir.logical_len(), SMALL_N);
    assert_eq!(ir.batch_count(), SAMPLE50_PLANES);

    let program = ProgramIr::one_dim_fft(ir).unwrap();
    assert_eq!(program.resources[0].elements, SAMPLE50_PLANES * SMALL_N);
    assert_eq!(program.resources[1].elements, SAMPLE50_PLANES * SMALL_N);
    assert!(
        program
            .passes
            .iter()
            .all(|pass| !pass.name.contains("convolution_multiply"))
    );
    assert!(
        program
            .resources
            .iter()
            .all(|resource| !resource.name.contains("convolution_kernel"))
    );

    #[cfg(feature = "vulkan-runtime")]
    {
        let shaders = vkfft_rs::backend::vulkan::VulkanGlslBackend
            .lower_one_dim_fft(ir)
            .unwrap();
        assert_eq!(shaders.len(), program.passes.len());
        for shader in shaders {
            assert_eq!(shader.compile_spirv().unwrap().words[0], 0x0723_0203);
        }
    }
}

#[test]
fn kernel_convolution_sample52_batch_coordinate_systems_keep_pinned_capacity_and_compact_r2c() {
    let profile = nvidia_vulkan_32k();
    let c2c = TransformIr::build(
        independent_kernel_config(PINNED_N, Precision::F32),
        Direction::Forward,
        profile,
    )
    .unwrap();
    let TransformIr::Complex1d(OneDimFftIr::Recursive(c2c_recursive)) = &c2c else {
        panic!("sample_52 C2C kernel preparation must use recursive two-upload Stockham");
    };
    assert_eq!(c2c_recursive.batch_count, SAMPLE52_SYSTEMS);
    let c2c_four_step = c2c_recursive
        .four_step_plan
        .as_ref()
        .expect("sample_52 C2C kernel preparation must retain Four-step metadata");
    let mut c2c_uploads = c2c_four_step.uploads.iter().collect::<Vec<_>>();
    c2c_uploads.sort_by_key(|upload| upload.axis_upload_id);
    assert_eq!(
        c2c_uploads
            .iter()
            .map(|upload| (upload.fft_len, upload.transform_count))
            .collect::<Vec<_>>(),
        vec![(128, 256), (64, 512)]
    );
    for (upload, expected) in c2c_uploads
        .iter()
        .zip([(16usize, 16usize, true), (16usize, 8usize, false)])
    {
        let block = upload
            .axis_block
            .as_ref()
            .expect("sample_52 kernel-preparation upload must retain AxisBlock");
        assert_eq!(
            (block.local_size_x, block.local_size_y),
            (expected.0, expected.1)
        );
        assert_eq!(block.grouped_batch, 16);
        assert!(block.transforms_on_x);
        assert_eq!(block.axis_swapped, expected.2);
    }
    let c2c_program = ProgramIr::one_dim_fft(match &c2c {
        TransformIr::Complex1d(ir) => ir,
        _ => unreachable!(),
    })
    .unwrap();
    assert_eq!(
        c2c_program.resources[0].elements,
        SAMPLE52_SYSTEMS * PINNED_N
    );
    assert_eq!(
        c2c_program.resources[1].elements,
        SAMPLE52_SYSTEMS * PINNED_N
    );
    assert!(
        c2c_program
            .resources
            .iter()
            .all(|resource| !resource.name.contains("convolution_kernel"))
    );

    let real_len = PINNED_N * 2;
    let r2c = TransformIr::build(
        independent_real_kernel_config(real_len, Precision::F32),
        Direction::Forward,
        profile,
    )
    .unwrap();
    let TransformIr::Real(real) = &r2c else {
        panic!("sample_52 R2C kernel preparation must materialize RealFftIr");
    };
    assert_eq!(real.batch_count, SAMPLE52_SYSTEMS);
    assert_eq!(real.transform.logical_len(), PINNED_N);
    let OneDimFftIr::Recursive(real_recursive) = &real.transform else {
        panic!("sample_52 R2C half-size child must use recursive two-upload Stockham");
    };
    assert_eq!(real_recursive.batch_count, SAMPLE52_SYSTEMS);
    let real_four_step = real_recursive
        .four_step_plan
        .as_ref()
        .expect("sample_52 R2C half-size child must retain Four-step metadata");
    let mut real_uploads = real_four_step.uploads.iter().collect::<Vec<_>>();
    real_uploads.sort_by_key(|upload| upload.axis_upload_id);
    assert_eq!(
        real_uploads
            .iter()
            .map(|upload| (upload.fft_len, upload.transform_count))
            .collect::<Vec<_>>(),
        vec![(128, 256), (64, 512)]
    );

    let real_program = ProgramIr::real_fft(real).unwrap();
    assert_eq!(
        real_program.resources[0].elements,
        SAMPLE52_SYSTEMS * real_len
    );
    assert_eq!(
        real_program.resources[1].elements,
        SAMPLE52_SYSTEMS * (real_len / 2 + 1)
    );
    assert!(
        real_program
            .resources
            .iter()
            .all(|resource| !resource.name.contains("convolution_kernel"))
    );
    assert!(
        real_program
            .passes
            .iter()
            .all(|pass| !pass.name.contains("convolution_multiply"))
    );

    #[cfg(feature = "vulkan-runtime")]
    {
        let shaders = vkfft_rs::backend::vulkan::VulkanGlslBackend
            .lower_real_fft(real)
            .unwrap();
        assert_eq!(shaders.len(), real_program.passes.len());
        for shader in shaders {
            assert_eq!(shader.compile_spirv().unwrap().words[0], 0x0723_0203);
        }
    }
}

#[test]
fn kernel_convolution_sample52_2d_b2c2_keeps_four_independent_compact_spectra() {
    let profile = nvidia_vulkan_32k();
    let transform = TransformIr::build(
        sample52_nd_kernel_config(Precision::F32),
        Direction::Forward,
        profile,
    )
    .unwrap();
    let TransformIr::RealNd(ir) = &transform else {
        panic!("sample_52 2D kernel preparation must materialize NdRealFftIr");
    };
    assert_eq!(ir.batch_count, SAMPLE52_SYSTEMS);
    assert_eq!(ir.dimensions, SAMPLE52_ND_DIMENSIONS);
    assert_eq!(ir.full_tensor_len, sample52_nd_full_tensor_len());
    assert_eq!(ir.compact_tensor_len, sample52_nd_compact_tensor_len());
    assert!(ir.zero_pad_pass.is_none());
    assert_eq!(
        ir.real_axis.batch_count,
        SAMPLE52_SYSTEMS * SAMPLE52_ND_DIMENSIONS[0]
    );
    assert!(
        ir.complex_axes
            .iter()
            .all(|axis| axis.pack.batch_count == SAMPLE52_SYSTEMS
                && axis.scatter.batch_count == SAMPLE52_SYSTEMS)
    );

    let program = ProgramIr::nd_real_fft(ir).unwrap();
    assert_eq!(
        program.resources[0].elements,
        SAMPLE52_SYSTEMS * sample52_nd_full_tensor_len()
    );
    assert_eq!(
        program.resources[1].elements,
        SAMPLE52_SYSTEMS * sample52_nd_compact_tensor_len()
    );
    assert!(
        program
            .passes
            .iter()
            .all(|pass| !pass.name.contains("convolution_multiply"))
    );
    assert!(
        program
            .resources
            .iter()
            .all(|resource| !resource.name.contains("convolution_kernel"))
    );

    #[cfg(feature = "vulkan-runtime")]
    {
        let shaders = vkfft_rs::backend::vulkan::VulkanGlslBackend
            .lower_nd_real_fft(ir)
            .unwrap();
        assert_eq!(shaders.len(), program.passes.len());
        for shader in shaders {
            assert_eq!(shader.compile_spirv().unwrap().words[0], 0x0723_0203);
        }
    }
}

#[test]
fn kernel_convolution_sample51_c9_nd_padding_keeps_plane_and_true_boundary_ownership() {
    let profile = nvidia_vulkan_32k();
    let transform = TransformIr::build(
        sample51_kernel_config(Precision::F32),
        Direction::Forward,
        profile,
    )
    .unwrap();
    let TransformIr::RealNd(ir) = &transform else {
        panic!("sample_51 kernel preparation must materialize NdRealFftIr");
    };
    assert_eq!(ir.batch_count, SAMPLE51_PLANES);
    assert_eq!(ir.dimensions, SAMPLE51_DIMENSIONS);
    assert_eq!(ir.full_tensor_len, sample51_full_tensor_len());
    assert_eq!(ir.compact_tensor_len, sample51_compact_tensor_len());
    assert_eq!(
        ir.zero_pad_pass
            .as_ref()
            .expect("sample_51 kernel preparation must retain spatial zero padding")
            .batch_count,
        SAMPLE51_PLANES
    );
    assert_eq!(
        ir.real_axis.batch_count,
        SAMPLE51_PLANES * SAMPLE51_DIMENSIONS[0] * SAMPLE51_DIMENSIONS[1]
    );
    assert!(
        ir.complex_axes
            .iter()
            .all(|axis| axis.pack.batch_count == SAMPLE51_PLANES
                && axis.scatter.batch_count == SAMPLE51_PLANES)
    );

    let program = ProgramIr::nd_real_fft(ir).unwrap();
    assert_eq!(
        program.resources[0].elements,
        SAMPLE51_PLANES * sample51_full_tensor_len()
    );
    assert_eq!(
        program.resources[1].elements,
        SAMPLE51_PLANES * sample51_compact_tensor_len()
    );
    assert!(
        program
            .passes
            .iter()
            .all(|pass| !pass.name.contains("convolution_multiply"))
    );
    assert!(
        program
            .resources
            .iter()
            .all(|resource| !resource.name.contains("convolution_kernel"))
    );

    #[cfg(feature = "vulkan-runtime")]
    {
        let shaders = vkfft_rs::backend::vulkan::VulkanGlslBackend
            .lower_nd_real_fft(ir)
            .unwrap();
        assert_eq!(shaders.len(), program.passes.len());
        for shader in shaders {
            assert_eq!(shader.compile_spirv().unwrap().words[0], 0x0723_0203);
        }
    }
}

#[cfg(any(feature = "cuda-runtime", feature = "opencl-runtime"))]
fn run_native<R: vkfft_rs::backend::NativeRuntime>(runtime: &R) {
    use vkfft_rs::backend::{
        NativeTransformInput32, NativeTransformInput64, NativeTransformOutput32,
        NativeTransformOutput64,
    };

    let expected = direct_dft64(&input64());
    let f32 = TransformIr::build(
        kernel_config(SMALL_N, Precision::F32),
        Direction::Forward,
        runtime.device_profile(),
    )
    .unwrap();
    let actual32 = match runtime
        .execute_transform_f32(&f32, NativeTransformInput32::Complex(&input32()))
        .unwrap()
    {
        NativeTransformOutput32::Complex(values) => values,
        NativeTransformOutput32::Real(_) => panic!("kernelConvolution C2C returned real output"),
    };
    assert_close32(&actual32, &expected, 3.0e-4, runtime.device_name());

    if runtime.device_profile().supports_f64 {
        let f64 = TransformIr::build(
            kernel_config(SMALL_N, Precision::F64),
            Direction::Forward,
            runtime.device_profile(),
        )
        .unwrap();
        let actual64 = match runtime
            .execute_transform_f64(&f64, NativeTransformInput64::Complex(&input64()))
            .unwrap()
        {
            NativeTransformOutput64::Complex(values) => values,
            NativeTransformOutput64::Real(_) => {
                panic!("kernelConvolution C2C returned real output")
            }
        };
        assert_close64(&actual64, &expected, 3.0e-10, runtime.device_name());
    }

    let real_expected = direct_real_dft64(&real_input64());
    let r2c32 = TransformIr::build(
        real_kernel_config(SMALL_N, Precision::F32),
        Direction::Forward,
        runtime.device_profile(),
    )
    .unwrap();
    let actual_r2c32 = match runtime
        .execute_transform_f32(&r2c32, NativeTransformInput32::Real(&real_input32()))
        .unwrap()
    {
        NativeTransformOutput32::Complex(values) => values,
        NativeTransformOutput32::Real(_) => panic!("kernelConvolution R2C returned real output"),
    };
    assert_close32(
        &actual_r2c32,
        &real_expected,
        3.0e-4,
        &format!("{} R2C F32", runtime.device_name()),
    );
    if runtime.device_profile().supports_f64 {
        let r2c64 = TransformIr::build(
            real_kernel_config(SMALL_N, Precision::F64),
            Direction::Forward,
            runtime.device_profile(),
        )
        .unwrap();
        let actual_r2c64 = match runtime
            .execute_transform_f64(&r2c64, NativeTransformInput64::Real(&real_input64()))
            .unwrap()
        {
            NativeTransformOutput64::Complex(values) => values,
            NativeTransformOutput64::Real(_) => {
                panic!("kernelConvolution R2C returned real output")
            }
        };
        assert_close64(
            &actual_r2c64,
            &real_expected,
            3.0e-10,
            &format!("{} R2C F64", runtime.device_name()),
        );
    }

    let sample50_expected = sample50_expected64();
    let sample50_c2c32 = TransformIr::build(
        sample50_kernel_config(Precision::F32),
        Direction::Forward,
        runtime.device_profile(),
    )
    .unwrap();
    let sample50_actual32 = match runtime
        .execute_transform_f32(
            &sample50_c2c32,
            NativeTransformInput32::Complex(&sample50_input32()),
        )
        .unwrap()
    {
        NativeTransformOutput32::Complex(values) => values,
        NativeTransformOutput32::Real(_) => {
            panic!("sample_50 kernelConvolution C2C returned real output")
        }
    };
    assert_close32(
        &sample50_actual32,
        &sample50_expected,
        5.0e-4,
        &format!("{} sample_50 C9 C2C F32", runtime.device_name()),
    );
    if runtime.device_profile().supports_f64 {
        let sample50_c2c64 = TransformIr::build(
            sample50_kernel_config(Precision::F64),
            Direction::Forward,
            runtime.device_profile(),
        )
        .unwrap();
        let sample50_input64 = sample50_input64();
        let sample50_actual64 = match runtime
            .execute_transform_f64(
                &sample50_c2c64,
                NativeTransformInput64::Complex(&sample50_input64),
            )
            .unwrap()
        {
            NativeTransformOutput64::Complex(values) => values,
            NativeTransformOutput64::Real(_) => {
                panic!("sample_50 F64 kernelConvolution C2C returned real output")
            }
        };
        assert_close64(
            &sample50_actual64,
            &sample50_expected,
            5.0e-10,
            &format!("{} sample_50 C9 C2C F64", runtime.device_name()),
        );
    }

    let sample52_expected = sample52_real_expected64();
    let sample52_r2c32 = TransformIr::build(
        independent_real_kernel_config(SMALL_N, Precision::F32),
        Direction::Forward,
        runtime.device_profile(),
    )
    .unwrap();
    let sample52_actual32 = match runtime
        .execute_transform_f32(
            &sample52_r2c32,
            NativeTransformInput32::Real(&sample52_real_input32()),
        )
        .unwrap()
    {
        NativeTransformOutput32::Complex(values) => values,
        NativeTransformOutput32::Real(_) => {
            panic!("sample_52 kernelConvolution R2C returned real output")
        }
    };
    assert_close32(
        &sample52_actual32,
        &sample52_expected,
        5.0e-4,
        &format!("{} sample_52 B2C2 R2C F32", runtime.device_name()),
    );
    if runtime.device_profile().supports_f64 {
        let sample52_r2c64 = TransformIr::build(
            independent_real_kernel_config(SMALL_N, Precision::F64),
            Direction::Forward,
            runtime.device_profile(),
        )
        .unwrap();
        let sample52_input64 = sample52_real_input64();
        let sample52_actual64 = match runtime
            .execute_transform_f64(
                &sample52_r2c64,
                NativeTransformInput64::Real(&sample52_input64),
            )
            .unwrap()
        {
            NativeTransformOutput64::Complex(values) => values,
            NativeTransformOutput64::Real(_) => {
                panic!("sample_52 F64 kernelConvolution R2C returned real output")
            }
        };
        assert_close64(
            &sample52_actual64,
            &sample52_expected,
            5.0e-10,
            &format!("{} sample_52 B2C2 R2C F64", runtime.device_name()),
        );
    }

    let sample52_nd_expected = sample52_nd_expected64();
    let sample52_nd_r2c32 = TransformIr::build(
        sample52_nd_kernel_config(Precision::F32),
        Direction::Forward,
        runtime.device_profile(),
    )
    .unwrap();
    let sample52_nd_actual32 = match runtime
        .execute_transform_f32(
            &sample52_nd_r2c32,
            NativeTransformInput32::Real(&sample52_nd_real_input32()),
        )
        .unwrap()
    {
        NativeTransformOutput32::Complex(values) => values,
        NativeTransformOutput32::Real(_) => {
            panic!("sample_52 2D kernelConvolution R2C returned real output")
        }
    };
    assert_close32(
        &sample52_nd_actual32,
        &sample52_nd_expected,
        5.0e-4,
        &format!("{} sample_52 2D B2C2 R2C F32", runtime.device_name()),
    );
    if runtime.device_profile().supports_f64 {
        let sample52_nd_r2c64 = TransformIr::build(
            sample52_nd_kernel_config(Precision::F64),
            Direction::Forward,
            runtime.device_profile(),
        )
        .unwrap();
        let sample52_nd_input64 = sample52_nd_real_input64();
        let sample52_nd_actual64 = match runtime
            .execute_transform_f64(
                &sample52_nd_r2c64,
                NativeTransformInput64::Real(&sample52_nd_input64),
            )
            .unwrap()
        {
            NativeTransformOutput64::Complex(values) => values,
            NativeTransformOutput64::Real(_) => {
                panic!("sample_52 2D F64 kernelConvolution R2C returned real output")
            }
        };
        assert_close64(
            &sample52_nd_actual64,
            &sample52_nd_expected,
            5.0e-10,
            &format!("{} sample_52 2D B2C2 R2C F64", runtime.device_name()),
        );
    }

    let sample51_expected = sample51_expected64();
    let sample51_r2c32 = TransformIr::build(
        sample51_kernel_config(Precision::F32),
        Direction::Forward,
        runtime.device_profile(),
    )
    .unwrap();
    let sample51_actual32 = match runtime
        .execute_transform_f32(
            &sample51_r2c32,
            NativeTransformInput32::Real(&sample51_real_input32()),
        )
        .unwrap()
    {
        NativeTransformOutput32::Complex(values) => values,
        NativeTransformOutput32::Real(_) => {
            panic!("sample_51 kernelConvolution R2C returned real output")
        }
    };
    assert_close32(
        &sample51_actual32,
        &sample51_expected,
        7.0e-4,
        &format!("{} sample_51 C9 padded ND R2C F32", runtime.device_name()),
    );
    if runtime.device_profile().supports_f64 {
        let sample51_r2c64 = TransformIr::build(
            sample51_kernel_config(Precision::F64),
            Direction::Forward,
            runtime.device_profile(),
        )
        .unwrap();
        let sample51_input64 = sample51_real_input64();
        let sample51_actual64 = match runtime
            .execute_transform_f64(
                &sample51_r2c64,
                NativeTransformInput64::Real(&sample51_input64),
            )
            .unwrap()
        {
            NativeTransformOutput64::Complex(values) => values,
            NativeTransformOutput64::Real(_) => {
                panic!("sample_51 F64 kernelConvolution R2C returned real output")
            }
        };
        assert_close64(
            &sample51_actual64,
            &sample51_expected,
            7.0e-10,
            &format!("{} sample_51 C9 padded ND R2C F64", runtime.device_name()),
        );
    }
}

fn gpu_test_lock() -> &'static std::sync::Mutex<()> {
    static LOCK: std::sync::OnceLock<std::sync::Mutex<()>> = std::sync::OnceLock::new();
    LOCK.get_or_init(|| std::sync::Mutex::new(()))
}

#[cfg(feature = "cuda-runtime")]
#[test]
fn cuda_kernel_convolution_forward_matches_direct_dft_or_skips() {
    use vkfft_rs::backend::cuda::runtime::CudaExecutionContext;

    let _guard = gpu_test_lock().lock().unwrap();
    let availability = CudaExecutionContext::probe();
    if !availability.compiler_available || availability.device_count == 0 {
        return;
    }
    let runtime = CudaExecutionContext::new(0).expect("CUDA context failed after successful probe");
    run_native(&runtime);
}

#[cfg(feature = "opencl-runtime")]
#[test]
fn opencl_kernel_convolution_forward_matches_direct_dft_or_skips() {
    use vkfft_rs::backend::opencl::runtime::OpenClExecutionContext;

    let _guard = gpu_test_lock().lock().unwrap();
    let availability = OpenClExecutionContext::probe();
    if !availability.compiler_available || availability.device_count == 0 {
        return;
    }
    let runtime =
        OpenClExecutionContext::new(0).expect("OpenCL context failed after successful probe");
    run_native(&runtime);
}

#[cfg(feature = "vulkan-runtime")]
#[test]
fn vulkan_kernel_convolution_forward_matches_direct_dft_or_skips() {
    use vkfft_rs::backend::vulkan::runtime::{
        TransformInput32, TransformInput64, TransformOutput32, TransformOutput64,
        VulkanExecutionContext,
    };

    let _guard = gpu_test_lock().lock().unwrap();
    let runtime = match VulkanExecutionContext::new() {
        Ok(runtime) => runtime,
        Err(VkFftError::VulkanUnavailable(_)) => return,
        Err(error) => panic!("Vulkan context failed after loader/device discovery: {error}"),
    };
    let expected = direct_dft64(&input64());
    let f32 = TransformIr::build(
        kernel_config(SMALL_N, Precision::F32),
        Direction::Forward,
        runtime.device_profile(),
    )
    .unwrap();
    let actual32 = match runtime
        .execute_transform_f32(&f32, TransformInput32::Complex(&input32()))
        .unwrap()
    {
        TransformOutput32::Complex(values) => values,
        TransformOutput32::Real(_) => panic!("kernelConvolution C2C returned real output"),
    };
    assert_close32(&actual32, &expected, 3.0e-4, "Vulkan F32");

    if runtime.device_profile().supports_f64 {
        let f64 = TransformIr::build(
            kernel_config(SMALL_N, Precision::F64),
            Direction::Forward,
            runtime.device_profile(),
        )
        .unwrap();
        let actual64 = match runtime
            .execute_transform_f64(&f64, TransformInput64::Complex(&input64()))
            .unwrap()
        {
            TransformOutput64::Complex(values) => values,
            TransformOutput64::Real(_) => panic!("kernelConvolution C2C returned real output"),
        };
        assert_close64(&actual64, &expected, 3.0e-10, "Vulkan F64");
    }

    let real_expected = direct_real_dft64(&real_input64());
    let r2c32 = TransformIr::build(
        real_kernel_config(SMALL_N, Precision::F32),
        Direction::Forward,
        runtime.device_profile(),
    )
    .unwrap();
    let actual_r2c32 = match runtime
        .execute_transform_f32(&r2c32, TransformInput32::Real(&real_input32()))
        .unwrap()
    {
        TransformOutput32::Complex(values) => values,
        TransformOutput32::Real(_) => panic!("kernelConvolution R2C returned real output"),
    };
    assert_close32(&actual_r2c32, &real_expected, 3.0e-4, "Vulkan R2C F32");
    if runtime.device_profile().supports_f64 {
        let r2c64 = TransformIr::build(
            real_kernel_config(SMALL_N, Precision::F64),
            Direction::Forward,
            runtime.device_profile(),
        )
        .unwrap();
        let actual_r2c64 = match runtime
            .execute_transform_f64(&r2c64, TransformInput64::Real(&real_input64()))
            .unwrap()
        {
            TransformOutput64::Complex(values) => values,
            TransformOutput64::Real(_) => panic!("kernelConvolution R2C returned real output"),
        };
        assert_close64(&actual_r2c64, &real_expected, 3.0e-10, "Vulkan R2C F64");
    }

    let sample50_expected = sample50_expected64();
    let sample50_c2c32 = TransformIr::build(
        sample50_kernel_config(Precision::F32),
        Direction::Forward,
        runtime.device_profile(),
    )
    .unwrap();
    let sample50_actual32 = match runtime
        .execute_transform_f32(
            &sample50_c2c32,
            TransformInput32::Complex(&sample50_input32()),
        )
        .unwrap()
    {
        TransformOutput32::Complex(values) => values,
        TransformOutput32::Real(_) => {
            panic!("sample_50 Vulkan kernelConvolution C2C returned real output")
        }
    };
    assert_close32(
        &sample50_actual32,
        &sample50_expected,
        5.0e-4,
        "Vulkan sample_50 C9 C2C F32",
    );
    if runtime.device_profile().supports_f64 {
        let sample50_c2c64 = TransformIr::build(
            sample50_kernel_config(Precision::F64),
            Direction::Forward,
            runtime.device_profile(),
        )
        .unwrap();
        let sample50_input64 = sample50_input64();
        let sample50_actual64 = match runtime
            .execute_transform_f64(
                &sample50_c2c64,
                TransformInput64::Complex(&sample50_input64),
            )
            .unwrap()
        {
            TransformOutput64::Complex(values) => values,
            TransformOutput64::Real(_) => {
                panic!("sample_50 Vulkan F64 kernelConvolution C2C returned real output")
            }
        };
        assert_close64(
            &sample50_actual64,
            &sample50_expected,
            5.0e-10,
            "Vulkan sample_50 C9 C2C F64",
        );
    }

    let sample52_expected = sample52_real_expected64();
    let sample52_r2c32 = TransformIr::build(
        independent_real_kernel_config(SMALL_N, Precision::F32),
        Direction::Forward,
        runtime.device_profile(),
    )
    .unwrap();
    let sample52_actual32 = match runtime
        .execute_transform_f32(
            &sample52_r2c32,
            TransformInput32::Real(&sample52_real_input32()),
        )
        .unwrap()
    {
        TransformOutput32::Complex(values) => values,
        TransformOutput32::Real(_) => {
            panic!("sample_52 Vulkan kernelConvolution R2C returned real output")
        }
    };
    assert_close32(
        &sample52_actual32,
        &sample52_expected,
        5.0e-4,
        "Vulkan sample_52 B2C2 R2C F32",
    );
    if runtime.device_profile().supports_f64 {
        let sample52_r2c64 = TransformIr::build(
            independent_real_kernel_config(SMALL_N, Precision::F64),
            Direction::Forward,
            runtime.device_profile(),
        )
        .unwrap();
        let sample52_input64 = sample52_real_input64();
        let sample52_actual64 = match runtime
            .execute_transform_f64(&sample52_r2c64, TransformInput64::Real(&sample52_input64))
            .unwrap()
        {
            TransformOutput64::Complex(values) => values,
            TransformOutput64::Real(_) => {
                panic!("sample_52 Vulkan F64 kernelConvolution R2C returned real output")
            }
        };
        assert_close64(
            &sample52_actual64,
            &sample52_expected,
            5.0e-10,
            "Vulkan sample_52 B2C2 R2C F64",
        );
    }

    let sample52_nd_expected = sample52_nd_expected64();
    let sample52_nd_r2c32 = TransformIr::build(
        sample52_nd_kernel_config(Precision::F32),
        Direction::Forward,
        runtime.device_profile(),
    )
    .unwrap();
    let sample52_nd_actual32 = match runtime
        .execute_transform_f32(
            &sample52_nd_r2c32,
            TransformInput32::Real(&sample52_nd_real_input32()),
        )
        .unwrap()
    {
        TransformOutput32::Complex(values) => values,
        TransformOutput32::Real(_) => {
            panic!("sample_52 2D Vulkan kernelConvolution R2C returned real output")
        }
    };
    assert_close32(
        &sample52_nd_actual32,
        &sample52_nd_expected,
        5.0e-4,
        "Vulkan sample_52 2D B2C2 R2C F32",
    );
    if runtime.device_profile().supports_f64 {
        let sample52_nd_r2c64 = TransformIr::build(
            sample52_nd_kernel_config(Precision::F64),
            Direction::Forward,
            runtime.device_profile(),
        )
        .unwrap();
        let sample52_nd_input64 = sample52_nd_real_input64();
        let sample52_nd_actual64 = match runtime
            .execute_transform_f64(
                &sample52_nd_r2c64,
                TransformInput64::Real(&sample52_nd_input64),
            )
            .unwrap()
        {
            TransformOutput64::Complex(values) => values,
            TransformOutput64::Real(_) => {
                panic!("sample_52 2D Vulkan F64 kernelConvolution R2C returned real output")
            }
        };
        assert_close64(
            &sample52_nd_actual64,
            &sample52_nd_expected,
            5.0e-10,
            "Vulkan sample_52 2D B2C2 R2C F64",
        );
    }

    let sample51_expected = sample51_expected64();
    let sample51_r2c32 = TransformIr::build(
        sample51_kernel_config(Precision::F32),
        Direction::Forward,
        runtime.device_profile(),
    )
    .unwrap();
    let sample51_actual32 = match runtime
        .execute_transform_f32(
            &sample51_r2c32,
            TransformInput32::Real(&sample51_real_input32()),
        )
        .unwrap()
    {
        TransformOutput32::Complex(values) => values,
        TransformOutput32::Real(_) => {
            panic!("sample_51 Vulkan kernelConvolution R2C returned real output")
        }
    };
    assert_close32(
        &sample51_actual32,
        &sample51_expected,
        7.0e-4,
        "Vulkan sample_51 C9 padded ND R2C F32",
    );
    if runtime.device_profile().supports_f64 {
        let sample51_r2c64 = TransformIr::build(
            sample51_kernel_config(Precision::F64),
            Direction::Forward,
            runtime.device_profile(),
        )
        .unwrap();
        let sample51_input64 = sample51_real_input64();
        let sample51_actual64 = match runtime
            .execute_transform_f64(&sample51_r2c64, TransformInput64::Real(&sample51_input64))
            .unwrap()
        {
            TransformOutput64::Complex(values) => values,
            TransformOutput64::Real(_) => {
                panic!("sample_51 Vulkan F64 kernelConvolution R2C returned real output")
            }
        };
        assert_close64(
            &sample51_actual64,
            &sample51_expected,
            7.0e-10,
            "Vulkan sample_51 C9 padded ND R2C F64",
        );
    }
}
