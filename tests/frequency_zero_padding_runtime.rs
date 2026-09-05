#![cfg(any(
    feature = "cuda-runtime",
    feature = "opencl-runtime",
    feature = "vulkan-runtime"
))]

use std::f64::consts::TAU;

use vkfft_rs::{Complex32, Direction, FftConfig, TransformIr, ZeroPaddingDomain};

const LENGTH: usize = 16;
const ZERO_LEFT: usize = 4;
const ZERO_RIGHT: usize = 8;

fn config(inverse: bool) -> FftConfig {
    FftConfig::new(vec![LENGTH])
        .with_inverse_normalization(inverse)
        .with_zero_padding(0, ZERO_LEFT, ZERO_RIGHT)
        .unwrap()
        .with_zero_padding_domain(ZeroPaddingDomain::Frequency)
}

fn impulse() -> Vec<Complex32> {
    let mut input = vec![Complex32::new(0.0, 0.0); LENGTH];
    input[0] = Complex32::new(1.0, 0.0);
    input
}

fn full_unit_spectrum() -> Vec<Complex32> {
    vec![Complex32::new(1.0, 0.0); LENGTH]
}

fn expected_masked_inverse() -> Vec<Complex32> {
    (0..LENGTH)
        .map(|n| {
            let mut re = 0.0f64;
            let mut im = 0.0f64;
            for k in 0..LENGTH {
                if (ZERO_LEFT..ZERO_RIGHT).contains(&k) {
                    continue;
                }
                let angle = TAU * (k * n) as f64 / LENGTH as f64;
                re += angle.cos();
                im += angle.sin();
            }
            Complex32::new((re / LENGTH as f64) as f32, (im / LENGTH as f64) as f32)
        })
        .collect()
}

fn assert_forward(values: &[Complex32], label: &str) {
    assert_eq!(values.len(), LENGTH);
    let max_error = values
        .iter()
        .enumerate()
        .map(|(index, value)| {
            let expected = if (ZERO_LEFT..ZERO_RIGHT).contains(&index) {
                Complex32::new(0.0, 0.0)
            } else {
                Complex32::new(1.0, 0.0)
            };
            ((value.re - expected.re).powi(2) + (value.im - expected.im).powi(2)).sqrt()
        })
        .fold(0.0f32, f32::max);
    assert!(
        max_error <= 2.0e-5,
        "{label} frequency-forward error {max_error:e}"
    );
}

fn assert_inverse(values: &[Complex32], label: &str) {
    let expected = expected_masked_inverse();
    assert_eq!(values.len(), expected.len());
    let max_error = values
        .iter()
        .zip(expected)
        .map(|(value, expected)| {
            ((value.re - expected.re).powi(2) + (value.im - expected.im).powi(2)).sqrt()
        })
        .fold(0.0f32, f32::max);
    assert!(
        max_error <= 3.0e-5,
        "{label} frequency-inverse error {max_error:e}"
    );
}

#[cfg(any(feature = "cuda-runtime", feature = "opencl-runtime"))]
fn run_native<R: vkfft_rs::backend::NativeRuntime>(runtime: &R) {
    use vkfft_rs::backend::{NativeTransformInput32, NativeTransformOutput32};

    let profile = runtime.device_profile();
    let forward = TransformIr::build(config(false), Direction::Forward, profile).unwrap();
    let spectrum = match runtime
        .execute_transform_f32(&forward, NativeTransformInput32::Complex(&impulse()))
        .unwrap()
    {
        NativeTransformOutput32::Complex(values) => values,
        NativeTransformOutput32::Real(_) => panic!("frequency C2C forward returned real output"),
    };
    assert_forward(&spectrum, runtime.device_name());

    let inverse = TransformIr::build(config(true), Direction::Inverse, profile).unwrap();
    let restored = match runtime
        .execute_transform_f32(
            &inverse,
            NativeTransformInput32::Complex(&full_unit_spectrum()),
        )
        .unwrap()
    {
        NativeTransformOutput32::Complex(values) => values,
        NativeTransformOutput32::Real(_) => panic!("frequency C2C inverse returned real output"),
    };
    assert_inverse(&restored, runtime.device_name());
}

fn gpu_test_lock() -> &'static std::sync::Mutex<()> {
    static LOCK: std::sync::OnceLock<std::sync::Mutex<()>> = std::sync::OnceLock::new();
    LOCK.get_or_init(|| std::sync::Mutex::new(()))
}

#[cfg(feature = "cuda-runtime")]
#[test]
fn cuda_frequency_zero_padding_c2c_or_skips() {
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
fn opencl_frequency_zero_padding_c2c_or_skips() {
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
fn vulkan_frequency_zero_padding_c2c_or_skips() {
    use vkfft_rs::{
        VkFftError,
        backend::vulkan::runtime::{TransformInput32, TransformOutput32, VulkanExecutionContext},
    };

    let _guard = gpu_test_lock().lock().unwrap();
    let runtime = match VulkanExecutionContext::new() {
        Ok(runtime) => runtime,
        Err(VkFftError::VulkanUnavailable(_)) => return,
        Err(error) => panic!("Vulkan context failed after loader/device discovery: {error}"),
    };
    let profile = runtime.device_profile();
    let forward = TransformIr::build(config(false), Direction::Forward, profile).unwrap();
    let spectrum = match runtime
        .execute_transform_f32(&forward, TransformInput32::Complex(&impulse()))
        .unwrap()
    {
        TransformOutput32::Complex(values) => values,
        TransformOutput32::Real(_) => panic!("Vulkan frequency C2C forward returned real output"),
    };
    assert_forward(&spectrum, "Vulkan");

    let inverse = TransformIr::build(config(true), Direction::Inverse, profile).unwrap();
    let restored = match runtime
        .execute_transform_f32(&inverse, TransformInput32::Complex(&full_unit_spectrum()))
        .unwrap()
    {
        TransformOutput32::Complex(values) => values,
        TransformOutput32::Real(_) => panic!("Vulkan frequency C2C inverse returned real output"),
    };
    assert_inverse(&restored, "Vulkan");
}
