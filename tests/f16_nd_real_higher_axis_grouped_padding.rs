#![cfg(any(
    feature = "cuda-runtime",
    feature = "opencl-runtime",
    feature = "vulkan-runtime"
))]

use vkfft_rs::{Direction, FftConfig, Precision, TransformIr, TransformKind};

const DIMENSIONS: [usize; 2] = [278_528, 14];
const BATCH_COUNT: usize = 5;
const GROUPED_BATCH: usize = 3;
const HALF_SPECTRUM: usize = 8;

fn config(kind: TransformKind, bandwidth_boost: usize) -> FftConfig {
    let config = FftConfig::new(DIMENSIONS.to_vec())
        .with_batch_count(BATCH_COUNT)
        .with_grouped_batch(0, GROUPED_BATCH)
        .unwrap()
        .with_precision(Precision::F16StorageF32Compute)
        .with_transform(kind)
        .with_bandwidth_boost(bandwidth_boost)
        .with_zero_padding(0, 1, 2)
        .unwrap();
    if kind == TransformKind::ComplexToReal {
        config.with_inverse_normalization(true)
    } else {
        config
    }
}

fn impulse_input() -> Vec<f32> {
    let tensor_len = DIMENSIONS.iter().product::<usize>();
    let mut input = vec![0.0f32; tensor_len * BATCH_COUNT];
    for batch in 0..BATCH_COUNT {
        input[batch * tensor_len] = 1.0;
    }
    input
}

fn assert_impulse_spectrum(values: &[vkfft_rs::Complex32], label: &str) {
    let expected_len = DIMENSIONS[0] * HALF_SPECTRUM * BATCH_COUNT;
    assert_eq!(
        values.len(),
        expected_len,
        "{label} compact spectrum length"
    );
    let max_error = values
        .iter()
        .map(|value| ((value.re - 1.0).powi(2) + value.im.powi(2)).sqrt())
        .fold(0.0f32, f32::max);
    assert!(
        max_error <= 2.0e-2,
        "{label} F16 ND-R2C impulse spectrum error {max_error:e}"
    );
}

fn assert_round_trip(actual: &[f32], expected: &[f32], label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label} restored real length");
    let max_error = actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_error <= 2.0e-2,
        "{label} F16 ND-R2C round-trip error {max_error:e}"
    );
}

#[cfg(any(feature = "cuda-runtime", feature = "opencl-runtime"))]
fn run_native<R: vkfft_rs::backend::NativeRuntime>(runtime: &R) {
    use vkfft_rs::backend::{NativeTransformInput32, NativeTransformOutput32};

    let profile = runtime.device_profile();
    let input = impulse_input();
    for bandwidth_boost in [0usize, 2] {
        let forward = TransformIr::build(
            config(TransformKind::RealToComplex, bandwidth_boost),
            Direction::Forward,
            profile,
        )
        .unwrap();
        let spectrum = match runtime
            .execute_transform_f32(&forward, NativeTransformInput32::Real(&input))
            .unwrap()
        {
            NativeTransformOutput32::Complex(values) => values,
            NativeTransformOutput32::Real(_) => panic!("F16 ND-R2C returned real output"),
        };
        let label = format!(
            "{} F16 ND-R2C padded grouped N278528x14 b{bandwidth_boost}",
            runtime.device_name()
        );
        assert_impulse_spectrum(&spectrum, &label);

        let inverse = TransformIr::build(
            config(TransformKind::ComplexToReal, bandwidth_boost),
            Direction::Inverse,
            profile,
        )
        .unwrap();
        let restored = match runtime
            .execute_transform_f32(&inverse, NativeTransformInput32::Complex(&spectrum))
            .unwrap()
        {
            NativeTransformOutput32::Real(values) => values,
            NativeTransformOutput32::Complex(_) => panic!("F16 ND-C2R returned complex output"),
        };
        assert_round_trip(&restored, &input, &label);
    }
}

#[cfg(any(
    feature = "cuda-runtime",
    feature = "opencl-runtime",
    feature = "vulkan-runtime"
))]
fn gpu_test_lock() -> &'static std::sync::Mutex<()> {
    static LOCK: std::sync::OnceLock<std::sync::Mutex<()>> = std::sync::OnceLock::new();
    LOCK.get_or_init(|| std::sync::Mutex::new(()))
}

#[cfg(feature = "cuda-runtime")]
#[test]
fn cuda_f16_nd_r2c_padded_grouped_higher_axis_n278528x14_or_skips() {
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
fn opencl_f16_nd_r2c_padded_grouped_higher_axis_n278528x14_or_skips() {
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
fn vulkan_f16_nd_r2c_padded_grouped_higher_axis_n278528x14_or_skips() {
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
    let input = impulse_input();
    for bandwidth_boost in [0usize, 2] {
        let forward = TransformIr::build(
            config(TransformKind::RealToComplex, bandwidth_boost),
            Direction::Forward,
            profile,
        )
        .unwrap();
        let spectrum = match runtime
            .execute_transform_f32(&forward, TransformInput32::Real(&input))
            .unwrap()
        {
            TransformOutput32::Complex(values) => values,
            TransformOutput32::Real(_) => panic!("Vulkan F16 ND-R2C returned real output"),
        };
        let label = format!("Vulkan F16 ND-R2C padded grouped N278528x14 b{bandwidth_boost}");
        assert_impulse_spectrum(&spectrum, &label);

        let inverse = TransformIr::build(
            config(TransformKind::ComplexToReal, bandwidth_boost),
            Direction::Inverse,
            profile,
        )
        .unwrap();
        let restored = match runtime
            .execute_transform_f32(&inverse, TransformInput32::Complex(&spectrum))
            .unwrap()
        {
            TransformOutput32::Real(values) => values,
            TransformOutput32::Complex(_) => panic!("Vulkan F16 ND-C2R returned complex output"),
        };
        assert_round_trip(&restored, &input, &label);
    }
}
