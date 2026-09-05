#![cfg(any(
    feature = "cuda-runtime",
    feature = "opencl-runtime",
    feature = "vulkan-runtime",
    feature = "level-zero-runtime",
    feature = "metal-runtime"
))]

use vkfft_rs::{Complex32, Complex64, Direction, FftConfig, TransformIr, ZeroPaddingDomain};

const BATCH: usize = 2;

fn batch_stride_config() -> FftConfig {
    FftConfig::new(vec![3, 4])
        .with_batch_count(BATCH)
        .with_input_buffer_batch_stride(17)
        .with_output_buffer_batch_stride(19)
}

fn row_stride_config() -> FftConfig {
    FftConfig::new(vec![3, 4])
        .with_batch_count(BATCH)
        .with_input_buffer_axis_stride(0, 7)
        .unwrap()
        .with_output_buffer_axis_stride(0, 9)
        .unwrap()
}

fn row_stride_spatial_padding_config() -> FftConfig {
    row_stride_config().with_zero_padding(1, 1, 3).unwrap()
}

fn row_stride_frequency_padding_config() -> FftConfig {
    row_stride_config()
        .with_zero_padding(1, 1, 3)
        .unwrap()
        .with_zero_padding_domain(ZeroPaddingDomain::Frequency)
}

fn plane_stride_config() -> FftConfig {
    FftConfig::new(vec![2, 3, 4])
        .with_batch_count(BATCH)
        .with_input_buffer_axis_stride(1, 6)
        .unwrap()
        .with_input_buffer_axis_stride(0, 20)
        .unwrap()
        .with_output_buffer_axis_stride(1, 7)
        .unwrap()
        .with_output_buffer_axis_stride(0, 24)
        .unwrap()
}

fn input(config: &FftConfig) -> Vec<Complex32> {
    let len = config.dimensions.iter().product::<usize>();
    (0..len * config.batch_count)
        .map(|index| {
            let batch = index / len;
            let local = index % len;
            let x = local as f32;
            if batch == 0 {
                Complex32::new((0.17 * x).sin() + 0.01 * x, (0.11 * x).cos() - 0.02 * x)
            } else {
                Complex32::new(
                    7.0 + (0.07 * x).cos() + 0.03 * x,
                    -5.0 + (0.19 * x).sin() - 0.01 * x,
                )
            }
        })
        .collect()
}

fn expected(transform: &TransformIr, input: &[Complex32]) -> Vec<Complex32> {
    let input64 = input
        .iter()
        .map(|value| Complex64::new(f64::from(value.re), f64::from(value.im)))
        .collect::<Vec<_>>();
    transform
        .execute_complex_reference(&input64)
        .unwrap()
        .into_iter()
        .map(|value| Complex32::new(value.re as f32, value.im as f32))
        .collect()
}

fn assert_close(actual: &[Complex32], expected: &[Complex32], label: &str) {
    assert_eq!(actual.len(), expected.len());
    let max_error = actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| {
            ((actual.re - expected.re).powi(2) + (actual.im - expected.im).powi(2)).sqrt()
        })
        .fold(0.0f32, f32::max);
    assert!(max_error <= 5.0e-5, "{label} max error {max_error:e}");
}

fn gpu_test_lock() -> &'static std::sync::Mutex<()> {
    static LOCK: std::sync::OnceLock<std::sync::Mutex<()>> = std::sync::OnceLock::new();
    LOCK.get_or_init(|| std::sync::Mutex::new(()))
}

#[cfg(any(
    feature = "cuda-runtime",
    feature = "opencl-runtime",
    feature = "level-zero-runtime",
    feature = "metal-runtime"
))]
fn run_native<R: vkfft_rs::backend::NativeRuntime>(runtime: &R) {
    use vkfft_rs::backend::{NativeTransformInput32, NativeTransformOutput32};

    for config in [
        batch_stride_config(),
        row_stride_config(),
        plane_stride_config(),
        row_stride_spatial_padding_config(),
        row_stride_frequency_padding_config(),
    ] {
        let logical_config = config.clone();
        let transform =
            TransformIr::build(config, Direction::Forward, runtime.device_profile()).unwrap();
        let input = input(&logical_config);
        let expected = expected(&transform, &input);
        let actual = match runtime
            .execute_transform_f32(&transform, NativeTransformInput32::Complex(&input))
            .unwrap()
        {
            NativeTransformOutput32::Complex(values) => values,
            NativeTransformOutput32::Real(_) => {
                panic!("formatted-stride C2C returned real output")
            }
        };
        assert_close(&actual, &expected, runtime.device_name());
    }
}

#[cfg(feature = "vulkan-runtime")]
fn run_vulkan(runtime: &vkfft_rs::backend::vulkan::runtime::VulkanExecutionContext) {
    use vkfft_rs::backend::vulkan::runtime::{TransformInput32, TransformOutput32};

    for config in [
        batch_stride_config(),
        row_stride_config(),
        plane_stride_config(),
        row_stride_spatial_padding_config(),
        row_stride_frequency_padding_config(),
    ] {
        let logical_config = config.clone();
        let transform =
            TransformIr::build(config, Direction::Forward, runtime.device_profile()).unwrap();
        let input = input(&logical_config);
        let expected = expected(&transform, &input);
        let actual = match runtime
            .execute_transform_f32(&transform, TransformInput32::Complex(&input))
            .unwrap()
        {
            TransformOutput32::Complex(values) => values,
            TransformOutput32::Real(_) => panic!("formatted-stride C2C returned real output"),
        };
        assert_close(&actual, &expected, "Vulkan");
    }
}

#[cfg(feature = "cuda-runtime")]
#[test]
fn cuda_formatted_strides_or_skips() {
    use vkfft_rs::backend::cuda::runtime::CudaExecutionContext;
    use vkfft_rs::backend::{NativeTransformInput32, NativeTransformOutput32};

    let _guard = gpu_test_lock().lock().unwrap();
    let availability = CudaExecutionContext::probe();
    if !availability.compiler_available || availability.device_count == 0 {
        return;
    }
    let runtime = CudaExecutionContext::new(0).expect("CUDA context failed after successful probe");
    run_native(&runtime);

    let config = row_stride_config();
    let transform =
        TransformIr::build(config.clone(), Direction::Forward, runtime.device_profile()).unwrap();
    let input = input(&config);
    let expected = expected(&transform, &input);
    let actual = match runtime
        .submit_transform_f32(&transform, NativeTransformInput32::Complex(&input))
        .unwrap()
        .wait()
        .unwrap()
    {
        NativeTransformOutput32::Complex(values) => values,
        NativeTransformOutput32::Real(_) => {
            panic!("async CUDA formatted-stride C2C returned real output")
        }
    };
    assert_close(&actual, &expected, "CUDA async");
}

#[cfg(feature = "opencl-runtime")]
#[test]
fn opencl_formatted_strides_or_skips() {
    use vkfft_rs::backend::opencl::runtime::OpenClExecutionContext;
    use vkfft_rs::backend::{NativeTransformInput32, NativeTransformOutput32};

    let _guard = gpu_test_lock().lock().unwrap();
    let availability = OpenClExecutionContext::probe();
    if !availability.compiler_available || availability.device_count == 0 {
        return;
    }
    let runtime =
        OpenClExecutionContext::new(0).expect("OpenCL context failed after successful probe");
    run_native(&runtime);

    let config = row_stride_config();
    let transform =
        TransformIr::build(config.clone(), Direction::Forward, runtime.device_profile()).unwrap();
    let input = input(&config);
    let expected = expected(&transform, &input);
    let actual = match runtime
        .submit_transform_f32(&transform, NativeTransformInput32::Complex(&input))
        .unwrap()
        .wait()
        .unwrap()
    {
        NativeTransformOutput32::Complex(values) => values,
        NativeTransformOutput32::Real(_) => {
            panic!("async OpenCL formatted-stride C2C returned real output")
        }
    };
    assert_close(&actual, &expected, "OpenCL async");
}

#[cfg(feature = "level-zero-runtime")]
#[test]
fn level_zero_formatted_strides_or_skips() {
    use vkfft_rs::backend::level_zero::runtime::LevelZeroExecutionContext;
    use vkfft_rs::backend::{NativeTransformInput32, NativeTransformOutput32};

    let _guard = gpu_test_lock().lock().unwrap();
    let require = std::env::var_os("VKFFT_REQUIRE_LEVEL_ZERO_RUNTIME").is_some();
    let availability = LevelZeroExecutionContext::probe();
    if !availability.available() {
        assert!(
            !require,
            "strict Level Zero formatted-stride gate is unavailable: {}",
            availability.detail
        );
        return;
    }
    let runtime = LevelZeroExecutionContext::new(0)
        .expect("Level Zero context failed after successful probe");
    run_native(&runtime);

    let config = row_stride_config();
    let transform =
        TransformIr::build(config.clone(), Direction::Forward, runtime.device_profile()).unwrap();
    let input = input(&config);
    let expected = expected(&transform, &input);
    let actual = match runtime
        .submit_transform_f32(&transform, NativeTransformInput32::Complex(&input))
        .unwrap()
        .wait()
        .unwrap()
    {
        NativeTransformOutput32::Complex(values) => values,
        NativeTransformOutput32::Real(_) => {
            panic!("async Level Zero formatted-stride C2C returned real output")
        }
    };
    assert_close(&actual, &expected, "Level Zero async");
}

#[cfg(feature = "metal-runtime")]
#[test]
fn metal_formatted_strides_or_skips() {
    use vkfft_rs::backend::metal::runtime::MetalExecutionContext;
    use vkfft_rs::backend::{NativeTransformInput32, NativeTransformOutput32};

    let _guard = gpu_test_lock().lock().unwrap();
    let require = std::env::var_os("VKFFT_REQUIRE_METAL_RUNTIME").is_some();
    let availability = MetalExecutionContext::probe();
    if !availability.available() {
        assert!(
            !require,
            "strict Metal formatted-stride gate is unavailable: {}",
            availability.detail
        );
        return;
    }
    let runtime =
        MetalExecutionContext::new(0).expect("Metal context failed after successful probe");
    run_native(&runtime);

    let config = row_stride_config();
    let transform =
        TransformIr::build(config.clone(), Direction::Forward, runtime.device_profile()).unwrap();
    let input = input(&config);
    let expected = expected(&transform, &input);
    let actual = match runtime
        .submit_transform_f32(&transform, NativeTransformInput32::Complex(&input))
        .unwrap()
        .wait()
        .unwrap()
    {
        NativeTransformOutput32::Complex(values) => values,
        NativeTransformOutput32::Real(_) => {
            panic!("async Metal formatted-stride C2C returned real output")
        }
    };
    assert_close(&actual, &expected, "Metal async");
}

#[cfg(feature = "vulkan-runtime")]
#[test]
fn vulkan_formatted_strides_or_skips() {
    use vkfft_rs::{VkFftError, backend::vulkan::runtime::VulkanExecutionContext};

    let _guard = gpu_test_lock().lock().unwrap();
    let runtime = match VulkanExecutionContext::new() {
        Ok(runtime) => runtime,
        Err(VkFftError::VulkanUnavailable(_)) => return,
        Err(error) => panic!("Vulkan context failed after loader/device discovery: {error}"),
    };
    run_vulkan(&runtime);
}
