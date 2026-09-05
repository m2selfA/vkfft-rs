#![cfg(any(
    feature = "cuda-runtime",
    feature = "opencl-runtime",
    feature = "vulkan-runtime"
))]

use vkfft_rs::{DctType, Direction, FftConfig, TransformIr, TransformKind};

fn config_2d(normalize_inverse: bool) -> FftConfig {
    FftConfig::new(vec![3, 4])
        .with_batch_count(2)
        .with_transform(TransformKind::Dct(DctType::II))
        .with_inverse_normalization(normalize_inverse)
        .with_input_buffer_axis_stride(0, 7)
        .unwrap()
        .with_output_buffer_axis_stride(0, 9)
        .unwrap()
}

fn config_2d_padded(normalize_inverse: bool) -> FftConfig {
    config_2d(normalize_inverse)
        .with_zero_padding(0, 1, 3)
        .unwrap()
        .with_zero_padding(1, 2, 4)
        .unwrap()
}

fn config_3d(normalize_inverse: bool) -> FftConfig {
    FftConfig::new(vec![2, 3, 4])
        .with_batch_count(2)
        .with_transform(TransformKind::Dct(DctType::II))
        .with_inverse_normalization(normalize_inverse)
        .with_input_buffer_axis_stride(1, 6)
        .unwrap()
        .with_input_buffer_axis_stride(0, 20)
        .unwrap()
        .with_output_buffer_axis_stride(1, 7)
        .unwrap()
        .with_output_buffer_axis_stride(0, 24)
        .unwrap()
}

fn input_for(config: &FftConfig) -> Vec<f32> {
    let tensor_len = config.dimensions.iter().product::<usize>();
    (0..tensor_len * config.batch_count)
        .map(|index| {
            let x = index as f32;
            (0.137 * x).sin() + 0.09 * (0.071 * x).cos() - 0.002 * x
        })
        .collect()
}

fn reference_f32(transform: &TransformIr, input: &[f32]) -> Vec<f32> {
    let input64 = input
        .iter()
        .map(|value| f64::from(*value))
        .collect::<Vec<_>>();
    transform
        .execute_r2r_reference(&input64)
        .unwrap()
        .into_iter()
        .map(|value| value as f32)
        .collect()
}

fn assert_close(actual: &[f32], expected: &[f32], label: &str) {
    assert_eq!(actual.len(), expected.len());
    let max_error = actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0f32, f32::max);
    assert!(max_error <= 8.0e-5, "{label} max error {max_error:e}");
}

fn gpu_test_lock() -> &'static std::sync::Mutex<()> {
    static LOCK: std::sync::OnceLock<std::sync::Mutex<()>> = std::sync::OnceLock::new();
    LOCK.get_or_init(|| std::sync::Mutex::new(()))
}

#[cfg(any(feature = "cuda-runtime", feature = "opencl-runtime"))]
fn run_native<R: vkfft_rs::backend::NativeRuntime>(runtime: &R) {
    use vkfft_rs::backend::{NativeTransformInput32, NativeTransformOutput32};

    let profile = runtime.device_profile();
    for (config, inverse_config) in [
        (config_2d(false), config_2d(true)),
        (config_3d(false), config_3d(true)),
        (config_2d_padded(false), config_2d_padded(true)),
    ] {
        let forward = TransformIr::build(config.clone(), Direction::Forward, profile).unwrap();
        let input = input_for(&config);
        let expected_forward = reference_f32(&forward, &input);
        let actual_forward = match runtime
            .execute_transform_f32(&forward, NativeTransformInput32::Real(&input))
            .unwrap()
        {
            NativeTransformOutput32::Real(values) => values,
            NativeTransformOutput32::Complex(_) => {
                panic!("formatted ND R2R forward returned complex output")
            }
        };
        assert_close(&actual_forward, &expected_forward, runtime.device_name());

        let inverse = TransformIr::build(inverse_config, Direction::Inverse, profile).unwrap();
        let expected_inverse = reference_f32(&inverse, &actual_forward);
        let actual_inverse = match runtime
            .execute_transform_f32(&inverse, NativeTransformInput32::Real(&actual_forward))
            .unwrap()
        {
            NativeTransformOutput32::Real(values) => values,
            NativeTransformOutput32::Complex(_) => {
                panic!("formatted ND R2R inverse returned complex output")
            }
        };
        assert_close(&actual_inverse, &expected_inverse, runtime.device_name());
    }
}

#[cfg(feature = "vulkan-runtime")]
fn run_vulkan(runtime: &vkfft_rs::backend::vulkan::runtime::VulkanExecutionContext) {
    use vkfft_rs::backend::vulkan::runtime::{TransformInput32, TransformOutput32};

    let profile = runtime.device_profile();
    for (config, inverse_config) in [
        (config_2d(false), config_2d(true)),
        (config_3d(false), config_3d(true)),
        (config_2d_padded(false), config_2d_padded(true)),
    ] {
        let forward = TransformIr::build(config.clone(), Direction::Forward, profile).unwrap();
        let input = input_for(&config);
        let expected_forward = reference_f32(&forward, &input);
        let actual_forward = match runtime
            .execute_transform_f32(&forward, TransformInput32::Real(&input))
            .unwrap()
        {
            TransformOutput32::Real(values) => values,
            TransformOutput32::Complex(_) => {
                panic!("Vulkan formatted ND R2R forward returned complex output")
            }
        };
        assert_close(&actual_forward, &expected_forward, "Vulkan");

        let inverse = TransformIr::build(inverse_config, Direction::Inverse, profile).unwrap();
        let expected_inverse = reference_f32(&inverse, &actual_forward);
        let actual_inverse = match runtime
            .execute_transform_f32(&inverse, TransformInput32::Real(&actual_forward))
            .unwrap()
        {
            TransformOutput32::Real(values) => values,
            TransformOutput32::Complex(_) => {
                panic!("Vulkan formatted ND R2R inverse returned complex output")
            }
        };
        assert_close(&actual_inverse, &expected_inverse, "Vulkan");
    }
}

#[cfg(feature = "cuda-runtime")]
#[test]
fn cuda_formatted_r2r_strides_or_skips() {
    use vkfft_rs::backend::cuda::runtime::CudaExecutionContext;
    use vkfft_rs::backend::{NativeTransformInput32, NativeTransformOutput32};

    let _guard = gpu_test_lock()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let availability = CudaExecutionContext::probe();
    if !availability.compiler_available || availability.device_count == 0 {
        return;
    }
    let runtime = CudaExecutionContext::new(0).expect("CUDA context failed after successful probe");
    run_native(&runtime);

    let config = config_2d(false);
    let transform =
        TransformIr::build(config.clone(), Direction::Forward, runtime.device_profile()).unwrap();
    let input = input_for(&config);
    let expected = reference_f32(&transform, &input);
    let actual = match runtime
        .submit_transform_f32(&transform, NativeTransformInput32::Real(&input))
        .unwrap()
        .wait()
        .unwrap()
    {
        NativeTransformOutput32::Real(values) => values,
        NativeTransformOutput32::Complex(_) => {
            panic!("async CUDA formatted ND R2R returned complex output")
        }
    };
    assert_close(&actual, &expected, "CUDA async");
}

#[cfg(feature = "opencl-runtime")]
#[test]
fn opencl_formatted_r2r_strides_or_skips() {
    use vkfft_rs::backend::opencl::runtime::OpenClExecutionContext;
    use vkfft_rs::backend::{NativeTransformInput32, NativeTransformOutput32};

    let _guard = gpu_test_lock()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let availability = OpenClExecutionContext::probe();
    if !availability.compiler_available || availability.device_count == 0 {
        return;
    }
    let runtime =
        OpenClExecutionContext::new(0).expect("OpenCL context failed after successful probe");
    run_native(&runtime);

    let config = config_2d(false);
    let transform =
        TransformIr::build(config.clone(), Direction::Forward, runtime.device_profile()).unwrap();
    let input = input_for(&config);
    let expected = reference_f32(&transform, &input);
    let actual = match runtime
        .submit_transform_f32(&transform, NativeTransformInput32::Real(&input))
        .unwrap()
        .wait()
        .unwrap()
    {
        NativeTransformOutput32::Real(values) => values,
        NativeTransformOutput32::Complex(_) => {
            panic!("async OpenCL formatted ND R2R returned complex output")
        }
    };
    assert_close(&actual, &expected, "OpenCL async");
}

#[cfg(feature = "vulkan-runtime")]
#[test]
fn vulkan_formatted_r2r_strides_or_skips() {
    use vkfft_rs::{VkFftError, backend::vulkan::runtime::VulkanExecutionContext};

    let _guard = gpu_test_lock()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let runtime = match VulkanExecutionContext::new() {
        Ok(runtime) => runtime,
        Err(VkFftError::VulkanUnavailable(_)) => return,
        Err(error) => panic!("Vulkan context failed after loader/device discovery: {error}"),
    };
    run_vulkan(&runtime);
}
