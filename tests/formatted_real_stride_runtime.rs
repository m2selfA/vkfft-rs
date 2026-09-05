#![cfg(any(
    feature = "cuda-runtime",
    feature = "opencl-runtime",
    feature = "vulkan-runtime"
))]

use vkfft_rs::{Complex32, Complex64, Direction, FftConfig, TransformIr, TransformKind};

fn cases() -> Vec<(FftConfig, FftConfig)> {
    vec![
        (
            FftConfig::new(vec![3, 8])
                .with_batch_count(2)
                .with_transform(TransformKind::RealToComplex)
                .with_input_buffer_axis_stride(0, 11)
                .unwrap()
                .with_output_buffer_axis_stride(0, 7)
                .unwrap(),
            FftConfig::new(vec![3, 8])
                .with_batch_count(2)
                .with_transform(TransformKind::ComplexToReal)
                .with_inverse_normalization(true)
                .with_input_buffer_axis_stride(0, 7)
                .unwrap()
                .with_output_buffer_axis_stride(0, 11)
                .unwrap(),
        ),
        (
            FftConfig::new(vec![2, 3, 8])
                .with_batch_count(2)
                .with_transform(TransformKind::RealToComplex)
                .with_input_buffer_axis_stride(1, 11)
                .unwrap()
                .with_input_buffer_axis_stride(0, 40)
                .unwrap()
                .with_output_buffer_axis_stride(1, 7)
                .unwrap()
                .with_output_buffer_axis_stride(0, 25)
                .unwrap(),
            FftConfig::new(vec![2, 3, 8])
                .with_batch_count(2)
                .with_transform(TransformKind::ComplexToReal)
                .with_inverse_normalization(true)
                .with_input_buffer_axis_stride(1, 7)
                .unwrap()
                .with_input_buffer_axis_stride(0, 25)
                .unwrap()
                .with_output_buffer_axis_stride(1, 11)
                .unwrap()
                .with_output_buffer_axis_stride(0, 40)
                .unwrap(),
        ),
    ]
}

fn padded_case() -> (FftConfig, FftConfig) {
    (
        FftConfig::new(vec![3, 8])
            .with_batch_count(2)
            .with_transform(TransformKind::RealToComplex)
            .with_input_buffer_axis_stride(0, 11)
            .unwrap()
            .with_output_buffer_axis_stride(0, 7)
            .unwrap()
            .with_zero_padding(0, 1, 2)
            .unwrap()
            .with_zero_padding(1, 2, 4)
            .unwrap(),
        FftConfig::new(vec![3, 8])
            .with_batch_count(2)
            .with_transform(TransformKind::ComplexToReal)
            .with_inverse_normalization(true)
            .with_input_buffer_axis_stride(0, 7)
            .unwrap()
            .with_output_buffer_axis_stride(0, 11)
            .unwrap()
            .with_zero_padding(0, 1, 2)
            .unwrap()
            .with_zero_padding(1, 2, 4)
            .unwrap(),
    )
}

fn real_input(config: &FftConfig) -> Vec<f32> {
    let len = config.dimensions.iter().product::<usize>();
    (0..len * config.batch_count)
        .map(|index| {
            let batch = index / len;
            let local = index % len;
            let x = local as f32;
            if batch == 0 {
                (0.137 * x).sin() + 0.021 * x - (0.043 * x).cos()
            } else {
                5.0 + (0.071 * x).cos() - 0.017 * x + (0.113 * x).sin()
            }
        })
        .collect()
}

fn complex_reference_f32(transform: &TransformIr, input: &[f32]) -> Vec<Complex32> {
    let input64 = input
        .iter()
        .map(|value| f64::from(*value))
        .collect::<Vec<_>>();
    transform
        .execute_r2c_reference(&input64)
        .unwrap()
        .into_iter()
        .map(|value| Complex32::new(value.re as f32, value.im as f32))
        .collect()
}

fn real_reference_f32(transform: &TransformIr, input: &[Complex32]) -> Vec<f32> {
    let input64 = input
        .iter()
        .map(|value| Complex64::new(f64::from(value.re), f64::from(value.im)))
        .collect::<Vec<_>>();
    transform
        .execute_c2r_reference(&input64)
        .unwrap()
        .into_iter()
        .map(|value| value as f32)
        .collect()
}

fn assert_complex_close(actual: &[Complex32], expected: &[Complex32], label: &str) {
    assert_eq!(actual.len(), expected.len());
    let max_error = actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| {
            ((actual.re - expected.re).powi(2) + (actual.im - expected.im).powi(2)).sqrt()
        })
        .fold(0.0f32, f32::max);
    assert!(
        max_error <= 2.5e-4,
        "{label} complex max error {max_error:e}"
    );
}

fn assert_real_close(actual: &[f32], expected: &[f32], label: &str) {
    assert_eq!(actual.len(), expected.len());
    let max_error = actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0f32, f32::max);
    assert!(max_error <= 3.5e-4, "{label} real max error {max_error:e}");
}

fn gpu_test_lock() -> &'static std::sync::Mutex<()> {
    static LOCK: std::sync::OnceLock<std::sync::Mutex<()>> = std::sync::OnceLock::new();
    LOCK.get_or_init(|| std::sync::Mutex::new(()))
}

#[cfg(any(feature = "cuda-runtime", feature = "opencl-runtime"))]
fn run_native<R: vkfft_rs::backend::NativeRuntime>(runtime: &R) {
    use vkfft_rs::backend::{NativeTransformInput32, NativeTransformOutput32};

    for (forward_config, inverse_config) in cases() {
        let input = real_input(&forward_config);
        let forward =
            TransformIr::build(forward_config, Direction::Forward, runtime.device_profile())
                .unwrap();
        let expected_spectrum = complex_reference_f32(&forward, &input);
        let spectrum = match runtime
            .execute_transform_f32(&forward, NativeTransformInput32::Real(&input))
            .unwrap()
        {
            NativeTransformOutput32::Complex(values) => values,
            NativeTransformOutput32::Real(_) => {
                panic!("formatted ND R2C returned real output")
            }
        };
        assert_complex_close(&spectrum, &expected_spectrum, runtime.device_name());

        let inverse =
            TransformIr::build(inverse_config, Direction::Inverse, runtime.device_profile())
                .unwrap();
        let expected_real = real_reference_f32(&inverse, &spectrum);
        let restored = match runtime
            .execute_transform_f32(&inverse, NativeTransformInput32::Complex(&spectrum))
            .unwrap()
        {
            NativeTransformOutput32::Real(values) => values,
            NativeTransformOutput32::Complex(_) => {
                panic!("formatted ND C2R returned complex output")
            }
        };
        assert_real_close(&restored, &expected_real, runtime.device_name());
        assert_real_close(&restored, &input, runtime.device_name());
    }

    let (forward_config, inverse_config) = padded_case();
    let input = real_input(&forward_config);
    let forward =
        TransformIr::build(forward_config, Direction::Forward, runtime.device_profile()).unwrap();
    let expected_spectrum = complex_reference_f32(&forward, &input);
    let spectrum = match runtime
        .execute_transform_f32(&forward, NativeTransformInput32::Real(&input))
        .unwrap()
    {
        NativeTransformOutput32::Complex(values) => values,
        NativeTransformOutput32::Real(_) => {
            panic!("formatted+padding ND R2C returned real output")
        }
    };
    assert_complex_close(&spectrum, &expected_spectrum, runtime.device_name());
    let inverse =
        TransformIr::build(inverse_config, Direction::Inverse, runtime.device_profile()).unwrap();
    let expected_real = real_reference_f32(&inverse, &spectrum);
    let restored = match runtime
        .execute_transform_f32(&inverse, NativeTransformInput32::Complex(&spectrum))
        .unwrap()
    {
        NativeTransformOutput32::Real(values) => values,
        NativeTransformOutput32::Complex(_) => {
            panic!("formatted+padding ND C2R returned complex output")
        }
    };
    assert_real_close(&restored, &expected_real, runtime.device_name());
}

#[cfg(feature = "vulkan-runtime")]
fn run_vulkan(runtime: &vkfft_rs::backend::vulkan::runtime::VulkanExecutionContext) {
    use vkfft_rs::backend::vulkan::runtime::{TransformInput32, TransformOutput32};

    for (forward_config, inverse_config) in cases() {
        let input = real_input(&forward_config);
        let forward =
            TransformIr::build(forward_config, Direction::Forward, runtime.device_profile())
                .unwrap();
        let expected_spectrum = complex_reference_f32(&forward, &input);
        let spectrum = match runtime
            .execute_transform_f32(&forward, TransformInput32::Real(&input))
            .unwrap()
        {
            TransformOutput32::Complex(values) => values,
            TransformOutput32::Real(_) => panic!("Vulkan formatted ND R2C returned real output"),
        };
        assert_complex_close(&spectrum, &expected_spectrum, "Vulkan");

        let inverse =
            TransformIr::build(inverse_config, Direction::Inverse, runtime.device_profile())
                .unwrap();
        let expected_real = real_reference_f32(&inverse, &spectrum);
        let restored = match runtime
            .execute_transform_f32(&inverse, TransformInput32::Complex(&spectrum))
            .unwrap()
        {
            TransformOutput32::Real(values) => values,
            TransformOutput32::Complex(_) => {
                panic!("Vulkan formatted ND C2R returned complex output")
            }
        };
        assert_real_close(&restored, &expected_real, "Vulkan");
        assert_real_close(&restored, &input, "Vulkan");
    }

    let (forward_config, inverse_config) = padded_case();
    let input = real_input(&forward_config);
    let forward =
        TransformIr::build(forward_config, Direction::Forward, runtime.device_profile()).unwrap();
    let expected_spectrum = complex_reference_f32(&forward, &input);
    let spectrum = match runtime
        .execute_transform_f32(&forward, TransformInput32::Real(&input))
        .unwrap()
    {
        TransformOutput32::Complex(values) => values,
        TransformOutput32::Real(_) => panic!("Vulkan formatted+padding ND R2C returned real"),
    };
    assert_complex_close(&spectrum, &expected_spectrum, "Vulkan");
    let inverse =
        TransformIr::build(inverse_config, Direction::Inverse, runtime.device_profile()).unwrap();
    let expected_real = real_reference_f32(&inverse, &spectrum);
    let restored = match runtime
        .execute_transform_f32(&inverse, TransformInput32::Complex(&spectrum))
        .unwrap()
    {
        TransformOutput32::Real(values) => values,
        TransformOutput32::Complex(_) => {
            panic!("Vulkan formatted+padding ND C2R returned complex")
        }
    };
    assert_real_close(&restored, &expected_real, "Vulkan");
}

#[cfg(feature = "cuda-runtime")]
#[test]
fn cuda_formatted_real_strides_or_skips() {
    use vkfft_rs::backend::cuda::runtime::CudaExecutionContext;
    use vkfft_rs::backend::{NativeTransformInput32, NativeTransformOutput32};

    let _guard = gpu_test_lock().lock().unwrap();
    let availability = CudaExecutionContext::probe();
    if !availability.compiler_available || availability.device_count == 0 {
        return;
    }
    let runtime = CudaExecutionContext::new(0).expect("CUDA context failed after successful probe");
    run_native(&runtime);

    let (forward_config, inverse_config) = cases().remove(0);
    let input = real_input(&forward_config);
    let forward =
        TransformIr::build(forward_config, Direction::Forward, runtime.device_profile()).unwrap();
    let spectrum = match runtime
        .submit_transform_f32(&forward, NativeTransformInput32::Real(&input))
        .unwrap()
        .wait()
        .unwrap()
    {
        NativeTransformOutput32::Complex(values) => values,
        NativeTransformOutput32::Real(_) => panic!("async CUDA formatted R2C returned real"),
    };
    let inverse =
        TransformIr::build(inverse_config, Direction::Inverse, runtime.device_profile()).unwrap();
    let restored = match runtime
        .submit_transform_f32(&inverse, NativeTransformInput32::Complex(&spectrum))
        .unwrap()
        .wait()
        .unwrap()
    {
        NativeTransformOutput32::Real(values) => values,
        NativeTransformOutput32::Complex(_) => {
            panic!("async CUDA formatted C2R returned complex")
        }
    };
    assert_real_close(&restored, &input, "CUDA async");
}

#[cfg(feature = "opencl-runtime")]
#[test]
fn opencl_formatted_real_strides_or_skips() {
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

    let (forward_config, inverse_config) = cases().remove(0);
    let input = real_input(&forward_config);
    let forward =
        TransformIr::build(forward_config, Direction::Forward, runtime.device_profile()).unwrap();
    let spectrum = match runtime
        .submit_transform_f32(&forward, NativeTransformInput32::Real(&input))
        .unwrap()
        .wait()
        .unwrap()
    {
        NativeTransformOutput32::Complex(values) => values,
        NativeTransformOutput32::Real(_) => panic!("async OpenCL formatted R2C returned real"),
    };
    let inverse =
        TransformIr::build(inverse_config, Direction::Inverse, runtime.device_profile()).unwrap();
    let restored = match runtime
        .submit_transform_f32(&inverse, NativeTransformInput32::Complex(&spectrum))
        .unwrap()
        .wait()
        .unwrap()
    {
        NativeTransformOutput32::Real(values) => values,
        NativeTransformOutput32::Complex(_) => {
            panic!("async OpenCL formatted C2R returned complex")
        }
    };
    assert_real_close(&restored, &input, "OpenCL async");
}

#[cfg(feature = "vulkan-runtime")]
#[test]
fn vulkan_formatted_real_strides_or_skips() {
    use vkfft_rs::{VkFftError, backend::vulkan::runtime::VulkanExecutionContext};

    let _guard = gpu_test_lock().lock().unwrap();
    let runtime = match VulkanExecutionContext::new() {
        Ok(runtime) => runtime,
        Err(VkFftError::VulkanUnavailable(_)) => return,
        Err(error) => panic!("Vulkan context failed after loader/device discovery: {error}"),
    };
    run_vulkan(&runtime);
}
