#![cfg(any(
    feature = "cuda-runtime",
    feature = "opencl-runtime",
    feature = "vulkan-runtime"
))]

use vkfft_rs::{Complex32, DctType, Direction, FftConfig, TransformIr, TransformKind};

const ROWS: usize = 4;
const COLS: usize = 8;
const LEN: usize = ROWS * COLS;

fn config(omitted_axis: usize, inverse: bool) -> FftConfig {
    FftConfig::new(vec![ROWS, COLS])
        .with_inverse_normalization(inverse)
        .with_omit_dimension(omitted_axis, true)
        .unwrap()
}

fn real_config(kind: TransformKind, inverse: bool) -> FftConfig {
    FftConfig::new(vec![ROWS, COLS])
        .with_transform(kind)
        .with_inverse_normalization(inverse)
        .with_omit_dimension(0, true)
        .unwrap()
}

fn r2r_config(omitted_axis: usize) -> FftConfig {
    FftConfig::new(vec![ROWS, COLS])
        .with_transform(TransformKind::Dct(DctType::II))
        .with_omit_dimension(omitted_axis, true)
        .unwrap()
}

fn real_input() -> Vec<f32> {
    (0..LEN)
        .map(|index| {
            let x = index as f32;
            (0.173 * x).sin() + 0.011 * x - (0.037 * x).cos()
        })
        .collect()
}

fn impulse_for_active_axis(omitted_axis: usize) -> Vec<Complex32> {
    let mut input = vec![Complex32::new(0.0, 0.0); LEN];
    match omitted_axis {
        0 => {
            for row in 0..ROWS {
                input[row * COLS] = Complex32::new(1.0, 0.0);
            }
        }
        1 => {
            input
                .iter_mut()
                .take(COLS)
                .for_each(|value| *value = Complex32::new(1.0, 0.0));
        }
        _ => unreachable!(),
    }
    input
}

fn expected_inverse(omitted_axis: usize) -> Vec<Complex32> {
    let mut output = vec![Complex32::new(0.0, 0.0); LEN];
    match omitted_axis {
        0 => {
            for row in 0..ROWS {
                output[row * COLS] = Complex32::new(1.0, 0.0);
            }
        }
        1 => {
            output
                .iter_mut()
                .take(COLS)
                .for_each(|value| *value = Complex32::new(1.0, 0.0));
        }
        _ => unreachable!(),
    }
    output
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
    assert!(max_error <= 3.0e-5, "{label} max error {max_error:e}");
}

fn assert_real_close(actual: &[f32], expected: &[f32], label: &str) {
    assert_eq!(actual.len(), expected.len());
    let max_error = actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0f32, f32::max);
    assert!(max_error <= 5.0e-5, "{label} max real error {max_error:e}");
}

fn complex_reference_f32(values: &[vkfft_rs::Complex64]) -> Vec<Complex32> {
    values
        .iter()
        .map(|value| Complex32::new(value.re as f32, value.im as f32))
        .collect()
}

fn real_reference_f32(values: &[f64]) -> Vec<f32> {
    values.iter().map(|value| *value as f32).collect()
}

#[cfg(any(feature = "cuda-runtime", feature = "opencl-runtime"))]
fn run_native<R: vkfft_rs::backend::NativeRuntime>(runtime: &R) {
    use vkfft_rs::backend::{NativeTransformInput32, NativeTransformOutput32};

    let profile = runtime.device_profile();
    for omitted_axis in [0usize, 1] {
        let forward =
            TransformIr::build(config(omitted_axis, false), Direction::Forward, profile).unwrap();
        let input = impulse_for_active_axis(omitted_axis);
        let actual = match runtime
            .execute_transform_f32(&forward, NativeTransformInput32::Complex(&input))
            .unwrap()
        {
            NativeTransformOutput32::Complex(values) => values,
            NativeTransformOutput32::Real(_) => panic!("omitDimension C2C returned real output"),
        };
        assert_close(
            &actual,
            &vec![Complex32::new(1.0, 0.0); LEN],
            runtime.device_name(),
        );

        let inverse =
            TransformIr::build(config(omitted_axis, true), Direction::Inverse, profile).unwrap();
        let spectrum = vec![Complex32::new(1.0, 0.0); LEN];
        let actual = match runtime
            .execute_transform_f32(&inverse, NativeTransformInput32::Complex(&spectrum))
            .unwrap()
        {
            NativeTransformOutput32::Complex(values) => values,
            NativeTransformOutput32::Real(_) => panic!("omitDimension C2C returned real output"),
        };
        assert_close(
            &actual,
            &expected_inverse(omitted_axis),
            runtime.device_name(),
        );
    }

    let input = real_input();
    let input64 = input
        .iter()
        .map(|value| f64::from(*value))
        .collect::<Vec<_>>();
    let real_forward = TransformIr::build(
        real_config(TransformKind::RealToComplex, false),
        Direction::Forward,
        profile,
    )
    .unwrap();
    let expected = complex_reference_f32(&real_forward.execute_r2c_reference(&input64).unwrap());
    let spectrum = match runtime
        .execute_transform_f32(&real_forward, NativeTransformInput32::Real(&input))
        .unwrap()
    {
        NativeTransformOutput32::Complex(values) => values,
        NativeTransformOutput32::Real(_) => panic!("omitDimension R2C returned real output"),
    };
    assert_close(&spectrum, &expected, runtime.device_name());

    let real_inverse = TransformIr::build(
        real_config(TransformKind::ComplexToReal, true),
        Direction::Inverse,
        profile,
    )
    .unwrap();
    let spectrum64 = spectrum
        .iter()
        .map(|value| vkfft_rs::Complex64::new(f64::from(value.re), f64::from(value.im)))
        .collect::<Vec<_>>();
    let expected = real_reference_f32(&real_inverse.execute_c2r_reference(&spectrum64).unwrap());
    let restored = match runtime
        .execute_transform_f32(&real_inverse, NativeTransformInput32::Complex(&spectrum))
        .unwrap()
    {
        NativeTransformOutput32::Real(values) => values,
        NativeTransformOutput32::Complex(_) => panic!("omitDimension C2R returned complex output"),
    };
    assert_real_close(&restored, &expected, runtime.device_name());

    for omitted_axis in [0usize, 1] {
        let r2r =
            TransformIr::build(r2r_config(omitted_axis), Direction::Forward, profile).unwrap();
        let expected = real_reference_f32(&r2r.execute_r2r_reference(&input64).unwrap());
        let actual = match runtime
            .execute_transform_f32(&r2r, NativeTransformInput32::Real(&input))
            .unwrap()
        {
            NativeTransformOutput32::Real(values) => values,
            NativeTransformOutput32::Complex(_) => {
                panic!("omitDimension DCT-II returned complex output")
            }
        };
        assert_real_close(&actual, &expected, runtime.device_name());
    }
}

fn gpu_test_lock() -> &'static std::sync::Mutex<()> {
    static LOCK: std::sync::OnceLock<std::sync::Mutex<()>> = std::sync::OnceLock::new();
    LOCK.get_or_init(|| std::sync::Mutex::new(()))
}

#[cfg(feature = "vulkan-runtime")]
fn run_vulkan(runtime: &vkfft_rs::backend::vulkan::runtime::VulkanExecutionContext) {
    use vkfft_rs::backend::vulkan::runtime::{TransformInput32, TransformOutput32};

    let profile = runtime.device_profile();
    for omitted_axis in [0usize, 1] {
        let forward =
            TransformIr::build(config(omitted_axis, false), Direction::Forward, profile).unwrap();
        let input = impulse_for_active_axis(omitted_axis);
        let actual = match runtime
            .execute_transform_f32(&forward, TransformInput32::Complex(&input))
            .unwrap()
        {
            TransformOutput32::Complex(values) => values,
            TransformOutput32::Real(_) => panic!("Vulkan omitDimension C2C returned real output"),
        };
        assert_close(&actual, &vec![Complex32::new(1.0, 0.0); LEN], "Vulkan");

        let inverse =
            TransformIr::build(config(omitted_axis, true), Direction::Inverse, profile).unwrap();
        let spectrum = vec![Complex32::new(1.0, 0.0); LEN];
        let actual = match runtime
            .execute_transform_f32(&inverse, TransformInput32::Complex(&spectrum))
            .unwrap()
        {
            TransformOutput32::Complex(values) => values,
            TransformOutput32::Real(_) => panic!("Vulkan omitDimension C2C returned real output"),
        };
        assert_close(&actual, &expected_inverse(omitted_axis), "Vulkan");
    }

    let input = real_input();
    let input64 = input
        .iter()
        .map(|value| f64::from(*value))
        .collect::<Vec<_>>();
    let real_forward = TransformIr::build(
        real_config(TransformKind::RealToComplex, false),
        Direction::Forward,
        profile,
    )
    .unwrap();
    let expected = complex_reference_f32(&real_forward.execute_r2c_reference(&input64).unwrap());
    let spectrum = match runtime
        .execute_transform_f32(&real_forward, TransformInput32::Real(&input))
        .unwrap()
    {
        TransformOutput32::Complex(values) => values,
        TransformOutput32::Real(_) => panic!("Vulkan omitDimension R2C returned real output"),
    };
    assert_close(&spectrum, &expected, "Vulkan");

    let real_inverse = TransformIr::build(
        real_config(TransformKind::ComplexToReal, true),
        Direction::Inverse,
        profile,
    )
    .unwrap();
    let spectrum64 = spectrum
        .iter()
        .map(|value| vkfft_rs::Complex64::new(f64::from(value.re), f64::from(value.im)))
        .collect::<Vec<_>>();
    let expected = real_reference_f32(&real_inverse.execute_c2r_reference(&spectrum64).unwrap());
    let restored = match runtime
        .execute_transform_f32(&real_inverse, TransformInput32::Complex(&spectrum))
        .unwrap()
    {
        TransformOutput32::Real(values) => values,
        TransformOutput32::Complex(_) => panic!("Vulkan omitDimension C2R returned complex output"),
    };
    assert_real_close(&restored, &expected, "Vulkan");

    for omitted_axis in [0usize, 1] {
        let r2r =
            TransformIr::build(r2r_config(omitted_axis), Direction::Forward, profile).unwrap();
        let expected = real_reference_f32(&r2r.execute_r2r_reference(&input64).unwrap());
        let actual = match runtime
            .execute_transform_f32(&r2r, TransformInput32::Real(&input))
            .unwrap()
        {
            TransformOutput32::Real(values) => values,
            TransformOutput32::Complex(_) => {
                panic!("Vulkan omitDimension DCT-II returned complex output")
            }
        };
        assert_real_close(&actual, &expected, "Vulkan");
    }
}

#[cfg(feature = "cuda-runtime")]
#[test]
fn cuda_omit_dimension_families_or_skips() {
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
fn opencl_omit_dimension_families_or_skips() {
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
fn vulkan_omit_dimension_families_or_skips() {
    use vkfft_rs::{VkFftError, backend::vulkan::runtime::VulkanExecutionContext};

    let _guard = gpu_test_lock().lock().unwrap();
    let runtime = match VulkanExecutionContext::new() {
        Ok(runtime) => runtime,
        Err(VkFftError::VulkanUnavailable(_)) => return,
        Err(error) => panic!("Vulkan context failed after loader/device discovery: {error}"),
    };
    run_vulkan(&runtime);
}
