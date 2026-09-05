#![cfg(any(
    feature = "cuda-runtime",
    feature = "opencl-runtime",
    feature = "vulkan-runtime"
))]

use vkfft_rs::{
    Complex64, ComplexDoubleDouble, DeviceProfile, Direction, DoubleDouble, FftConfig, Precision,
    TransformIr, ZeroPaddingDomain,
};

const DIMENSIONS: [usize; 2] = [3, 4];
const BATCH_COUNT: usize = 2;

fn config(precision: Precision, inverse: bool) -> FftConfig {
    let (input_row, output_row) = if inverse { (9, 7) } else { (7, 9) };
    FftConfig::new(DIMENSIONS.to_vec())
        .with_batch_count(BATCH_COUNT)
        .with_precision(precision)
        .with_inverse_normalization(inverse)
        .with_input_buffer_axis_stride(0, input_row)
        .unwrap()
        .with_output_buffer_axis_stride(0, output_row)
        .unwrap()
}

fn padded_config(precision: Precision, inverse: bool, domain: ZeroPaddingDomain) -> FftConfig {
    config(precision, inverse)
        .with_zero_padding(0, 1, 2)
        .unwrap()
        .with_zero_padding(1, 1, 3)
        .unwrap()
        .with_zero_padding_domain(domain)
}

fn dd_error(actual: DoubleDouble, expected: DoubleDouble) -> f64 {
    let delta = (actual - expected).abs();
    delta.hi.abs() + delta.lo.abs()
}

fn complex_dd_error(actual: ComplexDoubleDouble, expected: ComplexDoubleDouble) -> f64 {
    dd_error(actual.re, expected.re) + dd_error(actual.im, expected.im)
}

fn full_dd_input() -> Vec<ComplexDoubleDouble> {
    let tensor_len = DIMENSIONS.iter().product::<usize>();
    let zero = ComplexDoubleDouble::default();
    let mut input = vec![zero; tensor_len * BATCH_COUNT];
    for batch in 0..BATCH_COUNT {
        let amplitude = ComplexDoubleDouble::new(
            DoubleDouble::from_parts(1.0 + 0.25 * batch as f64, 5.0e-17),
            DoubleDouble::from_parts(-0.25 - 0.125 * batch as f64, -3.0e-17),
        );
        assert_ne!(amplitude.re.lo, 0.0);
        assert_ne!(amplitude.im.lo, 0.0);
        input[batch * tensor_len] = amplitude;
    }
    input
}

fn f64_input() -> Vec<Complex64> {
    let tensor_len = DIMENSIONS.iter().product::<usize>();
    let mut input = vec![Complex64::default(); tensor_len * BATCH_COUNT];
    for batch in 0..BATCH_COUNT {
        input[batch * tensor_len] =
            Complex64::new(1.0 + 0.25 * batch as f64, -0.25 - 0.125 * batch as f64);
    }
    input
}

fn run_with<Full, F64>(
    profile: DeviceProfile,
    device_name: &str,
    mut execute_full: Full,
    mut execute_f64: F64,
) where
    Full: FnMut(
        &TransformIr,
        &[ComplexDoubleDouble],
    ) -> Result<Vec<ComplexDoubleDouble>, vkfft_rs::VkFftError>,
    F64: FnMut(&TransformIr, &[Complex64]) -> Result<Vec<Complex64>, vkfft_rs::VkFftError>,
{
    if !profile.supports_f64 {
        return;
    }
    let tensor_len = DIMENSIONS.iter().product::<usize>();

    let full_input = full_dd_input();
    let full_forward = TransformIr::build(
        config(Precision::DoubleDouble, false),
        Direction::Forward,
        profile,
    )
    .unwrap();
    let TransformIr::ComplexNdDoubleDouble(full_forward_nd) = &full_forward else {
        panic!("formatted full-DD plan did not build ND DD IR");
    };
    assert!(full_forward_nd.input_formatted_copy.is_some());
    assert!(full_forward_nd.output_formatted_copy.is_some());
    let full_spectrum = execute_full(&full_forward, &full_input).unwrap();
    for batch in 0..BATCH_COUNT {
        let expected = full_input[batch * tensor_len];
        for value in &full_spectrum[batch * tensor_len..(batch + 1) * tensor_len] {
            let error = complex_dd_error(*value, expected);
            assert!(
                error <= 2.0e-24,
                "{device_name} formatted full-DD impulse spectrum error {error:e}"
            );
        }
    }
    let full_inverse = TransformIr::build(
        config(Precision::DoubleDouble, true),
        Direction::Inverse,
        profile,
    )
    .unwrap();
    let full_restored = execute_full(&full_inverse, &full_spectrum).unwrap();
    let full_round_trip = full_restored
        .iter()
        .copied()
        .zip(full_input.iter().copied())
        .map(|(actual, expected)| complex_dd_error(actual, expected))
        .fold(0.0f64, f64::max);
    assert!(
        full_round_trip <= 5.0e-23,
        "{device_name} formatted full-DD round-trip error {full_round_trip:e}"
    );

    let f64_input = f64_input();
    let f64_forward = TransformIr::build(
        config(Precision::DoubleDoubleF64Storage, false),
        Direction::Forward,
        profile,
    )
    .unwrap();
    let TransformIr::ComplexNdDoubleDouble(f64_forward_nd) = &f64_forward else {
        panic!("formatted DD/F64 plan did not build ND DD IR");
    };
    assert!(f64_forward_nd.input_formatted_copy.is_some());
    assert!(f64_forward_nd.output_formatted_copy.is_some());
    let f64_spectrum = execute_f64(&f64_forward, &f64_input).unwrap();
    for batch in 0..BATCH_COUNT {
        let expected = f64_input[batch * tensor_len];
        for value in &f64_spectrum[batch * tensor_len..(batch + 1) * tensor_len] {
            let error = (value.re - expected.re).abs() + (value.im - expected.im).abs();
            assert!(
                error <= 2.0e-13,
                "{device_name} formatted DD/F64 impulse spectrum error {error:e}"
            );
        }
    }
    let f64_inverse = TransformIr::build(
        config(Precision::DoubleDoubleF64Storage, true),
        Direction::Inverse,
        profile,
    )
    .unwrap();
    let f64_restored = execute_f64(&f64_inverse, &f64_spectrum).unwrap();
    let f64_round_trip = f64_restored
        .iter()
        .copied()
        .zip(f64_input.iter().copied())
        .map(|(actual, expected)| (actual.re - expected.re).abs() + (actual.im - expected.im).abs())
        .fold(0.0f64, f64::max);
    assert!(
        f64_round_trip <= 5.0e-13,
        "{device_name} formatted DD/F64 round-trip error {f64_round_trip:e}"
    );

    let full_padded_input = (0..tensor_len * BATCH_COUNT)
        .map(|index| {
            let x = index as f64;
            ComplexDoubleDouble::new(
                DoubleDouble::from_parts(
                    (0.071 * x).sin() + 0.013 * x,
                    (index + 1) as f64 * 4.0e-32,
                ),
                DoubleDouble::from_parts(
                    (0.047 * x).cos() - 0.009 * x,
                    -(index as f64 + 1.0) * 3.0e-32,
                ),
            )
        })
        .collect::<Vec<_>>();
    let f64_padded_input = full_padded_input
        .iter()
        .map(|value| Complex64::new(value.re.to_f64(), value.im.to_f64()))
        .collect::<Vec<_>>();
    for domain in [ZeroPaddingDomain::Spatial, ZeroPaddingDomain::Frequency] {
        let full_forward = TransformIr::build(
            padded_config(Precision::DoubleDouble, false, domain),
            Direction::Forward,
            profile,
        )
        .unwrap();
        let full_expected = full_forward
            .execute_double_double_reference(&full_padded_input)
            .unwrap();
        let full_actual = execute_full(&full_forward, &full_padded_input).unwrap();
        let full_error = full_actual
            .iter()
            .copied()
            .zip(full_expected.iter().copied())
            .map(|(actual, expected)| complex_dd_error(actual, expected))
            .fold(0.0f64, f64::max);
        assert!(
            full_error <= 5.0e-22,
            "{device_name} formatted+{domain:?} full-DD forward error {full_error:e}"
        );
        let full_inverse = TransformIr::build(
            padded_config(Precision::DoubleDouble, true, domain),
            Direction::Inverse,
            profile,
        )
        .unwrap();
        let full_inverse_expected = full_inverse
            .execute_double_double_reference(&full_actual)
            .unwrap();
        let full_inverse_actual = execute_full(&full_inverse, &full_actual).unwrap();
        let full_inverse_error = full_inverse_actual
            .iter()
            .copied()
            .zip(full_inverse_expected.iter().copied())
            .map(|(actual, expected)| complex_dd_error(actual, expected))
            .fold(0.0f64, f64::max);
        assert!(
            full_inverse_error <= 5.0e-22,
            "{device_name} formatted+{domain:?} full-DD inverse error {full_inverse_error:e}"
        );

        let f64_forward = TransformIr::build(
            padded_config(Precision::DoubleDoubleF64Storage, false, domain),
            Direction::Forward,
            profile,
        )
        .unwrap();
        let f64_expected = f64_forward
            .execute_complex_reference(&f64_padded_input)
            .unwrap();
        let f64_actual = execute_f64(&f64_forward, &f64_padded_input).unwrap();
        let f64_error = f64_actual
            .iter()
            .zip(&f64_expected)
            .map(|(actual, expected)| {
                (actual.re - expected.re).abs() + (actual.im - expected.im).abs()
            })
            .fold(0.0f64, f64::max);
        assert!(
            f64_error <= 5.0e-12,
            "{device_name} formatted+{domain:?} DD/F64 forward error {f64_error:e}"
        );
        let f64_inverse = TransformIr::build(
            padded_config(Precision::DoubleDoubleF64Storage, true, domain),
            Direction::Inverse,
            profile,
        )
        .unwrap();
        let f64_inverse_expected = f64_inverse.execute_complex_reference(&f64_actual).unwrap();
        let f64_inverse_actual = execute_f64(&f64_inverse, &f64_actual).unwrap();
        let f64_inverse_error = f64_inverse_actual
            .iter()
            .zip(&f64_inverse_expected)
            .map(|(actual, expected)| {
                (actual.re - expected.re).abs() + (actual.im - expected.im).abs()
            })
            .fold(0.0f64, f64::max);
        assert!(
            f64_inverse_error <= 5.0e-12,
            "{device_name} formatted+{domain:?} DD/F64 inverse error {f64_inverse_error:e}"
        );
    }
}

fn gpu_test_lock() -> &'static std::sync::Mutex<()> {
    static LOCK: std::sync::OnceLock<std::sync::Mutex<()>> = std::sync::OnceLock::new();
    LOCK.get_or_init(|| std::sync::Mutex::new(()))
}

#[cfg(feature = "cuda-runtime")]
#[test]
fn cuda_dd_formatted_strides_or_skips() {
    use vkfft_rs::backend::cuda::runtime::CudaExecutionContext;

    let _guard = gpu_test_lock()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let availability = CudaExecutionContext::probe();
    if !availability.compiler_available || availability.device_count == 0 {
        return;
    }
    let context = CudaExecutionContext::new(0).expect("CUDA context failed after successful probe");
    let profile = context.device_profile();
    let name = format!("CUDA {}", context.device_name());
    run_with(
        profile,
        &name,
        |ir, input| context.execute_transform_double_double(ir, input),
        |ir, input| context.execute_transform_double_double_f64_storage(ir, input),
    );
}

#[cfg(feature = "opencl-runtime")]
#[test]
fn opencl_dd_formatted_strides_or_skips() {
    use vkfft_rs::backend::opencl::runtime::OpenClExecutionContext;

    let _guard = gpu_test_lock()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let availability = OpenClExecutionContext::probe();
    if !availability.compiler_available || availability.device_count == 0 {
        return;
    }
    let context =
        OpenClExecutionContext::new(0).expect("OpenCL context failed after successful probe");
    let profile = context.device_profile();
    let name = format!("OpenCL {}", context.device_name());
    run_with(
        profile,
        &name,
        |ir, input| context.execute_transform_double_double(ir, input),
        |ir, input| context.execute_transform_double_double_f64_storage(ir, input),
    );
}

#[cfg(feature = "vulkan-runtime")]
#[test]
fn vulkan_dd_formatted_strides_or_skips() {
    use vkfft_rs::{VkFftError, backend::vulkan::runtime::VulkanExecutionContext};

    let _guard = gpu_test_lock()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let context = match VulkanExecutionContext::new() {
        Ok(context) => context,
        Err(VkFftError::VulkanUnavailable(_)) => return,
        Err(error) => panic!("Vulkan context failed after loader/device discovery: {error}"),
    };
    let profile = context.device_profile();
    run_with(
        profile,
        "Vulkan",
        |ir, input| context.execute_transform_double_double(ir, input),
        |ir, input| context.execute_transform_double_double_f64_storage(ir, input),
    );
}
