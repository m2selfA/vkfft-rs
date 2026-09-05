#![cfg(any(
    feature = "cuda-runtime",
    feature = "opencl-runtime",
    feature = "vulkan-runtime"
))]

use vkfft_rs::{
    DctType, DeviceProfile, Direction, DoubleDouble, FftConfig, Precision, TransformIr,
    TransformKind,
};

const DIMENSIONS: [usize; 2] = [3, 4];
const BATCH_COUNT: usize = 2;

fn config(precision: Precision, normalize_inverse: bool) -> FftConfig {
    FftConfig::new(DIMENSIONS.to_vec())
        .with_batch_count(BATCH_COUNT)
        .with_transform(TransformKind::Dct(DctType::II))
        .with_precision(precision)
        .with_inverse_normalization(normalize_inverse)
        .with_input_buffer_axis_stride(0, 7)
        .unwrap()
        .with_output_buffer_axis_stride(0, 9)
        .unwrap()
}

fn padded_config(precision: Precision, normalize_inverse: bool) -> FftConfig {
    config(precision, normalize_inverse)
        .with_zero_padding(0, 1, 2)
        .unwrap()
        .with_zero_padding(1, 1, 3)
        .unwrap()
}

fn dd_input() -> Vec<DoubleDouble> {
    let tensor_len = DIMENSIONS.iter().product::<usize>();
    (0..tensor_len * BATCH_COUNT)
        .map(|index| {
            let x = index as f64;
            DoubleDouble::from_parts(
                (0.137 * x).sin() + 0.09 * (0.071 * x).cos() - 0.002 * x,
                (index + 1) as f64 * 3.0e-32,
            )
        })
        .collect()
}

fn f64_input() -> Vec<f64> {
    dd_input().into_iter().map(DoubleDouble::to_f64).collect()
}

fn dd_error(actual: DoubleDouble, expected: DoubleDouble) -> f64 {
    let delta = (actual - expected).abs();
    delta.hi.abs() + delta.lo.abs()
}

fn gpu_test_lock() -> &'static std::sync::Mutex<()> {
    static LOCK: std::sync::OnceLock<std::sync::Mutex<()>> = std::sync::OnceLock::new();
    LOCK.get_or_init(|| std::sync::Mutex::new(()))
}

fn run_with<Full, F64>(
    profile: DeviceProfile,
    label: &str,
    mut execute_full: Full,
    mut execute_f64: F64,
) where
    Full: FnMut(&TransformIr, &[DoubleDouble]) -> Result<Vec<DoubleDouble>, vkfft_rs::VkFftError>,
    F64: FnMut(&TransformIr, &[f64]) -> Result<Vec<f64>, vkfft_rs::VkFftError>,
{
    if !profile.supports_f64 {
        return;
    }

    let full_input = dd_input();
    let full_forward = TransformIr::build(
        config(Precision::DoubleDouble, false),
        Direction::Forward,
        profile,
    )
    .unwrap();
    let TransformIr::RealToRealNdDoubleDouble(full_nd) = &full_forward else {
        panic!("formatted full-DD R2R plan did not build ND DD R2R IR");
    };
    assert!(full_nd.input_formatted_copy.is_some());
    assert!(full_nd.output_formatted_copy.is_some());
    let full_expected = full_forward
        .execute_double_double_r2r_reference(&full_input)
        .unwrap();
    let full_actual = execute_full(&full_forward, &full_input).unwrap();
    let forward_error = full_actual
        .iter()
        .copied()
        .zip(full_expected.iter().copied())
        .map(|(actual, expected)| dd_error(actual, expected))
        .fold(0.0, f64::max);
    assert!(
        forward_error <= 5.0e-17,
        "{label} full-DD formatted R2R error {forward_error:e}"
    );

    let full_inverse = TransformIr::build(
        config(Precision::DoubleDouble, true),
        Direction::Inverse,
        profile,
    )
    .unwrap();
    let full_restored = execute_full(&full_inverse, &full_actual).unwrap();
    let round_trip_error = full_restored
        .iter()
        .copied()
        .zip(full_input.iter().copied())
        .map(|(actual, expected)| dd_error(actual, expected))
        .fold(0.0, f64::max);
    assert!(
        round_trip_error <= 5.0e-16,
        "{label} full-DD formatted R2R round-trip error {round_trip_error:e}"
    );

    let f64_input = f64_input();
    let f64_forward = TransformIr::build(
        config(Precision::DoubleDoubleF64Storage, false),
        Direction::Forward,
        profile,
    )
    .unwrap();
    let TransformIr::RealToRealNdDoubleDouble(f64_nd) = &f64_forward else {
        panic!("formatted DD/F64 R2R plan did not build ND DD R2R IR");
    };
    assert!(f64_nd.input_formatted_copy.is_some());
    assert!(f64_nd.output_formatted_copy.is_some());
    let f64_expected = f64_forward.execute_r2r_reference(&f64_input).unwrap();
    let f64_actual = execute_f64(&f64_forward, &f64_input).unwrap();
    let f64_error = f64_actual
        .iter()
        .zip(&f64_expected)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0, f64::max);
    assert!(
        f64_error <= 2.0e-10,
        "{label} DD/F64 formatted R2R error {f64_error:e}"
    );

    let f64_inverse = TransformIr::build(
        config(Precision::DoubleDoubleF64Storage, true),
        Direction::Inverse,
        profile,
    )
    .unwrap();
    let f64_restored = execute_f64(&f64_inverse, &f64_actual).unwrap();
    let f64_round_trip_error = f64_restored
        .iter()
        .zip(&f64_input)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0, f64::max);
    assert!(
        f64_round_trip_error <= 2.0e-11,
        "{label} DD/F64 formatted R2R round-trip error {f64_round_trip_error:e}"
    );

    let full_padded_forward = TransformIr::build(
        padded_config(Precision::DoubleDouble, false),
        Direction::Forward,
        profile,
    )
    .unwrap();
    let full_padded_expected = full_padded_forward
        .execute_double_double_r2r_reference(&full_input)
        .unwrap();
    let full_padded_actual = execute_full(&full_padded_forward, &full_input).unwrap();
    let padded_forward_error = full_padded_actual
        .iter()
        .copied()
        .zip(full_padded_expected.iter().copied())
        .map(|(actual, expected)| dd_error(actual, expected))
        .fold(0.0, f64::max);
    assert!(
        padded_forward_error <= 5.0e-17,
        "{label} full-DD formatted+padding R2R forward error {padded_forward_error:e}"
    );
    let full_padded_inverse = TransformIr::build(
        padded_config(Precision::DoubleDouble, true),
        Direction::Inverse,
        profile,
    )
    .unwrap();
    let full_padded_inverse_expected = full_padded_inverse
        .execute_double_double_r2r_reference(&full_padded_actual)
        .unwrap();
    let full_padded_inverse_actual =
        execute_full(&full_padded_inverse, &full_padded_actual).unwrap();
    let padded_inverse_error = full_padded_inverse_actual
        .iter()
        .copied()
        .zip(full_padded_inverse_expected.iter().copied())
        .map(|(actual, expected)| dd_error(actual, expected))
        .fold(0.0, f64::max);
    assert!(
        padded_inverse_error <= 5.0e-16,
        "{label} full-DD formatted+padding R2R inverse error {padded_inverse_error:e}"
    );

    let f64_padded_forward = TransformIr::build(
        padded_config(Precision::DoubleDoubleF64Storage, false),
        Direction::Forward,
        profile,
    )
    .unwrap();
    let f64_padded_expected = f64_padded_forward
        .execute_r2r_reference(&f64_input)
        .unwrap();
    let f64_padded_actual = execute_f64(&f64_padded_forward, &f64_input).unwrap();
    let f64_padded_forward_error = f64_padded_actual
        .iter()
        .zip(&f64_padded_expected)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0, f64::max);
    assert!(
        f64_padded_forward_error <= 2.0e-10,
        "{label} DD/F64 formatted+padding R2R forward error {f64_padded_forward_error:e}"
    );
    let f64_padded_inverse = TransformIr::build(
        padded_config(Precision::DoubleDoubleF64Storage, true),
        Direction::Inverse,
        profile,
    )
    .unwrap();
    let f64_padded_inverse_expected = f64_padded_inverse
        .execute_r2r_reference(&f64_padded_actual)
        .unwrap();
    let f64_padded_inverse_actual = execute_f64(&f64_padded_inverse, &f64_padded_actual).unwrap();
    let f64_padded_inverse_error = f64_padded_inverse_actual
        .iter()
        .zip(&f64_padded_inverse_expected)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0, f64::max);
    assert!(
        f64_padded_inverse_error <= 2.0e-10,
        "{label} DD/F64 formatted+padding R2R inverse error {f64_padded_inverse_error:e}"
    );
}

#[cfg(feature = "cuda-runtime")]
#[test]
fn cuda_dd_formatted_r2r_strides_or_skips() {
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
    let label = format!("CUDA {}", context.device_name());
    run_with(
        profile,
        &label,
        |ir, input| context.execute_transform_double_double_r2r(ir, input),
        |ir, input| context.execute_transform_double_double_r2r_f64_storage(ir, input),
    );
}

#[cfg(feature = "opencl-runtime")]
#[test]
fn opencl_dd_formatted_r2r_strides_or_skips() {
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
    let label = format!("OpenCL {}", context.device_name());
    run_with(
        profile,
        &label,
        |ir, input| context.execute_transform_double_double_r2r(ir, input),
        |ir, input| context.execute_transform_double_double_r2r_f64_storage(ir, input),
    );
}

#[cfg(feature = "vulkan-runtime")]
#[test]
fn vulkan_dd_formatted_r2r_strides_or_skips() {
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
        |ir, input| context.execute_transform_double_double_r2r(ir, input),
        |ir, input| context.execute_transform_double_double_r2r_f64_storage(ir, input),
    );
}
