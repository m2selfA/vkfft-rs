#![cfg(any(
    feature = "cuda-runtime",
    feature = "opencl-runtime",
    feature = "vulkan-runtime"
))]

use vkfft_rs::{
    Complex64, ComplexDoubleDouble, DeviceProfile, Direction, DoubleDouble, FftConfig, Precision,
    TransformIr, TransformKind,
};

const DIMENSIONS: [usize; 2] = [3, 8];
const BATCH_COUNT: usize = 2;

fn r2c_config(precision: Precision) -> FftConfig {
    FftConfig::new(DIMENSIONS.to_vec())
        .with_batch_count(BATCH_COUNT)
        .with_transform(TransformKind::RealToComplex)
        .with_precision(precision)
        .with_input_buffer_axis_stride(0, 11)
        .unwrap()
        .with_output_buffer_axis_stride(0, 7)
        .unwrap()
}

fn c2r_config(precision: Precision) -> FftConfig {
    FftConfig::new(DIMENSIONS.to_vec())
        .with_batch_count(BATCH_COUNT)
        .with_transform(TransformKind::ComplexToReal)
        .with_precision(precision)
        .with_inverse_normalization(true)
        .with_input_buffer_axis_stride(0, 7)
        .unwrap()
        .with_output_buffer_axis_stride(0, 11)
        .unwrap()
}

fn r2c_padded_config(precision: Precision) -> FftConfig {
    r2c_config(precision)
        .with_zero_padding(0, 1, 2)
        .unwrap()
        .with_zero_padding(1, 2, 4)
        .unwrap()
}

fn c2r_padded_config(precision: Precision) -> FftConfig {
    c2r_config(precision)
        .with_zero_padding(0, 1, 2)
        .unwrap()
        .with_zero_padding(1, 2, 4)
        .unwrap()
}

fn dd_input() -> Vec<DoubleDouble> {
    let tensor_len = DIMENSIONS.iter().product::<usize>();
    (0..tensor_len * BATCH_COUNT)
        .map(|index| {
            let x = index as f64;
            DoubleDouble::from_parts(
                (0.137 * x).sin() + 0.09 * (0.071 * x).cos() - 0.002 * x,
                (index + 1) as f64 * 5.0e-32,
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

fn complex_dd_error(actual: ComplexDoubleDouble, expected: ComplexDoubleDouble) -> f64 {
    dd_error(actual.re, expected.re) + dd_error(actual.im, expected.im)
}

fn gpu_test_lock() -> &'static std::sync::Mutex<()> {
    static LOCK: std::sync::OnceLock<std::sync::Mutex<()>> = std::sync::OnceLock::new();
    LOCK.get_or_init(|| std::sync::Mutex::new(()))
}

fn run_with<R2cDd, C2rDd, R2cF64, C2rF64>(
    profile: DeviceProfile,
    label: &str,
    mut r2c_dd: R2cDd,
    mut c2r_dd: C2rDd,
    mut r2c_f64: R2cF64,
    mut c2r_f64: C2rF64,
) where
    R2cDd: FnMut(
        &vkfft_rs::DoubleDoubleNdRealFftIr,
        &[DoubleDouble],
    ) -> Result<Vec<ComplexDoubleDouble>, vkfft_rs::VkFftError>,
    C2rDd: FnMut(
        &vkfft_rs::DoubleDoubleNdRealFftIr,
        &[ComplexDoubleDouble],
    ) -> Result<Vec<DoubleDouble>, vkfft_rs::VkFftError>,
    R2cF64: FnMut(
        &vkfft_rs::DoubleDoubleNdRealFftIr,
        &[f64],
    ) -> Result<Vec<Complex64>, vkfft_rs::VkFftError>,
    C2rF64: FnMut(
        &vkfft_rs::DoubleDoubleNdRealFftIr,
        &[Complex64],
    ) -> Result<Vec<f64>, vkfft_rs::VkFftError>,
{
    if !profile.supports_f64 {
        return;
    }

    let input = dd_input();
    assert!(input.iter().all(|value| value.lo != 0.0));
    let forward = TransformIr::build(
        r2c_config(Precision::DoubleDouble),
        Direction::Forward,
        profile,
    )
    .unwrap();
    let TransformIr::RealNdDoubleDouble(forward_ir) = &forward else {
        panic!("formatted full-DD R2C did not build DD ND real IR");
    };
    assert!(forward_ir.formatted_io.input_formatted_copy.is_some());
    assert!(forward_ir.formatted_io.output_formatted_copy.is_some());
    let forward_program = vkfft_rs::ProgramIr::double_double_nd_real(forward_ir).unwrap();
    assert_eq!(
        forward_program.passes.first().unwrap().name,
        "vkfft_dd_nd_real_gather_formatted_input"
    );
    assert_eq!(
        forward_program.passes.last().unwrap().name,
        "vkfft_dd_nd_real_scatter_formatted_output"
    );
    assert_eq!(
        forward_program.resources[0].elements,
        forward_ir.formatted_io.input_external_layout.batch_stride * BATCH_COUNT
    );
    assert_eq!(
        forward_program.resources[1].elements,
        forward_ir.formatted_io.output_external_layout.batch_stride * BATCH_COUNT
    );
    let expected = forward.execute_double_double_r2c_reference(&input).unwrap();
    let actual = r2c_dd(forward_ir, &input).unwrap();
    let forward_error = actual
        .iter()
        .copied()
        .zip(expected.iter().copied())
        .map(|(actual, expected)| complex_dd_error(actual, expected))
        .fold(0.0, f64::max);
    assert!(
        forward_error <= 8.0e-16,
        "{label} full-DD formatted R2C error {forward_error:e}"
    );

    let inverse = TransformIr::build(
        c2r_config(Precision::DoubleDouble),
        Direction::Inverse,
        profile,
    )
    .unwrap();
    let TransformIr::RealNdDoubleDouble(inverse_ir) = &inverse else {
        panic!("formatted full-DD C2R did not build DD ND real IR");
    };
    assert!(inverse_ir.formatted_io.input_formatted_copy.is_some());
    assert!(inverse_ir.formatted_io.output_formatted_copy.is_some());
    let inverse_program = vkfft_rs::ProgramIr::double_double_nd_real(inverse_ir).unwrap();
    assert_eq!(
        inverse_program.passes.first().unwrap().name,
        "vkfft_dd_nd_real_gather_formatted_input"
    );
    assert_eq!(
        inverse_program.passes.last().unwrap().name,
        "vkfft_dd_nd_real_scatter_formatted_output"
    );
    let restored = c2r_dd(inverse_ir, &actual).unwrap();
    let round_trip_error = restored
        .iter()
        .copied()
        .zip(input.iter().copied())
        .map(|(actual, expected)| dd_error(actual, expected))
        .fold(0.0, f64::max);
    assert!(
        round_trip_error <= 2.0e-15,
        "{label} full-DD formatted real round-trip error {round_trip_error:e}"
    );

    let f64_input = f64_input();
    let f64_forward = TransformIr::build(
        r2c_config(Precision::DoubleDoubleF64Storage),
        Direction::Forward,
        profile,
    )
    .unwrap();
    let TransformIr::RealNdDoubleDouble(f64_forward_ir) = &f64_forward else {
        panic!("formatted DD/F64 R2C did not build DD ND real IR");
    };
    let f64_expected = f64_forward.execute_r2c_reference(&f64_input).unwrap();
    let f64_actual = r2c_f64(f64_forward_ir, &f64_input).unwrap();
    let f64_forward_error = f64_actual
        .iter()
        .zip(&f64_expected)
        .map(|(actual, expected)| (*actual - *expected).norm_sqr().sqrt())
        .fold(0.0, f64::max);
    assert!(
        f64_forward_error <= 2.0e-10,
        "{label} DD/F64 formatted R2C error {f64_forward_error:e}"
    );

    let f64_inverse = TransformIr::build(
        c2r_config(Precision::DoubleDoubleF64Storage),
        Direction::Inverse,
        profile,
    )
    .unwrap();
    let TransformIr::RealNdDoubleDouble(f64_inverse_ir) = &f64_inverse else {
        panic!("formatted DD/F64 C2R did not build DD ND real IR");
    };
    let f64_restored = c2r_f64(f64_inverse_ir, &f64_actual).unwrap();
    let f64_round_trip_error = f64_restored
        .iter()
        .zip(&f64_input)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0, f64::max);
    assert!(
        f64_round_trip_error <= 2.0e-11,
        "{label} DD/F64 formatted real round-trip error {f64_round_trip_error:e}"
    );

    let padded_forward = TransformIr::build(
        r2c_padded_config(Precision::DoubleDouble),
        Direction::Forward,
        profile,
    )
    .unwrap();
    let TransformIr::RealNdDoubleDouble(padded_forward_ir) = &padded_forward else {
        panic!("formatted+padding full-DD R2C did not build DD ND real IR");
    };
    let padded_expected = padded_forward
        .execute_double_double_r2c_reference(&input)
        .unwrap();
    let padded_actual = r2c_dd(padded_forward_ir, &input).unwrap();
    let padded_forward_error = padded_actual
        .iter()
        .copied()
        .zip(padded_expected.iter().copied())
        .map(|(actual, expected)| complex_dd_error(actual, expected))
        .fold(0.0, f64::max);
    assert!(
        padded_forward_error <= 8.0e-16,
        "{label} full-DD formatted+padding R2C error {padded_forward_error:e}"
    );

    let padded_inverse = TransformIr::build(
        c2r_padded_config(Precision::DoubleDouble),
        Direction::Inverse,
        profile,
    )
    .unwrap();
    let TransformIr::RealNdDoubleDouble(padded_inverse_ir) = &padded_inverse else {
        panic!("formatted+padding full-DD C2R did not build DD ND real IR");
    };
    let padded_inverse_expected = padded_inverse
        .execute_double_double_c2r_reference(&padded_actual)
        .unwrap();
    let padded_inverse_actual = c2r_dd(padded_inverse_ir, &padded_actual).unwrap();
    let padded_inverse_error = padded_inverse_actual
        .iter()
        .copied()
        .zip(padded_inverse_expected.iter().copied())
        .map(|(actual, expected)| dd_error(actual, expected))
        .fold(0.0, f64::max);
    assert!(
        padded_inverse_error <= 2.0e-15,
        "{label} full-DD formatted+padding C2R error {padded_inverse_error:e}"
    );

    let f64_padded_forward = TransformIr::build(
        r2c_padded_config(Precision::DoubleDoubleF64Storage),
        Direction::Forward,
        profile,
    )
    .unwrap();
    let TransformIr::RealNdDoubleDouble(f64_padded_forward_ir) = &f64_padded_forward else {
        panic!("formatted+padding DD/F64 R2C did not build DD ND real IR");
    };
    let f64_padded_expected = f64_padded_forward
        .execute_r2c_reference(&f64_input)
        .unwrap();
    let f64_padded_actual = r2c_f64(f64_padded_forward_ir, &f64_input).unwrap();
    let f64_padded_forward_error = f64_padded_actual
        .iter()
        .zip(&f64_padded_expected)
        .map(|(actual, expected)| (*actual - *expected).norm_sqr().sqrt())
        .fold(0.0, f64::max);
    assert!(
        f64_padded_forward_error <= 2.0e-10,
        "{label} DD/F64 formatted+padding R2C error {f64_padded_forward_error:e}"
    );

    let f64_padded_inverse = TransformIr::build(
        c2r_padded_config(Precision::DoubleDoubleF64Storage),
        Direction::Inverse,
        profile,
    )
    .unwrap();
    let TransformIr::RealNdDoubleDouble(f64_padded_inverse_ir) = &f64_padded_inverse else {
        panic!("formatted+padding DD/F64 C2R did not build DD ND real IR");
    };
    let f64_padded_inverse_expected = f64_padded_inverse
        .execute_c2r_reference(&f64_padded_actual)
        .unwrap();
    let f64_padded_inverse_actual = c2r_f64(f64_padded_inverse_ir, &f64_padded_actual).unwrap();
    let f64_padded_inverse_error = f64_padded_inverse_actual
        .iter()
        .zip(&f64_padded_inverse_expected)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0, f64::max);
    assert!(
        f64_padded_inverse_error <= 2.0e-11,
        "{label} DD/F64 formatted+padding C2R error {f64_padded_inverse_error:e}"
    );
}

#[cfg(feature = "cuda-runtime")]
#[test]
fn cuda_dd_formatted_real_strides_or_skips() {
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
        |ir, input| context.execute_double_double_nd_r2c(ir, input),
        |ir, input| context.execute_double_double_nd_c2r(ir, input),
        |ir, input| context.execute_double_double_nd_r2c_f64_storage(ir, input),
        |ir, input| context.execute_double_double_nd_c2r_f64_storage(ir, input),
    );
}

#[cfg(feature = "opencl-runtime")]
#[test]
fn opencl_dd_formatted_real_strides_or_skips() {
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
        |ir, input| context.execute_double_double_nd_r2c(ir, input),
        |ir, input| context.execute_double_double_nd_c2r(ir, input),
        |ir, input| context.execute_double_double_nd_r2c_f64_storage(ir, input),
        |ir, input| context.execute_double_double_nd_c2r_f64_storage(ir, input),
    );
}

#[cfg(feature = "vulkan-runtime")]
#[test]
fn vulkan_dd_formatted_real_strides_or_skips() {
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
        |ir, input| context.execute_double_double_nd_r2c(ir, input),
        |ir, input| context.execute_double_double_nd_c2r(ir, input),
        |ir, input| context.execute_double_double_nd_r2c_f64_storage(ir, input),
        |ir, input| context.execute_double_double_nd_c2r_f64_storage(ir, input),
    );
}
