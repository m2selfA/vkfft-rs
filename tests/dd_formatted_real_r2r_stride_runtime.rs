#![cfg(any(
    feature = "cuda-runtime",
    feature = "opencl-runtime",
    feature = "vulkan-runtime"
))]

use vkfft_rs::{
    Complex64, ComplexDoubleDouble, DctType, DeviceProfile, Direction, DoubleDouble,
    DoubleDoubleNdR2rIr, DoubleDoubleNdRealFftIr, FftConfig, Precision, TransformIr, TransformKind,
};

type StrideCase = (Vec<usize>, Vec<(usize, usize)>, Vec<(usize, usize)>);

fn dd_error(actual: DoubleDouble, expected: DoubleDouble) -> f64 {
    let delta = (actual - expected).abs();
    delta.hi.abs() + delta.lo.abs()
}

fn complex_dd_error(actual: ComplexDoubleDouble, expected: ComplexDoubleDouble) -> f64 {
    dd_error(actual.re, expected.re) + dd_error(actual.im, expected.im)
}

fn apply_strides(
    mut config: FftConfig,
    input_strides: &[(usize, usize)],
    output_strides: &[(usize, usize)],
) -> FftConfig {
    for &(axis, stride) in input_strides {
        config = config.with_input_buffer_axis_stride(axis, stride).unwrap();
    }
    for &(axis, stride) in output_strides {
        config = config.with_output_buffer_axis_stride(axis, stride).unwrap();
    }
    config
}

fn run_with<FullR2C, FullC2R, F64R2C, F64C2R, FullR2R, F64R2R>(
    profile: DeviceProfile,
    label: &str,
    mut full_r2c: FullR2C,
    mut full_c2r: FullC2R,
    mut f64_r2c: F64R2C,
    mut f64_c2r: F64C2R,
    mut full_r2r: FullR2R,
    mut f64_r2r: F64R2R,
) where
    FullR2C: FnMut(
        &DoubleDoubleNdRealFftIr,
        &[DoubleDouble],
    ) -> Result<Vec<ComplexDoubleDouble>, vkfft_rs::VkFftError>,
    FullC2R: FnMut(
        &DoubleDoubleNdRealFftIr,
        &[ComplexDoubleDouble],
    ) -> Result<Vec<DoubleDouble>, vkfft_rs::VkFftError>,
    F64R2C: FnMut(&DoubleDoubleNdRealFftIr, &[f64]) -> Result<Vec<Complex64>, vkfft_rs::VkFftError>,
    F64C2R: FnMut(&DoubleDoubleNdRealFftIr, &[Complex64]) -> Result<Vec<f64>, vkfft_rs::VkFftError>,
    FullR2R: FnMut(
        &DoubleDoubleNdR2rIr,
        &[DoubleDouble],
    ) -> Result<Vec<DoubleDouble>, vkfft_rs::VkFftError>,
    F64R2R: FnMut(&DoubleDoubleNdR2rIr, &[f64]) -> Result<Vec<f64>, vkfft_rs::VkFftError>,
{
    let real_cases: &[StrideCase] = &[
        (vec![3, 8], vec![(0, 11)], vec![(0, 7)]),
        (vec![2, 3, 8], vec![(1, 11), (0, 40)], vec![(1, 7), (0, 24)]),
    ];
    for (dimensions, full_strides, compact_strides) in real_cases {
        let full_len = dimensions.iter().product::<usize>();
        let full_input = (0..2 * full_len)
            .map(|index| {
                let x = index as f64;
                DoubleDouble::from_parts(
                    (0.131 * x).sin() + 0.002 * x,
                    (index + 1) as f64 * 4.0e-32,
                )
            })
            .collect::<Vec<_>>();
        let f64_input = full_input
            .iter()
            .map(|value| value.to_f64())
            .collect::<Vec<_>>();

        let full_forward = TransformIr::build(
            apply_strides(
                FftConfig::new(dimensions.clone())
                    .with_batch_count(2)
                    .with_transform(TransformKind::RealToComplex)
                    .with_precision(Precision::DoubleDouble),
                full_strides,
                compact_strides,
            ),
            Direction::Forward,
            profile,
        )
        .unwrap();
        let TransformIr::RealNdDoubleDouble(full_forward_ir) = &full_forward else {
            panic!("{label} formatted full-DD R2C did not build DD ND real IR");
        };
        let full_expected = full_forward
            .execute_double_double_r2c_reference(&full_input)
            .unwrap();
        let full_actual = full_r2c(full_forward_ir, &full_input).unwrap();
        let full_error = full_actual
            .iter()
            .copied()
            .zip(full_expected.iter().copied())
            .map(|(actual, expected)| complex_dd_error(actual, expected))
            .fold(0.0f64, f64::max);
        assert!(
            full_error <= 3.0e-11,
            "{label} formatted full-DD {:?} R2C error {full_error:e}",
            dimensions
        );

        let full_inverse = TransformIr::build(
            apply_strides(
                FftConfig::new(dimensions.clone())
                    .with_batch_count(2)
                    .with_transform(TransformKind::ComplexToReal)
                    .with_precision(Precision::DoubleDouble)
                    .with_inverse_normalization(true),
                compact_strides,
                full_strides,
            ),
            Direction::Inverse,
            profile,
        )
        .unwrap();
        let TransformIr::RealNdDoubleDouble(full_inverse_ir) = &full_inverse else {
            panic!("{label} formatted full-DD C2R did not build DD ND real IR");
        };
        let full_inverse_expected = full_inverse
            .execute_double_double_c2r_reference(&full_expected)
            .unwrap();
        let full_inverse_actual = full_c2r(full_inverse_ir, &full_expected).unwrap();
        let full_inverse_error = full_inverse_actual
            .iter()
            .copied()
            .zip(full_inverse_expected.iter().copied())
            .map(|(actual, expected)| dd_error(actual, expected))
            .fold(0.0f64, f64::max);
        assert!(
            full_inverse_error <= 3.0e-11,
            "{label} formatted full-DD {:?} C2R error {full_inverse_error:e}",
            dimensions
        );

        let f64_forward = TransformIr::build(
            apply_strides(
                FftConfig::new(dimensions.clone())
                    .with_batch_count(2)
                    .with_transform(TransformKind::RealToComplex)
                    .with_precision(Precision::DoubleDoubleF64Storage),
                full_strides,
                compact_strides,
            ),
            Direction::Forward,
            profile,
        )
        .unwrap();
        let TransformIr::RealNdDoubleDouble(f64_forward_ir) = &f64_forward else {
            panic!("{label} formatted DD/F64 R2C did not build DD ND real IR");
        };
        let f64_expected = f64_forward.execute_r2c_reference(&f64_input).unwrap();
        let f64_actual = f64_r2c(f64_forward_ir, &f64_input).unwrap();
        let f64_error = f64_actual
            .iter()
            .zip(&f64_expected)
            .map(|(actual, expected)| {
                (actual.re - expected.re).abs() + (actual.im - expected.im).abs()
            })
            .fold(0.0f64, f64::max);
        assert!(
            f64_error <= 3.0e-10,
            "{label} formatted DD/F64 {:?} R2C error {f64_error:e}",
            dimensions
        );

        let f64_inverse = TransformIr::build(
            apply_strides(
                FftConfig::new(dimensions.clone())
                    .with_batch_count(2)
                    .with_transform(TransformKind::ComplexToReal)
                    .with_precision(Precision::DoubleDoubleF64Storage)
                    .with_inverse_normalization(true),
                compact_strides,
                full_strides,
            ),
            Direction::Inverse,
            profile,
        )
        .unwrap();
        let TransformIr::RealNdDoubleDouble(f64_inverse_ir) = &f64_inverse else {
            panic!("{label} formatted DD/F64 C2R did not build DD ND real IR");
        };
        let f64_inverse_expected = f64_inverse.execute_c2r_reference(&f64_expected).unwrap();
        let f64_inverse_actual = f64_c2r(f64_inverse_ir, &f64_expected).unwrap();
        let f64_inverse_error = f64_inverse_actual
            .iter()
            .zip(&f64_inverse_expected)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0f64, f64::max);
        assert!(
            f64_inverse_error <= 3.0e-10,
            "{label} formatted DD/F64 {:?} C2R error {f64_inverse_error:e}",
            dimensions
        );
    }

    let r2r_cases: &[StrideCase] = &[
        (vec![3, 4], vec![(0, 7)], vec![(0, 9)]),
        (vec![2, 3, 4], vec![(1, 6), (0, 20)], vec![(1, 7), (0, 24)]),
    ];
    for (dimensions, input_strides, output_strides) in r2r_cases {
        let tensor_len = dimensions.iter().product::<usize>();
        let full_input = (0..2 * tensor_len)
            .map(|index| {
                let x = index as f64;
                DoubleDouble::from_parts(
                    (0.173 * x).cos() - 0.003 * x,
                    -(index as f64 + 1.0) * 5.0e-32,
                )
            })
            .collect::<Vec<_>>();
        let f64_input = full_input
            .iter()
            .map(|value| value.to_f64())
            .collect::<Vec<_>>();

        let full_transform = TransformIr::build(
            apply_strides(
                FftConfig::new(dimensions.clone())
                    .with_batch_count(2)
                    .with_transform(TransformKind::Dct(DctType::II))
                    .with_precision(Precision::DoubleDouble),
                input_strides,
                output_strides,
            ),
            Direction::Forward,
            profile,
        )
        .unwrap();
        let TransformIr::RealToRealNdDoubleDouble(full_ir) = &full_transform else {
            panic!("{label} formatted full-DD R2R did not build DD ND R2R IR");
        };
        let full_expected = full_transform
            .execute_double_double_r2r_reference(&full_input)
            .unwrap();
        let full_actual = full_r2r(full_ir, &full_input).unwrap();
        let full_error = full_actual
            .iter()
            .copied()
            .zip(full_expected.iter().copied())
            .map(|(actual, expected)| dd_error(actual, expected))
            .fold(0.0f64, f64::max);
        assert!(
            full_error <= 3.0e-11,
            "{label} formatted full-DD {:?} R2R error {full_error:e}",
            dimensions
        );

        let f64_transform = TransformIr::build(
            apply_strides(
                FftConfig::new(dimensions.clone())
                    .with_batch_count(2)
                    .with_transform(TransformKind::Dct(DctType::II))
                    .with_precision(Precision::DoubleDoubleF64Storage),
                input_strides,
                output_strides,
            ),
            Direction::Forward,
            profile,
        )
        .unwrap();
        let TransformIr::RealToRealNdDoubleDouble(f64_ir) = &f64_transform else {
            panic!("{label} formatted DD/F64 R2R did not build DD ND R2R IR");
        };
        let f64_expected = f64_transform.execute_r2r_reference(&f64_input).unwrap();
        let f64_actual = f64_r2r(f64_ir, &f64_input).unwrap();
        let f64_error = f64_actual
            .iter()
            .zip(&f64_expected)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0f64, f64::max);
        assert!(
            f64_error <= 3.0e-10,
            "{label} formatted DD/F64 {:?} R2R error {f64_error:e}",
            dimensions
        );
    }
}

fn gpu_test_lock() -> &'static std::sync::Mutex<()> {
    static LOCK: std::sync::OnceLock<std::sync::Mutex<()>> = std::sync::OnceLock::new();
    LOCK.get_or_init(|| std::sync::Mutex::new(()))
}

#[cfg(feature = "cuda-runtime")]
#[test]
fn cuda_dd_formatted_real_r2r_strides_or_skips() {
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
        |ir, input| context.execute_double_double_nd_r2r(ir, input),
        |ir, input| context.execute_double_double_nd_r2r_f64_storage(ir, input),
    );
}

#[cfg(feature = "opencl-runtime")]
#[test]
fn opencl_dd_formatted_real_r2r_strides_or_skips() {
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
        |ir, input| context.execute_double_double_nd_r2r(ir, input),
        |ir, input| context.execute_double_double_nd_r2r_f64_storage(ir, input),
    );
}

#[cfg(feature = "vulkan-runtime")]
#[test]
fn vulkan_dd_formatted_real_r2r_strides_or_skips() {
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
        |ir, input| context.execute_double_double_nd_r2r(ir, input),
        |ir, input| context.execute_double_double_nd_r2r_f64_storage(ir, input),
    );
}
