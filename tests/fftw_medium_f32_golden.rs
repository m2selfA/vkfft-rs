use vkfft_rs::{
    AxisAlgorithm, Backend, Complex32, Complex64, DctType, DeviceProfile, Direction, FftConfig,
    FftPlan, GpuVendor, PlannerTuning, Precision, PrecisionMetrics, TransformIr, TransformKind,
    complex_precision_metrics, real_precision_metrics,
};

const FIXTURE: &str = include_str!("fixtures/fftw_precision_medium_f32_v1.txt");

#[derive(Debug)]
struct Case {
    name: String,
    kind: String,
    shape: Vec<usize>,
    values: Vec<f64>,
    complex: bool,
}

fn parse_cases() -> Vec<Case> {
    let mut lines = FIXTURE.lines();
    assert_eq!(lines.next(), Some("schema 1"));
    assert_eq!(lines.next(), Some("source fftw-3.3.9"));
    assert_eq!(
        lines.next(),
        Some("upstream-samples VkFFT-1.3.4-sample11-sample14-sample15-sample16")
    );
    assert_eq!(
        lines.next(),
        Some("input-formulas complex-f32-v1 real-f32-v1")
    );
    let mut out = Vec::new();
    while let Some(case) = lines.next() {
        let name = case.strip_prefix("case ").unwrap().to_string();
        let kind = lines
            .next()
            .unwrap()
            .strip_prefix("kind ")
            .unwrap()
            .to_string();
        let shape = lines
            .next()
            .unwrap()
            .strip_prefix("shape ")
            .unwrap()
            .split_whitespace()
            .map(|v| v.parse().unwrap())
            .collect();
        let input_formula = lines.next().unwrap().strip_prefix("input ").unwrap();
        assert_eq!(
            input_formula,
            if kind == "c2c" {
                "complex-f32-v1"
            } else {
                "real-f32-v1"
            }
        );
        let count: usize = lines
            .next()
            .unwrap()
            .strip_prefix("output ")
            .unwrap()
            .parse()
            .unwrap();
        let complex = kind != "dct2";
        let mut values = Vec::with_capacity(if complex { count * 2 } else { count });
        for _ in 0..count {
            let value = lines.next().unwrap();
            if complex {
                let mut parts = value.split_whitespace();
                values.push(parts.next().unwrap().parse().unwrap());
                values.push(parts.next().unwrap().parse().unwrap());
                assert!(parts.next().is_none());
            } else {
                values.push(value.parse().unwrap());
            }
        }
        out.push(Case {
            name,
            kind,
            shape,
            values,
            complex,
        });
    }
    assert_eq!(out.len(), 7);
    out
}

fn reference_device() -> DeviceProfile {
    DeviceProfile {
        supports_f64: true,
        ..DeviceProfile::generic(Backend::Vulkan, GpuVendor::Nvidia)
    }
}

fn config_for_case(case: &Case) -> FftConfig {
    let mut config = FftConfig::new(case.shape.clone());
    config = match case.kind.as_str() {
        "c2c" => config,
        "r2c" => config.with_transform(TransformKind::RealToComplex),
        "dct2" => config.with_transform(TransformKind::Dct(DctType::II)),
        other => panic!("unsupported F32 golden kind {other}"),
    };
    if case.name == "c2c-bluestein-n103" {
        let mut tuning = PlannerTuning::portable();
        tuning.max_rader_fft_prime = 100;
        config = config.with_tuning(tuning);
    }
    config
}

fn complex_input32(n: usize) -> Vec<Complex32> {
    (0..n)
        .map(|i| {
            let x = i as f32;
            Complex32::new(
                (0.137f32 * x).sin() + 0.0013f32 * x,
                (0.071f32 * x).cos() - 0.0009f32 * x,
            )
        })
        .collect()
}

fn real_input32(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let x = i as f32;
            (0.173f32 * x).sin() + 0.003f32 * x - 0.27f32 * (0.071f32 * x).cos()
        })
        .collect()
}

fn expected_complex(case: &Case) -> Vec<Complex64> {
    assert!(case.complex);
    case.values
        .chunks_exact(2)
        .map(|values| Complex64::new(values[0], values[1]))
        .collect()
}

fn assert_algorithm_families(cases: &[Case]) {
    for case in cases.iter().filter(|case| case.kind == "c2c") {
        let plan = FftPlan::build(config_for_case(case)).unwrap();
        let expected = match case.name.as_str() {
            "c2c-bluestein-n103" => "bluestein",
            "c2c-rader-n257" => "rader",
            "c2c-smooth-n1536" | "c2c-large-smooth-n4096" => "stockham",
            other => panic!("unexpected F32 C2C golden {other}"),
        };
        let actual = match plan.axes[0].algorithm {
            AxisAlgorithm::Stockham { .. } => "stockham",
            AxisAlgorithm::Rader { .. } => "rader",
            AxisAlgorithm::Bluestein { .. } => "bluestein",
        };
        assert_eq!(
            actual, expected,
            "algorithm family changed for {}",
            case.name
        );
    }
}

fn assert_metrics(metrics: PrecisionMetrics, length: usize, label: &str, gpu: bool) {
    assert!(
        metrics.is_finite(),
        "non-finite F32 FFTW metric for {label}: {metrics:?}"
    );
    let tolerance = if gpu { 1.0e-4 } else { 2.0e-10 } * length as f64;
    assert!(
        metrics.max_difference <= tolerance,
        "F32 FFTW max difference for {label} exceeded {tolerance:e}: {metrics:?}"
    );
}

#[test]
fn committed_f32_fftw_vectors_match_reference_and_planner_families() {
    let cases = parse_cases();
    assert_algorithm_families(&cases);
    for case in &cases {
        let config = config_for_case(case).with_precision(Precision::F64);
        let ir = TransformIr::build(config, Direction::Forward, reference_device()).unwrap();
        let count = case.shape.iter().product::<usize>();
        match (case.kind.as_str(), case.complex) {
            ("c2c", true) => {
                let input = complex_input32(count)
                    .into_iter()
                    .map(|value| Complex64::new(value.re as f64, value.im as f64))
                    .collect::<Vec<_>>();
                let expected = expected_complex(case);
                let actual = ir.execute_complex_reference(&input).unwrap();
                assert_metrics(
                    complex_precision_metrics(&actual, &expected).unwrap(),
                    count,
                    &case.name,
                    false,
                );
            }
            ("r2c", true) => {
                let input = real_input32(count)
                    .into_iter()
                    .map(|value| value as f64)
                    .collect::<Vec<_>>();
                let expected = expected_complex(case);
                let actual = ir.execute_r2c_reference(&input).unwrap();
                assert_metrics(
                    complex_precision_metrics(&actual, &expected).unwrap(),
                    count,
                    &case.name,
                    false,
                );
            }
            ("dct2", false) => {
                let input = real_input32(count)
                    .into_iter()
                    .map(|value| value as f64)
                    .collect::<Vec<_>>();
                let actual = ir.execute_r2r_reference(&input).unwrap();
                assert_metrics(
                    real_precision_metrics(&actual, &case.values).unwrap(),
                    count,
                    &case.name,
                    false,
                );
            }
            _ => unreachable!(),
        }
    }
}

#[cfg(any(
    feature = "cuda-runtime",
    feature = "opencl-runtime",
    feature = "level-zero-runtime",
    feature = "metal-runtime"
))]
fn run_native_medium_f32_fftw<R: vkfft_rs::backend::NativeRuntime>(runtime: &R) {
    use vkfft_rs::backend::{NativeTransformInput32, NativeTransformOutput32};

    let cases = parse_cases();
    assert_algorithm_families(&cases);
    for case in &cases {
        let count = case.shape.iter().product::<usize>();
        let ir = TransformIr::build(
            config_for_case(case),
            Direction::Forward,
            runtime.device_profile(),
        )
        .unwrap();
        match (case.kind.as_str(), case.complex) {
            ("c2c", true) => {
                let input = complex_input32(count);
                let actual = match vkfft_rs::backend::NativeRuntime::execute_transform_f32(
                    runtime,
                    &ir,
                    NativeTransformInput32::Complex(&input),
                )
                .unwrap()
                {
                    NativeTransformOutput32::Complex(values) => values,
                    NativeTransformOutput32::Real(_) => panic!("F32 C2C returned real output"),
                };
                let actual = actual
                    .into_iter()
                    .map(|value| Complex64::new(value.re as f64, value.im as f64))
                    .collect::<Vec<_>>();
                let expected = expected_complex(case);
                assert_metrics(
                    complex_precision_metrics(&actual, &expected).unwrap(),
                    count,
                    &format!("{} on {}", case.name, runtime.device_name()),
                    true,
                );
            }
            ("r2c", true) => {
                let input = real_input32(count);
                let actual = match vkfft_rs::backend::NativeRuntime::execute_transform_f32(
                    runtime,
                    &ir,
                    NativeTransformInput32::Real(&input),
                )
                .unwrap()
                {
                    NativeTransformOutput32::Complex(values) => values,
                    NativeTransformOutput32::Real(_) => panic!("F32 R2C returned real output"),
                };
                let actual = actual
                    .into_iter()
                    .map(|value| Complex64::new(value.re as f64, value.im as f64))
                    .collect::<Vec<_>>();
                let expected = expected_complex(case);
                assert_metrics(
                    complex_precision_metrics(&actual, &expected).unwrap(),
                    count,
                    &format!("{} on {}", case.name, runtime.device_name()),
                    true,
                );
            }
            ("dct2", false) => {
                let input = real_input32(count);
                let actual = match vkfft_rs::backend::NativeRuntime::execute_transform_f32(
                    runtime,
                    &ir,
                    NativeTransformInput32::Real(&input),
                )
                .unwrap()
                {
                    NativeTransformOutput32::Real(values) => values,
                    NativeTransformOutput32::Complex(_) => {
                        panic!("F32 DCT-II returned complex output")
                    }
                };
                let actual = actual
                    .into_iter()
                    .map(|value| value as f64)
                    .collect::<Vec<_>>();
                assert_metrics(
                    real_precision_metrics(&actual, &case.values).unwrap(),
                    count,
                    &format!("{} on {}", case.name, runtime.device_name()),
                    true,
                );
            }
            _ => unreachable!(),
        }
    }
}

#[cfg(any(
    feature = "cuda-runtime",
    feature = "opencl-runtime",
    feature = "vulkan-runtime",
    feature = "level-zero-runtime",
    feature = "metal-runtime"
))]
fn gpu_test_lock() -> &'static std::sync::Mutex<()> {
    static LOCK: std::sync::OnceLock<std::sync::Mutex<()>> = std::sync::OnceLock::new();
    LOCK.get_or_init(|| std::sync::Mutex::new(()))
}

#[cfg(feature = "cuda-runtime")]
#[test]
fn cuda_f32_matches_medium_fftw_vectors_or_skips_without_device() {
    use vkfft_rs::backend::cuda::runtime::CudaExecutionContext;

    let _guard = gpu_test_lock().lock().unwrap();
    let availability = CudaExecutionContext::probe();
    if !availability.compiler_available || availability.device_count == 0 {
        return;
    }
    let context = CudaExecutionContext::new(0).expect("CUDA context failed after successful probe");
    run_native_medium_f32_fftw(&context);
}

#[cfg(feature = "opencl-runtime")]
#[test]
fn opencl_f32_matches_medium_fftw_vectors_or_skips_without_device() {
    use vkfft_rs::backend::opencl::runtime::OpenClExecutionContext;

    let _guard = gpu_test_lock().lock().unwrap();
    let availability = OpenClExecutionContext::probe();
    if !availability.compiler_available || availability.device_count == 0 {
        return;
    }
    let context =
        OpenClExecutionContext::new(0).expect("OpenCL context failed after successful probe");
    run_native_medium_f32_fftw(&context);
}

#[cfg(feature = "level-zero-runtime")]
#[test]
fn level_zero_f32_matches_medium_fftw_vectors_or_skips_without_device() {
    use vkfft_rs::backend::level_zero::runtime::LevelZeroExecutionContext;

    let _guard = gpu_test_lock().lock().unwrap();
    let require = std::env::var_os("VKFFT_REQUIRE_LEVEL_ZERO_RUNTIME").is_some();
    let availability = LevelZeroExecutionContext::probe();
    if !availability.available() {
        assert!(
            !require,
            "strict Level Zero F32 FFTW gate is unavailable: {}",
            availability.detail
        );
        return;
    }
    let context = LevelZeroExecutionContext::new(0)
        .expect("Level Zero context failed after successful probe");
    run_native_medium_f32_fftw(&context);
}

#[cfg(feature = "metal-runtime")]
#[test]
fn metal_f32_matches_medium_fftw_vectors_or_skips_without_device() {
    use vkfft_rs::backend::metal::runtime::MetalExecutionContext;

    let _guard = gpu_test_lock().lock().unwrap();
    let require = std::env::var_os("VKFFT_REQUIRE_METAL_RUNTIME").is_some();
    let availability = MetalExecutionContext::probe();
    if !availability.available() {
        assert!(
            !require,
            "strict Metal F32 FFTW gate is unavailable: {}",
            availability.detail
        );
        return;
    }
    let context =
        MetalExecutionContext::new(0).expect("Metal context failed after successful probe");
    run_native_medium_f32_fftw(&context);
}

#[cfg(feature = "vulkan-runtime")]
#[test]
fn vulkan_f32_matches_medium_fftw_vectors_or_skips_without_device() {
    use vkfft_rs::{
        VkFftError,
        backend::vulkan::runtime::{TransformInput32, TransformOutput32, VulkanExecutionContext},
    };

    let _guard = gpu_test_lock().lock().unwrap();
    let context = match VulkanExecutionContext::new() {
        Ok(context) => context,
        Err(VkFftError::VulkanUnavailable(_)) => return,
        Err(error) => panic!("Vulkan context failed after loader/device discovery: {error}"),
    };
    let profile = context.device_profile();
    let cases = parse_cases();
    assert_algorithm_families(&cases);
    for case in &cases {
        let count = case.shape.iter().product::<usize>();
        let ir = TransformIr::build(config_for_case(case), Direction::Forward, profile).unwrap();
        match (case.kind.as_str(), case.complex) {
            ("c2c", true) => {
                let input = complex_input32(count);
                let actual = match context
                    .execute_transform_f32(&ir, TransformInput32::Complex(&input))
                    .unwrap()
                {
                    TransformOutput32::Complex(values) => values,
                    TransformOutput32::Real(_) => panic!("Vulkan F32 C2C returned real output"),
                };
                let actual = actual
                    .into_iter()
                    .map(|value| Complex64::new(value.re as f64, value.im as f64))
                    .collect::<Vec<_>>();
                let expected = expected_complex(case);
                assert_metrics(
                    complex_precision_metrics(&actual, &expected).unwrap(),
                    count,
                    &format!("{} on Vulkan", case.name),
                    true,
                );
            }
            ("r2c", true) => {
                let input = real_input32(count);
                let actual = match context
                    .execute_transform_f32(&ir, TransformInput32::Real(&input))
                    .unwrap()
                {
                    TransformOutput32::Complex(values) => values,
                    TransformOutput32::Real(_) => panic!("Vulkan F32 R2C returned real output"),
                };
                let actual = actual
                    .into_iter()
                    .map(|value| Complex64::new(value.re as f64, value.im as f64))
                    .collect::<Vec<_>>();
                let expected = expected_complex(case);
                assert_metrics(
                    complex_precision_metrics(&actual, &expected).unwrap(),
                    count,
                    &format!("{} on Vulkan", case.name),
                    true,
                );
            }
            ("dct2", false) => {
                let input = real_input32(count);
                let actual = match context
                    .execute_transform_f32(&ir, TransformInput32::Real(&input))
                    .unwrap()
                {
                    TransformOutput32::Real(values) => values,
                    TransformOutput32::Complex(_) => {
                        panic!("Vulkan F32 DCT-II returned complex output")
                    }
                };
                let actual = actual
                    .into_iter()
                    .map(|value| value as f64)
                    .collect::<Vec<_>>();
                assert_metrics(
                    real_precision_metrics(&actual, &case.values).unwrap(),
                    count,
                    &format!("{} on Vulkan", case.name),
                    true,
                );
            }
            _ => unreachable!(),
        }
    }
}
