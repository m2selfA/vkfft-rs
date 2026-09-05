use vkfft_rs::{
    AxisAlgorithm, Backend, Complex64, DctType, DeviceProfile, Direction, FftConfig, FftPlan,
    GpuVendor, PlannerTuning, Precision, PrecisionMetrics, TransformIr, TransformKind,
    complex_precision_metrics, real_precision_metrics,
};

const FIXTURE: &str = include_str!("fixtures/fftw_precision_medium_v1.txt");

#[derive(Debug, Clone)]
enum GoldenOutput {
    Complex(Vec<Complex64>),
    Real(Vec<f64>),
}

#[derive(Debug, Clone)]
struct GoldenCase {
    name: String,
    kind: String,
    shape: Vec<usize>,
    input_formula: String,
    output: GoldenOutput,
}

fn device() -> DeviceProfile {
    DeviceProfile {
        shared_memory_bytes: 64 * 1024,
        shared_memory_pow2_bytes: 64 * 1024,
        supports_f64: true,
        ..DeviceProfile::generic(Backend::Vulkan, GpuVendor::Nvidia)
    }
}

fn complex_input(count: usize) -> Vec<Complex64> {
    (0..count)
        .map(|index| {
            let x = index as f64;
            Complex64::new(
                (0.137 * x).sin() + 0.0013 * x,
                (0.071 * x).cos() - 0.0009 * x,
            )
        })
        .collect()
}

fn real_input(count: usize) -> Vec<f64> {
    (0..count)
        .map(|index| {
            let x = index as f64;
            (0.173 * x).sin() + 0.003 * x - 0.27 * (0.071 * x).cos()
        })
        .collect()
}

fn parse_cases() -> Vec<GoldenCase> {
    let mut lines = FIXTURE.lines();
    assert_eq!(lines.next(), Some("schema 1"));
    assert_eq!(lines.next(), Some("source fftw-3.3.9"));
    assert_eq!(
        lines.next(),
        Some("upstream-samples VkFFT-1.3.4-sample11-sample14-sample15-sample16")
    );
    assert_eq!(lines.next(), Some("input-formulas complex-v1 real-v1"));

    let mut cases = Vec::new();
    while let Some(case_line) = lines.next() {
        let name = case_line
            .strip_prefix("case ")
            .expect("malformed medium golden case")
            .to_owned();
        let kind = lines
            .next()
            .expect("missing medium golden kind")
            .strip_prefix("kind ")
            .expect("malformed medium golden kind")
            .to_owned();
        let shape = lines
            .next()
            .expect("missing medium golden shape")
            .strip_prefix("shape ")
            .expect("malformed medium golden shape")
            .split_whitespace()
            .map(|value| value.parse::<usize>().expect("invalid medium golden shape"))
            .collect::<Vec<_>>();
        assert!(!shape.is_empty());
        let input_formula = lines
            .next()
            .expect("missing medium golden input formula")
            .strip_prefix("input ")
            .expect("malformed medium golden input formula")
            .to_owned();
        let output_count = lines
            .next()
            .expect("missing medium golden output count")
            .strip_prefix("output ")
            .expect("malformed medium golden output count")
            .parse::<usize>()
            .expect("invalid medium golden output count");
        let output = match kind.as_str() {
            "c2c" | "r2c" => GoldenOutput::Complex(
                (0..output_count)
                    .map(|_| {
                        let mut values = lines
                            .next()
                            .expect("missing complex medium golden output")
                            .split_whitespace()
                            .map(|value| {
                                value
                                    .parse::<f64>()
                                    .expect("invalid complex medium golden value")
                            });
                        let re = values.next().expect("missing complex golden real part");
                        let im = values
                            .next()
                            .expect("missing complex golden imaginary part");
                        assert!(values.next().is_none());
                        Complex64::new(re, im)
                    })
                    .collect(),
            ),
            "dct2" => GoldenOutput::Real(
                (0..output_count)
                    .map(|_| {
                        lines
                            .next()
                            .expect("missing real medium golden output")
                            .parse::<f64>()
                            .expect("invalid real medium golden value")
                    })
                    .collect(),
            ),
            other => panic!("unsupported medium golden kind {other}"),
        };
        cases.push(GoldenCase {
            name,
            kind,
            shape,
            input_formula,
            output,
        });
    }
    assert_eq!(cases.len(), 7);
    cases
}

fn config_for_case(case: &GoldenCase) -> FftConfig {
    let mut config = FftConfig::new(case.shape.clone()).with_precision(Precision::F64);
    config = match case.kind.as_str() {
        "c2c" => config,
        "r2c" => config.with_transform(TransformKind::RealToComplex),
        "dct2" => config.with_transform(TransformKind::Dct(DctType::II)),
        other => panic!("unsupported medium golden kind {other}"),
    };
    if case.name == "c2c-bluestein-n103" {
        let mut tuning = PlannerTuning::portable();
        tuning.max_rader_fft_prime = 100;
        config = config.with_tuning(tuning);
    }
    config
}

fn assert_metrics(metrics: PrecisionMetrics, length: usize, label: &str, gpu: bool) {
    assert!(
        metrics.is_finite(),
        "non-finite FFTW metric for {label}: {metrics:?}"
    );
    let tolerance = if gpu { 1.0e-8 } else { 2.0e-10 } * length as f64;
    assert!(
        metrics.max_difference <= tolerance,
        "FFTW max difference for {label} exceeded {tolerance:e}: {metrics:?}"
    );
}

fn assert_algorithm_families(cases: &[GoldenCase]) {
    for case in cases.iter().filter(|case| case.kind == "c2c") {
        let plan = FftPlan::build(config_for_case(case)).unwrap();
        let expected = match case.name.as_str() {
            "c2c-bluestein-n103" => "bluestein",
            "c2c-rader-n257" => "rader",
            "c2c-smooth-n1536" | "c2c-large-smooth-n4096" => "stockham",
            other => panic!("unexpected C2C medium golden {other}"),
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

#[test]
fn committed_medium_fftw_vectors_match_reference_and_algorithm_families() {
    let cases = parse_cases();
    assert_algorithm_families(&cases);
    for case in &cases {
        let count = case.shape.iter().product::<usize>();
        let ir = TransformIr::build(config_for_case(case), Direction::Forward, device()).unwrap();
        match (&case.output, case.input_formula.as_str()) {
            (GoldenOutput::Complex(expected), "complex-v1") => {
                let input = complex_input(count);
                let actual = ir.execute_complex_reference(&input).unwrap();
                assert_metrics(
                    complex_precision_metrics(&actual, expected).unwrap(),
                    count,
                    &case.name,
                    false,
                );
            }
            (GoldenOutput::Complex(expected), "real-v1") => {
                let input = real_input(count);
                let actual = ir.execute_r2c_reference(&input).unwrap();
                assert_metrics(
                    complex_precision_metrics(&actual, expected).unwrap(),
                    count,
                    &case.name,
                    false,
                );
            }
            (GoldenOutput::Real(expected), "real-v1") => {
                let input = real_input(count);
                let actual = ir.execute_r2r_reference(&input).unwrap();
                assert_metrics(
                    real_precision_metrics(&actual, expected).unwrap(),
                    count,
                    &case.name,
                    false,
                );
            }
            _ => panic!("medium golden input/output mismatch for {}", case.name),
        }
    }
}

#[cfg(any(
    feature = "cuda-runtime",
    feature = "opencl-runtime",
    feature = "level-zero-runtime"
))]
fn run_native_medium_fftw<R: vkfft_rs::backend::NativeRuntime>(runtime: &R) {
    use vkfft_rs::backend::{NativeTransformInput64, NativeTransformOutput64};

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
        match (&case.output, case.input_formula.as_str()) {
            (GoldenOutput::Complex(expected), "complex-v1") => {
                let input = complex_input(count);
                let actual = match vkfft_rs::backend::NativeRuntime::execute_transform_f64(
                    runtime,
                    &ir,
                    NativeTransformInput64::Complex(&input),
                )
                .unwrap()
                {
                    NativeTransformOutput64::Complex(values) => values,
                    NativeTransformOutput64::Real(_) => panic!("C2C returned real output"),
                };
                assert_metrics(
                    complex_precision_metrics(&actual, expected).unwrap(),
                    count,
                    &format!("{} on {}", case.name, runtime.device_name()),
                    true,
                );
            }
            (GoldenOutput::Complex(expected), "real-v1") => {
                let input = real_input(count);
                let actual = match vkfft_rs::backend::NativeRuntime::execute_transform_f64(
                    runtime,
                    &ir,
                    NativeTransformInput64::Real(&input),
                )
                .unwrap()
                {
                    NativeTransformOutput64::Complex(values) => values,
                    NativeTransformOutput64::Real(_) => panic!("R2C returned real output"),
                };
                assert_metrics(
                    complex_precision_metrics(&actual, expected).unwrap(),
                    count,
                    &format!("{} on {}", case.name, runtime.device_name()),
                    true,
                );
            }
            (GoldenOutput::Real(expected), "real-v1") => {
                let input = real_input(count);
                let actual = match vkfft_rs::backend::NativeRuntime::execute_transform_f64(
                    runtime,
                    &ir,
                    NativeTransformInput64::Real(&input),
                )
                .unwrap()
                {
                    NativeTransformOutput64::Real(values) => values,
                    NativeTransformOutput64::Complex(_) => panic!("DCT-II returned complex output"),
                };
                assert_metrics(
                    real_precision_metrics(&actual, expected).unwrap(),
                    count,
                    &format!("{} on {}", case.name, runtime.device_name()),
                    true,
                );
            }
            _ => panic!("medium golden input/output mismatch for {}", case.name),
        }
    }
}

#[cfg(any(
    feature = "cuda-runtime",
    feature = "opencl-runtime",
    feature = "vulkan-runtime",
    feature = "level-zero-runtime"
))]
fn gpu_test_lock() -> &'static std::sync::Mutex<()> {
    static LOCK: std::sync::OnceLock<std::sync::Mutex<()>> = std::sync::OnceLock::new();
    LOCK.get_or_init(|| std::sync::Mutex::new(()))
}

#[cfg(feature = "cuda-runtime")]
#[test]
fn cuda_f64_matches_medium_fftw_vectors_or_skips_without_device() {
    use vkfft_rs::backend::{NativeRuntime, cuda::runtime::CudaExecutionContext};

    let _guard = gpu_test_lock().lock().unwrap();
    let availability = CudaExecutionContext::probe();
    if !availability.compiler_available || availability.device_count == 0 {
        return;
    }
    let context = CudaExecutionContext::new(0).expect("CUDA context failed after successful probe");
    if !NativeRuntime::device_profile(&context).supports_f64 {
        return;
    }
    run_native_medium_fftw(&context);
}

#[cfg(feature = "opencl-runtime")]
#[test]
fn opencl_f64_matches_medium_fftw_vectors_or_skips_without_device() {
    use vkfft_rs::backend::{NativeRuntime, opencl::runtime::OpenClExecutionContext};

    let _guard = gpu_test_lock().lock().unwrap();
    let availability = OpenClExecutionContext::probe();
    if !availability.compiler_available || availability.device_count == 0 {
        return;
    }
    let context =
        OpenClExecutionContext::new(0).expect("OpenCL context failed after successful probe");
    if !NativeRuntime::device_profile(&context).supports_f64 {
        return;
    }
    run_native_medium_fftw(&context);
}

#[cfg(feature = "level-zero-runtime")]
#[test]
fn level_zero_f64_matches_medium_fftw_vectors_or_skips_without_device() {
    use vkfft_rs::backend::{NativeRuntime, level_zero::runtime::LevelZeroExecutionContext};

    let _guard = gpu_test_lock().lock().unwrap();
    let require = std::env::var_os("VKFFT_REQUIRE_LEVEL_ZERO_F64_RUNTIME").is_some();
    let availability = LevelZeroExecutionContext::probe();
    if !availability.available() {
        assert!(
            !require,
            "strict Level Zero F64 FFTW gate is unavailable: {}",
            availability.detail
        );
        return;
    }
    let context = LevelZeroExecutionContext::new(0)
        .expect("Level Zero context failed after successful probe");
    if !NativeRuntime::device_profile(&context).supports_f64 {
        assert!(!require, "strict Level Zero F64 gate found no FP64 support");
        return;
    }
    run_native_medium_fftw(&context);
}

#[cfg(feature = "vulkan-runtime")]
#[test]
fn vulkan_f64_matches_medium_fftw_vectors_or_skips_without_device() {
    use vkfft_rs::{
        VkFftError,
        backend::vulkan::runtime::{TransformInput64, TransformOutput64, VulkanExecutionContext},
    };

    let _guard = gpu_test_lock().lock().unwrap();
    let context = match VulkanExecutionContext::new() {
        Ok(context) => context,
        Err(VkFftError::VulkanUnavailable(_)) => return,
        Err(error) => panic!("Vulkan context failed after loader/device discovery: {error}"),
    };
    let profile = context.device_profile();
    if !profile.supports_f64 {
        return;
    }
    let cases = parse_cases();
    assert_algorithm_families(&cases);
    for case in &cases {
        let count = case.shape.iter().product::<usize>();
        let ir = TransformIr::build(config_for_case(case), Direction::Forward, profile).unwrap();
        match (&case.output, case.input_formula.as_str()) {
            (GoldenOutput::Complex(expected), "complex-v1") => {
                let input = complex_input(count);
                let actual = match context
                    .execute_transform_f64(&ir, TransformInput64::Complex(&input))
                    .unwrap()
                {
                    TransformOutput64::Complex(values) => values,
                    TransformOutput64::Real(_) => panic!("Vulkan C2C returned real output"),
                };
                assert_metrics(
                    complex_precision_metrics(&actual, expected).unwrap(),
                    count,
                    &format!("{} on Vulkan", case.name),
                    true,
                );
            }
            (GoldenOutput::Complex(expected), "real-v1") => {
                let input = real_input(count);
                let actual = match context
                    .execute_transform_f64(&ir, TransformInput64::Real(&input))
                    .unwrap()
                {
                    TransformOutput64::Complex(values) => values,
                    TransformOutput64::Real(_) => panic!("Vulkan R2C returned real output"),
                };
                assert_metrics(
                    complex_precision_metrics(&actual, expected).unwrap(),
                    count,
                    &format!("{} on Vulkan", case.name),
                    true,
                );
            }
            (GoldenOutput::Real(expected), "real-v1") => {
                let input = real_input(count);
                let actual = match context
                    .execute_transform_f64(&ir, TransformInput64::Real(&input))
                    .unwrap()
                {
                    TransformOutput64::Real(values) => values,
                    TransformOutput64::Complex(_) => {
                        panic!("Vulkan DCT-II returned complex output")
                    }
                };
                assert_metrics(
                    real_precision_metrics(&actual, expected).unwrap(),
                    count,
                    &format!("{} on Vulkan", case.name),
                    true,
                );
            }
            _ => panic!("medium golden input/output mismatch for {}", case.name),
        }
    }
}
