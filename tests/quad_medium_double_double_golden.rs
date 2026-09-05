use vkfft_rs::{
    AxisAlgorithm, Backend, ComplexDoubleDouble, DctType, DeviceProfile, Direction, DoubleDouble,
    FftConfig, FftPlan, GpuVendor, PlannerTuning, Precision, TransformIr, TransformKind,
};

const FIXTURE: &str = include_str!("fixtures/quad_precision_medium_double_double_v1.txt");

#[derive(Debug)]
enum Input {
    Complex(Vec<ComplexDoubleDouble>),
    Real(Vec<DoubleDouble>),
}

#[derive(Debug)]
enum Output {
    Complex(Vec<ComplexDoubleDouble>),
    Real(Vec<DoubleDouble>),
}

#[derive(Debug)]
struct Case {
    name: String,
    kind: String,
    shape: Vec<usize>,
    input: Input,
    output: Output,
}

fn parse_u64_bits(value: &str) -> u64 {
    u64::from_str_radix(value, 16).unwrap()
}

fn parse_dd(parts: &mut dyn Iterator<Item = &str>) -> DoubleDouble {
    let hi_bits = parse_u64_bits(parts.next().unwrap());
    let lo_bits = parse_u64_bits(parts.next().unwrap());
    let hi = f64::from_bits(hi_bits);
    let lo = f64::from_bits(lo_bits);
    let value = DoubleDouble::from_parts(hi, lo);
    assert_eq!(value.hi.to_bits(), hi_bits, "non-canonical DD high word");
    assert_eq!(value.lo.to_bits(), lo_bits, "non-canonical DD low word");
    value
}

fn parse_dd_line(line: &str) -> DoubleDouble {
    let mut parts = line.split_whitespace();
    let value = parse_dd(&mut parts);
    assert!(parts.next().is_none());
    value
}

fn parse_complex_dd_line(line: &str) -> ComplexDoubleDouble {
    let mut parts = line.split_whitespace();
    let re = parse_dd(&mut parts);
    let im = parse_dd(&mut parts);
    assert!(parts.next().is_none());
    ComplexDoubleDouble::new(re, im)
}

fn parse_cases() -> Vec<Case> {
    let mut lines = FIXTURE.lines();
    assert_eq!(lines.next(), Some("schema 1"));
    assert_eq!(
        lines.next(),
        Some("source libquadmath-direct-113bit-kahan-v1")
    );
    assert_eq!(
        lines.next(),
        Some("rounding double-double-hi-lo-rne-from-f128-v1")
    );
    assert_eq!(
        lines.next(),
        Some("input-storage double-double-binary64-v1")
    );

    let mut cases = Vec::new();
    while let Some(case_line) = lines.next() {
        let name = case_line.strip_prefix("case ").unwrap().to_owned();
        let kind = lines
            .next()
            .unwrap()
            .strip_prefix("kind ")
            .unwrap()
            .to_owned();
        let shape = lines
            .next()
            .unwrap()
            .strip_prefix("shape ")
            .unwrap()
            .split_whitespace()
            .map(|value| value.parse::<usize>().unwrap())
            .collect::<Vec<_>>();
        let input_format = lines.next().unwrap().strip_prefix("input-format ").unwrap();
        let input_count = lines
            .next()
            .unwrap()
            .strip_prefix("input-count ")
            .unwrap()
            .parse::<usize>()
            .unwrap();
        let output_format = lines
            .next()
            .unwrap()
            .strip_prefix("output-format ")
            .unwrap();
        let output_count = lines
            .next()
            .unwrap()
            .strip_prefix("output-count ")
            .unwrap()
            .parse::<usize>()
            .unwrap();

        let input = match input_format {
            "complex-double-double-binary64" => Input::Complex(
                (0..input_count)
                    .map(|_| parse_complex_dd_line(lines.next().unwrap()))
                    .collect(),
            ),
            "real-double-double-binary64" => Input::Real(
                (0..input_count)
                    .map(|_| parse_dd_line(lines.next().unwrap()))
                    .collect(),
            ),
            other => panic!("unsupported full-DD quad input format {other}"),
        };
        assert_eq!(lines.next(), Some("output-data"));
        let output = match output_format {
            "complex-double-double-binary64" => Output::Complex(
                (0..output_count)
                    .map(|_| parse_complex_dd_line(lines.next().unwrap()))
                    .collect(),
            ),
            "real-double-double-binary64" => Output::Real(
                (0..output_count)
                    .map(|_| parse_dd_line(lines.next().unwrap()))
                    .collect(),
            ),
            other => panic!("unsupported full-DD quad output format {other}"),
        };
        cases.push(Case {
            name,
            kind,
            shape,
            input,
            output,
        });
    }
    assert_eq!(cases.len(), 7);
    cases
}

fn reference_device() -> DeviceProfile {
    DeviceProfile {
        shared_memory_bytes: 64 * 1024,
        shared_memory_pow2_bytes: 64 * 1024,
        supports_f64: true,
        ..DeviceProfile::generic(Backend::Vulkan, GpuVendor::Nvidia)
    }
}

fn config_for_case(case: &Case) -> FftConfig {
    let mut config = FftConfig::new(case.shape.clone()).with_precision(Precision::DoubleDouble);
    config = match case.kind.as_str() {
        "c2c" => config,
        "r2c" => config.with_transform(TransformKind::RealToComplex),
        "dct2" => config.with_transform(TransformKind::Dct(DctType::II)),
        other => panic!("unsupported full-DD quad kind {other}"),
    };
    if case.name == "c2c-bluestein-n103" {
        let mut tuning = PlannerTuning::portable();
        tuning.max_rader_fft_prime = 100;
        config = config.with_tuning(tuning);
    }
    config
}

fn assert_algorithm_families(cases: &[Case]) {
    for case in cases.iter().filter(|case| case.kind == "c2c") {
        let plan = FftPlan::build(config_for_case(case)).unwrap();
        let expected = match case.name.as_str() {
            "c2c-bluestein-n103" => "bluestein",
            "c2c-rader-n257" => "rader",
            "c2c-smooth-n1536" | "c2c-large-smooth-n4096" => "stockham",
            other => panic!("unexpected full-DD C2C golden {other}"),
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

fn assert_low_words_are_exercised(cases: &[Case]) {
    let mut input_low_words = 0usize;
    let mut output_low_words = 0usize;
    for case in cases {
        match &case.input {
            Input::Complex(values) => {
                input_low_words += values
                    .iter()
                    .filter(|value| value.re.lo != 0.0 || value.im.lo != 0.0)
                    .count();
            }
            Input::Real(values) => {
                input_low_words += values.iter().filter(|value| value.lo != 0.0).count();
            }
        }
        match &case.output {
            Output::Complex(values) => {
                output_low_words += values
                    .iter()
                    .filter(|value| value.re.lo != 0.0 || value.im.lo != 0.0)
                    .count();
            }
            Output::Real(values) => {
                output_low_words += values.iter().filter(|value| value.lo != 0.0).count();
            }
        }
    }
    assert!(
        input_low_words > 0,
        "full-DD fixture must exercise input low words"
    );
    assert!(
        output_low_words > 0,
        "full-DD fixture must exercise output low words"
    );
}

#[derive(Debug, Default, Clone, Copy)]
struct ErrorEnvelope {
    max_absolute: f64,
    max_relative: f64,
    exact_pairs: usize,
    scalar_count: usize,
}

fn update_error(envelope: &mut ErrorEnvelope, actual: DoubleDouble, expected: DoubleDouble) {
    assert!(actual.is_finite());
    assert!(expected.is_finite());
    let normalized = DoubleDouble::from_parts(actual.hi, actual.lo);
    assert_eq!(normalized.hi.to_bits(), actual.hi.to_bits());
    assert_eq!(normalized.lo.to_bits(), actual.lo.to_bits());
    if actual.hi.to_bits() == expected.hi.to_bits() && actual.lo.to_bits() == expected.lo.to_bits()
    {
        envelope.exact_pairs += 1;
    }
    envelope.scalar_count += 1;
    let difference = (actual - expected).abs();
    let absolute = difference.hi.abs() + difference.lo.abs();
    envelope.max_absolute = envelope.max_absolute.max(absolute);
    let expected_abs = expected.hi.abs() + expected.lo.abs();
    if expected_abs != 0.0 {
        envelope.max_relative = envelope.max_relative.max(absolute / expected_abs);
    }
}

fn assert_envelope(envelope: ErrorEnvelope, length: usize, label: &str, gpu: bool) {
    // The focused real-device sweep observed <=3.03e-28 on the Rust reference and
    // <=4.22e-27 on CUDA/OpenCL/Vulkan over the seven-case corpus. Keep an explicit
    // full-DD absolute gate with modest headroom; relative error is diagnostic only
    // because mathematically-zero R2C imaginary bins legitimately make it ill-defined.
    let absolute_limit = if gpu { 2.0e-30 } else { 2.0e-31 } * length.max(1) as f64;
    assert!(
        envelope.max_absolute <= absolute_limit,
        "quad full-DD mismatch for {label}: {envelope:?}, abs_limit={absolute_limit:e}"
    );
}

fn compare_complex(
    actual: &[ComplexDoubleDouble],
    expected: &[ComplexDoubleDouble],
    length: usize,
    label: &str,
    gpu: bool,
) {
    assert_eq!(actual.len(), expected.len());
    let mut envelope = ErrorEnvelope::default();
    for (value, expected) in actual.iter().zip(expected) {
        update_error(&mut envelope, value.re, expected.re);
        update_error(&mut envelope, value.im, expected.im);
    }
    eprintln!("quad-full-dd {label}: {envelope:?}");
    assert_envelope(envelope, length, label, gpu);
}

fn compare_real(
    actual: &[DoubleDouble],
    expected: &[DoubleDouble],
    length: usize,
    label: &str,
    gpu: bool,
) {
    assert_eq!(actual.len(), expected.len());
    let mut envelope = ErrorEnvelope::default();
    for (value, expected) in actual.iter().zip(expected) {
        update_error(&mut envelope, *value, *expected);
    }
    eprintln!("quad-full-dd {label}: {envelope:?}");
    assert_envelope(envelope, length, label, gpu);
}

#[test]
fn committed_quad_vectors_match_full_double_double_reference_and_families() {
    let cases = parse_cases();
    assert_algorithm_families(&cases);
    assert_low_words_are_exercised(&cases);
    for case in &cases {
        let count = case.shape.iter().product::<usize>();
        let ir = TransformIr::build(
            config_for_case(case),
            Direction::Forward,
            reference_device(),
        )
        .unwrap();
        match (&case.input, &case.output) {
            (Input::Complex(input), Output::Complex(expected)) => {
                let actual = ir.execute_double_double_reference(input).unwrap();
                compare_complex(&actual, expected, count, &case.name, false);
            }
            (Input::Real(input), Output::Complex(expected)) => {
                let actual = ir.execute_double_double_r2c_reference(input).unwrap();
                compare_complex(&actual, expected, count, &case.name, false);
            }
            (Input::Real(input), Output::Real(expected)) => {
                let actual = ir.execute_double_double_r2r_reference(input).unwrap();
                compare_real(&actual, expected, count, &case.name, false);
            }
            _ => panic!("full-DD quad input/output mismatch for {}", case.name),
        }
    }
}

#[cfg(any(
    feature = "cuda-runtime",
    feature = "opencl-runtime",
    feature = "vulkan-runtime",
    feature = "level-zero-runtime"
))]
fn run_quad_full_dd<ComplexFn, R2cFn, R2rFn>(
    profile: DeviceProfile,
    device_name: &str,
    mut execute_complex: ComplexFn,
    mut execute_r2c: R2cFn,
    mut execute_r2r: R2rFn,
) where
    ComplexFn: FnMut(
        &TransformIr,
        &[ComplexDoubleDouble],
    ) -> Result<Vec<ComplexDoubleDouble>, vkfft_rs::VkFftError>,
    R2cFn: FnMut(
        &vkfft_rs::DoubleDoubleNdRealFftIr,
        &[DoubleDouble],
    ) -> Result<Vec<ComplexDoubleDouble>, vkfft_rs::VkFftError>,
    R2rFn: FnMut(&TransformIr, &[DoubleDouble]) -> Result<Vec<DoubleDouble>, vkfft_rs::VkFftError>,
{
    if !profile.supports_f64 {
        return;
    }
    let cases = parse_cases();
    assert_algorithm_families(&cases);
    assert_low_words_are_exercised(&cases);
    for case in &cases {
        let count = case.shape.iter().product::<usize>();
        let ir = TransformIr::build(config_for_case(case), Direction::Forward, profile).unwrap();
        let label = format!("{} on {device_name}", case.name);
        match (&case.input, &case.output) {
            (Input::Complex(input), Output::Complex(expected)) => {
                let actual = execute_complex(&ir, input).unwrap();
                compare_complex(&actual, expected, count, &label, true);
            }
            (Input::Real(input), Output::Complex(expected)) => {
                let TransformIr::RealNdDoubleDouble(real) = &ir else {
                    panic!("full-DD ND R2C must build RealNdDoubleDouble IR")
                };
                let actual = execute_r2c(real, input).unwrap();
                compare_complex(&actual, expected, count, &label, true);
            }
            (Input::Real(input), Output::Real(expected)) => {
                let actual = execute_r2r(&ir, input).unwrap();
                compare_real(&actual, expected, count, &label, true);
            }
            _ => panic!("full-DD quad input/output mismatch for {}", case.name),
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
fn cuda_full_double_double_matches_quad_medium_oracle_or_skips_without_device() {
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
    let device_name = format!("CUDA {}", context.device_name());
    run_quad_full_dd(
        profile,
        &device_name,
        |ir, input| context.execute_transform_double_double(ir, input),
        |ir, input| context.execute_double_double_nd_r2c(ir, input),
        |ir, input| context.execute_transform_double_double_r2r(ir, input),
    );
}

#[cfg(feature = "opencl-runtime")]
#[test]
fn opencl_full_double_double_matches_quad_medium_oracle_or_skips_without_device() {
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
    let device_name = format!("OpenCL {}", context.device_name());
    run_quad_full_dd(
        profile,
        &device_name,
        |ir, input| context.execute_transform_double_double(ir, input),
        |ir, input| context.execute_double_double_nd_r2c(ir, input),
        |ir, input| context.execute_transform_double_double_r2r(ir, input),
    );
}

#[cfg(feature = "level-zero-runtime")]
#[test]
fn level_zero_full_double_double_matches_quad_medium_oracle_or_skips_without_device() {
    use vkfft_rs::backend::{NativeRuntime, level_zero::runtime::LevelZeroExecutionContext};

    let _guard = gpu_test_lock()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let require = std::env::var_os("VKFFT_REQUIRE_LEVEL_ZERO_F64_RUNTIME").is_some();
    let availability = LevelZeroExecutionContext::probe();
    if !availability.available() {
        assert!(
            !require,
            "strict Level Zero full-DD quad gate is unavailable: {}",
            availability.detail
        );
        return;
    }
    let context = LevelZeroExecutionContext::new(0)
        .expect("Level Zero context failed after successful probe");
    let profile = NativeRuntime::device_profile(&context);
    if !profile.supports_f64 {
        assert!(
            !require,
            "strict Level Zero full-DD quad gate found no FP64 support"
        );
        return;
    }
    let device_name = format!("Level Zero {}", context.device_name());
    run_quad_full_dd(
        profile,
        &device_name,
        |ir, input| context.execute_transform_double_double(ir, input),
        |ir, input| context.execute_double_double_nd_r2c(ir, input),
        |ir, input| context.execute_transform_double_double_r2r(ir, input),
    );
}

#[cfg(feature = "vulkan-runtime")]
#[test]
fn vulkan_full_double_double_matches_quad_medium_oracle_or_skips_without_device() {
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
    run_quad_full_dd(
        profile,
        "Vulkan",
        |ir, input| context.execute_transform_double_double(ir, input),
        |ir, input| context.execute_double_double_nd_r2c(ir, input),
        |ir, input| context.execute_transform_double_double_r2r(ir, input),
    );
}
