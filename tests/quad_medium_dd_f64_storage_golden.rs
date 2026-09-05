use vkfft_rs::{
    AxisAlgorithm, Backend, Complex64, DctType, DeviceProfile, Direction, FftConfig, FftPlan,
    GpuVendor, PlannerTuning, Precision, TransformIr, TransformKind,
};

const FIXTURE: &str = include_str!("fixtures/quad_precision_medium_dd_f64_storage_v1.txt");

#[derive(Debug)]
enum Input {
    Complex(Vec<Complex64>),
    Real(Vec<f64>),
}

#[derive(Debug)]
enum OutputBits {
    Complex(Vec<[u64; 2]>),
    Real(Vec<u64>),
}

#[derive(Debug)]
struct Case {
    name: String,
    kind: String,
    shape: Vec<usize>,
    input: Input,
    output: OutputBits,
}

fn parse_u64_bits(value: &str) -> u64 {
    u64::from_str_radix(value, 16).unwrap()
}

fn parse_cases() -> Vec<Case> {
    let mut lines = FIXTURE.lines();
    assert_eq!(lines.next(), Some("schema 1"));
    assert_eq!(lines.next(), Some("source libquadmath-direct-113bit-v1"));
    assert_eq!(lines.next(), Some("rounding binary64-rne-from-f128-v1"));
    assert_eq!(lines.next(), Some("input-storage binary64-exact-v1"));

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
            "complex-binary64" => Input::Complex(
                (0..input_count)
                    .map(|_| {
                        let mut parts = lines.next().unwrap().split_whitespace();
                        let re = f64::from_bits(parse_u64_bits(parts.next().unwrap()));
                        let im = f64::from_bits(parse_u64_bits(parts.next().unwrap()));
                        assert!(parts.next().is_none());
                        Complex64::new(re, im)
                    })
                    .collect(),
            ),
            "real-binary64" => Input::Real(
                (0..input_count)
                    .map(|_| f64::from_bits(parse_u64_bits(lines.next().unwrap())))
                    .collect(),
            ),
            other => panic!("unsupported quad input format {other}"),
        };
        assert_eq!(lines.next(), Some("output-data"));
        let output = match output_format {
            "complex-binary64" => OutputBits::Complex(
                (0..output_count)
                    .map(|_| {
                        let mut parts = lines.next().unwrap().split_whitespace();
                        let re = parse_u64_bits(parts.next().unwrap());
                        let im = parse_u64_bits(parts.next().unwrap());
                        assert!(parts.next().is_none());
                        [re, im]
                    })
                    .collect(),
            ),
            "real-binary64" => OutputBits::Real(
                (0..output_count)
                    .map(|_| parse_u64_bits(lines.next().unwrap()))
                    .collect(),
            ),
            other => panic!("unsupported quad output format {other}"),
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
    let mut config =
        FftConfig::new(case.shape.clone()).with_precision(Precision::DoubleDoubleF64Storage);
    config = match case.kind.as_str() {
        "c2c" => config,
        "r2c" => config.with_transform(TransformKind::RealToComplex),
        "dct2" => config.with_transform(TransformKind::Dct(DctType::II)),
        other => panic!("unsupported DD/F64-storage quad kind {other}"),
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
            other => panic!("unexpected DD/F64-storage C2C golden {other}"),
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

fn f64_ordered(bits: u64) -> u64 {
    if bits & 0x8000_0000_0000_0000 != 0 {
        0x8000_0000_0000_0000u64 - (bits & 0x7fff_ffff_ffff_ffff)
    } else {
        0x8000_0000_0000_0000u64 + bits
    }
}

fn f64_ulp_distance(left: u64, right: u64) -> u64 {
    f64_ordered(left).abs_diff(f64_ordered(right))
}

#[derive(Debug, Default, Clone, Copy)]
struct ErrorEnvelope {
    max_ulps: u64,
    max_absolute: f64,
    max_relative: f64,
}

fn update_error(envelope: &mut ErrorEnvelope, actual: f64, expected_bits: u64) {
    assert!(actual.is_finite());
    let expected = f64::from_bits(expected_bits);
    assert!(expected.is_finite());
    envelope.max_ulps = envelope
        .max_ulps
        .max(f64_ulp_distance(actual.to_bits(), expected_bits));
    let absolute = (actual - expected).abs();
    envelope.max_absolute = envelope.max_absolute.max(absolute);
    if expected != 0.0 {
        envelope.max_relative = envelope.max_relative.max(absolute / expected.abs());
    }
}

fn assert_envelope(envelope: ErrorEnvelope, length: usize, label: &str, gpu: bool) {
    let absolute_limit = if gpu { 3.0e-12 } else { 8.0e-13 } * length as f64;
    let relative_limit = if gpu { 3.0e-12 } else { 8.0e-13 } * length as f64;
    assert!(
        envelope.max_ulps <= 64
            || envelope.max_absolute <= absolute_limit
            || envelope.max_relative <= relative_limit,
        "quad DD/F64-storage mismatch for {label}: {envelope:?}, abs_limit={absolute_limit:e}, rel_limit={relative_limit:e}"
    );
}

fn compare_complex(
    actual: &[Complex64],
    expected: &[[u64; 2]],
    length: usize,
    label: &str,
    gpu: bool,
) {
    assert_eq!(actual.len(), expected.len());
    let mut envelope = ErrorEnvelope::default();
    for (value, bits) in actual.iter().zip(expected) {
        update_error(&mut envelope, value.re, bits[0]);
        update_error(&mut envelope, value.im, bits[1]);
    }
    eprintln!("quad-dd {label}: {envelope:?}");
    assert_envelope(envelope, length, label, gpu);
}

fn compare_real(actual: &[f64], expected: &[u64], length: usize, label: &str, gpu: bool) {
    assert_eq!(actual.len(), expected.len());
    let mut envelope = ErrorEnvelope::default();
    for (value, bits) in actual.iter().zip(expected) {
        update_error(&mut envelope, *value, *bits);
    }
    eprintln!("quad-dd {label}: {envelope:?}");
    assert_envelope(envelope, length, label, gpu);
}

#[test]
fn committed_quad_vectors_match_double_double_f64_storage_reference_and_families() {
    let cases = parse_cases();
    assert_algorithm_families(&cases);
    for case in &cases {
        let count = case.shape.iter().product::<usize>();
        let ir = TransformIr::build(
            config_for_case(case),
            Direction::Forward,
            reference_device(),
        )
        .unwrap();
        match (&case.input, &case.output) {
            (Input::Complex(input), OutputBits::Complex(expected)) => {
                let actual = ir.execute_complex_reference(input).unwrap();
                compare_complex(&actual, expected, count, &case.name, false);
            }
            (Input::Real(input), OutputBits::Complex(expected)) => {
                let actual = ir.execute_r2c_reference(input).unwrap();
                compare_complex(&actual, expected, count, &case.name, false);
            }
            (Input::Real(input), OutputBits::Real(expected)) => {
                let actual = ir.execute_r2r_reference(input).unwrap();
                compare_real(&actual, expected, count, &case.name, false);
            }
            _ => panic!(
                "quad DD/F64-storage input/output mismatch for {}",
                case.name
            ),
        }
    }
}

#[cfg(any(
    feature = "cuda-runtime",
    feature = "opencl-runtime",
    feature = "vulkan-runtime",
    feature = "level-zero-runtime"
))]
fn run_quad_dd_f64_storage<ComplexFn, R2cFn, R2rFn>(
    profile: DeviceProfile,
    device_name: &str,
    mut execute_complex: ComplexFn,
    mut execute_r2c: R2cFn,
    mut execute_r2r: R2rFn,
) where
    ComplexFn: FnMut(&TransformIr, &[Complex64]) -> Result<Vec<Complex64>, vkfft_rs::VkFftError>,
    R2cFn: FnMut(
        &vkfft_rs::DoubleDoubleNdRealFftIr,
        &[f64],
    ) -> Result<Vec<Complex64>, vkfft_rs::VkFftError>,
    R2rFn: FnMut(&TransformIr, &[f64]) -> Result<Vec<f64>, vkfft_rs::VkFftError>,
{
    if !profile.supports_f64 {
        return;
    }
    let cases = parse_cases();
    assert_algorithm_families(&cases);
    for case in &cases {
        let count = case.shape.iter().product::<usize>();
        let ir = TransformIr::build(config_for_case(case), Direction::Forward, profile).unwrap();
        let label = format!("{} on {device_name}", case.name);
        match (&case.input, &case.output) {
            (Input::Complex(input), OutputBits::Complex(expected)) => {
                let actual = execute_complex(&ir, input).unwrap();
                compare_complex(&actual, expected, count, &label, true);
            }
            (Input::Real(input), OutputBits::Complex(expected)) => {
                let TransformIr::RealNdDoubleDouble(real) = &ir else {
                    panic!("DD/F64-storage ND R2C must build RealNdDoubleDouble IR")
                };
                let actual = execute_r2c(real, input).unwrap();
                compare_complex(&actual, expected, count, &label, true);
            }
            (Input::Real(input), OutputBits::Real(expected)) => {
                let actual = execute_r2r(&ir, input).unwrap();
                compare_real(&actual, expected, count, &label, true);
            }
            _ => panic!(
                "quad DD/F64-storage input/output mismatch for {}",
                case.name
            ),
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
fn cuda_dd_f64_storage_matches_quad_medium_oracle_or_skips_without_device() {
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
    let device_name = context.device_name().to_owned();
    run_quad_dd_f64_storage(
        profile,
        &device_name,
        |ir, input| context.execute_transform_double_double_f64_storage(ir, input),
        |ir, input| context.execute_double_double_nd_r2c_f64_storage(ir, input),
        |ir, input| context.execute_transform_double_double_r2r_f64_storage(ir, input),
    );
}

#[cfg(feature = "opencl-runtime")]
#[test]
fn opencl_dd_f64_storage_matches_quad_medium_oracle_or_skips_without_device() {
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
    let device_name = context.device_name().to_owned();
    run_quad_dd_f64_storage(
        profile,
        &device_name,
        |ir, input| context.execute_transform_double_double_f64_storage(ir, input),
        |ir, input| context.execute_double_double_nd_r2c_f64_storage(ir, input),
        |ir, input| context.execute_transform_double_double_r2r_f64_storage(ir, input),
    );
}

#[cfg(feature = "level-zero-runtime")]
#[test]
fn level_zero_dd_f64_storage_matches_quad_medium_oracle_or_skips_without_device() {
    use vkfft_rs::backend::{NativeRuntime, level_zero::runtime::LevelZeroExecutionContext};

    let _guard = gpu_test_lock()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let require = std::env::var_os("VKFFT_REQUIRE_LEVEL_ZERO_F64_RUNTIME").is_some();
    let availability = LevelZeroExecutionContext::probe();
    if !availability.available() {
        assert!(
            !require,
            "strict Level Zero DD/F64-storage quad gate is unavailable: {}",
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
            "strict Level Zero DD/F64-storage quad gate found no FP64 support"
        );
        return;
    }
    let device_name = format!("Level Zero {}", context.device_name());
    run_quad_dd_f64_storage(
        profile,
        &device_name,
        |ir, input| context.execute_transform_double_double_f64_storage(ir, input),
        |ir, input| context.execute_double_double_nd_r2c_f64_storage(ir, input),
        |ir, input| context.execute_transform_double_double_r2r_f64_storage(ir, input),
    );
}

#[cfg(feature = "vulkan-runtime")]
#[test]
fn vulkan_dd_f64_storage_matches_quad_medium_oracle_or_skips_without_device() {
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
    run_quad_dd_f64_storage(
        profile,
        "Vulkan",
        |ir, input| context.execute_transform_double_double_f64_storage(ir, input),
        |ir, input| context.execute_double_double_nd_r2c_f64_storage(ir, input),
        |ir, input| context.execute_transform_double_double_r2r_f64_storage(ir, input),
    );
}
