use vkfft_rs::{
    AxisAlgorithm, Backend, Complex32, Complex64, DctType, DeviceProfile, Direction, FftConfig,
    FftPlan, GpuVendor, PlannerTuning, Precision, TransformIr, TransformKind,
};

const FIXTURE: &str = include_str!("fixtures/fftw_precision_medium_f64_f32_storage_v1.txt");

#[derive(Debug)]
struct Case {
    name: String,
    kind: String,
    shape: Vec<usize>,
    bits: Vec<u32>,
    complex: bool,
}

fn parse_f32_bits(value: &str) -> u32 {
    u32::from_str_radix(value, 16).unwrap()
}

fn parse_cases() -> Vec<Case> {
    let mut lines = FIXTURE.lines();
    assert_eq!(lines.next(), Some("schema 1"));
    assert_eq!(lines.next(), Some("source fftw-3.3.9"));
    assert_eq!(
        lines.next(),
        Some("upstream-samples VkFFT-1.3.4-sample11-sample14-sample15-sample16")
    );
    assert_eq!(lines.next(), Some("storage binary32-rne-from-f64-v1"));
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
            .map(|value| value.parse().unwrap())
            .collect::<Vec<_>>();
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
        let mut bits = Vec::with_capacity(if complex { count * 2 } else { count });
        for _ in 0..count {
            let value = lines.next().unwrap();
            if complex {
                let mut parts = value.split_whitespace();
                bits.push(parse_f32_bits(parts.next().unwrap()));
                bits.push(parse_f32_bits(parts.next().unwrap()));
                assert!(parts.next().is_none());
            } else {
                bits.push(parse_f32_bits(value));
            }
        }
        out.push(Case {
            name,
            kind,
            shape,
            bits,
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

fn config_for_case(case: &Case, precision: Precision) -> FftConfig {
    let mut config = FftConfig::new(case.shape.clone()).with_precision(precision);
    config = match case.kind.as_str() {
        "c2c" => config,
        "r2c" => config.with_transform(TransformKind::RealToComplex),
        "dct2" => config.with_transform(TransformKind::Dct(DctType::II)),
        other => panic!("unsupported F64/F32-storage golden kind {other}"),
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

fn complex_input64(n: usize) -> Vec<Complex64> {
    complex_input32(n)
        .into_iter()
        .map(|value| Complex64::new(value.re as f64, value.im as f64))
        .collect()
}

fn real_input64(n: usize) -> Vec<f64> {
    real_input32(n)
        .into_iter()
        .map(|value| value as f64)
        .collect()
}

fn f32_complex_output_bits(values: &[Complex64]) -> Vec<u32> {
    values
        .iter()
        .flat_map(|value| [(value.re as f32).to_bits(), (value.im as f32).to_bits()])
        .collect()
}

fn f32_real_output_bits(values: &[f64]) -> Vec<u32> {
    values
        .iter()
        .map(|value| (*value as f32).to_bits())
        .collect()
}

fn assert_algorithm_families(cases: &[Case]) {
    for case in cases.iter().filter(|case| case.kind == "c2c") {
        let plan = FftPlan::build(config_for_case(case, Precision::F64ComputeF32Storage)).unwrap();
        let expected = match case.name.as_str() {
            "c2c-bluestein-n103" => "bluestein",
            "c2c-rader-n257" => "rader",
            "c2c-smooth-n1536" | "c2c-large-smooth-n4096" => "stockham",
            other => panic!("unexpected F64/F32-storage C2C golden {other}"),
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

#[cfg(any(
    feature = "cuda-runtime",
    feature = "opencl-runtime",
    feature = "vulkan-runtime",
    feature = "level-zero-runtime"
))]
fn f32_ordered(bits: u32) -> u32 {
    if bits & 0x8000_0000 != 0 {
        0x8000_0000u32 - (bits & 0x7fff_ffff)
    } else {
        0x8000_0000u32 + bits
    }
}

#[cfg(any(
    feature = "cuda-runtime",
    feature = "opencl-runtime",
    feature = "vulkan-runtime",
    feature = "level-zero-runtime"
))]
fn f32_ulp_distance(left: u32, right: u32) -> u32 {
    f32_ordered(left).abs_diff(f32_ordered(right))
}

#[cfg(any(
    feature = "cuda-runtime",
    feature = "opencl-runtime",
    feature = "vulkan-runtime",
    feature = "level-zero-runtime"
))]
fn assert_scalar_matches_f32_oracle(actual: f32, expected_bits: u32, label: &str, index: usize) {
    assert!(
        actual.is_finite(),
        "non-finite F64/F32-storage output for {label}[{index}]"
    );
    let expected = f32::from_bits(expected_bits);
    assert!(
        expected.is_finite(),
        "non-finite FFTW F64/F32-storage oracle for {label}[{index}]"
    );
    let actual_bits = actual.to_bits();
    let ulps = f32_ulp_distance(actual_bits, expected_bits);
    let absolute = (actual - expected).abs();
    assert!(
        ulps <= 4 || absolute <= 1.0e-6,
        "F64/F32-storage mismatch for {label}[{index}]: actual={actual:e} (0x{actual_bits:08x}) expected={expected:e} (0x{expected_bits:08x}) ulps={ulps} abs={absolute:e}"
    );
}

#[cfg(any(
    feature = "cuda-runtime",
    feature = "opencl-runtime",
    feature = "vulkan-runtime",
    feature = "level-zero-runtime"
))]
fn assert_complex_matches_f32_oracle(actual: &[Complex32], case: &Case, label: &str) {
    assert_eq!(actual.len() * 2, case.bits.len());
    for (index, (value, bits)) in actual.iter().zip(case.bits.chunks_exact(2)).enumerate() {
        assert_scalar_matches_f32_oracle(value.re, bits[0], &format!("{label}.re"), index);
        assert_scalar_matches_f32_oracle(value.im, bits[1], &format!("{label}.im"), index);
    }
}

#[cfg(any(
    feature = "cuda-runtime",
    feature = "opencl-runtime",
    feature = "vulkan-runtime",
    feature = "level-zero-runtime"
))]
fn assert_real_matches_f32_oracle(actual: &[f32], case: &Case, label: &str) {
    assert_eq!(actual.len(), case.bits.len());
    for (index, (value, bits)) in actual.iter().zip(&case.bits).enumerate() {
        assert_scalar_matches_f32_oracle(*value, *bits, label, index);
    }
}

#[test]
fn committed_f64_compute_f32_storage_fftw_bits_match_reference_and_planner_families() {
    let cases = parse_cases();
    assert_algorithm_families(&cases);
    for case in &cases {
        let count = case.shape.iter().product::<usize>();
        let ir = TransformIr::build(
            config_for_case(case, Precision::F64),
            Direction::Forward,
            reference_device(),
        )
        .unwrap();
        let actual_bits = match (case.kind.as_str(), case.complex) {
            ("c2c", true) => f32_complex_output_bits(
                &ir.execute_complex_reference(&complex_input64(count))
                    .unwrap(),
            ),
            ("r2c", true) => {
                f32_complex_output_bits(&ir.execute_r2c_reference(&real_input64(count)).unwrap())
            }
            ("dct2", false) => {
                f32_real_output_bits(&ir.execute_r2r_reference(&real_input64(count)).unwrap())
            }
            _ => unreachable!(),
        };
        assert_eq!(
            actual_bits, case.bits,
            "F32-storage FFTW fixture changed for {}",
            case.name
        );
    }
}

#[cfg(any(
    feature = "cuda-runtime",
    feature = "opencl-runtime",
    feature = "level-zero-runtime"
))]
fn run_native_medium_f64_f32_storage_fftw<R: vkfft_rs::backend::NativeRuntime>(runtime: &R) {
    use vkfft_rs::backend::{NativeTransformInput32, NativeTransformOutput32};

    let cases = parse_cases();
    assert_algorithm_families(&cases);
    for case in &cases {
        let count = case.shape.iter().product::<usize>();
        let ir = TransformIr::build(
            config_for_case(case, Precision::F64ComputeF32Storage),
            Direction::Forward,
            runtime.device_profile(),
        )
        .unwrap();
        let label = format!("{} on {}", case.name, runtime.device_name());
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
                    NativeTransformOutput32::Real(_) => {
                        panic!("F64/F32-storage C2C returned real output")
                    }
                };
                assert_complex_matches_f32_oracle(&actual, case, &label);
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
                    NativeTransformOutput32::Real(_) => {
                        panic!("F64/F32-storage R2C returned real output")
                    }
                };
                assert_complex_matches_f32_oracle(&actual, case, &label);
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
                        panic!("F64/F32-storage DCT-II returned complex output")
                    }
                };
                assert_real_matches_f32_oracle(&actual, case, &label);
            }
            _ => unreachable!(),
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
fn cuda_f64_compute_f32_storage_matches_medium_fftw_oracle_or_skips_without_device() {
    use vkfft_rs::backend::cuda::runtime::CudaExecutionContext;

    let _guard = gpu_test_lock().lock().unwrap();
    let availability = CudaExecutionContext::probe();
    if !availability.compiler_available || availability.device_count == 0 {
        return;
    }
    let context = CudaExecutionContext::new(0).expect("CUDA context failed after successful probe");
    if !context.device_profile().supports_f64 {
        return;
    }
    run_native_medium_f64_f32_storage_fftw(&context);
}

#[cfg(feature = "opencl-runtime")]
#[test]
fn opencl_f64_compute_f32_storage_matches_medium_fftw_oracle_or_skips_without_device() {
    use vkfft_rs::backend::opencl::runtime::OpenClExecutionContext;

    let _guard = gpu_test_lock().lock().unwrap();
    let availability = OpenClExecutionContext::probe();
    if !availability.compiler_available || availability.device_count == 0 {
        return;
    }
    let context =
        OpenClExecutionContext::new(0).expect("OpenCL context failed after successful probe");
    if !context.device_profile().supports_f64 {
        return;
    }
    run_native_medium_f64_f32_storage_fftw(&context);
}

#[cfg(feature = "level-zero-runtime")]
#[test]
fn level_zero_f64_compute_f32_storage_matches_medium_fftw_oracle_or_skips_without_device() {
    use vkfft_rs::backend::{NativeRuntime, level_zero::runtime::LevelZeroExecutionContext};

    let _guard = gpu_test_lock().lock().unwrap();
    let require = std::env::var_os("VKFFT_REQUIRE_LEVEL_ZERO_F64_RUNTIME").is_some();
    let availability = LevelZeroExecutionContext::probe();
    if !availability.available() {
        assert!(
            !require,
            "strict Level Zero F64/F32-storage FFTW gate is unavailable: {}",
            availability.detail
        );
        return;
    }
    let context = LevelZeroExecutionContext::new(0)
        .expect("Level Zero context failed after successful probe");
    if !NativeRuntime::device_profile(&context).supports_f64 {
        assert!(
            !require,
            "strict Level Zero F64/F32-storage gate found no FP64 support"
        );
        return;
    }
    run_native_medium_f64_f32_storage_fftw(&context);
}

#[cfg(feature = "vulkan-runtime")]
#[test]
fn vulkan_f64_compute_f32_storage_matches_medium_fftw_oracle_or_skips_without_device() {
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
    if !profile.supports_f64 {
        return;
    }
    let cases = parse_cases();
    assert_algorithm_families(&cases);
    for case in &cases {
        let count = case.shape.iter().product::<usize>();
        let ir = TransformIr::build(
            config_for_case(case, Precision::F64ComputeF32Storage),
            Direction::Forward,
            profile,
        )
        .unwrap();
        let label = format!("{} on Vulkan", case.name);
        match (case.kind.as_str(), case.complex) {
            ("c2c", true) => {
                let input = complex_input32(count);
                let actual = match context
                    .execute_transform_f32(&ir, TransformInput32::Complex(&input))
                    .unwrap()
                {
                    TransformOutput32::Complex(values) => values,
                    TransformOutput32::Real(_) => {
                        panic!("Vulkan F64/F32-storage C2C returned real output")
                    }
                };
                assert_complex_matches_f32_oracle(&actual, case, &label);
            }
            ("r2c", true) => {
                let input = real_input32(count);
                let actual = match context
                    .execute_transform_f32(&ir, TransformInput32::Real(&input))
                    .unwrap()
                {
                    TransformOutput32::Complex(values) => values,
                    TransformOutput32::Real(_) => {
                        panic!("Vulkan F64/F32-storage R2C returned real output")
                    }
                };
                assert_complex_matches_f32_oracle(&actual, case, &label);
            }
            ("dct2", false) => {
                let input = real_input32(count);
                let actual = match context
                    .execute_transform_f32(&ir, TransformInput32::Real(&input))
                    .unwrap()
                {
                    TransformOutput32::Real(values) => values,
                    TransformOutput32::Complex(_) => {
                        panic!("Vulkan F64/F32-storage DCT-II returned complex output")
                    }
                };
                assert_real_matches_f32_oracle(&actual, case, &label);
            }
            _ => unreachable!(),
        }
    }
}
