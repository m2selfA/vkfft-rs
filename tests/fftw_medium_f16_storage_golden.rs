use vkfft_rs::{
    AxisAlgorithm, Backend, Binary16, Complex32, Complex64, DctType, DeviceProfile, Direction,
    FftConfig, FftPlan, GpuVendor, PlannerTuning, Precision, TransformIr, TransformKind,
};

const FIXTURE: &str = include_str!("fixtures/fftw_precision_medium_f16_storage_v1.txt");

#[derive(Debug)]
struct Case {
    name: String,
    kind: String,
    shape: Vec<usize>,
    bits: Vec<u16>,
    complex: bool,
}

fn parse_half_bits(value: &str) -> u16 {
    u16::from_str_radix(value, 16).unwrap()
}

fn parse_cases() -> Vec<Case> {
    let mut lines = FIXTURE.lines();
    assert_eq!(lines.next(), Some("schema 1"));
    assert_eq!(lines.next(), Some("source fftw-3.3.9"));
    assert_eq!(
        lines.next(),
        Some("upstream-samples VkFFT-1.3.4-sample11-sample14-sample15-sample16")
    );
    assert_eq!(lines.next(), Some("storage binary16-rne-from-f32-v1"));
    assert_eq!(
        lines.next(),
        Some("input-formulas complex-f16-storage-v1 real-f16-storage-v1")
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
                "complex-f16-storage-v1"
            } else {
                "real-f16-storage-v1"
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
                bits.push(parse_half_bits(parts.next().unwrap()));
                bits.push(parse_half_bits(parts.next().unwrap()));
                assert!(parts.next().is_none());
            } else {
                bits.push(parse_half_bits(value));
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
        other => panic!("unsupported F16-storage golden kind {other}"),
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

fn quantized_complex_input64(n: usize) -> Vec<Complex64> {
    complex_input32(n)
        .into_iter()
        .map(|value| {
            Complex64::new(
                Binary16::from_f32(value.re).to_f32() as f64,
                Binary16::from_f32(value.im).to_f32() as f64,
            )
        })
        .collect()
}

fn quantized_real_input64(n: usize) -> Vec<f64> {
    real_input32(n)
        .into_iter()
        .map(|value| Binary16::from_f32(value).to_f32() as f64)
        .collect()
}

fn quantized_complex_output_bits(values: &[Complex64]) -> Vec<u16> {
    values
        .iter()
        .flat_map(|value| {
            [
                Binary16::from_f32(value.re as f32).to_bits(),
                Binary16::from_f32(value.im as f32).to_bits(),
            ]
        })
        .collect()
}

fn quantized_real_output_bits(values: &[f64]) -> Vec<u16> {
    values
        .iter()
        .map(|value| Binary16::from_f32(*value as f32).to_bits())
        .collect()
}

fn assert_algorithm_families(cases: &[Case]) {
    for case in cases.iter().filter(|case| case.kind == "c2c") {
        let plan = FftPlan::build(config_for_case(case, Precision::F16StorageF32Compute)).unwrap();
        let expected = match case.name.as_str() {
            "c2c-bluestein-n103" => "bluestein",
            "c2c-rader-n257" => "rader",
            "c2c-smooth-n1536" | "c2c-large-smooth-n4096" => "stockham",
            other => panic!("unexpected F16-storage C2C golden {other}"),
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
    feature = "level-zero-runtime",
    feature = "metal-runtime"
))]
fn half_ordered(bits: u16) -> u32 {
    if bits & 0x8000 != 0 {
        0x8000u32 - u32::from(bits & 0x7fff)
    } else {
        0x8000u32 + u32::from(bits)
    }
}

#[cfg(any(
    feature = "cuda-runtime",
    feature = "opencl-runtime",
    feature = "vulkan-runtime",
    feature = "level-zero-runtime",
    feature = "metal-runtime"
))]
fn half_ulp_distance(left: u16, right: u16) -> u32 {
    half_ordered(left).abs_diff(half_ordered(right))
}

#[cfg(any(
    feature = "cuda-runtime",
    feature = "opencl-runtime",
    feature = "vulkan-runtime",
    feature = "level-zero-runtime",
    feature = "metal-runtime"
))]
fn assert_scalar_matches_half_oracle(actual: f32, expected_bits: u16, label: &str, index: usize) {
    assert!(
        actual.is_finite(),
        "non-finite F16-storage output for {label}[{index}]"
    );
    let actual_bits = Binary16::from_f32(actual).to_bits();
    assert_eq!(
        Binary16::from_bits(actual_bits).to_f32(),
        actual,
        "{label}[{index}] was not caller-visible binary16 storage"
    );
    let expected = Binary16::from_bits(expected_bits).to_f32();
    assert!(
        expected.is_finite(),
        "non-finite FFTW F16-storage oracle for {label}[{index}]"
    );
    let ulps = half_ulp_distance(actual_bits, expected_bits);
    let absolute = (actual - expected).abs();
    assert!(
        ulps <= 2 || absolute <= 2.0e-3,
        "F16-storage mismatch for {label}[{index}]: actual={actual:e} (0x{actual_bits:04x}) expected={expected:e} (0x{expected_bits:04x}) ulps={ulps} abs={absolute:e}"
    );
}

#[cfg(any(
    feature = "cuda-runtime",
    feature = "opencl-runtime",
    feature = "vulkan-runtime",
    feature = "level-zero-runtime",
    feature = "metal-runtime"
))]
fn assert_complex_matches_half_oracle(actual: &[Complex32], case: &Case, label: &str) {
    assert_eq!(actual.len() * 2, case.bits.len());
    for (index, (value, bits)) in actual.iter().zip(case.bits.chunks_exact(2)).enumerate() {
        assert_scalar_matches_half_oracle(value.re, bits[0], &format!("{label}.re"), index);
        assert_scalar_matches_half_oracle(value.im, bits[1], &format!("{label}.im"), index);
    }
}

#[cfg(any(
    feature = "cuda-runtime",
    feature = "opencl-runtime",
    feature = "vulkan-runtime",
    feature = "level-zero-runtime",
    feature = "metal-runtime"
))]
fn assert_real_matches_half_oracle(actual: &[f32], case: &Case, label: &str) {
    assert_eq!(actual.len(), case.bits.len());
    for (index, (value, bits)) in actual.iter().zip(&case.bits).enumerate() {
        assert_scalar_matches_half_oracle(*value, *bits, label, index);
    }
}

#[test]
fn committed_f16_storage_fftw_bits_match_quantized_reference_and_planner_families() {
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
            ("c2c", true) => quantized_complex_output_bits(
                &ir.execute_complex_reference(&quantized_complex_input64(count))
                    .unwrap(),
            ),
            ("r2c", true) => quantized_complex_output_bits(
                &ir.execute_r2c_reference(&quantized_real_input64(count))
                    .unwrap(),
            ),
            ("dct2", false) => quantized_real_output_bits(
                &ir.execute_r2r_reference(&quantized_real_input64(count))
                    .unwrap(),
            ),
            _ => unreachable!(),
        };
        assert_eq!(
            actual_bits, case.bits,
            "quantized FFTW fixture changed for {}",
            case.name
        );
    }
}

#[cfg(any(
    feature = "cuda-runtime",
    feature = "opencl-runtime",
    feature = "level-zero-runtime",
    feature = "metal-runtime"
))]
fn run_native_medium_f16_storage_fftw<R: vkfft_rs::backend::NativeRuntime>(runtime: &R) {
    use vkfft_rs::backend::{NativeTransformInput32, NativeTransformOutput32};

    let cases = parse_cases();
    assert_algorithm_families(&cases);
    for case in &cases {
        let count = case.shape.iter().product::<usize>();
        let ir = TransformIr::build(
            config_for_case(case, Precision::F16StorageF32Compute),
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
                        panic!("F16-storage C2C returned real output")
                    }
                };
                assert_complex_matches_half_oracle(&actual, case, &label);
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
                        panic!("F16-storage R2C returned real output")
                    }
                };
                assert_complex_matches_half_oracle(&actual, case, &label);
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
                        panic!("F16-storage DCT-II returned complex output")
                    }
                };
                assert_real_matches_half_oracle(&actual, case, &label);
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
fn cuda_f16_storage_matches_medium_fftw_oracle_or_skips_without_device() {
    use vkfft_rs::backend::cuda::runtime::CudaExecutionContext;

    let _guard = gpu_test_lock().lock().unwrap();
    let availability = CudaExecutionContext::probe();
    if !availability.compiler_available || availability.device_count == 0 {
        return;
    }
    let context = CudaExecutionContext::new(0).expect("CUDA context failed after successful probe");
    run_native_medium_f16_storage_fftw(&context);
}

#[cfg(feature = "opencl-runtime")]
#[test]
fn opencl_f16_storage_matches_medium_fftw_oracle_or_skips_without_device() {
    use vkfft_rs::backend::opencl::runtime::OpenClExecutionContext;

    let _guard = gpu_test_lock().lock().unwrap();
    let availability = OpenClExecutionContext::probe();
    if !availability.compiler_available || availability.device_count == 0 {
        return;
    }
    let context =
        OpenClExecutionContext::new(0).expect("OpenCL context failed after successful probe");
    run_native_medium_f16_storage_fftw(&context);
}

#[cfg(feature = "level-zero-runtime")]
#[test]
fn level_zero_f16_storage_matches_medium_fftw_oracle_or_skips_without_device() {
    use vkfft_rs::backend::level_zero::runtime::LevelZeroExecutionContext;

    let _guard = gpu_test_lock().lock().unwrap();
    let require = std::env::var_os("VKFFT_REQUIRE_LEVEL_ZERO_RUNTIME").is_some();
    let availability = LevelZeroExecutionContext::probe();
    if !availability.available() {
        assert!(
            !require,
            "strict Level Zero F16-storage FFTW gate is unavailable: {}",
            availability.detail
        );
        return;
    }
    let context = LevelZeroExecutionContext::new(0)
        .expect("Level Zero context failed after successful probe");
    run_native_medium_f16_storage_fftw(&context);
}

#[cfg(feature = "metal-runtime")]
#[test]
fn metal_f16_storage_matches_medium_fftw_oracle_or_skips_without_device() {
    use vkfft_rs::backend::metal::runtime::MetalExecutionContext;

    let _guard = gpu_test_lock().lock().unwrap();
    let require = std::env::var_os("VKFFT_REQUIRE_METAL_RUNTIME").is_some();
    let availability = MetalExecutionContext::probe();
    if !availability.available() {
        assert!(
            !require,
            "strict Metal F16-storage FFTW gate is unavailable: {}",
            availability.detail
        );
        return;
    }
    let context =
        MetalExecutionContext::new(0).expect("Metal context failed after successful probe");
    run_native_medium_f16_storage_fftw(&context);
}

#[cfg(feature = "vulkan-runtime")]
#[test]
fn vulkan_f16_storage_matches_medium_fftw_oracle_or_skips_without_device() {
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
        let ir = TransformIr::build(
            config_for_case(case, Precision::F16StorageF32Compute),
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
                        panic!("Vulkan F16-storage C2C returned real output")
                    }
                };
                assert_complex_matches_half_oracle(&actual, case, &label);
            }
            ("r2c", true) => {
                let input = real_input32(count);
                let actual = match context
                    .execute_transform_f32(&ir, TransformInput32::Real(&input))
                    .unwrap()
                {
                    TransformOutput32::Complex(values) => values,
                    TransformOutput32::Real(_) => {
                        panic!("Vulkan F16-storage R2C returned real output")
                    }
                };
                assert_complex_matches_half_oracle(&actual, case, &label);
            }
            ("dct2", false) => {
                let input = real_input32(count);
                let actual = match context
                    .execute_transform_f32(&ir, TransformInput32::Real(&input))
                    .unwrap()
                {
                    TransformOutput32::Real(values) => values,
                    TransformOutput32::Complex(_) => {
                        panic!("Vulkan F16-storage DCT-II returned complex output")
                    }
                };
                assert_real_matches_half_oracle(&actual, case, &label);
            }
            _ => unreachable!(),
        }
    }
}
