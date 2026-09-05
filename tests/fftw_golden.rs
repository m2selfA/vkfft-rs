use vkfft_rs::{
    Backend, Complex64, DctType, DeviceProfile, Direction, FftConfig, GpuVendor, Precision,
    TransformIr, TransformKind, complex_precision_metrics, real_precision_metrics,
};

const FIXTURE: &str = include_str!("fixtures/fftw_precision_v1.txt");

fn device() -> DeviceProfile {
    DeviceProfile {
        shared_memory_bytes: 64 * 1024,
        shared_memory_pow2_bytes: 64 * 1024,
        supports_f64: true,
        ..DeviceProfile::generic(Backend::Vulkan, GpuVendor::Nvidia)
    }
}

fn header<'a>(
    lines: &mut impl Iterator<Item = &'a str>,
    expected_case: &str,
    expected_kind: &str,
) -> usize {
    let case_line = format!("case {expected_case}");
    let kind_line = format!("kind {expected_kind}");
    assert_eq!(lines.next(), Some(case_line.as_str()));
    assert_eq!(lines.next(), Some(kind_line.as_str()));
    let length = lines.next().expect("missing golden length");
    length
        .strip_prefix("length ")
        .expect("malformed golden length")
        .parse()
        .expect("invalid golden length")
}

#[derive(Debug)]
struct GoldenVectors {
    c2c_input: Vec<Complex64>,
    c2c_expected: Vec<Complex64>,
    r2c_input: Vec<f64>,
    r2c_expected: Vec<Complex64>,
    dct_input: Vec<f64>,
    dct_expected: Vec<f64>,
}

fn parse_golden_vectors() -> GoldenVectors {
    let mut lines = FIXTURE.lines();
    assert_eq!(lines.next(), Some("schema 1"));
    assert_eq!(lines.next(), Some("source fftw-3.3.9"));
    assert_eq!(
        lines.next(),
        Some("upstream-samples VkFFT-1.3.4-sample11-sample15-sample16")
    );

    let n = header(&mut lines, "c2c-n47", "c2c");
    let mut c2c_input = Vec::with_capacity(n);
    let mut c2c_expected = Vec::with_capacity(n);
    for _ in 0..n {
        let values = lines
            .next()
            .expect("missing C2C golden row")
            .split_whitespace()
            .map(|value| value.parse::<f64>().expect("invalid C2C golden value"))
            .collect::<Vec<_>>();
        assert_eq!(values.len(), 4);
        c2c_input.push(Complex64::new(values[0], values[1]));
        c2c_expected.push(Complex64::new(values[2], values[3]));
    }

    let n = header(&mut lines, "r2c-n45", "r2c");
    let mut r2c_input = Vec::with_capacity(n);
    for _ in 0..n {
        r2c_input.push(
            lines
                .next()
                .expect("missing R2C input")
                .parse::<f64>()
                .expect("invalid R2C input"),
        );
    }
    let output_len: usize = lines
        .next()
        .expect("missing R2C output header")
        .strip_prefix("output ")
        .expect("malformed R2C output header")
        .parse()
        .expect("invalid R2C output length");
    assert_eq!(output_len, n / 2 + 1);
    let mut r2c_expected = Vec::with_capacity(output_len);
    for _ in 0..output_len {
        let mut values = lines
            .next()
            .expect("missing R2C output")
            .split_whitespace()
            .map(|value| value.parse::<f64>().expect("invalid R2C golden value"));
        let re = values.next().expect("missing R2C real value");
        let im = values.next().expect("missing R2C imaginary value");
        assert!(values.next().is_none());
        r2c_expected.push(Complex64::new(re, im));
    }

    let n = header(&mut lines, "dct2-n37", "dct2");
    let mut dct_input = Vec::with_capacity(n);
    let mut dct_expected = Vec::with_capacity(n);
    for _ in 0..n {
        let mut values = lines
            .next()
            .expect("missing DCT golden row")
            .split_whitespace()
            .map(|value| value.parse::<f64>().expect("invalid DCT golden value"));
        dct_input.push(values.next().expect("missing DCT input"));
        dct_expected.push(values.next().expect("missing DCT output"));
        assert!(values.next().is_none());
    }
    assert!(lines.next().is_none());

    GoldenVectors {
        c2c_input,
        c2c_expected,
        r2c_input,
        r2c_expected,
        dct_input,
        dct_expected,
    }
}

#[cfg(any(feature = "cuda-runtime", feature = "opencl-runtime"))]
fn run_native_fftw_f64<R: vkfft_rs::backend::NativeRuntime>(runtime: &R) {
    use vkfft_rs::{
        Precision,
        backend::{NativeTransformInput64, NativeTransformOutput64},
    };

    let golden = parse_golden_vectors();
    let profile = runtime.device_profile();
    assert!(profile.supports_f64);

    let c2c = TransformIr::build(
        FftConfig::new(vec![golden.c2c_input.len()]).with_precision(Precision::F64),
        Direction::Forward,
        profile,
    )
    .unwrap();
    let actual = match vkfft_rs::backend::NativeRuntime::execute_transform_f64(
        runtime,
        &c2c,
        NativeTransformInput64::Complex(&golden.c2c_input),
    )
    .unwrap()
    {
        NativeTransformOutput64::Complex(values) => values,
        NativeTransformOutput64::Real(_) => panic!("F64 C2C returned real output"),
    };
    let metrics = complex_precision_metrics(&actual, &golden.c2c_expected).unwrap();
    assert!(
        metrics.max_difference <= 1.0e-8 * golden.c2c_input.len() as f64,
        "native F64 C2C FFTW error on {}: {metrics:?}",
        runtime.device_name()
    );

    let r2c = TransformIr::build(
        FftConfig::new(vec![golden.r2c_input.len()])
            .with_precision(Precision::F64)
            .with_transform(TransformKind::RealToComplex),
        Direction::Forward,
        profile,
    )
    .unwrap();
    let actual = match vkfft_rs::backend::NativeRuntime::execute_transform_f64(
        runtime,
        &r2c,
        NativeTransformInput64::Real(&golden.r2c_input),
    )
    .unwrap()
    {
        NativeTransformOutput64::Complex(values) => values,
        NativeTransformOutput64::Real(_) => panic!("F64 R2C returned real output"),
    };
    let metrics = complex_precision_metrics(&actual, &golden.r2c_expected).unwrap();
    assert!(
        metrics.max_difference <= 1.0e-8 * golden.r2c_input.len() as f64,
        "native F64 R2C FFTW error on {}: {metrics:?}",
        runtime.device_name()
    );

    let dct = TransformIr::build(
        FftConfig::new(vec![golden.dct_input.len()])
            .with_precision(Precision::F64)
            .with_transform(TransformKind::Dct(DctType::II)),
        Direction::Forward,
        profile,
    )
    .unwrap();
    let actual = match vkfft_rs::backend::NativeRuntime::execute_transform_f64(
        runtime,
        &dct,
        NativeTransformInput64::Real(&golden.dct_input),
    )
    .unwrap()
    {
        NativeTransformOutput64::Real(values) => values,
        NativeTransformOutput64::Complex(_) => panic!("F64 DCT-II returned complex output"),
    };
    let metrics = real_precision_metrics(&actual, &golden.dct_expected).unwrap();
    assert!(
        metrics.max_difference <= 1.0e-8 * golden.dct_input.len() as f64,
        "native F64 DCT-II FFTW error on {}: {metrics:?}",
        runtime.device_name()
    );
}

#[cfg(any(
    feature = "cuda-runtime",
    feature = "opencl-runtime",
    feature = "vulkan-runtime"
))]
fn gpu_test_lock() -> &'static std::sync::Mutex<()> {
    static LOCK: std::sync::OnceLock<std::sync::Mutex<()>> = std::sync::OnceLock::new();
    LOCK.get_or_init(|| std::sync::Mutex::new(()))
}

#[test]
fn committed_fftw_precision_vectors_match_reference_transforms() {
    let golden = parse_golden_vectors();
    assert_eq!(golden.c2c_input.len(), 47);
    assert_eq!(golden.r2c_input.len(), 45);
    assert_eq!(golden.dct_input.len(), 37);

    let c2c = TransformIr::build(
        FftConfig::new(vec![golden.c2c_input.len()]).with_precision(Precision::F64),
        Direction::Forward,
        device(),
    )
    .unwrap();
    let actual = c2c.execute_complex_reference(&golden.c2c_input).unwrap();
    let metrics = complex_precision_metrics(&actual, &golden.c2c_expected).unwrap();
    assert!(
        metrics.max_difference <= 2.0e-11,
        "C2C FFTW max error: {metrics:?}"
    );

    let r2c = TransformIr::build(
        FftConfig::new(vec![golden.r2c_input.len()])
            .with_precision(Precision::F64)
            .with_transform(TransformKind::RealToComplex),
        Direction::Forward,
        device(),
    )
    .unwrap();
    let actual = r2c.execute_r2c_reference(&golden.r2c_input).unwrap();
    let metrics = complex_precision_metrics(&actual, &golden.r2c_expected).unwrap();
    assert!(
        metrics.max_difference <= 2.0e-11,
        "R2C FFTW max error: {metrics:?}"
    );

    let dct = TransformIr::build(
        FftConfig::new(vec![golden.dct_input.len()])
            .with_precision(Precision::F64)
            .with_transform(TransformKind::Dct(DctType::II)),
        Direction::Forward,
        device(),
    )
    .unwrap();
    let actual = dct.execute_r2r_reference(&golden.dct_input).unwrap();
    let metrics = real_precision_metrics(&actual, &golden.dct_expected).unwrap();
    assert!(
        metrics.max_difference <= 2.0e-11,
        "DCT-II FFTW max error: {metrics:?}"
    );
}

#[cfg(feature = "cuda-runtime")]
#[test]
fn cuda_f64_matches_committed_fftw_vectors_or_skips_without_device() {
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
    run_native_fftw_f64(&context);
}

#[cfg(feature = "opencl-runtime")]
#[test]
fn opencl_f64_matches_committed_fftw_vectors_or_skips_without_device() {
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
    run_native_fftw_f64(&context);
}

#[cfg(feature = "vulkan-runtime")]
#[test]
fn vulkan_f64_matches_committed_fftw_vectors_or_skips_without_device() {
    use vkfft_rs::{
        Precision, VkFftError,
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
    let golden = parse_golden_vectors();

    let c2c = TransformIr::build(
        FftConfig::new(vec![golden.c2c_input.len()]).with_precision(Precision::F64),
        Direction::Forward,
        profile,
    )
    .unwrap();
    let actual = match context
        .execute_transform_f64(&c2c, TransformInput64::Complex(&golden.c2c_input))
        .unwrap()
    {
        TransformOutput64::Complex(values) => values,
        TransformOutput64::Real(_) => panic!("Vulkan F64 C2C returned real output"),
    };
    let metrics = complex_precision_metrics(&actual, &golden.c2c_expected).unwrap();
    assert!(
        metrics.max_difference <= 1.0e-8 * golden.c2c_input.len() as f64,
        "Vulkan F64 C2C FFTW error: {metrics:?}"
    );

    let r2c = TransformIr::build(
        FftConfig::new(vec![golden.r2c_input.len()])
            .with_precision(Precision::F64)
            .with_transform(TransformKind::RealToComplex),
        Direction::Forward,
        profile,
    )
    .unwrap();
    let actual = match context
        .execute_transform_f64(&r2c, TransformInput64::Real(&golden.r2c_input))
        .unwrap()
    {
        TransformOutput64::Complex(values) => values,
        TransformOutput64::Real(_) => panic!("Vulkan F64 R2C returned real output"),
    };
    let metrics = complex_precision_metrics(&actual, &golden.r2c_expected).unwrap();
    assert!(
        metrics.max_difference <= 1.0e-8 * golden.r2c_input.len() as f64,
        "Vulkan F64 R2C FFTW error: {metrics:?}"
    );

    let dct = TransformIr::build(
        FftConfig::new(vec![golden.dct_input.len()])
            .with_precision(Precision::F64)
            .with_transform(TransformKind::Dct(DctType::II)),
        Direction::Forward,
        profile,
    )
    .unwrap();
    let actual = match context
        .execute_transform_f64(&dct, TransformInput64::Real(&golden.dct_input))
        .unwrap()
    {
        TransformOutput64::Real(values) => values,
        TransformOutput64::Complex(_) => panic!("Vulkan F64 DCT-II returned complex output"),
    };
    let metrics = real_precision_metrics(&actual, &golden.dct_expected).unwrap();
    assert!(
        metrics.max_difference <= 1.0e-8 * golden.dct_input.len() as f64,
        "Vulkan F64 DCT-II FFTW error: {metrics:?}"
    );
}
