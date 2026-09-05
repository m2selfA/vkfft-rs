#![cfg(any(
    feature = "cuda-runtime",
    feature = "opencl-runtime",
    feature = "vulkan-runtime",
    feature = "level-zero-runtime",
    feature = "metal-runtime"
))]

use vkfft_rs::{
    Complex32, DeviceProfile, Direction, FftConfig, OneDimFftIr, Precision, RealFftAlgorithm,
    ScalarType, TransformIr, TransformKind, VkFftError,
};
#[cfg(any(
    feature = "cuda-runtime",
    feature = "opencl-runtime",
    feature = "vulkan-runtime",
    feature = "level-zero-runtime"
))]
use vkfft_rs::{Complex64, ComplexDoubleDouble, DoubleDouble};

#[cfg(any(
    feature = "cuda-runtime",
    feature = "opencl-runtime",
    feature = "vulkan-runtime",
    feature = "level-zero-runtime"
))]
const SMALL_LENGTH: usize = 64;
#[cfg(any(
    feature = "cuda-runtime",
    feature = "opencl-runtime",
    feature = "vulkan-runtime",
    feature = "level-zero-runtime"
))]
const LARGE_LENGTH: usize = 4096;
#[cfg(any(
    feature = "cuda-runtime",
    feature = "opencl-runtime",
    feature = "vulkan-runtime",
    feature = "level-zero-runtime"
))]
const SHARED_BYTES: usize = 32 * 1024;
const F16_LENGTH: usize = 94;
const F16_SHARED_BYTES: usize = 4 * 1024;

#[cfg(any(
    feature = "cuda-runtime",
    feature = "opencl-runtime",
    feature = "vulkan-runtime",
    feature = "level-zero-runtime"
))]
fn constrained_profile(mut profile: DeviceProfile) -> Option<DeviceProfile> {
    if !profile.supports_f64
        || profile.shared_memory_bytes < SHARED_BYTES
        || profile.shared_memory_pow2_bytes < SHARED_BYTES
    {
        return None;
    }
    profile.shared_memory_bytes = SHARED_BYTES;
    profile.shared_memory_pow2_bytes = SHARED_BYTES;
    Some(profile)
}

fn constrained_f16_profile(mut profile: DeviceProfile) -> Option<DeviceProfile> {
    if profile.shared_memory_bytes < F16_SHARED_BYTES
        || profile.shared_memory_pow2_bytes < F16_SHARED_BYTES
        || profile.max_threads_per_block < 128
        || profile.max_workgroup_size[0] < 128
        || profile.max_workgroup_size[1] < 128
    {
        return None;
    }
    profile.shared_memory_bytes = F16_SHARED_BYTES;
    profile.shared_memory_pow2_bytes = F16_SHARED_BYTES;
    profile.max_threads_per_block = 128;
    profile.max_workgroup_size[0] = 128;
    profile.max_workgroup_size[1] = 128;
    Some(profile)
}

#[cfg(any(
    feature = "cuda-runtime",
    feature = "opencl-runtime",
    feature = "vulkan-runtime",
    feature = "level-zero-runtime"
))]
fn dd_error(actual: DoubleDouble, expected: DoubleDouble) -> f64 {
    let difference = (actual - expected).abs();
    difference.hi.abs() + difference.lo.abs()
}

#[cfg(any(
    feature = "cuda-runtime",
    feature = "opencl-runtime",
    feature = "vulkan-runtime",
    feature = "level-zero-runtime"
))]
fn complex_dd_error(actual: ComplexDoubleDouble, expected: ComplexDoubleDouble) -> f64 {
    dd_error(actual.re, expected.re) + dd_error(actual.im, expected.im)
}

fn build_real_ir(
    length: usize,
    precision: Precision,
    transform: TransformKind,
    direction: Direction,
    normalize_inverse: bool,
    profile: DeviceProfile,
) -> TransformIr {
    TransformIr::build(
        FftConfig::new(vec![length])
            .with_precision(precision)
            .with_transform(transform)
            .with_inverse_normalization(normalize_inverse),
        direction,
        profile,
    )
    .unwrap()
}

#[cfg(any(
    feature = "cuda-runtime",
    feature = "opencl-runtime",
    feature = "vulkan-runtime",
    feature = "level-zero-runtime"
))]
fn run_dd_real_shapes<R2cDd, C2rDd, R2cF64, C2rF64>(
    profile: DeviceProfile,
    label: &str,
    mut execute_r2c_dd: R2cDd,
    mut execute_c2r_dd: C2rDd,
    mut execute_r2c_f64: R2cF64,
    mut execute_c2r_f64: C2rF64,
) where
    R2cDd: FnMut(
        &vkfft_rs::DoubleDoubleRealFftIr,
        &[DoubleDouble],
    ) -> Result<Vec<ComplexDoubleDouble>, VkFftError>,
    C2rDd: FnMut(
        &vkfft_rs::DoubleDoubleRealFftIr,
        &[ComplexDoubleDouble],
    ) -> Result<Vec<DoubleDouble>, VkFftError>,
    R2cF64: FnMut(&vkfft_rs::DoubleDoubleRealFftIr, &[f64]) -> Result<Vec<Complex64>, VkFftError>,
    C2rF64: FnMut(&vkfft_rs::DoubleDoubleRealFftIr, &[Complex64]) -> Result<Vec<f64>, VkFftError>,
{
    let Some(profile) = constrained_profile(profile) else {
        return;
    };

    for (length, expected_even_half_size) in [(SMALL_LENGTH, false), (LARGE_LENGTH, true)] {
        let child_len = if expected_even_half_size {
            length / 2
        } else {
            length
        };

        let dd_forward = build_real_ir(
            length,
            Precision::DoubleDouble,
            TransformKind::RealToComplex,
            Direction::Forward,
            false,
            profile,
        );
        let dd_inverse = build_real_ir(
            length,
            Precision::DoubleDouble,
            TransformKind::ComplexToReal,
            Direction::Inverse,
            true,
            profile,
        );
        let TransformIr::RealDoubleDouble(dd_forward_ir) = &dd_forward else {
            panic!("DD R2C must build one-dimensional real DD IR")
        };
        let TransformIr::RealDoubleDouble(dd_inverse_ir) = &dd_inverse else {
            panic!("DD C2R must build one-dimensional real DD IR")
        };
        assert_eq!(dd_forward_ir.even_half_size, expected_even_half_size);
        assert_eq!(dd_inverse_ir.even_half_size, expected_even_half_size);
        assert_eq!(dd_forward_ir.transform.sequence_len(), child_len);
        assert_eq!(dd_inverse_ir.transform.sequence_len(), child_len);

        let impulse = DoubleDouble::from_parts(1.0, 1.0e-31);
        let mut dd_input = vec![DoubleDouble::ZERO; length];
        dd_input[0] = impulse;
        let dd_spectrum = execute_r2c_dd(dd_forward_ir, &dd_input).unwrap();
        let expected_bin = ComplexDoubleDouble::new(impulse, DoubleDouble::ZERO);
        let max_forward_error = dd_spectrum
            .iter()
            .copied()
            .map(|value| complex_dd_error(value, expected_bin))
            .fold(0.0f64, f64::max);
        assert!(
            max_forward_error <= 2.0e-22,
            "{label} full-DD N{length} R2C mismatch: {max_forward_error:e}"
        );
        let dd_restored = execute_c2r_dd(dd_inverse_ir, &dd_spectrum).unwrap();
        let max_round_trip_error = dd_restored
            .iter()
            .copied()
            .zip(dd_input.iter().copied())
            .map(|(actual, expected)| dd_error(actual, expected))
            .fold(0.0f64, f64::max);
        assert!(
            max_round_trip_error <= 2.0e-20,
            "{label} full-DD N{length} round trip mismatch: {max_round_trip_error:e}"
        );

        let f64_forward = build_real_ir(
            length,
            Precision::DoubleDoubleF64Storage,
            TransformKind::RealToComplex,
            Direction::Forward,
            false,
            profile,
        );
        let f64_inverse = build_real_ir(
            length,
            Precision::DoubleDoubleF64Storage,
            TransformKind::ComplexToReal,
            Direction::Inverse,
            true,
            profile,
        );
        let TransformIr::RealDoubleDouble(f64_forward_ir) = &f64_forward else {
            panic!("DD/F64 R2C must build one-dimensional real DD IR")
        };
        let TransformIr::RealDoubleDouble(f64_inverse_ir) = &f64_inverse else {
            panic!("DD/F64 C2R must build one-dimensional real DD IR")
        };
        assert_eq!(f64_forward_ir.even_half_size, expected_even_half_size);
        assert_eq!(f64_inverse_ir.even_half_size, expected_even_half_size);
        assert_eq!(f64_forward_ir.transform.sequence_len(), child_len);
        assert_eq!(f64_inverse_ir.transform.sequence_len(), child_len);

        let mut f64_input = vec![0.0f64; length];
        f64_input[0] = 1.0;
        let f64_spectrum = execute_r2c_f64(f64_forward_ir, &f64_input).unwrap();
        let max_f64_forward_error = f64_spectrum
            .iter()
            .map(|value| (value.re - 1.0).abs().max(value.im.abs()))
            .fold(0.0f64, f64::max);
        assert!(
            max_f64_forward_error <= 2.0e-12,
            "{label} DD/F64 N{length} R2C mismatch: {max_f64_forward_error:e}"
        );
        let f64_restored = execute_c2r_f64(f64_inverse_ir, &f64_spectrum).unwrap();
        let max_f64_round_trip_error = f64_restored
            .iter()
            .zip(&f64_input)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0f64, f64::max);
        assert!(
            max_f64_round_trip_error <= 2.0e-12,
            "{label} DD/F64 N{length} round trip mismatch: {max_f64_round_trip_error:e}"
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
fn assert_f16_real_shape(ir: &TransformIr, label: &str) {
    let TransformIr::Real(real) = ir else {
        panic!("{label} F16 real shape must build ordinary real IR")
    };
    assert_eq!(real.algorithm, RealFftAlgorithm::FullComplex);
    assert_eq!(real.scalar, ScalarType::F32);
    assert_eq!(real.external_scalar, ScalarType::F16);
    let OneDimFftIr::Bluestein(bluestein) = &real.transform else {
        panic!("{label} F16 N94 child must use Bluestein")
    };
    assert_eq!(bluestein.logical_len, F16_LENGTH);
    assert_eq!(bluestein.convolution_len, 256);
    assert_eq!(bluestein.scalar, ScalarType::F32);
    assert_eq!(bluestein.external_scalar, ScalarType::F32);
    assert_eq!(bluestein.forward_fft.scalar, ScalarType::F32);
    assert_eq!(bluestein.forward_fft.external_scalar, ScalarType::F32);
    assert_eq!(bluestein.inverse_fft.scalar, ScalarType::F32);
    assert_eq!(bluestein.inverse_fft.external_scalar, ScalarType::F32);
}

#[cfg(any(
    feature = "cuda-runtime",
    feature = "opencl-runtime",
    feature = "vulkan-runtime",
    feature = "level-zero-runtime",
    feature = "metal-runtime"
))]
fn run_f16_real_shape<R2c, C2r>(
    profile: DeviceProfile,
    label: &str,
    mut execute_r2c: R2c,
    mut execute_c2r: C2r,
) where
    R2c: FnMut(&TransformIr, &[f32]) -> Result<Vec<Complex32>, VkFftError>,
    C2r: FnMut(&TransformIr, &[Complex32]) -> Result<Vec<f32>, VkFftError>,
{
    let Some(profile) = constrained_f16_profile(profile) else {
        return;
    };
    let forward = build_real_ir(
        F16_LENGTH,
        Precision::F16StorageF32Compute,
        TransformKind::RealToComplex,
        Direction::Forward,
        false,
        profile,
    );
    let inverse = build_real_ir(
        F16_LENGTH,
        Precision::F16StorageF32Compute,
        TransformKind::ComplexToReal,
        Direction::Inverse,
        true,
        profile,
    );
    assert_f16_real_shape(&forward, label);
    assert_f16_real_shape(&inverse, label);

    let mut input = vec![0.0f32; F16_LENGTH];
    input[0] = 1.0;
    let spectrum = execute_r2c(&forward, &input).unwrap();
    assert_eq!(spectrum.len(), F16_LENGTH / 2 + 1);
    let max_forward_error = spectrum
        .iter()
        .map(|value| (value.re - 1.0).abs().max(value.im.abs()))
        .fold(0.0f32, f32::max);
    assert!(
        max_forward_error <= 2.0e-3,
        "{label} F16 N94 R2C mismatch: {max_forward_error:e}"
    );

    let restored = execute_c2r(&inverse, &spectrum).unwrap();
    assert_eq!(restored.len(), F16_LENGTH);
    let max_round_trip_error = restored
        .iter()
        .zip(&input)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_round_trip_error <= 2.0e-3,
        "{label} F16 N94 round trip mismatch: {max_round_trip_error:e}"
    );
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
fn cuda_real_shape_dd_and_f16_execute_or_skip_without_device() {
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
    run_dd_real_shapes(
        profile,
        &label,
        |ir, input| context.execute_double_double_r2c(ir, input),
        |ir, input| context.execute_double_double_c2r(ir, input),
        |ir, input| context.execute_double_double_r2c_f64_storage(ir, input),
        |ir, input| context.execute_double_double_c2r_f64_storage(ir, input),
    );
    use vkfft_rs::backend::{NativeRuntime, NativeTransformInput32, NativeTransformOutput32};
    run_f16_real_shape(
        profile,
        &label,
        |ir, input| match NativeRuntime::execute_transform_f32(
            &context,
            ir,
            NativeTransformInput32::Real(input),
        )? {
            NativeTransformOutput32::Complex(values) => Ok(values),
            NativeTransformOutput32::Real(_) => panic!("CUDA F16 R2C returned real output"),
        },
        |ir, input| match NativeRuntime::execute_transform_f32(
            &context,
            ir,
            NativeTransformInput32::Complex(input),
        )? {
            NativeTransformOutput32::Real(values) => Ok(values),
            NativeTransformOutput32::Complex(_) => panic!("CUDA F16 C2R returned complex output"),
        },
    );
}

#[cfg(feature = "opencl-runtime")]
#[test]
fn opencl_real_shape_dd_and_f16_execute_or_skip_without_device() {
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
    run_dd_real_shapes(
        profile,
        &label,
        |ir, input| context.execute_double_double_r2c(ir, input),
        |ir, input| context.execute_double_double_c2r(ir, input),
        |ir, input| context.execute_double_double_r2c_f64_storage(ir, input),
        |ir, input| context.execute_double_double_c2r_f64_storage(ir, input),
    );
    use vkfft_rs::backend::{NativeRuntime, NativeTransformInput32, NativeTransformOutput32};
    run_f16_real_shape(
        profile,
        &label,
        |ir, input| match NativeRuntime::execute_transform_f32(
            &context,
            ir,
            NativeTransformInput32::Real(input),
        )? {
            NativeTransformOutput32::Complex(values) => Ok(values),
            NativeTransformOutput32::Real(_) => panic!("OpenCL F16 R2C returned real output"),
        },
        |ir, input| match NativeRuntime::execute_transform_f32(
            &context,
            ir,
            NativeTransformInput32::Complex(input),
        )? {
            NativeTransformOutput32::Real(values) => Ok(values),
            NativeTransformOutput32::Complex(_) => panic!("OpenCL F16 C2R returned complex output"),
        },
    );
}

#[cfg(feature = "level-zero-runtime")]
#[test]
fn level_zero_f16_real_shape_executes_or_skips_without_device() {
    use vkfft_rs::backend::level_zero::runtime::LevelZeroExecutionContext;
    use vkfft_rs::backend::{NativeRuntime, NativeTransformInput32, NativeTransformOutput32};

    let _guard = gpu_test_lock()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let require = std::env::var_os("VKFFT_REQUIRE_LEVEL_ZERO_RUNTIME").is_some();
    let availability = LevelZeroExecutionContext::probe();
    if !availability.available() {
        assert!(
            !require,
            "strict Level Zero F16 real-shape gate is unavailable: {}",
            availability.detail
        );
        return;
    }
    let context = LevelZeroExecutionContext::new(0)
        .expect("Level Zero context failed after successful probe");
    let profile = context.device_profile();
    let label = format!("Level Zero {}", context.device_name());
    run_f16_real_shape(
        profile,
        &label,
        |ir, input| match NativeRuntime::execute_transform_f32(
            &context,
            ir,
            NativeTransformInput32::Real(input),
        )? {
            NativeTransformOutput32::Complex(values) => Ok(values),
            NativeTransformOutput32::Real(_) => {
                panic!("Level Zero F16 R2C returned real output")
            }
        },
        |ir, input| match NativeRuntime::execute_transform_f32(
            &context,
            ir,
            NativeTransformInput32::Complex(input),
        )? {
            NativeTransformOutput32::Real(values) => Ok(values),
            NativeTransformOutput32::Complex(_) => {
                panic!("Level Zero F16 C2R returned complex output")
            }
        },
    );
}

#[cfg(feature = "level-zero-runtime")]
#[test]
fn level_zero_dd_real_shape_executes_or_skips_without_f64_device() {
    use vkfft_rs::backend::NativeRuntime;
    use vkfft_rs::backend::level_zero::runtime::LevelZeroExecutionContext;

    let _guard = gpu_test_lock()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let require = std::env::var_os("VKFFT_REQUIRE_LEVEL_ZERO_F64_RUNTIME").is_some();
    let availability = LevelZeroExecutionContext::probe();
    if !availability.available() {
        assert!(
            !require,
            "strict Level Zero DD real-shape gate is unavailable: {}",
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
            "strict Level Zero DD real-shape gate found no FP64 support"
        );
        return;
    }
    let label = format!("Level Zero {}", context.device_name());
    run_dd_real_shapes(
        profile,
        &label,
        |ir, input| context.execute_double_double_r2c(ir, input),
        |ir, input| context.execute_double_double_c2r(ir, input),
        |ir, input| context.execute_double_double_r2c_f64_storage(ir, input),
        |ir, input| context.execute_double_double_c2r_f64_storage(ir, input),
    );
}

#[cfg(feature = "metal-runtime")]
#[test]
fn metal_f16_real_shape_executes_round_trip_or_skips_without_device() {
    use vkfft_rs::backend::metal::runtime::MetalExecutionContext;
    use vkfft_rs::backend::{NativeRuntime, NativeTransformInput32, NativeTransformOutput32};

    let _guard = gpu_test_lock()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let require = std::env::var_os("VKFFT_REQUIRE_METAL_RUNTIME").is_some();
    let availability = MetalExecutionContext::probe();
    if !availability.available() {
        assert!(
            !require,
            "strict Metal F16 real round-trip gate is unavailable: {}",
            availability.detail
        );
        return;
    }
    let context =
        MetalExecutionContext::new(0).expect("Metal context failed after successful probe");
    let profile = context.device_profile();
    let label = format!("Metal {}", context.device_name());
    run_f16_real_shape(
        profile,
        &label,
        |ir, input| match NativeRuntime::execute_transform_f32(
            &context,
            ir,
            NativeTransformInput32::Real(input),
        )? {
            NativeTransformOutput32::Complex(values) => Ok(values),
            NativeTransformOutput32::Real(_) => panic!("Metal F16 R2C returned real output"),
        },
        |ir, input| match NativeRuntime::execute_transform_f32(
            &context,
            ir,
            NativeTransformInput32::Complex(input),
        )? {
            NativeTransformOutput32::Real(values) => Ok(values),
            NativeTransformOutput32::Complex(_) => panic!("Metal F16 C2R returned complex output"),
        },
    );
}

#[cfg(feature = "vulkan-runtime")]
#[test]
fn vulkan_real_shape_dd_and_f16_execute_or_skip_without_device() {
    use vkfft_rs::backend::vulkan::runtime::VulkanExecutionContext;

    let _guard = gpu_test_lock()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let context = match VulkanExecutionContext::new() {
        Ok(context) => context,
        Err(VkFftError::VulkanUnavailable(_)) => return,
        Err(error) => panic!("Vulkan context failed after loader/device discovery: {error}"),
    };
    let profile = context.device_profile();
    run_dd_real_shapes(
        profile,
        "Vulkan",
        |ir, input| context.execute_double_double_r2c(ir, input),
        |ir, input| context.execute_double_double_c2r(ir, input),
        |ir, input| context.execute_double_double_r2c_f64_storage(ir, input),
        |ir, input| context.execute_double_double_c2r_f64_storage(ir, input),
    );
    use vkfft_rs::backend::vulkan::runtime::{TransformInput32, TransformOutput32};
    run_f16_real_shape(
        profile,
        "Vulkan",
        |ir, input| match context.execute_transform_f32(ir, TransformInput32::Real(input))? {
            TransformOutput32::Complex(values) => Ok(values),
            TransformOutput32::Real(_) => panic!("Vulkan F16 R2C returned real output"),
        },
        |ir, input| match context.execute_transform_f32(ir, TransformInput32::Complex(input))? {
            TransformOutput32::Real(values) => Ok(values),
            TransformOutput32::Complex(_) => panic!("Vulkan F16 C2R returned complex output"),
        },
    );
}
