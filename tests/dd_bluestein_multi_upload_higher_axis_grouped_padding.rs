#![cfg(any(
    feature = "cuda-runtime",
    feature = "opencl-runtime",
    feature = "vulkan-runtime"
))]

use vkfft_rs::{
    ComplexDoubleDouble, DeviceProfile, Direction, DoubleDouble, FftConfig, Precision, TransformIr,
};

const DIMENSIONS: [usize; 2] = [2053, 8];
const BATCH_COUNT: usize = 5;
const GROUPED_BATCH: usize = 3;
const SHARED_BYTES: usize = 32 * 1024;

fn config(bandwidth_boost: usize) -> FftConfig {
    FftConfig::new(DIMENSIONS.to_vec())
        .with_batch_count(BATCH_COUNT)
        .with_grouped_batch(0, GROUPED_BATCH)
        .unwrap()
        .with_precision(Precision::DoubleDouble)
        .with_bandwidth_boost(bandwidth_boost)
        .with_zero_padding(0, 1, 2)
        .unwrap()
}

fn planning_profile(mut profile: DeviceProfile) -> Option<DeviceProfile> {
    if !profile.supports_f64
        || profile.shared_memory_bytes < SHARED_BYTES
        || profile.shared_memory_pow2_bytes < SHARED_BYTES
        || profile.max_threads_per_block < 1024
        || profile.max_workgroup_size[0] < 1024
        || profile.max_workgroup_size[1] < 1024
    {
        return None;
    }
    profile.shared_memory_bytes = SHARED_BYTES;
    profile.shared_memory_pow2_bytes = SHARED_BYTES;
    profile.max_threads_per_block = 1024;
    profile.max_workgroup_size[0] = 1024;
    profile.max_workgroup_size[1] = 1024;
    Some(profile)
}

fn dd_error(actual: DoubleDouble, expected: DoubleDouble) -> f64 {
    let delta = (actual - expected).abs();
    delta.hi.abs() + delta.lo.abs()
}

fn complex_dd_error(actual: ComplexDoubleDouble, expected: ComplexDoubleDouble) -> f64 {
    dd_error(actual.re, expected.re) + dd_error(actual.im, expected.im)
}

fn input_and_masked_expected() -> (
    Vec<ComplexDoubleDouble>,
    Vec<ComplexDoubleDouble>,
    ComplexDoubleDouble,
) {
    let tensor_len = DIMENSIONS.iter().product::<usize>();
    let zero = ComplexDoubleDouble::new(DoubleDouble::ZERO, DoubleDouble::ZERO);
    let impulse = ComplexDoubleDouble::new(
        DoubleDouble::from_parts(1.0, 1.0e-31),
        DoubleDouble::from_parts(0.0, 0.0),
    );
    let sentinel = ComplexDoubleDouble::new(
        DoubleDouble::from_parts(0.75, 2.0e-31),
        DoubleDouble::from_parts(-0.25, -1.0e-31),
    );
    let mut input = vec![zero; tensor_len * BATCH_COUNT];
    let mut expected = input.clone();
    for batch in 0..BATCH_COUNT {
        let base = batch * tensor_len;
        input[base] = impulse;
        expected[base] = impulse;
        // Outer axis index 1 belongs to zero-padding [1,2). This must be removed
        // before the Bluestein wrapper packs its higher-axis transforms.
        input[base + 8 + 3] = sentinel;
    }
    (input, expected, impulse)
}

fn assert_impulse_spectrum(
    values: &[ComplexDoubleDouble],
    expected: ComplexDoubleDouble,
    label: &str,
) {
    let max_error = values
        .iter()
        .copied()
        .map(|value| complex_dd_error(value, expected))
        .fold(0.0f64, f64::max);
    assert!(
        max_error <= 2.0e-14,
        "{label} grouped padded DD p2053 Bluestein impulse spectrum error {max_error:e}"
    );
}

fn assert_round_trip(
    actual: &[ComplexDoubleDouble],
    expected: &[ComplexDoubleDouble],
    label: &str,
) {
    assert_eq!(actual.len(), expected.len());
    let max_error = actual
        .iter()
        .copied()
        .zip(expected.iter().copied())
        .map(|(actual, expected)| complex_dd_error(actual, expected))
        .fold(0.0f64, f64::max);
    assert!(
        max_error <= 2.0e-12,
        "{label} grouped padded DD p2053 Bluestein round-trip error {max_error:e}"
    );
}

#[cfg(any(
    feature = "cuda-runtime",
    feature = "opencl-runtime",
    feature = "vulkan-runtime"
))]
fn run_with<Execute>(profile: DeviceProfile, device_name: &str, mut execute: Execute)
where
    Execute: FnMut(
        &TransformIr,
        &[ComplexDoubleDouble],
    ) -> Result<Vec<ComplexDoubleDouble>, vkfft_rs::VkFftError>,
{
    let Some(profile) = planning_profile(profile) else {
        return;
    };
    let (input, expected, impulse) = input_and_masked_expected();
    for bandwidth_boost in [0usize, 2] {
        let label = format!("{device_name} DD p2053x8 b{bandwidth_boost}");
        let forward =
            TransformIr::build(config(bandwidth_boost), Direction::Forward, profile).unwrap();
        assert!(matches!(forward, TransformIr::ComplexNdDoubleDouble(_)));
        let spectrum = execute(&forward, &input).unwrap();
        assert_impulse_spectrum(&spectrum, impulse, &label);

        let inverse = TransformIr::build(
            config(bandwidth_boost).with_inverse_normalization(true),
            Direction::Inverse,
            profile,
        )
        .unwrap();
        assert!(matches!(inverse, TransformIr::ComplexNdDoubleDouble(_)));
        let restored = execute(&inverse, &spectrum).unwrap();
        assert_round_trip(&restored, &expected, &label);
    }
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

#[cfg(feature = "cuda-runtime")]
#[test]
fn cuda_dd_bluestein_p2053_multi_upload_padded_grouped_higher_axis_or_skips() {
    use vkfft_rs::backend::cuda::runtime::CudaExecutionContext;

    let _guard = gpu_test_lock().lock().unwrap();
    let availability = CudaExecutionContext::probe();
    if !availability.compiler_available || availability.device_count == 0 {
        return;
    }
    let context = CudaExecutionContext::new(0).expect("CUDA context failed after successful probe");
    let profile = context.device_profile();
    let name = format!("CUDA {}", context.device_name());
    run_with(profile, &name, |ir, input| {
        context.execute_transform_double_double(ir, input)
    });
}

#[cfg(feature = "opencl-runtime")]
#[test]
fn opencl_dd_bluestein_p2053_multi_upload_padded_grouped_higher_axis_or_skips() {
    use vkfft_rs::backend::opencl::runtime::OpenClExecutionContext;

    let _guard = gpu_test_lock().lock().unwrap();
    let availability = OpenClExecutionContext::probe();
    if !availability.compiler_available || availability.device_count == 0 {
        return;
    }
    let context =
        OpenClExecutionContext::new(0).expect("OpenCL context failed after successful probe");
    let profile = context.device_profile();
    let name = format!("OpenCL {}", context.device_name());
    run_with(profile, &name, |ir, input| {
        context.execute_transform_double_double(ir, input)
    });
}

#[cfg(feature = "vulkan-runtime")]
#[test]
fn vulkan_dd_bluestein_p2053_multi_upload_padded_grouped_higher_axis_or_skips() {
    use vkfft_rs::{VkFftError, backend::vulkan::runtime::VulkanExecutionContext};

    let _guard = gpu_test_lock().lock().unwrap();
    let context = match VulkanExecutionContext::new() {
        Ok(context) => context,
        Err(VkFftError::VulkanUnavailable(_)) => return,
        Err(error) => panic!("Vulkan context failed after loader/device discovery: {error}"),
    };
    let profile = context.device_profile();
    run_with(profile, "Vulkan", |ir, input| {
        context.execute_transform_double_double(ir, input)
    });
}
