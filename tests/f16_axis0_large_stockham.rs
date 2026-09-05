#![cfg(any(
    feature = "cuda-runtime",
    feature = "opencl-runtime",
    feature = "vulkan-runtime"
))]

use vkfft_rs::{Complex32, Direction, FftConfig, GpuVendor, OneDimFftIr, Precision, TransformIr};

const LENGTH: usize = 524_288;

fn config() -> FftConfig {
    FftConfig::new(vec![LENGTH]).with_precision(Precision::F16StorageF32Compute)
}

fn constrained_nvidia_profile(
    mut profile: vkfft_rs::DeviceProfile,
) -> Option<vkfft_rs::DeviceProfile> {
    if profile.vendor != GpuVendor::Nvidia
        || profile.shared_memory_bytes < 48 * 1024
        || profile.shared_memory_pow2_bytes < 32 * 1024
        || profile.max_threads_per_block < 1024
        || profile.max_workgroup_size[0] < 1024
        || profile.max_workgroup_size[1] < 1024
    {
        return None;
    }
    profile.shared_memory_bytes = 48 * 1024;
    profile.shared_memory_pow2_bytes = 32 * 1024;
    profile.max_threads_per_block = 1024;
    profile.max_workgroup_size[0] = 1024;
    profile.max_workgroup_size[1] = 1024;
    profile.coalesced_memory_bytes = 64;
    Some(profile)
}

fn assert_three_upload_schedule(ir: &TransformIr, label: &str) {
    let TransformIr::Complex1d(OneDimFftIr::Recursive(recursive)) = ir else {
        panic!("{label} F16 N524288 must materialize recursive Four-step IR");
    };
    let four_step = recursive
        .four_step_plan
        .as_ref()
        .expect("F16 N524288 must retain three-upload Four-step metadata");
    assert_eq!(four_step.uploads.len(), 3, "{label} upload count changed");

    let mut by_id = four_step.uploads.iter().collect::<Vec<_>>();
    by_id.sort_by_key(|upload| upload.axis_upload_id);
    assert_eq!(
        by_id
            .iter()
            .map(|upload| upload.fft_len)
            .collect::<Vec<_>>(),
        vec![128, 64, 64],
        "{label} F16 N524288 logical split changed"
    );
    for (upload, expected_xy) in by_id.iter().zip([[16, 16], [16, 8], [16, 8]]) {
        let block = upload
            .axis_block
            .as_ref()
            .expect("F16 N524288 upload must retain a physical AxisBlock");
        assert_eq!(
            [block.local_size_x, block.local_size_y],
            expected_xy,
            "{label} upload {} AxisBlock changed",
            upload.axis_upload_id
        );
        assert_eq!(block.grouped_batch, 16);
        assert!(block.transforms_on_x);
        assert_eq!(block.axis_swapped, upload.axis_upload_id == 0);
    }
}

fn impulse_input() -> Vec<Complex32> {
    let mut input = vec![Complex32::new(0.0, 0.0); LENGTH];
    input[0] = Complex32::new(1.0, 0.0);
    input
}

fn assert_impulse_spectrum(values: &[Complex32], label: &str) {
    let max_error = values
        .iter()
        .map(|value| ((value.re - 1.0).powi(2) + value.im.powi(2)).sqrt())
        .fold(0.0f32, f32::max);
    assert!(
        max_error <= 2.0e-2,
        "{label} F16 N524288 impulse spectrum error {max_error:e}"
    );
}

fn assert_round_trip(actual: &[Complex32], label: &str) {
    assert_eq!(actual.len(), LENGTH);
    let impulse_error = ((actual[0].re - 1.0).powi(2) + actual[0].im.powi(2)).sqrt();
    let tail_error = actual[1..]
        .iter()
        .map(|value| (value.re.powi(2) + value.im.powi(2)).sqrt())
        .fold(0.0f32, f32::max);
    assert!(
        impulse_error <= 2.0e-2 && tail_error <= 2.0e-2,
        "{label} F16 N524288 round-trip errors impulse={impulse_error:e} tail={tail_error:e}"
    );
}

#[cfg(any(feature = "cuda-runtime", feature = "opencl-runtime"))]
fn run_native<R: vkfft_rs::backend::NativeRuntime>(runtime: &R) {
    use vkfft_rs::backend::{NativeTransformInput32, NativeTransformOutput32};

    let Some(profile) = constrained_nvidia_profile(runtime.device_profile()) else {
        return;
    };
    let forward = TransformIr::build(config(), Direction::Forward, profile).unwrap();
    assert_three_upload_schedule(&forward, runtime.device_name());
    let input = impulse_input();
    let spectrum = match runtime
        .execute_transform_f32(&forward, NativeTransformInput32::Complex(&input))
        .unwrap()
    {
        NativeTransformOutput32::Complex(values) => values,
        NativeTransformOutput32::Real(_) => panic!("F16 N524288 C2C returned real output"),
    };
    assert_impulse_spectrum(&spectrum, runtime.device_name());

    let inverse = TransformIr::build(
        config().with_inverse_normalization(true),
        Direction::Inverse,
        profile,
    )
    .unwrap();
    assert_three_upload_schedule(&inverse, runtime.device_name());
    let restored = match runtime
        .execute_transform_f32(&inverse, NativeTransformInput32::Complex(&spectrum))
        .unwrap()
    {
        NativeTransformOutput32::Complex(values) => values,
        NativeTransformOutput32::Real(_) => panic!("F16 N524288 inverse returned real output"),
    };
    assert_round_trip(&restored, runtime.device_name());
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
fn cuda_f16_axis0_n524288_three_upload_matches_impulse_and_round_trip_or_skips() {
    use vkfft_rs::backend::cuda::runtime::CudaExecutionContext;

    let _guard = gpu_test_lock().lock().unwrap();
    let availability = CudaExecutionContext::probe();
    if !availability.compiler_available || availability.device_count == 0 {
        return;
    }
    let runtime = CudaExecutionContext::new(0).expect("CUDA context failed after successful probe");
    run_native(&runtime);
}

#[cfg(feature = "opencl-runtime")]
#[test]
fn opencl_f16_axis0_n524288_three_upload_matches_impulse_and_round_trip_or_skips() {
    use vkfft_rs::backend::opencl::runtime::OpenClExecutionContext;

    let _guard = gpu_test_lock().lock().unwrap();
    let availability = OpenClExecutionContext::probe();
    if !availability.compiler_available || availability.device_count == 0 {
        return;
    }
    let runtime =
        OpenClExecutionContext::new(0).expect("OpenCL context failed after successful probe");
    run_native(&runtime);
}

#[cfg(feature = "vulkan-runtime")]
#[test]
fn vulkan_f16_axis0_n524288_three_upload_matches_impulse_and_round_trip_or_skips() {
    use vkfft_rs::{
        VkFftError,
        backend::vulkan::runtime::{TransformInput32, TransformOutput32, VulkanExecutionContext},
    };

    let _guard = gpu_test_lock().lock().unwrap();
    let runtime = match VulkanExecutionContext::new() {
        Ok(runtime) => runtime,
        Err(VkFftError::VulkanUnavailable(_)) => return,
        Err(error) => panic!("Vulkan context failed after loader/device discovery: {error}"),
    };
    let Some(profile) = constrained_nvidia_profile(runtime.device_profile()) else {
        return;
    };
    let forward = TransformIr::build(config(), Direction::Forward, profile).unwrap();
    assert_three_upload_schedule(&forward, "Vulkan");
    let input = impulse_input();
    let spectrum = match runtime
        .execute_transform_f32(&forward, TransformInput32::Complex(&input))
        .unwrap()
    {
        TransformOutput32::Complex(values) => values,
        TransformOutput32::Real(_) => panic!("Vulkan F16 N524288 C2C returned real output"),
    };
    assert_impulse_spectrum(&spectrum, "Vulkan");

    let inverse = TransformIr::build(
        config().with_inverse_normalization(true),
        Direction::Inverse,
        profile,
    )
    .unwrap();
    assert_three_upload_schedule(&inverse, "Vulkan");
    let restored = match runtime
        .execute_transform_f32(&inverse, TransformInput32::Complex(&spectrum))
        .unwrap()
    {
        TransformOutput32::Complex(values) => values,
        TransformOutput32::Real(_) => panic!("Vulkan F16 N524288 inverse returned real output"),
    };
    assert_round_trip(&restored, "Vulkan");
}
