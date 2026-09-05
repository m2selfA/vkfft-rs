#![cfg(any(
    feature = "cuda-runtime",
    feature = "opencl-runtime",
    feature = "vulkan-runtime"
))]

use vkfft_rs::{Complex32, Direction, FftConfig, GpuVendor, TransformIr};

const DIMENSIONS: [usize; 2] = [263, 8];
const BATCH_COUNT: usize = 5;
const GROUPED_BATCH: usize = 3;
const SHARED_BYTES: usize = 32 * 1024;

fn config(bandwidth_boost: usize) -> FftConfig {
    FftConfig::new(DIMENSIONS.to_vec())
        .with_batch_count(BATCH_COUNT)
        .with_grouped_batch(0, GROUPED_BATCH)
        .unwrap()
        .with_bandwidth_boost(bandwidth_boost)
        .with_zero_padding(0, 1, 2)
        .unwrap()
}

fn input_and_masked_expected() -> (Vec<Complex32>, Vec<Complex32>) {
    let tensor_len = DIMENSIONS.iter().product::<usize>();
    let mut input = vec![Complex32::new(0.0, 0.0); tensor_len * BATCH_COUNT];
    let mut expected = input.clone();
    for batch in 0..BATCH_COUNT {
        let base = batch * tensor_len;
        input[base] = Complex32::new(1.0, 0.0);
        expected[base] = Complex32::new(1.0, 0.0);
        // Axis-0 index 1 is inside zero-padding [1, 2); this non-zero sentinel must
        // disappear before the Bluestein wrapper/child sees the packed transform.
        input[base + 8 + 3] = Complex32::new(0.75, -0.25);
    }
    (input, expected)
}

fn assert_impulse_spectrum(values: &[Complex32], label: &str) {
    let max_error = values
        .iter()
        .map(|value| ((value.re - 1.0).powi(2) + value.im.powi(2)).sqrt())
        .fold(0.0f32, f32::max);
    assert!(
        max_error <= 5.0e-3,
        "{label} grouped padded p263 Bluestein impulse spectrum error {max_error:e}"
    );
}

fn assert_round_trip(actual: &[Complex32], expected: &[Complex32], label: &str) {
    assert_eq!(actual.len(), expected.len());
    let max_error = actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| {
            ((actual.re - expected.re).powi(2) + (actual.im - expected.im).powi(2)).sqrt()
        })
        .fold(0.0f32, f32::max);
    assert!(
        max_error <= 5.0e-3,
        "{label} grouped padded p263 Bluestein round-trip error {max_error:e}"
    );
}

fn planning_profile(mut profile: vkfft_rs::DeviceProfile) -> Option<vkfft_rs::DeviceProfile> {
    if profile.shared_memory_bytes < SHARED_BYTES
        || profile.shared_memory_pow2_bytes < SHARED_BYTES
        || profile.max_threads_per_block < 512
        || profile.max_workgroup_size[0] < 512
        || profile.max_workgroup_size[1] < 512
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

fn assert_bluestein_structure(ir: &TransformIr, vendor: GpuVendor, label: &str) {
    let TransformIr::ComplexNd(nd) = ir else {
        panic!("{label} p263x8 probe did not build ComplexNd");
    };
    let outer = nd.axes.iter().find(|axis| axis.axis == 0).unwrap();
    let vkfft_rs::OneDimFftIr::Bluestein(pipeline) = &outer.transform else {
        panic!("{label} p263 higher axis did not select device-default Bluestein");
    };
    assert_eq!(
        pipeline.batch_count, 40,
        "{label} Bluestein transform count"
    );
    assert_eq!(
        pipeline.grouped_batch, GROUPED_BATCH,
        "{label} wrapper group"
    );
    let wrapper = pipeline
        .preprocess
        .axis_batch_block
        .expect("higher-axis Bluestein wrapper block");
    assert_eq!(
        wrapper.grouped_batch, GROUPED_BATCH,
        "{label} wrapper block group"
    );
    assert!(
        wrapper.transforms_on_x,
        "{label} wrapper must keep transforms on X"
    );
    assert!(!wrapper.axis_swapped, "{label} wrapper must remain no-swap");
    assert_eq!(pipeline.multiply.axis_batch_block, Some(wrapper));
    assert_eq!(pipeline.postprocess.axis_batch_block, Some(wrapper));

    let (expected_m, expected_threads) = match vendor {
        GpuVendor::Nvidia => (567usize, 81usize),
        GpuVendor::Amd => (625usize, 125usize),
        _ => return,
    };
    assert_eq!(pipeline.convolution_len, expected_m, "{label} Bluestein M");
    for child in [&pipeline.forward_fft, &pipeline.inverse_fft] {
        let vkfft_rs::RecursiveFftNodeIr::Stockham(kernel) = &child.root else {
            panic!("{label} p263 M{expected_m} convolution child is not Stockham");
        };
        assert_eq!(kernel.batch_count, 40, "{label} child transform count");
        assert_eq!(
            kernel.workgroup_grouping.transforms_per_workgroup, GROUPED_BATCH,
            "{label} child group"
        );
        assert_eq!(
            kernel.workgroup_grouping.threads_per_transform, expected_threads,
            "{label} child lanes"
        );
        assert_eq!(
            kernel.workgroup_grouping.axis_layout,
            vkfft_rs::kernel_ir::StockhamWorkgroupAxisLayout::TransformsXThreadsY,
            "{label} child X/Y ownership"
        );
        assert_eq!(
            [
                kernel.workgroup_size.x as usize,
                kernel.workgroup_size.y as usize
            ],
            [GROUPED_BATCH, expected_threads],
            "{label} child local size"
        );
    }
}

#[cfg(any(feature = "cuda-runtime", feature = "opencl-runtime"))]
fn run_native<R: vkfft_rs::backend::NativeRuntime>(runtime: &R) {
    use vkfft_rs::backend::{NativeTransformInput32, NativeTransformOutput32};

    let Some(profile) = planning_profile(runtime.device_profile()) else {
        return;
    };
    let (input, expected) = input_and_masked_expected();
    for bandwidth_boost in [0usize, 2] {
        let label = format!("{} p263x8 b{bandwidth_boost}", runtime.device_name());
        let forward =
            TransformIr::build(config(bandwidth_boost), Direction::Forward, profile).unwrap();
        assert_bluestein_structure(&forward, profile.vendor, &label);
        let spectrum = match runtime
            .execute_transform_f32(&forward, NativeTransformInput32::Complex(&input))
            .unwrap()
        {
            NativeTransformOutput32::Complex(values) => values,
            NativeTransformOutput32::Real(_) => panic!("{label} returned real forward output"),
        };
        assert_impulse_spectrum(&spectrum, &label);

        let inverse = TransformIr::build(
            config(bandwidth_boost).with_inverse_normalization(true),
            Direction::Inverse,
            profile,
        )
        .unwrap();
        assert_bluestein_structure(&inverse, profile.vendor, &label);
        let restored = match runtime
            .execute_transform_f32(&inverse, NativeTransformInput32::Complex(&spectrum))
            .unwrap()
        {
            NativeTransformOutput32::Complex(values) => values,
            NativeTransformOutput32::Real(_) => panic!("{label} returned real inverse output"),
        };
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
fn cuda_f32_bluestein_p263_padded_grouped_higher_axis_or_skips() {
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
fn opencl_f32_bluestein_p263_padded_grouped_higher_axis_or_skips() {
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
fn vulkan_f32_bluestein_p263_padded_grouped_higher_axis_or_skips() {
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
    let Some(profile) = planning_profile(runtime.device_profile()) else {
        return;
    };
    let (input, expected) = input_and_masked_expected();
    for bandwidth_boost in [0usize, 2] {
        let label = format!("Vulkan p263x8 b{bandwidth_boost}");
        let forward =
            TransformIr::build(config(bandwidth_boost), Direction::Forward, profile).unwrap();
        assert_bluestein_structure(&forward, profile.vendor, &label);
        let spectrum = match runtime
            .execute_transform_f32(&forward, TransformInput32::Complex(&input))
            .unwrap()
        {
            TransformOutput32::Complex(values) => values,
            TransformOutput32::Real(_) => panic!("{label} returned real forward output"),
        };
        assert_impulse_spectrum(&spectrum, &label);

        let inverse = TransformIr::build(
            config(bandwidth_boost).with_inverse_normalization(true),
            Direction::Inverse,
            profile,
        )
        .unwrap();
        assert_bluestein_structure(&inverse, profile.vendor, &label);
        let restored = match runtime
            .execute_transform_f32(&inverse, TransformInput32::Complex(&spectrum))
            .unwrap()
        {
            TransformOutput32::Complex(values) => values,
            TransformOutput32::Real(_) => panic!("{label} returned real inverse output"),
        };
        assert_round_trip(&restored, &expected, &label);
    }
}
