#![cfg(any(
    feature = "cuda-runtime",
    feature = "opencl-runtime",
    feature = "vulkan-runtime"
))]

use vkfft_rs::{Complex32, Direction, FftConfig, GpuVendor, TransformIr};

const DIMENSIONS: [usize; 2] = [2053, 8];
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
        // This sample sits in outer-axis zero-padding [1,2). The forward spectrum can
        // be all ones only when the padding boundary executes before Bluestein packing.
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
        max_error <= 7.5e-3,
        "{label} grouped padded p2053 Bluestein impulse spectrum error {max_error:e}"
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
        max_error <= 7.5e-3,
        "{label} grouped padded p2053 Bluestein round-trip error {max_error:e}"
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
        panic!("{label} p2053x8 probe did not build ComplexNd");
    };
    let outer = nd.axes.iter().find(|axis| axis.axis == 0).unwrap();
    let vkfft_rs::OneDimFftIr::Bluestein(pipeline) = &outer.transform else {
        panic!("{label} p2053 higher axis did not select device-default Bluestein");
    };
    assert_eq!(pipeline.batch_count, 40, "{label} wrapper transform count");
    assert_eq!(
        pipeline.grouped_batch, GROUPED_BATCH,
        "{label} wrapper group"
    );
    let wrapper = pipeline
        .preprocess
        .axis_batch_block
        .expect("higher-axis p2053 Bluestein wrapper block");
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

    let (expected_m, expected_split, expected_blocks) = match vendor {
        GpuVendor::Nvidia => (
            4368usize,
            [78usize, 56usize],
            [(2240usize, 13usize), (3120usize, 8usize)],
        ),
        GpuVendor::Amd => (
            4375usize,
            [125usize, 35usize],
            [(1400usize, 25usize), (5000usize, 7usize)],
        ),
        _ => return,
    };
    assert_eq!(pipeline.convolution_len, expected_m, "{label} Bluestein M");

    let forward_schedule = pipeline
        .forward_fft
        .stockham_upload_schedule
        .as_ref()
        .expect("p2053 Bluestein forward upload schedule");
    let inverse_schedule = pipeline
        .inverse_fft
        .stockham_upload_schedule
        .as_ref()
        .expect("p2053 Bluestein inverse upload schedule");
    assert_eq!(
        forward_schedule, inverse_schedule,
        "{label} forward/inverse schedule"
    );
    assert_eq!(forward_schedule.upload_count, 2, "{label} upload count");
    assert_eq!(
        forward_schedule.axis_split, expected_split,
        "{label} axisSplit"
    );
    assert_eq!(
        forward_schedule.batch_count, 40,
        "{label} child batch count"
    );

    for child in [&pipeline.forward_fft, &pipeline.inverse_fft] {
        let four_step = child
            .four_step_plan
            .as_ref()
            .expect("p2053 Bluestein child Four-step plan");
        assert_eq!(four_step.batch_count, 40, "{label} Four-step batch count");
        assert_eq!(four_step.uploads.len(), 2, "{label} Four-step upload count");
        for axis_upload_id in 0..2 {
            let upload = four_step
                .uploads
                .iter()
                .find(|upload| upload.axis_upload_id == axis_upload_id)
                .unwrap();
            let (expected_transform_count, expected_threads) = expected_blocks[axis_upload_id];
            assert_eq!(
                upload.fft_len, expected_split[axis_upload_id],
                "{label} u{axis_upload_id} N"
            );
            assert_eq!(
                upload.transform_count, expected_transform_count,
                "{label} u{axis_upload_id} transform count"
            );
            let block = upload.axis_block.expect("p2053 Bluestein upload block");
            assert_eq!(
                block.grouped_batch, GROUPED_BATCH,
                "{label} u{axis_upload_id} group"
            );
            assert_eq!(
                block.threads_per_transform, expected_threads,
                "{label} u{axis_upload_id} lanes"
            );
            assert_eq!(
                [block.local_size_x, block.local_size_y],
                [GROUPED_BATCH, expected_threads],
                "{label} u{axis_upload_id} local size"
            );
            assert!(
                block.transforms_on_x,
                "{label} u{axis_upload_id} must be on-X"
            );
            assert!(
                !block.axis_swapped,
                "{label} u{axis_upload_id} must remain no-swap"
            );
        }
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
        let label = format!("{} p2053x8 b{bandwidth_boost}", runtime.device_name());
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
fn cuda_f32_bluestein_p2053_multi_upload_padded_grouped_higher_axis_or_skips() {
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
fn opencl_f32_bluestein_p2053_multi_upload_padded_grouped_higher_axis_or_skips() {
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
fn vulkan_f32_bluestein_p2053_multi_upload_padded_grouped_higher_axis_or_skips() {
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
        let label = format!("Vulkan p2053x8 b{bandwidth_boost}");
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
