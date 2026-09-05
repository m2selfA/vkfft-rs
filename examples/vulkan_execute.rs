#[cfg(feature = "vulkan-runtime")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use vkfft_rs::backend::{
        KernelBackend,
        vulkan::{VulkanGlslBackend, runtime::VulkanExecutionContext},
    };
    use vkfft_rs::{Complex32, Direction, FftConfig, FftPlan, KernelIr, VkFftError};

    let device_index = std::env::args()
        .nth(1)
        .map(|value| value.parse::<usize>())
        .transpose()?;
    let context_result = match device_index {
        Some(index) => VulkanExecutionContext::new_with_device_index(index),
        None => VulkanExecutionContext::new(),
    };
    let context = match context_result {
        Ok(context) => context,
        Err(VkFftError::VulkanUnavailable(reason)) if device_index.is_none() => {
            eprintln!("Vulkan unavailable: {reason}");
            return Ok(());
        }
        Err(error) => return Err(error.into()),
    };
    let length = 12usize;
    let plan = FftPlan::build(FftConfig::new(vec![length]))?;
    let ir = KernelIr::stockham_1d(&plan, Direction::Forward, context.device_profile())?;
    let shader = VulkanGlslBackend.lower(&ir)?.compile_spirv()?;
    let input = (0..length)
        .map(|index| Complex32::new(index as f32, -(index as f32) * 0.25))
        .collect::<Vec<_>>();
    let output = context.execute_complex32(&shader, &input)?;

    eprintln!("device: {}", context.device_name());
    for (index, value) in output.iter().enumerate() {
        println!("{index}: {} {:+}i", value.re, value.im);
    }
    Ok(())
}

#[cfg(not(feature = "vulkan-runtime"))]
fn main() {
    eprintln!("enable the `vulkan-runtime` feature to run this example");
}
