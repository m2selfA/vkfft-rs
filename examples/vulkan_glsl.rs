use vkfft_rs::backend::{KernelBackend, vulkan::VulkanGlslBackend};
use vkfft_rs::{Backend, DeviceProfile, Direction, FftConfig, FftPlan, GpuVendor, KernelIr};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let plan = FftPlan::build(FftConfig::new(vec![256]).with_batch_count(8))?;
    let device = DeviceProfile::generic(Backend::Vulkan, GpuVendor::Nvidia);
    let ir = KernelIr::stockham_1d(&plan, Direction::Forward, device)?;
    let shader = VulkanGlslBackend.lower(&ir)?;
    let spirv = shader.compile_spirv()?;

    eprintln!(
        "workgroup={:?}, dispatch={:?}, shared={} bytes, spirv={} words",
        shader.workgroup_size,
        shader.dispatch,
        shader.required_shared_memory_bytes,
        spirv.words.len()
    );
    print!("{}", shader.glsl);
    Ok(())
}
