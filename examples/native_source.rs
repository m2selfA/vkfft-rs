use std::env;

use vkfft_rs::backend::KernelBackend;
use vkfft_rs::backend::cuda::CudaSourceBackend;
use vkfft_rs::backend::hip::HipSourceBackend;
use vkfft_rs::backend::level_zero::LevelZeroSourceBackend;
use vkfft_rs::backend::metal::MetalSourceBackend;
use vkfft_rs::backend::opencl::OpenClSourceBackend;
use vkfft_rs::{
    Backend, DeviceProfile, Direction, FftConfig, FftPlan, GpuVendor, KernelIr, Precision,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = env::args().skip(1).collect::<Vec<_>>();
    let backend_name = args.first().map(String::as_str).unwrap_or("cuda");
    let length = args
        .get(1)
        .map(|value| value.parse::<usize>())
        .transpose()?
        .unwrap_or(256);
    let precision = match args.get(2).map(String::as_str).unwrap_or("f32") {
        "f32" => Precision::F32,
        "f64" => Precision::F64,
        other => return Err(format!("unsupported precision `{other}`; use f32 or f64").into()),
    };
    let (backend, vendor) = match backend_name {
        "cuda" => (Backend::Cuda, GpuVendor::Nvidia),
        "hip" => (Backend::Hip, GpuVendor::Amd),
        "opencl" => (Backend::OpenCl, GpuVendor::Amd),
        "level-zero" | "level_zero" => (Backend::LevelZero, GpuVendor::Intel),
        "metal" => (Backend::Metal, GpuVendor::Apple),
        other => return Err(format!("unknown backend `{other}`").into()),
    };
    let mut device = DeviceProfile::generic(backend, vendor);
    device.shared_memory_bytes = 64 * 1024;
    device.shared_memory_pow2_bytes = 64 * 1024;
    device.max_threads_per_block = 1024;
    device.supports_f64 = backend != Backend::Metal;
    let direct_rader = match args.get(3).map(String::as_str).unwrap_or("stockham") {
        "stockham" => false,
        "direct-rader" => true,
        other => {
            return Err(
                format!("unsupported source mode `{other}`; use stockham or direct-rader").into(),
            );
        }
    };
    let plan = FftPlan::build(FftConfig::new(vec![length]).with_precision(precision))?;
    let source = if direct_rader {
        let rader = vkfft_rs::RaderDirectIr::build(&plan, Direction::Forward, device)?;
        match backend {
            Backend::Cuda => CudaSourceBackend.lower_rader_direct(&rader)?.source,
            Backend::Hip => HipSourceBackend.lower_rader_direct(&rader)?.source,
            Backend::OpenCl => OpenClSourceBackend.lower_rader_direct(&rader)?.source,
            Backend::LevelZero => LevelZeroSourceBackend.lower_rader_direct(&rader)?.source,
            Backend::Metal => MetalSourceBackend.lower_rader_direct(&rader)?.source,
            _ => unreachable!(),
        }
    } else {
        let kernel = KernelIr::stockham_1d(&plan, Direction::Forward, device)?;
        match backend {
            Backend::Cuda => CudaSourceBackend.lower(&kernel)?.source,
            Backend::Hip => HipSourceBackend.lower(&kernel)?.source,
            Backend::OpenCl => OpenClSourceBackend.lower(&kernel)?.source,
            Backend::LevelZero => LevelZeroSourceBackend.lower(&kernel)?.source,
            Backend::Metal => MetalSourceBackend.lower(&kernel)?.source,
            _ => unreachable!(),
        }
    };
    print!("{source}");
    Ok(())
}
