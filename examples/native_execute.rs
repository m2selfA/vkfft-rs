use std::env;

#[cfg(feature = "native-runtime")]
use vkfft_rs::{Complex32, Direction, FftConfig, TransformIr};

#[cfg(feature = "native-runtime")]
use vkfft_rs::backend::NativeRuntime;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = env::args().skip(1).collect::<Vec<_>>();
    let backend = args.first().map(String::as_str).unwrap_or("cuda");
    let length = args
        .get(1)
        .map(|value| value.parse::<usize>())
        .transpose()?
        .unwrap_or(256);
    let device_index = args
        .get(2)
        .map(|value| value.parse::<usize>())
        .transpose()?
        .unwrap_or(0);
    match backend {
        "cuda" => run_cuda(length, device_index),
        "hip" => run_hip(length, device_index),
        "opencl" => run_opencl(length, device_index),
        "level-zero" | "level_zero" => run_level_zero(length, device_index),
        "metal" => run_metal(length, device_index),
        other => Err(format!(
            "unsupported runtime `{other}`; use cuda, hip, opencl, level-zero, or metal"
        )
        .into()),
    }
}

#[cfg(feature = "cuda-runtime")]
fn run_cuda(length: usize, device_index: usize) -> Result<(), Box<dyn std::error::Error>> {
    let context = vkfft_rs::backend::cuda::runtime::CudaExecutionContext::new(device_index)?;
    run_native(&context, length)
}

#[cfg(not(feature = "cuda-runtime"))]
fn run_cuda(_length: usize, _device_index: usize) -> Result<(), Box<dyn std::error::Error>> {
    Err("rebuild with --features cuda-runtime".into())
}

#[cfg(feature = "hip-runtime")]
fn run_hip(length: usize, device_index: usize) -> Result<(), Box<dyn std::error::Error>> {
    let context = vkfft_rs::backend::hip::runtime::HipExecutionContext::new(device_index)?;
    run_native(&context, length)
}

#[cfg(not(feature = "hip-runtime"))]
fn run_hip(_length: usize, _device_index: usize) -> Result<(), Box<dyn std::error::Error>> {
    Err("rebuild with --features hip-runtime".into())
}

#[cfg(feature = "opencl-runtime")]
fn run_opencl(length: usize, device_index: usize) -> Result<(), Box<dyn std::error::Error>> {
    let context = vkfft_rs::backend::opencl::runtime::OpenClExecutionContext::new(device_index)?;
    run_native(&context, length)
}

#[cfg(not(feature = "opencl-runtime"))]
fn run_opencl(_length: usize, _device_index: usize) -> Result<(), Box<dyn std::error::Error>> {
    Err("rebuild with --features opencl-runtime".into())
}

#[cfg(feature = "level-zero-runtime")]
fn run_level_zero(length: usize, device_index: usize) -> Result<(), Box<dyn std::error::Error>> {
    let context =
        vkfft_rs::backend::level_zero::runtime::LevelZeroExecutionContext::new(device_index)?;
    run_native(&context, length)
}

#[cfg(not(feature = "level-zero-runtime"))]
fn run_level_zero(_length: usize, _device_index: usize) -> Result<(), Box<dyn std::error::Error>> {
    Err("rebuild with --features level-zero-runtime".into())
}

#[cfg(feature = "metal-runtime")]
fn run_metal(length: usize, device_index: usize) -> Result<(), Box<dyn std::error::Error>> {
    let context = vkfft_rs::backend::metal::runtime::MetalExecutionContext::new(device_index)?;
    run_native(&context, length)
}

#[cfg(not(feature = "metal-runtime"))]
fn run_metal(_length: usize, _device_index: usize) -> Result<(), Box<dyn std::error::Error>> {
    Err("rebuild with --features metal-runtime".into())
}

#[cfg(feature = "native-runtime")]
fn run_native<R: NativeRuntime>(
    runtime: &R,
    length: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let input = (0..length)
        .map(|index| {
            let x = index as f32;
            Complex32::new((0.071 * x).sin() + x * 0.0002, (0.029 * x).cos())
        })
        .collect::<Vec<_>>();
    let ir = TransformIr::build(
        FftConfig::new(vec![length]),
        Direction::Forward,
        runtime.device_profile(),
    )?;
    let source =
        vkfft_rs::backend::NativeSourceBackend::new(runtime.backend()).lower_transform(&ir)?;
    let output = runtime.execute_program_complex32(&source, &input)?;
    println!(
        "backend={:?}, device={}, length={}, passes={}, output[0]={:?}",
        runtime.backend(),
        runtime.device_name(),
        length,
        source.shaders.len(),
        output.first()
    );
    Ok(())
}
