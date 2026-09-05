use std::env;
#[cfg(feature = "vulkan-runtime")]
use std::time::Instant;

#[cfg(feature = "vulkan-runtime")]
use vkfft_rs::{
    BufferAccess, Complex32, Direction, FftConfig, FftPlan, OneDimFftIr, PlannerTuning, ProgramIr,
    TransformIr,
};

#[cfg(feature = "vulkan-runtime")]
use vkfft_rs::backend::vulkan::{VulkanGlslBackend, VulkanSpirvShader};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BenchmarkMode {
    Default,
    /// Reproduce the N4089 `[87,47]` p29-FFT low-upload witness with a roomy
    /// two-shared fused kernel: portable Rader tuning plus a 32 KiB planner budget.
    N4089P29Fft32k,
    /// Reproduce the same N4089 p29-FFT witness below the two-shared footprint so
    /// the reduced single-shared capacity fallback is selected.
    N4089P29Fft16k,
}

impl BenchmarkMode {
    fn parse(value: Option<&str>) -> Result<Self, Box<dyn std::error::Error>> {
        match value.unwrap_or("default") {
            "default" => Ok(Self::Default),
            "n4089-p29-32k" => Ok(Self::N4089P29Fft32k),
            "n4089-p29-16k" => Ok(Self::N4089P29Fft16k),
            other => Err(format!(
                "unsupported benchmark mode `{other}`; use default, n4089-p29-32k, or n4089-p29-16k"
            )
            .into()),
        }
    }

    #[cfg(feature = "vulkan-runtime")]
    fn label(self) -> &'static str {
        match self {
            Self::Default => "default",
            Self::N4089P29Fft32k => "n4089-p29-32k",
            Self::N4089P29Fft16k => "n4089-p29-16k",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ExecutionMode {
    HostRoundTrip,
    DeviceResidentChain,
    DeviceResidentChainInto,
}

impl ExecutionMode {
    fn parse(value: Option<&str>) -> Result<Self, Box<dyn std::error::Error>> {
        match value.unwrap_or("host-roundtrip") {
            "host-roundtrip" | "host" => Ok(Self::HostRoundTrip),
            "device-resident-chain" | "resident" => Ok(Self::DeviceResidentChain),
            "device-resident-chain-into" | "resident-into" => Ok(Self::DeviceResidentChainInto),
            other => Err(format!(
                "unsupported execution mode `{other}`; use host-roundtrip, device-resident-chain, or device-resident-chain-into"
            )
            .into()),
        }
    }

    #[cfg(feature = "vulkan-runtime")]
    fn label(self) -> &'static str {
        match self {
            Self::HostRoundTrip => "host-roundtrip",
            Self::DeviceResidentChain => "device-resident-chain",
            Self::DeviceResidentChainInto => "device-resident-chain-into",
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = env::args().skip(1).collect::<Vec<_>>();
    if args
        .first()
        .is_some_and(|value| matches!(value.as_str(), "-h" | "--help"))
    {
        print_help();
        return Ok(());
    }
    let length = parse_arg(&args, 0, 4096)?;
    let batch_count = parse_arg(&args, 1, 64)?;
    let iterations = parse_arg(&args, 2, 20)?;
    let benchmark_mode = BenchmarkMode::parse(args.get(3).map(String::as_str))?;
    let execution_mode = ExecutionMode::parse(args.get(4).map(String::as_str))?;
    let device_index = args
        .get(5)
        .map(|value| value.parse::<usize>())
        .transpose()?;
    if iterations == 0 || batch_count == 0 || length == 0 {
        return Err("length, batch_count, and iterations must be non-zero".into());
    }
    run_vulkan(
        length,
        batch_count,
        iterations,
        benchmark_mode,
        execution_mode,
        device_index,
    )
}

fn print_help() {
    println!(
        "vkfft-rs Vulkan benchmark\n\
         \n\
         Usage:\n\
           cargo run --example vulkan_benchmark --features vulkan-runtime -- [length] [batch] [iterations] [mode] [execution-mode] [device-index]\n\
         \n\
         Modes:\n\
           default           Use the runtime device profile unchanged\n\
           n4089-p29-32k     N4089 p29-FFT roomy two-shared witness (requires length=4089)\n\
           n4089-p29-16k     N4089 p29-FFT reduced-shared witness (requires length=4089)\n\
         \n\
         Execution modes:\n\
           host-roundtrip               Read each FFT back to host before the next ProgramIr (default)\n\
           device-resident-chain        Keep the forward output device-resident until inverse completes\n\
           device-resident-chain-into   Also write the final dense-F32 result into caller-owned output\n\
         \n\
         Device selection:\n\
           device-index                 Optional Vulkan compute-device ordinal; omitted keeps automatic best-device selection\n"
    );
}

fn parse_arg(
    args: &[String],
    index: usize,
    default: usize,
) -> Result<usize, Box<dyn std::error::Error>> {
    Ok(args
        .get(index)
        .map(|value| value.parse::<usize>())
        .transpose()?
        .unwrap_or(default))
}

#[cfg(feature = "vulkan-runtime")]
fn estimated_program_storage_bytes(
    program: &ProgramIr,
) -> Result<usize, Box<dyn std::error::Error>> {
    let complex_bytes = program
        .scalar
        .bytes()
        .checked_mul(2)
        .ok_or("benchmark complex scalar byte count overflow")?;
    let mut total = 0usize;
    for pass in &program.passes {
        for binding in &pass.bindings {
            let resource = program.resource(binding.resource)?;
            let access_count = match binding.access {
                BufferAccess::ReadOnly | BufferAccess::WriteOnly => 1usize,
                BufferAccess::ReadWrite => 2usize,
            };
            let bytes = resource
                .elements
                .checked_mul(complex_bytes)
                .and_then(|value| value.checked_mul(access_count))
                .ok_or("benchmark pass-graph storage byte count overflow")?;
            total = total
                .checked_add(bytes)
                .ok_or("benchmark pass-graph storage byte sum overflow")?;
        }
    }
    Ok(total)
}

#[cfg(feature = "vulkan-runtime")]
fn benchmark_config(
    length: usize,
    batch_count: usize,
    normalize_inverse: bool,
    benchmark_mode: BenchmarkMode,
) -> FftConfig {
    let config = FftConfig::new(vec![length])
        .with_batch_count(batch_count)
        .with_inverse_normalization(normalize_inverse);
    match benchmark_mode {
        BenchmarkMode::Default => config,
        BenchmarkMode::N4089P29Fft32k | BenchmarkMode::N4089P29Fft16k => {
            config.with_tuning(PlannerTuning::portable())
        }
    }
}

#[cfg(feature = "vulkan-runtime")]
fn fused_fft_rader_reports(
    transform: &TransformIr,
) -> Result<Vec<vkfft_rs::FusedFftRaderStaticResourceReport>, Box<dyn std::error::Error>> {
    match transform {
        TransformIr::Complex1d(OneDimFftIr::Recursive(recursive)) => {
            Ok(recursive.fused_fft_rader_static_resource_reports()?)
        }
        _ => Ok(Vec::new()),
    }
}

#[cfg(feature = "vulkan-runtime")]
fn lower_vulkan_transform(
    transform: &TransformIr,
) -> Result<(ProgramIr, Vec<VulkanSpirvShader>), Box<dyn std::error::Error>> {
    let TransformIr::Complex1d(one_dim) = transform else {
        return Err(
            "Vulkan benchmark currently supports one-dimensional C2C transforms only".into(),
        );
    };
    let sources = VulkanGlslBackend.lower_one_dim_fft(one_dim)?;
    let mut shaders = Vec::with_capacity(sources.len());
    for source in sources {
        shaders.push(source.compile_spirv()?);
    }
    Ok((ProgramIr::one_dim_fft(one_dim)?, shaders))
}

#[cfg(feature = "vulkan-runtime")]
fn print_vulkan_resource_reports(
    direction: &str,
    transform: &TransformIr,
    program: &ProgramIr,
    shaders: &[VulkanSpirvShader],
) -> Result<(), Box<dyn std::error::Error>> {
    if program.passes.len() != shaders.len() {
        return Err(format!(
            "Vulkan shader count {} does not match program pass count {}",
            shaders.len(),
            program.passes.len()
        )
        .into());
    }
    for (pass, shader) in program.passes.iter().zip(shaders) {
        println!(
            "vulkan_shader_resource direction={direction} pass={} shared_bytes={} workgroup={}x{}x{} dispatch={}x{}x{} required_subgroup_size={:?} require_full_subgroups={} metric_scope=typed_lowered_spirv_metadata_not_compiler_register_allocation",
            pass.name,
            shader.required_shared_memory_bytes,
            shader.workgroup_size.x,
            shader.workgroup_size.y,
            shader.workgroup_size.z,
            shader.dispatch.x,
            shader.dispatch.y,
            shader.dispatch.z,
            shader.required_subgroup_size,
            shader.require_full_subgroups,
        );
    }
    for report in fused_fft_rader_reports(transform)? {
        let Some((pass, shader)) = program
            .passes
            .iter()
            .zip(shaders)
            .find(|(pass, _)| pass.name == report.pass_name)
        else {
            return Err(format!(
                "typed static resource report has no matching Vulkan pass `{}`",
                report.pass_name
            )
            .into());
        };
        if shader.required_shared_memory_bytes != report.required_shared_memory_bytes
            || shader.workgroup_size != report.workgroup_size
        {
            return Err(format!(
                "typed/lowered Vulkan static resource mismatch for `{}`: typed shared={} workgroup={:?}, lowered shared={} workgroup={:?}",
                pass.name,
                report.required_shared_memory_bytes,
                report.workgroup_size,
                shader.required_shared_memory_bytes,
                shader.workgroup_size,
            )
            .into());
        }
        println!(
            "static_resource direction={direction} pass={} shared_bytes={} workgroup={}x{}x{} uniform_barriers={} max_logical_register_complex_values_per_invocation={} metric_scope=typed_ir_not_hardware_occupancy",
            report.pass_name,
            report.required_shared_memory_bytes,
            report.workgroup_size.x,
            report.workgroup_size.y,
            report.workgroup_size.z,
            report.uniform_barrier_count,
            report.max_logical_register_complex_values_per_invocation,
        );
    }
    println!(
        "compiled_resource direction={direction} backend=vulkan available=false reason=no_authoritative_register_allocation_query_in_current_runtime metric_scope=compiler_driver_authoritative_not_achieved_occupancy"
    );
    println!(
        "theoretical_occupancy direction={direction} backend=vulkan available=false reason=compiler_register_allocation_unavailable metric_scope=coarse_residency_model_not_profiler_measurement"
    );
    Ok(())
}

#[cfg(feature = "vulkan-runtime")]
fn run_vulkan(
    length: usize,
    batch_count: usize,
    iterations: usize,
    benchmark_mode: BenchmarkMode,
    execution_mode: ExecutionMode,
    device_index: Option<usize>,
) -> Result<(), Box<dyn std::error::Error>> {
    let context = match device_index {
        Some(device_index) => {
            vkfft_rs::backend::vulkan::runtime::VulkanExecutionContext::new_with_device_index(
                device_index,
            )?
        }
        None => vkfft_rs::backend::vulkan::runtime::VulkanExecutionContext::new()?,
    };
    let mut planner_device = context.device_profile();
    let witness_shared_bytes = match benchmark_mode {
        BenchmarkMode::Default => None,
        BenchmarkMode::N4089P29Fft32k => Some(32 * 1024),
        BenchmarkMode::N4089P29Fft16k => Some(16 * 1024),
    };
    if let Some(witness_shared_bytes) = witness_shared_bytes {
        if length != 4089 {
            return Err(format!(
                "{} benchmark mode requires length=4089",
                benchmark_mode.label()
            )
            .into());
        }
        if planner_device.shared_memory_bytes < witness_shared_bytes {
            return Err(format!(
                "{} requires at least {witness_shared_bytes} physical shared-memory bytes, device reports {}",
                benchmark_mode.label(),
                planner_device.shared_memory_bytes
            )
            .into());
        }
        planner_device.shared_memory_bytes = witness_shared_bytes;
        planner_device.shared_memory_pow2_bytes = witness_shared_bytes;
    }

    let config = benchmark_config(length, batch_count, false, benchmark_mode);
    let plan = FftPlan::build(config.clone())?;
    let algorithm = plan.axes[0].algorithm.kind();
    let forward = TransformIr::build(config, Direction::Forward, planner_device)?;
    let inverse = TransformIr::build(
        benchmark_config(length, batch_count, true, benchmark_mode),
        Direction::Inverse,
        planner_device,
    )?;
    let (forward_program, forward_shaders) = lower_vulkan_transform(&forward)?;
    let (inverse_program, inverse_shaders) = lower_vulkan_transform(&inverse)?;

    let input = (0..length * batch_count)
        .map(|index| {
            let x = index as f32;
            Complex32::new((0.0031 * x).sin() + x * 1.0e-7, (0.0023 * x).cos())
        })
        .collect::<Vec<_>>();

    // Warm pipeline/LUT/transient-buffer caches before timing steady-state synchronous execution.
    let programs = [
        (&forward_program, forward_shaders.as_slice()),
        (&inverse_program, inverse_shaders.as_slice()),
    ];
    let mut warm = input.clone();
    let mut warm_output = vec![Complex32::default(); input.len()];
    for _ in 0..3 {
        match execution_mode {
            ExecutionMode::HostRoundTrip => {
                let spectrum =
                    context.execute_program_complex32(&forward_program, &forward_shaders, &warm)?;
                warm = context.execute_program_complex32(
                    &inverse_program,
                    &inverse_shaders,
                    &spectrum,
                )?;
            }
            ExecutionMode::DeviceResidentChain => {
                warm = context.execute_program_chain_complex32(&programs, &warm)?;
            }
            ExecutionMode::DeviceResidentChainInto => {
                context.execute_program_chain_complex32_into(&programs, &warm, &mut warm_output)?;
                std::mem::swap(&mut warm, &mut warm_output);
            }
        }
    }

    let start = Instant::now();
    let mut restored = input.clone();
    for _ in 0..iterations {
        match execution_mode {
            ExecutionMode::HostRoundTrip => {
                let spectrum = context.execute_program_complex32(
                    &forward_program,
                    &forward_shaders,
                    &input,
                )?;
                restored = context.execute_program_complex32(
                    &inverse_program,
                    &inverse_shaders,
                    &spectrum,
                )?;
            }
            ExecutionMode::DeviceResidentChain => {
                restored = context.execute_program_chain_complex32(&programs, &input)?;
            }
            ExecutionMode::DeviceResidentChainInto => {
                context.execute_program_chain_complex32_into(&programs, &input, &mut restored)?;
            }
        }
    }
    let elapsed = start.elapsed();
    let seconds = elapsed.as_secs_f64();
    let pair_seconds = seconds / iterations as f64;
    let logical_bytes_per_pair = 4usize
        .checked_mul(length)
        .and_then(|value| value.checked_mul(batch_count))
        .and_then(|value| value.checked_mul(std::mem::size_of::<Complex32>()))
        .ok_or("benchmark logical byte count overflow")?;
    let logical_gbps = logical_bytes_per_pair as f64 / pair_seconds / 1.0e9;
    let estimated_storage_bytes_per_pair = estimated_program_storage_bytes(&forward_program)?
        .checked_add(estimated_program_storage_bytes(&inverse_program)?)
        .ok_or("benchmark pair storage byte count overflow")?;
    let estimated_storage_gbps = estimated_storage_bytes_per_pair as f64 / pair_seconds / 1.0e9;
    let input_buffer_bytes = length
        .checked_mul(batch_count)
        .and_then(|value| value.checked_mul(std::mem::size_of::<Complex32>()))
        .ok_or("benchmark input buffer byte count overflow")?;
    let estimated_transfer_equivalents =
        estimated_storage_bytes_per_pair as f64 / input_buffer_bytes as f64;
    let max_error = restored
        .iter()
        .zip(&input)
        .map(|(actual, expected)| {
            let dr = actual.re as f64 - expected.re as f64;
            let di = actual.im as f64 - expected.im as f64;
            (dr * dr + di * di).sqrt()
        })
        .fold(0.0, f64::max);

    println!(
        "backend=Vulkan device={} device_index={} algorithm={algorithm:?} benchmark_mode={} execution_mode={} planner_shared_bytes={} length={length} batch={batch_count} iterations={iterations}",
        context.device_name(),
        device_index
            .map(|device_index| device_index.to_string())
            .unwrap_or_else(|| "auto".to_owned()),
        benchmark_mode.label(),
        execution_mode.label(),
        planner_device.shared_memory_bytes,
    );
    println!(
        "forward_passes={} inverse_passes={} pair_ms={:.3} timing_scope=host_wall_clock_synchronous_submission logical_io_gbps={:.3} estimated_pass_graph_storage_gbps={:.3} estimated_transfer_equivalents={:.2} max_roundtrip_error={:.3e}",
        forward_shaders.len(),
        inverse_shaders.len(),
        pair_seconds * 1.0e3,
        logical_gbps,
        estimated_storage_gbps,
        estimated_transfer_equivalents,
        max_error,
    );
    println!(
        "traffic_definition=logical_io uses 4*N*batch*sizeof(complex32) bytes per FFT+iFFT pair; estimated_pass_graph_storage sums every ProgramIr binding's full resource extent (ReadWrite counted twice), including scratch/LUT/auxiliary traffic. It is an upstream-style upload/transfer estimate, not a hardware DRAM counter."
    );
    print_vulkan_resource_reports("forward", &forward, &forward_program, &forward_shaders)?;
    print_vulkan_resource_reports("inverse", &inverse, &inverse_program, &inverse_shaders)?;
    println!(
        "vulkan_resource_definition=shared/workgroup/dispatch/subgroup fields come from typed lowered SPIR-V metadata. Fused static barrier/register-value counts come from typed IR. The current Vulkan runtime does not claim compiler register allocation, theoretical occupancy, or achieved profiler occupancy."
    );
    Ok(())
}

#[cfg(not(feature = "vulkan-runtime"))]
fn run_vulkan(
    _length: usize,
    _batch_count: usize,
    _iterations: usize,
    _benchmark_mode: BenchmarkMode,
    _execution_mode: ExecutionMode,
    _device_index: Option<usize>,
) -> Result<(), Box<dyn std::error::Error>> {
    Err("rebuild with --features vulkan-runtime".into())
}
