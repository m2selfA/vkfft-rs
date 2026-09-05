use std::env;
#[cfg(feature = "native-runtime")]
use std::time::Instant;

#[cfg(feature = "native-runtime")]
use vkfft_rs::{
    BufferAccess, Complex32, Direction, FftConfig, FftPlan, OneDimFftIr, PlannerTuning, ProgramIr,
    TransformIr,
};

#[cfg(feature = "native-runtime")]
use vkfft_rs::backend::{
    NativeCompiledResourceMetrics, NativeProgramSource, NativeRuntime, NativeSourceBackend,
};

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

    #[cfg(feature = "native-runtime")]
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

    #[cfg(feature = "native-runtime")]
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
    let backend = args.first().map(String::as_str).unwrap_or("cuda");
    let length = parse_arg(&args, 1, 4096)?;
    let batch_count = parse_arg(&args, 2, 64)?;
    let iterations = parse_arg(&args, 3, 20)?;
    let benchmark_mode = BenchmarkMode::parse(args.get(4).map(String::as_str))?;
    let device_index = parse_arg(&args, 5, 0)?;
    let execution_mode = ExecutionMode::parse(args.get(6).map(String::as_str))?;
    if iterations == 0 || batch_count == 0 || length == 0 {
        return Err("length, batch_count, and iterations must be non-zero".into());
    }
    match backend {
        "cuda" => run_cuda(
            length,
            batch_count,
            iterations,
            benchmark_mode,
            device_index,
            execution_mode,
        ),
        "hip" => run_hip(
            length,
            batch_count,
            iterations,
            benchmark_mode,
            device_index,
            execution_mode,
        ),
        "opencl" => run_opencl(
            length,
            batch_count,
            iterations,
            benchmark_mode,
            device_index,
            execution_mode,
        ),
        "level-zero" | "level_zero" => run_level_zero(
            length,
            batch_count,
            iterations,
            benchmark_mode,
            device_index,
            execution_mode,
        ),
        "metal" => run_metal(
            length,
            batch_count,
            iterations,
            benchmark_mode,
            device_index,
            execution_mode,
        ),
        other => Err(format!(
            "unsupported runtime `{other}`; use cuda, hip, opencl, level-zero, or metal"
        )
        .into()),
    }
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

#[cfg(any(
    feature = "hip-runtime",
    feature = "level-zero-runtime",
    feature = "metal-runtime"
))]
fn require_host_roundtrip(
    execution_mode: ExecutionMode,
    backend: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    if execution_mode != ExecutionMode::HostRoundTrip {
        return Err(format!(
            "device-resident execution is not implemented for `{backend}`; use host-roundtrip"
        )
        .into());
    }
    Ok(())
}

#[cfg(feature = "cuda-runtime")]
fn run_cuda(
    length: usize,
    batch_count: usize,
    iterations: usize,
    benchmark_mode: BenchmarkMode,
    device_index: usize,
    execution_mode: ExecutionMode,
) -> Result<(), Box<dyn std::error::Error>> {
    let context = vkfft_rs::backend::cuda::runtime::CudaExecutionContext::new(device_index)?;
    match execution_mode {
        ExecutionMode::HostRoundTrip => run_native(
            &context,
            length,
            batch_count,
            iterations,
            benchmark_mode,
            execution_mode,
        ),
        ExecutionMode::DeviceResidentChain => {
            run_cuda_resident_chain(&context, length, batch_count, iterations, benchmark_mode)
        }
        ExecutionMode::DeviceResidentChainInto => {
            run_cuda_resident_chain_into(&context, length, batch_count, iterations, benchmark_mode)
        }
    }
}

#[cfg(not(feature = "cuda-runtime"))]
fn run_cuda(
    _length: usize,
    _batch_count: usize,
    _iterations: usize,
    _benchmark_mode: BenchmarkMode,
    _device_index: usize,
    _execution_mode: ExecutionMode,
) -> Result<(), Box<dyn std::error::Error>> {
    Err("rebuild with --features cuda-runtime".into())
}

#[cfg(feature = "hip-runtime")]
fn run_hip(
    length: usize,
    batch_count: usize,
    iterations: usize,
    benchmark_mode: BenchmarkMode,
    device_index: usize,
    execution_mode: ExecutionMode,
) -> Result<(), Box<dyn std::error::Error>> {
    require_host_roundtrip(execution_mode, "hip")?;
    let context = vkfft_rs::backend::hip::runtime::HipExecutionContext::new(device_index)?;
    run_native(
        &context,
        length,
        batch_count,
        iterations,
        benchmark_mode,
        execution_mode,
    )
}

#[cfg(not(feature = "hip-runtime"))]
fn run_hip(
    _length: usize,
    _batch_count: usize,
    _iterations: usize,
    _benchmark_mode: BenchmarkMode,
    _device_index: usize,
    _execution_mode: ExecutionMode,
) -> Result<(), Box<dyn std::error::Error>> {
    Err("rebuild with --features hip-runtime".into())
}

#[cfg(feature = "opencl-runtime")]
fn run_opencl(
    length: usize,
    batch_count: usize,
    iterations: usize,
    benchmark_mode: BenchmarkMode,
    device_index: usize,
    execution_mode: ExecutionMode,
) -> Result<(), Box<dyn std::error::Error>> {
    let context = vkfft_rs::backend::opencl::runtime::OpenClExecutionContext::new(device_index)?;
    match execution_mode {
        ExecutionMode::HostRoundTrip => run_native(
            &context,
            length,
            batch_count,
            iterations,
            benchmark_mode,
            execution_mode,
        ),
        ExecutionMode::DeviceResidentChain => {
            run_opencl_resident_chain(&context, length, batch_count, iterations, benchmark_mode)
        }
        ExecutionMode::DeviceResidentChainInto => run_opencl_resident_chain_into(
            &context,
            length,
            batch_count,
            iterations,
            benchmark_mode,
        ),
    }
}

#[cfg(not(feature = "opencl-runtime"))]
fn run_opencl(
    _length: usize,
    _batch_count: usize,
    _iterations: usize,
    _benchmark_mode: BenchmarkMode,
    _device_index: usize,
    _execution_mode: ExecutionMode,
) -> Result<(), Box<dyn std::error::Error>> {
    Err("rebuild with --features opencl-runtime".into())
}

#[cfg(feature = "level-zero-runtime")]
fn run_level_zero(
    length: usize,
    batch_count: usize,
    iterations: usize,
    benchmark_mode: BenchmarkMode,
    device_index: usize,
    execution_mode: ExecutionMode,
) -> Result<(), Box<dyn std::error::Error>> {
    require_host_roundtrip(execution_mode, "level-zero")?;
    let context =
        vkfft_rs::backend::level_zero::runtime::LevelZeroExecutionContext::new(device_index)?;
    run_native(
        &context,
        length,
        batch_count,
        iterations,
        benchmark_mode,
        execution_mode,
    )
}

#[cfg(not(feature = "level-zero-runtime"))]
fn run_level_zero(
    _length: usize,
    _batch_count: usize,
    _iterations: usize,
    _benchmark_mode: BenchmarkMode,
    _device_index: usize,
    _execution_mode: ExecutionMode,
) -> Result<(), Box<dyn std::error::Error>> {
    Err("rebuild with --features level-zero-runtime".into())
}

#[cfg(feature = "metal-runtime")]
fn run_metal(
    length: usize,
    batch_count: usize,
    iterations: usize,
    benchmark_mode: BenchmarkMode,
    device_index: usize,
    execution_mode: ExecutionMode,
) -> Result<(), Box<dyn std::error::Error>> {
    require_host_roundtrip(execution_mode, "metal")?;
    let context = vkfft_rs::backend::metal::runtime::MetalExecutionContext::new(device_index)?;
    run_native(
        &context,
        length,
        batch_count,
        iterations,
        benchmark_mode,
        execution_mode,
    )
}

#[cfg(not(feature = "metal-runtime"))]
fn run_metal(
    _length: usize,
    _batch_count: usize,
    _iterations: usize,
    _benchmark_mode: BenchmarkMode,
    _device_index: usize,
    _execution_mode: ExecutionMode,
) -> Result<(), Box<dyn std::error::Error>> {
    Err("rebuild with --features metal-runtime".into())
}

#[cfg(feature = "native-runtime")]
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

#[cfg(feature = "native-runtime")]
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

#[cfg(feature = "native-runtime")]
fn print_static_resource_reports(
    direction: &str,
    transform: &TransformIr,
    source: &NativeProgramSource,
) -> Result<(), Box<dyn std::error::Error>> {
    for report in fused_fft_rader_reports(transform)? {
        let Some((_, shader)) = source
            .program
            .passes
            .iter()
            .zip(&source.shaders)
            .find(|(pass, _)| pass.name == report.pass_name)
        else {
            return Err(format!(
                "typed static resource report has no matching lowered pass `{}`",
                report.pass_name
            )
            .into());
        };
        if shader.required_shared_memory_bytes != report.required_shared_memory_bytes
            || shader.workgroup_size != report.workgroup_size
        {
            return Err(format!(
                "typed/lowered static resource mismatch for `{}`: typed shared={} workgroup={:?}, lowered shared={} workgroup={:?}",
                report.pass_name,
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
    Ok(())
}

#[cfg(feature = "native-runtime")]
fn print_compiled_resource_reports<R: NativeRuntime>(
    direction: &str,
    runtime: &R,
    source: &NativeProgramSource,
) -> Result<(), Box<dyn std::error::Error>> {
    let reports = runtime.compiled_pass_resource_reports(source)?;
    if reports.is_empty() {
        println!(
            "compiled_resource direction={direction} backend={:?} available=false metric_scope=compiler_driver_authoritative_not_achieved_occupancy",
            runtime.backend()
        );
        return Ok(());
    }
    if reports.len() != source.program.passes.len() {
        return Err(format!(
            "compiled resource report count {} does not match program pass count {}",
            reports.len(),
            source.program.passes.len()
        )
        .into());
    }
    for (pass, report) in source.program.passes.iter().zip(&reports) {
        if pass.name != report.pass_name {
            return Err(format!(
                "compiled resource report pass mismatch: program=`{}`, report=`{}`",
                pass.name, report.pass_name
            )
            .into());
        }
        match &report.metrics {
            NativeCompiledResourceMetrics::Cuda {
                registers_per_thread,
                static_shared_memory_bytes_per_block,
                local_memory_bytes_per_thread,
                max_threads_per_block,
            } => println!(
                "compiled_resource direction={direction} pass={} backend=cuda registers_per_thread={} static_shared_bytes_per_block={} local_bytes_per_thread={} max_threads_per_block={} metric_scope=compiler_driver_authoritative_not_achieved_occupancy",
                pass.name,
                registers_per_thread,
                static_shared_memory_bytes_per_block,
                local_memory_bytes_per_thread,
                max_threads_per_block,
            ),
            NativeCompiledResourceMetrics::Hip {
                registers_per_thread,
                static_shared_memory_bytes_per_block,
                local_memory_bytes_per_thread,
                max_threads_per_block,
            } => println!(
                "compiled_resource direction={direction} pass={} backend=hip registers_per_thread={} static_shared_bytes_per_block={} local_bytes_per_thread={} max_threads_per_block={} metric_scope=compiler_driver_authoritative_not_achieved_occupancy",
                pass.name,
                registers_per_thread,
                static_shared_memory_bytes_per_block,
                local_memory_bytes_per_thread,
                max_threads_per_block,
            ),
            NativeCompiledResourceMetrics::OpenCl {
                local_memory_bytes_per_workgroup,
                private_memory_bytes_per_work_item,
                max_workgroup_size,
                preferred_workgroup_size_multiple,
            } => println!(
                "compiled_resource direction={direction} pass={} backend=opencl local_bytes_per_workgroup={} private_bytes_per_work_item={} max_workgroup_size={} preferred_workgroup_multiple={} metric_scope=compiler_driver_authoritative_not_achieved_occupancy",
                pass.name,
                local_memory_bytes_per_workgroup,
                private_memory_bytes_per_work_item,
                max_workgroup_size,
                preferred_workgroup_size_multiple,
            ),
            NativeCompiledResourceMetrics::LevelZero {
                local_memory_bytes_per_workgroup,
                private_memory_bytes_per_thread,
                spill_memory_bytes,
                required_group_size,
                required_num_subgroups,
                required_subgroup_size,
                max_subgroup_size,
                max_num_subgroups,
            } => println!(
                "compiled_resource direction={direction} pass={} backend=level-zero local_bytes_per_workgroup={} private_bytes_per_thread={} spill_bytes={} required_group_size={:?} required_num_subgroups={} required_subgroup_size={} max_subgroup_size={} max_num_subgroups={} metric_scope=compiler_driver_authoritative_not_achieved_occupancy",
                pass.name,
                local_memory_bytes_per_workgroup,
                private_memory_bytes_per_thread,
                spill_memory_bytes,
                required_group_size,
                required_num_subgroups,
                required_subgroup_size,
                max_subgroup_size,
                max_num_subgroups,
            ),
            NativeCompiledResourceMetrics::Metal {
                static_threadgroup_memory_bytes,
                max_threads_per_threadgroup,
                thread_execution_width,
            } => println!(
                "compiled_resource direction={direction} pass={} backend=metal static_threadgroup_memory_bytes={} max_threads_per_threadgroup={} thread_execution_width={} metric_scope=compiler_driver_authoritative_not_achieved_occupancy",
                pass.name,
                static_threadgroup_memory_bytes,
                max_threads_per_threadgroup,
                thread_execution_width,
            ),
        }
    }
    let occupancy_reports = runtime.theoretical_occupancy_reports(source, &reports)?;
    if occupancy_reports.is_empty() {
        println!(
            "theoretical_occupancy direction={direction} backend={:?} available=false metric_scope=coarse_residency_model_not_profiler_measurement",
            runtime.backend()
        );
    } else {
        for report in occupancy_reports {
            println!(
                "theoretical_occupancy direction={direction} pass={} backend={:?} compute_units={} subgroup_size={} workgroup_threads={} subgroups_per_workgroup={} max_subgroups_per_compute_unit={} blocks_by_thread_subgroups={} blocks_by_registers={:?} blocks_by_shared_memory={:?} architectural_blocks_per_compute_unit={:?} resident_blocks_per_compute_unit_upper_bound={} resident_subgroups_per_compute_unit_upper_bound={} occupancy_upper_bound_percent={:.2} limiting_resources={:?} metric_scope=coarse_residency_upper_bound_not_profiler_measurement",
                report.pass_name,
                report.backend,
                report.compute_unit_count,
                report.subgroup_size,
                report.workgroup_threads,
                report.subgroups_per_workgroup,
                report.max_subgroups_per_compute_unit,
                report.blocks_limited_by_thread_subgroups,
                report.blocks_limited_by_registers,
                report.blocks_limited_by_shared_memory,
                report.architectural_blocks_per_compute_unit,
                report.resident_blocks_per_compute_unit_upper_bound,
                report.resident_subgroups_per_compute_unit_upper_bound,
                report.occupancy_basis_points_upper_bound as f64 / 100.0,
                report.limiting_resources,
            );
        }
    }
    Ok(())
}

#[cfg(feature = "native-runtime")]
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

#[cfg(feature = "native-runtime")]
fn run_native<R: NativeRuntime>(
    runtime: &R,
    length: usize,
    batch_count: usize,
    iterations: usize,
    benchmark_mode: BenchmarkMode,
    execution_mode: ExecutionMode,
) -> Result<(), Box<dyn std::error::Error>> {
    debug_assert_eq!(execution_mode, ExecutionMode::HostRoundTrip);
    run_native_with_pair(
        runtime,
        length,
        batch_count,
        iterations,
        benchmark_mode,
        execution_mode,
        |runtime, forward, inverse, input, output| {
            let spectrum = runtime.execute_program_complex32(forward, input)?;
            *output = runtime.execute_program_complex32(inverse, &spectrum)?;
            Ok(())
        },
    )
}

#[cfg(feature = "cuda-runtime")]
fn run_cuda_resident_chain(
    runtime: &vkfft_rs::backend::cuda::runtime::CudaExecutionContext,
    length: usize,
    batch_count: usize,
    iterations: usize,
    benchmark_mode: BenchmarkMode,
) -> Result<(), Box<dyn std::error::Error>> {
    run_native_with_pair(
        runtime,
        length,
        batch_count,
        iterations,
        benchmark_mode,
        ExecutionMode::DeviceResidentChain,
        |runtime, forward, inverse, input, output| {
            *output = runtime.execute_program_chain_complex32(&[forward, inverse], input)?;
            Ok(())
        },
    )
}

#[cfg(feature = "opencl-runtime")]
fn run_opencl_resident_chain(
    runtime: &vkfft_rs::backend::opencl::runtime::OpenClExecutionContext,
    length: usize,
    batch_count: usize,
    iterations: usize,
    benchmark_mode: BenchmarkMode,
) -> Result<(), Box<dyn std::error::Error>> {
    run_native_with_pair(
        runtime,
        length,
        batch_count,
        iterations,
        benchmark_mode,
        ExecutionMode::DeviceResidentChain,
        |runtime, forward, inverse, input, output| {
            *output = runtime.execute_program_chain_complex32(&[forward, inverse], input)?;
            Ok(())
        },
    )
}

#[cfg(feature = "opencl-runtime")]
fn run_opencl_resident_chain_into(
    runtime: &vkfft_rs::backend::opencl::runtime::OpenClExecutionContext,
    length: usize,
    batch_count: usize,
    iterations: usize,
    benchmark_mode: BenchmarkMode,
) -> Result<(), Box<dyn std::error::Error>> {
    run_native_with_pair(
        runtime,
        length,
        batch_count,
        iterations,
        benchmark_mode,
        ExecutionMode::DeviceResidentChainInto,
        |runtime, forward, inverse, input, output| {
            runtime.execute_program_chain_complex32_into(
                &[forward, inverse],
                input,
                output.as_mut_slice(),
            )
        },
    )
}

#[cfg(feature = "cuda-runtime")]
fn run_cuda_resident_chain_into(
    runtime: &vkfft_rs::backend::cuda::runtime::CudaExecutionContext,
    length: usize,
    batch_count: usize,
    iterations: usize,
    benchmark_mode: BenchmarkMode,
) -> Result<(), Box<dyn std::error::Error>> {
    run_native_with_pair(
        runtime,
        length,
        batch_count,
        iterations,
        benchmark_mode,
        ExecutionMode::DeviceResidentChainInto,
        |runtime, forward, inverse, input, output| {
            runtime.execute_program_chain_complex32_into(
                &[forward, inverse],
                input,
                output.as_mut_slice(),
            )
        },
    )
}

#[cfg(feature = "native-runtime")]
fn run_native_with_pair<R, F>(
    runtime: &R,
    length: usize,
    batch_count: usize,
    iterations: usize,
    benchmark_mode: BenchmarkMode,
    execution_mode: ExecutionMode,
    mut execute_pair: F,
) -> Result<(), Box<dyn std::error::Error>>
where
    R: NativeRuntime,
    F: FnMut(
        &R,
        &NativeProgramSource,
        &NativeProgramSource,
        &[Complex32],
        &mut Vec<Complex32>,
    ) -> vkfft_rs::Result<()>,
{
    let mut planner_device = runtime.device_profile();
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
    let lowering = NativeSourceBackend::new(runtime.backend());
    let forward_source = lowering.lower_transform(&forward)?;
    let inverse_source = lowering.lower_transform(&inverse)?;

    let input = (0..length * batch_count)
        .map(|index| {
            let x = index as f32;
            Complex32::new((0.0031 * x).sin() + x * 1.0e-7, (0.0023 * x).cos())
        })
        .collect::<Vec<_>>();

    // Warm compiler/module/LUT/buffer caches before timing steady-state execution.
    let mut warm = input.clone();
    let mut warm_output = vec![Complex32::default(); input.len()];
    for _ in 0..3 {
        execute_pair(
            runtime,
            &forward_source,
            &inverse_source,
            &warm,
            &mut warm_output,
        )?;
        std::mem::swap(&mut warm, &mut warm_output);
    }

    let mut restored = input.clone();
    let start = Instant::now();
    for _ in 0..iterations {
        execute_pair(
            runtime,
            &forward_source,
            &inverse_source,
            &input,
            &mut restored,
        )?;
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
    let estimated_storage_bytes_per_pair =
        estimated_program_storage_bytes(&forward_source.program)?
            .checked_add(estimated_program_storage_bytes(&inverse_source.program)?)
            .ok_or("benchmark pair storage byte count overflow")?;
    let estimated_storage_gbps = estimated_storage_bytes_per_pair as f64 / pair_seconds / 1.0e9;
    let input_buffer_bytes = length
        .checked_mul(batch_count)
        .and_then(|value| value.checked_mul(std::mem::size_of::<Complex32>()))
        .ok_or("benchmark input buffer byte count overflow")?;
    let estimated_transfer_equivalents =
        estimated_storage_bytes_per_pair as f64 / input_buffer_bytes as f64;
    let host_staging_transfer_equivalents = match execution_mode {
        ExecutionMode::HostRoundTrip => 4usize,
        ExecutionMode::DeviceResidentChain | ExecutionMode::DeviceResidentChainInto => 2usize,
    };
    let host_codec_copy_equivalents = match execution_mode {
        ExecutionMode::HostRoundTrip => 4usize,
        ExecutionMode::DeviceResidentChain => 2usize,
        ExecutionMode::DeviceResidentChainInto => 0usize,
    };
    let host_staging_bytes_per_pair = input_buffer_bytes
        .checked_mul(host_staging_transfer_equivalents)
        .ok_or("benchmark host staging byte count overflow")?;
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
        "backend={:?} device={} algorithm={algorithm:?} benchmark_mode={} execution_mode={} planner_shared_bytes={} length={length} batch={batch_count} iterations={iterations}",
        runtime.backend(),
        runtime.device_name(),
        benchmark_mode.label(),
        execution_mode.label(),
        planner_device.shared_memory_bytes,
    );
    println!(
        "forward_passes={} inverse_passes={} pair_ms={:.3} logical_io_gbps={:.3} estimated_pass_graph_storage_gbps={:.3} estimated_transfer_equivalents={:.2} host_staging_bytes_per_pair={} host_staging_transfer_equivalents={} host_codec_copy_equivalents={} max_roundtrip_error={:.3e}",
        forward_source.shaders.len(),
        inverse_source.shaders.len(),
        pair_seconds * 1.0e3,
        logical_gbps,
        estimated_storage_gbps,
        estimated_transfer_equivalents,
        host_staging_bytes_per_pair,
        host_staging_transfer_equivalents,
        host_codec_copy_equivalents,
        max_error,
    );
    println!(
        "traffic_definition=logical_io uses 4*N*batch*sizeof(complex32) bytes per FFT+iFFT pair; estimated_pass_graph_storage sums every ProgramIr binding's full resource extent (ReadWrite counted twice), including scratch/LUT/auxiliary traffic. host_staging counts the dense-F32 benchmark's actual GPU caller boundary: host-roundtrip performs H2D+D2H for both programs (4 buffers), while both resident modes perform only the first H2D and final D2H (2 buffers). host_codec_copy_equivalents counts full-buffer CPU staging copies/allocations around those transfers: 4 for host-roundtrip, 2 for Vec-returning resident-chain, and 0 for caller-owned resident-chain-into. These are host-boundary counts, not hardware DRAM counters."
    );
    print_static_resource_reports("forward", &forward, &forward_source)?;
    print_static_resource_reports("inverse", &inverse, &inverse_source)?;
    println!(
        "static_resource_definition=shared_bytes/workgroup/barrier/register-value counts come from typed IR plus lowered shader metadata. max_logical_register_complex_values_per_invocation counts logical complex values retained by the generated register schedule; it is not compiler register allocation or achieved occupancy."
    );
    print_compiled_resource_reports("forward", runtime, &forward_source)?;
    print_compiled_resource_reports("inverse", runtime, &inverse_source)?;
    println!(
        "compiled_resource_definition=compiled fields come from the concrete backend compiler/driver after source compilation. They are hardware-target-specific allocation/limit attributes, not achieved occupancy or runtime counter samples."
    );
    println!(
        "theoretical_occupancy_definition=CUDA uses concrete compiler registers/static-shared plus driver SM limits, rounds workgroups to whole warps, and reports a coarse resident-warp upper bound. It ignores allocation granularity, scheduling, memory latency, concurrent kernels, and profiler counters; unsupported backends report unavailable instead of estimating."
    );
    Ok(())
}
