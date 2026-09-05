use std::env;
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::PathBuf;

#[cfg(any(
    feature = "vulkan-runtime",
    feature = "cuda-runtime",
    feature = "hip-runtime",
    feature = "opencl-runtime",
    feature = "level-zero-runtime",
    feature = "metal-runtime"
))]
use vkfft_rs::{
    Backend, Complex32, Complex64, DctType, Direction, FftConfig, Precision, PrecisionCaseReport,
    PrecisionTransformFamily, TransformIr, TransformKind, complex_precision_metrics,
    real_precision_metrics,
};

#[cfg(any(
    feature = "vulkan-runtime",
    feature = "cuda-runtime",
    feature = "hip-runtime",
    feature = "opencl-runtime",
    feature = "level-zero-runtime"
))]
use vkfft_rs::PrecisionMetrics;

#[cfg(any(
    feature = "hip-runtime",
    feature = "level-zero-runtime",
    feature = "metal-runtime"
))]
use vkfft_rs::{DeviceProfile, GpuVendor};

#[cfg(any(
    feature = "cuda-runtime",
    feature = "hip-runtime",
    feature = "opencl-runtime",
    feature = "level-zero-runtime",
    feature = "metal-runtime"
))]
use vkfft_rs::backend::{NativeRuntime, NativeTransformInput32, NativeTransformOutput32};

#[cfg(any(
    feature = "cuda-runtime",
    feature = "hip-runtime",
    feature = "opencl-runtime",
    feature = "level-zero-runtime"
))]
use vkfft_rs::backend::{NativeTransformInput64, NativeTransformOutput64};

#[cfg(any(
    feature = "vulkan-runtime",
    feature = "cuda-runtime",
    feature = "hip-runtime",
    feature = "opencl-runtime",
    feature = "level-zero-runtime",
    feature = "metal-runtime"
))]
const PAPER_C2C_LENGTH: usize = 1usize << 27;

#[derive(Debug, Clone)]
struct Options {
    backend: String,
    device_index: Option<usize>,
    output: Option<PathBuf>,
    paper: bool,
}

#[cfg(any(
    feature = "vulkan-runtime",
    feature = "cuda-runtime",
    feature = "hip-runtime",
    feature = "opencl-runtime",
    feature = "level-zero-runtime",
    feature = "metal-runtime"
))]
#[derive(Debug, Clone, Copy)]
enum CaseKind {
    C2c,
    R2c,
    R2rDct2,
}

#[cfg(any(
    feature = "vulkan-runtime",
    feature = "cuda-runtime",
    feature = "hip-runtime",
    feature = "opencl-runtime",
    feature = "level-zero-runtime",
    feature = "metal-runtime"
))]
#[derive(Debug, Clone)]
struct Case {
    family: PrecisionTransformFamily,
    shape: Vec<usize>,
    kind: CaseKind,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let options = parse_options()?;
    #[cfg(not(any(
        feature = "vulkan-runtime",
        feature = "cuda-runtime",
        feature = "hip-runtime",
        feature = "opencl-runtime",
        feature = "level-zero-runtime",
        feature = "metal-runtime"
    )))]
    let _ = options.device_index;
    let mut writer: Box<dyn Write> = match &options.output {
        Some(path) => Box::new(BufWriter::new(File::create(path)?)),
        None => Box::new(BufWriter::new(io::stdout().lock())),
    };

    if options.paper {
        eprintln!(
            "warning: --paper adds a 2^27-sample C2C precision case and can require multiple GiB of host and device memory"
        );
    }

    match options.backend.as_str() {
        "vulkan" => run_vulkan(&options, &mut writer)?,
        "cuda" => run_cuda(&options, &mut writer)?,
        "hip" => run_hip(&options, &mut writer)?,
        "opencl" => run_opencl(&options, &mut writer)?,
        "level-zero" | "level_zero" => run_level_zero(&options, &mut writer)?,
        "metal" => run_metal(&options, &mut writer)?,
        other => {
            return Err(format!(
                "unsupported backend `{other}`; use vulkan, cuda, hip, opencl, level-zero, or metal"
            )
            .into());
        }
    }
    writer.flush()?;
    Ok(())
}

fn parse_options() -> Result<Options, Box<dyn std::error::Error>> {
    let mut backend = "vulkan".to_owned();
    let mut device_index = None;
    let mut output = None;
    let mut paper = false;
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--backend" => {
                backend = args.next().ok_or("--backend requires a value")?;
            }
            "--device" => {
                device_index = Some(
                    args.next()
                        .ok_or("--device requires an index")?
                        .parse::<usize>()?,
                );
            }
            "--output" => {
                output = Some(PathBuf::from(
                    args.next().ok_or("--output requires a path")?,
                ));
            }
            "--paper" => paper = true,
            "-h" | "--help" => {
                print_help();
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument `{other}`; use --help").into()),
        }
    }
    Ok(Options {
        backend,
        device_index,
        output,
        paper,
    })
}

fn print_help() {
    println!(
        "vkfft-rs precision report\n\
         \n\
         Usage:\n\
           cargo run --release --example precision_report --features <runtime-feature> -- [options]\n\
         \n\
         Options:\n\
           --backend <vulkan|cuda|hip|opencl|level-zero|metal>  Runtime backend (default: vulkan)\n\
           --device <index>                                     Device index (Vulkan default: auto; other backends: 0)\n\
           --output <path>                       Write JSONL to a file instead of stdout\n\
           --paper                               Add a 2^27-sample 1D C2C precision case\n\
           -h, --help                            Show this help\n\
         \n\
         Runtime features:\n\
           vulkan-runtime, cuda-runtime, hip-runtime, opencl-runtime, level-zero-runtime, metal-runtime\n\
         \n\
         Precision note:\n\
           Native devices with F64 support emit paired F32/F64 GPU records. Metal, and Level Zero devices without F64, emit F32 GPU records only while still using an independent F64 CPU reference."
    );
}

#[cfg(any(
    feature = "vulkan-runtime",
    feature = "cuda-runtime",
    feature = "hip-runtime",
    feature = "opencl-runtime",
    feature = "level-zero-runtime",
    feature = "metal-runtime"
))]
fn normal_cases() -> Vec<Case> {
    vec![
        Case {
            family: PrecisionTransformFamily::C2cNd,
            shape: vec![17, 34],
            kind: CaseKind::C2c,
        },
        Case {
            family: PrecisionTransformFamily::R2c1d,
            shape: vec![65],
            kind: CaseKind::R2c,
        },
        Case {
            family: PrecisionTransformFamily::R2cNd,
            shape: vec![17, 34],
            kind: CaseKind::R2c,
        },
        Case {
            family: PrecisionTransformFamily::R2r1d,
            shape: vec![257],
            kind: CaseKind::R2rDct2,
        },
        Case {
            family: PrecisionTransformFamily::R2rNd,
            shape: vec![17, 34],
            kind: CaseKind::R2rDct2,
        },
    ]
}

#[cfg(any(
    feature = "vulkan-runtime",
    feature = "cuda-runtime",
    feature = "hip-runtime",
    feature = "opencl-runtime",
    feature = "level-zero-runtime",
    feature = "metal-runtime"
))]
fn cases(paper: bool) -> Vec<Case> {
    let mut cases = normal_cases();
    if paper {
        cases.push(Case {
            family: PrecisionTransformFamily::C2c1d,
            shape: vec![PAPER_C2C_LENGTH],
            kind: CaseKind::C2c,
        });
    }
    cases
}

#[cfg(any(
    feature = "vulkan-runtime",
    feature = "cuda-runtime",
    feature = "hip-runtime",
    feature = "opencl-runtime",
    feature = "level-zero-runtime",
    feature = "metal-runtime"
))]
fn complex_input(shape: &[usize]) -> Vec<Complex32> {
    let len = shape.iter().product::<usize>();
    (0..len)
        .map(|index| {
            let x = index as f32;
            Complex32::new(
                (0.031 * x).sin() + 0.00021 * x,
                (0.047 * x).cos() - 0.00013 * x,
            )
        })
        .collect()
}

#[cfg(any(
    feature = "vulkan-runtime",
    feature = "cuda-runtime",
    feature = "hip-runtime",
    feature = "opencl-runtime",
    feature = "level-zero-runtime",
    feature = "metal-runtime"
))]
fn real_input(case: &Case) -> Vec<f32> {
    let len = case.shape.iter().product::<usize>();
    (0..len)
        .map(|index| {
            let x = index as f32;
            match case.family {
                PrecisionTransformFamily::R2c1d => {
                    (0.083 * x).sin() + 0.23 * (0.029 * x).cos() + 0.0007 * x
                }
                PrecisionTransformFamily::R2cNd => {
                    (0.057 * x).sin() + 0.19 * (0.017 * x).cos() - 0.00031 * x
                }
                PrecisionTransformFamily::R2r1d => {
                    (0.041 * x).sin() + 0.13 * (0.023 * x).cos() + 0.00011 * x
                }
                PrecisionTransformFamily::R2rNd => {
                    (0.037 * x).sin() + 0.17 * (0.019 * x).cos() + 0.00009 * x
                }
                _ => unreachable!("real input requested for a complex-only precision case"),
            }
        })
        .collect()
}

#[cfg(any(
    feature = "vulkan-runtime",
    feature = "cuda-runtime",
    feature = "hip-runtime",
    feature = "opencl-runtime",
    feature = "level-zero-runtime",
    feature = "metal-runtime"
))]
fn config(case: &Case, precision: Precision) -> FftConfig {
    let config = FftConfig::new(case.shape.clone()).with_precision(precision);
    match case.kind {
        CaseKind::C2c => config,
        CaseKind::R2c => config.with_transform(TransformKind::RealToComplex),
        CaseKind::R2rDct2 => config.with_transform(TransformKind::Dct(DctType::II)),
    }
}

#[cfg(any(
    feature = "vulkan-runtime",
    feature = "cuda-runtime",
    feature = "hip-runtime",
    feature = "opencl-runtime",
    feature = "level-zero-runtime"
))]
fn reports(
    backend: Backend,
    device: &str,
    case: &Case,
    f32_metrics: PrecisionMetrics,
    f64_metrics: PrecisionMetrics,
) -> [PrecisionCaseReport; 2] {
    [
        PrecisionCaseReport {
            backend,
            device: device.to_owned(),
            family: case.family,
            shape: case.shape.clone(),
            precision: Precision::F32,
            metrics: f32_metrics,
        },
        PrecisionCaseReport {
            backend,
            device: device.to_owned(),
            family: case.family,
            shape: case.shape.clone(),
            precision: Precision::F64,
            metrics: f64_metrics,
        },
    ]
}

#[cfg(any(
    feature = "vulkan-runtime",
    feature = "cuda-runtime",
    feature = "hip-runtime",
    feature = "opencl-runtime",
    feature = "level-zero-runtime"
))]
fn emit_reports(writer: &mut dyn Write, reports: [PrecisionCaseReport; 2]) -> io::Result<()> {
    for report in reports {
        writeln!(writer, "{}", report.to_json_line())?;
    }
    Ok(())
}

#[cfg(feature = "vulkan-runtime")]
fn run_vulkan(options: &Options, writer: &mut dyn Write) -> Result<(), Box<dyn std::error::Error>> {
    use vkfft_rs::backend::vulkan::runtime::{
        TransformInput32, TransformInput64, TransformOutput32, TransformOutput64,
        VulkanExecutionContext,
    };

    let context = match options.device_index {
        Some(index) => VulkanExecutionContext::new_with_device_index(index)?,
        None => VulkanExecutionContext::new()?,
    };
    let profile = context.device_profile();
    if !profile.supports_f64 {
        return Err("selected Vulkan device does not support F64 precision".into());
    }
    let device = context.device_name();
    for case in cases(options.paper) {
        let (f32_metrics, f64_metrics) = match case.kind {
            CaseKind::C2c => {
                let input32 = complex_input(&case.shape);
                let input64 = input32
                    .iter()
                    .map(|value| Complex64::new(value.re as f64, value.im as f64))
                    .collect::<Vec<_>>();
                let ir32 =
                    TransformIr::build(config(&case, Precision::F32), Direction::Forward, profile)?;
                let ir64 =
                    TransformIr::build(config(&case, Precision::F64), Direction::Forward, profile)?;
                let reference = ir64.execute_complex_reference(&input64)?;
                let actual32 = match context
                    .execute_transform_f32(&ir32, TransformInput32::Complex(&input32))?
                {
                    TransformOutput32::Complex(values) => values,
                    TransformOutput32::Real(_) => return Err("C2C returned real output".into()),
                };
                let actual32 = actual32
                    .iter()
                    .map(|value| Complex64::new(value.re as f64, value.im as f64))
                    .collect::<Vec<_>>();
                let f32_metrics = complex_precision_metrics(&actual32, &reference)?;
                drop(actual32);
                let actual64 = match context
                    .execute_transform_f64(&ir64, TransformInput64::Complex(&input64))?
                {
                    TransformOutput64::Complex(values) => values,
                    TransformOutput64::Real(_) => return Err("F64 C2C returned real output".into()),
                };
                (
                    f32_metrics,
                    complex_precision_metrics(&actual64, &reference)?,
                )
            }
            CaseKind::R2c => {
                let input32 = real_input(&case);
                let input64 = input32
                    .iter()
                    .map(|value| *value as f64)
                    .collect::<Vec<_>>();
                let ir32 =
                    TransformIr::build(config(&case, Precision::F32), Direction::Forward, profile)?;
                let ir64 =
                    TransformIr::build(config(&case, Precision::F64), Direction::Forward, profile)?;
                let reference = ir64.execute_r2c_reference(&input64)?;
                let actual32 =
                    match context.execute_transform_f32(&ir32, TransformInput32::Real(&input32))? {
                        TransformOutput32::Complex(values) => values,
                        TransformOutput32::Real(_) => return Err("R2C returned real output".into()),
                    };
                let actual32 = actual32
                    .iter()
                    .map(|value| Complex64::new(value.re as f64, value.im as f64))
                    .collect::<Vec<_>>();
                let f32_metrics = complex_precision_metrics(&actual32, &reference)?;
                drop(actual32);
                let actual64 = match context
                    .execute_transform_f64(&ir64, TransformInput64::Real(&input64))?
                {
                    TransformOutput64::Complex(values) => values,
                    TransformOutput64::Real(_) => return Err("F64 R2C returned real output".into()),
                };
                (
                    f32_metrics,
                    complex_precision_metrics(&actual64, &reference)?,
                )
            }
            CaseKind::R2rDct2 => {
                let input32 = real_input(&case);
                let input64 = input32
                    .iter()
                    .map(|value| *value as f64)
                    .collect::<Vec<_>>();
                let ir32 =
                    TransformIr::build(config(&case, Precision::F32), Direction::Forward, profile)?;
                let ir64 =
                    TransformIr::build(config(&case, Precision::F64), Direction::Forward, profile)?;
                let reference = ir64.execute_r2r_reference(&input64)?;
                let actual32 =
                    match context.execute_transform_f32(&ir32, TransformInput32::Real(&input32))? {
                        TransformOutput32::Real(values) => values,
                        TransformOutput32::Complex(_) => {
                            return Err("R2R returned complex output".into());
                        }
                    };
                let actual32 = actual32
                    .iter()
                    .map(|value| *value as f64)
                    .collect::<Vec<_>>();
                let f32_metrics = real_precision_metrics(&actual32, &reference)?;
                drop(actual32);
                let actual64 =
                    match context.execute_transform_f64(&ir64, TransformInput64::Real(&input64))? {
                        TransformOutput64::Real(values) => values,
                        TransformOutput64::Complex(_) => {
                            return Err("F64 R2R returned complex output".into());
                        }
                    };
                (f32_metrics, real_precision_metrics(&actual64, &reference)?)
            }
        };
        ensure_finite(case.family, f32_metrics, f64_metrics)?;
        emit_reports(
            writer,
            reports(Backend::Vulkan, &device, &case, f32_metrics, f64_metrics),
        )?;
    }
    Ok(())
}

#[cfg(not(feature = "vulkan-runtime"))]
fn run_vulkan(
    _options: &Options,
    _writer: &mut dyn Write,
) -> Result<(), Box<dyn std::error::Error>> {
    Err("rebuild with --features vulkan-runtime".into())
}

#[cfg(any(
    feature = "cuda-runtime",
    feature = "hip-runtime",
    feature = "opencl-runtime",
    feature = "level-zero-runtime"
))]
fn run_native<R: NativeRuntime>(
    runtime: &R,
    options: &Options,
    writer: &mut dyn Write,
    clear_runtime_caches: impl Fn(&R) -> vkfft_rs::Result<()>,
) -> Result<(), Box<dyn std::error::Error>> {
    let profile = runtime.device_profile();
    if !profile.supports_f64 {
        return Err(format!(
            "selected {:?} device does not support F64 precision",
            runtime.backend()
        )
        .into());
    }
    for case in cases(options.paper) {
        clear_runtime_caches(runtime)?;
        let (f32_metrics, f64_metrics) = match case.kind {
            CaseKind::C2c => {
                let input32 = complex_input(&case.shape);
                let input64 = input32
                    .iter()
                    .map(|value| Complex64::new(value.re as f64, value.im as f64))
                    .collect::<Vec<_>>();
                let ir32 =
                    TransformIr::build(config(&case, Precision::F32), Direction::Forward, profile)?;
                let ir64 =
                    TransformIr::build(config(&case, Precision::F64), Direction::Forward, profile)?;
                let reference = ir64.execute_complex_reference(&input64)?;
                let actual32_raw = match runtime
                    .execute_transform_f32(&ir32, NativeTransformInput32::Complex(&input32))?
                {
                    NativeTransformOutput32::Complex(values) => values,
                    NativeTransformOutput32::Real(_) => {
                        return Err("C2C returned real output".into());
                    }
                };
                let actual32 = actual32_raw
                    .iter()
                    .map(|value| Complex64::new(value.re as f64, value.im as f64))
                    .collect::<Vec<_>>();
                let f32_metrics = complex_precision_metrics(&actual32, &reference)?;
                drop(actual32);
                drop(actual32_raw);
                drop(input32);
                drop(ir32);
                clear_runtime_caches(runtime)?;
                let actual64 = match runtime
                    .execute_transform_f64(&ir64, NativeTransformInput64::Complex(&input64))?
                {
                    NativeTransformOutput64::Complex(values) => values,
                    NativeTransformOutput64::Real(_) => {
                        return Err("F64 C2C returned real output".into());
                    }
                };
                (
                    f32_metrics,
                    complex_precision_metrics(&actual64, &reference)?,
                )
            }
            CaseKind::R2c => {
                let input32 = real_input(&case);
                let input64 = input32
                    .iter()
                    .map(|value| *value as f64)
                    .collect::<Vec<_>>();
                let ir32 =
                    TransformIr::build(config(&case, Precision::F32), Direction::Forward, profile)?;
                let ir64 =
                    TransformIr::build(config(&case, Precision::F64), Direction::Forward, profile)?;
                let reference = ir64.execute_r2c_reference(&input64)?;
                let actual32_raw = match runtime
                    .execute_transform_f32(&ir32, NativeTransformInput32::Real(&input32))?
                {
                    NativeTransformOutput32::Complex(values) => values,
                    NativeTransformOutput32::Real(_) => {
                        return Err("R2C returned real output".into());
                    }
                };
                let actual32 = actual32_raw
                    .iter()
                    .map(|value| Complex64::new(value.re as f64, value.im as f64))
                    .collect::<Vec<_>>();
                let f32_metrics = complex_precision_metrics(&actual32, &reference)?;
                drop(actual32);
                drop(actual32_raw);
                drop(input32);
                drop(ir32);
                clear_runtime_caches(runtime)?;
                let actual64 = match runtime
                    .execute_transform_f64(&ir64, NativeTransformInput64::Real(&input64))?
                {
                    NativeTransformOutput64::Complex(values) => values,
                    NativeTransformOutput64::Real(_) => {
                        return Err("F64 R2C returned real output".into());
                    }
                };
                (
                    f32_metrics,
                    complex_precision_metrics(&actual64, &reference)?,
                )
            }
            CaseKind::R2rDct2 => {
                let input32 = real_input(&case);
                let input64 = input32
                    .iter()
                    .map(|value| *value as f64)
                    .collect::<Vec<_>>();
                let ir32 =
                    TransformIr::build(config(&case, Precision::F32), Direction::Forward, profile)?;
                let ir64 =
                    TransformIr::build(config(&case, Precision::F64), Direction::Forward, profile)?;
                let reference = ir64.execute_r2r_reference(&input64)?;
                let actual32_raw = match runtime
                    .execute_transform_f32(&ir32, NativeTransformInput32::Real(&input32))?
                {
                    NativeTransformOutput32::Real(values) => values,
                    NativeTransformOutput32::Complex(_) => {
                        return Err("R2R returned complex output".into());
                    }
                };
                let actual32 = actual32_raw
                    .iter()
                    .map(|value| *value as f64)
                    .collect::<Vec<_>>();
                let f32_metrics = real_precision_metrics(&actual32, &reference)?;
                drop(actual32);
                drop(actual32_raw);
                drop(input32);
                drop(ir32);
                clear_runtime_caches(runtime)?;
                let actual64 = match runtime
                    .execute_transform_f64(&ir64, NativeTransformInput64::Real(&input64))?
                {
                    NativeTransformOutput64::Real(values) => values,
                    NativeTransformOutput64::Complex(_) => {
                        return Err("F64 R2R returned complex output".into());
                    }
                };
                (f32_metrics, real_precision_metrics(&actual64, &reference)?)
            }
        };
        ensure_finite(case.family, f32_metrics, f64_metrics)?;
        emit_reports(
            writer,
            reports(
                runtime.backend(),
                runtime.device_name(),
                &case,
                f32_metrics,
                f64_metrics,
            ),
        )?;
    }
    Ok(())
}

#[cfg(any(
    feature = "vulkan-runtime",
    feature = "cuda-runtime",
    feature = "hip-runtime",
    feature = "opencl-runtime",
    feature = "level-zero-runtime"
))]
fn ensure_finite(
    family: PrecisionTransformFamily,
    f32_metrics: PrecisionMetrics,
    f64_metrics: PrecisionMetrics,
) -> Result<(), Box<dyn std::error::Error>> {
    if !f32_metrics.is_finite() || !f64_metrics.is_finite() {
        return Err(format!(
            "non-finite precision metrics for {family:?}: f32={f32_metrics:?}, f64={f64_metrics:?}"
        )
        .into());
    }
    Ok(())
}

#[cfg(feature = "cuda-runtime")]
fn run_cuda(options: &Options, writer: &mut dyn Write) -> Result<(), Box<dyn std::error::Error>> {
    let context = vkfft_rs::backend::cuda::runtime::CudaExecutionContext::new(
        options.device_index.unwrap_or(0),
    )?;
    run_native(&context, options, writer, |context| {
        context.clear_runtime_caches()
    })
}

#[cfg(not(feature = "cuda-runtime"))]
fn run_cuda(_options: &Options, _writer: &mut dyn Write) -> Result<(), Box<dyn std::error::Error>> {
    Err("rebuild with --features cuda-runtime".into())
}

#[cfg(feature = "hip-runtime")]
fn run_hip(options: &Options, writer: &mut dyn Write) -> Result<(), Box<dyn std::error::Error>> {
    let context = vkfft_rs::backend::hip::runtime::HipExecutionContext::new(
        options.device_index.unwrap_or(0),
    )?;
    if context.device_profile().supports_f64 {
        run_native(&context, options, writer, |context| {
            context.clear_runtime_caches()
        })
    } else {
        run_native_f32_only(&context, options, writer)
    }
}

#[cfg(not(feature = "hip-runtime"))]
fn run_hip(_options: &Options, _writer: &mut dyn Write) -> Result<(), Box<dyn std::error::Error>> {
    Err("rebuild with --features hip-runtime".into())
}

#[cfg(feature = "opencl-runtime")]
fn run_opencl(options: &Options, writer: &mut dyn Write) -> Result<(), Box<dyn std::error::Error>> {
    let context = vkfft_rs::backend::opencl::runtime::OpenClExecutionContext::new(
        options.device_index.unwrap_or(0),
    )?;
    run_native(&context, options, writer, |context| {
        context.clear_runtime_caches()
    })
}

#[cfg(not(feature = "opencl-runtime"))]
fn run_opencl(
    _options: &Options,
    _writer: &mut dyn Write,
) -> Result<(), Box<dyn std::error::Error>> {
    Err("rebuild with --features opencl-runtime".into())
}

#[cfg(feature = "level-zero-runtime")]
fn run_level_zero(
    options: &Options,
    writer: &mut dyn Write,
) -> Result<(), Box<dyn std::error::Error>> {
    let context = vkfft_rs::backend::level_zero::runtime::LevelZeroExecutionContext::new(
        options.device_index.unwrap_or(0),
    )?;
    if context.device_profile().supports_f64 {
        run_native(&context, options, writer, |context| {
            context.clear_runtime_caches()
        })
    } else {
        run_native_f32_only(&context, options, writer)
    }
}

#[cfg(not(feature = "level-zero-runtime"))]
fn run_level_zero(
    _options: &Options,
    _writer: &mut dyn Write,
) -> Result<(), Box<dyn std::error::Error>> {
    Err("rebuild with --features level-zero-runtime".into())
}

#[cfg(any(
    feature = "hip-runtime",
    feature = "level-zero-runtime",
    feature = "metal-runtime"
))]
fn run_native_f32_only<R: NativeRuntime>(
    runtime: &R,
    options: &Options,
    writer: &mut dyn Write,
) -> Result<(), Box<dyn std::error::Error>> {
    let profile = runtime.device_profile();
    let reference_profile = DeviceProfile::generic(Backend::Vulkan, GpuVendor::Nvidia);
    for case in cases(options.paper) {
        let f32_metrics = match case.kind {
            CaseKind::C2c => {
                let input32 = complex_input(&case.shape);
                let input64 = input32
                    .iter()
                    .map(|value| Complex64::new(value.re as f64, value.im as f64))
                    .collect::<Vec<_>>();
                let ir32 =
                    TransformIr::build(config(&case, Precision::F32), Direction::Forward, profile)?;
                let reference_ir = TransformIr::build(
                    config(&case, Precision::F64),
                    Direction::Forward,
                    reference_profile,
                )?;
                let reference = reference_ir.execute_complex_reference(&input64)?;
                let actual32 = match runtime
                    .execute_transform_f32(&ir32, NativeTransformInput32::Complex(&input32))?
                {
                    NativeTransformOutput32::Complex(values) => values,
                    NativeTransformOutput32::Real(_) => {
                        return Err("C2C returned real output".into());
                    }
                };
                let actual64 = actual32
                    .iter()
                    .map(|value| Complex64::new(value.re as f64, value.im as f64))
                    .collect::<Vec<_>>();
                complex_precision_metrics(&actual64, &reference)?
            }
            CaseKind::R2c => {
                let input32 = real_input(&case);
                let input64 = input32
                    .iter()
                    .map(|value| *value as f64)
                    .collect::<Vec<_>>();
                let ir32 =
                    TransformIr::build(config(&case, Precision::F32), Direction::Forward, profile)?;
                let reference_ir = TransformIr::build(
                    config(&case, Precision::F64),
                    Direction::Forward,
                    reference_profile,
                )?;
                let reference = reference_ir.execute_r2c_reference(&input64)?;
                let actual32 = match runtime
                    .execute_transform_f32(&ir32, NativeTransformInput32::Real(&input32))?
                {
                    NativeTransformOutput32::Complex(values) => values,
                    NativeTransformOutput32::Real(_) => {
                        return Err("R2C returned real output".into());
                    }
                };
                let actual64 = actual32
                    .iter()
                    .map(|value| Complex64::new(value.re as f64, value.im as f64))
                    .collect::<Vec<_>>();
                complex_precision_metrics(&actual64, &reference)?
            }
            CaseKind::R2rDct2 => {
                let input32 = real_input(&case);
                let input64 = input32
                    .iter()
                    .map(|value| *value as f64)
                    .collect::<Vec<_>>();
                let ir32 =
                    TransformIr::build(config(&case, Precision::F32), Direction::Forward, profile)?;
                let reference_ir = TransformIr::build(
                    config(&case, Precision::F64),
                    Direction::Forward,
                    reference_profile,
                )?;
                let reference = reference_ir.execute_r2r_reference(&input64)?;
                let actual32 = match runtime
                    .execute_transform_f32(&ir32, NativeTransformInput32::Real(&input32))?
                {
                    NativeTransformOutput32::Real(values) => values,
                    NativeTransformOutput32::Complex(_) => {
                        return Err("R2R returned complex output".into());
                    }
                };
                let actual64 = actual32
                    .iter()
                    .map(|value| *value as f64)
                    .collect::<Vec<_>>();
                real_precision_metrics(&actual64, &reference)?
            }
        };
        if !f32_metrics.is_finite() {
            return Err(format!(
                "non-finite {:?} F32 precision metrics for {:?}: {f32_metrics:?}",
                runtime.backend(),
                case.family
            )
            .into());
        }
        writeln!(
            writer,
            "{}",
            PrecisionCaseReport {
                backend: runtime.backend(),
                device: runtime.device_name().to_owned(),
                family: case.family,
                shape: case.shape,
                precision: Precision::F32,
                metrics: f32_metrics,
            }
            .to_json_line()
        )?;
    }
    Ok(())
}

#[cfg(feature = "metal-runtime")]
fn run_metal(options: &Options, writer: &mut dyn Write) -> Result<(), Box<dyn std::error::Error>> {
    let context = vkfft_rs::backend::metal::runtime::MetalExecutionContext::new(
        options.device_index.unwrap_or(0),
    )?;
    run_native_f32_only(&context, options, writer)
}

#[cfg(not(feature = "metal-runtime"))]
fn run_metal(
    _options: &Options,
    _writer: &mut dyn Write,
) -> Result<(), Box<dyn std::error::Error>> {
    Err("rebuild with --features metal-runtime".into())
}
