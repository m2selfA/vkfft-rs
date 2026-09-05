use std::env;
use std::fs::{self, File};
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};

use vkfft_rs::{
    Backend, DeviceProfile, GpuVendor, Precision, SchedulerAxisClassification,
    SchedulerPhysicalAxisProbeContext, SchedulerRaderUploadProbe, SchedulerSnapshotRecord,
    compare_upstream_scheduler_snapshot_jsonl, fixed_upstream_scheduler_snapshot_corpus,
    scheduler_axis_classification_probe,
    scheduler_rader_upload_axis_block_probe_records_with_context,
    scheduler_rader_upload_probe_record,
};

struct LoadedRaderUploadProbe {
    probe: SchedulerRaderUploadProbe,
    physical_axis: SchedulerPhysicalAxisProbeContext,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut output = None;
    let mut check = None;
    let mut case_filter = None;
    let mut rader_upload_file = None;
    let mut classification_file = None;
    let mut axis_block_file = None;
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--output" => {
                output = Some(PathBuf::from(
                    args.next().ok_or("--output requires a path")?,
                ));
            }
            "--check" => {
                check = Some(PathBuf::from(args.next().ok_or("--check requires a path")?));
            }
            "--case" => case_filter = Some(args.next().ok_or("--case requires a value")?),
            "--rader-upload-file" => {
                rader_upload_file = Some(PathBuf::from(
                    args.next().ok_or("--rader-upload-file requires a path")?,
                ));
            }
            "--classification-file" => {
                classification_file = Some(PathBuf::from(
                    args.next().ok_or("--classification-file requires a path")?,
                ));
            }
            "--axis-block-file" => {
                axis_block_file = Some(PathBuf::from(
                    args.next().ok_or("--axis-block-file requires a path")?,
                ));
            }
            "-h" | "--help" => {
                print_help();
                return Ok(());
            }
            other => return Err(format!("unknown argument `{other}`; use --help").into()),
        }
    }
    if usize::from(rader_upload_file.is_some())
        + usize::from(classification_file.is_some())
        + usize::from(axis_block_file.is_some())
        > 1
    {
        return Err(
            "--rader-upload-file, --classification-file, and --axis-block-file are mutually exclusive"
                .into(),
        );
    }

    if let Some(path) = classification_file {
        if check.is_some() {
            return Err("--check is only valid for JSONL scheduler snapshot modes".into());
        }
        let probes = load_rader_upload_probes(&path)?;
        let lines = probes
            .iter()
            .filter(|loaded| {
                case_filter
                    .as_ref()
                    .is_none_or(|filter| loaded.probe.case_name == *filter)
            })
            .map(|loaded| render_classification_line(&loaded.probe))
            .collect::<Result<Vec<_>, _>>()?;
        if lines.is_empty() {
            return Err("requested scheduler classification case was not found".into());
        }
        return write_lines(output, &lines);
    }

    if let Some(path) = axis_block_file {
        let probes = load_rader_upload_probes(&path)?;
        let records = probes
            .iter()
            .filter(|loaded| {
                case_filter
                    .as_ref()
                    .is_none_or(|filter| loaded.probe.case_name == *filter)
            })
            .map(|loaded| {
                scheduler_rader_upload_axis_block_probe_records_with_context(
                    &loaded.probe,
                    loaded.physical_axis,
                )
                .map_err(|error| {
                    format!(
                        "failed to build Rust physical probe `{}`: {error}",
                        loaded.probe.case_name
                    )
                })
            })
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
        if records.is_empty() {
            return Err("requested scheduler physical axis-block case was not found".into());
        }
        if let Some(path) = check {
            let reference = fs::read_to_string(&path)?;
            compare_upstream_scheduler_snapshot_jsonl(&reference, &records).map_err(|error| {
                format!(
                    "scheduler axis-block candidate differs from upstream reference {}: {error}",
                    path.display()
                )
            })?;
        }
        let lines = records
            .iter()
            .map(SchedulerSnapshotRecord::to_json_line)
            .collect::<Vec<_>>();
        return write_lines(output, &lines);
    }

    let records = match rader_upload_file {
        Some(path) => load_rader_upload_probes(&path)?
            .iter()
            .map(|loaded| {
                scheduler_rader_upload_probe_record(&loaded.probe).map_err(|error| {
                    format!(
                        "failed to build Rust probe `{}`: {error}",
                        loaded.probe.case_name
                    )
                })
            })
            .collect::<Result<Vec<_>, _>>()?,
        None => fixed_upstream_scheduler_snapshot_corpus()?,
    }
    .into_iter()
    .filter(|record| {
        case_filter
            .as_ref()
            .is_none_or(|filter| record.case_name == *filter)
    })
    .collect::<Vec<_>>();
    if records.is_empty() {
        return Err("requested scheduler snapshot case was not found".into());
    }

    if let Some(path) = check {
        let reference = fs::read_to_string(&path)?;
        compare_upstream_scheduler_snapshot_jsonl(&reference, &records).map_err(|error| {
            format!(
                "scheduler candidate differs from upstream reference {}: {error}",
                path.display()
            )
        })?;
    }

    let lines = records
        .iter()
        .map(SchedulerSnapshotRecord::to_json_line)
        .collect::<Vec<_>>();
    write_lines(output, &lines)
}

fn write_lines(
    output: Option<PathBuf>,
    lines: &[String],
) -> Result<(), Box<dyn std::error::Error>> {
    let mut writer: Box<dyn Write> = match output {
        Some(path) => Box::new(BufWriter::new(File::create(path)?)),
        None => Box::new(BufWriter::new(io::stdout().lock())),
    };
    for line in lines {
        writeln!(writer, "{line}")?;
    }
    writer.flush()?;
    Ok(())
}

fn render_classification_line(
    probe: &SchedulerRaderUploadProbe,
) -> Result<String, Box<dyn std::error::Error>> {
    let classification = scheduler_axis_classification_probe(probe)?;
    let line = match classification {
        SchedulerAxisClassification::Stockham => {
            format!(
                "{}\tstockham\t{}\t-\t-",
                probe.case_name, probe.sequence_len
            )
        }
        SchedulerAxisClassification::Bluestein { padded_len } => {
            format!("{}\tbluestein\t{padded_len}\t-\t-", probe.case_name)
        }
        SchedulerAxisClassification::Rader {
            direct_primes,
            fft_primes,
        } => format!(
            "{}\trader\t{}\t{}\t{}",
            probe.case_name,
            probe.sequence_len,
            comma_usize(&direct_primes),
            comma_usize(&fft_primes)
        ),
    };
    Ok(line)
}

fn comma_usize(values: &[usize]) -> String {
    if values.is_empty() {
        return "-".to_owned();
    }
    values
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(",")
}

fn load_rader_upload_probes(
    path: &Path,
) -> Result<Vec<LoadedRaderUploadProbe>, Box<dyn std::error::Error>> {
    let source = fs::read_to_string(path)?;
    let mut probes = Vec::new();
    for (line_index, raw_line) in source.lines().enumerate() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let fields = line.split('\t').collect::<Vec<_>>();
        if !matches!(fields.len(), 15 | 17 | 18 | 21) {
            return Err(format!(
                "{}:{}: expected 15, 17, 18, or 21 tab-separated fields, got {}",
                path.display(),
                line_index + 1,
                fields.len()
            )
            .into());
        }
        let precision = parse_precision(fields[2])?;
        let mut device = parse_profile(fields[1])?;
        device.shared_memory_bytes = parse_usize(fields[4], "shared_memory_bytes")?;
        device.shared_memory_pow2_bytes = parse_usize(fields[5], "shared_memory_pow2_bytes")?;
        device.max_threads_per_block = parse_usize(fields[6], "max_threads_per_block")?;
        let max_workgroup = parse_usize(fields[7], "max_workgroup")?;
        device.max_workgroup_size = [max_workgroup, max_workgroup, 64];
        if matches!(precision, Precision::F16StorageF32Compute) {
            device.coalesced_memory_bytes *= 2;
        }
        let strided_axis = parse_bool01(fields[8], "strided_axis")?;
        let fastest_axis_len = parse_usize(fields[9], "fastest_axis_len")?;
        if strided_axis && fastest_axis_len == 0 {
            return Err(format!(
                "{}:{}: strided probe requires a non-zero fastest_axis_len",
                path.display(),
                line_index + 1
            )
            .into());
        }
        let (
            batch_count,
            grouped_batch_override,
            axis1_grouped_batch_override,
            perform_zero_padding,
            upstream_axis_id,
            middle_axis_len,
            tuning_offset,
        ) = if matches!(fields.len(), 17 | 18 | 21) {
            let batch_count = parse_usize(fields[11], "batch_count")?;
            if batch_count == 0 {
                return Err(format!(
                    "{}:{}: batch_count must be non-zero",
                    path.display(),
                    line_index + 1
                )
                .into());
            }
            let grouped_batch = parse_usize(fields[12], "grouped_batch_override")?;
            let perform_zero_padding = if matches!(fields.len(), 18 | 21) {
                parse_bool01(fields[13], "zero_padding")?
            } else {
                false
            };
            if fields.len() == 21 {
                let upstream_axis_id = parse_usize(fields[14], "upstream_axis_id")?;
                let middle_axis_len = parse_usize(fields[15], "middle_axis_len")?;
                let axis1_grouped = parse_usize(fields[16], "axis1_grouped_batch_override")?;
                if !strided_axis || upstream_axis_id != 2 || middle_axis_len == 0 {
                    return Err(format!(
                        "{}:{}: 21-field probe requires strided upstream_axis_id=2 and middle_axis_len > 0",
                        path.display(),
                        line_index + 1
                    )
                    .into());
                }
                (
                    batch_count,
                    (grouped_batch != 0).then_some(grouped_batch),
                    (axis1_grouped != 0).then_some(axis1_grouped),
                    perform_zero_padding,
                    upstream_axis_id,
                    middle_axis_len,
                    17,
                )
            } else {
                let grouped_batch_override = (grouped_batch != 0).then_some(grouped_batch);
                (
                    batch_count,
                    grouped_batch_override,
                    if strided_axis {
                        grouped_batch_override
                    } else {
                        None
                    },
                    perform_zero_padding,
                    usize::from(strided_axis),
                    1,
                    if fields.len() == 18 { 14 } else { 13 },
                )
            }
        } else {
            (1, None, None, false, usize::from(strided_axis), 1, 11)
        };
        probes.push(LoadedRaderUploadProbe {
            probe: SchedulerRaderUploadProbe {
                case_name: fields[0].to_owned(),
                sequence_len: parse_usize(fields[3], "sequence_len")?,
                precision,
                device,
                strided_axis,
                fastest_axis_len,
                bandwidth_boost: parse_usize(fields[10], "bandwidth_boost")?,
                batch_count,
                grouped_batch_override,
                perform_zero_padding,
                min_rader_direct_prime: parse_usize(
                    fields[tuning_offset],
                    "min_rader_direct_prime",
                )?,
                max_rader_direct_prime: parse_usize(
                    fields[tuning_offset + 1],
                    "max_rader_direct_prime",
                )?,
                min_rader_fft_prime: parse_usize(fields[tuning_offset + 2], "min_rader_fft_prime")?,
                max_rader_fft_prime: parse_usize(fields[tuning_offset + 3], "max_rader_fft_prime")?,
            },
            physical_axis: SchedulerPhysicalAxisProbeContext {
                upstream_axis_id,
                middle_axis_len,
                axis1_grouped_batch_override,
            },
        });
    }
    if probes.is_empty() {
        return Err(format!("{} contains no scheduler probes", path.display()).into());
    }
    Ok(probes)
}

fn parse_profile(value: &str) -> Result<DeviceProfile, Box<dyn std::error::Error>> {
    let (backend, vendor, coalesced_memory_bytes) = match value {
        "nv-vk" => (Backend::Vulkan, GpuVendor::Nvidia, 32),
        "amd-vk" => (Backend::Vulkan, GpuVendor::Amd, 32),
        "intel-cl" => (Backend::OpenCl, GpuVendor::Intel, 64),
        "intel-vk" => (Backend::Vulkan, GpuVendor::Intel, 64),
        "intel-l0" => (Backend::LevelZero, GpuVendor::Intel, 64),
        other => return Err(format!("unsupported scheduler sweep profile `{other}`").into()),
    };
    let mut device = DeviceProfile::generic(backend, vendor);
    device.coalesced_memory_bytes = coalesced_memory_bytes;
    device.shared_banks = 32;
    device.supports_f64 = true;
    Ok(device)
}

fn parse_precision(value: &str) -> Result<Precision, Box<dyn std::error::Error>> {
    match value {
        "f32" => Ok(Precision::F32),
        "f16" => Ok(Precision::F16StorageF32Compute),
        "f64f32" => Ok(Precision::F64ComputeF32Storage),
        "dd" => Ok(Precision::DoubleDouble),
        other => Err(format!("unsupported scheduler sweep precision `{other}`").into()),
    }
}

fn parse_usize(value: &str, field: &str) -> Result<usize, Box<dyn std::error::Error>> {
    value
        .parse::<usize>()
        .map_err(|error| format!("invalid {field} `{value}`: {error}").into())
}

fn parse_bool01(value: &str, field: &str) -> Result<bool, Box<dyn std::error::Error>> {
    match value {
        "0" => Ok(false),
        "1" => Ok(true),
        other => Err(format!("invalid {field} `{other}`; expected 0 or 1").into()),
    }
}

fn print_help() {
    println!(
        "vkfft-rs scheduler snapshot report\n\
         \n\
         Usage:\n\
           cargo run --example scheduler_report -- [options]\n\
         \n\
         Options:\n\
           --case <name>                Emit/check one selected case\n\
           --rader-upload-file <path>   Build parameterized Rader-upload JSONL from TSV\n\
           --classification-file <path> Build parameterized algorithm classification TSV\n\
           --output <path>              Write report instead of stdout\n\
           --check <path>               Compare JSONL candidates with upstream reference\n\
           -h, --help                   Show this help"
    );
}
