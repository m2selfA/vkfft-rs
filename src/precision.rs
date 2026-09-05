//! Precision-reporting helpers shared by correctness sweeps and benchmark tooling.
//!
//! VkFFT's sample 11 reports four FFTW-comparison metrics: average/max absolute
//! difference and average/max per-bin relative epsilon. This module keeps those
//! definitions backend-neutral and provides a stable JSON-lines record so real-GPU
//! precision matrices can be persisted without adding a serialization dependency.

use crate::complex::Complex64;
use crate::config::{Backend, Precision};
use crate::error::{Result, VkFftError};

pub const PRECISION_REPORT_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PrecisionMetrics {
    pub avg_difference: f64,
    pub max_difference: f64,
    pub avg_eps: f64,
    pub max_eps: f64,
}

impl PrecisionMetrics {
    pub fn is_finite(self) -> bool {
        self.avg_difference.is_finite()
            && self.max_difference.is_finite()
            && self.avg_eps.is_finite()
            && self.max_eps.is_finite()
    }
}

pub fn complex_precision_metrics(
    actual: &[Complex64],
    reference: &[Complex64],
) -> Result<PrecisionMetrics> {
    precision_metrics_by(actual, reference, |actual, reference| {
        let difference = (*actual - *reference).norm_sqr().sqrt();
        let reference_norm = reference.norm_sqr().sqrt();
        (difference, reference_norm)
    })
}

pub fn real_precision_metrics(actual: &[f64], reference: &[f64]) -> Result<PrecisionMetrics> {
    precision_metrics_by(actual, reference, |actual, reference| {
        ((actual - reference).abs(), reference.abs())
    })
}

fn precision_metrics_by<T, F>(
    actual: &[T],
    reference: &[T],
    mut sample: F,
) -> Result<PrecisionMetrics>
where
    F: FnMut(&T, &T) -> (f64, f64),
{
    if actual.len() != reference.len() {
        return Err(VkFftError::InputLengthMismatch {
            expected: reference.len(),
            actual: actual.len(),
        });
    }
    if actual.is_empty() {
        return Err(VkFftError::InvalidKernelIr(
            "precision metrics require at least one sample",
        ));
    }

    let mut difference_sum = 0.0;
    let mut max_difference = 0.0_f64;
    let mut eps_sum = 0.0;
    let mut max_eps = 0.0_f64;
    for (actual, reference) in actual.iter().zip(reference) {
        let (difference, reference_norm) = sample(actual, reference);
        let eps = if reference_norm > f64::MIN_POSITIVE {
            difference / reference_norm
        } else {
            difference
        };
        difference_sum += difference;
        max_difference = max_difference.max(difference);
        eps_sum += eps;
        max_eps = max_eps.max(eps);
    }
    let count = actual.len() as f64;
    Ok(PrecisionMetrics {
        avg_difference: difference_sum / count,
        max_difference,
        avg_eps: eps_sum / count,
        max_eps,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrecisionTransformFamily {
    C2c1d,
    C2cNd,
    R2c1d,
    R2cNd,
    R2r1d,
    R2rNd,
}

impl PrecisionTransformFamily {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::C2c1d => "c2c-1d",
            Self::C2cNd => "c2c-nd",
            Self::R2c1d => "r2c-1d",
            Self::R2cNd => "r2c-nd",
            Self::R2r1d => "r2r-1d",
            Self::R2rNd => "r2r-nd",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct PrecisionCaseReport {
    pub backend: Backend,
    pub device: String,
    pub family: PrecisionTransformFamily,
    pub shape: Vec<usize>,
    pub precision: Precision,
    pub metrics: PrecisionMetrics,
}

impl PrecisionCaseReport {
    pub fn to_json_line(&self) -> String {
        let mut output = String::new();
        output.push('{');
        push_json_u64_field(
            &mut output,
            "schema_version",
            PRECISION_REPORT_SCHEMA_VERSION as u64,
            false,
        );
        push_json_string_field(
            &mut output,
            "upstream_vkfft_commit",
            crate::UPSTREAM_VKFFT_COMMIT,
            true,
        );
        push_json_string_field(
            &mut output,
            "vkfft_rs_version",
            env!("CARGO_PKG_VERSION"),
            true,
        );
        push_json_string_field(&mut output, "backend", &format!("{:?}", self.backend), true);
        push_json_string_field(&mut output, "device", &self.device, true);
        push_json_string_field(&mut output, "family", self.family.as_str(), true);
        output.push_str(",\"shape\":[");
        for (index, dimension) in self.shape.iter().enumerate() {
            if index != 0 {
                output.push(',');
            }
            output.push_str(&dimension.to_string());
        }
        output.push(']');
        push_json_string_field(
            &mut output,
            "precision",
            &format!("{:?}", self.precision),
            true,
        );
        push_json_f64_field(
            &mut output,
            "avg_difference",
            self.metrics.avg_difference,
            true,
        );
        push_json_f64_field(
            &mut output,
            "max_difference",
            self.metrics.max_difference,
            true,
        );
        push_json_f64_field(&mut output, "avg_eps", self.metrics.avg_eps, true);
        push_json_f64_field(&mut output, "max_eps", self.metrics.max_eps, true);
        output.push('}');
        output
    }
}

fn push_json_u64_field(output: &mut String, name: &str, value: u64, comma: bool) {
    if comma {
        output.push(',');
    }
    output.push('"');
    output.push_str(name);
    output.push_str("\":");
    output.push_str(&value.to_string());
}

fn push_json_f64_field(output: &mut String, name: &str, value: f64, comma: bool) {
    if comma {
        output.push(',');
    }
    output.push('"');
    output.push_str(name);
    output.push_str("\":");
    if value.is_finite() {
        output.push_str(&format!("{value:.17e}"));
    } else {
        output.push_str("null");
    }
}

fn push_json_string_field(output: &mut String, name: &str, value: &str, comma: bool) {
    if comma {
        output.push(',');
    }
    output.push('"');
    output.push_str(name);
    output.push_str("\":\"");
    push_json_escaped(output, value);
    output.push('"');
}

fn push_json_escaped(output: &mut String, value: &str) {
    for ch in value.chars() {
        match ch {
            '"' => output.push_str("\\\""),
            '\\' => output.push_str("\\\\"),
            '\n' => output.push_str("\\n"),
            '\r' => output.push_str("\\r"),
            '\t' => output.push_str("\\t"),
            ch if ch <= '\u{1f}' => {
                output.push_str(&format!("\\u{:04x}", ch as u32));
            }
            ch => output.push(ch),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn complex_and_real_metrics_match_known_values() {
        let complex_reference = [Complex64::new(3.0, 4.0), Complex64::new(0.0, 0.0)];
        let complex_actual = [Complex64::new(3.0, 5.0), Complex64::new(0.0, 2.0)];
        let metrics = complex_precision_metrics(&complex_actual, &complex_reference).unwrap();
        assert_eq!(metrics.avg_difference, 1.5);
        assert_eq!(metrics.max_difference, 2.0);
        assert_eq!(metrics.avg_eps, 1.1);
        assert_eq!(metrics.max_eps, 2.0);

        let metrics = real_precision_metrics(&[12.0, 1.0], &[10.0, 0.0]).unwrap();
        assert_eq!(metrics.avg_difference, 1.5);
        assert_eq!(metrics.max_difference, 2.0);
        assert_eq!(metrics.avg_eps, 0.6);
        assert_eq!(metrics.max_eps, 1.0);
    }

    #[test]
    fn metrics_reject_empty_or_mismatched_inputs() {
        assert!(complex_precision_metrics(&[], &[]).is_err());
        assert!(real_precision_metrics(&[1.0], &[]).is_err());
    }

    #[test]
    fn precision_report_is_stable_jsonl_and_escapes_device_name() {
        let report = PrecisionCaseReport {
            backend: Backend::Cuda,
            device: "GPU \\\"A\\\"\n0".to_owned(),
            family: PrecisionTransformFamily::C2cNd,
            shape: vec![17, 34],
            precision: Precision::F64,
            metrics: PrecisionMetrics {
                avg_difference: 1.25e-12,
                max_difference: 2.5e-12,
                avg_eps: 3.75e-13,
                max_eps: 4.5e-13,
            },
        };
        let line = report.to_json_line();
        assert!(line.starts_with("{\"schema_version\":1,\"upstream_vkfft_commit\":"));
        assert!(line.contains(&format!(
            "\"upstream_vkfft_commit\":\"{}\"",
            crate::UPSTREAM_VKFFT_COMMIT
        )));
        assert!(line.contains("\"vkfft_rs_version\":\"0.1.0\""));
        assert!(line.contains("\"backend\":\"Cuda\""));
        assert!(line.contains("\"device\":\"GPU \\\\\\\"A\\\\\\\"\\n0\""));
        assert!(line.contains("\"family\":\"c2c-nd\""));
        assert!(line.contains("\"shape\":[17,34]"));
        assert!(line.contains("\"precision\":\"F64\""));
        assert!(!line.ends_with('\n'));
    }
}
