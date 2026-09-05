use crate::config::{
    DctType, DeviceProfile, DstType, FftConfig, PlannerTuning, Precision, TransformKind,
    has_fixed_upstream_gpu_scheduler_profile, upstream_coalesced_memory_bytes_for_precision,
    upstream_effective_rader_tuning,
};
use crate::error::{Result, VkFftError};
use crate::scheduler::upstream_bluestein_auto_padding;

/// Prime radices handled directly by VkFFT's Stockham path.
pub const STOCKHAM_PRIMES: [usize; 6] = [2, 3, 5, 7, 11, 13];
const DOUBLE_DOUBLE_STOCKHAM_PRIMES: [usize; 4] = [2, 3, 5, 7];

/// Composite kernels explicitly called out by the VkFFT paper as merged radix kernels.
pub const MERGED_RADICES: [usize; 10] = [32, 16, 15, 14, 12, 10, 9, 8, 6, 4];

/// Small-prime set used by the current upstream Bluestein padding search.
pub const BLUESTEIN_SMOOTH_PRIMES: [usize; 4] = [2, 3, 5, 7];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlgorithmKind {
    Stockham,
    Rader,
    Bluestein,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RadixPlan {
    pub prime_factors: Vec<usize>,
    pub merged_radices: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RaderMode {
    FftConvolution { convolution: RadixPlan },
    DirectMultiplication,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RaderPrimePlan {
    pub prime: usize,
    pub multiplicity: usize,
    pub generator: usize,
    pub mode: RaderMode,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AxisAlgorithm {
    Stockham {
        radix: RadixPlan,
    },
    Rader {
        stockham: RadixPlan,
        primes: Vec<RaderPrimePlan>,
    },
    Bluestein {
        convolution_len: usize,
        convolution: RadixPlan,
    },
}

impl AxisAlgorithm {
    pub const fn kind(&self) -> AlgorithmKind {
        match self {
            Self::Stockham { .. } => AlgorithmKind::Stockham,
            Self::Rader { .. } => AlgorithmKind::Rader,
            Self::Bluestein { .. } => AlgorithmKind::Bluestein,
        }
    }
}

/// Resource-scoring class for a C2C axis after mapping this crate's row-major
/// dimension order onto VkFFT's `nonStridedAxisId` convention. In this crate the
/// last tensor dimension is naturally contiguous; extracted 1D children must
/// therefore preserve the parent axis class explicitly instead of becoming
/// contiguous merely because their temporary config has one dimension.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum C2cDeviceAxisClass {
    Contiguous,
    Strided,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AxisPlan {
    pub axis: usize,
    pub logical_len: usize,
    pub effective_fft_len: usize,
    pub algorithm: AxisAlgorithm,
    /// Conservative complex-element scratch estimate for algorithm-owned tables/buffers.
    pub estimated_scratch_complex: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FftPlan {
    pub config: FftConfig,
    pub axes: Vec<AxisPlan>,
    /// Preserve the physical parent-axis class for extracted one-dimensional C2C
    /// children so lower upload scheduling does not accidentally treat a strided
    /// tensor axis as contiguous merely because its temporary config is 1D.
    pub(crate) c2c_device_axis_class_override: Option<C2cDeviceAxisClass>,
    /// Preserve upstream `useBluesteinFFT[axis]` into the convolution Stockham
    /// scheduler. This changes unit-stride first-upload capacity selection and
    /// enables the automatic strided bandwidth-boost specialization.
    pub(crate) c2c_device_use_bluestein_fft_override: bool,
}

impl FftPlan {
    pub fn build(config: FftConfig) -> Result<Self> {
        Self::build_inner(config, None, None)
    }

    /// Device-scored high-level planning entry point. Default configurations resolve
    /// fixed vendor/precision tuning here; an explicit `with_tuning` override remains
    /// untouched because it disables device-default resolution. Low-level callers that
    /// need deterministic portable classification continue to use `build`.
    pub fn build_for_device(config: FftConfig, device: DeviceProfile) -> Result<Self> {
        let config = config.resolve_tuning_for_device(device);
        Self::build_inner(config, Some(device), None)
    }

    /// Build an extracted one-dimensional C2C child while preserving whether its
    /// parent tensor axis was naturally contiguous or strided.
    pub(crate) fn build_c2c_child_for_device(
        config: FftConfig,
        device: DeviceProfile,
        axis_class: C2cDeviceAxisClass,
    ) -> Result<Self> {
        if !matches!(config.transform, TransformKind::ComplexToComplex)
            || config.dimensions.len() != 1
        {
            return Err(VkFftError::UnsupportedKernelPath(
                "device-scored C2C child planning requires a one-dimensional C2C config",
            ));
        }
        let config = config.resolve_tuning_for_device(device);
        Self::build_inner(config, Some(device), Some(axis_class))
    }

    pub(crate) fn build_c2c_bluestein_child(
        config: FftConfig,
        axis_class: C2cDeviceAxisClass,
    ) -> Result<Self> {
        if config.dimensions.len() != 1 || config.transform != TransformKind::ComplexToComplex {
            return Err(VkFftError::UnsupportedKernelPath(
                "Bluestein convolution child planning requires a one-dimensional C2C config",
            ));
        }
        let mut plan = Self::build_inner(config, None, Some(axis_class))?;
        plan.c2c_device_use_bluestein_fft_override = true;
        Ok(plan)
    }

    pub(crate) fn build_c2c_bluestein_child_for_device(
        config: FftConfig,
        device: DeviceProfile,
        axis_class: C2cDeviceAxisClass,
    ) -> Result<Self> {
        if config.dimensions.len() != 1 || config.transform != TransformKind::ComplexToComplex {
            return Err(VkFftError::UnsupportedKernelPath(
                "device-scored Bluestein convolution child planning requires a one-dimensional C2C config",
            ));
        }
        let config = config.resolve_tuning_for_device(device);
        let mut plan = Self::build_inner(config, Some(device), Some(axis_class))?;
        plan.c2c_device_use_bluestein_fft_override = true;
        Ok(plan)
    }

    fn build_inner(
        config: FftConfig,
        device: Option<DeviceProfile>,
        c2c_axis_class_override: Option<C2cDeviceAxisClass>,
    ) -> Result<Self> {
        config.validate()?;
        if config.perform_convolution {
            return Err(VkFftError::UnsupportedKernelPath(
                "performConvolution is an application pipeline; build ConvolutionIr instead of a plain FftPlan",
            ));
        }
        let execution_batch_count = config.kernel_preparation_system_count()?;
        let mut axes = Vec::with_capacity(config.dimensions.len());

        for (axis, &logical_len) in config.dimensions.iter().enumerate() {
            let omitted = config.axis_is_omitted(axis);
            let effective_fft_len = if omitted {
                logical_len
            } else {
                effective_axis_len(config.transform, logical_len, axis)?
            };
            let algorithm = if omitted {
                // Omitted axes retain stable full-dimensional AxisPlan indexing but never
                // enter executable IR. Use the portable logical-length classifier as a
                // harmless placeholder rather than applying transform-family shape rewrites.
                plan_axis_algorithm(logical_len, config.tuning)?
            } else if matches!(config.transform, TransformKind::ComplexToComplex) {
                if let Some(device) = device {
                    if has_fixed_upstream_gpu_scheduler_profile(device) {
                        let axis_class = c2c_axis_class_override.unwrap_or_else(|| {
                            if axis + 1 == config.dimensions.len() {
                                C2cDeviceAxisClass::Contiguous
                            } else {
                                C2cDeviceAxisClass::Strided
                            }
                        });
                        plan_c2c_axis_algorithm_for_device(
                            effective_fft_len,
                            execution_batch_count,
                            config.tuning,
                            config.precision,
                            axis_class,
                            device,
                        )?
                    } else {
                        plan_axis_algorithm(effective_fft_len, config.tuning)?
                    }
                } else {
                    plan_axis_algorithm(effective_fft_len, config.tuning)?
                }
            } else {
                plan_axis_algorithm(effective_fft_len, config.tuning)?
            };
            let estimated_scratch_complex = if omitted {
                0
            } else {
                scratch_estimate(&algorithm)?
            };
            axes.push(AxisPlan {
                axis,
                logical_len,
                effective_fft_len,
                algorithm,
                estimated_scratch_complex,
            });
        }

        Ok(Self {
            config,
            axes,
            c2c_device_axis_class_override: c2c_axis_class_override,
            c2c_device_use_bluestein_fft_override: false,
        })
    }
}

fn effective_axis_len(transform: TransformKind, len: usize, axis: usize) -> Result<usize> {
    match transform {
        TransformKind::Dct(DctType::I) => len
            .checked_mul(2)
            .and_then(|value| value.checked_sub(2))
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "DCT-I effective FFT length",
            }),
        TransformKind::Dst(DstType::I) => len
            .checked_mul(2)
            .and_then(|value| value.checked_add(2))
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "DST-I effective FFT length",
            }),
        TransformKind::Dct(DctType::IV) | TransformKind::Dst(DstType::IV)
            if len.is_multiple_of(2) =>
        {
            let result = len / 2;
            if result == 0 {
                Err(VkFftError::InvalidTransformLength {
                    axis,
                    transform: "DCT/DST-IV",
                    length: len,
                })
            } else {
                Ok(result)
            }
        }
        _ => Ok(len),
    }
}

pub fn plan_axis_algorithm(len: usize, tuning: PlannerTuning) -> Result<AxisAlgorithm> {
    plan_axis_algorithm_with_rader_selector(len, tuning, &STOCKHAM_PRIMES, |prime| {
        Ok(rader_mode(prime, tuning))
    })
}

fn plan_axis_algorithm_with_rader_selector<F>(
    len: usize,
    tuning: PlannerTuning,
    stockham_primes: &[usize],
    mut select_rader_mode: F,
) -> Result<AxisAlgorithm>
where
    F: FnMut(usize) -> Result<Option<RaderMode>>,
{
    tuning.validate()?;
    if len == 0 {
        return Err(VkFftError::ZeroLength { axis: 0 });
    }
    if len == 1 {
        return Ok(AxisAlgorithm::Stockham {
            radix: RadixPlan {
                prime_factors: Vec::new(),
                merged_radices: Vec::new(),
            },
        });
    }

    let factors = prime_factorization(len);
    let mut stockham_factors = Vec::new();
    let mut rader_primes = Vec::<RaderPrimePlan>::new();

    let mut cursor = 0;
    while cursor < factors.len() {
        let prime = factors[cursor];
        let mut multiplicity = 1;
        while cursor + multiplicity < factors.len() && factors[cursor + multiplicity] == prime {
            multiplicity += 1;
        }

        if stockham_primes.contains(&prime) {
            stockham_factors.extend(core::iter::repeat_n(prime, multiplicity));
        } else if let Some(mode) = select_rader_mode(prime)? {
            let generator = primitive_root(prime).expect("prime factor must have a primitive root");
            rader_primes.push(RaderPrimePlan {
                prime,
                multiplicity,
                generator,
                mode,
            });
        } else {
            return plan_bluestein(len);
        }

        cursor += multiplicity;
    }

    let stockham_len = checked_product(&stockham_factors, "Stockham radix product")?;
    let stockham = RadixPlan {
        prime_factors: stockham_factors,
        merged_radices: merged_radix_schedule(stockham_len),
    };

    if rader_primes.is_empty() {
        Ok(AxisAlgorithm::Stockham { radix: stockham })
    } else {
        Ok(AxisAlgorithm::Rader {
            stockham,
            primes: rader_primes,
        })
    }
}

fn plan_c2c_axis_algorithm_for_device(
    len: usize,
    batch_count: usize,
    tuning: PlannerTuning,
    precision: Precision,
    axis_class: C2cDeviceAxisClass,
    device: DeviceProfile,
) -> Result<AxisAlgorithm> {
    let complex_bytes = precision.compute_complex_bytes();
    let coalesced_bytes = upstream_coalesced_memory_bytes_for_precision(device, precision);
    let used_shared_memory_bytes = if len.is_power_of_two() {
        device.shared_memory_pow2_bytes
    } else {
        device.shared_memory_bytes
    };
    let mut fft_rader_limit = c2c_rader_fft_shared_limit(
        used_shared_memory_bytes,
        len,
        axis_class,
        complex_bytes,
        coalesced_bytes,
        tuning.max_rader_fft_prime,
    );
    let effective_tuning = upstream_effective_rader_tuning(tuning, device, precision);
    let direct_rader_limit = effective_tuning.max_rader_direct_prime;

    let stockham_primes: &[usize] = if matches!(
        precision,
        Precision::DoubleDouble | Precision::DoubleDoubleF64Storage
    ) {
        &DOUBLE_DOUBLE_STOCKHAM_PRIMES
    } else {
        &STOCKHAM_PRIMES
    };
    let algorithm =
        plan_axis_algorithm_with_rader_selector(len, tuning, stockham_primes, |prime| {
            let in_fft_scan = prime >= tuning.min_rader_direct_prime
                && prime >= tuning.min_rader_fft_prime
                && prime < fft_rader_limit;
            if in_fft_scan {
                if rader_fft_passes_upstream_safe_prime_gate(prime, tuning) {
                    return Ok(Some(rader_fft_mode(prime)));
                }

                let reservation_bytes = (prime - 1).checked_mul(complex_bytes).ok_or(
                    VkFftError::ArithmeticOverflow {
                        operation: "device-scored direct Rader shared-memory reservation",
                    },
                )?;
                fft_rader_limit = c2c_rader_fft_shared_limit(
                    used_shared_memory_bytes.saturating_sub(reservation_bytes),
                    len,
                    axis_class,
                    complex_bytes,
                    coalesced_bytes,
                    tuning.max_rader_fft_prime,
                );
            }

            if prime >= tuning.min_rader_direct_prime && prime < direct_rader_limit {
                return Ok(Some(RaderMode::DirectMultiplication));
            }

            let fft_eligible =
                prime >= tuning.min_rader_fft_prime && prime < tuning.max_rader_fft_prime;
            if tuning.allow_recursive_fft_rader
                && fft_eligible
                && rader_convolution_is_recursively_plannable(
                    prime - 1,
                    effective_tuning,
                    stockham_primes,
                )
            {
                return Ok(Some(rader_fft_mode(prime)));
            }

            Ok(None)
        })?;

    if matches!(algorithm, AxisAlgorithm::Bluestein { .. }) {
        let convolution_len = upstream_bluestein_auto_padding(len, batch_count, precision, device)?;
        let prime_factors = prime_factorization(convolution_len);
        return Ok(AxisAlgorithm::Bluestein {
            convolution_len,
            convolution: RadixPlan {
                merged_radices: merged_radix_schedule(convolution_len),
                prime_factors,
            },
        });
    }
    Ok(algorithm)
}

fn c2c_rader_fft_shared_limit(
    available_shared_memory_bytes: usize,
    fft_len: usize,
    axis_class: C2cDeviceAxisClass,
    complex_bytes: usize,
    coalesced_bytes: usize,
    configured_fft_limit: usize,
) -> usize {
    let max_sequence_len_shared = available_shared_memory_bytes / complex_bytes;
    let max_sequence_len_strided = if coalesced_bytes > complex_bytes {
        available_shared_memory_bytes / coalesced_bytes
    } else {
        max_sequence_len_shared
    };
    let device_limit =
        if axis_class == C2cDeviceAxisClass::Contiguous && fft_len <= max_sequence_len_shared {
            max_sequence_len_shared
        } else {
            max_sequence_len_strided
        };
    device_limit.min(configured_fft_limit)
}

fn rader_mode(prime: usize, tuning: PlannerTuning) -> Option<RaderMode> {
    rader_mode_with_stockham_primes(prime, tuning, &STOCKHAM_PRIMES)
}

fn rader_mode_with_stockham_primes(
    prime: usize,
    tuning: PlannerTuning,
    stockham_primes: &[usize],
) -> Option<RaderMode> {
    let fft_eligible = prime >= tuning.min_rader_fft_prime && prime < tuning.max_rader_fft_prime;
    // VkFFTConstructRaderTree's safe-prime check divides `(p - 1)` by every factor
    // below fixMinRaderPrimeMult. That knob is also the direct-Rader minimum, so the
    // condition is exactly that every prime factor of `(p - 1)` is below this limit.
    if fft_eligible && rader_fft_passes_upstream_safe_prime_gate(prime, tuning) {
        return Some(rader_fft_mode(prime));
    }

    if prime >= tuning.min_rader_direct_prime && prime < tuning.max_rader_direct_prime {
        return Some(RaderMode::DirectMultiplication);
    }

    // Preserve recursive/sub-Rader as an explicit Rust extension without changing
    // VkFFT's default algorithm selection.
    if tuning.allow_recursive_fft_rader
        && fft_eligible
        && rader_convolution_is_recursively_plannable(prime - 1, tuning, stockham_primes)
    {
        return Some(rader_fft_mode(prime));
    }

    None
}

fn rader_fft_passes_upstream_safe_prime_gate(prime: usize, tuning: PlannerTuning) -> bool {
    prime > 2
        && prime_factorization(prime - 1)
            .into_iter()
            .all(|factor| factor < tuning.min_rader_direct_prime)
}

fn rader_fft_mode(prime: usize) -> RaderMode {
    let factors = prime_factorization(prime - 1);
    RaderMode::FftConvolution {
        convolution: RadixPlan {
            merged_radices: merged_radix_schedule(prime - 1),
            prime_factors: factors,
        },
    }
}

/// Return true when an FFT-Rader convolution length can be decomposed entirely into
/// Stockham factors and smaller Rader primes. Every recursive prime factor is strictly
/// smaller than the parent prime (`q | p-1`), so this terminates without an explicit
/// depth limit. This is used only by the explicit recursive-Rader extension; the
/// default planner follows upstream's stricter safe-prime gate.
fn rader_convolution_is_recursively_plannable(
    len: usize,
    tuning: PlannerTuning,
    stockham_primes: &[usize],
) -> bool {
    prime_factorization(len).into_iter().all(|prime| {
        stockham_primes.contains(&prime)
            || rader_mode_with_stockham_primes(prime, tuning, stockham_primes).is_some()
    })
}

fn plan_bluestein(len: usize) -> Result<AxisAlgorithm> {
    let minimum = len
        .checked_mul(2)
        .and_then(|value| value.checked_sub(1))
        .ok_or(VkFftError::ArithmeticOverflow {
            operation: "Bluestein minimum convolution length",
        })?;
    let convolution_len = next_smooth_at_least(minimum, &BLUESTEIN_SMOOTH_PRIMES)?;
    let factors = prime_factorization(convolution_len);

    Ok(AxisAlgorithm::Bluestein {
        convolution_len,
        convolution: RadixPlan {
            merged_radices: merged_radix_schedule(convolution_len),
            prime_factors: factors,
        },
    })
}

fn scratch_estimate(algorithm: &AxisAlgorithm) -> Result<usize> {
    match algorithm {
        AxisAlgorithm::Stockham { .. } => Ok(0),
        AxisAlgorithm::Rader { primes, .. } => {
            let mut total = 0usize;
            for prime in primes {
                let per_prime = prime
                    .prime
                    .checked_sub(1)
                    .and_then(|value| value.checked_mul(prime.multiplicity))
                    .ok_or(VkFftError::ArithmeticOverflow {
                        operation: "Rader scratch estimate",
                    })?;
                total = total
                    .checked_add(per_prime)
                    .ok_or(VkFftError::ArithmeticOverflow {
                        operation: "Rader scratch estimate",
                    })?;
            }
            Ok(total)
        }
        AxisAlgorithm::Bluestein {
            convolution_len, ..
        } => convolution_len
            .checked_mul(3)
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "Bluestein scratch estimate",
            }),
    }
}

pub fn prime_factorization(mut value: usize) -> Vec<usize> {
    let mut factors = Vec::new();
    if value < 2 {
        return factors;
    }

    while value.is_multiple_of(2) {
        factors.push(2);
        value /= 2;
    }

    let mut divisor = 3usize;
    while divisor <= value / divisor {
        while value.is_multiple_of(divisor) {
            factors.push(divisor);
            value /= divisor;
        }
        divisor += 2;
    }

    if value > 1 {
        factors.push(value);
    }
    factors
}

pub fn is_smooth(mut value: usize, primes: &[usize]) -> bool {
    if value == 0 {
        return false;
    }
    for &prime in primes {
        while value.is_multiple_of(prime) {
            value /= prime;
        }
    }
    value == 1
}

pub fn merged_radix_schedule(mut value: usize) -> Vec<usize> {
    if value <= 1 {
        return Vec::new();
    }

    const FALLBACK_RADICES: [usize; 6] = [13, 11, 7, 5, 3, 2];
    let mut stages = Vec::new();
    while value > 1 {
        let mut selected = None;
        for radix in MERGED_RADICES.into_iter().chain(FALLBACK_RADICES) {
            if value.is_multiple_of(radix) {
                selected = Some(radix);
                break;
            }
        }

        match selected {
            Some(radix) => {
                stages.push(radix);
                value /= radix;
            }
            None => break,
        }
    }
    stages
}

pub fn primitive_root(prime: usize) -> Option<usize> {
    if prime == 2 {
        return Some(1);
    }
    if prime < 2 || prime_factorization(prime).len() != 1 {
        return None;
    }

    let phi = prime - 1;
    let mut unique_factors = prime_factorization(phi);
    unique_factors.dedup();

    'candidate: for generator in 2..prime {
        for &factor in &unique_factors {
            if mod_pow(generator, phi / factor, prime) == 1 {
                continue 'candidate;
            }
        }
        return Some(generator);
    }
    None
}

fn mod_pow(mut base: usize, mut exponent: usize, modulus: usize) -> usize {
    let modulus_u128 = modulus as u128;
    let mut result = 1u128;
    let mut base_u128 = (base % modulus) as u128;
    while exponent > 0 {
        if exponent & 1 == 1 {
            result = (result * base_u128) % modulus_u128;
        }
        base_u128 = (base_u128 * base_u128) % modulus_u128;
        exponent >>= 1;
    }
    base = result as usize;
    base
}

pub fn next_smooth_at_least(target: usize, primes: &[usize]) -> Result<usize> {
    if target <= 1 {
        return Ok(1);
    }
    if primes.is_empty() || primes.iter().any(|&prime| prime < 2) {
        return Err(VkFftError::InvalidPlannerTuning(
            "smooth-number prime set must contain values >= 2",
        ));
    }

    let mut ordered = primes.to_vec();
    ordered.sort_unstable();
    ordered.dedup();

    let mut best = usize::MAX;
    if ordered.binary_search(&2).is_ok() {
        best = target.checked_next_power_of_two().unwrap_or(usize::MAX);
    }

    fn visit(primes: &[usize], index: usize, current: usize, target: usize, best: &mut usize) {
        if current >= *best {
            return;
        }
        if index == primes.len() {
            if current >= target {
                *best = current;
            }
            return;
        }

        let prime = primes[index];
        let mut value = current;
        loop {
            visit(primes, index + 1, value, target, best);
            match value.checked_mul(prime) {
                Some(next) if next < *best => value = next,
                _ => break,
            }
        }
    }

    visit(&ordered, 0, 1, target, &mut best);
    if best == usize::MAX {
        Err(VkFftError::ArithmeticOverflow {
            operation: "smooth convolution length",
        })
    } else {
        Ok(best)
    }
}

fn checked_product(values: &[usize], operation: &'static str) -> Result<usize> {
    values.iter().try_fold(1usize, |acc, &value| {
        acc.checked_mul(value)
            .ok_or(VkFftError::ArithmeticOverflow { operation })
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{Backend, FftConfig, GpuVendor};

    #[test]
    fn factorization_is_stable() {
        assert_eq!(prime_factorization(360), vec![2, 2, 2, 3, 3, 5]);
        assert_eq!(prime_factorization(47), vec![47]);
    }

    #[test]
    fn stockham_lengths_stay_stockham() {
        let algorithm =
            plan_axis_algorithm(2 * 3 * 5 * 7 * 11 * 13, PlannerTuning::portable()).unwrap();
        assert_eq!(algorithm.kind(), AlgorithmKind::Stockham);
    }

    #[test]
    fn smooth_rader_prime_uses_fft_convolution() {
        let algorithm = plan_axis_algorithm(17, PlannerTuning::portable()).unwrap();
        let AxisAlgorithm::Rader { primes, .. } = algorithm else {
            panic!("17 should use Rader");
        };
        assert_eq!(primes[0].prime, 17);
        assert!(matches!(primes[0].mode, RaderMode::FftConvolution { .. }));
    }

    #[test]
    fn safe_prime_uses_direct_rader_when_enabled() {
        let algorithm = plan_axis_algorithm(47, PlannerTuning::portable()).unwrap();
        let AxisAlgorithm::Rader { primes, .. } = algorithm else {
            panic!("47 should use Rader");
        };
        assert!(matches!(primes[0].mode, RaderMode::DirectMultiplication));
    }

    #[test]
    fn device_scored_double_double_uses_quad_stockham_prime_set() {
        let device = DeviceProfile::generic(Backend::Vulkan, GpuVendor::Nvidia);

        let ordinary = FftPlan::build_for_device(FftConfig::new(vec![13]), device).unwrap();
        assert!(matches!(
            ordinary.axes[0].algorithm,
            AxisAlgorithm::Stockham { .. }
        ));

        let portable_dd =
            FftPlan::build(FftConfig::new(vec![13]).with_precision(Precision::DoubleDouble))
                .unwrap();
        assert!(matches!(
            portable_dd.axes[0].algorithm,
            AxisAlgorithm::Stockham { .. }
        ));

        let dd_config = FftConfig::new(vec![13]).with_precision(Precision::DoubleDouble);
        let device_dd = FftPlan::build_for_device(dd_config, device).unwrap();
        let AxisAlgorithm::Rader { stockham, primes } = &device_dd.axes[0].algorithm else {
            panic!("device-scored DD p13 must use the Quad/Rader path");
        };
        assert!(stockham.prime_factors.is_empty());
        assert_eq!(primes.len(), 1);
        assert_eq!(primes[0].prime, 13);
        assert!(matches!(primes[0].mode, RaderMode::DirectMultiplication));

        let explicit_portable = FftPlan::build_for_device(
            FftConfig::new(vec![13])
                .with_precision(Precision::DoubleDouble)
                .with_tuning(PlannerTuning::portable()),
            device,
        )
        .unwrap();
        assert!(matches!(
            explicit_portable.axes[0].algorithm,
            AxisAlgorithm::Bluestein { .. }
        ));
    }

    #[test]
    fn device_scored_c2c_rader_limits_follow_thread_shared_and_strided_capacity() {
        assert_eq!(prime_factorization(7681), vec![7681]);
        let base = DeviceProfile::generic(Backend::Vulkan, GpuVendor::Nvidia);

        // Low-level planning remains device-neutral and keeps the initialization
        // FFT-Rader range, even when a real device would need a strided upload.
        let portable = FftPlan::build(FftConfig::new(vec![7681])).unwrap();
        let AxisAlgorithm::Rader { primes, .. } = &portable.axes[0].algorithm else {
            panic!("portable p7681 should remain FFT-Rader");
        };
        assert!(matches!(primes[0].mode, RaderMode::FftConvolution { .. }));

        // 32 KiB F32 holds 4096 contiguous complex values, but p7681 exceeds
        // that non-strided capacity. Upstream therefore uses the strided
        // 32KiB/32B = 1024 prime limit and falls back to Bluestein.
        let constrained = FftPlan::build_for_device(FftConfig::new(vec![7681]), base).unwrap();
        assert!(matches!(
            constrained.axes[0].algorithm,
            AxisAlgorithm::Bluestein { .. }
        ));

        // Unknown backend/vendor combinations do not inherit NVIDIA's
        // device-scored shared-memory policy. Their typed planner remains on the
        // portable/fail-soft algorithm classification.
        let unknown = DeviceProfile::generic(Backend::Vulkan, GpuVendor::Other(0x1234));
        let unknown_plan = FftPlan::build_for_device(FftConfig::new(vec![7681]), unknown).unwrap();
        let AxisAlgorithm::Rader { primes, .. } = &unknown_plan.axes[0].algorithm else {
            panic!("unknown profile should retain portable p7681 FFT-Rader classification");
        };
        assert!(matches!(primes[0].mode, RaderMode::FftConvolution { .. }));

        // At 64 KiB the same prime fits the non-strided 8192-complex limit and
        // its smooth p-1 safe-prime convolution becomes FFT-Rader again.
        let wide = DeviceProfile {
            shared_memory_bytes: 64 * 1024,
            shared_memory_pow2_bytes: 64 * 1024,
            ..base
        };
        let wide_plan = FftPlan::build_for_device(FftConfig::new(vec![7681]), wide).unwrap();
        let AxisAlgorithm::Rader { primes, .. } = &wide_plan.axes[0].algorithm else {
            panic!("64 KiB p7681 should use FFT-Rader");
        };
        assert!(matches!(primes[0].mode, RaderMode::FftConvolution { .. }));

        // This crate stores tensors row-major with the last dimension contiguous.
        // Map that last local axis to VkFFT's nonStridedAxisId=0. p4001 fits the
        // 4096-complex contiguous budget on local axis 1, while local axis 0 is
        // evaluated through the strided 1024-entry budget.
        let nd = FftPlan::build_for_device(FftConfig::new(vec![4001, 4001]), base).unwrap();
        assert!(matches!(
            nd.axes[0].algorithm,
            AxisAlgorithm::Bluestein { .. }
        ));
        assert!(matches!(nd.axes[1].algorithm, AxisAlgorithm::Rader { .. }));

        // Extracted one-dimensional children must preserve the parent axis class;
        // otherwise every temporary 1D config would incorrectly look contiguous.
        let child_config = FftConfig::new(vec![4001]);
        let contiguous_child = FftPlan::build_c2c_child_for_device(
            child_config.clone(),
            base,
            C2cDeviceAxisClass::Contiguous,
        )
        .unwrap();
        assert!(matches!(
            contiguous_child.axes[0].algorithm,
            AxisAlgorithm::Rader { .. }
        ));
        let strided_child = FftPlan::build_c2c_child_for_device(
            child_config.clone(),
            base,
            C2cDeviceAxisClass::Strided,
        )
        .unwrap();
        let AxisAlgorithm::Bluestein {
            convolution_len: strided_padded,
            ..
        } = strided_child.axes[0].algorithm
        else {
            panic!("strided p4001 should use Bluestein");
        };
        assert_eq!(strided_padded, 8_192);

        // Portable planning keeps its deterministic 2/3/5/7-smooth padding; only
        // device-scored planning consumes the fixed upstream auto-padding table.
        let mut portable_bluestein = PlannerTuning::portable();
        portable_bluestein.max_rader_fft_prime = 100;
        let portable_child = FftPlan::build(child_config.with_tuning(portable_bluestein)).unwrap();
        let AxisAlgorithm::Bluestein {
            convolution_len: portable_padded,
            ..
        } = portable_child.axes[0].algorithm
        else {
            panic!("forced portable p4001 should use Bluestein");
        };
        assert_eq!(portable_padded, 8_064);

        // Explicit user thresholds still encounter VkFFTScheduler's physical
        // thread/coalescing cap. p83 fails the safe-prime FFT gate; on a
        // 128-thread F32 profile the direct-Rader cap is 63, so it cannot use
        // the user's otherwise-permissive direct range.
        let mut explicit = PlannerTuning::portable();
        explicit.max_rader_direct_prime = 1_000;
        let thread_limited = DeviceProfile {
            max_threads_per_block: 128,
            max_workgroup_size: [128, 128, 64],
            ..base
        };
        let explicit_plan = FftPlan::build_for_device(
            FftConfig::new(vec![83]).with_tuning(explicit),
            thread_limited,
        )
        .unwrap();
        assert!(matches!(
            explicit_plan.axes[0].algorithm,
            AxisAlgorithm::Bluestein { .. }
        ));

        // Upstream doubles coalescedMemory in half-storage mode while keeping F32
        // compute complexSize. On this 128-thread NVIDIA profile that narrows the
        // Direct-Rader exclusive cap from 63 to 31, so p47 diverges by precision.
        let f32_p47 = FftPlan::build_for_device(FftConfig::new(vec![47]), thread_limited).unwrap();
        let AxisAlgorithm::Rader { primes, .. } = &f32_p47.axes[0].algorithm else {
            panic!("F32 p47 should remain Direct-Rader under the 63-prime cap");
        };
        assert!(matches!(primes[0].mode, RaderMode::DirectMultiplication));
        let f16_p47 = FftPlan::build_for_device(
            FftConfig::new(vec![47]).with_precision(Precision::F16StorageF32Compute),
            thread_limited,
        )
        .unwrap();
        assert!(matches!(
            f16_p47.axes[0].algorithm,
            AxisAlgorithm::Bluestein { .. }
        ));
    }

    #[test]
    fn unsafe_direct_rader_reservation_can_push_a_later_safe_prime_to_bluestein() {
        let length = 83usize * 1009usize;
        assert_eq!(prime_factorization(length), vec![83, 1009]);
        assert!(!rader_fft_passes_upstream_safe_prime_gate(
            83,
            PlannerTuning::portable()
        ));
        assert!(rader_fft_passes_upstream_safe_prime_gate(
            1009,
            PlannerTuning::portable()
        ));

        let base = DeviceProfile::generic(Backend::Vulkan, GpuVendor::Nvidia);
        // Because N is larger than the contiguous 32 KiB F32 capacity, the
        // initial FFT-Rader limit is the strided 32768/32 = 1024. p83 fails
        // the safe-prime gate and reserves 82*8 = 656 bytes; the recomputed
        // limit becomes floor((32768-656)/32) = 1003, excluding p1009.
        let constrained = FftPlan::build_for_device(FftConfig::new(vec![length]), base).unwrap();
        assert!(matches!(
            constrained.axes[0].algorithm,
            AxisAlgorithm::Bluestein { .. }
        ));

        // 33 KiB leaves floor((33792-656)/32) = 1035 strided entries, so the
        // same p1009 candidate survives after p83's reservation. The axis can
        // then keep p83 as direct Rader and p1009 as FFT-convolution Rader.
        let wider = DeviceProfile {
            shared_memory_bytes: 33 * 1024,
            ..base
        };
        let plan = FftPlan::build_for_device(FftConfig::new(vec![length]), wider).unwrap();
        let AxisAlgorithm::Rader { primes, .. } = &plan.axes[0].algorithm else {
            panic!("33 KiB reservation case should retain the mixed Rader axis");
        };
        assert_eq!(primes.len(), 2);
        assert_eq!(primes[0].prime, 83);
        assert!(matches!(primes[0].mode, RaderMode::DirectMultiplication));
        assert_eq!(primes[1].prime, 1009);
        assert!(matches!(primes[1].mode, RaderMode::FftConvolution { .. }));
    }

    #[test]
    fn nested_sub_rader_prime_is_opt_in_beyond_upstream_safe_prime_gate() {
        let upstream = plan_axis_algorithm(107, PlannerTuning::portable()).unwrap();
        assert!(matches!(upstream, AxisAlgorithm::Bluestein { .. }));

        let recursive = PlannerTuning::portable().with_recursive_fft_rader(true);
        let algorithm = plan_axis_algorithm(107, recursive).unwrap();
        let AxisAlgorithm::Rader { primes, .. } = algorithm else {
            panic!("107 should use nested FFT-Rader");
        };
        let RaderMode::FftConvolution { convolution } = &primes[0].mode else {
            panic!("107 should use FFT-convolution Rader");
        };
        assert_eq!(convolution.prime_factors, vec![2, 53]);
        let nested = plan_axis_algorithm(106, recursive).unwrap();
        let AxisAlgorithm::Rader { primes, .. } = nested else {
            panic!("106-point convolution should contain a Rader sub-prime");
        };
        assert_eq!(primes[0].prime, 53);
        assert!(matches!(primes[0].mode, RaderMode::FftConvolution { .. }));
    }

    #[test]
    fn recursive_rader_eligibility_uses_precision_specific_stockham_primes() {
        let device = DeviceProfile::generic(Backend::Vulkan, GpuVendor::Nvidia);
        let tuning = PlannerTuning::portable().with_recursive_fft_rader(true);

        let ordinary =
            FftPlan::build_for_device(FftConfig::new(vec![419]).with_tuning(tuning), device)
                .unwrap();
        assert!(matches!(
            ordinary.axes[0].algorithm,
            AxisAlgorithm::Rader { .. }
        ));

        let double_double = FftPlan::build_for_device(
            FftConfig::new(vec![419])
                .with_precision(Precision::DoubleDouble)
                .with_tuning(tuning),
            device,
        )
        .unwrap();
        assert!(matches!(
            double_double.axes[0].algorithm,
            AxisAlgorithm::Bluestein { .. }
        ));
    }

    #[test]
    fn unsupported_prime_uses_bluestein() {
        let length = 65_537usize;
        let algorithm = plan_axis_algorithm(length, PlannerTuning::portable()).unwrap();
        let AxisAlgorithm::Bluestein {
            convolution_len,
            convolution,
        } = algorithm
        else {
            panic!("prime above the configured Rader range should use Bluestein");
        };
        assert!(convolution_len >= 2 * length - 1);
        assert!(is_smooth(convolution_len, &BLUESTEIN_SMOOTH_PRIMES));
        assert_eq!(
            convolution.prime_factors.iter().product::<usize>(),
            convolution_len
        );
    }

    #[test]
    fn grouped_bluestein_is_available_across_ordinary_and_double_double_precision() {
        let mut tuning = PlannerTuning::portable();
        tuning.max_rader_fft_prime = 100;
        for precision in [
            Precision::F32,
            Precision::F16StorageF32Compute,
            Precision::F64,
            Precision::F64ComputeF32Storage,
            Precision::DoubleDouble,
            Precision::DoubleDoubleF64Storage,
        ] {
            let config = FftConfig::new(vec![103])
                .with_batch_count(7)
                .with_precision(precision)
                .with_tuning(tuning)
                .with_grouped_batch(0, 3)
                .unwrap();
            let plan = FftPlan::build(config).unwrap();
            assert_eq!(plan.config.grouped_batch_for_axis(0), Some(3));
            assert!(matches!(
                plan.axes[0].algorithm,
                AxisAlgorithm::Bluestein { .. }
            ));
        }
    }

    #[test]
    fn plan_is_per_axis() {
        let plan = FftPlan::build(FftConfig::new(vec![16, 17, 1031, 65_537])).unwrap();
        assert_eq!(plan.axes[0].algorithm.kind(), AlgorithmKind::Stockham);
        assert_eq!(plan.axes[1].algorithm.kind(), AlgorithmKind::Rader);
        assert_eq!(plan.axes[2].algorithm.kind(), AlgorithmKind::Bluestein);
        assert_eq!(plan.axes[3].algorithm.kind(), AlgorithmKind::Bluestein);
    }

    #[test]
    fn primitive_root_covers_multiplicative_group() {
        for prime in [17usize, 47, 59, 83] {
            let generator = primitive_root(prime).unwrap();
            let mut seen = std::collections::BTreeSet::new();
            let mut value = 1usize;
            for _ in 0..prime - 1 {
                seen.insert(value);
                value = (value * generator) % prime;
            }
            assert_eq!(seen.len(), prime - 1);
        }
    }

    #[test]
    fn smooth_search_finds_minimum_candidate() {
        let value = next_smooth_at_least(2061, &BLUESTEIN_SMOOTH_PRIMES).unwrap();
        assert!(value >= 2061);
        assert!(is_smooth(value, &BLUESTEIN_SMOOTH_PRIMES));
        assert!((2061..value).all(|candidate| !is_smooth(candidate, &BLUESTEIN_SMOOTH_PRIMES)));
    }
}
