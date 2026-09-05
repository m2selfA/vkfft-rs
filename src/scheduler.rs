//! Typed host-side scheduling metadata ported from VkFFT's GPU scheduler.
//!
//! The arithmetic/register tables are backend-neutral; device families select a
//! typed policy matching the fixed upstream initialization defaults. Vulkan/OpenCL
//! preserve vendor-specific NVIDIA/AMD/Intel register-boost, wave and Four-step
//! thresholds, while CUDA, HIP, Level Zero and Metal select their native defaults.
//! Power-of-two and ordinary smooth 2/3/5/7/11/13 Stockham, FFT-Rader occupancy,
//! grouped containers and `raderTranspose` all consume the same policy-driven core.

use crate::config::{
    Backend, DeviceProfile, GpuVendor, PlannerTuning, Precision,
    has_fixed_upstream_gpu_scheduler_profile, upstream_coalesced_memory_bytes_for_precision,
    upstream_effective_rader_tuning,
};
use crate::error::{Result, VkFftError};

// Fixed-commit `initializeBluesteinAutoPadding` threshold tables. The first
// tuple element is the inclusive sequence-length threshold and the second is
// the padded Bluestein convolution size selected until the next threshold.
// Keep these literal: they are benchmark-derived upstream initialization data.
const NVIDIA_WIDE_BLUESTEIN_AUTO_PADDING: &[(usize, usize)] = &[
    (17, 36),
    (19, 40),
    (23, 48),
    (29, 64),
    (34, 70),
    (37, 80),
    (41, 90),
    (46, 96),
    (51, 104),
    (53, 128),
    (67, 144),
    (73, 160),
    (82, 256),
    (129, 288),
    (145, 512),
    (257, 625),
    (314, 750),
    (376, 756),
    (379, 768),
    (386, 1024),
    (513, 1056),
    (529, 1200),
    (601, 1225),
    (614, 1250),
    (626, 1296),
    (649, 1331),
    (667, 1440),
    (721, 1456),
    (730, 1560),
    (781, 2048),
    (1025, 2187),
    (1095, 2304),
    (1153, 2688),
    (1345, 2730),
    (1366, 2925),
    (1464, 3000),
    (1501, 4096),
    (2049, 4368),
    (2185, 4608),
    (2305, 4900),
    (2364, 4900),
    (2451, 5184),
    (2593, 5625),
    (2814, 5760),
    (2881, 6000),
    (3001, 6048),
    (3026, 6144),
    (3073, 6561),
    (3282, 8192),
];

const NVIDIA_F32_BLUESTEIN_AUTO_PADDING: &[(usize, usize)] = &[
    (17, 36),
    (19, 40),
    (23, 48),
    (29, 64),
    (34, 70),
    (37, 80),
    (41, 96),
    (51, 104),
    (53, 112),
    (57, 120),
    (61, 128),
    (67, 144),
    (73, 150),
    (76, 160),
    (82, 256),
    (129, 384),
    (193, 512),
    (257, 567),
    (285, 625),
    (314, 768),
    (386, 832),
    (417, 1024),
    (513, 1152),
    (577, 1200),
    (601, 1296),
    (649, 1536),
    (769, 2048),
    (1025, 2187),
    (1095, 2304),
    (1153, 2500),
    (1251, 2592),
    (1297, 2816),
    (1409, 3072),
    (1537, 4096),
    (2049, 4368),
    (2185, 4563),
    (2283, 4576),
    (2289, 4608),
    (2305, 5184),
    (2593, 5625),
    (2814, 5632),
    (2817, 6000),
    (3001, 6144),
    (3073, 6561),
    (3282, 8192),
];

const OTHER_WIDE_BLUESTEIN_AUTO_PADDING: &[(usize, usize)] = &[
    (17, 36),
    (19, 40),
    (23, 56),
    (29, 64),
    (34, 70),
    (37, 78),
    (41, 81),
    (43, 90),
    (46, 125),
    (67, 150),
    (76, 175),
    (89, 189),
    (97, 198),
    (101, 243),
    (123, 256),
    (129, 270),
    (136, 512),
    (257, 625),
    (314, 640),
    (321, 702),
    (353, 750),
    (376, 756),
    (379, 768),
    (386, 875),
    (439, 1024),
    (513, 1296),
    (649, 1300),
    (651, 1323),
    (663, 1344),
    (673, 1512),
    (757, 1792),
    (897, 2016),
    (1009, 2048),
    (1025, 2187),
    (1095, 3136),
    (1569, 3159),
    (1581, 3430),
    (1717, 3584),
    (1793, 4096),
    (2049, 4224),
    (2113, 4375),
    (2189, 4480),
    (2241, 4704),
    (2353, 4928),
    (2465, 4992),
    (2497, 5005),
    (2504, 5103),
    (2553, 5376),
    (2689, 5632),
    (2817, 5824),
    (2913, 6048),
    (3026, 6144),
    (3073, 6875),
    (3439, 8192),
];

const OTHER_F32_BLUESTEIN_AUTO_PADDING: &[(usize, usize)] = &[
    (17, 36),
    (19, 42),
    (23, 64),
    (34, 81),
    (43, 88),
    (46, 125),
    (67, 150),
    (76, 162),
    (82, 175),
    (89, 256),
    (129, 512),
    (257, 625),
    (314, 768),
    (386, 1024),
    (513, 1296),
    (649, 2048),
    (1025, 2187),
    (1095, 2304),
    (1153, 2500),
    (1251, 2592),
    (1297, 3072),
    (1537, 3125),
    (1564, 3136),
    (1569, 4096),
    (2049, 4375),
    (2189, 4608),
    (2305, 5184),
    (2593, 6561),
    (3282, 8192),
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StockhamTwiddleSource {
    OnTheFly,
    LookupTable,
}

/// Backend/vendor scheduling defaults from the fixed upstream `vkFFT_InitializeApp.h`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GpuSchedulerPolicy {
    pub register_boost: usize,
    pub register_boost_four_step: usize,
    pub register_boost_non_power_of_two: bool,
    pub reorder_four_step: bool,
    pub swap_to_two_stage_four_step: usize,
    pub swap_to_three_stage_four_step: usize,
    pub subgroup_width: usize,
    pub coalesced_memory_bytes: usize,
    pub stockham_twiddle_source: StockhamTwiddleSource,
    pub four_step_twiddle_source: StockhamTwiddleSource,
}

fn precision_uses_lut(precision: Precision) -> bool {
    matches!(
        precision,
        Precision::F64
            | Precision::F64ComputeF32Storage
            | Precision::DoubleDouble
            | Precision::DoubleDoubleF64Storage
    )
}

/// Return the fixed-commit auto-padding table result when `sequence_len` is
/// covered by `initializeBluesteinAutoPadding`. `None` means the scheduler must
/// continue into the generic good-sequence search rather than guessing a pad.
pub(crate) fn upstream_bluestein_auto_padding_from_table(
    sequence_len: usize,
    precision: Precision,
    device: DeviceProfile,
) -> Option<usize> {
    if !has_fixed_upstream_gpu_scheduler_profile(device) {
        return None;
    }
    let wide = precision_uses_lut(precision);
    let table = match (device.vendor, wide) {
        (GpuVendor::Nvidia, true) => NVIDIA_WIDE_BLUESTEIN_AUTO_PADDING,
        (GpuVendor::Nvidia, false) => NVIDIA_F32_BLUESTEIN_AUTO_PADDING,
        (_, true) => OTHER_WIDE_BLUESTEIN_AUTO_PADDING,
        (_, false) => OTHER_F32_BLUESTEIN_AUTO_PADDING,
    };
    let required = sequence_len.checked_mul(2)?.checked_sub(1)?;
    for (index, &(threshold, padded)) in table.iter().enumerate() {
        if sequence_len < threshold {
            break;
        }
        let Some(&(next_threshold, _)) = table.get(index + 1) else {
            return (required <= padded).then_some(padded);
        };
        if sequence_len < next_threshold {
            return (required <= padded).then_some(padded);
        }
    }
    None
}

fn is_bluestein_small_radix_smooth(mut value: usize) -> bool {
    for prime in [2usize, 3, 5, 7] {
        while value.is_multiple_of(prime) {
            value /= prime;
        }
    }
    value == 1
}

fn upstream_bluestein_generic_padding(
    sequence_len: usize,
    batch_count: usize,
    precision: Precision,
    device: DeviceProfile,
) -> Result<usize> {
    if !has_fixed_upstream_gpu_scheduler_profile(device) {
        return Err(VkFftError::UnsupportedKernelPath(
            "fixed upstream Bluestein padding requires a known GPU scheduler profile",
        ));
    }
    let required = sequence_len
        .checked_mul(2)
        .and_then(|value| value.checked_sub(1))
        .ok_or(VkFftError::ArithmeticOverflow {
            operation: "fixed upstream Bluestein minimum padded length",
        })?;
    let max_rhs = sequence_len
        .checked_mul(batch_count)
        .ok_or(VkFftError::ArithmeticOverflow {
            operation: "fixed upstream Bluestein RHS estimate",
        })?;
    let complex_bytes = precision.compute_complex_bytes();
    let used_shared_memory = if sequence_len.is_power_of_two() {
        device.shared_memory_pow2_bytes
    } else {
        device.shared_memory_bytes
    };
    let max_sequence_len_shared = used_shared_memory / complex_bytes;
    let mut candidate = required;

    loop {
        let next_power =
            candidate
                .checked_next_power_of_two()
                .ok_or(VkFftError::ArithmeticOverflow {
                    operation: "fixed upstream Bluestein next power of two",
                })?;
        let three_quarters_reached = (next_power as u128) * 3 <= (candidate as u128) * 4;
        let power_fits_or_required_does_not =
            next_power <= max_sequence_len_shared || required > max_sequence_len_shared;
        if sequence_len < 128 || (three_quarters_reached && power_fits_or_required_does_not) {
            candidate = next_power;
        }

        if is_bluestein_small_radix_smooth(candidate) {
            // Upstream passes integer `max_rhs / candidate`; the register scorer
            // clamps its active-y estimate to one when that quotient is zero.
            let rhs_transform_count = (max_rhs / candidate).max(1);
            let is_good = if matches!(
                precision,
                Precision::DoubleDouble | Precision::DoubleDoubleF64Storage
            ) {
                // VkFFTGetRegistersPerThread dispatches directly to the Quad table
                // for both native-DD and DD-compute/F64-storage modes.
                plan_gpu_double_double_quad_registers(candidate, rhs_transform_count)?
                    .is_good_sequence
            } else if candidate.is_power_of_two() {
                plan_gpu_power_of_two_radix_registers(candidate, rhs_transform_count, 1)?
                    .is_good_sequence
            } else {
                plan_gpu_small_mixed_radix_registers(candidate, rhs_transform_count)?
                    .is_good_sequence
            };
            if is_good {
                return Ok(candidate);
            }
        }
        candidate = candidate
            .checked_add(1)
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "fixed upstream Bluestein padding search",
            })?;
    }
}

/// Fixed-commit default Bluestein auto-padding: benchmark-derived table first,
/// then the scheduler's power-of-two/small-radix `isGoodSequence` search.
pub(crate) fn upstream_bluestein_auto_padding(
    sequence_len: usize,
    batch_count: usize,
    precision: Precision,
    device: DeviceProfile,
) -> Result<usize> {
    if let Some(padded) =
        upstream_bluestein_auto_padding_from_table(sequence_len, precision, device)
    {
        return Ok(padded);
    }
    upstream_bluestein_generic_padding(sequence_len, batch_count, precision, device)
}

/// Whether the fixed upstream initialization has an explicit policy for this
/// backend/vendor pair. Unknown combinations deliberately retain portable IR
/// instead of inheriting a neighboring vendor's performance assumptions.
pub const fn has_specialized_gpu_scheduler_policy(device: DeviceProfile) -> bool {
    has_fixed_upstream_gpu_scheduler_profile(device)
}

/// Resolve the scheduler policy for a concrete backend/vendor pair. The policy is
/// deliberately separate from physical limits in `DeviceProfile`: callers can feed
/// queried shared-memory/thread limits while retaining VkFFT's architecture defaults.
pub fn plan_gpu_scheduler_policy(
    precision: Precision,
    device: DeviceProfile,
) -> Result<GpuSchedulerPolicy> {
    if !has_specialized_gpu_scheduler_policy(device) {
        return Err(VkFftError::UnsupportedKernelPath(
            "fixed upstream scheduler policy is unavailable for this backend/vendor pair",
        ));
    }
    let wide = precision_uses_lut(precision);
    // Fixed upstream keeps LUT selection and Four-step topology as separate
    // precision classifications. `doublePrecisionFloatMemory` uses the F64 LUT
    // but remains in the F32/F16 swap-threshold class.
    let reduced_four_step_threshold_precision = matches!(
        precision,
        Precision::F64 | Precision::DoubleDouble | Precision::DoubleDoubleF64Storage
    );
    let lut = if wide {
        StockhamTwiddleSource::LookupTable
    } else {
        StockhamTwiddleSource::OnTheFly
    };
    let coalesced_memory_bytes = upstream_coalesced_memory_bytes_for_precision(device, precision);
    let policy = match device.backend {
        Backend::Vulkan | Backend::OpenCl => match device.vendor {
            GpuVendor::Nvidia => GpuSchedulerPolicy {
                register_boost: 4,
                register_boost_four_step: 1,
                register_boost_non_power_of_two: false,
                reorder_four_step: true,
                swap_to_two_stage_four_step: 4_194_305,
                swap_to_three_stage_four_step: 4_194_305,
                subgroup_width: 32,
                coalesced_memory_bytes,
                stockham_twiddle_source: lut,
                four_step_twiddle_source: lut,
            },
            GpuVendor::Amd => GpuSchedulerPolicy {
                register_boost: if device.shared_memory_bytes >= 65_536 {
                    2
                } else {
                    4
                },
                register_boost_four_step: 1,
                register_boost_non_power_of_two: false,
                reorder_four_step: true,
                swap_to_two_stage_four_step: if reduced_four_step_threshold_precision {
                    262_144
                } else {
                    524_288
                },
                swap_to_three_stage_four_step: if reduced_four_step_threshold_precision {
                    262_144
                } else {
                    524_288
                },
                subgroup_width: 64,
                coalesced_memory_bytes,
                stockham_twiddle_source: lut,
                four_step_twiddle_source: lut,
            },
            GpuVendor::Intel => GpuSchedulerPolicy {
                register_boost: if device.shared_memory_bytes >= 65_536 {
                    1
                } else {
                    2
                },
                register_boost_four_step: 1,
                register_boost_non_power_of_two: false,
                reorder_four_step: true,
                swap_to_two_stage_four_step: if reduced_four_step_threshold_precision {
                    262_144
                } else {
                    524_288
                },
                swap_to_three_stage_four_step: if reduced_four_step_threshold_precision {
                    262_144
                } else {
                    524_288
                },
                subgroup_width: 32,
                coalesced_memory_bytes,
                stockham_twiddle_source: StockhamTwiddleSource::LookupTable,
                four_step_twiddle_source: StockhamTwiddleSource::LookupTable,
            },
            GpuVendor::Apple | GpuVendor::Other(_) => GpuSchedulerPolicy {
                register_boost: 1,
                register_boost_four_step: 1,
                register_boost_non_power_of_two: false,
                reorder_four_step: true,
                swap_to_two_stage_four_step: if reduced_four_step_threshold_precision {
                    262_144
                } else {
                    524_288
                },
                swap_to_three_stage_four_step: if reduced_four_step_threshold_precision {
                    262_144
                } else {
                    524_288
                },
                subgroup_width: 32,
                coalesced_memory_bytes,
                stockham_twiddle_source: lut,
                four_step_twiddle_source: lut,
            },
        },
        Backend::Cuda => GpuSchedulerPolicy {
            register_boost: 1,
            register_boost_four_step: 1,
            register_boost_non_power_of_two: false,
            reorder_four_step: true,
            swap_to_two_stage_four_step: 4_194_305,
            swap_to_three_stage_four_step: 4_194_305,
            subgroup_width: 32,
            coalesced_memory_bytes,
            stockham_twiddle_source: lut,
            four_step_twiddle_source: lut,
        },
        Backend::Hip => {
            GpuSchedulerPolicy {
                register_boost: 1,
                register_boost_four_step: 1,
                register_boost_non_power_of_two: false,
                reorder_four_step: true,
                swap_to_two_stage_four_step: if reduced_four_step_threshold_precision {
                    1_048_576
                } else {
                    2_097_152
                },
                swap_to_three_stage_four_step: if reduced_four_step_threshold_precision {
                    1_048_576
                } else {
                    2_097_152
                },
                // Fixed upstream obtains HIP warpSize from the runtime. Keep 64 as the
                // synthetic/fail-soft default, but honor an exact runtime-discovered
                // wave32 profile on gfx10+ hardware.
                subgroup_width: if device.subgroup.size == 0 {
                    64
                } else {
                    device.subgroup.size
                },
                coalesced_memory_bytes,
                stockham_twiddle_source: lut,
                // Upstream HIP initializes useLUT_4step=-1 independently of useLUT.
                four_step_twiddle_source: StockhamTwiddleSource::OnTheFly,
            }
        }
        Backend::LevelZero => GpuSchedulerPolicy {
            register_boost: if device.shared_memory_bytes >= 65_536 {
                1
            } else {
                2
            },
            register_boost_four_step: 1,
            register_boost_non_power_of_two: false,
            reorder_four_step: true,
            swap_to_two_stage_four_step: if reduced_four_step_threshold_precision {
                262_144
            } else {
                524_288
            },
            swap_to_three_stage_four_step: if reduced_four_step_threshold_precision {
                262_144
            } else {
                524_288
            },
            subgroup_width: device.subgroup.size.max(1),
            coalesced_memory_bytes,
            stockham_twiddle_source: StockhamTwiddleSource::LookupTable,
            four_step_twiddle_source: StockhamTwiddleSource::LookupTable,
        },
        Backend::Metal => GpuSchedulerPolicy {
            register_boost: 1,
            register_boost_four_step: 1,
            register_boost_non_power_of_two: false,
            reorder_four_step: true,
            swap_to_two_stage_four_step: if reduced_four_step_threshold_precision {
                262_144
            } else {
                524_288
            },
            swap_to_three_stage_four_step: if reduced_four_step_threshold_precision {
                262_144
            } else {
                524_288
            },
            subgroup_width: device.subgroup.size.max(1),
            coalesced_memory_bytes,
            stockham_twiddle_source: lut,
            four_step_twiddle_source: lut,
        },
        Backend::CpuReference => {
            return Err(VkFftError::UnsupportedKernelPath(
                "GPU scheduler policy is not defined for the CPU reference backend",
            ));
        }
    };
    Ok(policy)
}

/// Fixed-commit GPU `useLUT` policy for Stockham stage twiddles.
pub fn plan_gpu_stockham_twiddle_source(
    precision: Precision,
    device: DeviceProfile,
) -> StockhamTwiddleSource {
    plan_gpu_scheduler_policy(precision, device)
        .map(|policy| policy.stockham_twiddle_source)
        .unwrap_or(StockhamTwiddleSource::OnTheFly)
}

/// Compatibility wrapper preserving the original NVIDIA/Vulkan API contract.
pub fn plan_nvidia_vulkan_stockham_twiddle_source(
    precision: Precision,
    device: DeviceProfile,
) -> StockhamTwiddleSource {
    if device.backend == Backend::Vulkan && device.vendor == GpuVendor::Nvidia {
        plan_gpu_stockham_twiddle_source(precision, device)
    } else {
        StockhamTwiddleSource::OnTheFly
    }
}

/// Scheduling knobs initialized by upstream VkFFT for NVIDIA Vulkan devices.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NvidiaVulkanSchedulerTuning {
    pub register_boost: usize,
    pub register_boost_four_step: usize,
    pub register_boost_non_power_of_two: bool,
    pub reorder_four_step: bool,
    pub swap_to_two_stage_four_step: usize,
    pub swap_to_three_stage_four_step: usize,
}

impl Default for NvidiaVulkanSchedulerTuning {
    fn default() -> Self {
        let policy = plan_gpu_scheduler_policy(
            Precision::F32,
            DeviceProfile::generic(Backend::Vulkan, GpuVendor::Nvidia),
        )
        .expect("NVIDIA Vulkan scheduler policy is always defined");
        Self {
            register_boost: policy.register_boost,
            register_boost_four_step: policy.register_boost_four_step,
            register_boost_non_power_of_two: policy.register_boost_non_power_of_two,
            reorder_four_step: policy.reorder_four_step,
            swap_to_two_stage_four_step: policy.swap_to_two_stage_four_step,
            swap_to_three_stage_four_step: policy.swap_to_three_stage_four_step,
        }
    }
}

pub const VKFFT_RADIX_TABLE_LEN: usize = 33;

/// Register/radix metadata corresponding to the non-Rader portion of
/// `VkFFTGetRegistersPerThread` + `VkFFTOptimizeRadixKernels`.
///
/// `stage_radices` is the optimized radix sequence that VkFFT would feed into
/// code generation after extracting the optional register-boost stage. The
/// current correctness-first Stockham kernel does not execute this sequence yet;
/// it is carried as typed scheduling metadata for the register-resident lowering.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RadixRegisterSchedule {
    pub fft_len: usize,
    pub rhs_transform_count: usize,
    pub register_boost: usize,
    pub registers_per_thread_per_radix: [usize; VKFFT_RADIX_TABLE_LEN],
    pub stage_radix_multipliers: [usize; VKFFT_RADIX_TABLE_LEN],
    pub stage_radices: Vec<usize>,
    pub register_boost_stage_radix: Option<usize>,
    pub registers_per_thread: usize,
    pub min_registers_per_thread: usize,
    pub is_good_sequence: bool,
    pub max_non_power_of_two_radix: usize,
    pub required_local_registers: usize,
}

impl RadixRegisterSchedule {
    pub fn validate(&self) -> Result<()> {
        if self.fft_len == 0
            || self.rhs_transform_count == 0
            || self.register_boost == 0
            || !self.register_boost.is_power_of_two()
            || self.register_boost >= VKFFT_RADIX_TABLE_LEN
            || self.registers_per_thread == 0
            || self.min_registers_per_thread == 0
            || self.registers_per_thread < self.min_registers_per_thread
            || self.max_non_power_of_two_radix == 0
            || self.required_local_registers == 0
        {
            return Err(VkFftError::InvalidKernelIr(
                "radix register schedule metadata is inconsistent",
            ));
        }
        let mut product = 1usize;
        for &radix in &self.stage_radices {
            if !(2..VKFFT_RADIX_TABLE_LEN).contains(&radix)
                || self.registers_per_thread_per_radix[radix] == 0
            {
                return Err(VkFftError::InvalidKernelIr(
                    "radix register schedule contains an unsupported stage",
                ));
            }
            product = product
                .checked_mul(radix)
                .ok_or(VkFftError::ArithmeticOverflow {
                    operation: "radix register stage product",
                })?;
        }
        if product != self.fft_len {
            return Err(VkFftError::InvalidKernelIr(
                "radix register stages do not cover the FFT length",
            ));
        }
        let mut multiplier_product = 1usize;
        for (radix, &count) in self.stage_radix_multipliers.iter().enumerate().skip(2) {
            if count == 0 {
                continue;
            }
            multiplier_product = multiplier_product
                .checked_mul(checked_pow(
                    radix,
                    count,
                    "radix register multiplier product",
                )?)
                .ok_or(VkFftError::ArithmeticOverflow {
                    operation: "radix register multiplier product",
                })?;
        }
        if let Some(radix) = self.register_boost_stage_radix {
            if !(2..VKFFT_RADIX_TABLE_LEN).contains(&radix) {
                return Err(VkFftError::InvalidKernelIr(
                    "register-boost stage radix is out of range",
                ));
            }
            multiplier_product =
                multiplier_product
                    .checked_mul(radix)
                    .ok_or(VkFftError::ArithmeticOverflow {
                        operation: "radix register boost-stage product",
                    })?;
        }
        if multiplier_product != self.fft_len {
            return Err(VkFftError::InvalidKernelIr(
                "optimized radix multipliers do not cover the FFT length",
            ));
        }
        Ok(())
    }
}

/// Per-stage lane ownership used by upstream FFT-Rader's optional shared-layout
/// transpose. `ContainerMajor` is the stage-0/non-transposed mapping, while
/// `TransposedContainerInterleaved` is used after stage 0 when `raderTranspose` is
/// enabled. The formulas are prime/radix agnostic; stage-specific logical group
/// sizes come from the optimized Rader register table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RaderFftStageLaneLayout {
    ContainerMajor,
    TransposedContainerInterleaved,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RaderFftStageLaneSchedule {
    pub stage_index: usize,
    pub radix: usize,
    pub logical_storage_per_thread: usize,
    pub sub_logical_group_size: usize,
    pub active_threads: usize,
    pub layout: RaderFftStageLaneLayout,
}

/// Typed form of the `raderTranspose` lane geometry from `vkFFT_RaderKernels.h`.
/// Every boost-1 smooth Rader container that satisfies the upstream gate can consume
/// this layout; resource and stage-geometry validation remain fail-soft.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RaderFftTransposeSchedule {
    pub container_fft_num: usize,
    pub container_fft_dim: usize,
    pub workgroup_threads: usize,
    pub stages: Vec<RaderFftStageLaneSchedule>,
}

impl RaderFftTransposeSchedule {
    pub fn validate(&self) -> Result<()> {
        if self.container_fft_num < 8
            || self.container_fft_dim == 0
            || self.workgroup_threads == 0
            || self.stages.len() < 2
        {
            return Err(VkFftError::InvalidKernelIr(
                "Rader transpose schedule metadata is inconsistent",
            ));
        }
        let mut maximum_threads = 0usize;
        for (index, stage) in self.stages.iter().enumerate() {
            let expected_layout = if index == 0 {
                RaderFftStageLaneLayout::ContainerMajor
            } else {
                RaderFftStageLaneLayout::TransposedContainerInterleaved
            };
            if stage.stage_index != index
                || stage.radix < 2
                || stage.logical_storage_per_thread == 0
                || stage.sub_logical_group_size == 0
                || stage.layout != expected_layout
                || stage.active_threads
                    != self
                        .container_fft_num
                        .checked_mul(stage.sub_logical_group_size)
                        .ok_or(VkFftError::ArithmeticOverflow {
                            operation: "Rader transpose stage thread count",
                        })?
            {
                return Err(VkFftError::InvalidKernelIr(
                    "Rader transpose stage metadata is inconsistent",
                ));
            }
            maximum_threads = maximum_threads.max(stage.active_threads);
        }
        if maximum_threads != self.workgroup_threads {
            return Err(VkFftError::InvalidKernelIr(
                "Rader transpose workgroup size does not match its stages",
            ));
        }
        Ok(())
    }

    /// Return `(raderIDx, raderIDx2)` for one active invocation of a stage, using
    /// exactly the two ownership formulas selected by upstream `raderTranspose`.
    /// Threads beyond a stage's active range are intentionally reported as inactive.
    pub fn lane_coordinates(
        &self,
        stage_index: usize,
        local_invocation: usize,
    ) -> Result<Option<(usize, usize)>> {
        self.validate()?;
        let stage = self
            .stages
            .get(stage_index)
            .ok_or(VkFftError::InvalidKernelIr(
                "Rader transpose stage index is out of range",
            ))?;
        if local_invocation >= stage.active_threads {
            return Ok(None);
        }
        Ok(Some(match stage.layout {
            RaderFftStageLaneLayout::ContainerMajor => (
                local_invocation % stage.sub_logical_group_size,
                local_invocation / stage.sub_logical_group_size,
            ),
            RaderFftStageLaneLayout::TransposedContainerInterleaved => (
                local_invocation / self.container_fft_num,
                local_invocation % self.container_fft_num,
            ),
        }))
    }
    /// Convert one logical `(raderIDx, raderIDx2)` pair back to the physical local
    /// invocation used by the stage. This is the inverse of `lane_coordinates` and
    /// lets subgroup proofs reason about the real transposed ownership rather than a
    /// fictitious contiguous container stripe.
    pub fn physical_invocation(
        &self,
        stage_index: usize,
        rader_idx: usize,
        container: usize,
    ) -> Result<usize> {
        self.validate()?;
        let stage = self
            .stages
            .get(stage_index)
            .ok_or(VkFftError::InvalidKernelIr(
                "Rader transpose stage index is out of range",
            ))?;
        if rader_idx >= stage.sub_logical_group_size || container >= self.container_fft_num {
            return Err(VkFftError::InvalidKernelIr(
                "Rader transpose logical lane/container is out of range",
            ));
        }
        let physical = match stage.layout {
            RaderFftStageLaneLayout::ContainerMajor => container
                .checked_mul(stage.sub_logical_group_size)
                .and_then(|base| base.checked_add(rader_idx)),
            RaderFftStageLaneLayout::TransposedContainerInterleaved => rader_idx
                .checked_mul(self.container_fft_num)
                .and_then(|base| base.checked_add(container)),
        }
        .ok_or(VkFftError::ArithmeticOverflow {
            operation: "Rader transpose physical invocation",
        })?;
        if physical >= stage.active_threads {
            return Err(VkFftError::InvalidKernelIr(
                "Rader transpose physical invocation exceeds the active stage",
            ));
        }
        Ok(physical)
    }
}

/// Register/occupancy plan for the internal `(p - 1)` FFT of an FFT-convolution
/// Rader prime.
///
/// `container_fft_num` and `min_rader_fft_thread_num` use upstream VkFFT's names:
/// `VkFFTConstructRaderTree` records how many Rader containers belong to one outer
/// FFT workgroup, and `VkFFTGetRaderFFTThreadsNum` computes the minimum local thread
/// count needed to execute all of them together. `execution_*` records the physical
/// Rust launch shape. The first executable grouped slice is the two-container
/// 256-point convolution used by `514 = 2 * 257`; unsupported container shapes keep
/// the conservative one-container launch visible in the same metadata. The first
/// executable transposed slice is eight p257 `[16,16]` containers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RaderFftRegisterSchedule {
    pub prime: usize,
    pub convolution_len: usize,
    pub outer_fft_len: usize,
    pub container_fft_num: usize,
    pub min_rader_fft_thread_num: usize,
    pub execution_container_fft_num: usize,
    pub execution_workgroup_count: usize,
    pub execution_threads_per_workgroup: usize,
    /// Present exactly when upstream enables `raderTranspose` for this contiguous
    /// container: `containerFFTNum >= 8 && numStages > 1`.
    pub rader_transpose: Option<RaderFftTransposeSchedule>,
    pub internal_fft: RadixRegisterSchedule,
}

impl RaderFftRegisterSchedule {
    pub fn validate(&self) -> Result<()> {
        self.internal_fft.validate()?;
        if self.prime < 3
            || self.convolution_len + 1 != self.prime
            || self.container_fft_num == 0
            || self.execution_container_fft_num == 0
            || self.execution_container_fft_num > self.container_fft_num
            || self.min_rader_fft_thread_num == 0
            || self.execution_threads_per_workgroup == 0
            || self.execution_workgroup_count == 0
            || self.internal_fft.fft_len != self.convolution_len
            || self.internal_fft.register_boost != 1
            || self.internal_fft.register_boost_stage_radix.is_some()
            || self.internal_fft.stage_radices.is_empty()
            || self.prime.checked_mul(self.container_fft_num) != Some(self.outer_fft_len)
            || !self
                .internal_fft
                .rhs_transform_count
                .is_multiple_of(self.container_fft_num)
            || !self
                .internal_fft
                .rhs_transform_count
                .is_multiple_of(self.execution_container_fft_num)
        {
            return Err(VkFftError::InvalidKernelIr(
                "Rader FFT register schedule metadata is inconsistent",
            ));
        }
        let local_threads = rader_fft_threads_per_container(&self.internal_fft)?;
        let upstream_threads = self.container_fft_num.checked_mul(local_threads).ok_or(
            VkFftError::ArithmeticOverflow {
                operation: "Rader FFT upstream thread estimate",
            },
        )?;
        let execution_threads = self
            .execution_container_fft_num
            .checked_mul(local_threads)
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "Rader FFT execution thread estimate",
            })?;
        let execution_workgroups =
            self.internal_fft.rhs_transform_count / self.execution_container_fft_num;
        if upstream_threads != self.min_rader_fft_thread_num
            || execution_threads != self.execution_threads_per_workgroup
            || execution_workgroups != self.execution_workgroup_count
        {
            return Err(VkFftError::InvalidKernelIr(
                "Rader FFT thread/workgroup estimates do not match the register schedule",
            ));
        }
        let transpose_expected =
            self.container_fft_num >= 8 && self.internal_fft.stage_radices.len() > 1;
        match &self.rader_transpose {
            Some(transpose) if transpose_expected => {
                transpose.validate()?;
                if transpose.container_fft_num != self.container_fft_num
                    || transpose.container_fft_dim != self.convolution_len
                    || transpose.workgroup_threads != self.min_rader_fft_thread_num
                    || transpose.stages.len() != self.internal_fft.stage_radices.len()
                    || transpose
                        .stages
                        .iter()
                        .zip(&self.internal_fft.stage_radices)
                        .any(|(stage, radix)| stage.radix != *radix)
                {
                    return Err(VkFftError::InvalidKernelIr(
                        "Rader transpose schedule does not match the internal FFT",
                    ));
                }
            }
            None if !transpose_expected => {}
            _ => {
                return Err(VkFftError::InvalidKernelIr(
                    "Rader transpose presence does not match the upstream gate",
                ));
            }
        }
        Ok(())
    }

    pub fn upstream_workgroup_count(&self) -> usize {
        self.internal_fft.rhs_transform_count / self.container_fft_num
    }

    pub fn upstream_grouping_is_executable(&self) -> bool {
        self.execution_container_fft_num == self.container_fft_num
    }
}

/// Plan the policy-independent FFT-Rader register/container arithmetic for a GPU
/// profile. This accepts smooth `p - 1` factorizations covered completely by VkFFT's
/// small-radix container schedule; backend-specific limits enter through `DeviceProfile`.
pub fn plan_gpu_rader_fft_registers(
    prime: usize,
    batch_count: usize,
    device: DeviceProfile,
) -> Result<RaderFftRegisterSchedule> {
    plan_gpu_rader_fft_registers_for_containers(prime, batch_count, prime, 1, device)
}

/// Port the first multi-container slice of `VkFFTConstructRaderTree`,
/// `VkFFTOptimizeRaderFFTRegisters`, and `VkFFTGetRaderFFTThreadsNum`.
///
/// The covered outer factor is any Stockham-smooth `container_fft_num` with
/// `outer_fft_len == prime * container_fft_num`. Power-of-two and mixed-radix outer
/// register states use the shared `VkFFTGetRegistersPerThread` arithmetic before
/// coupling to the internal Rader optimize-shared table.
/// Before upstream's `containerFFTNum >= 8 && numStages > 1` transpose gate, every
/// covered boost-1 schedule can execute its containers as independent shared stripes.
/// At the gate, the typed upstream lane geometry is used for every smooth schedule
/// that fits the physical workgroup; shared-memory limits are validated when the
/// schedule is attached to the kernel.
pub fn plan_gpu_rader_fft_registers_for_containers(
    prime: usize,
    batch_count: usize,
    outer_fft_len: usize,
    container_fft_num: usize,
    device: DeviceProfile,
) -> Result<RaderFftRegisterSchedule> {
    if device.backend == Backend::CpuReference {
        return Err(VkFftError::UnsupportedKernelPath(
            "Rader FFT register scheduler requires a GPU DeviceProfile",
        ));
    }
    if prime < 3 {
        return Err(VkFftError::UnsupportedKernelPath(
            "Rader FFT register scheduler requires a prime length of at least three",
        ));
    }
    if batch_count == 0 {
        return Err(VkFftError::ZeroBatchCount);
    }
    if container_fft_num == 0
        || outer_fft_len
            != prime
                .checked_mul(container_fft_num)
                .ok_or(VkFftError::ArithmeticOverflow {
                    operation: "Rader outer FFT/container product",
                })?
        || !batch_count.is_multiple_of(container_fft_num)
    {
        return Err(VkFftError::UnsupportedKernelPath(
            "Rader outer-register coupling requires consistent container dimensions",
        ));
    }
    let convolution_len = prime - 1;
    let (mut registers, _, _) = rader_optimize_shared_register_table(convolution_len)?;
    let outer_batch_count = batch_count / container_fft_num;
    let _first_optimizer_min_registers = optimize_single_rader_container_registers(
        outer_fft_len,
        container_fft_num,
        container_fft_num,
        outer_batch_count,
        &mut registers,
    )?;
    // Fixed upstream runs VkFFTOptimizeRaderFFTRegisters twice around the global
    // register-harmonization phase. For a single type-0 container the first pass
    // already carries the harmonized table into the second call; re-evaluating the
    // outer state from the same physical dimensions reproduces the second-pass
    // container promotion without introducing a separate standalone state machine.
    let _second_optimizer_min_registers = optimize_single_rader_container_registers(
        outer_fft_len,
        container_fft_num,
        container_fft_num,
        outer_batch_count,
        &mut registers,
    )?;

    let mut multipliers = rader_small_radix_multipliers(convolution_len)?;
    let (max_non_power_of_two_radix, required_local_registers) =
        optimize_radix_kernels(&mut registers, &mut multipliers, 1);
    let mut stage_radices = Vec::new();
    for radix in (2..VKFFT_RADIX_TABLE_LEN).rev() {
        stage_radices.extend(core::iter::repeat_n(radix, multipliers[radix]));
    }
    scale_rader_stage_registers_to_thread_limit(
        convolution_len,
        container_fft_num,
        device.max_threads_per_block,
        &stage_radices,
        &mut registers,
    )?;
    let (registers_per_thread, min_registers_per_thread) =
        min_max_stage_registers(&stage_radices, &registers);
    if registers_per_thread == 0 || min_registers_per_thread == 0 {
        return Err(VkFftError::InvalidKernelIr(
            "Rader FFT register scheduler produced no executable stages",
        ));
    }
    let internal_fft = RadixRegisterSchedule {
        fft_len: convolution_len,
        rhs_transform_count: batch_count,
        register_boost: 1,
        registers_per_thread_per_radix: registers,
        stage_radix_multipliers: multipliers,
        stage_radices,
        register_boost_stage_radix: None,
        registers_per_thread,
        min_registers_per_thread,
        is_good_sequence: !(registers_per_thread > 16
            || registers_per_thread >= 2 * min_registers_per_thread),
        max_non_power_of_two_radix,
        required_local_registers,
    };
    internal_fft.validate()?;
    let local_threads = rader_fft_threads_per_container(&internal_fft)?;
    let min_rader_fft_thread_num =
        container_fft_num
            .checked_mul(local_threads)
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "Rader FFT multi-container thread estimate",
            })?;
    if min_rader_fft_thread_num > device.max_threads_per_block {
        return Err(VkFftError::ResourceLimitExceeded {
            resource: "upstream-grouped Rader FFT threads per workgroup",
            required: min_rader_fft_thread_num,
            available: device.max_threads_per_block,
        });
    }

    let rader_transpose =
        build_rader_fft_transpose_schedule(container_fft_num, convolution_len, &internal_fft)?;

    // Before the transpose gate, grouped Stockham execution isolates every container
    // in its own shared stripe. At the gate, the upstream transposed lane geometry is
    // executable whenever its exact physical workgroup fits the device.
    let grouped_shape_supported = rader_transpose.is_none();
    let transposed_shape_supported = rader_transpose
        .as_ref()
        .is_some_and(|transpose| transpose.workgroup_threads <= device.max_threads_per_block);
    let execution_container_fft_num = if grouped_shape_supported || transposed_shape_supported {
        container_fft_num
    } else {
        1
    };
    let execution_workgroup_count = batch_count / execution_container_fft_num;
    let execution_threads_per_workgroup = execution_container_fft_num
        .checked_mul(local_threads)
        .ok_or(VkFftError::ArithmeticOverflow {
            operation: "Rader FFT grouped execution thread estimate",
        })?;

    let schedule = RaderFftRegisterSchedule {
        prime,
        convolution_len,
        outer_fft_len,
        container_fft_num,
        min_rader_fft_thread_num,
        execution_container_fft_num,
        execution_workgroup_count,
        execution_threads_per_workgroup,
        rader_transpose,
        internal_fft,
    };
    schedule.validate()?;
    Ok(schedule)
}

/// Compatibility wrapper preserving the original NVIDIA/Vulkan API contract.
pub fn plan_nvidia_vulkan_rader_fft_registers(
    prime: usize,
    batch_count: usize,
    device: DeviceProfile,
) -> Result<RaderFftRegisterSchedule> {
    plan_nvidia_vulkan_rader_fft_registers_for_containers(prime, batch_count, prime, 1, device)
}

/// Compatibility wrapper preserving the original NVIDIA/Vulkan API contract.
pub fn plan_nvidia_vulkan_rader_fft_registers_for_containers(
    prime: usize,
    batch_count: usize,
    outer_fft_len: usize,
    container_fft_num: usize,
    device: DeviceProfile,
) -> Result<RaderFftRegisterSchedule> {
    if device.backend != Backend::Vulkan || device.vendor != GpuVendor::Nvidia {
        return Err(VkFftError::UnsupportedKernelPath(
            "Rader FFT register scheduler slice requires an NVIDIA Vulkan DeviceProfile",
        ));
    }
    plan_gpu_rader_fft_registers_for_containers(
        prime,
        batch_count,
        outer_fft_len,
        container_fft_num,
        device,
    )
}

fn build_rader_fft_transpose_schedule(
    container_fft_num: usize,
    container_fft_dim: usize,
    internal_fft: &RadixRegisterSchedule,
) -> Result<Option<RaderFftTransposeSchedule>> {
    if container_fft_num < 8 || internal_fft.stage_radices.len() <= 1 {
        return Ok(None);
    }
    let mut stages = Vec::with_capacity(internal_fft.stage_radices.len());
    let mut workgroup_threads = 0usize;
    for (stage_index, &radix) in internal_fft.stage_radices.iter().enumerate() {
        let logical_storage_per_thread = internal_fft.registers_per_thread_per_radix[radix]
            .checked_mul(internal_fft.register_boost)
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "Rader transpose logical storage per thread",
            })?;
        if logical_storage_per_thread == 0 {
            return Err(VkFftError::InvalidKernelIr(
                "Rader transpose stage has no register storage",
            ));
        }
        let sub_logical_group_size = ceil_div(container_fft_dim, logical_storage_per_thread)?;
        let active_threads = container_fft_num
            .checked_mul(sub_logical_group_size)
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "Rader transpose active thread count",
            })?;
        workgroup_threads = workgroup_threads.max(active_threads);
        stages.push(RaderFftStageLaneSchedule {
            stage_index,
            radix,
            logical_storage_per_thread,
            sub_logical_group_size,
            active_threads,
            layout: if stage_index == 0 {
                RaderFftStageLaneLayout::ContainerMajor
            } else {
                RaderFftStageLaneLayout::TransposedContainerInterleaved
            },
        });
    }
    let schedule = RaderFftTransposeSchedule {
        container_fft_num,
        container_fft_dim,
        workgroup_threads,
        stages,
    };
    schedule.validate()?;
    Ok(Some(schedule))
}

fn rader_outer_register_state(
    outer_fft_len: usize,
    container_fft_num: usize,
    outer_batch_count: usize,
) -> Result<([usize; VKFFT_RADIX_TABLE_LEN], usize, usize)> {
    if container_fft_num == 1 || container_fft_num.is_power_of_two() {
        return rader_outer_power_of_two_register_state(
            outer_fft_len,
            container_fft_num,
            outer_batch_count,
        );
    }
    if outer_fft_len == 0 || outer_batch_count == 0 {
        return Err(VkFftError::UnsupportedKernelPath(
            "Rader outer register state requires valid dimensions",
        ));
    }

    // VkFFTConstructRaderTree permits any Stockham-smooth outer container product,
    // not only 2^k. VkFFTGetRegistersPerThread uses one shared radix-register table
    // across GPU APIs; backend/vendor policy is applied by the surrounding upload
    // and occupancy scheduler. Couple the outer stage extrema to the inner Rader
    // table in VkFFTOptimizeRaderFFTRegisters below.
    let outer = plan_gpu_small_mixed_radix_registers(container_fft_num, outer_batch_count)?;
    let (max_registers, min_registers) =
        min_max_stage_registers(&outer.stage_radices, &outer.registers_per_thread_per_radix);
    if max_registers == 0 || min_registers == 0 {
        return Err(VkFftError::InvalidKernelIr(
            "Rader outer smooth register state has no active stages",
        ));
    }
    Ok((
        outer.registers_per_thread_per_radix,
        max_registers,
        min_registers,
    ))
}

fn rader_outer_power_of_two_register_state(
    outer_fft_len: usize,
    container_fft_num: usize,
    outer_batch_count: usize,
) -> Result<([usize; VKFFT_RADIX_TABLE_LEN], usize, usize)> {
    if container_fft_num == 1 {
        return Ok(([0; VKFFT_RADIX_TABLE_LEN], 2, 2));
    }
    if !container_fft_num.is_power_of_two() || outer_fft_len == 0 || outer_batch_count == 0 {
        return Err(VkFftError::UnsupportedKernelPath(
            "Rader outer power-of-two register state requires valid dimensions",
        ));
    }

    // This is the pure-radix-2 branch of VkFFTGetRegistersPerThread with
    // `useRader != 0`: the radix multiplicity belongs to the smooth outer factor,
    // while stage-count scoring still sees the full `A * p` upload length.
    let exponent = container_fft_num.trailing_zeros() as usize;
    let (floor_log2_fft_len, ceil_log2_fft_len) = exact_log2_bounds(outer_fft_len);
    let active_threads_y = (outer_batch_count / 64).max(1);
    let mut test_min_stages = usize::MAX;
    let mut max_radix_min_stages = 1usize;
    for candidate in 1..=3usize {
        // Upstream scores ceil(log2(N) / candidate). For integer N this is
        // exactly ceil(ceil(log2(N)) / candidate), so keep the decision in
        // integer arithmetic rather than routing launch metadata through f64.
        let stages = ceil_div(ceil_log2_fft_len, candidate)?;
        if stages < test_min_stages {
            test_min_stages = stages;
            max_radix_min_stages = candidate;
        }
    }
    let mut max_loc_multipliers_pow2 = 0usize;
    for candidate in (1..=max_radix_min_stages).rev() {
        let active_threads_x =
            active_threads_y
                .checked_mul(outer_fft_len)
                .ok_or(VkFftError::ArithmeticOverflow {
                    operation: "Rader outer active thread estimate",
                })?
                / (1usize << candidate);
        if active_threads_x >= 128 {
            max_loc_multipliers_pow2 = candidate;
            break;
        }
    }
    max_loc_multipliers_pow2 = max_loc_multipliers_pow2.max(3);

    let truncated_log2 = floor_log2_fft_len;
    let mut final_loc_multipliers_pow2 = 1usize;
    let mut num_stages_min = truncated_log2;
    for candidate in 2..=max_loc_multipliers_pow2 {
        let stages = ceil_div(truncated_log2, candidate)?;
        if stages < num_stages_min {
            final_loc_multipliers_pow2 = candidate;
            num_stages_min = stages;
        }
    }

    let mut registers = [0usize; VKFFT_RADIX_TABLE_LEN];
    let register_exponent = exponent.min(final_loc_multipliers_pow2);
    registers[2] = 1usize << register_exponent;
    if exponent < 3 {
        registers[2] = 1usize << exponent;
    }
    registers[32] = if registers[2].is_multiple_of(32) {
        registers[2]
    } else {
        0
    };
    registers[16] = if registers[2].is_multiple_of(16) {
        registers[2]
    } else {
        0
    };
    registers[8] = if registers[2].is_multiple_of(8) {
        registers[2]
    } else {
        0
    };
    registers[4] = if registers[2].is_multiple_of(4) {
        registers[2]
    } else {
        0
    };

    let mut max_registers = 0usize;
    let mut min_registers = usize::MAX;
    for value in registers.iter().copied().filter(|value| *value != 0) {
        max_registers = max_registers.max(value);
        min_registers = min_registers.min(value);
    }
    if min_registers == usize::MAX {
        min_registers = 2;
        max_registers = 2;
    }
    Ok((registers, max_registers, min_registers))
}

fn optimize_single_rader_container_registers(
    outer_fft_len: usize,
    smooth_outer_factor: usize,
    container_fft_num: usize,
    outer_batch_count: usize,
    rader_registers: &mut [usize; VKFFT_RADIX_TABLE_LEN],
) -> Result<usize> {
    // `smooth_outer_factor` is VkFFT's loc_multipliers product after *all* Rader
    // primes have been removed. `container_fft_num = fftDim / prime` is different in
    // mixed Direct+FFT-Rader axes because it still includes the other Rader primes.
    let (outer_registers_per_radix, outer_registers, outer_min_registers) =
        rader_outer_register_state(outer_fft_len, smooth_outer_factor, outer_batch_count)?;
    optimize_single_rader_container_registers_from_outer_state(
        outer_fft_len,
        container_fft_num,
        outer_registers_per_radix,
        outer_registers,
        outer_min_registers,
        rader_registers,
    )
}

fn optimize_single_rader_container_registers_from_outer_state(
    outer_fft_len: usize,
    container_fft_num: usize,
    mut outer_registers_per_radix: [usize; VKFFT_RADIX_TABLE_LEN],
    mut outer_registers: usize,
    mut outer_min_registers: usize,
    rader_registers: &mut [usize; VKFFT_RADIX_TABLE_LEN],
) -> Result<usize> {
    if outer_fft_len == 0
        || container_fft_num == 0
        || outer_registers == 0
        || outer_min_registers == 0
    {
        return Err(VkFftError::InvalidKernelIr(
            "Rader optimizer outer register state must be non-zero",
        ));
    }

    let (_, mut rader_registers_per_thread, mut rader_min_registers) =
        rader_register_min_max(rader_registers)?;
    if rader_min_registers / outer_min_registers >= 2 {
        outer_min_registers *= rader_min_registers / outer_min_registers;
        for value in &mut outer_registers_per_radix {
            if *value > 0 && *value < outer_min_registers {
                *value *= ceil_div(outer_min_registers, *value)?;
            }
            outer_registers = outer_registers.max(*value);
        }
    } else if outer_min_registers / rader_min_registers >= 2 {
        rader_min_registers *= outer_min_registers / rader_min_registers;
        for value in rader_registers.iter_mut() {
            if *value > 0 && *value < rader_min_registers {
                *value *= ceil_div(rader_min_registers, *value)?;
            }
            rader_registers_per_thread = rader_registers_per_thread.max(*value);
        }
    }

    if rader_min_registers < outer_min_registers {
        for (radix, value) in rader_registers.iter_mut().enumerate().skip(2) {
            if *value == 0 {
                continue;
            }
            while *value < outer_min_registers {
                *value = (*value)
                    .checked_add(radix)
                    .ok_or(VkFftError::ArithmeticOverflow {
                        operation: "Rader register alignment",
                    })?;
            }
            rader_registers_per_thread = rader_registers_per_thread.max(*value);
        }
    }

    // The actual type-0 container count, not the smooth outer factor, recovers the
    // Rader prime and controls occupancy.
    if container_fft_num == 0 || !outer_fft_len.is_multiple_of(container_fft_num) {
        return Err(VkFftError::UnsupportedKernelPath(
            "Rader optimizer requires an exact physical container count",
        ));
    }
    let prime = outer_fft_len / container_fft_num;
    let container_fft_dim = prime - 1;

    for (radix, value) in rader_registers.iter_mut().enumerate().skip(2) {
        if *value == 0 {
            continue;
        }
        loop {
            let outer_threads = ceil_div(outer_fft_len, outer_min_registers)?;
            if rader_container_occupancy_fits(
                outer_threads,
                container_fft_num,
                container_fft_dim,
                *value,
            )? {
                break;
            }
            *value = (*value)
                .checked_add(radix)
                .ok_or(VkFftError::ArithmeticOverflow {
                    operation: "Rader occupancy register scaling",
                })?;
        }
        rader_registers_per_thread = rader_registers_per_thread.max(*value);
    }
    outer_registers = outer_registers.max(rader_registers_per_thread);

    // Upstream's second loop raises every Rader radix toward the global maximum.
    for (radix, value) in rader_registers.iter_mut().enumerate().skip(2) {
        if *value == 0 {
            continue;
        }
        while (*value)
            .checked_add(radix)
            .is_some_and(|next| next <= outer_registers + 1)
        {
            *value += radix;
        }
    }
    // The final optimizer loop keeps the global minimum at the lower of the ordinary
    // outer state and each type-0 container's optimized minimum.
    let (_, _, final_rader_min_registers) = rader_register_min_max(rader_registers)?;
    Ok(outer_min_registers.min(final_rader_min_registers))
}

fn rader_container_occupancy_fits(
    outer_threads: usize,
    container_fft_num: usize,
    container_fft_dim: usize,
    registers: usize,
) -> Result<bool> {
    if outer_threads == 0 || container_fft_num == 0 || container_fft_dim == 0 || registers == 0 {
        return Err(VkFftError::InvalidKernelIr(
            "Rader occupancy comparison requires non-zero dimensions",
        ));
    }
    if registers < container_fft_dim {
        let scaling = ceil_div(container_fft_dim, registers)?;
        let required =
            container_fft_num
                .checked_mul(scaling)
                .ok_or(VkFftError::ArithmeticOverflow {
                    operation: "Rader container occupancy",
                })?;
        Ok(outer_threads >= required)
    } else {
        // Upstream uses scaling = 1 / floor(registers / containerFFTDim) here.
        // Compare the rational inequality without floating point:
        // outer_threads >= containerFFTNum / packed_containers.
        let packed_containers = registers / container_fft_dim;
        let capacity =
            outer_threads
                .checked_mul(packed_containers)
                .ok_or(VkFftError::ArithmeticOverflow {
                    operation: "Rader packed-container occupancy",
                })?;
        Ok(capacity >= container_fft_num)
    }
}

fn rader_register_min_max(
    registers: &[usize; VKFFT_RADIX_TABLE_LEN],
) -> Result<(usize, usize, usize)> {
    let mut count = 0usize;
    let mut max_registers = 0usize;
    let mut min_registers = usize::MAX;
    for value in registers.iter().copied().filter(|value| *value != 0) {
        count += 1;
        max_registers = max_registers.max(value);
        min_registers = min_registers.min(value);
    }
    if count == 0 || min_registers == usize::MAX {
        return Err(VkFftError::InvalidKernelIr(
            "Rader register table has no supported radix",
        ));
    }
    Ok((count, max_registers, min_registers))
}

fn scale_rader_stage_registers_to_thread_limit(
    convolution_len: usize,
    container_fft_num: usize,
    max_threads_per_block: usize,
    stage_radices: &[usize],
    registers: &mut [usize; VKFFT_RADIX_TABLE_LEN],
) -> Result<()> {
    if convolution_len == 0 || container_fft_num == 0 || max_threads_per_block == 0 {
        return Err(VkFftError::InvalidKernelIr(
            "Rader thread-limit scaling requires non-zero dimensions",
        ));
    }
    if container_fft_num > max_threads_per_block {
        return Err(VkFftError::ResourceLimitExceeded {
            resource: "Rader container count per workgroup",
            required: container_fft_num,
            available: max_threads_per_block,
        });
    }
    let per_container_limit = max_threads_per_block / container_fft_num;
    for &radix in stage_radices {
        let value = registers.get_mut(radix).ok_or(VkFftError::InvalidKernelIr(
            "Rader stage radix exceeds the register table",
        ))?;
        if *value == 0 || !(*value).is_multiple_of(radix) {
            return Err(VkFftError::InvalidKernelIr(
                "Rader thread-limit scaling found an invalid register count",
            ));
        }
        while ceil_div(convolution_len, *value)? > per_container_limit {
            *value = (*value)
                .checked_add(radix)
                .ok_or(VkFftError::ArithmeticOverflow {
                    operation: "Rader thread-limit register scaling",
                })?;
        }
    }
    Ok(())
}

fn rader_fft_threads_per_container(schedule: &RadixRegisterSchedule) -> Result<usize> {
    schedule
        .stage_radices
        .iter()
        .try_fold(0usize, |current, &radix| {
            let registers = schedule.registers_per_thread_per_radix[radix];
            if registers == 0 {
                return Err(VkFftError::InvalidKernelIr(
                    "Rader FFT stage is missing its register count",
                ));
            }
            Ok(current.max(ceil_div(schedule.fft_len, registers)?))
        })
}

fn rader_small_radix_multipliers_with_residual(
    fft_len: usize,
    min_rader_direct_prime: usize,
) -> Result<([usize; VKFFT_RADIX_TABLE_LEN], usize)> {
    if fft_len < 2 {
        return Err(VkFftError::UnsupportedKernelPath(
            "Rader small-radix factorization requires an FFT length of at least two",
        ));
    }
    let mut remaining = fft_len;
    let mut multipliers = [0usize; VKFFT_RADIX_TABLE_LEN];
    // VkFFTConstructRaderTree peels every factor below fixMinRaderPrimeMult before
    // recursively constructing the residual sub-prime tree. Composite radices are
    // intentionally visited as upstream does; prime factors have already been removed
    // by the time their composite multiples are reached.
    for (radix, multiplier) in multipliers
        .iter_mut()
        .enumerate()
        .take(min_rader_direct_prime)
        .skip(2)
    {
        while remaining.is_multiple_of(radix) {
            remaining /= radix;
            *multiplier += 1;
        }
    }
    Ok((multipliers, remaining))
}

fn rader_small_radix_multipliers(fft_len: usize) -> Result<[usize; VKFFT_RADIX_TABLE_LEN]> {
    let (multipliers, remaining) = rader_small_radix_multipliers_with_residual(fft_len, 17)?;
    if remaining != 1 {
        return Err(VkFftError::UnsupportedKernelPath(
            "Rader convolution still requires nested/sub-prime scheduling",
        ));
    }
    Ok(multipliers)
}

fn rader_optimize_shared_register_table(
    fft_len: usize,
) -> Result<([usize; VKFFT_RADIX_TABLE_LEN], usize, usize)> {
    if fft_len < 2 {
        return Err(VkFftError::UnsupportedKernelPath(
            "Rader optimize-shared register table requires an FFT length of at least two",
        ));
    }

    // Direct port of VkFFTGetRegistersPerThreadOptimizeShared's factor search. A
    // subtle but important upstream property is that failure to cover the whole FFT
    // dimension does not make this routine fail: the final attempted stage list is
    // retained. VkFFTConstructRaderTree uses exactly that partial small-radix table for
    // a parent such as 106=2*53, while p53 is represented by a nested Rader container.
    let mut selected = Vec::new();
    for stage_count in 1usize..20 {
        let mut remaining = fft_len;
        let mut stages = Vec::with_capacity(stage_count);
        let minimum = upstream_floor_nth_root(remaining, stage_count);
        if minimum > 16 {
            continue;
        }
        let mut candidate = minimum.max(1);
        while candidate <= 16 && stages.len() < stage_count {
            if remaining.is_multiple_of(candidate) {
                remaining /= candidate;
                stages.push(candidate);
                let slots_left = stage_count - stages.len();
                if slots_left == 0 {
                    break;
                }
                candidate = upstream_floor_nth_root(remaining, slots_left).max(1);
            } else {
                candidate += 1;
            }
        }
        selected = stages;
        if remaining == 1 && selected.len() == stage_count {
            break;
        }
    }
    if selected.is_empty() {
        return Err(VkFftError::UnsupportedKernelPath(
            "VkFFT optimize-shared search produced no usable Rader stage candidates",
        ));
    }

    let mut registers = [0usize; VKFFT_RADIX_TABLE_LEN];
    for stage in selected {
        for (radix, registers_for_radix) in registers
            .iter_mut()
            .enumerate()
            .take(stage.min(VKFFT_RADIX_TABLE_LEN - 1) + 1)
            .skip(2)
        {
            if stage.is_multiple_of(radix) {
                *registers_for_radix = (*registers_for_radix).max(stage);
            }
        }
    }
    let mut registers_per_thread = registers.iter().copied().max().unwrap_or(0);
    if registers_per_thread == 0 {
        return Err(VkFftError::InvalidKernelIr(
            "Rader optimize-shared search produced an empty register table",
        ));
    }

    // Match VkFFT's second pass, which scales each supported radix register count to
    // the nearest multiple around the global maximum.
    for value in &mut registers {
        if *value == 0 {
            continue;
        }
        *value *= nearest_register_multiple_multiplier(registers_per_thread, *value);
    }

    registers_per_thread = 0;
    let mut min_registers_per_thread = usize::MAX;
    for value in registers.iter().copied().filter(|value| *value != 0) {
        registers_per_thread = registers_per_thread.max(value);
        min_registers_per_thread = min_registers_per_thread.min(value);
    }
    if min_registers_per_thread == usize::MAX {
        return Err(VkFftError::InvalidKernelIr(
            "Rader optimize-shared register table has no supported radix",
        ));
    }
    Ok((registers, registers_per_thread, min_registers_per_thread))
}

/// Upstream `forceRaderTwoUpload` policy. FFT-Rader axes are prevented from staying
/// in a single upload when any Rader prime would be replicated across more than 512
/// outer containers, or across more containers than the physical thread limit.
pub fn plan_gpu_force_rader_two_upload(
    sequence_len: usize,
    fft_rader_primes: &[usize],
    device: DeviceProfile,
) -> Result<bool> {
    if sequence_len == 0 || device.max_threads_per_block == 0 {
        return Err(VkFftError::InvalidKernelIr(
            "Rader upload pressure requires non-zero dimensions",
        ));
    }
    if device.backend == Backend::CpuReference {
        return Ok(false);
    }
    for &prime in fft_rader_primes {
        if prime < 2 || !sequence_len.is_multiple_of(prime) {
            return Err(VkFftError::InvalidKernelIr(
                "FFT-Rader prime does not divide the scheduled axis",
            ));
        }
        let containers = sequence_len / prime;
        if containers > 512 || containers > device.max_threads_per_block {
            return Ok(true);
        }
    }
    Ok(false)
}

/// Compatibility wrapper preserving the original NVIDIA/Vulkan policy surface.
pub fn plan_nvidia_vulkan_force_rader_two_upload(
    sequence_len: usize,
    fft_rader_primes: &[usize],
    device: DeviceProfile,
) -> Result<bool> {
    if device.backend != Backend::Vulkan || device.vendor != GpuVendor::Nvidia {
        return Ok(false);
    }
    plan_gpu_force_rader_two_upload(sequence_len, fft_rader_primes, device)
}

/// Why a non-power-of-two Rader axis requires multiple physical uploads.
///
/// Upstream computes ordinary capacity/bandwidth `numPasses` first and only applies
/// `forceRaderTwoUpload` when that ordinary result is still one pass. Keep those two
/// decisions explicit so capacity-driven splits are not mislabeled as forced Rader.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RaderUploadReason {
    CapacityOrBandwidth,
    ForcedRaderPressure,
}

/// Exact fixed-policy geometry for a non-power-of-two Rader axis that requires two
/// or three uploads. This is deliberately separate from `StockhamUploadSchedule`:
/// one split factor may itself contain a Rader prime.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RaderUploadSchedule {
    pub sequence_len: usize,
    pub used_shared_memory_bytes: usize,
    pub max_sequence_len_shared: usize,
    pub max_sequence_len_strided: usize,
    pub upload_count: usize,
    pub axis_split: Vec<usize>,
    pub reason: RaderUploadReason,
}

impl RaderUploadSchedule {
    pub fn validate(&self) -> Result<()> {
        if self.sequence_len == 0
            || self.used_shared_memory_bytes == 0
            || self.max_sequence_len_shared == 0
            || self.max_sequence_len_strided == 0
            || !matches!(self.upload_count, 2 | 3)
            || self.axis_split.len() != self.upload_count
            || self.axis_split.contains(&0)
        {
            return Err(VkFftError::InvalidKernelIr(
                "Rader upload schedule metadata is inconsistent",
            ));
        }
        let product = self.axis_split.iter().try_fold(1usize, |product, factor| {
            product
                .checked_mul(*factor)
                .ok_or(VkFftError::ArithmeticOverflow {
                    operation: "Rader upload axis split product",
                })
        })?;
        if product != self.sequence_len {
            return Err(VkFftError::InvalidKernelIr(
                "Rader upload split does not cover the sequence",
            ));
        }
        Ok(())
    }
}

/// Backward-compatible name retained while IR/backend call sites migrate to the
/// generic upload-schedule terminology.
pub type RaderForcedUploadSchedule = RaderUploadSchedule;

/// Materialize the fixed upstream divisor search after `forceRaderTwoUpload` has
/// raised a non-power-of-two axis to at least two passes.
///
/// The covered fixed GPU policies all use `reorderFourStep=1`,
/// `registerBoost4Step=1`, and disable non-power-of-two register boost. Therefore
/// every split factor is bounded by the final strided shared-memory capacity. The
/// scheduler first tries the square-root two-upload split and, when that cannot fit,
/// promotes to the same cube-root/square-root three-upload search used by the smooth
/// non-power-of-two branch.
pub fn plan_gpu_rader_upload_split(
    sequence_len: usize,
    fft_rader_primes: &[usize],
    direct_rader_primes: &[usize],
    precision: Precision,
    device: DeviceProfile,
) -> Result<Option<RaderForcedUploadSchedule>> {
    plan_gpu_rader_upload_split_with_axis_context(
        sequence_len,
        fft_rader_primes,
        direct_rader_primes,
        precision,
        device,
        StockhamUploadAxisContext::default(),
    )
}

pub(crate) fn plan_gpu_rader_upload_split_with_axis_context(
    sequence_len: usize,
    fft_rader_primes: &[usize],
    direct_rader_primes: &[usize],
    precision: Precision,
    device: DeviceProfile,
    axis_context: StockhamUploadAxisContext,
) -> Result<Option<RaderForcedUploadSchedule>> {
    // Upstream computes the ordinary shared-capacity/bandwidth pass count first.
    // `forceRaderTwoUpload` is only a final promotion from one pass to two; it must
    // not gate normal-capacity Rader axes that already require multiple uploads.
    let force_rader_two_upload =
        plan_gpu_force_rader_two_upload(sequence_len, fft_rader_primes, device)?;
    let policy = plan_gpu_scheduler_policy(precision, device)?;
    if !policy.reorder_four_step
        || policy.register_boost_four_step != 1
        || policy.register_boost_non_power_of_two
    {
        return Err(VkFftError::UnsupportedKernelPath(
            "Rader forced upload split currently covers the fixed reorderFourStep boost-1 policy",
        ));
    }
    let complex_bytes = match precision {
        Precision::F32 | Precision::F16StorageF32Compute => 8usize,
        Precision::F64 | Precision::F64ComputeF32Storage => 16usize,
        Precision::DoubleDouble | Precision::DoubleDoubleF64Storage => 32usize,
    };
    for &prime in direct_rader_primes {
        if prime < 2 || !sequence_len.is_multiple_of(prime) {
            return Err(VkFftError::InvalidKernelIr(
                "direct-Rader prime does not divide the scheduled axis",
            ));
        }
    }
    let reserve_elements = direct_rader_primes
        .iter()
        .copied()
        .max()
        .unwrap_or(1)
        .saturating_sub(1);
    let reserve_bytes =
        reserve_elements
            .checked_mul(complex_bytes)
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "Rader forced upload shared reservation",
            })?;
    let Some(used_shared_memory_bytes) = device.shared_memory_bytes.checked_sub(reserve_bytes)
    else {
        // A device-aware recursive child can legitimately reclassify an FFT-Rader prime as
        // Direct-Rader and materialize that prime as its own pass with an external LUT. In
        // that case the parent Rader multi-upload optimization must not hard-fail merely
        // because the monolithic shared-memory reservation model no longer fits. Retain the
        // already selected recursive tree and skip this optional upload rematerialization.
        return Ok(None);
    };
    if used_shared_memory_bytes < complex_bytes {
        return Ok(None);
    }
    let max_sequence_len_shared = used_shared_memory_bytes / complex_bytes;
    let max_sequence_len_strided = if policy.coalesced_memory_bytes > complex_bytes {
        used_shared_memory_bytes / policy.coalesced_memory_bytes
    } else {
        max_sequence_len_shared
    };
    if max_sequence_len_strided == 0 {
        return Ok(None);
    }

    // Match the normal scheduler ordering exactly: establish the base pass count,
    // optionally reduce it through strided half-bandwidth scoring, apply vendor
    // 2/3-stage thresholds, and only then promote a remaining single pass because
    // forceRaderTwoUpload requires multiple uploads for the Rader container pressure.
    let initial_capacity = if axis_context.strided_axis {
        max_sequence_len_strided
    } else {
        max_sequence_len_shared
    };
    let mut upload_count = 1usize;
    if ceil_div(sequence_len, initial_capacity)? > 1 {
        upload_count = exponent_covering(sequence_len, max_sequence_len_strided)?;
    }
    let (bandwidth_upload_count, max_single_strided_half_bandwidth) =
        apply_strided_bandwidth_boost(
            sequence_len,
            upload_count,
            max_sequence_len_shared,
            max_sequence_len_strided,
            max_sequence_len_shared,
            used_shared_memory_bytes,
            complex_bytes,
            policy.coalesced_memory_bytes,
            policy.reorder_four_step,
            axis_context,
        )?;
    upload_count = bandwidth_upload_count;
    if sequence_len >= policy.swap_to_two_stage_four_step && upload_count < 3 {
        upload_count = 2;
    }
    if sequence_len >= policy.swap_to_three_stage_four_step
        && policy.swap_to_three_stage_four_step >= 65_536
    {
        upload_count = 3;
    }
    let ordinary_upload_count = upload_count;
    if upload_count == 1 && force_rader_two_upload {
        upload_count = 2;
    }
    if upload_count == 1 {
        return Ok(None);
    }
    if upload_count > 3 {
        return Ok(None);
    }

    let mut axis_split = match upload_count {
        2 => match generic_sqrt_divisor_split_with_limits(
            sequence_len,
            max_sequence_len_strided,
            max_single_strided_half_bandwidth,
        )? {
            Some(axis_split) => Vec::from(axis_split),
            // Fixed upstream promotes a failed non-power-of-two two-pass divisor search
            // to the three-pass branch instead of abandoning multi-upload materialization.
            // This matters for narrow strided capacity (notably F16 coalescing), where no
            // legal two-factor split exists even though three bounded factors do.
            None => {
                upload_count = 3;
                match generic_three_upload_divisor_split_with_limits(
                    sequence_len,
                    max_sequence_len_strided,
                    max_single_strided_half_bandwidth,
                ) {
                    Ok(axis_split) => axis_split,
                    Err(VkFftError::UnsupportedKernelPath(_)) => return Ok(None),
                    Err(error) => return Err(error),
                }
            }
        },
        3 => match generic_three_upload_divisor_split_with_limits(
            sequence_len,
            max_sequence_len_strided,
            max_single_strided_half_bandwidth,
        ) {
            Ok(axis_split) => axis_split,
            Err(VkFftError::UnsupportedKernelPath(_)) => return Ok(None),
            Err(error) => return Err(error),
        },
        _ => unreachable!("Rader multi-upload scheduling yields exactly two or three uploads"),
    };
    // Fixed upstream applies the common reorderFourStep factor preference after the
    // divisor split for every non-Bluestein multi-upload axis, including Rader roots.
    if policy.reorder_four_step && !axis_context.use_bluestein_fft {
        prefer_four_step_first_factor(&mut axis_split);
    }
    let reason = if ordinary_upload_count == 1 && force_rader_two_upload {
        RaderUploadReason::ForcedRaderPressure
    } else {
        RaderUploadReason::CapacityOrBandwidth
    };
    let schedule = RaderUploadSchedule {
        sequence_len,
        used_shared_memory_bytes,
        max_sequence_len_shared,
        max_sequence_len_strided,
        upload_count,
        axis_split,
        reason,
    };
    schedule.validate()?;
    Ok(Some(schedule))
}

/// Result of the first upstream-equivalent upload scheduler slice.
/// Compatibility wrapper for callers that still use the historical forced-Rader name.
pub fn plan_gpu_rader_forced_upload_split(
    sequence_len: usize,
    fft_rader_primes: &[usize],
    direct_rader_primes: &[usize],
    precision: Precision,
    device: DeviceProfile,
) -> Result<Option<RaderUploadSchedule>> {
    plan_gpu_rader_upload_split(
        sequence_len,
        fft_rader_primes,
        direct_rader_primes,
        precision,
        device,
    )
}

/// Result of the first upstream-equivalent upload scheduler slice.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StockhamUploadSchedule {
    pub sequence_len: usize,
    pub batch_count: usize,
    pub used_shared_memory_bytes: usize,
    pub max_sequence_len_shared: usize,
    pub max_sequence_len_strided: usize,
    pub register_boost: usize,
    pub upload_count: usize,
    pub axis_split: Vec<usize>,
    pub radix_schedules: Vec<RadixRegisterSchedule>,
}

impl StockhamUploadSchedule {
    pub fn validate(&self) -> Result<()> {
        if self.sequence_len == 0
            || self.batch_count == 0
            || self.used_shared_memory_bytes == 0
            || self.max_sequence_len_shared == 0
            || self.max_sequence_len_strided == 0
            || self.register_boost == 0
            || self.upload_count == 0
            || self.axis_split.len() != self.upload_count
            || self.radix_schedules.len() != self.upload_count
        {
            return Err(VkFftError::InvalidKernelIr(
                "Stockham upload schedule metadata is inconsistent",
            ));
        }
        let product = self.axis_split.iter().try_fold(1usize, |product, factor| {
            product
                .checked_mul(*factor)
                .ok_or(VkFftError::ArithmeticOverflow {
                    operation: "Stockham upload axis split product",
                })
        })?;
        if self.axis_split.contains(&0) || product != self.sequence_len {
            return Err(VkFftError::InvalidKernelIr(
                "Stockham upload axis split does not cover the sequence",
            ));
        }
        let total_elements = self.sequence_len.checked_mul(self.batch_count).ok_or(
            VkFftError::ArithmeticOverflow {
                operation: "Stockham upload total element count",
            },
        )?;
        for (&axis_len, radix) in self.axis_split.iter().zip(&self.radix_schedules) {
            radix.validate()?;
            if radix.fft_len != axis_len
                || radix.register_boost != self.register_boost
                || radix.rhs_transform_count != total_elements / axis_len
            {
                return Err(VkFftError::InvalidKernelIr(
                    "Stockham upload radix metadata does not match its axis split",
                ));
            }
        }
        Ok(())
    }
}

/// Exact first-axis/single-upload workgroup block from upstream `VkFFTSplitAxisBlock`.
/// `grouped_batch` is upstream `groupedBatch`; `axis_swapped` records the bank-conflict
/// avoidance branch that exchanges localSizeX/localSizeY after both logical dimensions
/// have been clamped against shared memory and device limits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StockhamAxisBlockSchedule {
    pub threads_per_transform: usize,
    pub grouped_batch: usize,
    /// Physical ownership of independent transforms. This is deliberately separate
    /// from upstream `axisSwapped`: axis-0 uploads above zero already place grouped
    /// transforms on X and FFT threads on Y while keeping `axisSwapped == 0`.
    pub transforms_on_x: bool,
    pub axis_swapped: bool,
    pub local_size_x: usize,
    pub local_size_y: usize,
}

impl StockhamAxisBlockSchedule {
    pub fn validate(self, batch_count: usize, device: DeviceProfile) -> Result<()> {
        if self.threads_per_transform == 0
            || self.grouped_batch == 0
            || self.grouped_batch > batch_count
            || self.local_size_x == 0
            || self.local_size_y == 0
        {
            return Err(VkFftError::InvalidKernelIr(
                "Stockham axis-block schedule has inconsistent dimensions",
            ));
        }
        let expected = if self.transforms_on_x {
            [self.grouped_batch, self.threads_per_transform]
        } else {
            [self.threads_per_transform, self.grouped_batch]
        };
        let total = self.local_size_x.checked_mul(self.local_size_y).ok_or(
            VkFftError::ArithmeticOverflow {
                operation: "Stockham axis-block workgroup product",
            },
        )?;
        if [self.local_size_x, self.local_size_y] != expected
            || self.local_size_x > device.max_workgroup_size[0]
            || self.local_size_y > device.max_workgroup_size[1]
            || total > device.max_threads_per_block
        {
            return Err(VkFftError::InvalidKernelIr(
                "Stockham axis-block schedule exceeds device workgroup limits",
            ));
        }
        Ok(())
    }
}

/// Exact fixed-upstream register table returned by `VkFFTGetRegistersPerThreadQuad`.
/// Quad/DD has its own 2/3/5/7 decision tree; it is deliberately not derived from the
/// ordinary F64 table. The fixed baseline leaves radix 11/13 unsupported in Quad mode,
/// so callers must fail soft rather than borrowing ordinary-precision register counts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DoubleDoubleQuadRegisterSchedule {
    pub fft_len: usize,
    pub rhs_transform_count: usize,
    pub registers_per_thread_per_radix: [usize; VKFFT_RADIX_TABLE_LEN],
    pub registers_per_thread: usize,
    pub min_registers_per_thread: usize,
    pub is_good_sequence: bool,
}

/// Multi-upload scheduling metadata for fixed-upstream Quad/DD Stockham.
/// This is intentionally distinct from [`StockhamUploadSchedule`]: Quad uses its
/// own register table and 32-byte complex compute footprint, so ordinary F64
/// register schedules are never reused as a proxy. `register_boost` is the final
/// scheduler boost after the one-upload/four-step admission logic, not a claim
/// that the current execution IR materializes that boost as a dedicated kernel stage.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DoubleDoubleStockhamUploadSchedule {
    pub sequence_len: usize,
    pub batch_count: usize,
    pub used_shared_memory_bytes: usize,
    pub max_sequence_len_shared: usize,
    pub max_sequence_len_strided: usize,
    pub register_boost: usize,
    pub upload_count: usize,
    pub axis_split: Vec<usize>,
    pub quad_schedules: Vec<DoubleDoubleQuadRegisterSchedule>,
}

impl DoubleDoubleStockhamUploadSchedule {
    pub fn validate(&self) -> Result<()> {
        if self.sequence_len < 2
            || self.batch_count == 0
            || self.used_shared_memory_bytes < 32
            || self.max_sequence_len_shared == 0
            || self.max_sequence_len_strided == 0
            || self.register_boost == 0
            || !matches!(self.upload_count, 1..=3)
            || self.axis_split.len() != self.upload_count
            || self.quad_schedules.len() != self.upload_count
            || self.axis_split.contains(&0)
        {
            return Err(VkFftError::InvalidKernelIr(
                "double-double Stockham upload schedule metadata is inconsistent",
            ));
        }
        let product = self.axis_split.iter().try_fold(1usize, |product, factor| {
            product
                .checked_mul(*factor)
                .ok_or(VkFftError::ArithmeticOverflow {
                    operation: "double-double Stockham upload axis split product",
                })
        })?;
        if product != self.sequence_len {
            return Err(VkFftError::InvalidKernelIr(
                "double-double Stockham upload split does not cover the sequence",
            ));
        }
        let total_elements = self.sequence_len.checked_mul(self.batch_count).ok_or(
            VkFftError::ArithmeticOverflow {
                operation: "double-double Stockham upload total element count",
            },
        )?;
        for (&axis_len, quad) in self.axis_split.iter().zip(&self.quad_schedules) {
            if quad.fft_len != axis_len
                || quad.rhs_transform_count != total_elements / axis_len
                || quad.min_registers_per_thread == 0
            {
                return Err(VkFftError::InvalidKernelIr(
                    "double-double Stockham upload Quad metadata does not match its axis split",
                ));
            }
        }
        Ok(())
    }
}

pub fn plan_gpu_double_double_quad_registers(
    fft_len: usize,
    rhs_transform_count: usize,
) -> Result<DoubleDoubleQuadRegisterSchedule> {
    if fft_len < 2 || rhs_transform_count == 0 {
        return Err(VkFftError::UnsupportedKernelPath(
            "double-double Quad register scheduling requires N>=2 and non-zero RHS",
        ));
    }

    let mut remaining = fft_len;
    let mut exponent2 = 0usize;
    let mut exponent3 = 0usize;
    let mut exponent5 = 0usize;
    let mut exponent7 = 0usize;
    for (prime, exponent) in [
        (2usize, &mut exponent2),
        (3usize, &mut exponent3),
        (5usize, &mut exponent5),
        (7usize, &mut exponent7),
    ] {
        while remaining.is_multiple_of(prime) {
            remaining /= prime;
            *exponent += 1;
        }
    }
    if remaining != 1 {
        return Err(VkFftError::UnsupportedKernelPath(
            "fixed-upstream Quad register table supports only factors 2/3/5/7",
        ));
    }

    let mut registers = [0usize; VKFFT_RADIX_TABLE_LEN];
    if exponent2 > 0 {
        if exponent3 > 0 {
            if exponent5 > 0 {
                registers[2] = 6;
                registers[3] = 6;
                registers[5] = 5;
                if exponent7 > 0 {
                    registers[7] = 7;
                }
            } else if exponent7 > 0 {
                registers[2] = if exponent2 <= 2 { 6 } else { 8 };
                registers[3] = 6;
                registers[7] = 7;
            } else {
                registers[2] = 6;
                registers[3] = 6;
            }
        } else if exponent5 > 0 {
            if exponent7 > 0 {
                registers[2] = if exponent2 == 1 { 6 } else { 8 };
                registers[5] = 5;
                registers[7] = 7;
            } else {
                registers[2] = 4;
                registers[5] = 5;
            }
        } else if exponent7 > 0 {
            registers[2] = 8;
            registers[7] = 7;
        } else {
            // Literal pure-power-of-two branch from the fixed baseline. `max_rhs/64`
            // estimates work balance across 64 CUs and is clamped to one.
            let active_threads_y = (rhs_transform_count / 64).max(1);
            let mut test_min_stages = usize::MAX;
            let mut max_radix_min_stages = 1usize;
            for exponent_per_stage in 1usize..=3 {
                let stages = exponent2.div_ceil(exponent_per_stage);
                if stages < test_min_stages {
                    test_min_stages = stages;
                    max_radix_min_stages = exponent_per_stage;
                }
            }

            let mut max_multiplier_pow2 = 0usize;
            for exponent_per_stage in (1..=max_radix_min_stages).rev() {
                let divisor = 1usize << exponent_per_stage;
                let active_threads_x = active_threads_y.checked_mul(fft_len).ok_or(
                    VkFftError::ArithmeticOverflow {
                        operation: "double-double Quad active thread estimate",
                    },
                )? / divisor;
                if active_threads_x >= 128 {
                    max_multiplier_pow2 = exponent_per_stage;
                    break;
                }
            }
            max_multiplier_pow2 = max_multiplier_pow2.max(3);

            let mut final_multiplier_pow2 = 1usize;
            let mut min_stages = exponent2;
            for exponent_per_stage in 2..=max_multiplier_pow2 {
                let stages = exponent2.div_ceil(exponent_per_stage);
                if stages < min_stages {
                    final_multiplier_pow2 = exponent_per_stage;
                    min_stages = stages;
                }
            }
            let used_exponent = exponent2.min(final_multiplier_pow2);
            registers[2] = 1usize << used_exponent;
            if exponent2 < 3 {
                registers[2] = 1usize << exponent2;
            }
        }
    } else if exponent3 > 0 {
        if exponent5 > 0 {
            if exponent7 > 0 {
                registers[3] = 6;
                registers[5] = 5;
                registers[7] = 7;
            } else {
                registers[3] = 3;
                registers[5] = 5;
            }
        } else if exponent7 > 0 {
            registers[3] = 6;
            registers[7] = 7;
        } else {
            registers[3] = if exponent3 == 1 { 3 } else { 9 };
        }
    } else if exponent5 > 0 {
        registers[5] = 5;
        if exponent7 > 0 {
            registers[7] = 7;
        }
    } else if exponent7 > 0 {
        registers[7] = 7;
    } else {
        return Err(VkFftError::UnsupportedKernelPath(
            "double-double Quad factorization produced no supported radix",
        ));
    }

    registers[32] = if registers[2].is_multiple_of(32) {
        registers[2]
    } else {
        0
    };
    registers[16] = if registers[2].is_multiple_of(16) {
        registers[2]
    } else {
        0
    };
    registers[8] = if registers[2].is_multiple_of(8) {
        registers[2]
    } else {
        0
    };
    registers[4] = if registers[2].is_multiple_of(4) {
        registers[2]
    } else {
        0
    };
    if registers[2] >= 12 && registers[3] >= 12 {
        registers[12] = registers[2].min(registers[3]);
        if !registers[12].is_multiple_of(12) {
            registers[12] = 0;
        }
    }
    registers[6] = registers[2].min(registers[3]);
    registers[9] = if registers[3].is_multiple_of(9) {
        registers[3]
    } else {
        0
    };
    registers[10] = registers[2].min(registers[5]);
    registers[14] = registers[2].min(registers[7]);
    registers[15] = registers[3].min(registers[5]);

    let mut registers_per_thread = 0usize;
    let mut min_registers_per_thread = usize::MAX;
    for value in registers.iter().copied().filter(|value| *value != 0) {
        registers_per_thread = registers_per_thread.max(value);
        min_registers_per_thread = min_registers_per_thread.min(value);
    }
    if registers_per_thread == 0 || min_registers_per_thread == usize::MAX {
        return Err(VkFftError::InvalidKernelIr(
            "double-double Quad register table produced no executable radix",
        ));
    }
    let is_good_sequence =
        !(registers_per_thread > 16 || registers_per_thread >= 2 * min_registers_per_thread);
    Ok(DoubleDoubleQuadRegisterSchedule {
        fft_len,
        rhs_transform_count,
        registers_per_thread_per_radix: registers,
        registers_per_thread,
        min_registers_per_thread,
        is_good_sequence,
    })
}

/// Plan the first fixed-upstream multi-upload Stockham slice for Quad/DD compute.
/// The current DD execution surface is boost-1, so this deliberately preserves
/// `registerBoost4Step = 1` instead of borrowing ordinary-precision boost scoring.
/// Power-of-two and smooth non-power-of-two splits reuse the already-ported
/// upstream divisor searches; each resulting component is then scored by the
/// dedicated Quad register table.
pub fn plan_gpu_double_double_stockham_uploads_for_batches(
    sequence_len: usize,
    batch_count: usize,
    device: DeviceProfile,
) -> Result<DoubleDoubleStockhamUploadSchedule> {
    plan_gpu_double_double_stockham_uploads_for_batches_with_axis_context(
        sequence_len,
        batch_count,
        device,
        StockhamUploadAxisContext::default(),
    )
}

pub(crate) fn plan_gpu_double_double_stockham_uploads_for_batches_with_axis_context(
    sequence_len: usize,
    batch_count: usize,
    device: DeviceProfile,
    axis_context: StockhamUploadAxisContext,
) -> Result<DoubleDoubleStockhamUploadSchedule> {
    if sequence_len < 2 {
        return Err(VkFftError::UnsupportedKernelPath(
            "double-double Stockham upload scheduling requires N>=2",
        ));
    }
    if batch_count == 0 {
        return Err(VkFftError::ZeroBatchCount);
    }
    let policy = plan_gpu_scheduler_policy(Precision::DoubleDouble, device)?;
    const DD_COMPLEX_BYTES: usize = 32;
    let used_shared_memory_bytes = if sequence_len.is_power_of_two() {
        device.shared_memory_pow2_bytes
    } else {
        device.shared_memory_bytes
    };
    if used_shared_memory_bytes < DD_COMPLEX_BYTES {
        return Err(VkFftError::ResourceLimitExceeded {
            resource: "double-double Stockham upload shared memory",
            required: DD_COMPLEX_BYTES,
            available: used_shared_memory_bytes,
        });
    }
    let max_sequence_len_shared = used_shared_memory_bytes / DD_COMPLEX_BYTES;
    let max_sequence_len_strided = if policy.coalesced_memory_bytes > DD_COMPLEX_BYTES {
        used_shared_memory_bytes / policy.coalesced_memory_bytes
    } else {
        max_sequence_len_shared
    };
    if max_sequence_len_strided == 0 {
        return Err(VkFftError::ResourceLimitExceeded {
            resource: "double-double Stockham upload strided shared-memory capacity",
            required: 1,
            available: 0,
        });
    }

    // Quad/DD uses the same policy-level registerBoost admission as upstream. If the
    // initial one-upload capacity is exceeded, registerBoost4Step takes over (boost-1
    // for the fixed policies). Bluestein explicitly forces configuration.registerBoost=1.
    let register_boost_limit = if axis_context.use_bluestein_fft {
        1
    } else {
        policy.register_boost
    };
    let mut register_boost = largest_square_divisor_boost(sequence_len, register_boost_limit);
    let initial_non_strided = checked_mul(
        max_sequence_len_shared,
        register_boost,
        "double-double Stockham initial non-strided register boost",
    )?;
    let initial_strided = checked_mul(
        max_sequence_len_strided,
        register_boost,
        "double-double Stockham initial strided register boost",
    )?;
    let initial_capacity = if axis_context.strided_axis {
        initial_strided
    } else {
        initial_non_strided
    };
    let mut upload_count = 1usize;
    if ceil_div(sequence_len, initial_capacity)? > 1 {
        register_boost =
            largest_square_divisor_boost(sequence_len, policy.register_boost_four_step);
        let four_step_non_strided = checked_mul(
            max_sequence_len_shared,
            register_boost,
            "double-double Stockham four-step non-strided register boost",
        )?;
        let four_step_strided = checked_mul(
            max_sequence_len_strided,
            register_boost,
            "double-double Stockham four-step strided register boost",
        )?;
        upload_count = if !axis_context.strided_axis
            && (!policy.reorder_four_step || axis_context.use_bluestein_fft)
        {
            upload_count_with_first_capacity(
                sequence_len,
                four_step_non_strided,
                four_step_strided,
            )?
        } else {
            exponent_covering(sequence_len, four_step_strided)?
        };
    }
    if axis_context.strided_axis && axis_context.use_bluestein_fft {
        let four_step_strided = checked_mul(
            max_sequence_len_strided,
            register_boost,
            "double-double strided Bluestein legacy pass capacity",
        )?;
        upload_count =
            upstream_strided_bluestein_initial_upload_count(sequence_len, four_step_strided)?;
    }

    let denominator = if !axis_context.strided_axis
        && (axis_context.use_bluestein_fft || !policy.reorder_four_step || upload_count == 1)
    {
        let later = checked_pow(
            max_sequence_len_strided,
            upload_count.saturating_sub(1),
            "double-double Stockham unit-stride register-boost denominator",
        )?;
        checked_mul(
            later,
            max_sequence_len_shared,
            "double-double Stockham unit-stride register-boost denominator",
        )?
    } else {
        checked_pow(
            max_sequence_len_strided,
            upload_count,
            "double-double Stockham strided register-boost denominator",
        )?
    };
    register_boost = ceil_div(sequence_len, denominator)?;
    let required_boost = register_boost;
    let mut selected_boost = None;
    for candidate in required_boost..=register_boost_limit {
        let square = candidate
            .checked_mul(candidate)
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "double-double Stockham register boost square",
            })?;
        if sequence_len.is_multiple_of(square) {
            selected_boost = Some(candidate);
            break;
        }
    }
    if let Some(selected) = selected_boost {
        register_boost = selected;
    }
    let must_drop_boost = if sequence_len.is_power_of_two() {
        selected_boost.is_none()
    } else {
        selected_boost.is_none() || !policy.register_boost_non_power_of_two
    };
    if must_drop_boost && register_boost > 1 {
        register_boost = 1;
        upload_count = upload_count
            .checked_add(1)
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "double-double Stockham upload count",
            })?;
    }
    let max_single_strided = checked_mul(
        max_sequence_len_strided,
        register_boost,
        "double-double Stockham final strided register boost",
    )?;
    let max_single_non_strided = checked_mul(
        max_sequence_len_shared,
        register_boost,
        "double-double Stockham final non-strided register boost",
    )?;
    let (bandwidth_upload_count, max_single_strided_half_bandwidth) =
        apply_strided_bandwidth_boost(
            sequence_len,
            upload_count,
            max_single_non_strided,
            max_single_strided,
            max_sequence_len_shared,
            used_shared_memory_bytes,
            DD_COMPLEX_BYTES,
            policy.coalesced_memory_bytes,
            policy.reorder_four_step,
            axis_context,
        )?;
    upload_count = bandwidth_upload_count;

    if sequence_len >= policy.swap_to_two_stage_four_step && upload_count < 3 {
        upload_count = 2;
    }
    if sequence_len >= policy.swap_to_three_stage_four_step
        && policy.swap_to_three_stage_four_step >= 65_536
    {
        upload_count = 3;
    }
    if upload_count > 3 {
        return Err(VkFftError::UnsupportedKernelPath(
            "double-double Stockham scheduler requires more than three uploads",
        ));
    }

    let unit_stride_first_stage =
        !axis_context.strided_axis && (!policy.reorder_four_step || axis_context.use_bluestein_fft);
    let two_upload_quotient_limit = if unit_stride_first_stage {
        max_sequence_len_shared
    } else {
        max_single_strided_half_bandwidth
    };
    let mut axis_split = match upload_count {
        1 => vec![sequence_len],
        2 if sequence_len.is_power_of_two() => power_of_two_two_upload_split(
            sequence_len,
            max_sequence_len_shared,
            max_single_strided,
            max_single_strided_half_bandwidth,
            register_boost,
            unit_stride_first_stage,
            device,
        )?,
        2 => generic_sqrt_divisor_split_with_limits(
            sequence_len,
            max_single_strided,
            two_upload_quotient_limit,
        )?
        .map(Vec::from)
        .ok_or(VkFftError::UnsupportedKernelPath(
            "double-double Stockham scheduler could not find a two-upload divisor split",
        ))?,
        3 if sequence_len.is_power_of_two() => {
            let split = power_of_two_three_upload_split(
                sequence_len,
                used_shared_memory_bytes,
                max_sequence_len_shared,
                max_single_strided,
                max_single_strided_half_bandwidth,
                register_boost,
                unit_stride_first_stage,
                device,
            )?;
            let executable = if unit_stride_first_stage {
                split
                    .first()
                    .is_some_and(|value| *value <= max_sequence_len_shared)
                    && split
                        .iter()
                        .skip(1)
                        .all(|value| *value <= max_single_strided)
            } else {
                split.iter().all(|value| *value <= max_single_strided)
            };
            if executable {
                split
            } else if unit_stride_first_stage {
                generic_three_upload_unit_stride_split(
                    sequence_len,
                    max_sequence_len_shared,
                    max_single_strided,
                )?
            } else {
                generic_power_of_two_three_upload_split(
                    sequence_len,
                    max_single_strided,
                    max_single_strided,
                )?
            }
        }
        3 if unit_stride_first_stage => generic_three_upload_unit_stride_split(
            sequence_len,
            max_sequence_len_shared,
            max_single_strided,
        )?,
        3 => generic_three_upload_divisor_split_with_limits(
            sequence_len,
            max_single_strided,
            max_single_strided_half_bandwidth,
        )?,
        _ => unreachable!("validated DD upload count is between one and three"),
    };
    if !sequence_len.is_power_of_two() && upload_count > 1 && !unit_stride_first_stage {
        prefer_four_step_first_factor(&mut axis_split);
    }
    let total_elements =
        sequence_len
            .checked_mul(batch_count)
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "double-double Stockham upload total element count",
            })?;
    let quad_schedules = axis_split
        .iter()
        .map(|&axis_len| plan_gpu_double_double_quad_registers(axis_len, total_elements / axis_len))
        .collect::<Result<Vec<_>>>()?;
    let schedule = DoubleDoubleStockhamUploadSchedule {
        sequence_len,
        batch_count,
        used_shared_memory_bytes,
        max_sequence_len_shared,
        max_sequence_len_strided,
        register_boost,
        upload_count,
        axis_split,
        quad_schedules,
    };
    schedule.validate()?;
    Ok(schedule)
}

const DD_COMPLEX_BYTES: usize = 32;

fn double_double_stockham_usable_shared_memory_bytes(device: DeviceProfile) -> usize {
    // NVIDIA's OpenCL -> PTX path reserves one additional 32-byte DD complex in static
    // shared memory. A kernel whose explicit cooperative stripe consumes CL_DEVICE_LOCAL_MEM_SIZE
    // therefore fails ptxas by exactly 0x20 bytes. CUDA and Vulkan accept the exact physical
    // limit, so keep the reserve backend-specific instead of weakening the shared DeviceProfile.
    let native_reserve = if device.backend == Backend::OpenCl && device.vendor == GpuVendor::Nvidia
    {
        DD_COMPLEX_BYTES
    } else {
        0
    };
    device.shared_memory_bytes.saturating_sub(native_reserve)
}

/// Fixed-upstream automatic axis-0 block for a single-upload DD Stockham transform.
///
/// This is intentionally separate from the user-grouped path below. Upstream keeps the
/// automatic physical group at one transform even when multiple independent batches are
/// present, and on devices with more than 136 KiB of shared memory the final
/// `scaleRegistersNum` pass may multiply the ordinary Quad register table so a
/// near-capacity transform fits the physical thread limit. The returned block therefore
/// carries the post-scale caller geometry rather than the raw Quad-table floor.
pub fn plan_gpu_double_double_axis0_stockham_block(
    sequence_len: usize,
    batch_count: usize,
    device: DeviceProfile,
) -> Result<Option<StockhamAxisBlockSchedule>> {
    plan_gpu_double_double_axis0_stockham_block_with_axis_context(
        sequence_len,
        batch_count,
        device,
        StockhamUploadAxisContext::default(),
    )
}

/// Automatic axis-0 block for a one-upload DD Stockham transform used as a
/// Bluestein convolution child. The child-specific upload context must be scored
/// before materializing the otherwise group-one physical block.
pub(crate) fn plan_gpu_double_double_axis0_bluestein_stockham_block(
    sequence_len: usize,
    batch_count: usize,
    device: DeviceProfile,
) -> Result<Option<StockhamAxisBlockSchedule>> {
    plan_gpu_double_double_axis0_stockham_block_with_axis_context(
        sequence_len,
        batch_count,
        device,
        StockhamUploadAxisContext {
            use_bluestein_fft: true,
            ..StockhamUploadAxisContext::default()
        },
    )
}

fn plan_gpu_double_double_axis0_stockham_block_with_axis_context(
    sequence_len: usize,
    batch_count: usize,
    device: DeviceProfile,
    axis_context: StockhamUploadAxisContext,
) -> Result<Option<StockhamAxisBlockSchedule>> {
    if sequence_len < 2
        || batch_count == 0
        || device.max_threads_per_block == 0
        || device.max_workgroup_size[0] == 0
    {
        return Ok(None);
    }

    let upload_schedule =
        match plan_gpu_double_double_stockham_uploads_for_batches_with_axis_context(
            sequence_len,
            batch_count,
            device,
            axis_context,
        ) {
            Ok(schedule) => schedule,
            Err(VkFftError::UnsupportedKernelPath(_))
            | Err(VkFftError::ResourceLimitExceeded { .. }) => return Ok(None),
            Err(error) => return Err(error),
        };
    if upload_schedule.upload_count != 1 {
        return Ok(None);
    }

    let quad = match plan_gpu_double_double_quad_registers(sequence_len, batch_count) {
        Ok(schedule) => schedule,
        Err(VkFftError::UnsupportedKernelPath(_)) => return Ok(None),
        Err(error) => return Err(error),
    };
    let physical_thread_limit = device
        .max_threads_per_block
        .min(device.max_workgroup_size[0]);
    if physical_thread_limit == 0 || quad.min_registers_per_thread == 0 {
        return Ok(None);
    }
    let scale_denominator = quad
        .min_registers_per_thread
        .checked_mul(physical_thread_limit)
        .ok_or(VkFftError::ArithmeticOverflow {
            operation: "double-double Stockham global register scaling denominator",
        })?;
    let scale_registers_num = ceil_div(sequence_len, scale_denominator)?.max(1);
    let scaled_min_registers = quad
        .min_registers_per_thread
        .checked_mul(scale_registers_num)
        .ok_or(VkFftError::ArithmeticOverflow {
            operation: "double-double Stockham global minimum-register scaling",
        })?;
    let threads_per_transform = ceil_div(sequence_len, scaled_min_registers)?.max(1);
    if threads_per_transform > physical_thread_limit {
        return Ok(None);
    }

    let usable_shared_memory_bytes = double_double_stockham_usable_shared_memory_bytes(device);
    let max_sequence_len_shared = usable_shared_memory_bytes / DD_COMPLEX_BYTES;
    if max_sequence_len_shared == 0 || sequence_len > max_sequence_len_shared {
        return Ok(None);
    }

    let block = StockhamAxisBlockSchedule {
        threads_per_transform,
        grouped_batch: 1,
        transforms_on_x: false,
        axis_swapped: false,
        local_size_x: threads_per_transform,
        local_size_y: 1,
    };
    if block.validate(batch_count, device).is_err() {
        return Ok(None);
    }
    Ok(Some(block))
}

/// Executable fixed-upstream double-double `VkFFTSplitAxisBlock` user-group slice.
/// Any single-upload Stockham length covered by the exact Quad 2/3/5/7 table may
/// materialize when its requested group fits the fixed-upstream device/thread/shared
/// geometry. Double-double compute values occupy 32 bytes per complex value.
pub fn plan_gpu_double_double_axis0_user_grouped_stockham_block(
    sequence_len: usize,
    batch_count: usize,
    grouped_batch_override: Option<usize>,
    device: DeviceProfile,
) -> Result<Option<StockhamAxisBlockSchedule>> {
    plan_gpu_double_double_axis0_user_grouped_stockham_block_with_zero_padding(
        sequence_len,
        batch_count,
        grouped_batch_override,
        false,
        device,
    )
}

pub(crate) fn plan_gpu_double_double_axis0_user_grouped_stockham_block_with_zero_padding(
    sequence_len: usize,
    batch_count: usize,
    grouped_batch_override: Option<usize>,
    perform_zero_padding: bool,
    device: DeviceProfile,
) -> Result<Option<StockhamAxisBlockSchedule>> {
    let Some(requested) = grouped_batch_override else {
        return Ok(None);
    };
    if requested <= 1
        || sequence_len < 2
        || batch_count == 0
        || device.max_threads_per_block == 0
        || device.max_workgroup_size[0] == 0
        || device.max_workgroup_size[1] == 0
    {
        return Ok(None);
    }
    let policy = match plan_gpu_scheduler_policy(Precision::DoubleDouble, device) {
        Ok(policy) => policy,
        Err(VkFftError::UnsupportedKernelPath(_)) => return Ok(None),
        Err(error) => return Err(error),
    };
    let quad = match plan_gpu_double_double_quad_registers(sequence_len, batch_count) {
        Ok(schedule) => schedule,
        Err(VkFftError::UnsupportedKernelPath(_)) => return Ok(None),
        Err(error) => return Err(error),
    };
    let threads_per_transform = ceil_div(sequence_len, quad.min_registers_per_thread)?.max(1);
    if threads_per_transform > device.max_threads_per_block {
        return Ok(None);
    }

    let usable_shared_memory_bytes = double_double_stockham_usable_shared_memory_bytes(device);
    let max_sequence_len_shared = usable_shared_memory_bytes / DD_COMPLEX_BYTES;
    if max_sequence_len_shared == 0 || sequence_len > max_sequence_len_shared {
        return Ok(None);
    }
    let max_batch_coalesced = (policy.coalesced_memory_bytes / DD_COMPLEX_BYTES).max(1);
    let mut grouped_batch = (max_sequence_len_shared / sequence_len)
        .max(max_batch_coalesced)
        .min(requested)
        .min(batch_count)
        .max(1);
    grouped_batch = grouped_batch.min(device.max_workgroup_size[1]).max(1);
    if grouped_batch
        .checked_mul(threads_per_transform)
        .is_some_and(|threads| threads > device.max_threads_per_block)
    {
        grouped_batch = (device.max_threads_per_block / threads_per_transform).max(1);
    }
    while grouped_batch > 1
        && grouped_batch
            .checked_mul(sequence_len)
            .is_some_and(|elements| elements > max_sequence_len_shared)
    {
        grouped_batch /= 2;
    }

    let axis_swapped = !perform_zero_padding
        && (sequence_len.is_multiple_of(2)
            || threads_per_transform < device.shared_banks.max(1) / 4)
        && grouped_batch > 1
        && grouped_batch
            .checked_mul(sequence_len)
            .is_some_and(|elements| elements < max_sequence_len_shared);
    let (local_size_x, local_size_y) = if axis_swapped {
        (grouped_batch, threads_per_transform)
    } else {
        (threads_per_transform, grouped_batch)
    };
    let block = StockhamAxisBlockSchedule {
        threads_per_transform,
        grouped_batch,
        transforms_on_x: axis_swapped,
        axis_swapped,
        local_size_x,
        local_size_y,
    };
    if block.validate(batch_count, device).is_err() {
        return Ok(None);
    }
    Ok(Some(block))
}

/// Executable fixed-upstream double-double standalone FFT-Rader axis-0 user-group slice.
/// The prime-length caller block is intentionally distinct from the `(p-1)` convolution
/// child block: independent batches use the outer Rader register coupling, while the
/// internal FFT demand comes from the exact Quad table rather than the ordinary F64 table.
pub fn plan_gpu_double_double_axis0_fft_rader_user_grouped_batch_block(
    prime: usize,
    batch_count: usize,
    grouped_batch_override: Option<usize>,
    device: DeviceProfile,
) -> Result<Option<StockhamAxisBlockSchedule>> {
    plan_gpu_double_double_axis0_fft_rader_user_grouped_batch_block_with_zero_padding(
        prime,
        batch_count,
        grouped_batch_override,
        false,
        device,
    )
}

pub(crate) fn plan_gpu_double_double_axis0_fft_rader_user_grouped_batch_block_with_zero_padding(
    prime: usize,
    batch_count: usize,
    grouped_batch_override: Option<usize>,
    perform_zero_padding: bool,
    device: DeviceProfile,
) -> Result<Option<StockhamAxisBlockSchedule>> {
    plan_gpu_double_double_axis0_fft_rader_user_grouped_batch_block_with_zero_padding_and_tuning(
        prime,
        batch_count,
        grouped_batch_override,
        perform_zero_padding,
        PlannerTuning::portable(),
        device,
    )
}

pub(crate) fn plan_gpu_double_double_axis0_fft_rader_user_grouped_batch_block_with_zero_padding_and_tuning(
    prime: usize,
    batch_count: usize,
    grouped_batch_override: Option<usize>,
    perform_zero_padding: bool,
    tuning: PlannerTuning,
    device: DeviceProfile,
) -> Result<Option<StockhamAxisBlockSchedule>> {
    let Some(requested) = grouped_batch_override else {
        return Ok(None);
    };
    if requested <= 1
        || prime < 3
        || batch_count == 0
        || device.max_threads_per_block == 0
        || device.max_workgroup_size[0] == 0
        || device.max_workgroup_size[1] == 0
    {
        return Ok(None);
    }
    let policy = match plan_gpu_scheduler_policy(Precision::DoubleDouble, device) {
        Ok(policy) => policy,
        Err(VkFftError::UnsupportedKernelPath(_)) => return Ok(None),
        Err(error) => return Err(error),
    };
    // VkFFT's prime-length caller is scheduled from the Rader-container register state.
    // Only the split `(p-1)` convolution child consumes the Quad/DD Stockham table.
    let effective_tuning = upstream_effective_rader_tuning(tuning, device, Precision::DoubleDouble);
    let Some(threads_per_transform) =
        plan_gpu_double_double_axis0_fft_rader_threads_for_prime_multiplicities_with_tuning(
            prime,
            &[(prime, 1)],
            batch_count,
            effective_tuning,
            device,
        )?
    else {
        return Ok(None);
    };
    if threads_per_transform > device.max_threads_per_block {
        return Ok(None);
    }

    const DD_COMPLEX_BYTES: usize = 32;
    let max_sequence_len_shared = device.shared_memory_bytes / DD_COMPLEX_BYTES;
    if max_sequence_len_shared == 0 {
        return Ok(None);
    }
    let max_batch_coalesced = (policy.coalesced_memory_bytes / DD_COMPLEX_BYTES).max(1);
    let mut grouped_batch = (max_sequence_len_shared / prime)
        .max(max_batch_coalesced)
        .min(requested)
        .min(batch_count)
        .max(1);
    grouped_batch = grouped_batch.min(device.max_workgroup_size[1]).max(1);
    if grouped_batch
        .checked_mul(threads_per_transform)
        .is_some_and(|threads| threads > device.max_threads_per_block)
    {
        grouped_batch = (device.max_threads_per_block / threads_per_transform).max(1);
    }
    while grouped_batch > 1
        && grouped_batch
            .checked_mul(prime)
            .is_some_and(|elements| elements > max_sequence_len_shared)
    {
        grouped_batch /= 2;
    }
    if grouped_batch <= 1 {
        return Ok(None);
    }

    let axis_swapped = !perform_zero_padding
        && (prime.is_multiple_of(2) || threads_per_transform < device.shared_banks.max(1) / 4)
        && grouped_batch > 1
        && grouped_batch
            .checked_mul(prime)
            .is_some_and(|elements| elements < max_sequence_len_shared);
    let (local_size_x, local_size_y) = if axis_swapped {
        (grouped_batch, threads_per_transform)
    } else {
        (threads_per_transform, grouped_batch)
    };
    let block = StockhamAxisBlockSchedule {
        threads_per_transform,
        grouped_batch,
        transforms_on_x: axis_swapped,
        axis_swapped,
        local_size_x,
        local_size_y,
    };
    if block.validate(batch_count, device).is_err() {
        return Ok(None);
    }
    Ok(Some(block))
}

/// Fixed-upstream automatic higher-axis block for a one-upload DD Stockham child.
/// Near the large-shared-memory ceiling the scheduler scales the Quad register table
/// before `VkFFTSplitAxisBlock`; one transform stays on X and the scaled FFT lanes stay on Y.
pub fn plan_gpu_double_double_other_axis_stockham_block(
    sequence_len: usize,
    batch_count: usize,
    device: DeviceProfile,
) -> Result<Option<StockhamAxisBlockSchedule>> {
    plan_gpu_double_double_other_axis_stockham_block_with_axis_context(
        sequence_len,
        batch_count,
        device,
        StockhamUploadAxisContext {
            strided_axis: true,
            bandwidth_boost: 1,
            use_bluestein_fft: false,
            perform_convolution: false,
        },
    )
}

/// Higher-axis counterpart of the large shared-memory Bluestein convolution block.
/// This is deliberately distinct from the contiguous scorer: upstream keeps the
/// Bluestein upload context while applying the axis_id>=1 transforms-X/threads-Y layout.
pub(crate) fn plan_gpu_double_double_other_axis_bluestein_stockham_block(
    sequence_len: usize,
    batch_count: usize,
    device: DeviceProfile,
) -> Result<Option<StockhamAxisBlockSchedule>> {
    plan_gpu_double_double_other_axis_stockham_block_with_axis_context(
        sequence_len,
        batch_count,
        device,
        StockhamUploadAxisContext {
            strided_axis: true,
            bandwidth_boost: 1,
            use_bluestein_fft: true,
            perform_convolution: false,
        },
    )
}

fn plan_gpu_double_double_other_axis_stockham_block_with_axis_context(
    sequence_len: usize,
    batch_count: usize,
    device: DeviceProfile,
    axis_context: StockhamUploadAxisContext,
) -> Result<Option<StockhamAxisBlockSchedule>> {
    if sequence_len < 2
        || batch_count == 0
        || device.max_threads_per_block == 0
        || device.max_workgroup_size[0] == 0
        || device.max_workgroup_size[1] == 0
    {
        return Ok(None);
    }
    let upload_schedule =
        match plan_gpu_double_double_stockham_uploads_for_batches_with_axis_context(
            sequence_len,
            batch_count,
            device,
            axis_context,
        ) {
            Ok(schedule) => schedule,
            Err(VkFftError::UnsupportedKernelPath(_))
            | Err(VkFftError::ResourceLimitExceeded { .. }) => return Ok(None),
            Err(error) => return Err(error),
        };
    if upload_schedule.upload_count != 1 {
        return Ok(None);
    }

    let quad = match plan_gpu_double_double_quad_registers(sequence_len, batch_count) {
        Ok(schedule) => schedule,
        Err(VkFftError::UnsupportedKernelPath(_)) => return Ok(None),
        Err(error) => return Err(error),
    };
    let physical_thread_limit = device
        .max_threads_per_block
        .min(device.max_workgroup_size[1]);
    if physical_thread_limit == 0 || quad.min_registers_per_thread == 0 {
        return Ok(None);
    }
    let scale_denominator = quad
        .min_registers_per_thread
        .checked_mul(physical_thread_limit)
        .ok_or(VkFftError::ArithmeticOverflow {
            operation: "double-double higher-axis Stockham global register scaling denominator",
        })?;
    let scale_registers_num = ceil_div(sequence_len, scale_denominator)?.max(1);
    let scaled_min_registers = quad
        .min_registers_per_thread
        .checked_mul(scale_registers_num)
        .ok_or(VkFftError::ArithmeticOverflow {
            operation: "double-double higher-axis Stockham global minimum-register scaling",
        })?;
    let threads_per_transform = ceil_div(sequence_len, scaled_min_registers)?.max(1);
    if threads_per_transform > physical_thread_limit {
        return Ok(None);
    }

    let usable_shared_memory_bytes = double_double_stockham_usable_shared_memory_bytes(device);
    if sequence_len > usable_shared_memory_bytes / DD_COMPLEX_BYTES {
        return Ok(None);
    }
    let block = StockhamAxisBlockSchedule {
        threads_per_transform,
        grouped_batch: 1,
        transforms_on_x: true,
        axis_swapped: false,
        local_size_x: 1,
        local_size_y: threads_per_transform,
    };
    if block.validate(batch_count, device).is_err() {
        return Ok(None);
    }
    Ok(Some(block))
}

/// Fixed-upstream `axis_id >= 1` user-grouped double-double Stockham slice.
/// The Quad register table supplies FFT-thread demand while the higher-axis branch
/// keeps independent sequences on X and FFT threads on Y exactly like upstream.
/// As in the ordinary branch, `axis1_grouped_batch_override` is the literal
/// `configuration.groupedBatch[1]` gate after mapping row-major axes to VkFFT order.
pub fn plan_gpu_double_double_other_axis_user_grouped_stockham_block(
    sequence_len: usize,
    batch_count: usize,
    fastest_axis_len: usize,
    grouped_batch_override: Option<usize>,
    axis1_grouped_batch_override: Option<usize>,
    device: DeviceProfile,
) -> Result<Option<StockhamAxisBlockSchedule>> {
    if grouped_batch_override == Some(0)
        || sequence_len < 2
        || batch_count == 0
        || fastest_axis_len == 0
        || device.max_threads_per_block == 0
        || device.max_workgroup_size[0] == 0
        || device.max_workgroup_size[1] == 0
    {
        return Ok(None);
    }
    let policy = match plan_gpu_scheduler_policy(Precision::DoubleDouble, device) {
        Ok(policy) => policy,
        Err(VkFftError::UnsupportedKernelPath(_)) => return Ok(None),
        Err(error) => return Err(error),
    };
    let quad = match plan_gpu_double_double_quad_registers(sequence_len, batch_count) {
        Ok(schedule) => schedule,
        Err(VkFftError::UnsupportedKernelPath(_)) => return Ok(None),
        Err(error) => return Err(error),
    };
    let threads_per_transform = ceil_div(sequence_len, quad.min_registers_per_thread)?.max(1);
    if threads_per_transform > device.max_threads_per_block
        || threads_per_transform > device.max_workgroup_size[1]
    {
        return Ok(None);
    }

    let usable_shared_memory_bytes = double_double_stockham_usable_shared_memory_bytes(device);
    let max_sequence_len_shared = usable_shared_memory_bytes / DD_COMPLEX_BYTES;
    let coalesced = policy.coalesced_memory_bytes.max(DD_COMPLEX_BYTES);
    let max_single_size_strided = if coalesced > DD_COMPLEX_BYTES {
        usable_shared_memory_bytes / coalesced
    } else {
        max_sequence_len_shared
    };
    if max_sequence_len_shared == 0 || max_single_size_strided == 0 {
        return Ok(None);
    }
    let max_batch_coalesced = (policy.coalesced_memory_bytes / DD_COMPLEX_BYTES).max(1);
    let mut grouped_batch = if max_single_size_strided / sequence_len > 1 {
        (max_single_size_strided / sequence_len)
            .checked_mul(max_batch_coalesced)
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "double-double higher-axis initial grouped batch",
            })?
    } else {
        max_batch_coalesced
    };

    if let Some(requested) = grouped_batch_override {
        // Fixed upstream's early user branch runs before automatic warp/coalescing tuning.
        let mut user_grouped_batch = fastest_axis_len.min(grouped_batch).max(1);
        let upstream_axis1_gate = axis1_grouped_batch_override.unwrap_or(0);
        if user_grouped_batch > upstream_axis1_gate {
            user_grouped_batch = requested;
        }
        user_grouped_batch = user_grouped_batch
            .min(batch_count)
            .min(device.max_workgroup_size[0])
            .max(1);
        if user_grouped_batch
            .checked_mul(threads_per_transform)
            .is_some_and(|threads| threads > device.max_threads_per_block)
        {
            // Preserve the fixed upstream assignment literally.
            user_grouped_batch = device.max_threads_per_block / user_grouped_batch;
            if user_grouped_batch == 0 {
                return Ok(None);
            }
        }
        let block = StockhamAxisBlockSchedule {
            threads_per_transform,
            grouped_batch: user_grouped_batch,
            transforms_on_x: true,
            axis_swapped: false,
            local_size_x: user_grouped_batch,
            local_size_y: threads_per_transform,
        };
        if block.validate(batch_count, device).is_err() {
            return Ok(None);
        }
        return Ok(Some(block));
    }

    if grouped_batch < max_batch_coalesced {
        grouped_batch = max_batch_coalesced;
    }
    grouped_batch = (grouped_batch / max_batch_coalesced) * max_batch_coalesced;
    if sequence_len > max_single_size_strided {
        grouped_batch = (max_sequence_len_shared / sequence_len).max(1);
    }
    let warp_size = policy.subgroup_width.max(1);
    if grouped_batch > warp_size {
        grouped_batch = (grouped_batch / warp_size) * warp_size;
    }
    if grouped_batch > 2 * max_batch_coalesced {
        grouped_batch = (grouped_batch / (2 * max_batch_coalesced)) * (2 * max_batch_coalesced);
    }
    if grouped_batch > 4 * max_batch_coalesced {
        grouped_batch = (grouped_batch / (4 * max_batch_coalesced)) * (4 * max_batch_coalesced);
    }
    grouped_batch = grouped_batch.max(1);
    let mut grouped = fastest_axis_len.min(grouped_batch).max(1);
    if device.vendor == GpuVendor::Nvidia {
        const AIM_THREADS: usize = 128;
        while grouped
            .checked_mul(threads_per_transform)
            .is_some_and(|threads| threads >= 2 * AIM_THREADS)
            && grouped > max_batch_coalesced
        {
            grouped = (grouped / 2).max(max_batch_coalesced);
        }
    }
    grouped = grouped
        .min(batch_count)
        .min(device.max_workgroup_size[0])
        .max(1);
    if grouped
        .checked_mul(threads_per_transform)
        .is_some_and(|threads| threads > device.max_threads_per_block)
    {
        let original = grouped;
        for divisor in 1..=original {
            let candidate = original / divisor;
            if candidate > 0
                && candidate
                    .checked_mul(threads_per_transform)
                    .is_some_and(|threads| threads <= device.max_threads_per_block)
            {
                grouped = candidate;
                break;
            }
        }
    }
    let block = StockhamAxisBlockSchedule {
        threads_per_transform,
        grouped_batch: grouped,
        transforms_on_x: true,
        axis_swapped: false,
        local_size_x: grouped,
        local_size_y: threads_per_transform,
    };
    if block.validate(batch_count, device).is_err() {
        return Ok(None);
    }
    Ok(Some(block))
}

fn plan_gpu_other_axis_rader_block_from_threads(
    sequence_len: usize,
    batch_count: usize,
    fastest_axis_len: usize,
    complex_bytes: usize,
    threads_per_transform: usize,
    grouped_batch_override: Option<usize>,
    axis1_grouped_batch_override: Option<usize>,
    device: DeviceProfile,
) -> Result<Option<StockhamAxisBlockSchedule>> {
    let precision = match complex_bytes {
        8 => Precision::F32,
        16 => Precision::F64,
        32 => Precision::DoubleDouble,
        _ => return Ok(None),
    };
    plan_gpu_other_axis_rader_block_from_threads_for_precision(
        sequence_len,
        batch_count,
        fastest_axis_len,
        precision,
        complex_bytes,
        threads_per_transform,
        grouped_batch_override,
        axis1_grouped_batch_override,
        device,
    )
}

fn plan_gpu_other_axis_rader_block_from_threads_for_precision(
    sequence_len: usize,
    batch_count: usize,
    fastest_axis_len: usize,
    precision: Precision,
    complex_bytes: usize,
    threads_per_transform: usize,
    grouped_batch_override: Option<usize>,
    axis1_grouped_batch_override: Option<usize>,
    device: DeviceProfile,
) -> Result<Option<StockhamAxisBlockSchedule>> {
    if sequence_len < 2
        || batch_count == 0
        || fastest_axis_len == 0
        || complex_bytes == 0
        || threads_per_transform == 0
        || threads_per_transform > device.max_threads_per_block
        || threads_per_transform > device.max_workgroup_size[1]
        || device.max_workgroup_size[0] == 0
    {
        return Ok(None);
    }
    let policy = plan_gpu_scheduler_policy(precision, device)?;
    let max_sequence_len_shared = device.shared_memory_bytes / complex_bytes;
    if max_sequence_len_shared == 0 {
        return Ok(None);
    }
    let coalesced = policy.coalesced_memory_bytes.max(complex_bytes);
    let max_single_size_strided = if coalesced > complex_bytes {
        device.shared_memory_bytes / coalesced
    } else {
        max_sequence_len_shared
    };
    if max_single_size_strided == 0 {
        return Ok(None);
    }
    let max_batch_coalesced = (policy.coalesced_memory_bytes / complex_bytes).max(1);
    let mut grouped_batch = if max_single_size_strided / sequence_len > 1 {
        (max_single_size_strided / sequence_len)
            .checked_mul(max_batch_coalesced)
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "higher-axis Rader initial grouped batch",
            })?
    } else {
        max_batch_coalesced
    };

    if let Some(grouped_batch_override) = grouped_batch_override {
        if grouped_batch
            .checked_mul(threads_per_transform)
            .is_some_and(|threads| threads > device.max_threads_per_block)
        {
            grouped_batch = max_batch_coalesced;
        }
        let mut grouped = fastest_axis_len.min(grouped_batch).max(1);
        let axis1_gate = axis1_grouped_batch_override.unwrap_or(0);
        if grouped > axis1_gate {
            grouped = grouped_batch_override;
        }
        grouped = grouped
            .min(batch_count)
            .min(device.max_workgroup_size[0])
            .max(1);
        if grouped
            .checked_mul(threads_per_transform)
            .is_some_and(|threads| threads > device.max_threads_per_block)
        {
            // Preserve fixed upstream's axis_id>=1 user branch literally.
            grouped = device.max_threads_per_block / grouped;
            if grouped == 0 {
                return Ok(None);
            }
            grouped = grouped
                .min(batch_count)
                .min(device.max_workgroup_size[0])
                .max(1);
        }
        let block = StockhamAxisBlockSchedule {
            threads_per_transform,
            grouped_batch: grouped,
            transforms_on_x: true,
            axis_swapped: false,
            local_size_x: grouped,
            local_size_y: threads_per_transform,
        };
        if block.validate(batch_count, device).is_err() {
            return Ok(None);
        }
        return Ok(Some(block));
    }

    if grouped_batch < max_batch_coalesced {
        grouped_batch = max_batch_coalesced;
    }
    grouped_batch = (grouped_batch / max_batch_coalesced) * max_batch_coalesced;
    if sequence_len > max_single_size_strided {
        grouped_batch = (max_sequence_len_shared / sequence_len).max(1);
    }
    let warp_size = policy.subgroup_width.max(1);
    if grouped_batch > warp_size {
        grouped_batch = (grouped_batch / warp_size) * warp_size;
    }
    if grouped_batch > 2 * max_batch_coalesced {
        grouped_batch = (grouped_batch / (2 * max_batch_coalesced)) * (2 * max_batch_coalesced);
    }
    if grouped_batch > 4 * max_batch_coalesced {
        grouped_batch = (grouped_batch / (4 * max_batch_coalesced)) * (4 * max_batch_coalesced);
    }
    grouped_batch = grouped_batch.max(1);
    if grouped_batch
        .checked_mul(threads_per_transform)
        .is_some_and(|threads| threads > device.max_threads_per_block)
    {
        grouped_batch = max_batch_coalesced;
    }

    // Fixed upstream's automatic axis_id>=1 branch does not perform the
    // aimThreads refill used by axis0 follow-up uploads. Keep the coalescing/
    // shared-memory floor above, then apply only the NVIDIA oversized-group clamp.
    const AIM_THREADS: usize = 128;
    let mut grouped = fastest_axis_len.min(grouped_batch).max(1);
    if device.vendor == GpuVendor::Nvidia {
        while grouped
            .checked_mul(threads_per_transform)
            .is_some_and(|threads| threads >= 2 * AIM_THREADS)
            && grouped > max_batch_coalesced
        {
            grouped = (grouped / 2).max(max_batch_coalesced);
        }
    }
    grouped = grouped
        .min(batch_count)
        .min(device.max_workgroup_size[0])
        .max(1);
    if grouped
        .checked_mul(threads_per_transform)
        .is_some_and(|threads| threads > device.max_threads_per_block)
    {
        let original = grouped;
        for divisor in 1..=original {
            let candidate = original / divisor;
            if candidate > 0
                && candidate
                    .checked_mul(threads_per_transform)
                    .is_some_and(|threads| threads <= device.max_threads_per_block)
            {
                grouped = candidate;
                break;
            }
        }
    }
    let block = StockhamAxisBlockSchedule {
        threads_per_transform,
        grouped_batch: grouped,
        transforms_on_x: true,
        axis_swapped: false,
        local_size_x: grouped,
        local_size_y: threads_per_transform,
    };
    if block.validate(batch_count, device).is_err() {
        return Ok(None);
    }
    Ok(Some(block))
}

/// Higher-axis physical ownership for invocation-local elementwise wrapper passes.
/// Reuse the same fixed-upstream axis_id>=1 batch scorer as Rader/Stockham callers,
/// but cap invocation-local element lanes at the 128-thread aim used by wrapper kernels.
/// Logical groupedBatch remains a caller contract; when it is not explicitly set,
/// this helper may still group several physical transforms on X.
pub fn plan_gpu_other_axis_elementwise_wrapper_block(
    element_len: usize,
    batch_count: usize,
    fastest_axis_len: usize,
    complex_bytes: usize,
    grouped_batch_override: Option<usize>,
    axis1_grouped_batch_override: Option<usize>,
    device: DeviceProfile,
) -> Result<Option<StockhamAxisBlockSchedule>> {
    if element_len == 0 || batch_count == 0 || fastest_axis_len == 0 {
        return Ok(None);
    }
    let threads_per_transform = element_len
        .min(128)
        .min(device.max_workgroup_size[1])
        .min(device.max_threads_per_block);
    if threads_per_transform == 0 {
        return Ok(None);
    }
    plan_gpu_other_axis_rader_block_from_threads(
        element_len,
        batch_count,
        fastest_axis_len,
        complex_bytes,
        threads_per_transform,
        grouped_batch_override,
        axis1_grouped_batch_override,
        device,
    )
}

pub fn plan_gpu_other_axis_bluestein_wrapper_block(
    convolution_len: usize,
    batch_count: usize,
    fastest_axis_len: usize,
    complex_bytes: usize,
    grouped_batch_override: Option<usize>,
    axis1_grouped_batch_override: Option<usize>,
    device: DeviceProfile,
) -> Result<Option<StockhamAxisBlockSchedule>> {
    plan_gpu_other_axis_elementwise_wrapper_block(
        convolution_len,
        batch_count,
        fastest_axis_len,
        complex_bytes,
        grouped_batch_override,
        axis1_grouped_batch_override,
        device,
    )
}

/// Fixed-upstream higher-axis parent geometry for a composite Rader/Cooley root.
/// The caller supplies the already-selected parent thread floor; only independent
/// transform ownership is remapped to X/Y, leaving every nested Rader child alone.
pub fn plan_gpu_other_axis_composite_rader_batch_block(
    logical_len: usize,
    batch_count: usize,
    fastest_axis_len: usize,
    complex_bytes: usize,
    threads_per_transform: usize,
    grouped_batch_override: Option<usize>,
    axis1_grouped_batch_override: Option<usize>,
    device: DeviceProfile,
) -> Result<Option<StockhamAxisBlockSchedule>> {
    let precision = match complex_bytes {
        8 => Precision::F32,
        16 => Precision::F64,
        32 => Precision::DoubleDouble,
        _ => return Ok(None),
    };
    plan_gpu_other_axis_composite_rader_batch_block_for_precision(
        logical_len,
        batch_count,
        fastest_axis_len,
        precision,
        complex_bytes,
        threads_per_transform,
        grouped_batch_override,
        axis1_grouped_batch_override,
        device,
    )
}

pub(crate) fn plan_gpu_other_axis_composite_rader_batch_block_for_precision(
    logical_len: usize,
    batch_count: usize,
    fastest_axis_len: usize,
    precision: Precision,
    complex_bytes: usize,
    threads_per_transform: usize,
    grouped_batch_override: Option<usize>,
    axis1_grouped_batch_override: Option<usize>,
    device: DeviceProfile,
) -> Result<Option<StockhamAxisBlockSchedule>> {
    plan_gpu_other_axis_rader_block_from_threads_for_precision(
        logical_len,
        batch_count,
        fastest_axis_len,
        precision,
        complex_bytes,
        threads_per_transform,
        grouped_batch_override,
        axis1_grouped_batch_override,
        device,
    )
}

/// Exact fixed-upstream type-1/direct-Rader thread estimate for a single-upload
/// composite axis containing exactly one direct-Rader prime and a smooth outer
/// container product. This ports the `useRaderMult`/`estimate_rader_threadnum`
/// coupling after the ordinary outer register table has been selected. The returned
/// value is the physical lane count for one axis-0 transform; independent-batch
/// grouping is applied separately by `VkFFTSplitAxisBlock`-style helpers.
pub fn plan_gpu_axis0_composite_direct_rader_threads(
    sequence_len: usize,
    direct_prime: usize,
    batch_count: usize,
    device: DeviceProfile,
) -> Result<Option<usize>> {
    if sequence_len < 2
        || direct_prime < 3
        || batch_count == 0
        || !sequence_len.is_multiple_of(direct_prime)
        || device.max_threads_per_block == 0
        || device.max_workgroup_size[0] == 0
    {
        return Ok(None);
    }
    let container_fft_num = sequence_len / direct_prime;
    if container_fft_num <= 1 {
        return Ok(None);
    }
    let (_, _, base_min_registers) =
        match rader_outer_register_state(sequence_len, container_fft_num, batch_count) {
            Ok(state) => state,
            Err(VkFftError::UnsupportedKernelPath(_)) => return Ok(None),
            Err(error) => return Err(error),
        };
    if base_min_registers == 0 {
        return Ok(None);
    }

    // For axis 0 / upload 0 / one upload, fixed upstream sets maxBatchCoalesced=1.
    // Direct Rader owns `(p + 1) / 2` lanes per simultaneously active container.
    let direct_threads = direct_prime.div_ceil(2);
    let physical_thread_limit = device
        .max_threads_per_block
        .min(device.max_workgroup_size[0]);
    if direct_threads > physical_thread_limit {
        return Ok(None);
    }

    let mut scale_registers_rader = 0usize;
    loop {
        let rader_min_registers = (base_min_registers / 2)
            .checked_add(scale_registers_rader)
            .and_then(|value| value.checked_mul(2))
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "composite direct-Rader minimum register scaling",
            })?;
        if rader_min_registers == 0 {
            return Ok(None);
        }
        let denominator = rader_min_registers.checked_mul(direct_threads).ok_or(
            VkFftError::ArithmeticOverflow {
                operation: "composite direct-Rader container denominator",
            },
        )?;
        let temp_rader = ceil_div(sequence_len, denominator)?.max(1);
        let mut active_rader = ceil_div(container_fft_num, temp_rader)?.max(1);

        // Preserve upstream's fractional active-container rounding: when the ceil
        // overshoot is at least one half and one fewer active container still fits
        // maxThreadsNum, use that smaller active count.
        if active_rader > 1 {
            let covered =
                active_rader
                    .checked_mul(temp_rader)
                    .ok_or(VkFftError::ArithmeticOverflow {
                        operation: "composite direct-Rader active container coverage",
                    })?;
            let overshoot = covered - container_fft_num;
            let round_down = overshoot
                .checked_mul(2)
                .is_some_and(|twice| twice >= temp_rader);
            let fewer_threads = ceil_div(container_fft_num, active_rader - 1)?
                .checked_mul(direct_threads)
                .ok_or(VkFftError::ArithmeticOverflow {
                    operation: "composite direct-Rader rounded thread estimate",
                })?;
            if round_down && fewer_threads <= physical_thread_limit {
                active_rader -= 1;
            }
        }

        let mut estimate_rader_threads = ceil_div(container_fft_num, active_rader)?
            .checked_mul(direct_threads)
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "composite direct-Rader thread estimate",
            })?;
        // Literal integer branch from upstream: the ordinary transform lane count may
        // still dominate the type-1 Rader estimate.
        estimate_rader_threads = estimate_rader_threads.max(sequence_len / rader_min_registers);
        let low_register_pressure = ((sequence_len / base_min_registers) > 256
            || estimate_rader_threads > 256)
            && rader_min_registers <= 4;
        if estimate_rader_threads <= physical_thread_limit && !low_register_pressure {
            return Ok(Some(estimate_rader_threads.max(1)));
        }

        scale_registers_rader =
            scale_registers_rader
                .checked_add(1)
                .ok_or(VkFftError::ArithmeticOverflow {
                    operation: "composite direct-Rader register scale iteration",
                })?;
        // A direct prime that passed the planner's thread cap should converge before
        // this bound. Fail soft rather than inventing a launch shape if a custom
        // profile violates that invariant.
        if scale_registers_rader > sequence_len {
            return Ok(None);
        }
    }
}

/// Multiplicity-aware fixed-upstream type-1/direct-Rader thread estimate. VkFFT
/// creates one `raderContainer` per distinct prime, stores repeated occurrences in
/// `multiplier`, removes every occurrence from the ordinary outer radix product, and
/// still scores the type-1 container only once in the shared register-scale loop.
pub fn plan_gpu_axis0_composite_direct_rader_threads_for_prime_multiplicities(
    sequence_len: usize,
    direct_prime_multiplicities: &[(usize, usize)],
    batch_count: usize,
    device: DeviceProfile,
) -> Result<Option<usize>> {
    plan_gpu_axis0_composite_direct_rader_threads_for_prime_multiplicities_with_max_batch_coalesced(
        sequence_len,
        direct_prime_multiplicities,
        batch_count,
        1,
        device,
    )
}

pub(crate) fn plan_gpu_axis0_composite_direct_rader_threads_for_prime_multiplicities_with_max_batch_coalesced(
    sequence_len: usize,
    direct_prime_multiplicities: &[(usize, usize)],
    batch_count: usize,
    max_batch_coalesced: usize,
    device: DeviceProfile,
) -> Result<Option<usize>> {
    if sequence_len < 2
        || direct_prime_multiplicities.is_empty()
        || batch_count == 0
        || max_batch_coalesced == 0
        || device.max_threads_per_block == 0
        || device.max_workgroup_size[0] == 0
    {
        return Ok(None);
    }
    let physical_thread_limit = device
        .max_threads_per_block
        .min(device.max_workgroup_size[0]);
    let mut rader_product = 1usize;
    for (index, &(prime, multiplicity)) in direct_prime_multiplicities.iter().enumerate() {
        if prime < 3
            || multiplicity == 0
            || !sequence_len.is_multiple_of(prime)
            || direct_prime_multiplicities[..index]
                .iter()
                .any(|&(previous, _)| previous == prime)
        {
            return Ok(None);
        }
        for _ in 0..multiplicity {
            rader_product =
                rader_product
                    .checked_mul(prime)
                    .ok_or(VkFftError::ArithmeticOverflow {
                        operation: "composite direct-Rader prime-power product",
                    })?;
        }
    }
    if !sequence_len.is_multiple_of(rader_product) {
        return Ok(None);
    }
    let smooth_outer_product = sequence_len / rader_product;
    let (_, _, base_min_registers) =
        match rader_outer_register_state(sequence_len, smooth_outer_product, batch_count) {
            Ok(state) => state,
            Err(VkFftError::UnsupportedKernelPath(_)) => return Ok(None),
            Err(error) => return Err(error),
        };
    if base_min_registers == 0 {
        return Ok(None);
    }
    // `VkFFTOptimizeRaderFFTRegisters` merges every Rader container into the
    // parent min/max after optimizing type-0 containers. Type-1 Direct-Rader
    // containers carry a fixed two-register state, so direct-only parents must
    // feed that same minimum into the subsequent scale_registers_rader loop.
    let type1_scale_base_min_registers = base_min_registers.min(2);
    let Some((rader_min_registers, _)) =
        scale_axis0_direct_rader_type1_register_floor_with_max_batch_coalesced(
            sequence_len,
            direct_prime_multiplicities,
            type1_scale_base_min_registers,
            max_batch_coalesced,
            physical_thread_limit,
        )?
    else {
        return Ok(None);
    };

    let mut estimate_rader_threads = sequence_len / rader_min_registers;
    for &(prime, _) in direct_prime_multiplicities {
        let Some(local_estimate) =
            axis0_direct_rader_local_thread_estimate_with_max_batch_coalesced(
                sequence_len,
                prime,
                rader_min_registers,
                max_batch_coalesced,
                physical_thread_limit,
            )?
        else {
            return Ok(None);
        };
        estimate_rader_threads = estimate_rader_threads.max(local_estimate);
    }
    Ok(Some(estimate_rader_threads.max(1)))
}

/// AxisBlockSplitter's per-transform type-1 lane estimate with the scheduler's
/// pass-local `maxBatchCoalesced` participating only in the active-container
/// round-down feasibility check. The returned value is still lanes per transform;
/// physical transform grouping is materialized separately by the axis-block planner.
fn axis0_direct_rader_local_thread_estimate_with_max_batch_coalesced(
    sequence_len: usize,
    prime: usize,
    rader_min_registers: usize,
    max_batch_coalesced: usize,
    physical_thread_limit: usize,
) -> Result<Option<usize>> {
    if prime < 3
        || rader_min_registers == 0
        || max_batch_coalesced == 0
        || !sequence_len.is_multiple_of(prime)
    {
        return Ok(None);
    }
    let direct_threads = prime.div_ceil(2);
    if direct_threads
        .checked_mul(max_batch_coalesced)
        .is_none_or(|threads| threads > physical_thread_limit)
    {
        return Ok(None);
    }
    let container_fft_num = sequence_len / prime;
    let denominator =
        rader_min_registers
            .checked_mul(direct_threads)
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "axis0 direct-Rader type-1 denominator",
            })?;
    let temp_rader = ceil_div(sequence_len, denominator)?.max(1);
    let mut active_rader = ceil_div(container_fft_num, temp_rader)?.max(1);
    if active_rader > 1 {
        let covered =
            active_rader
                .checked_mul(temp_rader)
                .ok_or(VkFftError::ArithmeticOverflow {
                    operation: "axis0 direct-Rader type-1 active coverage",
                })?;
        let overshoot = covered - container_fft_num;
        let round_down = overshoot
            .checked_mul(2)
            .is_some_and(|twice| twice >= temp_rader);
        let fewer_threads = ceil_div(container_fft_num, active_rader - 1)?
            .checked_mul(direct_threads)
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "axis0 direct-Rader type-1 rounded estimate",
            })?;
        let fewer_total = fewer_threads.checked_mul(max_batch_coalesced).ok_or(
            VkFftError::ArithmeticOverflow {
                operation: "axis0 direct-Rader coalesced rounded estimate",
            },
        )?;
        if round_down && fewer_total <= physical_thread_limit {
            active_rader -= 1;
        }
    }
    Ok(Some(
        ceil_div(container_fft_num, active_rader)?
            .checked_mul(direct_threads)
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "axis0 direct-Rader type-1 local estimate",
            })?,
    ))
}

/// Fixed-upstream `scale_registers_rader` loop for axis-0 one-upload type-1
/// containers. Returns the final even Rader register floor and the scheduler's
/// total direct-Rader thread pressure before AxisBlockSplitter performs its
/// final per-transform multiple-of-(p+1)/2 rounding.
fn scale_axis0_direct_rader_type1_register_floor_with_max_batch_coalesced(
    sequence_len: usize,
    direct_prime_multiplicities: &[(usize, usize)],
    base_min_registers: usize,
    max_batch_coalesced: usize,
    physical_thread_limit: usize,
) -> Result<Option<(usize, usize)>> {
    if direct_prime_multiplicities.is_empty()
        || base_min_registers == 0
        || max_batch_coalesced == 0
        || physical_thread_limit == 0
    {
        return Ok(None);
    }
    let mut scale_registers_rader = 0usize;
    loop {
        let rader_min_registers = (base_min_registers / 2)
            .checked_add(scale_registers_rader)
            .and_then(|value| value.checked_mul(2))
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "axis0 direct-Rader shared register scaling",
            })?;
        if rader_min_registers == 0 {
            return Ok(None);
        }
        let mut estimate_rader_threads = (sequence_len / rader_min_registers)
            .checked_mul(max_batch_coalesced)
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "axis0 direct-Rader coalesced ordinary pressure",
            })?;
        for &(prime, multiplicity) in direct_prime_multiplicities {
            if multiplicity == 0 {
                return Ok(None);
            }
            let Some(local_estimate) =
                axis0_direct_rader_local_thread_estimate_with_max_batch_coalesced(
                    sequence_len,
                    prime,
                    rader_min_registers,
                    max_batch_coalesced,
                    physical_thread_limit,
                )?
            else {
                return Ok(None);
            };
            let local_total = local_estimate.checked_mul(max_batch_coalesced).ok_or(
                VkFftError::ArithmeticOverflow {
                    operation: "axis0 direct-Rader coalesced local pressure",
                },
            )?;
            estimate_rader_threads = estimate_rader_threads.max(local_total);
        }
        let low_register_pressure = ((sequence_len / base_min_registers) > 256
            || estimate_rader_threads > 256)
            && rader_min_registers <= 4;
        if estimate_rader_threads <= physical_thread_limit && !low_register_pressure {
            return Ok(Some((rader_min_registers, estimate_rader_threads.max(1))));
        }
        scale_registers_rader =
            scale_registers_rader
                .checked_add(1)
                .ok_or(VkFftError::ArithmeticOverflow {
                    operation: "axis0 direct-Rader scale iteration",
                })?;
        if scale_registers_rader > sequence_len {
            return Ok(None);
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MultiFftRaderContainerKind {
    Fft,
    Direct,
}

#[derive(Debug, Clone)]
struct MultiFftRaderContainerRegisterState {
    kind: MultiFftRaderContainerKind,
    prime: usize,
    container_fft_num: usize,
    registers_per_thread_per_radix: [usize; VKFFT_RADIX_TABLE_LEN],
    registers_per_thread: usize,
    min_registers_per_thread: usize,
    stage_radix_multipliers: [usize; VKFFT_RADIX_TABLE_LEN],
    subcontainers: Vec<MultiFftRaderContainerRegisterState>,
}

fn nested_rader_container_kind(
    prime: usize,
    tuning: PlannerTuning,
) -> Option<MultiFftRaderContainerKind> {
    if prime < 3 {
        return None;
    }
    let fft_eligible = prime >= tuning.min_rader_fft_prime && prime < tuning.max_rader_fft_prime;
    let safe_fft = fft_eligible
        && distinct_prime_factors(prime - 1)
            .into_iter()
            .all(|(factor, _)| factor < tuning.min_rader_direct_prime);
    if safe_fft {
        return Some(MultiFftRaderContainerKind::Fft);
    }
    if prime >= tuning.min_rader_direct_prime && prime < tuning.max_rader_direct_prime {
        return Some(MultiFftRaderContainerKind::Direct);
    }
    if tuning.allow_recursive_fft_rader && fft_eligible {
        return Some(MultiFftRaderContainerKind::Fft);
    }
    None
}

fn distinct_prime_factors(mut value: usize) -> Vec<(usize, usize)> {
    let mut factors = Vec::new();
    let mut candidate = 2usize;
    while candidate <= value / candidate {
        if value.is_multiple_of(candidate) {
            let mut multiplicity = 0usize;
            while value.is_multiple_of(candidate) {
                value /= candidate;
                multiplicity += 1;
            }
            factors.push((candidate, multiplicity));
        }
        candidate += if candidate == 2 { 1 } else { 2 };
    }
    if value > 1 {
        factors.push((value, 1));
    }
    factors
}

fn build_nested_fft_rader_subcontainers(
    residual: usize,
    fft_radix_part: usize,
    tuning: PlannerTuning,
) -> Result<Option<Vec<MultiFftRaderContainerRegisterState>>> {
    if residual == 1 {
        return Ok(Some(Vec::new()));
    }
    if fft_radix_part == 0 {
        return Ok(None);
    }
    let mut classified = Vec::new();
    for (prime, _) in distinct_prime_factors(residual) {
        let Some(kind) = nested_rader_container_kind(prime, tuning) else {
            return Ok(None);
        };
        classified.push((prime, kind));
    }

    // VkFFTConstructRaderTree materializes containers in two distinct scans over the
    // residual: all type-0 FFT-Rader primes first, then the rejected/direct type-1
    // primes. Preserve that order because VkFFTOptimizeRaderFFTRegisters carries a
    // shared min/max register state from one container into the next.
    let mut subcontainers = Vec::with_capacity(classified.len());
    for desired_kind in [
        MultiFftRaderContainerKind::Fft,
        MultiFftRaderContainerKind::Direct,
    ] {
        for &(prime, kind) in &classified {
            if kind != desired_kind {
                continue;
            }
            let container_fft_num = fft_radix_part.checked_mul(residual / prime).ok_or(
                VkFftError::ArithmeticOverflow {
                    operation: "nested Rader container count",
                },
            )?;
            let container = match kind {
                MultiFftRaderContainerKind::Fft => {
                    let Some(container) = build_multi_fft_rader_container_state_with_tuning(
                        prime,
                        container_fft_num,
                        tuning,
                    )?
                    else {
                        return Ok(None);
                    };
                    container
                }
                MultiFftRaderContainerKind::Direct => MultiFftRaderContainerRegisterState {
                    kind,
                    prime,
                    container_fft_num,
                    registers_per_thread_per_radix: [0usize; VKFFT_RADIX_TABLE_LEN],
                    registers_per_thread: 2,
                    min_registers_per_thread: 2,
                    stage_radix_multipliers: [0usize; VKFFT_RADIX_TABLE_LEN],
                    subcontainers: Vec::new(),
                },
            };
            subcontainers.push(container);
        }
    }
    Ok(Some(subcontainers))
}

fn build_multi_fft_rader_container_state_with_tuning(
    prime: usize,
    container_fft_num: usize,
    tuning: PlannerTuning,
) -> Result<Option<MultiFftRaderContainerRegisterState>> {
    if prime < 3 || container_fft_num == 0 {
        return Ok(None);
    }
    let (stage_radix_multipliers, residual) =
        rader_small_radix_multipliers_with_residual(prime - 1, tuning.min_rader_direct_prime)?;
    let (registers_per_thread_per_radix, registers_per_thread, min_registers_per_thread) =
        rader_optimize_shared_register_table(prime - 1)?;
    let Some(subcontainers) =
        build_nested_fft_rader_subcontainers(residual, container_fft_num, tuning)?
    else {
        return Ok(None);
    };
    Ok(Some(MultiFftRaderContainerRegisterState {
        kind: MultiFftRaderContainerKind::Fft,
        prime,
        container_fft_num,
        registers_per_thread_per_radix,
        registers_per_thread,
        min_registers_per_thread,
        stage_radix_multipliers,
        subcontainers,
    }))
}

#[cfg(test)]
fn build_multi_fft_rader_container_state(
    prime: usize,
    container_fft_num: usize,
) -> Result<Option<MultiFftRaderContainerRegisterState>> {
    build_multi_fft_rader_container_state_with_tuning(
        prime,
        container_fft_num,
        PlannerTuning::portable(),
    )
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MultiFftRaderParentSchedule {
    pub(crate) global_scale_registers_num: usize,
    pub(crate) final_min_registers: usize,
    pub(crate) min_rader_fft_thread_num: usize,
    pub(crate) threads_per_transform: usize,
}

/// One exact, non-recursive invocation of fixed-upstream
/// `VkFFTOptimizeRaderFFTRegisters` for a set of smooth type-0 containers.
/// All containers deliberately share the same outer register state: processing them
/// independently loses the global max/min feedback that the upstream loop carries from
/// one Rader prime into the next.
fn optimize_smooth_fft_rader_containers_once(
    sequence_len: usize,
    outer_registers_per_radix: &mut [usize; VKFFT_RADIX_TABLE_LEN],
    outer_registers_per_thread: &mut usize,
    outer_min_registers: &mut usize,
    containers: &mut [MultiFftRaderContainerRegisterState],
) -> Result<()> {
    if sequence_len == 0 || *outer_min_registers == 0 {
        return Err(VkFftError::InvalidKernelIr(
            "multi FFT-Rader optimizer requires a non-zero outer register floor",
        ));
    }

    for container in containers.iter_mut() {
        if container.kind == MultiFftRaderContainerKind::Direct {
            continue;
        }
        if container.min_registers_per_thread / *outer_min_registers >= 2 {
            *outer_min_registers *= container.min_registers_per_thread / *outer_min_registers;
            for value in outer_registers_per_radix.iter_mut() {
                if *value > 0 && *value < *outer_min_registers {
                    *value *= ceil_div(*outer_min_registers, *value)?;
                }
                *outer_registers_per_thread = (*outer_registers_per_thread).max(*value);
            }
        } else if *outer_min_registers / container.min_registers_per_thread >= 2 {
            container.min_registers_per_thread *=
                *outer_min_registers / container.min_registers_per_thread;
            for value in container.registers_per_thread_per_radix.iter_mut() {
                if *value > 0 && *value < container.min_registers_per_thread {
                    *value *= ceil_div(container.min_registers_per_thread, *value)?;
                }
                container.registers_per_thread = container.registers_per_thread.max(*value);
            }
        }

        if container.min_registers_per_thread < *outer_min_registers {
            for (radix, value) in container
                .registers_per_thread_per_radix
                .iter_mut()
                .enumerate()
                .skip(2)
            {
                if *value == 0 {
                    continue;
                }
                while *value < *outer_min_registers {
                    *value = value
                        .checked_add(radix)
                        .ok_or(VkFftError::ArithmeticOverflow {
                            operation: "multi FFT-Rader register alignment",
                        })?;
                }
                container.registers_per_thread = container.registers_per_thread.max(*value);
            }
        }

        let outer_threads = ceil_div(sequence_len, *outer_min_registers)?;
        let container_fft_dim = container.prime - 1;
        for (radix, value) in container
            .registers_per_thread_per_radix
            .iter_mut()
            .enumerate()
            .skip(2)
        {
            if *value == 0 {
                continue;
            }
            while !rader_container_occupancy_fits(
                outer_threads,
                container.container_fft_num,
                container_fft_dim,
                *value,
            )? {
                *value = value
                    .checked_add(radix)
                    .ok_or(VkFftError::ArithmeticOverflow {
                        operation: "multi FFT-Rader occupancy register scaling",
                    })?;
            }
            container.registers_per_thread = container.registers_per_thread.max(*value);
        }
        *outer_registers_per_thread =
            (*outer_registers_per_thread).max(container.registers_per_thread);
    }

    // Upstream's second loop in the same optimizer invocation raises every type-0
    // radix toward the global register maximum established by all sibling containers.
    for container in containers.iter_mut() {
        if container.kind == MultiFftRaderContainerKind::Direct {
            continue;
        }
        for (radix, value) in container
            .registers_per_thread_per_radix
            .iter_mut()
            .enumerate()
            .skip(2)
        {
            if *value == 0 {
                continue;
            }
            while value
                .checked_add(radix)
                .is_some_and(|next| next <= *outer_registers_per_thread + 1)
            {
                *value += radix;
            }
        }
        let (_, max_registers, min_registers) =
            rader_register_min_max(&container.registers_per_thread_per_radix)?;
        container.registers_per_thread = max_registers;
        container.min_registers_per_thread = min_registers;
    }

    // VkFFTOptimizeRaderFFTRegisters recursively optimizes sub-prime containers with
    // the same top-level fftDim and the same outer register state. Parent cached
    // min/max values are merged only after the recursive calls return.
    for container in containers.iter_mut() {
        if !container.subcontainers.is_empty() {
            optimize_smooth_fft_rader_containers_once(
                sequence_len,
                outer_registers_per_radix,
                outer_registers_per_thread,
                outer_min_registers,
                &mut container.subcontainers,
            )?;
        }
    }

    for container in containers.iter() {
        *outer_min_registers = (*outer_min_registers).min(container.min_registers_per_thread);
        *outer_registers_per_thread =
            (*outer_registers_per_thread).max(container.registers_per_thread);
    }
    Ok(())
}

fn optimize_and_collect_recursive_fft_rader_pressure(
    container: &mut MultiFftRaderContainerRegisterState,
    final_min_registers: &mut usize,
    min_rader_fft_thread_num: &mut usize,
) -> Result<()> {
    if container.kind == MultiFftRaderContainerKind::Direct {
        return Ok(());
    }
    let (_, _) = optimize_radix_kernels(
        &mut container.registers_per_thread_per_radix,
        &mut container.stage_radix_multipliers,
        1,
    );
    let mut saw_parent_stage = false;
    for radix in (2..VKFFT_RADIX_TABLE_LEN).rev() {
        for _ in 0..container.stage_radix_multipliers[radix] {
            let registers = container.registers_per_thread_per_radix[radix];
            // Fixed upstream may keep a locally peeled radix in `stageRadix` even when
            // `VkFFTGetRegistersPerThreadOptimizeShared` has no register allocation for
            // that radix. The custom p103 -> 102 = 2*3*17 cross-Bluestein case is the
            // minimal witness: radix 17 remains a stage with register count zero, while
            // only the executable radix-3/radix-2 stages contribute to parent min-register
            // and Rader-thread pressure. Treat the zero-allocation stage as
            // non-contributing instead of rejecting an otherwise valid recursive tree.
            if registers == 0 {
                continue;
            }
            saw_parent_stage = true;
            *final_min_registers = (*final_min_registers).min(registers);
            let stage_threads = container
                .container_fft_num
                .checked_mul(ceil_div(container.prime - 1, registers)?)
                .ok_or(VkFftError::ArithmeticOverflow {
                    operation: "recursive FFT-Rader child thread pressure",
                })?;
            *min_rader_fft_thread_num = (*min_rader_fft_thread_num).max(stage_threads);
        }
    }
    for subcontainer in &mut container.subcontainers {
        optimize_and_collect_recursive_fft_rader_pressure(
            subcontainer,
            final_min_registers,
            min_rader_fft_thread_num,
        )?;
    }
    if !saw_parent_stage && container.subcontainers.is_empty() {
        return Err(VkFftError::InvalidKernelIr(
            "recursive FFT-Rader container has no executable parent or sub-prime stage",
        ));
    }
    Ok(())
}

fn smooth_outer_primitive_multipliers(
    smooth_outer_factor: usize,
) -> Result<[usize; VKFFT_RADIX_TABLE_LEN]> {
    if smooth_outer_factor == 0 {
        return Err(VkFftError::UnsupportedKernelPath(
            "Rader smooth outer factor must be non-zero",
        ));
    }
    let mut remaining = smooth_outer_factor;
    let mut multipliers = [0usize; VKFFT_RADIX_TABLE_LEN];
    for prime in [2usize, 3, 5, 7, 11, 13] {
        while remaining.is_multiple_of(prime) {
            remaining /= prime;
            multipliers[prime] += 1;
        }
    }
    if remaining != 1 {
        return Err(VkFftError::UnsupportedKernelPath(
            "Rader smooth outer factor contains a non-Stockham prime",
        ));
    }
    Ok(multipliers)
}

pub(crate) fn plan_gpu_axis0_multi_fft_rader_parent_schedule(
    sequence_len: usize,
    fft_prime_multiplicities: &[(usize, usize)],
    batch_count: usize,
    device: DeviceProfile,
) -> Result<Option<MultiFftRaderParentSchedule>> {
    plan_gpu_axis0_multi_fft_rader_parent_schedule_with_tuning(
        sequence_len,
        fft_prime_multiplicities,
        batch_count,
        PlannerTuning::portable(),
        device,
    )
}

fn plan_gpu_axis0_multi_fft_rader_parent_schedule_with_tuning(
    sequence_len: usize,
    fft_prime_multiplicities: &[(usize, usize)],
    batch_count: usize,
    tuning: PlannerTuning,
    device: DeviceProfile,
) -> Result<Option<MultiFftRaderParentSchedule>> {
    plan_gpu_axis0_multi_fft_rader_parent_schedule_with_max_batch_coalesced(
        sequence_len,
        fft_prime_multiplicities,
        batch_count,
        1,
        false,
        tuning,
        device,
    )
}

fn plan_gpu_axis0_multi_fft_rader_parent_schedule_with_max_batch_coalesced(
    sequence_len: usize,
    fft_prime_multiplicities: &[(usize, usize)],
    batch_count: usize,
    max_batch_coalesced: usize,
    shrink_first_upload_coalescing: bool,
    tuning: PlannerTuning,
    device: DeviceProfile,
) -> Result<Option<MultiFftRaderParentSchedule>> {
    if sequence_len < 2
        || fft_prime_multiplicities.is_empty()
        || batch_count == 0
        || max_batch_coalesced == 0
        || device.max_threads_per_block == 0
        || device.max_workgroup_size[0] == 0
    {
        return Ok(None);
    }
    let physical_thread_limit = device
        .max_threads_per_block
        .min(device.max_workgroup_size[0]);
    let mut rader_product = 1usize;
    let mut containers = Vec::with_capacity(fft_prime_multiplicities.len());
    for (index, &(prime, multiplicity)) in fft_prime_multiplicities.iter().enumerate() {
        if prime < 3
            || multiplicity == 0
            || !sequence_len.is_multiple_of(prime)
            || fft_prime_multiplicities[..index]
                .iter()
                .any(|&(previous, _)| previous == prime)
        {
            return Ok(None);
        }
        for _ in 0..multiplicity {
            rader_product =
                rader_product
                    .checked_mul(prime)
                    .ok_or(VkFftError::ArithmeticOverflow {
                        operation: "multi FFT-Rader prime-power product",
                    })?;
        }
        let Some(container) =
            build_multi_fft_rader_container_state_with_tuning(prime, sequence_len / prime, tuning)?
        else {
            return Ok(None);
        };
        containers.push(container);
    }
    if !sequence_len.is_multiple_of(rader_product) {
        return Ok(None);
    }
    let smooth_outer_factor = sequence_len / rader_product;
    let (mut outer_registers_per_radix, mut outer_registers_per_thread, mut outer_min_registers) =
        match rader_outer_register_state(sequence_len, smooth_outer_factor, batch_count) {
            Ok(state) => state,
            Err(VkFftError::UnsupportedKernelPath(_)) => return Ok(None),
            Err(error) => return Err(error),
        };
    let mut outer_stage_multipliers = match smooth_outer_primitive_multipliers(smooth_outer_factor)
    {
        Ok(multipliers) => multipliers,
        Err(VkFftError::UnsupportedKernelPath(_)) => return Ok(None),
        Err(error) => return Err(error),
    };
    // Fixed upstream invokes VkFFTOptimizeRaderFFTRegisters twice with
    // `scaleRegistersNum` in between. For pure type-0 axes the global branch is driven
    // only by the ordinary parent pressure; type-0 child pressure is measured later by
    // VkFFTGetRaderFFTThreadsNum.
    optimize_smooth_fft_rader_containers_once(
        sequence_len,
        &mut outer_registers_per_radix,
        &mut outer_registers_per_thread,
        &mut outer_min_registers,
        &mut containers,
    )?;
    let coalesced_elements =
        sequence_len
            .checked_mul(max_batch_coalesced)
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "pure FFT-Rader coalesced element pressure",
            })?;
    let global_scale_registers_num =
        if ceil_div(coalesced_elements, outer_min_registers)? > physical_thread_limit {
            scale_axis0_rader_register_tables_with_max_batch_coalesced(
                sequence_len,
                max_batch_coalesced,
                shrink_first_upload_coalescing,
                physical_thread_limit,
                &mut outer_registers_per_radix,
                &mut outer_registers_per_thread,
                &mut outer_min_registers,
                &mut containers,
            )?
        } else {
            1
        };
    optimize_smooth_fft_rader_containers_once(
        sequence_len,
        &mut outer_registers_per_radix,
        &mut outer_registers_per_thread,
        &mut outer_min_registers,
        &mut containers,
    )?;

    let mut final_min_registers = usize::MAX;
    if smooth_outer_factor > 1 {
        let (_, _) = optimize_radix_kernels(
            &mut outer_registers_per_radix,
            &mut outer_stage_multipliers,
            1,
        );
        for radix in 2..VKFFT_RADIX_TABLE_LEN {
            if outer_stage_multipliers[radix] == 0 {
                continue;
            }
            let registers = outer_registers_per_radix[radix];
            if registers == 0 {
                return Err(VkFftError::InvalidKernelIr(
                    "Rader smooth outer stage has no register allocation",
                ));
            }
            final_min_registers = final_min_registers.min(registers);
        }
    }
    let mut min_rader_fft_thread_num = 0usize;
    for container in &mut containers {
        optimize_and_collect_recursive_fft_rader_pressure(
            container,
            &mut final_min_registers,
            &mut min_rader_fft_thread_num,
        )?;
    }
    if final_min_registers == usize::MAX || final_min_registers == 0 {
        return Ok(None);
    }
    let ordinary_threads = ceil_div(sequence_len, final_min_registers)?;
    let threads_per_transform = ordinary_threads.max(min_rader_fft_thread_num);
    // The uncovered upstream branch globally scales register tables when this pressure
    // exceeds the physical workgroup limit. Do not silently approximate that branch.
    if threads_per_transform > physical_thread_limit {
        return Ok(None);
    }
    Ok(Some(MultiFftRaderParentSchedule {
        global_scale_registers_num,
        final_min_registers,
        min_rader_fft_thread_num,
        threads_per_transform,
    }))
}

/// Exact fixed-upstream axis-0 parent lane floor for one or more smooth FFT-Rader
/// type-0 containers plus an optional Stockham-smooth outer factor, with no type-1
/// Direct-Rader factor in the upload.
pub fn plan_gpu_axis0_multi_fft_rader_threads_for_prime_multiplicities(
    sequence_len: usize,
    fft_prime_multiplicities: &[(usize, usize)],
    batch_count: usize,
    device: DeviceProfile,
) -> Result<Option<usize>> {
    Ok(plan_gpu_axis0_multi_fft_rader_parent_schedule(
        sequence_len,
        fft_prime_multiplicities,
        batch_count,
        device,
    )?
    .map(|schedule| schedule.threads_per_transform))
}

pub(crate) fn plan_gpu_axis0_multi_fft_rader_threads_for_prime_multiplicities_with_tuning(
    sequence_len: usize,
    fft_prime_multiplicities: &[(usize, usize)],
    batch_count: usize,
    tuning: PlannerTuning,
    device: DeviceProfile,
) -> Result<Option<usize>> {
    Ok(plan_gpu_axis0_multi_fft_rader_parent_schedule_with_tuning(
        sequence_len,
        fft_prime_multiplicities,
        batch_count,
        tuning,
        device,
    )?
    .map(|schedule| schedule.threads_per_transform))
}

#[cfg(test)]
pub(crate) fn plan_gpu_axis0_multi_fft_rader_threads_for_prime_multiplicities_with_max_batch_coalesced(
    sequence_len: usize,
    fft_prime_multiplicities: &[(usize, usize)],
    batch_count: usize,
    max_batch_coalesced: usize,
    device: DeviceProfile,
) -> Result<Option<usize>> {
    Ok(
        plan_gpu_axis0_multi_fft_rader_parent_schedule_with_max_batch_coalesced(
            sequence_len,
            fft_prime_multiplicities,
            batch_count,
            max_batch_coalesced,
            false,
            PlannerTuning::portable(),
            device,
        )?
        .map(|schedule| schedule.threads_per_transform),
    )
}

#[cfg(test)]
pub(crate) fn plan_gpu_axis0_multi_fft_rader_threads_for_forced_upload(
    sequence_len: usize,
    fft_prime_multiplicities: &[(usize, usize)],
    batch_count: usize,
    max_batch_coalesced: usize,
    axis_upload_id: usize,
    device: DeviceProfile,
) -> Result<Option<usize>> {
    plan_gpu_axis0_multi_fft_rader_threads_with_pass_context_and_tuning(
        sequence_len,
        fft_prime_multiplicities,
        batch_count,
        max_batch_coalesced,
        axis_upload_id == 0,
        PlannerTuning::portable(),
        device,
    )
}

pub(crate) fn plan_gpu_axis0_multi_fft_rader_threads_with_pass_context_and_tuning(
    sequence_len: usize,
    fft_prime_multiplicities: &[(usize, usize)],
    batch_count: usize,
    max_batch_coalesced: usize,
    shrink_first_upload_coalescing: bool,
    tuning: PlannerTuning,
    device: DeviceProfile,
) -> Result<Option<usize>> {
    Ok(
        plan_gpu_axis0_multi_fft_rader_parent_schedule_with_max_batch_coalesced(
            sequence_len,
            fft_prime_multiplicities,
            batch_count,
            max_batch_coalesced,
            shrink_first_upload_coalescing,
            tuning,
            device,
        )?
        .map(|schedule| schedule.threads_per_transform),
    )
}

/// Exact axis-0 one-upload parent floor for one or more type-1 Direct-Rader
/// Fixed-upstream `scaleRegistersNum` branch for an axis-0 Rader pass. The
/// scheduler's pass-local `maxBatchCoalesced` participates in the pressure test,
/// while the resulting register tables still describe one transform. Upstream scales
/// the ordinary table first, then only scales type-0 FFT-Rader tables if their
/// still-unscaled minimum keeps the coalesced axis over the workgroup limit. Container
/// cached min/max fields intentionally remain untouched until the second
/// `VkFFTOptimizeRaderFFTRegisters` pass, matching the C implementation.
fn scale_axis0_rader_register_tables_with_max_batch_coalesced(
    sequence_len: usize,
    max_batch_coalesced: usize,
    shrink_first_upload_coalescing: bool,
    physical_thread_limit: usize,
    outer_registers_per_radix: &mut [usize; VKFFT_RADIX_TABLE_LEN],
    registers_per_thread: &mut usize,
    min_registers_per_thread: &mut usize,
    containers: &mut [MultiFftRaderContainerRegisterState],
) -> Result<usize> {
    if sequence_len == 0
        || max_batch_coalesced == 0
        || physical_thread_limit == 0
        || *registers_per_thread == 0
        || *min_registers_per_thread == 0
    {
        return Err(VkFftError::InvalidKernelIr(
            "axis-0 global Rader register scaling requires non-zero dimensions",
        ));
    }
    let mut effective_max_batch_coalesced = max_batch_coalesced;
    if shrink_first_upload_coalescing && effective_max_batch_coalesced > 1 {
        effective_max_batch_coalesced = physical_thread_limit
            .checked_mul(*min_registers_per_thread)
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "axis-0 upload-0 coalescing shrink numerator",
            })?
            / sequence_len;
        effective_max_batch_coalesced = effective_max_batch_coalesced.max(1);
    }
    let coalesced_elements = sequence_len
        .checked_mul(effective_max_batch_coalesced)
        .ok_or(VkFftError::ArithmeticOverflow {
            operation: "axis-0 global Rader coalesced element pressure",
        })?;

    let axis_threads = ceil_div(coalesced_elements, *min_registers_per_thread)?;
    let scale_registers_num = if axis_threads > physical_thread_limit {
        let denominator = min_registers_per_thread
            .checked_mul(physical_thread_limit)
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "axis-0 global Rader register scaling denominator",
            })?;
        ceil_div(coalesced_elements, denominator)?.max(1)
    } else {
        1
    };

    *min_registers_per_thread = min_registers_per_thread
        .checked_mul(scale_registers_num)
        .ok_or(VkFftError::ArithmeticOverflow {
            operation: "axis-0 global Rader minimum register scaling",
        })?;
    *registers_per_thread = registers_per_thread
        .checked_mul(scale_registers_num)
        .ok_or(VkFftError::ArithmeticOverflow {
            operation: "axis-0 global Rader maximum register scaling",
        })?;
    for value in outer_registers_per_radix
        .iter_mut()
        .filter(|value| **value > 0)
    {
        *value = value
            .checked_mul(scale_registers_num)
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "axis-0 global ordinary register-table scaling",
            })?;
    }

    let mut new_min_registers = usize::MAX;
    for value in outer_registers_per_radix
        .iter()
        .copied()
        .filter(|value| *value > 0)
    {
        new_min_registers = new_min_registers.min(value);
    }
    for container in containers.iter() {
        for value in container
            .registers_per_thread_per_radix
            .iter()
            .copied()
            .filter(|value| *value > 0)
        {
            new_min_registers = new_min_registers.min(value);
        }
    }
    if new_min_registers == usize::MAX || new_min_registers == 0 {
        return Err(VkFftError::InvalidKernelIr(
            "axis-0 global Rader scaling found no active register table",
        ));
    }

    if ceil_div(coalesced_elements, new_min_registers)? > physical_thread_limit {
        for container in containers.iter_mut() {
            for value in container
                .registers_per_thread_per_radix
                .iter_mut()
                .filter(|value| **value > 0)
            {
                *value = value.checked_mul(scale_registers_num).ok_or(
                    VkFftError::ArithmeticOverflow {
                        operation: "axis-0 global FFT-Rader register-table scaling",
                    },
                )?;
            }
        }
    } else {
        *min_registers_per_thread = new_min_registers;
    }

    if *min_registers_per_thread > *registers_per_thread {
        core::mem::swap(min_registers_per_thread, registers_per_thread);
    }
    for value in outer_registers_per_radix
        .iter()
        .copied()
        .filter(|value| *value > 0)
    {
        *registers_per_thread = (*registers_per_thread).max(value);
        *min_registers_per_thread = (*min_registers_per_thread).min(value);
    }
    for container in containers.iter() {
        for value in container
            .registers_per_thread_per_radix
            .iter()
            .copied()
            .filter(|value| *value > 0)
        {
            *registers_per_thread = (*registers_per_thread).max(value);
            *min_registers_per_thread = (*min_registers_per_thread).min(value);
        }
    }
    Ok(scale_registers_num)
}

pub fn plan_gpu_axis0_mixed_direct_multi_fft_rader_threads(
    sequence_len: usize,
    direct_prime_multiplicities: &[(usize, usize)],
    fft_prime_multiplicities: &[(usize, usize)],
    batch_count: usize,
    device: DeviceProfile,
) -> Result<Option<usize>> {
    plan_gpu_axis0_mixed_direct_multi_fft_rader_threads_with_tuning(
        sequence_len,
        direct_prime_multiplicities,
        fft_prime_multiplicities,
        batch_count,
        PlannerTuning::portable(),
        device,
    )
}

pub(crate) fn plan_gpu_axis0_mixed_direct_multi_fft_rader_threads_with_tuning(
    sequence_len: usize,
    direct_prime_multiplicities: &[(usize, usize)],
    fft_prime_multiplicities: &[(usize, usize)],
    batch_count: usize,
    tuning: PlannerTuning,
    device: DeviceProfile,
) -> Result<Option<usize>> {
    plan_gpu_axis0_mixed_direct_multi_fft_rader_threads_with_pass_context(
        sequence_len,
        direct_prime_multiplicities,
        fft_prime_multiplicities,
        batch_count,
        1,
        false,
        tuning,
        device,
    )
}

#[cfg(test)]
pub(crate) fn plan_gpu_axis0_mixed_direct_multi_fft_rader_threads_with_max_batch_coalesced(
    sequence_len: usize,
    direct_prime_multiplicities: &[(usize, usize)],
    fft_prime_multiplicities: &[(usize, usize)],
    batch_count: usize,
    max_batch_coalesced: usize,
    device: DeviceProfile,
) -> Result<Option<usize>> {
    plan_gpu_axis0_mixed_direct_multi_fft_rader_threads_with_pass_context(
        sequence_len,
        direct_prime_multiplicities,
        fft_prime_multiplicities,
        batch_count,
        max_batch_coalesced,
        false,
        PlannerTuning::portable(),
        device,
    )
}

pub(crate) fn plan_gpu_axis0_mixed_direct_multi_fft_rader_threads_with_pass_context(
    sequence_len: usize,
    direct_prime_multiplicities: &[(usize, usize)],
    fft_prime_multiplicities: &[(usize, usize)],
    batch_count: usize,
    max_batch_coalesced: usize,
    shrink_first_upload_coalescing: bool,
    tuning: PlannerTuning,
    device: DeviceProfile,
) -> Result<Option<usize>> {
    if sequence_len < 2
        || direct_prime_multiplicities.is_empty()
        || fft_prime_multiplicities.is_empty()
        || batch_count == 0
        || max_batch_coalesced == 0
        || device.max_threads_per_block == 0
        || device.max_workgroup_size[0] == 0
    {
        return Ok(None);
    }
    let physical_thread_limit = device
        .max_threads_per_block
        .min(device.max_workgroup_size[0]);
    let mut rader_product = 1usize;
    for (index, &(prime, multiplicity)) in direct_prime_multiplicities.iter().enumerate() {
        if prime < 3
            || multiplicity == 0
            || !sequence_len.is_multiple_of(prime)
            || prime.div_ceil(2) > physical_thread_limit
            || direct_prime_multiplicities[..index]
                .iter()
                .any(|&(previous, _)| previous == prime)
        {
            return Ok(None);
        }
        for _ in 0..multiplicity {
            rader_product =
                rader_product
                    .checked_mul(prime)
                    .ok_or(VkFftError::ArithmeticOverflow {
                        operation: "mixed multi-Rader direct prime-power product",
                    })?;
        }
    }

    let mut containers = Vec::with_capacity(fft_prime_multiplicities.len());
    for (index, &(prime, multiplicity)) in fft_prime_multiplicities.iter().enumerate() {
        if prime < 3
            || multiplicity == 0
            || !sequence_len.is_multiple_of(prime)
            || direct_prime_multiplicities
                .iter()
                .any(|&(direct, _)| direct == prime)
            || fft_prime_multiplicities[..index]
                .iter()
                .any(|&(previous, _)| previous == prime)
        {
            return Ok(None);
        }
        for _ in 0..multiplicity {
            rader_product =
                rader_product
                    .checked_mul(prime)
                    .ok_or(VkFftError::ArithmeticOverflow {
                        operation: "mixed multi-Rader FFT prime-power product",
                    })?;
        }
        let Some(container) =
            build_multi_fft_rader_container_state_with_tuning(prime, sequence_len / prime, tuning)?
        else {
            return Ok(None);
        };
        containers.push(container);
    }
    if !sequence_len.is_multiple_of(rader_product) {
        return Ok(None);
    }
    let smooth_outer_factor = sequence_len / rader_product;
    let (mut outer_registers_per_radix, mut outer_registers_per_thread, mut outer_min_registers) =
        match rader_outer_register_state(sequence_len, smooth_outer_factor, batch_count) {
            Ok(state) => state,
            Err(VkFftError::UnsupportedKernelPath(_)) => return Ok(None),
            Err(error) => return Err(error),
        };
    let mut outer_stage_multipliers = match smooth_outer_primitive_multipliers(smooth_outer_factor)
    {
        Ok(multipliers) => multipliers,
        Err(VkFftError::UnsupportedKernelPath(_)) => return Ok(None),
        Err(error) => return Err(error),
    };
    optimize_smooth_fft_rader_containers_once(
        sequence_len,
        &mut outer_registers_per_radix,
        &mut outer_registers_per_thread,
        &mut outer_min_registers,
        &mut containers,
    )?;
    // `VkFFTOptimizeRaderFFTRegisters` skips type-1 containers in its optimization
    // loops, but its final min/max merge includes every Rader container. A Direct-Rader
    // container has the fixed 2-register state, so mixed Direct+FFT parents must feed
    // that two-register minimum into the subsequent `scale_registers_rader` loop.
    let type1_scale_base_min_registers = outer_min_registers.min(2);
    let Some((rader_min_registers, estimate_rader_threads)) =
        scale_axis0_direct_rader_type1_register_floor_with_max_batch_coalesced(
            sequence_len,
            direct_prime_multiplicities,
            type1_scale_base_min_registers,
            max_batch_coalesced,
            physical_thread_limit,
        )?
    else {
        return Ok(None);
    };

    // Port the register harmonization immediately following scale_registers_rader.
    outer_registers_per_thread = outer_registers_per_thread.max(rader_min_registers);
    for value in outer_registers_per_radix.iter_mut() {
        if *value > 0 && outer_registers_per_thread / *value >= 2 {
            *value *= outer_registers_per_thread / *value;
        }
    }
    for container in &mut containers {
        for value in container.registers_per_thread_per_radix.iter_mut() {
            if *value > 0 && outer_registers_per_thread / *value >= 2 {
                *value *= outer_registers_per_thread / *value;
            }
        }
    }
    let mut new_min_registers = usize::MAX;
    let mut new_max_registers = outer_registers_per_thread;
    for value in outer_registers_per_radix
        .iter()
        .copied()
        .filter(|value| *value > 0)
    {
        new_min_registers = new_min_registers.min(value);
        new_max_registers = new_max_registers.max(value);
    }
    for container in &containers {
        for value in container
            .registers_per_thread_per_radix
            .iter()
            .copied()
            .filter(|value| *value > 0)
        {
            new_min_registers = new_min_registers.min(value);
            new_max_registers = new_max_registers.max(value);
        }
    }
    if new_min_registers == usize::MAX || new_min_registers == 0 {
        return Ok(None);
    }
    outer_min_registers = new_min_registers;
    outer_registers_per_thread = new_max_registers;
    let coalesced_elements =
        sequence_len
            .checked_mul(max_batch_coalesced)
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "mixed multi-Rader coalesced element pressure",
            })?;
    if ceil_div(coalesced_elements, outer_min_registers)? > physical_thread_limit
        || estimate_rader_threads > physical_thread_limit
    {
        scale_axis0_rader_register_tables_with_max_batch_coalesced(
            sequence_len,
            max_batch_coalesced,
            shrink_first_upload_coalescing,
            physical_thread_limit,
            &mut outer_registers_per_radix,
            &mut outer_registers_per_thread,
            &mut outer_min_registers,
            &mut containers,
        )?;
    }
    optimize_smooth_fft_rader_containers_once(
        sequence_len,
        &mut outer_registers_per_radix,
        &mut outer_registers_per_thread,
        &mut outer_min_registers,
        &mut containers,
    )?;

    // Upstream stores `rader_min_registers` separately from the harmonized parent
    // table, then seeds the final `VkFFTMinMaxRegisterCheck` with that Direct-Rader
    // floor. Ordinary outer stages and type-0 containers may only lower it afterward.
    let mut final_min_registers = rader_min_registers;
    if smooth_outer_factor > 1 {
        let (_, _) = optimize_radix_kernels(
            &mut outer_registers_per_radix,
            &mut outer_stage_multipliers,
            1,
        );
        for radix in 2..VKFFT_RADIX_TABLE_LEN {
            if outer_stage_multipliers[radix] == 0 {
                continue;
            }
            let registers = outer_registers_per_radix[radix];
            if registers == 0 {
                return Err(VkFftError::InvalidKernelIr(
                    "mixed multi-Rader outer stage has no register allocation",
                ));
            }
            final_min_registers = final_min_registers.min(registers);
        }
    }
    let mut min_rader_fft_thread_num = 0usize;
    for container in &mut containers {
        optimize_and_collect_recursive_fft_rader_pressure(
            container,
            &mut final_min_registers,
            &mut min_rader_fft_thread_num,
        )?;
    }

    let base_axis_threads = ceil_div(sequence_len, final_min_registers)?;
    let mut final_direct_threads = 0usize;
    for &(prime, _) in direct_prime_multiplicities {
        let direct_threads = prime.div_ceil(2);
        let Some(local_estimate) =
            axis0_direct_rader_local_thread_estimate_with_max_batch_coalesced(
                sequence_len,
                prime,
                rader_min_registers,
                max_batch_coalesced,
                physical_thread_limit,
            )?
        else {
            return Ok(None);
        };
        let rounded_axis_threads = ceil_div(base_axis_threads, direct_threads)?
            .checked_mul(direct_threads)
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "mixed multi-Rader AxisBlockSplitter direct rounding",
            })?;
        final_direct_threads = final_direct_threads
            .max(local_estimate)
            .max(rounded_axis_threads);
    }
    // `VkFFTSplitAxisBlock` clamps the final axis block after Direct-Rader rounding
    // and FFT-Rader child pressure are combined. Scheduler-side pressure has already
    // had its chance to promote uploads; a small final granularity overflow (for
    // example N3553: 258 rounded lanes on a 256-thread profile) clamps here.
    let threads_per_transform = final_direct_threads
        .max(min_rader_fft_thread_num)
        .min(physical_thread_limit);
    Ok(Some(threads_per_transform.max(1)))
}

/// Quad/DD counterpart of the mixed Direct + multiple FFT-Rader one-upload parent
/// scheduler. The Rader container and type-1 arithmetic is shared with ordinary
/// precision; only the smooth outer radix state comes from
/// `VkFFTGetRegistersPerThreadQuad` before the first joint type-0 optimizer.
pub fn plan_gpu_double_double_axis0_mixed_direct_multi_fft_rader_threads(
    sequence_len: usize,
    direct_prime_multiplicities: &[(usize, usize)],
    fft_prime_multiplicities: &[(usize, usize)],
    batch_count: usize,
    device: DeviceProfile,
) -> Result<Option<usize>> {
    plan_gpu_double_double_axis0_mixed_direct_multi_fft_rader_threads_with_tuning(
        sequence_len,
        direct_prime_multiplicities,
        fft_prime_multiplicities,
        batch_count,
        PlannerTuning::portable(),
        device,
    )
}

pub(crate) fn plan_gpu_double_double_axis0_mixed_direct_multi_fft_rader_threads_with_tuning(
    sequence_len: usize,
    direct_prime_multiplicities: &[(usize, usize)],
    fft_prime_multiplicities: &[(usize, usize)],
    batch_count: usize,
    tuning: PlannerTuning,
    device: DeviceProfile,
) -> Result<Option<usize>> {
    plan_gpu_double_double_axis0_mixed_direct_multi_fft_rader_threads_with_pass_context(
        sequence_len,
        direct_prime_multiplicities,
        fft_prime_multiplicities,
        batch_count,
        1,
        false,
        tuning,
        device,
    )
}

pub(crate) fn plan_gpu_double_double_axis0_mixed_direct_multi_fft_rader_threads_for_forced_upload_with_tuning(
    sequence_len: usize,
    direct_prime_multiplicities: &[(usize, usize)],
    fft_prime_multiplicities: &[(usize, usize)],
    batch_count: usize,
    max_batch_coalesced: usize,
    axis_upload_id: usize,
    tuning: PlannerTuning,
    device: DeviceProfile,
) -> Result<Option<usize>> {
    plan_gpu_double_double_axis0_mixed_direct_multi_fft_rader_threads_with_pass_context(
        sequence_len,
        direct_prime_multiplicities,
        fft_prime_multiplicities,
        batch_count,
        max_batch_coalesced,
        axis_upload_id == 0,
        tuning,
        device,
    )
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Axis0RaderSplitState {
    pub(crate) base_axis_threads: usize,
    pub(crate) rader_min_registers: usize,
    pub(crate) min_rader_fft_thread_num: usize,
    pub(crate) direct_prime_multiplicities: Vec<(usize, usize)>,
}

fn axis0_direct_rader_threads_from_split_state(
    sequence_len: usize,
    state: &Axis0RaderSplitState,
    max_batch_coalesced: usize,
    physical_thread_limit: usize,
) -> Result<Option<usize>> {
    if sequence_len == 0
        || state.base_axis_threads == 0
        || state.rader_min_registers == 0
        || state.direct_prime_multiplicities.is_empty()
        || max_batch_coalesced == 0
        || physical_thread_limit == 0
    {
        return Ok(None);
    }
    let mut final_direct_threads = 0usize;
    for &(prime, multiplicity) in &state.direct_prime_multiplicities {
        if multiplicity == 0 {
            return Ok(None);
        }
        let direct_threads = prime.div_ceil(2);
        let Some(local_estimate) =
            axis0_direct_rader_local_thread_estimate_with_max_batch_coalesced(
                sequence_len,
                prime,
                state.rader_min_registers,
                max_batch_coalesced,
                physical_thread_limit,
            )?
        else {
            return Ok(None);
        };
        let rounded_axis_threads = ceil_div(state.base_axis_threads, direct_threads)?
            .checked_mul(direct_threads)
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "axis0 Rader split-state Direct rounding",
            })?;
        final_direct_threads = final_direct_threads
            .max(local_estimate)
            .max(rounded_axis_threads);
    }
    Ok(Some(final_direct_threads.min(physical_thread_limit).max(1)))
}

fn axis0_rader_threads_from_split_state(
    sequence_len: usize,
    state: &Axis0RaderSplitState,
    max_batch_coalesced: usize,
    physical_thread_limit: usize,
) -> Result<Option<usize>> {
    if sequence_len == 0
        || state.base_axis_threads == 0
        || state.rader_min_registers == 0
        || max_batch_coalesced == 0
        || physical_thread_limit == 0
    {
        return Ok(None);
    }
    let final_direct_threads = axis0_direct_rader_threads_from_split_state(
        sequence_len,
        state,
        max_batch_coalesced,
        physical_thread_limit,
    )?
    .unwrap_or(0);
    Ok(Some(
        final_direct_threads
            .max(state.min_rader_fft_thread_num)
            .min(physical_thread_limit)
            .max(1),
    ))
}

pub(crate) fn plan_gpu_double_double_axis0_mixed_direct_multi_fft_rader_split_state_for_forced_upload_with_tuning(
    sequence_len: usize,
    direct_prime_multiplicities: &[(usize, usize)],
    fft_prime_multiplicities: &[(usize, usize)],
    batch_count: usize,
    max_batch_coalesced: usize,
    axis_upload_id: usize,
    tuning: PlannerTuning,
    device: DeviceProfile,
) -> Result<Option<Axis0RaderSplitState>> {
    plan_gpu_double_double_axis0_mixed_direct_multi_fft_rader_split_state_with_pass_context(
        sequence_len,
        direct_prime_multiplicities,
        fft_prime_multiplicities,
        batch_count,
        max_batch_coalesced,
        axis_upload_id == 0,
        tuning,
        device,
    )
}

fn plan_gpu_double_double_axis0_mixed_direct_multi_fft_rader_threads_with_pass_context(
    sequence_len: usize,
    direct_prime_multiplicities: &[(usize, usize)],
    fft_prime_multiplicities: &[(usize, usize)],
    batch_count: usize,
    max_batch_coalesced: usize,
    shrink_first_upload_coalescing: bool,
    tuning: PlannerTuning,
    device: DeviceProfile,
) -> Result<Option<usize>> {
    let physical_thread_limit = device
        .max_threads_per_block
        .min(device.max_workgroup_size[0]);
    let Some(state) =
        plan_gpu_double_double_axis0_mixed_direct_multi_fft_rader_split_state_with_pass_context(
            sequence_len,
            direct_prime_multiplicities,
            fft_prime_multiplicities,
            batch_count,
            max_batch_coalesced,
            shrink_first_upload_coalescing,
            tuning,
            device,
        )?
    else {
        return Ok(None);
    };
    axis0_rader_threads_from_split_state(
        sequence_len,
        &state,
        max_batch_coalesced,
        physical_thread_limit,
    )
}

fn plan_gpu_double_double_axis0_mixed_direct_multi_fft_rader_split_state_with_pass_context(
    sequence_len: usize,
    direct_prime_multiplicities: &[(usize, usize)],
    fft_prime_multiplicities: &[(usize, usize)],
    batch_count: usize,
    max_batch_coalesced: usize,
    shrink_first_upload_coalescing: bool,
    tuning: PlannerTuning,
    device: DeviceProfile,
) -> Result<Option<Axis0RaderSplitState>> {
    if sequence_len < 2
        || direct_prime_multiplicities.is_empty()
        || fft_prime_multiplicities.is_empty()
        || batch_count == 0
        || max_batch_coalesced == 0
        || device.max_threads_per_block == 0
        || device.max_workgroup_size[0] == 0
    {
        return Ok(None);
    }
    let physical_thread_limit = device
        .max_threads_per_block
        .min(device.max_workgroup_size[0]);
    let mut rader_product = 1usize;
    for (index, &(prime, multiplicity)) in direct_prime_multiplicities.iter().enumerate() {
        if prime < 3
            || multiplicity == 0
            || !sequence_len.is_multiple_of(prime)
            || prime.div_ceil(2) > physical_thread_limit
            || direct_prime_multiplicities[..index]
                .iter()
                .any(|&(previous, _)| previous == prime)
        {
            return Ok(None);
        }
        for _ in 0..multiplicity {
            rader_product =
                rader_product
                    .checked_mul(prime)
                    .ok_or(VkFftError::ArithmeticOverflow {
                        operation: "double-double mixed multi-Rader direct prime-power product",
                    })?;
        }
    }

    let mut containers = Vec::with_capacity(fft_prime_multiplicities.len());
    for (index, &(prime, multiplicity)) in fft_prime_multiplicities.iter().enumerate() {
        if prime < 3
            || multiplicity == 0
            || !sequence_len.is_multiple_of(prime)
            || direct_prime_multiplicities
                .iter()
                .any(|&(direct, _)| direct == prime)
            || fft_prime_multiplicities[..index]
                .iter()
                .any(|&(previous, _)| previous == prime)
        {
            return Ok(None);
        }
        for _ in 0..multiplicity {
            rader_product =
                rader_product
                    .checked_mul(prime)
                    .ok_or(VkFftError::ArithmeticOverflow {
                        operation: "double-double mixed multi-Rader FFT prime-power product",
                    })?;
        }
        let Some(container) =
            build_multi_fft_rader_container_state_with_tuning(prime, sequence_len / prime, tuning)?
        else {
            return Ok(None);
        };
        containers.push(container);
    }
    if !sequence_len.is_multiple_of(rader_product) {
        return Ok(None);
    }
    let smooth_outer_factor = sequence_len / rader_product;
    let (mut outer_registers_per_radix, mut outer_registers_per_thread, mut outer_min_registers) =
        if smooth_outer_factor == 1 {
            ([0usize; VKFFT_RADIX_TABLE_LEN], 2usize, 2usize)
        } else {
            let total_elements =
                sequence_len
                    .checked_mul(batch_count)
                    .ok_or(VkFftError::ArithmeticOverflow {
                        operation: "double-double mixed multi-Rader total element count",
                    })?;
            let rhs_transform_count = total_elements / smooth_outer_factor;
            let quad = match plan_gpu_double_double_quad_registers(
                smooth_outer_factor,
                rhs_transform_count,
            ) {
                Ok(schedule) => schedule,
                Err(VkFftError::UnsupportedKernelPath(_)) => return Ok(None),
                Err(error) => return Err(error),
            };
            (
                quad.registers_per_thread_per_radix,
                quad.registers_per_thread,
                quad.min_registers_per_thread,
            )
        };
    let mut outer_stage_multipliers = match smooth_outer_primitive_multipliers(smooth_outer_factor)
    {
        Ok(multipliers) => multipliers,
        Err(VkFftError::UnsupportedKernelPath(_)) => return Ok(None),
        Err(error) => return Err(error),
    };

    optimize_smooth_fft_rader_containers_once(
        sequence_len,
        &mut outer_registers_per_radix,
        &mut outer_registers_per_thread,
        &mut outer_min_registers,
        &mut containers,
    )?;
    // As in the ordinary path and fixed upstream, the first FFT-Rader optimizer merges
    // every container into the parent min/max even though only type-0 containers take
    // part in the optimize-shared loop. A type-1 Direct-Rader contributes its fixed
    // two-register state before `scale_registers_rader` in Quad/DD precision too.
    let type1_scale_base_min_registers = outer_min_registers.min(2);
    let Some((rader_min_registers, estimate_rader_threads)) =
        scale_axis0_direct_rader_type1_register_floor_with_max_batch_coalesced(
            sequence_len,
            direct_prime_multiplicities,
            type1_scale_base_min_registers,
            max_batch_coalesced,
            physical_thread_limit,
        )?
    else {
        return Ok(None);
    };

    outer_registers_per_thread = outer_registers_per_thread.max(rader_min_registers);
    for value in outer_registers_per_radix.iter_mut() {
        if *value > 0 && outer_registers_per_thread / *value >= 2 {
            *value *= outer_registers_per_thread / *value;
        }
    }
    for container in &mut containers {
        for value in container.registers_per_thread_per_radix.iter_mut() {
            if *value > 0 && outer_registers_per_thread / *value >= 2 {
                *value *= outer_registers_per_thread / *value;
            }
        }
    }
    let mut new_min_registers = usize::MAX;
    let mut new_max_registers = outer_registers_per_thread;
    for value in outer_registers_per_radix
        .iter()
        .copied()
        .filter(|value| *value > 0)
    {
        new_min_registers = new_min_registers.min(value);
        new_max_registers = new_max_registers.max(value);
    }
    for container in &containers {
        for value in container
            .registers_per_thread_per_radix
            .iter()
            .copied()
            .filter(|value| *value > 0)
        {
            new_min_registers = new_min_registers.min(value);
            new_max_registers = new_max_registers.max(value);
        }
    }
    if new_min_registers == usize::MAX || new_min_registers == 0 {
        return Ok(None);
    }
    outer_min_registers = new_min_registers;
    outer_registers_per_thread = new_max_registers;
    let coalesced_elements =
        sequence_len
            .checked_mul(max_batch_coalesced)
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "double-double mixed multi-Rader coalesced element pressure",
            })?;
    if ceil_div(coalesced_elements, outer_min_registers)? > physical_thread_limit
        || estimate_rader_threads > physical_thread_limit
    {
        scale_axis0_rader_register_tables_with_max_batch_coalesced(
            sequence_len,
            max_batch_coalesced,
            shrink_first_upload_coalescing,
            physical_thread_limit,
            &mut outer_registers_per_radix,
            &mut outer_registers_per_thread,
            &mut outer_min_registers,
            &mut containers,
        )?;
    }
    optimize_smooth_fft_rader_containers_once(
        sequence_len,
        &mut outer_registers_per_radix,
        &mut outer_registers_per_thread,
        &mut outer_min_registers,
        &mut containers,
    )?;

    let mut final_min_registers = rader_min_registers;
    if smooth_outer_factor > 1 {
        let (_, _) = optimize_radix_kernels(
            &mut outer_registers_per_radix,
            &mut outer_stage_multipliers,
            1,
        );
        for radix in 2..VKFFT_RADIX_TABLE_LEN {
            if outer_stage_multipliers[radix] == 0 {
                continue;
            }
            let registers = outer_registers_per_radix[radix];
            if registers == 0 {
                return Err(VkFftError::InvalidKernelIr(
                    "double-double mixed multi-Rader outer stage has no register allocation",
                ));
            }
            final_min_registers = final_min_registers.min(registers);
        }
    }

    let mut min_rader_fft_thread_num = 0usize;
    for container in &mut containers {
        optimize_and_collect_recursive_fft_rader_pressure(
            container,
            &mut final_min_registers,
            &mut min_rader_fft_thread_num,
        )?;
    }

    Ok(Some(Axis0RaderSplitState {
        base_axis_threads: ceil_div(sequence_len, final_min_registers)?,
        rader_min_registers,
        min_rader_fft_thread_num,
        direct_prime_multiplicities: direct_prime_multiplicities.to_vec(),
    }))
}

/// Quad/DD counterpart of the pure type-0 FFT-Rader parent scheduler. Rader container
/// optimize-shared arithmetic is precision-independent, but the ordinary smooth outer
/// factor must come from VkFFTGetRegistersPerThreadQuad rather than the F32/F64 table.
pub fn plan_gpu_double_double_axis0_fft_rader_threads_for_prime_multiplicities(
    sequence_len: usize,
    fft_prime_multiplicities: &[(usize, usize)],
    batch_count: usize,
    device: DeviceProfile,
) -> Result<Option<usize>> {
    plan_gpu_double_double_axis0_fft_rader_threads_for_prime_multiplicities_with_tuning(
        sequence_len,
        fft_prime_multiplicities,
        batch_count,
        PlannerTuning::portable(),
        device,
    )
}

pub(crate) fn plan_gpu_double_double_axis0_fft_rader_threads_for_prime_multiplicities_with_tuning(
    sequence_len: usize,
    fft_prime_multiplicities: &[(usize, usize)],
    batch_count: usize,
    tuning: PlannerTuning,
    device: DeviceProfile,
) -> Result<Option<usize>> {
    plan_gpu_double_double_axis0_fft_rader_threads_for_prime_multiplicities_with_pass_context(
        sequence_len,
        fft_prime_multiplicities,
        batch_count,
        1,
        false,
        tuning,
        device,
    )
}

pub(crate) fn plan_gpu_double_double_axis0_fft_rader_threads_for_forced_upload_with_tuning(
    sequence_len: usize,
    fft_prime_multiplicities: &[(usize, usize)],
    batch_count: usize,
    max_batch_coalesced: usize,
    axis_upload_id: usize,
    tuning: PlannerTuning,
    device: DeviceProfile,
) -> Result<Option<usize>> {
    plan_gpu_double_double_axis0_fft_rader_threads_for_prime_multiplicities_with_pass_context(
        sequence_len,
        fft_prime_multiplicities,
        batch_count,
        max_batch_coalesced,
        axis_upload_id == 0,
        tuning,
        device,
    )
}

fn plan_gpu_double_double_axis0_fft_rader_threads_for_prime_multiplicities_with_pass_context(
    sequence_len: usize,
    fft_prime_multiplicities: &[(usize, usize)],
    batch_count: usize,
    max_batch_coalesced: usize,
    shrink_first_upload_coalescing: bool,
    tuning: PlannerTuning,
    device: DeviceProfile,
) -> Result<Option<usize>> {
    if sequence_len < 2
        || fft_prime_multiplicities.is_empty()
        || batch_count == 0
        || max_batch_coalesced == 0
        || device.max_threads_per_block == 0
        || device.max_workgroup_size[0] == 0
    {
        return Ok(None);
    }
    let physical_thread_limit = device
        .max_threads_per_block
        .min(device.max_workgroup_size[0]);
    let mut rader_product = 1usize;
    let mut containers = Vec::with_capacity(fft_prime_multiplicities.len());
    for (index, &(prime, multiplicity)) in fft_prime_multiplicities.iter().enumerate() {
        if prime < 3
            || multiplicity == 0
            || !sequence_len.is_multiple_of(prime)
            || fft_prime_multiplicities[..index]
                .iter()
                .any(|&(previous, _)| previous == prime)
        {
            return Ok(None);
        }
        for _ in 0..multiplicity {
            rader_product =
                rader_product
                    .checked_mul(prime)
                    .ok_or(VkFftError::ArithmeticOverflow {
                        operation: "double-double FFT-Rader prime-power product",
                    })?;
        }
        let Some(container) =
            build_multi_fft_rader_container_state_with_tuning(prime, sequence_len / prime, tuning)?
        else {
            return Ok(None);
        };
        containers.push(container);
    }
    if !sequence_len.is_multiple_of(rader_product) {
        return Ok(None);
    }
    let smooth_outer_factor = sequence_len / rader_product;
    let (mut outer_registers_per_radix, mut outer_registers_per_thread, mut outer_min_registers) =
        if smooth_outer_factor == 1 {
            ([0usize; VKFFT_RADIX_TABLE_LEN], 2usize, 2usize)
        } else {
            let total_elements =
                sequence_len
                    .checked_mul(batch_count)
                    .ok_or(VkFftError::ArithmeticOverflow {
                        operation: "double-double FFT-Rader total element count",
                    })?;
            let rhs_transform_count = total_elements / smooth_outer_factor;
            let quad = match plan_gpu_double_double_quad_registers(
                smooth_outer_factor,
                rhs_transform_count,
            ) {
                Ok(schedule) => schedule,
                Err(VkFftError::UnsupportedKernelPath(_)) => return Ok(None),
                Err(error) => return Err(error),
            };
            (
                quad.registers_per_thread_per_radix,
                quad.registers_per_thread,
                quad.min_registers_per_thread,
            )
        };
    let mut outer_stage_multipliers = match smooth_outer_primitive_multipliers(smooth_outer_factor)
    {
        Ok(multipliers) => multipliers,
        Err(VkFftError::UnsupportedKernelPath(_)) => return Ok(None),
        Err(error) => return Err(error),
    };

    optimize_smooth_fft_rader_containers_once(
        sequence_len,
        &mut outer_registers_per_radix,
        &mut outer_registers_per_thread,
        &mut outer_min_registers,
        &mut containers,
    )?;
    let coalesced_elements =
        sequence_len
            .checked_mul(max_batch_coalesced)
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "double-double FFT-Rader coalesced element pressure",
            })?;
    if ceil_div(coalesced_elements, outer_min_registers)? > physical_thread_limit {
        scale_axis0_rader_register_tables_with_max_batch_coalesced(
            sequence_len,
            max_batch_coalesced,
            shrink_first_upload_coalescing,
            physical_thread_limit,
            &mut outer_registers_per_radix,
            &mut outer_registers_per_thread,
            &mut outer_min_registers,
            &mut containers,
        )?;
    }
    optimize_smooth_fft_rader_containers_once(
        sequence_len,
        &mut outer_registers_per_radix,
        &mut outer_registers_per_thread,
        &mut outer_min_registers,
        &mut containers,
    )?;

    let mut final_min_registers = usize::MAX;
    if smooth_outer_factor > 1 {
        let (_, _) = optimize_radix_kernels(
            &mut outer_registers_per_radix,
            &mut outer_stage_multipliers,
            1,
        );
        for radix in 2..VKFFT_RADIX_TABLE_LEN {
            if outer_stage_multipliers[radix] == 0 {
                continue;
            }
            let registers = outer_registers_per_radix[radix];
            if registers == 0 {
                return Err(VkFftError::InvalidKernelIr(
                    "double-double FFT-Rader outer stage has no register allocation",
                ));
            }
            final_min_registers = final_min_registers.min(registers);
        }
    }
    let mut min_rader_fft_thread_num = 0usize;
    for container in &mut containers {
        optimize_and_collect_recursive_fft_rader_pressure(
            container,
            &mut final_min_registers,
            &mut min_rader_fft_thread_num,
        )?;
    }
    if final_min_registers == usize::MAX || final_min_registers == 0 {
        return Ok(None);
    }
    let threads_per_transform =
        ceil_div(sequence_len, final_min_registers)?.max(min_rader_fft_thread_num);
    if threads_per_transform > physical_thread_limit {
        return Ok(None);
    }
    Ok(Some(threads_per_transform.max(1)))
}

/// Fixed-upstream single-upload mixed Direct+FFT-Rader parent thread estimate for
/// one type-0 FFT-Rader container plus one or more type-1 direct-Rader containers.
/// The ordinary register state removes *all* Rader prime powers; the type-0 first
/// optimizer pass then couples its real `fftDim / prime` container count back into
/// that smooth state before the type-1 shared scale loop runs.
pub fn plan_gpu_axis0_mixed_direct_fft_rader_threads(
    sequence_len: usize,
    direct_prime_multiplicities: &[(usize, usize)],
    fft_prime: usize,
    batch_count: usize,
    device: DeviceProfile,
) -> Result<Option<usize>> {
    plan_gpu_axis0_mixed_direct_multi_fft_rader_threads(
        sequence_len,
        direct_prime_multiplicities,
        &[(fft_prime, 1)],
        batch_count,
        device,
    )
}

/// Quad/DD mixed Direct+FFT-Rader parent thread estimate. Unlike ordinary precision,
/// a non-trivial smooth outer factor must feed VkFFT's Quad 2/3/5/7 register table
/// into the type-0 first optimizer before the shared type-1 scale loop runs.
pub fn plan_gpu_double_double_axis0_mixed_direct_fft_rader_threads(
    sequence_len: usize,
    direct_prime_multiplicities: &[(usize, usize)],
    fft_prime: usize,
    batch_count: usize,
    device: DeviceProfile,
) -> Result<Option<usize>> {
    plan_gpu_double_double_axis0_mixed_direct_multi_fft_rader_threads(
        sequence_len,
        direct_prime_multiplicities,
        &[(fft_prime, 1)],
        batch_count,
        device,
    )
}

/// Multi-prime form of the fixed-upstream type-1/direct-Rader thread estimate.
/// Every listed prime is a distinct multiplicity-one type-1 Rader container in the
/// same single upload. Upstream rescales all of them through one shared
/// `scale_registers_rader` loop and keeps the maximum thread estimate.
pub fn plan_gpu_axis0_composite_direct_rader_threads_for_primes(
    sequence_len: usize,
    direct_primes: &[usize],
    batch_count: usize,
    device: DeviceProfile,
) -> Result<Option<usize>> {
    if sequence_len < 2
        || direct_primes.is_empty()
        || batch_count == 0
        || device.max_threads_per_block == 0
        || device.max_workgroup_size[0] == 0
    {
        return Ok(None);
    }
    let physical_thread_limit = device
        .max_threads_per_block
        .min(device.max_workgroup_size[0]);
    let mut rader_product = 1usize;
    for (index, &prime) in direct_primes.iter().enumerate() {
        if prime < 3
            || !sequence_len.is_multiple_of(prime)
            || direct_primes[..index].contains(&prime)
            || prime.div_ceil(2) > physical_thread_limit
        {
            return Ok(None);
        }
        rader_product = rader_product
            .checked_mul(prime)
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "composite direct-Rader prime product",
            })?;
    }
    if !sequence_len.is_multiple_of(rader_product) {
        return Ok(None);
    }
    let smooth_outer_product = sequence_len / rader_product;
    let (_, _, base_min_registers) =
        match rader_outer_register_state(sequence_len, smooth_outer_product, batch_count) {
            Ok(state) => state,
            Err(VkFftError::UnsupportedKernelPath(_)) => return Ok(None),
            Err(error) => return Err(error),
        };
    if base_min_registers == 0 {
        return Ok(None);
    }

    let mut scale_registers_rader = 0usize;
    loop {
        let rader_min_registers = (base_min_registers / 2)
            .checked_add(scale_registers_rader)
            .and_then(|value| value.checked_mul(2))
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "multi direct-Rader minimum register scaling",
            })?;
        if rader_min_registers == 0 {
            return Ok(None);
        }
        let mut estimate_rader_threads = sequence_len / rader_min_registers;
        for &prime in direct_primes {
            let direct_threads = prime.div_ceil(2);
            let container_fft_num = sequence_len / prime;
            let denominator = rader_min_registers.checked_mul(direct_threads).ok_or(
                VkFftError::ArithmeticOverflow {
                    operation: "multi direct-Rader container denominator",
                },
            )?;
            let temp_rader = ceil_div(sequence_len, denominator)?.max(1);
            let mut active_rader = ceil_div(container_fft_num, temp_rader)?.max(1);
            if active_rader > 1 {
                let covered =
                    active_rader
                        .checked_mul(temp_rader)
                        .ok_or(VkFftError::ArithmeticOverflow {
                            operation: "multi direct-Rader active container coverage",
                        })?;
                let overshoot = covered - container_fft_num;
                let round_down = overshoot
                    .checked_mul(2)
                    .is_some_and(|twice| twice >= temp_rader);
                let fewer_threads = ceil_div(container_fft_num, active_rader - 1)?
                    .checked_mul(direct_threads)
                    .ok_or(VkFftError::ArithmeticOverflow {
                        operation: "multi direct-Rader rounded thread estimate",
                    })?;
                if round_down && fewer_threads <= physical_thread_limit {
                    active_rader -= 1;
                }
            }
            let local_estimate = ceil_div(container_fft_num, active_rader)?
                .checked_mul(direct_threads)
                .ok_or(VkFftError::ArithmeticOverflow {
                    operation: "multi direct-Rader thread estimate",
                })?;
            estimate_rader_threads = estimate_rader_threads.max(local_estimate);
        }
        let low_register_pressure = ((sequence_len / base_min_registers) > 256
            || estimate_rader_threads > 256)
            && rader_min_registers <= 4;
        if estimate_rader_threads <= physical_thread_limit && !low_register_pressure {
            return Ok(Some(estimate_rader_threads.max(1)));
        }
        scale_registers_rader =
            scale_registers_rader
                .checked_add(1)
                .ok_or(VkFftError::ArithmeticOverflow {
                    operation: "multi direct-Rader register scale iteration",
                })?;
        if scale_registers_rader > sequence_len {
            return Ok(None);
        }
    }
}

/// Quad/DD multiplicity-aware counterpart. Repeated direct primes are removed from
/// the smooth outer factor with their full multiplier, but one type-1 container per
/// distinct prime participates in the shared scale loop.
pub fn plan_gpu_double_double_axis0_composite_direct_rader_threads_for_prime_multiplicities(
    sequence_len: usize,
    direct_prime_multiplicities: &[(usize, usize)],
    batch_count: usize,
    device: DeviceProfile,
) -> Result<Option<usize>> {
    plan_gpu_double_double_axis0_composite_direct_rader_threads_for_prime_multiplicities_with_max_batch_coalesced(
        sequence_len,
        direct_prime_multiplicities,
        batch_count,
        1,
        device,
    )
}

pub(crate) fn plan_gpu_double_double_axis0_composite_direct_rader_split_state_for_prime_multiplicities_with_max_batch_coalesced(
    sequence_len: usize,
    direct_prime_multiplicities: &[(usize, usize)],
    batch_count: usize,
    max_batch_coalesced: usize,
    device: DeviceProfile,
) -> Result<Option<Axis0RaderSplitState>> {
    if sequence_len < 2
        || direct_prime_multiplicities.is_empty()
        || batch_count == 0
        || max_batch_coalesced == 0
        || device.max_threads_per_block == 0
        || device.max_workgroup_size[0] == 0
    {
        return Ok(None);
    }
    let physical_thread_limit = device
        .max_threads_per_block
        .min(device.max_workgroup_size[0]);
    let mut rader_product = 1usize;
    for (index, &(prime, multiplicity)) in direct_prime_multiplicities.iter().enumerate() {
        if prime < 3
            || multiplicity == 0
            || !sequence_len.is_multiple_of(prime)
            || direct_prime_multiplicities[..index]
                .iter()
                .any(|&(previous, _)| previous == prime)
            || prime.div_ceil(2) > physical_thread_limit
        {
            return Ok(None);
        }
        for _ in 0..multiplicity {
            rader_product =
                rader_product
                    .checked_mul(prime)
                    .ok_or(VkFftError::ArithmeticOverflow {
                        operation: "double-double direct-Rader split-state prime-power product",
                    })?;
        }
    }
    if !sequence_len.is_multiple_of(rader_product) {
        return Ok(None);
    }
    let smooth_outer_product = sequence_len / rader_product;
    let base_min_registers = if smooth_outer_product == 1 {
        2usize
    } else {
        let total_elements =
            sequence_len
                .checked_mul(batch_count)
                .ok_or(VkFftError::ArithmeticOverflow {
                    operation: "double-double direct-Rader split-state total elements",
                })?;
        let rhs_transform_count = total_elements / smooth_outer_product;
        match plan_gpu_double_double_quad_registers(smooth_outer_product, rhs_transform_count) {
            Ok(schedule) => schedule.min_registers_per_thread,
            Err(VkFftError::UnsupportedKernelPath(_)) => return Ok(None),
            Err(error) => return Err(error),
        }
    };
    let type1_scale_base_min_registers = base_min_registers.min(2);
    let Some((rader_min_registers, _)) =
        scale_axis0_direct_rader_type1_register_floor_with_max_batch_coalesced(
            sequence_len,
            direct_prime_multiplicities,
            type1_scale_base_min_registers,
            max_batch_coalesced,
            physical_thread_limit,
        )?
    else {
        return Ok(None);
    };
    Ok(Some(Axis0RaderSplitState {
        base_axis_threads: ceil_div(sequence_len, rader_min_registers)?,
        rader_min_registers,
        min_rader_fft_thread_num: 0,
        direct_prime_multiplicities: direct_prime_multiplicities.to_vec(),
    }))
}

pub(crate) fn plan_gpu_double_double_axis0_composite_direct_rader_threads_for_prime_multiplicities_with_max_batch_coalesced(
    sequence_len: usize,
    direct_prime_multiplicities: &[(usize, usize)],
    batch_count: usize,
    max_batch_coalesced: usize,
    device: DeviceProfile,
) -> Result<Option<usize>> {
    if sequence_len < 2
        || direct_prime_multiplicities.is_empty()
        || batch_count == 0
        || max_batch_coalesced == 0
        || device.max_threads_per_block == 0
        || device.max_workgroup_size[0] == 0
    {
        return Ok(None);
    }
    let physical_thread_limit = device
        .max_threads_per_block
        .min(device.max_workgroup_size[0]);
    let mut rader_product = 1usize;
    for (index, &(prime, multiplicity)) in direct_prime_multiplicities.iter().enumerate() {
        if prime < 3
            || multiplicity == 0
            || !sequence_len.is_multiple_of(prime)
            || direct_prime_multiplicities[..index]
                .iter()
                .any(|&(previous, _)| previous == prime)
            || prime.div_ceil(2) > physical_thread_limit
        {
            return Ok(None);
        }
        for _ in 0..multiplicity {
            rader_product =
                rader_product
                    .checked_mul(prime)
                    .ok_or(VkFftError::ArithmeticOverflow {
                        operation: "double-double composite direct-Rader prime-power product",
                    })?;
        }
    }
    if !sequence_len.is_multiple_of(rader_product) {
        return Ok(None);
    }
    let smooth_outer_product = sequence_len / rader_product;
    let base_min_registers = if smooth_outer_product == 1 {
        2usize
    } else {
        let total_elements =
            sequence_len
                .checked_mul(batch_count)
                .ok_or(VkFftError::ArithmeticOverflow {
                    operation: "double-double multiplicity-aware direct-Rader total elements",
                })?;
        let rhs_transform_count = total_elements / smooth_outer_product;
        match plan_gpu_double_double_quad_registers(smooth_outer_product, rhs_transform_count) {
            Ok(schedule) => schedule.min_registers_per_thread,
            Err(VkFftError::UnsupportedKernelPath(_)) => return Ok(None),
            Err(error) => return Err(error),
        }
    };
    if base_min_registers == 0 {
        return Ok(None);
    }

    let type1_scale_base_min_registers = base_min_registers.min(2);
    let Some((rader_min_registers, _)) =
        scale_axis0_direct_rader_type1_register_floor_with_max_batch_coalesced(
            sequence_len,
            direct_prime_multiplicities,
            type1_scale_base_min_registers,
            max_batch_coalesced,
            physical_thread_limit,
        )?
    else {
        return Ok(None);
    };
    let mut estimate_rader_threads = sequence_len / rader_min_registers;
    for &(prime, _) in direct_prime_multiplicities {
        let Some(local_estimate) =
            axis0_direct_rader_local_thread_estimate_with_max_batch_coalesced(
                sequence_len,
                prime,
                rader_min_registers,
                max_batch_coalesced,
                physical_thread_limit,
            )?
        else {
            return Ok(None);
        };
        estimate_rader_threads = estimate_rader_threads.max(local_estimate);
    }
    Ok(Some(estimate_rader_threads.max(1)))
}

/// Fixed-upstream higher-axis lane floor for the narrow DD composite family containing
/// exactly one type-1 Direct-Rader prime and one smooth outer factor. In this context
/// `VkFFTSplitAxisBlock` keeps one Direct-Rader lane group per smooth outer transform:
/// `smooth_outer * ceil(prime / 2)`. Keep repeated/multi-prime composites on the existing
/// fail-soft path until they have their own pinned upstream witness.
pub(crate) fn plan_gpu_double_double_other_axis_composite_direct_rader_threads_for_prime_multiplicities(
    sequence_len: usize,
    direct_prime_multiplicities: &[(usize, usize)],
    batch_count: usize,
    device: DeviceProfile,
) -> Result<Option<usize>> {
    if sequence_len < 2
        || batch_count == 0
        || direct_prime_multiplicities.len() != 1
        || direct_prime_multiplicities[0].1 != 1
        || device.max_threads_per_block == 0
        || device.max_workgroup_size[1] == 0
    {
        return Ok(None);
    }
    let prime = direct_prime_multiplicities[0].0;
    if prime < 3 || !sequence_len.is_multiple_of(prime) {
        return Ok(None);
    }
    let smooth_outer_product = sequence_len / prime;
    if smooth_outer_product <= 1 {
        return Ok(None);
    }
    let threads = smooth_outer_product.checked_mul(prime.div_ceil(2)).ok_or(
        VkFftError::ArithmeticOverflow {
            operation: "DD higher-axis single Direct-Rader composite thread count",
        },
    )?;
    let physical_thread_limit = device
        .max_threads_per_block
        .min(device.max_workgroup_size[1]);
    if threads == 0 || threads > physical_thread_limit {
        return Ok(None);
    }
    Ok(Some(threads))
}

/// Quad/DD form of the fixed-upstream type-1/direct-Rader composite thread estimate.
/// The type-1 active-container loop is the same as ordinary precision, but the smooth
/// non-Rader outer factor obtains its minimum register demand from VkFFT's Quad
/// 2/3/5/7 table using the exact number of RHS transforms materialized by that factor.
pub fn plan_gpu_double_double_axis0_composite_direct_rader_threads_for_primes(
    sequence_len: usize,
    direct_primes: &[usize],
    batch_count: usize,
    device: DeviceProfile,
) -> Result<Option<usize>> {
    if sequence_len < 2
        || direct_primes.is_empty()
        || batch_count == 0
        || device.max_threads_per_block == 0
        || device.max_workgroup_size[0] == 0
    {
        return Ok(None);
    }
    let physical_thread_limit = device
        .max_threads_per_block
        .min(device.max_workgroup_size[0]);
    let mut rader_product = 1usize;
    for (index, &prime) in direct_primes.iter().enumerate() {
        if prime < 3
            || !sequence_len.is_multiple_of(prime)
            || direct_primes[..index].contains(&prime)
            || prime.div_ceil(2) > physical_thread_limit
        {
            return Ok(None);
        }
        rader_product = rader_product
            .checked_mul(prime)
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "double-double composite direct-Rader prime product",
            })?;
    }
    if !sequence_len.is_multiple_of(rader_product) {
        return Ok(None);
    }
    let smooth_outer_product = sequence_len / rader_product;
    let base_min_registers = if smooth_outer_product == 1 {
        // With no ordinary radix stage, fixed upstream retains the baseline two-register
        // floor before the shared type-1 scaling loop.
        2usize
    } else {
        let total_elements =
            sequence_len
                .checked_mul(batch_count)
                .ok_or(VkFftError::ArithmeticOverflow {
                    operation: "double-double composite direct-Rader total element count",
                })?;
        let rhs_transform_count = total_elements / smooth_outer_product;
        match plan_gpu_double_double_quad_registers(smooth_outer_product, rhs_transform_count) {
            Ok(schedule) => schedule.min_registers_per_thread,
            Err(VkFftError::UnsupportedKernelPath(_)) => return Ok(None),
            Err(error) => return Err(error),
        }
    };
    if base_min_registers == 0 {
        return Ok(None);
    }

    let type1_scale_base_min_registers = base_min_registers.min(2);
    let mut scale_registers_rader = 0usize;
    loop {
        let rader_min_registers = (type1_scale_base_min_registers / 2)
            .checked_add(scale_registers_rader)
            .and_then(|value| value.checked_mul(2))
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "double-double composite direct-Rader minimum register scaling",
            })?;
        if rader_min_registers == 0 {
            return Ok(None);
        }
        let mut estimate_rader_threads = sequence_len / rader_min_registers;
        for &prime in direct_primes {
            let direct_threads = prime.div_ceil(2);
            let container_fft_num = sequence_len / prime;
            let denominator = rader_min_registers.checked_mul(direct_threads).ok_or(
                VkFftError::ArithmeticOverflow {
                    operation: "double-double composite direct-Rader container denominator",
                },
            )?;
            let temp_rader = ceil_div(sequence_len, denominator)?.max(1);
            let mut active_rader = ceil_div(container_fft_num, temp_rader)?.max(1);
            if active_rader > 1 {
                let covered =
                    active_rader
                        .checked_mul(temp_rader)
                        .ok_or(VkFftError::ArithmeticOverflow {
                            operation: "double-double composite direct-Rader active coverage",
                        })?;
                let overshoot = covered - container_fft_num;
                let round_down = overshoot
                    .checked_mul(2)
                    .is_some_and(|twice| twice >= temp_rader);
                let fewer_threads = ceil_div(container_fft_num, active_rader - 1)?
                    .checked_mul(direct_threads)
                    .ok_or(VkFftError::ArithmeticOverflow {
                        operation: "double-double composite direct-Rader rounded estimate",
                    })?;
                if round_down && fewer_threads <= physical_thread_limit {
                    active_rader -= 1;
                }
            }
            let local_estimate = ceil_div(container_fft_num, active_rader)?
                .checked_mul(direct_threads)
                .ok_or(VkFftError::ArithmeticOverflow {
                    operation: "double-double composite direct-Rader thread estimate",
                })?;
            estimate_rader_threads = estimate_rader_threads.max(local_estimate);
        }
        let low_register_pressure = ((sequence_len / base_min_registers) > 256
            || estimate_rader_threads > 256)
            && rader_min_registers <= 4;
        if estimate_rader_threads <= physical_thread_limit && !low_register_pressure {
            return Ok(Some(estimate_rader_threads.max(1)));
        }
        scale_registers_rader =
            scale_registers_rader
                .checked_add(1)
                .ok_or(VkFftError::ArithmeticOverflow {
                    operation: "double-double composite direct-Rader scale iteration",
                })?;
        if scale_registers_rader > sequence_len {
            return Ok(None);
        }
    }
}

/// Fixed-upstream `axis_id >= 1` geometry for a standalone direct-Rader prime.
/// Unlike axis 0, independent parent sequences stay on X and Rader threads stay on Y.
pub fn plan_gpu_other_axis_direct_rader_batch_block(
    prime: usize,
    batch_count: usize,
    fastest_axis_len: usize,
    complex_bytes: usize,
    grouped_batch_override: Option<usize>,
    axis1_grouped_batch_override: Option<usize>,
    device: DeviceProfile,
) -> Result<Option<StockhamAxisBlockSchedule>> {
    let precision = match complex_bytes {
        8 => Precision::F32,
        16 => Precision::F64,
        32 => Precision::DoubleDouble,
        _ => return Ok(None),
    };
    plan_gpu_other_axis_direct_rader_batch_block_for_precision(
        prime,
        batch_count,
        fastest_axis_len,
        precision,
        complex_bytes,
        grouped_batch_override,
        axis1_grouped_batch_override,
        device,
    )
}

pub(crate) fn plan_gpu_other_axis_direct_rader_batch_block_for_precision(
    prime: usize,
    batch_count: usize,
    fastest_axis_len: usize,
    precision: Precision,
    complex_bytes: usize,
    grouped_batch_override: Option<usize>,
    axis1_grouped_batch_override: Option<usize>,
    device: DeviceProfile,
) -> Result<Option<StockhamAxisBlockSchedule>> {
    if prime < 3 || batch_count == 0 {
        return Ok(None);
    }
    let (_, _, min_registers_per_thread) = rader_outer_register_state(prime, 1, batch_count)?;
    if min_registers_per_thread == 0 {
        return Ok(None);
    }
    let direct_threads = prime.div_ceil(2);
    let initial_threads = ceil_div(prime, min_registers_per_thread)?.max(1);
    let threads_per_transform = ceil_div(initial_threads, direct_threads)?
        .checked_mul(direct_threads)
        .ok_or(VkFftError::ArithmeticOverflow {
            operation: "higher-axis direct Rader thread alignment",
        })?
        .max(direct_threads);
    plan_gpu_other_axis_rader_block_from_threads_for_precision(
        prime,
        batch_count,
        fastest_axis_len,
        precision,
        complex_bytes,
        threads_per_transform,
        grouped_batch_override,
        axis1_grouped_batch_override,
        device,
    )
}

/// Fixed-upstream `axis_id >= 1` geometry for a standalone FFT-convolution Rader
/// prime. The caller keeps the prime-level Rader thread floor while its `(p-1)`
/// Stockham convolution remains a distinct typed child.
pub fn plan_gpu_other_axis_fft_rader_batch_block(
    prime: usize,
    batch_count: usize,
    fastest_axis_len: usize,
    schedule: &RaderFftRegisterSchedule,
    complex_bytes: usize,
    grouped_batch_override: Option<usize>,
    axis1_grouped_batch_override: Option<usize>,
    device: DeviceProfile,
) -> Result<Option<StockhamAxisBlockSchedule>> {
    let precision = match complex_bytes {
        8 => Precision::F32,
        16 => Precision::F64,
        _ => return Ok(None),
    };
    plan_gpu_other_axis_fft_rader_batch_block_for_precision(
        prime,
        batch_count,
        fastest_axis_len,
        schedule,
        precision,
        complex_bytes,
        grouped_batch_override,
        axis1_grouped_batch_override,
        device,
    )
}

pub(crate) fn plan_gpu_other_axis_fft_rader_batch_block_for_precision(
    prime: usize,
    batch_count: usize,
    fastest_axis_len: usize,
    schedule: &RaderFftRegisterSchedule,
    precision: Precision,
    complex_bytes: usize,
    grouped_batch_override: Option<usize>,
    axis1_grouped_batch_override: Option<usize>,
    device: DeviceProfile,
) -> Result<Option<StockhamAxisBlockSchedule>> {
    schedule.validate()?;
    if prime < 3
        || batch_count == 0
        || schedule.prime != prime
        || schedule.outer_fft_len != prime
        || schedule.container_fft_num != 1
        || schedule.execution_container_fft_num != 1
        || schedule.internal_fft.rhs_transform_count != batch_count
        || schedule.internal_fft.register_boost != 1
    {
        return Ok(None);
    }
    let policy = plan_gpu_scheduler_policy(precision, device)?;
    let max_batch_coalesced = (policy.coalesced_memory_bytes / complex_bytes).max(1);
    let Some(parent) = plan_gpu_axis0_multi_fft_rader_parent_schedule_with_max_batch_coalesced(
        prime,
        &[(prime, 1)],
        batch_count,
        max_batch_coalesced,
        false,
        PlannerTuning::portable(),
        device,
    )?
    else {
        return Ok(None);
    };
    let threads_per_transform = parent.threads_per_transform.max(1);
    plan_gpu_other_axis_rader_block_from_threads_for_precision(
        prime,
        batch_count,
        fastest_axis_len,
        precision,
        complex_bytes,
        threads_per_transform,
        grouped_batch_override,
        axis1_grouped_batch_override,
        device,
    )
}

/// Double-double `axis_id >= 1` FFT-Rader caller geometry. The prime-length caller
/// follows VkFFT's precision-independent Rader-container optimizer; only the split
/// `(p-1)` child pass uses the Quad/DD Stockham register table.
pub fn plan_gpu_double_double_other_axis_fft_rader_batch_block(
    prime: usize,
    batch_count: usize,
    fastest_axis_len: usize,
    grouped_batch_override: Option<usize>,
    axis1_grouped_batch_override: Option<usize>,
    device: DeviceProfile,
) -> Result<Option<StockhamAxisBlockSchedule>> {
    plan_gpu_double_double_other_axis_fft_rader_batch_block_with_tuning(
        prime,
        batch_count,
        fastest_axis_len,
        grouped_batch_override,
        axis1_grouped_batch_override,
        PlannerTuning::portable(),
        device,
    )
}

pub(crate) fn plan_gpu_double_double_other_axis_fft_rader_batch_block_with_tuning(
    prime: usize,
    batch_count: usize,
    fastest_axis_len: usize,
    grouped_batch_override: Option<usize>,
    axis1_grouped_batch_override: Option<usize>,
    tuning: PlannerTuning,
    device: DeviceProfile,
) -> Result<Option<StockhamAxisBlockSchedule>> {
    if prime < 3 || batch_count == 0 {
        return Ok(None);
    }
    let effective_tuning = upstream_effective_rader_tuning(tuning, device, Precision::DoubleDouble);
    let Some(threads_per_transform) =
        plan_gpu_double_double_axis0_fft_rader_threads_for_prime_multiplicities_with_tuning(
            prime,
            &[(prime, 1)],
            batch_count,
            effective_tuning,
            device,
        )?
    else {
        return Ok(None);
    };
    plan_gpu_other_axis_rader_block_from_threads(
        prime,
        batch_count,
        fastest_axis_len,
        32,
        threads_per_transform,
        grouped_batch_override,
        axis1_grouped_batch_override,
        device,
    )
}

/// Port the axis-0/upload-0 `VkFFTSplitAxisBlock` geometry for a standalone
/// direct-multiplication Rader prime. Ordinary direct kernels need no workgroup
/// barriers; the DD physical consumer may additionally use barriered shared reduction,
/// but keeps partial final groups safe by making inactive lanes participate with zero
/// values. `grouped_batch_override` follows upstream's early user-override branch, including
/// its conservative reset to `locMaxBatchCoalesced == 1` when the initial
/// shared-capacity grouping would exceed the physical thread limit.
pub fn plan_gpu_axis0_direct_rader_batch_block(
    prime: usize,
    batch_count: usize,
    complex_bytes: usize,
    perform_zero_padding: bool,
    grouped_batch_override: Option<usize>,
    device: DeviceProfile,
) -> Result<Option<StockhamAxisBlockSchedule>> {
    if prime < 3
        || batch_count == 0
        || complex_bytes == 0
        || device.max_threads_per_block == 0
        || device.max_workgroup_size[0] == 0
        || device.max_workgroup_size[1] == 0
    {
        return Ok(None);
    }
    let precision = match complex_bytes {
        8 => Precision::F32,
        16 => Precision::F64,
        32 => Precision::DoubleDouble,
        _ => return Ok(None),
    };
    let policy = plan_gpu_scheduler_policy(precision, device)?;
    let (_, _, min_registers_per_thread) = rader_outer_register_state(prime, 1, batch_count)?;
    if min_registers_per_thread == 0 {
        return Ok(None);
    }

    // For one standalone type-1 Rader container, upstream's active_rader is one,
    // so the direct-multiplication floor is exactly (p + 1) / 2 threads. The
    // preceding register-derived axisBlock is rounded up to this same multiple.
    let direct_threads = prime.div_ceil(2);
    let initial_threads = ceil_div(prime, min_registers_per_thread)?.max(1);
    let aligned_threads = ceil_div(initial_threads, direct_threads)?
        .checked_mul(direct_threads)
        .ok_or(VkFftError::ArithmeticOverflow {
            operation: "direct Rader axis-block thread alignment",
        })?
        .max(direct_threads);
    if aligned_threads > device.max_threads_per_block
        || aligned_threads > device.max_workgroup_size[0]
    {
        return Ok(None);
    }
    let threads_per_transform = aligned_threads;

    let max_sequence_len_shared = device.shared_memory_bytes / complex_bytes;
    let max_sequence_len_shared_pow2 = device.shared_memory_pow2_bytes / complex_bytes;
    if max_sequence_len_shared == 0 || max_sequence_len_shared_pow2 == 0 {
        return Ok(None);
    }
    let aim_threads = 128usize;
    let warp_size = policy.subgroup_width.max(1);
    let max_batch_coalesced = (policy.coalesced_memory_bytes / complex_bytes).max(1);

    let mut grouped_batch = if let Some(requested) = grouped_batch_override {
        let mut upstream_grouped = (max_sequence_len_shared / prime).max(max_batch_coalesced);
        // axis-0, upload-0, one-upload direct Rader uses locMaxBatchCoalesced=1.
        if threads_per_transform
            .checked_mul(upstream_grouped)
            .is_none_or(|threads| threads > device.max_threads_per_block)
        {
            upstream_grouped = 1;
        }
        upstream_grouped.min(requested).min(batch_count).max(1)
    } else {
        let estimate_batch = if upstream_near_single_subgroup(threads_per_transform, warp_size) {
            (aim_threads / warp_size).max(1)
        } else {
            (aim_threads / threads_per_transform).max(1)
        };
        let mut grouped = if threads_per_transform < aim_threads {
            estimate_batch
        } else {
            1
        }
        .min(batch_count)
        .max(1);
        let can_round_for_swap = !perform_zero_padding
            && (prime.is_multiple_of(2) || threads_per_transform < device.shared_banks.max(1) / 4)
            && grouped > 1
            && grouped
                .checked_mul(prime)
                .is_some_and(|elements| elements < max_sequence_len_shared_pow2);
        if can_round_for_swap {
            grouped = grouped.next_power_of_two().min(batch_count);
        }
        if device.vendor == GpuVendor::Nvidia {
            while grouped
                .checked_mul(threads_per_transform)
                .is_some_and(|threads| threads >= 2 * aim_threads)
                && grouped > max_batch_coalesced
            {
                grouped = (grouped / 2).max(max_batch_coalesced);
            }
        }
        grouped
    };

    grouped_batch = grouped_batch.min(device.max_workgroup_size[1]).max(1);
    if grouped_batch
        .checked_mul(threads_per_transform)
        .is_some_and(|threads| threads > device.max_threads_per_block)
    {
        grouped_batch = (device.max_threads_per_block / threads_per_transform).max(1);
    }
    while grouped_batch > 1
        && grouped_batch
            .checked_mul(prime)
            .is_some_and(|elements| elements > max_sequence_len_shared)
    {
        grouped_batch /= 2;
    }

    let axis_swapped = !perform_zero_padding
        && (prime.is_multiple_of(2) || threads_per_transform < device.shared_banks.max(1) / 4)
        && grouped_batch > 1
        && grouped_batch
            .checked_mul(prime)
            .is_some_and(|elements| elements < max_sequence_len_shared);
    let (local_size_x, local_size_y) = if axis_swapped {
        (grouped_batch, threads_per_transform)
    } else {
        (threads_per_transform, grouped_batch)
    };
    let block = StockhamAxisBlockSchedule {
        threads_per_transform,
        grouped_batch,
        transforms_on_x: axis_swapped,
        axis_swapped,
        local_size_x,
        local_size_y,
    };
    if block.validate(batch_count, device).is_err() {
        return Ok(None);
    }
    Ok(Some(block))
}

/// Port the default axis-0/upload-0 `VkFFTSplitAxisBlock` geometry for a standalone
/// FFT-convolution Rader prime. `container_fft_num == 1` is intentional: this helper
/// groups independent top-level batches and must never reinterpret them as Rader
/// containers (which would incorrectly enable `raderTranspose`). The physical X
/// dimension follows the outer prime-length register state, so p257 uses 17 physical
/// lanes even though its internal 256-point convolution has only 16 active lanes.
pub fn plan_gpu_axis0_fft_rader_batch_block(
    prime: usize,
    batch_count: usize,
    schedule: &RaderFftRegisterSchedule,
    complex_bytes: usize,
    perform_zero_padding: bool,
    device: DeviceProfile,
) -> Result<Option<StockhamAxisBlockSchedule>> {
    plan_gpu_axis0_fft_rader_batch_block_with_grouped_batch(
        prime,
        batch_count,
        schedule,
        complex_bytes,
        perform_zero_padding,
        None,
        device,
    )
}

/// User-override-aware form of [`plan_gpu_axis0_fft_rader_batch_block`]. For the
/// covered standalone FFT-Rader slice, `Some(n)` mirrors upstream's early
/// `configuration.groupedBatch[0]` branch: start from the shared-memory grouping,
/// cap it by `n`, then apply physical Y/total-thread/shared limits without the
/// default aimThreads/NVIDIA widening-shrinking heuristics.
pub fn plan_gpu_axis0_fft_rader_batch_block_with_grouped_batch(
    prime: usize,
    batch_count: usize,
    schedule: &RaderFftRegisterSchedule,
    complex_bytes: usize,
    perform_zero_padding: bool,
    grouped_batch_override: Option<usize>,
    device: DeviceProfile,
) -> Result<Option<StockhamAxisBlockSchedule>> {
    let precision = match complex_bytes {
        8 => Precision::F32,
        16 => Precision::F64,
        _ => return Ok(None),
    };
    plan_gpu_axis0_fft_rader_batch_block_with_grouped_batch_for_precision(
        prime,
        batch_count,
        schedule,
        precision,
        complex_bytes,
        perform_zero_padding,
        grouped_batch_override,
        device,
    )
}

pub(crate) fn plan_gpu_axis0_fft_rader_batch_block_with_grouped_batch_for_precision(
    prime: usize,
    batch_count: usize,
    schedule: &RaderFftRegisterSchedule,
    precision: Precision,
    complex_bytes: usize,
    perform_zero_padding: bool,
    grouped_batch_override: Option<usize>,
    device: DeviceProfile,
) -> Result<Option<StockhamAxisBlockSchedule>> {
    schedule.validate()?;
    if precision.compute_complex_bytes() != complex_bytes
        || prime < 3
        || schedule.prime != prime
        || schedule.outer_fft_len != prime
        || schedule.container_fft_num != 1
        || schedule.execution_container_fft_num != 1
        || schedule.internal_fft.rhs_transform_count != batch_count
        || schedule.internal_fft.register_boost != 1
        || schedule.rader_transpose.is_some()
        || complex_bytes == 0
        || device.max_threads_per_block == 0
        || device.max_workgroup_size[0] == 0
        || device.max_workgroup_size[1] == 0
    {
        return Ok(None);
    }
    let policy = plan_gpu_scheduler_policy(precision, device)?;
    let Some(parent_threads) = plan_gpu_axis0_multi_fft_rader_threads_for_prime_multiplicities(
        prime,
        &[(prime, 1)],
        batch_count,
        device,
    )?
    else {
        return Ok(None);
    };

    let mut threads_per_transform = parent_threads.max(schedule.min_rader_fft_thread_num);
    if threads_per_transform > device.max_threads_per_block
        || threads_per_transform > device.max_workgroup_size[0]
    {
        return Ok(None);
    }
    threads_per_transform = threads_per_transform.max(1);

    let aim_threads = 128usize;
    let warp_size = policy.subgroup_width.max(1);
    let max_batch_coalesced = (policy.coalesced_memory_bytes / complex_bytes).max(1);
    let max_sequence_len_shared = device.shared_memory_bytes / complex_bytes;
    let max_sequence_len_shared_pow2 = device.shared_memory_pow2_bytes / complex_bytes;
    if max_sequence_len_shared == 0 || max_sequence_len_shared_pow2 == 0 {
        return Ok(None);
    }
    let mut grouped_batch = if let Some(requested) = grouped_batch_override {
        (max_sequence_len_shared / prime)
            .max(max_batch_coalesced)
            .min(requested)
            .min(batch_count)
            .max(1)
    } else {
        let estimate_batch = if upstream_near_single_subgroup(threads_per_transform, warp_size) {
            (aim_threads / warp_size).max(1)
        } else {
            (aim_threads / threads_per_transform).max(1)
        };
        // `useRader != 0` makes the second half of upstream's batching predicate true.
        if threads_per_transform < aim_threads {
            estimate_batch
        } else {
            1
        }
        .min(batch_count)
        .max(1)
    };

    if grouped_batch_override.is_none() {
        let can_round_for_swap = !perform_zero_padding
            && (prime.is_multiple_of(2) || threads_per_transform < device.shared_banks.max(1) / 4)
            && grouped_batch > 1
            && grouped_batch
                .checked_mul(prime)
                .is_some_and(|elements| elements < max_sequence_len_shared_pow2);
        if can_round_for_swap {
            grouped_batch = grouped_batch.next_power_of_two().min(batch_count);
        }
        if device.vendor == GpuVendor::Nvidia {
            while grouped_batch
                .checked_mul(threads_per_transform)
                .is_some_and(|threads| threads >= 2 * aim_threads)
                && grouped_batch > max_batch_coalesced
            {
                grouped_batch = (grouped_batch / 2).max(max_batch_coalesced);
            }
        }
    }
    grouped_batch = grouped_batch.min(device.max_workgroup_size[1]).max(1);
    if grouped_batch
        .checked_mul(threads_per_transform)
        .is_some_and(|threads| threads > device.max_threads_per_block)
    {
        grouped_batch = (device.max_threads_per_block / threads_per_transform).max(1);
    }
    while grouped_batch > 1
        && grouped_batch
            .checked_mul(prime)
            .is_some_and(|elements| elements > max_sequence_len_shared)
    {
        grouped_batch /= 2;
    }
    // A single logical transform can still need a physical caller block when the
    // prime-length Rader envelope has a wider thread floor than its `(p - 1)`
    // convolution child. Fixed upstream keeps that distinction for batch=1 too
    // (for example p257 is 17 caller lanes over a 16-lane convolution child and
    // p7681 is 769 over 768). Preserve `None` only when a 1x block would carry no
    // additional physical information.
    if grouped_batch <= 1 && threads_per_transform == schedule.execution_threads_per_workgroup {
        return Ok(None);
    }

    let axis_swapped = !perform_zero_padding
        && (prime.is_multiple_of(2) || threads_per_transform < device.shared_banks.max(1) / 4)
        && grouped_batch > 1
        && grouped_batch
            .checked_mul(prime)
            .is_some_and(|elements| elements < max_sequence_len_shared);
    let (local_size_x, local_size_y) = if axis_swapped {
        (grouped_batch, threads_per_transform)
    } else {
        (threads_per_transform, grouped_batch)
    };
    let block = StockhamAxisBlockSchedule {
        threads_per_transform,
        grouped_batch,
        transforms_on_x: axis_swapped,
        axis_swapped,
        local_size_x,
        local_size_y,
    };
    if block.validate(batch_count, device).is_err() {
        return Ok(None);
    }
    Ok(Some(block))
}

/// Port the default axis-0/upload-0 block-splitter slice used by one-upload C2C
/// Stockham without spatial zero padding.
pub fn plan_gpu_axis0_single_upload_block(
    upload: &StockhamUploadSchedule,
    complex_bytes: usize,
    device: DeviceProfile,
) -> Result<Option<StockhamAxisBlockSchedule>> {
    plan_gpu_axis0_single_upload_block_with_zero_padding(upload, complex_bytes, false, device)
}

/// Zero-padding-aware form of [`plan_gpu_axis0_single_upload_block`]. Fixed upstream
/// keeps the same grouped-batch calculation when spatial padding is enabled but
/// suppresses the bank-conflict `axisSwapped` exchange, so physical ownership stays
/// threads-X/transforms-Y. Rader/Bluestein/user groupedBatch remain outside this slice.
pub fn plan_gpu_axis0_single_upload_block_with_zero_padding(
    upload: &StockhamUploadSchedule,
    complex_bytes: usize,
    perform_zero_padding: bool,
    device: DeviceProfile,
) -> Result<Option<StockhamAxisBlockSchedule>> {
    plan_gpu_axis0_single_upload_block_with_grouped_batch(
        upload,
        complex_bytes,
        perform_zero_padding,
        None,
        device,
    )
}

/// User-override-aware form of the axis-0 single-upload splitter. `Some(n)` mirrors
/// fixed upstream `configuration.groupedBatch[0]`: it caps the precomputed shared-
/// memory grouping before thread/shared limits are applied. Partial final workgroups
/// are represented explicitly through ceil-dispatch plus inactive-batch guards.
pub fn plan_gpu_axis0_single_upload_block_with_grouped_batch(
    upload: &StockhamUploadSchedule,
    complex_bytes: usize,
    perform_zero_padding: bool,
    grouped_batch_override: Option<usize>,
    device: DeviceProfile,
) -> Result<Option<StockhamAxisBlockSchedule>> {
    upload.validate()?;
    if upload.upload_count != 1
        || upload.register_boost != 1
        || complex_bytes == 0
        || device.max_threads_per_block == 0
        || device.max_workgroup_size[0] == 0
        || device.max_workgroup_size[1] == 0
    {
        return Ok(None);
    }
    let radix = &upload.radix_schedules[0];
    let threads_per_transform =
        ceil_div(upload.sequence_len, radix.min_registers_per_thread)?.max(1);
    if threads_per_transform > device.max_workgroup_size[0]
        || threads_per_transform > device.max_threads_per_block
    {
        return Ok(None);
    }

    let policy = plan_gpu_scheduler_policy(
        if complex_bytes == 8 {
            Precision::F32
        } else if complex_bytes == 16 {
            Precision::F64
        } else {
            return Ok(None);
        },
        device,
    )?;
    let aim_threads = 128usize;
    let warp_size = policy.subgroup_width.max(1);
    let max_batch_coalesced = (policy.coalesced_memory_bytes / complex_bytes).max(1);
    let estimate_batch = if upstream_near_single_subgroup(threads_per_transform, warp_size) {
        (aim_threads / warp_size).max(1)
    } else {
        (aim_threads / threads_per_transform).max(1)
    };
    let mut grouped_batch =
        if threads_per_transform < aim_threads && threads_per_transform < warp_size {
            estimate_batch
        } else {
            1
        };
    grouped_batch = grouped_batch.min(upload.batch_count).max(1);

    let max_sequence_len_shared = device.shared_memory_bytes / complex_bytes;
    let max_sequence_len_shared_pow2 = device.shared_memory_pow2_bytes / complex_bytes;
    if max_sequence_len_shared == 0 || max_sequence_len_shared_pow2 == 0 {
        return Ok(None);
    }
    if let Some(grouped_batch_override) = grouped_batch_override {
        let mut user_grouped_batch = (max_sequence_len_shared / upload.sequence_len)
            .max(max_batch_coalesced)
            .min(grouped_batch_override)
            .min(upload.batch_count)
            .max(1);
        user_grouped_batch = user_grouped_batch.min(device.max_workgroup_size[1]).max(1);
        if user_grouped_batch
            .checked_mul(threads_per_transform)
            .is_some_and(|threads| threads > device.max_threads_per_block)
        {
            user_grouped_batch = (device.max_threads_per_block / threads_per_transform).max(1);
        }
        while user_grouped_batch > 1
            && user_grouped_batch
                .checked_mul(upload.sequence_len / upload.register_boost)
                .is_some_and(|elements| elements > max_sequence_len_shared)
        {
            user_grouped_batch /= 2;
        }
        let axis_swapped = !perform_zero_padding
            && (upload.sequence_len.is_multiple_of(2)
                || threads_per_transform < device.shared_banks.max(1) / 4)
            && user_grouped_batch > 1
            && user_grouped_batch
                .checked_mul(upload.sequence_len)
                .is_some_and(|elements| elements < max_sequence_len_shared);
        let (local_size_x, local_size_y) = if axis_swapped {
            (user_grouped_batch, threads_per_transform)
        } else {
            (threads_per_transform, user_grouped_batch)
        };
        let block = StockhamAxisBlockSchedule {
            threads_per_transform,
            grouped_batch: user_grouped_batch,
            transforms_on_x: axis_swapped,
            axis_swapped,
            local_size_x,
            local_size_y,
        };
        if block.validate(upload.batch_count, device).is_err() {
            return Ok(None);
        }
        return Ok(Some(block));
    }
    // Default no-zero-padding/single-upload bank-conflict branch: round the batch
    // dimension to a power of two when the whole grouped transform still fits the
    // power-of-two shared-memory capacity.
    let can_swap = !perform_zero_padding
        && (upload.sequence_len.is_multiple_of(2)
            || threads_per_transform < device.shared_banks.max(1) / 4)
        && grouped_batch > 1
        && grouped_batch
            .checked_mul(upload.sequence_len)
            .is_some_and(|elements| elements < max_sequence_len_shared_pow2);
    if can_swap {
        grouped_batch = grouped_batch.next_power_of_two();
    }

    if device.vendor == GpuVendor::Nvidia {
        while grouped_batch
            .checked_mul(threads_per_transform)
            .is_some_and(|threads| threads >= 2 * aim_threads)
            && grouped_batch > max_batch_coalesced
        {
            grouped_batch = (grouped_batch / 2).max(max_batch_coalesced);
        }
    }
    grouped_batch = grouped_batch.min(device.max_workgroup_size[1]).max(1);
    if grouped_batch
        .checked_mul(threads_per_transform)
        .is_some_and(|threads| threads > device.max_threads_per_block)
    {
        grouped_batch = (device.max_threads_per_block / threads_per_transform).max(1);
    }
    while grouped_batch > 1
        && grouped_batch
            .checked_mul(upload.sequence_len)
            .is_some_and(|elements| elements > max_sequence_len_shared)
    {
        grouped_batch /= 2;
    }

    let axis_swapped = !perform_zero_padding
        && (upload.sequence_len.is_multiple_of(2)
            || threads_per_transform < device.shared_banks.max(1) / 4)
        && grouped_batch > 1
        && grouped_batch
            .checked_mul(upload.sequence_len)
            .is_some_and(|elements| elements < max_sequence_len_shared);
    let (local_size_x, local_size_y) = if axis_swapped {
        (grouped_batch, threads_per_transform)
    } else {
        (threads_per_transform, grouped_batch)
    };
    let block = StockhamAxisBlockSchedule {
        threads_per_transform,
        grouped_batch,
        transforms_on_x: axis_swapped,
        axis_swapped,
        local_size_x,
        local_size_y,
    };
    if block.validate(upload.batch_count, device).is_err() {
        return Ok(None);
    }
    Ok(Some(block))
}

/// Port the fixed-upstream `axis_id >= 1` block splitter for one Stockham
/// component of a two/three-upload Four-step axis. Higher axes keep independent
/// sequences on X and FFT threads on Y for every upload; unlike axis 0, neither
/// upload id nor `stageStartSize` changes that physical ownership. The common
/// pre-branch groupedBatch widening/strided-capacity logic is still upload-aware.
pub fn plan_gpu_other_axis_four_step_upload_block(
    upload: &StockhamUploadSchedule,
    axis_upload_id: usize,
    transform_count: usize,
    fastest_axis_len: usize,
    complex_bytes: usize,
    device: DeviceProfile,
) -> Result<Option<StockhamAxisBlockSchedule>> {
    plan_gpu_other_axis_four_step_upload_block_with_grouped_batch(
        upload,
        axis_upload_id,
        transform_count,
        fastest_axis_len,
        complex_bytes,
        None,
        None,
        device,
    )
}

/// User-grouped form of [`plan_gpu_other_axis_four_step_upload_block`]. This
/// mirrors the early `configuration.groupedBatch[axis_id]` branch from the fixed
/// upstream `VkFFT_AxisBlockSplitter.h`, including its literal `groupedBatch[1]`
/// gate and thread-limit assignment. `axis1_grouped_batch_override` is the
/// row-major second-fastest public axis mapped to upstream axis id 1.
pub fn plan_gpu_other_axis_four_step_upload_block_with_grouped_batch(
    upload: &StockhamUploadSchedule,
    axis_upload_id: usize,
    transform_count: usize,
    fastest_axis_len: usize,
    complex_bytes: usize,
    grouped_batch_override: Option<usize>,
    axis1_grouped_batch_override: Option<usize>,
    device: DeviceProfile,
) -> Result<Option<StockhamAxisBlockSchedule>> {
    let precision = match complex_bytes {
        8 => Precision::F32,
        16 => Precision::F64,
        _ => return Ok(None),
    };
    plan_gpu_other_axis_four_step_upload_block_with_grouped_batch_for_precision(
        upload,
        axis_upload_id,
        transform_count,
        fastest_axis_len,
        precision,
        complex_bytes,
        grouped_batch_override,
        axis1_grouped_batch_override,
        device,
    )
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct OtherAxisFourStepPhysicalContext {
    pub has_direct_rader: bool,
    pub available_shared_memory_bytes: usize,
}

pub(crate) fn plan_gpu_other_axis_four_step_block_from_shape_for_precision(
    upload_count: usize,
    axis_upload_id: usize,
    fft_len: usize,
    transform_count: usize,
    fastest_axis_len: usize,
    threads_per_transform: usize,
    precision: Precision,
    complex_bytes: usize,
    grouped_batch_override: Option<usize>,
    axis1_grouped_batch_override: Option<usize>,
    physical: OtherAxisFourStepPhysicalContext,
    device: DeviceProfile,
) -> Result<Option<StockhamAxisBlockSchedule>> {
    if !matches!(upload_count, 2 | 3)
        || axis_upload_id >= upload_count
        || fft_len == 0
        || transform_count == 0
        || fastest_axis_len == 0
        || threads_per_transform == 0
        || complex_bytes == 0
        || physical.available_shared_memory_bytes == 0
        || device.max_threads_per_block == 0
        || device.max_workgroup_size[0] == 0
        || device.max_workgroup_size[1] == 0
        || threads_per_transform > device.max_threads_per_block
        || threads_per_transform > device.max_workgroup_size[1]
    {
        return Ok(None);
    }
    let policy = plan_gpu_scheduler_policy(precision, device)?;
    let max_sequence_len_shared = physical.available_shared_memory_bytes / complex_bytes;
    if max_sequence_len_shared == 0 {
        return Ok(None);
    }
    let coalesced = policy.coalesced_memory_bytes.max(complex_bytes);
    let max_single_size_strided = if coalesced > complex_bytes {
        physical.available_shared_memory_bytes / coalesced
    } else {
        max_sequence_len_shared
    };
    if max_single_size_strided == 0 {
        return Ok(None);
    }
    let mut max_batch_coalesced = (policy.coalesced_memory_bytes / complex_bytes).max(1);
    let mut grouped_batch = if max_single_size_strided / fft_len > 1 {
        (max_single_size_strided / fft_len)
            .checked_mul(max_batch_coalesced)
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "higher-axis Four-step initial grouped batch",
            })?
    } else {
        max_batch_coalesced
    };

    if let Some(requested) = grouped_batch_override {
        if requested == 0 {
            return Ok(None);
        }
        // The upstream user-grouped higher-axis branch still applies Direct-Rader
        // pressure to the automatic pre-clipped groupedBatch before consulting the
        // configured groupedBatch gate. This can collapse a wide shared-memory group
        // to maxBatchCoalesced even when the explicit group itself would fit.
        if physical.has_direct_rader
            && threads_per_transform
                .checked_mul(grouped_batch)
                .is_some_and(|threads| threads > device.max_threads_per_block)
        {
            grouped_batch = max_batch_coalesced.max(1);
        }
        let mut axis_block_0 = fastest_axis_len.min(grouped_batch).max(1);
        let upstream_axis1_gate = axis1_grouped_batch_override.unwrap_or(0);
        if axis_block_0 > upstream_axis1_gate {
            axis_block_0 = requested;
        }
        axis_block_0 = axis_block_0
            .min(transform_count)
            .min(device.max_workgroup_size[0])
            .max(1);
        if axis_block_0
            .checked_mul(threads_per_transform)
            .is_some_and(|threads| threads > device.max_threads_per_block)
        {
            axis_block_0 = device.max_threads_per_block / axis_block_0;
            if axis_block_0 == 0 {
                return Ok(None);
            }
            axis_block_0 = axis_block_0
                .min(transform_count)
                .min(device.max_workgroup_size[0])
                .max(1);
        }
        let block = StockhamAxisBlockSchedule {
            threads_per_transform,
            grouped_batch: axis_block_0,
            transforms_on_x: true,
            axis_swapped: false,
            local_size_x: axis_block_0,
            local_size_y: threads_per_transform,
        };
        if block.validate(transform_count, device).is_err() {
            return Ok(None);
        }
        return Ok(Some(block));
    }

    if device.vendor == GpuVendor::Nvidia {
        let widen =
            upload_count == 3 || (upload_count == 2 && (axis_upload_id > 0 || fft_len <= 512));
        if widen {
            for bytes in [64usize, 128] {
                let candidate = bytes / complex_bytes;
                if candidate > 0
                    && fft_len
                        .checked_mul(candidate)
                        .is_some_and(|elements| elements <= max_sequence_len_shared)
                {
                    grouped_batch = candidate;
                    max_batch_coalesced = candidate;
                }
            }
        }
    } else if axis_upload_id == 0 {
        if upload_count == 2
            && fft_len
                .checked_mul(max_batch_coalesced)
                .is_some_and(|elements| elements <= max_sequence_len_shared)
        {
            grouped_batch = ceil_div(grouped_batch, 2)?;
        }
        if upload_count == 3 && fft_len < max_sequence_len_shared / (2 * complex_bytes) {
            grouped_batch = ceil_div(grouped_batch, 2)?;
        }
    }
    if grouped_batch < max_batch_coalesced {
        grouped_batch = max_batch_coalesced;
    }
    grouped_batch = (grouped_batch / max_batch_coalesced) * max_batch_coalesced;
    if fft_len > max_single_size_strided {
        grouped_batch = (max_sequence_len_shared / fft_len).max(1);
    }
    let warp_size = policy.subgroup_width.max(1);
    if grouped_batch > warp_size {
        grouped_batch = (grouped_batch / warp_size) * warp_size;
    }
    if grouped_batch > 2 * max_batch_coalesced {
        grouped_batch = (grouped_batch / (2 * max_batch_coalesced)) * (2 * max_batch_coalesced);
    }
    if grouped_batch > 4 * max_batch_coalesced {
        grouped_batch = (grouped_batch / (4 * max_batch_coalesced)) * (4 * max_batch_coalesced);
    }
    grouped_batch = grouped_batch.max(1);

    // Fixed upstream applies the Direct-Rader thread-pressure reset before clipping
    // groupedBatch to the fastest higher-axis extent. This ordering matters when a
    // large pre-clipped group would overflow maxThreadsNum even though the final X
    // tile itself would fit (for example AMD F64-compute/F32-storage N68=4*p17).
    if physical.has_direct_rader
        && threads_per_transform
            .checked_mul(grouped_batch)
            .is_some_and(|threads| threads > device.max_threads_per_block)
    {
        grouped_batch = max_batch_coalesced.max(1);
    }

    let mut axis_block_0 = fastest_axis_len.min(grouped_batch).max(1);
    if device.vendor == GpuVendor::Nvidia {
        let aim_threads = 128usize;
        while axis_block_0
            .checked_mul(threads_per_transform)
            .is_some_and(|threads| threads >= 2 * aim_threads)
            && axis_block_0 > max_batch_coalesced
        {
            axis_block_0 = (axis_block_0 / 2).max(max_batch_coalesced);
        }
    }
    axis_block_0 = axis_block_0
        .min(transform_count)
        .min(device.max_workgroup_size[0])
        .max(1);
    if axis_block_0
        .checked_mul(threads_per_transform)
        .is_some_and(|threads| threads > device.max_threads_per_block)
    {
        let original = axis_block_0;
        for divisor in 1..=original {
            let candidate = original / divisor;
            if candidate > 0
                && candidate
                    .checked_mul(threads_per_transform)
                    .is_some_and(|threads| threads <= device.max_threads_per_block)
            {
                axis_block_0 = candidate;
                break;
            }
        }
    }
    let block = StockhamAxisBlockSchedule {
        threads_per_transform,
        grouped_batch: axis_block_0,
        transforms_on_x: true,
        axis_swapped: false,
        local_size_x: axis_block_0,
        local_size_y: threads_per_transform,
    };
    if block.validate(transform_count, device).is_err() {
        return Ok(None);
    }
    Ok(Some(block))
}

pub(crate) fn plan_gpu_other_axis_four_step_upload_block_with_grouped_batch_for_precision(
    upload: &StockhamUploadSchedule,
    axis_upload_id: usize,
    transform_count: usize,
    fastest_axis_len: usize,
    precision: Precision,
    complex_bytes: usize,
    grouped_batch_override: Option<usize>,
    axis1_grouped_batch_override: Option<usize>,
    device: DeviceProfile,
) -> Result<Option<StockhamAxisBlockSchedule>> {
    upload.validate()?;
    if !matches!(upload.upload_count, 2 | 3)
        || upload.register_boost != 1
        || axis_upload_id >= upload.upload_count
        || transform_count == 0
        || fastest_axis_len == 0
        || complex_bytes == 0
        || device.max_threads_per_block == 0
        || device.max_workgroup_size[0] == 0
        || device.max_workgroup_size[1] == 0
    {
        return Ok(None);
    }
    let radix = &upload.radix_schedules[axis_upload_id];
    let fft_len = upload.axis_split[axis_upload_id];
    if radix.fft_len != fft_len || radix.rhs_transform_count != transform_count {
        return Ok(None);
    }
    let threads_per_transform =
        (ceil_div(fft_len, radix.min_registers_per_thread)? / upload.register_boost).max(1);
    plan_gpu_other_axis_four_step_block_from_shape_for_precision(
        upload.upload_count,
        axis_upload_id,
        fft_len,
        transform_count,
        fastest_axis_len,
        threads_per_transform,
        precision,
        complex_bytes,
        grouped_batch_override,
        axis1_grouped_batch_override,
        OtherAxisFourStepPhysicalContext {
            has_direct_rader: false,
            available_shared_memory_bytes: device.shared_memory_bytes,
        },
        device,
    )
}

/// Fixed-upstream `axis_id >= 1` Four-step block scorer for Quad/DD uploads.
/// This mirrors [`plan_gpu_other_axis_four_step_upload_block_with_grouped_batch`]
/// but consumes the dedicated Quad register schedule and the 32-byte DD complex
/// footprint instead of borrowing ordinary F64 register metadata.
pub fn plan_gpu_double_double_other_axis_four_step_upload_block(
    upload: &DoubleDoubleStockhamUploadSchedule,
    axis_upload_id: usize,
    transform_count: usize,
    fastest_axis_len: usize,
    grouped_batch_override: Option<usize>,
    axis1_grouped_batch_override: Option<usize>,
    device: DeviceProfile,
) -> Result<Option<StockhamAxisBlockSchedule>> {
    upload.validate()?;
    if !matches!(upload.upload_count, 2 | 3)
        || axis_upload_id >= upload.upload_count
        || transform_count == 0
        || fastest_axis_len == 0
        || device.max_threads_per_block == 0
        || device.max_workgroup_size[0] == 0
        || device.max_workgroup_size[1] == 0
    {
        return Ok(None);
    }
    let policy = plan_gpu_scheduler_policy(Precision::DoubleDouble, device)?;
    let quad = &upload.quad_schedules[axis_upload_id];
    let fft_len = upload.axis_split[axis_upload_id];
    if quad.fft_len != fft_len || quad.rhs_transform_count != transform_count {
        return Ok(None);
    }
    let threads_per_transform = ceil_div(fft_len, quad.min_registers_per_thread)?.max(1);
    if threads_per_transform > device.max_threads_per_block
        || threads_per_transform > device.max_workgroup_size[1]
    {
        return Ok(None);
    }

    // VkFFTSplitAxisBlock receives both allowedSharedMemory and
    // allowedSharedMemoryPow2. The upload splitter may have used the Pow2 budget,
    // but higher-axis groupedBatch scoring uses the full allowedSharedMemory.
    let usable_shared_memory_bytes = double_double_stockham_usable_shared_memory_bytes(device);
    let max_sequence_len_shared = usable_shared_memory_bytes / DD_COMPLEX_BYTES;
    if max_sequence_len_shared == 0 {
        return Ok(None);
    }
    let coalesced = policy.coalesced_memory_bytes.max(DD_COMPLEX_BYTES);
    let max_single_size_strided = if coalesced > DD_COMPLEX_BYTES {
        usable_shared_memory_bytes / coalesced
    } else {
        max_sequence_len_shared
    };
    if max_single_size_strided == 0 {
        return Ok(None);
    }
    let mut max_batch_coalesced = (policy.coalesced_memory_bytes / DD_COMPLEX_BYTES).max(1);
    let mut grouped_batch = if max_single_size_strided / fft_len > 1 {
        (max_single_size_strided / fft_len)
            .checked_mul(max_batch_coalesced)
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "double-double higher-axis Four-step initial grouped batch",
            })?
    } else {
        max_batch_coalesced
    };

    if let Some(requested) = grouped_batch_override {
        if requested == 0 {
            return Ok(None);
        }
        let mut axis_block_0 = fastest_axis_len.min(grouped_batch).max(1);
        let upstream_axis1_gate = axis1_grouped_batch_override.unwrap_or(0);
        if axis_block_0 > upstream_axis1_gate {
            axis_block_0 = requested;
        }
        axis_block_0 = axis_block_0
            .min(transform_count)
            .min(device.max_workgroup_size[0])
            .max(1);
        if axis_block_0
            .checked_mul(threads_per_transform)
            .is_some_and(|threads| threads > device.max_threads_per_block)
        {
            axis_block_0 = device.max_threads_per_block / axis_block_0;
            if axis_block_0 == 0 {
                return Ok(None);
            }
            axis_block_0 = axis_block_0
                .min(transform_count)
                .min(device.max_workgroup_size[0])
                .max(1);
        }
        let block = StockhamAxisBlockSchedule {
            threads_per_transform,
            grouped_batch: axis_block_0,
            transforms_on_x: true,
            axis_swapped: false,
            local_size_x: axis_block_0,
            local_size_y: threads_per_transform,
        };
        if block.validate(transform_count, device).is_err() {
            return Ok(None);
        }
        return Ok(Some(block));
    }

    if device.vendor == GpuVendor::Nvidia {
        let widen = upload.upload_count == 3
            || (upload.upload_count == 2 && (axis_upload_id > 0 || fft_len <= 512));
        if widen {
            for bytes in [64usize, 128] {
                let candidate = bytes / DD_COMPLEX_BYTES;
                if candidate > 0
                    && fft_len
                        .checked_mul(candidate)
                        .is_some_and(|elements| elements <= max_sequence_len_shared)
                {
                    grouped_batch = candidate;
                    max_batch_coalesced = candidate;
                }
            }
        }
    } else if axis_upload_id == 0 {
        if upload.upload_count == 2
            && fft_len
                .checked_mul(max_batch_coalesced)
                .is_some_and(|elements| elements <= max_sequence_len_shared)
        {
            grouped_batch = ceil_div(grouped_batch, 2)?;
        }
        if upload.upload_count == 3 && fft_len < max_sequence_len_shared / (2 * DD_COMPLEX_BYTES) {
            grouped_batch = ceil_div(grouped_batch, 2)?;
        }
    }
    if grouped_batch < max_batch_coalesced {
        grouped_batch = max_batch_coalesced;
    }
    grouped_batch = (grouped_batch / max_batch_coalesced) * max_batch_coalesced;
    if fft_len > max_single_size_strided {
        grouped_batch = (max_sequence_len_shared / fft_len).max(1);
    }
    let warp_size = policy.subgroup_width.max(1);
    if grouped_batch > warp_size {
        grouped_batch = (grouped_batch / warp_size) * warp_size;
    }
    if grouped_batch > 2 * max_batch_coalesced {
        grouped_batch = (grouped_batch / (2 * max_batch_coalesced)) * (2 * max_batch_coalesced);
    }
    if grouped_batch > 4 * max_batch_coalesced {
        grouped_batch = (grouped_batch / (4 * max_batch_coalesced)) * (4 * max_batch_coalesced);
    }
    grouped_batch = grouped_batch.max(1);

    let mut axis_block_0 = fastest_axis_len.min(grouped_batch).max(1);
    if device.vendor == GpuVendor::Nvidia {
        const AIM_THREADS: usize = 128;
        while axis_block_0
            .checked_mul(threads_per_transform)
            .is_some_and(|threads| threads >= 2 * AIM_THREADS)
            && axis_block_0 > max_batch_coalesced
        {
            axis_block_0 = (axis_block_0 / 2).max(max_batch_coalesced);
        }
    }
    axis_block_0 = axis_block_0
        .min(transform_count)
        .min(device.max_workgroup_size[0])
        .max(1);
    if axis_block_0
        .checked_mul(threads_per_transform)
        .is_some_and(|threads| threads > device.max_threads_per_block)
    {
        let original = axis_block_0;
        for divisor in 1..=original {
            let candidate = original / divisor;
            if candidate > 0
                && candidate
                    .checked_mul(threads_per_transform)
                    .is_some_and(|threads| threads <= device.max_threads_per_block)
            {
                axis_block_0 = candidate;
                break;
            }
        }
    }
    let block = StockhamAxisBlockSchedule {
        threads_per_transform,
        grouped_batch: axis_block_0,
        transforms_on_x: true,
        axis_swapped: false,
        local_size_x: axis_block_0,
        local_size_y: threads_per_transform,
    };
    if block.validate(transform_count, device).is_err() {
        return Ok(None);
    }
    Ok(Some(block))
}

/// Port the default `axis_id >= 1`, single-upload, boost-1 Stockham slice of
/// upstream `VkFFTSplitAxisBlock`. VkFFT keeps grouped independent sequences on X
/// and FFT threads on Y for strided/higher axes; unlike axis 0, this branch never
/// performs the bank-conflict X/Y swap. `fastest_axis_len` corresponds to
/// `actualFFTSizePerAxis[axis_id][0]` (VkFFT's physical `size[0]`); for this crate's
/// row-major tensors it is the last logical dimension, not the whole inner stride.
pub fn plan_gpu_other_axis_single_upload_block(
    upload: &StockhamUploadSchedule,
    fastest_axis_len: usize,
    complex_bytes: usize,
    device: DeviceProfile,
) -> Result<Option<StockhamAxisBlockSchedule>> {
    plan_gpu_other_axis_single_upload_block_with_grouped_batch(
        upload,
        fastest_axis_len,
        complex_bytes,
        None,
        None,
        device,
    )
}

/// User-override-aware `axis_id >= 1` branch from the fixed upstream
/// `VkFFTSplitAxisBlock`. The public tensor axes are row-major, so callers map
/// the second-fastest logical axis to `axis1_grouped_batch_override`, upstream's
/// literal `configuration.groupedBatch[1]` gate for all strided axes.
/// The user value controls physical independent sequences on local X; it is not
/// multiplied by the flattened ND line count.
pub fn plan_gpu_other_axis_single_upload_block_with_grouped_batch(
    upload: &StockhamUploadSchedule,
    fastest_axis_len: usize,
    complex_bytes: usize,
    grouped_batch_override: Option<usize>,
    axis1_grouped_batch_override: Option<usize>,
    device: DeviceProfile,
) -> Result<Option<StockhamAxisBlockSchedule>> {
    let precision = match complex_bytes {
        8 => Precision::F32,
        16 => Precision::F64,
        _ => return Ok(None),
    };
    plan_gpu_other_axis_single_upload_block_with_grouped_batch_for_precision(
        upload,
        fastest_axis_len,
        precision,
        complex_bytes,
        grouped_batch_override,
        axis1_grouped_batch_override,
        device,
    )
}

pub(crate) fn plan_gpu_other_axis_single_upload_block_with_grouped_batch_for_precision(
    upload: &StockhamUploadSchedule,
    fastest_axis_len: usize,
    precision: Precision,
    complex_bytes: usize,
    grouped_batch_override: Option<usize>,
    axis1_grouped_batch_override: Option<usize>,
    device: DeviceProfile,
) -> Result<Option<StockhamAxisBlockSchedule>> {
    upload.validate()?;
    if precision.compute_complex_bytes() != complex_bytes
        || upload.upload_count != 1
        || fastest_axis_len == 0
        || complex_bytes == 0
        || device.max_threads_per_block == 0
        || device.max_workgroup_size[0] == 0
        || device.max_workgroup_size[1] == 0
    {
        return Ok(None);
    }
    if grouped_batch_override.is_some() && upload.register_boost != 1 {
        return Ok(None);
    }
    let policy = plan_gpu_scheduler_policy(precision, device)?;
    let radix = &upload.radix_schedules[0];
    if radix.fft_len != upload.sequence_len || radix.rhs_transform_count != upload.batch_count {
        return Ok(None);
    }
    let threads_per_transform = (ceil_div(upload.sequence_len, radix.min_registers_per_thread)?
        / upload.register_boost)
        .max(1);
    if threads_per_transform > device.max_threads_per_block
        || threads_per_transform > device.max_workgroup_size[1]
    {
        return Ok(None);
    }

    let max_sequence_len_shared = device.shared_memory_bytes / complex_bytes;
    let coalesced = policy.coalesced_memory_bytes.max(complex_bytes);
    let max_single_size_strided = if coalesced > complex_bytes {
        device.shared_memory_bytes / coalesced
    } else {
        max_sequence_len_shared
    };
    if max_sequence_len_shared == 0 || max_single_size_strided == 0 {
        return Ok(None);
    }
    let max_batch_coalesced = (policy.coalesced_memory_bytes / complex_bytes).max(1);
    let mut grouped_batch = if max_single_size_strided / upload.sequence_len > 1 {
        (max_single_size_strided / upload.sequence_len)
            .checked_mul(max_batch_coalesced)
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "higher-axis initial grouped batch",
            })?
    } else {
        max_batch_coalesced
    };
    if let Some(requested) = grouped_batch_override {
        if requested == 0 {
            return Ok(None);
        }
        // Fixed upstream's early user branch executes before the later automatic
        // warp/coalescing/NVIDIA aimThreads normalization below. For axis_id>=1 it
        // first caps by physical size[0], then applies the user groupedBatch gate.
        let mut user_grouped_batch = fastest_axis_len.min(grouped_batch).max(1);
        let upstream_axis1_gate = axis1_grouped_batch_override.unwrap_or(0);
        if user_grouped_batch > upstream_axis1_gate {
            user_grouped_batch = requested;
        }
        user_grouped_batch = user_grouped_batch
            .min(upload.batch_count)
            .min(device.max_workgroup_size[0])
            .max(1);
        if user_grouped_batch
            .checked_mul(threads_per_transform)
            .is_some_and(|threads| threads > device.max_threads_per_block)
        {
            // Preserve the fixed upstream assignment literally: axisBlock[0] is
            // replaced with maxThreadsNum / axisBlock[0] (not / axisBlock[1]).
            user_grouped_batch = device.max_threads_per_block / user_grouped_batch;
            if user_grouped_batch == 0 {
                return Ok(None);
            }
        }
        let block = StockhamAxisBlockSchedule {
            threads_per_transform,
            grouped_batch: user_grouped_batch,
            transforms_on_x: true,
            axis_swapped: false,
            local_size_x: user_grouped_batch,
            local_size_y: threads_per_transform,
        };
        if block.validate(upload.batch_count, device).is_err() {
            return Ok(None);
        }
        return Ok(Some(block));
    }
    grouped_batch = grouped_batch.max(max_batch_coalesced);
    grouped_batch = (grouped_batch / max_batch_coalesced) * max_batch_coalesced;
    if upload.sequence_len > max_single_size_strided {
        grouped_batch = (max_sequence_len_shared / upload.sequence_len).max(1);
    }
    let warp_size = policy.subgroup_width.max(1);
    if grouped_batch > warp_size {
        grouped_batch = (grouped_batch / warp_size) * warp_size;
    }
    if grouped_batch > 2 * max_batch_coalesced {
        grouped_batch = (grouped_batch / (2 * max_batch_coalesced)) * (2 * max_batch_coalesced);
    }
    if grouped_batch > 4 * max_batch_coalesced {
        grouped_batch = (grouped_batch / (4 * max_batch_coalesced)) * (4 * max_batch_coalesced);
    }
    grouped_batch = grouped_batch
        .max(1)
        .min(fastest_axis_len)
        .min(upload.batch_count);

    let aim_threads = 128usize;
    if device.vendor == GpuVendor::Nvidia {
        while grouped_batch
            .checked_mul(threads_per_transform)
            .is_some_and(|threads| threads >= 2 * aim_threads)
            && grouped_batch > max_batch_coalesced
        {
            grouped_batch = (grouped_batch / 2).max(max_batch_coalesced);
        }
    }
    grouped_batch = grouped_batch.min(device.max_workgroup_size[0]).max(1);
    if grouped_batch
        .checked_mul(threads_per_transform)
        .is_some_and(|threads| threads > device.max_threads_per_block)
    {
        let original = grouped_batch;
        for divisor in 1..=original {
            let candidate = original / divisor;
            if candidate > 0
                && candidate
                    .checked_mul(threads_per_transform)
                    .is_some_and(|threads| threads <= device.max_threads_per_block)
            {
                grouped_batch = candidate;
                break;
            }
        }
    }

    let block = StockhamAxisBlockSchedule {
        threads_per_transform,
        grouped_batch,
        transforms_on_x: true,
        axis_swapped: false,
        local_size_x: grouped_batch,
        local_size_y: threads_per_transform,
    };
    if block.validate(upload.batch_count, device).is_err() {
        return Ok(None);
    }
    Ok(Some(block))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FourStepAxisBlockRequest {
    pub axis_upload_id: usize,
    pub stage_start_size: usize,
    pub transform_count: usize,
    pub outer_batch_count: usize,
    pub perform_zero_padding: bool,
    pub grouped_batch_override: Option<usize>,
}

/// Shared implementation of fixed upstream's early user-groupedBatch branch for
/// a Four-step upload. Unlike the Stockham-specific entry point below, this only
/// needs the physical thread count of one upload component. That makes the
/// ownership geometry reusable by forced-Rader uploads whose component may be a
/// recursive Cooley-Tukey tree rather than one Stockham kernel. `upload_count == 1`
/// reuses the same upstream user-groupedBatch branch for a top-level composite
/// Cooley boundary that has no physical Four-step upload split.
///
/// Keep the policy lookup below compute-byte-shaped. This helper receives a
/// post-scheduler component shape rather than the original storage precision.
/// Pinned upstream F16 N323=`p17*p19`, batch32/groupedBatch16 still resolves to
/// 21 lanes x group12 even though the application-level coalesced width is 64 B;
/// blindly retagging this shape helper as F16 would incorrectly reduce it to 8.
pub fn plan_gpu_axis0_four_step_grouped_block_from_shape(
    upload_count: usize,
    fft_len: usize,
    threads_per_transform: usize,
    request: FourStepAxisBlockRequest,
    complex_bytes: usize,
    device: DeviceProfile,
) -> Result<Option<StockhamAxisBlockSchedule>> {
    let precision = match complex_bytes {
        8 => Precision::F32,
        16 => Precision::F64,
        _ => return Ok(None),
    };
    plan_gpu_axis0_four_step_grouped_block_from_shape_for_precision(
        upload_count,
        fft_len,
        threads_per_transform,
        request,
        precision,
        complex_bytes,
        device,
    )
}

pub(crate) fn plan_gpu_axis0_four_step_grouped_block_from_shape_for_precision(
    upload_count: usize,
    fft_len: usize,
    threads_per_transform: usize,
    request: FourStepAxisBlockRequest,
    precision: Precision,
    complex_bytes: usize,
    device: DeviceProfile,
) -> Result<Option<StockhamAxisBlockSchedule>> {
    plan_gpu_axis0_four_step_grouped_block_from_shape_impl(
        upload_count,
        fft_len,
        threads_per_transform,
        request,
        precision,
        complex_bytes,
        None,
        None,
        device,
    )
}

pub(crate) fn plan_gpu_axis0_direct_rader_four_step_grouped_block_from_shape_for_precision(
    upload_count: usize,
    fft_len: usize,
    threads_per_transform: usize,
    direct_rader_prime: usize,
    request: FourStepAxisBlockRequest,
    precision: Precision,
    complex_bytes: usize,
    device: DeviceProfile,
) -> Result<Option<StockhamAxisBlockSchedule>> {
    if direct_rader_prime < 3 || !fft_len.is_multiple_of(direct_rader_prime) {
        return Ok(None);
    }
    plan_gpu_axis0_four_step_grouped_block_from_shape_impl(
        upload_count,
        fft_len,
        threads_per_transform,
        request,
        precision,
        complex_bytes,
        None,
        Some(direct_rader_prime),
        device,
    )
}

pub(crate) fn plan_gpu_axis0_rader_four_step_grouped_block_from_split_state_for_precision(
    upload_count: usize,
    fft_len: usize,
    state: &Axis0RaderSplitState,
    request: FourStepAxisBlockRequest,
    precision: Precision,
    complex_bytes: usize,
    device: DeviceProfile,
) -> Result<Option<StockhamAxisBlockSchedule>> {
    let max_batch_coalesced =
        (plan_gpu_scheduler_policy(precision, device)?.coalesced_memory_bytes / complex_bytes)
            .max(1);
    let physical_thread_limit = device
        .max_threads_per_block
        .min(device.max_workgroup_size[0]);
    let Some(threads_per_transform) = axis0_rader_threads_from_split_state(
        fft_len,
        state,
        max_batch_coalesced,
        physical_thread_limit,
    )?
    else {
        return Ok(None);
    };
    plan_gpu_axis0_four_step_grouped_block_from_shape_impl(
        upload_count,
        fft_len,
        threads_per_transform,
        request,
        precision,
        complex_bytes,
        Some(state),
        None,
        device,
    )
}

fn plan_gpu_axis0_four_step_grouped_block_from_shape_impl(
    upload_count: usize,
    fft_len: usize,
    threads_per_transform: usize,
    request: FourStepAxisBlockRequest,
    precision: Precision,
    complex_bytes: usize,
    rader_state: Option<&Axis0RaderSplitState>,
    direct_rader_prime: Option<usize>,
    device: DeviceProfile,
) -> Result<Option<StockhamAxisBlockSchedule>> {
    if precision.compute_complex_bytes() != complex_bytes {
        return Ok(None);
    }
    let FourStepAxisBlockRequest {
        axis_upload_id,
        stage_start_size,
        transform_count,
        outer_batch_count,
        perform_zero_padding,
        grouped_batch_override,
    } = request;
    let Some(grouped_batch_override) = grouped_batch_override else {
        return Ok(None);
    };
    if !matches!(upload_count, 1..=3)
        || axis_upload_id >= upload_count
        || fft_len == 0
        || threads_per_transform == 0
        || stage_start_size == 0
        || transform_count == 0
        || outer_batch_count == 0
        || !transform_count.is_multiple_of(outer_batch_count)
        || grouped_batch_override == 0
        || device.max_threads_per_block == 0
        || device.max_workgroup_size[0] == 0
        || device.max_workgroup_size[1] == 0
    {
        return Ok(None);
    }
    let policy = plan_gpu_scheduler_policy(precision, device)?;
    let max_direct_prime = rader_state
        .and_then(|state| {
            state
                .direct_prime_multiplicities
                .iter()
                .map(|(prime, _)| *prime)
                .max()
        })
        .or(direct_rader_prime);
    let direct_reserve = if let Some(prime) = max_direct_prime {
        prime.saturating_sub(1).checked_mul(complex_bytes).ok_or(
            VkFftError::ArithmeticOverflow {
                operation: "user-grouped Four-step Direct-Rader shared-memory reserve",
            },
        )?
    } else {
        0
    };
    if direct_reserve >= device.shared_memory_bytes
        || direct_reserve >= device.shared_memory_pow2_bytes
    {
        return Ok(None);
    }
    let allowed_shared_memory = device.shared_memory_bytes - direct_reserve;
    if allowed_shared_memory < complex_bytes {
        return Ok(None);
    }
    let max_sequence_len_shared = allowed_shared_memory / complex_bytes;
    let coalesced = policy.coalesced_memory_bytes.max(complex_bytes);
    let max_single_size_strided = if coalesced > complex_bytes {
        allowed_shared_memory / coalesced
    } else {
        max_sequence_len_shared
    };
    let aim_threads = 128usize;
    let max_batch_coalesced = (policy.coalesced_memory_bytes / complex_bytes).max(1);
    let mut grouped_batch = if max_single_size_strided / fft_len > 1 {
        (max_single_size_strided / fft_len)
            .checked_mul(max_batch_coalesced)
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "Four-step user grouped batch",
            })?
    } else {
        max_batch_coalesced
    };
    // In the upstream user-grouped branch Direct-Rader pressure is applied to the
    // pre-clipped automatic groupedBatch before axis0 consumes the user override and
    // before higher uploads perform their aimThreads scale. A large Direct parent can
    // therefore reset a seemingly valid wide group all the way to maxBatchCoalesced.
    let direct_threads = if let Some(state) = rader_state {
        axis0_direct_rader_threads_from_split_state(
            fft_len,
            state,
            max_batch_coalesced,
            device.max_threads_per_block,
        )?
    } else if direct_rader_prime.is_some() {
        Some(threads_per_transform)
    } else {
        None
    };
    if let Some(direct_threads) = direct_threads
        && grouped_batch
            .checked_mul(direct_threads)
            .is_some_and(|threads| threads > device.max_threads_per_block)
    {
        grouped_batch = max_batch_coalesced;
    }

    let block = if axis_upload_id == 0 {
        let axis_block_0 = threads_per_transform
            .min(device.max_threads_per_block)
            .min(device.max_workgroup_size[0]);
        if axis_block_0 == 0 {
            return Ok(None);
        }
        let mut axis_block_1 = grouped_batch
            .min(grouped_batch_override)
            .min(transform_count)
            .min(device.max_workgroup_size[1])
            .max(1);
        if axis_block_0
            .checked_mul(axis_block_1)
            .is_some_and(|threads| threads > device.max_threads_per_block)
        {
            axis_block_1 = (device.max_threads_per_block / axis_block_0).max(1);
        }
        while axis_block_1 > 1
            && axis_block_1
                .checked_mul(fft_len)
                .is_some_and(|elements| elements > max_sequence_len_shared)
        {
            axis_block_1 /= 2;
        }
        let axis_swapped = !perform_zero_padding
            && (fft_len.is_multiple_of(2) || axis_block_0 < device.shared_banks.max(1) / 4)
            && axis_block_1 > 1
            && axis_block_1
                .checked_mul(fft_len)
                .is_some_and(|elements| elements < max_sequence_len_shared);
        if axis_swapped {
            StockhamAxisBlockSchedule {
                threads_per_transform,
                grouped_batch: axis_block_1,
                transforms_on_x: true,
                axis_swapped: true,
                local_size_x: axis_block_1,
                local_size_y: axis_block_0,
            }
        } else {
            StockhamAxisBlockSchedule {
                threads_per_transform,
                grouped_batch: axis_block_1,
                transforms_on_x: false,
                axis_swapped: false,
                local_size_x: axis_block_0,
                local_size_y: axis_block_1,
            }
        }
    } else {
        let axis_block_1 = threads_per_transform;
        if axis_block_1 > device.max_workgroup_size[1] {
            return Ok(None);
        }
        let scale = aim_threads / axis_block_1 / grouped_batch.max(1);
        if scale > 1
            && fft_len
                .checked_mul(grouped_batch)
                .and_then(|value| value.checked_mul(scale))
                .is_some_and(|elements| elements <= max_sequence_len_shared)
        {
            grouped_batch =
                grouped_batch
                    .checked_mul(scale)
                    .ok_or(VkFftError::ArithmeticOverflow {
                        operation: "user Four-step higher-upload grouped batch scale",
                    })?;
        }
        let mut axis_block_0 = stage_start_size.min(grouped_batch).max(1);
        if device.vendor == GpuVendor::Nvidia {
            while axis_block_0
                .checked_mul(axis_block_1)
                .is_some_and(|threads| threads >= 2 * aim_threads)
                && axis_block_0 > max_batch_coalesced
            {
                axis_block_0 /= 2;
                if axis_block_0 < max_batch_coalesced {
                    axis_block_0 = max_batch_coalesced;
                }
            }
        }
        axis_block_0 = axis_block_0.min(device.max_workgroup_size[0]).max(1);
        if axis_block_0
            .checked_mul(axis_block_1)
            .is_some_and(|threads| threads > device.max_threads_per_block)
        {
            for divisor in 1..=axis_block_0 {
                let candidate = axis_block_0 / divisor;
                if candidate > 0
                    && candidate
                        .checked_mul(axis_block_1)
                        .is_some_and(|threads| threads <= device.max_threads_per_block)
                {
                    axis_block_0 = candidate;
                    break;
                }
            }
        }
        if axis_block_0 <= 1 {
            return Ok(None);
        }
        StockhamAxisBlockSchedule {
            threads_per_transform,
            grouped_batch: axis_block_0,
            transforms_on_x: true,
            axis_swapped: false,
            local_size_x: axis_block_0,
            local_size_y: axis_block_1,
        }
    };
    if block.validate(transform_count, device).is_err() {
        return Ok(None);
    }
    Ok(Some(block))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
struct Axis0FourStepPhysicalContext {
    use_bluestein_fft: bool,
    direct_rader_prime: Option<usize>,
    precision: Option<Precision>,
}

/// Default fixed-upstream axis-0 Four-step block splitter when a caller already
/// knows the exact per-transform FFT lane floor. This is the physical grouping half
/// of `VkFFTSplitAxisBlock`: Rader/recursive components can reuse it after their own
/// register/container scorer has established `threads_per_transform`.
pub(crate) fn plan_gpu_axis0_four_step_default_block_from_shape(
    upload_count: usize,
    fft_len: usize,
    threads_per_transform: usize,
    request: FourStepAxisBlockRequest,
    complex_bytes: usize,
    device: DeviceProfile,
) -> Result<Option<StockhamAxisBlockSchedule>> {
    plan_gpu_axis0_four_step_default_block_from_shape_impl(
        upload_count,
        fft_len,
        threads_per_transform,
        request,
        complex_bytes,
        device,
        None,
        Axis0FourStepPhysicalContext::default(),
    )
}

pub(crate) fn plan_gpu_axis0_four_step_default_block_from_shape_for_precision(
    upload_count: usize,
    fft_len: usize,
    threads_per_transform: usize,
    request: FourStepAxisBlockRequest,
    precision: Precision,
    complex_bytes: usize,
    device: DeviceProfile,
) -> Result<Option<StockhamAxisBlockSchedule>> {
    if precision.compute_complex_bytes() != complex_bytes {
        return Ok(None);
    }
    plan_gpu_axis0_four_step_default_block_from_shape_impl(
        upload_count,
        fft_len,
        threads_per_transform,
        request,
        complex_bytes,
        device,
        None,
        Axis0FourStepPhysicalContext {
            precision: Some(precision),
            ..Axis0FourStepPhysicalContext::default()
        },
    )
}

/// Direct-Rader variant of the fixed-upstream Four-step physical shape scorer.
/// `VkFFTSplitAxisBlock` keeps `useRaderMult` live after the caller lane floor is
/// known: the direct-prime shared-memory reserve and the groupedBatch x Rader-thread
/// cap must therefore still participate in the final physical block choice.
pub(crate) fn plan_gpu_axis0_direct_rader_four_step_default_block_from_shape_for_precision(
    upload_count: usize,
    fft_len: usize,
    threads_per_transform: usize,
    direct_rader_prime: usize,
    request: FourStepAxisBlockRequest,
    precision: Precision,
    complex_bytes: usize,
    device: DeviceProfile,
) -> Result<Option<StockhamAxisBlockSchedule>> {
    if direct_rader_prime < 3
        || !fft_len.is_multiple_of(direct_rader_prime)
        || precision.compute_complex_bytes() != complex_bytes
    {
        return Ok(None);
    }
    plan_gpu_axis0_four_step_default_block_from_shape_impl(
        upload_count,
        fft_len,
        threads_per_transform,
        request,
        complex_bytes,
        device,
        None,
        Axis0FourStepPhysicalContext {
            precision: Some(precision),
            direct_rader_prime: Some(direct_rader_prime),
            ..Axis0FourStepPhysicalContext::default()
        },
    )
}

pub(crate) fn plan_gpu_axis0_rader_four_step_default_block_from_split_state(
    upload_count: usize,
    fft_len: usize,
    state: &Axis0RaderSplitState,
    request: FourStepAxisBlockRequest,
    complex_bytes: usize,
    device: DeviceProfile,
) -> Result<Option<StockhamAxisBlockSchedule>> {
    plan_gpu_axis0_four_step_default_block_from_shape_impl(
        upload_count,
        fft_len,
        state.base_axis_threads,
        request,
        complex_bytes,
        device,
        Some(state),
        Axis0FourStepPhysicalContext::default(),
    )
}

pub(crate) fn plan_gpu_axis0_bluestein_four_step_default_block_from_shape(
    upload_count: usize,
    fft_len: usize,
    threads_per_transform: usize,
    request: FourStepAxisBlockRequest,
    complex_bytes: usize,
    device: DeviceProfile,
) -> Result<Option<StockhamAxisBlockSchedule>> {
    plan_gpu_axis0_four_step_default_block_from_shape_impl(
        upload_count,
        fft_len,
        threads_per_transform,
        request,
        complex_bytes,
        device,
        None,
        Axis0FourStepPhysicalContext {
            use_bluestein_fft: true,
            ..Axis0FourStepPhysicalContext::default()
        },
    )
}

pub(crate) fn plan_gpu_axis0_bluestein_rader_four_step_default_block_from_split_state(
    upload_count: usize,
    fft_len: usize,
    state: &Axis0RaderSplitState,
    request: FourStepAxisBlockRequest,
    complex_bytes: usize,
    device: DeviceProfile,
) -> Result<Option<StockhamAxisBlockSchedule>> {
    plan_gpu_axis0_four_step_default_block_from_shape_impl(
        upload_count,
        fft_len,
        state.base_axis_threads,
        request,
        complex_bytes,
        device,
        Some(state),
        Axis0FourStepPhysicalContext {
            use_bluestein_fft: true,
            ..Axis0FourStepPhysicalContext::default()
        },
    )
}

fn plan_gpu_axis0_four_step_default_block_from_shape_impl(
    upload_count: usize,
    fft_len: usize,
    threads_per_transform: usize,
    request: FourStepAxisBlockRequest,
    complex_bytes: usize,
    device: DeviceProfile,
    rader_state: Option<&Axis0RaderSplitState>,
    physical_context: Axis0FourStepPhysicalContext,
) -> Result<Option<StockhamAxisBlockSchedule>> {
    let FourStepAxisBlockRequest {
        axis_upload_id,
        stage_start_size,
        transform_count,
        outer_batch_count,
        perform_zero_padding,
        grouped_batch_override,
    } = request;
    if !matches!(upload_count, 2 | 3)
        || axis_upload_id >= upload_count
        || fft_len == 0
        || threads_per_transform == 0
        || stage_start_size == 0
        || transform_count == 0
        || outer_batch_count == 0
        || !transform_count.is_multiple_of(outer_batch_count)
        || grouped_batch_override.is_some()
        || device.max_threads_per_block == 0
        || device.max_workgroup_size[0] == 0
        || device.max_workgroup_size[1] == 0
    {
        return Ok(None);
    }
    let precision = if let Some(precision) = physical_context.precision {
        if precision.compute_complex_bytes() != complex_bytes {
            return Ok(None);
        }
        precision
    } else {
        match complex_bytes {
            8 => Precision::F32,
            16 => Precision::F64,
            32 => Precision::DoubleDouble,
            _ => return Ok(None),
        }
    };
    if threads_per_transform > device.max_threads_per_block {
        return Ok(None);
    }
    let policy = plan_gpu_scheduler_policy(precision, device)?;
    let max_direct_prime = rader_state
        .and_then(|state| {
            state
                .direct_prime_multiplicities
                .iter()
                .map(|(prime, _)| *prime)
                .max()
        })
        .or(physical_context.direct_rader_prime);
    let direct_reserve = if let Some(max_direct_prime) = max_direct_prime {
        max_direct_prime
            .saturating_sub(1)
            .checked_mul(complex_bytes)
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "Four-step Rader direct shared-memory reserve",
            })?
    } else {
        0
    };
    let Some(allowed_shared_memory) = device.shared_memory_bytes.checked_sub(direct_reserve) else {
        return Ok(None);
    };
    let Some(allowed_shared_memory_pow2) =
        device.shared_memory_pow2_bytes.checked_sub(direct_reserve)
    else {
        return Ok(None);
    };
    if allowed_shared_memory < complex_bytes || allowed_shared_memory_pow2 < complex_bytes {
        return Ok(None);
    }
    let max_sequence_len_shared = allowed_shared_memory / complex_bytes;
    let max_sequence_len_shared_pow2 = allowed_shared_memory_pow2 / complex_bytes;
    let coalesced = policy.coalesced_memory_bytes.max(complex_bytes);
    let max_single_size_strided = if coalesced > complex_bytes {
        allowed_shared_memory / coalesced
    } else {
        max_sequence_len_shared
    };
    let aim_threads = 128usize;
    let warp_size = policy.subgroup_width.max(1);
    let mut max_batch_coalesced = (policy.coalesced_memory_bytes / complex_bytes).max(1);
    let mut grouped_batch = if physical_context.use_bluestein_fft && axis_upload_id == 0 {
        (max_sequence_len_shared / fft_len).max(max_batch_coalesced)
    } else if max_single_size_strided / fft_len > 1 {
        (max_single_size_strided / fft_len)
            .checked_mul(max_batch_coalesced)
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "Four-step shape initial grouped batch",
            })?
    } else {
        max_batch_coalesced
    };

    if device.vendor == GpuVendor::Nvidia {
        let widen =
            upload_count == 3 || (upload_count == 2 && (axis_upload_id > 0 || fft_len <= 512));
        if widen {
            for bytes in [64usize, 128] {
                let candidate = bytes / complex_bytes;
                if candidate > 0
                    && fft_len
                        .checked_mul(candidate)
                        .is_some_and(|elements| elements <= max_sequence_len_shared)
                {
                    grouped_batch = candidate;
                    max_batch_coalesced = candidate;
                }
            }
        }
    } else if axis_upload_id == 0 {
        if upload_count == 2
            && fft_len
                .checked_mul(max_batch_coalesced)
                .is_some_and(|elements| elements <= max_sequence_len_shared)
        {
            grouped_batch = ceil_div(grouped_batch, 2)?;
        }
        if upload_count == 3 && fft_len < max_sequence_len_shared / (2 * complex_bytes) {
            grouped_batch = ceil_div(grouped_batch, 2)?;
        }
    }

    if grouped_batch < max_batch_coalesced {
        grouped_batch = max_batch_coalesced;
    }
    grouped_batch = (grouped_batch / max_batch_coalesced) * max_batch_coalesced;
    if fft_len > max_single_size_strided {
        grouped_batch = (max_sequence_len_shared / fft_len).max(1);
    }
    if grouped_batch > warp_size {
        grouped_batch = (grouped_batch / warp_size) * warp_size;
    }
    if grouped_batch > 2 * max_batch_coalesced {
        grouped_batch = (grouped_batch / (2 * max_batch_coalesced)) * (2 * max_batch_coalesced);
    }
    if grouped_batch > 4 * max_batch_coalesced {
        grouped_batch = (grouped_batch / (4 * max_batch_coalesced)) * (4 * max_batch_coalesced);
    }
    grouped_batch = grouped_batch.max(1);

    let physical_thread_limit = device
        .max_threads_per_block
        .min(device.max_workgroup_size[0]);
    let threads_per_transform = if let Some(state) = rader_state {
        let Some(threads) = axis0_rader_threads_from_split_state(
            fft_len,
            state,
            max_batch_coalesced,
            physical_thread_limit,
        )?
        else {
            return Ok(None);
        };
        threads
    } else {
        threads_per_transform
    };
    if threads_per_transform == 0 || threads_per_transform > physical_thread_limit {
        return Ok(None);
    }
    let has_direct_rader = rader_state
        .is_some_and(|state| !state.direct_prime_multiplicities.is_empty())
        || physical_context.direct_rader_prime.is_some();
    if has_direct_rader
        && threads_per_transform
            .checked_mul(grouped_batch)
            .is_some_and(|threads| threads > device.max_threads_per_block)
    {
        grouped_batch = max_batch_coalesced.max(1);
    }

    let reduce_dimension = |dimension: usize, other: usize| -> usize {
        if dimension.saturating_mul(other) <= device.max_threads_per_block {
            return dimension;
        }
        for divisor in 1..=dimension {
            let candidate = dimension / divisor;
            if candidate > 0 && candidate.saturating_mul(other) <= device.max_threads_per_block {
                return candidate;
            }
        }
        1
    };

    let transforms_per_outer_batch = transform_count / outer_batch_count;
    let block = if axis_upload_id == 0 {
        let axis_block_0 = threads_per_transform
            .min(device.max_threads_per_block)
            .min(device.max_workgroup_size[0]);
        if axis_block_0 == 0 {
            return Ok(None);
        }
        let mut axis_block_1 = if physical_context.use_bluestein_fft {
            let estimate_batch = if axis_block_0 / warp_size == 1
                && axis_block_0.saturating_mul(2) < warp_size.saturating_mul(3)
            {
                aim_threads / warp_size
            } else {
                aim_threads / axis_block_0
            }
            .max(1);
            if axis_block_0 < aim_threads && (axis_block_0 < warp_size || rader_state.is_some()) {
                estimate_batch
            } else {
                1
            }
        } else {
            grouped_batch
        };
        let current = axis_block_1;
        if current > 0 && !transforms_per_outer_batch.is_multiple_of(current) {
            for candidate in current..current.saturating_mul(2) {
                if candidate
                    .checked_mul(fft_len)
                    .and_then(|value| value.checked_mul(complex_bytes))
                    .is_some_and(|bytes| bytes <= allowed_shared_memory)
                {
                    axis_block_1 = candidate;
                    break;
                }
            }
        }
        let can_round_for_swap = !perform_zero_padding
            && !physical_context.use_bluestein_fft
            && (fft_len.is_multiple_of(2) || axis_block_0 < device.shared_banks.max(1) / 4)
            && axis_block_1 > 1
            && axis_block_1
                .checked_mul(fft_len)
                .is_some_and(|elements| elements < max_sequence_len_shared_pow2);
        if can_round_for_swap {
            axis_block_1 = axis_block_1.next_power_of_two();
        }
        axis_block_1 = axis_block_1.min(transforms_per_outer_batch).max(1);
        if device.vendor == GpuVendor::Nvidia {
            while axis_block_1
                .checked_mul(axis_block_0)
                .is_some_and(|threads| threads >= 2 * aim_threads)
                && axis_block_1 > max_batch_coalesced
            {
                axis_block_1 = (axis_block_1 / 2).max(max_batch_coalesced);
            }
        }
        axis_block_1 = axis_block_1.min(device.max_workgroup_size[1]).max(1);
        axis_block_1 = reduce_dimension(axis_block_1, axis_block_0).max(1);
        while axis_block_1 > 1
            && axis_block_1
                .checked_mul(fft_len)
                .is_some_and(|elements| elements > max_sequence_len_shared)
        {
            axis_block_1 /= 2;
        }
        if axis_block_1
            .checked_mul(fft_len)
            .is_none_or(|elements| elements > max_sequence_len_shared)
        {
            return Ok(None);
        }
        let axis_swapped = !perform_zero_padding
            && !physical_context.use_bluestein_fft
            && (fft_len.is_multiple_of(2) || axis_block_0 < device.shared_banks.max(1) / 4)
            && axis_block_1 > 1
            && axis_block_1
                .checked_mul(fft_len)
                .is_some_and(|elements| elements < max_sequence_len_shared);
        if axis_swapped {
            StockhamAxisBlockSchedule {
                threads_per_transform,
                grouped_batch: axis_block_1,
                transforms_on_x: true,
                axis_swapped: true,
                local_size_x: axis_block_1,
                local_size_y: axis_block_0,
            }
        } else {
            StockhamAxisBlockSchedule {
                threads_per_transform,
                grouped_batch: axis_block_1,
                transforms_on_x: false,
                axis_swapped: false,
                local_size_x: axis_block_0,
                local_size_y: axis_block_1,
            }
        }
    } else {
        let axis_block_1 = threads_per_transform;
        if axis_block_1 > device.max_workgroup_size[1] {
            return Ok(None);
        }
        let scale = aim_threads / axis_block_1 / grouped_batch.max(1);
        if scale > 1
            && fft_len
                .checked_mul(grouped_batch)
                .and_then(|value| value.checked_mul(scale))
                .is_some_and(|elements| elements <= max_sequence_len_shared)
        {
            grouped_batch =
                grouped_batch
                    .checked_mul(scale)
                    .ok_or(VkFftError::ArithmeticOverflow {
                        operation: "Four-step shape higher-upload grouped batch scale",
                    })?;
        }
        let mut axis_block_0 = stage_start_size.min(grouped_batch).max(1);
        if device.vendor == GpuVendor::Nvidia {
            while axis_block_0
                .checked_mul(axis_block_1)
                .is_some_and(|threads| threads >= 2 * aim_threads)
                && axis_block_0 > max_batch_coalesced
            {
                axis_block_0 = (axis_block_0 / 2).max(max_batch_coalesced);
            }
        }
        axis_block_0 = axis_block_0.min(device.max_workgroup_size[0]).max(1);
        axis_block_0 = reduce_dimension(axis_block_0, axis_block_1).max(1);
        StockhamAxisBlockSchedule {
            threads_per_transform,
            grouped_batch: axis_block_0,
            transforms_on_x: true,
            axis_swapped: false,
            local_size_x: axis_block_0,
            local_size_y: axis_block_1,
        }
    };
    if (rader_state.is_none() && block.grouped_batch <= 1)
        || block.validate(transform_count, device).is_err()
    {
        return Ok(None);
    }
    Ok(Some(block))
}

/// Port the default axis-0 `reorderFourStep` block splitter for one concrete
/// upload. This is the non-Rader, no-zero-padding, no-user-groupedBatch branch of
/// fixed upstream `VkFFTSplitAxisBlock`. Upload 0 starts as threads-X/batches-Y and
/// may set upstream `axisSwapped`; higher uploads natively use batches-X/threads-Y
/// without setting `axisSwapped`.
pub fn plan_gpu_axis0_four_step_upload_block(
    upload: &StockhamUploadSchedule,
    axis_upload_id: usize,
    stage_start_size: usize,
    transform_count: usize,
    outer_batch_count: usize,
    complex_bytes: usize,
    device: DeviceProfile,
) -> Result<Option<StockhamAxisBlockSchedule>> {
    plan_gpu_axis0_four_step_upload_block_with_grouped_batch(
        upload,
        FourStepAxisBlockRequest {
            axis_upload_id,
            stage_start_size,
            transform_count,
            outer_batch_count,
            perform_zero_padding: false,
            grouped_batch_override: None,
        },
        complex_bytes,
        device,
    )
}

/// User-override/zero-padding-aware Four-step axis-0 splitter. When
/// `grouped_batch_override` is set, all uploads follow upstream's early user branch:
/// upload 0 is capped by the requested value, while higher uploads intentionally
/// skip the default NVIDIA 64/128-byte widening and retain `axisSwapped == 0`.
pub fn plan_gpu_axis0_four_step_upload_block_with_grouped_batch(
    upload: &StockhamUploadSchedule,
    request: FourStepAxisBlockRequest,
    complex_bytes: usize,
    device: DeviceProfile,
) -> Result<Option<StockhamAxisBlockSchedule>> {
    let precision = match complex_bytes {
        8 => Precision::F32,
        16 => Precision::F64,
        _ => return Ok(None),
    };
    plan_gpu_axis0_four_step_upload_block_with_grouped_batch_for_precision(
        upload,
        request,
        precision,
        complex_bytes,
        device,
    )
}

pub(crate) fn plan_gpu_axis0_four_step_upload_block_with_grouped_batch_for_precision(
    upload: &StockhamUploadSchedule,
    request: FourStepAxisBlockRequest,
    precision: Precision,
    complex_bytes: usize,
    device: DeviceProfile,
) -> Result<Option<StockhamAxisBlockSchedule>> {
    if precision.compute_complex_bytes() != complex_bytes {
        return Ok(None);
    }
    let FourStepAxisBlockRequest {
        axis_upload_id,
        stage_start_size,
        transform_count,
        outer_batch_count,
        perform_zero_padding,
        grouped_batch_override,
    } = request;
    upload.validate()?;
    if !matches!(upload.upload_count, 2 | 3)
        || upload.register_boost != 1
        || axis_upload_id >= upload.upload_count
        || stage_start_size == 0
        || transform_count == 0
        || outer_batch_count == 0
        || !transform_count.is_multiple_of(outer_batch_count)
        || device.max_threads_per_block == 0
        || device.max_workgroup_size[0] == 0
        || device.max_workgroup_size[1] == 0
    {
        return Ok(None);
    }
    let policy = plan_gpu_scheduler_policy(precision, device)?;
    let radix = &upload.radix_schedules[axis_upload_id];
    let fft_len = upload.axis_split[axis_upload_id];
    if radix.fft_len != fft_len || radix.rhs_transform_count != transform_count {
        return Ok(None);
    }
    let threads_per_transform =
        (ceil_div(fft_len, radix.min_registers_per_thread)? / upload.register_boost).max(1);
    if threads_per_transform > device.max_threads_per_block {
        return Ok(None);
    }

    let allowed_shared_memory = device.shared_memory_bytes;
    let allowed_shared_memory_pow2 = device.shared_memory_pow2_bytes;
    if allowed_shared_memory < complex_bytes || allowed_shared_memory_pow2 < complex_bytes {
        return Ok(None);
    }
    let max_sequence_len_shared = allowed_shared_memory / complex_bytes;
    let max_sequence_len_shared_pow2 = allowed_shared_memory_pow2 / complex_bytes;
    let coalesced = policy.coalesced_memory_bytes.max(complex_bytes);
    let max_single_size_strided = if coalesced > complex_bytes {
        allowed_shared_memory / coalesced
    } else {
        max_sequence_len_shared
    };
    let aim_threads = 128usize;
    let warp_size = policy.subgroup_width.max(1);
    let mut max_batch_coalesced = (policy.coalesced_memory_bytes / complex_bytes).max(1);
    let mut grouped_batch = if max_single_size_strided / fft_len > 1 {
        (max_single_size_strided / fft_len)
            .checked_mul(max_batch_coalesced)
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "Four-step initial grouped batch",
            })?
    } else {
        max_batch_coalesced
    };

    if grouped_batch_override.is_some() {
        return plan_gpu_axis0_four_step_grouped_block_from_shape_for_precision(
            upload.upload_count,
            fft_len,
            threads_per_transform,
            FourStepAxisBlockRequest {
                axis_upload_id,
                stage_start_size,
                transform_count,
                outer_batch_count,
                perform_zero_padding,
                grouped_batch_override,
            },
            precision,
            complex_bytes,
            device,
        );
    }

    if device.vendor == GpuVendor::Nvidia {
        let widen = upload.upload_count == 3
            || (upload.upload_count == 2 && (axis_upload_id > 0 || fft_len <= 512));
        if widen {
            for bytes in [64usize, 128] {
                let candidate = bytes / complex_bytes;
                if candidate > 0
                    && fft_len
                        .checked_mul(candidate)
                        .is_some_and(|elements| elements <= max_sequence_len_shared)
                {
                    grouped_batch = candidate;
                    max_batch_coalesced = candidate;
                }
            }
        }
    } else if axis_upload_id == 0 {
        if upload.upload_count == 2
            && fft_len
                .checked_mul(max_batch_coalesced)
                .is_some_and(|elements| elements <= max_sequence_len_shared)
        {
            grouped_batch = ceil_div(grouped_batch, 2)?;
        }
        if upload.upload_count == 3 && fft_len < max_sequence_len_shared / (2 * complex_bytes) {
            grouped_batch = ceil_div(grouped_batch, 2)?;
        }
    }

    if grouped_batch < max_batch_coalesced {
        grouped_batch = max_batch_coalesced;
    }
    grouped_batch = (grouped_batch / max_batch_coalesced) * max_batch_coalesced;
    if fft_len > max_single_size_strided {
        grouped_batch = (max_sequence_len_shared / fft_len).max(1);
    }
    if grouped_batch > warp_size {
        grouped_batch = (grouped_batch / warp_size) * warp_size;
    }
    if grouped_batch > 2 * max_batch_coalesced {
        grouped_batch = (grouped_batch / (2 * max_batch_coalesced)) * (2 * max_batch_coalesced);
    }
    if grouped_batch > 4 * max_batch_coalesced {
        grouped_batch = (grouped_batch / (4 * max_batch_coalesced)) * (4 * max_batch_coalesced);
    }
    grouped_batch = grouped_batch.max(1);

    let reduce_dimension = |dimension: usize, other: usize| -> usize {
        if dimension.saturating_mul(other) <= device.max_threads_per_block {
            return dimension;
        }
        for divisor in 1..=dimension {
            let candidate = dimension / divisor;
            if candidate > 0 && candidate.saturating_mul(other) <= device.max_threads_per_block {
                return candidate;
            }
        }
        1
    };

    let transforms_per_outer_batch = transform_count / outer_batch_count;
    let (grouped_batch, axis_swapped, transforms_on_x, local_size_x, local_size_y) =
        if axis_upload_id == 0 {
            let axis_block_0 = threads_per_transform
                .min(device.max_threads_per_block)
                .min(device.max_workgroup_size[0]);
            if axis_block_0 == 0 {
                return Ok(None);
            }
            let mut axis_block_1 = grouped_batch;

            let current = axis_block_1;
            if current > 0 && !transforms_per_outer_batch.is_multiple_of(current) {
                for candidate in current..current.saturating_mul(2) {
                    if candidate
                        .checked_mul(fft_len)
                        .and_then(|value| value.checked_mul(complex_bytes))
                        .is_some_and(|bytes| bytes <= allowed_shared_memory)
                    {
                        axis_block_1 = candidate;
                        break;
                    }
                }
            }
            let can_round_for_swap = !perform_zero_padding
                && (fft_len.is_multiple_of(2) || axis_block_0 < device.shared_banks.max(1) / 4)
                && axis_block_1 > 1
                && axis_block_1
                    .checked_mul(fft_len)
                    .is_some_and(|elements| elements < max_sequence_len_shared_pow2);
            if can_round_for_swap {
                axis_block_1 = axis_block_1.next_power_of_two();
            }
            axis_block_1 = axis_block_1.min(transforms_per_outer_batch).max(1);
            if device.vendor == GpuVendor::Nvidia {
                while axis_block_1
                    .checked_mul(axis_block_0)
                    .is_some_and(|threads| threads >= 2 * aim_threads)
                    && axis_block_1 > max_batch_coalesced
                {
                    axis_block_1 = (axis_block_1 / 2).max(max_batch_coalesced);
                }
            }
            axis_block_1 = axis_block_1.min(device.max_workgroup_size[1]).max(1);
            axis_block_1 = reduce_dimension(axis_block_1, axis_block_0).max(1);
            while axis_block_1 > 1
                && axis_block_1
                    .checked_mul(fft_len / upload.register_boost)
                    .is_some_and(|elements| elements > max_sequence_len_shared)
            {
                axis_block_1 /= 2;
            }
            if axis_block_1
                .checked_mul(fft_len / upload.register_boost)
                .is_none_or(|elements| elements > max_sequence_len_shared)
            {
                return Ok(None);
            }
            let final_swap = !perform_zero_padding
                && (fft_len.is_multiple_of(2) || axis_block_0 < device.shared_banks.max(1) / 4)
                && axis_block_1 > 1
                && axis_block_1
                    .checked_mul(fft_len)
                    .is_some_and(|elements| elements < max_sequence_len_shared);
            if final_swap {
                (axis_block_1, true, true, axis_block_1, axis_block_0)
            } else {
                (axis_block_1, false, false, axis_block_0, axis_block_1)
            }
        } else {
            let axis_block_1 = threads_per_transform;
            if axis_block_1 > device.max_workgroup_size[1] {
                return Ok(None);
            }
            let scale = aim_threads / axis_block_1 / grouped_batch.max(1);
            if scale > 1
                && fft_len
                    .checked_mul(grouped_batch)
                    .and_then(|value| value.checked_mul(scale))
                    .is_some_and(|elements| elements <= max_sequence_len_shared)
            {
                grouped_batch =
                    grouped_batch
                        .checked_mul(scale)
                        .ok_or(VkFftError::ArithmeticOverflow {
                            operation: "Four-step higher-upload grouped batch scale",
                        })?;
            }
            let mut axis_block_0 = stage_start_size.min(grouped_batch).max(1);
            if device.vendor == GpuVendor::Nvidia {
                while axis_block_0
                    .checked_mul(axis_block_1)
                    .is_some_and(|threads| threads >= 2 * aim_threads)
                    && axis_block_0 > max_batch_coalesced
                {
                    axis_block_0 = (axis_block_0 / 2).max(max_batch_coalesced);
                }
            }
            axis_block_0 = axis_block_0.min(device.max_workgroup_size[0]).max(1);
            axis_block_0 = reduce_dimension(axis_block_0, axis_block_1).max(1);
            (axis_block_0, false, true, axis_block_0, axis_block_1)
        };

    let block = StockhamAxisBlockSchedule {
        threads_per_transform,
        grouped_batch,
        transforms_on_x,
        axis_swapped,
        local_size_x,
        local_size_y,
    };
    if block.validate(transform_count, device).is_err() {
        return Ok(None);
    }
    Ok(Some(block))
}

/// Port the default NVIDIA/Vulkan power-of-two Stockham upload decision from
/// `vkFFT_Scheduler.h`.
///
/// Scope is intentionally explicit: Rader reservation and user launch overrides are not
/// modeled here. Axis class, upstream `performBandwidthBoost`, and the application-level
/// `performConvolution` capacity rule are carried by the internal typed context; the public
/// wrapper preserves ordinary-FFT defaults.
pub fn plan_gpu_power_of_two_stockham_uploads(
    sequence_len: usize,
    precision: Precision,
    device: DeviceProfile,
) -> Result<StockhamUploadSchedule> {
    plan_gpu_power_of_two_stockham_uploads_for_batches(sequence_len, 1, precision, device)
}

/// Compatibility wrapper preserving the original NVIDIA/Vulkan API contract.
pub fn plan_nvidia_vulkan_power_of_two_stockham_uploads(
    sequence_len: usize,
    precision: Precision,
    device: DeviceProfile,
) -> Result<StockhamUploadSchedule> {
    plan_nvidia_vulkan_power_of_two_stockham_uploads_for_batches(sequence_len, 1, precision, device)
}

/// Batch-aware form of the policy-driven power-of-two scheduler. The batch
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct StockhamUploadAxisContext {
    pub strided_axis: bool,
    pub bandwidth_boost: usize,
    pub use_bluestein_fft: bool,
    pub perform_convolution: bool,
}

fn upload_count_with_first_capacity(
    sequence_len: usize,
    first_capacity: usize,
    later_capacity: usize,
) -> Result<usize> {
    if first_capacity == 0 || later_capacity == 0 {
        return Err(VkFftError::InvalidKernelIr(
            "Stockham upload capacities must be non-zero",
        ));
    }
    let mut remaining = ceil_div(sequence_len, first_capacity)?;
    let mut uploads = 1usize;
    while remaining > 1 {
        remaining = ceil_div(remaining, later_capacity)?;
        uploads = uploads
            .checked_add(1)
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "Stockham upload count with first-axis capacity",
            })?;
    }
    Ok(uploads)
}

fn upstream_strided_bluestein_initial_upload_count(
    sequence_len: usize,
    max_single_strided: usize,
) -> Result<usize> {
    if max_single_strided == 0 {
        return Err(VkFftError::InvalidKernelIr(
            "strided Bluestein upload capacity must be non-zero",
        ));
    }
    let floor_ratio = sequence_len / max_single_strided;
    if floor_ratio <= 1 {
        return Ok(1);
    }
    1usize
        .checked_add(exponent_covering(floor_ratio, max_single_strided)?)
        .ok_or(VkFftError::ArithmeticOverflow {
            operation: "strided Bluestein upload count",
        })
}

fn apply_strided_bandwidth_boost(
    sequence_len: usize,
    current_upload_count: usize,
    max_single_non_strided: usize,
    max_single_strided: usize,
    max_sequence_len_shared: usize,
    used_shared_memory_bytes: usize,
    complex_bytes: usize,
    coalesced_memory_bytes: usize,
    reorder_four_step: bool,
    axis_context: StockhamUploadAxisContext,
) -> Result<(usize, usize)> {
    let effective_boost = if axis_context.strided_axis {
        if axis_context.bandwidth_boost > 0 {
            axis_context.bandwidth_boost
        } else if axis_context.use_bluestein_fft {
            1
        } else {
            0
        }
    } else {
        0
    };
    if effective_boost == 0 {
        return Ok((current_upload_count, max_single_strided));
    }

    let boosted_coalesced = coalesced_memory_bytes / effective_boost;
    let half_bandwidth_capacity = if boosted_coalesced > complex_bytes {
        used_shared_memory_bytes / boosted_coalesced
    } else {
        max_sequence_len_shared
    };
    if half_bandwidth_capacity == 0 {
        return Ok((current_upload_count, max_single_strided));
    }

    // Direct port of the fixed-upstream performBandwidthBoost pass-count probe.
    // The half-bandwidth capacity is checked first. Only when it still needs more
    // than one pass does Bluestein/no-reorder switch the first stage to the full
    // non-strided capacity before dividing the remainder by ordinary strided capacity.
    let initial_capacity = if axis_context.strided_axis {
        half_bandwidth_capacity
    } else {
        max_single_non_strided
    };
    let mut candidate_upload_count = 1usize;
    if ceil_div(sequence_len, initial_capacity)? > 1 {
        let first_capacity = if !reorder_four_step || axis_context.use_bluestein_fft {
            max_single_non_strided
        } else {
            half_bandwidth_capacity
        };
        let mut remaining = ceil_div(sequence_len, first_capacity)?;
        for _ in 0..5 {
            remaining = ceil_div(remaining, max_single_strided)?;
            candidate_upload_count =
                candidate_upload_count
                    .checked_add(1)
                    .ok_or(VkFftError::ArithmeticOverflow {
                        operation: "half-bandwidth Stockham upload count",
                    })?;
            if remaining == 1 {
                break;
            }
        }
    }
    if candidate_upload_count < current_upload_count {
        Ok((candidate_upload_count, half_bandwidth_capacity))
    } else {
        Ok((current_upload_count, max_single_strided))
    }
}

/// count contributes to VkFFT's `max_rhs / locAxisSplit[k]` occupancy estimate
/// used by `VkFFTGetRegistersPerThread` for each upload.
pub fn plan_gpu_power_of_two_stockham_uploads_for_batches(
    sequence_len: usize,
    batch_count: usize,
    precision: Precision,
    device: DeviceProfile,
) -> Result<StockhamUploadSchedule> {
    plan_gpu_power_of_two_stockham_uploads_for_batches_with_axis_context(
        sequence_len,
        batch_count,
        precision,
        device,
        StockhamUploadAxisContext::default(),
    )
}

pub(crate) fn plan_gpu_power_of_two_stockham_uploads_for_batches_with_axis_context(
    sequence_len: usize,
    batch_count: usize,
    precision: Precision,
    device: DeviceProfile,
    axis_context: StockhamUploadAxisContext,
) -> Result<StockhamUploadSchedule> {
    let tuning = plan_gpu_scheduler_policy(precision, device)?;
    if sequence_len == 0 || !sequence_len.is_power_of_two() {
        return Err(VkFftError::UnsupportedKernelPath(
            "NVIDIA Vulkan scheduler slice currently requires a power-of-two length",
        ));
    }
    if batch_count == 0 {
        return Err(VkFftError::ZeroBatchCount);
    }
    let complex_bytes = match precision {
        Precision::F32 | Precision::F16StorageF32Compute => 8usize,
        Precision::F64 | Precision::F64ComputeF32Storage => 16usize,
        other => {
            return Err(VkFftError::UnsupportedPrecision {
                backend: "NVIDIA Vulkan scheduler baseline",
                precision: precision_name(other),
            });
        }
    };
    let used_shared_memory_bytes = device.shared_memory_pow2_bytes;
    if used_shared_memory_bytes < complex_bytes {
        return Err(VkFftError::ResourceLimitExceeded {
            resource: "NVIDIA Vulkan scheduler shared memory",
            required: complex_bytes,
            available: used_shared_memory_bytes,
        });
    }
    let max_sequence_len_shared = used_shared_memory_bytes / complex_bytes;
    let max_sequence_len_strided = if tuning.coalesced_memory_bytes > complex_bytes {
        used_shared_memory_bytes / tuning.coalesced_memory_bytes
    } else {
        max_sequence_len_shared
    };
    if max_sequence_len_strided == 0 {
        return Err(VkFftError::ResourceLimitExceeded {
            resource: "NVIDIA Vulkan scheduler strided shared-memory capacity",
            required: 1,
            available: 0,
        });
    }

    // Pinned initializeVkFFT mutates these policy knobs before VkFFTScheduler runs:
    // Bluestein forces registerBoost=1, while performConvolution additionally disables
    // reorderFourStep and forces both ordinary/Four-step register boosts to one.
    let reorder_four_step = tuning.reorder_four_step && !axis_context.perform_convolution;
    let register_boost_limit = if axis_context.use_bluestein_fft || axis_context.perform_convolution
    {
        1
    } else {
        tuning.register_boost
    };
    let register_boost_four_step_limit = if axis_context.perform_convolution {
        1
    } else {
        tuning.register_boost_four_step
    };
    let mut register_boost = largest_square_divisor_boost(sequence_len, register_boost_limit);
    // Fixed upstream deliberately excludes registerBoost from the initial single-upload
    // capacity probe while performConvolution is active. The boost value itself is still
    // computed and may be reconsidered later after the upload count has been selected.
    let initial_non_strided = if axis_context.perform_convolution {
        max_sequence_len_shared
    } else {
        checked_mul(
            max_sequence_len_shared,
            register_boost,
            "GPU scheduler non-strided register boost",
        )?
    };
    let initial_strided = if axis_context.perform_convolution {
        max_sequence_len_strided
    } else {
        checked_mul(
            max_sequence_len_strided,
            register_boost,
            "GPU scheduler strided register boost",
        )?
    };
    let initial_capacity = if axis_context.strided_axis {
        initial_strided
    } else {
        initial_non_strided
    };
    let mut upload_count = 1usize;

    if ceil_div(sequence_len, initial_capacity)? > 1 {
        register_boost = largest_square_divisor_boost(sequence_len, register_boost_four_step_limit);
        let four_step_non_strided = if axis_context.perform_convolution {
            max_sequence_len_shared
        } else {
            checked_mul(
                max_sequence_len_shared,
                register_boost,
                "GPU four-step non-strided register boost",
            )?
        };
        let four_step_strided = if axis_context.perform_convolution {
            max_sequence_len_strided
        } else {
            checked_mul(
                max_sequence_len_strided,
                register_boost,
                "GPU four-step strided register boost",
            )?
        };
        upload_count = if !axis_context.strided_axis
            && (!reorder_four_step || axis_context.use_bluestein_fft)
        {
            upload_count_with_first_capacity(
                sequence_len,
                four_step_non_strided,
                four_step_strided,
            )?
        } else {
            exponent_covering(sequence_len, four_step_strided)?
        };
    }
    if axis_context.strided_axis && axis_context.use_bluestein_fft {
        let four_step_strided = checked_mul(
            max_sequence_len_strided,
            register_boost,
            "GPU strided Bluestein legacy pass capacity",
        )?;
        upload_count =
            upstream_strided_bluestein_initial_upload_count(sequence_len, four_step_strided)?;
    }

    let denominator = if !axis_context.strided_axis
        && (axis_context.use_bluestein_fft || !reorder_four_step || upload_count == 1)
    {
        let later = checked_pow(
            max_sequence_len_strided,
            upload_count.saturating_sub(1),
            "GPU scheduler unit-stride register-boost denominator",
        )?;
        checked_mul(
            later,
            max_sequence_len_shared,
            "GPU scheduler unit-stride register-boost denominator",
        )?
    } else {
        checked_pow(
            max_sequence_len_strided,
            upload_count,
            "GPU scheduler strided register-boost denominator",
        )?
    };
    register_boost = ceil_div(sequence_len, denominator)?;
    let required_boost = register_boost;
    let mut selected_boost = None;
    for candidate in required_boost..=register_boost_limit {
        let square = candidate
            .checked_mul(candidate)
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "NVIDIA scheduler register boost square",
            })?;
        if sequence_len.is_multiple_of(square) {
            selected_boost = Some(candidate);
            break;
        }
    }
    if let Some(selected) = selected_boost {
        register_boost = selected;
    }
    if selected_boost.is_none() && register_boost > 1 {
        register_boost = 1;
        upload_count = upload_count
            .checked_add(1)
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "NVIDIA scheduler upload count",
            })?;
    }
    let max_single_strided = checked_mul(
        max_sequence_len_strided,
        register_boost,
        "NVIDIA scheduler final strided register boost",
    )?;
    let max_single_non_strided = checked_mul(
        max_sequence_len_shared,
        register_boost,
        "GPU scheduler final non-strided register boost",
    )?;
    let (bandwidth_upload_count, max_single_strided_half_bandwidth) =
        apply_strided_bandwidth_boost(
            sequence_len,
            upload_count,
            max_single_non_strided,
            max_single_strided,
            max_sequence_len_shared,
            used_shared_memory_bytes,
            complex_bytes,
            tuning.coalesced_memory_bytes,
            reorder_four_step,
            axis_context,
        )?;
    upload_count = bandwidth_upload_count;

    if sequence_len >= tuning.swap_to_two_stage_four_step && upload_count < 3 {
        upload_count = 2;
    }
    if sequence_len >= tuning.swap_to_three_stage_four_step
        && tuning.swap_to_three_stage_four_step >= 65_536
    {
        upload_count = 3;
    }
    if upload_count > 3 {
        return Err(VkFftError::UnsupportedKernelPath(
            "NVIDIA Vulkan scheduler requires more than three Stockham uploads",
        ));
    }

    let unit_stride_first_stage =
        !axis_context.strided_axis && (!reorder_four_step || axis_context.use_bluestein_fft);
    let axis_split = match upload_count {
        1 => vec![sequence_len],
        2 => power_of_two_two_upload_split(
            sequence_len,
            max_sequence_len_shared,
            max_single_strided,
            max_single_strided_half_bandwidth,
            register_boost,
            unit_stride_first_stage,
            device,
        )?,
        3 => power_of_two_three_upload_split(
            sequence_len,
            used_shared_memory_bytes,
            max_sequence_len_shared,
            max_single_strided,
            max_single_strided_half_bandwidth,
            register_boost,
            unit_stride_first_stage,
            device,
        )?,
        _ => unreachable!("validated upload count is between one and three"),
    };
    let total_elements =
        sequence_len
            .checked_mul(batch_count)
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "NVIDIA scheduler total element count",
            })?;
    let radix_schedules = axis_split
        .iter()
        .map(|&axis_len| {
            plan_nvidia_vulkan_power_of_two_radix_registers(
                axis_len,
                total_elements / axis_len,
                register_boost,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    let schedule = StockhamUploadSchedule {
        sequence_len,
        batch_count,
        used_shared_memory_bytes,
        max_sequence_len_shared,
        max_sequence_len_strided,
        register_boost,
        upload_count,
        axis_split,
        radix_schedules,
    };
    schedule.validate()?;
    Ok(schedule)
}

/// Compatibility wrapper preserving the original NVIDIA/Vulkan API contract.
pub fn plan_nvidia_vulkan_power_of_two_stockham_uploads_for_batches(
    sequence_len: usize,
    batch_count: usize,
    precision: Precision,
    device: DeviceProfile,
) -> Result<StockhamUploadSchedule> {
    if device.backend != Backend::Vulkan || device.vendor != GpuVendor::Nvidia {
        return Err(VkFftError::UnsupportedKernelPath(
            "NVIDIA Vulkan scheduler slice requires an NVIDIA Vulkan DeviceProfile",
        ));
    }
    plan_gpu_power_of_two_stockham_uploads_for_batches(sequence_len, batch_count, precision, device)
}

/// Plan the policy-driven one-dimensional smooth Stockham upload layout.
/// Power-of-two inputs preserve the existing specialized scheduler; ordinary smooth
/// non-power-of-two inputs follow the generic VkFFT scheduler branch with
/// `registerBoostNonPow2 = 0`, `registerBoost4Step = 1`, and `reorderFourStep = 1`.
pub fn plan_gpu_smooth_stockham_uploads(
    sequence_len: usize,
    precision: Precision,
    device: DeviceProfile,
) -> Result<StockhamUploadSchedule> {
    plan_gpu_smooth_stockham_uploads_for_batches(sequence_len, 1, precision, device)
}

/// Compatibility wrapper preserving the original NVIDIA/Vulkan API contract.
pub fn plan_nvidia_vulkan_smooth_stockham_uploads(
    sequence_len: usize,
    precision: Precision,
    device: DeviceProfile,
) -> Result<StockhamUploadSchedule> {
    plan_nvidia_vulkan_smooth_stockham_uploads_for_batches(sequence_len, 1, precision, device)
}

/// Batch-aware policy-driven smooth Stockham upload scheduler.
pub fn plan_gpu_smooth_stockham_uploads_for_batches(
    sequence_len: usize,
    batch_count: usize,
    precision: Precision,
    device: DeviceProfile,
) -> Result<StockhamUploadSchedule> {
    plan_gpu_smooth_stockham_uploads_for_batches_with_axis_context(
        sequence_len,
        batch_count,
        precision,
        device,
        StockhamUploadAxisContext::default(),
    )
}

pub(crate) fn plan_gpu_smooth_stockham_uploads_for_batches_with_axis_context(
    sequence_len: usize,
    batch_count: usize,
    precision: Precision,
    device: DeviceProfile,
    axis_context: StockhamUploadAxisContext,
) -> Result<StockhamUploadSchedule> {
    if sequence_len.is_power_of_two() {
        return plan_gpu_power_of_two_stockham_uploads_for_batches_with_axis_context(
            sequence_len,
            batch_count,
            precision,
            device,
            axis_context,
        );
    }
    let tuning = plan_gpu_scheduler_policy(precision, device)?;
    if sequence_len < 3 {
        return Err(VkFftError::UnsupportedKernelPath(
            "non-power-of-two smooth upload scheduling requires an FFT length >= 3",
        ));
    }
    if batch_count == 0 {
        return Err(VkFftError::ZeroBatchCount);
    }
    let complex_bytes = match precision {
        Precision::F32 | Precision::F16StorageF32Compute => 8usize,
        Precision::F64 | Precision::F64ComputeF32Storage => 16usize,
        other => {
            return Err(VkFftError::UnsupportedPrecision {
                backend: "NVIDIA Vulkan smooth scheduler",
                precision: precision_name(other),
            });
        }
    };
    // Half-storage keeps F32 compute/register/shared values upstream; only external
    // storage and scheduler coalescing change. Reuse the F32 complex footprint while
    // `plan_gpu_scheduler_policy` supplies the precision-adjusted coalescing width.
    let _ = plan_gpu_small_mixed_radix_registers(sequence_len, batch_count)?;

    let used_shared_memory_bytes = device.shared_memory_bytes;
    if used_shared_memory_bytes < complex_bytes {
        return Err(VkFftError::ResourceLimitExceeded {
            resource: "NVIDIA Vulkan smooth scheduler shared memory",
            required: complex_bytes,
            available: used_shared_memory_bytes,
        });
    }
    let max_sequence_len_shared = used_shared_memory_bytes / complex_bytes;
    let max_sequence_len_strided = if tuning.coalesced_memory_bytes > complex_bytes {
        used_shared_memory_bytes / tuning.coalesced_memory_bytes
    } else {
        max_sequence_len_shared
    };
    if max_sequence_len_strided == 0 {
        return Err(VkFftError::ResourceLimitExceeded {
            resource: "NVIDIA Vulkan smooth scheduler strided shared-memory capacity",
            required: 1,
            available: 0,
        });
    }

    // Mirror the same pinned initializeVkFFT policy as the power-of-two path.
    let reorder_four_step = tuning.reorder_four_step && !axis_context.perform_convolution;
    let register_boost_limit = if axis_context.use_bluestein_fft || axis_context.perform_convolution
    {
        1
    } else {
        tuning.register_boost
    };
    let register_boost_four_step_limit = if axis_context.perform_convolution {
        1
    } else {
        tuning.register_boost_four_step
    };
    let register_boost_non_power_of_two =
        tuning.register_boost_non_power_of_two && !axis_context.perform_convolution;
    let mut register_boost = largest_square_divisor_boost(sequence_len, register_boost_limit);
    let initial_non_strided = if axis_context.perform_convolution {
        max_sequence_len_shared
    } else {
        checked_mul(
            max_sequence_len_shared,
            register_boost,
            "GPU smooth scheduler initial non-strided boost",
        )?
    };
    let initial_strided = if axis_context.perform_convolution {
        max_sequence_len_strided
    } else {
        checked_mul(
            max_sequence_len_strided,
            register_boost,
            "GPU smooth scheduler initial strided boost",
        )?
    };
    let initial_capacity = if axis_context.strided_axis {
        initial_strided
    } else {
        initial_non_strided
    };
    let mut upload_count = 1usize;
    if ceil_div(sequence_len, initial_capacity)? > 1 {
        register_boost = largest_square_divisor_boost(sequence_len, register_boost_four_step_limit);
        let four_step_non_strided = if axis_context.perform_convolution {
            max_sequence_len_shared
        } else {
            checked_mul(
                max_sequence_len_shared,
                register_boost,
                "GPU smooth scheduler four-step non-strided boost",
            )?
        };
        let four_step_strided = if axis_context.perform_convolution {
            max_sequence_len_strided
        } else {
            checked_mul(
                max_sequence_len_strided,
                register_boost,
                "GPU smooth scheduler four-step strided boost",
            )?
        };
        upload_count = if !axis_context.strided_axis
            && (!reorder_four_step || axis_context.use_bluestein_fft)
        {
            upload_count_with_first_capacity(
                sequence_len,
                four_step_non_strided,
                four_step_strided,
            )?
        } else {
            exponent_covering(sequence_len, four_step_strided)?
        };
    }
    if axis_context.strided_axis && axis_context.use_bluestein_fft {
        let four_step_strided = checked_mul(
            max_sequence_len_strided,
            register_boost,
            "GPU smooth strided Bluestein legacy pass capacity",
        )?;
        upload_count =
            upstream_strided_bluestein_initial_upload_count(sequence_len, four_step_strided)?;
    }

    let denominator = if !axis_context.strided_axis
        && (axis_context.use_bluestein_fft || !reorder_four_step || upload_count == 1)
    {
        let later = checked_pow(
            max_sequence_len_strided,
            upload_count.saturating_sub(1),
            "GPU smooth scheduler unit-stride register-boost denominator",
        )?;
        checked_mul(
            later,
            max_sequence_len_shared,
            "GPU smooth scheduler unit-stride register-boost denominator",
        )?
    } else {
        checked_pow(
            max_sequence_len_strided,
            upload_count,
            "GPU smooth scheduler strided register-boost denominator",
        )?
    };
    register_boost = ceil_div(sequence_len, denominator)?;
    let required_boost = register_boost;
    let mut selected_boost = None;
    for candidate in required_boost..=register_boost_limit {
        let square = candidate
            .checked_mul(candidate)
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "NVIDIA smooth scheduler register boost square",
            })?;
        if sequence_len.is_multiple_of(square) {
            selected_boost = Some(candidate);
            break;
        }
    }
    if let Some(selected) = selected_boost {
        register_boost = selected;
    }
    if (selected_boost.is_none() || !register_boost_non_power_of_two) && register_boost > 1 {
        register_boost = 1;
        upload_count = upload_count
            .checked_add(1)
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "NVIDIA smooth scheduler upload count",
            })?;
    }
    let max_single_strided = checked_mul(
        max_sequence_len_strided,
        register_boost,
        "NVIDIA smooth scheduler final strided register boost",
    )?;
    let max_single_non_strided = checked_mul(
        max_sequence_len_shared,
        register_boost,
        "GPU smooth scheduler final non-strided register boost",
    )?;
    let (bandwidth_upload_count, max_single_strided_half_bandwidth) =
        apply_strided_bandwidth_boost(
            sequence_len,
            upload_count,
            max_single_non_strided,
            max_single_strided,
            max_sequence_len_shared,
            used_shared_memory_bytes,
            complex_bytes,
            tuning.coalesced_memory_bytes,
            reorder_four_step,
            axis_context,
        )?;
    upload_count = bandwidth_upload_count;

    if sequence_len >= tuning.swap_to_two_stage_four_step && upload_count < 3 {
        upload_count = 2;
    }
    if sequence_len >= tuning.swap_to_three_stage_four_step
        && tuning.swap_to_three_stage_four_step >= 65_536
    {
        upload_count = 3;
    }
    if upload_count > 3 {
        return Err(VkFftError::UnsupportedKernelPath(
            "NVIDIA Vulkan smooth scheduler requires more than three Stockham uploads",
        ));
    }

    let unit_stride_first_stage =
        !axis_context.strided_axis && (!reorder_four_step || axis_context.use_bluestein_fft);
    let two_upload_quotient_limit = if unit_stride_first_stage {
        max_sequence_len_shared
    } else {
        max_single_strided_half_bandwidth
    };
    let mut axis_split = match upload_count {
        1 => vec![sequence_len],
        2 => generic_sqrt_divisor_split_with_limits(
            sequence_len,
            max_single_strided,
            two_upload_quotient_limit,
        )?
        .map(Vec::from)
        .ok_or(VkFftError::UnsupportedKernelPath(
            "GPU smooth scheduler could not find a two-upload divisor split",
        ))?,
        3 if unit_stride_first_stage => generic_three_upload_unit_stride_split(
            sequence_len,
            max_sequence_len_shared,
            max_single_strided,
        )?,
        3 => generic_three_upload_divisor_split_with_limits(
            sequence_len,
            max_single_strided,
            max_single_strided_half_bandwidth,
        )?,
        _ => unreachable!("validated upload count is between one and three"),
    };
    // Fixed upstream only applies the reorderFourStep 2/4/8 factor preference when
    // Bluestein is not active. A strided Bluestein convolution still uses the generic
    // divisor search, but its locAxisSplit order is preserved because that order feeds
    // Bluestein's upload/stageStartSize semantics directly.
    if !unit_stride_first_stage && reorder_four_step && !axis_context.use_bluestein_fft {
        prefer_four_step_first_factor(&mut axis_split);
    }

    let total_elements =
        sequence_len
            .checked_mul(batch_count)
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "NVIDIA smooth scheduler total element count",
            })?;
    let radix_schedules = axis_split
        .iter()
        .map(|&axis_len| {
            let rhs = total_elements / axis_len;
            if axis_len.is_power_of_two() {
                plan_gpu_power_of_two_radix_registers(axis_len, rhs, register_boost)
            } else {
                plan_gpu_small_mixed_radix_registers(axis_len, rhs)
            }
        })
        .collect::<Result<Vec<_>>>()?;
    let schedule = StockhamUploadSchedule {
        sequence_len,
        batch_count,
        used_shared_memory_bytes,
        max_sequence_len_shared,
        max_sequence_len_strided,
        register_boost,
        upload_count,
        axis_split,
        radix_schedules,
    };
    schedule.validate()?;
    Ok(schedule)
}

/// Compatibility wrapper preserving the original NVIDIA/Vulkan API contract.
pub fn plan_nvidia_vulkan_smooth_stockham_uploads_for_batches(
    sequence_len: usize,
    batch_count: usize,
    precision: Precision,
    device: DeviceProfile,
) -> Result<StockhamUploadSchedule> {
    if device.backend != Backend::Vulkan || device.vendor != GpuVendor::Nvidia {
        return Err(VkFftError::UnsupportedKernelPath(
            "NVIDIA Vulkan smooth scheduler requires an NVIDIA Vulkan DeviceProfile",
        ));
    }
    plan_gpu_smooth_stockham_uploads_for_batches(sequence_len, batch_count, precision, device)
}

/// Port the exact non-Rader `2^a * 3^b` branch of `VkFFTGetRegistersPerThread`
/// for NVIDIA/Vulkan boost-1 Stockham, followed by the existing
/// `VkFFTOptimizeRadixKernels` port. This remains the focused helper for that branch;
/// `plan_nvidia_vulkan_small_mixed_radix_registers` dispatches the complete smooth
/// non-power-of-two 2/3/5/7/11/13 decision tree.
pub fn plan_nvidia_vulkan_two_three_radix_registers(
    fft_len: usize,
    rhs_transform_count: usize,
) -> Result<RadixRegisterSchedule> {
    if fft_len < 6 || rhs_transform_count == 0 {
        return Err(VkFftError::UnsupportedKernelPath(
            "2x3 radix register scheduling requires a non-trivial FFT and RHS count",
        ));
    }
    let mut remaining = fft_len;
    let mut exponent2 = 0usize;
    while remaining.is_multiple_of(2) {
        remaining /= 2;
        exponent2 += 1;
    }
    let mut exponent3 = 0usize;
    while remaining.is_multiple_of(3) {
        remaining /= 3;
        exponent3 += 1;
    }
    if remaining != 1 || exponent2 == 0 || exponent3 == 0 {
        return Err(VkFftError::UnsupportedKernelPath(
            "first mixed-radix Stockham register slice covers exactly 2^a * 3^b",
        ));
    }

    // Exact no-5/no-7/no-11/no-13 branch from VkFFTGetRegistersPerThread.
    // Equal 2/3 multiplicities and the single-radix-2 case use six registers;
    // other exponent2 >= 2 cases use twelve for both primitive radices.
    let primitive_registers = if exponent2 == exponent3 || exponent2 == 1 {
        6usize
    } else {
        12usize
    };
    let mut registers = [0usize; VKFFT_RADIX_TABLE_LEN];
    registers[2] = primitive_registers;
    registers[3] = primitive_registers;
    registers[32] = if registers[2].is_multiple_of(32) {
        registers[2]
    } else {
        0
    };
    registers[16] = if registers[2].is_multiple_of(16) {
        registers[2]
    } else {
        0
    };
    registers[8] = if registers[2].is_multiple_of(8) {
        registers[2]
    } else {
        0
    };
    registers[4] = if registers[2].is_multiple_of(4) {
        registers[2]
    } else {
        0
    };
    if registers[2] >= 12 && registers[3] >= 12 {
        registers[12] = registers[2].min(registers[3]);
        if !registers[12].is_multiple_of(12) {
            registers[12] = 0;
        }
    }
    registers[6] = registers[2].min(registers[3]);
    registers[9] = if registers[3].is_multiple_of(9) {
        registers[3]
    } else {
        0
    };

    let mut multipliers = [0usize; VKFFT_RADIX_TABLE_LEN];
    multipliers[2] = exponent2;
    multipliers[3] = exponent3;
    let (max_non_power_of_two_radix, required_local_registers) =
        optimize_radix_kernels(&mut registers, &mut multipliers, 1);
    let mut stage_radices = Vec::new();
    for radix in (2..VKFFT_RADIX_TABLE_LEN).rev() {
        stage_radices.extend(core::iter::repeat_n(radix, multipliers[radix]));
    }
    let (registers_per_thread, min_registers_per_thread) =
        min_max_stage_registers(&stage_radices, &registers);
    if registers_per_thread == 0 || min_registers_per_thread == 0 {
        return Err(VkFftError::InvalidKernelIr(
            "2x3 radix scheduler produced no executable stages",
        ));
    }
    let schedule = RadixRegisterSchedule {
        fft_len,
        rhs_transform_count,
        register_boost: 1,
        registers_per_thread_per_radix: registers,
        stage_radix_multipliers: multipliers,
        stage_radices,
        register_boost_stage_radix: None,
        registers_per_thread,
        min_registers_per_thread,
        is_good_sequence: !(registers_per_thread > 16
            || registers_per_thread >= 2 * min_registers_per_thread),
        max_non_power_of_two_radix,
        required_local_registers,
    };
    schedule.validate()?;
    Ok(schedule)
}

/// Backend-neutral entry to the ordinary non-power-of-two register arithmetic from
/// `VkFFTGetRegistersPerThread`. Upstream shares this radix decision table across GPU
/// APIs; backend/vendor policy is applied by the surrounding upload/boost scheduler.
pub fn plan_gpu_small_mixed_radix_registers(
    fft_len: usize,
    rhs_transform_count: usize,
) -> Result<RadixRegisterSchedule> {
    plan_nvidia_vulkan_small_mixed_radix_registers(fft_len, rhs_transform_count)
}

/// Broader exact ordinary non-power-of-two slice from `VkFFTGetRegistersPerThread`.
///
/// Together with the dedicated `2^a * 3^b` helper above, this covers the complete
/// ordinary smooth non-power-of-two decision tree over VkFFT's radices
/// 2/3/5/7/11/13: pure odd powers, every two-prime family, and all higher-factor
/// combinations. Primitive register counts are direct ports of the matching upstream
/// branches, including their exponent-2 dependent choices. Composite
/// 4/6/8/9/10/12/14/15/16/32 entries are then derived exactly as upstream does before
/// the shared radix optimizer runs.
pub fn plan_nvidia_vulkan_small_mixed_radix_registers(
    fft_len: usize,
    rhs_transform_count: usize,
) -> Result<RadixRegisterSchedule> {
    if fft_len < 3 || rhs_transform_count == 0 {
        return Err(VkFftError::UnsupportedKernelPath(
            "non-power-of-two register scheduling requires an FFT length >= 3 and RHS count",
        ));
    }
    let primes = [2usize, 3, 5, 7, 11, 13];
    let mut remaining = fft_len;
    let mut exponents = [0usize; 6];
    for (slot, prime) in primes.into_iter().enumerate() {
        while remaining.is_multiple_of(prime) {
            remaining /= prime;
            exponents[slot] += 1;
        }
    }
    if remaining != 1 {
        return Err(VkFftError::UnsupportedKernelPath(
            "mixed-radix register slice requires factors from 2/3/5/7/11/13 only",
        ));
    }
    let active = exponents
        .iter()
        .enumerate()
        .filter_map(|(index, exponent)| (*exponent > 0).then_some(index))
        .collect::<Vec<_>>();
    if active.as_slice() == [0] || active.is_empty() {
        return Err(VkFftError::UnsupportedKernelPath(
            "power-of-two or empty factorization belongs to a different scheduler branch",
        ));
    }
    if active.as_slice() == [0, 1] {
        return plan_nvidia_vulkan_two_three_radix_registers(fft_len, rhs_transform_count);
    }

    let mut registers = [0usize; VKFFT_RADIX_TABLE_LEN];
    match active.as_slice() {
        [1] => {
            registers[3] = if exponents[1] == 1 { 3 } else { 9 };
        }
        [2] => registers[5] = 5,
        [3] => registers[7] = 7,
        [4] => registers[11] = 11,
        [5] => registers[13] = 13,
        [0, 2] => {
            registers[2] = if exponents[0] <= 2 { 10 } else { 8 };
            registers[5] = 10;
        }
        [0, 3] => {
            if exponents[0] <= 2 {
                registers[2] = 14;
                registers[7] = 14;
            } else {
                registers[2] = 8;
                registers[7] = 7;
            }
        }
        [0, 4] => {
            registers[2] = if exponents[0] == 1 { 10 } else { 8 };
            registers[11] = 11;
        }
        [0, 5] => {
            registers[2] = if exponents[0] <= 2 { 12 } else { 8 };
            registers[13] = 13;
        }
        [1, 2] => {
            registers[3] = 15;
            registers[5] = 15;
        }
        [1, 3] => {
            registers[3] = if exponents[1] == 1 { 6 } else { 9 };
            registers[7] = 7;
        }
        [1, 4] => {
            registers[3] = 9;
            registers[11] = 11;
        }
        [1, 5] => {
            registers[3] = if exponents[1] == 1 { 12 } else { 9 };
            registers[13] = 13;
        }
        // Exact 2+odd leaves with additional 11/13 factors and no radix 3.
        [0, 2, 4] | [0, 2, 5] | [0, 2, 4, 5] => {
            registers[2] = if exponents[0] <= 2 { 10 } else { 8 };
            registers[5] = 10;
            if exponents[4] > 0 {
                registers[11] = 11;
            }
            if exponents[5] > 0 {
                registers[13] = 13;
            }
        }
        [0, 3, 4] | [0, 3, 5] | [0, 3, 4, 5] => {
            registers[2] = match exponents[0] {
                1 | 2 => 14,
                3 => 8,
                _ => 16,
            };
            registers[7] = 14;
            if exponents[4] > 0 {
                registers[11] = 11;
            }
            if exponents[5] > 0 {
                registers[13] = 13;
            }
        }
        [0, 4, 5] => {
            registers[2] = if exponents[0] <= 2 { 12 } else { 8 };
            registers[11] = 11;
            registers[13] = 13;
        }
        [0, 2, 3, 4] | [0, 2, 3, 5] | [0, 2, 3, 4, 5] => {
            registers[2] = match exponents[0] {
                1 | 2 => 10,
                3 => 8,
                _ => 16,
            };
            registers[5] = 10;
            registers[7] = 14;
            if exponents[4] > 0 {
                registers[11] = 11;
            }
            if exponents[5] > 0 {
                registers[13] = 13;
            }
        }

        // Exact radix-3 families with additional odd factors and no radix 2.
        [1, 2, 4] | [1, 2, 5] | [1, 2, 4, 5] => {
            registers[3] = 15;
            registers[5] = 15;
            if exponents[4] > 0 {
                registers[11] = 11;
            }
            if exponents[5] > 0 {
                registers[13] = 13;
            }
        }
        [1, 3, 4] | [1, 3, 5] | [1, 3, 4, 5] => {
            if exponents[1] == 1 {
                registers[3] = 12;
                registers[7] = 14;
            } else {
                registers[3] = 9;
                registers[7] = 7;
            }
            if exponents[4] > 0 {
                registers[11] = 11;
            }
            if exponents[5] > 0 {
                registers[13] = 13;
            }
        }
        [1, 4, 5] => {
            registers[3] = if exponents[1] == 1 { 12 } else { 9 };
            registers[11] = 11;
            registers[13] = 13;
        }
        [1, 2, 3, 4] | [1, 2, 3, 5] | [1, 2, 3, 4, 5] => {
            registers[3] = 15;
            registers[5] = 15;
            registers[7] = 14;
            if exponents[4] > 0 {
                registers[11] = 11;
            }
            if exponents[5] > 0 {
                registers[13] = 13;
            }
        }

        // Exact 2x3 leaves augmented by 11/13 and/or the existing 5/7 branches.
        [0, 1, 4, 5] => {
            if exponents[0] == 1 {
                registers[2] = 6;
                registers[3] = 6;
            } else {
                registers[2] = 12;
                registers[3] = 12;
            }
            registers[11] = 11;
            registers[13] = 13;
        }
        [0, 1, 2, 4] | [0, 1, 2, 5] | [0, 1, 2, 4, 5] => {
            if exponents[0] == 1 {
                registers[2] = 10;
                registers[3] = 15;
            } else {
                registers[2] = 12;
                registers[3] = 12;
            }
            registers[5] = 10;
            if exponents[4] > 0 {
                registers[11] = 11;
            }
            if exponents[5] > 0 {
                registers[13] = 13;
            }
        }
        [0, 1, 3, 4] | [0, 1, 3, 5] | [0, 1, 3, 4, 5] => {
            registers[2] = 12;
            registers[3] = 12;
            registers[7] = 14;
            if exponents[4] > 0 {
                registers[11] = 11;
            }
            if exponents[5] > 0 {
                registers[13] = 13;
            }
        }
        [0, 1, 2, 3, 4] | [0, 1, 2, 3, 5] | [0, 1, 2, 3, 4, 5] => {
            match exponents[0] {
                1 => {
                    registers[2] = 14;
                    registers[3] = 15;
                }
                2 | 3 => {
                    registers[2] = 12;
                    registers[3] = 12;
                }
                _ => {
                    registers[2] = 16;
                    registers[3] = 12;
                }
            }
            registers[5] = 15;
            registers[7] = 14;
            if exponents[4] > 0 {
                registers[11] = 11;
            }
            if exponents[5] > 0 {
                registers[13] = 13;
            }
        }

        // Common exact three-/four-factor leaves with 11/13 absent.
        [0, 1, 2] => {
            if exponents[0] == 1 {
                registers[2] = 6;
                registers[3] = 6;
                registers[5] = 5;
            } else {
                registers[2] = 12;
                registers[3] = 12;
                registers[5] = 10;
            }
        }
        [0, 1, 3] => {
            if exponents[0] <= 2 {
                registers[2] = 6;
                registers[3] = 6;
                registers[7] = 7;
            } else {
                registers[2] = 8;
                registers[3] = 6;
                registers[7] = 7;
            }
        }
        [0, 1, 4] => {
            if exponents[0] == 1 {
                registers[2] = 6;
                registers[3] = 6;
            } else {
                registers[2] = 12;
                registers[3] = 12;
            }
            registers[11] = 11;
        }
        [0, 1, 5] => {
            if exponents[0] == 1 {
                registers[2] = 6;
                registers[3] = 6;
            } else {
                registers[2] = 12;
                registers[3] = 12;
            }
            registers[13] = 13;
        }
        [0, 2, 3] => {
            registers[2] = if exponents[0] <= 2 { 10 } else { 8 };
            registers[5] = 10;
            registers[7] = 7;
        }
        [1, 2, 3] => {
            registers[3] = 15;
            registers[5] = 15;
            registers[7] = 14;
        }
        [0, 1, 2, 3] => {
            match exponents[0] {
                1 => {
                    registers[2] = 14;
                    registers[3] = 15;
                }
                2 | 3 => {
                    registers[2] = 12;
                    registers[3] = 12;
                }
                _ => {
                    registers[2] = 16;
                    registers[3] = 12;
                }
            }
            registers[5] = 15;
            registers[7] = 14;
        }
        factors if factors.iter().all(|index| *index >= 2) => {
            for &index in factors {
                let prime = primes[index];
                registers[prime] = prime;
            }
        }
        _ => {
            return Err(VkFftError::InvalidKernelIr(
                "exhaustive smooth-factor register-table dispatch missed a supported factor subset",
            ));
        }
    }

    derive_vkfft_composite_registers(&mut registers);
    let mut multipliers = [0usize; VKFFT_RADIX_TABLE_LEN];
    for (index, exponent) in exponents.into_iter().enumerate() {
        multipliers[primes[index]] = exponent;
    }
    finalize_mixed_radix_register_schedule(fft_len, rhs_transform_count, registers, multipliers)
}

fn derive_vkfft_composite_registers(registers: &mut [usize; VKFFT_RADIX_TABLE_LEN]) {
    registers[32] = if registers[2].is_multiple_of(32) {
        registers[2]
    } else {
        0
    };
    registers[16] = if registers[2].is_multiple_of(16) {
        registers[2]
    } else {
        0
    };
    registers[8] = if registers[2].is_multiple_of(8) {
        registers[2]
    } else {
        0
    };
    registers[4] = if registers[2].is_multiple_of(4) {
        registers[2]
    } else {
        0
    };
    registers[12] = if registers[2] >= 12 && registers[3] >= 12 {
        let value = registers[2].min(registers[3]);
        if value.is_multiple_of(12) { value } else { 0 }
    } else {
        0
    };
    registers[6] = registers[2].min(registers[3]);
    registers[9] = if registers[3].is_multiple_of(9) {
        registers[3]
    } else {
        0
    };
    registers[10] = registers[2].min(registers[5]);
    registers[14] = registers[2].min(registers[7]);
    registers[15] = registers[3].min(registers[5]);
}

fn finalize_mixed_radix_register_schedule(
    fft_len: usize,
    rhs_transform_count: usize,
    mut registers: [usize; VKFFT_RADIX_TABLE_LEN],
    mut multipliers: [usize; VKFFT_RADIX_TABLE_LEN],
) -> Result<RadixRegisterSchedule> {
    let (max_non_power_of_two_radix, required_local_registers) =
        optimize_radix_kernels(&mut registers, &mut multipliers, 1);
    let mut stage_radices = Vec::new();
    for radix in (2..VKFFT_RADIX_TABLE_LEN).rev() {
        stage_radices.extend(core::iter::repeat_n(radix, multipliers[radix]));
    }
    let (registers_per_thread, min_registers_per_thread) =
        min_max_stage_registers(&stage_radices, &registers);
    if registers_per_thread == 0 || min_registers_per_thread == 0 {
        return Err(VkFftError::InvalidKernelIr(
            "mixed-radix scheduler produced no executable stages",
        ));
    }
    // VkFFT's axis-level stage array applies the same final no-registerBoost
    // minimum-register reorder used by the power-of-two path. The analogous
    // Rader-container-local reorder is intentionally absent upstream, but an
    // ordinary Stockham axis must put its lowest register-demand stage first.
    if min_registers_per_thread != registers_per_thread
        && let Some(index) = stage_radices
            .iter()
            .position(|&radix| registers[radix] == min_registers_per_thread)
    {
        stage_radices.swap(0, index);
    }
    let schedule = RadixRegisterSchedule {
        fft_len,
        rhs_transform_count,
        register_boost: 1,
        registers_per_thread_per_radix: registers,
        stage_radix_multipliers: multipliers,
        stage_radices,
        register_boost_stage_radix: None,
        registers_per_thread,
        min_registers_per_thread,
        is_good_sequence: !(registers_per_thread > 16
            || registers_per_thread >= 2 * min_registers_per_thread),
        max_non_power_of_two_radix,
        required_local_registers,
    };
    schedule.validate()?;
    Ok(schedule)
}

/// Backend-neutral entry to the shared power-of-two register arithmetic from
/// `VkFFTGetRegistersPerThread`. The caller supplies the register boost selected by
/// its `GpuSchedulerPolicy`; the legacy NVIDIA/Vulkan function remains as a
/// compatibility name for existing callers.
pub fn plan_gpu_power_of_two_radix_registers(
    fft_len: usize,
    rhs_transform_count: usize,
    register_boost: usize,
) -> Result<RadixRegisterSchedule> {
    plan_nvidia_vulkan_power_of_two_radix_registers(fft_len, rhs_transform_count, register_boost)
}

/// Port the Vulkan power-of-two branch of `VkFFTGetRegistersPerThread`, followed
/// by the non-Rader body of `VkFFTOptimizeRadixKernels` and VkFFT's final stage
/// ordering/register-boost extraction.
pub fn plan_nvidia_vulkan_power_of_two_radix_registers(
    fft_len: usize,
    rhs_transform_count: usize,
    register_boost: usize,
) -> Result<RadixRegisterSchedule> {
    if fft_len == 0 || !fft_len.is_power_of_two() {
        return Err(VkFftError::UnsupportedKernelPath(
            "power-of-two radix register scheduling requires a power-of-two FFT length",
        ));
    }
    if rhs_transform_count == 0 {
        return Err(VkFftError::InvalidKernelIr(
            "radix register scheduler requires a non-zero RHS transform count",
        ));
    }
    if register_boost == 0
        || !register_boost.is_power_of_two()
        || register_boost >= VKFFT_RADIX_TABLE_LEN
    {
        return Err(VkFftError::InvalidKernelIr(
            "power-of-two radix scheduler requires a supported power-of-two register boost",
        ));
    }
    if fft_len == 1 {
        let schedule = RadixRegisterSchedule {
            fft_len,
            rhs_transform_count,
            register_boost,
            registers_per_thread_per_radix: [0; VKFFT_RADIX_TABLE_LEN],
            stage_radix_multipliers: [0; VKFFT_RADIX_TABLE_LEN],
            stage_radices: Vec::new(),
            register_boost_stage_radix: None,
            registers_per_thread: 2,
            min_registers_per_thread: 2,
            is_good_sequence: true,
            max_non_power_of_two_radix: 1,
            required_local_registers: 1,
        };
        schedule.validate()?;
        return Ok(schedule);
    }

    let exponent = fft_len.trailing_zeros() as usize;
    let (mut registers, _, _, is_good_sequence) =
        power_of_two_register_table(fft_len, exponent, rhs_transform_count)?;
    let mut multipliers = [0usize; VKFFT_RADIX_TABLE_LEN];
    multipliers[2] = exponent;
    let (max_non_power_of_two_radix, required_local_registers) =
        optimize_radix_kernels(&mut registers, &mut multipliers, register_boost);

    let mut register_boost_stage_radix = None;
    if register_boost > 1 {
        if multipliers[register_boost] > 0 {
            multipliers[register_boost] -= 1;
            register_boost_stage_radix = Some(register_boost);
        } else if let Some(radix) = (2..VKFFT_RADIX_TABLE_LEN)
            .rev()
            .find(|&radix| multipliers[radix] > 0)
        {
            multipliers[radix] -= 1;
            register_boost_stage_radix = Some(radix);
        }
    }

    let mut stage_radices = Vec::new();
    for radix in (2..VKFFT_RADIX_TABLE_LEN).rev() {
        stage_radices.extend(core::iter::repeat_n(radix, multipliers[radix]));
    }

    // VkFFT recomputes min/max register demand before appending the extracted
    // register-boost stage. If that extraction consumed the only stage, the
    // scheduler falls back to two registers before appending it.
    let (mut registers_per_thread, mut min_registers_per_thread) =
        min_max_stage_registers(&stage_radices, &registers);
    if stage_radices.is_empty() {
        registers_per_thread = 2;
        min_registers_per_thread = 2;
    }
    if let Some(radix) = register_boost_stage_radix {
        stage_radices.push(radix);
    } else if min_registers_per_thread != registers_per_thread
        && let Some(index) = stage_radices
            .iter()
            .position(|&radix| registers[radix] == min_registers_per_thread)
    {
        stage_radices.swap(0, index);
    }

    let schedule = RadixRegisterSchedule {
        fft_len,
        rhs_transform_count,
        register_boost,
        registers_per_thread_per_radix: registers,
        stage_radix_multipliers: multipliers,
        stage_radices,
        register_boost_stage_radix,
        registers_per_thread,
        min_registers_per_thread,
        is_good_sequence,
        max_non_power_of_two_radix,
        required_local_registers,
    };
    schedule.validate()?;
    Ok(schedule)
}

fn power_of_two_register_table(
    fft_len: usize,
    exponent: usize,
    rhs_transform_count: usize,
) -> Result<([usize; VKFFT_RADIX_TABLE_LEN], usize, usize, bool)> {
    let mut registers = [0usize; VKFFT_RADIX_TABLE_LEN];
    let active_threads_y = (rhs_transform_count / 64).max(1);
    let mut test_min_stages = usize::MAX;
    let mut max_radix_min_stages = 1usize;
    for candidate in 1..=3 {
        let stages = ceil_div(exponent, candidate)?;
        if stages < test_min_stages {
            test_min_stages = stages;
            max_radix_min_stages = candidate;
        }
    }
    let mut max_loc_multipliers_pow2 = 0usize;
    for candidate in (1..=max_radix_min_stages).rev() {
        let divisor = 1usize << candidate;
        let active_threads_x =
            active_threads_y
                .checked_mul(fft_len)
                .ok_or(VkFftError::ArithmeticOverflow {
                    operation: "power-of-two register scheduler active thread estimate",
                })?
                / divisor;
        if active_threads_x >= 128 {
            max_loc_multipliers_pow2 = candidate;
            break;
        }
    }
    max_loc_multipliers_pow2 = max_loc_multipliers_pow2.max(3);

    let mut final_loc_multipliers_pow2 = 1usize;
    let mut min_stage_count = exponent;
    for candidate in 2..=max_loc_multipliers_pow2 {
        let stages = ceil_div(exponent, candidate)?;
        if stages < min_stage_count {
            final_loc_multipliers_pow2 = candidate;
            min_stage_count = stages;
        }
    }
    let register_exponent = if exponent > final_loc_multipliers_pow2 {
        final_loc_multipliers_pow2
    } else {
        exponent
    };
    registers[2] = 1usize << register_exponent;
    if exponent < 3 {
        registers[2] = 1usize << exponent;
    }
    registers[32] = if registers[2].is_multiple_of(32) {
        registers[2]
    } else {
        0
    };
    registers[16] = if registers[2].is_multiple_of(16) {
        registers[2]
    } else {
        0
    };
    registers[8] = if registers[2].is_multiple_of(8) {
        registers[2]
    } else {
        0
    };
    registers[4] = if registers[2].is_multiple_of(4) {
        registers[2]
    } else {
        0
    };
    registers[6] = registers[2].min(registers[3]);
    registers[9] = if registers[3].is_multiple_of(9) {
        registers[3]
    } else {
        0
    };
    registers[10] = registers[2].min(registers[5]);
    registers[14] = registers[2].min(registers[7]);
    registers[15] = registers[3].min(registers[5]);

    let mut max_registers = 0usize;
    let mut min_registers = usize::MAX;
    for &value in &registers {
        if value > 0 {
            max_registers = max_registers.max(value);
            min_registers = min_registers.min(value);
        }
    }
    if min_registers == usize::MAX {
        min_registers = 2;
        max_registers = 2;
    }
    let is_good_sequence = !(max_registers > 16 || max_registers >= 2 * min_registers);
    Ok((registers, max_registers, min_registers, is_good_sequence))
}

fn optimize_radix_kernels(
    registers: &mut [usize; VKFFT_RADIX_TABLE_LEN],
    multipliers: &mut [usize; VKFFT_RADIX_TABLE_LEN],
    register_boost: usize,
) -> (usize, usize) {
    if (registers[32] > 0 || registers[2].is_multiple_of(32))
        && registers[32].is_multiple_of(32)
        && multipliers[2] >= 5
    {
        multipliers[32] = multipliers[2] / 5;
        multipliers[2] -= multipliers[32] * 5;
        if registers[2].is_multiple_of(32) {
            registers[32] = registers[2];
        }
    }
    if (registers[16] > 0 || registers[2].is_multiple_of(16))
        && registers[16].is_multiple_of(16)
        && multipliers[2] >= 4
    {
        multipliers[16] = multipliers[2] / 4;
        multipliers[2] -= multipliers[16] * 4;
        if registers[2].is_multiple_of(16) {
            registers[16] = registers[2];
        }
    }
    if registers[15] > 0
        && registers[15].is_multiple_of(15)
        && multipliers[3] >= 1
        && multipliers[5] >= 1
    {
        multipliers[15] = multipliers[3].min(multipliers[5]);
        multipliers[3] -= multipliers[15];
        multipliers[5] -= multipliers[15];
    }
    if registers[14] > 0
        && registers[14].is_multiple_of(14)
        && multipliers[2] >= 1
        && multipliers[7] >= 1
    {
        multipliers[14] = multipliers[2].min(multipliers[7]);
        multipliers[2] -= multipliers[14];
        multipliers[7] -= multipliers[14];
    }
    if registers[12] > 0
        && registers[12].is_multiple_of(12)
        && multipliers[2] >= 2
        && multipliers[3] >= 1
    {
        multipliers[12] = multipliers[3].min(multipliers[2] / 2);
        multipliers[2] -= 2 * multipliers[12];
        multipliers[3] -= multipliers[12];
    }
    if registers[10] > 0
        && registers[10].is_multiple_of(10)
        && multipliers[2] >= 1
        && multipliers[5] >= 1
    {
        multipliers[10] = multipliers[2].min(multipliers[5]);
        multipliers[2] -= multipliers[10];
        multipliers[5] -= multipliers[10];
    }
    if registers[9] > 0 && registers[9].is_multiple_of(9) && multipliers[3] >= 2 {
        multipliers[9] = multipliers[3] / 2;
        multipliers[3] -= multipliers[9] * 2;
    }
    if (registers[8] > 0 || registers[2].is_multiple_of(8))
        && registers[8].is_multiple_of(8)
        && multipliers[2] >= 3
    {
        multipliers[8] = multipliers[2] / 3;
        multipliers[2] -= multipliers[8] * 3;
        if registers[2].is_multiple_of(8) {
            registers[8] = registers[2];
        }
    }
    if registers[6] > 0
        && registers[6].is_multiple_of(6)
        && multipliers[2] >= 1
        && multipliers[3] >= 1
    {
        multipliers[6] = multipliers[2].min(multipliers[3]);
        multipliers[2] -= multipliers[6];
        multipliers[3] -= multipliers[6];
    }
    if (registers[4] > 0 || registers[2].is_multiple_of(4))
        && registers[4].is_multiple_of(4)
        && multipliers[2] >= 2
    {
        multipliers[4] = multipliers[2] / 2;
        multipliers[2] -= multipliers[4] * 2;
        if registers[2].is_multiple_of(4) {
            registers[4] = registers[2];
        }
    }

    if register_boost == 2 && multipliers[2] == 0 {
        if multipliers[4] > 0 {
            multipliers[4] -= 1;
            multipliers[2] = 2;
        } else if multipliers[8] > 0 {
            multipliers[8] -= 1;
            multipliers[4] += 1;
            multipliers[2] += 1;
        } else if multipliers[16] > 0 {
            multipliers[16] -= 1;
            multipliers[8] += 1;
            multipliers[2] += 1;
        } else if multipliers[32] > 0 {
            multipliers[32] -= 1;
            multipliers[16] += 1;
            multipliers[2] += 1;
        }
    }
    if register_boost == 4 && multipliers[4] == 0 {
        if multipliers[8] > 0 {
            multipliers[8] -= 1;
            multipliers[4] += 1;
            multipliers[2] += 1;
        } else if multipliers[16] > 0 {
            multipliers[16] -= 1;
            if multipliers[2] == 0 {
                multipliers[4] = 2;
            } else {
                multipliers[4] += 1;
                multipliers[2] -= 1;
                multipliers[8] += 1;
            }
        } else if multipliers[32] > 0 {
            multipliers[32] -= 1;
            if multipliers[2] == 0 {
                multipliers[8] += 1;
                multipliers[4] += 1;
            } else {
                multipliers[16] += 1;
                multipliers[4] += 1;
                multipliers[2] -= 1;
            }
        }
    }

    let mut max_non_power_of_two_radix = 1usize;
    let mut required_local_registers = 1usize;
    for (radix, &multiplier) in multipliers.iter().enumerate().skip(2) {
        let used_local_registers = if multiplier == 0 {
            0
        } else {
            match radix {
                6 | 9 | 12 => 3,
                10 | 15 => 5,
                14 => 7,
                _ => radix,
            }
        };
        if multiplier > 0 && !radix.is_power_of_two() {
            max_non_power_of_two_radix = max_non_power_of_two_radix.max(radix);
            required_local_registers = required_local_registers.max(used_local_registers);
        }
    }
    (max_non_power_of_two_radix, required_local_registers)
}

fn min_max_stage_registers(
    stage_radices: &[usize],
    registers: &[usize; VKFFT_RADIX_TABLE_LEN],
) -> (usize, usize) {
    let mut max_registers = 0usize;
    let mut min_registers = usize::MAX;
    for &radix in stage_radices {
        let value = registers[radix];
        if value > 0 {
            max_registers = max_registers.max(value);
            min_registers = min_registers.min(value);
        }
    }
    if max_registers == 0 {
        (0, 0)
    } else {
        (max_registers, min_registers)
    }
}

fn power_of_two_two_upload_split(
    sequence_len: usize,
    max_sequence_len_shared: usize,
    max_single_strided: usize,
    max_single_strided_half_bandwidth: usize,
    register_boost: usize,
    unit_stride_first_stage: bool,
    device: DeviceProfile,
) -> Result<Vec<usize>> {
    // Upstream keeps the dedicated power-of-two split for every vendor except
    // NVIDIA once N exceeds 262144. NVIDIA then falls through to the generic
    // divisor search; the quotient limit still depends on the first-stage stride.
    if device.vendor == GpuVendor::Nvidia && sequence_len > 262_144 {
        let quotient_limit = if unit_stride_first_stage {
            max_sequence_len_shared
        } else {
            max_single_strided_half_bandwidth
        };
        return generic_sqrt_divisor_split_with_limits(
            sequence_len,
            max_single_strided,
            quotient_limit,
        )?
        .map(Vec::from)
        .ok_or(VkFftError::UnsupportedKernelPath(
            "GPU scheduler could not find a two-upload power-of-two split",
        ));
    }

    let mut first = if unit_stride_first_stage {
        let max_pow8_shared = power_of_eight_floor(max_sequence_len_shared)?.max(1);
        if sequence_len / max_pow8_shared <= max_single_strided {
            max_pow8_shared
        } else if sequence_len / max_sequence_len_shared <= max_single_strided {
            max_sequence_len_shared
        } else {
            let boosted_shared = max_sequence_len_shared.checked_mul(register_boost).ok_or(
                VkFftError::ArithmeticOverflow {
                    operation: "power-of-two unit-stride two-upload boosted shared capacity",
                },
            )?;
            if sequence_len / boosted_shared < max_single_strided_half_bandwidth {
                let max_shift = exact_log2_bounds(register_boost.max(1)).0;
                let mut selected = None;
                for shift in 1..=max_shift {
                    let candidate = max_sequence_len_shared.checked_shl(shift as u32).ok_or(
                        VkFftError::ArithmeticOverflow {
                            operation: "power-of-two unit-stride two-upload shared scale",
                        },
                    )?;
                    if sequence_len / candidate <= max_single_strided {
                        selected = Some(candidate);
                        break;
                    }
                }
                selected.unwrap_or(boosted_shared)
            } else {
                boosted_shared
            }
        }
    } else {
        let mut max_pow8_strided = power_of_eight_floor(max_single_strided)?.min(512);
        if max_pow8_strided == 0 {
            max_pow8_strided = 1;
        }
        if sequence_len / max_pow8_strided <= max_single_strided {
            max_pow8_strided
        } else if sequence_len / max_single_strided < max_single_strided_half_bandwidth {
            max_single_strided
        } else {
            max_single_strided_half_bandwidth
        }
    };
    if first == 0 || !sequence_len.is_multiple_of(first) {
        return Err(VkFftError::UnsupportedKernelPath(
            "power-of-two two-upload split is incompatible with the selected capacity",
        ));
    }
    let mut second = sequence_len / first;
    if second < 64 {
        first /= 64 / second.max(1);
        second = 64;
    }
    if second > first {
        core::mem::swap(&mut first, &mut second);
    }
    Ok(vec![first, second])
}

fn power_of_two_three_upload_split(
    sequence_len: usize,
    used_shared_memory_bytes: usize,
    max_sequence_len_shared: usize,
    max_single_strided: usize,
    max_single_strided_half_bandwidth: usize,
    register_boost: usize,
    unit_stride_first_stage: bool,
    device: DeviceProfile,
) -> Result<Vec<usize>> {
    if device.vendor == GpuVendor::Nvidia && sequence_len > 262_144 {
        return if unit_stride_first_stage {
            generic_three_upload_unit_stride_split(
                sequence_len,
                max_sequence_len_shared,
                max_single_strided,
            )
        } else {
            generic_power_of_two_three_upload_split(
                sequence_len,
                max_single_strided,
                max_single_strided_half_bandwidth,
            )
        };
    }
    if used_shared_memory_bytes < 128 || max_sequence_len_shared == 0 || max_single_strided == 0 {
        return Err(VkFftError::UnsupportedKernelPath(
            "power-of-two three-upload split requires executable shared-memory capacities",
        ));
    }

    let max_pow8_strided = power_of_eight_floor(max_single_strided)?.max(1);
    let first = if unit_stride_first_stage {
        // Upstream unit-stride branch prefers the full shared-memory first stage;
        // the remaining two stages use ordinary strided capacity.
        let max_pow8_shared = power_of_eight_floor(max_sequence_len_shared)?.max(1);
        let square_capacity = max_single_strided.checked_mul(max_single_strided).ok_or(
            VkFftError::ArithmeticOverflow {
                operation: "power-of-two unit-stride three-upload square capacity",
            },
        )?;
        if sequence_len / max_pow8_shared <= square_capacity {
            max_pow8_shared
        } else if sequence_len / max_sequence_len_shared <= square_capacity {
            max_sequence_len_shared
        } else {
            let boosted_shared = max_sequence_len_shared.checked_mul(register_boost).ok_or(
                VkFftError::ArithmeticOverflow {
                    operation: "power-of-two unit-stride three-upload boosted shared capacity",
                },
            )?;
            if sequence_len / boosted_shared <= square_capacity {
                let max_shift = exact_log2_bounds(register_boost.max(1)).0;
                let mut selected = None;
                for shift in 0..=max_shift {
                    let candidate = max_sequence_len_shared.checked_shl(shift as u32).ok_or(
                        VkFftError::ArithmeticOverflow {
                            operation: "power-of-two unit-stride three-upload shared scale",
                        },
                    )?;
                    if sequence_len / candidate <= square_capacity {
                        selected = Some(candidate);
                        break;
                    }
                }
                selected.unwrap_or(boosted_shared)
            } else {
                boosted_shared
            }
        }
    } else {
        // Fixed upstream reorderFourStep branch: reserve a 128-byte-coalesced first
        // stage to reduce TLB pressure, then choose the remaining two power-of-two
        // factors from the ordinary strided capacity.
        let max_single_strided_128 = used_shared_memory_bytes / 128;
        let max_pow8_128 = power_of_eight_floor(max_single_strided_128)?.max(1);
        let remaining_capacity = max_pow8_strided.checked_mul(max_single_strided).ok_or(
            VkFftError::ArithmeticOverflow {
                operation: "power-of-two three-upload remaining capacity",
            },
        )?;
        if sequence_len / max_pow8_128 <= remaining_capacity {
            max_pow8_128
        } else if max_pow8_128.checked_mul(2).is_some_and(|candidate| {
            candidate <= max_single_strided_128 && sequence_len / candidate <= remaining_capacity
        }) {
            max_pow8_128 * 2
        } else if max_pow8_128.checked_mul(4).is_some_and(|candidate| {
            candidate <= max_single_strided_128 && sequence_len / candidate <= remaining_capacity
        }) {
            max_pow8_128 * 4
        } else {
            let square_capacity = max_single_strided.checked_mul(max_single_strided).ok_or(
                VkFftError::ArithmeticOverflow {
                    operation: "power-of-two three-upload square strided capacity",
                },
            )?;
            if sequence_len / max_single_strided <= square_capacity {
                let ratio = max_single_strided / max_single_strided_128.max(1);
                let max_scale_log2 = exact_log2_bounds(ratio.max(1)).0;
                let mut selected = None;
                for shift in 0..=max_scale_log2 {
                    let candidate = max_single_strided_128.checked_shl(shift as u32).ok_or(
                        VkFftError::ArithmeticOverflow {
                            operation: "power-of-two three-upload 128-byte stage scale",
                        },
                    )?;
                    if sequence_len / candidate <= square_capacity {
                        selected = Some(candidate);
                        break;
                    }
                }
                selected.unwrap_or(max_single_strided)
            } else {
                max_single_strided_half_bandwidth
            }
        }
    };
    if first == 0 || !sequence_len.is_multiple_of(first) {
        return Err(VkFftError::UnsupportedKernelPath(
            "power-of-two three-upload first split is incompatible with the sequence",
        ));
    }

    let remainder = sequence_len / first;
    let mut second;
    let mut third;
    if remainder < max_pow8_strided {
        let remainder_log2 = exact_log2_bounds(remainder.max(1)).0;
        second = 1usize.checked_shl((remainder_log2 / 2) as u32).ok_or(
            VkFftError::ArithmeticOverflow {
                operation: "power-of-two three-upload balanced second factor",
            },
        )?;
        third = remainder / second;
    } else {
        if remainder / max_pow8_strided <= max_single_strided {
            second = max_pow8_strided;
            third = remainder / second;
        } else {
            second = max_single_strided;
            third = remainder / second;
        }
        if third < 64 {
            second /= 64 / third.max(1);
            third = 64;
        }
    }
    if third > second {
        core::mem::swap(&mut second, &mut third);
    }
    if first == 0
        || second == 0
        || third == 0
        || first
            .checked_mul(second)
            .and_then(|value| value.checked_mul(third))
            != Some(sequence_len)
    {
        return Err(VkFftError::UnsupportedKernelPath(
            "power-of-two three-upload split does not cover the sequence",
        ));
    }
    Ok(vec![first, second, third])
}

fn generic_power_of_two_three_upload_split(
    sequence_len: usize,
    max_single_strided: usize,
    quotient_limit: usize,
) -> Result<Vec<usize>> {
    let mut outer = highest_power_of_two_at_most(ceil_cuberoot(sequence_len)?);
    while outer > 0 {
        if outer <= max_single_strided && sequence_len.is_multiple_of(outer) {
            let remainder = sequence_len / outer;
            let mut middle = highest_power_of_two_at_most(ceil_sqrt(remainder)?);
            while middle > 0 {
                if middle <= max_single_strided
                    && remainder.is_multiple_of(middle)
                    && remainder / middle <= quotient_limit
                {
                    return Ok(vec![remainder / middle, outer, middle]);
                }
                middle /= 2;
            }
        }
        outer /= 2;
    }
    Err(VkFftError::UnsupportedKernelPath(
        "GPU scheduler could not find a three-upload power-of-two divisor split",
    ))
}

fn generic_three_upload_unit_stride_split(
    sequence_len: usize,
    first_stage_limit: usize,
    max_single_strided: usize,
) -> Result<Vec<usize>> {
    if first_stage_limit == 0 || max_single_strided == 0 {
        return Err(VkFftError::UnsupportedKernelPath(
            "unit-stride three-upload search requires non-zero capacities",
        ));
    }
    // Direct port of the upstream generic unit-stride branch: prefer the largest
    // legal first factor near maxSequenceLengthSharedMemory, then balance the
    // remaining two strided factors around sqrt(remainder).
    let mut first = first_stage_limit.min(sequence_len);
    while first > 0 {
        if sequence_len.is_multiple_of(first) {
            let remainder = sequence_len / first;
            let mut second = ceil_sqrt(remainder)?;
            while second > 0 {
                if second <= max_single_strided
                    && remainder.is_multiple_of(second)
                    && remainder / second <= max_single_strided
                {
                    return Ok(vec![first, second, remainder / second]);
                }
                second -= 1;
            }
        }
        first -= 1;
    }
    Err(VkFftError::UnsupportedKernelPath(
        "GPU scheduler could not find a unit-stride three-upload divisor split",
    ))
}

fn generic_sqrt_divisor_split_with_limits(
    sequence_len: usize,
    divisor_limit: usize,
    quotient_limit: usize,
) -> Result<Option<[usize; 2]>> {
    let mut candidate = ceil_sqrt(sequence_len)?;
    while candidate > 0 {
        if candidate <= divisor_limit
            && sequence_len.is_multiple_of(candidate)
            && sequence_len / candidate <= quotient_limit
        {
            return Ok(Some([sequence_len / candidate, candidate]));
        }
        candidate -= 1;
    }
    Ok(None)
}

fn generic_three_upload_divisor_split_with_limits(
    sequence_len: usize,
    max_single_strided: usize,
    quotient_limit: usize,
) -> Result<Vec<usize>> {
    let mut outer = ceil_cuberoot(sequence_len)?;
    while outer > 0 {
        if outer <= max_single_strided && sequence_len.is_multiple_of(outer) {
            let remainder = sequence_len / outer;
            let mut middle = ceil_sqrt(remainder)?;
            while middle > 0 {
                if middle <= max_single_strided
                    && remainder.is_multiple_of(middle)
                    && remainder / middle <= quotient_limit
                {
                    return Ok(vec![remainder / middle, outer, middle]);
                }
                middle -= 1;
            }
        }
        outer -= 1;
    }
    Err(VkFftError::UnsupportedKernelPath(
        "GPU smooth scheduler could not find a three-upload divisor split",
    ))
}

fn prefer_four_step_first_factor(axis_split: &mut [usize]) {
    for divisor in [2usize, 4, 8] {
        if axis_split
            .first()
            .is_some_and(|first| !first.is_multiple_of(divisor))
            && let Some(index) = axis_split
                .iter()
                .position(|factor| factor.is_multiple_of(divisor))
        {
            axis_split.swap(0, index);
        }
    }
}

fn largest_square_divisor_boost(sequence_len: usize, limit: usize) -> usize {
    let mut selected = 1usize;
    for candidate in 1..=limit.max(1) {
        if candidate
            .checked_mul(candidate)
            .is_some_and(|square| sequence_len.is_multiple_of(square))
        {
            selected = candidate;
        }
    }
    selected
}

fn exponent_covering(target: usize, base: usize) -> Result<usize> {
    if base <= 1 {
        if target <= 1 {
            return Ok(1);
        }
        return Err(VkFftError::ResourceLimitExceeded {
            resource: "NVIDIA scheduler strided upload base",
            required: 2,
            available: base,
        });
    }
    let mut exponent = 1usize;
    let mut covered = base;
    while covered < target {
        covered = covered
            .checked_mul(base)
            .ok_or(VkFftError::ArithmeticOverflow {
                operation: "NVIDIA scheduler upload coverage",
            })?;
        exponent += 1;
    }
    Ok(exponent)
}

fn power_of_eight_floor(value: usize) -> Result<usize> {
    if value == 0 {
        return Err(VkFftError::ResourceLimitExceeded {
            resource: "NVIDIA scheduler power-of-eight capacity",
            required: 1,
            available: 0,
        });
    }
    let log2 = usize::BITS as usize - 1 - value.leading_zeros() as usize;
    Ok(1usize << (3 * (log2 / 3)))
}

fn highest_power_of_two_at_most(value: usize) -> usize {
    if value == 0 {
        0
    } else {
        1usize << (usize::BITS as usize - 1 - value.leading_zeros() as usize)
    }
}

fn ceil_sqrt(value: usize) -> Result<usize> {
    if value <= 1 {
        return Ok(value);
    }
    let mut low = 1usize;
    let mut high = value.min(1usize << (usize::BITS as usize).div_ceil(2));
    while low < high {
        let mid = low + (high - low) / 2;
        if mid >= ceil_div(value, mid)? {
            high = mid;
        } else {
            low = mid + 1;
        }
    }
    Ok(low)
}

fn ceil_cuberoot(value: usize) -> Result<usize> {
    if value <= 1 {
        return Ok(value);
    }
    let mut low = 1usize;
    let mut high = highest_power_of_two_at_most(value).saturating_mul(2);
    while low < high {
        let mid = low + (high - low) / 2;
        let square = mid.checked_mul(mid);
        let cube_at_least_value = square
            .and_then(|square| square.checked_mul(mid))
            .is_none_or(|cube| cube >= value);
        if cube_at_least_value {
            high = mid;
        } else {
            low = mid + 1;
        }
    }
    Ok(low)
}

/// Preserve the fixed-upstream `floor(pow(double, 1.0 / degree))` semantics used by
/// `VkFFTGetRegistersPerThreadOptimizeShared`. This intentionally is not replaced by
/// an integer nth-root: libm rounding at exact roots (for example 64^(1/3)) affects
/// the upstream candidate search and is therefore part of the scheduling contract.
fn upstream_floor_nth_root(value: usize, degree: usize) -> usize {
    debug_assert!(degree > 0);
    (value as f64).powf(1.0 / degree as f64).floor() as usize
}

/// Choose the same floor/ceil multiple as VkFFT's ratio comparison without using
/// floating point. `maximum` is the current global register maximum and is therefore
/// at least `value`.
fn nearest_register_multiple_multiplier(maximum: usize, value: usize) -> usize {
    debug_assert!(maximum >= value && value > 0);
    let floor = maximum / value;
    let remainder = maximum % value;
    if remainder == 0 {
        return floor;
    }
    let ceil = floor + 1;
    // Compare (value * ceil) / maximum > maximum / (value * floor).
    // Rewriting around maximum = value * floor + remainder keeps each cross product
    // bounded by usize^2, which fits exactly in u128 on supported 32/64-bit hosts.
    let floor_side = (value - remainder) as u128 * (maximum - remainder) as u128;
    let ceil_side = remainder as u128 * maximum as u128;
    if floor_side > ceil_side { floor } else { ceil }
}

/// Exact form of upstream's `threads / subgroup == 1 && threads / subgroup < 1.5`
/// batch heuristic. Cross multiplication avoids scheduler decisions through f64.
fn upstream_near_single_subgroup(threads: usize, subgroup: usize) -> bool {
    debug_assert!(subgroup > 0);
    threads / subgroup == 1 && (threads as u128) * 2 < (subgroup as u128) * 3
}

/// Exact floor/ceiling base-2 logarithms for a non-zero integer. Scheduler
/// decisions use these bounds instead of floating-point `log2`, keeping fixed-
/// commit stage-count scoring deterministic for every representable `usize`.
fn exact_log2_bounds(value: usize) -> (usize, usize) {
    debug_assert!(value > 0);
    let floor = value.ilog2() as usize;
    let ceil = floor + usize::from(!value.is_power_of_two());
    (floor, ceil)
}

fn ceil_div(value: usize, divisor: usize) -> Result<usize> {
    if divisor == 0 {
        return Err(VkFftError::InvalidKernelIr(
            "NVIDIA scheduler attempted division by zero",
        ));
    }
    value
        .checked_add(divisor - 1)
        .ok_or(VkFftError::ArithmeticOverflow {
            operation: "NVIDIA scheduler ceil division",
        })
        .map(|adjusted| adjusted / divisor)
}

fn checked_mul(lhs: usize, rhs: usize, operation: &'static str) -> Result<usize> {
    lhs.checked_mul(rhs)
        .ok_or(VkFftError::ArithmeticOverflow { operation })
}

fn checked_pow(base: usize, exponent: usize, operation: &'static str) -> Result<usize> {
    let exponent = u32::try_from(exponent).map_err(|_| VkFftError::ValueOutOfRange {
        field: "NVIDIA scheduler upload exponent",
    })?;
    base.checked_pow(exponent)
        .ok_or(VkFftError::ArithmeticOverflow { operation })
}

fn precision_name(precision: Precision) -> &'static str {
    match precision {
        Precision::F16StorageF32Compute => "f16-storage/f32-compute",
        Precision::F32 => "f32",
        Precision::F64 => "f64",
        Precision::F64ComputeF32Storage => "f64-compute/f32-storage",
        Precision::DoubleDouble => "double-double",
        Precision::DoubleDoubleF64Storage => "double-double/f64-storage",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn higher_axis_f16_forced_rader_keeps_full_upload0_coalescing_pressure() {
        let mut device = DeviceProfile::generic(Backend::Vulkan, GpuVendor::Amd);
        device.shared_memory_bytes = 2048;
        device.shared_memory_pow2_bytes = 2048;
        device.max_threads_per_block = 64;
        device.max_workgroup_size = [64, 64, 64];
        device.coalesced_memory_bytes = 64;
        let mut tuning = PlannerTuning::for_device(device, Precision::F16StorageF32Compute);
        tuning.min_rader_direct_prime = 17;
        tuning.max_rader_direct_prime = 89;
        tuning.min_rader_fft_prime = 17;
        tuning.max_rader_fft_prime = 16_384;
        tuning.validate().unwrap();

        let axis0_upload0 =
            plan_gpu_axis0_multi_fft_rader_parent_schedule_with_max_batch_coalesced(
                62,
                &[(31, 1)],
                31,
                8,
                true,
                tuning,
                device,
            )
            .unwrap()
            .expect("axis-0 upload0 should have a parent schedule");
        assert_eq!(axis0_upload0.global_scale_registers_num, 1);
        assert_eq!(axis0_upload0.final_min_registers, 4);
        assert_eq!(axis0_upload0.min_rader_fft_thread_num, 12);
        assert_eq!(axis0_upload0.threads_per_transform, 16);

        let higher_axis = plan_gpu_axis0_multi_fft_rader_parent_schedule_with_max_batch_coalesced(
            62,
            &[(31, 1)],
            31,
            8,
            false,
            tuning,
            device,
        )
        .unwrap()
        .expect("higher-axis upload0 should have a parent schedule");
        assert_eq!(higher_axis.global_scale_registers_num, 2);
        assert_eq!(higher_axis.final_min_registers, 8);
        assert_eq!(higher_axis.min_rader_fft_thread_num, 6);
        assert_eq!(higher_axis.threads_per_transform, 8);
    }

    fn nvidia_vulkan(shared_memory_bytes: usize) -> DeviceProfile {
        DeviceProfile {
            shared_memory_bytes,
            shared_memory_pow2_bytes: shared_memory_bytes.next_power_of_two() / 2,
            coalesced_memory_bytes: 32,
            ..DeviceProfile::generic(Backend::Vulkan, GpuVendor::Nvidia)
        }
    }

    #[test]
    fn bluestein_auto_padding_tables_match_fixed_upstream_boundaries() {
        let nvidia = nvidia_vulkan(48 * 1024);
        assert_eq!(
            upstream_bluestein_auto_padding_from_table(41, Precision::F32, nvidia),
            Some(96)
        );
        assert_eq!(
            upstream_bluestein_auto_padding_from_table(41, Precision::F64, nvidia),
            Some(90)
        );
        // Explicit/custom planner tuning can expose Bluestein lengths that the
        // benchmark table never selects under fixed defaults. A table interval
        // must never return a circular-convolution size below the 2N-1 minimum.
        assert_eq!(
            upstream_bluestein_auto_padding_from_table(66, Precision::DoubleDouble, nvidia,),
            None
        );
        assert_eq!(
            upstream_bluestein_auto_padding_from_table(206, Precision::F32, nvidia),
            Some(512)
        );
        assert_eq!(
            upstream_bluestein_auto_padding_from_table(256, Precision::F32, nvidia),
            Some(512)
        );
        assert_eq!(
            upstream_bluestein_auto_padding_from_table(257, Precision::F32, nvidia),
            Some(567)
        );
        assert_eq!(
            upstream_bluestein_auto_padding_from_table(4096, Precision::F32, nvidia),
            Some(8192)
        );
        assert_eq!(
            upstream_bluestein_auto_padding_from_table(4097, Precision::F32, nvidia),
            None
        );

        let amd = DeviceProfile::generic(Backend::Vulkan, GpuVendor::Amd);
        assert_eq!(
            upstream_bluestein_auto_padding_from_table(19, Precision::F32, amd),
            Some(42)
        );
        assert_eq!(
            upstream_bluestein_auto_padding_from_table(19, Precision::F64, amd),
            Some(40)
        );
        assert_eq!(
            upstream_bluestein_auto_padding_from_table(3438, Precision::F64, amd),
            Some(6875)
        );
        assert_eq!(
            upstream_bluestein_auto_padding_from_table(3439, Precision::F64, amd),
            Some(8192)
        );

        let unknown = DeviceProfile::generic(Backend::Vulkan, GpuVendor::Other(0x1234));
        assert_eq!(
            upstream_bluestein_auto_padding_from_table(206, Precision::F32, unknown),
            None
        );

        assert_eq!(
            upstream_bluestein_generic_padding(10, 1, Precision::F32, nvidia).unwrap(),
            32
        );
        assert_eq!(
            upstream_bluestein_auto_padding(4106, 1, Precision::F32, nvidia).unwrap(),
            8232
        );
        // Quad/DD uses the same table family, but table misses must score smooth
        // candidates through VkFFTGetRegistersPerThreadQuad rather than fail soft.
        for precision in [Precision::DoubleDouble, Precision::DoubleDoubleF64Storage] {
            assert_eq!(
                upstream_bluestein_auto_padding_from_table(4106, precision, nvidia),
                None
            );
            assert_eq!(
                upstream_bluestein_auto_padding(4106, 1, precision, nvidia).unwrap(),
                8232
            );
            let quad = plan_gpu_double_double_quad_registers(8232, 1).unwrap();
            assert!(quad.is_good_sequence);
        }
    }

    #[test]
    fn exact_rader_outer_log2_scoring_matches_legacy_formula_without_float_in_production() {
        for prime in [17usize, 19, 29, 53, 257, 4001, 12289] {
            for exponent in 1..=20u32 {
                let containers = 1usize << exponent;
                let Some(outer_fft_len) = prime.checked_mul(containers) else {
                    continue;
                };
                let (floor, ceil) = exact_log2_bounds(outer_fft_len);
                let legacy = (outer_fft_len as f64).log2();
                assert_eq!(floor, legacy as usize, "N={outer_fft_len}");
                for candidate in 1..=3usize {
                    assert_eq!(
                        ceil_div(ceil, candidate).unwrap(),
                        (legacy / candidate as f64).ceil() as usize,
                        "N={outer_fft_len}, radix-log2 candidate={candidate}"
                    );
                }
            }
        }

        let high_bit = usize::BITS - 2;
        let above_exact_f64_integer_range = (1usize << high_bit) + 1;
        assert_eq!(
            exact_log2_bounds(above_exact_f64_integer_range),
            (high_bit as usize, high_bit as usize + 1)
        );
        assert_eq!(
            exact_log2_bounds(usize::MAX),
            (usize::BITS as usize - 1, usize::BITS as usize)
        );
    }

    #[test]
    fn optimize_shared_preserves_upstream_double_pow_root_rounding() {
        assert_eq!(upstream_floor_nth_root(64, 3), 3);
        assert_eq!(upstream_floor_nth_root(81, 4), 3);
        assert_eq!(upstream_floor_nth_root(256, 4), 4);
    }

    #[test]
    fn exact_scheduler_integer_scoring_matches_legacy_ratio_and_subgroup_formulas() {
        for maximum in 1usize..=512 {
            for value in 1usize..=maximum {
                let ratio = maximum as f64 / value as f64;
                let ratio_floor = ratio.floor() as usize;
                let ratio_ceil = ratio.ceil() as usize;
                let ratio2 = (value * ratio_ceil) as f64 / maximum as f64;
                let ratio3 = maximum as f64 / (value * ratio_floor) as f64;
                let legacy = if ratio2 > ratio3 {
                    ratio_floor
                } else {
                    ratio_ceil
                };
                assert_eq!(
                    nearest_register_multiple_multiplier(maximum, value),
                    legacy,
                    "maximum={maximum}, value={value}"
                );
            }
        }

        for subgroup in 1usize..=128 {
            for threads in 1usize..=4 * subgroup {
                let legacy = threads / subgroup == 1 && (threads as f64 / subgroup as f64) < 1.5;
                assert_eq!(
                    upstream_near_single_subgroup(threads, subgroup),
                    legacy,
                    "threads={threads}, subgroup={subgroup}"
                );
            }
        }
    }

    #[test]
    fn higher_axis_rader_blocks_keep_prime_threads_on_y() {
        let mut device = nvidia_vulkan(48 * 1024);
        device.max_threads_per_block = 1024;
        device.max_workgroup_size = [1024, 1024, 64];

        let direct = plan_gpu_other_axis_direct_rader_batch_block(47, 40, 8, 8, None, None, device)
            .unwrap()
            .expect("higher-axis p47 should retain a physical block");
        assert_eq!(direct.threads_per_transform, 24);
        assert_eq!(direct.grouped_batch, 4);
        assert_eq!([direct.local_size_x, direct.local_size_y], [4, 24]);
        assert!(direct.transforms_on_x);
        assert!(!direct.axis_swapped);
        direct.validate(40, device).unwrap();

        let direct_user =
            plan_gpu_other_axis_direct_rader_batch_block(47, 40, 8, 8, Some(3), Some(3), device)
                .unwrap()
                .expect("higher-axis p47/G3 should retain the requested parent group");
        assert_eq!(direct_user.grouped_batch, 3);
        assert_eq!(
            [direct_user.local_size_x, direct_user.local_size_y],
            [3, 24]
        );
        direct_user.validate(40, device).unwrap();

        let schedule =
            plan_gpu_rader_fft_registers_for_containers(257, 40, 257, 1, device).unwrap();
        let fft =
            plan_gpu_other_axis_fft_rader_batch_block(257, 40, 8, &schedule, 8, None, None, device)
                .unwrap()
                .expect("higher-axis p257 should retain a physical block");
        assert_eq!(fft.threads_per_transform, 17);
        assert_eq!(fft.grouped_batch, 8);
        assert_eq!([fft.local_size_x, fft.local_size_y], [8, 17]);
        assert!(fft.transforms_on_x);
        assert!(!fft.axis_swapped);
        fft.validate(40, device).unwrap();

        let fft_user = plan_gpu_other_axis_fft_rader_batch_block(
            257,
            40,
            8,
            &schedule,
            8,
            Some(3),
            Some(3),
            device,
        )
        .unwrap()
        .expect("higher-axis p257/G3 should retain the requested parent group");
        assert_eq!(fft_user.grouped_batch, 3);
        assert_eq!([fft_user.local_size_x, fft_user.local_size_y], [3, 17]);
        fft_user.validate(40, device).unwrap();

        for (prime, expected_threads) in [(31usize, 7usize), (281, 41)] {
            let schedule =
                plan_gpu_rader_fft_registers_for_containers(prime, 40, prime, 1, device).unwrap();
            let block = plan_gpu_other_axis_fft_rader_batch_block(
                prime,
                40,
                8,
                &schedule,
                8,
                Some(3),
                Some(3),
                device,
            )
            .unwrap()
            .expect("higher-axis FFT-Rader residual must retain the exact caller floor");
            assert_eq!(block.threads_per_transform, expected_threads);
            assert_eq!(block.grouped_batch, 3);
            assert!(block.transforms_on_x);
            assert!(!block.axis_swapped);
            assert_eq!(
                [block.local_size_x, block.local_size_y],
                [3, expected_threads]
            );
            block.validate(40, device).unwrap();
        }
    }

    #[test]
    fn higher_axis_thread_limited_wide_types_match_upstream_grouping() {
        let mut device = nvidia_vulkan(48 * 1024);
        device.max_threads_per_block = 128;
        device.max_workgroup_size = [1024, 1024, 64];

        let stockham = plan_gpu_double_double_other_axis_user_grouped_stockham_block(
            64, 8, 8, None, None, device,
        )
        .unwrap()
        .expect("thread-limited DD N64 Stockham should keep the upstream X batch tile");
        assert_eq!(stockham.threads_per_transform, 8);
        assert_eq!(stockham.grouped_batch, 8);
        assert_eq!([stockham.local_size_x, stockham.local_size_y], [8, 8]);

        let f32 = plan_gpu_other_axis_direct_rader_batch_block(47, 40, 8, 8, None, None, device)
            .unwrap()
            .expect("thread-limited F32 p47 should retain a physical block");
        assert_eq!(f32.threads_per_transform, 24);
        assert_eq!(f32.grouped_batch, 4);

        let f64 = plan_gpu_other_axis_direct_rader_batch_block(47, 40, 8, 16, None, None, device)
            .unwrap()
            .expect("thread-limited F64 p47 should retain the coalescing-floor group");
        assert_eq!(f64.threads_per_transform, 24);
        assert_eq!(f64.grouped_batch, 2);
        assert_eq!([f64.local_size_x, f64.local_size_y], [2, 24]);

        let dd = plan_gpu_other_axis_direct_rader_batch_block(47, 40, 8, 32, None, None, device)
            .unwrap()
            .expect("thread-limited DD p47 should retain the coalescing-floor group");
        assert_eq!(dd.threads_per_transform, 24);
        assert_eq!(dd.grouped_batch, 1);
        assert_eq!([dd.local_size_x, dd.local_size_y], [1, 24]);

        let dd_fft =
            plan_gpu_double_double_other_axis_fft_rader_batch_block(257, 40, 8, None, None, device)
                .unwrap()
                .expect("thread-limited DD p257 should retain the upstream FFT-Rader group");
        assert_eq!(dd_fft.threads_per_transform, 17);
        assert_eq!(dd_fft.grouped_batch, 4);
        assert_eq!([dd_fft.local_size_x, dd_fft.local_size_y], [4, 17]);
    }

    #[test]
    fn amd_dd_power_of_two_higher_axis_grouping_uses_full_shared_memory() {
        let device = DeviceProfile {
            shared_memory_bytes: 48 * 1024,
            shared_memory_pow2_bytes: 32 * 1024,
            max_threads_per_block: 1024,
            max_workgroup_size: [1024, 1024, 64],
            supports_f64: true,
            ..DeviceProfile::generic(Backend::Vulkan, GpuVendor::Amd)
        };
        let upload =
            plan_gpu_double_double_stockham_uploads_for_batches(16_384, 1, device).unwrap();
        assert_eq!(upload.used_shared_memory_bytes, 32 * 1024);
        assert_eq!(upload.upload_count, 2);
        assert_eq!(upload.axis_split, vec![256, 64]);

        let high = plan_gpu_double_double_other_axis_four_step_upload_block(
            &upload,
            1,
            upload.quad_schedules[1].rhs_transform_count,
            64,
            None,
            None,
            device,
        )
        .unwrap()
        .expect("AMD DD N64 high upload should consume the full 48KiB grouping budget");
        assert_eq!(high.threads_per_transform, 8);
        assert_eq!(high.grouped_batch, 24);
        assert_eq!([high.local_size_x, high.local_size_y], [24, 8]);
        assert_eq!(64 * high.grouped_batch * DD_COMPLEX_BYTES, 48 * 1024);
        assert!(high.transforms_on_x);
        assert!(!high.axis_swapped);

        let low = plan_gpu_double_double_other_axis_four_step_upload_block(
            &upload,
            0,
            upload.quad_schedules[0].rhs_transform_count,
            64,
            None,
            None,
            device,
        )
        .unwrap()
        .expect("AMD DD N256 low upload should retain an executable block");
        assert_eq!(low.grouped_batch, 2);
        assert!(low.transforms_on_x);
    }

    #[test]
    fn axis0_direct_rader_batch_block_matches_p47_upstream_geometry() {
        let mut device = nvidia_vulkan(48 * 1024);
        device.max_threads_per_block = 1024;
        device.max_workgroup_size = [1024, 1024, 64];

        let block = plan_gpu_axis0_direct_rader_batch_block(47, 32, 8, false, None, device)
            .unwrap()
            .expect("p47/batch32 should use the upstream direct-Rader axis block");
        assert_eq!(block.threads_per_transform, 24);
        assert_eq!(block.grouped_batch, 5);
        assert!(!block.transforms_on_x);
        assert!(!block.axis_swapped);
        assert_eq!([block.local_size_x, block.local_size_y], [24, 5]);
        block.validate(32, device).unwrap();

        let user = plan_gpu_axis0_direct_rader_batch_block(47, 32, 8, false, Some(3), device)
            .unwrap()
            .expect("p47 groupedBatch override should use the upstream user branch");
        assert_eq!(user.threads_per_transform, 24);
        assert_eq!(user.grouped_batch, 1);
        assert_eq!([user.local_size_x, user.local_size_y], [24, 1]);
        user.validate(32, device).unwrap();

        let dd_user = plan_gpu_axis0_direct_rader_batch_block(47, 7, 32, false, Some(3), device)
            .unwrap()
            .expect("DD p47/batch7/G3 should retain the caller group");
        assert_eq!(dd_user.threads_per_transform, 24);
        assert_eq!(dd_user.grouped_batch, 3);
        assert!(!dd_user.transforms_on_x);
        assert!(!dd_user.axis_swapped);
        assert_eq!([dd_user.local_size_x, dd_user.local_size_y], [24, 3]);
        dd_user.validate(7, device).unwrap();
    }

    #[test]
    fn axis0_fft_rader_batch_block_matches_p257_upstream_geometry() {
        let mut device = nvidia_vulkan(48 * 1024);
        device.max_threads_per_block = 1024;
        device.max_workgroup_size = [1024, 1024, 64];
        let schedule =
            plan_gpu_rader_fft_registers_for_containers(257, 32, 257, 1, device).unwrap();
        assert_eq!(schedule.container_fft_num, 1);
        assert_eq!(schedule.execution_container_fft_num, 1);
        assert_eq!(schedule.internal_fft.stage_radices, vec![16, 16]);
        assert_eq!(schedule.internal_fft.min_registers_per_thread, 16);
        assert_eq!(schedule.min_rader_fft_thread_num, 16);

        let block = plan_gpu_axis0_fft_rader_batch_block(257, 32, &schedule, 8, false, device)
            .unwrap()
            .expect("p257/batch32 should use the upstream Rader axis block");
        assert_eq!(block.threads_per_transform, 17);
        assert_eq!(block.grouped_batch, 7);
        assert!(!block.transforms_on_x);
        assert!(!block.axis_swapped);
        assert_eq!([block.local_size_x, block.local_size_y], [17, 7]);
        block.validate(32, device).unwrap();

        let user = plan_gpu_axis0_fft_rader_batch_block_with_grouped_batch(
            257,
            32,
            &schedule,
            8,
            false,
            Some(5),
            device,
        )
        .unwrap()
        .expect("p257 groupedBatch=5 should use the upstream user Rader branch");
        assert_eq!(user.threads_per_transform, 17);
        assert_eq!(user.grouped_batch, 5);
        assert!(!user.transforms_on_x);
        assert!(!user.axis_swapped);
        assert_eq!([user.local_size_x, user.local_size_y], [17, 5]);
        user.validate(32, device).unwrap();

        let mut wide_device = device;
        wide_device.shared_memory_bytes = 64 * 1024;
        wide_device.shared_memory_pow2_bytes = 64 * 1024;
        for (prime, expected_caller_threads, expected_internal_threads) in [
            (17usize, 2usize, 1usize),
            (29, 5, 4),
            (31, 7, 6),
            (257, 17, 16),
            (2_161, 145, 144),
            (4_993, 385, 384),
            (7_681, 769, 768),
        ] {
            let single =
                plan_gpu_rader_fft_registers_for_containers(prime, 1, prime, 1, wide_device)
                    .unwrap();
            assert_eq!(
                single.execution_threads_per_workgroup, expected_internal_threads,
                "p{prime} internal convolution thread floor"
            );
            let single_block =
                plan_gpu_axis0_fft_rader_batch_block(prime, 1, &single, 8, false, wide_device)
                    .unwrap()
                    .expect("batch-one FFT-Rader must preserve a wider prime caller floor");
            assert_eq!(single_block.threads_per_transform, expected_caller_threads);
            assert_eq!(single_block.grouped_batch, 1);
            assert!(!single_block.transforms_on_x);
            assert!(!single_block.axis_swapped);
            assert_eq!(
                [single_block.local_size_x, single_block.local_size_y],
                [expected_caller_threads, 1]
            );
            single_block.validate(1, wide_device).unwrap();
        }

        for (prime, expected_threads) in [(17usize, 2usize), (31, 7)] {
            let batched =
                plan_gpu_rader_fft_registers_for_containers(prime, 32, prime, 1, device).unwrap();
            let block = plan_gpu_axis0_fft_rader_batch_block(prime, 32, &batched, 8, false, device)
                .unwrap()
                .expect("batch32 FFT-Rader residual must use the upstream caller block");
            assert_eq!(block.threads_per_transform, expected_threads);
            assert_eq!(block.grouped_batch, 32);
            assert!(block.transforms_on_x);
            assert!(block.axis_swapped);
            assert_eq!(
                [block.local_size_x, block.local_size_y],
                [32, expected_threads]
            );
            block.validate(32, device).unwrap();
        }

        for (prime, expected_radices) in [
            (769usize, vec![16usize, 16, 3]),
            (1_009, vec![16, 7, 3, 3]),
            (2_017, vec![16, 14, 3, 3]),
            (2_161, vec![16, 15, 3, 3]),
            (2_801, vec![16, 7, 5, 5]),
            (4_993, vec![16, 13, 8, 3]),
            (7_841, vec![16, 14, 7, 5]),
        ] {
            let schedule =
                plan_gpu_rader_fft_registers_for_containers(prime, 1, prime, 1, wide_device)
                    .unwrap();
            assert_eq!(
                schedule.internal_fft.stage_radices, expected_radices,
                "p{prime} must preserve the final fixed-upstream Rader radix order"
            );
        }

        for (prime, radices, expected_threads, expected_padded_group) in [
            (19usize, vec![6usize, 3], 4usize, 32usize),
            (29, vec![7, 4], 5, 25),
        ] {
            let smooth =
                plan_gpu_rader_fft_registers_for_containers(prime, 32, prime, 1, device).unwrap();
            assert_eq!(smooth.internal_fft.stage_radices, radices);
            assert_eq!(smooth.container_fft_num, 1);
            assert_eq!(smooth.execution_container_fft_num, 1);
            assert!(smooth.rader_transpose.is_none());
            let smooth_block =
                plan_gpu_axis0_fft_rader_batch_block(prime, 32, &smooth, 8, false, device)
                    .unwrap()
                    .expect("smooth standalone Rader should use axis-level batch grouping");
            assert_eq!(smooth_block.threads_per_transform, expected_threads);
            assert_eq!(smooth_block.grouped_batch, 32);
            assert!(smooth_block.transforms_on_x);
            assert!(smooth_block.axis_swapped);
            assert_eq!(
                [smooth_block.local_size_x, smooth_block.local_size_y],
                [32, expected_threads]
            );
            let padded = plan_gpu_axis0_fft_rader_batch_block(prime, 32, &smooth, 8, true, device)
                .unwrap()
                .expect("zero-padded smooth Rader should keep grouping without axis swap");
            assert_eq!(padded.threads_per_transform, expected_threads);
            assert_eq!(padded.grouped_batch, expected_padded_group);
            assert!(!padded.transforms_on_x);
            assert!(!padded.axis_swapped);
            assert_eq!(
                [padded.local_size_x, padded.local_size_y],
                [expected_threads, expected_padded_group]
            );
        }
    }

    #[test]
    fn axis0_single_upload_block_matches_upstream_default_batch_geometry() {
        let mut device = nvidia_vulkan(48 * 1024);
        device.max_threads_per_block = 1024;
        device.max_workgroup_size = [1024, 1024, 64];
        let upload =
            plan_gpu_power_of_two_stockham_uploads_for_batches(64, 32, Precision::F32, device)
                .unwrap();
        assert_eq!(upload.upload_count, 1);
        assert_eq!(upload.register_boost, 1);
        assert_eq!(upload.radix_schedules[0].stage_radices, vec![8, 8]);

        let block = plan_gpu_axis0_single_upload_block(&upload, 8, device)
            .unwrap()
            .expect("64-point batched Stockham should use the covered axis block");
        assert_eq!(block.threads_per_transform, 8);
        assert_eq!(block.grouped_batch, 16);
        assert!(block.axis_swapped);
        assert_eq!([block.local_size_x, block.local_size_y], [16, 8]);
        block.validate(32, device).unwrap();

        let padded = plan_gpu_axis0_single_upload_block_with_zero_padding(&upload, 8, true, device)
            .unwrap()
            .expect("zero padding should keep groupedBatch while suppressing axisSwapped");
        assert_eq!(padded.threads_per_transform, 8);
        assert_eq!(padded.grouped_batch, 16);
        assert!(!padded.transforms_on_x);
        assert!(!padded.axis_swapped);
        assert_eq!([padded.local_size_x, padded.local_size_y], [8, 16]);
        padded.validate(32, device).unwrap();

        let user = plan_gpu_axis0_single_upload_block_with_grouped_batch(
            &upload,
            8,
            false,
            Some(8),
            device,
        )
        .unwrap()
        .expect("groupedBatch=8 should produce the upstream user-override block");
        assert_eq!(user.grouped_batch, 8);
        assert!(user.transforms_on_x);
        assert!(user.axis_swapped);
        assert_eq!([user.local_size_x, user.local_size_y], [8, 8]);

        let user_padded = plan_gpu_axis0_single_upload_block_with_grouped_batch(
            &upload,
            8,
            true,
            Some(8),
            device,
        )
        .unwrap()
        .expect("padded groupedBatch=8 should keep the user cap without axis swap");
        assert_eq!(user_padded.grouped_batch, 8);
        assert!(!user_padded.transforms_on_x);
        assert!(!user_padded.axis_swapped);
        assert_eq!([user_padded.local_size_x, user_padded.local_size_y], [8, 8]);

        let mixed_upload =
            plan_gpu_smooth_stockham_uploads_for_batches(18, 32, Precision::F32, device).unwrap();
        assert_eq!(mixed_upload.upload_count, 1);
        assert_eq!(mixed_upload.register_boost, 1);
        assert_eq!(mixed_upload.radix_schedules[0].stage_radices, vec![6, 3]);
        let mixed = plan_gpu_axis0_single_upload_block(&mixed_upload, 8, device)
            .unwrap()
            .expect("18-point smooth Stockham should share the same axis-block scheduler");
        assert_eq!(mixed.threads_per_transform, 3);
        assert_eq!(mixed.grouped_batch, 32);
        assert!(mixed.axis_swapped);
        assert_eq!([mixed.local_size_x, mixed.local_size_y], [32, 3]);

        let mut total_limited = device;
        total_limited.max_threads_per_block = 64;
        let limited = plan_gpu_axis0_single_upload_block(&upload, 8, total_limited)
            .unwrap()
            .expect("total-thread limit should shrink the batch block");
        assert_eq!(limited.grouped_batch, 8);
        assert!(limited.axis_swapped);
        assert_eq!([limited.local_size_x, limited.local_size_y], [8, 8]);

        let mut x_limited = device;
        x_limited.max_workgroup_size[0] = 8;
        assert!(
            plan_gpu_axis0_single_upload_block(&upload, 8, x_limited)
                .unwrap()
                .is_none(),
            "post-swap X must still obey the physical per-dimension limit"
        );
    }

    #[test]
    fn higher_axis_single_upload_block_uses_fastest_dimension_on_x() {
        let mut device = nvidia_vulkan(48 * 1024);
        device.max_threads_per_block = 1024;
        device.max_workgroup_size = [1024, 1024, 64];

        let n3 =
            plan_gpu_smooth_stockham_uploads_for_batches(3, 8, Precision::F32, device).unwrap();
        let block3 = plan_gpu_other_axis_single_upload_block(&n3, 8, 8, device)
            .unwrap()
            .expect("higher-axis N=3 should group all eight fastest-axis sequences");
        assert_eq!(block3.threads_per_transform, 1);
        assert_eq!(block3.grouped_batch, 8);
        assert_eq!([block3.local_size_x, block3.local_size_y], [8, 1]);
        assert!(block3.transforms_on_x);
        assert!(!block3.axis_swapped);

        let n64 =
            plan_gpu_power_of_two_stockham_uploads_for_batches(64, 64, Precision::F32, device)
                .unwrap();
        let block64 = plan_gpu_other_axis_single_upload_block(&n64, 64, 8, device)
            .unwrap()
            .expect("higher-axis N=64 should use the upstream NVIDIA aimThreads shrink");
        assert_eq!(block64.threads_per_transform, 8);
        assert_eq!(block64.grouped_batch, 16);
        assert_eq!([block64.local_size_x, block64.local_size_y], [16, 8]);
        assert!(block64.transforms_on_x);
        assert!(!block64.axis_swapped);

        let n64_three_dim =
            plan_gpu_power_of_two_stockham_uploads_for_batches(64, 512, Precision::F32, device)
                .unwrap();
        let capped = plan_gpu_other_axis_single_upload_block(&n64_three_dim, 8, 8, device)
            .unwrap()
            .expect("higher-axis grouping must be capped by physical size[0], not inner stride");
        assert_eq!(capped.grouped_batch, 8);
        assert_eq!([capped.local_size_x, capped.local_size_y], [8, 8]);
    }

    #[test]
    fn double_double_quad_register_table_matches_fixed_upstream_mixed_radix_leaves() {
        let cases = [
            (3usize, vec![(3usize, 3usize)]),
            (9, vec![(3, 9), (9, 9)]),
            (5, vec![(5, 5)]),
            (7, vec![(7, 7)]),
            (15, vec![(3, 3), (5, 5), (15, 3)]),
            (21, vec![(3, 6), (7, 7)]),
            (30, vec![(2, 6), (3, 6), (5, 5), (6, 6), (10, 5), (15, 5)]),
            (42, vec![(2, 6), (3, 6), (7, 7), (6, 6), (14, 6)]),
            (70, vec![(2, 6), (5, 5), (7, 7), (10, 5), (14, 6)]),
            (168, vec![(2, 8), (3, 6), (7, 7), (8, 8), (6, 6), (14, 7)]),
            (
                210,
                vec![
                    (2, 6),
                    (3, 6),
                    (5, 5),
                    (7, 7),
                    (6, 6),
                    (10, 5),
                    (14, 6),
                    (15, 5),
                ],
            ),
        ];
        for (length, expected) in cases {
            let schedule = plan_gpu_double_double_quad_registers(length, 7).unwrap();
            for (radix, registers) in expected {
                assert_eq!(
                    schedule.registers_per_thread_per_radix[radix], registers,
                    "Quad register mismatch for N={length}, radix={radix}"
                );
            }
            assert_eq!(schedule.registers_per_thread_per_radix[11], 0);
            assert_eq!(schedule.registers_per_thread_per_radix[13], 0);
            assert!(schedule.min_registers_per_thread > 0);
            assert!(schedule.registers_per_thread >= schedule.min_registers_per_thread);
        }

        let n16 = plan_gpu_double_double_quad_registers(16, 7).unwrap();
        assert_eq!(n16.registers_per_thread_per_radix[2], 4);
        assert_eq!(n16.min_registers_per_thread, 4);
        let n64 = plan_gpu_double_double_quad_registers(64, 7).unwrap();
        assert_eq!(n64.registers_per_thread_per_radix[2], 8);
        assert_eq!(n64.min_registers_per_thread, 8);

        assert!(matches!(
            plan_gpu_double_double_quad_registers(11, 7),
            Err(VkFftError::UnsupportedKernelPath(_))
        ));
        assert!(matches!(
            plan_gpu_double_double_quad_registers(33, 7),
            Err(VkFftError::UnsupportedKernelPath(_))
        ));
    }

    #[test]
    fn double_double_strided_axis_and_bandwidth_boost_follow_64b_coalescing() {
        let mut intel = DeviceProfile::generic(Backend::Vulkan, GpuVendor::Intel);
        intel.shared_memory_bytes = 32 * 1024;
        intel.shared_memory_pow2_bytes = 32 * 1024;
        intel.max_threads_per_block = 1024;
        intel.max_workgroup_size = [1024, 1024, 64];

        let contiguous =
            plan_gpu_double_double_stockham_uploads_for_batches(1024, 1, intel).unwrap();
        assert_eq!(contiguous.register_boost, 1);
        assert_eq!(contiguous.upload_count, 1);
        assert_eq!(contiguous.axis_split, vec![1024]);

        let strided = plan_gpu_double_double_stockham_uploads_for_batches_with_axis_context(
            1024,
            1,
            intel,
            StockhamUploadAxisContext {
                strided_axis: true,
                ..StockhamUploadAxisContext::default()
            },
        )
        .unwrap();
        assert_eq!(strided.max_sequence_len_shared, 1024);
        assert_eq!(strided.max_sequence_len_strided, 512);
        assert_eq!(strided.register_boost, 2);
        assert_eq!(strided.upload_count, 1);
        assert_eq!(strided.axis_split, vec![1024]);

        let boost1 = plan_gpu_double_double_stockham_uploads_for_batches_with_axis_context(
            1024,
            1,
            intel,
            StockhamUploadAxisContext {
                strided_axis: true,
                bandwidth_boost: 1,
                ..StockhamUploadAxisContext::default()
            },
        )
        .unwrap();
        assert_eq!(boost1.register_boost, 2);
        assert_eq!(boost1.upload_count, 1);
        assert_eq!(boost1.axis_split, vec![1024]);

        let boost2 = plan_gpu_double_double_stockham_uploads_for_batches_with_axis_context(
            1024,
            1,
            intel,
            StockhamUploadAxisContext {
                strided_axis: true,
                bandwidth_boost: 2,
                ..StockhamUploadAxisContext::default()
            },
        )
        .unwrap();
        assert_eq!(boost2.register_boost, 2);
        assert_eq!(boost2.upload_count, 1);
        assert_eq!(boost2.axis_split, vec![1024]);

        let bluestein_auto = plan_gpu_double_double_stockham_uploads_for_batches_with_axis_context(
            625,
            1,
            intel,
            StockhamUploadAxisContext {
                strided_axis: true,
                use_bluestein_fft: true,
                ..StockhamUploadAxisContext::default()
            },
        )
        .unwrap();
        assert_eq!(bluestein_auto.upload_count, 2);
        assert_eq!(bluestein_auto.axis_split, vec![25, 25]);

        let bluestein_boost2 =
            plan_gpu_double_double_stockham_uploads_for_batches_with_axis_context(
                625,
                1,
                intel,
                StockhamUploadAxisContext {
                    strided_axis: true,
                    bandwidth_boost: 2,
                    use_bluestein_fft: true,
                    perform_convolution: false,
                },
            )
            .unwrap();
        assert_eq!(bluestein_boost2.upload_count, 1);
        assert_eq!(bluestein_boost2.axis_split, vec![625]);
    }

    #[test]
    fn strided_bluestein_preserves_upstream_divisor_order_without_four_step_preference() {
        let mut nvidia = DeviceProfile::generic(Backend::Vulkan, GpuVendor::Nvidia);
        nvidia.shared_memory_bytes = 32 * 1024;
        nvidia.shared_memory_pow2_bytes = 32 * 1024;
        nvidia.max_threads_per_block = 1024;
        nvidia.max_workgroup_size = [1024, 1024, 64];

        let bluestein = plan_gpu_smooth_stockham_uploads_for_batches_with_axis_context(
            4_368,
            40,
            Precision::F32,
            nvidia,
            StockhamUploadAxisContext {
                strided_axis: true,
                use_bluestein_fft: true,
                ..StockhamUploadAxisContext::default()
            },
        )
        .unwrap();
        assert_eq!(bluestein.upload_count, 2);
        assert_eq!(bluestein.axis_split, vec![78, 56]);
        assert_eq!(
            bluestein
                .radix_schedules
                .iter()
                .map(|radix| (radix.fft_len, radix.rhs_transform_count))
                .collect::<Vec<_>>(),
            vec![(78, 2_240), (56, 3_120)]
        );

        // Ordinary reorderFourStep keeps the historical factor preference: the factor
        // divisible by the larger 2/4/8 divisor is promoted to locAxisSplit[0].
        let ordinary = plan_gpu_smooth_stockham_uploads_for_batches_with_axis_context(
            4_368,
            40,
            Precision::F32,
            nvidia,
            StockhamUploadAxisContext {
                strided_axis: true,
                ..StockhamUploadAxisContext::default()
            },
        )
        .unwrap();
        assert_eq!(ordinary.upload_count, 2);
        assert_eq!(ordinary.axis_split, vec![56, 78]);
    }

    #[test]
    fn bluestein_strided_legacy_bandwidth_probe_matches_fixed_upstream_boundaries() {
        let mut intel = DeviceProfile::generic(Backend::Vulkan, GpuVendor::Intel);
        intel.shared_memory_bytes = 32 * 1024;
        intel.shared_memory_pow2_bytes = 32 * 1024;
        intel.max_threads_per_block = 1024;
        intel.max_workgroup_size = [1024, 1024, 64];

        for sequence_len in [625usize, 1_024usize] {
            let auto = if sequence_len.is_power_of_two() {
                plan_gpu_power_of_two_stockham_uploads_for_batches_with_axis_context(
                    sequence_len,
                    1,
                    Precision::F32,
                    intel,
                    StockhamUploadAxisContext {
                        strided_axis: true,
                        use_bluestein_fft: true,
                        ..StockhamUploadAxisContext::default()
                    },
                )
            } else {
                plan_gpu_smooth_stockham_uploads_for_batches_with_axis_context(
                    sequence_len,
                    1,
                    Precision::F32,
                    intel,
                    StockhamUploadAxisContext {
                        strided_axis: true,
                        use_bluestein_fft: true,
                        ..StockhamUploadAxisContext::default()
                    },
                )
            }
            .unwrap();
            assert_eq!(auto.upload_count, 2, "auto B=1 N={sequence_len}");

            let boosted = if sequence_len.is_power_of_two() {
                plan_gpu_power_of_two_stockham_uploads_for_batches_with_axis_context(
                    sequence_len,
                    1,
                    Precision::F32,
                    intel,
                    StockhamUploadAxisContext {
                        strided_axis: true,
                        bandwidth_boost: 2,
                        use_bluestein_fft: true,
                        perform_convolution: false,
                    },
                )
            } else {
                plan_gpu_smooth_stockham_uploads_for_batches_with_axis_context(
                    sequence_len,
                    1,
                    Precision::F32,
                    intel,
                    StockhamUploadAxisContext {
                        strided_axis: true,
                        bandwidth_boost: 2,
                        use_bluestein_fft: true,
                        perform_convolution: false,
                    },
                )
            }
            .unwrap();
            assert_eq!(boosted.upload_count, 1, "explicit B=2 N={sequence_len}");
            assert_eq!(boosted.axis_split, vec![sequence_len]);
        }
    }

    #[test]
    fn double_double_upload_scheduler_uses_quad_capacity_and_upstream_splits() {
        let device = nvidia_vulkan(48 * 1024);

        let two = plan_gpu_double_double_stockham_uploads_for_batches(8_192, 7, device).unwrap();
        assert_eq!(two.used_shared_memory_bytes, 32 * 1024);
        assert_eq!(two.max_sequence_len_shared, 1_024);
        assert_eq!(two.max_sequence_len_strided, 1_024);
        assert_eq!(two.upload_count, 2);
        assert_eq!(two.axis_split, vec![128, 64]);
        assert_eq!(
            two.quad_schedules
                .iter()
                .map(|quad| quad.fft_len)
                .collect::<Vec<_>>(),
            two.axis_split
        );
        two.validate().unwrap();

        let wider = plan_gpu_double_double_stockham_uploads_for_batches(32_768, 1, device).unwrap();
        assert_eq!(wider.upload_count, 2);
        assert_eq!(wider.axis_split, vec![512, 64]);
        assert_eq!(
            wider
                .quad_schedules
                .iter()
                .map(|quad| quad.fft_len)
                .collect::<Vec<_>>(),
            wider.axis_split
        );
        wider.validate().unwrap();

        let very_wide =
            plan_gpu_double_double_stockham_uploads_for_batches(1_679_616, 1, device).unwrap();
        assert_eq!(very_wide.used_shared_memory_bytes, 48 * 1024);
        assert_eq!(very_wide.max_sequence_len_shared, 1_536);
        assert_eq!(very_wide.max_sequence_len_strided, 1_536);
        assert_eq!(very_wide.upload_count, 2);
        assert_eq!(very_wide.axis_split, vec![1_296, 1_296]);
        very_wide.validate().unwrap();

        let smooth = plan_gpu_double_double_stockham_uploads_for_batches(4_116, 3, device).unwrap();
        assert_eq!(smooth.used_shared_memory_bytes, 48 * 1024);
        assert_eq!(smooth.upload_count, 2);
        assert_eq!(smooth.axis_split, vec![84, 49]);
        assert!(
            smooth
                .quad_schedules
                .iter()
                .all(|quad| quad.min_registers_per_thread > 0)
        );
        smooth.validate().unwrap();

        let three =
            plan_gpu_double_double_stockham_uploads_for_batches(8_388_608, 1, device).unwrap();
        assert_eq!(three.upload_count, 3);
        assert_eq!(three.axis_split, vec![256, 128, 256]);
        assert_eq!(
            three
                .quad_schedules
                .iter()
                .map(|quad| quad.fft_len)
                .collect::<Vec<_>>(),
            three.axis_split
        );
        three.validate().unwrap();

        let three_wide =
            plan_gpu_double_double_stockham_uploads_for_batches(9_765_625, 1, device).unwrap();
        assert_eq!(three_wide.used_shared_memory_bytes, 48 * 1024);
        assert_eq!(three_wide.max_sequence_len_shared, 1_536);
        assert_eq!(three_wide.max_sequence_len_strided, 1_536);
        assert_eq!(three_wide.upload_count, 3);
        assert_eq!(three_wide.axis_split, vec![625, 125, 125]);
        assert_eq!(
            three_wide
                .quad_schedules
                .iter()
                .map(|quad| quad.fft_len)
                .collect::<Vec<_>>(),
            three_wide.axis_split
        );
        three_wide.validate().unwrap();
    }

    #[test]
    fn double_double_three_upload_scheduler_reaches_875_point_component() {
        let device = nvidia_vulkan(48 * 1024);
        let schedule =
            plan_gpu_double_double_stockham_uploads_for_batches(102_942_875, 1, device).unwrap();
        assert_eq!(schedule.used_shared_memory_bytes, 48 * 1024);
        assert_eq!(schedule.max_sequence_len_shared, 1_536);
        assert_eq!(schedule.max_sequence_len_strided, 1_536);
        assert_eq!(schedule.upload_count, 3);
        assert_eq!(schedule.axis_split, vec![875, 343, 343]);
        assert_eq!(
            schedule
                .quad_schedules
                .iter()
                .map(|quad| quad.fft_len)
                .collect::<Vec<_>>(),
            schedule.axis_split
        );
        schedule.validate().unwrap();
    }

    #[test]
    fn double_double_three_upload_scheduler_reaches_full_1536_capacity() {
        let device = nvidia_vulkan(48 * 1024);
        let schedule =
            plan_gpu_double_double_stockham_uploads_for_batches(3_319_142_400, 1, device).unwrap();
        assert_eq!(schedule.used_shared_memory_bytes, 48 * 1024);
        assert_eq!(schedule.max_sequence_len_shared, 1_536);
        assert_eq!(schedule.max_sequence_len_strided, 1_536);
        assert_eq!(schedule.upload_count, 3);
        assert_eq!(schedule.axis_split, vec![1536, 1470, 1470]);
        assert_eq!(
            schedule
                .quad_schedules
                .iter()
                .map(|quad| quad.fft_len)
                .collect::<Vec<_>>(),
            schedule.axis_split
        );
        schedule.validate().unwrap();
    }

    #[test]
    fn double_double_two_upload_scheduler_reaches_3125_point_component_on_128k() {
        let mut device = nvidia_vulkan(128 * 1024);
        device.shared_memory_pow2_bytes = 128 * 1024;
        let schedule =
            plan_gpu_double_double_stockham_uploads_for_batches(1_953_125, 1, device).unwrap();
        assert_eq!(schedule.used_shared_memory_bytes, 128 * 1024);
        assert_eq!(schedule.max_sequence_len_shared, 4_096);
        assert_eq!(schedule.max_sequence_len_strided, 4_096);
        assert_eq!(schedule.upload_count, 2);
        assert_eq!(schedule.axis_split, vec![3125, 625]);
        assert_eq!(
            schedule
                .quad_schedules
                .iter()
                .map(|quad| quad.fft_len)
                .collect::<Vec<_>>(),
            schedule.axis_split
        );
        schedule.validate().unwrap();
    }

    #[test]
    fn double_double_three_upload_scheduler_reaches_2401_point_component_on_96k() {
        let mut device = nvidia_vulkan(96 * 1024);
        device.shared_memory_pow2_bytes = 64 * 1024;
        let schedule =
            plan_gpu_double_double_stockham_uploads_for_batches(282_475_249, 1, device).unwrap();
        assert_eq!(schedule.used_shared_memory_bytes, 96 * 1024);
        assert_eq!(schedule.max_sequence_len_shared, 3_072);
        assert_eq!(schedule.max_sequence_len_strided, 3_072);
        assert_eq!(schedule.upload_count, 3);
        assert_eq!(schedule.axis_split, vec![2401, 343, 343]);
        assert_eq!(
            schedule
                .quad_schedules
                .iter()
                .map(|quad| quad.fft_len)
                .collect::<Vec<_>>(),
            schedule.axis_split
        );
        schedule.validate().unwrap();
    }

    #[test]
    fn double_double_three_upload_scheduler_reaches_full_4096_leaf_capacity_on_128k() {
        let mut device = nvidia_vulkan(128 * 1024);
        device.shared_memory_pow2_bytes = 128 * 1024;
        let schedule =
            plan_gpu_double_double_stockham_uploads_for_batches(68_719_476_736, 1, device).unwrap();
        assert_eq!(schedule.used_shared_memory_bytes, 128 * 1024);
        assert_eq!(schedule.max_sequence_len_shared, 4_096);
        assert_eq!(schedule.max_sequence_len_strided, 4_096);
        assert_eq!(schedule.upload_count, 3);
        assert_eq!(schedule.axis_split, vec![4096, 4096, 4096]);
        assert_eq!(
            schedule
                .quad_schedules
                .iter()
                .map(|quad| quad.fft_len)
                .collect::<Vec<_>>(),
            schedule.axis_split
        );
        schedule.validate().unwrap();
    }

    #[test]
    fn double_double_large_shared_single_upload_stockham_matches_upstream_global_scaling() {
        let mut device = nvidia_vulkan(164 * 1024);
        device.shared_memory_pow2_bytes = 164 * 1024;
        device.max_threads_per_block = 1024;
        device.max_workgroup_size = [1024, 1024, 64];

        for (sequence_len, expected_threads) in [
            (4_800usize, 960usize),
            (5_000usize, 625usize),
            (5_120usize, 640usize),
        ] {
            let upload =
                plan_gpu_double_double_stockham_uploads_for_batches(sequence_len, 1, device)
                    .unwrap();
            assert_eq!(upload.upload_count, 1);
            assert_eq!(upload.axis_split, vec![sequence_len]);

            let block = plan_gpu_double_double_axis0_stockham_block(sequence_len, 1, device)
                .unwrap()
                .expect("164KiB DD single-upload transform should have an automatic axis block");
            assert_eq!(block.threads_per_transform, expected_threads);
            assert_eq!(block.grouped_batch, 1);
            assert_eq!(
                [block.local_size_x, block.local_size_y],
                [expected_threads, 1]
            );
            assert!(!block.transforms_on_x);
            assert!(!block.axis_swapped);
            block.validate(1, device).unwrap();

            let batched = plan_gpu_double_double_axis0_stockham_block(sequence_len, 32, device)
                .unwrap()
                .expect(
                    "164KiB DD batch32 single-upload transform should keep the same caller floor",
                );
            assert_eq!(batched.threads_per_transform, expected_threads);
            assert_eq!(batched.grouped_batch, 1);
            assert_eq!(
                [batched.local_size_x, batched.local_size_y],
                [expected_threads, 1]
            );
            assert!(!batched.transforms_on_x);
            assert!(!batched.axis_swapped);
            batched.validate(32, device).unwrap();

            let higher = plan_gpu_double_double_other_axis_stockham_block(sequence_len, 40, device)
                .unwrap()
                .expect("164KiB DD higher-axis single-upload transform should keep one transform per workgroup");
            assert_eq!(higher.threads_per_transform, expected_threads);
            assert_eq!(higher.grouped_batch, 1);
            assert_eq!(
                [higher.local_size_x, higher.local_size_y],
                [1, expected_threads]
            );
            assert!(higher.transforms_on_x);
            assert!(!higher.axis_swapped);
            higher.validate(40, device).unwrap();
        }

        let small_device = nvidia_vulkan(48 * 1024);
        assert!(
            plan_gpu_double_double_axis0_stockham_block(5_000, 1, small_device)
                .unwrap()
                .is_none(),
            "N5000 must remain resource-driven and require multiple uploads on 48KiB"
        );
        assert!(
            plan_gpu_double_double_other_axis_stockham_block(5_000, 40, small_device)
                .unwrap()
                .is_none(),
            "higher-axis N5000 must also remain multi-upload on 48KiB"
        );
    }

    #[test]
    fn double_double_fft_rader_user_group_separates_rader_caller_from_quad_child_floor() {
        let mut device = nvidia_vulkan(48 * 1024);
        device.max_threads_per_block = 1024;
        device.max_workgroup_size = [1024, 1024, 64];
        let quad = plan_gpu_double_double_quad_registers(256, 7).unwrap();
        assert_eq!(quad.min_registers_per_thread, 8);
        assert_eq!(256usize.div_ceil(quad.min_registers_per_thread), 32);

        let caller = plan_gpu_double_double_axis0_fft_rader_user_grouped_batch_block(
            257,
            7,
            Some(3),
            device,
        )
        .unwrap()
        .expect("DD p257/batch7/G3 should retain the caller group");
        assert_eq!(caller.threads_per_transform, 17);
        assert_eq!(caller.grouped_batch, 3);
        assert_eq!([caller.local_size_x, caller.local_size_y], [17, 3]);
        assert!(!caller.transforms_on_x);
        assert!(!caller.axis_swapped);
        caller.validate(7, device).unwrap();

        let child =
            plan_gpu_double_double_axis0_user_grouped_stockham_block(256, 7, Some(3), device)
                .unwrap()
                .expect("DD p257 internal N256 child should use its own Quad block");
        assert_eq!(child.threads_per_transform, 32);
        assert_eq!(child.grouped_batch, 3);
        assert_eq!([child.local_size_x, child.local_size_y], [3, 32]);
        assert!(child.transforms_on_x);
        assert!(child.axis_swapped);
    }

    #[test]
    fn double_double_axis0_zero_padding_disables_bank_conflict_swap() {
        let mut device = nvidia_vulkan(48 * 1024);
        device.max_threads_per_block = 1024;
        device.max_workgroup_size = [1024, 1024, 64];

        let stockham =
            plan_gpu_double_double_axis0_user_grouped_stockham_block(64, 7, Some(3), device)
                .unwrap()
                .expect("unpadded DD N64/G3 should use a physical block");
        assert_eq!([stockham.local_size_x, stockham.local_size_y], [3, 8]);
        assert!(stockham.axis_swapped);
        let stockham_padded =
            plan_gpu_double_double_axis0_user_grouped_stockham_block_with_zero_padding(
                64,
                7,
                Some(3),
                true,
                device,
            )
            .unwrap()
            .expect("padded DD N64/G3 should keep the unswapped block");
        assert_eq!(
            [stockham_padded.local_size_x, stockham_padded.local_size_y],
            [8, 3]
        );
        assert!(!stockham_padded.transforms_on_x);
        assert!(!stockham_padded.axis_swapped);

        let direct = plan_gpu_axis0_direct_rader_batch_block(
            13,
            7,
            DD_COMPLEX_BYTES,
            false,
            Some(3),
            device,
        )
        .unwrap()
        .expect("unpadded DD p13/G3 direct-Rader should use a physical block");
        assert_eq!(direct.threads_per_transform, 7);
        assert_eq!([direct.local_size_x, direct.local_size_y], [3, 7]);
        assert!(direct.axis_swapped);
        let direct_padded =
            plan_gpu_axis0_direct_rader_batch_block(13, 7, DD_COMPLEX_BYTES, true, Some(3), device)
                .unwrap()
                .expect("padded DD p13/G3 direct-Rader should keep the unswapped block");
        assert_eq!(
            [direct_padded.local_size_x, direct_padded.local_size_y],
            [7, 3]
        );
        assert!(!direct_padded.axis_swapped);

        let fft =
            plan_gpu_double_double_axis0_fft_rader_user_grouped_batch_block(17, 7, Some(3), device)
                .unwrap()
                .expect("unpadded DD p17/G3 FFT-Rader should use a physical block");
        assert_eq!(fft.threads_per_transform, 2);
        assert_eq!([fft.local_size_x, fft.local_size_y], [3, 2]);
        assert!(fft.axis_swapped);
        let fft_padded =
            plan_gpu_double_double_axis0_fft_rader_user_grouped_batch_block_with_zero_padding(
                17,
                7,
                Some(3),
                true,
                device,
            )
            .unwrap()
            .expect("padded DD p17/G3 FFT-Rader should keep the unswapped block");
        assert_eq!([fft_padded.local_size_x, fft_padded.local_size_y], [2, 3]);
        assert!(!fft_padded.axis_swapped);
    }

    #[test]
    fn double_double_power_of_two_user_grouped_batch_uses_quad_register_floor() {
        let mut device = nvidia_vulkan(48 * 1024);
        device.max_threads_per_block = 1024;
        device.max_workgroup_size = [1024, 1024, 64];
        let block =
            plan_gpu_double_double_axis0_user_grouped_stockham_block(64, 7, Some(3), device)
                .unwrap()
                .expect("DD N=64 groupedBatch=3 should have an exact Quad axis block");
        assert_eq!(block.threads_per_transform, 8);
        assert_eq!(block.grouped_batch, 3);
        assert_eq!([block.local_size_x, block.local_size_y], [3, 8]);
        assert!(block.transforms_on_x);
        assert!(block.axis_swapped);

        let n16 = plan_gpu_double_double_axis0_user_grouped_stockham_block(16, 7, Some(3), device)
            .unwrap()
            .expect("DD N=16 groupedBatch=3 should keep a multi-lane Quad block");
        assert_eq!(n16.threads_per_transform, 4);
        assert_eq!(n16.grouped_batch, 3);
        assert_eq!([n16.local_size_x, n16.local_size_y], [3, 4]);
        assert!(n16.transforms_on_x);
        assert!(n16.axis_swapped);

        let n72 = plan_gpu_double_double_axis0_user_grouped_stockham_block(72, 7, Some(3), device)
            .unwrap()
            .expect("DD mixed-radix N=72 should be resource-gated rather than length-capped");
        assert_eq!(n72.threads_per_transform, 12);
        assert_eq!(n72.grouped_batch, 3);
        assert_eq!([n72.local_size_x, n72.local_size_y], [3, 12]);
        assert!(n72.transforms_on_x);
        assert!(n72.axis_swapped);

        let n512 =
            plan_gpu_double_double_axis0_user_grouped_stockham_block(512, 7, Some(3), device)
                .unwrap()
                .expect("DD N=512/G=3 should fit the exact 48KiB shared stripe");
        assert_eq!(n512.threads_per_transform, 64);
        assert_eq!(n512.grouped_batch, 3);
        assert_eq!([n512.local_size_x, n512.local_size_y], [64, 3]);
        assert!(!n512.transforms_on_x);
        assert!(!n512.axis_swapped);

        let n1024 =
            plan_gpu_double_double_axis0_user_grouped_stockham_block(1024, 7, Some(3), device)
                .unwrap()
                .expect("raw fixed-upstream resource clamp should still yield a group-1 fallback");
        assert_eq!(n1024.threads_per_transform, 128);
        assert_eq!(n1024.grouped_batch, 1);
        assert_eq!([n1024.local_size_x, n1024.local_size_y], [128, 1]);

        let higher = plan_gpu_double_double_other_axis_user_grouped_stockham_block(
            64,
            56,
            8,
            Some(3),
            Some(3),
            device,
        )
        .unwrap()
        .expect("DD higher-axis N=64 groupedBatch=3 should use the Quad X/Y tile");
        assert_eq!(higher.threads_per_transform, 8);
        assert_eq!(higher.grouped_batch, 3);
        assert_eq!([higher.local_size_x, higher.local_size_y], [3, 8]);
        assert!(higher.transforms_on_x);
        assert!(!higher.axis_swapped);

        let higher_axis2 = plan_gpu_double_double_other_axis_user_grouped_stockham_block(
            64,
            56,
            8,
            Some(5),
            Some(3),
            device,
        )
        .unwrap()
        .expect("DD axis>=2 grouping should preserve the fixed upstream axis-1 gate");
        assert_eq!(higher_axis2.grouped_batch, 5);
        assert_eq!(
            [higher_axis2.local_size_x, higher_axis2.local_size_y],
            [5, 8]
        );
        assert!(higher_axis2.transforms_on_x);
        assert!(!higher_axis2.axis_swapped);

        assert!(
            plan_gpu_double_double_axis0_user_grouped_stockham_block(32, 7, None, device,)
                .unwrap()
                .is_none()
        );
        let n48 = plan_gpu_double_double_axis0_user_grouped_stockham_block(48, 7, Some(3), device)
            .unwrap()
            .expect("DD mixed-radix N=48 should consume the exact Quad 2x3 table");
        assert_eq!(n48.threads_per_transform, 8);
        assert_eq!([n48.local_size_x, n48.local_size_y], [3, 8]);
        assert!(n48.axis_swapped);
        assert!(
            plan_gpu_double_double_axis0_user_grouped_stockham_block(11, 7, Some(3), device)
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn nvidia_opencl_dd_stockham_reserves_one_complex_of_static_shared_headroom() {
        let mut device = DeviceProfile {
            shared_memory_bytes: 48 * 1024,
            shared_memory_pow2_bytes: 32 * 1024,
            coalesced_memory_bytes: 32,
            max_threads_per_block: 1024,
            max_workgroup_size: [1024, 1024, 64],
            supports_f64: true,
            ..DeviceProfile::generic(Backend::OpenCl, GpuVendor::Nvidia)
        };

        let n128 =
            plan_gpu_double_double_axis0_user_grouped_stockham_block(128, 136, Some(136), device)
                .unwrap()
                .expect("OpenCL DD N128 should keep a cooperative block below the ptxas limit");
        assert_eq!(n128.grouped_batch, 11);
        assert!(
            128usize * n128.grouped_batch * DD_COMPLEX_BYTES + DD_COMPLEX_BYTES
                <= device.shared_memory_bytes
        );

        let n512 =
            plan_gpu_double_double_axis0_user_grouped_stockham_block(512, 7, Some(3), device)
                .unwrap()
                .expect("OpenCL DD N512/G3 should shrink rather than fill all 48KiB");
        assert_eq!(n512.grouped_batch, 2);
        assert!(n512.axis_swapped);

        assert!(
            plan_gpu_double_double_axis0_user_grouped_stockham_block(1536, 7, Some(1), device)
                .unwrap()
                .is_none()
        );

        let higher = plan_gpu_double_double_other_axis_user_grouped_stockham_block(
            512,
            7,
            8,
            Some(3),
            Some(3),
            device,
        )
        .unwrap()
        .expect("OpenCL DD higher-axis N512 should use the same shared headroom");
        assert_eq!(higher.grouped_batch, 2);

        // The reserve is deliberately OpenCL/NVIDIA-specific: fixed Vulkan geometry keeps
        // the exact 48KiB N512/G3 contract used by the upstream-equivalent scheduler tests.
        device.backend = Backend::Vulkan;
        let vulkan =
            plan_gpu_double_double_axis0_user_grouped_stockham_block(512, 7, Some(3), device)
                .unwrap()
                .expect("Vulkan DD N512/G3 should retain the exact physical limit");
        assert_eq!(vulkan.grouped_batch, 3);
    }

    #[test]
    fn higher_axis_user_grouped_batch_matches_fixed_upstream_x_tile() {
        let mut device = nvidia_vulkan(48 * 1024);
        device.max_threads_per_block = 1024;
        device.max_workgroup_size = [1024, 1024, 64];

        let n3 =
            plan_gpu_smooth_stockham_uploads_for_batches(3, 40, Precision::F32, device).unwrap();
        let axis1 = plan_gpu_other_axis_single_upload_block_with_grouped_batch(
            &n3,
            8,
            8,
            Some(3),
            Some(3),
            device,
        )
        .unwrap()
        .expect("axis-1 user groupedBatch should override the physical X tile");
        assert_eq!(axis1.threads_per_transform, 1);
        assert_eq!(axis1.grouped_batch, 3);
        assert_eq!([axis1.local_size_x, axis1.local_size_y], [3, 1]);
        assert!(axis1.transforms_on_x);
        assert!(!axis1.axis_swapped);

        let n64 =
            plan_gpu_power_of_two_stockham_uploads_for_batches(64, 64, Precision::F32, device)
                .unwrap();
        let user = plan_gpu_other_axis_single_upload_block_with_grouped_batch(
            &n64,
            64,
            8,
            Some(7),
            Some(7),
            device,
        )
        .unwrap()
        .expect("user groupedBatch should run before the default NVIDIA aimThreads shrink");
        assert_eq!(user.threads_per_transform, 8);
        assert_eq!(user.grouped_batch, 7);
        assert_eq!([user.local_size_x, user.local_size_y], [7, 8]);

        // Fixed upstream literally gates axes >=2 against groupedBatch[1] and then
        // assigns groupedBatch[axis_id]. Preserve that quirk instead of flattening
        // the full inner line count into the physical X tile.
        let axis2 = plan_gpu_other_axis_single_upload_block_with_grouped_batch(
            &n64,
            8,
            8,
            Some(5),
            Some(3),
            device,
        )
        .unwrap()
        .expect("axis-2 groupedBatch should use the fixed upstream axis-1 gate");
        assert_eq!(axis2.grouped_batch, 5);
        assert_eq!([axis2.local_size_x, axis2.local_size_y], [5, 8]);
    }

    #[test]
    fn higher_axis_four_step_blocks_keep_transforms_on_x_across_uploads() {
        let mut device = nvidia_vulkan(32 * 1024);
        device.shared_memory_pow2_bytes = 32 * 1024;
        device.max_threads_per_block = 1024;
        device.max_workgroup_size = [1024, 1024, 64];

        let schedule =
            plan_gpu_smooth_stockham_uploads_for_batches(6_144, 40, Precision::F32, device)
                .unwrap();
        assert_eq!(schedule.upload_count, 2);
        assert_eq!(schedule.axis_split, vec![96, 64]);

        let upload1 = plan_gpu_other_axis_four_step_upload_block(&schedule, 1, 3_840, 8, 8, device)
            .unwrap()
            .expect("higher-axis upload1 should retain an executable X/Y block");
        let upload0 = plan_gpu_other_axis_four_step_upload_block(&schedule, 0, 2_560, 8, 8, device)
            .unwrap()
            .expect("higher-axis upload0 should retain an executable X/Y block");
        assert_eq!(upload1.grouped_batch, 8);
        assert_eq!(upload0.grouped_batch, 8);
        assert_eq!([upload1.local_size_x, upload1.local_size_y], [8, 8]);
        assert_eq!([upload0.local_size_x, upload0.local_size_y], [8, 8]);
        assert!(upload1.transforms_on_x && upload0.transforms_on_x);
        assert!(!upload1.axis_swapped && !upload0.axis_swapped);

        let user1 = plan_gpu_other_axis_four_step_upload_block_with_grouped_batch(
            &schedule,
            1,
            3_840,
            8,
            8,
            Some(3),
            Some(3),
            device,
        )
        .unwrap()
        .expect("higher-axis upload1 should preserve groupedBatch=3");
        let user0 = plan_gpu_other_axis_four_step_upload_block_with_grouped_batch(
            &schedule,
            0,
            2_560,
            8,
            8,
            Some(3),
            Some(3),
            device,
        )
        .unwrap()
        .expect("higher-axis upload0 should preserve groupedBatch=3");
        assert_eq!([user1.grouped_batch, user0.grouped_batch], [3, 3]);
        assert_eq!([user1.local_size_x, user1.local_size_y], [3, 8]);
        assert_eq!([user0.local_size_x, user0.local_size_y], [3, 8]);
        user1.validate(3_840, device).unwrap();
        user0.validate(2_560, device).unwrap();
    }

    #[test]
    fn higher_axis_four_step_blocks_cover_three_upload_stockham() {
        let mut device = nvidia_vulkan(32 * 1024);
        device.shared_memory_pow2_bytes = 32 * 1024;
        device.max_threads_per_block = 1024;
        device.max_workgroup_size = [1024, 1024, 64];

        let schedule =
            plan_gpu_smooth_stockham_uploads_for_batches(1_572_864, 8, Precision::F32, device)
                .unwrap();
        assert_eq!(schedule.upload_count, 3);
        assert_eq!(schedule.axis_split, vec![128, 96, 128]);
        let counts = [98_304usize, 131_072, 98_304];
        let expected_y = [16usize, 8, 16];
        for ((axis_upload_id, transform_count), expected_y) in
            [2usize, 1, 0].into_iter().zip(counts).zip(expected_y)
        {
            let block = plan_gpu_other_axis_four_step_upload_block_with_grouped_batch(
                &schedule,
                axis_upload_id,
                transform_count,
                8,
                8,
                Some(3),
                Some(3),
                device,
            )
            .unwrap()
            .expect("higher-axis three-upload component should keep an executable block");
            assert_eq!(block.grouped_batch, 3);
            assert_eq!([block.local_size_x, block.local_size_y], [3, expected_y]);
            assert!(block.transforms_on_x);
            assert!(!block.axis_swapped);
            block.validate(transform_count, device).unwrap();
        }
    }

    #[test]
    fn strided_axis_class_changes_default_power_of_two_upload_count() {
        let mut device = nvidia_vulkan(48 * 1024);
        device.shared_memory_pow2_bytes = 32 * 1024;
        device.max_threads_per_block = 1024;
        device.max_workgroup_size = [1024, 1024, 64];

        let contiguous = plan_gpu_power_of_two_stockham_uploads_for_batches_with_axis_context(
            8_192,
            1,
            Precision::F32,
            device,
            StockhamUploadAxisContext::default(),
        )
        .unwrap();
        assert_eq!(contiguous.upload_count, 1);
        assert_eq!(contiguous.axis_split, vec![8_192]);

        let strided = plan_gpu_power_of_two_stockham_uploads_for_batches_with_axis_context(
            8_192,
            1,
            Precision::F32,
            device,
            StockhamUploadAxisContext {
                strided_axis: true,
                ..StockhamUploadAxisContext::default()
            },
        )
        .unwrap();
        assert_eq!(strided.upload_count, 2);
        assert_eq!(strided.axis_split.iter().product::<usize>(), 8_192);
        assert_ne!(strided.axis_split, contiguous.axis_split);
    }

    #[test]
    fn perform_convolution_suppresses_register_boost_capacity_for_n8192() {
        let mut device = nvidia_vulkan(32 * 1024);
        device.shared_memory_pow2_bytes = 32 * 1024;
        device.max_threads_per_block = 1024;
        device.max_workgroup_size = [1024, 1024, 64];

        let ordinary = plan_gpu_power_of_two_stockham_uploads_for_batches_with_axis_context(
            8_192,
            1,
            Precision::F32,
            device,
            StockhamUploadAxisContext::default(),
        )
        .unwrap();
        assert_eq!(ordinary.register_boost, 2);
        assert_eq!(ordinary.upload_count, 1);
        assert_eq!(ordinary.axis_split, vec![8_192]);

        let convolution = plan_gpu_power_of_two_stockham_uploads_for_batches_with_axis_context(
            8_192,
            1,
            Precision::F32,
            device,
            StockhamUploadAxisContext {
                perform_convolution: true,
                ..StockhamUploadAxisContext::default()
            },
        )
        .unwrap();
        assert_eq!(convolution.register_boost, 1);
        assert_eq!(convolution.upload_count, 2);
        assert_eq!(convolution.axis_split, vec![128, 64]);
    }

    #[test]
    fn bluestein_context_forces_register_boost_one_before_upload_scoring() {
        let mut device = nvidia_vulkan(48 * 1024);
        device.shared_memory_pow2_bytes = 32 * 1024;
        device.max_threads_per_block = 1024;
        device.max_workgroup_size = [1024, 1024, 64];

        let ordinary = plan_gpu_power_of_two_stockham_uploads_for_batches_with_axis_context(
            16_384,
            1,
            Precision::F32,
            device,
            StockhamUploadAxisContext::default(),
        )
        .unwrap();
        assert_eq!(ordinary.register_boost, 4);
        assert_eq!(ordinary.upload_count, 1);
        assert_eq!(ordinary.axis_split, vec![16_384]);

        let bluestein = plan_gpu_power_of_two_stockham_uploads_for_batches_with_axis_context(
            16_384,
            1,
            Precision::F32,
            device,
            StockhamUploadAxisContext {
                use_bluestein_fft: true,
                ..StockhamUploadAxisContext::default()
            },
        )
        .unwrap();
        assert_eq!(bluestein.register_boost, 1);
        assert_eq!(bluestein.upload_count, 2);
        assert_eq!(bluestein.axis_split, vec![256, 64]);
    }

    #[test]
    fn bluestein_unit_stride_pow2_split_uses_shared_first_stage() {
        let mut device = nvidia_vulkan(48 * 1024);
        device.shared_memory_pow2_bytes = 32 * 1024;
        device.max_threads_per_block = 1024;
        device.max_workgroup_size = [1024, 1024, 64];
        let context = StockhamUploadAxisContext {
            use_bluestein_fft: true,
            ..StockhamUploadAxisContext::default()
        };

        let two = plan_gpu_power_of_two_stockham_uploads_for_batches_with_axis_context(
            65_536,
            1,
            Precision::F32,
            device,
            context,
        )
        .unwrap();
        assert_eq!(two.register_boost, 1);
        assert_eq!(two.upload_count, 2);
        assert_eq!(two.axis_split, vec![1_024, 64]);

        let three = plan_gpu_power_of_two_stockham_uploads_for_batches_with_axis_context(
            8_388_608,
            1,
            Precision::F32,
            device,
            context,
        )
        .unwrap();
        assert_eq!(three.register_boost, 1);
        assert_eq!(three.upload_count, 3);
        assert_eq!(three.axis_split, vec![4_096, 32, 64]);
    }

    #[test]
    fn strided_bandwidth_boost_reduces_upload_count_only_when_profitable() {
        let mut device = nvidia_vulkan(48 * 1024);
        device.shared_memory_pow2_bytes = 32 * 1024;
        device.max_threads_per_block = 1024;
        device.max_workgroup_size = [1024, 1024, 64];
        let length = 2_097_152usize;

        let baseline = plan_gpu_power_of_two_stockham_uploads_for_batches_with_axis_context(
            length,
            1,
            Precision::F32,
            device,
            StockhamUploadAxisContext {
                strided_axis: true,
                ..StockhamUploadAxisContext::default()
            },
        )
        .unwrap();
        assert_eq!(baseline.upload_count, 3);

        let neutral = plan_gpu_power_of_two_stockham_uploads_for_batches_with_axis_context(
            length,
            1,
            Precision::F32,
            device,
            StockhamUploadAxisContext {
                strided_axis: true,
                bandwidth_boost: 1,
                ..StockhamUploadAxisContext::default()
            },
        )
        .unwrap();
        assert_eq!(neutral.upload_count, baseline.upload_count);
        assert_eq!(neutral.axis_split, baseline.axis_split);

        let boosted = plan_gpu_power_of_two_stockham_uploads_for_batches_with_axis_context(
            length,
            1,
            Precision::F32,
            device,
            StockhamUploadAxisContext {
                strided_axis: true,
                bandwidth_boost: 2,
                ..StockhamUploadAxisContext::default()
            },
        )
        .unwrap();
        assert_eq!(boosted.upload_count, 2);
        // Upstream assigns the relaxed half-bandwidth quotient to locAxisSplit[0].
        assert_eq!(boosted.axis_split, vec![2_048, 1_024]);
    }

    #[test]
    fn smooth_strided_bandwidth_boost_reduces_three_upload_to_two() {
        let mut device = nvidia_vulkan(48 * 1024);
        device.shared_memory_pow2_bytes = 32 * 1024;
        device.max_threads_per_block = 1024;
        device.max_workgroup_size = [1024, 1024, 64];
        let length = 3_145_728usize;

        let baseline = plan_gpu_smooth_stockham_uploads_for_batches_with_axis_context(
            length,
            1,
            Precision::F32,
            device,
            StockhamUploadAxisContext {
                strided_axis: true,
                ..StockhamUploadAxisContext::default()
            },
        )
        .unwrap();
        assert_eq!(baseline.upload_count, 3);
        assert_eq!(baseline.axis_split, vec![192, 128, 128]);

        let boosted = plan_gpu_smooth_stockham_uploads_for_batches_with_axis_context(
            length,
            1,
            Precision::F32,
            device,
            StockhamUploadAxisContext {
                strided_axis: true,
                bandwidth_boost: 2,
                ..StockhamUploadAxisContext::default()
            },
        )
        .unwrap();
        assert_eq!(boosted.upload_count, 2);
        // Only locAxisSplit[0] may consume the relaxed half-bandwidth capacity.
        assert_eq!(boosted.axis_split, vec![2_048, 1_536]);
    }

    #[test]
    fn four_step_axis_blocks_match_upstream_two_and_three_upload_geometry() {
        let mut device = nvidia_vulkan(48 * 1024);
        device.max_threads_per_block = 1024;
        device.max_workgroup_size = [1024, 1024, 64];

        let two = plan_gpu_power_of_two_stockham_uploads_for_batches(
            1_048_576,
            1,
            Precision::F32,
            device,
        )
        .unwrap();
        assert_eq!(two.axis_split, vec![1024, 1024]);
        let upload1 = plan_gpu_axis0_four_step_upload_block(&two, 1, 1024, 1024, 1, 8, device)
            .unwrap()
            .expect("two-upload high pass should group Four-step transforms");
        let upload0 = plan_gpu_axis0_four_step_upload_block(&two, 0, 1, 1024, 1, 8, device)
            .unwrap()
            .expect("two-upload low pass should group Four-step transforms");
        for block in [upload1, upload0] {
            assert_eq!(block.threads_per_transform, 128);
            assert_eq!(block.grouped_batch, 4);
            assert!(block.transforms_on_x);
            assert_eq!([block.local_size_x, block.local_size_y], [4, 128]);
            block.validate(1024, device).unwrap();
        }
        assert!(!upload1.axis_swapped);
        assert!(upload0.axis_swapped);

        let user_upload1 = plan_gpu_axis0_four_step_upload_block_with_grouped_batch(
            &two,
            FourStepAxisBlockRequest {
                axis_upload_id: 1,
                stage_start_size: 1024,
                transform_count: 1024,
                outer_batch_count: 1,
                perform_zero_padding: false,
                grouped_batch_override: Some(2),
            },
            8,
            device,
        )
        .unwrap()
        .expect("two-upload user branch should keep upload1 grouped at four");
        let user_upload0 = plan_gpu_axis0_four_step_upload_block_with_grouped_batch(
            &two,
            FourStepAxisBlockRequest {
                axis_upload_id: 0,
                stage_start_size: 1,
                transform_count: 1024,
                outer_batch_count: 1,
                perform_zero_padding: false,
                grouped_batch_override: Some(2),
            },
            8,
            device,
        )
        .unwrap()
        .expect("two-upload user branch should cap upload0 at groupedBatch=2");
        assert_eq!(user_upload1.grouped_batch, 4);
        assert_eq!(
            [user_upload1.local_size_x, user_upload1.local_size_y],
            [4, 128]
        );
        assert!(!user_upload1.axis_swapped);
        assert_eq!(user_upload0.grouped_batch, 2);
        assert_eq!(
            [user_upload0.local_size_x, user_upload0.local_size_y],
            [2, 128]
        );
        assert!(user_upload0.axis_swapped);

        let three = plan_gpu_power_of_two_stockham_uploads_for_batches(
            8_388_608,
            1,
            Precision::F32,
            device,
        )
        .unwrap();
        assert_eq!(three.axis_split, vec![256, 128, 256]);
        let upload2 =
            plan_gpu_axis0_four_step_upload_block(&three, 2, 32_768, 32_768, 1, 8, device)
                .unwrap()
                .expect("three-upload high pass should group Four-step transforms");
        let upload1 = plan_gpu_axis0_four_step_upload_block(&three, 1, 256, 65_536, 1, 8, device)
            .unwrap()
            .expect("three-upload middle pass should group Four-step transforms");
        let upload0 = plan_gpu_axis0_four_step_upload_block(&three, 0, 1, 32_768, 1, 8, device)
            .unwrap()
            .expect("three-upload low pass should group Four-step transforms");
        assert_eq!(
            [
                upload2.grouped_batch,
                upload1.grouped_batch,
                upload0.grouped_batch,
            ],
            [16, 16, 16]
        );
        assert_eq!(
            [
                [upload2.local_size_x, upload2.local_size_y],
                [upload1.local_size_x, upload1.local_size_y],
                [upload0.local_size_x, upload0.local_size_y],
            ],
            [[16, 32], [16, 16], [16, 32]]
        );
        assert!(!upload2.axis_swapped && !upload1.axis_swapped && upload0.axis_swapped);
        assert!(upload2.transforms_on_x && upload1.transforms_on_x && upload0.transforms_on_x);

        let convolution_three =
            plan_gpu_power_of_two_stockham_uploads_for_batches_with_axis_context(
                8_388_608,
                1,
                Precision::F32,
                device,
                StockhamUploadAxisContext {
                    perform_convolution: true,
                    ..StockhamUploadAxisContext::default()
                },
            )
            .unwrap();
        assert_eq!(convolution_three.axis_split, vec![4096, 32, 64]);
        let convolution_u0 =
            plan_gpu_axis0_four_step_upload_block(&convolution_three, 0, 1, 2_048, 1, 8, device)
                .unwrap()
                .expect("three-upload convolution upload0 must preserve upstream groupedBatch=1");
        assert_eq!(convolution_u0.threads_per_transform, 512);
        assert_eq!(convolution_u0.grouped_batch, 1);
        assert!(!convolution_u0.transforms_on_x);
        assert!(!convolution_u0.axis_swapped);
        assert_eq!(
            [convolution_u0.local_size_x, convolution_u0.local_size_y],
            [512, 1]
        );
        convolution_u0.validate(2_048, device).unwrap();

        let user_middle = plan_gpu_axis0_four_step_upload_block_with_grouped_batch(
            &three,
            FourStepAxisBlockRequest {
                axis_upload_id: 1,
                stage_start_size: 256,
                transform_count: 65_536,
                outer_batch_count: 1,
                perform_zero_padding: false,
                grouped_batch_override: Some(2),
            },
            8,
            device,
        )
        .unwrap()
        .expect(
            "partial final Four-step workgroups make the fixed non-divisor user branch executable",
        );
        assert_eq!(user_middle.grouped_batch, 12);
        assert_eq!(
            [user_middle.local_size_x, user_middle.local_size_y],
            [12, 16]
        );
        assert_eq!(65_536usize.div_ceil(user_middle.grouped_batch), 5_462);
    }

    #[test]
    fn grouped_four_step_precision_aware_path_preserves_f16_storage_coalescing() {
        let device = DeviceProfile {
            shared_memory_bytes: 32 * 1024,
            shared_memory_pow2_bytes: 32 * 1024,
            max_threads_per_block: 1024,
            max_workgroup_size: [1024, 1024, 64],
            ..DeviceProfile::generic(Backend::Vulkan, GpuVendor::Amd)
        };
        let request = FourStepAxisBlockRequest {
            axis_upload_id: 1,
            stage_start_size: 512,
            transform_count: 2_560,
            outer_batch_count: 5,
            perform_zero_padding: true,
            grouped_batch_override: Some(3),
        };
        let compute_shaped =
            plan_gpu_axis0_four_step_grouped_block_from_shape(2, 272, 17, request, 8, device)
                .unwrap()
                .unwrap();
        assert_eq!(compute_shaped.grouped_batch, 12);
        assert_eq!(
            [compute_shaped.local_size_x, compute_shaped.local_size_y],
            [12, 17]
        );

        let f16_storage = plan_gpu_axis0_four_step_grouped_block_from_shape_for_precision(
            2,
            272,
            17,
            request,
            Precision::F16StorageF32Compute,
            8,
            device,
        )
        .unwrap()
        .unwrap();
        assert_eq!(f16_storage.grouped_batch, 8);
        assert_eq!(
            [f16_storage.local_size_x, f16_storage.local_size_y],
            [8, 17]
        );
        assert!(f16_storage.transforms_on_x);
        assert!(!f16_storage.axis_swapped);
    }

    #[test]
    fn large_power_of_two_split_preserves_non_nvidia_special_branch() {
        let amd = DeviceProfile {
            shared_memory_bytes: 48 * 1024,
            shared_memory_pow2_bytes: 32 * 1024,
            max_threads_per_block: 1024,
            max_workgroup_size: [1024, 1024, 64],
            ..DeviceProfile::generic(Backend::Vulkan, GpuVendor::Amd)
        };
        let amd_schedule =
            plan_gpu_power_of_two_stockham_uploads_for_batches(524_288, 1, Precision::F32, amd)
                .unwrap();
        assert_eq!(amd_schedule.upload_count, 3);
        assert_eq!(amd_schedule.axis_split, vec![64, 128, 64]);

        let intel = DeviceProfile {
            shared_memory_bytes: 48 * 1024,
            shared_memory_pow2_bytes: 32 * 1024,
            max_threads_per_block: 1024,
            max_workgroup_size: [1024, 1024, 64],
            ..DeviceProfile::generic(Backend::Vulkan, GpuVendor::Intel)
        };
        let intel_schedule =
            plan_gpu_power_of_two_stockham_uploads_for_batches(524_288, 1, Precision::F32, intel)
                .unwrap();
        assert_eq!(intel_schedule.upload_count, 3);
        assert_eq!(intel_schedule.axis_split, vec![64, 128, 64]);

        let hip = DeviceProfile {
            shared_memory_bytes: 48 * 1024,
            shared_memory_pow2_bytes: 32 * 1024,
            max_threads_per_block: 1024,
            max_workgroup_size: [1024, 1024, 64],
            ..DeviceProfile::generic(Backend::Hip, GpuVendor::Amd)
        };
        let hip_schedule =
            plan_gpu_power_of_two_stockham_uploads_for_batches(2_097_152, 1, Precision::F32, hip)
                .unwrap();
        assert_eq!(hip_schedule.upload_count, 3);
        assert_eq!(hip_schedule.axis_split, vec![64, 512, 64]);

        // NVIDIA alone switches power-of-two N>262144 to the generic divisor
        // search. At the same 32 KiB pow2 capacity N=524288 remains two-upload;
        // upstream stores the relaxed quotient in locAxisSplit[0]: 1024 x 512.
        let nvidia = DeviceProfile {
            shared_memory_bytes: 48 * 1024,
            shared_memory_pow2_bytes: 32 * 1024,
            max_threads_per_block: 1024,
            max_workgroup_size: [1024, 1024, 64],
            ..DeviceProfile::generic(Backend::Vulkan, GpuVendor::Nvidia)
        };
        let nvidia_schedule =
            plan_gpu_power_of_two_stockham_uploads_for_batches(524_288, 1, Precision::F32, nvidia)
                .unwrap();
        assert_eq!(nvidia_schedule.upload_count, 2);
        assert_eq!(nvidia_schedule.axis_split, vec![1024, 512]);
    }

    #[test]
    fn gpu_scheduler_policy_matches_fixed_upstream_backend_defaults() {
        let amd_48k = DeviceProfile {
            shared_memory_bytes: 48 * 1024,
            shared_memory_pow2_bytes: 32 * 1024,
            ..DeviceProfile::generic(Backend::Vulkan, GpuVendor::Amd)
        };
        let amd_f32 = plan_gpu_scheduler_policy(Precision::F32, amd_48k).unwrap();
        assert_eq!(amd_f32.register_boost, 4);
        assert_eq!(amd_f32.subgroup_width, 64);
        assert_eq!(amd_f32.coalesced_memory_bytes, 32);
        assert_eq!(amd_f32.swap_to_three_stage_four_step, 524_288);
        assert_eq!(
            amd_f32.stockham_twiddle_source,
            StockhamTwiddleSource::OnTheFly
        );
        let amd_f64 = plan_gpu_scheduler_policy(Precision::F64, amd_48k).unwrap();
        assert_eq!(amd_f64.swap_to_three_stage_four_step, 262_144);
        assert_eq!(
            amd_f64.stockham_twiddle_source,
            StockhamTwiddleSource::LookupTable
        );

        let amd_64k = DeviceProfile {
            shared_memory_bytes: 64 * 1024,
            shared_memory_pow2_bytes: 64 * 1024,
            ..amd_48k
        };
        assert_eq!(
            plan_gpu_scheduler_policy(Precision::F32, amd_64k)
                .unwrap()
                .register_boost,
            2
        );

        let cuda = plan_gpu_scheduler_policy(
            Precision::F32,
            DeviceProfile::generic(Backend::Cuda, GpuVendor::Nvidia),
        )
        .unwrap();
        assert_eq!(cuda.register_boost, 1);
        assert_eq!(cuda.swap_to_three_stage_four_step, 4_194_305);
        assert_eq!(cuda.subgroup_width, 32);

        for (backend, vendor) in [
            (Backend::Vulkan, GpuVendor::Amd),
            (Backend::Vulkan, GpuVendor::Intel),
            (Backend::OpenCl, GpuVendor::Amd),
            (Backend::OpenCl, GpuVendor::Intel),
        ] {
            let mut device = DeviceProfile::generic(backend, vendor);
            device.shared_memory_bytes = 64 * 1024;
            device.shared_memory_pow2_bytes = 64 * 1024;
            device.supports_f64 = true;
            let native_f64 = plan_gpu_scheduler_policy(Precision::F64, device).unwrap();
            let mixed = plan_gpu_scheduler_policy(Precision::F64ComputeF32Storage, device).unwrap();
            assert_eq!(native_f64.swap_to_three_stage_four_step, 262_144);
            assert_eq!(mixed.swap_to_three_stage_four_step, 524_288);
            assert_eq!(
                mixed.stockham_twiddle_source,
                StockhamTwiddleSource::LookupTable
            );
        }

        let hip_f64 = plan_gpu_scheduler_policy(
            Precision::F64,
            DeviceProfile::generic(Backend::Hip, GpuVendor::Amd),
        )
        .unwrap();
        assert_eq!(hip_f64.register_boost, 1);
        assert_eq!(hip_f64.swap_to_three_stage_four_step, 1_048_576);
        assert_eq!(hip_f64.subgroup_width, 64);
        let hip_f64_f32_storage = plan_gpu_scheduler_policy(
            Precision::F64ComputeF32Storage,
            DeviceProfile::generic(Backend::Hip, GpuVendor::Amd),
        )
        .unwrap();
        assert_eq!(hip_f64_f32_storage.register_boost, 1);
        assert_eq!(hip_f64_f32_storage.swap_to_three_stage_four_step, 2_097_152);
        assert_eq!(
            hip_f64_f32_storage.stockham_twiddle_source,
            StockhamTwiddleSource::LookupTable
        );
        assert_eq!(
            hip_f64_f32_storage.four_step_twiddle_source,
            StockhamTwiddleSource::OnTheFly
        );
        let hip_dd = plan_gpu_scheduler_policy(
            Precision::DoubleDouble,
            DeviceProfile::generic(Backend::Hip, GpuVendor::Amd),
        )
        .unwrap();
        assert_eq!(hip_dd.swap_to_three_stage_four_step, 1_048_576);
        let mut hip_wave32 = DeviceProfile::generic(Backend::Hip, GpuVendor::Amd);
        hip_wave32.subgroup = crate::SubgroupProfile {
            size: 32,
            min_size: 32,
            max_size: 32,
            required_size_compute_supported: false,
            compute_supported: true,
            basic_supported: true,
            shuffle_supported: true,
            shuffle_relative_supported: true,
            compute_full_subgroups: true,
        };
        assert_eq!(
            plan_gpu_scheduler_policy(Precision::F32, hip_wave32)
                .unwrap()
                .subgroup_width,
            32
        );
        assert_eq!(
            hip_f64.stockham_twiddle_source,
            StockhamTwiddleSource::LookupTable
        );
        assert_eq!(
            hip_f64.four_step_twiddle_source,
            StockhamTwiddleSource::OnTheFly
        );

        let level_zero = plan_gpu_scheduler_policy(
            Precision::F32,
            DeviceProfile::generic(Backend::LevelZero, GpuVendor::Intel),
        )
        .unwrap();
        assert_eq!(level_zero.register_boost, 2);
        assert_eq!(level_zero.coalesced_memory_bytes, 64);
        assert_eq!(
            level_zero.stockham_twiddle_source,
            StockhamTwiddleSource::LookupTable
        );
        let mut level_zero_64k = DeviceProfile::generic(Backend::LevelZero, GpuVendor::Intel);
        level_zero_64k.shared_memory_bytes = 64 * 1024;
        level_zero_64k.shared_memory_pow2_bytes = 64 * 1024;
        let level_zero_f64 = plan_gpu_scheduler_policy(Precision::F64, level_zero_64k).unwrap();
        assert_eq!(level_zero_f64.register_boost, 1);
        assert_eq!(level_zero_f64.swap_to_three_stage_four_step, 262_144);
        let level_zero_f64_f32_storage =
            plan_gpu_scheduler_policy(Precision::F64ComputeF32Storage, level_zero_64k).unwrap();
        assert_eq!(level_zero_f64_f32_storage.register_boost, 1);
        assert_eq!(
            level_zero_f64_f32_storage.swap_to_three_stage_four_step,
            524_288
        );
        assert_eq!(
            level_zero_f64_f32_storage.stockham_twiddle_source,
            StockhamTwiddleSource::LookupTable
        );

        let metal = plan_gpu_scheduler_policy(
            Precision::F32,
            DeviceProfile::generic(Backend::Metal, GpuVendor::Apple),
        )
        .unwrap();
        assert_eq!(metal.register_boost, 1);
        assert_eq!(metal.coalesced_memory_bytes, 64);
        assert_eq!(
            metal.stockham_twiddle_source,
            StockhamTwiddleSource::OnTheFly
        );
    }

    #[test]
    fn amd_vulkan_policy_drives_register_boost_uploads_and_rader_grouping() {
        let mut amd_48k = DeviceProfile {
            shared_memory_bytes: 48 * 1024,
            shared_memory_pow2_bytes: 32 * 1024,
            max_threads_per_block: 1024,
            ..DeviceProfile::generic(Backend::Vulkan, GpuVendor::Amd)
        };
        let boost4 =
            plan_gpu_power_of_two_stockham_uploads(16_384, Precision::F32, amd_48k).unwrap();
        assert_eq!(boost4.upload_count, 1);
        assert_eq!(boost4.register_boost, 4);

        amd_48k.shared_memory_bytes = 64 * 1024;
        amd_48k.shared_memory_pow2_bytes = 64 * 1024;
        let boost2 =
            plan_gpu_power_of_two_stockham_uploads(16_384, Precision::F32, amd_48k).unwrap();
        assert_eq!(boost2.upload_count, 1);
        assert_eq!(boost2.register_boost, 2);

        let rader = plan_gpu_rader_fft_registers_for_containers(19, 8, 152, 8, amd_48k).unwrap();
        assert_eq!(rader.container_fft_num, 8);
        assert_eq!(rader.internal_fft.stage_radices, vec![6, 3]);
        assert!(rader.rader_transpose.is_some());
        assert!(rader.upstream_grouping_is_executable());

        let large_f64 =
            plan_gpu_power_of_two_stockham_uploads(262_144, Precision::F64, amd_48k).unwrap();
        assert_eq!(large_f64.upload_count, 3);
    }

    #[test]
    fn nvidia_stockham_twiddle_policy_matches_upstream_precision_defaults() {
        let mut device = nvidia_vulkan(48 * 1024);
        device.supports_f64 = true;
        assert_eq!(
            plan_nvidia_vulkan_stockham_twiddle_source(Precision::F32, device),
            StockhamTwiddleSource::OnTheFly
        );
        assert_eq!(
            plan_nvidia_vulkan_stockham_twiddle_source(Precision::F64, device),
            StockhamTwiddleSource::LookupTable
        );
        let amd = DeviceProfile::generic(Backend::Vulkan, GpuVendor::Amd);
        assert_eq!(
            plan_nvidia_vulkan_stockham_twiddle_source(Precision::F64, amd),
            StockhamTwiddleSource::OnTheFly
        );
    }

    #[test]
    fn force_rader_two_upload_matches_upstream_container_pressure_gate() {
        let mut device = nvidia_vulkan(48 * 1024);
        device.max_threads_per_block = 1024;
        assert!(!plan_nvidia_vulkan_force_rader_two_upload(17 * 512, &[17], device).unwrap());
        assert!(plan_nvidia_vulkan_force_rader_two_upload(17 * 513, &[17], device).unwrap());

        let mut narrow = device;
        narrow.max_threads_per_block = 128;
        assert!(!plan_nvidia_vulkan_force_rader_two_upload(19 * 128, &[19], narrow).unwrap());
        assert!(plan_nvidia_vulkan_force_rader_two_upload(19 * 129, &[19], narrow).unwrap());
        assert!(!plan_nvidia_vulkan_force_rader_two_upload(47 * 1024, &[], device).unwrap());
    }

    #[test]
    fn normal_capacity_rader_split_precedes_force_promotion() {
        let mut device = nvidia_vulkan(32 * 1024);
        device.max_threads_per_block = 1024;
        device.max_workgroup_size = [1024, 1024, 64];

        // Pinned upstream: ordinary F32 Direct-Rader capacity already requires two
        // uploads even though no FFT-Rader container triggers forceRaderTwoUpload.
        let sequence_len = 3usize * 29 * 47;
        assert_eq!(sequence_len, 4_089);
        assert!(!plan_gpu_force_rader_two_upload(sequence_len, &[], device).unwrap());
        let direct = plan_gpu_rader_upload_split_with_axis_context(
            sequence_len,
            &[],
            &[29, 47],
            Precision::F32,
            device,
            StockhamUploadAxisContext::default(),
        )
        .unwrap()
        .expect("normal Direct-Rader capacity must enter the common multi-upload path");
        assert_eq!(direct.upload_count, 2);
        assert_eq!(direct.axis_split, vec![87, 47]);
        assert_eq!(direct.reason, RaderUploadReason::CapacityOrBandwidth);

        // Pinned upstream DD default: p11/p23 Direct-Rader reserves reduce the usable
        // 32 KiB capacity enough that N1012 splits as 44 x 23 without a force flag.
        let dd_sequence_len = 4usize * 11 * 23;
        assert!(!plan_gpu_force_rader_two_upload(dd_sequence_len, &[], device).unwrap());
        let dd = plan_gpu_rader_upload_split_with_axis_context(
            dd_sequence_len,
            &[],
            &[11, 23],
            Precision::DoubleDouble,
            device,
            StockhamUploadAxisContext::default(),
        )
        .unwrap()
        .expect("normal DD Direct-Rader capacity must enter the common multi-upload path");
        assert_eq!(dd.upload_count, 2);
        assert_eq!(dd.axis_split, vec![44, 23]);
        assert_eq!(dd.reason, RaderUploadReason::CapacityOrBandwidth);

        // Pure repeated FFT-Rader is the complementary witness: there is no Direct
        // reservation, but N8303 exceeds the ordinary 48 KiB/F32 shared capacity.
        let mut wide = nvidia_vulkan(48 * 1024);
        wide.max_threads_per_block = 1024;
        wide.max_workgroup_size = [1024, 1024, 64];
        let repeated = 19usize * 19 * 23;
        assert!(!plan_gpu_force_rader_two_upload(repeated, &[19, 23], wide).unwrap());
        let fft = plan_gpu_rader_upload_split_with_axis_context(
            repeated,
            &[19, 23],
            &[],
            Precision::F32,
            wide,
            StockhamUploadAxisContext::default(),
        )
        .unwrap()
        .expect("normal FFT-Rader capacity must enter the common multi-upload path");
        assert_eq!(fft.upload_count, 2);
        assert_eq!(fft.axis_split, vec![361, 23]);
        assert_eq!(fft.reason, RaderUploadReason::CapacityOrBandwidth);
    }

    #[test]
    fn f16_rader_failed_two_pass_divisor_promotes_to_three() {
        let mut device = nvidia_vulkan(8 * 1024);
        device.shared_memory_pow2_bytes = 8 * 1024;
        device.max_threads_per_block = 1024;
        device.max_workgroup_size = [1024, 1024, 64];
        let sequence_len = 11usize * 17 * 47;
        let schedule = plan_gpu_rader_upload_split_with_axis_context(
            sequence_len,
            &[],
            &[17, 47],
            Precision::F16StorageF32Compute,
            device,
            StockhamUploadAxisContext::default(),
        )
        .unwrap()
        .expect("F16 Rader capacity must promote the failed two-pass split to three uploads");
        assert_eq!(schedule.upload_count, 3);
        assert_eq!(schedule.axis_split, vec![47, 17, 11]);
        assert_eq!(schedule.reason, RaderUploadReason::CapacityOrBandwidth);
    }

    #[test]
    fn rader_upload_skips_when_pass_local_direct_reservation_exceeds_shared_memory() {
        let mut constrained = nvidia_vulkan(256);
        constrained.shared_memory_pow2_bytes = 256;
        let schedule = plan_gpu_rader_upload_split_with_axis_context(
            106,
            &[],
            &[53],
            Precision::F32,
            constrained,
            StockhamUploadAxisContext::default(),
        )
        .unwrap();
        assert_eq!(schedule, None);
    }

    #[test]
    fn forced_rader_upload_uses_upstream_non_power_of_two_divisor_geometry() {
        let mut device = nvidia_vulkan(32 * 1024);
        device.max_threads_per_block = 1024;
        let schedule =
            plan_gpu_rader_forced_upload_split(17 * 8192, &[17], &[], Precision::F32, device)
                .unwrap()
                .expect("p17 across 8192 containers must enter the forced two-upload branch");
        assert_eq!(schedule.max_sequence_len_shared, 4096);
        assert_eq!(schedule.max_sequence_len_strided, 1024);
        assert_eq!(schedule.upload_count, 2);
        assert_eq!(schedule.axis_split, vec![512, 272]);
        assert_eq!(schedule.reason, RaderUploadReason::CapacityOrBandwidth);
        schedule.validate().unwrap();

        // The force gate is still false at exactly 512 containers, but N8704 already
        // exceeds the ordinary 32 KiB/F32 capacity. Fixed upstream therefore keeps the
        // normal two-upload result rather than returning to one pass.
        assert!(!plan_gpu_force_rader_two_upload(17 * 512, &[17], device).unwrap());
        let capacity =
            plan_gpu_rader_forced_upload_split(17 * 512, &[17], &[], Precision::F32, device)
                .unwrap()
                .expect("normal capacity must split N8704 even below the force threshold");
        assert_eq!(capacity.upload_count, 2);
        assert_eq!(capacity.axis_split, vec![128, 68]);
        assert_eq!(capacity.reason, RaderUploadReason::CapacityOrBandwidth);

        let mut thread_limited = nvidia_vulkan(48 * 1024);
        thread_limited.max_threads_per_block = 128;
        let schedule = plan_gpu_rader_forced_upload_split(
            17 * 300,
            &[17],
            &[],
            Precision::F32,
            thread_limited,
        )
        .unwrap()
        .expect(
            "thread pressure should force a two-upload split even below the 512-container gate",
        );
        assert_eq!(schedule.upload_count, 2);
        assert_eq!(schedule.axis_split, vec![68, 75]);
        assert_eq!(schedule.reason, RaderUploadReason::ForcedRaderPressure);

        let schedule = plan_gpu_rader_forced_upload_split(
            17 * 65_536,
            &[17],
            &[],
            Precision::F32,
            device,
        )
        .unwrap()
        .expect(
            "a forced Rader axis above the two-factor capacity should promote to three uploads",
        );
        assert_eq!(schedule.upload_count, 3);
        assert_eq!(schedule.axis_split, vec![128, 68, 128]);
        schedule.validate().unwrap();

        let mut intel = DeviceProfile::generic(Backend::Vulkan, GpuVendor::Intel);
        intel.shared_memory_bytes = 32 * 1024;
        intel.shared_memory_pow2_bytes = 32 * 1024;
        intel.max_threads_per_block = 1024;
        intel.max_workgroup_size = [1024, 1024, 64];
        let sequence_len = 17 * 16_384;
        let strided = plan_gpu_rader_upload_split_with_axis_context(
            sequence_len,
            &[17],
            &[],
            Precision::F32,
            intel,
            StockhamUploadAxisContext {
                strided_axis: true,
                ..StockhamUploadAxisContext::default()
            },
        )
        .unwrap()
        .expect("Intel strided p17 composite must retain forced-Rader scheduling");
        assert_eq!(strided.max_sequence_len_shared, 4_096);
        assert_eq!(strided.max_sequence_len_strided, 512);
        assert_eq!(strided.upload_count, 3);
        assert_eq!(strided.axis_split, vec![64, 68, 64]);

        let boosted = plan_gpu_rader_upload_split_with_axis_context(
            sequence_len,
            &[17],
            &[],
            Precision::F32,
            intel,
            StockhamUploadAxisContext {
                strided_axis: true,
                bandwidth_boost: 2,
                ..StockhamUploadAxisContext::default()
            },
        )
        .unwrap()
        .expect("B=2 must keep the forced-Rader branch while reducing its upload count");
        assert_eq!(boosted.upload_count, 2);
        assert_eq!(boosted.axis_split, vec![544, 512]);
        boosted.validate().unwrap();
    }

    #[test]
    fn forced_rader_upload_uses_quad_compute_footprint_for_double_double() {
        let mut device = nvidia_vulkan(32 * 1024);
        device.max_threads_per_block = 1024;
        for precision in [Precision::DoubleDouble, Precision::DoubleDoubleF64Storage] {
            let schedule =
                plan_gpu_rader_forced_upload_split(17 * 8192, &[17], &[], precision, device)
                    .unwrap()
                    .expect("DD p17 across 8192 containers must enter the forced upload branch");
            assert_eq!(schedule.used_shared_memory_bytes, 32 * 1024);
            assert_eq!(schedule.max_sequence_len_shared, 1024);
            assert_eq!(schedule.max_sequence_len_strided, 1024);
            assert_eq!(schedule.upload_count, 2);
            assert_eq!(schedule.axis_split, vec![512, 272]);
            schedule.validate().unwrap();
        }
    }

    #[test]
    fn composite_direct_rader_thread_estimate_matches_upstream_type1_coupling() {
        let device = nvidia_vulkan(48 * 1024);
        assert_eq!(
            plan_gpu_axis0_composite_direct_rader_threads(2 * 47, 47, 5, device).unwrap(),
            Some(48)
        );
        assert_eq!(
            plan_gpu_axis0_composite_direct_rader_threads(4 * 47, 47, 5, device).unwrap(),
            Some(48)
        );
        assert_eq!(
            plan_gpu_axis0_composite_direct_rader_threads(47, 47, 5, device).unwrap(),
            None
        );

        let mut too_narrow = device;
        too_narrow.max_threads_per_block = 32;
        too_narrow.max_workgroup_size[0] = 32;
        // This helper starts after algorithm classification. If called directly on
        // an otherwise-impossible 32-thread p47 profile, upstream's register scaling
        // can still compress the physical estimate from 48 to 24 lanes; the earlier
        // direct-prime thread cap is what prevents the real planner from selecting p47.
        assert_eq!(
            plan_gpu_axis0_composite_direct_rader_threads(2 * 47, 47, 5, too_narrow).unwrap(),
            Some(24)
        );
    }

    #[test]
    fn multi_direct_rader_thread_estimate_shares_upstream_register_scale() {
        let device = nvidia_vulkan(48 * 1024);
        // The default synthetic NVIDIA profile is 256-thread limited. The shared
        // upstream scale loop therefore advances until both type-1 containers fit at
        // 216 lanes. A 1024-thread profile stops earlier at 432 lanes.
        assert_eq!(
            plan_gpu_axis0_composite_direct_rader_threads_for_primes(
                47 * 53,
                &[47, 53],
                2,
                device,
            )
            .unwrap(),
            Some(216)
        );
        let mut wide = device;
        wide.max_threads_per_block = 1024;
        wide.max_workgroup_size[0] = 1024;
        assert_eq!(
            plan_gpu_axis0_composite_direct_rader_threads_for_primes(47 * 53, &[47, 53], 2, wide,)
                .unwrap(),
            Some(432)
        );
    }

    #[test]
    fn single_fft_rader_smooth_outer_parent_uses_shared_optimizer() {
        let mut device = nvidia_vulkan(48 * 1024);
        device.max_threads_per_block = 1024;
        device.max_workgroup_size[0] = 1024;
        assert_eq!(
            plan_gpu_axis0_multi_fft_rader_threads_for_prime_multiplicities(
                4 * 17,
                &[(17, 1)],
                75,
                device,
            )
            .unwrap(),
            Some(5)
        );
        assert_eq!(
            plan_gpu_axis0_multi_fft_rader_threads_for_prime_multiplicities(
                16 * 17,
                &[(17, 1)],
                512,
                device,
            )
            .unwrap(),
            Some(17)
        );
    }

    #[test]
    fn double_double_single_fft_rader_smooth_outer_uses_quad_parent_state() {
        let mut device = nvidia_vulkan(48 * 1024);
        device.max_threads_per_block = 1024;
        device.max_workgroup_size[0] = 1024;
        assert_eq!(
            plan_gpu_double_double_axis0_fft_rader_threads_for_prime_multiplicities(
                4 * 17,
                &[(17, 1)],
                75,
                device,
            )
            .unwrap(),
            Some(5)
        );
        assert_eq!(
            plan_gpu_double_double_axis0_fft_rader_threads_for_prime_multiplicities(
                16 * 17,
                &[(17, 1)],
                512,
                device,
            )
            .unwrap(),
            Some(17)
        );
        assert_eq!(
            plan_gpu_double_double_axis0_fft_rader_threads_for_prime_multiplicities(
                19 * 19 * 23,
                &[(19, 2), (23, 1)],
                1,
                device,
            )
            .unwrap(),
            Some(437)
        );
    }

    #[test]
    fn four_step_shape_splitter_matches_stockham_physical_rules_across_vendors() {
        for (backend, vendor) in [
            (Backend::Vulkan, GpuVendor::Nvidia),
            (Backend::Vulkan, GpuVendor::Amd),
            (Backend::OpenCl, GpuVendor::Intel),
        ] {
            let device = DeviceProfile {
                shared_memory_bytes: 4 * 1024,
                shared_memory_pow2_bytes: 4 * 1024,
                max_threads_per_block: 256,
                max_workgroup_size: [256, 256, 64],
                supports_f64: true,
                ..DeviceProfile::generic(backend, vendor)
            };
            let length = 8192usize;
            let upload = plan_gpu_smooth_stockham_uploads(length, Precision::F32, device).unwrap();
            assert!(matches!(upload.upload_count, 2 | 3));
            let mut stage_start_size = 1usize;
            let mut compared = 0usize;
            for axis_upload_id in 0..upload.upload_count {
                let fft_len = upload.axis_split[axis_upload_id];
                let transform_count = length / fft_len;
                let stockham = plan_gpu_axis0_four_step_upload_block(
                    &upload,
                    axis_upload_id,
                    stage_start_size,
                    transform_count,
                    1,
                    8,
                    device,
                )
                .unwrap();
                if let Some(stockham) = stockham
                    && stockham.grouped_batch > 1
                {
                    let shape = plan_gpu_axis0_four_step_default_block_from_shape(
                        upload.upload_count,
                        fft_len,
                        stockham.threads_per_transform,
                        FourStepAxisBlockRequest {
                            axis_upload_id,
                            stage_start_size,
                            transform_count,
                            outer_batch_count: 1,
                            perform_zero_padding: false,
                            grouped_batch_override: None,
                        },
                        8,
                        device,
                    )
                    .unwrap();
                    assert_eq!(
                        shape,
                        Some(stockham),
                        "shape-only splitter diverged for {backend:?}/{vendor:?} upload {axis_upload_id}"
                    );
                    compared += 1;
                }
                stage_start_size *= fft_len;
            }
            assert!(
                compared > 0,
                "no grouped Four-step upload exercised for {vendor:?}"
            );
        }
    }

    #[test]
    fn four_step_shape_splitter_uses_precision_specific_coalescing_width() {
        let mut device = nvidia_vulkan(48 * 1024);
        device.shared_memory_pow2_bytes = 32 * 1024;
        device.max_threads_per_block = 128;
        device.max_workgroup_size = [128, 128, 64];
        let request = FourStepAxisBlockRequest {
            axis_upload_id: 1,
            stage_start_size: 75,
            transform_count: 75,
            outer_batch_count: 1,
            perform_zero_padding: false,
            grouped_batch_override: None,
        };
        let ordinary =
            plan_gpu_axis0_four_step_default_block_from_shape(2, 68, 5, request, 8, device)
                .unwrap()
                .unwrap();
        assert_eq!(ordinary.threads_per_transform, 5);
        assert_eq!(ordinary.grouped_batch, 16);
        assert_eq!([ordinary.local_size_x, ordinary.local_size_y], [16, 5]);
        assert!(ordinary.transforms_on_x);
        assert!(!ordinary.axis_swapped);

        let quad = plan_gpu_axis0_four_step_default_block_from_shape(2, 68, 5, request, 32, device)
            .unwrap()
            .unwrap();
        assert_eq!(quad.threads_per_transform, 5);
        assert_eq!(quad.grouped_batch, 4);
        assert_eq!([quad.local_size_x, quad.local_size_y], [4, 5]);
        assert!(quad.transforms_on_x);
        assert!(!quad.axis_swapped);
    }

    #[test]
    fn pass_local_coalescing_changes_pure_type0_and_direct_only_parent_floors() {
        let mut wide = nvidia_vulkan(48 * 1024);
        wide.max_threads_per_block = 1024;
        wide.max_workgroup_size[0] = 1024;
        // Existing forced components stay unchanged when the workgroup is wide enough.
        assert_eq!(
            plan_gpu_axis0_multi_fft_rader_threads_for_prime_multiplicities_with_max_batch_coalesced(
                4 * 17,
                &[(17, 1)],
                75,
                4,
                wide,
            )
            .unwrap(),
            Some(5)
        );
        assert_eq!(
            plan_gpu_axis0_composite_direct_rader_threads_for_prime_multiplicities_with_max_batch_coalesced(
                2 * 17,
                &[(17, 1)],
                32,
                4,
                wide,
            )
            .unwrap(),
            Some(18)
        );

        // Under a narrow pass-local cap, upstream raises the register floor to make
        // maxBatchCoalesced transforms fit. Direct type-1 therefore drops from 18 to 9.
        let mut narrow = wide;
        narrow.max_threads_per_block = 64;
        narrow.max_workgroup_size[0] = 64;
        assert_eq!(
            plan_gpu_axis0_composite_direct_rader_threads_for_prime_multiplicities_with_max_batch_coalesced(
                2 * 17,
                &[(17, 1)],
                32,
                1,
                narrow,
            )
            .unwrap(),
            Some(18)
        );
        assert_eq!(
            plan_gpu_axis0_composite_direct_rader_threads_for_prime_multiplicities_with_max_batch_coalesced(
                2 * 17,
                &[(17, 1)],
                32,
                4,
                narrow,
            )
            .unwrap(),
            Some(9)
        );

        // Type-1 Direct-Rader contributes its fixed two-register container minimum
        // to the parent before scale_registers_rader. N55=5*p11 therefore reaches
        // a 30-lane parent even though the isolated radix-5 Quad table has min=5.
        assert_eq!(
            plan_gpu_axis0_composite_direct_rader_threads_for_prime_multiplicities_with_max_batch_coalesced(
                5 * 11,
                &[(11, 1)],
                125,
                1,
                wide,
            )
            .unwrap(),
            Some(30)
        );
        assert_eq!(
            plan_gpu_double_double_axis0_composite_direct_rader_threads_for_prime_multiplicities_with_max_batch_coalesced(
                5 * 11,
                &[(11, 1)],
                125,
                1,
                wide,
            )
            .unwrap(),
            Some(30)
        );

        // Pure type-0 has no direct active-container rounding, but scaleRegistersNum
        // still sees the coalesced parent pressure and moves N272 from 17 to 16 lanes.
        assert_eq!(
            plan_gpu_axis0_multi_fft_rader_threads_for_prime_multiplicities_with_max_batch_coalesced(
                16 * 17,
                &[(17, 1)],
                512,
                1,
                narrow,
            )
            .unwrap(),
            Some(17)
        );
        assert_eq!(
            plan_gpu_axis0_multi_fft_rader_threads_for_prime_multiplicities_with_max_batch_coalesced(
                16 * 17,
                &[(17, 1)],
                512,
                4,
                narrow,
            )
            .unwrap(),
            Some(16)
        );
        // Scheduler k==0 is special: after the initial coalesced-pressure test,
        // fixed upstream shrinks maxBatchCoalesced before scaleRegistersNum. The same
        // N272 pass therefore stays at the one-transform 17-lane floor for upload0,
        // while a later upload keeps all four coalesced transforms and scales to 16.
        assert_eq!(
            plan_gpu_axis0_multi_fft_rader_threads_for_forced_upload(
                16 * 17,
                &[(17, 1)],
                512,
                4,
                0,
                narrow,
            )
            .unwrap(),
            Some(17)
        );
        assert_eq!(
            plan_gpu_axis0_multi_fft_rader_threads_for_forced_upload(
                16 * 17,
                &[(17, 1)],
                512,
                4,
                1,
                narrow,
            )
            .unwrap(),
            Some(16)
        );

        let mut medium = wide;
        medium.max_threads_per_block = 128;
        medium.max_workgroup_size[0] = 128;
        assert_eq!(
            plan_gpu_axis0_multi_fft_rader_threads_for_prime_multiplicities_with_max_batch_coalesced(
                32 * 17,
                &[(17, 1)],
                512,
                1,
                medium,
            )
            .unwrap(),
            Some(34)
        );
        assert_eq!(
            plan_gpu_axis0_multi_fft_rader_threads_for_prime_multiplicities_with_max_batch_coalesced(
                32 * 17,
                &[(17, 1)],
                512,
                4,
                medium,
            )
            .unwrap(),
            Some(32)
        );
    }

    #[test]
    fn nested_direct_rader_subcontainer_feeds_recursive_optimizer_state() {
        let (registers, max_registers, min_registers) =
            rader_optimize_shared_register_table(282).unwrap();
        assert_eq!(registers[2], 4);
        assert_eq!(registers[3], 3);
        assert_eq!(max_registers, 4);
        assert_eq!(min_registers, 3);
        let state = build_multi_fft_rader_container_state(283, 2)
            .unwrap()
            .expect("p283 should retain nested p47 Direct-Rader under portable tuning");
        assert_eq!(state.kind, MultiFftRaderContainerKind::Fft);
        assert_eq!(state.prime, 283);
        assert_eq!(state.container_fft_num, 2);
        assert_eq!(state.subcontainers.len(), 1);
        assert_eq!(
            state.subcontainers[0].kind,
            MultiFftRaderContainerKind::Direct
        );
        assert_eq!(state.subcontainers[0].prime, 47);
        assert_eq!(state.subcontainers[0].container_fft_num, 2);
        assert_eq!(state.subcontainers[0].min_registers_per_thread, 2);
        assert_eq!(state.subcontainers[0].registers_per_thread, 2);

        let mut device = nvidia_vulkan(48 * 1024);
        device.max_threads_per_block = 1024;
        device.max_workgroup_size[0] = 1024;
        let schedule =
            plan_gpu_axis0_multi_fft_rader_parent_schedule(2 * 283, &[(283, 1)], 2, device)
                .unwrap()
                .expect("N566 nested p283/p47 should have an exact parent schedule");
        // Nested p47 is type-1: it contributes its cached two-register minimum to
        // recursive optimizer state but is intentionally absent from
        // minRaderFFTThreadNum. The p283 type-0 parent stages need 188 lanes, while
        // the shared minimum of two makes the outer N566 axis the final 283-lane
        // limiter.
        assert_eq!(schedule.global_scale_registers_num, 1);
        assert_eq!(schedule.final_min_registers, 2);
        assert_eq!(schedule.min_rader_fft_thread_num, 188);
        assert_eq!(schedule.threads_per_transform, 283);

        let mut narrow = device;
        narrow.max_threads_per_block = 256;
        narrow.max_workgroup_size[0] = 256;
        let narrow_schedule =
            plan_gpu_axis0_multi_fft_rader_parent_schedule(2 * 283, &[(283, 1)], 2, narrow)
                .unwrap()
                .expect("N566 nested p283/p47 should scale into a 256-thread profile");
        assert_eq!(narrow_schedule.global_scale_registers_num, 2);
        assert_eq!(narrow_schedule.final_min_registers, 4);
        assert_eq!(narrow_schedule.min_rader_fft_thread_num, 72);
        assert_eq!(narrow_schedule.threads_per_transform, 142);
    }

    #[test]
    fn double_double_nested_rader_peeling_uses_min_direct_threshold() {
        let device = nvidia_vulkan(48 * 1024);
        let tuning = PlannerTuning::for_device(device, Precision::DoubleDouble)
            .with_recursive_fft_rader(true);
        assert_eq!(tuning.min_rader_direct_prime, 11);
        assert_eq!(tuning.max_rader_direct_prime, 29);

        let container = build_multi_fft_rader_container_state_with_tuning(53, 2, tuning)
            .unwrap()
            .expect("DD p53 FFT-Rader container should remain representable");
        assert_eq!(container.subcontainers.len(), 1);
        assert_eq!(container.subcontainers[0].prime, 13);
        assert_eq!(
            container.subcontainers[0].kind,
            MultiFftRaderContainerKind::Direct
        );
        assert_eq!(container.subcontainers[0].container_fft_num, 2);
        let threads =
            plan_gpu_double_double_axis0_fft_rader_threads_for_prime_multiplicities_with_tuning(
                2 * 53,
                &[(53, 1)],
                2,
                tuning,
                device,
            )
            .unwrap();
        assert_eq!(threads, Some(9));

        let ordinary = build_multi_fft_rader_container_state_with_tuning(
            53,
            2,
            PlannerTuning::portable().with_recursive_fft_rader(true),
        )
        .unwrap()
        .expect("ordinary p53 FFT-Rader container should remain representable");
        assert!(ordinary.subcontainers.is_empty());
    }

    #[test]
    fn nested_rader_container_order_matches_upstream_fft_then_direct_passes() {
        let tuning = PlannerTuning::portable().with_recursive_fft_rader(true);
        let residual = 47usize * 53;
        let containers = build_nested_fft_rader_subcontainers(residual, 2, tuning)
            .unwrap()
            .expect("47*53 residual should be recursively schedulable");
        assert_eq!(containers.len(), 2);
        assert_eq!(containers[0].kind, MultiFftRaderContainerKind::Fft);
        assert_eq!(containers[0].prime, 53);
        assert_eq!(containers[0].container_fft_num, 94);
        assert_eq!(containers[1].kind, MultiFftRaderContainerKind::Direct);
        assert_eq!(containers[1].prime, 47);
        assert_eq!(containers[1].container_fft_num, 106);
    }

    #[test]
    fn double_double_nested_fft_then_direct_order_reaches_n886_parent() {
        let device = nvidia_vulkan(48 * 1024);
        let tuning = PlannerTuning::for_device(device, Precision::DoubleDouble)
            .with_recursive_fft_rader(true);
        assert_eq!(tuning.min_rader_direct_prime, 11);
        assert_eq!(tuning.min_rader_fft_prime, 17);
        let parent = build_multi_fft_rader_container_state_with_tuning(443, 2, tuning)
            .unwrap()
            .expect("DD p443 should retain recursive 442=2*13*17 Rader state");
        assert_eq!(parent.subcontainers.len(), 2);
        assert_eq!(
            parent.subcontainers[0].kind,
            MultiFftRaderContainerKind::Fft
        );
        assert_eq!(parent.subcontainers[0].prime, 17);
        assert_eq!(
            parent.subcontainers[1].kind,
            MultiFftRaderContainerKind::Direct
        );
        assert_eq!(parent.subcontainers[1].prime, 13);
        let threads =
            plan_gpu_double_double_axis0_fft_rader_threads_for_prime_multiplicities_with_tuning(
                2 * 443,
                &[(443, 1)],
                1,
                tuning,
                device,
            )
            .unwrap()
            .expect("DD N886 should have an exact recursive Rader parent floor");
        assert_eq!(threads, 37);
        let mut wide = device;
        wide.max_threads_per_block = 1024;
        wide.max_workgroup_size[0] = 1024;
        let wide_threads =
            plan_gpu_double_double_axis0_fft_rader_threads_for_prime_multiplicities_with_tuning(
                2 * 443,
                &[(443, 1)],
                1,
                tuning,
                wide,
            )
            .unwrap()
            .expect("DD N886 should keep an exact parent floor on a 1024-thread profile");
        assert_eq!(wide_threads, 74);
    }

    #[test]
    fn nested_rader_container_classification_respects_custom_tuning() {
        let mut fft_nested_tuning = PlannerTuning::portable().with_recursive_fft_rader(true);
        fft_nested_tuning.max_rader_direct_prime = 47;
        fft_nested_tuning.validate().unwrap();
        let fft_nested =
            build_multi_fft_rader_container_state_with_tuning(283, 2, fft_nested_tuning)
                .unwrap()
                .expect("custom tuning should retain p283 as FFT-Rader");
        assert_eq!(fft_nested.subcontainers.len(), 1);
        assert_eq!(fft_nested.subcontainers[0].prime, 47);
        assert_eq!(
            fft_nested.subcontainers[0].kind,
            MultiFftRaderContainerKind::Fft
        );

        let mut narrow = nvidia_vulkan(48 * 1024);
        narrow.max_threads_per_block = 256;
        narrow.max_workgroup_size[0] = 256;
        let tuned_schedule = plan_gpu_axis0_multi_fft_rader_parent_schedule_with_tuning(
            2 * 283,
            &[(283, 1)],
            2,
            fft_nested_tuning,
            narrow,
        )
        .unwrap()
        .expect("custom nested p47 FFT-Rader should have an exact N566 parent schedule");
        let dd_tuned_threads =
            plan_gpu_double_double_axis0_fft_rader_threads_for_prime_multiplicities_with_tuning(
                2 * 283,
                &[(283, 1)],
                2,
                fft_nested_tuning,
                narrow,
            )
            .unwrap()
            .expect("custom DD nested p47 FFT-Rader should have an exact N566 parent schedule");
        assert_eq!(dd_tuned_threads, 57);
        assert_eq!(tuned_schedule.global_scale_registers_num, 1);
        assert_eq!(tuned_schedule.final_min_registers, 10);
        assert_eq!(tuned_schedule.min_rader_fft_thread_num, 48);
        assert_eq!(tuned_schedule.threads_per_transform, 57);
    }

    #[test]
    fn nested_rader_parent_schedule_must_use_device_capped_direct_range() {
        let tuning = PlannerTuning::portable().with_recursive_fft_rader(true);
        let mut device = nvidia_vulkan(48 * 1024);
        device.max_threads_per_block = 64;
        device.max_workgroup_size[0] = 64;

        let raw = plan_gpu_axis0_multi_fft_rader_parent_schedule_with_tuning(
            2 * 283,
            &[(283, 1)],
            2,
            tuning,
            device,
        )
        .unwrap()
        .expect("raw recursive N566 schedule should remain representable");

        let capped = crate::config::upstream_effective_rader_tuning(tuning, device, Precision::F32);
        assert_eq!(capped.max_rader_direct_prime, 31);
        let expected = plan_gpu_axis0_multi_fft_rader_parent_schedule_with_tuning(
            2 * 283,
            &[(283, 1)],
            2,
            capped,
            device,
        )
        .unwrap()
        .expect("device-capped recursive N566 schedule should remain representable");

        assert_eq!(raw.global_scale_registers_num, 5);
        assert_eq!(raw.final_min_registers, 10);
        assert_eq!(raw.min_rader_fft_thread_num, 30);
        assert_eq!(raw.threads_per_transform, 57);
        assert_eq!(expected.global_scale_registers_num, 3);
        assert_eq!(expected.final_min_registers, 30);
        assert_eq!(expected.min_rader_fft_thread_num, 16);
        assert_eq!(expected.threads_per_transform, 19);
        assert_ne!(raw, expected);
    }

    #[test]
    fn nested_sub_rader_type0_parent_recurses_register_and_thread_pressure() {
        let (registers, max_registers, min_registers) =
            rader_optimize_shared_register_table(106).unwrap();
        assert_eq!(registers[2], 2);
        assert_eq!(max_registers, 2);
        assert_eq!(min_registers, 2);

        let state = build_multi_fft_rader_container_state(107, 2)
            .unwrap()
            .expect("p107 should retain its safe p53 subcontainer");
        assert_eq!(state.prime, 107);
        assert_eq!(state.container_fft_num, 2);
        assert_eq!(state.stage_radix_multipliers[2], 1);
        assert_eq!(state.subcontainers.len(), 1);
        assert_eq!(state.subcontainers[0].prime, 53);
        assert_eq!(state.subcontainers[0].container_fft_num, 2);
        assert!(state.subcontainers[0].subcontainers.is_empty());

        let mut device = nvidia_vulkan(48 * 1024);
        device.max_threads_per_block = 256;
        device.max_workgroup_size[0] = 256;
        let schedule =
            plan_gpu_axis0_multi_fft_rader_parent_schedule(2 * 107, &[(107, 1)], 2, device)
                .unwrap()
                .expect("N214 nested p107/p53 should have an exact parent schedule");
        assert_eq!(schedule.global_scale_registers_num, 1);
        // Recursive p53 optimization raises the shared minimum to 12. The parent
        // p107 radix-2 table is then aligned to 12 and the second optimizer's
        // toward-global-max loop advances it once more to 14, so the p107 container
        // pressure is 2*ceil(106/14)=16 lanes. The outer axis still needs
        // ceil(214/12)=18 lanes and therefore remains the final limiter.
        assert_eq!(schedule.final_min_registers, 12);
        assert_eq!(schedule.min_rader_fft_thread_num, 16);
        assert_eq!(schedule.threads_per_transform, 18);
        assert_eq!(
            plan_gpu_axis0_multi_fft_rader_threads_for_prime_multiplicities(
                2 * 107,
                &[(107, 1)],
                2,
                device,
            )
            .unwrap(),
            Some(18)
        );
        assert_eq!(
            plan_gpu_double_double_axis0_fft_rader_threads_for_prime_multiplicities(
                2 * 107,
                &[(107, 1)],
                2,
                device,
            )
            .unwrap(),
            Some(18)
        );
    }

    #[test]
    fn nested_sub_rader_and_sibling_type0_share_joint_optimizer_state() {
        let mut device = nvidia_vulkan(48 * 1024);
        device.max_threads_per_block = 1024;
        device.max_workgroup_size[0] = 1024;
        let schedule = plan_gpu_axis0_multi_fft_rader_parent_schedule(
            19 * 107,
            &[(19, 1), (107, 1)],
            2,
            device,
        )
        .unwrap()
        .expect("N2033 p19+p107/p53 should fit the recursive joint type0 slice");
        assert_eq!(schedule.global_scale_registers_num, 1);
        assert_eq!(schedule.final_min_registers, 12);
        assert_eq!(schedule.min_rader_fft_thread_num, 214);
        assert_eq!(schedule.threads_per_transform, 214);
        assert_eq!(
            plan_gpu_double_double_axis0_fft_rader_threads_for_prime_multiplicities(
                19 * 107,
                &[(19, 1), (107, 1)],
                2,
                device,
            )
            .unwrap(),
            Some(214)
        );
    }

    #[test]
    fn multiple_fft_rader_type0_containers_share_upstream_optimizer_state() {
        let mut device = nvidia_vulkan(48 * 1024);
        device.max_threads_per_block = 256;
        device.max_workgroup_size[0] = 256;
        let schedule =
            plan_gpu_axis0_multi_fft_rader_parent_schedule(19 * 29, &[(19, 1), (29, 1)], 2, device)
                .unwrap()
                .expect("p19+p29 should fit the exact smooth multi-type0 slice");
        // After both optimizer passes the active p19 stages retain a six-register
        // minimum, so the ordinary parent needs ceil(551/6)=92 lanes. The p19
        // container is the tighter FFT-Rader child at 29*ceil(18/6)=87 lanes;
        // p29 remains below that pressure. AxisBlockSplitter therefore keeps 92.
        assert_eq!(schedule.final_min_registers, 6);
        assert_eq!(schedule.min_rader_fft_thread_num, 87);
        assert_eq!(schedule.threads_per_transform, 92);
        assert_eq!(schedule.global_scale_registers_num, 1);
        assert_eq!(
            plan_gpu_axis0_multi_fft_rader_threads_for_prime_multiplicities(
                19 * 29,
                &[(19, 1), (29, 1)],
                2,
                device,
            )
            .unwrap(),
            Some(92)
        );

        let mut wide = device;
        wide.max_threads_per_block = 1024;
        wide.max_workgroup_size[0] = 1024;
        let scaled = plan_gpu_axis0_multi_fft_rader_parent_schedule(
            19 * 29 * 31,
            &[(19, 1), (29, 1), (31, 1)],
            1,
            wide,
        )
        .unwrap()
        .expect("p19+p29+p31 should be recovered by global register scaling");
        assert_eq!(scaled.threads_per_transform, 899);
        assert_eq!(scaled.global_scale_registers_num, 3);

        let repeated = plan_gpu_axis0_multi_fft_rader_parent_schedule(
            19 * 19 * 23,
            &[(19, 2), (23, 1)],
            1,
            wide,
        )
        .unwrap()
        .expect("p19^2+p23 should remain within one upload");
        assert_eq!(repeated.threads_per_transform, 437);
        assert_eq!(repeated.global_scale_registers_num, 2);
    }

    #[test]
    fn zero_register_fft_rader_stage_does_not_block_cross_bluestein_parent_scoring() {
        let mut tuning = PlannerTuning::portable();
        tuning.min_rader_direct_prime = 29;
        tuning.min_rader_fft_prime = 29;
        tuning.validate().unwrap();

        let mut nvidia_48k = nvidia_vulkan(48 * 1024);
        nvidia_48k.max_threads_per_block = 1024;
        nvidia_48k.max_workgroup_size[0] = 1024;
        let n206 = plan_gpu_axis0_multi_fft_rader_parent_schedule_with_tuning(
            2 * 103,
            &[(103, 1)],
            1,
            tuning,
            nvidia_48k,
        )
        .unwrap()
        .expect("N206 p103 cross-Bluestein parent should remain schedulable");
        assert_eq!(n206.final_min_registers, 2);
        assert_eq!(n206.min_rader_fft_thread_num, 68);
        assert_eq!(n206.threads_per_transform, 103);

        let mut nvidia_256k = nvidia_vulkan(256 * 1024);
        nvidia_256k.shared_memory_pow2_bytes = 256 * 1024;
        nvidia_256k.max_threads_per_block = 1024;
        nvidia_256k.max_workgroup_size[0] = 1024;
        let n3193 = plan_gpu_axis0_multi_fft_rader_parent_schedule_with_tuning(
            31 * 103,
            &[(31, 1), (103, 1)],
            1,
            tuning,
            nvidia_256k,
        )
        .unwrap()
        .expect("N3193 p31+p103 cross-Bluestein parent should remain schedulable");
        assert_eq!(n3193.final_min_registers, 5);
        assert_eq!(n3193.min_rader_fft_thread_num, 618);
        assert_eq!(n3193.threads_per_transform, 639);
    }

    #[test]
    fn double_double_forced_pass_context_uses_coalesced_parent_pressure() {
        let mut device = DeviceProfile::generic(Backend::OpenCl, GpuVendor::Intel);
        device.shared_memory_bytes = 48 * 1024;
        device.shared_memory_pow2_bytes = 32 * 1024;
        device.max_threads_per_block = 32;
        device.max_workgroup_size[0] = 32;
        device.supports_f64 = true;
        assert_eq!(
            crate::config::upstream_coalesced_memory_bytes(device) / 32,
            2
        );

        // Direct-only N34 is the minimal DD witness: two coalesced transforms force the
        // shared type-1 register floor from two to four and halve the parent lane count.
        assert_eq!(
            plan_gpu_double_double_axis0_composite_direct_rader_threads_for_prime_multiplicities_with_max_batch_coalesced(
                2 * 17,
                &[(17, 1)],
                32,
                1,
                device,
            )
            .unwrap(),
            Some(18)
        );
        assert_eq!(
            plan_gpu_double_double_axis0_composite_direct_rader_threads_for_prime_multiplicities_with_max_batch_coalesced(
                2 * 17,
                &[(17, 1)],
                32,
                2,
                device,
            )
            .unwrap(),
            Some(9)
        );

        // Pure type-0 N272 observes the same pass-local pressure through the global
        // scaleRegistersNum branch. maxBatch=1 remains the legacy one-upload result.
        let recursive = PlannerTuning::portable().with_recursive_fft_rader(true);
        assert_eq!(
            plan_gpu_double_double_axis0_fft_rader_threads_for_prime_multiplicities_with_tuning(
                16 * 17,
                &[(17, 1)],
                32,
                recursive,
                device,
            )
            .unwrap(),
            Some(17)
        );
        assert_eq!(
            plan_gpu_double_double_axis0_fft_rader_threads_for_forced_upload_with_tuning(
                16 * 17,
                &[(17, 1)],
                32,
                2,
                1,
                recursive,
                device,
            )
            .unwrap(),
            Some(16)
        );

        // Mixed p17 Direct + p19 FFT-Rader couples the same maxBatch=2 pressure into
        // both the type-1 register loop and the type-0 global register-table scaling.
        let mut mixed_tuning = recursive;
        mixed_tuning.min_rader_direct_prime = 17;
        mixed_tuning.max_rader_direct_prime = 31;
        mixed_tuning.min_rader_fft_prime = 19;
        mixed_tuning.validate().unwrap();
        assert_eq!(
            plan_gpu_double_double_axis0_mixed_direct_multi_fft_rader_threads_with_tuning(
                17 * 19,
                &[(17, 1)],
                &[(19, 1)],
                32,
                mixed_tuning,
                device,
            )
            .unwrap(),
            Some(27)
        );
        assert_eq!(
            plan_gpu_double_double_axis0_mixed_direct_multi_fft_rader_threads_for_forced_upload_with_tuning(
                17 * 19,
                &[(17, 1)],
                &[(19, 1)],
                32,
                2,
                1,
                mixed_tuning,
                device,
            )
            .unwrap(),
            Some(17)
        );
    }

    #[test]
    fn double_double_mixed_multi_fft_rader_uses_quad_outer_and_global_scaling() {
        let mut device = nvidia_vulkan(48 * 1024);
        device.max_threads_per_block = 1024;
        device.max_workgroup_size[0] = 1024;
        assert_eq!(
            plan_gpu_double_double_axis0_mixed_direct_multi_fft_rader_threads(
                17 * 19 * 29,
                &[(17, 1)],
                &[(19, 1), (29, 1)],
                1,
                device,
            )
            .unwrap(),
            Some(999)
        );
        assert_eq!(
            plan_gpu_double_double_axis0_mixed_direct_multi_fft_rader_threads(
                2 * 17 * 19 * 23,
                &[(17, 1)],
                &[(19, 1), (23, 1)],
                1,
                device,
            )
            .unwrap(),
            Some(990)
        );
        assert_eq!(
            plan_gpu_double_double_axis0_mixed_direct_fft_rader_threads(
                17 * 19,
                &[(17, 1)],
                19,
                29,
                device,
            )
            .unwrap(),
            Some(171)
        );
    }

    #[test]
    fn mixed_direct_multi_fft_rader_runs_joint_type0_then_type1_scheduler() {
        let mut device = nvidia_vulkan(48 * 1024);
        device.max_threads_per_block = 256;
        device.max_workgroup_size[0] = 256;
        // Custom fixed-upstream thresholds: p11 is below the FFT scan and therefore
        // remains type-1 Direct-Rader, while safe p17/p19 are both type-0 FFT-Rader.
        // The shared first optimizer merges the p11 Direct container's two-register
        // state before type-1 scaling. Direct granularity rounds the base block to
        // 258 lanes; fixed-upstream AxisBlockSplitter clamps that final block to the
        // 256-thread physical ceiling after lower p17/p19 FFT child pressure.
        assert_eq!(
            plan_gpu_axis0_mixed_direct_multi_fft_rader_threads(
                11 * 17 * 19,
                &[(11, 1)],
                &[(17, 1), (19, 1)],
                2,
                device,
            )
            .unwrap(),
            Some(256)
        );

        let mut wide = device;
        wide.max_threads_per_block = 1024;
        wide.max_workgroup_size[0] = 1024;
        assert_eq!(
            plan_gpu_axis0_mixed_direct_multi_fft_rader_threads(
                17 * 19 * 23,
                &[(17, 1)],
                &[(19, 1), (23, 1)],
                2,
                wide,
            )
            .unwrap(),
            Some(990)
        );
        // p17 Direct + p19/p29 FFT-Rader crosses the one-upload 1024-thread floor
        // before upstream's global `scaleRegistersNum` pass. Keep this assertion as a
        // deterministic oracle for that branch rather than falling back to `None`.
        assert_eq!(
            plan_gpu_axis0_mixed_direct_multi_fft_rader_threads(
                17 * 19 * 29,
                &[(17, 1)],
                &[(19, 1), (29, 1)],
                2,
                wide,
            )
            .unwrap(),
            Some(999)
        );
        assert_eq!(
            plan_gpu_axis0_mixed_direct_multi_fft_rader_threads_with_max_batch_coalesced(
                17 * 19,
                &[(17, 1)],
                &[(19, 1)],
                29,
                4,
                wide,
            )
            .unwrap(),
            Some(63)
        );
    }

    #[test]
    fn mixed_direct_fft_rader_uses_type0_first_optimizer_before_type1_scaling() {
        let device = nvidia_vulkan(48 * 1024);
        // Splitting the optimizer's two geometries is observable: the p31 FFT-Rader
        // has 17 physical containers, while the type-0-only optimizer reaches a
        // 4-register floor. The full upstream optimizer then merges the p17 Direct
        // container's fixed two-register state before `scale_registers_rader`, yielding
        // the final 108-lane AxisBlock instead of the stale 144-lane estimate.
        let (mut registers, _, _) = rader_optimize_shared_register_table(30).unwrap();
        assert_eq!(
            optimize_single_rader_container_registers(17 * 31, 1, 17, 2, &mut registers).unwrap(),
            4
        );
        assert_eq!(
            plan_gpu_axis0_mixed_direct_fft_rader_threads(17 * 31, &[(17, 1)], 31, 2, device,)
                .unwrap(),
            Some(108)
        );
    }

    #[test]
    fn double_double_mixed_direct_fft_rader_matches_zero_smooth_outer_slice() {
        let device = nvidia_vulkan(48 * 1024);
        assert_eq!(
            plan_gpu_double_double_axis0_mixed_direct_fft_rader_threads(
                17 * 31,
                &[(17, 1)],
                31,
                2,
                device,
            )
            .unwrap(),
            Some(108)
        );
        let with_smooth_outer = plan_gpu_double_double_axis0_mixed_direct_fft_rader_threads(
            2 * 17 * 31,
            &[(17, 1)],
            31,
            2,
            device,
        )
        .unwrap()
        .expect("DD mixed Rader with radix-2 outer factor should use Quad register state");
        assert!(with_smooth_outer > 0 && with_smooth_outer <= device.max_threads_per_block);
    }

    #[test]
    fn double_double_mixed_rader_can_share_parent_floor_despite_quad_outer_table() {
        let mut device = nvidia_vulkan(48 * 1024);
        device.max_threads_per_block = 1024;
        device.max_workgroup_size[0] = 1024;
        let length = 15usize * 17 * 31;
        let ordinary =
            plan_gpu_axis0_mixed_direct_fft_rader_threads(length, &[(17, 1)], 31, 1, device)
                .unwrap()
                .expect("ordinary mixed Rader N7905 should produce a parent thread floor");
        let quad = plan_gpu_double_double_axis0_mixed_direct_fft_rader_threads(
            length,
            &[(17, 1)],
            31,
            1,
            device,
        )
        .unwrap()
        .expect("DD mixed Rader N7905 should produce a Quad parent thread floor");
        assert_eq!(ordinary, 990);
        assert_eq!(quad, 990);
        assert_eq!(ordinary, quad);

        let ordinary_outer = plan_gpu_small_mixed_radix_registers(15, length / 15).unwrap();
        let quad_outer = plan_gpu_double_double_quad_registers(15, length / 15).unwrap();
        assert_eq!(ordinary_outer.min_registers_per_thread, 15);
        assert_eq!(quad_outer.min_registers_per_thread, 3);
    }

    #[test]
    fn double_double_composite_direct_rader_thread_estimate_uses_quad_outer_floor() {
        let device = nvidia_vulkan(48 * 1024);
        assert_eq!(
            plan_gpu_double_double_axis0_composite_direct_rader_threads_for_primes(
                2 * 47,
                &[47],
                5,
                device,
            )
            .unwrap(),
            Some(48)
        );
        assert_eq!(
            plan_gpu_double_double_axis0_composite_direct_rader_threads_for_primes(
                47 * 53,
                &[47, 53],
                2,
                device,
            )
            .unwrap(),
            Some(216)
        );
        let mut wide = device;
        wide.max_threads_per_block = 1024;
        wide.max_workgroup_size[0] = 1024;
        assert_eq!(
            plan_gpu_double_double_axis0_composite_direct_rader_threads_for_primes(
                47 * 53,
                &[47, 53],
                2,
                wide,
            )
            .unwrap(),
            Some(432)
        );
    }

    #[test]
    fn repeated_direct_rader_multiplicity_removes_full_prime_power_from_outer_radix() {
        let device = nvidia_vulkan(48 * 1024);
        assert_eq!(
            plan_gpu_axis0_composite_direct_rader_threads_for_prime_multiplicities(
                17 * 17,
                &[(17, 2)],
                2,
                device,
            )
            .unwrap(),
            Some(153)
        );
        assert_eq!(
            plan_gpu_double_double_axis0_composite_direct_rader_threads_for_prime_multiplicities(
                17 * 17,
                &[(17, 2)],
                2,
                device,
            )
            .unwrap(),
            Some(153)
        );
    }

    #[test]
    fn rader_occupancy_preserves_upstream_fractional_scaling_branch() {
        assert!(!rader_container_occupancy_fits(1, 8, 16, 32).unwrap());
        assert!(rader_container_occupancy_fits(4, 8, 16, 32).unwrap());
        assert!(rader_container_occupancy_fits(2, 8, 16, 64).unwrap());
        assert!(!rader_container_occupancy_fits(3, 8, 16, 16).unwrap());
        assert!(rader_container_occupancy_fits(8, 8, 16, 16).unwrap());
    }

    #[test]
    fn rader_smooth_container_uses_optimize_shared_register_schedule() {
        let device = nvidia_vulkan(48 * 1024);
        let r17 = plan_nvidia_vulkan_rader_fft_registers(17, 2, device).unwrap();
        assert_eq!(r17.convolution_len, 16);
        assert_eq!(r17.container_fft_num, 1);
        assert_eq!(r17.min_rader_fft_thread_num, 1);
        assert_eq!(r17.execution_threads_per_workgroup, 1);
        assert_eq!(r17.execution_workgroup_count, 2);
        assert!(r17.upstream_grouping_is_executable());
        assert_eq!(r17.internal_fft.rhs_transform_count, 2);
        assert_eq!(r17.internal_fft.stage_radices, vec![16]);
        assert_eq!(r17.internal_fft.registers_per_thread, 16);
        assert_eq!(r17.internal_fft.min_registers_per_thread, 16);
        assert_eq!(r17.internal_fft.registers_per_thread_per_radix[16], 16);

        let r257 = plan_nvidia_vulkan_rader_fft_registers(257, 3, device).unwrap();
        assert_eq!(r257.convolution_len, 256);
        assert_eq!(r257.container_fft_num, 1);
        assert_eq!(r257.min_rader_fft_thread_num, 16);
        assert_eq!(r257.execution_threads_per_workgroup, 16);
        assert_eq!(r257.execution_workgroup_count, 3);
        assert!(r257.upstream_grouping_is_executable());
        assert_eq!(r257.internal_fft.rhs_transform_count, 3);
        assert_eq!(r257.internal_fft.stage_radices, vec![16, 16]);
        assert_eq!(r257.internal_fft.registers_per_thread, 16);
        assert_eq!(r257.internal_fft.min_registers_per_thread, 16);
        assert_eq!(r257.internal_fft.registers_per_thread_per_radix[16], 16);
        assert_eq!(r257.internal_fft.register_boost, 1);
        assert_eq!(r257.internal_fft.register_boost_stage_radix, None);

        let mixed =
            plan_nvidia_vulkan_rader_fft_registers_for_containers(257, 4, 514, 2, device).unwrap();
        assert_eq!(mixed.outer_fft_len, 514);
        assert_eq!(mixed.container_fft_num, 2);
        assert_eq!(mixed.min_rader_fft_thread_num, 32);
        assert_eq!(mixed.upstream_workgroup_count(), 2);
        assert_eq!(mixed.execution_container_fft_num, 2);
        assert_eq!(mixed.execution_workgroup_count, 2);
        assert_eq!(mixed.execution_threads_per_workgroup, 32);
        assert!(mixed.upstream_grouping_is_executable());
        assert_eq!(mixed.internal_fft.stage_radices, vec![16, 16]);
        assert_eq!(mixed.internal_fft.registers_per_thread_per_radix[16], 16);

        let grouped_four =
            plan_nvidia_vulkan_rader_fft_registers_for_containers(257, 8, 1028, 4, device).unwrap();
        assert_eq!(grouped_four.outer_fft_len, 1028);
        assert_eq!(grouped_four.container_fft_num, 4);
        assert_eq!(grouped_four.min_rader_fft_thread_num, 64);
        assert_eq!(grouped_four.execution_container_fft_num, 4);
        assert_eq!(grouped_four.execution_workgroup_count, 2);
        assert_eq!(grouped_four.execution_threads_per_workgroup, 64);
        assert!(grouped_four.upstream_grouping_is_executable());
        assert!(grouped_four.rader_transpose.is_none());
        assert_eq!(grouped_four.internal_fft.stage_radices, vec![16, 16]);

        for (prime, containers, batch_count, expected_radices) in [
            (19usize, 3usize, 6usize, vec![6usize, 3usize]),
            (29usize, 6usize, 12usize, vec![7usize, 4usize]),
        ] {
            let smooth_outer = plan_nvidia_vulkan_rader_fft_registers_for_containers(
                prime,
                batch_count,
                prime * containers,
                containers,
                device,
            )
            .unwrap();
            assert_eq!(smooth_outer.container_fft_num, containers);
            assert_eq!(smooth_outer.execution_container_fft_num, containers);
            assert_eq!(
                smooth_outer.execution_workgroup_count,
                batch_count / containers
            );
            assert_eq!(smooth_outer.internal_fft.stage_radices, expected_radices);
            assert!(smooth_outer.rader_transpose.is_none());
            assert!(smooth_outer.upstream_grouping_is_executable());
            assert!(smooth_outer.execution_threads_per_workgroup <= device.max_threads_per_block);
        }

        let mut constrained = device;
        constrained.max_threads_per_block = 16;
        let scaled =
            plan_nvidia_vulkan_rader_fft_registers_for_containers(257, 2, 514, 2, constrained)
                .unwrap();
        assert_eq!(scaled.min_rader_fft_thread_num, 16);
        assert_eq!(scaled.execution_threads_per_workgroup, 16);
        assert!(scaled.internal_fft.registers_per_thread_per_radix[16] >= 32);

        let transposed_eight =
            plan_nvidia_vulkan_rader_fft_registers_for_containers(257, 16, 2056, 8, device)
                .unwrap();
        assert_eq!(transposed_eight.container_fft_num, 8);
        assert_eq!(transposed_eight.min_rader_fft_thread_num, 128);
        assert_eq!(transposed_eight.execution_container_fft_num, 8);
        assert_eq!(transposed_eight.execution_workgroup_count, 2);
        assert_eq!(transposed_eight.execution_threads_per_workgroup, 128);
        assert!(transposed_eight.upstream_grouping_is_executable());
        let transpose = transposed_eight
            .rader_transpose
            .as_ref()
            .expect("eight two-stage p257 containers should enable upstream raderTranspose");
        assert_eq!(transpose.workgroup_threads, 128);
        assert_eq!(transpose.stages.len(), 2);
        assert_eq!(transpose.stages[0].sub_logical_group_size, 16);
        assert_eq!(transpose.stages[0].active_threads, 128);
        assert_eq!(
            transpose.stages[0].layout,
            RaderFftStageLaneLayout::ContainerMajor
        );
        assert_eq!(
            transpose.stages[1].layout,
            RaderFftStageLaneLayout::TransposedContainerInterleaved
        );
        assert_eq!(transpose.lane_coordinates(0, 0).unwrap(), Some((0, 0)));
        assert_eq!(transpose.lane_coordinates(0, 15).unwrap(), Some((15, 0)));
        assert_eq!(transpose.lane_coordinates(0, 16).unwrap(), Some((0, 1)));
        assert_eq!(transpose.lane_coordinates(0, 127).unwrap(), Some((15, 7)));
        assert_eq!(transpose.lane_coordinates(1, 0).unwrap(), Some((0, 0)));
        assert_eq!(transpose.lane_coordinates(1, 7).unwrap(), Some((0, 7)));
        assert_eq!(transpose.lane_coordinates(1, 8).unwrap(), Some((1, 0)));
        assert_eq!(transpose.lane_coordinates(1, 127).unwrap(), Some((15, 7)));
        assert_eq!(transpose.lane_coordinates(1, 128).unwrap(), None);

        // p29's 28-point convolution keeps its VkFFTGetRaderFFTStages construction
        // order [7,4] even when eight containers change the per-radix register scaling.
        // Upstream's final no-registerBoost stage0 swap applies only to the top-level
        // axis stageRadix array; the analogous internal-container reorder is commented out.
        for batch_count in [8usize, 16usize] {
            let transposed_p29 = plan_nvidia_vulkan_rader_fft_registers_for_containers(
                29,
                batch_count,
                232,
                8,
                device,
            )
            .unwrap();
            assert_eq!(transposed_p29.container_fft_num, 8);
            assert_eq!(transposed_p29.internal_fft.stage_radices, vec![7, 4]);
            assert!(transposed_p29.rader_transpose.is_some());
            assert!(transposed_p29.upstream_grouping_is_executable());
        }

        let transposed_sixteen =
            plan_nvidia_vulkan_rader_fft_registers_for_containers(257, 32, 4112, 16, device)
                .unwrap();
        assert_eq!(transposed_sixteen.container_fft_num, 16);
        assert_eq!(transposed_sixteen.min_rader_fft_thread_num, 256);
        assert_eq!(transposed_sixteen.execution_container_fft_num, 16);
        assert_eq!(transposed_sixteen.execution_workgroup_count, 2);
        assert_eq!(transposed_sixteen.execution_threads_per_workgroup, 256);
        assert!(transposed_sixteen.upstream_grouping_is_executable());
        let transpose16 = transposed_sixteen
            .rader_transpose
            .as_ref()
            .expect("sixteen p257 containers should reuse upstream raderTranspose geometry");
        assert_eq!(transpose16.workgroup_threads, 256);
        assert!(
            transpose16
                .stages
                .iter()
                .all(|stage| stage.active_threads == 256)
        );
        assert_eq!(
            transpose16.lane_coordinates(0, 255).unwrap(),
            Some((15, 15))
        );
        assert_eq!(
            transpose16.lane_coordinates(1, 255).unwrap(),
            Some((15, 15))
        );
        assert_eq!(transpose16.lane_coordinates(1, 256).unwrap(), None);

        let mut wide_device = device;
        wide_device.max_threads_per_block = 1024;
        let transposed_thirty_two =
            plan_nvidia_vulkan_rader_fft_registers_for_containers(257, 64, 8224, 32, wide_device)
                .unwrap();
        assert_eq!(transposed_thirty_two.container_fft_num, 32);
        assert_eq!(transposed_thirty_two.min_rader_fft_thread_num, 512);
        assert_eq!(transposed_thirty_two.execution_container_fft_num, 32);
        assert_eq!(transposed_thirty_two.execution_workgroup_count, 2);
        assert_eq!(transposed_thirty_two.execution_threads_per_workgroup, 512);
        let transpose32 = transposed_thirty_two.rader_transpose.as_ref().unwrap();
        assert_eq!(transpose32.workgroup_threads, 512);
        assert!(
            transpose32
                .stages
                .iter()
                .all(|stage| stage.active_threads == 512)
        );
        assert_eq!(
            transpose32.lane_coordinates(0, 511).unwrap(),
            Some((15, 31))
        );
        assert_eq!(
            transpose32.lane_coordinates(1, 511).unwrap(),
            Some((15, 31))
        );

        let one_stage_eight =
            plan_nvidia_vulkan_rader_fft_registers_for_containers(17, 16, 136, 8, device).unwrap();
        assert_eq!(one_stage_eight.internal_fft.stage_radices, vec![16]);
        assert_eq!(one_stage_eight.container_fft_num, 8);
        assert_eq!(one_stage_eight.min_rader_fft_thread_num, 8);
        assert_eq!(one_stage_eight.execution_container_fft_num, 8);
        assert_eq!(one_stage_eight.execution_workgroup_count, 2);
        assert_eq!(one_stage_eight.execution_threads_per_workgroup, 8);
        assert!(one_stage_eight.upstream_grouping_is_executable());
        assert!(one_stage_eight.rader_transpose.is_none());

        for (prime, expected_radices) in [
            (19usize, vec![6usize, 3]),
            (29usize, vec![7usize, 4]),
            (53usize, vec![13usize, 4]),
        ] {
            let transposed = plan_nvidia_vulkan_rader_fft_registers_for_containers(
                prime,
                16,
                prime * 8,
                8,
                device,
            )
            .unwrap();
            assert_eq!(transposed.container_fft_num, 8);
            assert_eq!(transposed.execution_container_fft_num, 8);
            assert!(transposed.upstream_grouping_is_executable());
            assert_eq!(transposed.internal_fft.stage_radices, expected_radices);
            let transpose = transposed.rader_transpose.as_ref().expect(
                "eight multi-stage smooth Rader containers should use upstream raderTranspose",
            );
            assert_eq!(transpose.container_fft_dim, prime - 1);
            assert_eq!(transpose.container_fft_num, 8);
            assert_eq!(
                transpose.workgroup_threads,
                transposed.execution_threads_per_workgroup
            );
            assert_eq!(transpose.stages.len(), 2);
            assert_eq!(
                transpose.stages[0].layout,
                RaderFftStageLaneLayout::ContainerMajor
            );
            assert_eq!(
                transpose.stages[1].layout,
                RaderFftStageLaneLayout::TransposedContainerInterleaved
            );
            assert!(transpose.workgroup_threads <= device.max_threads_per_block);
        }

        let r19 = plan_nvidia_vulkan_rader_fft_registers(19, 2, device).unwrap();
        assert_eq!(r19.convolution_len, 18);
        assert_eq!(r19.internal_fft.stage_radices, vec![6, 3]);
        assert_eq!(r19.internal_fft.registers_per_thread, 6);
        assert_eq!(r19.internal_fft.min_registers_per_thread, 6);
        assert_eq!(r19.execution_threads_per_workgroup, 3);
        assert_eq!(r19.execution_workgroup_count, 2);
        assert!(r19.rader_transpose.is_none());

        let r29 = plan_nvidia_vulkan_rader_fft_registers(29, 2, device).unwrap();
        assert_eq!(r29.convolution_len, 28);
        assert_eq!(r29.internal_fft.stage_radices, vec![7, 4]);
        assert_eq!(r29.internal_fft.registers_per_thread, 8);
        assert_eq!(r29.internal_fft.min_registers_per_thread, 7);
        assert_eq!(r29.execution_threads_per_workgroup, 4);
        assert_eq!(r29.execution_workgroup_count, 2);

        for (prime, expected_radices, expected_threads) in [
            (19usize, vec![6usize, 3], 6usize),
            (29, vec![7, 4], 8),
            (53, vec![13, 4], 8),
        ] {
            let grouped = plan_nvidia_vulkan_rader_fft_registers_for_containers(
                prime,
                4,
                prime * 2,
                2,
                device,
            )
            .unwrap();
            assert_eq!(grouped.container_fft_num, 2);
            assert_eq!(grouped.execution_container_fft_num, 2);
            assert_eq!(grouped.execution_workgroup_count, 2);
            assert_eq!(grouped.execution_threads_per_workgroup, expected_threads);
            assert!(grouped.upstream_grouping_is_executable());
            assert!(grouped.rader_transpose.is_none());
            assert_eq!(grouped.internal_fft.stage_radices, expected_radices);
        }

        assert!(matches!(
            plan_nvidia_vulkan_rader_fft_registers(47, 1, device),
            Err(VkFftError::UnsupportedKernelPath(_))
        ));
    }

    #[test]
    fn nvidia_register_boost_matches_one_upload_power_of_two_policy() {
        let device = nvidia_vulkan(48 * 1024);
        let schedule =
            plan_nvidia_vulkan_power_of_two_stockham_uploads(16_384, Precision::F32, device)
                .unwrap();
        assert_eq!(schedule.used_shared_memory_bytes, 32 * 1024);
        assert_eq!(schedule.max_sequence_len_shared, 4096);
        assert_eq!(schedule.max_sequence_len_strided, 1024);
        assert_eq!(schedule.register_boost, 4);
        assert_eq!(schedule.upload_count, 1);
        assert_eq!(schedule.axis_split, vec![16_384]);
    }

    #[test]
    fn nvidia_two_upload_large_power_of_two_uses_sqrt_split() {
        let device = nvidia_vulkan(48 * 1024);
        let schedule =
            plan_nvidia_vulkan_power_of_two_stockham_uploads(1_048_576, Precision::F32, device)
                .unwrap();
        assert_eq!(schedule.register_boost, 1);
        assert_eq!(schedule.upload_count, 2);
        assert_eq!(schedule.axis_split, vec![1024, 1024]);
    }

    #[test]
    fn nvidia_three_upload_threshold_uses_cube_root_divisor_search() {
        let device = nvidia_vulkan(48 * 1024);
        let schedule =
            plan_nvidia_vulkan_power_of_two_stockham_uploads(8_388_608, Precision::F32, device)
                .unwrap();
        assert_eq!(schedule.register_boost, 1);
        assert_eq!(schedule.upload_count, 3);
        assert_eq!(schedule.axis_split, vec![256, 128, 256]);
    }

    #[test]
    fn nvidia_smooth_non_power_of_two_uploads_follow_generic_divisor_searches() {
        let device = DeviceProfile::generic(Backend::Vulkan, GpuVendor::Nvidia);

        let one = plan_nvidia_vulkan_smooth_stockham_uploads(3840, Precision::F32, device).unwrap();
        assert_eq!(one.register_boost, 1);
        assert_eq!(one.upload_count, 1);
        assert_eq!(one.axis_split, vec![3840]);
        assert_eq!(one.radix_schedules[0].stage_radices, vec![10, 12, 4, 4, 2]);
        assert_eq!(
            one.radix_schedules[0]
                .stage_radices
                .iter()
                .product::<usize>(),
            3840
        );

        let two = plan_nvidia_vulkan_smooth_stockham_uploads(6144, Precision::F32, device).unwrap();
        assert_eq!(two.register_boost, 1);
        assert_eq!(two.upload_count, 2);
        assert_eq!(two.axis_split, vec![96, 64]);
        assert_eq!(
            two.radix_schedules[0]
                .stage_radices
                .iter()
                .product::<usize>(),
            96
        );
        assert_eq!(
            two.radix_schedules[1]
                .stage_radices
                .iter()
                .product::<usize>(),
            64
        );

        let three =
            plan_nvidia_vulkan_smooth_stockham_uploads(1_572_864, Precision::F32, device).unwrap();
        assert_eq!(three.register_boost, 1);
        assert_eq!(three.upload_count, 3);
        assert_eq!(three.axis_split, vec![128, 96, 128]);
        assert!(three.axis_split.iter().all(|factor| *factor <= 1024));
        three.validate().unwrap();
    }

    #[test]
    fn nvidia_f64_capacity_uses_sixteen_byte_complex_values() {
        let device = nvidia_vulkan(48 * 1024);
        let schedule =
            plan_nvidia_vulkan_power_of_two_stockham_uploads(8192, Precision::F64, device).unwrap();
        assert_eq!(schedule.max_sequence_len_shared, 2048);
        assert_eq!(schedule.max_sequence_len_strided, 1024);
        assert_eq!(schedule.register_boost, 4);
        assert_eq!(schedule.upload_count, 1);
    }

    #[test]
    fn two_three_register_schedule_matches_upstream_table_and_merge_order() {
        let cases = [
            (6usize, vec![6], 6usize),
            (12, vec![12], 12),
            (18, vec![6, 3], 6),
            (24, vec![12, 2], 12),
            (36, vec![6, 6], 6),
            (54, vec![6, 3, 3], 6),
            (72, vec![12, 6], 12),
        ];
        for (length, expected_radices, expected_registers) in cases {
            let schedule = plan_nvidia_vulkan_two_three_radix_registers(length, 2).unwrap();
            assert_eq!(schedule.fft_len, length);
            assert_eq!(schedule.register_boost, 1);
            assert_eq!(schedule.stage_radices, expected_radices, "N={length}");
            assert_eq!(
                schedule.registers_per_thread, expected_registers,
                "N={length}"
            );
            assert!(schedule.register_boost_stage_radix.is_none());
        }
        assert!(matches!(
            plan_nvidia_vulkan_two_three_radix_registers(60, 2),
            Err(VkFftError::UnsupportedKernelPath(_))
        ));
        assert!(matches!(
            plan_nvidia_vulkan_two_three_radix_registers(27, 2),
            Err(VkFftError::UnsupportedKernelPath(_))
        ));
    }

    #[test]
    fn backend_neutral_register_entries_preserve_exact_legacy_tables() {
        let generic_pow2 = plan_gpu_power_of_two_radix_registers(1024, 7, 1).unwrap();
        let legacy_pow2 = plan_nvidia_vulkan_power_of_two_radix_registers(1024, 7, 1).unwrap();
        assert_eq!(generic_pow2, legacy_pow2);

        for length in [18usize, 60, 143, 3840] {
            let generic = plan_gpu_small_mixed_radix_registers(length, 5).unwrap();
            let legacy = plan_nvidia_vulkan_small_mixed_radix_registers(length, 5).unwrap();
            assert_eq!(
                generic, legacy,
                "shared VkFFT register table diverged for N={length}"
            );
        }
    }

    #[test]
    fn broader_small_mixed_register_table_matches_upstream_decision_leaves() {
        let cases = [
            (3usize, vec![(3usize, 3usize)]),
            (5, vec![(5, 5)]),
            (7, vec![(7, 7)]),
            (9, vec![(3, 9)]),
            (11, vec![(11, 11)]),
            (13, vec![(13, 13)]),
            (10, vec![(2, 10), (5, 10)]),
            (14, vec![(2, 14), (7, 14)]),
            (15, vec![(3, 15), (5, 15)]),
            (21, vec![(3, 6), (7, 7)]),
            (22, vec![(2, 10), (11, 11)]),
            (25, vec![(5, 5)]),
            (26, vec![(2, 12), (13, 13)]),
            (27, vec![(3, 9)]),
            (30, vec![(2, 6), (3, 6), (5, 5)]),
            (35, vec![(5, 5), (7, 7)]),
            (39, vec![(3, 12), (13, 13)]),
            (40, vec![(2, 8), (5, 10)]),
            (42, vec![(2, 6), (3, 6), (7, 7)]),
            (44, vec![(2, 8), (11, 11)]),
            (49, vec![(7, 7)]),
            (55, vec![(5, 5), (11, 11)]),
            (60, vec![(2, 12), (3, 12), (5, 10)]),
            (66, vec![(2, 6), (3, 6), (11, 11)]),
            (70, vec![(2, 10), (5, 10), (7, 7)]),
            (78, vec![(2, 6), (3, 6), (13, 13)]),
            (63, vec![(3, 9), (7, 7)]),
            (65, vec![(5, 5), (13, 13)]),
            (77, vec![(7, 7), (11, 11)]),
            (91, vec![(7, 7), (13, 13)]),
            (105, vec![(3, 15), (5, 15), (7, 14)]),
            (110, vec![(2, 10), (5, 10), (11, 11)]),
            (130, vec![(2, 10), (5, 10), (13, 13)]),
            (154, vec![(2, 14), (7, 14), (11, 11)]),
            (165, vec![(3, 15), (5, 15), (11, 11)]),
            (182, vec![(2, 14), (7, 14), (13, 13)]),
            (195, vec![(3, 15), (5, 15), (13, 13)]),
            (231, vec![(3, 12), (7, 14), (11, 11)]),
            (273, vec![(3, 12), (7, 14), (13, 13)]),
            (286, vec![(2, 12), (11, 11), (13, 13)]),
            (330, vec![(2, 10), (3, 15), (5, 10), (11, 11)]),
            (390, vec![(2, 10), (3, 15), (5, 10), (13, 13)]),
            (429, vec![(3, 12), (11, 11), (13, 13)]),
            (462, vec![(2, 12), (3, 12), (7, 14), (11, 11)]),
            (546, vec![(2, 12), (3, 12), (7, 14), (13, 13)]),
            (770, vec![(2, 10), (5, 10), (7, 14), (11, 11)]),
            (858, vec![(2, 6), (3, 6), (11, 11), (13, 13)]),
            (117, vec![(3, 9), (13, 13)]),
            (121, vec![(11, 11)]),
            (143, vec![(11, 11), (13, 13)]),
            (169, vec![(13, 13)]),
            (210, vec![(2, 14), (3, 15), (5, 15), (7, 14)]),
            (420, vec![(2, 12), (3, 12), (5, 15), (7, 14)]),
        ];
        for (length, primitive_registers) in cases {
            let schedule = plan_nvidia_vulkan_small_mixed_radix_registers(length, 2).unwrap();
            assert_eq!(schedule.fft_len, length);
            assert_eq!(schedule.register_boost, 1);
            for (radix, registers) in primitive_registers {
                assert_eq!(
                    schedule.registers_per_thread_per_radix[radix], registers,
                    "primitive register mismatch for N={length}, radix={radix}"
                );
            }
            assert_eq!(schedule.stage_radices.iter().product::<usize>(), length);
            assert!(schedule.register_boost_stage_radix.is_none());
        }

        for length in [17usize, 34, 46, 51, 69] {
            assert!(matches!(
                plan_nvidia_vulkan_small_mixed_radix_registers(length, 2),
                Err(VkFftError::UnsupportedKernelPath(_))
            ));
        }
    }

    #[test]
    fn every_smooth_factor_combination_selects_a_register_table_leaf() {
        let odd_primes = [5usize, 7, 11, 13];
        let mut covered = 0usize;
        for exponent2 in 0usize..=4 {
            for exponent3 in 0usize..=2 {
                for mask in 0usize..(1usize << odd_primes.len()) {
                    let mut length = 1usize << exponent2;
                    length = length.checked_mul(3usize.pow(exponent3 as u32)).unwrap();
                    for (bit, prime) in odd_primes.iter().enumerate() {
                        if mask & (1usize << bit) != 0 {
                            length = length.checked_mul(*prime).unwrap();
                        }
                    }
                    if length < 3 || length.is_power_of_two() {
                        continue;
                    }
                    covered += 1;
                    let schedule =
                        plan_nvidia_vulkan_small_mixed_radix_registers(length, 2).unwrap_or_else(
                            |error| panic!(
                                "smooth factor combination N={length} (2^{exponent2} 3^{exponent3} mask={mask:#06b}) failed: {error}"
                            ),
                        );
                    assert_eq!(schedule.stage_radices.iter().product::<usize>(), length);
                    assert_eq!(schedule.register_boost, 1);
                }
            }
        }
        assert_eq!(covered, 235);
    }

    #[test]
    fn power_of_two_register_schedule_matches_upstream_radix_eight_grouping() {
        let schedule = plan_nvidia_vulkan_power_of_two_radix_registers(1024, 1, 1).unwrap();
        assert_eq!(schedule.registers_per_thread_per_radix[2], 8);
        assert_eq!(schedule.registers_per_thread_per_radix[4], 8);
        assert_eq!(schedule.registers_per_thread_per_radix[8], 8);
        assert_eq!(schedule.registers_per_thread, 8);
        assert_eq!(schedule.min_registers_per_thread, 8);
        assert!(schedule.is_good_sequence);
        assert_eq!(schedule.register_boost_stage_radix, None);
        assert_eq!(schedule.stage_radices, vec![8, 8, 8, 2]);
        assert_eq!(schedule.stage_radix_multipliers[8], 3);
        assert_eq!(schedule.stage_radix_multipliers[2], 1);
    }

    #[test]
    fn register_boost_extracts_vkfft_final_stage() {
        let schedule = plan_nvidia_vulkan_power_of_two_radix_registers(16_384, 1, 4).unwrap();
        assert_eq!(schedule.register_boost_stage_radix, Some(4));
        assert_eq!(schedule.stage_radices, vec![8, 8, 8, 8, 4]);
        assert_eq!(schedule.stage_radix_multipliers[8], 4);
        assert_eq!(schedule.stage_radix_multipliers[4], 0);
        assert_eq!(schedule.registers_per_thread, 8);
        assert_eq!(schedule.min_registers_per_thread, 8);
    }

    #[test]
    fn boost_only_stage_uses_vkfft_two_register_fallback() {
        let schedule = plan_nvidia_vulkan_power_of_two_radix_registers(4, 1, 4).unwrap();
        assert_eq!(schedule.register_boost_stage_radix, Some(4));
        assert_eq!(schedule.stage_radices, vec![4]);
        assert_eq!(schedule.registers_per_thread, 2);
        assert_eq!(schedule.min_registers_per_thread, 2);
    }

    #[test]
    fn direct_rader_four_step_shape_preserves_f16_coalescing_width() {
        let device = DeviceProfile {
            shared_memory_bytes: 8 * 1024,
            shared_memory_pow2_bytes: 8 * 1024,
            max_threads_per_block: 256,
            max_workgroup_size: [256, 256, 64],
            ..DeviceProfile::generic(Backend::Vulkan, GpuVendor::Amd)
        };
        let request = FourStepAxisBlockRequest {
            axis_upload_id: 1,
            stage_start_size: 64,
            transform_count: 2_176,
            outer_batch_count: 1,
            perform_zero_padding: false,
            grouped_batch_override: None,
        };
        let f32 = plan_gpu_axis0_direct_rader_four_step_default_block_from_shape_for_precision(
            3,
            47,
            24,
            47,
            request,
            Precision::F32,
            8,
            device,
        )
        .unwrap()
        .unwrap();
        let f16 = plan_gpu_axis0_direct_rader_four_step_default_block_from_shape_for_precision(
            3,
            47,
            24,
            47,
            request,
            Precision::F16StorageF32Compute,
            8,
            device,
        )
        .unwrap()
        .unwrap();
        assert_eq!((f32.grouped_batch, f32.local_size_x), (4, 4));
        assert_eq!((f16.grouped_batch, f16.local_size_x), (8, 8));
        assert_eq!((f16.threads_per_transform, f16.local_size_y), (24, 24));
    }

    #[test]
    fn four_step_shape_respects_mixed_storage_coalescing_width() {
        let device = DeviceProfile {
            shared_memory_bytes: 8 * 1024,
            shared_memory_pow2_bytes: 8 * 1024,
            max_threads_per_block: 1024,
            max_workgroup_size: [1024, 1024, 64],
            ..DeviceProfile::generic(Backend::Vulkan, GpuVendor::Amd)
        };
        let request = FourStepAxisBlockRequest {
            axis_upload_id: 1,
            stage_start_size: 47,
            transform_count: 517,
            outer_batch_count: 1,
            perform_zero_padding: false,
            grouped_batch_override: None,
        };
        let f32 = plan_gpu_axis0_four_step_default_block_from_shape_for_precision(
            3,
            17,
            2,
            request,
            Precision::F32,
            8,
            device,
        )
        .unwrap()
        .unwrap();
        let f16 = plan_gpu_axis0_four_step_default_block_from_shape_for_precision(
            3,
            17,
            2,
            request,
            Precision::F16StorageF32Compute,
            8,
            device,
        )
        .unwrap()
        .unwrap();
        assert_eq!((f32.grouped_batch, f32.local_size_x), (47, 47));
        assert_eq!((f16.grouped_batch, f16.local_size_x), (32, 32));
        assert_eq!((f16.threads_per_transform, f16.local_size_y), (2, 2));
    }

    #[test]
    fn batch_aware_upload_schedule_embeds_per_upload_register_metadata() {
        let device = nvidia_vulkan(48 * 1024);
        let schedule = plan_nvidia_vulkan_power_of_two_stockham_uploads_for_batches(
            1_048_576,
            2,
            Precision::F32,
            device,
        )
        .unwrap();
        assert_eq!(schedule.batch_count, 2);
        assert_eq!(schedule.axis_split, vec![1024, 1024]);
        assert_eq!(schedule.radix_schedules.len(), 2);
        for radix in &schedule.radix_schedules {
            assert_eq!(radix.fft_len, 1024);
            assert_eq!(radix.rhs_transform_count, 2048);
            assert_eq!(radix.register_boost, 1);
            assert_eq!(radix.stage_radices, vec![8, 8, 8, 2]);
        }
    }
}
