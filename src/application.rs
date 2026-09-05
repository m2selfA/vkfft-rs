//! High-level typed transform planning facade.
//!
//! Lower-level IR modules remain public for backend work, while `TransformIr`
//! provides the normal Application -> Plan -> Code entry point: build one host
//! plan from `FftConfig`, select the appropriate executable IR family, then hand
//! that typed object to a CPU oracle or backend runtime.

use crate::complex::Complex64;
use crate::config::{DeviceProfile, Direction, FftConfig, Precision, TransformKind};
use crate::double_double::{ComplexDoubleDouble, DoubleDouble};
use crate::double_double_ir::{
    DoubleDoubleNdFftIr, DoubleDoubleNdR2rIr, DoubleDoubleNdRealFftIr, DoubleDoubleOneDimIr,
    DoubleDoubleR2rIr, DoubleDoubleRealFftIr, execute_double_double_c2r_ir,
    execute_double_double_c2r_ir_f64_storage, execute_double_double_nd_c2r_ir,
    execute_double_double_nd_c2r_ir_f64_storage, execute_double_double_nd_ir,
    execute_double_double_nd_ir_f64_storage, execute_double_double_nd_r2c_ir,
    execute_double_double_nd_r2c_ir_f64_storage, execute_double_double_nd_r2r_ir,
    execute_double_double_nd_r2r_ir_f64_storage, execute_double_double_one_dim_ir,
    execute_double_double_one_dim_ir_f64_storage, execute_double_double_r2c_ir,
    execute_double_double_r2c_ir_f64_storage, execute_double_double_r2r_ir,
    execute_double_double_r2r_ir_f64_storage,
};
use crate::error::{Result, VkFftError};
use crate::nd_ir::{NdFftIr, execute_nd_fft_ir};
use crate::nd_real_ir::{NdRealFftIr, execute_nd_c2r_ir, execute_nd_r2c_ir};
use crate::one_dim_ir::{OneDimFftIr, execute_one_dim_fft_ir};
use crate::planner::FftPlan;
use crate::r2r_ir::{NdR2rIr, R2rIr, execute_nd_r2r_ir, execute_r2r_ir};
use crate::real_ir::{RealFftIr, RealFftKind, execute_c2r_ir, execute_r2c_ir};

#[derive(Debug, Clone, PartialEq)]
pub enum TransformIr {
    Complex1d(OneDimFftIr),
    Complex1dDoubleDouble(DoubleDoubleOneDimIr),
    ComplexNdDoubleDouble(DoubleDoubleNdFftIr),
    ComplexNd(NdFftIr),
    RealDoubleDouble(DoubleDoubleRealFftIr),
    RealNdDoubleDouble(DoubleDoubleNdRealFftIr),
    Real(RealFftIr),
    RealNd(NdRealFftIr),
    RealToRealDoubleDouble(DoubleDoubleR2rIr),
    RealToRealNdDoubleDouble(DoubleDoubleNdR2rIr),
    RealToReal(R2rIr),
    RealToRealNd(NdR2rIr),
}

impl TransformIr {
    pub fn build(config: FftConfig, direction: Direction, device: DeviceProfile) -> Result<Self> {
        match config.transform {
            TransformKind::RealToComplex if direction != Direction::Forward => {
                return Err(VkFftError::UnsupportedKernelPath(
                    "RealToComplex is a forward transform; use Direction::Forward",
                ));
            }
            TransformKind::ComplexToReal if direction != Direction::Inverse => {
                return Err(VkFftError::UnsupportedKernelPath(
                    "ComplexToReal is an inverse transform; use Direction::Inverse",
                ));
            }
            _ => {}
        }
        let config = config.resolve_tuning_for_device(device);
        let plan = FftPlan::build_for_device(config, device)?;
        match plan.config.transform {
            TransformKind::ComplexToComplex => {
                if plan.config.dimensions.len() == 1 {
                    if matches!(
                        plan.config.precision,
                        Precision::DoubleDouble | Precision::DoubleDoubleF64Storage
                    ) {
                        Ok(Self::Complex1dDoubleDouble(
                            DoubleDoubleOneDimIr::build_for_device(&plan, direction, device)?,
                        ))
                    } else {
                        Ok(Self::Complex1d(
                            OneDimFftIr::build(&plan, direction, device)?
                                .with_axis0_single_upload_block(device)?,
                        ))
                    }
                } else if matches!(
                    plan.config.precision,
                    Precision::DoubleDouble | Precision::DoubleDoubleF64Storage
                ) {
                    let axis1_grouped_batch_override = plan
                        .config
                        .dimensions
                        .len()
                        .checked_sub(2)
                        .and_then(|axis| plan.config.grouped_batch_for_axis(axis));
                    Ok(Self::ComplexNdDoubleDouble(
                        DoubleDoubleNdFftIr::build_for_device(&plan, direction, device)?
                            .with_grouped_stockham_axis_blocks(
                                axis1_grouped_batch_override,
                                device,
                            )?,
                    ))
                } else {
                    Ok(Self::ComplexNd(NdFftIr::build(&plan, direction, device)?))
                }
            }
            TransformKind::RealToComplex | TransformKind::ComplexToReal => {
                if matches!(
                    plan.config.precision,
                    Precision::DoubleDouble | Precision::DoubleDoubleF64Storage
                ) {
                    if plan.config.dimensions.len() == 1 {
                        Ok(Self::RealDoubleDouble(
                            DoubleDoubleRealFftIr::build_for_device(&plan, device)?
                                .with_grouped_stockham_child_block(device)?,
                        ))
                    } else {
                        let axis1_grouped_batch_override = plan
                            .config
                            .dimensions
                            .len()
                            .checked_sub(2)
                            .and_then(|axis| plan.config.grouped_batch_for_axis(axis));
                        Ok(Self::RealNdDoubleDouble(
                            DoubleDoubleNdRealFftIr::build_for_device(&plan, device)?
                                .with_grouped_stockham_axis_blocks(
                                    axis1_grouped_batch_override,
                                    device,
                                )?,
                        ))
                    }
                } else if plan.config.dimensions.len() == 1 {
                    Ok(Self::Real(RealFftIr::build_for_device_plan(&plan, device)?))
                } else {
                    Ok(Self::RealNd(NdRealFftIr::build_for_device_plan(
                        &plan, device,
                    )?))
                }
            }
            TransformKind::Dct(_) | TransformKind::Dst(_) => {
                if matches!(
                    plan.config.precision,
                    Precision::DoubleDouble | Precision::DoubleDoubleF64Storage
                ) {
                    if plan.config.dimensions.len() == 1 {
                        Ok(Self::RealToRealDoubleDouble(
                            DoubleDoubleR2rIr::build_for_device(&plan, direction, device)?,
                        ))
                    } else {
                        let axis1_grouped_batch_override = plan
                            .config
                            .dimensions
                            .len()
                            .checked_sub(2)
                            .and_then(|axis| plan.config.grouped_batch_for_axis(axis));
                        Ok(Self::RealToRealNdDoubleDouble(
                            DoubleDoubleNdR2rIr::build_for_device(&plan, direction, device)?
                                .with_grouped_stockham_axis_blocks(
                                    axis1_grouped_batch_override,
                                    device,
                                )?,
                        ))
                    }
                } else if plan.config.dimensions.len() == 1 {
                    Ok(Self::RealToReal(R2rIr::build(&plan, direction, device)?))
                } else {
                    Ok(Self::RealToRealNd(NdR2rIr::build(
                        &plan, direction, device,
                    )?))
                }
            }
        }
    }

    pub fn direction(&self) -> Direction {
        match self {
            Self::Complex1d(ir) => ir.direction(),
            Self::Complex1dDoubleDouble(ir) => ir.direction(),
            Self::ComplexNdDoubleDouble(ir) => ir.direction,
            Self::ComplexNd(ir) => ir.direction,
            Self::RealDoubleDouble(ir) => ir.direction,
            Self::RealNdDoubleDouble(ir) => match ir.kind {
                RealFftKind::RealToComplex => Direction::Forward,
                RealFftKind::ComplexToReal => Direction::Inverse,
            },
            Self::Real(ir) => match ir.kind {
                RealFftKind::RealToComplex => Direction::Forward,
                RealFftKind::ComplexToReal => Direction::Inverse,
            },
            Self::RealNd(ir) => match ir.kind {
                RealFftKind::RealToComplex => Direction::Forward,
                RealFftKind::ComplexToReal => Direction::Inverse,
            },
            Self::RealToRealDoubleDouble(ir) => ir.direction,
            Self::RealToRealNdDoubleDouble(ir) => ir.direction,
            Self::RealToReal(ir) => ir.direction,
            Self::RealToRealNd(ir) => ir.direction,
        }
    }

    pub fn batch_count(&self) -> usize {
        match self {
            Self::Complex1d(ir) => ir.batch_count(),
            Self::Complex1dDoubleDouble(ir) => ir.batch_count(),
            Self::ComplexNdDoubleDouble(ir) => ir.batch_count,
            Self::ComplexNd(ir) => ir.batch_count,
            Self::RealDoubleDouble(ir) => ir.batch_count,
            Self::RealNdDoubleDouble(ir) => ir.batch_count,
            Self::Real(ir) => ir.batch_count,
            Self::RealNd(ir) => ir.batch_count,
            Self::RealToRealDoubleDouble(ir) => ir.batch_count,
            Self::RealToRealNdDoubleDouble(ir) => ir.batch_count,
            Self::RealToReal(ir) => ir.batch_count,
            Self::RealToRealNd(ir) => ir.batch_count,
        }
    }

    pub fn execute_complex_reference(&self, input: &[Complex64]) -> Result<Vec<Complex64>> {
        match self {
            Self::Complex1d(ir) => execute_one_dim_fft_ir(ir, input),
            Self::ComplexNd(ir) => execute_nd_fft_ir(ir, input),
            Self::Complex1dDoubleDouble(ir) => {
                execute_double_double_one_dim_ir_f64_storage(ir, input)
            }
            Self::ComplexNdDoubleDouble(ir) => execute_double_double_nd_ir_f64_storage(ir, input),
            _ => Err(VkFftError::UnsupportedKernelPath(
                "complex reference execution requires a C2C transform",
            )),
        }
    }

    pub fn execute_double_double_reference(
        &self,
        input: &[ComplexDoubleDouble],
    ) -> Result<Vec<ComplexDoubleDouble>> {
        match self {
            Self::Complex1dDoubleDouble(ir) => execute_double_double_one_dim_ir(ir, input),
            Self::ComplexNdDoubleDouble(ir) => execute_double_double_nd_ir(ir, input),
            _ => Err(VkFftError::UnsupportedKernelPath(
                "double-double reference execution requires a double-double C2C transform",
            )),
        }
    }

    pub fn execute_r2c_reference(&self, input: &[f64]) -> Result<Vec<Complex64>> {
        match self {
            Self::Real(ir) if ir.kind == RealFftKind::RealToComplex => execute_r2c_ir(ir, input),
            Self::RealNd(ir) if ir.kind == RealFftKind::RealToComplex => {
                execute_nd_r2c_ir(ir, input)
            }
            Self::RealDoubleDouble(ir) if ir.kind == RealFftKind::RealToComplex => {
                execute_double_double_r2c_ir_f64_storage(ir, input)
            }
            Self::RealNdDoubleDouble(ir) if ir.kind == RealFftKind::RealToComplex => {
                execute_double_double_nd_r2c_ir_f64_storage(ir, input)
            }
            _ => Err(VkFftError::UnsupportedKernelPath(
                "R2C reference execution requires a RealToComplex transform",
            )),
        }
    }

    pub fn execute_c2r_reference(&self, input: &[Complex64]) -> Result<Vec<f64>> {
        match self {
            Self::Real(ir) if ir.kind == RealFftKind::ComplexToReal => execute_c2r_ir(ir, input),
            Self::RealNd(ir) if ir.kind == RealFftKind::ComplexToReal => {
                execute_nd_c2r_ir(ir, input)
            }
            Self::RealDoubleDouble(ir) if ir.kind == RealFftKind::ComplexToReal => {
                execute_double_double_c2r_ir_f64_storage(ir, input)
            }
            Self::RealNdDoubleDouble(ir) if ir.kind == RealFftKind::ComplexToReal => {
                execute_double_double_nd_c2r_ir_f64_storage(ir, input)
            }
            _ => Err(VkFftError::UnsupportedKernelPath(
                "C2R reference execution requires a ComplexToReal transform",
            )),
        }
    }

    pub fn execute_double_double_r2c_reference(
        &self,
        input: &[DoubleDouble],
    ) -> Result<Vec<ComplexDoubleDouble>> {
        match self {
            Self::RealDoubleDouble(ir) if ir.kind == RealFftKind::RealToComplex => {
                execute_double_double_r2c_ir(ir, input)
            }
            Self::RealNdDoubleDouble(ir) if ir.kind == RealFftKind::RealToComplex => {
                execute_double_double_nd_r2c_ir(ir, input)
            }
            _ => Err(VkFftError::UnsupportedKernelPath(
                "double-double R2C reference execution requires a DD RealToComplex transform",
            )),
        }
    }

    pub fn execute_double_double_c2r_reference(
        &self,
        input: &[ComplexDoubleDouble],
    ) -> Result<Vec<DoubleDouble>> {
        match self {
            Self::RealDoubleDouble(ir) if ir.kind == RealFftKind::ComplexToReal => {
                execute_double_double_c2r_ir(ir, input)
            }
            Self::RealNdDoubleDouble(ir) if ir.kind == RealFftKind::ComplexToReal => {
                execute_double_double_nd_c2r_ir(ir, input)
            }
            _ => Err(VkFftError::UnsupportedKernelPath(
                "double-double C2R reference execution requires a DD ComplexToReal transform",
            )),
        }
    }

    pub fn execute_r2r_reference(&self, input: &[f64]) -> Result<Vec<f64>> {
        match self {
            Self::RealToRealDoubleDouble(ir) => execute_double_double_r2r_ir_f64_storage(ir, input),
            Self::RealToRealNdDoubleDouble(ir) => {
                execute_double_double_nd_r2r_ir_f64_storage(ir, input)
            }
            Self::RealToReal(ir) => execute_r2r_ir(ir, input),
            Self::RealToRealNd(ir) => execute_nd_r2r_ir(ir, input),
            _ => Err(VkFftError::UnsupportedKernelPath(
                "R2R reference execution requires a DCT/DST transform",
            )),
        }
    }

    pub fn execute_double_double_r2r_reference(
        &self,
        input: &[DoubleDouble],
    ) -> Result<Vec<DoubleDouble>> {
        match self {
            Self::RealToRealDoubleDouble(ir) => execute_double_double_r2r_ir(ir, input),
            Self::RealToRealNdDoubleDouble(ir) => execute_double_double_nd_r2r_ir(ir, input),
            _ => Err(VkFftError::UnsupportedKernelPath(
                "double-double R2R reference execution requires a DD DCT/DST transform",
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{Backend, DctType, GpuVendor};

    fn device() -> DeviceProfile {
        DeviceProfile {
            shared_memory_bytes: 128 * 1024,
            shared_memory_pow2_bytes: 128 * 1024,
            ..DeviceProfile::generic(Backend::Vulkan, GpuVendor::Nvidia)
        }
    }

    fn sample_real(length: usize) -> Vec<f64> {
        (0..length)
            .map(|index| {
                let x = index as f64;
                (0.071 * x).sin() + 0.23 * (0.041 * x).cos() + 0.0007 * x
            })
            .collect()
    }

    fn max_real_error(lhs: &[f64], rhs: &[f64]) -> f64 {
        lhs.iter()
            .zip(rhs)
            .map(|(lhs, rhs)| (lhs - rhs).abs())
            .fold(0.0, f64::max)
    }

    fn max_complex_error(lhs: &[Complex64], rhs: &[Complex64]) -> f64 {
        lhs.iter()
            .zip(rhs)
            .map(|(lhs, rhs)| (*lhs - *rhs).norm_sqr().sqrt())
            .fold(0.0, f64::max)
    }

    fn direct_dct_ii(input: &[f64]) -> Vec<f64> {
        let n = input.len();
        (0..n)
            .map(|k| {
                input
                    .iter()
                    .enumerate()
                    .map(|(j, value)| {
                        2.0 * value
                            * (core::f64::consts::PI * (j as f64 + 0.5) * k as f64 / n as f64).cos()
                    })
                    .sum()
            })
            .collect()
    }

    fn direct_dct_iii(input: &[f64]) -> Vec<f64> {
        let n = input.len();
        (0..n)
            .map(|k| {
                input[0]
                    + input
                        .iter()
                        .enumerate()
                        .skip(1)
                        .map(|(j, value)| {
                            2.0 * value
                                * (core::f64::consts::PI * j as f64 * (k as f64 + 0.5) / n as f64)
                                    .cos()
                        })
                        .sum::<f64>()
            })
            .collect()
    }

    fn direct_nd_r2c(input: &[f64], rows: usize, cols: usize) -> Vec<Complex64> {
        let compact_cols = cols / 2 + 1;
        let mut output = vec![Complex64::default(); rows * compact_cols];
        for k0 in 0..rows {
            for k1 in 0..compact_cols {
                let mut sum = Complex64::default();
                for n0 in 0..rows {
                    for n1 in 0..cols {
                        let angle = -core::f64::consts::TAU
                            * (n0 as f64 * k0 as f64 / rows as f64
                                + n1 as f64 * k1 as f64 / cols as f64);
                        sum += Complex64::new(input[n0 * cols + n1], 0.0) * Complex64::exp_i(angle);
                    }
                }
                output[k0 * compact_cols + k1] = sum;
            }
        }
        output
    }

    #[test]
    fn public_grouped_batch_override_reaches_supported_stockham_and_rader_paths() {
        let mut profile = device();
        profile.shared_memory_bytes = 48 * 1024;
        profile.shared_memory_pow2_bytes = 32 * 1024;
        profile.max_threads_per_block = 1024;
        profile.max_workgroup_size = [1024, 1024, 64];

        let single = TransformIr::build(
            FftConfig::new(vec![64])
                .with_batch_count(32)
                .with_grouped_batch(0, 8)
                .unwrap(),
            Direction::Forward,
            profile,
        )
        .unwrap();
        let TransformIr::Complex1d(OneDimFftIr::Recursive(single)) = single else {
            panic!("groupedBatch single-upload path should remain recursive Stockham");
        };
        let crate::recursive_ir::RecursiveFftNodeIr::Stockham(single_kernel) = &single.root else {
            panic!("groupedBatch single-upload path should remain one Stockham kernel");
        };
        assert_eq!(single.axis0_grouped_batch_override, Some(8));
        assert_eq!(single_kernel.workgroup_grouping.transforms_per_workgroup, 8);
        assert_eq!(
            [
                single_kernel.workgroup_size.x,
                single_kernel.workgroup_size.y
            ],
            [8, 8]
        );
        assert_eq!(single_kernel.dispatch.x, 4);

        let two = TransformIr::build(
            FftConfig::new(vec![1_048_576])
                .with_grouped_batch(0, 2)
                .unwrap(),
            Direction::Forward,
            profile,
        )
        .unwrap();
        let TransformIr::Complex1d(OneDimFftIr::Recursive(two)) = two else {
            panic!("groupedBatch two-upload path should remain recursive Stockham");
        };
        let four_step = two.four_step_plan.as_ref().unwrap();
        assert_eq!(
            four_step
                .uploads
                .iter()
                .map(|upload| upload.axis_block.unwrap().grouped_batch)
                .collect::<Vec<_>>(),
            vec![4, 2]
        );
        let kernels = two.four_step_stockham_upload_kernels().unwrap().unwrap();
        assert_eq!(
            kernels
                .iter()
                .map(|kernel| [kernel.workgroup_size.x, kernel.workgroup_size.y])
                .collect::<Vec<_>>(),
            vec![[4, 128], [2, 128]]
        );
        assert_eq!(
            kernels
                .iter()
                .map(|kernel| kernel.dispatch.x)
                .collect::<Vec<_>>(),
            vec![256, 512]
        );

        let medium = TransformIr::build(
            FftConfig::new(vec![49_152])
                .with_grouped_batch(0, 5)
                .unwrap(),
            Direction::Forward,
            profile,
        )
        .unwrap();
        let TransformIr::Complex1d(OneDimFftIr::Recursive(medium)) = medium else {
            panic!("49,152-point groupedBatch path should remain recursive Stockham");
        };
        let medium_four_step = medium.four_step_plan.as_ref().unwrap();
        assert!(
            medium_four_step
                .uploads
                .iter()
                .all(|upload| upload.axis_block.is_some())
        );
        let medium_kernels = medium.four_step_stockham_upload_kernels().unwrap().unwrap();
        assert!(
            medium_kernels
                .iter()
                .all(|kernel| kernel.workgroup_grouping.transforms_per_workgroup > 1)
        );
        assert_eq!(
            medium_four_step
                .uploads
                .last()
                .and_then(|upload| upload.axis_block)
                .map(|block| block.grouped_batch),
            Some(5)
        );
        assert!(medium_kernels.iter().any(|kernel| {
            !kernel
                .batch_count
                .is_multiple_of(kernel.workgroup_grouping.transforms_per_workgroup)
        }));

        let three = TransformIr::build(
            FftConfig::new(vec![8_388_608])
                .with_grouped_batch(0, 2)
                .unwrap(),
            Direction::Forward,
            profile,
        )
        .unwrap();
        let TransformIr::Complex1d(OneDimFftIr::Recursive(three)) = three else {
            panic!("groupedBatch three-upload path should remain recursive Stockham");
        };
        let three_uploads = three.four_step_plan.as_ref().unwrap();
        assert_eq!(
            three_uploads
                .uploads
                .iter()
                .map(|upload| upload.axis_block.unwrap().grouped_batch)
                .collect::<Vec<_>>(),
            vec![6, 12, 2]
        );
        let three_kernels = three.four_step_stockham_upload_kernels().unwrap().unwrap();
        assert_eq!(
            three_kernels
                .iter()
                .map(|kernel| [kernel.workgroup_size.x, kernel.workgroup_size.y])
                .collect::<Vec<_>>(),
            vec![[6, 32], [12, 16], [2, 32]]
        );
        assert_eq!(
            three_kernels
                .iter()
                .map(|kernel| kernel.dispatch.x)
                .collect::<Vec<_>>(),
            vec![5462, 5462, 16384]
        );
        let rader = TransformIr::build(
            FftConfig::new(vec![257])
                .with_batch_count(32)
                .with_grouped_batch(0, 5)
                .unwrap(),
            Direction::Forward,
            profile,
        )
        .unwrap();
        let TransformIr::Complex1d(OneDimFftIr::Recursive(rader)) = rader else {
            panic!("groupedBatch p257 should remain recursive Rader");
        };
        let crate::recursive_ir::RecursiveFftNodeIr::FftRader(rader) = &rader.root else {
            panic!("groupedBatch p257 should keep an FFT-Rader root");
        };
        let block = rader
            .axis_batch_block
            .expect("groupedBatch p257 should carry its axis block");
        assert_eq!(block.threads_per_transform, 17);
        assert_eq!(block.grouped_batch, 5);
        assert_eq!([block.local_size_x, block.local_size_y], [17, 5]);
        let crate::recursive_ir::RecursiveFftNodeIr::Stockham(rader_kernel) =
            &rader.forward_recursive().unwrap().root
        else {
            panic!("groupedBatch p257 convolution should remain one Stockham root");
        };
        assert_eq!(rader_kernel.dispatch.x, 7);
        assert_eq!(
            rader
                .internal_register_schedule
                .as_ref()
                .map(|schedule| schedule.container_fft_num),
            Some(1)
        );

        let direct = TransformIr::build(
            FftConfig::new(vec![47])
                .with_batch_count(32)
                .with_grouped_batch(0, 3)
                .unwrap(),
            Direction::Forward,
            profile,
        )
        .unwrap();
        let TransformIr::Complex1d(OneDimFftIr::Recursive(direct)) = direct else {
            panic!("groupedBatch p47 should remain recursive direct Rader");
        };
        let crate::recursive_ir::RecursiveFftNodeIr::DirectRader(direct) = &direct.root else {
            panic!("groupedBatch p47 should keep a direct-Rader root");
        };
        let direct_block = direct
            .axis_batch_block
            .expect("groupedBatch p47 should carry its axis block");
        assert_eq!(direct_block.threads_per_transform, 24);
        assert_eq!(direct_block.grouped_batch, 1);
        assert_eq!([direct.workgroup_size.x, direct.workgroup_size.y], [24, 1]);
        assert_eq!(direct.dispatch.x, 32);
        let p17 = TransformIr::build(
            FftConfig::new(vec![17])
                .with_batch_count(32)
                .with_grouped_batch(0, 2)
                .unwrap(),
            Direction::Forward,
            profile,
        )
        .unwrap();
        let TransformIr::Complex1d(OneDimFftIr::Recursive(p17)) = p17 else {
            panic!("groupedBatch p17 should remain recursive Rader");
        };
        let crate::recursive_ir::RecursiveFftNodeIr::FftRader(p17) = &p17.root else {
            panic!("groupedBatch p17 should keep an FFT-Rader root");
        };
        let p17_block = p17
            .axis_batch_block
            .expect("groupedBatch p17 should carry its exact caller block");
        assert_eq!(p17_block.threads_per_transform, 2);
        assert_eq!(p17_block.grouped_batch, 2);
        assert!(p17_block.transforms_on_x);
        assert!(p17_block.axis_swapped);
        assert_eq!([p17_block.local_size_x, p17_block.local_size_y], [2, 2]);
    }

    #[test]
    fn high_level_real_shape_follows_fixed_upstream_big_even_threshold() {
        let roomy = device();
        let small = TransformIr::build(
            FftConfig::new(vec![30]).with_transform(TransformKind::RealToComplex),
            Direction::Forward,
            roomy,
        )
        .unwrap();
        let TransformIr::Real(small) = small else {
            panic!("small real transform should build ordinary real IR");
        };
        assert_eq!(small.algorithm, crate::RealFftAlgorithm::FullComplex);
        assert_eq!(small.transform_len(), 30);

        let constrained = DeviceProfile {
            shared_memory_bytes: 128,
            shared_memory_pow2_bytes: 128,
            ..DeviceProfile::generic(Backend::Vulkan, GpuVendor::Nvidia)
        };
        let big_even = TransformIr::build(
            FftConfig::new(vec![30]).with_transform(TransformKind::RealToComplex),
            Direction::Forward,
            constrained,
        )
        .unwrap();
        let TransformIr::Real(big_even) = big_even else {
            panic!("constrained real transform should build ordinary real IR");
        };
        assert_eq!(big_even.algorithm, crate::RealFftAlgorithm::EvenHalfSize);
        assert_eq!(big_even.transform_len(), 15);

        // N=94 initially fits exactly in 94 F32 complex values, but its p47
        // direct-Rader child reserves 46 complex values. Upstream's second R2C
        // check therefore switches the real transform to N/2.
        let direct_rader_reserved = DeviceProfile {
            shared_memory_bytes: 94 * 8,
            shared_memory_pow2_bytes: 94 * 8,
            ..DeviceProfile::generic(Backend::Vulkan, GpuVendor::Nvidia)
        };
        let reserved = TransformIr::build(
            FftConfig::new(vec![94]).with_transform(TransformKind::RealToComplex),
            Direction::Forward,
            direct_rader_reserved,
        )
        .unwrap();
        let TransformIr::Real(reserved) = reserved else {
            panic!("Rader-reserved real transform should build ordinary real IR");
        };
        assert_eq!(reserved.algorithm, crate::RealFftAlgorithm::EvenHalfSize);
        assert_eq!(reserved.transform_len(), 47);

        // A fully factorable FFT-Rader sequence can also force the two-upload
        // branch even while the raw shared-memory capacity still holds N.
        let force_two_upload = TransformIr::build(
            FftConfig::new(vec![8_704]).with_transform(TransformKind::RealToComplex),
            Direction::Forward,
            roomy,
        )
        .unwrap();
        let TransformIr::Real(force_two_upload) = force_two_upload else {
            panic!("force-two-upload real transform should build ordinary real IR");
        };
        assert_eq!(
            force_two_upload.algorithm,
            crate::RealFftAlgorithm::EvenHalfSize
        );
        assert_eq!(force_two_upload.transform_len(), 4_352);

        // p103 makes N=206 reach the Bluestein phase. NVIDIA/F32's fixed table
        // maps 193 <= N < 257 to padded size 512, so the scheduler's third R2C
        // check switches to N/2 when raw N fits but that padded convolution does not.
        let bluestein_recheck = DeviceProfile {
            shared_memory_bytes: 300 * 8,
            shared_memory_pow2_bytes: 300 * 8,
            ..DeviceProfile::generic(Backend::Vulkan, GpuVendor::Nvidia)
        };
        let padded_too_large = TransformIr::build(
            FftConfig::new(vec![206]).with_transform(TransformKind::RealToComplex),
            Direction::Forward,
            bluestein_recheck,
        )
        .unwrap();
        let TransformIr::Real(padded_too_large) = padded_too_large else {
            panic!("Bluestein-rechecked real transform should build ordinary real IR");
        };
        assert_eq!(
            padded_too_large.algorithm,
            crate::RealFftAlgorithm::EvenHalfSize
        );
        assert_eq!(padded_too_large.transform_len(), 103);

        let bluestein_fits = DeviceProfile {
            shared_memory_bytes: 512 * 8,
            shared_memory_pow2_bytes: 512 * 8,
            ..DeviceProfile::generic(Backend::Vulkan, GpuVendor::Nvidia)
        };
        let padded_fits = TransformIr::build(
            FftConfig::new(vec![206]).with_transform(TransformKind::RealToComplex),
            Direction::Forward,
            bluestein_fits,
        )
        .unwrap();
        let TransformIr::Real(padded_fits) = padded_fits else {
            panic!("roomy Bluestein real transform should build ordinary real IR");
        };
        assert_eq!(padded_fits.algorithm, crate::RealFftAlgorithm::FullComplex);
        assert_eq!(padded_fits.transform_len(), 206);

        // N=4106 = 2*p2053 is beyond the fixed NVIDIA/F32 table's final
        // 8192-padded coverage. The generic upstream search starts at 8211 and
        // selects the first good 2/3/5/7 sequence, 8232; a 6000-complex shared
        // budget therefore triggers the same third R2C switch to N/2.
        let generic_recheck = DeviceProfile {
            shared_memory_bytes: 6000 * 8,
            shared_memory_pow2_bytes: 6000 * 8,
            ..DeviceProfile::generic(Backend::Vulkan, GpuVendor::Nvidia)
        };
        let generic_padded = TransformIr::build(
            FftConfig::new(vec![4106]).with_transform(TransformKind::RealToComplex),
            Direction::Forward,
            generic_recheck,
        )
        .unwrap();
        let TransformIr::Real(generic_padded) = generic_padded else {
            panic!("generic-padded real transform should build ordinary real IR");
        };
        assert_eq!(
            generic_padded.algorithm,
            crate::RealFftAlgorithm::EvenHalfSize
        );
        assert_eq!(generic_padded.transform_len(), 2053);

        // Low-level IR construction intentionally keeps the previous portable
        // half-size-all-even behavior for direct backend/fusion testing.
        let portable_plan =
            FftPlan::build(FftConfig::new(vec![30]).with_transform(TransformKind::RealToComplex))
                .unwrap();
        let portable = RealFftIr::build(&portable_plan, roomy).unwrap();
        assert_eq!(portable.algorithm, crate::RealFftAlgorithm::EvenHalfSize);

        let nd_small = TransformIr::build(
            FftConfig::new(vec![3, 30]).with_transform(TransformKind::RealToComplex),
            Direction::Forward,
            roomy,
        )
        .unwrap();
        let TransformIr::RealNd(nd_small) = nd_small else {
            panic!("small ND real transform should build ND real IR");
        };
        assert_eq!(
            nd_small.real_axis.algorithm,
            crate::RealFftAlgorithm::FullComplex
        );
        assert_eq!(nd_small.real_axis.transform_len(), 30);
        assert_eq!(nd_small.compact_dimensions, vec![3, 16]);

        let nd_big_even = TransformIr::build(
            FftConfig::new(vec![3, 30]).with_transform(TransformKind::RealToComplex),
            Direction::Forward,
            constrained,
        )
        .unwrap();
        let TransformIr::RealNd(nd_big_even) = nd_big_even else {
            panic!("constrained ND real transform should build ND real IR");
        };
        assert_eq!(
            nd_big_even.real_axis.algorithm,
            crate::RealFftAlgorithm::EvenHalfSize
        );
        assert_eq!(nd_big_even.real_axis.transform_len(), 15);
        assert_eq!(nd_big_even.compact_dimensions, vec![3, 16]);
    }

    #[test]
    fn high_level_planner_uses_device_rader_defaults_unless_tuning_is_explicit() {
        let nvidia = DeviceProfile::generic(Backend::Vulkan, GpuVendor::Nvidia);
        let intel = DeviceProfile::generic(Backend::Vulkan, GpuVendor::Intel);

        let nvidia_p47 =
            TransformIr::build(FftConfig::new(vec![47]), Direction::Forward, nvidia).unwrap();
        assert!(matches!(
            nvidia_p47,
            TransformIr::Complex1d(OneDimFftIr::Recursive(ref ir))
                if matches!(ir.root, crate::RecursiveFftNodeIr::DirectRader(_))
        ));

        let intel_p47 =
            TransformIr::build(FftConfig::new(vec![47]), Direction::Forward, intel).unwrap();
        assert!(matches!(
            intel_p47,
            TransformIr::Complex1d(OneDimFftIr::Bluestein(_))
        ));

        let explicit_portable = TransformIr::build(
            FftConfig::new(vec![47]).with_tuning(crate::PlannerTuning::portable()),
            Direction::Forward,
            intel,
        )
        .unwrap();
        assert!(matches!(
            explicit_portable,
            TransformIr::Complex1d(OneDimFftIr::Recursive(ref ir))
                if matches!(ir.root, crate::RecursiveFftNodeIr::DirectRader(_))
        ));

        let amd = DeviceProfile::generic(Backend::Hip, GpuVendor::Amd);
        let amd_f64_p19 = TransformIr::build(
            FftConfig::new(vec![19]).with_precision(Precision::F64),
            Direction::Forward,
            amd,
        )
        .unwrap();
        assert!(matches!(
            amd_f64_p19,
            TransformIr::Complex1d(OneDimFftIr::Recursive(ref ir))
                if matches!(ir.root, crate::RecursiveFftNodeIr::DirectRader(_))
        ));

        let nvidia_dd_p47 = TransformIr::build(
            FftConfig::new(vec![47]).with_precision(Precision::DoubleDouble),
            Direction::Forward,
            nvidia,
        )
        .unwrap();
        assert!(matches!(
            nvidia_dd_p47,
            TransformIr::Complex1dDoubleDouble(DoubleDoubleOneDimIr::Bluestein(_))
        ));

        let constrained_nvidia = DeviceProfile {
            max_threads_per_block: 128,
            max_workgroup_size: [128, 128, 64],
            ..nvidia
        };
        let constrained_f32_p83 = TransformIr::build(
            FftConfig::new(vec![83]),
            Direction::Forward,
            constrained_nvidia,
        )
        .unwrap();
        assert!(matches!(
            constrained_f32_p83,
            TransformIr::Complex1d(OneDimFftIr::Bluestein(_))
        ));
        let constrained_f64_p83 = TransformIr::build(
            FftConfig::new(vec![83]).with_precision(Precision::F64),
            Direction::Forward,
            constrained_nvidia,
        )
        .unwrap();
        assert!(matches!(
            constrained_f64_p83,
            TransformIr::Complex1d(OneDimFftIr::Recursive(ref ir))
                if matches!(ir.root, crate::RecursiveFftNodeIr::DirectRader(_))
        ));
    }

    #[test]
    fn high_level_bluestein_uses_fixed_device_padding_and_child_scheduler() {
        let mut tuning = crate::PlannerTuning::portable();
        tuning.max_rader_fft_prime = 100;
        let mut profile = DeviceProfile::generic(Backend::Vulkan, GpuVendor::Nvidia);
        profile.shared_memory_bytes = 48 * 1024;
        profile.shared_memory_pow2_bytes = 32 * 1024;
        profile.max_threads_per_block = 1024;
        profile.max_workgroup_size = [1024, 1024, 64];

        let ir = TransformIr::build(
            FftConfig::new(vec![4001]).with_tuning(tuning),
            Direction::Forward,
            profile,
        )
        .unwrap();
        let TransformIr::Complex1d(OneDimFftIr::Bluestein(pipeline)) = ir else {
            panic!("forced high-level p4001 should use Bluestein");
        };
        assert_eq!(pipeline.convolution_len, 8_192);
        for fft in [&pipeline.forward_fft, &pipeline.inverse_fft] {
            let schedule = fft.stockham_upload_schedule.as_ref().unwrap();
            assert_eq!(schedule.register_boost, 1);
            assert_eq!(schedule.upload_count, 2);
            assert_eq!(schedule.axis_split, vec![128, 64]);
            assert!(fft.four_step_plan.is_some());
        }
        pipeline.validate().unwrap();
    }

    #[test]
    fn high_level_planner_routes_all_current_transform_families() {
        let mut bluestein_tuning = crate::PlannerTuning::portable();
        bluestein_tuning.max_rader_fft_prime = 100;
        let c2c = TransformIr::build(
            FftConfig::new(vec![103]).with_tuning(bluestein_tuning),
            Direction::Forward,
            device(),
        )
        .unwrap();
        assert!(matches!(
            c2c,
            TransformIr::Complex1d(OneDimFftIr::Bluestein(_))
        ));

        let nd =
            TransformIr::build(FftConfig::new(vec![3, 4]), Direction::Forward, device()).unwrap();
        assert!(matches!(nd, TransformIr::ComplexNd(_)));

        let r2c = TransformIr::build(
            FftConfig::new(vec![17]).with_transform(TransformKind::RealToComplex),
            Direction::Forward,
            device(),
        )
        .unwrap();
        assert!(matches!(
            r2c,
            TransformIr::Real(RealFftIr {
                kind: RealFftKind::RealToComplex,
                ..
            })
        ));

        let nd_r2c = TransformIr::build(
            FftConfig::new(vec![3, 8]).with_transform(TransformKind::RealToComplex),
            Direction::Forward,
            device(),
        )
        .unwrap();
        assert!(matches!(nd_r2c, TransformIr::RealNd(_)));

        let dct = TransformIr::build(
            FftConfig::new(vec![9]).with_transform(TransformKind::Dct(DctType::II)),
            Direction::Forward,
            device(),
        )
        .unwrap();
        assert!(matches!(dct, TransformIr::RealToReal(_)));

        let nd_dct = TransformIr::build(
            FftConfig::new(vec![3, 4]).with_transform(TransformKind::Dct(DctType::II)),
            Direction::Forward,
            device(),
        )
        .unwrap();
        assert!(matches!(nd_dct, TransformIr::RealToRealNd(_)));
    }

    #[test]
    fn medium_transform_family_goldens_cover_real_r2r_and_nd_paths() {
        for (length, force_bluestein) in [(206usize, true), (578usize, false)] {
            let left = length * 3 / 4;
            let mut tuning = crate::PlannerTuning::default();
            if force_bluestein {
                tuning = crate::PlannerTuning::portable();
                tuning.max_rader_fft_prime = 100;
            }
            let mut input = sample_real(length);
            for (offset, value) in input[left..].iter_mut().enumerate() {
                *value = 10_000.0 + offset as f64;
            }
            let mut manual = input.clone();
            manual[left..].fill(0.0);
            let forward = TransformIr::build(
                FftConfig::new(vec![length])
                    .with_transform(TransformKind::RealToComplex)
                    .with_tuning(tuning)
                    .with_zero_padding(0, left, length)
                    .unwrap(),
                Direction::Forward,
                device(),
            )
            .unwrap();
            match &forward {
                TransformIr::Real(ir) if force_bluestein => {
                    assert!(matches!(ir.transform, OneDimFftIr::Bluestein(_)));
                }
                TransformIr::Real(ir) => {
                    assert_eq!(ir.algorithm, crate::RealFftAlgorithm::FullComplex);
                    assert_eq!(ir.transform_len(), length);
                    assert!(matches!(ir.transform, OneDimFftIr::Recursive(_)));
                }
                _ => panic!("medium real golden did not route to 1D real IR"),
            }
            let actual_spectrum = forward.execute_r2c_reference(&input).unwrap();
            let complex = manual
                .iter()
                .map(|value| Complex64::new(*value, 0.0))
                .collect::<Vec<_>>();
            let expected_full = crate::reference::fft(&complex, Direction::Forward, false).unwrap();
            assert!(
                max_complex_error(&actual_spectrum, &expected_full[..length / 2 + 1])
                    <= 2.0e-8 * length as f64,
                "real R2C medium golden failed at N={length}"
            );

            let inverse = TransformIr::build(
                FftConfig::new(vec![length])
                    .with_transform(TransformKind::ComplexToReal)
                    .with_inverse_normalization(true)
                    .with_tuning(tuning)
                    .with_zero_padding(0, left, length)
                    .unwrap(),
                Direction::Inverse,
                device(),
            )
            .unwrap();
            let restored = inverse.execute_c2r_reference(&actual_spectrum).unwrap();
            assert!(restored[left..].iter().all(|value| *value == 0.0));
            assert!(
                max_real_error(&restored, &manual) <= 2.0e-8 * length as f64,
                "real C2R medium golden failed at N={length}"
            );
        }

        let length = 257usize;
        let input = sample_real(length);
        let dct2 = TransformIr::build(
            FftConfig::new(vec![length]).with_transform(TransformKind::Dct(DctType::II)),
            Direction::Forward,
            device(),
        )
        .unwrap();
        let TransformIr::RealToReal(dct2_ir) = &dct2 else {
            panic!("DCT-II medium golden did not route to R2R IR");
        };
        let reduction = dct2_ir
            .fft_reduction
            .as_ref()
            .expect("DCT-II medium golden must use FFT reduction");
        assert_eq!(reduction.fft_len, 257);
        assert_eq!(reduction.fft.direction(), Direction::Forward);
        let actual_dct2 = dct2.execute_r2r_reference(&input).unwrap();
        let expected_dct2 = direct_dct_ii(&input);
        assert!(max_real_error(&actual_dct2, &expected_dct2) <= 2.0e-8 * length as f64);

        let dct3 = TransformIr::build(
            FftConfig::new(vec![length]).with_transform(TransformKind::Dct(DctType::III)),
            Direction::Forward,
            device(),
        )
        .unwrap();
        let TransformIr::RealToReal(dct3_ir) = &dct3 else {
            panic!("DCT-III medium golden did not route to R2R IR");
        };
        let dct3_reduction = dct3_ir
            .fft_reduction
            .as_ref()
            .expect("DCT-III medium golden must use FFT reduction");
        assert_eq!(dct3_reduction.fft_len, 257);
        assert_eq!(dct3_reduction.fft.direction(), Direction::Inverse);
        let actual_dct3 = dct3.execute_r2r_reference(&input).unwrap();
        let expected_dct3 = direct_dct_iii(&input);
        assert!(max_real_error(&actual_dct3, &expected_dct3) <= 2.0e-8 * length as f64);

        let inverse_dct2 = TransformIr::build(
            FftConfig::new(vec![length])
                .with_transform(TransformKind::Dct(DctType::II))
                .with_inverse_normalization(true),
            Direction::Inverse,
            device(),
        )
        .unwrap();
        let restored = inverse_dct2.execute_r2r_reference(&actual_dct2).unwrap();
        assert!(max_real_error(&restored, &input) <= 2.0e-8 * length as f64);

        let (rows, cols) = (17usize, 34usize);
        let tensor_len = rows * cols;
        let nd_input = sample_real(tensor_len);
        let nd_forward = TransformIr::build(
            FftConfig::new(vec![rows, cols]).with_transform(TransformKind::RealToComplex),
            Direction::Forward,
            device(),
        )
        .unwrap();
        let actual_nd = nd_forward.execute_r2c_reference(&nd_input).unwrap();
        let expected_nd = direct_nd_r2c(&nd_input, rows, cols);
        assert!(
            max_complex_error(&actual_nd, &expected_nd) <= 3.0e-8 * tensor_len as f64,
            "ND real medium golden failed"
        );
        let nd_inverse = TransformIr::build(
            FftConfig::new(vec![rows, cols])
                .with_transform(TransformKind::ComplexToReal)
                .with_inverse_normalization(true),
            Direction::Inverse,
            device(),
        )
        .unwrap();
        let nd_restored = nd_inverse.execute_c2r_reference(&actual_nd).unwrap();
        assert!(max_real_error(&nd_restored, &nd_input) <= 3.0e-8 * tensor_len as f64);
    }

    #[test]
    fn fixed_upstream_sample_shape_corpus_materializes_and_executes() {
        // Representative dimensions are copied from VkFFT 1.3.4 commit
        // 066a17c17068c0f11c9298d848c2976c71fad1c1 sample 6/7/100/4.
        // Keep the benchmark's shape semantics while using bounded unit-test workloads.

        // sample_6_benchmark_VkFFT_single_r2c.cpp includes 64x64 2D and 32^3 3D R2C.
        let r2c_2d = TransformIr::build(
            FftConfig::new(vec![64, 64]).with_transform(TransformKind::RealToComplex),
            Direction::Forward,
            device(),
        )
        .unwrap();
        let TransformIr::RealNd(r2c_2d_ir) = &r2c_2d else {
            panic!("upstream sample-6 64x64 shape did not route to ND real IR");
        };
        crate::ProgramIr::nd_real_fft(r2c_2d_ir)
            .unwrap()
            .validate()
            .unwrap();
        let r2c_input = sample_real(64 * 64);
        let r2c_spectrum = r2c_2d.execute_r2c_reference(&r2c_input).unwrap();
        let c2r_2d = TransformIr::build(
            FftConfig::new(vec![64, 64])
                .with_transform(TransformKind::ComplexToReal)
                .with_inverse_normalization(true),
            Direction::Inverse,
            device(),
        )
        .unwrap();
        let restored = c2r_2d.execute_c2r_reference(&r2c_spectrum).unwrap();
        assert!(max_real_error(&restored, &r2c_input) <= 3.0e-8 * (64 * 64) as f64);

        let r2c_3d = TransformIr::build(
            FftConfig::new(vec![32, 32, 32]).with_transform(TransformKind::RealToComplex),
            Direction::Forward,
            device(),
        )
        .unwrap();
        let TransformIr::RealNd(r2c_3d_ir) = &r2c_3d else {
            panic!("upstream sample-6 32^3 shape did not route to ND real IR");
        };
        let r2c_3d_program = crate::ProgramIr::nd_real_fft(r2c_3d_ir).unwrap();
        r2c_3d_program.validate().unwrap();
        assert_eq!(r2c_3d_ir.dimensions, vec![32, 32, 32]);

        // sample_7_benchmark_VkFFT_single_Bluestein.cpp includes 179x179. Force the
        // portable planner below the prime so this corpus preserves the sample's
        // intended Bluestein pressure instead of accepting a newer Rader choice.
        let mut bluestein_tuning = crate::PlannerTuning::portable();
        bluestein_tuning.max_rader_fft_prime = 100;
        let prime_square = TransformIr::build(
            FftConfig::new(vec![179, 179]).with_tuning(bluestein_tuning),
            Direction::Forward,
            device(),
        )
        .unwrap();
        let TransformIr::ComplexNd(prime_square_ir) = &prime_square else {
            panic!("upstream sample-7 179x179 shape did not route to ND complex IR");
        };
        assert!(
            prime_square_ir
                .axes
                .iter()
                .all(|axis| matches!(axis.transform, OneDimFftIr::Bluestein(_)))
        );
        crate::ProgramIr::nd_fft(prime_square_ir)
            .unwrap()
            .validate()
            .unwrap();

        // sample_100_benchmark_VkFFT_single_nd_dct.cpp includes 64x64 and 300x300.
        // The small shape executes a DCT-II/III pair; the non-power-of-two shape
        // verifies the fixed-upstream N-point FFT-reduction graph materializes for both axes.
        let dct_input = sample_real(64 * 64);
        let dct_forward = TransformIr::build(
            FftConfig::new(vec![64, 64]).with_transform(TransformKind::Dct(DctType::II)),
            Direction::Forward,
            device(),
        )
        .unwrap();
        let dct_coefficients = dct_forward.execute_r2r_reference(&dct_input).unwrap();
        let dct_inverse = TransformIr::build(
            FftConfig::new(vec![64, 64])
                .with_transform(TransformKind::Dct(DctType::II))
                .with_inverse_normalization(true),
            Direction::Inverse,
            device(),
        )
        .unwrap();
        let dct_restored = dct_inverse
            .execute_r2r_reference(&dct_coefficients)
            .unwrap();
        assert!(max_real_error(&dct_restored, &dct_input) <= 3.0e-8 * (64 * 64) as f64);

        let dct_300 = TransformIr::build(
            FftConfig::new(vec![300, 300]).with_transform(TransformKind::Dct(DctType::II)),
            Direction::Forward,
            device(),
        )
        .unwrap();
        let TransformIr::RealToRealNd(dct_300_ir) = &dct_300 else {
            panic!("upstream sample-100 300x300 shape did not route to ND R2R IR");
        };
        assert!(dct_300_ir.axes.iter().all(|axis| {
            axis.transform
                .fft_reduction
                .as_ref()
                .is_some_and(|reduction| {
                    reduction.fft_len == 300 && reduction.fft.direction() == Direction::Forward
                })
        }));
        crate::ProgramIr::nd_r2r(dct_300_ir)
            .unwrap()
            .validate()
            .unwrap();

        // sample_4_benchmark_VkFFT_single_3d_zeropadding.cpp contains 16^3 and
        // zeros [ceil(N/2), N) on every axis. Inject garbage there and prove the
        // native boundary is equivalent to an explicitly zeroed ordinary transform.
        let dimensions = [16usize, 16, 16];
        let tensor_len = dimensions.iter().product::<usize>();
        let mut padded_input = (0..tensor_len)
            .map(|index| {
                let x = index as f64;
                Complex64::new(
                    (0.053 * x).sin() + 0.001 * x,
                    (0.031 * x).cos() - 0.0004 * x,
                )
            })
            .collect::<Vec<_>>();
        let mut manual_zero = padded_input.clone();
        for z in 0..16 {
            for y in 0..16 {
                for x in 0..16 {
                    if x >= 8 || y >= 8 || z >= 8 {
                        let index = z * 16 * 16 + y * 16 + x;
                        padded_input[index] = Complex64::new(10_000.0 + index as f64, -20_000.0);
                        manual_zero[index] = Complex64::default();
                    }
                }
            }
        }
        let zero_padded = TransformIr::build(
            FftConfig::new(dimensions.to_vec())
                .with_zero_padding(0, 8, 16)
                .unwrap()
                .with_zero_padding(1, 8, 16)
                .unwrap()
                .with_zero_padding(2, 8, 16)
                .unwrap(),
            Direction::Forward,
            device(),
        )
        .unwrap();
        let plain = TransformIr::build(
            FftConfig::new(dimensions.to_vec()),
            Direction::Forward,
            device(),
        )
        .unwrap();
        let padded_output = zero_padded
            .execute_complex_reference(&padded_input)
            .unwrap();
        let manual_output = plain.execute_complex_reference(&manual_zero).unwrap();
        assert!(max_complex_error(&padded_output, &manual_output) <= 3.0e-8 * tensor_len as f64);
    }

    #[test]
    fn double_double_fft_rader_and_bluestein_build_without_aliasing() {
        for precision in [
            crate::Precision::DoubleDouble,
            crate::Precision::DoubleDoubleF64Storage,
        ] {
            let rader = TransformIr::build(
                FftConfig::new(vec![17]).with_precision(precision),
                Direction::Forward,
                device(),
            )
            .unwrap();
            let TransformIr::Complex1dDoubleDouble(crate::DoubleDoubleOneDimIr::FftRader(rader)) =
                rader
            else {
                panic!("p17 DD plan should preserve planner-selected FFT Rader");
            };
            assert_eq!(rader.prime, 17);
            assert_eq!(rader.convolution_len, 16);

            let mut tuning = crate::PlannerTuning::portable();
            tuning.max_rader_fft_prime = 100;
            let bluestein = TransformIr::build(
                FftConfig::new(vec![103])
                    .with_precision(precision)
                    .with_tuning(tuning),
                Direction::Forward,
                device(),
            )
            .unwrap();
            let TransformIr::Complex1dDoubleDouble(crate::DoubleDoubleOneDimIr::Bluestein(
                bluestein,
            )) = bluestein
            else {
                panic!("forced p103 DD plan should preserve planner-selected Bluestein");
            };
            assert_eq!(bluestein.logical_len, 103);
            assert_eq!(bluestein.convolution_len, 256);
        }
    }

    #[test]
    fn f16_storage_f32_compute_stockham_has_typed_external_abi() {
        let ir = TransformIr::build(
            FftConfig::new(vec![64])
                .with_batch_count(3)
                .with_precision(crate::Precision::F16StorageF32Compute),
            Direction::Forward,
            device(),
        )
        .unwrap();
        let TransformIr::Complex1d(one_dim) = &ir else {
            panic!("F16-storage C2C should build a one-dimensional FFT");
        };
        assert_eq!(one_dim.scalar(), crate::ScalarType::F32);
        assert_eq!(one_dim.external_storage_scalar(), crate::ScalarType::F16);

        let program = crate::ProgramIr::one_dim_fft(one_dim).unwrap();
        assert_eq!(program.scalar, crate::ScalarType::F32);
        assert_eq!(
            program.input_resource().unwrap().scalar,
            crate::ScalarType::F16
        );
        assert_eq!(
            program.output_resource().unwrap().scalar,
            crate::ScalarType::F16
        );
        assert!(program.resources.iter().all(|resource| {
            matches!(
                resource.kind,
                crate::ProgramResourceKind::Input | crate::ProgramResourceKind::Output
            ) || resource.scalar == crate::ScalarType::F32
        }));
        let memory = program.memory_plan().unwrap();
        let input_allocation = memory
            .allocation_for(program.input_resource().unwrap().id)
            .unwrap();
        let input_allocation = &memory.allocations[input_allocation.0];
        assert_eq!(input_allocation.scalar, crate::ScalarType::F16);
        assert_eq!(
            input_allocation.elements * input_allocation.scalar.complex_bytes(),
            input_allocation.elements * 4
        );

        let shaders = crate::backend::vulkan::VulkanGlslBackend
            .lower_one_dim_fft(one_dim)
            .unwrap();
        assert_eq!(shaders.len(), 1);
        let shader = &shaders[0];
        assert!(
            shader
                .glsl
                .contains("readonly buffer VkFftInput { uint data[]; }")
        );
        assert!(shader.glsl.contains("buffer VkFftOutput { uint data[]; }"));
        assert!(shader.glsl.contains("unpackHalf2x16(vkfft_input.data["));
        assert!(shader.glsl.contains("packHalf2x16("));
        shader.compile_spirv().unwrap();

        for backend in [
            Backend::Cuda,
            Backend::Hip,
            Backend::OpenCl,
            Backend::LevelZero,
            Backend::Metal,
        ] {
            let native = crate::backend::NativeSourceBackend::new(backend)
                .lower_transform(&ir)
                .unwrap();
            let source = &native.shaders[0].source;
            assert_eq!(
                native.program.input_resource().unwrap().scalar,
                crate::ScalarType::F16
            );
            assert!(
                source.contains("uint* inputs") || source.contains("uint *inputs"),
                "{backend:?}: {source}"
            );
            assert!(
                source.contains("uint* outputs") || source.contains("uint *outputs"),
                "{backend:?}: {source}"
            );
            assert!(source.contains("vkfft_unpack_half2("), "{backend:?}");
            assert!(source.contains("vkfft_pack_half2("), "{backend:?}");
            assert!(!source.contains("unpackHalf2x16("), "{backend:?}");
            assert!(!source.contains("packHalf2x16("), "{backend:?}");
        }
    }

    #[test]
    fn f64_compute_f32_storage_stockham_has_typed_external_abi() {
        let mut profile = device();
        profile.supports_f64 = true;
        let ir = TransformIr::build(
            FftConfig::new(vec![64])
                .with_batch_count(3)
                .with_precision(crate::Precision::F64ComputeF32Storage),
            Direction::Forward,
            profile,
        )
        .unwrap();
        let TransformIr::Complex1d(one_dim) = &ir else {
            panic!("mixed-storage C2C should build a one-dimensional FFT");
        };
        assert_eq!(one_dim.scalar(), crate::ScalarType::F64);
        assert_eq!(one_dim.external_storage_scalar(), crate::ScalarType::F32);

        let program = crate::ProgramIr::one_dim_fft(one_dim).unwrap();
        assert_eq!(program.scalar, crate::ScalarType::F64);
        assert_eq!(
            program.input_resource().unwrap().scalar,
            crate::ScalarType::F32
        );
        assert_eq!(
            program.output_resource().unwrap().scalar,
            crate::ScalarType::F32
        );
        assert!(program.resources.iter().all(|resource| {
            matches!(
                resource.kind,
                crate::ProgramResourceKind::Input | crate::ProgramResourceKind::Output
            ) || resource.scalar == crate::ScalarType::F64
        }));
        let memory = program.memory_plan().unwrap();
        assert_eq!(
            memory.allocations[program
                .memory_plan()
                .unwrap()
                .allocation_for(program.input_resource().unwrap().id)
                .unwrap()
                .0]
                .scalar,
            crate::ScalarType::F32
        );

        let shaders = crate::backend::vulkan::VulkanGlslBackend
            .lower_one_dim_fft(one_dim)
            .unwrap();
        assert_eq!(shaders.len(), 1);
        let shader = &shaders[0];
        assert!(
            shader
                .glsl
                .contains("readonly buffer VkFftInput { vec2 data[]; }")
        );
        assert!(shader.glsl.contains("buffer VkFftOutput { vec2 data[]; }"));
        assert!(shader.glsl.contains("dvec2((vkfft_input.data["));
        assert!(shader.glsl.contains("= vec2("));
        shader.compile_spirv().unwrap();

        let native = crate::backend::NativeSourceBackend::new(Backend::Cuda)
            .lower_transform(&ir)
            .unwrap();
        assert_eq!(
            native.program.input_resource().unwrap().scalar,
            crate::ScalarType::F32
        );
        assert!(native.shaders[0].source.contains("const vec2* inputs"));
        assert!(native.shaders[0].source.contains("vec2* outputs"));
        assert!(!native.shaders[0].source.contains("const dvec2* inputs"));
        assert!(native.shaders[0].source.contains("vkfft_dv2("));
        assert!(native.shaders[0].source.contains("inputs["));
        assert!(native.shaders[0].source.contains("= vkfft_v2("));
    }

    #[test]
    fn high_level_transform_builds_and_executes_double_double_stockham() {
        let length = 8usize;
        let ir = TransformIr::build(
            FftConfig::new(vec![length]).with_precision(crate::Precision::DoubleDouble),
            Direction::Forward,
            device(),
        )
        .unwrap();
        let TransformIr::Complex1dDoubleDouble(crate::DoubleDoubleOneDimIr::Stockham(stockham)) =
            &ir
        else {
            panic!("double-double C2C should build dedicated Stockham IR");
        };
        assert_eq!(
            stockham.external_storage,
            crate::PrecisionStorage::DoubleDouble
        );
        let mut input = vec![crate::ComplexDoubleDouble::default(); length];
        input[0] = crate::ComplexDoubleDouble::new(
            crate::DoubleDouble::from_parts(1.0e16, 1.0),
            crate::DoubleDouble::ZERO,
        );
        for (index, value) in input.iter_mut().enumerate().skip(1) {
            value.re = crate::DoubleDouble::from_f64(index as f64 * 0.25);
            value.im = crate::DoubleDouble::from_f64(-(index as f64) * 0.125);
        }
        let actual = ir.execute_double_double_reference(&input).unwrap();
        let expected = crate::double_double_dft(&input, Direction::Forward, false).unwrap();
        for (actual, expected) in actual.into_iter().zip(expected) {
            let re = (actual.re - expected.re).abs();
            let im = (actual.im - expected.im).abs();
            let error = re.hi.abs() + re.lo.abs() + im.hi.abs() + im.lo.abs();
            assert!(error < 2.0e-28, "high-level DD error {error:e}");
        }
        for backend in [
            Backend::Cuda,
            Backend::Hip,
            Backend::OpenCl,
            Backend::LevelZero,
        ] {
            let native = crate::backend::NativeSourceBackend::new(backend)
                .lower_transform(&ir)
                .unwrap();
            assert_eq!(native.program.scalar, crate::ScalarType::DoubleDouble);
            assert_eq!(native.shaders.len(), 1);
        }
        let error = crate::backend::NativeSourceBackend::new(Backend::Metal)
            .lower_transform(&ir)
            .unwrap_err();
        assert!(matches!(error, VkFftError::UnsupportedPrecision { .. }));
    }

    #[test]
    fn high_level_double_double_f64_storage_uses_complex64_boundary() {
        let length = 12usize;
        let ir = TransformIr::build(
            FftConfig::new(vec![length]).with_precision(crate::Precision::DoubleDoubleF64Storage),
            Direction::Forward,
            device(),
        )
        .unwrap();
        let TransformIr::Complex1dDoubleDouble(crate::DoubleDoubleOneDimIr::Stockham(stockham)) =
            &ir
        else {
            panic!("DD/F64-storage C2C should build dedicated Stockham IR");
        };
        assert_eq!(stockham.external_storage, crate::PrecisionStorage::F64);
        let input = (0..length)
            .map(|index| {
                let x = index as f64;
                Complex64::new((0.19 * x).sin() + x * 0.001, (0.13 * x).cos())
            })
            .collect::<Vec<_>>();
        let actual = ir.execute_complex_reference(&input).unwrap();
        let expected = crate::double_double_dft(
            &input
                .iter()
                .copied()
                .map(crate::ComplexDoubleDouble::from_complex64)
                .collect::<Vec<_>>(),
            Direction::Forward,
            false,
        )
        .unwrap()
        .into_iter()
        .map(crate::ComplexDoubleDouble::to_complex64)
        .collect::<Vec<_>>();
        assert!(max_complex_error(&actual, &expected) < 2.0e-14 * length as f64);
    }

    #[test]
    fn high_level_planner_rejects_wrong_real_direction() {
        let error = TransformIr::build(
            FftConfig::new(vec![8]).with_transform(TransformKind::RealToComplex),
            Direction::Inverse,
            device(),
        )
        .unwrap_err();
        assert!(matches!(error, VkFftError::UnsupportedKernelPath(_)));
    }
}
