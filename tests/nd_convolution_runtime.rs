#![cfg_attr(
    not(any(
        feature = "cuda-runtime",
        feature = "opencl-runtime",
        feature = "vulkan-runtime"
    )),
    allow(dead_code, unused_imports)
)]

use vkfft_rs::backend::vulkan::VulkanGlslBackend;
use vkfft_rs::{
    Backend, Complex32, Complex64, ConvolutionConjugation, ConvolutionMatrixLayout, DeviceProfile,
    FftConfig, GpuVendor, NdConvolutionIr, Precision, ProgramIr, execute_nd_convolution_ir,
};

const DIMENSIONS: [usize; 2] = [3, 4];
const BATCH_COUNT: usize = 2;

fn tensor_len() -> usize {
    DIMENSIONS.iter().product()
}

fn device(backend: Backend, vendor: GpuVendor) -> DeviceProfile {
    let mut profile = DeviceProfile::generic(backend, vendor);
    profile.supports_f64 = true;
    profile
}

fn kernel() -> Vec<Complex64> {
    (0..tensor_len())
        .map(|index| {
            let x = index as f64;
            Complex64::new((0.29 * x).cos() + 0.013 * x, (0.17 * x).sin() - 0.009 * x)
        })
        .collect()
}

fn direct_dft_2d(input: &[Complex64]) -> Vec<Complex64> {
    let [rows, cols] = DIMENSIONS;
    let mut output = vec![Complex64::default(); rows * cols];
    for k0 in 0..rows {
        for k1 in 0..cols {
            let mut sum = Complex64::default();
            for n0 in 0..rows {
                for n1 in 0..cols {
                    let phase = -std::f64::consts::TAU
                        * (k0 as f64 * n0 as f64 / rows as f64
                            + k1 as f64 * n1 as f64 / cols as f64);
                    sum += input[n0 * cols + n1] * Complex64::new(phase.cos(), phase.sin());
                }
            }
            output[k0 * cols + k1] = sum;
        }
    }
    output
}

fn direct_idft_2d(input: &[Complex64]) -> Vec<Complex64> {
    let [rows, cols] = DIMENSIONS;
    let scale = 1.0 / (rows * cols) as f64;
    let mut output = vec![Complex64::default(); rows * cols];
    for n0 in 0..rows {
        for n1 in 0..cols {
            let mut sum = Complex64::default();
            for k0 in 0..rows {
                for k1 in 0..cols {
                    let phase = std::f64::consts::TAU
                        * (k0 as f64 * n0 as f64 / rows as f64
                            + k1 as f64 * n1 as f64 / cols as f64);
                    sum += input[k0 * cols + k1] * Complex64::new(phase.cos(), phase.sin());
                }
            }
            output[n0 * cols + n1] = sum.scale(scale);
        }
    }
    output
}

fn kernel_spectrum() -> Vec<Complex64> {
    direct_dft_2d(&kernel())
}

fn input64() -> Vec<Complex64> {
    let n = tensor_len();
    (0..n * BATCH_COUNT)
        .map(|index| {
            let batch = index / n;
            let local = (index % n) as f64;
            Complex64::new(
                0.31 * batch as f64 + (0.11 * local).sin() + 0.004 * local,
                -0.23 * batch as f64 + (0.07 * local).cos() - 0.003 * local,
            )
        })
        .collect()
}

fn direct_circular_convolution_2d(input: &[Complex64], kernel: &[Complex64]) -> Vec<Complex64> {
    let [rows, cols] = DIMENSIONS;
    let mut output = vec![Complex64::default(); rows * cols];
    for out0 in 0..rows {
        for out1 in 0..cols {
            let mut sum = Complex64::default();
            for n0 in 0..rows {
                for n1 in 0..cols {
                    let k0 = (out0 + rows - n0) % rows;
                    let k1 = (out1 + cols - n1) % cols;
                    sum += input[n0 * cols + n1] * kernel[k0 * cols + k1];
                }
            }
            output[out0 * cols + out1] = sum;
        }
    }
    output
}

fn expected64() -> Vec<Complex64> {
    let input = input64();
    let kernel = kernel();
    let n = tensor_len();
    let mut expected = Vec::with_capacity(n * BATCH_COUNT);
    for batch in 0..BATCH_COUNT {
        let base = batch * n;
        expected.extend(direct_circular_convolution_2d(
            &input[base..base + n],
            &kernel,
        ));
    }
    expected
}

fn expected_policy64(sequence_conjugation: bool, cross_power: bool) -> Vec<Complex64> {
    let input = input64();
    let kernel_spectrum = kernel_spectrum();
    let n = tensor_len();
    let mut expected = Vec::with_capacity(n * BATCH_COUNT);
    for batch in 0..BATCH_COUNT {
        let base = batch * n;
        let mut spectrum = direct_dft_2d(&input[base..base + n]);
        for (sequence, kernel) in spectrum.iter_mut().zip(&kernel_spectrum) {
            let prepared = if sequence_conjugation {
                sequence.conj()
            } else {
                *sequence
            };
            let product = prepared * *kernel;
            *sequence = if cross_power {
                product.scale(1.0 / product.norm_sqr().sqrt())
            } else {
                product
            };
        }
        expected.extend(direct_idft_2d(&spectrum));
    }
    expected
}

fn build_with_policy(
    profile: DeviceProfile,
    precision: Precision,
    sequence_conjugation: bool,
    cross_power: bool,
) -> NdConvolutionIr {
    let mut config = FftConfig::new(DIMENSIONS.to_vec())
        .with_batch_count(BATCH_COUNT)
        .with_precision(precision)
        .with_convolution(true);
    if sequence_conjugation {
        config = config.with_convolution_conjugation(ConvolutionConjugation::Sequence);
    }
    if cross_power {
        config = config.with_cross_power_spectrum_normalization(true);
    }
    NdConvolutionIr::build_from_spectrum(config, kernel_spectrum(), profile).unwrap()
}

fn build(profile: DeviceProfile, precision: Precision) -> NdConvolutionIr {
    build_with_policy(profile, precision, false, false)
}

fn build_sequence_cross_power(profile: DeviceProfile, precision: Precision) -> NdConvolutionIr {
    build_with_policy(profile, precision, true, true)
}

const FANOUT_KERNELS: usize = 3;

fn fanout_input64() -> Vec<Complex64> {
    input64()[..tensor_len()].to_vec()
}

fn fanout_kernel_spectrum() -> Vec<Complex64> {
    (0..FANOUT_KERNELS)
        .flat_map(|kernel_id| {
            let spatial = (0..tensor_len())
                .map(|index| {
                    let x = index as f64;
                    Complex64::new(
                        0.18 + 0.037 * kernel_id as f64 + (0.08 * x).cos(),
                        -0.09 + 0.021 * kernel_id as f64 + (0.12 * x).sin(),
                    )
                })
                .collect::<Vec<_>>();
            direct_dft_2d(&spatial)
        })
        .collect()
}

fn expected_fanout_policy64(sequence_conjugation: bool, cross_power: bool) -> Vec<Complex64> {
    let input_spectrum = direct_dft_2d(&fanout_input64());
    let kernels = fanout_kernel_spectrum();
    let mut output = Vec::with_capacity(FANOUT_KERNELS * tensor_len());
    for kernel_id in 0..FANOUT_KERNELS {
        let kernel_base = kernel_id * tensor_len();
        let mut spectrum = Vec::with_capacity(tensor_len());
        for index in 0..tensor_len() {
            let sequence = if sequence_conjugation {
                input_spectrum[index].conj()
            } else {
                input_spectrum[index]
            };
            let product = sequence * kernels[kernel_base + index];
            spectrum.push(if cross_power {
                product.scale(1.0 / product.norm_sqr().sqrt())
            } else {
                product
            });
        }
        output.extend(direct_idft_2d(&spectrum));
    }
    output
}

fn build_fanout(
    profile: DeviceProfile,
    precision: Precision,
    semantic_policy: bool,
) -> NdConvolutionIr {
    let mut config = FftConfig::new(DIMENSIONS.to_vec())
        .with_precision(precision)
        .with_convolution(true)
        .with_convolution_kernel_count(FANOUT_KERNELS);
    if semantic_policy {
        config = config
            .with_convolution_conjugation(ConvolutionConjugation::Sequence)
            .with_cross_power_spectrum_normalization(true);
    }
    NdConvolutionIr::build_from_spectrum(config, fanout_kernel_spectrum(), profile).unwrap()
}

fn matrix_input64(matrix_size: usize) -> Vec<Complex64> {
    let n = tensor_len();
    (0..matrix_size * n)
        .map(|index| {
            let coordinate = index / n;
            let local = (index % n) as f64;
            Complex64::new(
                0.17 * coordinate as f64 + (0.09 * local).sin() + 0.006 * local,
                -0.11 * coordinate as f64 + (0.13 * local).cos() - 0.004 * local,
            )
        })
        .collect()
}

fn matrix_kernel_spectrum(matrix_size: usize, symmetric: bool) -> Vec<Complex64> {
    let layout = ConvolutionMatrixLayout {
        matrix_size,
        symmetric_kernel: symmetric,
    };
    (0..layout.kernel_plane_count())
        .flat_map(|plane| {
            let spatial = (0..tensor_len())
                .map(|index| {
                    let x = index as f64;
                    Complex64::new(
                        0.23 + 0.041 * plane as f64 + (0.07 * x).cos(),
                        -0.14 + 0.019 * plane as f64 + (0.05 * x).sin(),
                    )
                })
                .collect::<Vec<_>>();
            direct_dft_2d(&spatial)
        })
        .collect()
}

fn expected_matrix_policy64(
    matrix_size: usize,
    symmetric: bool,
    sequence_conjugation: bool,
    cross_power: bool,
) -> Vec<Complex64> {
    let layout = ConvolutionMatrixLayout {
        matrix_size,
        symmetric_kernel: symmetric,
    };
    let input = matrix_input64(matrix_size);
    let kernel_spectrum = matrix_kernel_spectrum(matrix_size, symmetric);
    let n = tensor_len();
    let spectra = (0..matrix_size)
        .map(|coordinate| direct_dft_2d(&input[coordinate * n..(coordinate + 1) * n]))
        .collect::<Vec<_>>();
    let mut expected = vec![Complex64::default(); matrix_size * n];
    for output_coordinate in 0..matrix_size {
        let mut output_spectrum = vec![Complex64::default(); n];
        for index in 0..n {
            let mut sum = Complex64::default();
            for (input_coordinate, spectrum) in spectra.iter().enumerate() {
                let sequence = if sequence_conjugation {
                    spectrum[index].conj()
                } else {
                    spectrum[index]
                };
                let plane = layout
                    .kernel_plane_index(output_coordinate, input_coordinate)
                    .unwrap();
                sum += sequence * kernel_spectrum[plane * n + index];
            }
            output_spectrum[index] = if cross_power {
                sum.scale(1.0 / sum.norm_sqr().sqrt())
            } else {
                sum
            };
        }
        expected[output_coordinate * n..(output_coordinate + 1) * n]
            .copy_from_slice(&direct_idft_2d(&output_spectrum));
    }
    expected
}

fn build_matrix(
    profile: DeviceProfile,
    precision: Precision,
    matrix_size: usize,
    symmetric: bool,
    semantic_policy: bool,
) -> NdConvolutionIr {
    let mut config = FftConfig::new(DIMENSIONS.to_vec())
        .with_precision(precision)
        .with_convolution(true)
        .with_matrix_convolution(matrix_size)
        .with_symmetric_convolution_kernel(symmetric);
    if semantic_policy {
        config = config
            .with_convolution_conjugation(ConvolutionConjugation::Sequence)
            .with_cross_power_spectrum_normalization(true);
    }
    NdConvolutionIr::build_from_spectrum(
        config,
        matrix_kernel_spectrum(matrix_size, symmetric),
        profile,
    )
    .unwrap()
}

fn assert_close64(actual: &[Complex64], expected: &[Complex64], tolerance: f64, label: &str) {
    assert_eq!(actual.len(), expected.len());
    let error = actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (*actual - *expected).norm_sqr().sqrt())
        .fold(0.0, f64::max);
    assert!(error <= tolerance, "{label} ND convolution error {error:e}");
}

fn assert_close32(actual: &[Complex32], expected: &[Complex64], tolerance: f32, label: &str) {
    assert_eq!(actual.len(), expected.len());
    let error = actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| {
            let dr = actual.re - expected.re as f32;
            let di = actual.im - expected.im as f32;
            (dr * dr + di * di).sqrt()
        })
        .fold(0.0f32, f32::max);
    assert!(error <= tolerance, "{label} ND convolution error {error:e}");
}

#[test]
fn nd_convolution_program_and_spirv_keep_one_tensor_multiply_between_nd_children() {
    let profile = device(Backend::Vulkan, GpuVendor::Nvidia);
    for precision in [Precision::F32, Precision::F64] {
        let ir = build(profile, precision);
        let program = ProgramIr::nd_convolution(&ir).unwrap();
        let shaders = VulkanGlslBackend.lower_nd_convolution(&ir).unwrap();
        assert_eq!(program.passes.len(), shaders.len());
        let multiply_indices = program
            .passes
            .iter()
            .enumerate()
            .filter_map(|(index, pass)| (pass.name == ir.multiply.name).then_some(index))
            .collect::<Vec<_>>();
        assert_eq!(multiply_indices.len(), 1);
        let multiply_index = multiply_indices[0];
        assert!(multiply_index > 0 && multiply_index + 1 < program.passes.len());
        assert!(
            program.passes[..multiply_index]
                .iter()
                .all(|pass| pass.name.starts_with("vkfft_nd_convolution_forward_"))
        );
        assert!(
            program.passes[multiply_index + 1..]
                .iter()
                .all(|pass| pass.name.starts_with("vkfft_nd_convolution_inverse_"))
        );
        assert_eq!(ir.multiply.sequence_len, tensor_len());
        assert_eq!(ir.multiply.dispatch.x, BATCH_COUNT as u32);
        assert!(
            shaders[multiply_index]
                .glsl
                .contains("typed ConvolutionMultiplyIr")
        );
        let kernel = program
            .resources
            .iter()
            .find(|resource| resource.name == "nd_convolution_kernel_spectrum")
            .unwrap();
        assert_eq!(kernel.elements, tensor_len());
        for shader in shaders {
            assert_eq!(shader.compile_spirv().unwrap().words[0], 0x0723_0203);
        }
    }
}

#[test]
fn nd_convolution_policy_matrix_matches_independent_2d_dft_and_spirv() {
    let profile = device(Backend::Vulkan, GpuVendor::Nvidia);
    let input = input64();
    for (sequence_conjugation, cross_power) in [(true, false), (false, true), (true, true)] {
        let expected = expected_policy64(sequence_conjugation, cross_power);
        for precision in [Precision::F32, Precision::F64] {
            let ir = build_with_policy(profile, precision, sequence_conjugation, cross_power);
            assert_eq!(
                ir.multiply.policy.conjugation,
                if sequence_conjugation {
                    ConvolutionConjugation::Sequence
                } else {
                    ConvolutionConjugation::None
                }
            );
            assert_eq!(
                ir.multiply.policy.cross_power_spectrum_normalization,
                cross_power
            );
            let actual = execute_nd_convolution_ir(&ir, &input).unwrap();
            assert_close64(&actual, &expected, 2.0e-10, "CPU ND policy");

            let program = ProgramIr::nd_convolution(&ir).unwrap();
            let shaders = VulkanGlslBackend.lower_nd_convolution(&ir).unwrap();
            assert_eq!(program.passes.len(), shaders.len());
            let multiply_index = program
                .passes
                .iter()
                .position(|pass| pass.name == ir.multiply.name)
                .unwrap();
            let multiply = &shaders[multiply_index].glsl;
            assert!(multiply.contains("vkfft_convolution_product"));
            assert_eq!(multiply.contains(", -("), sequence_conjugation);
            assert_eq!(multiply.contains("vkfft_convolution_norm"), cross_power);
            assert_eq!(
                multiply.contains("inversesqrt(vkfft_convolution_norm)"),
                cross_power
            );
            for shader in shaders {
                assert_eq!(shader.compile_spirv().unwrap().words[0], 0x0723_0203);
            }
        }
    }
}

#[test]
fn nd_matrix_convolution_expands_coordinates_and_matches_independent_2d_dft() {
    let profile = device(Backend::Vulkan, GpuVendor::Nvidia);
    for (matrix_size, symmetric) in [(2usize, false), (2, true), (3, false)] {
        let input = matrix_input64(matrix_size);
        let expected = expected_matrix_policy64(matrix_size, symmetric, false, false);
        let layout = ConvolutionMatrixLayout {
            matrix_size,
            symmetric_kernel: symmetric,
        };
        for precision in [Precision::F32, Precision::F64] {
            let ir = build_matrix(profile, precision, matrix_size, symmetric, false);
            assert_eq!(ir.batch_count, 1);
            assert_eq!(ir.coordinate_count, matrix_size);
            assert_eq!(ir.matrix_layout, Some(layout));
            assert_eq!(ir.forward_fft.batch_count, matrix_size);
            assert_eq!(ir.inverse_fft.batch_count, matrix_size);
            assert_eq!(ir.multiply.batch_count, 1);
            assert_eq!(ir.multiply.coordinate_count, matrix_size);
            assert_eq!(ir.multiply.matrix_layout, Some(layout));
            assert_eq!(ir.multiply.dispatch.x, 1);
            assert_eq!(
                ir.kernel_spectrum().len(),
                layout.kernel_plane_count() * tensor_len()
            );

            let actual = execute_nd_convolution_ir(&ir, &input).unwrap();
            assert_close64(&actual, &expected, 3.0e-10, "CPU ND matrix");

            let program = ProgramIr::nd_convolution(&ir).unwrap();
            let input_resource = program
                .resources
                .iter()
                .find(|resource| resource.name == "input")
                .unwrap();
            let output_resource = program
                .resources
                .iter()
                .find(|resource| resource.name == "output")
                .unwrap();
            let kernel_resource = program
                .resources
                .iter()
                .find(|resource| resource.name == "nd_convolution_kernel_spectrum")
                .unwrap();
            assert_eq!(input_resource.elements, matrix_size * tensor_len());
            assert_eq!(output_resource.elements, matrix_size * tensor_len());
            assert_eq!(
                kernel_resource.elements,
                layout.kernel_plane_count() * tensor_len()
            );
            assert_eq!(
                input_resource.external_layout.unwrap().batch_count,
                matrix_size
            );
            assert_eq!(
                output_resource.external_layout.unwrap().batch_count,
                matrix_size
            );

            let shaders = VulkanGlslBackend.lower_nd_convolution(&ir).unwrap();
            assert_eq!(program.passes.len(), shaders.len());
            let multiply_index = program
                .passes
                .iter()
                .position(|pass| pass.name == ir.multiply.name)
                .unwrap();
            assert!(shaders[multiply_index].glsl.contains("vkfft_matrix_sum_0"));
            assert!(
                shaders[multiply_index]
                    .glsl
                    .contains("vkfft_matrix_input_0")
            );
            for shader in shaders {
                assert_eq!(shader.compile_spirv().unwrap().words[0], 0x0723_0203);
            }
        }
    }

    let input = matrix_input64(2);
    let expected = expected_matrix_policy64(2, false, true, true);
    for precision in [Precision::F32, Precision::F64] {
        let ir = build_matrix(profile, precision, 2, false, true);
        let actual = execute_nd_convolution_ir(&ir, &input).unwrap();
        assert_close64(&actual, &expected, 3.0e-10, "CPU ND matrix policy");
        let program = ProgramIr::nd_convolution(&ir).unwrap();
        let shaders = VulkanGlslBackend.lower_nd_convolution(&ir).unwrap();
        let multiply_index = program
            .passes
            .iter()
            .position(|pass| pass.name == ir.multiply.name)
            .unwrap();
        let multiply = &shaders[multiply_index].glsl;
        assert!(multiply.contains("vkfft_matrix_norm_0"));
        assert!(multiply.contains(", -("));
        for shader in shaders {
            assert_eq!(shader.compile_spirv().unwrap().words[0], 0x0723_0203);
        }
    }
}

#[test]
fn nd_scalar_multi_kernel_fanout_matches_independent_2d_dft_and_program_ownership() {
    let profile = device(Backend::Vulkan, GpuVendor::Nvidia);
    let input = fanout_input64();
    for semantic_policy in [false, true] {
        let expected = expected_fanout_policy64(semantic_policy, semantic_policy);
        for precision in [Precision::F32, Precision::F64] {
            let ir = build_fanout(profile, precision, semantic_policy);
            assert_eq!(ir.batch_count, 1);
            assert_eq!(ir.coordinate_count, 1);
            assert_eq!(ir.kernel_count, FANOUT_KERNELS);
            assert_eq!(ir.output_batch_count(), FANOUT_KERNELS);
            assert_eq!(ir.forward_fft.batch_count, 1);
            assert_eq!(ir.inverse_fft.batch_count, FANOUT_KERNELS);
            assert_eq!(ir.multiply.batch_count, 1);
            assert_eq!(ir.multiply.kernel_count, FANOUT_KERNELS);
            assert_eq!(ir.multiply.dispatch.x, FANOUT_KERNELS as u32);
            assert_eq!(ir.kernel_spectrum().len(), FANOUT_KERNELS * tensor_len());

            let actual = execute_nd_convolution_ir(&ir, &input).unwrap();
            assert_close64(&actual, &expected, 3.0e-10, "CPU ND fanout");

            let program = ProgramIr::nd_convolution(&ir).unwrap();
            let resource = |name: &str| {
                program
                    .resources
                    .iter()
                    .find(|resource| resource.name == name)
                    .unwrap()
            };
            assert_eq!(resource("input").elements, tensor_len());
            assert_eq!(resource("output").elements, FANOUT_KERNELS * tensor_len());
            assert_eq!(
                resource("nd_convolution_forward_spectrum").elements,
                tensor_len()
            );
            assert_eq!(
                resource("nd_convolution_multiplied_spectrum").elements,
                FANOUT_KERNELS * tensor_len()
            );
            assert_eq!(resource("input").external_layout.unwrap().batch_count, 1);
            assert_eq!(
                resource("output").external_layout.unwrap().batch_count,
                FANOUT_KERNELS
            );

            let shaders = VulkanGlslBackend.lower_nd_convolution(&ir).unwrap();
            assert_eq!(program.passes.len(), shaders.len());
            let multiply_index = program
                .passes
                .iter()
                .position(|pass| pass.name == ir.multiply.name)
                .unwrap();
            let multiply = &shaders[multiply_index].glsl;
            assert!(multiply.contains("typed ConvolutionMultiplyIr"));
            assert_eq!(multiply.contains("vkfft_convolution_norm"), semantic_policy);
            for shader in shaders {
                assert_eq!(shader.compile_spirv().unwrap().words[0], 0x0723_0203);
            }
        }
    }
}

#[cfg(any(feature = "cuda-runtime", feature = "opencl-runtime"))]
fn run_native<R: vkfft_rs::backend::NativeRuntime>(runtime: &R) {
    let expected = expected64();
    let policy_expected = expected_policy64(true, true);
    let input64 = input64();
    let input32 = input64
        .iter()
        .map(|value| Complex32::new(value.re as f32, value.im as f32))
        .collect::<Vec<_>>();
    let f32_ir = build(runtime.device_profile(), Precision::F32);
    let actual32 = runtime
        .execute_nd_convolution_f32(&f32_ir, &input32)
        .unwrap();
    assert_close32(&actual32, &expected, 8.0e-4, runtime.device_name());
    let f32_policy_ir = build_sequence_cross_power(runtime.device_profile(), Precision::F32);
    let actual32_policy = runtime
        .execute_nd_convolution_f32(&f32_policy_ir, &input32)
        .unwrap();
    assert_close32(
        &actual32_policy,
        &policy_expected,
        1.2e-3,
        runtime.device_name(),
    );
    let matrix_input64 = matrix_input64(2);
    let matrix_expected = expected_matrix_policy64(2, false, false, false);
    let matrix_input32 = matrix_input64
        .iter()
        .map(|value| Complex32::new(value.re as f32, value.im as f32))
        .collect::<Vec<_>>();
    let matrix_f32_ir = build_matrix(runtime.device_profile(), Precision::F32, 2, false, false);
    let matrix_actual32 = runtime
        .execute_nd_convolution_f32(&matrix_f32_ir, &matrix_input32)
        .unwrap();
    assert_close32(
        &matrix_actual32,
        &matrix_expected,
        1.2e-3,
        runtime.device_name(),
    );
    let fanout_input64 = fanout_input64();
    let fanout_expected = expected_fanout_policy64(false, false);
    let fanout_input32 = fanout_input64
        .iter()
        .map(|value| Complex32::new(value.re as f32, value.im as f32))
        .collect::<Vec<_>>();
    let fanout_f32_ir = build_fanout(runtime.device_profile(), Precision::F32, false);
    let fanout_actual32 = runtime
        .execute_nd_convolution_f32(&fanout_f32_ir, &fanout_input32)
        .unwrap();
    assert_close32(
        &fanout_actual32,
        &fanout_expected,
        1.2e-3,
        runtime.device_name(),
    );
    if runtime.device_profile().supports_f64 {
        let f64_ir = build(runtime.device_profile(), Precision::F64);
        let actual64 = runtime
            .execute_nd_convolution_f64(&f64_ir, &input64)
            .unwrap();
        assert_close64(&actual64, &expected, 8.0e-10, runtime.device_name());
        let f64_policy_ir = build_sequence_cross_power(runtime.device_profile(), Precision::F64);
        let actual64_policy = runtime
            .execute_nd_convolution_f64(&f64_policy_ir, &input64)
            .unwrap();
        assert_close64(
            &actual64_policy,
            &policy_expected,
            8.0e-10,
            runtime.device_name(),
        );
        let matrix_f64_ir = build_matrix(runtime.device_profile(), Precision::F64, 2, false, false);
        let matrix_actual64 = runtime
            .execute_nd_convolution_f64(&matrix_f64_ir, &matrix_input64)
            .unwrap();
        assert_close64(
            &matrix_actual64,
            &matrix_expected,
            8.0e-10,
            runtime.device_name(),
        );
        let fanout_f64_ir = build_fanout(runtime.device_profile(), Precision::F64, false);
        let fanout_actual64 = runtime
            .execute_nd_convolution_f64(&fanout_f64_ir, &fanout_input64)
            .unwrap();
        assert_close64(
            &fanout_actual64,
            &fanout_expected,
            8.0e-10,
            runtime.device_name(),
        );
    }
}

fn gpu_test_lock() -> &'static std::sync::Mutex<()> {
    static LOCK: std::sync::OnceLock<std::sync::Mutex<()>> = std::sync::OnceLock::new();
    LOCK.get_or_init(|| std::sync::Mutex::new(()))
}

#[cfg(feature = "cuda-runtime")]
#[test]
fn cuda_nd_convolution_or_skips() {
    use vkfft_rs::backend::cuda::runtime::CudaExecutionContext;
    let _guard = gpu_test_lock()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let availability = CudaExecutionContext::probe();
    if !availability.compiler_available || availability.device_count == 0 {
        return;
    }
    let runtime = CudaExecutionContext::new(0).expect("CUDA context failed after successful probe");
    run_native(&runtime);
}

#[cfg(feature = "opencl-runtime")]
#[test]
fn opencl_nd_convolution_or_skips() {
    use vkfft_rs::backend::opencl::runtime::OpenClExecutionContext;
    let _guard = gpu_test_lock()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let availability = OpenClExecutionContext::probe();
    if !availability.compiler_available || availability.device_count == 0 {
        return;
    }
    let runtime =
        OpenClExecutionContext::new(0).expect("OpenCL context failed after successful probe");
    run_native(&runtime);
}

#[cfg(feature = "vulkan-runtime")]
#[test]
fn vulkan_nd_convolution_or_skips() {
    use vkfft_rs::{VkFftError, backend::vulkan::runtime::VulkanExecutionContext};
    let _guard = gpu_test_lock()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let runtime = match VulkanExecutionContext::new() {
        Ok(runtime) => runtime,
        Err(VkFftError::VulkanUnavailable(_)) => return,
        Err(error) => panic!("Vulkan context failed after loader/device discovery: {error}"),
    };
    let expected = expected64();
    let policy_expected = expected_policy64(true, true);
    let input64 = input64();
    let input32 = input64
        .iter()
        .map(|value| Complex32::new(value.re as f32, value.im as f32))
        .collect::<Vec<_>>();
    let f32_ir = build(runtime.device_profile(), Precision::F32);
    let actual32 = runtime
        .execute_nd_convolution_f32(&f32_ir, &input32)
        .unwrap();
    assert_close32(&actual32, &expected, 8.0e-4, "Vulkan F32");
    let f32_policy_ir = build_sequence_cross_power(runtime.device_profile(), Precision::F32);
    let actual32_policy = runtime
        .execute_nd_convolution_f32(&f32_policy_ir, &input32)
        .unwrap();
    assert_close32(
        &actual32_policy,
        &policy_expected,
        1.2e-3,
        "Vulkan F32 sequence cross-power",
    );
    let matrix_input64 = matrix_input64(2);
    let matrix_expected = expected_matrix_policy64(2, false, false, false);
    let matrix_input32 = matrix_input64
        .iter()
        .map(|value| Complex32::new(value.re as f32, value.im as f32))
        .collect::<Vec<_>>();
    let matrix_f32_ir = build_matrix(runtime.device_profile(), Precision::F32, 2, false, false);
    let matrix_actual32 = runtime
        .execute_nd_convolution_f32(&matrix_f32_ir, &matrix_input32)
        .unwrap();
    assert_close32(
        &matrix_actual32,
        &matrix_expected,
        1.2e-3,
        "Vulkan F32 matrix",
    );
    let fanout_input64 = fanout_input64();
    let fanout_expected = expected_fanout_policy64(false, false);
    let fanout_input32 = fanout_input64
        .iter()
        .map(|value| Complex32::new(value.re as f32, value.im as f32))
        .collect::<Vec<_>>();
    let fanout_f32_ir = build_fanout(runtime.device_profile(), Precision::F32, false);
    let fanout_actual32 = runtime
        .execute_nd_convolution_f32(&fanout_f32_ir, &fanout_input32)
        .unwrap();
    assert_close32(
        &fanout_actual32,
        &fanout_expected,
        1.2e-3,
        "Vulkan F32 fanout",
    );
    if runtime.device_profile().supports_f64 {
        let f64_ir = build(runtime.device_profile(), Precision::F64);
        let actual64 = runtime
            .execute_nd_convolution_f64(&f64_ir, &input64)
            .unwrap();
        assert_close64(&actual64, &expected, 8.0e-10, "Vulkan F64");
        let f64_policy_ir = build_sequence_cross_power(runtime.device_profile(), Precision::F64);
        let actual64_policy = runtime
            .execute_nd_convolution_f64(&f64_policy_ir, &input64)
            .unwrap();
        assert_close64(
            &actual64_policy,
            &policy_expected,
            8.0e-10,
            "Vulkan F64 sequence cross-power",
        );
        let matrix_f64_ir = build_matrix(runtime.device_profile(), Precision::F64, 2, false, false);
        let matrix_actual64 = runtime
            .execute_nd_convolution_f64(&matrix_f64_ir, &matrix_input64)
            .unwrap();
        assert_close64(
            &matrix_actual64,
            &matrix_expected,
            8.0e-10,
            "Vulkan F64 matrix",
        );
        let fanout_f64_ir = build_fanout(runtime.device_profile(), Precision::F64, false);
        let fanout_actual64 = runtime
            .execute_nd_convolution_f64(&fanout_f64_ir, &fanout_input64)
            .unwrap();
        assert_close64(
            &fanout_actual64,
            &fanout_expected,
            8.0e-10,
            "Vulkan F64 fanout",
        );
    }
}
