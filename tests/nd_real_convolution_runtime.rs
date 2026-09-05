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
    Backend, Complex64, ConvolutionConjugation, DeviceProfile, FftConfig, GpuVendor,
    NdRealConvolutionIr, Precision, ProgramIr, ScalarType, TransformKind,
    execute_nd_real_convolution_ir,
};

const DIMENSIONS: [usize; 2] = [3, 4];
const KERNEL_COUNT: usize = 3;
const COORDINATE_COUNT: usize = 2;
const COORDINATE_KERNEL_COUNT: usize = 2;
const MATRIX_SIZE: usize = 3;
const MATRIX_KERNEL_COUNT: usize = 2;

fn tensor_len() -> usize {
    DIMENSIONS.iter().product()
}

fn compact_len() -> usize {
    DIMENSIONS[0] * (DIMENSIONS[1] / 2 + 1)
}

fn device(backend: Backend, vendor: GpuVendor) -> DeviceProfile {
    let mut profile = DeviceProfile::generic(backend, vendor);
    profile.supports_f64 = true;
    profile
}

fn input64() -> Vec<f64> {
    (0..tensor_len())
        .map(|index| {
            let x = index as f64;
            0.31 + (0.17 * x).sin() - 0.013 * x + 0.07 * (0.11 * x).cos()
        })
        .collect()
}

fn spatial_kernel(kernel_id: usize) -> Vec<f64> {
    (0..tensor_len())
        .map(|index| {
            let x = index as f64;
            0.09 + 0.023 * kernel_id as f64 + (0.13 * x + 0.07 * kernel_id as f64).cos()
                - 0.015 * (0.19 * x).sin()
        })
        .collect()
}

fn compact_dft_2d(input: &[f64]) -> Vec<Complex64> {
    let [rows, cols] = DIMENSIONS;
    let half_cols = cols / 2 + 1;
    let mut output = vec![Complex64::default(); rows * half_cols];
    for k0 in 0..rows {
        for k1 in 0..half_cols {
            let mut sum = Complex64::default();
            for n0 in 0..rows {
                for n1 in 0..cols {
                    let phase = -std::f64::consts::TAU
                        * (k0 as f64 * n0 as f64 / rows as f64
                            + k1 as f64 * n1 as f64 / cols as f64);
                    sum += Complex64::new(
                        input[n0 * cols + n1] * phase.cos(),
                        input[n0 * cols + n1] * phase.sin(),
                    );
                }
            }
            output[k0 * half_cols + k1] = sum;
        }
    }
    output
}

fn kernel_spectrum(kernel_count: usize) -> Vec<Complex64> {
    (0..kernel_count)
        .flat_map(|kernel_id| compact_dft_2d(&spatial_kernel(kernel_id)))
        .collect()
}

fn direct_circular_convolution(input: &[f64], kernel: &[f64]) -> Vec<f64> {
    let [rows, cols] = DIMENSIONS;
    let mut output = vec![0.0; rows * cols];
    for out0 in 0..rows {
        for out1 in 0..cols {
            let mut sum = 0.0;
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

fn expected64(kernel_count: usize) -> Vec<f64> {
    let input = input64();
    (0..kernel_count)
        .flat_map(|kernel_id| direct_circular_convolution(&input, &spatial_kernel(kernel_id)))
        .collect()
}

fn coordinate_input64() -> Vec<f64> {
    (0..COORDINATE_COUNT)
        .flat_map(|coordinate| {
            (0..tensor_len()).map(move |index| {
                let x = index as f64;
                0.21 + 0.17 * coordinate as f64 + (0.09 * x).sin() - 0.011 * x
                    + 0.03 * (0.07 * x + 0.13 * coordinate as f64).cos()
            })
        })
        .collect()
}

fn coordinate_spatial_kernel(kernel_id: usize, coordinate: usize) -> Vec<f64> {
    (0..tensor_len())
        .map(|index| {
            let x = index as f64;
            0.07 + 0.031 * kernel_id as f64
                + 0.019 * coordinate as f64
                + (0.12 * x + 0.05 * kernel_id as f64).cos()
                - 0.017 * (0.16 * x + 0.09 * coordinate as f64).sin()
        })
        .collect()
}

fn coordinate_kernel_spectrum() -> Vec<Complex64> {
    (0..COORDINATE_KERNEL_COUNT)
        .flat_map(|kernel_id| {
            (0..COORDINATE_COUNT).flat_map(move |coordinate| {
                compact_dft_2d(&coordinate_spatial_kernel(kernel_id, coordinate))
            })
        })
        .collect()
}

fn coordinate_expected64() -> Vec<f64> {
    let input = coordinate_input64();
    (0..COORDINATE_KERNEL_COUNT)
        .flat_map(|kernel_id| {
            let input = &input;
            (0..COORDINATE_COUNT).flat_map(move |coordinate| {
                let base = coordinate * tensor_len();
                direct_circular_convolution(
                    &input[base..base + tensor_len()],
                    &coordinate_spatial_kernel(kernel_id, coordinate),
                )
            })
        })
        .collect()
}

fn build_coordinates(profile: DeviceProfile, precision: Precision) -> NdRealConvolutionIr {
    NdRealConvolutionIr::build_from_spectrum(
        FftConfig::new(DIMENSIONS.to_vec())
            .with_transform(TransformKind::RealToComplex)
            .with_precision(precision)
            .with_convolution(true)
            .with_coordinate_features(COORDINATE_COUNT)
            .with_convolution_kernel_count(COORDINATE_KERNEL_COUNT),
        coordinate_kernel_spectrum(),
        profile,
    )
    .unwrap()
}

fn coordinate_padded_probe_input64() -> Vec<f64> {
    let mut input = coordinate_input64();
    for coordinate in 0..COORDINATE_COUNT {
        let base = coordinate * tensor_len();
        for index in 0..tensor_len() {
            if matrix_padding_contains(index) {
                input[base + index] = 30_000.0 + 100.0 * coordinate as f64 + index as f64;
            }
        }
    }
    input
}

fn coordinate_padded_expected64(input: &[f64]) -> Vec<f64> {
    let mut masked = input.to_vec();
    for coordinate in 0..COORDINATE_COUNT {
        let base = coordinate * tensor_len();
        for index in 0..tensor_len() {
            if matrix_padding_contains(index) {
                masked[base + index] = 0.0;
            }
        }
    }
    let mut output = Vec::with_capacity(COORDINATE_KERNEL_COUNT * COORDINATE_COUNT * tensor_len());
    for kernel_id in 0..COORDINATE_KERNEL_COUNT {
        for coordinate in 0..COORDINATE_COUNT {
            let base = coordinate * tensor_len();
            let mut row = direct_circular_convolution(
                &masked[base..base + tensor_len()],
                &coordinate_spatial_kernel(kernel_id, coordinate),
            );
            for (index, value) in row.iter_mut().enumerate() {
                if matrix_padding_contains(index) {
                    *value = 0.0;
                }
            }
            output.extend(row);
        }
    }
    output
}

fn build_padded_coordinates(profile: DeviceProfile, precision: Precision) -> NdRealConvolutionIr {
    NdRealConvolutionIr::build_from_spectrum(
        FftConfig::new(DIMENSIONS.to_vec())
            .with_transform(TransformKind::RealToComplex)
            .with_precision(precision)
            .with_convolution(true)
            .with_coordinate_features(COORDINATE_COUNT)
            .with_convolution_kernel_count(COORDINATE_KERNEL_COUNT)
            .with_zero_padding(0, 2, 3)
            .unwrap()
            .with_zero_padding(1, 2, 4)
            .unwrap(),
        coordinate_kernel_spectrum(),
        profile,
    )
    .unwrap()
}

fn matrix_input64() -> Vec<f64> {
    (0..MATRIX_SIZE)
        .flat_map(|coordinate| {
            (0..tensor_len()).map(move |index| {
                let x = index as f64;
                0.18 + 0.11 * coordinate as f64 + (0.08 * x + 0.03 * coordinate as f64).sin()
                    - 0.009 * x
                    + 0.025 * (0.14 * x).cos()
            })
        })
        .collect()
}

fn matrix_spatial_kernel_for(
    kernel_id: usize,
    output_coordinate: usize,
    input_coordinate: usize,
) -> Vec<f64> {
    (0..tensor_len())
        .map(|index| {
            let x = index as f64;
            0.045
                + 0.029 * kernel_id as f64
                + 0.017 * output_coordinate as f64
                + 0.013 * input_coordinate as f64
                + (0.10 * x + 0.07 * output_coordinate as f64 + 0.031 * kernel_id as f64).cos()
                - 0.012
                    * (0.18 * x + 0.05 * input_coordinate as f64 + 0.043 * kernel_id as f64).sin()
        })
        .collect()
}

fn matrix_spatial_kernel(output_coordinate: usize, input_coordinate: usize) -> Vec<f64> {
    matrix_spatial_kernel_for(0, output_coordinate, input_coordinate)
}

fn matrix_kernel_spectrum() -> Vec<Complex64> {
    (0..MATRIX_SIZE)
        .flat_map(|output_coordinate| {
            (0..MATRIX_SIZE).flat_map(move |input_coordinate| {
                compact_dft_2d(&matrix_spatial_kernel(output_coordinate, input_coordinate))
            })
        })
        .collect()
}

fn matrix_expected64() -> Vec<f64> {
    let input = matrix_input64();
    (0..MATRIX_SIZE)
        .flat_map(|output_coordinate| {
            let mut row = vec![0.0; tensor_len()];
            for input_coordinate in 0..MATRIX_SIZE {
                let base = input_coordinate * tensor_len();
                let term = direct_circular_convolution(
                    &input[base..base + tensor_len()],
                    &matrix_spatial_kernel(output_coordinate, input_coordinate),
                );
                for (sum, value) in row.iter_mut().zip(term) {
                    *sum += value;
                }
            }
            row
        })
        .collect()
}

fn matrix_multi_kernel_spectrum() -> Vec<Complex64> {
    let mut spectrum =
        Vec::with_capacity(MATRIX_KERNEL_COUNT * MATRIX_SIZE * MATRIX_SIZE * compact_len());
    for kernel_id in 0..MATRIX_KERNEL_COUNT {
        for output_coordinate in 0..MATRIX_SIZE {
            for input_coordinate in 0..MATRIX_SIZE {
                spectrum.extend(compact_dft_2d(&matrix_spatial_kernel_for(
                    kernel_id,
                    output_coordinate,
                    input_coordinate,
                )));
            }
        }
    }
    spectrum
}

fn matrix_multi_expected64() -> Vec<f64> {
    let input = matrix_input64();
    let mut output = Vec::with_capacity(MATRIX_KERNEL_COUNT * MATRIX_SIZE * tensor_len());
    for kernel_id in 0..MATRIX_KERNEL_COUNT {
        for output_coordinate in 0..MATRIX_SIZE {
            let mut row = vec![0.0; tensor_len()];
            for input_coordinate in 0..MATRIX_SIZE {
                let base = input_coordinate * tensor_len();
                let term = direct_circular_convolution(
                    &input[base..base + tensor_len()],
                    &matrix_spatial_kernel_for(kernel_id, output_coordinate, input_coordinate),
                );
                for (sum, value) in row.iter_mut().zip(term) {
                    *sum += value;
                }
            }
            output.extend(row);
        }
    }
    output
}

fn build_multi_kernel_matrix(profile: DeviceProfile, precision: Precision) -> NdRealConvolutionIr {
    NdRealConvolutionIr::build_from_spectrum(
        FftConfig::new(DIMENSIONS.to_vec())
            .with_transform(TransformKind::RealToComplex)
            .with_precision(precision)
            .with_convolution(true)
            .with_matrix_convolution(MATRIX_SIZE)
            .with_convolution_kernel_count(MATRIX_KERNEL_COUNT),
        matrix_multi_kernel_spectrum(),
        profile,
    )
    .unwrap()
}

fn build_matrix(profile: DeviceProfile, precision: Precision) -> NdRealConvolutionIr {
    NdRealConvolutionIr::build_from_spectrum(
        FftConfig::new(DIMENSIONS.to_vec())
            .with_transform(TransformKind::RealToComplex)
            .with_precision(precision)
            .with_convolution(true)
            .with_matrix_convolution(MATRIX_SIZE),
        matrix_kernel_spectrum(),
        profile,
    )
    .unwrap()
}

fn matrix_padding_contains(index: usize) -> bool {
    let [_rows, cols] = DIMENSIONS;
    let row = index / cols;
    let col = index % cols;
    row >= 2 || col >= 2
}

fn matrix_padded_probe_input64() -> Vec<f64> {
    let mut input = matrix_input64();
    for coordinate in 0..MATRIX_SIZE {
        let base = coordinate * tensor_len();
        for index in 0..tensor_len() {
            if matrix_padding_contains(index) {
                input[base + index] = 10_000.0 + 100.0 * coordinate as f64 + index as f64;
            }
        }
    }
    input
}

fn matrix_padded_expected64(input: &[f64]) -> Vec<f64> {
    let mut masked = input.to_vec();
    for coordinate in 0..MATRIX_SIZE {
        let base = coordinate * tensor_len();
        for index in 0..tensor_len() {
            if matrix_padding_contains(index) {
                masked[base + index] = 0.0;
            }
        }
    }
    let mut output = (0..MATRIX_SIZE)
        .flat_map(|output_coordinate| {
            let mut row = vec![0.0; tensor_len()];
            for input_coordinate in 0..MATRIX_SIZE {
                let base = input_coordinate * tensor_len();
                let term = direct_circular_convolution(
                    &masked[base..base + tensor_len()],
                    &matrix_spatial_kernel(output_coordinate, input_coordinate),
                );
                for (sum, value) in row.iter_mut().zip(term) {
                    *sum += value;
                }
            }
            row
        })
        .collect::<Vec<_>>();
    for coordinate in 0..MATRIX_SIZE {
        let base = coordinate * tensor_len();
        for index in 0..tensor_len() {
            if matrix_padding_contains(index) {
                output[base + index] = 0.0;
            }
        }
    }
    output
}

fn build_padded_matrix(profile: DeviceProfile, precision: Precision) -> NdRealConvolutionIr {
    NdRealConvolutionIr::build_from_spectrum(
        FftConfig::new(DIMENSIONS.to_vec())
            .with_transform(TransformKind::RealToComplex)
            .with_precision(precision)
            .with_convolution(true)
            .with_matrix_convolution(MATRIX_SIZE)
            .with_zero_padding(0, 2, 3)
            .unwrap()
            .with_zero_padding(1, 2, 4)
            .unwrap(),
        matrix_kernel_spectrum(),
        profile,
    )
    .unwrap()
}

fn matrix_multi_padded_expected64(input: &[f64]) -> Vec<f64> {
    let mut masked = input.to_vec();
    for coordinate in 0..MATRIX_SIZE {
        let base = coordinate * tensor_len();
        for index in 0..tensor_len() {
            if matrix_padding_contains(index) {
                masked[base + index] = 0.0;
            }
        }
    }
    let mut output = Vec::with_capacity(MATRIX_KERNEL_COUNT * MATRIX_SIZE * tensor_len());
    for kernel_id in 0..MATRIX_KERNEL_COUNT {
        for output_coordinate in 0..MATRIX_SIZE {
            let mut row = vec![0.0; tensor_len()];
            for input_coordinate in 0..MATRIX_SIZE {
                let base = input_coordinate * tensor_len();
                let term = direct_circular_convolution(
                    &masked[base..base + tensor_len()],
                    &matrix_spatial_kernel_for(kernel_id, output_coordinate, input_coordinate),
                );
                for (sum, value) in row.iter_mut().zip(term) {
                    *sum += value;
                }
            }
            for (index, value) in row.iter_mut().enumerate() {
                if matrix_padding_contains(index) {
                    *value = 0.0;
                }
            }
            output.extend(row);
        }
    }
    output
}

fn build_padded_multi_kernel_matrix(
    profile: DeviceProfile,
    precision: Precision,
) -> NdRealConvolutionIr {
    NdRealConvolutionIr::build_from_spectrum(
        FftConfig::new(DIMENSIONS.to_vec())
            .with_transform(TransformKind::RealToComplex)
            .with_precision(precision)
            .with_convolution(true)
            .with_matrix_convolution(MATRIX_SIZE)
            .with_convolution_kernel_count(MATRIX_KERNEL_COUNT)
            .with_zero_padding(0, 2, 3)
            .unwrap()
            .with_zero_padding(1, 2, 4)
            .unwrap(),
        matrix_multi_kernel_spectrum(),
        profile,
    )
    .unwrap()
}

fn compact_inverse_dft_2d(spectrum: &[Complex64]) -> Vec<f64> {
    let [rows, cols] = DIMENSIONS;
    let half_cols = cols / 2 + 1;
    assert_eq!(spectrum.len(), rows * half_cols);
    let mut output = vec![0.0; rows * cols];
    for n0 in 0..rows {
        for n1 in 0..cols {
            let mut sum = 0.0;
            for k0 in 0..rows {
                for k1 in 0..cols {
                    let value = if k1 < half_cols {
                        spectrum[k0 * half_cols + k1]
                    } else {
                        let mirror0 = (rows - k0) % rows;
                        let mirror1 = cols - k1;
                        spectrum[mirror0 * half_cols + mirror1].conj()
                    };
                    let phase = std::f64::consts::TAU
                        * (k0 as f64 * n0 as f64 / rows as f64
                            + k1 as f64 * n1 as f64 / cols as f64);
                    sum += value.re * phase.cos() - value.im * phase.sin();
                }
            }
            output[n0 * cols + n1] = sum / (rows * cols) as f64;
        }
    }
    output
}

fn matrix_policy_padded_expected64(input: &[f64]) -> Vec<f64> {
    let mut masked = input.to_vec();
    for coordinate in 0..MATRIX_SIZE {
        let base = coordinate * tensor_len();
        for index in 0..tensor_len() {
            if matrix_padding_contains(index) {
                masked[base + index] = 0.0;
            }
        }
    }
    let input_spectra = (0..MATRIX_SIZE)
        .map(|coordinate| {
            let base = coordinate * tensor_len();
            compact_dft_2d(&masked[base..base + tensor_len()])
        })
        .collect::<Vec<_>>();
    let kernels = matrix_multi_kernel_spectrum();
    let mut output = Vec::with_capacity(MATRIX_KERNEL_COUNT * MATRIX_SIZE * tensor_len());
    for kernel_id in 0..MATRIX_KERNEL_COUNT {
        for output_coordinate in 0..MATRIX_SIZE {
            let mut row_spectrum = vec![Complex64::default(); compact_len()];
            for index in 0..compact_len() {
                let mut sum = Complex64::default();
                for (input_coordinate, input_spectrum) in input_spectra.iter().enumerate() {
                    let plane = output_coordinate * MATRIX_SIZE + input_coordinate;
                    let kernel_offset =
                        ((kernel_id * MATRIX_SIZE * MATRIX_SIZE + plane) * compact_len()) + index;
                    sum += input_spectrum[index].conj() * kernels[kernel_offset];
                }
                row_spectrum[index] = sum.scale(1.0 / sum.norm_sqr().sqrt());
            }
            output.extend(compact_inverse_dft_2d(&row_spectrum));
        }
    }
    for system in 0..MATRIX_KERNEL_COUNT * MATRIX_SIZE {
        let base = system * tensor_len();
        for index in 0..tensor_len() {
            if matrix_padding_contains(index) {
                output[base + index] = 0.0;
            }
        }
    }
    output
}

fn build_policy_padded_multi_kernel_matrix(
    profile: DeviceProfile,
    precision: Precision,
) -> NdRealConvolutionIr {
    NdRealConvolutionIr::build_from_spectrum(
        FftConfig::new(DIMENSIONS.to_vec())
            .with_transform(TransformKind::RealToComplex)
            .with_precision(precision)
            .with_convolution(true)
            .with_matrix_convolution(MATRIX_SIZE)
            .with_convolution_kernel_count(MATRIX_KERNEL_COUNT)
            .with_convolution_conjugation(ConvolutionConjugation::Sequence)
            .with_cross_power_spectrum_normalization(true)
            .with_zero_padding(0, 2, 3)
            .unwrap()
            .with_zero_padding(1, 2, 4)
            .unwrap(),
        matrix_multi_kernel_spectrum(),
        profile,
    )
    .unwrap()
}

fn build(profile: DeviceProfile, precision: Precision, kernel_count: usize) -> NdRealConvolutionIr {
    NdRealConvolutionIr::build_from_spectrum(
        FftConfig::new(DIMENSIONS.to_vec())
            .with_transform(TransformKind::RealToComplex)
            .with_precision(precision)
            .with_convolution(true)
            .with_convolution_kernel_count(kernel_count),
        kernel_spectrum(kernel_count),
        profile,
    )
    .unwrap()
}

fn build_formatted(profile: DeviceProfile, precision: Precision) -> NdRealConvolutionIr {
    NdRealConvolutionIr::build_from_spectrum(
        FftConfig::new(DIMENSIONS.to_vec())
            .with_transform(TransformKind::RealToComplex)
            .with_precision(precision)
            .with_convolution(true)
            .with_input_buffer_axis_stride(0, 7)
            .unwrap()
            .with_output_buffer_axis_stride(0, 9)
            .unwrap(),
        kernel_spectrum(1),
        profile,
    )
    .unwrap()
}

fn build_mixed(profile: DeviceProfile, precision: Precision) -> NdRealConvolutionIr {
    NdRealConvolutionIr::build_from_spectrum(
        FftConfig::new(DIMENSIONS.to_vec())
            .with_transform(TransformKind::RealToComplex)
            .with_precision(precision)
            .with_convolution(true),
        kernel_spectrum(1),
        profile,
    )
    .unwrap()
}

fn build_mixed_padded(profile: DeviceProfile, precision: Precision) -> NdRealConvolutionIr {
    NdRealConvolutionIr::build_from_spectrum(
        FftConfig::new(DIMENSIONS.to_vec())
            .with_transform(TransformKind::RealToComplex)
            .with_precision(precision)
            .with_convolution(true)
            .with_zero_padding(0, 2, 3)
            .unwrap()
            .with_zero_padding(1, 2, 4)
            .unwrap(),
        kernel_spectrum(1),
        profile,
    )
    .unwrap()
}

fn build_mixed_padded_fanout(profile: DeviceProfile, precision: Precision) -> NdRealConvolutionIr {
    NdRealConvolutionIr::build_from_spectrum(
        FftConfig::new(DIMENSIONS.to_vec())
            .with_transform(TransformKind::RealToComplex)
            .with_precision(precision)
            .with_convolution(true)
            .with_convolution_kernel_count(KERNEL_COUNT)
            .with_zero_padding(0, 2, 3)
            .unwrap()
            .with_zero_padding(1, 2, 4)
            .unwrap(),
        kernel_spectrum(KERNEL_COUNT),
        profile,
    )
    .unwrap()
}

fn formatted_padded_input64() -> Vec<f64> {
    let mut input = input64();
    for (index, value) in input.iter_mut().enumerate() {
        if matrix_padding_contains(index) {
            *value = 20_000.0 + index as f64;
        }
    }
    input
}

fn formatted_padded_expected64(input: &[f64]) -> Vec<f64> {
    let mut masked = input.to_vec();
    for (index, value) in masked.iter_mut().enumerate() {
        if matrix_padding_contains(index) {
            *value = 0.0;
        }
    }
    let mut output = direct_circular_convolution(&masked, &spatial_kernel(0));
    for (index, value) in output.iter_mut().enumerate() {
        if matrix_padding_contains(index) {
            *value = 0.0;
        }
    }
    output
}

fn padded_fanout_expected64(input: &[f64]) -> Vec<f64> {
    let mut masked = input.to_vec();
    for (index, value) in masked.iter_mut().enumerate() {
        if matrix_padding_contains(index) {
            *value = 0.0;
        }
    }
    (0..KERNEL_COUNT)
        .flat_map(|kernel_id| {
            let mut output = direct_circular_convolution(&masked, &spatial_kernel(kernel_id));
            for (index, value) in output.iter_mut().enumerate() {
                if matrix_padding_contains(index) {
                    *value = 0.0;
                }
            }
            output
        })
        .collect()
}

fn build_formatted_padded(profile: DeviceProfile, precision: Precision) -> NdRealConvolutionIr {
    NdRealConvolutionIr::build_from_spectrum(
        FftConfig::new(DIMENSIONS.to_vec())
            .with_transform(TransformKind::RealToComplex)
            .with_precision(precision)
            .with_convolution(true)
            .with_input_buffer_axis_stride(0, 7)
            .unwrap()
            .with_output_buffer_axis_stride(0, 9)
            .unwrap()
            .with_zero_padding(0, 2, 3)
            .unwrap()
            .with_zero_padding(1, 2, 4)
            .unwrap(),
        kernel_spectrum(1),
        profile,
    )
    .unwrap()
}

fn assert_close(actual: &[f64], expected: &[f64], tolerance: f64, label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label} length mismatch");
    let error = actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0_f64, f64::max);
    assert!(error <= tolerance, "{label} error {error:e}");
}

#[test]
fn nd_real_convolution_matches_direct_spatial_oracle_and_program_ownership() {
    let profile = device(Backend::Vulkan, GpuVendor::Nvidia);
    let input = input64();
    for kernel_count in [1usize, KERNEL_COUNT] {
        let expected = expected64(kernel_count);
        for precision in [Precision::F32, Precision::F64] {
            let ir = build(profile, precision, kernel_count);
            assert_eq!(ir.batch_count, 1);
            assert_eq!(ir.full_tensor_len, tensor_len());
            assert_eq!(ir.compact_tensor_len, compact_len());
            assert_eq!(ir.kernel_count, kernel_count);
            assert_eq!(ir.forward_r2c.batch_count, 1);
            assert_eq!(ir.inverse_c2r.batch_count, kernel_count);
            assert_eq!(ir.multiply.sequence_len, compact_len());
            assert_eq!(ir.multiply.kernel_count, kernel_count);
            assert_eq!(ir.multiply.dispatch.x, kernel_count as u32);
            assert_eq!(ir.kernel_spectrum().len(), kernel_count * compact_len());

            let actual = execute_nd_real_convolution_ir(&ir, &input).unwrap();
            assert_close(&actual, &expected, 4.0e-10, "CPU ND real convolution");

            let program = ProgramIr::nd_real_convolution(&ir).unwrap();
            let resource = |name: &str| {
                program
                    .resources
                    .iter()
                    .find(|resource| resource.name == name)
                    .unwrap()
            };
            assert_eq!(resource("input").elements, tensor_len());
            assert_eq!(resource("output").elements, kernel_count * tensor_len());
            assert_eq!(
                resource("nd_real_convolution_forward_spectrum").elements,
                compact_len()
            );
            assert_eq!(
                resource("nd_real_convolution_multiplied_spectrum").elements,
                kernel_count * compact_len()
            );
            assert_eq!(
                resource("nd_real_convolution_kernel_spectrum").elements,
                kernel_count * compact_len()
            );
            assert_eq!(resource("input").external_layout.unwrap().batch_count, 1);
            assert_eq!(
                resource("output").external_layout.unwrap().batch_count,
                kernel_count
            );

            let shaders = VulkanGlslBackend.lower_nd_real_convolution(&ir).unwrap();
            assert_eq!(program.passes.len(), shaders.len());
            let multiply_index = program
                .passes
                .iter()
                .position(|pass| pass.name == ir.multiply.name)
                .unwrap();
            assert!(multiply_index > 0 && multiply_index + 1 < program.passes.len());
            assert!(
                program.passes[..multiply_index]
                    .iter()
                    .all(|pass| pass.name.starts_with("vkfft_nd_real_convolution_forward_"))
            );
            assert!(
                program.passes[multiply_index + 1..]
                    .iter()
                    .all(|pass| pass.name.starts_with("vkfft_nd_real_convolution_inverse_"))
            );
            for shader in shaders {
                assert_eq!(shader.compile_spirv().unwrap().words[0], 0x0723_0203);
            }
        }
    }
}

#[test]
fn nd_real_mixed_storage_k1_keeps_only_true_caller_boundaries_narrow() {
    let profile = device(Backend::Vulkan, GpuVendor::Nvidia);
    let expected = expected64(1);
    for (precision, compute, storage, tolerance) in [
        (
            Precision::F16StorageF32Compute,
            ScalarType::F32,
            ScalarType::F16,
            8.0e-2,
        ),
        (
            Precision::F64ComputeF32Storage,
            ScalarType::F64,
            ScalarType::F32,
            2.0e-5,
        ),
    ] {
        let ir = build_mixed(profile, precision);
        assert_eq!(ir.scalar, compute);
        assert_eq!(ir.external_scalar, storage);
        assert_eq!(ir.forward_r2c.external_scalar, storage);
        assert_eq!(ir.inverse_c2r.external_scalar, storage);
        assert!(ir.forward_r2c.output_boundary_compute_storage);
        assert!(ir.inverse_c2r.input_boundary_compute_storage);
        assert!(!ir.forward_r2c.input_boundary_compute_storage);
        assert!(!ir.inverse_c2r.output_boundary_compute_storage);

        let program = ProgramIr::nd_real_convolution(&ir).unwrap();
        assert_eq!(
            program
                .resources
                .iter()
                .find(|resource| resource.name == "input")
                .unwrap()
                .scalar,
            storage
        );
        assert_eq!(
            program
                .resources
                .iter()
                .find(|resource| resource.name == "output")
                .unwrap()
                .scalar,
            storage
        );
        for name in [
            "nd_real_convolution_kernel_spectrum",
            "nd_real_convolution_forward_spectrum",
            "nd_real_convolution_multiplied_spectrum",
        ] {
            assert_eq!(
                program
                    .resources
                    .iter()
                    .find(|resource| resource.name == name)
                    .unwrap()
                    .scalar,
                compute,
                "{name} must stay in compute storage"
            );
        }

        let actual = execute_nd_real_convolution_ir(&ir, &input64()).unwrap();
        assert_close(
            &actual,
            &expected,
            tolerance,
            "CPU ND real mixed storage K1",
        );
        for shader in VulkanGlslBackend.lower_nd_real_convolution(&ir).unwrap() {
            assert_eq!(shader.compile_spirv().unwrap().words[0], 0x0723_0203);
        }
    }
}

#[test]
fn nd_real_mixed_storage_scalar_fanout_keeps_one_forward_spectrum_and_k_external_outputs() {
    let profile = device(Backend::Vulkan, GpuVendor::Nvidia);
    let expected = expected64(KERNEL_COUNT);
    for (precision, compute, storage, tolerance) in [
        (
            Precision::F16StorageF32Compute,
            ScalarType::F32,
            ScalarType::F16,
            8.0e-2,
        ),
        (
            Precision::F64ComputeF32Storage,
            ScalarType::F64,
            ScalarType::F32,
            2.0e-5,
        ),
    ] {
        let ir = build(profile, precision, KERNEL_COUNT);
        assert_eq!(ir.scalar, compute);
        assert_eq!(ir.external_scalar, storage);
        assert_eq!(ir.kernel_count, KERNEL_COUNT);
        assert_eq!(ir.coordinate_count, 1);
        assert_eq!(ir.forward_r2c.batch_count, 1);
        assert_eq!(ir.inverse_c2r.batch_count, KERNEL_COUNT);
        assert_eq!(ir.output_batch_count(), KERNEL_COUNT);
        assert_eq!(ir.multiply.kernel_count, KERNEL_COUNT);
        assert_eq!(ir.multiply.dispatch.x, KERNEL_COUNT as u32);
        assert!(ir.zero_padding.iter().all(Option::is_none));
        assert!(ir.forward_r2c.input_formatted_copy.is_none());
        assert!(ir.inverse_c2r.output_formatted_copy.is_none());
        assert!(ir.forward_r2c.output_boundary_compute_storage);
        assert!(ir.inverse_c2r.input_boundary_compute_storage);

        let program = ProgramIr::nd_real_convolution(&ir).unwrap();
        let resource = |name: &str| {
            program
                .resources
                .iter()
                .find(|resource| resource.name == name)
                .unwrap()
        };
        assert_eq!(resource("input").scalar, storage);
        assert_eq!(resource("output").scalar, storage);
        assert_eq!(
            resource("output")
                .external_layout
                .as_ref()
                .unwrap()
                .batch_count,
            KERNEL_COUNT
        );
        assert_eq!(
            resource("nd_real_convolution_forward_spectrum").elements,
            compact_len()
        );
        assert_eq!(
            resource("nd_real_convolution_multiplied_spectrum").elements,
            KERNEL_COUNT * compact_len()
        );
        assert_eq!(
            resource("nd_real_convolution_kernel_spectrum").elements,
            KERNEL_COUNT * compact_len()
        );
        for name in [
            "nd_real_convolution_kernel_spectrum",
            "nd_real_convolution_forward_spectrum",
            "nd_real_convolution_multiplied_spectrum",
        ] {
            assert_eq!(resource(name).scalar, compute, "{name}");
        }

        let actual = execute_nd_real_convolution_ir(&ir, &input64()).unwrap();
        assert_close(
            &actual,
            &expected,
            tolerance,
            "CPU ND real mixed storage scalar K3 fan-out",
        );
        for shader in VulkanGlslBackend.lower_nd_real_convolution(&ir).unwrap() {
            assert_eq!(shader.compile_spirv().unwrap().words[0], 0x0723_0203);
        }
    }
}

#[test]
fn nd_real_mixed_storage_scalar_fanout_spatial_padding_masks_one_input_and_k_outputs() {
    let profile = device(Backend::Vulkan, GpuVendor::Nvidia);
    let input = formatted_padded_input64();
    let expected = padded_fanout_expected64(&input);
    for (precision, compute, storage, tolerance) in [
        (
            Precision::F16StorageF32Compute,
            ScalarType::F32,
            ScalarType::F16,
            8.0e-2,
        ),
        (
            Precision::F64ComputeF32Storage,
            ScalarType::F64,
            ScalarType::F32,
            2.0e-5,
        ),
    ] {
        let ir = build_mixed_padded_fanout(profile, precision);
        assert_eq!(ir.kernel_count, KERNEL_COUNT);
        assert_eq!(ir.forward_r2c.batch_count, 1);
        assert_eq!(ir.inverse_c2r.batch_count, KERNEL_COUNT);
        let forward_zero = ir.forward_r2c.zero_pad_pass.as_ref().unwrap();
        let inverse_zero = ir.inverse_c2r.zero_pad_pass.as_ref().unwrap();
        assert_eq!(forward_zero.batch_count, 1);
        assert_eq!(inverse_zero.batch_count, KERNEL_COUNT);
        assert_eq!(forward_zero.input_storage_scalar, storage);
        assert_eq!(forward_zero.output_storage_scalar, compute);
        assert_eq!(inverse_zero.input_storage_scalar, compute);
        assert_eq!(inverse_zero.output_storage_scalar, storage);
        assert_eq!(forward_zero.ranges, ir.zero_padding);
        assert_eq!(inverse_zero.ranges, ir.zero_padding);
        assert!(ir.forward_r2c.input_formatted_copy.is_none());
        assert!(ir.inverse_c2r.output_formatted_copy.is_none());

        let program = ProgramIr::nd_real_convolution(&ir).unwrap();
        let resource = |name: &str| {
            program
                .resources
                .iter()
                .find(|resource| resource.name == name)
                .unwrap()
        };
        assert_eq!(resource("input").scalar, storage);
        assert_eq!(resource("output").scalar, storage);
        assert_eq!(
            resource("output")
                .external_layout
                .as_ref()
                .unwrap()
                .batch_count,
            KERNEL_COUNT
        );
        assert_eq!(
            resource("nd_real_convolution_forward_spectrum").elements,
            compact_len()
        );
        assert_eq!(
            resource("nd_real_convolution_multiplied_spectrum").elements,
            KERNEL_COUNT * compact_len()
        );
        for name in [
            "nd_real_convolution_kernel_spectrum",
            "nd_real_convolution_forward_spectrum",
            "nd_real_convolution_multiplied_spectrum",
        ] {
            assert_eq!(resource(name).scalar, compute, "{name}");
        }

        let actual = execute_nd_real_convolution_ir(&ir, &input).unwrap();
        assert_close(
            &actual,
            &expected,
            tolerance,
            "CPU ND real mixed storage scalar K3 spatial padding",
        );
        for shader in VulkanGlslBackend.lower_nd_real_convolution(&ir).unwrap() {
            assert_eq!(shader.compile_spirv().unwrap().words[0], 0x0723_0203);
        }
    }
}

#[test]
fn nd_real_mixed_storage_k1_spatial_padding_converts_only_at_masked_caller_edges() {
    let profile = device(Backend::Vulkan, GpuVendor::Nvidia);
    let input = formatted_padded_input64();
    let expected = formatted_padded_expected64(&input);
    for (precision, compute, storage, tolerance) in [
        (
            Precision::F16StorageF32Compute,
            ScalarType::F32,
            ScalarType::F16,
            8.0e-2,
        ),
        (
            Precision::F64ComputeF32Storage,
            ScalarType::F64,
            ScalarType::F32,
            2.0e-5,
        ),
    ] {
        let ir = build_mixed_padded(profile, precision);
        let forward_zero = ir.forward_r2c.zero_pad_pass.as_ref().unwrap();
        let inverse_zero = ir.inverse_c2r.zero_pad_pass.as_ref().unwrap();
        assert_eq!(forward_zero.ranges, ir.zero_padding);
        assert_eq!(inverse_zero.ranges, ir.zero_padding);
        assert_eq!(forward_zero.input_storage_scalar, storage);
        assert_eq!(forward_zero.output_storage_scalar, compute);
        assert_eq!(inverse_zero.input_storage_scalar, compute);
        assert_eq!(inverse_zero.output_storage_scalar, storage);
        assert!(ir.forward_r2c.output_boundary_compute_storage);
        assert!(ir.inverse_c2r.input_boundary_compute_storage);
        assert!(!ir.forward_r2c.input_boundary_compute_storage);
        assert!(!ir.inverse_c2r.output_boundary_compute_storage);

        let program = ProgramIr::nd_real_convolution(&ir).unwrap();
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
        assert_eq!(input_resource.scalar, storage);
        assert_eq!(output_resource.scalar, storage);
        for name in [
            "nd_real_convolution_kernel_spectrum",
            "nd_real_convolution_forward_spectrum",
            "nd_real_convolution_multiplied_spectrum",
        ] {
            assert_eq!(
                program
                    .resources
                    .iter()
                    .find(|resource| resource.name == name)
                    .unwrap()
                    .scalar,
                compute,
                "{name} must stay in compute storage"
            );
        }
        let multiply_index = program
            .passes
            .iter()
            .position(|pass| pass.name == ir.multiply.name)
            .unwrap();
        assert!(
            program.passes[..multiply_index]
                .iter()
                .any(|pass| pass.name.contains("zero_pad"))
        );
        assert!(
            program.passes[multiply_index + 1..]
                .iter()
                .any(|pass| pass.name.contains("zero_pad"))
        );

        let actual = execute_nd_real_convolution_ir(&ir, &input).unwrap();
        assert_close(
            &actual,
            &expected,
            tolerance,
            "CPU ND real mixed storage K1 spatial padding",
        );
        for shader in VulkanGlslBackend.lower_nd_real_convolution(&ir).unwrap() {
            assert_eq!(shader.compile_spirv().unwrap().words[0], 0x0723_0203);
        }
    }
}

#[test]
fn nd_real_formatted_k1_keeps_only_true_caller_boundaries_physical() {
    let profile = device(Backend::Vulkan, GpuVendor::Nvidia);
    let input = input64();
    let expected = expected64(1);
    for precision in [Precision::F32, Precision::F64] {
        let ir = build_formatted(profile, precision);
        assert!(ir.forward_r2c.input_formatted_copy.is_some());
        assert!(ir.forward_r2c.output_formatted_copy.is_none());
        assert!(ir.inverse_c2r.input_formatted_copy.is_none());
        assert!(ir.inverse_c2r.output_formatted_copy.is_some());
        assert_eq!(
            ir.forward_r2c.input_external_layout.axis_strides,
            vec![7, 1]
        );
        assert_eq!(
            ir.inverse_c2r.output_external_layout.axis_strides,
            vec![9, 1]
        );
        assert_eq!(
            ir.forward_r2c.output_external_layout.batch_stride,
            compact_len()
        );
        assert_eq!(
            ir.inverse_c2r.input_external_layout.batch_stride,
            compact_len()
        );

        let actual = execute_nd_real_convolution_ir(&ir, &input).unwrap();
        assert_close(&actual, &expected, 4.0e-10, "CPU ND real formatted K1");

        let program = ProgramIr::nd_real_convolution(&ir).unwrap();
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
        assert_eq!(
            input_resource.elements,
            ir.forward_r2c.input_external_layout.batch_stride
        );
        assert_eq!(
            output_resource.elements,
            ir.inverse_c2r.output_external_layout.batch_stride
        );
        assert_eq!(
            program
                .resources
                .iter()
                .find(|resource| resource.name == "nd_real_convolution_forward_spectrum")
                .unwrap()
                .elements,
            compact_len()
        );
        assert_eq!(
            program
                .resources
                .iter()
                .find(|resource| resource.name == "nd_real_convolution_multiplied_spectrum")
                .unwrap()
                .elements,
            compact_len()
        );
        let multiply_index = program
            .passes
            .iter()
            .position(|pass| pass.name == ir.multiply.name)
            .unwrap();
        assert!(
            program.passes[..multiply_index]
                .iter()
                .any(|pass| pass.name.contains("gather_formatted_input"))
        );
        assert!(
            program.passes[multiply_index + 1..]
                .iter()
                .any(|pass| pass.name.contains("scatter_formatted_output"))
        );
        assert!(!program.passes.iter().any(|pass| {
            pass.name.contains("gather_formatted_input") && pass.name.contains("inverse")
        }));
        assert!(!program.passes.iter().any(|pass| {
            pass.name.contains("scatter_formatted_output") && pass.name.contains("forward")
        }));
        for shader in VulkanGlslBackend.lower_nd_real_convolution(&ir).unwrap() {
            assert_eq!(shader.compile_spirv().unwrap().words[0], 0x0723_0203);
        }
    }
}

#[test]
fn nd_real_mixed_storage_formatted_k1_converts_only_in_physical_caller_copies() {
    let profile = device(Backend::Vulkan, GpuVendor::Nvidia);
    let expected = expected64(1);
    for (precision, compute, storage, tolerance) in [
        (
            Precision::F16StorageF32Compute,
            ScalarType::F32,
            ScalarType::F16,
            8.0e-2,
        ),
        (
            Precision::F64ComputeF32Storage,
            ScalarType::F64,
            ScalarType::F32,
            2.0e-5,
        ),
    ] {
        let ir = build_formatted(profile, precision);
        let gather = ir.forward_r2c.input_formatted_copy.as_ref().unwrap();
        let scatter = ir.inverse_c2r.output_formatted_copy.as_ref().unwrap();
        assert_eq!(gather.input_storage_scalar, storage);
        assert_eq!(gather.output_storage_scalar, compute);
        assert_eq!(scatter.input_storage_scalar, compute);
        assert_eq!(scatter.output_storage_scalar, storage);
        assert_eq!(gather.external_layout.axis_strides, vec![7, 1]);
        assert_eq!(scatter.external_layout.axis_strides, vec![9, 1]);
        assert!(ir.forward_r2c.output_boundary_compute_storage);
        assert!(ir.inverse_c2r.input_boundary_compute_storage);
        assert!(!ir.forward_r2c.input_boundary_compute_storage);
        assert!(!ir.inverse_c2r.output_boundary_compute_storage);

        let program = ProgramIr::nd_real_convolution(&ir).unwrap();
        let resource = |name: &str| {
            program
                .resources
                .iter()
                .find(|resource| resource.name == name)
                .unwrap()
        };
        assert_eq!(resource("input").scalar, storage);
        assert_eq!(resource("output").scalar, storage);
        for name in [
            "nd_real_convolution_kernel_spectrum",
            "nd_real_convolution_forward_spectrum",
            "nd_real_convolution_multiplied_spectrum",
        ] {
            assert_eq!(resource(name).scalar, compute, "{name}");
        }
        let multiply_index = program
            .passes
            .iter()
            .position(|pass| pass.name == ir.multiply.name)
            .unwrap();
        assert!(
            program.passes[..multiply_index]
                .iter()
                .any(|pass| pass.name.contains("gather_formatted_input"))
        );
        assert!(
            program.passes[multiply_index + 1..]
                .iter()
                .any(|pass| pass.name.contains("scatter_formatted_output"))
        );

        let actual = execute_nd_real_convolution_ir(&ir, &input64()).unwrap();
        assert_close(
            &actual,
            &expected,
            tolerance,
            "CPU ND real mixed storage formatted K1",
        );
        for shader in VulkanGlslBackend.lower_nd_real_convolution(&ir).unwrap() {
            assert_eq!(shader.compile_spirv().unwrap().words[0], 0x0723_0203);
        }
    }
}

#[test]
fn nd_real_mixed_storage_formatted_padding_orders_pitch_conversion_outside_logical_mask() {
    let profile = device(Backend::Vulkan, GpuVendor::Nvidia);
    let input = formatted_padded_input64();
    let expected = formatted_padded_expected64(&input);
    for (precision, compute, storage, tolerance) in [
        (
            Precision::F16StorageF32Compute,
            ScalarType::F32,
            ScalarType::F16,
            8.0e-2,
        ),
        (
            Precision::F64ComputeF32Storage,
            ScalarType::F64,
            ScalarType::F32,
            2.0e-5,
        ),
    ] {
        let ir = build_formatted_padded(profile, precision);
        let gather = ir.forward_r2c.input_formatted_copy.as_ref().unwrap();
        let scatter = ir.inverse_c2r.output_formatted_copy.as_ref().unwrap();
        let forward_zero = ir.forward_r2c.zero_pad_pass.as_ref().unwrap();
        let inverse_zero = ir.inverse_c2r.zero_pad_pass.as_ref().unwrap();
        assert_eq!(gather.input_storage_scalar, storage);
        assert_eq!(gather.output_storage_scalar, compute);
        assert_eq!(forward_zero.input_storage_scalar, compute);
        assert_eq!(forward_zero.output_storage_scalar, compute);
        assert_eq!(inverse_zero.input_storage_scalar, compute);
        assert_eq!(inverse_zero.output_storage_scalar, compute);
        assert_eq!(scatter.input_storage_scalar, compute);
        assert_eq!(scatter.output_storage_scalar, storage);
        assert_eq!(gather.external_layout.axis_strides, vec![7, 1]);
        assert_eq!(scatter.external_layout.axis_strides, vec![9, 1]);
        assert_eq!(forward_zero.ranges, ir.zero_padding);
        assert_eq!(inverse_zero.ranges, ir.zero_padding);
        assert!(ir.forward_r2c.output_boundary_compute_storage);
        assert!(ir.inverse_c2r.input_boundary_compute_storage);

        let program = ProgramIr::nd_real_convolution(&ir).unwrap();
        let resource = |name: &str| {
            program
                .resources
                .iter()
                .find(|resource| resource.name == name)
                .unwrap()
        };
        assert_eq!(resource("input").scalar, storage);
        assert_eq!(resource("output").scalar, storage);
        for name in [
            "nd_real_convolution_kernel_spectrum",
            "nd_real_convolution_forward_spectrum",
            "nd_real_convolution_multiplied_spectrum",
        ] {
            assert_eq!(resource(name).scalar, compute, "{name}");
        }
        let multiply_index = program
            .passes
            .iter()
            .position(|pass| pass.name == ir.multiply.name)
            .unwrap();
        let forward_names = program.passes[..multiply_index]
            .iter()
            .map(|pass| pass.name.as_str())
            .collect::<Vec<_>>();
        let inverse_names = program.passes[multiply_index + 1..]
            .iter()
            .map(|pass| pass.name.as_str())
            .collect::<Vec<_>>();
        let gather_index = forward_names
            .iter()
            .position(|name| name.contains("gather_formatted_input"))
            .unwrap();
        let forward_zero_index = forward_names
            .iter()
            .position(|name| name.contains("zero_pad"))
            .unwrap();
        let inverse_zero_index = inverse_names
            .iter()
            .rposition(|name| name.contains("zero_pad"))
            .unwrap();
        let scatter_index = inverse_names
            .iter()
            .rposition(|name| name.contains("scatter_formatted_output"))
            .unwrap();
        assert!(gather_index < forward_zero_index);
        assert!(inverse_zero_index < scatter_index);

        let actual = execute_nd_real_convolution_ir(&ir, &input).unwrap();
        assert_close(
            &actual,
            &expected,
            tolerance,
            "CPU ND real mixed storage formatted spatial padding K1",
        );
        for shader in VulkanGlslBackend.lower_nd_real_convolution(&ir).unwrap() {
            assert_eq!(shader.compile_spirv().unwrap().words[0], 0x0723_0203);
        }
    }
}

#[test]
fn nd_real_formatted_k1_spatial_padding_keeps_logical_mask_inside_physical_boundaries() {
    let profile = device(Backend::Vulkan, GpuVendor::Nvidia);
    let input = formatted_padded_input64();
    let expected = formatted_padded_expected64(&input);
    for precision in [Precision::F32, Precision::F64] {
        let ir = build_formatted_padded(profile, precision);
        assert!(ir.forward_r2c.input_formatted_copy.is_some());
        assert!(ir.inverse_c2r.output_formatted_copy.is_some());
        let forward_zero = ir.forward_r2c.zero_pad_pass.as_ref().unwrap();
        let inverse_zero = ir.inverse_c2r.zero_pad_pass.as_ref().unwrap();
        assert_eq!(forward_zero.ranges, ir.zero_padding);
        assert_eq!(inverse_zero.ranges, ir.zero_padding);

        let actual = execute_nd_real_convolution_ir(&ir, &input).unwrap();
        assert_close(
            &actual,
            &expected,
            8.0e-10,
            "CPU ND real formatted K1 spatial padding",
        );

        let program = ProgramIr::nd_real_convolution(&ir).unwrap();
        let multiply_index = program
            .passes
            .iter()
            .position(|pass| pass.name == ir.multiply.name)
            .unwrap();
        let forward_names = program.passes[..multiply_index]
            .iter()
            .map(|pass| pass.name.as_str())
            .collect::<Vec<_>>();
        let inverse_names = program.passes[multiply_index + 1..]
            .iter()
            .map(|pass| pass.name.as_str())
            .collect::<Vec<_>>();
        let gather = forward_names
            .iter()
            .position(|name| name.contains("gather_formatted_input"))
            .unwrap();
        let forward_zero_index = forward_names
            .iter()
            .position(|name| name.contains("zero_pad"))
            .unwrap();
        let inverse_zero_index = inverse_names
            .iter()
            .rposition(|name| name.contains("zero_pad"))
            .unwrap();
        let scatter = inverse_names
            .iter()
            .rposition(|name| name.contains("scatter_formatted_output"))
            .unwrap();
        assert!(gather < forward_zero_index);
        assert!(inverse_zero_index < scatter);
        for shader in VulkanGlslBackend.lower_nd_real_convolution(&ir).unwrap() {
            assert_eq!(shader.compile_spirv().unwrap().words[0], 0x0723_0203);
        }
    }
}

#[test]
fn nd_real_independent_coordinates_match_sample52_layout_and_spatial_oracle() {
    let profile = device(Backend::Vulkan, GpuVendor::Nvidia);
    let input = coordinate_input64();
    let expected = coordinate_expected64();
    for precision in [Precision::F32, Precision::F64] {
        let ir = build_coordinates(profile, precision);
        assert_eq!(ir.batch_count, 1);
        assert_eq!(ir.coordinate_count, COORDINATE_COUNT);
        assert_eq!(ir.kernel_count, COORDINATE_KERNEL_COUNT);
        assert_eq!(ir.output_batch_count(), COORDINATE_KERNEL_COUNT);
        assert_eq!(
            ir.output_system_count().unwrap(),
            COORDINATE_KERNEL_COUNT * COORDINATE_COUNT
        );
        assert_eq!(ir.forward_r2c.batch_count, COORDINATE_COUNT);
        assert_eq!(
            ir.inverse_c2r.batch_count,
            COORDINATE_KERNEL_COUNT * COORDINATE_COUNT
        );
        assert!(ir.multiply.independent_coordinates);
        assert!(ir.multiply.matrix_layout.is_none());
        assert_eq!(ir.multiply.coordinate_count, COORDINATE_COUNT);
        assert_eq!(ir.multiply.kernel_count, COORDINATE_KERNEL_COUNT);
        assert_eq!(ir.multiply.dispatch.x, COORDINATE_KERNEL_COUNT as u32);
        assert_eq!(
            ir.kernel_spectrum().len(),
            COORDINATE_KERNEL_COUNT * COORDINATE_COUNT * compact_len()
        );

        let actual = execute_nd_real_convolution_ir(&ir, &input).unwrap();
        assert_close(
            &actual,
            &expected,
            5.0e-10,
            "CPU ND real independent coordinates",
        );

        let program = ProgramIr::nd_real_convolution(&ir).unwrap();
        let resource = |name: &str| {
            program
                .resources
                .iter()
                .find(|resource| resource.name == name)
                .unwrap()
        };
        assert_eq!(resource("input").elements, COORDINATE_COUNT * tensor_len());
        assert_eq!(
            resource("output").elements,
            COORDINATE_KERNEL_COUNT * COORDINATE_COUNT * tensor_len()
        );
        assert_eq!(
            resource("nd_real_convolution_forward_spectrum").elements,
            COORDINATE_COUNT * compact_len()
        );
        assert_eq!(
            resource("nd_real_convolution_multiplied_spectrum").elements,
            COORDINATE_KERNEL_COUNT * COORDINATE_COUNT * compact_len()
        );
        assert_eq!(
            resource("nd_real_convolution_kernel_spectrum").elements,
            COORDINATE_KERNEL_COUNT * COORDINATE_COUNT * compact_len()
        );
        assert_eq!(
            resource("input").external_layout.unwrap().batch_count,
            COORDINATE_COUNT
        );
        assert_eq!(
            resource("output").external_layout.unwrap().batch_count,
            COORDINATE_KERNEL_COUNT * COORDINATE_COUNT
        );

        let shaders = VulkanGlslBackend.lower_nd_real_convolution(&ir).unwrap();
        assert_eq!(program.passes.len(), shaders.len());
        let multiply_index = program
            .passes
            .iter()
            .position(|pass| pass.name == ir.multiply.name)
            .unwrap();
        let multiply = &shaders[multiply_index].glsl;
        assert!(multiply.contains("kernel_id * 2u + 0u"));
        assert!(multiply.contains("kernel_id * 2u + 1u"));
        assert!(multiply.contains("vkfft_input.data[(0u *"));
        assert!(multiply.contains("vkfft_input.data[(1u *"));
        assert!(!multiply.contains("vkfft_matrix_sum"));
        for shader in shaders {
            assert_eq!(shader.compile_spirv().unwrap().words[0], 0x0723_0203);
        }
    }
}

#[test]
fn nd_real_mixed_storage_matrix_k1_keeps_row_sum_at_compute_precision() {
    let profile = device(Backend::Vulkan, GpuVendor::Nvidia);
    let input = matrix_input64();
    let expected = matrix_expected64();
    for (precision, compute, storage, tolerance) in [
        (
            Precision::F16StorageF32Compute,
            ScalarType::F32,
            ScalarType::F16,
            8.0e-2,
        ),
        (
            Precision::F64ComputeF32Storage,
            ScalarType::F64,
            ScalarType::F32,
            2.0e-5,
        ),
    ] {
        let ir = build_matrix(profile, precision);
        let matrix = ir.matrix_layout.expect("mixed real matrix layout missing");
        assert_eq!(matrix.matrix_size, MATRIX_SIZE);
        assert_eq!(ir.scalar, compute);
        assert_eq!(ir.external_scalar, storage);
        assert_eq!(ir.coordinate_count, MATRIX_SIZE);
        assert_eq!(ir.kernel_count, 1);
        assert_eq!(ir.forward_r2c.batch_count, MATRIX_SIZE);
        assert_eq!(ir.inverse_c2r.batch_count, MATRIX_SIZE);
        assert!(!ir.multiply.independent_coordinates);
        assert_eq!(ir.multiply.matrix_layout, Some(matrix));
        assert!(ir.zero_padding.iter().all(Option::is_none));
        assert!(ir.forward_r2c.input_formatted_copy.is_none());
        assert!(ir.inverse_c2r.output_formatted_copy.is_none());
        assert!(ir.forward_r2c.output_boundary_compute_storage);
        assert!(ir.inverse_c2r.input_boundary_compute_storage);

        let program = ProgramIr::nd_real_convolution(&ir).unwrap();
        let resource = |name: &str| {
            program
                .resources
                .iter()
                .find(|resource| resource.name == name)
                .unwrap()
        };
        assert_eq!(resource("input").scalar, storage);
        assert_eq!(resource("output").scalar, storage);
        for name in [
            "nd_real_convolution_kernel_spectrum",
            "nd_real_convolution_forward_spectrum",
            "nd_real_convolution_multiplied_spectrum",
        ] {
            assert_eq!(resource(name).scalar, compute, "{name}");
        }
        assert_eq!(resource("input").elements, MATRIX_SIZE * tensor_len());
        assert_eq!(resource("output").elements, MATRIX_SIZE * tensor_len());
        assert_eq!(
            resource("nd_real_convolution_kernel_spectrum").elements,
            MATRIX_SIZE * MATRIX_SIZE * compact_len()
        );

        let actual = execute_nd_real_convolution_ir(&ir, &input).unwrap();
        assert_close(
            &actual,
            &expected,
            tolerance,
            "CPU ND real mixed storage matrix K1",
        );
        let shaders = VulkanGlslBackend.lower_nd_real_convolution(&ir).unwrap();
        let multiply_index = program
            .passes
            .iter()
            .position(|pass| pass.name == ir.multiply.name)
            .unwrap();
        for coordinate in 0..MATRIX_SIZE {
            assert!(
                shaders[multiply_index]
                    .glsl
                    .contains(&format!("vkfft_matrix_sum_{coordinate}"))
            );
        }
        for shader in shaders {
            assert_eq!(shader.compile_spirv().unwrap().words[0], 0x0723_0203);
        }
    }
}

#[test]
fn nd_real_matrix_k1_matches_sample51_dense_ownership_and_spatial_oracle() {
    let profile = device(Backend::Vulkan, GpuVendor::Nvidia);
    let input = matrix_input64();
    let expected = matrix_expected64();
    for precision in [Precision::F32, Precision::F64] {
        let ir = build_matrix(profile, precision);
        let matrix = ir.matrix_layout.expect("real matrix layout missing");
        assert_eq!(matrix.matrix_size, MATRIX_SIZE);
        assert!(!matrix.symmetric_kernel);
        assert_eq!(ir.batch_count, 1);
        assert_eq!(ir.coordinate_count, MATRIX_SIZE);
        assert_eq!(ir.kernel_count, 1);
        assert_eq!(ir.forward_r2c.batch_count, MATRIX_SIZE);
        assert_eq!(ir.inverse_c2r.batch_count, MATRIX_SIZE);
        assert!(!ir.multiply.independent_coordinates);
        assert_eq!(ir.multiply.matrix_layout, Some(matrix));
        assert_eq!(ir.multiply.dispatch.x, 1);
        assert_eq!(
            ir.kernel_spectrum().len(),
            MATRIX_SIZE * MATRIX_SIZE * compact_len()
        );

        let actual = execute_nd_real_convolution_ir(&ir, &input).unwrap();
        assert_close(&actual, &expected, 6.0e-10, "CPU ND real matrix K1");

        let program = ProgramIr::nd_real_convolution(&ir).unwrap();
        let resource = |name: &str| {
            program
                .resources
                .iter()
                .find(|resource| resource.name == name)
                .unwrap()
        };
        assert_eq!(resource("input").elements, MATRIX_SIZE * tensor_len());
        assert_eq!(resource("output").elements, MATRIX_SIZE * tensor_len());
        assert_eq!(
            resource("nd_real_convolution_forward_spectrum").elements,
            MATRIX_SIZE * compact_len()
        );
        assert_eq!(
            resource("nd_real_convolution_multiplied_spectrum").elements,
            MATRIX_SIZE * compact_len()
        );
        assert_eq!(
            resource("nd_real_convolution_kernel_spectrum").elements,
            MATRIX_SIZE * MATRIX_SIZE * compact_len()
        );
        assert_eq!(
            resource("input").external_layout.unwrap().batch_count,
            MATRIX_SIZE
        );
        assert_eq!(
            resource("output").external_layout.unwrap().batch_count,
            MATRIX_SIZE
        );

        let shaders = VulkanGlslBackend.lower_nd_real_convolution(&ir).unwrap();
        let multiply_index = program
            .passes
            .iter()
            .position(|pass| pass.name == ir.multiply.name)
            .unwrap();
        let multiply = &shaders[multiply_index].glsl;
        for coordinate in 0..MATRIX_SIZE {
            assert!(multiply.contains(&format!("vkfft_matrix_sum_{coordinate}")));
            assert!(multiply.contains(&format!("vkfft_matrix_input_{coordinate}")));
        }
        assert!(!multiply.contains("kernel_id = batch"));
        for shader in shaders {
            assert_eq!(shader.compile_spirv().unwrap().words[0], 0x0723_0203);
        }
    }
}

#[test]
fn nd_real_mixed_storage_matrix_fanout_keeps_one_forward_and_k_row_sums() {
    let profile = device(Backend::Vulkan, GpuVendor::Nvidia);
    let input = matrix_input64();
    let expected = matrix_multi_expected64();
    for (precision, compute, storage, tolerance) in [
        (
            Precision::F16StorageF32Compute,
            ScalarType::F32,
            ScalarType::F16,
            8.0e-2,
        ),
        (
            Precision::F64ComputeF32Storage,
            ScalarType::F64,
            ScalarType::F32,
            2.0e-5,
        ),
    ] {
        let ir = build_multi_kernel_matrix(profile, precision);
        let matrix = ir
            .matrix_layout
            .expect("mixed real matrix K>1 layout missing");
        assert_eq!(matrix.matrix_size, MATRIX_SIZE);
        assert_eq!(ir.scalar, compute);
        assert_eq!(ir.external_scalar, storage);
        assert_eq!(ir.coordinate_count, MATRIX_SIZE);
        assert_eq!(ir.kernel_count, MATRIX_KERNEL_COUNT);
        assert_eq!(
            ir.output_system_count().unwrap(),
            MATRIX_KERNEL_COUNT * MATRIX_SIZE
        );
        assert_eq!(ir.forward_r2c.batch_count, MATRIX_SIZE);
        assert_eq!(
            ir.inverse_c2r.batch_count,
            MATRIX_KERNEL_COUNT * MATRIX_SIZE
        );
        assert!(ir.zero_padding.iter().all(Option::is_none));
        assert_eq!(ir.multiply.matrix_layout, Some(matrix));
        assert_eq!(ir.multiply.dispatch.x, MATRIX_KERNEL_COUNT as u32);
        assert!(ir.forward_r2c.output_boundary_compute_storage);
        assert!(ir.inverse_c2r.input_boundary_compute_storage);
        assert!(ir.forward_r2c.input_formatted_copy.is_none());
        assert!(ir.inverse_c2r.output_formatted_copy.is_none());

        let program = ProgramIr::nd_real_convolution(&ir).unwrap();
        let resource = |name: &str| {
            program
                .resources
                .iter()
                .find(|resource| resource.name == name)
                .unwrap()
        };
        assert_eq!(resource("input").scalar, storage);
        assert_eq!(resource("output").scalar, storage);
        for name in [
            "nd_real_convolution_kernel_spectrum",
            "nd_real_convolution_forward_spectrum",
            "nd_real_convolution_multiplied_spectrum",
        ] {
            assert_eq!(resource(name).scalar, compute, "{name}");
        }
        assert_eq!(resource("input").elements, MATRIX_SIZE * tensor_len());
        assert_eq!(
            resource("output").elements,
            MATRIX_KERNEL_COUNT * MATRIX_SIZE * tensor_len()
        );
        assert_eq!(
            resource("nd_real_convolution_forward_spectrum").elements,
            MATRIX_SIZE * compact_len()
        );
        assert_eq!(
            resource("nd_real_convolution_multiplied_spectrum").elements,
            MATRIX_KERNEL_COUNT * MATRIX_SIZE * compact_len()
        );
        assert_eq!(
            resource("nd_real_convolution_kernel_spectrum").elements,
            MATRIX_KERNEL_COUNT * MATRIX_SIZE * MATRIX_SIZE * compact_len()
        );

        let actual = execute_nd_real_convolution_ir(&ir, &input).unwrap();
        assert_close(
            &actual,
            &expected,
            tolerance,
            "CPU ND real mixed storage matrix K2 fan-out",
        );
        let shaders = VulkanGlslBackend.lower_nd_real_convolution(&ir).unwrap();
        let multiply_index = program
            .passes
            .iter()
            .position(|pass| pass.name == ir.multiply.name)
            .unwrap();
        assert!(
            shaders[multiply_index]
                .glsl
                .contains("uint kernel_id = batch;")
        );
        for coordinate in 0..MATRIX_SIZE {
            assert!(
                shaders[multiply_index]
                    .glsl
                    .contains(&format!("vkfft_matrix_sum_{coordinate}"))
            );
        }
        for shader in shaders {
            assert_eq!(shader.compile_spirv().unwrap().words[0], 0x0723_0203);
        }
    }
}

#[test]
fn nd_real_matrix_multi_kernel_matches_pinned_orthogonal_ownership() {
    let profile = device(Backend::Vulkan, GpuVendor::Nvidia);
    let input = matrix_input64();
    let expected = matrix_multi_expected64();
    for precision in [Precision::F32, Precision::F64] {
        let ir = build_multi_kernel_matrix(profile, precision);
        let matrix = ir.matrix_layout.expect("real matrix K>1 layout missing");
        assert_eq!(matrix.matrix_size, MATRIX_SIZE);
        assert_eq!(ir.batch_count, 1);
        assert_eq!(ir.coordinate_count, MATRIX_SIZE);
        assert_eq!(ir.kernel_count, MATRIX_KERNEL_COUNT);
        assert_eq!(
            ir.output_system_count().unwrap(),
            MATRIX_KERNEL_COUNT * MATRIX_SIZE
        );
        assert_eq!(ir.forward_r2c.batch_count, MATRIX_SIZE);
        assert_eq!(
            ir.inverse_c2r.batch_count,
            MATRIX_KERNEL_COUNT * MATRIX_SIZE
        );
        assert!(ir.zero_padding.iter().all(Option::is_none));
        assert!(!ir.multiply.independent_coordinates);
        assert_eq!(ir.multiply.matrix_layout, Some(matrix));
        assert_eq!(ir.multiply.dispatch.x, MATRIX_KERNEL_COUNT as u32);
        assert_eq!(
            ir.kernel_spectrum().len(),
            MATRIX_KERNEL_COUNT * MATRIX_SIZE * MATRIX_SIZE * compact_len()
        );

        let actual = execute_nd_real_convolution_ir(&ir, &input).unwrap();
        assert_close(
            &actual,
            &expected,
            8.0e-10,
            "CPU ND real matrix multi-kernel",
        );

        let program = ProgramIr::nd_real_convolution(&ir).unwrap();
        let resource = |name: &str| {
            program
                .resources
                .iter()
                .find(|resource| resource.name == name)
                .unwrap()
        };
        assert_eq!(resource("input").elements, MATRIX_SIZE * tensor_len());
        assert_eq!(
            resource("output").elements,
            MATRIX_KERNEL_COUNT * MATRIX_SIZE * tensor_len()
        );
        assert_eq!(
            resource("nd_real_convolution_forward_spectrum").elements,
            MATRIX_SIZE * compact_len()
        );
        assert_eq!(
            resource("nd_real_convolution_multiplied_spectrum").elements,
            MATRIX_KERNEL_COUNT * MATRIX_SIZE * compact_len()
        );
        assert_eq!(
            resource("nd_real_convolution_kernel_spectrum").elements,
            MATRIX_KERNEL_COUNT * MATRIX_SIZE * MATRIX_SIZE * compact_len()
        );
        assert_eq!(
            resource("input").external_layout.unwrap().batch_count,
            MATRIX_SIZE
        );
        assert_eq!(
            resource("output").external_layout.unwrap().batch_count,
            MATRIX_KERNEL_COUNT * MATRIX_SIZE
        );

        let shaders = VulkanGlslBackend.lower_nd_real_convolution(&ir).unwrap();
        let multiply_index = program
            .passes
            .iter()
            .position(|pass| pass.name == ir.multiply.name)
            .unwrap();
        let multiply = &shaders[multiply_index].glsl;
        assert!(multiply.contains("uint kernel_id = batch;"));
        assert!(multiply.contains(&format!(
            "vkfft_lut.data[((kernel_id * {}u + 0u) * {}u) + i]",
            MATRIX_SIZE * MATRIX_SIZE,
            compact_len()
        )));
        assert!(multiply.contains(&format!(
            "vkfft_output.data[((kernel_id * {}u + 0u) * {}u) + i]",
            MATRIX_SIZE,
            compact_len()
        )));
        for shader in shaders {
            assert_eq!(shader.compile_spirv().unwrap().words[0], 0x0723_0203);
        }
    }
}

#[test]
fn nd_real_mixed_storage_matrix_fanout_spatial_padding_masks_k_matrix_outputs() {
    let profile = device(Backend::Vulkan, GpuVendor::Nvidia);
    let input = matrix_padded_probe_input64();
    let expected = matrix_multi_padded_expected64(&input);
    for (precision, compute, storage, tolerance) in [
        (
            Precision::F16StorageF32Compute,
            ScalarType::F32,
            ScalarType::F16,
            8.0e-2,
        ),
        (
            Precision::F64ComputeF32Storage,
            ScalarType::F64,
            ScalarType::F32,
            2.0e-5,
        ),
    ] {
        let ir = build_padded_multi_kernel_matrix(profile, precision);
        let matrix = ir
            .matrix_layout
            .expect("mixed padded real matrix K>1 layout missing");
        assert_eq!(matrix.matrix_size, MATRIX_SIZE);
        assert_eq!(ir.scalar, compute);
        assert_eq!(ir.external_scalar, storage);
        assert_eq!(ir.coordinate_count, MATRIX_SIZE);
        assert_eq!(ir.kernel_count, MATRIX_KERNEL_COUNT);
        assert_eq!(
            ir.output_system_count().unwrap(),
            MATRIX_KERNEL_COUNT * MATRIX_SIZE
        );
        assert_eq!(
            ir.forward_r2c.zero_pad_pass.as_ref().unwrap().batch_count,
            MATRIX_SIZE
        );
        assert_eq!(
            ir.inverse_c2r.zero_pad_pass.as_ref().unwrap().batch_count,
            MATRIX_KERNEL_COUNT * MATRIX_SIZE
        );
        assert_eq!(ir.multiply.matrix_layout, Some(matrix));
        assert_eq!(ir.multiply.dispatch.x, MATRIX_KERNEL_COUNT as u32);
        assert!(ir.forward_r2c.output_boundary_compute_storage);
        assert!(ir.inverse_c2r.input_boundary_compute_storage);
        assert!(ir.forward_r2c.input_formatted_copy.is_none());
        assert!(ir.inverse_c2r.output_formatted_copy.is_none());

        let program = ProgramIr::nd_real_convolution(&ir).unwrap();
        let resource = |name: &str| {
            program
                .resources
                .iter()
                .find(|resource| resource.name == name)
                .unwrap()
        };
        assert_eq!(resource("input").scalar, storage);
        assert_eq!(resource("output").scalar, storage);
        for name in [
            "nd_real_convolution_kernel_spectrum",
            "nd_real_convolution_forward_spectrum",
            "nd_real_convolution_multiplied_spectrum",
        ] {
            assert_eq!(resource(name).scalar, compute, "{name}");
        }
        assert_eq!(resource("input").elements, MATRIX_SIZE * tensor_len());
        assert_eq!(
            resource("output").elements,
            MATRIX_KERNEL_COUNT * MATRIX_SIZE * tensor_len()
        );

        let actual = execute_nd_real_convolution_ir(&ir, &input).unwrap();
        assert_close(
            &actual,
            &expected,
            tolerance,
            "CPU ND real mixed storage matrix K2 spatial padding",
        );
        for shader in VulkanGlslBackend.lower_nd_real_convolution(&ir).unwrap() {
            assert_eq!(shader.compile_spirv().unwrap().words[0], 0x0723_0203);
        }
    }
}

#[test]
fn nd_real_mixed_storage_policy_padding_matrix_fanout_keeps_policy_in_compute_stage() {
    let profile = device(Backend::Vulkan, GpuVendor::Nvidia);
    let input = matrix_padded_probe_input64();
    let expected = matrix_policy_padded_expected64(&input);
    for (precision, compute, storage, tolerance) in [
        (
            Precision::F16StorageF32Compute,
            ScalarType::F32,
            ScalarType::F16,
            1.0e-1,
        ),
        (
            Precision::F64ComputeF32Storage,
            ScalarType::F64,
            ScalarType::F32,
            2.0e-5,
        ),
    ] {
        let ir = build_policy_padded_multi_kernel_matrix(profile, precision);
        assert_eq!(ir.scalar, compute);
        assert_eq!(ir.external_scalar, storage);
        assert_eq!(ir.coordinate_count, MATRIX_SIZE);
        assert_eq!(ir.kernel_count, MATRIX_KERNEL_COUNT);
        assert_eq!(
            ir.multiply.policy.conjugation,
            ConvolutionConjugation::Sequence
        );
        assert!(ir.multiply.policy.cross_power_spectrum_normalization);
        assert_eq!(
            ir.forward_r2c.zero_pad_pass.as_ref().unwrap().batch_count,
            MATRIX_SIZE
        );
        assert_eq!(
            ir.inverse_c2r.zero_pad_pass.as_ref().unwrap().batch_count,
            MATRIX_KERNEL_COUNT * MATRIX_SIZE
        );
        assert!(ir.forward_r2c.output_boundary_compute_storage);
        assert!(ir.inverse_c2r.input_boundary_compute_storage);

        let program = ProgramIr::nd_real_convolution(&ir).unwrap();
        let resource = |name: &str| {
            program
                .resources
                .iter()
                .find(|resource| resource.name == name)
                .unwrap()
        };
        assert_eq!(resource("input").scalar, storage);
        assert_eq!(resource("output").scalar, storage);
        for name in [
            "nd_real_convolution_kernel_spectrum",
            "nd_real_convolution_forward_spectrum",
            "nd_real_convolution_multiplied_spectrum",
        ] {
            assert_eq!(resource(name).scalar, compute, "{name}");
        }

        let actual = execute_nd_real_convolution_ir(&ir, &input).unwrap();
        assert_close(
            &actual,
            &expected,
            tolerance,
            "CPU ND real mixed policy/padding/matrix fanout",
        );
        let shaders = VulkanGlslBackend.lower_nd_real_convolution(&ir).unwrap();
        let multiply_index = program
            .passes
            .iter()
            .position(|pass| pass.name == ir.multiply.name)
            .unwrap();
        let multiply = &shaders[multiply_index].glsl;
        assert!(multiply.contains("inversesqrt"));
        assert!(multiply.contains("vec2((vkfft_input.data["));
        assert!(multiply.contains("-(vkfft_input.data["));
        for shader in shaders {
            assert_eq!(shader.compile_spirv().unwrap().words[0], 0x0723_0203);
        }
    }
}

#[test]
fn nd_real_policy_padding_fanout_orthogonality_matches_independent_frequency_oracle() {
    let profile = device(Backend::Vulkan, GpuVendor::Nvidia);
    let input = matrix_padded_probe_input64();
    let expected = matrix_policy_padded_expected64(&input);
    for precision in [Precision::F32, Precision::F64] {
        let ir = build_policy_padded_multi_kernel_matrix(profile, precision);
        assert_eq!(ir.coordinate_count, MATRIX_SIZE);
        assert_eq!(ir.kernel_count, MATRIX_KERNEL_COUNT);
        assert_eq!(
            ir.output_system_count().unwrap(),
            MATRIX_KERNEL_COUNT * MATRIX_SIZE
        );
        assert_eq!(
            ir.multiply.policy.conjugation,
            ConvolutionConjugation::Sequence
        );
        assert!(ir.multiply.policy.cross_power_spectrum_normalization);
        assert_eq!(
            ir.forward_r2c.zero_pad_pass.as_ref().unwrap().batch_count,
            MATRIX_SIZE
        );
        assert_eq!(
            ir.inverse_c2r.zero_pad_pass.as_ref().unwrap().batch_count,
            MATRIX_KERNEL_COUNT * MATRIX_SIZE
        );
        let actual = execute_nd_real_convolution_ir(&ir, &input).unwrap();
        assert_close(
            &actual,
            &expected,
            1.5e-9,
            "CPU ND real policy/padding/matrix fanout",
        );
        let program = ProgramIr::nd_real_convolution(&ir).unwrap();
        let shaders = VulkanGlslBackend.lower_nd_real_convolution(&ir).unwrap();
        let multiply_index = program
            .passes
            .iter()
            .position(|pass| pass.name == ir.multiply.name)
            .unwrap();
        assert!(shaders[multiply_index].glsl.contains("inversesqrt"));
        assert!(
            shaders[multiply_index]
                .glsl
                .contains("vec2((vkfft_input.data[")
        );
        assert!(shaders[multiply_index].glsl.contains("-(vkfft_input.data["));
        for shader in shaders {
            assert_eq!(shader.compile_spirv().unwrap().words[0], 0x0723_0203);
        }
    }

    for (ir, forward_batch, inverse_batch) in [
        (
            NdRealConvolutionIr::build_from_spectrum(
                FftConfig::new(DIMENSIONS.to_vec())
                    .with_transform(TransformKind::RealToComplex)
                    .with_convolution(true)
                    .with_convolution_kernel_count(KERNEL_COUNT)
                    .with_convolution_conjugation(ConvolutionConjugation::Sequence)
                    .with_cross_power_spectrum_normalization(true)
                    .with_zero_padding(0, 2, 3)
                    .unwrap(),
                kernel_spectrum(KERNEL_COUNT),
                profile,
            )
            .unwrap(),
            1,
            KERNEL_COUNT,
        ),
        (
            NdRealConvolutionIr::build_from_spectrum(
                FftConfig::new(DIMENSIONS.to_vec())
                    .with_transform(TransformKind::RealToComplex)
                    .with_convolution(true)
                    .with_coordinate_features(COORDINATE_COUNT)
                    .with_convolution_kernel_count(COORDINATE_KERNEL_COUNT)
                    .with_convolution_conjugation(ConvolutionConjugation::Sequence)
                    .with_cross_power_spectrum_normalization(true)
                    .with_zero_padding(0, 2, 3)
                    .unwrap(),
                coordinate_kernel_spectrum(),
                profile,
            )
            .unwrap(),
            COORDINATE_COUNT,
            COORDINATE_COUNT * COORDINATE_KERNEL_COUNT,
        ),
    ] {
        assert_eq!(
            ir.forward_r2c.zero_pad_pass.as_ref().unwrap().batch_count,
            forward_batch
        );
        assert_eq!(
            ir.inverse_c2r.zero_pad_pass.as_ref().unwrap().batch_count,
            inverse_batch
        );
        assert_eq!(
            ir.multiply.policy.conjugation,
            ConvolutionConjugation::Sequence
        );
        assert!(ir.multiply.policy.cross_power_spectrum_normalization);
    }
}

#[test]
fn nd_real_mixed_storage_independent_coordinates_fanout_keeps_c_forward_and_kc_outputs() {
    let profile = device(Backend::Vulkan, GpuVendor::Nvidia);
    let input = coordinate_input64();
    let expected = coordinate_expected64();
    for (precision, compute, storage, tolerance) in [
        (
            Precision::F16StorageF32Compute,
            ScalarType::F32,
            ScalarType::F16,
            8.0e-2,
        ),
        (
            Precision::F64ComputeF32Storage,
            ScalarType::F64,
            ScalarType::F32,
            2.0e-5,
        ),
    ] {
        let ir = build_coordinates(profile, precision);
        assert_eq!(ir.scalar, compute);
        assert_eq!(ir.external_scalar, storage);
        assert_eq!(ir.batch_count, 1);
        assert_eq!(ir.coordinate_count, COORDINATE_COUNT);
        assert_eq!(ir.kernel_count, COORDINATE_KERNEL_COUNT);
        assert_eq!(ir.forward_r2c.batch_count, COORDINATE_COUNT);
        assert_eq!(
            ir.inverse_c2r.batch_count,
            COORDINATE_COUNT * COORDINATE_KERNEL_COUNT
        );
        assert_eq!(
            ir.output_system_count().unwrap(),
            COORDINATE_COUNT * COORDINATE_KERNEL_COUNT
        );
        assert!(ir.multiply.independent_coordinates);
        assert!(ir.multiply.matrix_layout.is_none());
        assert_eq!(ir.multiply.coordinate_count, COORDINATE_COUNT);
        assert_eq!(ir.multiply.kernel_count, COORDINATE_KERNEL_COUNT);
        assert!(ir.zero_padding.iter().all(Option::is_none));
        assert!(ir.forward_r2c.input_formatted_copy.is_none());
        assert!(ir.inverse_c2r.output_formatted_copy.is_none());
        assert!(ir.forward_r2c.output_boundary_compute_storage);
        assert!(ir.inverse_c2r.input_boundary_compute_storage);

        let program = ProgramIr::nd_real_convolution(&ir).unwrap();
        let resource = |name: &str| {
            program
                .resources
                .iter()
                .find(|resource| resource.name == name)
                .unwrap()
        };
        assert_eq!(resource("input").scalar, storage);
        assert_eq!(resource("output").scalar, storage);
        assert_eq!(resource("input").elements, COORDINATE_COUNT * tensor_len());
        assert_eq!(
            resource("output").elements,
            COORDINATE_KERNEL_COUNT * COORDINATE_COUNT * tensor_len()
        );
        assert_eq!(
            resource("input")
                .external_layout
                .as_ref()
                .unwrap()
                .batch_count,
            COORDINATE_COUNT
        );
        assert_eq!(
            resource("output")
                .external_layout
                .as_ref()
                .unwrap()
                .batch_count,
            COORDINATE_KERNEL_COUNT * COORDINATE_COUNT
        );
        assert_eq!(
            resource("nd_real_convolution_forward_spectrum").elements,
            COORDINATE_COUNT * compact_len()
        );
        assert_eq!(
            resource("nd_real_convolution_multiplied_spectrum").elements,
            COORDINATE_KERNEL_COUNT * COORDINATE_COUNT * compact_len()
        );
        assert_eq!(
            resource("nd_real_convolution_kernel_spectrum").elements,
            COORDINATE_KERNEL_COUNT * COORDINATE_COUNT * compact_len()
        );
        for name in [
            "nd_real_convolution_kernel_spectrum",
            "nd_real_convolution_forward_spectrum",
            "nd_real_convolution_multiplied_spectrum",
        ] {
            assert_eq!(resource(name).scalar, compute, "{name}");
        }

        let actual = execute_nd_real_convolution_ir(&ir, &input).unwrap();
        assert_close(
            &actual,
            &expected,
            tolerance,
            "CPU ND real mixed storage independent coordinates C2/K2",
        );
        let shaders = VulkanGlslBackend.lower_nd_real_convolution(&ir).unwrap();
        let multiply_index = program
            .passes
            .iter()
            .position(|pass| pass.name == ir.multiply.name)
            .unwrap();
        let multiply = &shaders[multiply_index].glsl;
        assert!(multiply.contains("kernel_id * 2u + 0u"));
        assert!(multiply.contains("kernel_id * 2u + 1u"));
        assert!(!multiply.contains("vkfft_matrix_sum"));
        for shader in shaders {
            assert_eq!(shader.compile_spirv().unwrap().words[0], 0x0723_0203);
        }
    }
}

#[test]
fn nd_real_mixed_storage_independent_coordinates_spatial_padding_masks_c_inputs_and_kc_outputs() {
    let profile = device(Backend::Vulkan, GpuVendor::Nvidia);
    let input = coordinate_padded_probe_input64();
    let expected = coordinate_padded_expected64(&input);
    for (precision, compute, storage, tolerance) in [
        (
            Precision::F16StorageF32Compute,
            ScalarType::F32,
            ScalarType::F16,
            8.0e-2,
        ),
        (
            Precision::F64ComputeF32Storage,
            ScalarType::F64,
            ScalarType::F32,
            2.0e-5,
        ),
    ] {
        let ir = build_padded_coordinates(profile, precision);
        assert_eq!(ir.scalar, compute);
        assert_eq!(ir.external_scalar, storage);
        assert_eq!(ir.coordinate_count, COORDINATE_COUNT);
        assert_eq!(ir.kernel_count, COORDINATE_KERNEL_COUNT);
        assert_eq!(
            ir.forward_r2c.zero_pad_pass.as_ref().unwrap().batch_count,
            COORDINATE_COUNT
        );
        assert_eq!(
            ir.inverse_c2r.zero_pad_pass.as_ref().unwrap().batch_count,
            COORDINATE_COUNT * COORDINATE_KERNEL_COUNT
        );
        assert!(ir.forward_r2c.output_boundary_compute_storage);
        assert!(ir.inverse_c2r.input_boundary_compute_storage);
        assert!(ir.forward_r2c.input_formatted_copy.is_none());
        assert!(ir.inverse_c2r.output_formatted_copy.is_none());

        let program = ProgramIr::nd_real_convolution(&ir).unwrap();
        let resource = |name: &str| {
            program
                .resources
                .iter()
                .find(|resource| resource.name == name)
                .unwrap()
        };
        assert_eq!(resource("input").scalar, storage);
        assert_eq!(resource("output").scalar, storage);
        for name in [
            "nd_real_convolution_kernel_spectrum",
            "nd_real_convolution_forward_spectrum",
            "nd_real_convolution_multiplied_spectrum",
        ] {
            assert_eq!(resource(name).scalar, compute, "{name}");
        }
        assert_eq!(resource("input").elements, COORDINATE_COUNT * tensor_len());
        assert_eq!(
            resource("output").elements,
            COORDINATE_KERNEL_COUNT * COORDINATE_COUNT * tensor_len()
        );

        let actual = execute_nd_real_convolution_ir(&ir, &input).unwrap();
        assert_close(
            &actual,
            &expected,
            tolerance,
            "CPU ND real mixed storage independent coordinates C2/K2 spatial padding",
        );
        for shader in VulkanGlslBackend.lower_nd_real_convolution(&ir).unwrap() {
            assert_eq!(shader.compile_spirv().unwrap().words[0], 0x0723_0203);
        }
    }
}

#[test]
fn nd_real_mixed_storage_matrix_k1_spatial_padding_masks_all_matrix_edges() {
    let profile = device(Backend::Vulkan, GpuVendor::Nvidia);
    let input = matrix_padded_probe_input64();
    let expected = matrix_padded_expected64(&input);
    for (precision, compute, storage, tolerance) in [
        (
            Precision::F16StorageF32Compute,
            ScalarType::F32,
            ScalarType::F16,
            8.0e-2,
        ),
        (
            Precision::F64ComputeF32Storage,
            ScalarType::F64,
            ScalarType::F32,
            2.0e-5,
        ),
    ] {
        let ir = build_padded_matrix(profile, precision);
        let matrix = ir
            .matrix_layout
            .expect("mixed padded real matrix layout missing");
        assert_eq!(matrix.matrix_size, MATRIX_SIZE);
        assert_eq!(ir.scalar, compute);
        assert_eq!(ir.external_scalar, storage);
        assert_eq!(ir.coordinate_count, MATRIX_SIZE);
        assert_eq!(ir.kernel_count, 1);
        assert_eq!(
            ir.forward_r2c.zero_pad_pass.as_ref().unwrap().batch_count,
            MATRIX_SIZE
        );
        assert_eq!(
            ir.inverse_c2r.zero_pad_pass.as_ref().unwrap().batch_count,
            MATRIX_SIZE
        );
        assert_eq!(ir.multiply.matrix_layout, Some(matrix));
        assert!(!ir.multiply.independent_coordinates);
        assert!(ir.forward_r2c.output_boundary_compute_storage);
        assert!(ir.inverse_c2r.input_boundary_compute_storage);
        assert!(ir.forward_r2c.input_formatted_copy.is_none());
        assert!(ir.inverse_c2r.output_formatted_copy.is_none());

        let program = ProgramIr::nd_real_convolution(&ir).unwrap();
        let resource = |name: &str| {
            program
                .resources
                .iter()
                .find(|resource| resource.name == name)
                .unwrap()
        };
        assert_eq!(resource("input").scalar, storage);
        assert_eq!(resource("output").scalar, storage);
        for name in [
            "nd_real_convolution_kernel_spectrum",
            "nd_real_convolution_forward_spectrum",
            "nd_real_convolution_multiplied_spectrum",
        ] {
            assert_eq!(resource(name).scalar, compute, "{name}");
        }
        assert_eq!(resource("input").elements, MATRIX_SIZE * tensor_len());
        assert_eq!(resource("output").elements, MATRIX_SIZE * tensor_len());

        let actual = execute_nd_real_convolution_ir(&ir, &input).unwrap();
        assert_close(
            &actual,
            &expected,
            tolerance,
            "CPU ND real mixed storage matrix K1 spatial padding",
        );
        for shader in VulkanGlslBackend.lower_nd_real_convolution(&ir).unwrap() {
            assert_eq!(shader.compile_spirv().unwrap().words[0], 0x0723_0203);
        }
    }
}

#[test]
fn nd_real_matrix_k1_spatial_zero_padding_matches_sample51_boundary_ownership() {
    let profile = device(Backend::Vulkan, GpuVendor::Nvidia);
    let input = matrix_padded_probe_input64();
    let expected = matrix_padded_expected64(&input);
    for precision in [Precision::F32, Precision::F64] {
        let ir = build_padded_matrix(profile, precision);
        assert_eq!(ir.matrix_layout.unwrap().matrix_size, MATRIX_SIZE);
        assert_eq!(ir.zero_padding.len(), DIMENSIONS.len());
        let axis0 = ir.zero_padding[0].unwrap();
        let axis1 = ir.zero_padding[1].unwrap();
        assert_eq!((axis0.left, axis0.right), (2, 3));
        assert_eq!((axis1.left, axis1.right), (2, 4));
        let forward_zero = ir
            .forward_r2c
            .zero_pad_pass
            .as_ref()
            .expect("padded Real matrix R2C must own the input spatial mask");
        let inverse_zero = ir
            .inverse_c2r
            .zero_pad_pass
            .as_ref()
            .expect("padded Real matrix C2R must own the output spatial mask");
        assert_eq!(forward_zero.ranges, ir.zero_padding);
        assert_eq!(inverse_zero.ranges, ir.zero_padding);
        assert_eq!(forward_zero.batch_count, MATRIX_SIZE);
        assert_eq!(inverse_zero.batch_count, MATRIX_SIZE);

        let actual = execute_nd_real_convolution_ir(&ir, &input).unwrap();
        assert_close(
            &actual,
            &expected,
            8.0e-10,
            "CPU ND real matrix K1 spatial zero padding",
        );
        for coordinate in 0..MATRIX_SIZE {
            let base = coordinate * tensor_len();
            for index in 0..tensor_len() {
                if matrix_padding_contains(index) {
                    assert_eq!(actual[base + index], 0.0);
                }
            }
        }

        let program = ProgramIr::nd_real_convolution(&ir).unwrap();
        let shaders = VulkanGlslBackend.lower_nd_real_convolution(&ir).unwrap();
        assert_eq!(program.passes.len(), shaders.len());
        let multiply_index = program
            .passes
            .iter()
            .position(|pass| pass.name == ir.multiply.name)
            .unwrap();
        let forward_zero_index = program.passes[..multiply_index]
            .iter()
            .position(|pass| pass.name.contains("zero_pad"))
            .expect("R2C spatial zero-pad pass must precede the matrix midpoint");
        let inverse_zero_index = program.passes[multiply_index + 1..]
            .iter()
            .rposition(|pass| pass.name.contains("zero_pad"))
            .map(|index| multiply_index + 1 + index)
            .expect("C2R spatial zero-pad pass must follow the matrix midpoint");
        assert!(forward_zero_index < multiply_index);
        assert!(inverse_zero_index > multiply_index);
        for zero_index in [forward_zero_index, inverse_zero_index] {
            assert!(
                shaders[zero_index]
                    .glsl
                    .contains("generated by vkfft-rs from typed NdZeroPadPassIr")
            );
        }
        for shader in shaders {
            assert_eq!(shader.compile_spirv().unwrap().words[0], 0x0723_0203);
        }
    }
}

#[cfg(any(feature = "cuda-runtime", feature = "opencl-runtime"))]
fn run_native<R: vkfft_rs::backend::NativeRuntime>(runtime: &R) {
    let expected = expected64(KERNEL_COUNT);
    let input64 = input64();
    let input32 = input64
        .iter()
        .map(|value| *value as f32)
        .collect::<Vec<_>>();
    let f32_ir = build(runtime.device_profile(), Precision::F32, KERNEL_COUNT);
    let actual32 = runtime
        .execute_nd_real_convolution_f32(&f32_ir, &input32)
        .unwrap()
        .into_iter()
        .map(f64::from)
        .collect::<Vec<_>>();
    assert_close(&actual32, &expected, 1.5e-3, runtime.device_name());

    let mixed_expected = expected64(1);
    let f16_storage_ir = build_mixed(runtime.device_profile(), Precision::F16StorageF32Compute);
    let f16_storage_actual = runtime
        .execute_nd_real_convolution_f32(&f16_storage_ir, &input32)
        .unwrap()
        .into_iter()
        .map(f64::from)
        .collect::<Vec<_>>();
    assert_close(
        &f16_storage_actual,
        &mixed_expected,
        8.0e-2,
        runtime.device_name(),
    );

    let f16_storage_fanout_ir = build(
        runtime.device_profile(),
        Precision::F16StorageF32Compute,
        KERNEL_COUNT,
    );
    let f16_storage_fanout_actual = runtime
        .execute_nd_real_convolution_f32(&f16_storage_fanout_ir, &input32)
        .unwrap()
        .into_iter()
        .map(f64::from)
        .collect::<Vec<_>>();
    assert_close(
        &f16_storage_fanout_actual,
        &expected,
        8.0e-2,
        runtime.device_name(),
    );

    let mixed_padded_input64 = formatted_padded_input64();
    let mixed_padded_expected = formatted_padded_expected64(&mixed_padded_input64);
    let mixed_padded_input32 = mixed_padded_input64
        .iter()
        .map(|value| *value as f32)
        .collect::<Vec<_>>();
    let f16_storage_padded_ir =
        build_mixed_padded(runtime.device_profile(), Precision::F16StorageF32Compute);
    let f16_storage_padded_actual = runtime
        .execute_nd_real_convolution_f32(&f16_storage_padded_ir, &mixed_padded_input32)
        .unwrap()
        .into_iter()
        .map(f64::from)
        .collect::<Vec<_>>();
    assert_close(
        &f16_storage_padded_actual,
        &mixed_padded_expected,
        8.0e-2,
        runtime.device_name(),
    );

    let mixed_padded_fanout_expected = padded_fanout_expected64(&mixed_padded_input64);
    let f16_storage_padded_fanout_ir =
        build_mixed_padded_fanout(runtime.device_profile(), Precision::F16StorageF32Compute);
    let f16_storage_padded_fanout_actual = runtime
        .execute_nd_real_convolution_f32(&f16_storage_padded_fanout_ir, &mixed_padded_input32)
        .unwrap()
        .into_iter()
        .map(f64::from)
        .collect::<Vec<_>>();
    assert_close(
        &f16_storage_padded_fanout_actual,
        &mixed_padded_fanout_expected,
        8.0e-2,
        runtime.device_name(),
    );

    let formatted_expected = expected64(1);
    let formatted_f32 = build_formatted(runtime.device_profile(), Precision::F32);
    let formatted_actual32 = runtime
        .execute_nd_real_convolution_f32(&formatted_f32, &input32)
        .unwrap()
        .into_iter()
        .map(f64::from)
        .collect::<Vec<_>>();
    assert_close(
        &formatted_actual32,
        &formatted_expected,
        1.5e-3,
        runtime.device_name(),
    );

    let mixed_formatted_f16 =
        build_formatted(runtime.device_profile(), Precision::F16StorageF32Compute);
    let mixed_formatted_f16_actual = runtime
        .execute_nd_real_convolution_f32(&mixed_formatted_f16, &input32)
        .unwrap()
        .into_iter()
        .map(f64::from)
        .collect::<Vec<_>>();
    assert_close(
        &mixed_formatted_f16_actual,
        &formatted_expected,
        8.0e-2,
        runtime.device_name(),
    );

    let formatted_padded_input64 = formatted_padded_input64();
    let formatted_padded_expected = formatted_padded_expected64(&formatted_padded_input64);
    let formatted_padded_input32 = formatted_padded_input64
        .iter()
        .map(|value| *value as f32)
        .collect::<Vec<_>>();
    let formatted_padded_f32 = build_formatted_padded(runtime.device_profile(), Precision::F32);
    let formatted_padded_actual32 = runtime
        .execute_nd_real_convolution_f32(&formatted_padded_f32, &formatted_padded_input32)
        .unwrap()
        .into_iter()
        .map(f64::from)
        .collect::<Vec<_>>();
    assert_close(
        &formatted_padded_actual32,
        &formatted_padded_expected,
        2.0e-3,
        runtime.device_name(),
    );

    let mixed_formatted_padded_f16 =
        build_formatted_padded(runtime.device_profile(), Precision::F16StorageF32Compute);
    let mixed_formatted_padded_f16_actual = runtime
        .execute_nd_real_convolution_f32(&mixed_formatted_padded_f16, &formatted_padded_input32)
        .unwrap()
        .into_iter()
        .map(f64::from)
        .collect::<Vec<_>>();
    assert_close(
        &mixed_formatted_padded_f16_actual,
        &formatted_padded_expected,
        8.0e-2,
        runtime.device_name(),
    );

    let coordinate_expected = coordinate_expected64();
    let coordinate_input64 = coordinate_input64();
    let coordinate_input32 = coordinate_input64
        .iter()
        .map(|value| *value as f32)
        .collect::<Vec<_>>();
    let coordinate_f32_ir = build_coordinates(runtime.device_profile(), Precision::F32);
    let coordinate_actual32 = runtime
        .execute_nd_real_convolution_f32(&coordinate_f32_ir, &coordinate_input32)
        .unwrap()
        .into_iter()
        .map(f64::from)
        .collect::<Vec<_>>();
    assert_close(
        &coordinate_actual32,
        &coordinate_expected,
        1.8e-3,
        runtime.device_name(),
    );
    let coordinate_f16_ir =
        build_coordinates(runtime.device_profile(), Precision::F16StorageF32Compute);
    let coordinate_f16_actual = runtime
        .execute_nd_real_convolution_f32(&coordinate_f16_ir, &coordinate_input32)
        .unwrap()
        .into_iter()
        .map(f64::from)
        .collect::<Vec<_>>();
    assert_close(
        &coordinate_f16_actual,
        &coordinate_expected,
        8.0e-2,
        runtime.device_name(),
    );
    let coordinate_padded_input64 = coordinate_padded_probe_input64();
    let coordinate_padded_expected = coordinate_padded_expected64(&coordinate_padded_input64);
    let coordinate_padded_input32 = coordinate_padded_input64
        .iter()
        .map(|value| *value as f32)
        .collect::<Vec<_>>();
    let coordinate_padded_f16_ir =
        build_padded_coordinates(runtime.device_profile(), Precision::F16StorageF32Compute);
    let coordinate_padded_f16_actual = runtime
        .execute_nd_real_convolution_f32(&coordinate_padded_f16_ir, &coordinate_padded_input32)
        .unwrap()
        .into_iter()
        .map(f64::from)
        .collect::<Vec<_>>();
    assert_close(
        &coordinate_padded_f16_actual,
        &coordinate_padded_expected,
        8.0e-2,
        runtime.device_name(),
    );

    let matrix_expected = matrix_expected64();
    let matrix_input64 = matrix_input64();
    let matrix_input32 = matrix_input64
        .iter()
        .map(|value| *value as f32)
        .collect::<Vec<_>>();
    let matrix_f32_ir = build_matrix(runtime.device_profile(), Precision::F32);
    let matrix_actual32 = runtime
        .execute_nd_real_convolution_f32(&matrix_f32_ir, &matrix_input32)
        .unwrap()
        .into_iter()
        .map(f64::from)
        .collect::<Vec<_>>();
    assert_close(
        &matrix_actual32,
        &matrix_expected,
        2.5e-3,
        runtime.device_name(),
    );
    let matrix_f16_ir = build_matrix(runtime.device_profile(), Precision::F16StorageF32Compute);
    let matrix_f16_actual = runtime
        .execute_nd_real_convolution_f32(&matrix_f16_ir, &matrix_input32)
        .unwrap()
        .into_iter()
        .map(f64::from)
        .collect::<Vec<_>>();
    assert_close(
        &matrix_f16_actual,
        &matrix_expected,
        8.0e-2,
        runtime.device_name(),
    );

    let matrix_multi_expected = matrix_multi_expected64();
    let matrix_multi_f32_ir = build_multi_kernel_matrix(runtime.device_profile(), Precision::F32);
    let matrix_multi_actual32 = runtime
        .execute_nd_real_convolution_f32(&matrix_multi_f32_ir, &matrix_input32)
        .unwrap()
        .into_iter()
        .map(f64::from)
        .collect::<Vec<_>>();
    assert_close(
        &matrix_multi_actual32,
        &matrix_multi_expected,
        3.0e-3,
        runtime.device_name(),
    );
    let matrix_multi_f16_ir =
        build_multi_kernel_matrix(runtime.device_profile(), Precision::F16StorageF32Compute);
    let matrix_multi_f16_actual = runtime
        .execute_nd_real_convolution_f32(&matrix_multi_f16_ir, &matrix_input32)
        .unwrap()
        .into_iter()
        .map(f64::from)
        .collect::<Vec<_>>();
    assert_close(
        &matrix_multi_f16_actual,
        &matrix_multi_expected,
        8.0e-2,
        runtime.device_name(),
    );

    let padded_matrix_input64 = matrix_padded_probe_input64();
    let padded_matrix_expected = matrix_padded_expected64(&padded_matrix_input64);
    let padded_matrix_input32 = padded_matrix_input64
        .iter()
        .map(|value| *value as f32)
        .collect::<Vec<_>>();
    let padded_matrix_f32_ir = build_padded_matrix(runtime.device_profile(), Precision::F32);
    let padded_matrix_actual32 = runtime
        .execute_nd_real_convolution_f32(&padded_matrix_f32_ir, &padded_matrix_input32)
        .unwrap()
        .into_iter()
        .map(f64::from)
        .collect::<Vec<_>>();
    assert_close(
        &padded_matrix_actual32,
        &padded_matrix_expected,
        3.0e-3,
        runtime.device_name(),
    );
    let padded_matrix_f16_ir =
        build_padded_matrix(runtime.device_profile(), Precision::F16StorageF32Compute);
    let padded_matrix_f16_actual = runtime
        .execute_nd_real_convolution_f32(&padded_matrix_f16_ir, &padded_matrix_input32)
        .unwrap()
        .into_iter()
        .map(f64::from)
        .collect::<Vec<_>>();
    assert_close(
        &padded_matrix_f16_actual,
        &padded_matrix_expected,
        8.0e-2,
        runtime.device_name(),
    );
    let padded_matrix_multi_expected = matrix_multi_padded_expected64(&padded_matrix_input64);
    let padded_matrix_multi_f16_ir =
        build_padded_multi_kernel_matrix(runtime.device_profile(), Precision::F16StorageF32Compute);
    let padded_matrix_multi_f16_actual = runtime
        .execute_nd_real_convolution_f32(&padded_matrix_multi_f16_ir, &padded_matrix_input32)
        .unwrap()
        .into_iter()
        .map(f64::from)
        .collect::<Vec<_>>();
    assert_close(
        &padded_matrix_multi_f16_actual,
        &padded_matrix_multi_expected,
        8.0e-2,
        runtime.device_name(),
    );

    if runtime.device_profile().supports_f64 {
        let f64_ir = build(runtime.device_profile(), Precision::F64, KERNEL_COUNT);
        let actual64 = runtime
            .execute_nd_real_convolution_f64(&f64_ir, &input64)
            .unwrap();
        assert_close(&actual64, &expected, 1.0e-9, runtime.device_name());
        let f32_storage_ir = build_mixed(runtime.device_profile(), Precision::F64ComputeF32Storage);
        let f32_storage_actual = runtime
            .execute_nd_real_convolution_f64(&f32_storage_ir, &input64)
            .unwrap();
        assert_close(
            &f32_storage_actual,
            &mixed_expected,
            2.0e-5,
            runtime.device_name(),
        );
        let f32_storage_fanout_ir = build(
            runtime.device_profile(),
            Precision::F64ComputeF32Storage,
            KERNEL_COUNT,
        );
        let f32_storage_fanout_actual = runtime
            .execute_nd_real_convolution_f64(&f32_storage_fanout_ir, &input64)
            .unwrap();
        assert_close(
            &f32_storage_fanout_actual,
            &expected,
            2.0e-5,
            runtime.device_name(),
        );
        let f32_storage_padded_ir =
            build_mixed_padded(runtime.device_profile(), Precision::F64ComputeF32Storage);
        let f32_storage_padded_actual = runtime
            .execute_nd_real_convolution_f64(&f32_storage_padded_ir, &mixed_padded_input64)
            .unwrap();
        assert_close(
            &f32_storage_padded_actual,
            &mixed_padded_expected,
            2.0e-5,
            runtime.device_name(),
        );
        let f32_storage_padded_fanout_ir =
            build_mixed_padded_fanout(runtime.device_profile(), Precision::F64ComputeF32Storage);
        let f32_storage_padded_fanout_actual = runtime
            .execute_nd_real_convolution_f64(&f32_storage_padded_fanout_ir, &mixed_padded_input64)
            .unwrap();
        assert_close(
            &f32_storage_padded_fanout_actual,
            &mixed_padded_fanout_expected,
            2.0e-5,
            runtime.device_name(),
        );
        let formatted_f64 = build_formatted(runtime.device_profile(), Precision::F64);
        let formatted_actual64 = runtime
            .execute_nd_real_convolution_f64(&formatted_f64, &input64)
            .unwrap();
        assert_close(
            &formatted_actual64,
            &formatted_expected,
            1.0e-9,
            runtime.device_name(),
        );
        let mixed_formatted_f32 =
            build_formatted(runtime.device_profile(), Precision::F64ComputeF32Storage);
        let mixed_formatted_f32_actual = runtime
            .execute_nd_real_convolution_f64(&mixed_formatted_f32, &input64)
            .unwrap();
        assert_close(
            &mixed_formatted_f32_actual,
            &formatted_expected,
            2.0e-5,
            runtime.device_name(),
        );
        let formatted_padded_f64 = build_formatted_padded(runtime.device_profile(), Precision::F64);
        let formatted_padded_actual64 = runtime
            .execute_nd_real_convolution_f64(&formatted_padded_f64, &formatted_padded_input64)
            .unwrap();
        assert_close(
            &formatted_padded_actual64,
            &formatted_padded_expected,
            2.0e-9,
            runtime.device_name(),
        );
        let mixed_formatted_padded_f32 =
            build_formatted_padded(runtime.device_profile(), Precision::F64ComputeF32Storage);
        let mixed_formatted_padded_f32_actual = runtime
            .execute_nd_real_convolution_f64(&mixed_formatted_padded_f32, &formatted_padded_input64)
            .unwrap();
        assert_close(
            &mixed_formatted_padded_f32_actual,
            &formatted_padded_expected,
            2.0e-5,
            runtime.device_name(),
        );
        let coordinate_f64_ir = build_coordinates(runtime.device_profile(), Precision::F64);
        let coordinate_actual64 = runtime
            .execute_nd_real_convolution_f64(&coordinate_f64_ir, &coordinate_input64)
            .unwrap();
        assert_close(
            &coordinate_actual64,
            &coordinate_expected,
            1.2e-9,
            runtime.device_name(),
        );
        let coordinate_f32_storage_ir =
            build_coordinates(runtime.device_profile(), Precision::F64ComputeF32Storage);
        let coordinate_f32_storage_actual = runtime
            .execute_nd_real_convolution_f64(&coordinate_f32_storage_ir, &coordinate_input64)
            .unwrap();
        assert_close(
            &coordinate_f32_storage_actual,
            &coordinate_expected,
            2.0e-5,
            runtime.device_name(),
        );
        let coordinate_padded_f32_storage_ir =
            build_padded_coordinates(runtime.device_profile(), Precision::F64ComputeF32Storage);
        let coordinate_padded_f32_storage_actual = runtime
            .execute_nd_real_convolution_f64(
                &coordinate_padded_f32_storage_ir,
                &coordinate_padded_input64,
            )
            .unwrap();
        assert_close(
            &coordinate_padded_f32_storage_actual,
            &coordinate_padded_expected,
            2.0e-5,
            runtime.device_name(),
        );
        let matrix_f64_ir = build_matrix(runtime.device_profile(), Precision::F64);
        let matrix_actual64 = runtime
            .execute_nd_real_convolution_f64(&matrix_f64_ir, &matrix_input64)
            .unwrap();
        assert_close(
            &matrix_actual64,
            &matrix_expected,
            1.5e-9,
            runtime.device_name(),
        );
        let matrix_f32_storage_ir =
            build_matrix(runtime.device_profile(), Precision::F64ComputeF32Storage);
        let matrix_f32_storage_actual = runtime
            .execute_nd_real_convolution_f64(&matrix_f32_storage_ir, &matrix_input64)
            .unwrap();
        assert_close(
            &matrix_f32_storage_actual,
            &matrix_expected,
            2.0e-5,
            runtime.device_name(),
        );
        let matrix_multi_f64_ir =
            build_multi_kernel_matrix(runtime.device_profile(), Precision::F64);
        let matrix_multi_actual64 = runtime
            .execute_nd_real_convolution_f64(&matrix_multi_f64_ir, &matrix_input64)
            .unwrap();
        assert_close(
            &matrix_multi_actual64,
            &matrix_multi_expected,
            2.0e-9,
            runtime.device_name(),
        );
        let matrix_multi_f32_storage_ir =
            build_multi_kernel_matrix(runtime.device_profile(), Precision::F64ComputeF32Storage);
        let matrix_multi_f32_storage_actual = runtime
            .execute_nd_real_convolution_f64(&matrix_multi_f32_storage_ir, &matrix_input64)
            .unwrap();
        assert_close(
            &matrix_multi_f32_storage_actual,
            &matrix_multi_expected,
            2.0e-5,
            runtime.device_name(),
        );
        let padded_matrix_f64_ir = build_padded_matrix(runtime.device_profile(), Precision::F64);
        let padded_matrix_actual64 = runtime
            .execute_nd_real_convolution_f64(&padded_matrix_f64_ir, &padded_matrix_input64)
            .unwrap();
        assert_close(
            &padded_matrix_actual64,
            &padded_matrix_expected,
            2.0e-9,
            runtime.device_name(),
        );
        let padded_matrix_f32_storage_ir =
            build_padded_matrix(runtime.device_profile(), Precision::F64ComputeF32Storage);
        let padded_matrix_f32_storage_actual = runtime
            .execute_nd_real_convolution_f64(&padded_matrix_f32_storage_ir, &padded_matrix_input64)
            .unwrap();
        assert_close(
            &padded_matrix_f32_storage_actual,
            &padded_matrix_expected,
            2.0e-5,
            runtime.device_name(),
        );
        let padded_matrix_multi_f32_storage_ir = build_padded_multi_kernel_matrix(
            runtime.device_profile(),
            Precision::F64ComputeF32Storage,
        );
        let padded_matrix_multi_f32_storage_actual = runtime
            .execute_nd_real_convolution_f64(
                &padded_matrix_multi_f32_storage_ir,
                &padded_matrix_input64,
            )
            .unwrap();
        assert_close(
            &padded_matrix_multi_f32_storage_actual,
            &padded_matrix_multi_expected,
            2.0e-5,
            runtime.device_name(),
        );
    }
    run_policy_padded_native(runtime);
}

#[cfg(any(feature = "cuda-runtime", feature = "opencl-runtime"))]
fn run_policy_padded_native<R: vkfft_rs::backend::NativeRuntime>(runtime: &R) {
    let input64 = matrix_padded_probe_input64();
    let expected = matrix_policy_padded_expected64(&input64);
    let input32 = input64
        .iter()
        .map(|value| *value as f32)
        .collect::<Vec<_>>();
    let f32_ir = build_policy_padded_multi_kernel_matrix(runtime.device_profile(), Precision::F32);
    let actual32 = runtime
        .execute_nd_real_convolution_f32(&f32_ir, &input32)
        .unwrap()
        .into_iter()
        .map(f64::from)
        .collect::<Vec<_>>();
    assert_close(&actual32, &expected, 4.0e-3, runtime.device_name());
    let f16_ir = build_policy_padded_multi_kernel_matrix(
        runtime.device_profile(),
        Precision::F16StorageF32Compute,
    );
    let actual_f16 = runtime
        .execute_nd_real_convolution_f32(&f16_ir, &input32)
        .unwrap()
        .into_iter()
        .map(f64::from)
        .collect::<Vec<_>>();
    assert_close(&actual_f16, &expected, 1.0e-1, runtime.device_name());
    if runtime.device_profile().supports_f64 {
        let f64_ir =
            build_policy_padded_multi_kernel_matrix(runtime.device_profile(), Precision::F64);
        let actual64 = runtime
            .execute_nd_real_convolution_f64(&f64_ir, &input64)
            .unwrap();
        assert_close(&actual64, &expected, 3.0e-9, runtime.device_name());
        let f32_storage_ir = build_policy_padded_multi_kernel_matrix(
            runtime.device_profile(),
            Precision::F64ComputeF32Storage,
        );
        let actual_f32_storage = runtime
            .execute_nd_real_convolution_f64(&f32_storage_ir, &input64)
            .unwrap();
        assert_close(
            &actual_f32_storage,
            &expected,
            2.0e-5,
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
fn cuda_nd_real_convolution_or_skips() {
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
fn opencl_nd_real_convolution_or_skips() {
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
fn run_policy_padded_vulkan(runtime: &vkfft_rs::backend::vulkan::runtime::VulkanExecutionContext) {
    let input64 = matrix_padded_probe_input64();
    let expected = matrix_policy_padded_expected64(&input64);
    let input32 = input64
        .iter()
        .map(|value| *value as f32)
        .collect::<Vec<_>>();
    let f32_ir = build_policy_padded_multi_kernel_matrix(runtime.device_profile(), Precision::F32);
    let actual32 = runtime
        .execute_nd_real_convolution_f32(&f32_ir, &input32)
        .unwrap()
        .into_iter()
        .map(f64::from)
        .collect::<Vec<_>>();
    assert_close(
        &actual32,
        &expected,
        4.0e-3,
        "Vulkan F32 ND real policy/padding/matrix fanout",
    );
    let f16_ir = build_policy_padded_multi_kernel_matrix(
        runtime.device_profile(),
        Precision::F16StorageF32Compute,
    );
    let actual_f16 = runtime
        .execute_nd_real_convolution_f32(&f16_ir, &input32)
        .unwrap()
        .into_iter()
        .map(f64::from)
        .collect::<Vec<_>>();
    assert_close(
        &actual_f16,
        &expected,
        1.0e-1,
        "Vulkan F16-storage/F32-compute ND real policy/padding/matrix fanout",
    );
    if runtime.device_profile().supports_f64 {
        let f64_ir =
            build_policy_padded_multi_kernel_matrix(runtime.device_profile(), Precision::F64);
        let actual64 = runtime
            .execute_nd_real_convolution_f64(&f64_ir, &input64)
            .unwrap();
        assert_close(
            &actual64,
            &expected,
            3.0e-9,
            "Vulkan F64 ND real policy/padding/matrix fanout",
        );
        let f32_storage_ir = build_policy_padded_multi_kernel_matrix(
            runtime.device_profile(),
            Precision::F64ComputeF32Storage,
        );
        let actual_f32_storage = runtime
            .execute_nd_real_convolution_f64(&f32_storage_ir, &input64)
            .unwrap();
        assert_close(
            &actual_f32_storage,
            &expected,
            2.0e-5,
            "Vulkan F64-compute/F32-storage ND real policy/padding/matrix fanout",
        );
    }
}

#[cfg(feature = "vulkan-runtime")]
#[test]
fn vulkan_nd_real_convolution_or_skips() {
    use vkfft_rs::{VkFftError, backend::vulkan::runtime::VulkanExecutionContext};
    let _guard = gpu_test_lock()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let runtime = match VulkanExecutionContext::new() {
        Ok(runtime) => runtime,
        Err(VkFftError::VulkanUnavailable(_)) => return,
        Err(error) => panic!("Vulkan context failed after loader/device discovery: {error}"),
    };
    let expected = expected64(KERNEL_COUNT);
    let input64 = input64();
    let input32 = input64
        .iter()
        .map(|value| *value as f32)
        .collect::<Vec<_>>();
    let f32_ir = build(runtime.device_profile(), Precision::F32, KERNEL_COUNT);
    let actual32 = runtime
        .execute_nd_real_convolution_f32(&f32_ir, &input32)
        .unwrap()
        .into_iter()
        .map(f64::from)
        .collect::<Vec<_>>();
    assert_close(
        &actual32,
        &expected,
        1.5e-3,
        "Vulkan F32 ND real convolution",
    );

    let mixed_expected = expected64(1);
    let f16_storage_ir = build_mixed(runtime.device_profile(), Precision::F16StorageF32Compute);
    let f16_storage_actual = runtime
        .execute_nd_real_convolution_f32(&f16_storage_ir, &input32)
        .unwrap()
        .into_iter()
        .map(f64::from)
        .collect::<Vec<_>>();
    assert_close(
        &f16_storage_actual,
        &mixed_expected,
        8.0e-2,
        "Vulkan F16-storage/F32-compute ND real convolution K1",
    );

    let f16_storage_fanout_ir = build(
        runtime.device_profile(),
        Precision::F16StorageF32Compute,
        KERNEL_COUNT,
    );
    let f16_storage_fanout_actual = runtime
        .execute_nd_real_convolution_f32(&f16_storage_fanout_ir, &input32)
        .unwrap()
        .into_iter()
        .map(f64::from)
        .collect::<Vec<_>>();
    assert_close(
        &f16_storage_fanout_actual,
        &expected,
        8.0e-2,
        "Vulkan F16-storage/F32-compute ND real convolution K3 fan-out",
    );

    let mixed_padded_input64 = formatted_padded_input64();
    let mixed_padded_expected = formatted_padded_expected64(&mixed_padded_input64);
    let mixed_padded_input32 = mixed_padded_input64
        .iter()
        .map(|value| *value as f32)
        .collect::<Vec<_>>();
    let f16_storage_padded_ir =
        build_mixed_padded(runtime.device_profile(), Precision::F16StorageF32Compute);
    let f16_storage_padded_actual = runtime
        .execute_nd_real_convolution_f32(&f16_storage_padded_ir, &mixed_padded_input32)
        .unwrap()
        .into_iter()
        .map(f64::from)
        .collect::<Vec<_>>();
    assert_close(
        &f16_storage_padded_actual,
        &mixed_padded_expected,
        8.0e-2,
        "Vulkan F16-storage/F32-compute ND real convolution K1 spatial padding",
    );

    let mixed_padded_fanout_expected = padded_fanout_expected64(&mixed_padded_input64);
    let f16_storage_padded_fanout_ir =
        build_mixed_padded_fanout(runtime.device_profile(), Precision::F16StorageF32Compute);
    let f16_storage_padded_fanout_actual = runtime
        .execute_nd_real_convolution_f32(&f16_storage_padded_fanout_ir, &mixed_padded_input32)
        .unwrap()
        .into_iter()
        .map(f64::from)
        .collect::<Vec<_>>();
    assert_close(
        &f16_storage_padded_fanout_actual,
        &mixed_padded_fanout_expected,
        8.0e-2,
        "Vulkan F16-storage/F32-compute ND real convolution K3 spatial padding",
    );

    let formatted_expected = expected64(1);
    let formatted_f32 = build_formatted(runtime.device_profile(), Precision::F32);
    let formatted_actual32 = runtime
        .execute_nd_real_convolution_f32(&formatted_f32, &input32)
        .unwrap()
        .into_iter()
        .map(f64::from)
        .collect::<Vec<_>>();
    assert_close(
        &formatted_actual32,
        &formatted_expected,
        1.5e-3,
        "Vulkan F32 ND real formatted K1",
    );

    let mixed_formatted_f16 =
        build_formatted(runtime.device_profile(), Precision::F16StorageF32Compute);
    let mixed_formatted_f16_actual = runtime
        .execute_nd_real_convolution_f32(&mixed_formatted_f16, &input32)
        .unwrap()
        .into_iter()
        .map(f64::from)
        .collect::<Vec<_>>();
    assert_close(
        &mixed_formatted_f16_actual,
        &formatted_expected,
        8.0e-2,
        "Vulkan F16-storage/F32-compute ND real formatted K1",
    );

    let formatted_padded_input64 = formatted_padded_input64();
    let formatted_padded_expected = formatted_padded_expected64(&formatted_padded_input64);
    let formatted_padded_input32 = formatted_padded_input64
        .iter()
        .map(|value| *value as f32)
        .collect::<Vec<_>>();
    let formatted_padded_f32 = build_formatted_padded(runtime.device_profile(), Precision::F32);
    let formatted_padded_actual32 = runtime
        .execute_nd_real_convolution_f32(&formatted_padded_f32, &formatted_padded_input32)
        .unwrap()
        .into_iter()
        .map(f64::from)
        .collect::<Vec<_>>();
    assert_close(
        &formatted_padded_actual32,
        &formatted_padded_expected,
        2.0e-3,
        "Vulkan F32 ND real formatted K1 spatial padding",
    );

    let mixed_formatted_padded_f16 =
        build_formatted_padded(runtime.device_profile(), Precision::F16StorageF32Compute);
    let mixed_formatted_padded_f16_actual = runtime
        .execute_nd_real_convolution_f32(&mixed_formatted_padded_f16, &formatted_padded_input32)
        .unwrap()
        .into_iter()
        .map(f64::from)
        .collect::<Vec<_>>();
    assert_close(
        &mixed_formatted_padded_f16_actual,
        &formatted_padded_expected,
        8.0e-2,
        "Vulkan F16-storage/F32-compute ND real formatted K1 spatial padding",
    );

    let coordinate_expected = coordinate_expected64();
    let coordinate_input64 = coordinate_input64();
    let coordinate_input32 = coordinate_input64
        .iter()
        .map(|value| *value as f32)
        .collect::<Vec<_>>();
    let coordinate_f32_ir = build_coordinates(runtime.device_profile(), Precision::F32);
    let coordinate_actual32 = runtime
        .execute_nd_real_convolution_f32(&coordinate_f32_ir, &coordinate_input32)
        .unwrap()
        .into_iter()
        .map(f64::from)
        .collect::<Vec<_>>();
    assert_close(
        &coordinate_actual32,
        &coordinate_expected,
        1.8e-3,
        "Vulkan F32 ND real independent coordinates",
    );
    let coordinate_f16_ir =
        build_coordinates(runtime.device_profile(), Precision::F16StorageF32Compute);
    let coordinate_f16_actual = runtime
        .execute_nd_real_convolution_f32(&coordinate_f16_ir, &coordinate_input32)
        .unwrap()
        .into_iter()
        .map(f64::from)
        .collect::<Vec<_>>();
    assert_close(
        &coordinate_f16_actual,
        &coordinate_expected,
        8.0e-2,
        "Vulkan F16-storage/F32-compute ND real independent coordinates",
    );
    let coordinate_padded_input64 = coordinate_padded_probe_input64();
    let coordinate_padded_expected = coordinate_padded_expected64(&coordinate_padded_input64);
    let coordinate_padded_input32 = coordinate_padded_input64
        .iter()
        .map(|value| *value as f32)
        .collect::<Vec<_>>();
    let coordinate_padded_f16_ir =
        build_padded_coordinates(runtime.device_profile(), Precision::F16StorageF32Compute);
    let coordinate_padded_f16_actual = runtime
        .execute_nd_real_convolution_f32(&coordinate_padded_f16_ir, &coordinate_padded_input32)
        .unwrap()
        .into_iter()
        .map(f64::from)
        .collect::<Vec<_>>();
    assert_close(
        &coordinate_padded_f16_actual,
        &coordinate_padded_expected,
        8.0e-2,
        "Vulkan F16-storage/F32-compute ND real independent coordinates spatial padding",
    );

    let matrix_expected = matrix_expected64();
    let matrix_input64 = matrix_input64();
    let matrix_input32 = matrix_input64
        .iter()
        .map(|value| *value as f32)
        .collect::<Vec<_>>();
    let matrix_f32_ir = build_matrix(runtime.device_profile(), Precision::F32);
    let matrix_actual32 = runtime
        .execute_nd_real_convolution_f32(&matrix_f32_ir, &matrix_input32)
        .unwrap()
        .into_iter()
        .map(f64::from)
        .collect::<Vec<_>>();
    assert_close(
        &matrix_actual32,
        &matrix_expected,
        2.5e-3,
        "Vulkan F32 ND real matrix K1",
    );
    let matrix_f16_ir = build_matrix(runtime.device_profile(), Precision::F16StorageF32Compute);
    let matrix_f16_actual = runtime
        .execute_nd_real_convolution_f32(&matrix_f16_ir, &matrix_input32)
        .unwrap()
        .into_iter()
        .map(f64::from)
        .collect::<Vec<_>>();
    assert_close(
        &matrix_f16_actual,
        &matrix_expected,
        8.0e-2,
        "Vulkan F16-storage/F32-compute ND real matrix K1",
    );

    let matrix_multi_expected = matrix_multi_expected64();
    let matrix_multi_f32_ir = build_multi_kernel_matrix(runtime.device_profile(), Precision::F32);
    let matrix_multi_actual32 = runtime
        .execute_nd_real_convolution_f32(&matrix_multi_f32_ir, &matrix_input32)
        .unwrap()
        .into_iter()
        .map(f64::from)
        .collect::<Vec<_>>();
    assert_close(
        &matrix_multi_actual32,
        &matrix_multi_expected,
        3.0e-3,
        "Vulkan F32 ND real matrix K2",
    );
    let matrix_multi_f16_ir =
        build_multi_kernel_matrix(runtime.device_profile(), Precision::F16StorageF32Compute);
    let matrix_multi_f16_actual = runtime
        .execute_nd_real_convolution_f32(&matrix_multi_f16_ir, &matrix_input32)
        .unwrap()
        .into_iter()
        .map(f64::from)
        .collect::<Vec<_>>();
    assert_close(
        &matrix_multi_f16_actual,
        &matrix_multi_expected,
        8.0e-2,
        "Vulkan F16-storage/F32-compute ND real matrix K2",
    );

    let padded_matrix_input64 = matrix_padded_probe_input64();
    let padded_matrix_expected = matrix_padded_expected64(&padded_matrix_input64);
    let padded_matrix_input32 = padded_matrix_input64
        .iter()
        .map(|value| *value as f32)
        .collect::<Vec<_>>();
    let padded_matrix_f32_ir = build_padded_matrix(runtime.device_profile(), Precision::F32);
    let padded_matrix_actual32 = runtime
        .execute_nd_real_convolution_f32(&padded_matrix_f32_ir, &padded_matrix_input32)
        .unwrap()
        .into_iter()
        .map(f64::from)
        .collect::<Vec<_>>();
    assert_close(
        &padded_matrix_actual32,
        &padded_matrix_expected,
        3.0e-3,
        "Vulkan F32 ND real matrix K1 spatial zero padding",
    );
    let padded_matrix_f16_ir =
        build_padded_matrix(runtime.device_profile(), Precision::F16StorageF32Compute);
    let padded_matrix_f16_actual = runtime
        .execute_nd_real_convolution_f32(&padded_matrix_f16_ir, &padded_matrix_input32)
        .unwrap()
        .into_iter()
        .map(f64::from)
        .collect::<Vec<_>>();
    assert_close(
        &padded_matrix_f16_actual,
        &padded_matrix_expected,
        8.0e-2,
        "Vulkan F16-storage/F32-compute ND real matrix K1 spatial zero padding",
    );
    let padded_matrix_multi_expected = matrix_multi_padded_expected64(&padded_matrix_input64);
    let padded_matrix_multi_f16_ir =
        build_padded_multi_kernel_matrix(runtime.device_profile(), Precision::F16StorageF32Compute);
    let padded_matrix_multi_f16_actual = runtime
        .execute_nd_real_convolution_f32(&padded_matrix_multi_f16_ir, &padded_matrix_input32)
        .unwrap()
        .into_iter()
        .map(f64::from)
        .collect::<Vec<_>>();
    assert_close(
        &padded_matrix_multi_f16_actual,
        &padded_matrix_multi_expected,
        8.0e-2,
        "Vulkan F16-storage/F32-compute ND real matrix K2 spatial padding",
    );

    if runtime.device_profile().supports_f64 {
        let f64_ir = build(runtime.device_profile(), Precision::F64, KERNEL_COUNT);
        let actual64 = runtime
            .execute_nd_real_convolution_f64(&f64_ir, &input64)
            .unwrap();
        assert_close(
            &actual64,
            &expected,
            1.0e-9,
            "Vulkan F64 ND real convolution",
        );
        let f32_storage_ir = build_mixed(runtime.device_profile(), Precision::F64ComputeF32Storage);
        let f32_storage_actual = runtime
            .execute_nd_real_convolution_f64(&f32_storage_ir, &input64)
            .unwrap();
        assert_close(
            &f32_storage_actual,
            &mixed_expected,
            2.0e-5,
            "Vulkan F64-compute/F32-storage ND real convolution K1",
        );
        let f32_storage_fanout_ir = build(
            runtime.device_profile(),
            Precision::F64ComputeF32Storage,
            KERNEL_COUNT,
        );
        let f32_storage_fanout_actual = runtime
            .execute_nd_real_convolution_f64(&f32_storage_fanout_ir, &input64)
            .unwrap();
        assert_close(
            &f32_storage_fanout_actual,
            &expected,
            2.0e-5,
            "Vulkan F64-compute/F32-storage ND real convolution K3 fan-out",
        );
        let f32_storage_padded_ir =
            build_mixed_padded(runtime.device_profile(), Precision::F64ComputeF32Storage);
        let f32_storage_padded_actual = runtime
            .execute_nd_real_convolution_f64(&f32_storage_padded_ir, &mixed_padded_input64)
            .unwrap();
        assert_close(
            &f32_storage_padded_actual,
            &mixed_padded_expected,
            2.0e-5,
            "Vulkan F64-compute/F32-storage ND real convolution K1 spatial padding",
        );
        let f32_storage_padded_fanout_ir =
            build_mixed_padded_fanout(runtime.device_profile(), Precision::F64ComputeF32Storage);
        let f32_storage_padded_fanout_actual = runtime
            .execute_nd_real_convolution_f64(&f32_storage_padded_fanout_ir, &mixed_padded_input64)
            .unwrap();
        assert_close(
            &f32_storage_padded_fanout_actual,
            &mixed_padded_fanout_expected,
            2.0e-5,
            "Vulkan F64-compute/F32-storage ND real convolution K3 spatial padding",
        );
        let formatted_f64 = build_formatted(runtime.device_profile(), Precision::F64);
        let formatted_actual64 = runtime
            .execute_nd_real_convolution_f64(&formatted_f64, &input64)
            .unwrap();
        assert_close(
            &formatted_actual64,
            &formatted_expected,
            1.0e-9,
            "Vulkan F64 ND real formatted K1",
        );
        let mixed_formatted_f32 =
            build_formatted(runtime.device_profile(), Precision::F64ComputeF32Storage);
        let mixed_formatted_f32_actual = runtime
            .execute_nd_real_convolution_f64(&mixed_formatted_f32, &input64)
            .unwrap();
        assert_close(
            &mixed_formatted_f32_actual,
            &formatted_expected,
            2.0e-5,
            "Vulkan F64-compute/F32-storage ND real formatted K1",
        );
        let formatted_padded_f64 = build_formatted_padded(runtime.device_profile(), Precision::F64);
        let formatted_padded_actual64 = runtime
            .execute_nd_real_convolution_f64(&formatted_padded_f64, &formatted_padded_input64)
            .unwrap();
        assert_close(
            &formatted_padded_actual64,
            &formatted_padded_expected,
            2.0e-9,
            "Vulkan F64 ND real formatted K1 spatial padding",
        );
        let mixed_formatted_padded_f32 =
            build_formatted_padded(runtime.device_profile(), Precision::F64ComputeF32Storage);
        let mixed_formatted_padded_f32_actual = runtime
            .execute_nd_real_convolution_f64(&mixed_formatted_padded_f32, &formatted_padded_input64)
            .unwrap();
        assert_close(
            &mixed_formatted_padded_f32_actual,
            &formatted_padded_expected,
            2.0e-5,
            "Vulkan F64-compute/F32-storage ND real formatted K1 spatial padding",
        );
        let coordinate_f64_ir = build_coordinates(runtime.device_profile(), Precision::F64);
        let coordinate_actual64 = runtime
            .execute_nd_real_convolution_f64(&coordinate_f64_ir, &coordinate_input64)
            .unwrap();
        assert_close(
            &coordinate_actual64,
            &coordinate_expected,
            1.2e-9,
            "Vulkan F64 ND real independent coordinates",
        );
        let coordinate_f32_storage_ir =
            build_coordinates(runtime.device_profile(), Precision::F64ComputeF32Storage);
        let coordinate_f32_storage_actual = runtime
            .execute_nd_real_convolution_f64(&coordinate_f32_storage_ir, &coordinate_input64)
            .unwrap();
        assert_close(
            &coordinate_f32_storage_actual,
            &coordinate_expected,
            2.0e-5,
            "Vulkan F64-compute/F32-storage ND real independent coordinates",
        );
        let coordinate_padded_f32_storage_ir =
            build_padded_coordinates(runtime.device_profile(), Precision::F64ComputeF32Storage);
        let coordinate_padded_f32_storage_actual = runtime
            .execute_nd_real_convolution_f64(
                &coordinate_padded_f32_storage_ir,
                &coordinate_padded_input64,
            )
            .unwrap();
        assert_close(
            &coordinate_padded_f32_storage_actual,
            &coordinate_padded_expected,
            2.0e-5,
            "Vulkan F64-compute/F32-storage ND real independent coordinates spatial padding",
        );
        let matrix_f64_ir = build_matrix(runtime.device_profile(), Precision::F64);
        let matrix_actual64 = runtime
            .execute_nd_real_convolution_f64(&matrix_f64_ir, &matrix_input64)
            .unwrap();
        assert_close(
            &matrix_actual64,
            &matrix_expected,
            1.5e-9,
            "Vulkan F64 ND real matrix K1",
        );
        let matrix_f32_storage_ir =
            build_matrix(runtime.device_profile(), Precision::F64ComputeF32Storage);
        let matrix_f32_storage_actual = runtime
            .execute_nd_real_convolution_f64(&matrix_f32_storage_ir, &matrix_input64)
            .unwrap();
        assert_close(
            &matrix_f32_storage_actual,
            &matrix_expected,
            2.0e-5,
            "Vulkan F64-compute/F32-storage ND real matrix K1",
        );
        let matrix_multi_f64_ir =
            build_multi_kernel_matrix(runtime.device_profile(), Precision::F64);
        let matrix_multi_actual64 = runtime
            .execute_nd_real_convolution_f64(&matrix_multi_f64_ir, &matrix_input64)
            .unwrap();
        assert_close(
            &matrix_multi_actual64,
            &matrix_multi_expected,
            2.0e-9,
            "Vulkan F64 ND real matrix K2",
        );
        let matrix_multi_f32_storage_ir =
            build_multi_kernel_matrix(runtime.device_profile(), Precision::F64ComputeF32Storage);
        let matrix_multi_f32_storage_actual = runtime
            .execute_nd_real_convolution_f64(&matrix_multi_f32_storage_ir, &matrix_input64)
            .unwrap();
        assert_close(
            &matrix_multi_f32_storage_actual,
            &matrix_multi_expected,
            2.0e-5,
            "Vulkan F64-compute/F32-storage ND real matrix K2",
        );
        let padded_matrix_f64_ir = build_padded_matrix(runtime.device_profile(), Precision::F64);
        let padded_matrix_actual64 = runtime
            .execute_nd_real_convolution_f64(&padded_matrix_f64_ir, &padded_matrix_input64)
            .unwrap();
        assert_close(
            &padded_matrix_actual64,
            &padded_matrix_expected,
            2.0e-9,
            "Vulkan F64 ND real matrix K1 spatial zero padding",
        );
        let padded_matrix_f32_storage_ir =
            build_padded_matrix(runtime.device_profile(), Precision::F64ComputeF32Storage);
        let padded_matrix_f32_storage_actual = runtime
            .execute_nd_real_convolution_f64(&padded_matrix_f32_storage_ir, &padded_matrix_input64)
            .unwrap();
        assert_close(
            &padded_matrix_f32_storage_actual,
            &padded_matrix_expected,
            2.0e-5,
            "Vulkan F64-compute/F32-storage ND real matrix K1 spatial zero padding",
        );
        let padded_matrix_multi_f32_storage_ir = build_padded_multi_kernel_matrix(
            runtime.device_profile(),
            Precision::F64ComputeF32Storage,
        );
        let padded_matrix_multi_f32_storage_actual = runtime
            .execute_nd_real_convolution_f64(
                &padded_matrix_multi_f32_storage_ir,
                &padded_matrix_input64,
            )
            .unwrap();
        assert_close(
            &padded_matrix_multi_f32_storage_actual,
            &padded_matrix_multi_expected,
            2.0e-5,
            "Vulkan F64-compute/F32-storage ND real matrix K2 spatial padding",
        );
    }
    run_policy_padded_vulkan(&runtime);
}
