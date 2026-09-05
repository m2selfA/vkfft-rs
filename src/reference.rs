//! Correctness-oriented CPU reference transforms.
//!
//! This module is not intended to compete with the eventual GPU backends. It
//! gives the port a Rust-native oracle for validating planner and generated
//! kernels without depending on FFTW or the C VkFFT implementation.

use core::f64::consts::PI;

use crate::complex::Complex64;
use crate::config::Direction;
use crate::error::{Result, VkFftError};
use crate::planner::{BLUESTEIN_SMOOTH_PRIMES, STOCKHAM_PRIMES, next_smooth_at_least};

pub fn dft(input: &[Complex64], direction: Direction, normalize_inverse: bool) -> Vec<Complex64> {
    let mut output = dft_unscaled(input, direction);
    if direction == Direction::Inverse && normalize_inverse && !output.is_empty() {
        let scale = 1.0 / output.len() as f64;
        for value in &mut output {
            *value = value.scale(scale);
        }
    }
    output
}

pub fn fft(
    input: &[Complex64],
    direction: Direction,
    normalize_inverse: bool,
) -> Result<Vec<Complex64>> {
    if input.is_empty() {
        return Ok(Vec::new());
    }

    let mut output = fft_unscaled(input, direction)?;
    if direction == Direction::Inverse && normalize_inverse {
        let scale = 1.0 / output.len() as f64;
        for value in &mut output {
            *value = value.scale(scale);
        }
    }
    Ok(output)
}

fn dft_unscaled(input: &[Complex64], direction: Direction) -> Vec<Complex64> {
    let n = input.len();
    let mut output = vec![Complex64::new(0.0, 0.0); n];
    let sign = direction.exponent_sign();

    for (k, out) in output.iter_mut().enumerate() {
        let mut sum = Complex64::new(0.0, 0.0);
        for (index, &value) in input.iter().enumerate() {
            let angle = sign * 2.0 * PI * (index as f64) * (k as f64) / n as f64;
            sum += value * Complex64::exp_i(angle);
        }
        *out = sum;
    }
    output
}

fn fft_unscaled(input: &[Complex64], direction: Direction) -> Result<Vec<Complex64>> {
    let n = input.len();
    if n <= 1 {
        return Ok(input.to_vec());
    }

    if let Some(radix) = STOCKHAM_PRIMES
        .into_iter()
        .find(|radix| n.is_multiple_of(*radix))
    {
        return mixed_radix_unscaled(input, direction, radix);
    }

    bluestein_unscaled(input, direction)
}

fn mixed_radix_unscaled(
    input: &[Complex64],
    direction: Direction,
    radix: usize,
) -> Result<Vec<Complex64>> {
    let n = input.len();
    let inner_len = n / radix;
    let mut inner_ffts = Vec::with_capacity(radix);

    for lane in 0..radix {
        let mut sequence = Vec::with_capacity(inner_len);
        for index in 0..inner_len {
            sequence.push(input[lane + radix * index]);
        }
        inner_ffts.push(fft_unscaled(&sequence, direction)?);
    }

    let sign = direction.exponent_sign();
    let mut output = vec![Complex64::new(0.0, 0.0); n];
    for (k, out) in output.iter_mut().enumerate() {
        let inner_index = k % inner_len;
        let mut sum = Complex64::new(0.0, 0.0);
        for (lane, inner_fft) in inner_ffts.iter().enumerate().take(radix) {
            let angle = sign * 2.0 * PI * (lane as f64) * (k as f64) / n as f64;
            sum += inner_fft[inner_index] * Complex64::exp_i(angle);
        }
        *out = sum;
    }
    Ok(output)
}

fn bluestein_unscaled(input: &[Complex64], direction: Direction) -> Result<Vec<Complex64>> {
    let n = input.len();
    let minimum = n
        .checked_mul(2)
        .and_then(|value| value.checked_sub(1))
        .ok_or(VkFftError::ArithmeticOverflow {
            operation: "reference Bluestein convolution length",
        })?;
    let convolution_len = next_smooth_at_least(minimum, &BLUESTEIN_SMOOTH_PRIMES)?;
    let sign = direction.exponent_sign();

    let mut a = vec![Complex64::new(0.0, 0.0); convolution_len];
    let mut b = vec![Complex64::new(0.0, 0.0); convolution_len];

    for index in 0..n {
        let index_f64 = index as f64;
        let angle = sign * PI * index_f64 * index_f64 / n as f64;
        let chirp = Complex64::exp_i(angle);
        let inverse_chirp = Complex64::exp_i(-angle);
        a[index] = input[index] * chirp;
        b[index] = inverse_chirp;
        if index != 0 {
            b[convolution_len - index] = inverse_chirp;
        }
    }

    let a_frequency = fft_unscaled(&a, Direction::Forward)?;
    let b_frequency = fft_unscaled(&b, Direction::Forward)?;
    let mut product = Vec::with_capacity(convolution_len);
    for index in 0..convolution_len {
        product.push(a_frequency[index] * b_frequency[index]);
    }

    let mut convolution = fft_unscaled(&product, Direction::Inverse)?;
    let convolution_scale = 1.0 / convolution_len as f64;
    for value in &mut convolution {
        *value = value.scale(convolution_scale);
    }

    let mut output = Vec::with_capacity(n);
    for (index, value) in convolution.into_iter().take(n).enumerate() {
        let index_f64 = index as f64;
        let angle = sign * PI * index_f64 * index_f64 / n as f64;
        output.push(value * Complex64::exp_i(angle));
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(length: usize) -> Vec<Complex64> {
        (0..length)
            .map(|index| {
                let x = index as f64;
                Complex64::new((0.37 * x).sin() + x * 0.01, (0.19 * x).cos() - x * 0.02)
            })
            .collect()
    }

    fn max_error(lhs: &[Complex64], rhs: &[Complex64]) -> f64 {
        lhs.iter()
            .zip(rhs)
            .map(|(lhs, rhs)| {
                let delta = *lhs - *rhs;
                delta.norm_sqr().sqrt()
            })
            .fold(0.0, f64::max)
    }

    #[test]
    fn mixed_radix_and_bluestein_match_naive_dft() {
        for length in [2usize, 3, 4, 6, 7, 11, 13, 17, 19, 31, 47, 77] {
            let input = sample(length);
            let expected = dft(&input, Direction::Forward, false);
            let actual = fft(&input, Direction::Forward, false).unwrap();
            let tolerance = 5.0e-10 * length as f64;
            assert!(
                max_error(&actual, &expected) <= tolerance,
                "length {length} exceeded tolerance"
            );
        }
    }

    #[test]
    fn normalized_inverse_round_trips() {
        for length in [8usize, 17, 47, 64] {
            let input = sample(length);
            let transformed = fft(&input, Direction::Forward, false).unwrap();
            let restored = fft(&transformed, Direction::Inverse, true).unwrap();
            assert!(max_error(&restored, &input) < 5.0e-10 * length as f64);
        }
    }
}
