use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// A GPU-friendly complex number with a stable interleaved real/imaginary layout.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Complex<T> {
    pub re: T,
    pub im: T,
}

pub type Complex32 = Complex<f32>;
pub type Complex64 = Complex<f64>;

impl<T> Complex<T> {
    pub const fn new(re: T, im: T) -> Self {
        Self { re, im }
    }
}

impl<T> Add for Complex<T>
where
    T: Copy + Add<Output = T>,
{
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self::new(self.re + rhs.re, self.im + rhs.im)
    }
}

impl<T> AddAssign for Complex<T>
where
    T: Copy + AddAssign,
{
    fn add_assign(&mut self, rhs: Self) {
        self.re += rhs.re;
        self.im += rhs.im;
    }
}

impl<T> Sub for Complex<T>
where
    T: Copy + Sub<Output = T>,
{
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self::new(self.re - rhs.re, self.im - rhs.im)
    }
}

impl<T> SubAssign for Complex<T>
where
    T: Copy + SubAssign,
{
    fn sub_assign(&mut self, rhs: Self) {
        self.re -= rhs.re;
        self.im -= rhs.im;
    }
}

impl<T> Mul for Complex<T>
where
    T: Copy + Add<Output = T> + Sub<Output = T> + Mul<Output = T>,
{
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self::new(
            self.re * rhs.re - self.im * rhs.im,
            self.re * rhs.im + self.im * rhs.re,
        )
    }
}

impl<T> MulAssign for Complex<T>
where
    T: Copy + Add<Output = T> + Sub<Output = T> + Mul<Output = T>,
{
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<T> Neg for Complex<T>
where
    T: Neg<Output = T>,
{
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self::new(-self.re, -self.im)
    }
}

impl Complex64 {
    pub fn exp_i(angle: f64) -> Self {
        Self::new(angle.cos(), angle.sin())
    }

    pub const fn conj(self) -> Self {
        Self::new(self.re, -self.im)
    }

    pub fn norm_sqr(self) -> f64 {
        self.re * self.re + self.im * self.im
    }

    pub fn scale(self, scale: f64) -> Self {
        Self::new(self.re * scale, self.im * scale)
    }
}

impl Complex32 {
    pub fn exp_i(angle: f32) -> Self {
        Self::new(angle.cos(), angle.sin())
    }

    pub const fn conj(self) -> Self {
        Self::new(self.re, -self.im)
    }

    pub fn norm_sqr(self) -> f32 {
        self.re * self.re + self.im * self.im
    }

    pub fn scale(self, scale: f32) -> Self {
        Self::new(self.re * scale, self.im * scale)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn complex_layout_matches_two_scalars() {
        assert_eq!(
            core::mem::size_of::<Complex32>(),
            2 * core::mem::size_of::<f32>()
        );
        assert_eq!(
            core::mem::size_of::<Complex64>(),
            2 * core::mem::size_of::<f64>()
        );
    }

    #[test]
    fn multiplication_is_complex_multiplication() {
        let lhs = Complex64::new(2.0, 3.0);
        let rhs = Complex64::new(4.0, -5.0);
        assert_eq!(lhs * rhs, Complex64::new(23.0, 2.0));
    }
}
