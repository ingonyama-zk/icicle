/// Trait representing a polynomial ring: R = Base[X] / (X^DEGREE - MODULUS_COEFF)
pub trait PolynomialRing: Sized {
    /// Base field type
    type Base: Copy;

    /// Number of terms in the polynomial (polynomials are degree < DEGREE)
    const DEGREE: usize;

    /// The constant term in the modulus polynomial: X^DEGREE - MODULUS_COEFF
    const MODULUS_COEFF: i32;

    /// Returns the internal values (coefficients or evaluations)
    fn values(&self) -> &[Self::Base];

    /// Mutable access to internal values
    fn values_mut(&mut self) -> &mut [Self::Base];

    /// Construct a zero polynomial (all values = 0)
    fn zero() -> Self;

    /// Construct from a slice (should panic or assert if length ≠ DEGREE)
    fn from_slice(values: &[Self::Base]) -> Self;
}

#[macro_export]
macro_rules! impl_polynomial_ring {
    ($name:ident, $base:ty, $degree:expr, $modulus_coeff:expr) => {
        #[derive(Clone, Copy, Debug, PartialEq)]
        #[repr(C)]
        pub struct $name {
            values: [$base; $degree],
        }

        impl icicle_core::polynomial_ring::PolynomialRing for $name {
            type Base = $base;

            const DEGREE: usize = $degree;
            const MODULUS_COEFF: i32 = $modulus_coeff;

            fn values(&self) -> &[Self::Base] {
                &self.values
            }

            fn values_mut(&mut self) -> &mut [Self::Base] {
                &mut self.values
            }

            fn zero() -> Self {
                Self {
                    values: [<$base>::zero(); $degree],
                }
            }

            fn from_slice(input: &[Self::Base]) -> Self {
                assert_eq!(input.len(), Self::DEGREE);
                let mut values = [<$base>::zero(); $degree];
                values.copy_from_slice(input);
                Self { values }
            }
        }
    };
}

#[macro_export]
macro_rules! test_polynomial_ring {
    ($type:ty) => {
        mod test_polynomial_ring {
            use super::*;
            use icicle_core::tests::*;

            #[test]
            fn test_zero_and_from_slice() {
                check_zero_and_from_slice::<$type>();
            }

            #[test]
            fn test_vector_alloc() {
                check_vector_alloc::<$type>();
            }
        }
    };
}
