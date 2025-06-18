use crate::traits::FieldImpl;
use icicle_runtime::memory::{DeviceSlice, HostOrDeviceSlice, HostSlice};

/// Trait representing a polynomial ring: R = Base[X] / (X^DEGREE - MODULUS_COEFF)
pub trait PolynomialRing: Sized + Clone + PartialEq + core::fmt::Debug {
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

pub fn flatten_polynomials_host_slice<P>(input: &HostSlice<P>) -> &HostSlice<P::Base>
where
    P: PolynomialRing,
    P::Base: FieldImpl,
{
    let poly_len = input.len();
    let coeffs_per_poly = P::DEGREE; // or P::Config::N or similar

    // SAFETY:
    // - We assume that `P` is `#[repr(C)]` and contains exactly `[Zq; N]`
    // - So a `[P]` of length `n` can be viewed as a `[Zq]` of length `n * N`
    unsafe { HostSlice::from_raw_parts(input.as_ptr() as *const P::Base, poly_len * coeffs_per_poly) }
}

// pub fn flatten_polynomials_device_slice<P>(input: &DeviceSlice<P>) -> &DeviceSlice<P::Base>
// where
//     P: PolynomialRing,
//     P::Base: FieldImpl,
// {
//     let poly_len = input.len();
//     let coeffs_per_poly = P::DEGREE;

//     // SAFETY:
//     // - `P` must be `#[repr(C)]` or compatible and contain exactly `[Zq; DEGREE]`.
//     // - Device memory layout of `[P]` must match a flat layout of `[Zq]`.
//     // - `DeviceSlice<T>` must be `#[repr(transparent)]` over `[T]`.
//     unsafe { unsafe { DeviceSlice::from_raw_parts(input.as_ptr() as *const P::Base, poly_len * coeffs_per_poly) } }
// }

#[macro_export]
macro_rules! impl_polynomial_ring {
    ($polyring:ident, $base:ty, $degree:expr, $modulus_coeff:expr) => {
        #[derive(Clone, Copy, Debug, PartialEq)]
        #[repr(C)]
        pub struct $polyring {
            values: [$base; $degree],
        }

        use icicle_core::polynomial_ring::PolynomialRing;
        use icicle_core::traits::GenerateRandom;

        impl PolynomialRing for $polyring {
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

impl GenerateRandom<$polyring> for $polyring {
    fn generate_random(size: usize) -> Vec<$polyring> {
        use std::mem::{forget, ManuallyDrop};
        use std::slice;

        let flat_base_field_vec: Vec<$base> = <<$base as icicle_core::traits::FieldImpl>::Config as icicle_core::traits::GenerateRandom<$base>>::generate_random(
            size * Self::DEGREE,
        );

        let ptr = flat_base_field_vec.as_ptr() as *mut $polyring;
        let len = size;
        let cap = flat_base_field_vec.capacity() / Self::DEGREE;

        // Avoid double-drop
        forget(flat_base_field_vec);

        unsafe {
            Vec::from_raw_parts(ptr, len, cap)
        }
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

            #[test]
            fn test_flatten_slices() {
                check_polyring_flatten_host_memory::<$type>();
            }
        }
    };
}
