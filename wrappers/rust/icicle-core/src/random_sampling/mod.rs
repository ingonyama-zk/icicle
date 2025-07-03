use crate::field::PrimeField;
use crate::vec_ops::VecOpsConfig;
use crate::polynomial_ring::PolynomialRing;
use icicle_runtime::{errors::eIcicleError, memory::HostOrDeviceSlice};

pub mod tests;

/// Trait for random sampling operations on group elements.
pub trait RandomSampling<T: PrimeField> {
    fn random_sampling(
        size: usize,
        fast_mode: bool,
        seed: &[u8],
        cfg: &VecOpsConfig,
        output: &mut (impl HostOrDeviceSlice<T> + ?Sized),
    ) -> Result<(), eIcicleError>;
}

pub fn random_sampling<T>(
    size: usize,
    fast_mode: bool,
    seed: &[u8],
    cfg: &VecOpsConfig,
    output: &mut (impl HostOrDeviceSlice<T> + ?Sized),
) -> Result<(), eIcicleError>
where
    T: PrimeField,
    T: RandomSampling<T>,
{
    T::random_sampling(size, fast_mode, seed, cfg, output)
}

/// Implements RandomSampling for a scalar ring type using FFI.
#[macro_export]
macro_rules! impl_random_sampling {
    ($prefix:literal, $scalar_type:ty, $implement_for:ty) => {
        use icicle_core::random_sampling::RandomSampling;
        use icicle_core::vec_ops::VecOpsConfig;
        use icicle_runtime::eIcicleError;
        use icicle_runtime::memory::HostOrDeviceSlice;

        extern "C" {
            #[link_name = concat!($prefix, "_random_sampling")]
            fn random_sampling_ffi(
                size: usize,
                fast_mode: bool,
                seed: *const u8,
                seed_len: usize,
                cfg: *const VecOpsConfig,
                output: *mut $scalar_type,
            ) -> eIcicleError;
        }

        impl RandomSampling<$scalar_type> for $implement_for {
            fn random_sampling(
                size: usize,
                fast_mode: bool,
                seed: &[u8],
                cfg: &VecOpsConfig,
                output: &mut (impl HostOrDeviceSlice<$scalar_type> + ?Sized),
            ) -> Result<(), eIcicleError> {
                if output.is_on_device() && !output.is_on_active_device() {
                    eprintln!("Output is on an inactive device");
                    return Err(eIcicleError::InvalidArgument);
                }

                let mut cfg_clone = cfg.clone();
                cfg_clone.is_result_on_device = output.is_on_device();

                unsafe {
                    random_sampling_ffi(
                        size,
                        fast_mode,
                        seed.as_ptr(),
                        seed.len(),
                        &cfg_clone,
                        output.as_mut_ptr(),
                    )
                    .wrap()
                }
            }
        }
    };
}

/// Implements unit tests for RandomSampling on scalar ring types.
#[macro_export]
macro_rules! impl_random_sampling_tests {
    ($scalar_type: ident) => {
        mod test_scalar {
            use super::*;
            use icicle_core::random_sampling::tests::*;
            use icicle_runtime::test_utilities;

            /// Initializes devices before running tests.
            pub fn initialize() {
                test_utilities::test_load_and_init_devices();
            }

            #[test]
            fn test_random_sampling() {
                initialize();
                check_random_sampling::<$scalar_type>();
            }
        }
    };
}

pub trait ChallengeSpacePolynomialsSampling<T: PolynomialRing> {
    fn challenge_space_polynomials_sampling(
        seed: &[u8],
        cfg: &VecOpsConfig,
        ones: usize,
        twos: usize,
        norm: usize,
        output: &mut (impl HostOrDeviceSlice<T> + ?Sized),
    ) -> Result<(), eIcicleError>;
}

/// Samples Rq challenge polynomials with {0, 1, 2, -1, -2} coefficients.
///
/// This function samples challenge polynomials with specific coefficient patterns. The sampling process:
/// 1. Initializes a polynomial with coefficients consisting of `ones` number of ±1s, `twos` number of ±2s, and the
///    rest of the coefficients are 0s.
/// 2. Randomly flips the signs of the coefficients.
/// 3. Permutes the coefficients randomly.
/// 4. If `norm` is nonzero, only polynomials with operator norm less than or equal to `norm` are accepted.
///
/// # Parameters
/// - `seed`: Seed buffer for deterministic sampling.
/// - `cfg`: Vector operations configuration (e.g., backend, device).
/// - `ones`: Number of ±1 coefficients in each polynomial.
/// - `twos`: Number of ±2 coefficients in each polynomial.
/// - `norm`: If greater than zero, computes operator norm for the sampled polynomials and reject the ones with norm
/// greater than this value.
/// - `output`: Output buffer to store the sampled Rq polynomials. Should be of size `cfg.batch_size`.
///
/// # Returns
/// - `Ok(())` on success, or an error code on failure.
pub fn challenge_space_polynomials_sampling<T>(
    seed: &[u8],
    cfg: &VecOpsConfig,
    ones: usize,
    twos: usize,
    norm: usize,
    output: &mut (impl HostOrDeviceSlice<T> + ?Sized),
) -> Result<(), eIcicleError>
where
    T: PolynomialRing,
    T::Base: PrimeField,
    T: ChallengeSpacePolynomialsSampling<T>,
{
    T::challenge_space_polynomials_sampling(seed, cfg, ones, twos, norm, output)
}

/// Implements ChallengeSpacePolynomialsSampling for a PolyRing type using FFI.
#[macro_export]
macro_rules! impl_challenge_space_polynomials_sampling {
    ($prefix:literal, $poly_ring_type:ty) => {
        use icicle_core::random_sampling::ChallengeSpacePolynomialsSampling;

        extern "C" {
            #[link_name = concat!($prefix, "_sample_challenge_space_polynomials")]
            fn challenge_space_polynomials_sampling_ffi(
                seed: *const u8,
                seed_len: usize,
                size: usize,
                ones: u32,
                twos: u32,
                norm: u64,
                cfg: *const VecOpsConfig,
                output: *mut $poly_ring_type,
            ) -> eIcicleError;
        }

        impl ChallengeSpacePolynomialsSampling<$poly_ring_type> for $poly_ring_type {
            fn challenge_space_polynomials_sampling(
                seed: &[u8],
                cfg: &VecOpsConfig,
                ones: usize,
                twos: usize,
                norm: usize,
                output: &mut (impl HostOrDeviceSlice<$poly_ring_type> + ?Sized),
            ) -> Result<(), eIcicleError> {
                if output.is_on_device() && !output.is_on_active_device() {
                    eprintln!("Output is on an inactive device");
                    return Err(eIcicleError::InvalidArgument);
                }

                let mut cfg_clone = cfg.clone();
                cfg_clone.is_result_on_device = output.is_on_device();

                unsafe {
                    challenge_space_polynomials_sampling_ffi(
                        seed.as_ptr(),
                        seed.len(),
                        output.len(),
                        ones as u32,
                        twos as u32,
                        norm as u64,
                        &cfg_clone,
                        output.as_mut_ptr(),
                    )
                    .wrap()
                }
            }
        }
    };
}

/// Implements unit tests for ChallengeSpacePolynomialsSampling on PolyRing types.
#[macro_export]
macro_rules! impl_challenge_space_polynomials_sampling_tests {
    ($poly_ring_type: ident) => {
        mod test_poly_ring {
            use super::*;
            use icicle_core::random_sampling::tests::*;
            use icicle_runtime::test_utilities;

            /// Initializes devices before running tests.
            pub fn initialize() {
                test_utilities::test_load_and_init_devices();
            }

            #[test]
            fn test_challenge_space_polynomials_sampling() {
                initialize();
                check_challenge_space_polynomials_sampling::<$poly_ring_type>();
            }
        }
    };
}
