use crate::{polynomial_ring::PolynomialRing, vec_ops::VecOpsConfig};
use icicle_runtime::{eIcicleError, memory::HostOrDeviceSlice};

pub mod tests;

/// Balanced base decomposition API for ring elements.
///
/// This trait allows elements in a ring (e.g. finite fields, polynomial rings) to be
/// decomposed into a sequence of digits in a balanced base-b representation
/// (digits in the range [-b/2, b/2)), and later recomposed back into the original value.
///
/// ### Output layout:
/// For an input slice of `n` elements and digit count `d = count_digits(base)`:
///
/// - The output vector has length `n * d`.
/// - Digits are grouped **by digit index**, not by element:
///     - The first `n` entries are the **first digit** of all elements.
///     - The next `n` entries are the **second digit** of all elements.
///     - And so on, until all `d` digits are emitted.
///
/// This layout is consistent for both scalar fields (e.g. `Zq`) and polynomial rings (e.g. `Rq`),
/// where the digit decomposition is applied element-wise to the entire input slice.
pub trait BalancedDecomposition<T> {
    /// Computes the number of balanced base-b digits required to represent a field element.
    fn count_digits(base: u32) -> u32;

    /// Decomposes field elements into balanced base-b digits.
    ///
    /// The output buffer must have length `input.len() * count_digits(base)`.
    fn decompose(
        input: &(impl HostOrDeviceSlice<T> + ?Sized),
        output: &mut (impl HostOrDeviceSlice<T> + ?Sized),
        base: u32,
        cfg: &VecOpsConfig,
    ) -> Result<(), eIcicleError>;

    /// Recomposes field elements from balanced base-b digits.
    ///
    /// The input buffer must have length `output.len() * count_digits(base)`.
    fn recompose(
        input: &(impl HostOrDeviceSlice<T> + ?Sized),
        output: &mut (impl HostOrDeviceSlice<T> + ?Sized),
        base: u32,
        cfg: &VecOpsConfig,
    ) -> Result<(), eIcicleError>;
}

// Public floating functions around the trait
pub fn count_digits<T: PolynomialRing + BalancedDecomposition<T>>(base: u32) -> u32 {
    T::count_digits(base)
}

pub fn decompose<T: BalancedDecomposition<T>>(
    input: &(impl HostOrDeviceSlice<T> + ?Sized),
    output: &mut (impl HostOrDeviceSlice<T> + ?Sized),
    base: u32,
    cfg: &VecOpsConfig,
) -> Result<(), eIcicleError>
where
    T: BalancedDecomposition<T>,
{
    T::decompose(input, output, base, cfg)
}

pub fn recompose<T: BalancedDecomposition<T>>(
    input: &(impl HostOrDeviceSlice<T> + ?Sized),
    output: &mut (impl HostOrDeviceSlice<T> + ?Sized),
    base: u32,
    cfg: &VecOpsConfig,
) -> Result<(), eIcicleError>
where
    T: BalancedDecomposition<T>,
{
    T::recompose(input, output, base, cfg)
}

#[macro_export]
macro_rules! impl_balanced_decomposition {
    (
        $prefix: literal,
        $ring_type: ident
    ) => {
        extern "C" {
            #[link_name = concat!($prefix, "_balanced_decomposition_nof_digits")]
            fn balanced_decomposition_nof_digits(base: u32) -> u32;

            #[link_name = concat!($prefix, "_decompose_balanced_digits")]
            fn decompose_balanced_digits(
                input: *const $ring_type,
                input_size: usize,
                base: u32,
                cfg: *const VecOpsConfig,
                output: *mut $ring_type,
                output_size: usize,
            ) -> eIcicleError;

            #[link_name = concat!($prefix, "_recompose_from_balanced_digits")]
            fn recompose_from_balanced_digits(
                input: *const $ring_type,
                input_size: usize,
                base: u32,
                cfg: *const VecOpsConfig,
                output: *mut $ring_type,
                output_size: usize,
            ) -> eIcicleError;
        }

        fn balanced_decomposition_check_args(
            input: &(impl HostOrDeviceSlice<$ring_type> + ?Sized),
            output: &mut (impl HostOrDeviceSlice<$ring_type> + ?Sized),
            cfg: &mut VecOpsConfig,
        ) -> Result<(), eIcicleError> {
            if input.is_on_device() && !input.is_on_active_device() {
                eprintln!("Input is on an inactive device");
                return Err(eIcicleError::InvalidArgument);
            }

            if output.is_on_device() && !output.is_on_active_device() {
                eprintln!("Output is on an inactive device");
                return Err(eIcicleError::InvalidArgument);
            }

            cfg.is_a_on_device = input.is_on_device();
            cfg.is_result_on_device = output.is_on_device();

            Ok(())
        }

        impl BalancedDecomposition<$ring_type> for $ring_type {
            fn count_digits(base: u32) -> u32 {
                unsafe { balanced_decomposition_nof_digits(base) }
            }

            fn decompose(
                input: &(impl HostOrDeviceSlice<$ring_type> + ?Sized),
                output: &mut (impl HostOrDeviceSlice<$ring_type> + ?Sized),
                base: u32,
                cfg: &VecOpsConfig,
            ) -> Result<(), eIcicleError> {
                let mut cfg = cfg.clone();
                balanced_decomposition_check_args(input, output, &mut cfg)?;

                unsafe {
                    decompose_balanced_digits(
                        input.as_ptr(),
                        input.len() / (cfg.batch_size as usize),
                        base,
                        &cfg,
                        output.as_mut_ptr(),
                        output.len() / (cfg.batch_size as usize),
                    )
                    .wrap()
                }
            }

            fn recompose(
                input: &(impl HostOrDeviceSlice<$ring_type> + ?Sized),
                output: &mut (impl HostOrDeviceSlice<$ring_type> + ?Sized),
                base: u32,
                cfg: &VecOpsConfig,
            ) -> Result<(), eIcicleError> {
                let mut cfg = cfg.clone();
                balanced_decomposition_check_args(input, output, &mut cfg)?;

                unsafe {
                    recompose_from_balanced_digits(
                        input.as_ptr(),
                        input.len() / (cfg.batch_size as usize),
                        base,
                        &cfg,
                        output.as_mut_ptr(),
                        output.len() / (cfg.batch_size as usize),
                    )
                    .wrap()
                }
            }
        }
    };
}

#[macro_export]
macro_rules! impl_balanced_decomposition_tests {
    ($ring_type: ident) => {
        use icicle_core::balanced_decomposition::tests::*;
        use icicle_runtime::test_utilities;

        /// Initializes devices before running tests.
        pub fn initialize() {
            test_utilities::test_load_and_init_devices();
            test_utilities::test_set_main_device();
        }

        #[cfg(test)]
        mod tests {
            use super::*;

            #[test]
            fn test_balanced_decomposition() {
                initialize();
                test_utilities::test_set_main_device();
                check_balanced_decomposition::<$ring_type>();
                test_utilities::test_set_ref_device();
                check_balanced_decomposition::<$ring_type>();
            }
        }
    };
}
