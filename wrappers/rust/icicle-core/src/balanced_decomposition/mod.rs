use crate::{field::PrimeField, vec_ops::VecOpsConfig};
use icicle_runtime::{eIcicleError, memory::HostOrDeviceSlice};

pub mod tests;

/// Balanced base decomposition API for field/ring types.
///
/// This trait allows elements in a finite field/ring to be decomposed into a sequence of
/// digits in a balanced base-b representation (digits in the range [-b/2, b/2)),
/// and later recomposed back into the original field element.
pub trait BalancedDecomposition: PrimeField {
    /// Computes the number of balanced base-b digits required to represent a field element.
    fn count_digits(base: u32) -> u32;

    /// Decomposes field elements into balanced base-b digits.
    ///
    /// The output buffer must have length `input.len() * count_digits(base)`.
    fn decompose(
        input: &(impl HostOrDeviceSlice<Self> + ?Sized),
        output: &mut (impl HostOrDeviceSlice<Self> + ?Sized),
        base: u32,
        cfg: &VecOpsConfig,
    ) -> Result<(), eIcicleError>;

    /// Recomposes field elements from balanced base-b digits.
    ///
    /// The input buffer must have length `output.len() * count_digits(base)`.
    fn recompose(
        input: &(impl HostOrDeviceSlice<Self> + ?Sized),
        output: &mut (impl HostOrDeviceSlice<Self> + ?Sized),
        base: u32,
        cfg: &VecOpsConfig,
    ) -> Result<(), eIcicleError>;
}

// Public floating functions around the trait
pub fn count_digits<T: PrimeField + BalancedDecomposition>(base: u32) -> u32 {
    T::count_digits(base)
}

pub fn decompose<T: PrimeField + BalancedDecomposition>(
    input: &(impl HostOrDeviceSlice<T> + ?Sized),
    output: &mut (impl HostOrDeviceSlice<T> + ?Sized),
    base: u32,
    cfg: &VecOpsConfig,
) -> Result<(), eIcicleError> {
    T::decompose(input, output, base, cfg)
}

pub fn recompose<T: PrimeField + BalancedDecomposition>(
    input: &(impl HostOrDeviceSlice<T> + ?Sized),
    output: &mut (impl HostOrDeviceSlice<T> + ?Sized),
    base: u32,
    cfg: &VecOpsConfig,
) -> Result<(), eIcicleError> {
    T::recompose(input, output, base, cfg)
}

/// Internal macro to implement the `BalancedDecomposition` trait for a specific field backend.
#[macro_export]
macro_rules! impl_balanced_decomposition {
    (
        $field_prefix: literal,
        $field_type: ident,
        $field_cfg_type: ident
    ) => {
        use icicle_core::balanced_decomposition::BalancedDecomposition;

        extern "C" {
            #[link_name = concat!($field_prefix, "_balanced_decomposition_nof_digits")]
            fn balanced_decomposition_nof_digits(base: u32) -> u32;

            #[link_name = concat!($field_prefix, "_decompose_balanced_digits")]
            fn decompose_balanced_digits(
                input: *const $field_type,
                input_size: usize,
                base: u32,
                cfg: *const VecOpsConfig,
                output: *mut $field_type,
                output_size: usize,
            ) -> eIcicleError;

            #[link_name = concat!($field_prefix, "_recompose_from_balanced_digits")]
            fn recompose_from_balanced_digits(
                input: *const $field_type,
                input_size: usize,
                base: u32,
                cfg: *const VecOpsConfig,
                output: *mut $field_type,
                output_size: usize,
            ) -> eIcicleError;
        }

        fn balanced_decomposition_check_args(
            input: &(impl HostOrDeviceSlice<$field_type> + ?Sized),
            output: &mut (impl HostOrDeviceSlice<$field_type> + ?Sized),
            cfg: &mut VecOpsConfig,
        ) -> Result<(), eIcicleError> {
            if input.len() % (cfg.batch_size as usize) != 0 || output.len() % (cfg.batch_size as usize) != 0 {
                eprintln!(
                    "Batch size {} must divide input size {} and output size {}",
                    cfg.batch_size,
                    input.len(),
                    output.len()
                );
                return Err(eIcicleError::InvalidArgument);
            }

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

        impl BalancedDecomposition for $field_type {
            fn count_digits(base: u32) -> u32 {
                unsafe { balanced_decomposition_nof_digits(base) }
            }

            fn decompose(
                input: &(impl HostOrDeviceSlice<$field_type> + ?Sized),
                output: &mut (impl HostOrDeviceSlice<$field_type> + ?Sized),
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
                input: &(impl HostOrDeviceSlice<$field_type> + ?Sized),
                output: &mut (impl HostOrDeviceSlice<$field_type> + ?Sized),
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
    ($field_type: ident) => {
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
                check_balanced_decomposition::<$field_type>();
                test_utilities::test_set_ref_device();
                check_balanced_decomposition::<$field_type>();
            }
        }
    };
}
