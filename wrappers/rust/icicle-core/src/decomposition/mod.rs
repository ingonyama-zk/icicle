// (1) trait for BalancedDecomposition
// (2) floating functions to call the API (for consistency with ICICLE)
// (3) macro to implement the trait for a given type via C FFI
//          - Implemented for Field::Config
// (4) macro to test decomposition

use crate::{traits::FieldImpl, vec_ops::VecOpsConfig};
use icicle_runtime::{eIcicleError, memory::HostOrDeviceSlice};

pub mod tests;

pub trait BalancedDecomposition<T: FieldImpl> {
    fn compute_nof_digits(base: u32) -> u32;

    fn decompose(
        input: &(impl HostOrDeviceSlice<T> + ?Sized),
        output: &mut (impl HostOrDeviceSlice<T> + ?Sized),
        base: u32,
        cfg: &VecOpsConfig,
    ) -> Result<(), eIcicleError>;

    fn recompose(
        input: &(impl HostOrDeviceSlice<T> + ?Sized),
        output: &mut (impl HostOrDeviceSlice<T> + ?Sized),
        base: u32,
        cfg: &VecOpsConfig,
    ) -> Result<(), eIcicleError>;
}

#[macro_export]
macro_rules! impl_balanced_decomposition {
    (
        $field_prefix: literal,
        $field_type: ident,
        $field_cfg_type: ident
    ) => {
        use icicle_core::decomposition::BalancedDecomposition;

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
            // make sure batch divides input and output sizes
            let batch_divides_input_size = input.len() % (cfg.batch_size as usize) == 0;
            let batch_divides_output_size = input.len() % (cfg.batch_size as usize) == 0;
            if !batch_divides_input_size || !batch_divides_output_size {
                eprintln!(
                    "Batch-size={} does not divide input size={} or output size={}",
                    cfg.batch_size,
                    input.len(),
                    output.len()
                );
                return Err(eIcicleError::InvalidArgument);
            }

            // Ensure input/output are on the active device
            if input.is_on_device() && !input.is_on_active_device() {
                eprintln!("Input is allocated on an inactive device.");
                return Err(eIcicleError::InvalidArgument);
            }
            if output.is_on_device() && !output.is_on_active_device() {
                eprintln!("Output is allocated on an inactive device.");
                return Err(eIcicleError::InvalidArgument);
            }

            cfg.is_a_on_device = input.is_on_device();
            cfg.is_result_on_device = output.is_on_device();

            Ok(())
        }

        impl BalancedDecomposition<$field_type> for $field_cfg_type {
            fn compute_nof_digits(base: u32) -> u32 {
                unsafe { balanced_decomposition_nof_digits(base) }
            }

            fn decompose(
                input: &(impl HostOrDeviceSlice<$field_type> + ?Sized),
                output: &mut (impl HostOrDeviceSlice<$field_type> + ?Sized),
                base: u32,
                cfg: &VecOpsConfig,
            ) -> Result<(), eIcicleError> {
                let mut cfg_clone = cfg.clone();
                balanced_decomposition_check_args(input, output, &mut cfg_clone)?;

                unsafe {
                    decompose_balanced_digits(
                        input.as_ptr(),
                        (input.len() / (cfg_clone.batch_size as usize)) as usize,
                        base,
                        &cfg_clone as *const VecOpsConfig,
                        output.as_mut_ptr(),
                        (output.len() / (cfg_clone.batch_size as usize)) as usize,
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
                let mut cfg_clone = cfg.clone();
                balanced_decomposition_check_args(input, output, &mut cfg_clone)?;

                unsafe {
                    recompose_from_balanced_digits(
                        input.as_ptr(),
                        (input.len() / (cfg_clone.batch_size as usize)) as usize,
                        base,
                        &cfg_clone as *const VecOpsConfig,
                        output.as_mut_ptr(),
                        (output.len() / (cfg_clone.batch_size as usize)) as usize,
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
        use icicle_core::decomposition::tests::*;
        use icicle_runtime::{device::Device, runtime, test_utilities};

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
