use crate::traits::FieldImpl;
use icicle_runtime::{
    errors::eIcicleError,
    memory::HostOrDeviceSlice,
    program::Program,
};

use crate::vec_ops::{VecOpsConfig, VecOps};

/// Macro to implement program execution
#[macro_export]
macro_rules! impl_execute_program {
    ($field_prefix:literal, $field_prefix_ident:ident, $field:ident, $field_config:ident) => {
        mod $field_prefix_ident {
            use crate::vec_ops::{$field, HostOrDeviceSlice};
            use icicle_core::vec_ops::VecOpsConfig;
            use icicle_runtime::errors::eIcicleError;
            use icicle_runtime::program::Program;
        }

        impl VecOps<$field> for $field_config {
            fn execute_program(
                program: &Program,
                inputs: &[&(impl HostOrDeviceSlice<$field> + ?Sized)],
                outputs: &mut [&mut (impl HostOrDeviceSlice<$field> + ?Sized)],
                cfg: &VecOpsConfig,
            ) -> Result<(), eIcicleError> {
                unsafe {
                    $field_prefix_ident::execute_program_ffi(
                        program as *const Program,
                        inputs.as_ptr() as *const *const $field,
                        inputs.len() as u32,
                        outputs.as_mut_ptr() as *mut *mut $field,
                        outputs.len() as u32,
                        cfg as *const VecOpsConfig,
                    )
                    .wrap()
                }
            }
        }
    };
}

/// Macro to implement program tests
#[macro_export]
macro_rules! impl_program_tests {
    ($field_prefix_ident:ident, $field:ident) => {
        pub(crate) mod test_program {
            use super::*;
            use crate::program::$field_prefix_ident::{FieldProgram, FieldReturningValueProgram};
            use icicle_runtime::test_utilities;
            use icicle_runtime::{device::Device, runtime};
            use std::sync::Once;

            fn initialize() {
                test_utilities::test_load_and_init_devices();
                test_utilities::test_set_main_device();
            }

            #[test]
            pub fn test_program() {
                initialize();
                test_utilities::test_set_main_device();
                check_program::<$field, FieldProgram>();
                test_utilities::test_set_ref_device();
                check_program::<$field, FieldProgram>()
            }

            #[test]
            pub fn test_predefined_program() {
                initialize();
                test_utilities::test_set_main_device();
                check_predefined_program::<$field, FieldProgram>();
                test_utilities::test_set_ref_device();
                check_predefined_program::<$field, FieldProgram>()
            }
        }
    };
} 