use crate::traits::FieldImpl;
use icicle_runtime::{
    errors::eIcicleError,
    memory::HostOrDeviceSlice,
};

use super::{VecOpsConfig, VecOps, MixedVecOps};

/// Macro to implement vector operations for a field
#[macro_export]
macro_rules! impl_vec_ops_field {
    (
        $field_prefix:literal,
        $field_prefix_ident:ident,
        $field:ident,
        $field_config:ident
    ) => {
        mod $field_prefix_ident {
            use crate::vec_ops::{$field, HostOrDeviceSlice};
            use icicle_core::vec_ops::VecOpsConfig;
            use icicle_runtime::errors::eIcicleError;
        }

        impl VecOps<$field> for $field_config {
            impl_vec_add!($field_prefix, $field_prefix_ident, $field);
            impl_vec_accumulate!($field_prefix, $field_prefix_ident, $field);
            impl_vec_sub!($field_prefix, $field_prefix_ident, $field);
            impl_vec_mul!($field_prefix, $field_prefix_ident, $field);
            impl_vec_div!($field_prefix, $field_prefix_ident, $field);
            impl_vec_inv!($field_prefix, $field_prefix_ident, $field);
            impl_vec_sum!($field_prefix, $field_prefix_ident, $field);
            impl_vec_product!($field_prefix, $field_prefix_ident, $field);
            impl_scalar_add!($field_prefix, $field_prefix_ident, $field);
            impl_scalar_sub!($field_prefix, $field_prefix_ident, $field);
            impl_scalar_mul!($field_prefix, $field_prefix_ident, $field);
            impl_transpose!($field_prefix, $field_prefix_ident, $field);
            impl_bit_reverse!($field_prefix, $field_prefix_ident, $field);
            impl_bit_reverse_inplace!($field_prefix, $field_prefix_ident, $field);
            impl_slice!($field_prefix, $field_prefix_ident, $field);
        }
    };
}

/// Macro to implement mixed vector operations for different field types
#[macro_export]
macro_rules! impl_vec_ops_mixed_field {
    (
        $field_prefix:literal,
        $field_prefix_ident:ident,
        $ext_field:ident,
        $field:ident,
        $ext_field_config:ident
    ) => {
        mod $field_prefix_ident {
            use crate::vec_ops::{$ext_field, $field, HostOrDeviceSlice};
            use icicle_core::vec_ops::VecOpsConfig;
            use icicle_runtime::errors::eIcicleError;
        }

        impl MixedVecOps<$ext_field, $field> for $ext_field_config {
            fn mul(
                a: &(impl HostOrDeviceSlice<$ext_field> + ?Sized),
                b: &(impl HostOrDeviceSlice<$field> + ?Sized),
                result: &mut (impl HostOrDeviceSlice<$ext_field> + ?Sized),
                cfg: &VecOpsConfig,
            ) -> Result<(), eIcicleError> {
                unsafe {
                    $field_prefix_ident::vector_mul_ffi(
                        a.as_ptr(),
                        b.as_ptr(),
                        a.len() as u32,
                        cfg as *const VecOpsConfig,
                        result.as_mut_ptr(),
                    )
                    .wrap()
                }
            }
        }
    };
}

/// Macro to implement vector operation tests
#[macro_export]
macro_rules! impl_vec_ops_tests {
    (
      $field_prefix_ident: ident,
      $field:ident
    ) => {
        pub(crate) mod test_vecops {
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
            pub fn test_vec_ops_scalars_add() {
                initialize();
                let test_size = 1 << 14;
                check_vec_ops_scalars_add::<$field>(test_size);
            }

            #[test]
            pub fn test_vec_ops_scalars_sub() {
                initialize();
                let test_size = 1 << 14;
                check_vec_ops_scalars_sub::<$field>(test_size);
            }

            #[test]
            pub fn test_vec_ops_scalars_mul() {
                initialize();
                let test_size = 1 << 14;
                check_vec_ops_scalars_mul::<$field>(test_size);
            }

            #[test]
            pub fn test_vec_ops_scalars_div() {
                initialize();
                let test_size = 1 << 14;
                check_vec_ops_scalars_div::<$field>(test_size);
            }

            #[test]
            pub fn test_vec_ops_scalars_sum() {
                initialize();
                let test_size = 1 << 14;
                check_vec_ops_scalars_sum::<$field>(test_size);
            }

            #[test]
            pub fn test_vec_ops_scalars_product() {
                initialize();
                let test_size = 1 << 14;
                check_vec_ops_scalars_product::<$field>(test_size);
            }

            #[test]
            pub fn test_vec_ops_scalars_add_scalar() {
                initialize();
                let test_size = 1 << 14;
                check_vec_ops_scalars_add_scalar::<$field>(test_size);
            }

            #[test]
            pub fn test_vec_ops_scalars_sub_scalar() {
                initialize();
                let test_size = 1 << 14;
                check_vec_ops_scalars_sub_scalar::<$field>(test_size);
            }

            #[test]
            pub fn test_vec_ops_scalars_mul_scalar() {
                initialize();
                let test_size = 1 << 14;
                check_vec_ops_scalars_mul_scalar::<$field>(test_size);
            }

            #[test]
            pub fn test_vec_ops_scalars_accumulate() {
                initialize();
                let test_size = 1 << 14;
                check_vec_ops_scalars_accumulate::<$field>(test_size);
            }

            #[test]
            pub fn test_vec_ops_scalars_inv() {
                initialize();
                let test_size = 1 << 14;
                check_vec_ops_scalars_inv::<$field>(test_size);
            }

            #[test]
            pub fn test_matrix_transpose() {
                initialize();
                check_matrix_transpose::<$field>()
            }

            #[test]
            pub fn test_bit_reverse() {
                initialize();
                check_bit_reverse::<$field>()
            }

            #[test]
            pub fn test_bit_reverse_inplace() {
                initialize();
                check_bit_reverse_inplace::<$field>()
            }

            #[test]
            pub fn test_slice() {
                initialize();
                check_slice::<$field>()
            }
        }
    };
}

/// Macro to implement mixed vector operation tests
#[macro_export]
macro_rules! impl_mixed_vec_ops_tests {
    (
      $ext_field:ident,
      $field:ident
    ) => {
        pub(crate) mod test_mixed_vecops {
            use super::*;
            use icicle_runtime::test_utilities;
            use icicle_runtime::{device::Device, runtime};
            use std::sync::Once;

            fn initialize() {
                test_utilities::test_load_and_init_devices();
                test_utilities::test_set_main_device();
            }

            #[test]
            pub fn test_mixed_vec_ops_scalars() {
                initialize();
                check_mixed_vec_ops_scalars::<$ext_field, $field>()
            }
        }
    };
}

/// Macro to implement vector addition
#[macro_export]
macro_rules! impl_vec_add {
    ($field_prefix:literal, $field_prefix_ident:ident, $field:ident, $field_config:ident) => {
        mod $field_prefix_ident {
            use crate::vec_ops::{$field, HostOrDeviceSlice};
            use icicle_core::vec_ops::VecOpsConfig;
            use icicle_runtime::errors::eIcicleError;
        }

        impl VecOps<$field> for $field_config {
            fn add(
                a: &(impl HostOrDeviceSlice<$field> + ?Sized),
                b: &(impl HostOrDeviceSlice<$field> + ?Sized),
                result: &mut (impl HostOrDeviceSlice<$field> + ?Sized),
                cfg: &VecOpsConfig,
            ) -> Result<(), eIcicleError> {
                unsafe {
                    $field_prefix_ident::vector_add_ffi(
                        a.as_ptr(),
                        b.as_ptr(),
                        a.len() as u32,
                        cfg as *const VecOpsConfig,
                        result.as_mut_ptr(),
                    )
                    .wrap()
                }
            }
        }
    };
}

/// Macro to implement vector accumulation
#[macro_export]
macro_rules! impl_vec_accumulate {
    ($field_prefix:literal, $field_prefix_ident:ident, $field:ident, $field_config:ident) => {
        mod $field_prefix_ident {
            use crate::vec_ops::{$field, HostOrDeviceSlice};
            use icicle_core::vec_ops::VecOpsConfig;
            use icicle_runtime::errors::eIcicleError;
        }

        impl VecOps<$field> for $field_config {
            fn accumulate(
                a: &(impl HostOrDeviceSlice<$field> + ?Sized),
                result: &mut (impl HostOrDeviceSlice<$field> + ?Sized),
                cfg: &VecOpsConfig,
            ) -> Result<(), eIcicleError> {
                unsafe {
                    $field_prefix_ident::vector_accumulate_ffi(
                        a.as_ptr(),
                        a.len() as u32,
                        cfg as *const VecOpsConfig,
                        result.as_mut_ptr(),
                    )
                    .wrap()
                }
            }
        }
    };
}

/// Macro to implement vector subtraction
#[macro_export]
macro_rules! impl_vec_sub {
    ($field_prefix:literal, $field_prefix_ident:ident, $field:ident, $field_config:ident) => {
        mod $field_prefix_ident {
            use crate::vec_ops::{$field, HostOrDeviceSlice};
            use icicle_core::vec_ops::VecOpsConfig;
            use icicle_runtime::errors::eIcicleError;
        }

        impl VecOps<$field> for $field_config {
            fn sub(
                a: &(impl HostOrDeviceSlice<$field> + ?Sized),
                b: &(impl HostOrDeviceSlice<$field> + ?Sized),
                result: &mut (impl HostOrDeviceSlice<$field> + ?Sized),
                cfg: &VecOpsConfig,
            ) -> Result<(), eIcicleError> {
                unsafe {
                    $field_prefix_ident::vector_sub_ffi(
                        a.as_ptr(),
                        b.as_ptr(),
                        a.len() as u32,
                        cfg as *const VecOpsConfig,
                        result.as_mut_ptr(),
                    )
                    .wrap()
                }
            }
        }
    };
}

/// Macro to implement vector multiplication
#[macro_export]
macro_rules! impl_vec_mul {
    ($field_prefix:literal, $field_prefix_ident:ident, $field:ident, $field_config:ident) => {
        mod $field_prefix_ident {
            use crate::vec_ops::{$field, HostOrDeviceSlice};
            use icicle_core::vec_ops::VecOpsConfig;
            use icicle_runtime::errors::eIcicleError;
        }

        impl VecOps<$field> for $field_config {
            fn mul(
                a: &(impl HostOrDeviceSlice<$field> + ?Sized),
                b: &(impl HostOrDeviceSlice<$field> + ?Sized),
                result: &mut (impl HostOrDeviceSlice<$field> + ?Sized),
                cfg: &VecOpsConfig,
            ) -> Result<(), eIcicleError> {
                unsafe {
                    $field_prefix_ident::vector_mul_ffi(
                        a.as_ptr(),
                        b.as_ptr(),
                        a.len() as u32,
                        cfg as *const VecOpsConfig,
                        result.as_mut_ptr(),
                    )
                    .wrap()
                }
            }
        }
    };
}

/// Macro to implement vector division
#[macro_export]
macro_rules! impl_vec_div {
    ($field_prefix:literal, $field_prefix_ident:ident, $field:ident, $field_config:ident) => {
        mod $field_prefix_ident {
            use crate::vec_ops::{$field, HostOrDeviceSlice};
            use icicle_core::vec_ops::VecOpsConfig;
            use icicle_runtime::errors::eIcicleError;
        }

        impl VecOps<$field> for $field_config {
            fn div(
                a: &(impl HostOrDeviceSlice<$field> + ?Sized),
                b: &(impl HostOrDeviceSlice<$field> + ?Sized),
                result: &mut (impl HostOrDeviceSlice<$field> + ?Sized),
                cfg: &VecOpsConfig,
            ) -> Result<(), eIcicleError> {
                unsafe {
                    $field_prefix_ident::vector_div_ffi(
                        a.as_ptr(),
                        b.as_ptr(),
                        a.len() as u32,
                        cfg as *const VecOpsConfig,
                        result.as_mut_ptr(),
                    )
                    .wrap()
                }
            }
        }
    };
}

/// Macro to implement vector inverse
#[macro_export]
macro_rules! impl_vec_inv {
    ($field_prefix:literal, $field_prefix_ident:ident, $field:ident, $field_config:ident) => {
        mod $field_prefix_ident {
            use crate::vec_ops::{$field, HostOrDeviceSlice};
            use icicle_core::vec_ops::VecOpsConfig;
            use icicle_runtime::errors::eIcicleError;
        }

        impl VecOps<$field> for $field_config {
            fn inv(
                a: &(impl HostOrDeviceSlice<$field> + ?Sized),
                result: &mut (impl HostOrDeviceSlice<$field> + ?Sized),
                cfg: &VecOpsConfig,
            ) -> Result<(), eIcicleError> {
                unsafe {
                    $field_prefix_ident::vector_inv_ffi(
                        a.as_ptr(),
                        a.len() as u32,
                        cfg as *const VecOpsConfig,
                        result.as_mut_ptr(),
                    )
                    .wrap()
                }
            }
        }
    };
}

/// Macro to implement vector sum
#[macro_export]
macro_rules! impl_vec_sum {
    ($field_prefix:literal, $field_prefix_ident:ident, $field:ident, $field_config:ident) => {
        mod $field_prefix_ident {
            use crate::vec_ops::{$field, HostOrDeviceSlice};
            use icicle_core::vec_ops::VecOpsConfig;
            use icicle_runtime::errors::eIcicleError;
        }

        impl VecOps<$field> for $field_config {
            fn sum(
                a: &(impl HostOrDeviceSlice<$field> + ?Sized),
                result: &mut (impl HostOrDeviceSlice<$field> + ?Sized),
                cfg: &VecOpsConfig,
            ) -> Result<(), eIcicleError> {
                unsafe {
                    $field_prefix_ident::vector_sum_ffi(
                        a.as_ptr(),
                        a.len() as u32,
                        cfg as *const VecOpsConfig,
                        result.as_mut_ptr(),
                    )
                    .wrap()
                }
            }
        }
    };
}

/// Macro to implement vector product
#[macro_export]
macro_rules! impl_vec_product {
    ($field_prefix:literal, $field_prefix_ident:ident, $field:ident, $field_config:ident) => {
        mod $field_prefix_ident {
            use crate::vec_ops::{$field, HostOrDeviceSlice};
            use icicle_core::vec_ops::VecOpsConfig;
            use icicle_runtime::errors::eIcicleError;
        }

        impl VecOps<$field> for $field_config {
            fn product(
                a: &(impl HostOrDeviceSlice<$field> + ?Sized),
                result: &mut (impl HostOrDeviceSlice<$field> + ?Sized),
                cfg: &VecOpsConfig,
            ) -> Result<(), eIcicleError> {
                unsafe {
                    $field_prefix_ident::vector_product_ffi(
                        a.as_ptr(),
                        a.len() as u32,
                        cfg as *const VecOpsConfig,
                        result.as_mut_ptr(),
                    )
                    .wrap()
                }
            }
        }
    };
}

/// Macro to implement scalar addition
#[macro_export]
macro_rules! impl_scalar_add {
    ($field_prefix:literal, $field_prefix_ident:ident, $field:ident, $field_config:ident) => {
        mod $field_prefix_ident {
            use crate::vec_ops::{$field, HostOrDeviceSlice};
            use icicle_core::vec_ops::VecOpsConfig;
            use icicle_runtime::errors::eIcicleError;
        }

        impl VecOps<$field> for $field_config {
            fn scalar_add(
                a: &(impl HostOrDeviceSlice<$field> + ?Sized),
                scalar: $field,
                result: &mut (impl HostOrDeviceSlice<$field> + ?Sized),
                cfg: &VecOpsConfig,
            ) -> Result<(), eIcicleError> {
                unsafe {
                    $field_prefix_ident::scalar_add_ffi(
                        a.as_ptr(),
                        &scalar,
                        a.len() as u32,
                        cfg as *const VecOpsConfig,
                        result.as_mut_ptr(),
                    )
                    .wrap()
                }
            }
        }
    };
}

/// Macro to implement scalar subtraction
#[macro_export]
macro_rules! impl_scalar_sub {
    ($field_prefix:literal, $field_prefix_ident:ident, $field:ident, $field_config:ident) => {
        mod $field_prefix_ident {
            use crate::vec_ops::{$field, HostOrDeviceSlice};
            use icicle_core::vec_ops::VecOpsConfig;
            use icicle_runtime::errors::eIcicleError;
        }

        impl VecOps<$field> for $field_config {
            fn scalar_sub(
                a: &(impl HostOrDeviceSlice<$field> + ?Sized),
                scalar: $field,
                result: &mut (impl HostOrDeviceSlice<$field> + ?Sized),
                cfg: &VecOpsConfig,
            ) -> Result<(), eIcicleError> {
                unsafe {
                    $field_prefix_ident::scalar_sub_ffi(
                        a.as_ptr(),
                        &scalar,
                        a.len() as u32,
                        cfg as *const VecOpsConfig,
                        result.as_mut_ptr(),
                    )
                    .wrap()
                }
            }
        }
    };
}

/// Macro to implement scalar multiplication
#[macro_export]
macro_rules! impl_scalar_mul {
    ($field_prefix:literal, $field_prefix_ident:ident, $field:ident, $field_config:ident) => {
        mod $field_prefix_ident {
            use crate::vec_ops::{$field, HostOrDeviceSlice};
            use icicle_core::vec_ops::VecOpsConfig;
            use icicle_runtime::errors::eIcicleError;
        }

        impl VecOps<$field> for $field_config {
            fn scalar_mul(
                a: &(impl HostOrDeviceSlice<$field> + ?Sized),
                scalar: $field,
                result: &mut (impl HostOrDeviceSlice<$field> + ?Sized),
                cfg: &VecOpsConfig,
            ) -> Result<(), eIcicleError> {
                unsafe {
                    $field_prefix_ident::scalar_mul_ffi(
                        a.as_ptr(),
                        &scalar,
                        a.len() as u32,
                        cfg as *const VecOpsConfig,
                        result.as_mut_ptr(),
                    )
                    .wrap()
                }
            }
        }
    };
}

/// Macro to implement matrix transpose
#[macro_export]
macro_rules! impl_transpose {
    ($field_prefix:literal, $field_prefix_ident:ident, $field:ident, $field_config:ident) => {
        mod $field_prefix_ident {
            use crate::vec_ops::{$field, HostOrDeviceSlice};
            use icicle_core::vec_ops::VecOpsConfig;
            use icicle_runtime::errors::eIcicleError;
        }

        impl VecOps<$field> for $field_config {
            fn transpose(
                a: &(impl HostOrDeviceSlice<$field> + ?Sized),
                result: &mut (impl HostOrDeviceSlice<$field> + ?Sized),
                rows: u32,
                cols: u32,
                cfg: &VecOpsConfig,
            ) -> Result<(), eIcicleError> {
                unsafe {
                    $field_prefix_ident::transpose_ffi(
                        a.as_ptr(),
                        rows,
                        cols,
                        cfg as *const VecOpsConfig,
                        result.as_mut_ptr(),
                    )
                    .wrap()
                }
            }
        }
    };
}

/// Macro to implement bit reversal
#[macro_export]
macro_rules! impl_bit_reverse {
    ($field_prefix:literal, $field_prefix_ident:ident, $field:ident, $field_config:ident) => {
        mod $field_prefix_ident {
            use crate::vec_ops::{$field, HostOrDeviceSlice};
            use icicle_core::vec_ops::VecOpsConfig;
            use icicle_runtime::errors::eIcicleError;
        }

        impl VecOps<$field> for $field_config {
            fn bit_reverse(
                a: &(impl HostOrDeviceSlice<$field> + ?Sized),
                result: &mut (impl HostOrDeviceSlice<$field> + ?Sized),
                cfg: &VecOpsConfig,
            ) -> Result<(), eIcicleError> {
                unsafe {
                    $field_prefix_ident::bit_reverse_ffi(
                        a.as_ptr(),
                        a.len() as u32,
                        cfg as *const VecOpsConfig,
                        result.as_mut_ptr(),
                    )
                    .wrap()
                }
            }
        }
    };
}

/// Macro to implement in-place bit reversal
#[macro_export]
macro_rules! impl_bit_reverse_inplace {
    ($field_prefix:literal, $field_prefix_ident:ident, $field:ident, $field_config:ident) => {
        mod $field_prefix_ident {
            use crate::vec_ops::{$field, HostOrDeviceSlice};
            use icicle_core::vec_ops::VecOpsConfig;
            use icicle_runtime::errors::eIcicleError;
        }

        impl VecOps<$field> for $field_config {
            fn bit_reverse_inplace(
                a: &mut (impl HostOrDeviceSlice<$field> + ?Sized),
                cfg: &VecOpsConfig,
            ) -> Result<(), eIcicleError> {
                unsafe {
                    $field_prefix_ident::bit_reverse_inplace_ffi(
                        a.as_mut_ptr(),
                        a.len() as u32,
                        cfg as *const VecOpsConfig,
                    )
                    .wrap()
                }
            }
        }
    };
}

/// Macro to implement vector slicing
#[macro_export]
macro_rules! impl_slice {
    ($field_prefix:literal, $field_prefix_ident:ident, $field:ident, $field_config:ident) => {
        mod $field_prefix_ident {
            use crate::vec_ops::{$field, HostOrDeviceSlice};
            use icicle_core::vec_ops::VecOpsConfig;
            use icicle_runtime::errors::eIcicleError;
        }

        impl VecOps<$field> for $field_config {
            fn slice(
                a: &(impl HostOrDeviceSlice<$field> + ?Sized),
                result: &mut (impl HostOrDeviceSlice<$field> + ?Sized),
                start: u32,
                end: u32,
                cfg: &VecOpsConfig,
            ) -> Result<(), eIcicleError> {
                unsafe {
                    $field_prefix_ident::slice_ffi(
                        a.as_ptr(),
                        start,
                        end,
                        cfg as *const VecOpsConfig,
                        result.as_mut_ptr(),
                    )
                    .wrap()
                }
            }
        }
    };
}

/// Macro to implement program execution
#[macro_export]
macro_rules! impl_execute_program {
    ($field_prefix:literal, $field_prefix_ident:ident, $field:ident, $field_config:ident) => {
        mod $field_prefix_ident {
            use crate::vec_ops::{$field, HostOrDeviceSlice};
            use icicle_core::program::{Program, ProgramHandle};
            use icicle_core::vec_ops::VecOpsConfig;
            use icicle_runtime::errors::eIcicleError;
        }

        impl VecOps<$field> for $field_config {
            fn execute_program(
                program: &Program,
                result: &mut (impl HostOrDeviceSlice<$field> + ?Sized),
                cfg: &VecOpsConfig,
            ) -> Result<(), eIcicleError> {
                unsafe {
                    $field_prefix_ident::execute_program_ffi(
                        program as *const Program,
                        cfg as *const VecOpsConfig,
                        result.as_mut_ptr(),
                    )
                    .wrap()
                }
            }
        }
    };
}
