use crate::{ring::IntegerRing, traits::Invertible};

pub trait Field: IntegerRing + Invertible {}

#[macro_export]
macro_rules! impl_field {
    (
        $field:ident,
        $field_prefix:literal,
        $num_limbs:ident
    ) => {
        icicle_core::impl_integer_ring!($field, $field_prefix, $num_limbs);
        icicle_core::impl_invertible!($field, $field_prefix);
        impl icicle_core::field::Field for $field {}
    };
}

#[macro_export]
macro_rules! impl_field_tests {
    (
        $field:ident
    ) => {
        pub mod test_field {
            use super::*;
            use icicle_runtime::test_utilities;

            fn initialize() {
                test_utilities::test_load_and_init_devices();
                test_utilities::test_set_main_device();
            }

            #[test]
            fn test_field_convert_montgomery() {
                initialize();
                icicle_core::tests::check_montgomery_convert_host::<$field>();
                icicle_core::tests::check_montgomery_convert_device::<$field>();
            }

            #[test]
            fn test_field_arithmetic() {
                icicle_core::tests::check_field_arithmetic::<$field>();
            }
        }
    };
}
