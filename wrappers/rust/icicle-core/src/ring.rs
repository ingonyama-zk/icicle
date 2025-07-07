use crate::{bignum::BigNum, traits::Arithmetic};

pub trait IntegerRing: BigNum + Arithmetic {}

#[macro_export]
macro_rules! impl_integer_ring {
    (
        $ring:ident,
        $ring_prefix:literal,
        $num_limbs:ident,
        $use_ffi_for_eq:expr,
        $use_ffi_for_from_u32:expr
    ) => {
        icicle_core::impl_bignum!($ring, $ring_prefix, $num_limbs, $use_ffi_for_eq, $use_ffi_for_from_u32);

        impl icicle_core::ring::IntegerRing for $ring {}

        icicle_core::impl_arithmetic!($ring, $ring_prefix);
    };
}

#[macro_export]
macro_rules! impl_integer_ring_tests {
    (
        $ring:ident
    ) => {
        pub mod test_integer_ring {
            use super::*;
            use icicle_runtime::test_utilities;

            fn initialize() {
                test_utilities::test_load_and_init_devices();
                test_utilities::test_set_main_device();
            }

            #[test]
            fn test_integer_ring_convert_montgomery() {
                initialize();
                icicle_core::tests::check_montgomery_convert_host::<$ring>();
                icicle_core::tests::check_montgomery_convert_device::<$ring>();
            }

            #[test]
            fn test_integer_ring_arithmetic() {
                initialize();
                icicle_core::tests::check_ring_arithmetic::<$ring>();
            }
        }
    };
}
