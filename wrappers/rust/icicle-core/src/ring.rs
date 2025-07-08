use crate::{
    bignum::BigNum,
    traits::{Arithmetic, GenerateRandom, TryInverse},
};

pub trait IntegerRing: BigNum + GenerateRandom + Arithmetic + TryInverse {}

#[macro_export]
macro_rules! impl_integer_ring {
    (
        $ring:ident,
        $ring_prefix:literal,
        $num_limbs:ident
    ) => {
        icicle_core::impl_bignum!($ring, $ring_prefix, $num_limbs);

        impl icicle_core::ring::IntegerRing for $ring {}

        icicle_core::impl_arithmetic!($ring, $ring_prefix);
        icicle_core::impl_try_inverse!($ring, $ring_prefix);
        icicle_core::impl_generate_random!($ring, concat!($ring_prefix, "_generate_random"));
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
