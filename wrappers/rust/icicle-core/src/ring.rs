use crate::{bignum::BigNum, traits::Arithmetic};

pub trait IntegerRing: BigNum + From<u32> + Arithmetic {
    fn one() -> Self {
        Self::from(1)
    }

    fn from_u32(val: u32) -> Self {
        Self::from(val)
    }
}

#[macro_export]
macro_rules! impl_integer_ring {
    (
        $ring:ident,
        $ring_prefix:literal,
        $num_limbs:ident,
        $use_ffi_for_eq:expr,
        $use_ffi_for_from_u32:expr
    ) => {
        icicle_core::impl_bignum!($ring, $ring_prefix, $num_limbs, $use_ffi_for_eq);

        impl icicle_core::ring::IntegerRing for $ring {}

        impl From<u32> for $ring {
            fn from(val: u32) -> Self {
                if $use_ffi_for_from_u32 {
                    extern "C" {
                        #[link_name = concat!($ring_prefix, "_from_u32")]
                        pub(crate) fn from_u32(val: u32, result: *mut $ring);
                    }

                    let mut limbs = [0u32; $num_limbs];

                    unsafe {
                        from_u32(val, limbs.as_mut_ptr() as *mut $ring);
                    }

                    Self { limbs }
                } else {
                    let mut limbs = [0u32; $num_limbs];
                    limbs[0] = val;
                    Self { limbs }
                }
            }
        }

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
