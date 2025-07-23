use crate::{
    bignum::BigNum,
    traits::{Arithmetic, GenerateRandom, TryInverse},
};

pub trait IntegerRing: BigNum + GenerateRandom + Arithmetic + TryInverse {}

/// Returns the modulus of the ring as a little-endian byte array.
pub fn modulus<T>() -> Vec<u8>
where
    T: IntegerRing,
{
    // Compute modulus - 1 using field arithmetic
    let minus_one = T::zero() - T::one();
    let mut bytes = minus_one.to_bytes_le();

    // Add 1 to get the actual modulus (modulus = modulus - 1 + 1)
    let mut carry = 1u8;
    for byte in &mut bytes {
        let (sum, new_carry) = byte.overflowing_add(carry);
        *byte = sum;
        carry = new_carry as u8;
        if carry == 0 {
            break;
        }
    }

    // If there's still a carry, append it
    if carry != 0 {
        bytes.push(carry);
    }

    bytes
}

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
