use icicle_core::bignum::BigNum;
use icicle_core::{impl_integer_ring, impl_montgomery_convertible};
use icicle_runtime::{eIcicleError, memory::HostOrDeviceSlice, stream::IcicleStream};

pub(crate) const SCALAR_LIMBS: usize = 2;

impl_integer_ring!(ScalarRing, "babykoala", SCALAR_LIMBS);
impl_montgomery_convertible!(ScalarRing, "babykoala_scalar_convert_montgomery");

impl_integer_ring!(ScalarRingRns, "babykoala_rns", SCALAR_LIMBS);
impl_montgomery_convertible!(ScalarRingRns, "babykoala_rns_scalar_convert_montgomery");

#[cfg(test)]
mod tests {
    use super::{ScalarRing, ScalarRingRns};
    use icicle_core::impl_integer_ring_tests;

    impl_integer_ring_tests!(ScalarRing);
    mod rns {
        use super::*;
        impl_integer_ring_tests!(ScalarRingRns);
    }
}
