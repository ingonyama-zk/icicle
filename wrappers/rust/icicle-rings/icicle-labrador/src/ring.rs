use icicle_core::impl_integer_ring;
use icicle_core::{impl_generate_random_ffi, impl_montgomery_convertible_ffi, vec_ops::VecOpsConfig};
use icicle_runtime::{eIcicleError, memory::HostOrDeviceSlice, stream::IcicleStream};

pub(crate) const SCALAR_LIMBS: usize = 2;

impl_integer_ring!(ScalarRing, "labrador", SCALAR_LIMBS, true, true);
impl_montgomery_convertible_ffi!(ScalarRing, labrador_scalar_convert_montgomery);
impl_generate_random_ffi!(ScalarRing, labrador_generate_scalars);

impl_integer_ring!(ScalarRingRns, "labrador_rns", SCALAR_LIMBS, true, true);
impl_montgomery_convertible_ffi!(ScalarRingRns, labrador_rns_scalar_convert_montgomery);
impl_generate_random_ffi!(ScalarRingRns, labrador_rns_generate_scalars);

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
