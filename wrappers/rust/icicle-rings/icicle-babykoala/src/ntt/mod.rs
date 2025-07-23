use crate::polynomial_ring::PolyRing;
use crate::ring::{ScalarRing, ScalarRingRns};
use icicle_core::ntt::{NTTConfig, NTTDir, NTTDomain, NTTInitDomainConfig, NTT};
use icicle_core::{
    impl_negacyclic_ntt,
    negacyclic_ntt::{NegacyclicNtt, NegacyclicNttConfig},
};
use icicle_core::{impl_ntt, impl_ntt_without_domain};
use icicle_runtime::{eIcicleError, memory::HostOrDeviceSlice, IcicleError};

impl_ntt!("babykoala", babykoala, ScalarRing);
impl_ntt!("babykoala_rns", babykoala_rns, ScalarRingRns);
impl_negacyclic_ntt!("babykoala", PolyRing);

#[cfg(test)]
pub(crate) mod tests {
    use crate::polynomial_ring::PolyRing;
    use crate::ring::{ScalarRing, ScalarRingRns};
    use icicle_core::ntt::tests::*;
    use icicle_core::{impl_negacyclic_ntt_tests, impl_ntt_tests};
    use serial_test::{parallel, serial};

    impl_ntt_tests!(ScalarRing);
    impl_negacyclic_ntt_tests!(PolyRing);

    mod rns {
        use super::*;
        impl_ntt_tests!(ScalarRingRns);
    }
}
