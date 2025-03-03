use crate::ring::{ScalarCfg, ScalarCfgRns, ScalarRing, ScalarRingRns};
use icicle_core::ntt::{NTTConfig, NTTDir, NTTDomain, NTTInitDomainConfig, NTT};
use icicle_core::{impl_ntt, impl_ntt_without_domain};
use icicle_runtime::errors::eIcicleError;
use icicle_runtime::memory::HostOrDeviceSlice;

impl_ntt!("labrador", labrador, ScalarRing, ScalarCfg);
impl_ntt!("labrador_rns", labrador_rns, ScalarRingRns, ScalarCfgRns);

#[cfg(test)]
pub(crate) mod tests {
    use crate::ring::{ScalarRing, ScalarRingRns};
    use icicle_core::impl_ntt_tests;
    use icicle_core::ntt::tests::*;
    use serial_test::{parallel, serial};
    impl_ntt_tests!(ScalarRing);

    mod rns {
        use super::*;
        impl_ntt_tests!(ScalarRingRns);
    }
}
