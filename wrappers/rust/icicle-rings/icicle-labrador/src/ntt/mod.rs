use crate::ring::{LabradorScalarRing, LabradorScalarRingRns};
use icicle_core::ntt::{NTTConfig, NTTDir, NTTDomain, NTTInitDomainConfig, NTT};
use icicle_core::{impl_ntt, impl_ntt_without_domain};
use icicle_runtime::errors::eIcicleError;
use icicle_runtime::memory::HostOrDeviceSlice;

impl_ntt!("labrador", labrador, LabradorScalarRing);
impl_ntt!("labrador_rns", labrador_rns, LabradorScalarRingRns);

#[cfg(test)]
pub(crate) mod tests {
    use crate::ring::{LabradorScalarRing, LabradorScalarRingRns};
    use icicle_core::impl_ntt_tests;
    use icicle_core::ntt::tests::*;
    use serial_test::{parallel, serial};
    impl_ntt_tests!(LabradorScalarRing);

    mod rns {
        use super::*;
        impl_ntt_tests!(LabradorScalarRingRns);
    }
}
