use crate::field::{KoalabearExtensionField, KoalabearField};
use icicle_core::ntt::{NTTConfig, NTTDir, NTTDomain, NTTInitDomainConfig, NTT};
use icicle_core::{impl_ntt, impl_ntt_without_domain};
use icicle_runtime::errors::eIcicleError;
use icicle_runtime::memory::HostOrDeviceSlice;

impl_ntt!("koalabear", koalabear, KoalabearField);
impl_ntt_without_domain!(
    "koalabear_extension",
    KoalabearField,
    NTT,
    "_ntt",
    KoalabearExtensionField
);

#[cfg(test)]
pub(crate) mod tests {
    use crate::field::KoalabearField;
    use icicle_core::impl_ntt_tests;
    use icicle_core::ntt::tests::*;
    use serial_test::{parallel, serial};
    impl_ntt_tests!(KoalabearField);
}
