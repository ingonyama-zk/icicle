use crate::field::{ExtensionField, ScalarCfg, ScalarField};
use icicle_core::ntt::{NTTConfig, NTTDir, NTTDomain, NTTInitDomainConfig, NTT};
use icicle_core::{impl_ntt, impl_ntt_without_domain};
use icicle_runtime::errors::eIcicleError;
use icicle_runtime::memory::HostOrDeviceSlice;

impl_ntt!("babybear", babybear, ScalarField, ScalarCfg);
impl_ntt_without_domain!(
    "babybear_extension",
    ScalarField,
    ScalarCfg,
    NTT,
    "_ntt",
    ExtensionField
);

#[cfg(test)]
pub(crate) mod tests {
    use crate::field::ScalarField;
    use icicle_core::impl_ntt_tests;
    use icicle_core::ntt::tests::*;
    use serial_test::{parallel, serial};
    use std::sync::OnceLock;

    impl_ntt_tests!(ScalarField);
}

// TODO Yuval : V2 has tests against plonky3, do we still need it?
// Note that the NTT tests could not work for babybear since they rely on arkworks which is not implementing babybear
// UPDATE: team decided to keep it
