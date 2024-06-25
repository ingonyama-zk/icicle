use crate::field::{ScalarCfg, ScalarField};
use icicle_core::ntt::{NTTConfig, NTTDir, NTTDomain, NTTInitDomainConfig, NTT};
use icicle_core::{impl_ntt, impl_ntt_without_domain};
use icicle_runtime::errors::eIcicleError;
use icicle_runtime::memory::HostOrDeviceSlice;

impl_ntt!("stark252", stark252, ScalarField, ScalarCfg);

#[cfg(test)]
pub(crate) mod tests {
    use crate::field::ScalarField;
    use icicle_core::impl_ntt_tests;
    use icicle_core::ntt::tests::*;
    use serial_test::{parallel, serial};
    use std::sync::OnceLock;

    impl_ntt_tests!(ScalarField);
}

// TODO Yuval : V2 has tests against lambdaworks, do we still need it?
// UPDATE: team decided to keep it
