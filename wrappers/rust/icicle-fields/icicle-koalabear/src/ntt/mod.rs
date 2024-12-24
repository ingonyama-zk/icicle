use crate::field::{ExtensionField, ScalarCfg, ScalarField};
use icicle_core::ntt::{NTTConfig, NTTDir, NTTDomain, NTTInitDomainConfig, NTT};
use icicle_core::{impl_ntt, impl_ntt_without_domain};
use icicle_runtime::errors::eIcicleError;
use icicle_runtime::memory::HostOrDeviceSlice;

impl_ntt!("koalabear", koalabear, ScalarField, ScalarCfg);
impl_ntt_without_domain!(
    "koalabear_extension",
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
    impl_ntt_tests!(ScalarField);

    // Tests against risc0 and plonky3
    use super::ExtensionField;
    use icicle_core::{
        ntt::{initialize_domain, ntt_inplace, release_domain, NTTConfig, NTTDir, NTTInitDomainConfig},
        traits::{FieldImpl, GenerateRandom},
    };
    use icicle_runtime::memory::HostSlice;
    use risc0_core::field::{
        baby_bear::{Elem, ExtElem},
        Elem as FieldElem, RootsOfUnity,
    };
}
