use crate::curve::{ScalarCfg, ScalarField};

use icicle_core::ntt::{NTTConfig, NTTDir, NTTDomain, NTTInitDomainConfig, NTT};
use icicle_core::{impl_ntt, impl_ntt_without_domain};
use icicle_runtime::{eIcicleError, memory::HostOrDeviceSlice, IcicleError};

impl_ntt!("bn254", bn254, ScalarField, ScalarCfg);

#[cfg(test)]
pub(crate) mod tests {
    use crate::curve::ScalarField;
    use icicle_core::impl_ntt_tests;
    use icicle_core::ntt::tests::*;
    use serial_test::{parallel, serial};

    impl_ntt_tests!(ScalarField);
}
