use crate::curve::Bls12_381ScalarField;

use icicle_core::ntt::{NTTConfig, NTTDir, NTTDomain, NTTInitDomainConfig, NTT};
use icicle_core::{impl_ntt, impl_ntt_without_domain};
use icicle_runtime::{errors::eIcicleError, memory::HostOrDeviceSlice};

impl_ntt!("bls12_381", bls12_381, Bls12_381ScalarField);

#[cfg(test)]
pub(crate) mod tests {
    use crate::curve::Bls12_381ScalarField;
    use icicle_core::impl_ntt_tests;
    use icicle_core::ntt::tests::*;
    use serial_test::{parallel, serial};

    impl_ntt_tests!(Bls12_381ScalarField);
}
