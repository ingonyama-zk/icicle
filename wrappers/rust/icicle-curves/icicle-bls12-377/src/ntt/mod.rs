#[cfg(feature = "bw6-761")]
use crate::curve::Bls12_377BaseField;
use crate::curve::Bls12_377ScalarField;

use icicle_core::ntt::{NTTConfig, NTTDir, NTTDomain, NTTInitDomainConfig, NTT};
use icicle_core::{impl_ntt, impl_ntt_without_domain};
use icicle_runtime::{errors::eIcicleError, memory::HostOrDeviceSlice};

impl_ntt!("bls12_377", bls12_377, Bls12_377ScalarField);
#[cfg(feature = "bw6-761")]
impl_ntt!("bw6_761", bw6_761, Bls12_377BaseField);

#[cfg(test)]
pub(crate) mod tests {
    use crate::curve::Bls12_377ScalarField;
    use icicle_core::impl_ntt_tests;
    use icicle_core::ntt::tests::*;
    use serial_test::{parallel, serial};

    impl_ntt_tests!(Bls12_377ScalarField);
}
