#[cfg(feature = "bw6-761")]
use crate::curve::BaseField;
use crate::curve::ScalarField;

use icicle_core::ntt::{NTTConfig, NTTDir, NTTDomain, NTTInitDomainConfig, NTT};
use icicle_core::traits::FieldImpl;
use icicle_core::{impl_ntt, impl_ntt_without_domain};
use icicle_runtime::{errors::eIcicleError, memory::HostOrDeviceSlice};

impl_ntt!("bls12_377", bls12_377, ScalarField);
#[cfg(feature = "bw6-761")]
impl_ntt!("bw6_761", bw6_761, BaseField);

#[cfg(test)]
pub(crate) mod tests {
    use crate::curve::ScalarField;
    use icicle_core::impl_ntt_tests;
    use icicle_core::ntt::tests::*;
    use serial_test::{parallel, serial};

    impl_ntt_tests!(ScalarField);
}
