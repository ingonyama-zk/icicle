use crate::field::GoldilocksField;
use icicle_core::ntt::{NTTConfig, NTTDir, NTTDomain, NTTInitDomainConfig, NTT};
use icicle_core::{impl_ntt, impl_ntt_without_domain};
use icicle_runtime::errors::eIcicleError;
use icicle_runtime::memory::HostOrDeviceSlice;

impl_ntt!("goldilocks", goldilocks, GoldilocksField);

#[cfg(test)]
pub(crate) mod tests {
    use crate::field::GoldilocksField;
    use icicle_core::impl_ntt_tests;
    use icicle_core::ntt::tests::*;
    use serial_test::{parallel, serial};
    impl_ntt_tests!(GoldilocksField);
}
