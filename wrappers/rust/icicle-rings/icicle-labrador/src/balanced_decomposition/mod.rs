use crate::ring::ScalarRing;
use icicle_core::balanced_decomposition::BalancedDecomposition;
use icicle_core::impl_balanced_decomposition;
use icicle_core::vec_ops::VecOpsConfig;
use icicle_runtime::errors::eIcicleError;
use icicle_runtime::memory::HostOrDeviceSlice;

impl_balanced_decomposition!("labrador", ScalarRing);

#[cfg(test)]
pub(crate) mod tests {
    use crate::ring::ScalarRing;
    use icicle_core::balanced_decomposition::tests::*;
    use icicle_core::impl_balanced_decomposition_tests;
    impl_balanced_decomposition_tests!(ScalarRing);
}
