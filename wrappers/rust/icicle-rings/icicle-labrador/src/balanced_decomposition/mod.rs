use crate::ring::{ScalarCfg, ScalarRing};
use icicle_core::{impl_balanced_decomposition, vec_ops::VecOpsConfig};
use icicle_runtime::errors::eIcicleError;
use icicle_runtime::memory::HostOrDeviceSlice;

impl_balanced_decomposition!("labrador", ScalarRing, ScalarCfg);

#[cfg(test)]
pub(crate) mod tests {
    use crate::ring::ScalarRing;
    use icicle_core::impl_balanced_decomposition_tests;

    impl_balanced_decomposition_tests!(ScalarRing);
}
