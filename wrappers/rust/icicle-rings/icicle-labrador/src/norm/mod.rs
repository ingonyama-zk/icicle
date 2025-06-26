use crate::ring::{ScalarCfg, ScalarRing};
use icicle_core::norm::NormType;
use icicle_core::{impl_norm, vec_ops::VecOpsConfig};
use icicle_runtime::errors::eIcicleError;
use icicle_runtime::memory::HostOrDeviceSlice;

impl_norm!("labrador", ScalarRing, ScalarCfg);

#[cfg(test)]
pub(crate) mod tests {
    use crate::ring::ScalarRing;
    use icicle_core::impl_norm_tests;

    impl_norm_tests!(ScalarRing);
}
