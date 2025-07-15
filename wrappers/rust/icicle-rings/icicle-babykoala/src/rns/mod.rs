use crate::ring::{ScalarRing, ScalarRingRns};
use icicle_core::{impl_rns_conversions, vec_ops::VecOpsConfig};
use icicle_runtime::errors::eIcicleError;
use icicle_runtime::memory::HostOrDeviceSlice;

impl_rns_conversions!("babykoala", ScalarRing, ScalarRingRns);

#[cfg(test)]
pub(crate) mod tests {
    use crate::ring::{ScalarRing, ScalarRingRns};
    use icicle_core::impl_rns_conversions_tests;
    use icicle_core::rns::tests::*;
    impl_rns_conversions_tests!(ScalarRing, ScalarRingRns);
}
