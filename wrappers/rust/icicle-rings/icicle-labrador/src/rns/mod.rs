use crate::ring::{LabradorScalarRing, LabradorScalarRingRns};
use icicle_core::{impl_rns_conversions, vec_ops::VecOpsConfig};
use icicle_runtime::errors::eIcicleError;
use icicle_runtime::memory::HostOrDeviceSlice;

impl_rns_conversions!("labrador", LabradorScalarRing, LabradorScalarRingRns);

#[cfg(test)]
pub(crate) mod tests {
    use crate::ring::{LabradorScalarRing, LabradorScalarRingRns};
    use icicle_core::impl_rns_conversions_tests;
    use icicle_core::rns::tests::*;
    impl_rns_conversions_tests!(LabradorScalarRing, LabradorScalarRingRns);
}
