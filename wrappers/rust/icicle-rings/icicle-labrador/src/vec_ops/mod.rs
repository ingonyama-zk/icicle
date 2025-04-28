use crate::ring::{LabradorScalarRing, LabradorScalarRingRns};

use icicle_core::impl_vec_ops_field;
use icicle_core::vec_ops::{VecOps, VecOpsConfig};
use icicle_runtime::errors::eIcicleError;
use icicle_runtime::memory::HostOrDeviceSlice;

use icicle_core::program::Program;

impl_vec_ops_field!("labrador", labrador, LabradorScalarRing);
impl_vec_ops_field!("labrador_rns", labrador_rns, LabradorScalarRingRns);

#[cfg(test)]
pub(crate) mod tests {
    use crate::ring::{LabradorScalarRing, LabradorScalarRingRns};
    use icicle_core::impl_vec_ops_tests;
    use icicle_core::vec_ops::tests::*;

    impl_vec_ops_tests!(labrador, LabradorScalarRing);
    mod rns {
        use super::*;
        impl_vec_ops_tests!(labrador_rns, LabradorScalarRingRns);
    }
}
