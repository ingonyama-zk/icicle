use crate::ring::{ScalarCfg, ScalarCfgRns, ScalarRing, ScalarRingRns};

use icicle_core::impl_vec_ops_field;
use icicle_core::vec_ops::{VecOps, VecOpsConfig};
use icicle_runtime::errors::eIcicleError;
use icicle_runtime::memory::HostOrDeviceSlice;

use icicle_core::program::Program;
use icicle_core::traits::FieldImpl;

impl_vec_ops_field!("labrador", labrador, ScalarRing, ScalarCfg);
impl_vec_ops_field!("labrador_rns", labrador_rns, ScalarRingRns, ScalarCfgRns);

#[cfg(test)]
pub(crate) mod tests {
    use crate::ring::{ScalarRing, ScalarRingRns};
    use icicle_core::impl_vec_ops_tests;
    use icicle_core::vec_ops::tests::*;

    impl_vec_ops_tests!(labrador, ScalarRing);
    mod rns {
        use super::*;
        impl_vec_ops_tests!(labrador_rns, ScalarRingRns);
    }
}
