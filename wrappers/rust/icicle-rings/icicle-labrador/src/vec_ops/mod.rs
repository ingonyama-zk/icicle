use crate::ring::{ScalarCfg, ScalarRing};

use icicle_core::impl_vec_ops_field;
use icicle_core::vec_ops::{VecOps, VecOpsConfig};
use icicle_runtime::errors::eIcicleError;
use icicle_runtime::memory::HostOrDeviceSlice;

use icicle_core::program::Program;
use icicle_core::traits::FieldImpl;

impl_vec_ops_field!("labrador", labrador, ScalarRing, ScalarCfg);

#[cfg(test)]
pub(crate) mod tests {
    use crate::ring::ScalarRing;
    use icicle_core::impl_vec_ops_tests;
    use icicle_core::vec_ops::tests::*;

    impl_vec_ops_tests!(labrador, ScalarRing);
}
