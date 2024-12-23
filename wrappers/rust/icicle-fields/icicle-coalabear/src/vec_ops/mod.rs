use crate::field::{ScalarCfg, ScalarField};

use icicle_core::impl_vec_ops_field;
use icicle_core::vec_ops::{VecOps, VecOpsConfig};
use icicle_runtime::errors::eIcicleError;
use icicle_runtime::memory::HostOrDeviceSlice;

impl_vec_ops_field!("coalabear", coalabear, ScalarField, ScalarCfg);

#[cfg(test)]
pub(crate) mod tests {
    use crate::field::ScalarField;
    use icicle_core::impl_vec_ops_tests;
    use icicle_core::vec_ops::tests::*;

    impl_vec_ops_tests!(ScalarField);
}
