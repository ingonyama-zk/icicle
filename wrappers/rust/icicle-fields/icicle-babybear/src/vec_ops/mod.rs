use crate::field::{ExtensionCfg, ExtensionField, ScalarCfg, ScalarField};

use icicle_core::vec_ops::{CrossVecOps, VecOps, VecOpsConfig};
use icicle_core::{impl_vec_ops_cross_field, impl_vec_ops_field};
use icicle_runtime::errors::eIcicleError;
use icicle_runtime::memory::HostOrDeviceSlice;

impl_vec_ops_field!("babybear", babybear, ScalarField, ScalarCfg);
impl_vec_ops_field!("babybear_extension", babybear_extension, ExtensionField, ExtensionCfg);
impl_vec_ops_cross_field!(
    "babybear_extension",
    babybear_cross,
    ExtensionField,
    ScalarField,
    ExtensionCfg
);

#[cfg(test)]
pub(crate) mod tests {
    use crate::field::{ExtensionField, ScalarField};
    use icicle_core::vec_ops::tests::*;
    use icicle_core::{impl_cross_vec_ops_tests, impl_vec_ops_tests};

    impl_vec_ops_tests!(ScalarField);

    mod extension {
        use super::*;

        impl_vec_ops_tests!(ExtensionField);
        impl_cross_vec_ops_tests!(ExtensionField, ScalarField);
    }
}
