use crate::field::{ScalarCfg, ScalarField};
#[cfg(not(feature = "no_ext_field"))]
use crate::field::{ExtensionCfg, ExtensionField};

use icicle_core::vec_ops::{MixedVecOps, VecOps, VecOpsConfig};
use icicle_core::{impl_vec_ops_field, impl_vec_ops_mixed_field};
use icicle_runtime::errors::eIcicleError;
use icicle_runtime::memory::HostOrDeviceSlice;

impl_vec_ops_field!("babybear", babybear, ScalarField, ScalarCfg);
#[cfg(not(feature = "no_ext_field"))]
impl_vec_ops_field!("babybear_extension", babybear_extension, ExtensionField, ExtensionCfg);
#[cfg(not(feature = "no_ext_field"))]
impl_vec_ops_mixed_field!(
    "babybear_extension",
    babybear_mixed,
    ExtensionField,
    ScalarField,
    ExtensionCfg
);

#[cfg(test)]
pub(crate) mod tests {
    #[cfg(not(feature = "no_ext_field"))]
    use crate::field::ExtensionField;
    use crate::field::ScalarField;
    use icicle_core::vec_ops::tests::*;
    use icicle_core::{impl_mixed_vec_ops_tests, impl_vec_ops_tests};

    impl_vec_ops_tests!(ScalarField);
    #[cfg(not(feature = "no_ext_field"))]
    mod extension {
        use super::*;

        impl_vec_ops_tests!(ExtensionField);
        impl_mixed_vec_ops_tests!(ExtensionField, ScalarField);
    }
}
