use crate::field::{BabybearExtensionField, BabybearField};

use icicle_core::vec_ops::{MixedVecOps, VecOps, VecOpsConfig};
use icicle_core::{impl_vec_ops_field, impl_vec_ops_mixed_field};
use icicle_runtime::errors::eIcicleError;
use icicle_runtime::memory::HostOrDeviceSlice;

use icicle_core::field::PrimeField;
use icicle_core::program::Program;

impl_vec_ops_field!("babybear", babybear, BabybearField);
impl_vec_ops_field!("babybear_extension", babybear_extension, BabybearExtensionField);
impl_vec_ops_mixed_field!(
    "babybear_extension",
    babybear_mixed,
    BabybearExtensionField,
    BabybearField
);

#[cfg(test)]
pub(crate) mod tests {
    use crate::field::{BabybearExtensionField, BabybearField};
    use icicle_core::vec_ops::tests::*;
    use icicle_core::{impl_mixed_vec_ops_tests, impl_vec_ops_tests};

    impl_vec_ops_tests!(babybear, BabybearField);

    mod extension {
        use super::*;

        impl_vec_ops_tests!(babybear_extension, BabybearExtensionField);
        impl_mixed_vec_ops_tests!(BabybearExtensionField, BabybearField);
    }
}
