use crate::field::{ExtensionCfg, ExtensionField, ScalarCfg, ScalarField};

use icicle_core::impl_vec_ops_field;
use icicle_core::vec_ops::{VecOps, VecOpsConfig};
use icicle_runtime::errors::eIcicleError;
use icicle_runtime::memory::HostOrDeviceSlice;

impl_vec_ops_field!("m31", m31, ScalarField, ScalarCfg);
impl_vec_ops_field!("m31_extension", m31_extension, ExtensionField, ExtensionCfg);

#[cfg(test)]
pub(crate) mod tests {
    use crate::field::{ExtensionField, ScalarField};
    use icicle_core::impl_vec_ops_tests;
    use icicle_core::vec_ops::tests::*;

    impl_vec_ops_tests!(ScalarField);

    mod extension {
        use super::*;

        impl_vec_ops_tests!(ExtensionField);
    }
}

