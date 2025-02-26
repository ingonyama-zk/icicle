use crate::field::{ExtensionCfg, ExtensionField, ScalarCfg, ScalarField};

use icicle_core::vec_ops::{MixedVecOps, VecOps, VecOpsConfig};
use icicle_core::{impl_vec_ops_field, impl_vec_ops_mixed_field};
use icicle_runtime::errors::eIcicleError;
use icicle_runtime::memory::HostOrDeviceSlice;

use icicle_core::traits::FieldImpl;
use icicle_core::program::Program;

impl_vec_ops_field!("m31", m31, ScalarField, ScalarCfg);
impl_vec_ops_field!("m31_extension", m31_extension, ExtensionField, ExtensionCfg);
impl_vec_ops_mixed_field!("m31_extension", m31_mixed, ExtensionField, ScalarField, ExtensionCfg);

#[cfg(test)]
pub(crate) mod tests {
    use crate::field::{ExtensionField, ScalarField};
    use icicle_core::vec_ops::tests::*;
    use icicle_core::{impl_mixed_vec_ops_tests, impl_vec_ops_tests};

    impl_vec_ops_tests!(m31, ScalarField);

    mod extension {
        use super::*;

        impl_vec_ops_tests!(m31_extension, ExtensionField);
        impl_mixed_vec_ops_tests!(ExtensionField, ScalarField);
    }
}
