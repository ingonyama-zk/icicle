use crate::field::{M31ExtensionField, M31Field};

use icicle_core::vec_ops::{MixedVecOps, VecOps, VecOpsConfig};
use icicle_core::{impl_vec_ops_field, impl_vec_ops_mixed_field};
use icicle_runtime::errors::eIcicleError;
use icicle_runtime::memory::HostOrDeviceSlice;

use icicle_core::program::Program;

impl_vec_ops_field!("m31", m31, M31Field);
impl_vec_ops_field!("m31_extension", m31_extension, M31ExtensionField);
impl_vec_ops_mixed_field!("m31_extension", m31_mixed, M31ExtensionField, M31Field);

#[cfg(test)]
pub(crate) mod tests {
    use crate::field::{M31ExtensionField, M31Field};
    use icicle_core::vec_ops::tests::*;
    use icicle_core::{impl_mixed_vec_ops_tests, impl_vec_ops_tests};

    impl_vec_ops_tests!(m31, M31Field);

    mod extension {
        use super::*;

        impl_vec_ops_tests!(m31_extension, M31ExtensionField);
        impl_mixed_vec_ops_tests!(M31ExtensionField, M31Field);
    }
}
