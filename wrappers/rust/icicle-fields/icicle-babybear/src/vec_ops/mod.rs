use crate::field::{ExtensionField, ScalarField};

use icicle_core::vec_ops::*;
use icicle_core::{impl_vec_ops_field, impl_vec_ops_mixed_field};
use icicle_runtime::errors::IcicleError;
use icicle_runtime::memory::HostOrDeviceSlice;

impl_vec_ops_field!("babybear", babybear, ScalarField);
impl_vec_ops_field!("babybear_extension", babybear_extension, ExtensionField);
impl_vec_ops_mixed_field!("babybear_extension", babybear_mixed, ExtensionField, ScalarField);

#[cfg(test)]
pub(crate) mod tests {
    use crate::field::{ExtensionField, ScalarField};
    use icicle_core::vec_ops::tests::*;
    use icicle_core::{impl_mixed_vec_ops_tests, impl_vec_ops_tests};

    impl_vec_ops_tests!(babybear, ScalarField);

    mod extension {
        use super::*;

        impl_vec_ops_tests!(babybear_extension, ExtensionField);
        impl_mixed_vec_ops_tests!(ExtensionField, ScalarField);
    }
}
