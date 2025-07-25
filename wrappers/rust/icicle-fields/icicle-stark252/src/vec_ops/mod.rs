use crate::field::ScalarField;

use icicle_core::impl_vec_ops_field;
use icicle_core::vec_ops::*;
use icicle_runtime::errors::IcicleError;
use icicle_runtime::memory::HostOrDeviceSlice;

impl_vec_ops_field!("stark252", stark252, ScalarField);

#[cfg(test)]
pub(crate) mod tests {
    use crate::field::ScalarField;
    use icicle_core::impl_vec_ops_tests;
    use icicle_core::vec_ops::tests::*;

    impl_vec_ops_tests!(stark252, ScalarField);
}
