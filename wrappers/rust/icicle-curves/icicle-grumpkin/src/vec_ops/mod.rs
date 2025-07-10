use crate::curve::ScalarField;
use icicle_core::{impl_vec_ops_field, vec_ops::*};
use icicle_runtime::{errors::IcicleError, memory::HostOrDeviceSlice};

impl_vec_ops_field!("grumpkin", grumpkin, ScalarField);
#[cfg(test)]
pub(crate) mod tests {
    use crate::curve::ScalarField;
    use icicle_core::impl_vec_ops_tests;
    use icicle_core::vec_ops::tests::*;

    impl_vec_ops_tests!(grumpkin, ScalarField);
}
