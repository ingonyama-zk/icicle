use crate::curve::ScalarField;
use icicle_core::{impl_vec_ops_field, vec_ops::*};
use icicle_runtime::{memory::HostOrDeviceSlice, IcicleError};

impl_vec_ops_field!("bls12_377", bls12_377, ScalarField);

#[cfg(test)]
pub(crate) mod tests {
    use crate::curve::ScalarField;
    use icicle_core::impl_vec_ops_tests;
    use icicle_core::vec_ops::tests::*;

    impl_vec_ops_tests!(bls12_377, ScalarField);
}
