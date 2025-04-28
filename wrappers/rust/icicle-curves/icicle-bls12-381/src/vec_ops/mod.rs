use crate::curve::Bls12_381ScalarField;
use icicle_core::{
    impl_vec_ops_field,
    vec_ops::{/*BitReverseConfig,*/ VecOps, VecOpsConfig},
};
use icicle_runtime::{errors::eIcicleError, memory::HostOrDeviceSlice};

use icicle_core::program::Program;

impl_vec_ops_field!("bls12_381", bls12_381, Bls12_381ScalarField);
#[cfg(test)]
pub(crate) mod tests {
    use crate::curve::Bls12_381ScalarField;
    use icicle_core::impl_vec_ops_tests;
    use icicle_core::vec_ops::tests::*;

    impl_vec_ops_tests!(bls12_381, Bls12_381ScalarField);
}
