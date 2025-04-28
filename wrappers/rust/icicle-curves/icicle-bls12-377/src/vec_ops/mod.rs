#[cfg(feature = "bw6-761")]
use crate::curve::Bls12_377BaseField;
use crate::curve::Bls12_377ScalarField;
use icicle_core::{
    impl_vec_ops_field,
    vec_ops::{/*BitReverseConfig,*/ VecOps, VecOpsConfig},
};
use icicle_runtime::{errors::eIcicleError, memory::HostOrDeviceSlice};

use icicle_core::program::Program;

impl_vec_ops_field!("bls12_377", bls12_377, Bls12_377ScalarField);
#[cfg(feature = "bw6-761")]
impl_vec_ops_field!("bw6_761", bw6_761, Bls12_377BaseField);

#[cfg(test)]
pub(crate) mod tests {
    use crate::curve::Bls12_377ScalarField;
    use icicle_core::impl_vec_ops_tests;
    use icicle_core::vec_ops::tests::*;

    impl_vec_ops_tests!(bls12_377, Bls12_377ScalarField);
}
