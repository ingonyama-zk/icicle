use crate::curve::{ScalarCfg, ScalarField};
use icicle_core::{
    impl_vec_ops_field,
    vec_ops::{/*BitReverseConfig,*/ VecOps, VecOpsConfig},
};
use icicle_runtime::{errors::eIcicleError, memory::HostOrDeviceSlice};

impl_vec_ops_field!("bls12_377", bls12_377, ScalarField, ScalarCfg);
#[cfg(test)]
pub(crate) mod tests {
    use crate::curve::ScalarField;
    use icicle_core::impl_vec_ops_tests;
    use icicle_core::vec_ops::tests::*;

    impl_vec_ops_tests!(ScalarField);
}
