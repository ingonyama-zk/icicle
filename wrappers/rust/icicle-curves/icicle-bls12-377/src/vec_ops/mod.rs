#[cfg(feature = "bw6-761")]
use crate::curve::BaseField;
use crate::curve::ScalarField;
use icicle_core::{
    impl_vec_ops_field,
    vec_ops::{/*BitReverseConfig,*/ VecOps, VecOpsConfig},
};
use icicle_runtime::{errors::eIcicleError, memory::HostOrDeviceSlice};

impl_vec_ops_field!("bls12_377", bls12_377, ScalarField);
#[cfg(feature = "bw6-761")]
impl_vec_ops_field!("bw6_761", bw6_761, BaseField);

#[cfg(test)]
pub(crate) mod tests {
    use crate::curve::ScalarField;
    use icicle_core::impl_vec_ops_tests;
    use icicle_core::vec_ops::tests::*;

    impl_vec_ops_tests!(ScalarField);
}
