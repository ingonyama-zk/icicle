use crate::curve::{ScalarCfg, ScalarField};
use icicle_core::{
    impl_vec_ops_field,
    vec_ops::{/*BitReverseConfig,*/ VecOps, VecOpsConfig},
};
use icicle_runtime::{errors::eIcicleError, memory::HostOrDeviceSlice};

use icicle_core::program::Program;
use icicle_core::traits::FieldImpl;

impl_vec_ops_field!("bw6_761", bw6_761, ScalarField, ScalarCfg);

#[cfg(test)]
pub(crate) mod tests {
    use crate::curve::ScalarField;
    use icicle_core::impl_vec_ops_tests;
    use icicle_core::vec_ops::tests::*;

    impl_vec_ops_tests!(bw6_761, ScalarField);
}
