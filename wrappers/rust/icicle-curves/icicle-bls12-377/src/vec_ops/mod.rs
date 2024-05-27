#[cfg(feature = "bw6-761")]
use crate::curve::{BaseCfg, BaseField};
use crate::curve::{ScalarCfg, ScalarField};

use icicle_core::error::IcicleResult;
use icicle_core::impl_vec_ops_field;
use icicle_core::traits::IcicleResultWrap;
use icicle_core::vec_ops::{VecOps, VecOpsConfig, BitReverseConfig};
use icicle_cuda_runtime::device_context::DeviceContext;
use icicle_cuda_runtime::error::CudaError;
use icicle_cuda_runtime::memory::HostOrDeviceSlice;

impl_vec_ops_field!("bls12_377", bls12_377, ScalarField, ScalarCfg);
#[cfg(feature = "bw6-761")]
impl_vec_ops_field!("bw6_761", bw6_761, BaseField, BaseCfg);

#[cfg(test)]
pub(crate) mod tests {
    use crate::curve::ScalarField;
    use icicle_core::impl_vec_add_tests;
    use icicle_core::vec_ops::tests::*;

    impl_vec_add_tests!(ScalarField);
}
