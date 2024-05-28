use crate::field::{ExtensionCfg, ExtensionField, ScalarCfg, ScalarField};

use icicle_core::error::IcicleResult;
use icicle_core::impl_vec_ops_field;
use icicle_core::traits::IcicleResultWrap;
use icicle_core::vec_ops::{BitReverseConfig, VecOps, VecOpsConfig};
use icicle_cuda_runtime::device_context::DeviceContext;
use icicle_cuda_runtime::error::CudaError;
use icicle_cuda_runtime::memory::HostOrDeviceSlice;

impl_vec_ops_field!("babybear", babybear, ScalarField, ScalarCfg);
impl_vec_ops_field!("babybear_extension", babybear_extension, ExtensionField, ExtensionCfg);

#[cfg(test)]
pub(crate) mod tests {
    use crate::field::{ExtensionField, ScalarField};
    use icicle_core::impl_vec_add_tests;
    use icicle_core::vec_ops::tests::*;

    impl_vec_add_tests!(ScalarField);

    mod extension {
        use super::*;

        impl_vec_add_tests!(ExtensionField);
    }
}
