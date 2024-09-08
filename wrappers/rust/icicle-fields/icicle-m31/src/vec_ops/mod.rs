use crate::field::{CExtensionCfg, CExtensionField, ExtensionCfg, ExtensionField, ScalarCfg, ScalarField};

use icicle_core::error::IcicleResult;
use icicle_core::impl_vec_ops_field;
use icicle_core::traits::IcicleResultWrap;
use icicle_core::vec_ops::{BitReverseConfig, VecOps, VecOpsConfig};
use icicle_cuda_runtime::device_context::DeviceContext;
use icicle_cuda_runtime::error::CudaError;
use icicle_cuda_runtime::memory::HostOrDeviceSlice;

impl_vec_ops_field!("m31", m31, ScalarField, ScalarCfg);
impl_vec_ops_field!("m31_extension", m31_extension, ExtensionField, ExtensionCfg);
impl_vec_ops_field!("m31_cextension", m31_cextension, CExtensionField, CExtensionCfg);

#[cfg(test)]
pub(crate) mod tests {
    use crate::field::{CExtensionField, ExtensionField, ScalarField};
    use icicle_core::impl_vec_add_tests;
    use icicle_core::vec_ops::tests::*;

    impl_vec_add_tests!(ScalarField);
    mod cextension {
        use super::*;
        impl_vec_add_tests!(CExtensionField);
    }
    mod extension {
        use super::*;
        impl_vec_add_tests!(ExtensionField);
    }
}
