use crate::field::{ComplexExtensionCfg, ComplexExtensionField, ExtensionCfg, QuarticExtensionField, ScalarCfg, ScalarField};

use icicle_core::error::IcicleResult;
use icicle_core::impl_vec_ops_field;
use icicle_core::traits::IcicleResultWrap;
use icicle_core::vec_ops::{BitReverseConfig, VecOps, VecOpsConfig};
use icicle_cuda_runtime::device_context::DeviceContext;
use icicle_cuda_runtime::error::CudaError;
use icicle_cuda_runtime::memory::HostOrDeviceSlice;

impl_vec_ops_field!("m31", m31, ScalarField, ScalarCfg);
impl_vec_ops_field!("m31_q_extension", m31_q_extension, QuarticExtensionField, ExtensionCfg);
impl_vec_ops_field!("m31_c_extension", m31_c_extension, ComplexExtensionField, ComplexExtensionCfg);

#[cfg(test)]
pub(crate) mod tests {
    use crate::field::{ComplexExtensionField, QuarticExtensionField, ScalarField};
    use icicle_core::impl_vec_add_tests;
    use icicle_core::vec_ops::tests::*;

    impl_vec_add_tests!(ScalarField);
    mod complex_extension {
        use super::*;
        impl_vec_add_tests!(ComplexExtensionField);
    }
    mod extension {
        use super::*;
        impl_vec_add_tests!(QuarticExtensionField);
    }
}
