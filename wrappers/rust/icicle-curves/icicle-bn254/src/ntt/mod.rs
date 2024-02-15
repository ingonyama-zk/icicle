use crate::curve::{ScalarCfg, ScalarField};

use icicle_core::error::IcicleResult;
use icicle_core::impl_ntt;
use icicle_core::ntt::{NTTConfig, NTTDir, NTT};
use icicle_core::traits::IcicleResultWrap;
use icicle_cuda_runtime::device_context::DeviceContext;
use icicle_cuda_runtime::device_context::DEFAULT_DEVICE_ID;
use icicle_cuda_runtime::error::CudaError;
use icicle_cuda_runtime::memory::HostOrDeviceSlice;

impl_ntt!("bn254", bn254, ScalarField, ScalarCfg);

#[cfg(test)]
pub(crate) mod tests {
    use crate::curve::ScalarField;
    use crate::ntt::DEFAULT_DEVICE_ID;
    use icicle_core::impl_ntt_tests;
    use icicle_core::ntt::tests::*;
    use std::sync::OnceLock;

    impl_ntt_tests!(ScalarField);
}
