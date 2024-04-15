use crate::curve::{ScalarCfg, ScalarField};

use icicle_core::error::IcicleResult;
use icicle_core::{impl_ntt, impl_ntt_without_domain};
use icicle_core::ntt::{NTTConfig, NTTDir, NTT, NTTDomain};
use icicle_core::traits::IcicleResultWrap;
use icicle_cuda_runtime::device_context::DeviceContext;
use icicle_cuda_runtime::error::CudaError;
use icicle_cuda_runtime::memory::HostOrDeviceSlice;

impl_ntt!("<CURVE>", <CURVE>, ScalarField, ScalarCfg);

#[cfg(test)]
pub(crate) mod tests {
    use crate::curve::ScalarField;
    use icicle_cuda_runtime::device_context::DEFAULT_DEVICE_ID;
    use icicle_core::impl_ntt_tests;
    use icicle_core::ntt::tests::*;
    use std::sync::OnceLock;
    use serial_test::{serial, parallel};

    impl_ntt_tests!(ScalarField);
}
