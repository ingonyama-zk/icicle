use icicle_core::ecntt::ECNTT;
use icicle_core::error::IcicleResult;
use icicle_core::impl_ecntt;
use icicle_core::ntt::{NTTConfig, NTTDir};
use icicle_core::traits::IcicleResultWrap;
use icicle_cuda_runtime::device_context::DeviceContext;
use icicle_cuda_runtime::device_context::DEFAULT_DEVICE_ID;
use icicle_cuda_runtime::error::CudaError;
use icicle_cuda_runtime::memory::HostOrDeviceSlice;

use crate::curve::{CurveCfg, ScalarCfg, ScalarField, BaseCfg};
use icicle_core::ecntt::Projective;

impl_ecntt!("bls12_381", bls12_381, ScalarField, ScalarCfg, BaseCfg, CurveCfg);

#[cfg(test)]
pub(crate) mod tests {
    use crate::curve::{CurveCfg, BaseField, ScalarField};

    use icicle_core::impl_ecntt_tests;
    use icicle_core::ecntt::tests::*;
    use std::sync::OnceLock;

    impl_ecntt_tests!(
        ScalarField,
        BaseField,
        CurveCfg
    );
}
