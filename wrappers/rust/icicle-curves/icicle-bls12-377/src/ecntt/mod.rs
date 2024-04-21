#![cfg(feature = "ec_ntt")]
use icicle_core::error::IcicleResult;
use icicle_core::impl_ecntt;
use icicle_core::ntt::{NTTConfig, NTTDir};
use icicle_cuda_runtime::device_context::DeviceContext;
use icicle_cuda_runtime::device_context::DEFAULT_DEVICE_ID;
use icicle_cuda_runtime::error::CudaError;

use crate::curve::{CurveCfg, ScalarCfg, ScalarField};
use icicle_core::ecntt::Projective;

impl_ecntt!("bls12_377", bls12_377, ScalarField, ScalarCfg, CurveCfg);

#[cfg(test)]
pub(crate) mod tests {
    use crate::curve::{CurveCfg, ScalarField};

    use icicle_core::ecntt::tests::*;
    use icicle_core::impl_ecntt_tests;
    use std::sync::OnceLock;

    impl_ecntt_tests!(ScalarField, CurveCfg);
}
