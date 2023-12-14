use crate::curve::{CurveCfg, ScalarField};

use icicle_core::curve::CurveConfig;
use icicle_core::impl_ntt;
use icicle_core::ntt::{NTTConfig, NTT};
use icicle_cuda_runtime::device_context::DeviceContext;
use icicle_cuda_runtime::error::{CudaError, CudaResult, CudaResultWrap};

impl_ntt!("bls12_377", CurveCfg);

#[cfg(test)]
pub(crate) mod tests {
    use icicle_core::traits::ArkConvertible;
    use icicle_core::{
        curve::CurveConfig,
        impl_ntt_tests,
        ntt::{Ordering, NTT},
    };
    use icicle_cuda_runtime::device_context::get_default_device_context;

    use crate::curve::{generate_random_scalars, CurveCfg};

    use ark_bls12_377::Fr;
    use ark_ff::FftField;
    use ark_poly::{EvaluationDomain, GeneralEvaluationDomain};

    impl_ntt_tests!(CurveCfg);
}
