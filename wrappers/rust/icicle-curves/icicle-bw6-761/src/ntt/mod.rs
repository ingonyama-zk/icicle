use crate::curve::{ScalarField, SCALAR_LIMBS, ScalarCfg};

use icicle_core::impl_ntt;
use icicle_core::ntt::{NTTConfig, NTT};
use icicle_cuda_runtime::device_context::DeviceContext;
use icicle_cuda_runtime::error::{CudaError, CudaResult, CudaResultWrap};

impl_ntt!("bw6_761", SCALAR_LIMBS, ScalarField, ScalarCfg);

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

    use ark_bw6_761::Fr;
    use ark_ff::FftField;
    use ark_poly::{EvaluationDomain, GeneralEvaluationDomain};

    impl_ntt_tests!(CurveCfg);
}
