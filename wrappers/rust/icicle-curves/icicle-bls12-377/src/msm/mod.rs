use crate::curve::{CurveCfg, G1Affine, G1Projective, ScalarField};
use icicle_core::{
    curve::CurveConfig,
    impl_msm,
    msm::{MSMConfig, MSM},
};
use icicle_cuda_runtime::error::{CudaError, CudaResult, CudaResultWrap};

impl_msm!("bls12_377", CurveCfg);

#[cfg(test)]
pub(crate) mod tests {
    use ark_bls12_377::G1Projective as ArkG1Projective;
    use ark_ec::scalar_mul::variable_base::VariableBaseMSM;
    use icicle_core::{curve::CurveConfig, impl_msm_tests, msm::MSM};

    use crate::curve::CurveCfg;
    use icicle_core::traits::ArkConvertible;
    use icicle_cuda_runtime::memory::DeviceSlice;
    use icicle_cuda_runtime::stream::CudaStream;

    impl_msm_tests!(CurveCfg);
}
