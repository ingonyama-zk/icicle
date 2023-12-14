use crate::curve::{CurveCfg, G1Affine, G1Projective, ScalarField};
use icicle_core::{msm::{MSMConfig, MSM}, impl_msm, curve::CurveConfig};
use icicle_cuda_runtime::error::{CudaError, CudaResult, CudaResultWrap};

impl_msm!("bn254", CurveCfg);

#[cfg(test)]
pub(crate) mod tests {
    use ark_bn254::G1Projective as ArkG1Projective;
    use ark_ec::scalar_mul::variable_base::VariableBaseMSM;
    use icicle_core::{impl_msm_tests, msm::MSM, curve::CurveConfig};

    use crate::curve::{generate_random_affine_points, generate_random_scalars, CurveCfg};
    use icicle_core::traits::ArkConvertible;
    use icicle_cuda_runtime::memory::DeviceSlice;
    use icicle_cuda_runtime::stream::CudaStream;

    impl_msm_tests!(CurveCfg);
}
