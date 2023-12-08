use crate::curve::{G1Affine, G1Projective, ScalarField};
use icicle_core::msm::MSMConfig;
use icicle_cuda_runtime::error::{CudaError, CudaResult, CudaResultWrap};

extern "C" {
    #[link_name = "bn254MSMCuda"]
    fn msm_cuda<'a>(
        scalars: *const ScalarField,
        points: *const G1Affine,
        count: usize,
        config: MSMConfig<'a>,
        out: *mut G1Projective,
    ) -> CudaError;

    #[link_name = "bn254DefaultMSMConfig"]
    fn default_msm_config() -> MSMConfig<'static>;
}

pub fn get_default_msm_config() -> MSMConfig<'static> {
    unsafe { default_msm_config() }
}

pub fn msm<'a>(
    scalars: &[ScalarField],
    points: &[G1Affine],
    cfg: MSMConfig<'a>,
    results: &mut [G1Projective],
) -> CudaResult<()> {
    if points.len() != scalars.len() {
        return Err(CudaError::cudaErrorInvalidValue);
    }

    unsafe {
        msm_cuda(
            scalars as *const _ as *const ScalarField,
            points as *const _ as *const G1Affine,
            points.len(),
            cfg,
            results as *mut _ as *mut G1Projective,
        )
        .wrap()
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use ark_bn254::G1Projective as ArkG1Projective;
    use ark_ec::scalar_mul::variable_base::VariableBaseMSM;

    use crate::{
        curve::{generate_random_affine_points, generate_random_scalars, G1Projective},
        msm::{get_default_msm_config, msm},
    };
    use icicle_core::traits::ArkConvertible;
    use icicle_cuda_runtime::memory::DeviceSlice;
    use icicle_cuda_runtime::stream::CudaStream;

    #[test]
    fn test_msm() {
        let log_test_sizes = [20];

        for log_test_size in log_test_sizes {
            let count = 1 << log_test_size;
            let points = generate_random_affine_points(count);
            let scalars = generate_random_scalars(count);

            let mut msm_results = DeviceSlice::cuda_malloc(1).unwrap();
            let stream = CudaStream::create().unwrap();
            let mut cfg = get_default_msm_config();
            cfg.ctx
                .stream = &stream;
            cfg.is_async = true;
            cfg.are_results_on_device = true;
            msm(&scalars, &points, cfg, &mut msm_results.as_slice()).unwrap();

            // this happens on CPU in parallel to the GPU MSM computations
            let point_r_ark: Vec<_> = points
                .iter()
                .map(|x| x.to_ark())
                .collect();
            let scalars_r_ark: Vec<_> = scalars
                .iter()
                .map(|x| x.to_ark())
                .collect();
            let msm_result_ark: ArkG1Projective = VariableBaseMSM::msm(&point_r_ark, &scalars_r_ark).unwrap();

            let mut msm_host_result = vec![G1Projective::zero(); 1];
            msm_results
                .copy_to_host(&mut msm_host_result[..])
                .unwrap();
            stream
                .synchronize()
                .unwrap();
            stream
                .destroy()
                .unwrap();

            assert_eq!(msm_host_result[0].to_ark(), msm_result_ark);
        }
    }
}
