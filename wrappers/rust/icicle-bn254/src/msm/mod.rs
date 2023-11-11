use std::ffi::c_uint;

use crate::curve::{ScalarField, G1Affine, G1Projective};

use icicle_core::msm::MSMConfig;

extern "C" {
    #[link_name = "bn254MSMCuda"]
    fn msm_cuda(
        scalars: *const ScalarField,
        points: *const G1Affine,
        count: usize,
        config: MSMConfig,
        out: *mut G1Projective,
    ) -> c_uint;

    // #[link_name = "GetDefaultMSMConfig"]
    fn GetDefaultMSMConfig() -> MSMConfig;
}

pub fn get_default_msm_config() -> MSMConfig {
    unsafe { GetDefaultMSMConfig() }
}

pub fn msm(scalars: &[ScalarField], points: &[G1Affine], cfg: MSMConfig, results: &mut [G1Projective]) {
    if points.len() != scalars.len() {
        panic!("lengths of scalars and points are not equal")
    }

    unsafe {
        msm_cuda(
            scalars as *const _ as *const ScalarField,
            points as *const _ as *const G1Affine,
            points.len(),
            cfg,
            results as *mut _ as *mut G1Projective,
        )
    };
}

#[cfg(test)]
pub(crate) mod tests {
    use ark_bn254::{Fr, G1Affine as ArkG1Affine, G1Projective as ArkG1Projective};
    use ark_ec::scalar_mul::variable_base::VariableBaseMSM;

    use crate::{curve::{G1Projective, generate_random_affine_points, generate_random_scalars}, msm::{msm, get_default_msm_config}};
    use icicle_core::traits::ArkConvertible;
    use icicle_cuda_runtime::device_context::DeviceContext;

    #[test]
    fn test_msm() {
        let test_sizes = [20];

        for pow2 in test_sizes {
            let count = 1 << pow2;
            let points = generate_random_affine_points(count);
            let scalars = generate_random_scalars(count);

            let mut msm_results = [G1Projective::zero()];
            msm(&scalars, &points, get_default_msm_config(), &mut msm_results);
            let msm_result = msm_results[0];

            let point_r_ark: Vec<_> = points
                .iter()
                .map(|x| x.to_ark())
                .collect();
            let scalars_r_ark: Vec<_> = scalars
                .iter()
                .map(|x| x.to_ark())
                .collect();

            let msm_result_ark: ArkG1Projective = VariableBaseMSM::msm(&point_r_ark, &scalars_r_ark).unwrap();

            assert_eq!(msm_result.to_ark(), msm_result_ark);
        }
    }
}
