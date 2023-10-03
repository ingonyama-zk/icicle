use std::ffi::c_uint;

use self::bls12_381::*;

pub mod bls12_381;


extern "C" {
    #[link_name = "bls12_381_msm_cuda"]
    fn msm_cuda(
        scalars: *const ScalarField,
        points: *const PointAffineNoInfinity,
        count: usize,
        out: *mut Point,
    ) -> c_uint;
}

pub fn msm(
    scalars: &[ScalarField],
    points: &[PointAffineNoInfinity],
) -> Point {
    let count = points.len();
    if count != scalars.len() {
        todo!("variable length")
    }

    let mut out = Point::zero();
    unsafe {
        msm_cuda(
            scalars as *const _ as *const ScalarField,
            points as *const _ as *const PointAffineNoInfinity,
            scalars.len(),
            &mut out as *mut _ as *mut Point,
        )
    };

    out
}

#[cfg(test)]
pub(crate) mod tests {
    use ark_ec::msm::VariableBaseMSM;
    use ark_bls12_381::{Fr, G1Projective};
    use ark_ff::PrimeField;
    use ark_std::UniformRand;
    use rand::RngCore;

    use crate::{curves::{bls12_381::*, msm}, utils::get_rng};


    pub fn generate_random_points(
        count: usize,
        mut rng: Box<dyn RngCore>,
    ) -> Vec<PointAffineNoInfinity> {
        (0..count)
            .map(|_| Point::from_ark(G1Projective::rand(&mut rng)).to_xy_strip_z())
            .collect()
    }
    
    pub fn generate_random_points_proj(count: usize, mut rng: Box<dyn RngCore>) -> Vec<Point> {
        (0..count)
            .map(|_| Point::from_ark(G1Projective::rand(&mut rng)))
            .collect()
    }
    
    pub fn generate_random_scalars(count: usize, mut rng: Box<dyn RngCore>) -> Vec<ScalarField> {
        (0..count)
            .map(|_| ScalarField::from_ark(Fr::rand(&mut rng).into_repr()))
            .collect()
    }
    

    #[test]
    fn test_msm() {
        let test_sizes = [6, 9];

        for pow2 in test_sizes {
            let count = 1 << pow2;
            let seed = None; // set Some to provide seed
            let points = generate_random_points(count, get_rng(seed));
            let scalars = generate_random_scalars(count, get_rng(seed));

            let msm_result = msm(&scalars, &points);

            let point_r_ark: Vec<_> = points
                .iter()
                .map(|x| x.to_ark_repr())
                .collect();
            let scalars_r_ark: Vec<_> = scalars
                .iter()
                .map(|x| x.to_ark())
                .collect();

            let msm_result_ark = VariableBaseMSM::multi_scalar_mul(&point_r_ark, &scalars_r_ark);

            assert_eq!(msm_result.to_ark_affine(), msm_result_ark);
            assert_eq!(msm_result.to_ark(), msm_result_ark);
            assert_eq!(
                msm_result.to_ark_affine(),
                Point::from_ark(msm_result_ark).to_ark_affine()
            );
        }
    }
}
