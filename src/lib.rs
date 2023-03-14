use std::ffi::{c_int, c_uint};

use field::*;

pub mod field;
pub mod utils;

extern "C" {
    fn msm_cuda(
        out: *mut Point,
        points: *const PointAffineNoInfinity,
        scalars: *const ScalarField,
        count: usize, //TODO: is needed?
        device_id: usize,
    ) -> c_uint;

    fn ntt_cuda(inout: *mut ScalarField, n: usize, inverse: bool, device_id: usize) -> c_int;

    fn ecntt_cuda(inout: *mut Point, n: usize, inverse: bool, device_id: usize) -> c_int;

    fn vec_mod_mult_point(
        inout: *mut Point,
        scalars: *const ScalarField,
        n_elements: usize,
        device_id: usize,
    ) -> c_int;

    fn vec_mod_mult_scalar(
        inout: *mut ScalarField,
        scalars: *const ScalarField,
        n_elements: usize,
        device_id: usize,
    ) -> c_int;
}

pub fn msm(points: &[PointAffineNoInfinity], scalars: &[Scalar], device_id: usize) -> Point {
    let count = points.len();
    if count != scalars.len() {
        todo!("variable length")
    }

    let mut ret = Point::zero();
    unsafe {
        msm_cuda(
            &mut ret as *mut _ as *mut Point,
            points as *const _ as *const PointAffineNoInfinity,
            scalars as *const _ as *const ScalarField,
            scalars.len(),
            device_id,
        )
    };

    ret
}

/// Compute an in-place ECNTT on the input data.
fn ntt_internal(values: &mut [Scalar], device_id: usize, inverse: bool) -> i32 {
    let ret_code = unsafe { ntt_cuda(values as *mut _ as *mut ScalarField, values.len(), inverse, device_id) };
    ret_code
}

pub fn ntt(values: &mut [Scalar], device_id: usize) {
    ntt_internal(values, device_id, false);
}

pub fn intt(values: &mut [Scalar], device_id: usize) {
    ntt_internal(values, device_id, true);
}

/// Compute an in-place ECNTT on the input data.
fn ecntt_internal(values: &mut [Point],inverse: bool,  device_id: usize) -> i32 {
    unsafe { ecntt_cuda(values as *mut _ as *mut Point, values.len(), inverse, device_id) }
}

pub fn ecntt(values: &mut [Point], device_id: usize) {
    ecntt_internal(values, false, device_id);
}

/// Compute an in-place iECNTT on the input data.
pub fn iecntt(values: &mut [Point], device_id: usize) {
    ecntt_internal(values, true, device_id);
}

pub fn multp_vec(a: &mut [Point], b: &[Scalar], device_id: usize) {
    assert_eq!(a.len(), b.len());
    unsafe {
        vec_mod_mult_point(
            a as *mut _ as *mut Point,
            b as *const _ as *const ScalarField,
            a.len(),
            device_id,
        );
    }
}

pub fn mult_sc_vec(a: &mut [Scalar], b: &[Scalar], device_id: usize) {
    assert_eq!(a.len(), b.len());
    unsafe {
        vec_mod_mult_scalar(
            a as *mut _ as *mut ScalarField,
            b as *const _ as *const ScalarField,
            a.len(),
            device_id,
        );
    }
}

#[cfg(test)]
mod tests {
    use ark_bls12_381::{Fr, G1Affine, G1Projective};
    use ark_ec::msm::VariableBaseMSM;
    use ark_ff::UniformRand;
    use rand::{rngs::StdRng, RngCore, SeedableRng};

    use crate::{field::*, *};

    fn get_rng(seed: Option<u64>) -> Box<dyn RngCore> {
        let rng: Box<dyn RngCore> = match seed {
            Some(seed) => Box::new(StdRng::seed_from_u64(seed)),
            None => Box::new(rand::thread_rng()),
        };
        rng
    }

    fn generate_random_points(
        count: usize,
        mut rng: Box<dyn RngCore>,
    ) -> Vec<PointAffineNoInfinity> {
        (0..count)
            .map(|_| Point::from_ark(G1Projective::rand(&mut rng)).to_xy_strip_z())
            .collect()
    }
    fn generate_random_points_proj(count: usize, mut rng: Box<dyn RngCore>) -> Vec<Point> {
        (0..count)
            .map(|_| Point::from_ark(G1Projective::rand(&mut rng)))
            .collect()
    }

    fn generate_random_scalars(count: usize, mut rng: Box<dyn RngCore>) -> Vec<Scalar> {
        (0..count)
            .map(|_| Scalar::from_ark(&Fr::rand(&mut rng)))
            .collect()
    }

    #[test]
    fn test_msm() {
        let test_sizes = [7, 8, 12];

        for pow2 in test_sizes {
            let count = 1 << pow2;
            let seed = None; //set Some to provide seed
            let points = generate_random_points(count, get_rng(seed));
            let scalars = generate_random_scalars(count, get_rng(seed));

            let msm_result = msm(&points, &scalars, 0);

            let point_r_ark: Vec<_> = points.iter().map(|x| x.to_ark_repr()).collect();
            let scalars_r_ark: Vec<_> = scalars.iter().map(|x| x.to_ark_mod_p().0).collect();

            let msm_result_ark = VariableBaseMSM::multi_scalar_mul(&point_r_ark, &scalars_r_ark);

            assert_eq!(msm_result.to_ark_affine(), msm_result_ark);
            assert_eq!(msm_result.to_ark(), msm_result_ark);
            assert_eq!(
                msm_result.to_ark_affine(),
                Point::from_ark(msm_result_ark).to_ark_affine()
            );
        }
    }

    #[test]
    fn test_ntt() {
        use std::ops::Add;

        use ark_bls12_381::{Fr, G1Projective};
        use ark_ec::{AffineCurve, ProjectiveCurve};
        use ark_ff::{FftField, Field, Zero};
        use ark_std::UniformRand;

        fn random_points(nof_elements: usize) -> Vec<G1Projective> {
            let mut rng = ark_std::rand::thread_rng();
            let mut points: Vec<G1Projective> = Vec::new();
            for _ in 0..nof_elements {
                let p = G1Projective::rand(&mut rng);
                points.push(p);
            }
            points
        }

        fn ecntt_arc_naive(
            points: &Vec<G1Projective>,
            size: usize,
            inverse: bool,
        ) -> Vec<G1Projective> {
            let mut result: Vec<G1Projective> = Vec::new();
            for _ in 0..size {
                result.push(G1Projective::zero());
            }
            let rou: Fr;
            if !inverse {
                rou = Fr::get_root_of_unity(size).unwrap();
            } else {
                rou = Fr::inverse(&Fr::get_root_of_unity(size).unwrap()).unwrap();
            }
            for k in 0..size {
                for l in 0..size {
                    let pow: [u64; 1] = [(l * k).try_into().unwrap()];
                    let mul_rou = Fr::pow(&rou, &pow);
                    result[k] = result[k].add(points[l].into_affine().mul(mul_rou));
                }
            }
            if inverse {
                let size2 = size as u64;
                for k in 0..size {
                    let multfactor = Fr::inverse(&Fr::from(size2)).unwrap();
                    result[k] = result[k].into_affine().mul(multfactor);
                }
            }
            return result;
        }

        fn check_eq(points: &Vec<G1Projective>, points2: &Vec<G1Projective>) -> bool {
            let mut eq = true;
            for i in 0..points.len() {
                if points2[i].ne(&points[i]) {
                    eq = false;
                    break;
                }
            }
            return eq;
        }

        fn test_naive_ark_ecntt(size: usize) {
            let points = random_points(size);
            let result1: Vec<G1Projective> = ecntt_arc_naive(&points, size, false);
            let result2: Vec<G1Projective> = ecntt_arc_naive(&result1, size, true);
            assert!(!check_eq(&result2, &result1));
            assert!(check_eq(&result2, &points));
        }

        //NTT
        let seed = None; //some value to fix the rng
        let test_size = 1 << 8;

        let scalars = generate_random_scalars(test_size, get_rng(seed));

        let mut ntt_result = scalars.clone();
        ntt(&mut ntt_result, 0);

        assert_ne!(ntt_result, scalars);

        let mut intt_result = ntt_result.clone();

        intt(&mut intt_result, 0);

        assert_eq!(intt_result, scalars);

        //ECNTT
        let rng = get_rng(seed);

        let points_proj: Vec<_> = generate_random_points_proj(test_size, rng);

        //naive ark
        test_naive_ark_ecntt(test_size); // to ensure points conversion is ok, naive will match just
                                         // with reordering, cause current twiddle factors are
                                         // same as Ethereum, not Ark
        let points_proj_ark = points_proj
            .iter()
            .map(|p| p.to_ark())
            .collect::<Vec<G1Projective>>();

        let ecntt_result_naive = ecntt_arc_naive(&points_proj_ark, points_proj_ark.len(), false);

        let iecntt_result_naive = ecntt_arc_naive(&ecntt_result_naive, points_proj_ark.len(), true);

        assert_eq!(points_proj_ark, iecntt_result_naive);

        //ingo gpu
        let mut ecntt_result = points_proj.to_vec();
        ecntt(&mut ecntt_result, 0);

        assert_ne!(ecntt_result, points_proj);

        let mut iecntt_result = ecntt_result.clone();
        iecntt(&mut iecntt_result, 0);

        assert_eq!(
            iecntt_result_naive,
            points_proj
                .iter()
                .map(|p| p.to_ark_affine())
                .collect::<Vec<G1Affine>>()
        );
        assert_eq!(
            iecntt_result
                .iter()
                .map(|p| p.to_ark_affine())
                .collect::<Vec<G1Affine>>(),
            points_proj
                .iter()
                .map(|p| p.to_ark_affine())
                .collect::<Vec<G1Affine>>()
        );
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_vec_scalar_mul() {
        let mut intoo = [Scalar::one(), Scalar::one(), Scalar::zero()];
        let expected = [Scalar::one(), Scalar::zero(), Scalar::zero()];
        mult_sc_vec(&mut intoo, &expected, 0);
        assert_eq!(intoo, expected);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_equality() {
        let left = Point::zero();
        let right = Point::zero();
        assert_eq!(left, right);
        let right = Point::from_limbs(&[0; 12], &[2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], &[0; 12]);
        assert_eq!(left, right);
        let right = Point::from_limbs(
            &[2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            &[0; 12],
            &[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        );
        assert!(left != right);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_vec_point_mul() {
        let dummy_one = Point {
            x: BaseField::one(),
            y: BaseField::zero(),
            z: BaseField::one(),
        };

        let mut inout = [dummy_one, dummy_one, Point::zero()];
        let scalars = [Scalar::one(), Scalar::zero(), Scalar::zero()];
        let expected = [
            Point::zero(),
            Point {
                x: BaseField::zero(),
                y: BaseField::one(),
                z: BaseField::zero(),
            },
            Point {
                x: BaseField::zero(),
                y: BaseField::one(),
                z: BaseField::zero(),
            },
        ];
        multp_vec(&mut inout, &scalars, 0);
        assert_eq!(inout, expected);
    }
}
