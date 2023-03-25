use std::ffi::{c_int, c_uint};

use field::*;

pub mod field;
pub mod utils;

extern "C" {
    fn msm_cuda(
        out: *mut Point,
        points: *const PointAffineNoInfinity,
        scalars: *const ScalarField,
        count: usize,
        device_id: usize,
    ) -> c_uint;

    fn msm_batch_cuda(
        out: *mut Point,
        points: *const PointAffineNoInfinity,
        scalars: *const ScalarField,
        batch_size: usize,
        msm_size: usize,
        device_id: usize,
    ) -> c_uint;

    fn ntt_cuda(inout: *mut ScalarField, n: usize, inverse: bool, device_id: usize) -> c_int;

    fn ecntt_cuda(inout: *mut Point, n: usize, inverse: bool, device_id: usize) -> c_int;

    fn ntt_end2end_batch(
        inout: *mut ScalarField,
        arr_size: usize,
        n: usize,
        inverse: bool,
    ) -> c_int;

    fn ecntt_end2end_batch(inout: *mut Point, arr_size: usize, n: usize, inverse: bool) -> c_int;

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

pub fn msm_batch(
    points: &[PointAffineNoInfinity],
    scalars: &[Scalar],
    batch_size: usize,
    device_id: usize,
) -> Vec<Point> {
    let count = points.len();
    if count != scalars.len() {
        todo!("variable length")
    }

    let mut ret = vec![Point::zero(); batch_size];

    unsafe {
        msm_batch_cuda(
            &mut ret[0] as *mut _ as *mut Point,
            points as *const _ as *const PointAffineNoInfinity,
            scalars as *const _ as *const ScalarField,
            batch_size,
            count / batch_size,
            device_id,
        )
    };

    ret
}

/// Compute an in-place ECNTT on the input data.
fn ntt_internal(values: &mut [Scalar], device_id: usize, inverse: bool) -> i32 {
    let ret_code = unsafe {
        ntt_cuda(
            values as *mut _ as *mut ScalarField,
            values.len(),
            inverse,
            device_id,
        )
    };
    ret_code
}

pub fn ntt(values: &mut [Scalar], device_id: usize) {
    ntt_internal(values, device_id, false);
}

pub fn intt(values: &mut [Scalar], device_id: usize) {
    ntt_internal(values, device_id, true);
}

/// Compute an in-place ECNTT on the input data.
fn ntt_internal_batch(
    values: &mut [Scalar],
    device_id: usize,
    batch_size: usize,
    inverse: bool,
) -> i32 {
    unsafe {
        ntt_end2end_batch(
            values as *mut _ as *mut ScalarField,
            values.len(),
            batch_size,
            inverse,
        )
    }
}

pub fn ntt_batch(values: &mut [Scalar], batch_size: usize, device_id: usize) {
    ntt_internal_batch(values, 0, batch_size, false);
}

pub fn intt_batch(values: &mut [Scalar], batch_size: usize, device_id: usize) {
    ntt_internal_batch(values, 0, batch_size, true);
}

/// Compute an in-place ECNTT on the input data.
fn ecntt_internal(values: &mut [Point], inverse: bool, device_id: usize) -> i32 {
    unsafe {
        ecntt_cuda(
            values as *mut _ as *mut Point,
            values.len(),
            inverse,
            device_id,
        )
    }
}

pub fn ecntt(values: &mut [Point], device_id: usize) {
    ecntt_internal(values, false, device_id);
}

/// Compute an in-place iECNTT on the input data.
pub fn iecntt(values: &mut [Point], device_id: usize) {
    ecntt_internal(values, true, device_id);
}

/// Compute an in-place ECNTT on the input data.
fn ecntt_internal_batch(
    values: &mut [Point],
    device_id: usize,
    batch_size: usize,
    inverse: bool,
) -> i32 {
    unsafe {
        ecntt_end2end_batch(
            values as *mut _ as *mut Point,
            values.len(),
            batch_size,
            inverse,
        )
    }
}

pub fn ecntt_batch(values: &mut [Point], batch_size: usize, device_id: usize) {
    ecntt_internal_batch(values, 0, batch_size, false);
}

/// Compute an in-place iECNTT on the input data.
pub fn iecntt_batch(values: &mut [Point], batch_size: usize, device_id: usize) {
    ecntt_internal_batch(values, 0, batch_size, true);
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

    use std::ops::Add;

    use ark_bls12_381::{Fr, G1Affine, G1Projective};
    use ark_ec::{msm::VariableBaseMSM, AffineCurve, ProjectiveCurve};
    use ark_ff::{FftField, Field, Zero};
    use ark_std::UniformRand;
    use rand::{rngs::StdRng, RngCore, SeedableRng};

    use crate::{field::*, *};

    fn random_points_ark_proj(nof_elements: usize) -> Vec<G1Projective> {
        let mut rng = ark_std::rand::thread_rng();
        let mut points_ga: Vec<G1Projective> = Vec::new();
        for _ in 0..nof_elements {
            let aff = G1Projective::rand(&mut rng);
            points_ga.push(aff);
        }
        points_ga
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
        let points = random_points_ark_proj(size);
        let result1: Vec<G1Projective> = ecntt_arc_naive(&points, size, false);
        let result2: Vec<G1Projective> = ecntt_arc_naive(&result1, size, true);
        assert!(!check_eq(&result2, &result1));
        assert!(check_eq(&result2, &points));
    }
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
    fn test_batch_msm() {
        for batch_pow2 in 7..10 {
            for pow2 in [7, 8, 12] {
                let count = 1 << pow2;
                let batch_size = 1 << batch_pow2;
                let seed = None; //set Some to provide seed
                let points = generate_random_points(count, get_rng(seed));
                let scalars = generate_random_scalars(count, get_rng(seed));

                let scalars_batch: Vec<_> = vec![scalars.clone(); batch_size]
                    .into_iter()
                    .flatten()
                    .collect();
                let points_batch: Vec<_> = vec![points.clone(); batch_size]
                    .into_iter()
                    .flatten()
                    .collect();

                let msm_result = msm(&points, &scalars, 0);

                let expected = vec![msm_result; batch_size];

                let result = msm_batch(&points_batch, &scalars_batch, batch_size, 0);

                assert_eq!(expected, result, "total {} x {}", count, batch_size);
            }
        }
    }

    #[test]
    fn test_ntt_quick() {
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
        let points_proj = generate_random_points_proj(test_size, get_rng(seed));

        assert!(points_proj[0].to_ark().into_affine().is_on_curve());

        //naive ark
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
    fn test_ntt() {
        //NTT
        let seed = None; //some value to fix the rng
        let test_size = 1 << 5;

        let scalars = generate_random_scalars(test_size, get_rng(seed));

        let mut ntt_result = scalars.clone();
        ntt(&mut ntt_result, 0);

        assert_ne!(ntt_result, scalars);

        let mut intt_result = ntt_result.clone();

        intt(&mut intt_result, 0);

        assert_eq!(intt_result, scalars);

        //ECNTT
        let points_proj = generate_random_points_proj(test_size, get_rng(seed));

        test_naive_ark_ecntt(test_size);

        assert!(points_proj[0].to_ark().into_affine().is_on_curve());

        //naive ark
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
    fn test_ntt_batch() {
        //NTT
        let seed = None; //some value to fix the rng
        let test_size = 1 << 9;
        let batches = 4;

        let scalars_batch: Vec<Scalar> =
            generate_random_scalars(test_size * batches, get_rng(seed));

        let mut scalar_vec_of_vec: Vec<Vec<Scalar>> = Vec::new();

        for i in 0..batches {
            scalar_vec_of_vec.push(scalars_batch[i * test_size..(i + 1) * test_size].to_vec());
        }

        let mut ntt_result = scalars_batch.clone();

        // do batch ntt
        ntt_batch(&mut ntt_result, test_size, 0);

        let mut ntt_result_vec_of_vec = Vec::new();

        // do ntt for every chunk
        for i in 0..batches {
            ntt_result_vec_of_vec.push(scalar_vec_of_vec[i].clone());
            ntt(&mut ntt_result_vec_of_vec[i], 0);
        }

        // check that the ntt of each vec of scalars is equal to the intt of the specific batch
        for i in 0..batches {
            assert_eq!(
                ntt_result_vec_of_vec[i],
                ntt_result[i * test_size..(i + 1) * test_size]
            );
        }

        // check that ntt output is different from input
        assert_ne!(ntt_result, scalars_batch);

        let mut intt_result = ntt_result.clone();

        // do batch intt
        intt_batch(&mut intt_result, test_size, 0);

        let mut intt_result_vec_of_vec = Vec::new();

        // do intt for every chunk
        for i in 0..batches {
            intt_result_vec_of_vec.push(ntt_result_vec_of_vec[i].clone());
            intt(&mut intt_result_vec_of_vec[i], 0);
        }

        // check that the intt of each vec of scalars is equal to the intt of the specific batch
        for i in 0..batches {
            assert_eq!(
                intt_result_vec_of_vec[i],
                intt_result[i * test_size..(i + 1) * test_size]
            );
        }

        assert_eq!(intt_result, scalars_batch);

        // //ECNTT
        let points_proj = generate_random_points_proj(test_size * batches, get_rng(seed));

        let mut points_vec_of_vec: Vec<Vec<Point>> = Vec::new();

        for i in 0..batches {
            points_vec_of_vec.push(points_proj[i * test_size..(i + 1) * test_size].to_vec());
        }

        let mut ntt_result_points = points_proj.clone();

        // do batch ecintt
        ecntt_batch(&mut ntt_result_points, test_size, 0);

        let mut ntt_result_points_vec_of_vec = Vec::new();

        for i in 0..batches {
            ntt_result_points_vec_of_vec.push(points_vec_of_vec[i].clone());
            ecntt(&mut ntt_result_points_vec_of_vec[i], 0);
        }

        for i in 0..batches {
            assert_eq!(
                ntt_result_points_vec_of_vec[i],
                ntt_result_points[i * test_size..(i + 1) * test_size]
            );
        }

        assert_ne!(ntt_result_points, points_proj);

        let mut intt_result_points = ntt_result_points.clone();

        // do batch ecintt
        iecntt_batch(&mut intt_result_points, test_size, 0);

        let mut intt_result_points_vec_of_vec = Vec::new();

        // do ecintt for every chunk
        for i in 0..batches {
            intt_result_points_vec_of_vec.push(ntt_result_points_vec_of_vec[i].clone());
            iecntt(&mut intt_result_points_vec_of_vec[i], 0);
        }

        // check that the ecintt of each vec of scalars is equal to the intt of the specific batch
        for i in 0..batches {
            assert_eq!(
                intt_result_points_vec_of_vec[i],
                intt_result_points[i * test_size..(i + 1) * test_size]
            );
        }

        assert_eq!(intt_result_points, points_proj);
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
