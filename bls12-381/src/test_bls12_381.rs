use std::ffi::{c_int, c_uint};
use ark_std::UniformRand;
use rand::{rngs::StdRng, RngCore, SeedableRng};
use rustacuda::CudaFlags;
use rustacuda::memory::DeviceBox;
use rustacuda::prelude::{DeviceBuffer, Device, ContextFlags, Context};
use rustacuda_core::DevicePointer;
use std::mem::transmute;
pub use crate::basic_structs::scalar::ScalarTrait;
pub use crate::curve_structs::*;
use icicle_core::utils::{u32_vec_to_u64_vec, u64_vec_to_u32_vec};
use std::marker::PhantomData;
use std::convert::TryInto;
use ark_bls12_381::{Fq as Fq_BLS12_381, Fr as Fr_BLS12_381, G1Affine as G1Affine_BLS12_381, G1Projective as G1Projective_BLS12_381};
use ark_ec::AffineCurve;
use ark_ff::{BigInteger384, BigInteger256, PrimeField};
use rustacuda::memory::{CopyDestination, DeviceCopy};


impl Scalar {
    pub fn to_biginteger254(&self) -> BigInteger256 {
        BigInteger256::new(u32_vec_to_u64_vec(&self.limbs()).try_into().unwrap())
    }

    pub fn to_ark(&self) -> BigInteger256 {
        BigInteger256::new(u32_vec_to_u64_vec(&self.limbs()).try_into().unwrap())
    }

    pub fn from_biginteger256(ark: BigInteger256) -> Self {
        Self{ value: u64_vec_to_u32_vec(&ark.0).try_into().unwrap(), phantom : PhantomData}
    }

    pub fn to_biginteger256_transmute(&self) -> BigInteger256 {
        unsafe { transmute(*self) }
    }

    pub fn from_biginteger_transmute(v: BigInteger256) -> Scalar {
        Scalar{ value: unsafe{ transmute(v)}, phantom : PhantomData }
    }

    pub fn to_ark_transmute(&self) -> Fr_BLS12_381 {
        unsafe { std::mem::transmute(*self) }
    }

    pub fn from_ark_transmute(v: &Fr_BLS12_381) -> Scalar {
        unsafe { std::mem::transmute_copy(v) }
    }

    pub fn to_ark_mod_p(&self) -> Fr_BLS12_381 {
        Fr_BLS12_381::new(BigInteger256::new(u32_vec_to_u64_vec(&self.limbs()).try_into().unwrap()))
    }

    pub fn to_ark_repr(&self) -> Fr_BLS12_381 {
        Fr_BLS12_381::from_repr(BigInteger256::new(u32_vec_to_u64_vec(&self.limbs()).try_into().unwrap())).unwrap()
    }

    pub fn from_ark(v: BigInteger256) -> Scalar {
        Self { value : u64_vec_to_u32_vec(&v.0).try_into().unwrap(), phantom: PhantomData}
    }

}

impl Base {
    pub fn to_ark(&self) -> BigInteger384 {
        BigInteger384::new(u32_vec_to_u64_vec(&self.limbs()).try_into().unwrap())
    }

    pub fn from_ark(ark: BigInteger384) -> Self {
        Self::from_limbs(&u64_vec_to_u32_vec(&ark.0))
    }
}


impl Point {
    pub fn to_ark(&self) -> G1Projective_BLS12_381 {
        self.to_ark_affine().into_projective()
    }

    pub fn to_ark_affine(&self) -> G1Affine_BLS12_381 {
        //TODO: generic conversion
        use ark_ff::Field;
        use std::ops::Mul;
        let proj_x_field = Fq_BLS12_381::from_le_bytes_mod_order(&self.x.to_bytes_le());
        let proj_y_field = Fq_BLS12_381::from_le_bytes_mod_order(&self.y.to_bytes_le());
        let proj_z_field = Fq_BLS12_381::from_le_bytes_mod_order(&self.z.to_bytes_le());
        let inverse_z = proj_z_field.inverse().unwrap();
        let aff_x = proj_x_field.mul(inverse_z);
        let aff_y = proj_y_field.mul(inverse_z);
        G1Affine_BLS12_381::new(aff_x, aff_y, false)
    }

    pub fn from_ark(ark: G1Projective_BLS12_381) -> Point {
        use ark_ff::Field;
        let z_inv = ark.z.inverse().unwrap();
        let z_invsq = z_inv * z_inv;
        let z_invq3 = z_invsq * z_inv;
        Point {
            x: Base::from_ark((ark.x * z_invsq).into_repr()),
            y: Base::from_ark((ark.y * z_invq3).into_repr()),
            z: Base::one(),
        }
    }
}

impl PointAffineNoInfinity {

    pub fn to_ark(&self) -> G1Affine_BLS12_381 {
        G1Affine_BLS12_381::new(Fq_BLS12_381::new(self.x.to_ark()), Fq_BLS12_381::new(self.y.to_ark()), false)
    }

    pub fn to_ark_repr(&self) -> G1Affine_BLS12_381 {
        G1Affine_BLS12_381::new(
            Fq_BLS12_381::from_repr(self.x.to_ark()).unwrap(),
            Fq_BLS12_381::from_repr(self.y.to_ark()).unwrap(),
            false,
        )
    }

    pub fn from_ark(p: &G1Affine_BLS12_381) -> Self {
        PointAffineNoInfinity {
            x: Base::from_ark(p.x.into_repr()),
            y: Base::from_ark(p.y.into_repr()),
        }
    }
}

impl Point {
    pub fn to_affine(&self) -> PointAffineNoInfinity {
        let ark_affine = self.to_ark_affine();
        PointAffineNoInfinity {
            x: Base::from_ark(ark_affine.x.into_repr()),
            y: Base::from_ark(ark_affine.y.into_repr()),
        }
    }
}


#[cfg(test)]
pub(crate) mod tests_bls12_381 {
    use std::ops::Add;
    use ark_bls12_381::{Fr, G1Affine, G1Projective};
    use ark_ec::{msm::VariableBaseMSM, AffineCurve, ProjectiveCurve};
    use ark_ff::{FftField, Field, Zero, PrimeField};
    use ark_std::UniformRand;
    use rustacuda::prelude::{DeviceBuffer, CopyDestination};
    use crate::curve_structs::{Point, Scalar, Base};
    use crate::basic_structs::scalar::ScalarTrait;
    use crate::from_cuda::{generate_random_points, get_rng, generate_random_scalars, msm, msm_batch, set_up_scalars, commit, commit_batch, ntt, intt, generate_random_points_proj, ecntt, iecntt, ntt_batch, ecntt_batch, iecntt_batch, intt_batch, reverse_order_scalars_batch, interpolate_scalars_batch, set_up_points, reverse_order_points, interpolate_points, reverse_order_points_batch, interpolate_points_batch, evaluate_scalars, interpolate_scalars, reverse_order_scalars, evaluate_points, build_domain, evaluate_scalars_on_coset, evaluate_points_on_coset, mult_matrix_by_vec, mult_sc_vec, multp_vec,evaluate_scalars_batch, evaluate_points_batch, evaluate_scalars_on_coset_batch, evaluate_points_on_coset_batch};

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

    #[test]
    fn test_msm() {
        let test_sizes = [6, 9];

        for pow2 in test_sizes {
            let count = 1 << pow2;
            let seed = None; // set Some to provide seed
            let points = generate_random_points(count, get_rng(seed));
            let scalars = generate_random_scalars(count, get_rng(seed));

            let msm_result = msm(&points, &scalars, 0);

            let point_r_ark: Vec<_> = points.iter().map(|x| x.to_ark_repr()).collect();
            let scalars_r_ark: Vec<_> = scalars.iter().map(|x| x.to_ark()).collect();

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
        for batch_pow2 in [2, 4] {
            for pow2 in [4, 6] {
                let msm_size = 1 << pow2;
                let batch_size = 1 << batch_pow2;
                let seed = None; // set Some to provide seed
                let points_batch = generate_random_points(msm_size * batch_size, get_rng(seed));
                let scalars_batch = generate_random_scalars(msm_size * batch_size, get_rng(seed));

                let point_r_ark: Vec<_> = points_batch.iter().map(|x| x.to_ark_repr()).collect();
                let scalars_r_ark: Vec<_> = scalars_batch.iter().map(|x| x.to_ark()).collect();

                let expected: Vec<_> = point_r_ark
                    .chunks(msm_size)
                    .zip(scalars_r_ark.chunks(msm_size))
                    .map(|p| Point::from_ark(VariableBaseMSM::multi_scalar_mul(p.0, p.1)))
                    .collect();

                let result = msm_batch(&points_batch, &scalars_batch, batch_size, 0);

                assert_eq!(result, expected);
            }
        }
    }

    #[test]
    fn test_commit() {
        let test_size = 1 << 8;
        let seed = Some(0);
        let (mut scalars, mut d_scalars, _) = set_up_scalars(test_size, 0, false);
        let mut points = generate_random_points(test_size, get_rng(seed));
        let mut d_points = DeviceBuffer::from_slice(&points[..]).unwrap();

        let msm_result = msm(&points, &scalars, 0);
        let mut d_commit_result = commit(&mut d_points, &mut d_scalars);
        let mut h_commit_result = Point::zero();
        d_commit_result.copy_to(&mut h_commit_result).unwrap();

        assert_eq!(msm_result, h_commit_result);
        assert_ne!(msm_result, Point::zero());
        assert_ne!(h_commit_result, Point::zero());
    }

    #[test]
    fn test_batch_commit() {
        let batch_size = 4;
        let test_size = 1 << 12;
        let seed = Some(0);
        let (scalars, mut d_scalars, _) = set_up_scalars(test_size * batch_size, 0, false);
        let points = generate_random_points(test_size * batch_size, get_rng(seed));
        let mut d_points = DeviceBuffer::from_slice(&points[..]).unwrap();

        let msm_result = msm_batch(&points, &scalars, batch_size, 0);
        let mut d_commit_result = commit_batch(&mut d_points, &mut d_scalars, batch_size);
        let mut h_commit_result: Vec<Point> = (0..batch_size).map(|_| Point::zero()).collect();
        d_commit_result.copy_to(&mut h_commit_result[..]).unwrap();

        assert_eq!(msm_result, h_commit_result);
        for h in h_commit_result {
            assert_ne!(h, Point::zero());
        }
    }

    #[test]
    fn test_ntt() {
        //NTT
        let seed = None; //some value to fix the rng
        let test_size = 1 << 3;

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
        let test_size = 1 << 5;
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
    fn test_scalar_interpolation() {
        let log_test_size = 7;
        let test_size = 1 << log_test_size;
        let (mut evals_mut, mut d_evals, mut d_domain) = set_up_scalars(test_size, log_test_size, true);

        reverse_order_scalars(&mut d_evals);
        let mut d_coeffs = interpolate_scalars(&mut d_evals, &mut d_domain);
        intt(&mut evals_mut, 0);
        let mut h_coeffs: Vec<Scalar> = (0..test_size).map(|_| Scalar::zero()).collect();
        d_coeffs.copy_to(&mut h_coeffs[..]).unwrap();

        assert_eq!(h_coeffs, evals_mut);
    }

    #[test]
    fn test_scalar_batch_interpolation() {
        let batch_size = 4;
        let log_test_size = 10;
        let test_size = 1 << log_test_size;
        let (mut evals_mut, mut d_evals, mut d_domain) = set_up_scalars(test_size * batch_size, log_test_size, true);

        reverse_order_scalars_batch(&mut d_evals, batch_size);
        let mut d_coeffs = interpolate_scalars_batch(&mut d_evals, &mut d_domain, batch_size);
        intt_batch(&mut evals_mut, test_size, 0);
        let mut h_coeffs: Vec<Scalar> = (0..test_size * batch_size).map(|_| Scalar::zero()).collect();
        d_coeffs.copy_to(&mut h_coeffs[..]).unwrap();

        assert_eq!(h_coeffs, evals_mut);
    }

    #[test]
    fn test_point_interpolation() {
        let log_test_size = 6;
        let test_size = 1 << log_test_size;
        let (mut evals_mut, mut d_evals, mut d_domain) = set_up_points(test_size, log_test_size, true);

        reverse_order_points(&mut d_evals);
        let mut d_coeffs = interpolate_points(&mut d_evals, &mut d_domain);
        iecntt(&mut evals_mut[..], 0);
        let mut h_coeffs: Vec<Point> = (0..test_size).map(|_| Point::zero()).collect();
        d_coeffs.copy_to(&mut h_coeffs[..]).unwrap();
        
        assert_eq!(h_coeffs, *evals_mut);
        for h in h_coeffs.iter() {
            assert_ne!(*h, Point::zero());
        }
    }

    #[test]
    fn test_point_batch_interpolation() {
        let batch_size = 4;
        let log_test_size = 6;
        let test_size = 1 << log_test_size;
        let (mut evals_mut, mut d_evals, mut d_domain) = set_up_points(test_size * batch_size, log_test_size, true);

        reverse_order_points_batch(&mut d_evals, batch_size);
        let mut d_coeffs = interpolate_points_batch(&mut d_evals, &mut d_domain, batch_size);
        iecntt_batch(&mut evals_mut[..], test_size, 0);
        let mut h_coeffs: Vec<Point> = (0..test_size * batch_size).map(|_| Point::zero()).collect();
        d_coeffs.copy_to(&mut h_coeffs[..]).unwrap();
        
        assert_eq!(h_coeffs, *evals_mut);
        for h in h_coeffs.iter() {
            assert_ne!(*h, Point::zero());
        }
    }

    #[test]
    fn test_scalar_evaluation() {
        let log_test_domain_size = 8;
        let coeff_size = 1 << 6;
        let (h_coeffs, mut d_coeffs, mut d_domain) = set_up_scalars(coeff_size, log_test_domain_size, false);
        let (_, _, mut d_domain_inv) = set_up_scalars(0, log_test_domain_size, true);

        let mut d_evals = evaluate_scalars(&mut d_coeffs, &mut d_domain);
        let mut d_coeffs_domain = interpolate_scalars(&mut d_evals, &mut d_domain_inv);
        let mut h_coeffs_domain: Vec<Scalar> = (0..1 << log_test_domain_size).map(|_| Scalar::zero()).collect();
        d_coeffs_domain.copy_to(&mut h_coeffs_domain[..]).unwrap();

        assert_eq!(h_coeffs, h_coeffs_domain[..coeff_size]);
        for i in coeff_size.. (1 << log_test_domain_size) {
            assert_eq!(Scalar::zero(), h_coeffs_domain[i]);
        }
    }

    #[test]
    fn test_scalar_batch_evaluation() {
        let batch_size = 6;
        let log_test_domain_size = 8;
        let domain_size = 1 << log_test_domain_size;
        let coeff_size = 1 << 6;
        let (h_coeffs, mut d_coeffs, mut d_domain) = set_up_scalars(coeff_size * batch_size, log_test_domain_size, false);
        let (_, _, mut d_domain_inv) = set_up_scalars(0, log_test_domain_size, true);

        let mut d_evals = evaluate_scalars_batch(&mut d_coeffs, &mut d_domain, batch_size);
        let mut d_coeffs_domain = interpolate_scalars_batch(&mut d_evals, &mut d_domain_inv, batch_size);
        let mut h_coeffs_domain: Vec<Scalar> = (0..domain_size * batch_size).map(|_| Scalar::zero()).collect();
        d_coeffs_domain.copy_to(&mut h_coeffs_domain[..]).unwrap();

        for j in 0..batch_size {
            assert_eq!(h_coeffs[j * coeff_size..(j + 1) * coeff_size], h_coeffs_domain[j * domain_size..j * domain_size + coeff_size]);
            for i in coeff_size..domain_size {
                assert_eq!(Scalar::zero(), h_coeffs_domain[j * domain_size + i]);
            }
        }
    }

    #[test]
    fn test_point_evaluation() {
        let log_test_domain_size = 7;
        let coeff_size = 1 << 7;
        let (h_coeffs, mut d_coeffs, mut d_domain) = set_up_points(coeff_size, log_test_domain_size, false);
        let (_, _, mut d_domain_inv) = set_up_points(0, log_test_domain_size, true);

        let mut d_evals = evaluate_points(&mut d_coeffs, &mut d_domain);
        let mut d_coeffs_domain = interpolate_points(&mut d_evals, &mut d_domain_inv);
        let mut h_coeffs_domain: Vec<Point> = (0..1 << log_test_domain_size).map(|_| Point::zero()).collect();
        d_coeffs_domain.copy_to(&mut h_coeffs_domain[..]).unwrap();

        assert_eq!(h_coeffs[..], h_coeffs_domain[..coeff_size]);
        for i in coeff_size..(1 << log_test_domain_size) {
            assert_eq!(Point::zero(), h_coeffs_domain[i]);
        }
        for i in 0..coeff_size {
            assert_ne!(h_coeffs_domain[i], Point::zero());
        }
    }

    #[test]
    fn test_point_batch_evaluation() {
        let batch_size = 4;
        let log_test_domain_size = 6;
        let domain_size = 1 << log_test_domain_size;
        let coeff_size = 1 << 5;
        let (h_coeffs, mut d_coeffs, mut d_domain) = set_up_points(coeff_size * batch_size, log_test_domain_size, false);
        let (_, _, mut d_domain_inv) = set_up_points(0, log_test_domain_size, true);

        let mut d_evals = evaluate_points_batch(&mut d_coeffs, &mut d_domain, batch_size);
        let mut d_coeffs_domain = interpolate_points_batch(&mut d_evals, &mut d_domain_inv, batch_size);
        let mut h_coeffs_domain: Vec<Point> = (0..domain_size * batch_size).map(|_| Point::zero()).collect();
        d_coeffs_domain.copy_to(&mut h_coeffs_domain[..]).unwrap();

        for j in 0..batch_size {
            assert_eq!(h_coeffs[j * coeff_size..(j + 1) * coeff_size], h_coeffs_domain[j * domain_size..(j * domain_size + coeff_size)]);
            for i in coeff_size..domain_size {
                assert_eq!(Point::zero(), h_coeffs_domain[j * domain_size + i]);
            }
            for i in j * domain_size..(j * domain_size + coeff_size) {
                assert_ne!(h_coeffs_domain[i], Point::zero());
            }
        }
    }

    #[test]
    fn test_scalar_evaluation_on_trivial_coset() {
        // checks that the evaluations on the subgroup is the same as on the coset generated by 1
        let log_test_domain_size = 8;
        let coeff_size = 1 << 6;
        let (_, mut d_coeffs, mut d_domain) = set_up_scalars(coeff_size, log_test_domain_size, false);
        let (_, _, mut d_domain_inv) = set_up_scalars(coeff_size, log_test_domain_size, true);
        let mut d_trivial_coset_powers = build_domain(1 << log_test_domain_size, 0, false);

        let mut d_evals = evaluate_scalars(&mut d_coeffs, &mut d_domain);
        let mut h_coeffs: Vec<Scalar> = (0..1 << log_test_domain_size).map(|_| Scalar::zero()).collect();
        d_evals.copy_to(&mut h_coeffs[..]).unwrap();
        let mut d_evals_coset = evaluate_scalars_on_coset(&mut d_coeffs, &mut d_domain, &mut d_trivial_coset_powers);
        let mut h_evals_coset: Vec<Scalar> = (0..1 << log_test_domain_size).map(|_| Scalar::zero()).collect();
        d_evals_coset.copy_to(&mut h_evals_coset[..]).unwrap();

        assert_eq!(h_coeffs, h_evals_coset);
    }

    #[test]
    fn test_scalar_evaluation_on_coset() {
        // checks that evaluating a polynomial on a subgroup and its coset is the same as evaluating on a 2x larger subgroup 
        let log_test_size = 8;
        let test_size = 1 << log_test_size;
        let (_, mut d_coeffs, mut d_domain) = set_up_scalars(test_size, log_test_size, false);
        let (_, _, mut d_large_domain) = set_up_scalars(0, log_test_size + 1, false);
        let mut d_coset_powers = build_domain(test_size, log_test_size + 1, false);

        let mut d_evals_large = evaluate_scalars(&mut d_coeffs, &mut d_large_domain);
        let mut h_evals_large: Vec<Scalar> = (0..2 * test_size).map(|_| Scalar::zero()).collect();
        d_evals_large.copy_to(&mut h_evals_large[..]).unwrap();
        let mut d_evals = evaluate_scalars(&mut d_coeffs, &mut d_domain);
        let mut h_evals: Vec<Scalar> = (0..test_size).map(|_| Scalar::zero()).collect();
        d_evals.copy_to(&mut h_evals[..]).unwrap();
        let mut d_evals_coset = evaluate_scalars_on_coset(&mut d_coeffs, &mut d_domain, &mut d_coset_powers);
        let mut h_evals_coset: Vec<Scalar> = (0..test_size).map(|_| Scalar::zero()).collect();
        d_evals_coset.copy_to(&mut h_evals_coset[..]).unwrap();

        assert_eq!(h_evals[..], h_evals_large[..test_size]);
        assert_eq!(h_evals_coset[..], h_evals_large[test_size..2 * test_size]);
    }

    #[test]
    fn test_scalar_batch_evaluation_on_coset() {
        // checks that evaluating a polynomial on a subgroup and its coset is the same as evaluating on a 2x larger subgroup 
        let batch_size = 4;
        let log_test_size = 6;
        let test_size = 1 << log_test_size;
        let (_, mut d_coeffs, mut d_domain) = set_up_scalars(test_size * batch_size, log_test_size, false);
        let (_, _, mut d_large_domain) = set_up_scalars(0, log_test_size + 1, false);
        let mut d_coset_powers = build_domain(test_size, log_test_size + 1, false);

        let mut d_evals_large = evaluate_scalars_batch(&mut d_coeffs, &mut d_large_domain, batch_size);
        let mut h_evals_large: Vec<Scalar> = (0..2 * test_size * batch_size).map(|_| Scalar::zero()).collect();
        d_evals_large.copy_to(&mut h_evals_large[..]).unwrap();
        let mut d_evals = evaluate_scalars_batch(&mut d_coeffs, &mut d_domain, batch_size);
        let mut h_evals: Vec<Scalar> = (0..test_size * batch_size).map(|_| Scalar::zero()).collect();
        d_evals.copy_to(&mut h_evals[..]).unwrap();
        let mut d_evals_coset = evaluate_scalars_on_coset_batch(&mut d_coeffs, &mut d_domain, batch_size, &mut d_coset_powers);
        let mut h_evals_coset: Vec<Scalar> = (0..test_size * batch_size).map(|_| Scalar::zero()).collect();
        d_evals_coset.copy_to(&mut h_evals_coset[..]).unwrap();

        for i in 0..batch_size {
            assert_eq!(h_evals_large[2 * i * test_size..(2 * i + 1) * test_size], h_evals[i * test_size..(i + 1) * test_size]);
            assert_eq!(h_evals_large[(2 * i + 1) * test_size..(2 * i + 2) * test_size], h_evals_coset[i * test_size..(i + 1) * test_size]);
        }
    }

    #[test]
    fn test_point_evaluation_on_coset() {
        // checks that evaluating a polynomial on a subgroup and its coset is the same as evaluating on a 2x larger subgroup 
        let log_test_size = 8;
        let test_size = 1 << log_test_size;
        let (_, mut d_coeffs, mut d_domain) = set_up_points(test_size, log_test_size, false);
        let (_, _, mut d_large_domain) = set_up_points(0, log_test_size + 1, false);
        let mut d_coset_powers = build_domain(test_size, log_test_size + 1, false);

        let mut d_evals_large = evaluate_points(&mut d_coeffs, &mut d_large_domain);
        let mut h_evals_large: Vec<Point> = (0..2 * test_size).map(|_| Point::zero()).collect();
        d_evals_large.copy_to(&mut h_evals_large[..]).unwrap();
        let mut d_evals = evaluate_points(&mut d_coeffs, &mut d_domain);
        let mut h_evals: Vec<Point> = (0..test_size).map(|_| Point::zero()).collect();
        d_evals.copy_to(&mut h_evals[..]).unwrap();
        let mut d_evals_coset = evaluate_points_on_coset(&mut d_coeffs, &mut d_domain, &mut d_coset_powers);
        let mut h_evals_coset: Vec<Point> = (0..test_size).map(|_| Point::zero()).collect();
        d_evals_coset.copy_to(&mut h_evals_coset[..]).unwrap();

        assert_eq!(h_evals[..], h_evals_large[..test_size]);
        assert_eq!(h_evals_coset[..], h_evals_large[test_size..2 * test_size]);
        for i in 0..test_size {
            assert_ne!(h_evals[i], Point::zero());
            assert_ne!(h_evals_coset[i], Point::zero());
            assert_ne!(h_evals_large[2 * i], Point::zero());
            assert_ne!(h_evals_large[2 * i + 1], Point::zero());
        }
    }

    #[test]
    fn test_point_batch_evaluation_on_coset() {
        // checks that evaluating a polynomial on a subgroup and its coset is the same as evaluating on a 2x larger subgroup 
        let batch_size = 2;
        let log_test_size = 6;
        let test_size = 1 << log_test_size;
        let (_, mut d_coeffs, mut d_domain) = set_up_points(test_size * batch_size, log_test_size, false);
        let (_, _, mut d_large_domain) = set_up_points(0, log_test_size + 1, false);
        let mut d_coset_powers = build_domain(test_size, log_test_size + 1, false);

        let mut d_evals_large = evaluate_points_batch(&mut d_coeffs, &mut d_large_domain, batch_size);
        let mut h_evals_large: Vec<Point> = (0..2 * test_size * batch_size).map(|_| Point::zero()).collect();
        d_evals_large.copy_to(&mut h_evals_large[..]).unwrap();
        let mut d_evals = evaluate_points_batch(&mut d_coeffs, &mut d_domain, batch_size);
        let mut h_evals: Vec<Point> = (0..test_size * batch_size).map(|_| Point::zero()).collect();
        d_evals.copy_to(&mut h_evals[..]).unwrap();
        let mut d_evals_coset = evaluate_points_on_coset_batch(&mut d_coeffs, &mut d_domain, batch_size, &mut d_coset_powers);
        let mut h_evals_coset: Vec<Point> = (0..test_size * batch_size).map(|_| Point::zero()).collect();
        d_evals_coset.copy_to(&mut h_evals_coset[..]).unwrap();

        for i in 0..batch_size {
            assert_eq!(h_evals_large[2 * i * test_size..(2 * i + 1) * test_size], h_evals[i * test_size..(i + 1) * test_size]);
            assert_eq!(h_evals_large[(2 * i + 1) * test_size..(2 * i + 2) * test_size], h_evals_coset[i * test_size..(i + 1) * test_size]);
        }
        for i in 0..test_size * batch_size {
            assert_ne!(h_evals[i], Point::zero());
            assert_ne!(h_evals_coset[i], Point::zero());
            assert_ne!(h_evals_large[2 * i], Point::zero());
            assert_ne!(h_evals_large[2 * i + 1], Point::zero());
        }
    }

    // testing matrix multiplication by comparing the result of FFT with the naive multiplication by the DFT matrix
    #[test]
    fn test_matrix_multiplication() {
        let seed = None; // some value to fix the rng
        let test_size = 1 << 5;
        let rou = Fr::get_root_of_unity(test_size).unwrap();
        let matrix_flattened: Vec<Scalar> = (0..test_size).map(
            |row_num| { (0..test_size).map( 
                |col_num| {
                    let pow: [u64; 1] = [(row_num * col_num).try_into().unwrap()];
                    Scalar::from_ark(Fr::pow(&rou, &pow).into_repr())
                }).collect::<Vec<Scalar>>()
            }).flatten().collect::<Vec<_>>();
        let vector: Vec<Scalar> = generate_random_scalars(test_size, get_rng(seed));

        let result = mult_matrix_by_vec(&matrix_flattened, &vector, 0);
        let mut ntt_result = vector.clone();
        ntt(&mut ntt_result, 0);
        
        // we don't use the same roots of unity as arkworks, so the results are permutations
        // of one another and the only guaranteed fixed scalars are the following ones:
        assert_eq!(result[0], ntt_result[0]);
        assert_eq!(result[test_size >> 1], ntt_result[test_size >> 1]);
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
            x: Base::one(),
            y: Base::one(),
            z: Base::one(),
        };

        let mut inout = [dummy_one, dummy_one, Point::zero()];
        let scalars = [Scalar::one(), Scalar::zero(), Scalar::zero()];
        let expected = [dummy_one, Point::zero(), Point::zero()];
        multp_vec(&mut inout, &scalars, 0);
        assert_eq!(inout, expected);
    }
}
