use crate::curves::f251::*;
use ark_ff::PrimeField;
use ark_std::UniformRand;
use rand::{rngs::StdRng, RngCore, SeedableRng};
use rustacuda::memory::{CopyDestination, DeviceBox, DeviceCopy};
use rustacuda::prelude::*;
use rustacuda_core::DevicePointer;
use std::ffi::{c_int, c_uint};

extern "C" {
    fn build_domain_cuda_f251(
        domain_size: usize,
        logn: usize,
        inverse: bool,
        device_id: usize,
    ) -> DevicePointer<ScalarField_F251>;

    fn ntt_cuda_f251(inout: *mut ScalarField_F251, n: usize, inverse: bool, device_id: usize) -> c_int;

    fn ntt_batch_cuda_f251(inout: *mut ScalarField_F251, arr_size: usize, n: usize, inverse: bool) -> c_int;

    fn ntt_inplace_batch_cuda_f251(
        d_inout: DevicePointer<ScalarField_F251>,
        d_twiddles: DevicePointer<ScalarField_F251>,
        n: usize,
        batch_size: usize,
        inverse: bool,
        device_id: usize,
    ) -> c_int;

    fn interpolate_scalars_cuda_f251(
        d_out: DevicePointer<ScalarField_F251>,
        d_evaluations: DevicePointer<ScalarField_F251>,
        d_domain: DevicePointer<ScalarField_F251>,
        n: usize,
        device_id: usize,
    ) -> c_int;

    fn interpolate_scalars_batch_cuda_f251(
        d_out: DevicePointer<ScalarField_F251>,
        d_evaluations: DevicePointer<ScalarField_F251>,
        d_domain: DevicePointer<ScalarField_F251>,
        n: usize,
        batch_size: usize,
        device_id: usize,
    ) -> c_int;

    fn evaluate_scalars_cuda_f251(
        d_out: DevicePointer<ScalarField_F251>,
        d_coefficients: DevicePointer<ScalarField_F251>,
        d_domain: DevicePointer<ScalarField_F251>,
        domain_size: usize,
        n: usize,
        device_id: usize,
    ) -> c_int;

    fn evaluate_scalars_batch_cuda_f251(
        d_out: DevicePointer<ScalarField_F251>,
        d_coefficients: DevicePointer<ScalarField_F251>,
        d_domain: DevicePointer<ScalarField_F251>,
        domain_size: usize,
        n: usize,
        batch_size: usize,
        device_id: usize,
    ) -> c_int;

    fn evaluate_scalars_on_coset_cuda_f251(
        d_out: DevicePointer<ScalarField_F251>,
        d_coefficients: DevicePointer<ScalarField_F251>,
        d_domain: DevicePointer<ScalarField_F251>,
        domain_size: usize,
        n: usize,
        coset_powers: DevicePointer<ScalarField_F251>,
        device_id: usize,
    ) -> c_int;

    fn evaluate_scalars_on_coset_batch_cuda_f251(
        d_out: DevicePointer<ScalarField_F251>,
        d_coefficients: DevicePointer<ScalarField_F251>,
        d_domain: DevicePointer<ScalarField_F251>,
        domain_size: usize,
        n: usize,
        batch_size: usize,
        coset_powers: DevicePointer<ScalarField_F251>,
        device_id: usize,
    ) -> c_int;

    fn reverse_order_scalars_cuda_f251(d_arr: DevicePointer<ScalarField_F251>, n: usize, device_id: usize) -> c_int;

    fn reverse_order_scalars_batch_cuda_f251(
        d_arr: DevicePointer<ScalarField_F251>,
        n: usize,
        batch_size: usize,
        device_id: usize,
    ) -> c_int;

    fn vec_mod_mult_scalar_f251(
        inout: *mut ScalarField_F251,
        scalars: *const ScalarField_F251,
        n_elements: usize,
        device_id: usize,
    ) -> c_int;

    fn matrix_vec_mod_mult_f251(
        matrix_flattened: *const ScalarField_F251,
        input: *const ScalarField_F251,
        output: *mut ScalarField_F251,
        n_elements: usize,
        device_id: usize,
    ) -> c_int;
}

/// Compute an in-place NTT on the input data.
fn ntt_internal_f251(values: &mut [ScalarField_F251], device_id: usize, inverse: bool) -> i32 {
    let ret_code = unsafe {
        ntt_cuda_f251(
            values as *mut _ as *mut ScalarField_F251,
            values.len(),
            inverse,
            device_id,
        )
    };
    ret_code
}

pub fn ntt_f251(values: &mut [ScalarField_F251], device_id: usize) {
    ntt_internal_f251(values, device_id, false);
}

pub fn intt_f251(values: &mut [ScalarField_F251], device_id: usize) {
    ntt_internal_f251(values, device_id, true);
}

/// Compute an in-place NTT on the input data.
fn ntt_internal_batch_f251(values: &mut [ScalarField_F251], device_id: usize, batch_size: usize, inverse: bool) -> i32 {
    unsafe {
        ntt_batch_cuda_f251(
            values as *mut _ as *mut ScalarField_F251,
            values.len(),
            batch_size,
            inverse,
        )
    }
}

pub fn ntt_batch_f251(values: &mut [ScalarField_F251], batch_size: usize, device_id: usize) {
    ntt_internal_batch_f251(values, 0, batch_size, false);
}

pub fn intt_batch_f251(values: &mut [ScalarField_F251], batch_size: usize, device_id: usize) {
    ntt_internal_batch_f251(values, 0, batch_size, true);
}

pub fn build_domain_f251(domain_size: usize, logn: usize, inverse: bool) -> DeviceBuffer<ScalarField_F251> {
    unsafe { DeviceBuffer::from_raw_parts(build_domain_cuda_f251(domain_size, logn, inverse, 0), domain_size) }
}

pub fn reverse_order_scalars_f251(d_scalars: &mut DeviceBuffer<ScalarField_F251>) {
    unsafe {
        reverse_order_scalars_cuda_f251(d_scalars.as_device_ptr(), d_scalars.len(), 0);
    }
}

pub fn reverse_order_scalars_batch_f251(d_scalars: &mut DeviceBuffer<ScalarField_F251>, batch_size: usize) {
    unsafe {
        reverse_order_scalars_batch_cuda_f251(d_scalars.as_device_ptr(), d_scalars.len() / batch_size, batch_size, 0);
    }
}

pub fn interpolate_scalars_f251(
    d_evaluations: &mut DeviceBuffer<ScalarField_F251>,
    d_domain: &mut DeviceBuffer<ScalarField_F251>,
) -> DeviceBuffer<ScalarField_F251> {
    let mut res = unsafe { DeviceBuffer::uninitialized(d_domain.len()).unwrap() };
    unsafe {
        interpolate_scalars_cuda_f251(
            res.as_device_ptr(),
            d_evaluations.as_device_ptr(),
            d_domain.as_device_ptr(),
            d_domain.len(),
            0,
        )
    };
    return res;
}

pub fn interpolate_scalars_batch_f251(
    d_evaluations: &mut DeviceBuffer<ScalarField_F251>,
    d_domain: &mut DeviceBuffer<ScalarField_F251>,
    batch_size: usize,
) -> DeviceBuffer<ScalarField_F251> {
    let mut res = unsafe { DeviceBuffer::uninitialized(d_domain.len() * batch_size).unwrap() };
    unsafe {
        interpolate_scalars_batch_cuda_f251(
            res.as_device_ptr(),
            d_evaluations.as_device_ptr(),
            d_domain.as_device_ptr(),
            d_domain.len(),
            batch_size,
            0,
        )
    };
    return res;
}

pub fn evaluate_scalars_f251(
    d_coefficients: &mut DeviceBuffer<ScalarField_F251>,
    d_domain: &mut DeviceBuffer<ScalarField_F251>,
) -> DeviceBuffer<ScalarField_F251> {
    let mut res = unsafe { DeviceBuffer::uninitialized(d_domain.len()).unwrap() };
    unsafe {
        evaluate_scalars_cuda_f251(
            res.as_device_ptr(),
            d_coefficients.as_device_ptr(),
            d_domain.as_device_ptr(),
            d_domain.len(),
            d_coefficients.len(),
            0,
        );
    }
    return res;
}

pub fn evaluate_scalars_batch_f251(
    d_coefficients: &mut DeviceBuffer<ScalarField_F251>,
    d_domain: &mut DeviceBuffer<ScalarField_F251>,
    batch_size: usize,
) -> DeviceBuffer<ScalarField_F251> {
    let mut res = unsafe { DeviceBuffer::uninitialized(d_domain.len() * batch_size).unwrap() };
    unsafe {
        evaluate_scalars_batch_cuda_f251(
            res.as_device_ptr(),
            d_coefficients.as_device_ptr(),
            d_domain.as_device_ptr(),
            d_domain.len(),
            d_coefficients.len() / batch_size,
            batch_size,
            0,
        );
    }
    return res;
}

pub fn evaluate_scalars_on_coset_f251(
    d_coefficients: &mut DeviceBuffer<ScalarField_F251>,
    d_domain: &mut DeviceBuffer<ScalarField_F251>,
    coset_powers: &mut DeviceBuffer<ScalarField_F251>,
) -> DeviceBuffer<ScalarField_F251> {
    let mut res = unsafe { DeviceBuffer::uninitialized(d_domain.len()).unwrap() };
    unsafe {
        evaluate_scalars_on_coset_cuda_f251(
            res.as_device_ptr(),
            d_coefficients.as_device_ptr(),
            d_domain.as_device_ptr(),
            d_domain.len(),
            d_coefficients.len(),
            coset_powers.as_device_ptr(),
            0,
        );
    }
    return res;
}

pub fn evaluate_scalars_on_coset_batch_f251(
    d_coefficients: &mut DeviceBuffer<ScalarField_F251>,
    d_domain: &mut DeviceBuffer<ScalarField_F251>,
    batch_size: usize,
    coset_powers: &mut DeviceBuffer<ScalarField_F251>,
) -> DeviceBuffer<ScalarField_F251> {
    let mut res = unsafe { DeviceBuffer::uninitialized(d_domain.len() * batch_size).unwrap() };
    unsafe {
        evaluate_scalars_on_coset_batch_cuda_f251(
            res.as_device_ptr(),
            d_coefficients.as_device_ptr(),
            d_domain.as_device_ptr(),
            d_domain.len(),
            d_coefficients.len() / batch_size,
            batch_size,
            coset_powers.as_device_ptr(),
            0,
        );
    }
    return res;
}

pub fn ntt_inplace_batch_f251(
    d_inout: &mut DeviceBuffer<ScalarField_F251>,
    d_twiddles: &mut DeviceBuffer<ScalarField_F251>,
    batch_size: usize,
    inverse: bool,
    device_id: usize,
) -> i32 {
    unsafe {
        ntt_inplace_batch_cuda_f251(
            d_inout.as_device_ptr(),
            d_twiddles.as_device_ptr(),
            d_twiddles.len(),
            batch_size,
            inverse,
            device_id,
        )
    }
}

pub fn mult_sc_vec_f251(a: &mut [ScalarField_F251], b: &[ScalarField_F251], device_id: usize) {
    assert_eq!(a.len(), b.len());
    unsafe {
        vec_mod_mult_scalar_f251(
            a as *mut _ as *mut ScalarField_F251,
            b as *const _ as *const ScalarField_F251,
            a.len(),
            device_id,
        );
    }
}

// Multiply a matrix by a scalar:
//  `a` - flattenned matrix;
//  `b` - vector to multiply `a` by;
pub fn mult_matrix_by_vec_f251(
    a: &[ScalarField_F251],
    b: &[ScalarField_F251],
    device_id: usize,
) -> Vec<ScalarField_F251> {
    let mut c = Vec::with_capacity(b.len());
    for i in 0..b.len() {
        c.push(ScalarField_F251::zero());
    }
    unsafe {
        matrix_vec_mod_mult_f251(
            a as *const _ as *const ScalarField_F251,
            b as *const _ as *const ScalarField_F251,
            c.as_mut_slice() as *mut _ as *mut ScalarField_F251,
            b.len(),
            device_id,
        );
    }
    c
}

pub fn clone_buffer_f251<T: DeviceCopy>(buf: &mut DeviceBuffer<T>) -> DeviceBuffer<T> {
    let mut buf_cpy = unsafe { DeviceBuffer::uninitialized(buf.len()).unwrap() };
    unsafe { buf_cpy.copy_from(buf) };
    return buf_cpy;
}

pub fn get_rng_f251(seed: Option<u64>) -> Box<dyn RngCore> {
    //TODO: not curve specific
    let rng: Box<dyn RngCore> = match seed {
        Some(seed) => Box::new(StdRng::seed_from_u64(seed)),
        None => Box::new(rand::thread_rng()),
    };
    rng
}

fn set_up_device_f251() {
    // Set up the context, load the module, and create a stream to run kernels in.
    rustacuda::init(CudaFlags::empty()).unwrap();
    let device = Device::get_device(0).unwrap();
    let _ctx = Context::create_and_push(ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device).unwrap();
}

pub fn generate_random_scalars_f251(count: usize, mut rng: Box<dyn RngCore>) -> Vec<ScalarField_F251> {
    (0..count)
        .map(|_| {
            let mut limbs: Vec<_> = (0..SCALAR_LIMBS_F251)
                .map(|_| rng.next_u32())
                .collect();
            limbs[SCALAR_LIMBS_F251 - 1] = 0;
            ScalarField_F251::from_limbs(&limbs)
        })
        .collect()
}

pub fn set_up_scalars_f251(
    test_size: usize,
    log_domain_size: usize,
    inverse: bool,
) -> (
    Vec<ScalarField_F251>,
    DeviceBuffer<ScalarField_F251>,
    DeviceBuffer<ScalarField_F251>,
) {
    set_up_device_f251();

    let d_domain = build_domain_f251(1 << log_domain_size, log_domain_size, inverse);

    let seed = Some(0); // fix the rng to get two equal scalars
    let mut vector_mut = generate_random_scalars_f251(test_size, get_rng_f251(seed));

    let mut d_vector = DeviceBuffer::from_slice(&vector_mut[..]).unwrap();
    (vector_mut, d_vector, d_domain)
}

#[cfg(test)]
pub(crate) mod tests_f251 {
    use crate::test_f251::*;
    use crate::{curves::f251::*, *};
    use ark_ff::{FftField, Field, Zero};
    use ark_std::UniformRand;
    use rand::{rngs::StdRng, RngCore, SeedableRng};
    use std::ops::Add;

    #[test]
    fn test_ntt() {
        //NTT
        let seed = None; //some value to fix the rng
        let test_size = 1 << 3;

        let scalars = generate_random_scalars_f251(test_size, get_rng_f251(seed));

        let mut ntt_result = scalars.clone();
        ntt_f251(&mut ntt_result, 0);

        assert_ne!(ntt_result, scalars);

        let mut intt_result = ntt_result.clone();

        intt_f251(&mut intt_result, 0);

        assert_eq!(intt_result, scalars);
    }

    #[test]
    fn test_ntt_batch() {
        //NTT
        let seed = None; //some value to fix the rng
        let test_size = 1 << 5;
        let batches = 4;

        let scalars_batch: Vec<ScalarField_F251> =
            generate_random_scalars_f251(test_size * batches, get_rng_f251(seed));

        let mut scalar_vec_of_vec: Vec<Vec<ScalarField_F251>> = Vec::new();

        for i in 0..batches {
            scalar_vec_of_vec.push(scalars_batch[i * test_size..(i + 1) * test_size].to_vec());
        }

        let mut ntt_result = scalars_batch.clone();

        // do batch ntt
        ntt_batch_f251(&mut ntt_result, test_size, 0);

        let mut ntt_result_vec_of_vec = Vec::new();

        // do ntt for every chunk
        for i in 0..batches {
            ntt_result_vec_of_vec.push(scalar_vec_of_vec[i].clone());
            ntt_f251(&mut ntt_result_vec_of_vec[i], 0);
        }

        // check that the ntt of each vec of scalars is equal to the intt of the specific batch
        for i in 0..batches {
            assert_eq!(ntt_result_vec_of_vec[i], ntt_result[i * test_size..(i + 1) * test_size]);
        }

        // check that ntt output is different from input
        assert_ne!(ntt_result, scalars_batch);

        let mut intt_result = ntt_result.clone();

        // do batch intt
        intt_batch_f251(&mut intt_result, test_size, 0);

        let mut intt_result_vec_of_vec = Vec::new();

        // do intt for every chunk
        for i in 0..batches {
            intt_result_vec_of_vec.push(ntt_result_vec_of_vec[i].clone());
            intt_f251(&mut intt_result_vec_of_vec[i], 0);
        }

        // check that the intt of each vec of scalars is equal to the intt of the specific batch
        for i in 0..batches {
            assert_eq!(
                intt_result_vec_of_vec[i],
                intt_result[i * test_size..(i + 1) * test_size]
            );
        }

        assert_eq!(intt_result, scalars_batch);
    }

    #[test]
    fn test_scalar_interpolation() {
        let log_test_size = 7;
        let test_size = 1 << log_test_size;
        let (mut evals_mut, mut d_evals, mut d_domain) = set_up_scalars_f251(test_size, log_test_size, true);

        let mut d_coeffs = interpolate_scalars_f251(&mut d_evals, &mut d_domain);
        intt_f251(&mut evals_mut, 0);
        let mut h_coeffs: Vec<ScalarField_F251> = (0..test_size)
            .map(|_| ScalarField_F251::zero())
            .collect();
        d_coeffs
            .copy_to(&mut h_coeffs[..])
            .unwrap();

        assert_eq!(h_coeffs, evals_mut);
    }

    #[test]
    fn test_scalar_batch_interpolation() {
        let batch_size = 4;
        let log_test_size = 10;
        let test_size = 1 << log_test_size;
        let (mut evals_mut, mut d_evals, mut d_domain) =
            set_up_scalars_f251(test_size * batch_size, log_test_size, true);

        let mut d_coeffs = interpolate_scalars_batch_f251(&mut d_evals, &mut d_domain, batch_size);
        intt_batch_f251(&mut evals_mut, test_size, 0);
        let mut h_coeffs: Vec<ScalarField_F251> = (0..test_size * batch_size)
            .map(|_| ScalarField_F251::zero())
            .collect();
        d_coeffs
            .copy_to(&mut h_coeffs[..])
            .unwrap();

        assert_eq!(h_coeffs, evals_mut);
    }

    #[test]
    fn test_scalar_evaluation() {
        let log_test_domain_size = 8;
        let coeff_size = 1 << 6;
        let (h_coeffs, mut d_coeffs, mut d_domain) = set_up_scalars_f251(coeff_size, log_test_domain_size, false);
        let (_, _, mut d_domain_inv) = set_up_scalars_f251(0, log_test_domain_size, true);

        let mut d_evals = evaluate_scalars_f251(&mut d_coeffs, &mut d_domain);
        let mut d_coeffs_domain = interpolate_scalars_f251(&mut d_evals, &mut d_domain_inv);
        let mut h_coeffs_domain: Vec<ScalarField_F251> = (0..1 << log_test_domain_size)
            .map(|_| ScalarField_F251::zero())
            .collect();
        d_coeffs_domain
            .copy_to(&mut h_coeffs_domain[..])
            .unwrap();

        assert_eq!(h_coeffs, h_coeffs_domain[..coeff_size]);
        for i in coeff_size..(1 << log_test_domain_size) {
            assert_eq!(ScalarField_F251::zero(), h_coeffs_domain[i]);
        }
    }

    #[test]
    fn test_scalar_batch_evaluation() {
        let batch_size = 6;
        let log_test_domain_size = 8;
        let domain_size = 1 << log_test_domain_size;
        let coeff_size = 1 << 6;
        let (h_coeffs, mut d_coeffs, mut d_domain) =
            set_up_scalars_f251(coeff_size * batch_size, log_test_domain_size, false);
        let (_, _, mut d_domain_inv) = set_up_scalars_f251(0, log_test_domain_size, true);

        let mut d_evals = evaluate_scalars_batch_f251(&mut d_coeffs, &mut d_domain, batch_size);
        let mut d_coeffs_domain = interpolate_scalars_batch_f251(&mut d_evals, &mut d_domain_inv, batch_size);
        let mut h_coeffs_domain: Vec<ScalarField_F251> = (0..domain_size * batch_size)
            .map(|_| ScalarField_F251::zero())
            .collect();
        d_coeffs_domain
            .copy_to(&mut h_coeffs_domain[..])
            .unwrap();

        for j in 0..batch_size {
            assert_eq!(
                h_coeffs[j * coeff_size..(j + 1) * coeff_size],
                h_coeffs_domain[j * domain_size..j * domain_size + coeff_size]
            );
            for i in coeff_size..domain_size {
                assert_eq!(ScalarField_F251::zero(), h_coeffs_domain[j * domain_size + i]);
            }
        }
    }

    #[test]
    fn test_scalar_evaluation_on_trivial_coset() {
        // checks that the evaluations on the subgroup is the same as on the coset generated by 1
        let log_test_domain_size = 8;
        let coeff_size = 1 << 6;
        let (_, mut d_coeffs, mut d_domain) = set_up_scalars_f251(coeff_size, log_test_domain_size, false);
        let (_, _, mut d_domain_inv) = set_up_scalars_f251(coeff_size, log_test_domain_size, true);
        let mut d_trivial_coset_powers = build_domain_f251(1 << log_test_domain_size, 0, false);

        let mut d_evals = evaluate_scalars_f251(&mut d_coeffs, &mut d_domain);
        let mut h_coeffs: Vec<ScalarField_F251> = (0..1 << log_test_domain_size)
            .map(|_| ScalarField_F251::zero())
            .collect();
        d_evals
            .copy_to(&mut h_coeffs[..])
            .unwrap();
        let mut d_evals_coset =
            evaluate_scalars_on_coset_f251(&mut d_coeffs, &mut d_domain, &mut d_trivial_coset_powers);
        let mut h_evals_coset: Vec<ScalarField_F251> = (0..1 << log_test_domain_size)
            .map(|_| ScalarField_F251::zero())
            .collect();
        d_evals_coset
            .copy_to(&mut h_evals_coset[..])
            .unwrap();

        assert_eq!(h_coeffs, h_evals_coset);
    }

    #[test]
    fn test_scalar_evaluation_on_coset() {
        // checks that evaluating a polynomial on a subgroup and its coset is the same as evaluating on a 2x larger subgroup
        let log_test_size = 8;
        let test_size = 1 << log_test_size;
        let (_, mut d_coeffs, mut d_domain) = set_up_scalars_f251(test_size, log_test_size, false);
        let (_, _, mut d_large_domain) = set_up_scalars_f251(0, log_test_size + 1, false);
        let mut d_coset_powers = build_domain_f251(test_size, log_test_size + 1, false);

        let mut d_evals_large = evaluate_scalars_f251(&mut d_coeffs, &mut d_large_domain);
        let mut h_evals_large: Vec<ScalarField_F251> = (0..2 * test_size)
            .map(|_| ScalarField_F251::zero())
            .collect();
        d_evals_large
            .copy_to(&mut h_evals_large[..])
            .unwrap();
        let mut d_evals = evaluate_scalars_f251(&mut d_coeffs, &mut d_domain);
        let mut h_evals: Vec<ScalarField_F251> = (0..test_size)
            .map(|_| ScalarField_F251::zero())
            .collect();
        d_evals
            .copy_to(&mut h_evals[..])
            .unwrap();
        let mut d_evals_coset = evaluate_scalars_on_coset_f251(&mut d_coeffs, &mut d_domain, &mut d_coset_powers);
        let mut h_evals_coset: Vec<ScalarField_F251> = (0..test_size)
            .map(|_| ScalarField_F251::zero())
            .collect();
        d_evals_coset
            .copy_to(&mut h_evals_coset[..])
            .unwrap();

        assert_eq!(h_evals[..], h_evals_large[..test_size]);
        assert_eq!(h_evals_coset[..], h_evals_large[test_size..2 * test_size]);
    }

    #[test]
    fn test_scalar_batch_evaluation_on_coset() {
        // checks that evaluating a polynomial on a subgroup and its coset is the same as evaluating on a 2x larger subgroup
        let batch_size = 4;
        let log_test_size = 6;
        let test_size = 1 << log_test_size;
        let (_, mut d_coeffs, mut d_domain) = set_up_scalars_f251(test_size * batch_size, log_test_size, false);
        let (_, _, mut d_large_domain) = set_up_scalars_f251(0, log_test_size + 1, false);
        let mut d_coset_powers = build_domain_f251(test_size, log_test_size + 1, false);

        let mut d_evals_large = evaluate_scalars_batch_f251(&mut d_coeffs, &mut d_large_domain, batch_size);
        let mut h_evals_large: Vec<ScalarField_F251> = (0..2 * test_size * batch_size)
            .map(|_| ScalarField_F251::zero())
            .collect();
        d_evals_large
            .copy_to(&mut h_evals_large[..])
            .unwrap();
        let mut d_evals = evaluate_scalars_batch_f251(&mut d_coeffs, &mut d_domain, batch_size);
        let mut h_evals: Vec<ScalarField_F251> = (0..test_size * batch_size)
            .map(|_| ScalarField_F251::zero())
            .collect();
        d_evals
            .copy_to(&mut h_evals[..])
            .unwrap();
        let mut d_evals_coset =
            evaluate_scalars_on_coset_batch_f251(&mut d_coeffs, &mut d_domain, batch_size, &mut d_coset_powers);
        let mut h_evals_coset: Vec<ScalarField_F251> = (0..test_size * batch_size)
            .map(|_| ScalarField_F251::zero())
            .collect();
        d_evals_coset
            .copy_to(&mut h_evals_coset[..])
            .unwrap();

        for i in 0..batch_size {
            assert_eq!(
                h_evals_large[2 * i * test_size..(2 * i + 1) * test_size],
                h_evals[i * test_size..(i + 1) * test_size]
            );
            assert_eq!(
                h_evals_large[(2 * i + 1) * test_size..(2 * i + 2) * test_size],
                h_evals_coset[i * test_size..(i + 1) * test_size]
            );
        }
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_vec_scalar_mul() {
        let mut intoo = [
            ScalarField_F251::one(),
            ScalarField_F251::one(),
            ScalarField_F251::zero(),
        ];
        let expected = [
            ScalarField_F251::one(),
            ScalarField_F251::zero(),
            ScalarField_F251::zero(),
        ];
        mult_sc_vec_f251(&mut intoo, &expected, 0);
        assert_eq!(intoo, expected);
    }
}
