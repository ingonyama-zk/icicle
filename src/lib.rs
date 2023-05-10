use std::ffi::{c_int, c_uint};

use rand::{rngs::StdRng, RngCore, SeedableRng};

use field::*;
use ark_bls12_381::{Fr, G1Affine, G1Projective};
use ark_ff::PrimeField;
use ark_std::UniformRand;

use rustacuda::prelude::*;
use rustacuda_core::DevicePointer;
use rustacuda::memory::{DeviceBox, CopyDestination, DeviceCopy};

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

    fn commit_cuda(
        d_out: DevicePointer<Point>,
        d_scalars: DevicePointer<ScalarField>,
        d_points: DevicePointer<PointAffineNoInfinity>,
        count: usize,
        device_id: usize,
    ) -> c_uint;

    fn commit_batch_cuda(
        d_out: DevicePointer<Point>,
        d_scalars: DevicePointer<ScalarField>,
        d_points: DevicePointer<PointAffineNoInfinity>,
        count: usize,
        batch_size: usize,
        device_id: usize,
    ) -> c_uint;

    fn build_domain_cuda(domain_size: usize, logn: usize, inverse: bool, device_id: usize) -> DevicePointer<ScalarField>;

    fn ntt_cuda(inout: *mut ScalarField, n: usize, inverse: bool, device_id: usize) -> c_int;

    fn ecntt_cuda(inout: *mut Point, n: usize, inverse: bool, device_id: usize) -> c_int;

    fn ntt_end2end_batch(
        inout: *mut ScalarField,
        arr_size: usize,
        n: usize,
        inverse: bool,
    ) -> c_int;

    fn ecntt_end2end_batch(inout: *mut Point, arr_size: usize, n: usize, inverse: bool) -> c_int;

    fn interpolate_scalars_cuda(
        d_out: DevicePointer<ScalarField>,
        d_evaluations: DevicePointer<ScalarField>,
        d_domain: DevicePointer<ScalarField>, 
        n: usize,
        device_id: usize
    ) -> c_int;

    fn interpolate_scalars_batch_cuda(
        d_out: DevicePointer<ScalarField>,
        d_evaluations: DevicePointer<ScalarField>,
        d_domain: DevicePointer<ScalarField>,
        n: usize,
        batch_size: usize,
        device_id: usize
    ) -> c_int;

    fn interpolate_points_cuda(
        d_out: DevicePointer<Point>,
        d_evaluations: DevicePointer<Point>,
        d_domain: DevicePointer<ScalarField>,
        n: usize,
        device_id: usize
    ) -> c_int;

    fn interpolate_points_batch_cuda(
        d_out: DevicePointer<Point>,
        d_evaluations: DevicePointer<Point>,
        d_domain: DevicePointer<ScalarField>,
        n: usize,
        batch_size: usize,
        device_id: usize
    ) -> c_int;

    fn evaluate_scalars_cuda(
        d_out: DevicePointer<ScalarField>,
        d_coefficients: DevicePointer<ScalarField>,
        d_domain: DevicePointer<ScalarField>,
        domain_size: usize,
        n: usize,
        device_id: usize
    ) -> c_int;

    fn evaluate_scalars_batch_cuda(
        d_out: DevicePointer<ScalarField>,
        d_coefficients: DevicePointer<ScalarField>,
        d_domain: DevicePointer<ScalarField>,
        domain_size: usize,
        n: usize,
        batch_size: usize,
        device_id: usize
    ) -> c_int;

    fn evaluate_points_cuda(
        d_out: DevicePointer<Point>,
        d_coefficients: DevicePointer<Point>,
        d_domain: DevicePointer<ScalarField>,
        domain_size: usize,
        n: usize,
        device_id: usize
    ) -> c_int;

    fn evaluate_points_batch_cuda(
        d_out: DevicePointer<Point>,
        d_coefficients: DevicePointer<Point>,
        d_domain: DevicePointer<ScalarField>,
        domain_size: usize,
        n: usize,
        batch_size: usize,
        device_id: usize
    ) -> c_int;

    fn evaluate_scalars_on_coset_cuda(
        d_out: DevicePointer<ScalarField>,
        d_coefficients: DevicePointer<ScalarField>,
        d_domain: DevicePointer<ScalarField>,
        domain_size: usize,
        n: usize,
        coset_powers: DevicePointer<ScalarField>,
        device_id: usize
    ) -> c_int;

    fn evaluate_scalars_on_coset_batch_cuda(
        d_out: DevicePointer<ScalarField>,
        d_coefficients: DevicePointer<ScalarField>,
        d_domain: DevicePointer<ScalarField>,
        domain_size: usize,
        n: usize,
        batch_size: usize,
        coset_powers: DevicePointer<ScalarField>,
        device_id: usize
    ) -> c_int;

    fn evaluate_points_on_coset_cuda(
        d_out: DevicePointer<Point>,
        d_coefficients: DevicePointer<Point>,
        d_domain: DevicePointer<ScalarField>,
        domain_size: usize,
        n: usize,
        coset_powers: DevicePointer<ScalarField>,
        device_id: usize
    ) -> c_int;

    fn evaluate_points_on_coset_batch_cuda(
        d_out: DevicePointer<Point>,
        d_coefficients: DevicePointer<Point>,
        d_domain: DevicePointer<ScalarField>,
        domain_size: usize,
        n: usize,
        batch_size: usize,
        coset_powers: DevicePointer<ScalarField>,
        device_id: usize
    ) -> c_int;

    fn reverse_order_scalars_cuda(
        d_arr: DevicePointer<ScalarField>,
        n: usize,
        device_id: usize
    ) -> c_int;

    fn reverse_order_scalars_batch_cuda(
        d_arr: DevicePointer<ScalarField>,
        n: usize,
        batch_size: usize,
        device_id: usize
    ) -> c_int;

    fn reverse_order_points_cuda(
        d_arr: DevicePointer<Point>,
        n: usize,
        device_id: usize
    ) -> c_int;

    fn reverse_order_points_batch_cuda(
        d_arr: DevicePointer<Point>,
        n: usize,
        batch_size: usize,
        device_id: usize
    ) -> c_int;

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

    fn matrix_vec_mod_mult(
        matrix_flattened: *const ScalarField,
        input: *const ScalarField,
        output: *mut ScalarField,
        n_elements: usize,
        device_id: usize,
    ) -> c_int;
}

pub fn msm(points: &[PointAffineNoInfinity], scalars: &[ScalarField], device_id: usize) -> Point {
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
    scalars: &[ScalarField],
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

pub fn commit(
    points: &mut DeviceBuffer<PointAffineNoInfinity>,
    scalars: &mut DeviceBuffer<ScalarField>,
) -> DeviceBox<Point> {
    let mut res = DeviceBox::new(&Point::zero()).unwrap();
    unsafe {
        commit_cuda(
            res.as_device_ptr(),
            scalars.as_device_ptr(),
            points.as_device_ptr(),
            scalars.len(),
            0,
        );
    }
    return res;
}

pub fn commit_batch(
    points: &mut DeviceBuffer<PointAffineNoInfinity>,
    scalars: &mut DeviceBuffer<ScalarField>,
    batch_size: usize,
) -> DeviceBuffer<Point> {
    let mut res = unsafe { DeviceBuffer::uninitialized(batch_size).unwrap() };
    unsafe {
        commit_batch_cuda(
            res.as_device_ptr(),
            scalars.as_device_ptr(),
            points.as_device_ptr(),
            scalars.len() / batch_size,
            batch_size,
            0,
        );
    }
    return res;
}

/// Compute an in-place NTT on the input data.
fn ntt_internal(values: &mut [ScalarField], device_id: usize, inverse: bool) -> i32 {
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

pub fn ntt(values: &mut [ScalarField], device_id: usize) {
    ntt_internal(values, device_id, false);
}

pub fn intt(values: &mut [ScalarField], device_id: usize) {
    ntt_internal(values, device_id, true);
}

/// Compute an in-place NTT on the input data.
fn ntt_internal_batch(
    values: &mut [ScalarField],
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

pub fn ntt_batch(values: &mut [ScalarField], batch_size: usize, device_id: usize) {
    ntt_internal_batch(values, 0, batch_size, false);
}

pub fn intt_batch(values: &mut [ScalarField], batch_size: usize, device_id: usize) {
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

pub fn build_domain(domain_size: usize, logn: usize, inverse: bool) -> DeviceBuffer<ScalarField> {
    unsafe {
        DeviceBuffer::from_raw_parts(build_domain_cuda(
            domain_size,
            logn,
            inverse,
            0
        ), domain_size)
    }
}


pub fn reverse_order_scalars(
    d_scalars: &mut DeviceBuffer<ScalarField>,
) {
    unsafe { reverse_order_scalars_cuda(
        d_scalars.as_device_ptr(),
        d_scalars.len(),
        0
    ); }
}

pub fn reverse_order_scalars_batch(
    d_scalars: &mut DeviceBuffer<ScalarField>,
    batch_size: usize,
) {
    unsafe { reverse_order_scalars_batch_cuda(
        d_scalars.as_device_ptr(),
        d_scalars.len() / batch_size,
        batch_size,
        0
    ); }
}

pub fn reverse_order_points(
    d_points: &mut DeviceBuffer<Point>,
) {
    unsafe { reverse_order_points_cuda(
        d_points.as_device_ptr(),
        d_points.len(),
        0
    ); }
}

pub fn reverse_order_points_batch(
    d_points: &mut DeviceBuffer<Point>,
    batch_size: usize,
) {
    unsafe { reverse_order_points_batch_cuda(
        d_points.as_device_ptr(),
        d_points.len() / batch_size,
        batch_size,
        0
    ); }
}

pub fn interpolate_scalars(
    d_evaluations: &mut DeviceBuffer<ScalarField>,
    d_domain: &mut DeviceBuffer<ScalarField>
) -> DeviceBuffer<ScalarField> {
    let mut res = unsafe { DeviceBuffer::uninitialized(d_domain.len()).unwrap() };
    unsafe { interpolate_scalars_cuda(
        res.as_device_ptr(),
        d_evaluations.as_device_ptr(),
        d_domain.as_device_ptr(),
        d_domain.len(),
        0
    ) };
    return res;
}

pub fn interpolate_scalars_batch(
    d_evaluations: &mut DeviceBuffer<ScalarField>,
    d_domain: &mut DeviceBuffer<ScalarField>,
    batch_size: usize,
) -> DeviceBuffer<ScalarField> {
    let mut res = unsafe { DeviceBuffer::uninitialized(d_domain.len() * batch_size).unwrap() };
    unsafe { interpolate_scalars_batch_cuda(
        res.as_device_ptr(),
        d_evaluations.as_device_ptr(),
        d_domain.as_device_ptr(),
        d_domain.len(),
        batch_size,
        0
    ) };
    return res;
}

pub fn interpolate_points(
    d_evaluations: &mut DeviceBuffer<Point>,
    d_domain: &mut DeviceBuffer<ScalarField>,
) -> DeviceBuffer<Point> {
    let mut res = unsafe { DeviceBuffer::uninitialized(d_domain.len()).unwrap() };
    unsafe { interpolate_points_cuda(
        res.as_device_ptr(),
        d_evaluations.as_device_ptr(),
        d_domain.as_device_ptr(),
        d_domain.len(),
        0
    ) };
    return res;
}

pub fn interpolate_points_batch(
    d_evaluations: &mut DeviceBuffer<Point>,
    d_domain: &mut DeviceBuffer<ScalarField>,
    batch_size: usize,
) -> DeviceBuffer<Point> {
    let mut res = unsafe { DeviceBuffer::uninitialized(d_domain.len() * batch_size).unwrap() };
    unsafe { interpolate_points_batch_cuda(
        res.as_device_ptr(),
        d_evaluations.as_device_ptr(),
        d_domain.as_device_ptr(),
        d_domain.len(),
        batch_size,
        0
    ) };
    return res;
}

pub fn evaluate_scalars(
    d_coefficients: &mut DeviceBuffer<ScalarField>,
    d_domain: &mut DeviceBuffer<ScalarField>,
) -> DeviceBuffer<ScalarField> {
    let mut res = unsafe { DeviceBuffer::uninitialized(d_domain.len()).unwrap() };
    unsafe {
        evaluate_scalars_cuda(
            res.as_device_ptr(),
            d_coefficients.as_device_ptr(),
            d_domain.as_device_ptr(),
            d_domain.len(),
            d_coefficients.len(),
            0
        );
    }
    return res;
}

pub fn evaluate_scalars_batch(
    d_coefficients: &mut DeviceBuffer<ScalarField>,
    d_domain: &mut DeviceBuffer<ScalarField>,
    batch_size: usize,
) -> DeviceBuffer<ScalarField> {
    let mut res = unsafe { DeviceBuffer::uninitialized(d_domain.len() * batch_size).unwrap() };
    unsafe {
        evaluate_scalars_batch_cuda(
            res.as_device_ptr(),
            d_coefficients.as_device_ptr(),
            d_domain.as_device_ptr(),
            d_domain.len(),
            d_coefficients.len() / batch_size,
            batch_size,
            0
        );
    }
    return res;
}

pub fn evaluate_points(
    d_coefficients: &mut DeviceBuffer<Point>,
    d_domain: &mut DeviceBuffer<ScalarField>,
) -> DeviceBuffer<Point> {
    let mut res = unsafe { DeviceBuffer::uninitialized(d_domain.len()).unwrap() };
    unsafe {
        evaluate_points_cuda(
            res.as_device_ptr(),
            d_coefficients.as_device_ptr(),
            d_domain.as_device_ptr(),
            d_domain.len(),
            d_coefficients.len(),
            0
        );
    }
    return res;
}

pub fn evaluate_points_batch(
    d_coefficients: &mut DeviceBuffer<Point>,
    d_domain: &mut DeviceBuffer<ScalarField>,
    batch_size: usize,
) -> DeviceBuffer<Point> {
    let mut res = unsafe { DeviceBuffer::uninitialized(d_domain.len() * batch_size).unwrap() };
    unsafe {
        evaluate_points_batch_cuda(
            res.as_device_ptr(),
            d_coefficients.as_device_ptr(),
            d_domain.as_device_ptr(),
            d_domain.len(),
            d_coefficients.len() / batch_size,
            batch_size,
            0
        );
    }
    return res;
}

pub fn evaluate_scalars_on_coset(
    d_coefficients: &mut DeviceBuffer<ScalarField>,
    d_domain: &mut DeviceBuffer<ScalarField>,
    coset_powers: &mut DeviceBuffer<ScalarField>,
) -> DeviceBuffer<ScalarField> {
    let mut res = unsafe { DeviceBuffer::uninitialized(d_domain.len()).unwrap() };
    unsafe {
        evaluate_scalars_on_coset_cuda(
            res.as_device_ptr(),
            d_coefficients.as_device_ptr(),
            d_domain.as_device_ptr(),
            d_domain.len(),
            d_coefficients.len(),
            coset_powers.as_device_ptr(),
            0
        );
    }
    return res;
}

pub fn evaluate_scalars_on_coset_batch(
    d_coefficients: &mut DeviceBuffer<ScalarField>,
    d_domain: &mut DeviceBuffer<ScalarField>,
    batch_size: usize,
    coset_powers: &mut DeviceBuffer<ScalarField>,
) -> DeviceBuffer<ScalarField> {
    let mut res = unsafe { DeviceBuffer::uninitialized(d_domain.len() * batch_size).unwrap() };
    unsafe {
        evaluate_scalars_on_coset_batch_cuda(
            res.as_device_ptr(),
            d_coefficients.as_device_ptr(),
            d_domain.as_device_ptr(),
            d_domain.len(),
            d_coefficients.len() / batch_size,
            batch_size,
            coset_powers.as_device_ptr(),
            0
        );
    }
    return res;
}

pub fn evaluate_points_on_coset(
    d_coefficients: &mut DeviceBuffer<Point>,
    d_domain: &mut DeviceBuffer<ScalarField>,
    coset_powers: &mut DeviceBuffer<ScalarField>,
) -> DeviceBuffer<Point> {
    let mut res = unsafe { DeviceBuffer::uninitialized(d_domain.len()).unwrap() };
    unsafe {
        evaluate_points_on_coset_cuda(
            res.as_device_ptr(),
            d_coefficients.as_device_ptr(),
            d_domain.as_device_ptr(),
            d_domain.len(),
            d_coefficients.len(),
            coset_powers.as_device_ptr(),
            0
        );
    }
    return res;
}

pub fn evaluate_points_on_coset_batch(
    d_coefficients: &mut DeviceBuffer<Point>,
    d_domain: &mut DeviceBuffer<ScalarField>,
    batch_size: usize,
    coset_powers: &mut DeviceBuffer<ScalarField>,
) -> DeviceBuffer<Point> {
    let mut res = unsafe { DeviceBuffer::uninitialized(d_domain.len() * batch_size).unwrap() };
    unsafe {
        evaluate_points_on_coset_batch_cuda(
            res.as_device_ptr(),
            d_coefficients.as_device_ptr(),
            d_domain.as_device_ptr(),
            d_domain.len(),
            d_coefficients.len() / batch_size,
            batch_size,
            coset_powers.as_device_ptr(),
            0
        );
    }
    return res;
}

pub fn multp_vec(a: &mut [Point], b: &[ScalarField], device_id: usize) {
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

pub fn mult_sc_vec(a: &mut [ScalarField], b: &[ScalarField], device_id: usize) {
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

// Multiply a matrix by a scalar:
//  `a` - flattenned matrix;
//  `b` - vector to multiply `a` by;
pub fn mult_matrix_by_vec(a: &[ScalarField], b: &[ScalarField], device_id: usize) -> Vec<ScalarField> {
    let mut c = Vec::with_capacity(b.len());
    for i in 0..b.len() {
        c.push(ScalarField::zero());
    }
    unsafe {
        matrix_vec_mod_mult(
            a as *const _ as *const ScalarField,
            b as *const _ as *const ScalarField,
            c.as_mut_slice() as *mut _ as *mut ScalarField,
            b.len(),
            device_id,
        );
    }
    c
}

pub fn clone_buffer<T: DeviceCopy>(buf: &mut DeviceBuffer<T>) -> DeviceBuffer<T> {
    let mut buf_cpy = unsafe { DeviceBuffer::uninitialized(buf.len()).unwrap() };
    unsafe { buf_cpy.copy_from(buf) };
    return buf_cpy;
}

pub fn get_rng(seed: Option<u64>) -> Box<dyn RngCore> {
    let rng: Box<dyn RngCore> = match seed {
        Some(seed) => Box::new(StdRng::seed_from_u64(seed)),
        None => Box::new(rand::thread_rng()),
    };
    rng
}

fn set_up_device() {
    // Set up the context, load the module, and create a stream to run kernels in.
    rustacuda::init(CudaFlags::empty()).unwrap();
    let device = Device::get_device(0).unwrap();
    let _ctx = Context::create_and_push(ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device).unwrap();
}

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

pub fn set_up_points(test_size: usize, log_domain_size: usize, inverse: bool) -> (Vec<Point>, DeviceBuffer<Point>, DeviceBuffer<ScalarField>) {
    set_up_device();

    let d_domain = build_domain(1 << log_domain_size, log_domain_size, inverse);

    let seed = Some(0); // fix the rng to get two equal scalar 
    let vector = generate_random_points_proj(test_size, get_rng(seed));
    let mut vector_mut = vector.clone();

    let mut d_vector = DeviceBuffer::from_slice(&vector[..]).unwrap();
    (vector_mut, d_vector, d_domain)
}

pub fn set_up_scalars(test_size: usize, log_domain_size: usize, inverse: bool) -> (Vec<ScalarField>, DeviceBuffer<ScalarField>, DeviceBuffer<ScalarField>) {
    set_up_device();

    let d_domain = build_domain(1 << log_domain_size, log_domain_size, inverse);

    let seed = Some(0); // fix the rng to get two equal scalars
    let mut vector_mut = generate_random_scalars(test_size, get_rng(seed));

    let mut d_vector = DeviceBuffer::from_slice(&vector_mut[..]).unwrap();
    (vector_mut, d_vector, d_domain)
}


#[cfg(test)]
pub(crate) mod tests {
    use std::ops::Add;

    use ark_ec::{msm::VariableBaseMSM, AffineCurve, ProjectiveCurve};
    use ark_ff::{FftField, Field, Zero, PrimeField};

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
    fn test_msm_debug() {
        let test_sizes = [15];

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

        let scalars_batch: Vec<ScalarField> =
            generate_random_scalars(test_size * batches, get_rng(seed));

        let mut scalar_vec_of_vec: Vec<Vec<ScalarField>> = Vec::new();

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
        let mut h_coeffs: Vec<ScalarField> = (0..test_size).map(|_| ScalarField::zero()).collect();
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
        let mut h_coeffs: Vec<ScalarField> = (0..test_size * batch_size).map(|_| ScalarField::zero()).collect();
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
        let mut h_coeffs_domain: Vec<ScalarField> = (0..1 << log_test_domain_size).map(|_| ScalarField::zero()).collect();
        d_coeffs_domain.copy_to(&mut h_coeffs_domain[..]).unwrap();

        assert_eq!(h_coeffs, h_coeffs_domain[..coeff_size]);
        for i in coeff_size.. (1 << log_test_domain_size) {
            assert_eq!(ScalarField::zero(), h_coeffs_domain[i]);
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
        let mut h_coeffs_domain: Vec<ScalarField> = (0..domain_size * batch_size).map(|_| ScalarField::zero()).collect();
        d_coeffs_domain.copy_to(&mut h_coeffs_domain[..]).unwrap();

        for j in 0..batch_size {
            assert_eq!(h_coeffs[j * coeff_size..(j + 1) * coeff_size], h_coeffs_domain[j * domain_size..j * domain_size + coeff_size]);
            for i in coeff_size..domain_size {
                assert_eq!(ScalarField::zero(), h_coeffs_domain[j * domain_size + i]);
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
        let mut h_coeffs: Vec<ScalarField> = (0..1 << log_test_domain_size).map(|_| ScalarField::zero()).collect();
        d_evals.copy_to(&mut h_coeffs[..]).unwrap();
        let mut d_evals_coset = evaluate_scalars_on_coset(&mut d_coeffs, &mut d_domain, &mut d_trivial_coset_powers);
        let mut h_evals_coset: Vec<ScalarField> = (0..1 << log_test_domain_size).map(|_| ScalarField::zero()).collect();
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
        let mut h_evals_large: Vec<ScalarField> = (0..2 * test_size).map(|_| ScalarField::zero()).collect();
        d_evals_large.copy_to(&mut h_evals_large[..]).unwrap();
        let mut d_evals = evaluate_scalars(&mut d_coeffs, &mut d_domain);
        let mut h_evals: Vec<ScalarField> = (0..test_size).map(|_| ScalarField::zero()).collect();
        d_evals.copy_to(&mut h_evals[..]).unwrap();
        let mut d_evals_coset = evaluate_scalars_on_coset(&mut d_coeffs, &mut d_domain, &mut d_coset_powers);
        let mut h_evals_coset: Vec<ScalarField> = (0..test_size).map(|_| ScalarField::zero()).collect();
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
        let mut h_evals_large: Vec<ScalarField> = (0..2 * test_size * batch_size).map(|_| ScalarField::zero()).collect();
        d_evals_large.copy_to(&mut h_evals_large[..]).unwrap();
        let mut d_evals = evaluate_scalars_batch(&mut d_coeffs, &mut d_domain, batch_size);
        let mut h_evals: Vec<ScalarField> = (0..test_size * batch_size).map(|_| ScalarField::zero()).collect();
        d_evals.copy_to(&mut h_evals[..]).unwrap();
        let mut d_evals_coset = evaluate_scalars_on_coset_batch(&mut d_coeffs, &mut d_domain, batch_size, &mut d_coset_powers);
        let mut h_evals_coset: Vec<ScalarField> = (0..test_size * batch_size).map(|_| ScalarField::zero()).collect();
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
        let matrix_flattened: Vec<ScalarField> = (0..test_size).map(
            |row_num| { (0..test_size).map( 
                |col_num| {
                    let pow: [u64; 1] = [(row_num * col_num).try_into().unwrap()];
                    ScalarField::from_ark(Fr::pow(&rou, &pow).into_repr())
                }).collect::<Vec<ScalarField>>()
            }).flatten().collect::<Vec<_>>();
        let vector: Vec<ScalarField> = generate_random_scalars(test_size, get_rng(seed));

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
        let mut intoo = [ScalarField::one(), ScalarField::one(), ScalarField::zero()];
        let expected = [ScalarField::one(), ScalarField::zero(), ScalarField::zero()];
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
        let scalars = [ScalarField::one(), ScalarField::zero(), ScalarField::zero()];
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
