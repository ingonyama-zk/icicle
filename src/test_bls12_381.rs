use std::ffi::{c_int, c_uint};

use rand::{rngs::StdRng, RngCore, SeedableRng};


use crate::curves::bls12_381::*;

use ark_bls12_381::{Fr as Fr_BLS12_381, G1Projective as G1Projective_BLS12_381};
use ark_ff::PrimeField;
use ark_std::UniformRand;

use rustacuda::prelude::*;
use rustacuda_core::DevicePointer;
use rustacuda::memory::{DeviceBox, CopyDestination, DeviceCopy};

extern "C" {
    fn poseidon_multi_cuda_bls12_381(
        input: *const ScalarField_BLS12_381,
        out: *mut ScalarField_BLS12_381,
        number_of_blocks: usize,
        arity: c_uint,
        device_id: usize,
    ) -> c_uint;

    fn msm_cuda_bls12_381(
        out: *mut Point_BLS12_381,
        points: *const PointAffineNoInfinity_BLS12_381,
        scalars: *const ScalarField_BLS12_381,
        count: usize,
        device_id: usize,
    ) -> c_uint;

    fn msm_batch_cuda_bls12_381(
        out: *mut Point_BLS12_381,
        points: *const PointAffineNoInfinity_BLS12_381,
        scalars: *const ScalarField_BLS12_381,
        batch_size: usize,
        msm_size: usize,
        device_id: usize,
    ) -> c_uint;

    fn commit_cuda_bls12_381(
        d_out: DevicePointer<Point_BLS12_381>,
        d_scalars: DevicePointer<ScalarField_BLS12_381>,
        d_points: DevicePointer<PointAffineNoInfinity_BLS12_381>,
        count: usize,
        device_id: usize,
    ) -> c_uint;

    fn commit_batch_cuda_bls12_381(
        d_out: DevicePointer<Point_BLS12_381>,
        d_scalars: DevicePointer<ScalarField_BLS12_381>,
        d_points: DevicePointer<PointAffineNoInfinity_BLS12_381>,
        count: usize,
        batch_size: usize,
        device_id: usize,
    ) -> c_uint;

    fn build_domain_cuda_bls12_381(domain_size: usize, logn: usize, inverse: bool, device_id: usize) -> DevicePointer<ScalarField_BLS12_381>;

    fn ntt_cuda_bls12_381(inout: *mut ScalarField_BLS12_381, n: usize, inverse: bool, device_id: usize) -> c_int;

    fn ecntt_cuda_bls12_381(inout: *mut Point_BLS12_381, n: usize, inverse: bool, device_id: usize) -> c_int;

    fn ntt_batch_cuda_bls12_381(
        inout: *mut ScalarField_BLS12_381,
        arr_size: usize,
        n: usize,
        inverse: bool,
    ) -> c_int;

    fn ecntt_batch_cuda_bls12_381(inout: *mut Point_BLS12_381, arr_size: usize, n: usize, inverse: bool) -> c_int;

    fn interpolate_scalars_cuda_bls12_381(
        d_out: DevicePointer<ScalarField_BLS12_381>,
        d_evaluations: DevicePointer<ScalarField_BLS12_381>,
        d_domain: DevicePointer<ScalarField_BLS12_381>, 
        n: usize,
        device_id: usize
    ) -> c_int;

    fn interpolate_scalars_batch_cuda_bls12_381(
        d_out: DevicePointer<ScalarField_BLS12_381>,
        d_evaluations: DevicePointer<ScalarField_BLS12_381>,
        d_domain: DevicePointer<ScalarField_BLS12_381>,
        n: usize,
        batch_size: usize,
        device_id: usize
    ) -> c_int;

    fn interpolate_points_cuda_bls12_381(
        d_out: DevicePointer<Point_BLS12_381>,
        d_evaluations: DevicePointer<Point_BLS12_381>,
        d_domain: DevicePointer<ScalarField_BLS12_381>,
        n: usize,
        device_id: usize
    ) -> c_int;

    fn interpolate_points_batch_cuda_bls12_381(
        d_out: DevicePointer<Point_BLS12_381>,
        d_evaluations: DevicePointer<Point_BLS12_381>,
        d_domain: DevicePointer<ScalarField_BLS12_381>,
        n: usize,
        batch_size: usize,
        device_id: usize
    ) -> c_int;

    fn evaluate_scalars_cuda_bls12_381(
        d_out: DevicePointer<ScalarField_BLS12_381>,
        d_coefficients: DevicePointer<ScalarField_BLS12_381>,
        d_domain: DevicePointer<ScalarField_BLS12_381>,
        domain_size: usize,
        n: usize,
        device_id: usize
    ) -> c_int;

    fn evaluate_scalars_batch_cuda_bls12_381(
        d_out: DevicePointer<ScalarField_BLS12_381>,
        d_coefficients: DevicePointer<ScalarField_BLS12_381>,
        d_domain: DevicePointer<ScalarField_BLS12_381>,
        domain_size: usize,
        n: usize,
        batch_size: usize,
        device_id: usize
    ) -> c_int;

    fn evaluate_points_cuda_bls12_381(
        d_out: DevicePointer<Point_BLS12_381>,
        d_coefficients: DevicePointer<Point_BLS12_381>,
        d_domain: DevicePointer<ScalarField_BLS12_381>,
        domain_size: usize,
        n: usize,
        device_id: usize
    ) -> c_int;

    fn evaluate_points_batch_cuda_bls12_381(
        d_out: DevicePointer<Point_BLS12_381>,
        d_coefficients: DevicePointer<Point_BLS12_381>,
        d_domain: DevicePointer<ScalarField_BLS12_381>,
        domain_size: usize,
        n: usize,
        batch_size: usize,
        device_id: usize
    ) -> c_int;

    fn evaluate_scalars_on_coset_cuda_bls12_381(
        d_out: DevicePointer<ScalarField_BLS12_381>,
        d_coefficients: DevicePointer<ScalarField_BLS12_381>,
        d_domain: DevicePointer<ScalarField_BLS12_381>,
        domain_size: usize,
        n: usize,
        coset_powers: DevicePointer<ScalarField_BLS12_381>,
        device_id: usize
    ) -> c_int;

    fn evaluate_scalars_on_coset_batch_cuda_bls12_381(
        d_out: DevicePointer<ScalarField_BLS12_381>,
        d_coefficients: DevicePointer<ScalarField_BLS12_381>,
        d_domain: DevicePointer<ScalarField_BLS12_381>,
        domain_size: usize,
        n: usize,
        batch_size: usize,
        coset_powers: DevicePointer<ScalarField_BLS12_381>,
        device_id: usize
    ) -> c_int;

    fn evaluate_points_on_coset_cuda_bls12_381(
        d_out: DevicePointer<Point_BLS12_381>,
        d_coefficients: DevicePointer<Point_BLS12_381>,
        d_domain: DevicePointer<ScalarField_BLS12_381>,
        domain_size: usize,
        n: usize,
        coset_powers: DevicePointer<ScalarField_BLS12_381>,
        device_id: usize
    ) -> c_int;

    fn evaluate_points_on_coset_batch_cuda_bls12_381(
        d_out: DevicePointer<Point_BLS12_381>,
        d_coefficients: DevicePointer<Point_BLS12_381>,
        d_domain: DevicePointer<ScalarField_BLS12_381>,
        domain_size: usize,
        n: usize,
        batch_size: usize,
        coset_powers: DevicePointer<ScalarField_BLS12_381>,
        device_id: usize
    ) -> c_int;

    fn reverse_order_scalars_cuda_bls12_381(
        d_arr: DevicePointer<ScalarField_BLS12_381>,
        n: usize,
        device_id: usize
    ) -> c_int;

    fn reverse_order_scalars_batch_cuda_bls12_381(
        d_arr: DevicePointer<ScalarField_BLS12_381>,
        n: usize,
        batch_size: usize,
        device_id: usize
    ) -> c_int;

    fn reverse_order_points_cuda_bls12_381(
        d_arr: DevicePointer<Point_BLS12_381>,
        n: usize,
        device_id: usize
    ) -> c_int;

    fn reverse_order_points_batch_cuda_bls12_381(
        d_arr: DevicePointer<Point_BLS12_381>,
        n: usize,
        batch_size: usize,
        device_id: usize
    ) -> c_int;

    fn vec_mod_mult_point_bls12_381(
        inout: *mut Point_BLS12_381,
        scalars: *const ScalarField_BLS12_381,
        n_elements: usize,
        device_id: usize,
    ) -> c_int;

    fn vec_mod_mult_scalar_bls12_381(
        inout: *mut ScalarField_BLS12_381,
        scalars: *const ScalarField_BLS12_381,
        n_elements: usize,
        device_id: usize,
    ) -> c_int;

    fn matrix_vec_mod_mult_bls12_381(
        matrix_flattened: *const ScalarField_BLS12_381,
        input: *const ScalarField_BLS12_381,
        output: *mut ScalarField_BLS12_381,
        n_elements: usize,
        device_id: usize,
    ) -> c_int;
}

pub fn poseidon_multi_bls12_381(input: &[ScalarField_BLS12_381], arity: usize, device_id: usize) -> Vec<ScalarField_BLS12_381> {
    let number_of_blocks = input.len() / arity;
    let mut out = vec![ScalarField_BLS12_381::zero(); number_of_blocks];
    unsafe {
        poseidon_multi_cuda_bls12_381(
            input as *const _ as *const ScalarField_BLS12_381,
            out.as_mut_slice() as *mut _ as *mut ScalarField_BLS12_381,
            number_of_blocks,
            arity as c_uint,
            device_id);
    }

    out
}

pub fn msm_bls12_381(points: &[PointAffineNoInfinity_BLS12_381], scalars: &[ScalarField_BLS12_381], device_id: usize) -> Point_BLS12_381 {
    let count = points.len();
    if count != scalars.len() {
        todo!("variable length")
    }

    let mut ret = Point_BLS12_381::zero();
    unsafe {
        msm_cuda_bls12_381(
            &mut ret as *mut _ as *mut Point_BLS12_381,
            points as *const _ as *const PointAffineNoInfinity_BLS12_381,
            scalars as *const _ as *const ScalarField_BLS12_381,
            scalars.len(),
            device_id,
        )
    };

    ret
}

pub fn msm_batch_bls12_381(
    points: &[PointAffineNoInfinity_BLS12_381],
    scalars: &[ScalarField_BLS12_381],
    batch_size: usize,
    device_id: usize,
) -> Vec<Point_BLS12_381> {
    let count = points.len();
    if count != scalars.len() {
        todo!("variable length")
    }

    let mut ret = vec![Point_BLS12_381::zero(); batch_size];

    unsafe {
        msm_batch_cuda_bls12_381(
            &mut ret[0] as *mut _ as *mut Point_BLS12_381,
            points as *const _ as *const PointAffineNoInfinity_BLS12_381,
            scalars as *const _ as *const ScalarField_BLS12_381,
            batch_size,
            count / batch_size,
            device_id,
        )
    };

    ret
}

pub fn commit_bls12_381(
    points: &mut DeviceBuffer<PointAffineNoInfinity_BLS12_381>,
    scalars: &mut DeviceBuffer<ScalarField_BLS12_381>,
) -> DeviceBox<Point_BLS12_381> {
    let mut res = DeviceBox::new(&Point_BLS12_381::zero()).unwrap();
    unsafe {
        commit_cuda_bls12_381(
            res.as_device_ptr(),
            scalars.as_device_ptr(),
            points.as_device_ptr(),
            scalars.len(),
            0,
        );
    }
    return res;
}

pub fn commit_batch_bls12_381(
    points: &mut DeviceBuffer<PointAffineNoInfinity_BLS12_381>,
    scalars: &mut DeviceBuffer<ScalarField_BLS12_381>,
    batch_size: usize,
) -> DeviceBuffer<Point_BLS12_381> {
    let mut res = unsafe { DeviceBuffer::uninitialized(batch_size).unwrap() };
    unsafe {
        commit_batch_cuda_bls12_381(
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
fn ntt_internal_bls12_381(values: &mut [ScalarField_BLS12_381], device_id: usize, inverse: bool) -> i32 {
    let ret_code = unsafe {
        ntt_cuda_bls12_381(
            values as *mut _ as *mut ScalarField_BLS12_381,
            values.len(),
            inverse,
            device_id,
        )
    };
    ret_code
}

pub fn ntt_bls12_381(values: &mut [ScalarField_BLS12_381], device_id: usize) {
    ntt_internal_bls12_381(values, device_id, false);
}

pub fn intt_bls12_381(values: &mut [ScalarField_BLS12_381], device_id: usize) {
    ntt_internal_bls12_381(values, device_id, true);
}

/// Compute an in-place NTT on the input data.
fn ntt_internal_batch_bls12_381(
    values: &mut [ScalarField_BLS12_381],
    device_id: usize,
    batch_size: usize,
    inverse: bool,
) -> i32 {
    unsafe {
        ntt_batch_cuda_bls12_381(
            values as *mut _ as *mut ScalarField_BLS12_381,
            values.len(),
            batch_size,
            inverse,
        )
    }
}

pub fn ntt_batch_bls12_381(values: &mut [ScalarField_BLS12_381], batch_size: usize, device_id: usize) {
    ntt_internal_batch_bls12_381(values, 0, batch_size, false);
}

pub fn intt_batch_bls12_381(values: &mut [ScalarField_BLS12_381], batch_size: usize, device_id: usize) {
    ntt_internal_batch_bls12_381(values, 0, batch_size, true);
}

/// Compute an in-place ECNTT on the input data.
fn ecntt_internal_bls12_381(values: &mut [Point_BLS12_381], inverse: bool, device_id: usize) -> i32 {
    unsafe {
        ecntt_cuda_bls12_381(
            values as *mut _ as *mut Point_BLS12_381,
            values.len(),
            inverse,
            device_id,
        )
    }
}

pub fn ecntt_bls12_381(values: &mut [Point_BLS12_381], device_id: usize) {
    ecntt_internal_bls12_381(values, false, device_id);
}

/// Compute an in-place iECNTT on the input data.
pub fn iecntt_bls12_381(values: &mut [Point_BLS12_381], device_id: usize) {
    ecntt_internal_bls12_381(values, true, device_id);
}

/// Compute an in-place ECNTT on the input data.
fn ecntt_internal_batch_bls12_381(
    values: &mut [Point_BLS12_381],
    device_id: usize,
    batch_size: usize,
    inverse: bool,
) -> i32 {
    unsafe {
        ecntt_batch_cuda_bls12_381(
            values as *mut _ as *mut Point_BLS12_381,
            values.len(),
            batch_size,
            inverse,
        )
    }
}

pub fn ecntt_batch_bls12_381(values: &mut [Point_BLS12_381], batch_size: usize, device_id: usize) {
    ecntt_internal_batch_bls12_381(values, 0, batch_size, false);
}

/// Compute an in-place iECNTT on the input data.
pub fn iecntt_batch_bls12_381(values: &mut [Point_BLS12_381], batch_size: usize, device_id: usize) {
    ecntt_internal_batch_bls12_381(values, 0, batch_size, true);
}

pub fn build_domain_bls12_381(domain_size: usize, logn: usize, inverse: bool) -> DeviceBuffer<ScalarField_BLS12_381> {
    unsafe {
        DeviceBuffer::from_raw_parts(build_domain_cuda_bls12_381(
            domain_size,
            logn,
            inverse,
            0
        ), domain_size)
    }
}


pub fn reverse_order_scalars_bls12_381(
    d_scalars: &mut DeviceBuffer<ScalarField_BLS12_381>,
) {
    unsafe { reverse_order_scalars_cuda_bls12_381(
        d_scalars.as_device_ptr(),
        d_scalars.len(),
        0
    ); }
}

pub fn reverse_order_scalars_batch_bls12_381(
    d_scalars: &mut DeviceBuffer<ScalarField_BLS12_381>,
    batch_size: usize,
) {
    unsafe { reverse_order_scalars_batch_cuda_bls12_381(
        d_scalars.as_device_ptr(),
        d_scalars.len() / batch_size,
        batch_size,
        0
    ); }
}

pub fn reverse_order_points_bls12_381(
    d_points: &mut DeviceBuffer<Point_BLS12_381>,
) {
    unsafe { reverse_order_points_cuda_bls12_381(
        d_points.as_device_ptr(),
        d_points.len(),
        0
    ); }
}

pub fn reverse_order_points_batch_bls12_381(
    d_points: &mut DeviceBuffer<Point_BLS12_381>,
    batch_size: usize,
) {
    unsafe { reverse_order_points_batch_cuda_bls12_381(
        d_points.as_device_ptr(),
        d_points.len() / batch_size,
        batch_size,
        0
    ); }
}

pub fn interpolate_scalars_bls12_381(
    d_evaluations: &mut DeviceBuffer<ScalarField_BLS12_381>,
    d_domain: &mut DeviceBuffer<ScalarField_BLS12_381>
) -> DeviceBuffer<ScalarField_BLS12_381> {
    let mut res = unsafe { DeviceBuffer::uninitialized(d_domain.len()).unwrap() };
    unsafe { interpolate_scalars_cuda_bls12_381(
        res.as_device_ptr(),
        d_evaluations.as_device_ptr(),
        d_domain.as_device_ptr(),
        d_domain.len(),
        0
    ) };
    return res;
}

pub fn interpolate_scalars_batch_bls12_381(
    d_evaluations: &mut DeviceBuffer<ScalarField_BLS12_381>,
    d_domain: &mut DeviceBuffer<ScalarField_BLS12_381>,
    batch_size: usize,
) -> DeviceBuffer<ScalarField_BLS12_381> {
    let mut res = unsafe { DeviceBuffer::uninitialized(d_domain.len() * batch_size).unwrap() };
    unsafe { interpolate_scalars_batch_cuda_bls12_381(
        res.as_device_ptr(),
        d_evaluations.as_device_ptr(),
        d_domain.as_device_ptr(),
        d_domain.len(),
        batch_size,
        0
    ) };
    return res;
}

pub fn interpolate_points_bls12_381(
    d_evaluations: &mut DeviceBuffer<Point_BLS12_381>,
    d_domain: &mut DeviceBuffer<ScalarField_BLS12_381>,
) -> DeviceBuffer<Point_BLS12_381> {
    let mut res = unsafe { DeviceBuffer::uninitialized(d_domain.len()).unwrap() };
    unsafe { interpolate_points_cuda_bls12_381(
        res.as_device_ptr(),
        d_evaluations.as_device_ptr(),
        d_domain.as_device_ptr(),
        d_domain.len(),
        0
    ) };
    return res;
}

pub fn interpolate_points_batch_bls12_381(
    d_evaluations: &mut DeviceBuffer<Point_BLS12_381>,
    d_domain: &mut DeviceBuffer<ScalarField_BLS12_381>,
    batch_size: usize,
) -> DeviceBuffer<Point_BLS12_381> {
    let mut res = unsafe { DeviceBuffer::uninitialized(d_domain.len() * batch_size).unwrap() };
    unsafe { interpolate_points_batch_cuda_bls12_381(
        res.as_device_ptr(),
        d_evaluations.as_device_ptr(),
        d_domain.as_device_ptr(),
        d_domain.len(),
        batch_size,
        0
    ) };
    return res;
}

pub fn evaluate_scalars_bls12_381(
    d_coefficients: &mut DeviceBuffer<ScalarField_BLS12_381>,
    d_domain: &mut DeviceBuffer<ScalarField_BLS12_381>,
) -> DeviceBuffer<ScalarField_BLS12_381> {
    let mut res = unsafe { DeviceBuffer::uninitialized(d_domain.len()).unwrap() };
    unsafe {
        evaluate_scalars_cuda_bls12_381(
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

pub fn evaluate_scalars_batch_bls12_381(
    d_coefficients: &mut DeviceBuffer<ScalarField_BLS12_381>,
    d_domain: &mut DeviceBuffer<ScalarField_BLS12_381>,
    batch_size: usize,
) -> DeviceBuffer<ScalarField_BLS12_381> {
    let mut res = unsafe { DeviceBuffer::uninitialized(d_domain.len() * batch_size).unwrap() };
    unsafe {
        evaluate_scalars_batch_cuda_bls12_381(
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

pub fn evaluate_points_bls12_381(
    d_coefficients: &mut DeviceBuffer<Point_BLS12_381>,
    d_domain: &mut DeviceBuffer<ScalarField_BLS12_381>,
) -> DeviceBuffer<Point_BLS12_381> {
    let mut res = unsafe { DeviceBuffer::uninitialized(d_domain.len()).unwrap() };
    unsafe {
        evaluate_points_cuda_bls12_381(
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

pub fn evaluate_points_batch_bls12_381(
    d_coefficients: &mut DeviceBuffer<Point_BLS12_381>,
    d_domain: &mut DeviceBuffer<ScalarField_BLS12_381>,
    batch_size: usize,
) -> DeviceBuffer<Point_BLS12_381> {
    let mut res = unsafe { DeviceBuffer::uninitialized(d_domain.len() * batch_size).unwrap() };
    unsafe {
        evaluate_points_batch_cuda_bls12_381(
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

pub fn evaluate_scalars_on_coset_bls12_381(
    d_coefficients: &mut DeviceBuffer<ScalarField_BLS12_381>,
    d_domain: &mut DeviceBuffer<ScalarField_BLS12_381>,
    coset_powers: &mut DeviceBuffer<ScalarField_BLS12_381>,
) -> DeviceBuffer<ScalarField_BLS12_381> {
    let mut res = unsafe { DeviceBuffer::uninitialized(d_domain.len()).unwrap() };
    unsafe {
        evaluate_scalars_on_coset_cuda_bls12_381(
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

pub fn evaluate_scalars_on_coset_batch_bls12_381(
    d_coefficients: &mut DeviceBuffer<ScalarField_BLS12_381>,
    d_domain: &mut DeviceBuffer<ScalarField_BLS12_381>,
    batch_size: usize,
    coset_powers: &mut DeviceBuffer<ScalarField_BLS12_381>,
) -> DeviceBuffer<ScalarField_BLS12_381> {
    let mut res = unsafe { DeviceBuffer::uninitialized(d_domain.len() * batch_size).unwrap() };
    unsafe {
        evaluate_scalars_on_coset_batch_cuda_bls12_381(
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

pub fn evaluate_points_on_coset_bls12_381(
    d_coefficients: &mut DeviceBuffer<Point_BLS12_381>,
    d_domain: &mut DeviceBuffer<ScalarField_BLS12_381>,
    coset_powers: &mut DeviceBuffer<ScalarField_BLS12_381>,
) -> DeviceBuffer<Point_BLS12_381> {
    let mut res = unsafe { DeviceBuffer::uninitialized(d_domain.len()).unwrap() };
    unsafe {
        evaluate_points_on_coset_cuda_bls12_381(
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

pub fn evaluate_points_on_coset_batch_bls12_381(
    d_coefficients: &mut DeviceBuffer<Point_BLS12_381>,
    d_domain: &mut DeviceBuffer<ScalarField_BLS12_381>,
    batch_size: usize,
    coset_powers: &mut DeviceBuffer<ScalarField_BLS12_381>,
) -> DeviceBuffer<Point_BLS12_381> {
    let mut res = unsafe { DeviceBuffer::uninitialized(d_domain.len() * batch_size).unwrap() };
    unsafe {
        evaluate_points_on_coset_batch_cuda_bls12_381(
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

pub fn multp_vec_bls12_381(a: &mut [Point_BLS12_381], b: &[ScalarField_BLS12_381], device_id: usize) {
    assert_eq!(a.len(), b.len());
    unsafe {
        vec_mod_mult_point_bls12_381(
            a as *mut _ as *mut Point_BLS12_381,
            b as *const _ as *const ScalarField_BLS12_381,
            a.len(),
            device_id,
        );
    }
}

pub fn mult_sc_vec_bls12_381(a: &mut [ScalarField_BLS12_381], b: &[ScalarField_BLS12_381], device_id: usize) {
    assert_eq!(a.len(), b.len());
    unsafe {
        vec_mod_mult_scalar_bls12_381(
            a as *mut _ as *mut ScalarField_BLS12_381,
            b as *const _ as *const ScalarField_BLS12_381,
            a.len(),
            device_id,
        );
    }
}

// Multiply a matrix by a scalar:
//  `a` - flattenned matrix;
//  `b` - vector to multiply `a` by;
pub fn mult_matrix_by_vec_bls12_381(a: &[ScalarField_BLS12_381], b: &[ScalarField_BLS12_381], device_id: usize) -> Vec<ScalarField_BLS12_381> {
    let mut c = Vec::with_capacity(b.len());
    for i in 0..b.len() {
        c.push(ScalarField_BLS12_381::zero());
    }
    unsafe {
        matrix_vec_mod_mult_bls12_381(
            a as *const _ as *const ScalarField_BLS12_381,
            b as *const _ as *const ScalarField_BLS12_381,
            c.as_mut_slice() as *mut _ as *mut ScalarField_BLS12_381,
            b.len(),
            device_id,
        );
    }
    c
}

pub fn clone_buffer_bls12_381<T: DeviceCopy>(buf: &mut DeviceBuffer<T>) -> DeviceBuffer<T> {
    let mut buf_cpy = unsafe { DeviceBuffer::uninitialized(buf.len()).unwrap() };
    unsafe { buf_cpy.copy_from(buf) };
    return buf_cpy;
}

pub fn get_rng_bls12_381(seed: Option<u64>) -> Box<dyn RngCore> {
    let rng: Box<dyn RngCore> = match seed {
        Some(seed) => Box::new(StdRng::seed_from_u64(seed)),
        None => Box::new(rand::thread_rng()),
    };
    rng
}

fn set_up_device_bls12_381() {
    // Set up the context, load the module, and create a stream to run kernels in.
    rustacuda::init(CudaFlags::empty()).unwrap();
    let device = Device::get_device(0).unwrap();
    let _ctx = Context::create_and_push(ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device).unwrap();
}

pub fn generate_random_points_bls12_381(
    count: usize,
    mut rng: Box<dyn RngCore>,
) -> Vec<PointAffineNoInfinity_BLS12_381> {
    (0..count)
        .map(|_| Point_BLS12_381::from_ark(G1Projective_BLS12_381::rand(&mut rng)).to_xy_strip_z())
        .collect()
}

pub fn generate_random_points_proj_bls12_381(count: usize, mut rng: Box<dyn RngCore>) -> Vec<Point_BLS12_381> {
    (0..count)
        .map(|_| Point_BLS12_381::from_ark(G1Projective_BLS12_381::rand(&mut rng)))
        .collect()
}

pub fn generate_random_scalars_bls12_381(count: usize, mut rng: Box<dyn RngCore>) -> Vec<ScalarField_BLS12_381> {
    (0..count)
        .map(|_| ScalarField_BLS12_381::from_ark(Fr_BLS12_381::rand(&mut rng).into_repr()))
        .collect()
}

pub fn set_up_points_bls12_381(test_size: usize, log_domain_size: usize, inverse: bool) -> (Vec<Point_BLS12_381>, DeviceBuffer<Point_BLS12_381>, DeviceBuffer<ScalarField_BLS12_381>) {
    set_up_device_bls12_381();

    let d_domain = build_domain_bls12_381(1 << log_domain_size, log_domain_size, inverse);

    let seed = Some(0); // fix the rng to get two equal scalar 
    let vector = generate_random_points_proj_bls12_381(test_size, get_rng_bls12_381(seed));
    let mut vector_mut = vector.clone();

    let mut d_vector = DeviceBuffer::from_slice(&vector[..]).unwrap();
    (vector_mut, d_vector, d_domain)
}

pub fn set_up_scalars_bls12_381(test_size: usize, log_domain_size: usize, inverse: bool) -> (Vec<ScalarField_BLS12_381>, DeviceBuffer<ScalarField_BLS12_381>, DeviceBuffer<ScalarField_BLS12_381>) {
    set_up_device_bls12_381();

    let d_domain = build_domain_bls12_381(1 << log_domain_size, log_domain_size, inverse);

    let seed = Some(0); // fix the rng to get two equal scalars
    let mut vector_mut = generate_random_scalars_bls12_381(test_size, get_rng_bls12_381(seed));

    let mut d_vector = DeviceBuffer::from_slice(&vector_mut[..]).unwrap();
    (vector_mut, d_vector, d_domain)
}


#[cfg(test)]
pub(crate) mod tests_bls12_381 {
    use std::fs::{File, self};
    use std::ops::Add;
    use std::path::PathBuf;
    use ark_bls12_381::{Fr, G1Affine, G1Projective};
    use ark_ec::{msm::VariableBaseMSM, AffineCurve, ProjectiveCurve};
    use ark_ff::{FftField, Field, Zero};
    use ark_std::UniformRand;
    use rand::{rngs::StdRng, RngCore, SeedableRng};
    use crate::test_bls12_381::*;
    use crate::{curves::bls12_381::*, *};

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
            let points = generate_random_points_bls12_381(count, get_rng_bls12_381(seed));
            let scalars = generate_random_scalars_bls12_381(count, get_rng_bls12_381(seed));

            let msm_result = msm_bls12_381(&points, &scalars, 0);

            let point_r_ark: Vec<_> = points.iter().map(|x| x.to_ark_repr()).collect();
            let scalars_r_ark: Vec<_> = scalars.iter().map(|x| x.to_ark()).collect();

            let msm_result_ark = VariableBaseMSM::multi_scalar_mul(&point_r_ark, &scalars_r_ark);

            assert_eq!(msm_result.to_ark_affine(), msm_result_ark);
            assert_eq!(msm_result.to_ark(), msm_result_ark);
            assert_eq!(
                msm_result.to_ark_affine(),
                Point_BLS12_381::from_ark(msm_result_ark).to_ark_affine()
            );
        }
    }

    #[test]
    fn test_poseidon() {
        let arities = [2, 4, 8, 11];
        let number_of_blocks = 1024;

        for arity in arities {
            // Generate scalars sequence [0, 1, ... arity * number_of_blocks]
            let scalars: Vec<ScalarField_BLS12_381> = (0..arity * number_of_blocks).map(|i| ScalarField_BLS12_381::from_ark(Fr_BLS12_381::from(i as i32).into_repr())).collect();
            let out = poseidon_multi_bls12_381(&scalars, arity, 0);

            let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
            path.push(format!("test_vectors/poseidon_1024_{}", arity));

            let test_vector: Vec<ScalarField_BLS12_381> = serde_cbor::from_slice(&fs::read(path).unwrap()).unwrap();
            assert!(out.iter().zip(test_vector.iter()).all(|(l, r)| l.eq(r)));
        }
    }

    #[test]
    fn test_batch_msm() {
        for batch_pow2 in [2, 4] {
            for pow2 in [4, 6] {
                let msm_size = 1 << pow2;
                let batch_size = 1 << batch_pow2;
                let seed = None; // set Some to provide seed
                let points_batch = generate_random_points_bls12_381(msm_size * batch_size, get_rng_bls12_381(seed));
                let scalars_batch = generate_random_scalars_bls12_381(msm_size * batch_size, get_rng_bls12_381(seed));

                let point_r_ark: Vec<_> = points_batch.iter().map(|x| x.to_ark_repr()).collect();
                let scalars_r_ark: Vec<_> = scalars_batch.iter().map(|x| x.to_ark()).collect();

                let expected: Vec<_> = point_r_ark
                    .chunks(msm_size)
                    .zip(scalars_r_ark.chunks(msm_size))
                    .map(|p| Point_BLS12_381::from_ark(VariableBaseMSM::multi_scalar_mul(p.0, p.1)))
                    .collect();

                let result = msm_batch_bls12_381(&points_batch, &scalars_batch, batch_size, 0);

                assert_eq!(result, expected);
            }
        }
    }

    #[test]
    fn test_commit() {
        let test_size = 1 << 8;
        let seed = Some(0);
        let (mut scalars, mut d_scalars, _) = set_up_scalars_bls12_381(test_size, 0, false);
        let mut points = generate_random_points_bls12_381(test_size, get_rng_bls12_381(seed));
        let mut d_points = DeviceBuffer::from_slice(&points[..]).unwrap();

        let msm_result = msm_bls12_381(&points, &scalars, 0);
        let mut d_commit_result = commit_bls12_381(&mut d_points, &mut d_scalars);
        let mut h_commit_result = Point_BLS12_381::zero();
        d_commit_result.copy_to(&mut h_commit_result).unwrap();

        assert_eq!(msm_result, h_commit_result);
        assert_ne!(msm_result, Point_BLS12_381::zero());
        assert_ne!(h_commit_result, Point_BLS12_381::zero());
    }

    #[test]
    fn test_batch_commit() {
        let batch_size = 4;
        let test_size = 1 << 12;
        let seed = Some(0);
        let (scalars, mut d_scalars, _) = set_up_scalars_bls12_381(test_size * batch_size, 0, false);
        let points = generate_random_points_bls12_381(test_size * batch_size, get_rng_bls12_381(seed));
        let mut d_points = DeviceBuffer::from_slice(&points[..]).unwrap();

        let msm_result = msm_batch_bls12_381(&points, &scalars, batch_size, 0);
        let mut d_commit_result = commit_batch_bls12_381(&mut d_points, &mut d_scalars, batch_size);
        let mut h_commit_result: Vec<Point_BLS12_381> = (0..batch_size).map(|_| Point_BLS12_381::zero()).collect();
        d_commit_result.copy_to(&mut h_commit_result[..]).unwrap();

        assert_eq!(msm_result, h_commit_result);
        for h in h_commit_result {
            assert_ne!(h, Point_BLS12_381::zero());
        }
    }

    #[test]
    fn test_ntt() {
        //NTT
        let seed = None; //some value to fix the rng
        let test_size = 1 << 3;

        let scalars = generate_random_scalars_bls12_381(test_size, get_rng_bls12_381(seed));

        let mut ntt_result = scalars.clone();
        ntt_bls12_381(&mut ntt_result, 0);

        assert_ne!(ntt_result, scalars);

        let mut intt_result = ntt_result.clone();

        intt_bls12_381(&mut intt_result, 0);

        assert_eq!(intt_result, scalars);

        //ECNTT
        let points_proj = generate_random_points_proj_bls12_381(test_size, get_rng_bls12_381(seed));

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
        ecntt_bls12_381(&mut ecntt_result, 0);

        assert_ne!(ecntt_result, points_proj);

        let mut iecntt_result = ecntt_result.clone();
        iecntt_bls12_381(&mut iecntt_result, 0);

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

        let scalars_batch: Vec<ScalarField_BLS12_381> =
            generate_random_scalars_bls12_381(test_size * batches, get_rng_bls12_381(seed));

        let mut scalar_vec_of_vec: Vec<Vec<ScalarField_BLS12_381>> = Vec::new();

        for i in 0..batches {
            scalar_vec_of_vec.push(scalars_batch[i * test_size..(i + 1) * test_size].to_vec());
        }

        let mut ntt_result = scalars_batch.clone();

        // do batch ntt
        ntt_batch_bls12_381(&mut ntt_result, test_size, 0);

        let mut ntt_result_vec_of_vec = Vec::new();

        // do ntt for every chunk
        for i in 0..batches {
            ntt_result_vec_of_vec.push(scalar_vec_of_vec[i].clone());
            ntt_bls12_381(&mut ntt_result_vec_of_vec[i], 0);
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
        intt_batch_bls12_381(&mut intt_result, test_size, 0);

        let mut intt_result_vec_of_vec = Vec::new();

        // do intt for every chunk
        for i in 0..batches {
            intt_result_vec_of_vec.push(ntt_result_vec_of_vec[i].clone());
            intt_bls12_381(&mut intt_result_vec_of_vec[i], 0);
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
        let points_proj = generate_random_points_proj_bls12_381(test_size * batches, get_rng_bls12_381(seed));

        let mut points_vec_of_vec: Vec<Vec<Point_BLS12_381>> = Vec::new();

        for i in 0..batches {
            points_vec_of_vec.push(points_proj[i * test_size..(i + 1) * test_size].to_vec());
        }

        let mut ntt_result_points = points_proj.clone();

        // do batch ecintt
        ecntt_batch_bls12_381(&mut ntt_result_points, test_size, 0);

        let mut ntt_result_points_vec_of_vec = Vec::new();

        for i in 0..batches {
            ntt_result_points_vec_of_vec.push(points_vec_of_vec[i].clone());
            ecntt_bls12_381(&mut ntt_result_points_vec_of_vec[i], 0);
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
        iecntt_batch_bls12_381(&mut intt_result_points, test_size, 0);

        let mut intt_result_points_vec_of_vec = Vec::new();

        // do ecintt for every chunk
        for i in 0..batches {
            intt_result_points_vec_of_vec.push(ntt_result_points_vec_of_vec[i].clone());
            iecntt_bls12_381(&mut intt_result_points_vec_of_vec[i], 0);
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
        let (mut evals_mut, mut d_evals, mut d_domain) = set_up_scalars_bls12_381(test_size, log_test_size, true);

        reverse_order_scalars_bls12_381(&mut d_evals);
        let mut d_coeffs = interpolate_scalars_bls12_381(&mut d_evals, &mut d_domain);
        intt_bls12_381(&mut evals_mut, 0);
        let mut h_coeffs: Vec<ScalarField_BLS12_381> = (0..test_size).map(|_| ScalarField_BLS12_381::zero()).collect();
        d_coeffs.copy_to(&mut h_coeffs[..]).unwrap();

        assert_eq!(h_coeffs, evals_mut);
    }

    #[test]
    fn test_scalar_batch_interpolation() {
        let batch_size = 4;
        let log_test_size = 10;
        let test_size = 1 << log_test_size;
        let (mut evals_mut, mut d_evals, mut d_domain) = set_up_scalars_bls12_381(test_size * batch_size, log_test_size, true);

        reverse_order_scalars_batch_bls12_381(&mut d_evals, batch_size);
        let mut d_coeffs = interpolate_scalars_batch_bls12_381(&mut d_evals, &mut d_domain, batch_size);
        intt_batch_bls12_381(&mut evals_mut, test_size, 0);
        let mut h_coeffs: Vec<ScalarField_BLS12_381> = (0..test_size * batch_size).map(|_| ScalarField_BLS12_381::zero()).collect();
        d_coeffs.copy_to(&mut h_coeffs[..]).unwrap();

        assert_eq!(h_coeffs, evals_mut);
    }

    #[test]
    fn test_point_interpolation() {
        let log_test_size = 6;
        let test_size = 1 << log_test_size;
        let (mut evals_mut, mut d_evals, mut d_domain) = set_up_points_bls12_381(test_size, log_test_size, true);

        reverse_order_points_bls12_381(&mut d_evals);
        let mut d_coeffs = interpolate_points_bls12_381(&mut d_evals, &mut d_domain);
        iecntt_bls12_381(&mut evals_mut[..], 0);
        let mut h_coeffs: Vec<Point_BLS12_381> = (0..test_size).map(|_| Point_BLS12_381::zero()).collect();
        d_coeffs.copy_to(&mut h_coeffs[..]).unwrap();
        
        assert_eq!(h_coeffs, *evals_mut);
        for h in h_coeffs.iter() {
            assert_ne!(*h, Point_BLS12_381::zero());
        }
    }

    #[test]
    fn test_point_batch_interpolation() {
        let batch_size = 4;
        let log_test_size = 6;
        let test_size = 1 << log_test_size;
        let (mut evals_mut, mut d_evals, mut d_domain) = set_up_points_bls12_381(test_size * batch_size, log_test_size, true);

        reverse_order_points_batch_bls12_381(&mut d_evals, batch_size);
        let mut d_coeffs = interpolate_points_batch_bls12_381(&mut d_evals, &mut d_domain, batch_size);
        iecntt_batch_bls12_381(&mut evals_mut[..], test_size, 0);
        let mut h_coeffs: Vec<Point_BLS12_381> = (0..test_size * batch_size).map(|_| Point_BLS12_381::zero()).collect();
        d_coeffs.copy_to(&mut h_coeffs[..]).unwrap();
        
        assert_eq!(h_coeffs, *evals_mut);
        for h in h_coeffs.iter() {
            assert_ne!(*h, Point_BLS12_381::zero());
        }
    }

    #[test]
    fn test_scalar_evaluation() {
        let log_test_domain_size = 8;
        let coeff_size = 1 << 6;
        let (h_coeffs, mut d_coeffs, mut d_domain) = set_up_scalars_bls12_381(coeff_size, log_test_domain_size, false);
        let (_, _, mut d_domain_inv) = set_up_scalars_bls12_381(0, log_test_domain_size, true);

        let mut d_evals = evaluate_scalars_bls12_381(&mut d_coeffs, &mut d_domain);
        let mut d_coeffs_domain = interpolate_scalars_bls12_381(&mut d_evals, &mut d_domain_inv);
        let mut h_coeffs_domain: Vec<ScalarField_BLS12_381> = (0..1 << log_test_domain_size).map(|_| ScalarField_BLS12_381::zero()).collect();
        d_coeffs_domain.copy_to(&mut h_coeffs_domain[..]).unwrap();

        assert_eq!(h_coeffs, h_coeffs_domain[..coeff_size]);
        for i in coeff_size.. (1 << log_test_domain_size) {
            assert_eq!(ScalarField_BLS12_381::zero(), h_coeffs_domain[i]);
        }
    }

    #[test]
    fn test_scalar_batch_evaluation() {
        let batch_size = 6;
        let log_test_domain_size = 8;
        let domain_size = 1 << log_test_domain_size;
        let coeff_size = 1 << 6;
        let (h_coeffs, mut d_coeffs, mut d_domain) = set_up_scalars_bls12_381(coeff_size * batch_size, log_test_domain_size, false);
        let (_, _, mut d_domain_inv) = set_up_scalars_bls12_381(0, log_test_domain_size, true);

        let mut d_evals = evaluate_scalars_batch_bls12_381(&mut d_coeffs, &mut d_domain, batch_size);
        let mut d_coeffs_domain = interpolate_scalars_batch_bls12_381(&mut d_evals, &mut d_domain_inv, batch_size);
        let mut h_coeffs_domain: Vec<ScalarField_BLS12_381> = (0..domain_size * batch_size).map(|_| ScalarField_BLS12_381::zero()).collect();
        d_coeffs_domain.copy_to(&mut h_coeffs_domain[..]).unwrap();

        for j in 0..batch_size {
            assert_eq!(h_coeffs[j * coeff_size..(j + 1) * coeff_size], h_coeffs_domain[j * domain_size..j * domain_size + coeff_size]);
            for i in coeff_size..domain_size {
                assert_eq!(ScalarField_BLS12_381::zero(), h_coeffs_domain[j * domain_size + i]);
            }
        }
    }

    #[test]
    fn test_point_evaluation() {
        let log_test_domain_size = 7;
        let coeff_size = 1 << 7;
        let (h_coeffs, mut d_coeffs, mut d_domain) = set_up_points_bls12_381(coeff_size, log_test_domain_size, false);
        let (_, _, mut d_domain_inv) = set_up_points_bls12_381(0, log_test_domain_size, true);

        let mut d_evals = evaluate_points_bls12_381(&mut d_coeffs, &mut d_domain);
        let mut d_coeffs_domain = interpolate_points_bls12_381(&mut d_evals, &mut d_domain_inv);
        let mut h_coeffs_domain: Vec<Point_BLS12_381> = (0..1 << log_test_domain_size).map(|_| Point_BLS12_381::zero()).collect();
        d_coeffs_domain.copy_to(&mut h_coeffs_domain[..]).unwrap();

        assert_eq!(h_coeffs[..], h_coeffs_domain[..coeff_size]);
        for i in coeff_size..(1 << log_test_domain_size) {
            assert_eq!(Point_BLS12_381::zero(), h_coeffs_domain[i]);
        }
        for i in 0..coeff_size {
            assert_ne!(h_coeffs_domain[i], Point_BLS12_381::zero());
        }
    }

    #[test]
    fn test_point_batch_evaluation() {
        let batch_size = 4;
        let log_test_domain_size = 6;
        let domain_size = 1 << log_test_domain_size;
        let coeff_size = 1 << 5;
        let (h_coeffs, mut d_coeffs, mut d_domain) = set_up_points_bls12_381(coeff_size * batch_size, log_test_domain_size, false);
        let (_, _, mut d_domain_inv) = set_up_points_bls12_381(0, log_test_domain_size, true);

        let mut d_evals = evaluate_points_batch_bls12_381(&mut d_coeffs, &mut d_domain, batch_size);
        let mut d_coeffs_domain = interpolate_points_batch_bls12_381(&mut d_evals, &mut d_domain_inv, batch_size);
        let mut h_coeffs_domain: Vec<Point_BLS12_381> = (0..domain_size * batch_size).map(|_| Point_BLS12_381::zero()).collect();
        d_coeffs_domain.copy_to(&mut h_coeffs_domain[..]).unwrap();

        for j in 0..batch_size {
            assert_eq!(h_coeffs[j * coeff_size..(j + 1) * coeff_size], h_coeffs_domain[j * domain_size..(j * domain_size + coeff_size)]);
            for i in coeff_size..domain_size {
                assert_eq!(Point_BLS12_381::zero(), h_coeffs_domain[j * domain_size + i]);
            }
            for i in j * domain_size..(j * domain_size + coeff_size) {
                assert_ne!(h_coeffs_domain[i], Point_BLS12_381::zero());
            }
        }
    }

    #[test]
    fn test_scalar_evaluation_on_trivial_coset() {
        // checks that the evaluations on the subgroup is the same as on the coset generated by 1
        let log_test_domain_size = 8;
        let coeff_size = 1 << 6;
        let (_, mut d_coeffs, mut d_domain) = set_up_scalars_bls12_381(coeff_size, log_test_domain_size, false);
        let (_, _, mut d_domain_inv) = set_up_scalars_bls12_381(coeff_size, log_test_domain_size, true);
        let mut d_trivial_coset_powers = build_domain_bls12_381(1 << log_test_domain_size, 0, false);

        let mut d_evals = evaluate_scalars_bls12_381(&mut d_coeffs, &mut d_domain);
        let mut h_coeffs: Vec<ScalarField_BLS12_381> = (0..1 << log_test_domain_size).map(|_| ScalarField_BLS12_381::zero()).collect();
        d_evals.copy_to(&mut h_coeffs[..]).unwrap();
        let mut d_evals_coset = evaluate_scalars_on_coset_bls12_381(&mut d_coeffs, &mut d_domain, &mut d_trivial_coset_powers);
        let mut h_evals_coset: Vec<ScalarField_BLS12_381> = (0..1 << log_test_domain_size).map(|_| ScalarField_BLS12_381::zero()).collect();
        d_evals_coset.copy_to(&mut h_evals_coset[..]).unwrap();

        assert_eq!(h_coeffs, h_evals_coset);
    }

    #[test]
    fn test_scalar_evaluation_on_coset() {
        // checks that evaluating a polynomial on a subgroup and its coset is the same as evaluating on a 2x larger subgroup 
        let log_test_size = 8;
        let test_size = 1 << log_test_size;
        let (_, mut d_coeffs, mut d_domain) = set_up_scalars_bls12_381(test_size, log_test_size, false);
        let (_, _, mut d_large_domain) = set_up_scalars_bls12_381(0, log_test_size + 1, false);
        let mut d_coset_powers = build_domain_bls12_381(test_size, log_test_size + 1, false);

        let mut d_evals_large = evaluate_scalars_bls12_381(&mut d_coeffs, &mut d_large_domain);
        let mut h_evals_large: Vec<ScalarField_BLS12_381> = (0..2 * test_size).map(|_| ScalarField_BLS12_381::zero()).collect();
        d_evals_large.copy_to(&mut h_evals_large[..]).unwrap();
        let mut d_evals = evaluate_scalars_bls12_381(&mut d_coeffs, &mut d_domain);
        let mut h_evals: Vec<ScalarField_BLS12_381> = (0..test_size).map(|_| ScalarField_BLS12_381::zero()).collect();
        d_evals.copy_to(&mut h_evals[..]).unwrap();
        let mut d_evals_coset = evaluate_scalars_on_coset_bls12_381(&mut d_coeffs, &mut d_domain, &mut d_coset_powers);
        let mut h_evals_coset: Vec<ScalarField_BLS12_381> = (0..test_size).map(|_| ScalarField_BLS12_381::zero()).collect();
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
        let (_, mut d_coeffs, mut d_domain) = set_up_scalars_bls12_381(test_size * batch_size, log_test_size, false);
        let (_, _, mut d_large_domain) = set_up_scalars_bls12_381(0, log_test_size + 1, false);
        let mut d_coset_powers = build_domain_bls12_381(test_size, log_test_size + 1, false);

        let mut d_evals_large = evaluate_scalars_batch_bls12_381(&mut d_coeffs, &mut d_large_domain, batch_size);
        let mut h_evals_large: Vec<ScalarField_BLS12_381> = (0..2 * test_size * batch_size).map(|_| ScalarField_BLS12_381::zero()).collect();
        d_evals_large.copy_to(&mut h_evals_large[..]).unwrap();
        let mut d_evals = evaluate_scalars_batch_bls12_381(&mut d_coeffs, &mut d_domain, batch_size);
        let mut h_evals: Vec<ScalarField_BLS12_381> = (0..test_size * batch_size).map(|_| ScalarField_BLS12_381::zero()).collect();
        d_evals.copy_to(&mut h_evals[..]).unwrap();
        let mut d_evals_coset = evaluate_scalars_on_coset_batch_bls12_381(&mut d_coeffs, &mut d_domain, batch_size, &mut d_coset_powers);
        let mut h_evals_coset: Vec<ScalarField_BLS12_381> = (0..test_size * batch_size).map(|_| ScalarField_BLS12_381::zero()).collect();
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
        let (_, mut d_coeffs, mut d_domain) = set_up_points_bls12_381(test_size, log_test_size, false);
        let (_, _, mut d_large_domain) = set_up_points_bls12_381(0, log_test_size + 1, false);
        let mut d_coset_powers = build_domain_bls12_381(test_size, log_test_size + 1, false);

        let mut d_evals_large = evaluate_points_bls12_381(&mut d_coeffs, &mut d_large_domain);
        let mut h_evals_large: Vec<Point_BLS12_381> = (0..2 * test_size).map(|_| Point_BLS12_381::zero()).collect();
        d_evals_large.copy_to(&mut h_evals_large[..]).unwrap();
        let mut d_evals = evaluate_points_bls12_381(&mut d_coeffs, &mut d_domain);
        let mut h_evals: Vec<Point_BLS12_381> = (0..test_size).map(|_| Point_BLS12_381::zero()).collect();
        d_evals.copy_to(&mut h_evals[..]).unwrap();
        let mut d_evals_coset = evaluate_points_on_coset_bls12_381(&mut d_coeffs, &mut d_domain, &mut d_coset_powers);
        let mut h_evals_coset: Vec<Point_BLS12_381> = (0..test_size).map(|_| Point_BLS12_381::zero()).collect();
        d_evals_coset.copy_to(&mut h_evals_coset[..]).unwrap();

        assert_eq!(h_evals[..], h_evals_large[..test_size]);
        assert_eq!(h_evals_coset[..], h_evals_large[test_size..2 * test_size]);
        for i in 0..test_size {
            assert_ne!(h_evals[i], Point_BLS12_381::zero());
            assert_ne!(h_evals_coset[i], Point_BLS12_381::zero());
            assert_ne!(h_evals_large[2 * i], Point_BLS12_381::zero());
            assert_ne!(h_evals_large[2 * i + 1], Point_BLS12_381::zero());
        }
    }

    #[test]
    fn test_point_batch_evaluation_on_coset() {
        // checks that evaluating a polynomial on a subgroup and its coset is the same as evaluating on a 2x larger subgroup 
        let batch_size = 2;
        let log_test_size = 6;
        let test_size = 1 << log_test_size;
        let (_, mut d_coeffs, mut d_domain) = set_up_points_bls12_381(test_size * batch_size, log_test_size, false);
        let (_, _, mut d_large_domain) = set_up_points_bls12_381(0, log_test_size + 1, false);
        let mut d_coset_powers = build_domain_bls12_381(test_size, log_test_size + 1, false);

        let mut d_evals_large = evaluate_points_batch_bls12_381(&mut d_coeffs, &mut d_large_domain, batch_size);
        let mut h_evals_large: Vec<Point_BLS12_381> = (0..2 * test_size * batch_size).map(|_| Point_BLS12_381::zero()).collect();
        d_evals_large.copy_to(&mut h_evals_large[..]).unwrap();
        let mut d_evals = evaluate_points_batch_bls12_381(&mut d_coeffs, &mut d_domain, batch_size);
        let mut h_evals: Vec<Point_BLS12_381> = (0..test_size * batch_size).map(|_| Point_BLS12_381::zero()).collect();
        d_evals.copy_to(&mut h_evals[..]).unwrap();
        let mut d_evals_coset = evaluate_points_on_coset_batch_bls12_381(&mut d_coeffs, &mut d_domain, batch_size, &mut d_coset_powers);
        let mut h_evals_coset: Vec<Point_BLS12_381> = (0..test_size * batch_size).map(|_| Point_BLS12_381::zero()).collect();
        d_evals_coset.copy_to(&mut h_evals_coset[..]).unwrap();

        for i in 0..batch_size {
            assert_eq!(h_evals_large[2 * i * test_size..(2 * i + 1) * test_size], h_evals[i * test_size..(i + 1) * test_size]);
            assert_eq!(h_evals_large[(2 * i + 1) * test_size..(2 * i + 2) * test_size], h_evals_coset[i * test_size..(i + 1) * test_size]);
        }
        for i in 0..test_size * batch_size {
            assert_ne!(h_evals[i], Point_BLS12_381::zero());
            assert_ne!(h_evals_coset[i], Point_BLS12_381::zero());
            assert_ne!(h_evals_large[2 * i], Point_BLS12_381::zero());
            assert_ne!(h_evals_large[2 * i + 1], Point_BLS12_381::zero());
        }
    }

    // testing matrix multiplication by comparing the result of FFT with the naive multiplication by the DFT matrix
    #[test]
    fn test_matrix_multiplication() {
        let seed = None; // some value to fix the rng
        let test_size = 1 << 5;
        let rou = Fr::get_root_of_unity(test_size).unwrap();
        let matrix_flattened: Vec<ScalarField_BLS12_381> = (0..test_size).map(
            |row_num| { (0..test_size).map( 
                |col_num| {
                    let pow: [u64; 1] = [(row_num * col_num).try_into().unwrap()];
                    ScalarField_BLS12_381::from_ark(Fr::pow(&rou, &pow).into_repr())
                }).collect::<Vec<ScalarField_BLS12_381>>()
            }).flatten().collect::<Vec<_>>();
        let vector: Vec<ScalarField_BLS12_381> = generate_random_scalars_bls12_381(test_size, get_rng_bls12_381(seed));

        let result = mult_matrix_by_vec_bls12_381(&matrix_flattened, &vector, 0);
        let mut ntt_result = vector.clone();
        ntt_bls12_381(&mut ntt_result, 0);
        
        // we don't use the same roots of unity as arkworks, so the results are permutations
        // of one another and the only guaranteed fixed scalars are the following ones:
        assert_eq!(result[0], ntt_result[0]);
        assert_eq!(result[test_size >> 1], ntt_result[test_size >> 1]);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_vec_scalar_mul() {
        let mut intoo = [ScalarField_BLS12_381::one(), ScalarField_BLS12_381::one(), ScalarField_BLS12_381::zero()];
        let expected = [ScalarField_BLS12_381::one(), ScalarField_BLS12_381::zero(), ScalarField_BLS12_381::zero()];
        mult_sc_vec_bls12_381(&mut intoo, &expected, 0);
        assert_eq!(intoo, expected);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_vec_point_mul() {
        let dummy_one = Point_BLS12_381 {
            x: BaseField_BLS12_381::one(),
            y: BaseField_BLS12_381::zero(),
            z: BaseField_BLS12_381::one(),
        };

        let mut inout = [dummy_one, dummy_one, Point_BLS12_381::zero()];
        let scalars = [ScalarField_BLS12_381::one(), ScalarField_BLS12_381::zero(), ScalarField_BLS12_381::zero()];
        let expected = [
            Point_BLS12_381::zero(),
            Point_BLS12_381 {
                x: BaseField_BLS12_381::zero(),
                y: BaseField_BLS12_381::one(),
                z: BaseField_BLS12_381::zero(),
            },
            Point_BLS12_381 {
                x: BaseField_BLS12_381::zero(),
                y: BaseField_BLS12_381::one(),
                z: BaseField_BLS12_381::zero(),
            },
        ];
        multp_vec_bls12_381(&mut inout, &scalars, 0);
        assert_eq!(inout, expected);
    }
}
