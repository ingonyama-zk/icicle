use std::ffi::{c_int, c_uint};
use ark_std::UniformRand;
use rand::{rngs::StdRng, RngCore, SeedableRng};
use rustacuda::CudaFlags;
use rustacuda::memory::DeviceBox;
use rustacuda::prelude::{DeviceBuffer, Device, ContextFlags, Context};
use rustacuda_core::DevicePointer;
use std::mem::transmute;
use crate::basic_structs::scalar::ScalarTrait;
use crate::curves::bn254::*;
use crate::{utils::{u32_vec_to_u64_vec, u64_vec_to_u32_vec}};
use std::marker::PhantomData;
use std::convert::TryInto;
use ark_bn254::{Fq as Fq_BN254, Fr as Fr_BN254, G1Affine as G1Affine_BN254, G1Projective as G1Projective_BN254};
use ark_ec::AffineCurve;
use ark_ff::{BigInteger384, BigInteger256, PrimeField};
use rustacuda::memory::{CopyDestination, DeviceCopy};

extern "C" {
    fn msm_cuda_bn254(
        out: *mut Point,
        points: *const PointAffineNoInfinity,
        scalars: *const Scalar,
        count: usize,
        device_id: usize,
    ) -> c_uint;

    fn msm_batch_cuda_bn254(
        out: *mut Point,
        points: *const PointAffineNoInfinity,
        scalars: *const Scalar,
        batch_size: usize,
        msm_size: usize,
        device_id: usize,
    ) -> c_uint;

    fn commit_cuda_bn254(
        d_out: DevicePointer<Point>,
        d_scalars: DevicePointer<Scalar>,
        d_points: DevicePointer<PointAffineNoInfinity>,
        count: usize,
        device_id: usize,
    ) -> c_uint;

    fn commit_batch_cuda_bn254(
        d_out: DevicePointer<Point>,
        d_scalars: DevicePointer<Scalar>,
        d_points: DevicePointer<PointAffineNoInfinity>,
        count: usize,
        batch_size: usize,
        device_id: usize,
    ) -> c_uint;

    fn build_domain_cuda_bn254(domain_size: usize, logn: usize, inverse: bool, device_id: usize) -> DevicePointer<Scalar>;

    fn ntt_cuda_bn254(inout: *mut Scalar, n: usize, inverse: bool, device_id: usize) -> c_int;

    fn ecntt_cuda_bn254(inout: *mut Point, n: usize, inverse: bool, device_id: usize) -> c_int;

    fn ntt_batch_cuda_bn254(
        inout: *mut Scalar,
        arr_size: usize,
        n: usize,
        inverse: bool,
    ) -> c_int;

    fn ecntt_batch_cuda_bn254(inout: *mut Point, arr_size: usize, n: usize, inverse: bool) -> c_int;

    fn interpolate_scalars_cuda_bn254(
        d_out: DevicePointer<Scalar>,
        d_evaluations: DevicePointer<Scalar>,
        d_domain: DevicePointer<Scalar>, 
        n: usize,
        device_id: usize
    ) -> c_int;

    fn interpolate_scalars_batch_cuda_bn254(
        d_out: DevicePointer<Scalar>,
        d_evaluations: DevicePointer<Scalar>,
        d_domain: DevicePointer<Scalar>,
        n: usize,
        batch_size: usize,
        device_id: usize
    ) -> c_int;

    fn interpolate_points_cuda_bn254(
        d_out: DevicePointer<Point>,
        d_evaluations: DevicePointer<Point>,
        d_domain: DevicePointer<Scalar>,
        n: usize,
        device_id: usize
    ) -> c_int;

    fn interpolate_points_batch_cuda_bn254(
        d_out: DevicePointer<Point>,
        d_evaluations: DevicePointer<Point>,
        d_domain: DevicePointer<Scalar>,
        n: usize,
        batch_size: usize,
        device_id: usize
    ) -> c_int;

    fn evaluate_scalars_cuda_bn254(
        d_out: DevicePointer<Scalar>,
        d_coefficients: DevicePointer<Scalar>,
        d_domain: DevicePointer<Scalar>,
        domain_size: usize,
        n: usize,
        device_id: usize
    ) -> c_int;

    fn evaluate_scalars_batch_cuda_bn254(
        d_out: DevicePointer<Scalar>,
        d_coefficients: DevicePointer<Scalar>,
        d_domain: DevicePointer<Scalar>,
        domain_size: usize,
        n: usize,
        batch_size: usize,
        device_id: usize
    ) -> c_int;

    fn evaluate_points_cuda_bn254(
        d_out: DevicePointer<Point>,
        d_coefficients: DevicePointer<Point>,
        d_domain: DevicePointer<Scalar>,
        domain_size: usize,
        n: usize,
        device_id: usize
    ) -> c_int;

    fn evaluate_points_batch_cuda_bn254(
        d_out: DevicePointer<Point>,
        d_coefficients: DevicePointer<Point>,
        d_domain: DevicePointer<Scalar>,
        domain_size: usize,
        n: usize,
        batch_size: usize,
        device_id: usize
    ) -> c_int;

    fn evaluate_scalars_on_coset_cuda_bn254(
        d_out: DevicePointer<Scalar>,
        d_coefficients: DevicePointer<Scalar>,
        d_domain: DevicePointer<Scalar>,
        domain_size: usize,
        n: usize,
        coset_powers: DevicePointer<Scalar>,
        device_id: usize
    ) -> c_int;

    fn evaluate_scalars_on_coset_batch_cuda_bn254(
        d_out: DevicePointer<Scalar>,
        d_coefficients: DevicePointer<Scalar>,
        d_domain: DevicePointer<Scalar>,
        domain_size: usize,
        n: usize,
        batch_size: usize,
        coset_powers: DevicePointer<Scalar>,
        device_id: usize
    ) -> c_int;

    fn evaluate_points_on_coset_cuda_bn254(
        d_out: DevicePointer<Point>,
        d_coefficients: DevicePointer<Point>,
        d_domain: DevicePointer<Scalar>,
        domain_size: usize,
        n: usize,
        coset_powers: DevicePointer<Scalar>,
        device_id: usize
    ) -> c_int;

    fn evaluate_points_on_coset_batch_cuda_bn254(
        d_out: DevicePointer<Point>,
        d_coefficients: DevicePointer<Point>,
        d_domain: DevicePointer<Scalar>,
        domain_size: usize,
        n: usize,
        batch_size: usize,
        coset_powers: DevicePointer<Scalar>,
        device_id: usize
    ) -> c_int;

    fn reverse_order_scalars_cuda_bn254(
        d_arr: DevicePointer<Scalar>,
        n: usize,
        device_id: usize
    ) -> c_int;

    fn reverse_order_scalars_batch_cuda_bn254(
        d_arr: DevicePointer<Scalar>,
        n: usize,
        batch_size: usize,
        device_id: usize
    ) -> c_int;

    fn reverse_order_points_cuda_bn254(
        d_arr: DevicePointer<Point>,
        n: usize,
        device_id: usize
    ) -> c_int;

    fn reverse_order_points_batch_cuda_bn254(
        d_arr: DevicePointer<Point>,
        n: usize,
        batch_size: usize,
        device_id: usize
    ) -> c_int;

    fn vec_mod_mult_point_bn254(
        inout: *mut Point,
        scalars: *const Scalar,
        n_elements: usize,
        device_id: usize,
    ) -> c_int;

    fn vec_mod_mult_scalar_bn254(
        inout: *mut Scalar,
        scalars: *const Scalar,
        n_elements: usize,
        device_id: usize,
    ) -> c_int;

    fn matrix_vec_mod_mult_bn254(
        matrix_flattened: *const Scalar,
        input: *const Scalar,
        output: *mut Scalar,
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
        msm_cuda_bn254(
            &mut ret as *mut _ as *mut Point,
            points as *const _ as *const PointAffineNoInfinity,
            scalars as *const _ as *const Scalar,
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
        msm_batch_cuda_bn254(
            &mut ret[0] as *mut _ as *mut Point,
            points as *const _ as *const PointAffineNoInfinity,
            scalars as *const _ as *const Scalar,
            batch_size,
            count / batch_size,
            device_id,
        )
    };

    ret
}

pub fn commit(
    points: &mut DeviceBuffer<PointAffineNoInfinity>,
    scalars: &mut DeviceBuffer<Scalar>,
) -> DeviceBox<Point> {
    let mut res = DeviceBox::new(&Point::zero()).unwrap();
    unsafe {
        commit_cuda_bn254(
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
    scalars: &mut DeviceBuffer<Scalar>,
    batch_size: usize,
) -> DeviceBuffer<Point> {
    let mut res = unsafe { DeviceBuffer::uninitialized(batch_size).unwrap() };
    unsafe {
        commit_batch_cuda_bn254(
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
fn ntt_internal(values: &mut [Scalar], device_id: usize, inverse: bool) -> i32 {
    let ret_code = unsafe {
        ntt_cuda_bn254(
            values as *mut _ as *mut Scalar,
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

/// Compute an in-place NTT on the input data.
fn ntt_internal_batch(
    values: &mut [Scalar],
    device_id: usize,
    batch_size: usize,
    inverse: bool,
) -> i32 {
    unsafe {
        ntt_batch_cuda_bn254(
            values as *mut _ as *mut Scalar,
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
        ecntt_cuda_bn254(
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
        ecntt_batch_cuda_bn254(
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

pub fn build_domain(domain_size: usize, logn: usize, inverse: bool) -> DeviceBuffer<Scalar> {
    unsafe {
        DeviceBuffer::from_raw_parts(build_domain_cuda_bn254(
            domain_size,
            logn,
            inverse,
            0
        ), domain_size)
    }
}


pub fn reverse_order_scalars(
    d_scalars: &mut DeviceBuffer<Scalar>,
) {
    unsafe { reverse_order_scalars_cuda_bn254(
        d_scalars.as_device_ptr(),
        d_scalars.len(),
        0
    ); }
}

pub fn reverse_order_scalars_batch(
    d_scalars: &mut DeviceBuffer<Scalar>,
    batch_size: usize,
) {
    unsafe { reverse_order_scalars_batch_cuda_bn254(
        d_scalars.as_device_ptr(),
        d_scalars.len() / batch_size,
        batch_size,
        0
    ); }
}

pub fn reverse_order_points(
    d_points: &mut DeviceBuffer<Point>,
) {
    unsafe { reverse_order_points_cuda_bn254(
        d_points.as_device_ptr(),
        d_points.len(),
        0
    ); }
}

pub fn reverse_order_points_batch(
    d_points: &mut DeviceBuffer<Point>,
    batch_size: usize,
) {
    unsafe { reverse_order_points_batch_cuda_bn254(
        d_points.as_device_ptr(),
        d_points.len() / batch_size,
        batch_size,
        0
    ); }
}

pub fn interpolate_scalars(
    d_evaluations: &mut DeviceBuffer<Scalar>,
    d_domain: &mut DeviceBuffer<Scalar>
) -> DeviceBuffer<Scalar> {
    let mut res = unsafe { DeviceBuffer::uninitialized(d_domain.len()).unwrap() };
    unsafe { interpolate_scalars_cuda_bn254(
        res.as_device_ptr(),
        d_evaluations.as_device_ptr(),
        d_domain.as_device_ptr(),
        d_domain.len(),
        0
    ) };
    return res;
}

pub fn interpolate_scalars_batch(
    d_evaluations: &mut DeviceBuffer<Scalar>,
    d_domain: &mut DeviceBuffer<Scalar>,
    batch_size: usize,
) -> DeviceBuffer<Scalar> {
    let mut res = unsafe { DeviceBuffer::uninitialized(d_domain.len() * batch_size).unwrap() };
    unsafe { interpolate_scalars_batch_cuda_bn254(
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
    d_domain: &mut DeviceBuffer<Scalar>,
) -> DeviceBuffer<Point> {
    let mut res = unsafe { DeviceBuffer::uninitialized(d_domain.len()).unwrap() };
    unsafe { interpolate_points_cuda_bn254(
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
    d_domain: &mut DeviceBuffer<Scalar>,
    batch_size: usize,
) -> DeviceBuffer<Point> {
    let mut res = unsafe { DeviceBuffer::uninitialized(d_domain.len() * batch_size).unwrap() };
    unsafe { interpolate_points_batch_cuda_bn254(
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
    d_coefficients: &mut DeviceBuffer<Scalar>,
    d_domain: &mut DeviceBuffer<Scalar>,
) -> DeviceBuffer<Scalar> {
    let mut res = unsafe { DeviceBuffer::uninitialized(d_domain.len()).unwrap() };
    unsafe {
        evaluate_scalars_cuda_bn254(
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
    d_coefficients: &mut DeviceBuffer<Scalar>,
    d_domain: &mut DeviceBuffer<Scalar>,
    batch_size: usize,
) -> DeviceBuffer<Scalar> {
    let mut res = unsafe { DeviceBuffer::uninitialized(d_domain.len() * batch_size).unwrap() };
    unsafe {
        evaluate_scalars_batch_cuda_bn254(
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
    d_domain: &mut DeviceBuffer<Scalar>,
) -> DeviceBuffer<Point> {
    let mut res = unsafe { DeviceBuffer::uninitialized(d_domain.len()).unwrap() };
    unsafe {
        evaluate_points_cuda_bn254(
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
    d_domain: &mut DeviceBuffer<Scalar>,
    batch_size: usize,
) -> DeviceBuffer<Point> {
    let mut res = unsafe { DeviceBuffer::uninitialized(d_domain.len() * batch_size).unwrap() };
    unsafe {
        evaluate_points_batch_cuda_bn254(
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
    d_coefficients: &mut DeviceBuffer<Scalar>,
    d_domain: &mut DeviceBuffer<Scalar>,
    coset_powers: &mut DeviceBuffer<Scalar>,
) -> DeviceBuffer<Scalar> {
    let mut res = unsafe { DeviceBuffer::uninitialized(d_domain.len()).unwrap() };
    unsafe {
        evaluate_scalars_on_coset_cuda_bn254(
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
    d_coefficients: &mut DeviceBuffer<Scalar>,
    d_domain: &mut DeviceBuffer<Scalar>,
    batch_size: usize,
    coset_powers: &mut DeviceBuffer<Scalar>,
) -> DeviceBuffer<Scalar> {
    let mut res = unsafe { DeviceBuffer::uninitialized(d_domain.len() * batch_size).unwrap() };
    unsafe {
        evaluate_scalars_on_coset_batch_cuda_bn254(
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
    d_domain: &mut DeviceBuffer<Scalar>,
    coset_powers: &mut DeviceBuffer<Scalar>,
) -> DeviceBuffer<Point> {
    let mut res = unsafe { DeviceBuffer::uninitialized(d_domain.len()).unwrap() };
    unsafe {
        evaluate_points_on_coset_cuda_bn254(
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
    d_domain: &mut DeviceBuffer<Scalar>,
    batch_size: usize,
    coset_powers: &mut DeviceBuffer<Scalar>,
) -> DeviceBuffer<Point> {
    let mut res = unsafe { DeviceBuffer::uninitialized(d_domain.len() * batch_size).unwrap() };
    unsafe {
        evaluate_points_on_coset_batch_cuda_bn254(
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

pub fn multp_vec(a: &mut [Point], b: &[Scalar], device_id: usize) {
    assert_eq!(a.len(), b.len());
    unsafe {
        vec_mod_mult_point_bn254(
            a as *mut _ as *mut Point,
            b as *const _ as *const Scalar,
            a.len(),
            device_id,
        );
    }
}

pub fn mult_sc_vec(a: &mut [Scalar], b: &[Scalar], device_id: usize) {
    assert_eq!(a.len(), b.len());
    unsafe {
        vec_mod_mult_scalar_bn254(
            a as *mut _ as *mut Scalar,
            b as *const _ as *const Scalar,
            a.len(),
            device_id,
        );
    }
}

// Multiply a matrix by a scalar:
//  `a` - flattenned matrix;
//  `b` - vector to multiply `a` by;
pub fn mult_matrix_by_vec(a: &[Scalar], b: &[Scalar], device_id: usize) -> Vec<Scalar> {
    let mut c = Vec::with_capacity(b.len());
    for i in 0..b.len() {
        c.push(Scalar::zero());
    }
    unsafe {
        matrix_vec_mod_mult_bn254(
            a as *const _ as *const Scalar,
            b as *const _ as *const Scalar,
            c.as_mut_slice() as *mut _ as *mut Scalar,
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
        .map(|_| Point::from_ark(G1Projective_BN254::rand(&mut rng)).to_xy_strip_z())
        .collect()
}

pub fn generate_random_points_proj(count: usize, mut rng: Box<dyn RngCore>) -> Vec<Point> {
    (0..count)
        .map(|_| Point::from_ark(G1Projective_BN254::rand(&mut rng)))
        .collect()
}

pub fn generate_random_scalars(count: usize, mut rng: Box<dyn RngCore>) -> Vec<Scalar> {
    (0..count)
        .map(|_| Scalar::from_ark(Fr_BN254::rand(&mut rng).into_repr()))
        .collect()
}

pub fn set_up_points(test_size: usize, log_domain_size: usize, inverse: bool) -> (Vec<Point>, DeviceBuffer<Point>, DeviceBuffer<Scalar>) {
    set_up_device();

    let d_domain = build_domain(1 << log_domain_size, log_domain_size, inverse);

    let seed = Some(0); // fix the rng to get two equal scalar 
    let vector = generate_random_points_proj(test_size, get_rng(seed));
    let mut vector_mut = vector.clone();

    let mut d_vector = DeviceBuffer::from_slice(&vector[..]).unwrap();
    (vector_mut, d_vector, d_domain)
}

pub fn set_up_scalars(test_size: usize, log_domain_size: usize, inverse: bool) -> (Vec<Scalar>, DeviceBuffer<Scalar>, DeviceBuffer<Scalar>) {
    set_up_device();

    let d_domain = build_domain(1 << log_domain_size, log_domain_size, inverse);

    let seed = Some(0); // fix the rng to get two equal scalars
    let mut vector_mut = generate_random_scalars(test_size, get_rng(seed));

    let mut d_vector = DeviceBuffer::from_slice(&vector_mut[..]).unwrap();
    (vector_mut, d_vector, d_domain)
}

