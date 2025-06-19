use super::{add_scalars, mul_scalars, scalar_mul, sub_scalars, sum_scalars, VecOps, VecOpsConfig};
use crate::{polynomial_ring::PolynomialRing, traits::FieldImpl};
use icicle_runtime::{
    errors::eIcicleError,
    memory::{DeviceSlice, HostOrDeviceSlice, HostSlice},
    stream::IcicleStream,
};
use std::mem::size_of;

// TODO Yuval: generalize this and move to icicle-runtime

enum UnifiedSlice<'a, T: 'a> {
    Host(&'a HostSlice<T>),
    Device(&'a DeviceSlice<T>),
}

enum UnifiedSliceMut<'a, T: 'a> {
    Host(&'a mut HostSlice<T>),
    Device(&'a mut DeviceSlice<T>),
}

/// SAFETY: Caller must ensure layout of P as [P::Base; DEGREE]
unsafe fn reinterpret_slice<From, To>(input: &(impl HostOrDeviceSlice<From> + ?Sized)) -> UnifiedSlice<'_, To>
where
    From: Sized,
    To: Sized,
{
    let len = input.len();
    let flat_len = len * size_of::<From>() / size_of::<To>();
    let ptr = input.as_ptr() as *const To;

    if input.is_on_device() {
        UnifiedSlice::Device(DeviceSlice::from_raw_parts(ptr, flat_len))
    } else {
        UnifiedSlice::Host(HostSlice::from_raw_parts(ptr, flat_len))
    }
}

unsafe fn reinterpret_slice_mut<From, To>(
    input: &mut (impl HostOrDeviceSlice<From> + ?Sized),
) -> UnifiedSliceMut<'_, To>
where
    From: Sized,
    To: Sized,
{
    let len = input.len();
    let flat_len = len * size_of::<From>() / size_of::<To>();
    let ptr = input.as_mut_ptr() as *mut To;

    if input.is_on_device() {
        UnifiedSliceMut::Device(DeviceSlice::from_raw_parts_mut(ptr, flat_len))
    } else {
        UnifiedSliceMut::Host(HostSlice::from_raw_parts_mut(ptr, flat_len))
    }
}

impl<'a, T> HostOrDeviceSlice<T> for UnifiedSlice<'a, T> {
    fn is_on_device(&self) -> bool {
        match self {
            UnifiedSlice::Device(d) => d.is_on_device(),
            UnifiedSlice::Host(h) => h.is_on_device(),
        }
    }

    fn is_on_active_device(&self) -> bool {
        match self {
            UnifiedSlice::Device(d) => d.is_on_active_device(),
            UnifiedSlice::Host(h) => h.is_on_active_device(),
        }
    }

    unsafe fn as_ptr(&self) -> *const T {
        match self {
            UnifiedSlice::Device(d) => d.as_ptr(),
            UnifiedSlice::Host(h) => h.as_ptr(),
        }
    }

    unsafe fn as_mut_ptr(&mut self) -> *mut T {
        panic!("Cannot get mutable pointer from immutable UnifiedSlice")
    }

    fn len(&self) -> usize {
        match self {
            UnifiedSlice::Device(d) => d.len(),
            UnifiedSlice::Host(h) => h.len(),
        }
    }

    fn is_empty(&self) -> bool {
        match self {
            UnifiedSlice::Device(d) => d.is_empty(),
            UnifiedSlice::Host(h) => h.is_empty(),
        }
    }

    fn memset(&mut self, _: u8, _: usize) -> Result<(), eIcicleError> {
        panic!("Cannot memset immutable UnifiedSlice");
    }

    fn memset_async(&mut self, _: u8, _: usize, _: &IcicleStream) -> Result<(), eIcicleError> {
        panic!("Cannot memset_async immutable UnifiedSlice");
    }
}

impl<'a, T> HostOrDeviceSlice<T> for UnifiedSliceMut<'a, T> {
    fn is_on_device(&self) -> bool {
        match self {
            UnifiedSliceMut::Device(d) => d.is_on_device(),
            UnifiedSliceMut::Host(h) => h.is_on_device(),
        }
    }

    fn is_on_active_device(&self) -> bool {
        match self {
            UnifiedSliceMut::Device(d) => d.is_on_active_device(),
            UnifiedSliceMut::Host(h) => h.is_on_active_device(),
        }
    }

    unsafe fn as_ptr(&self) -> *const T {
        match self {
            UnifiedSliceMut::Device(d) => d.as_ptr(),
            UnifiedSliceMut::Host(h) => h.as_ptr(),
        }
    }

    unsafe fn as_mut_ptr(&mut self) -> *mut T {
        match self {
            UnifiedSliceMut::Device(d) => d.as_mut_ptr(),
            UnifiedSliceMut::Host(h) => h.as_mut_ptr(),
        }
    }

    fn len(&self) -> usize {
        match self {
            UnifiedSliceMut::Device(d) => d.len(),
            UnifiedSliceMut::Host(h) => h.len(),
        }
    }

    fn is_empty(&self) -> bool {
        match self {
            UnifiedSliceMut::Device(d) => d.is_empty(),
            UnifiedSliceMut::Host(h) => h.is_empty(),
        }
    }

    fn memset(&mut self, value: u8, size: usize) -> Result<(), eIcicleError> {
        match self {
            UnifiedSliceMut::Device(d) => d.memset(value, size),
            UnifiedSliceMut::Host(h) => h.memset(value, size),
        }
    }

    fn memset_async(&mut self, value: u8, size: usize, stream: &IcicleStream) -> Result<(), eIcicleError> {
        match self {
            UnifiedSliceMut::Device(d) => d.memset_async(value, size, stream),
            UnifiedSliceMut::Host(h) => h.memset_async(value, size, stream),
        }
    }
}

// <Rq,Zq> vector-mul
pub fn polyvec_mul_by_scalar<P>(
    input_polyvec: &(impl HostOrDeviceSlice<P> + ?Sized),
    input_scalarvec: &(impl HostOrDeviceSlice<P::Base> + ?Sized),
    result: &mut (impl HostOrDeviceSlice<P> + ?Sized),
    cfg: &VecOpsConfig,
) -> Result<(), eIcicleError>
where
    P: PolynomialRing,
    P::Base: FieldImpl,
    <P::Base as FieldImpl>::Config: VecOps<P::Base>,
{
    unsafe {
        let vec_flat = reinterpret_slice::<P, P::Base>(input_polyvec);
        let mut result_flat = reinterpret_slice_mut::<P, P::Base>(result);
        let mut local_cfg = cfg.clone();
        local_cfg.batch_size = input_scalarvec.len() as i32;
        scalar_mul(input_scalarvec, &vec_flat, &mut result_flat, &local_cfg)
    }
}

// <Rq,Rq> vector-mul
pub fn polyvec_mul<P>(
    input_polyvec_a: &(impl HostOrDeviceSlice<P> + ?Sized),
    input_polyvec_b: &(impl HostOrDeviceSlice<P> + ?Sized),
    result: &mut (impl HostOrDeviceSlice<P> + ?Sized),
    cfg: &VecOpsConfig,
) -> Result<(), eIcicleError>
where
    P: PolynomialRing,
    P::Base: FieldImpl,
    <P::Base as FieldImpl>::Config: VecOps<P::Base>,
{
    unsafe {
        let vec_a_flat = reinterpret_slice::<P, P::Base>(input_polyvec_a);
        let vec_b_flat = reinterpret_slice::<P, P::Base>(input_polyvec_b);
        let mut result_flat = reinterpret_slice_mut::<P, P::Base>(result);
        mul_scalars(&vec_a_flat, &vec_b_flat, &mut result_flat, cfg)
    }
}

// <Rq,Rq> vector-add
pub fn polyvec_add<P>(
    input_polyvec_a: &(impl HostOrDeviceSlice<P> + ?Sized),
    input_polyvec_b: &(impl HostOrDeviceSlice<P> + ?Sized),
    result: &mut (impl HostOrDeviceSlice<P> + ?Sized),
    cfg: &VecOpsConfig,
) -> Result<(), eIcicleError>
where
    P: PolynomialRing,
    P::Base: FieldImpl,
    <P::Base as FieldImpl>::Config: VecOps<P::Base>,
{
    unsafe {
        let vec_a_flat = reinterpret_slice::<P, P::Base>(input_polyvec_a);
        let vec_b_flat = reinterpret_slice::<P, P::Base>(input_polyvec_b);
        let mut result_flat = reinterpret_slice_mut::<P, P::Base>(result);
        add_scalars(&vec_a_flat, &vec_b_flat, &mut result_flat, cfg)
    }
}
// <Rq,Rq> vector-sub
pub fn polyvec_sub<P>(
    input_polyvec_a: &(impl HostOrDeviceSlice<P> + ?Sized),
    input_polyvec_b: &(impl HostOrDeviceSlice<P> + ?Sized),
    result: &mut (impl HostOrDeviceSlice<P> + ?Sized),
    cfg: &VecOpsConfig,
) -> Result<(), eIcicleError>
where
    P: PolynomialRing,
    P::Base: FieldImpl,
    <P::Base as FieldImpl>::Config: VecOps<P::Base>,
{
    unsafe {
        let vec_a_flat = reinterpret_slice::<P, P::Base>(input_polyvec_a);
        let vec_b_flat = reinterpret_slice::<P, P::Base>(input_polyvec_b);
        let mut result_flat = reinterpret_slice_mut::<P, P::Base>(result);
        sub_scalars(&vec_a_flat, &vec_b_flat, &mut result_flat, cfg)
    }
}
// <Rq> vector-reduce
pub fn polyvec_sum_reduce<P>(
    input_polyvec: &(impl HostOrDeviceSlice<P> + ?Sized),
    result: &mut (impl HostOrDeviceSlice<P> + ?Sized),
    cfg: &VecOpsConfig,
) -> Result<(), eIcicleError>
where
    P: PolynomialRing,
    P::Base: FieldImpl,
    <P::Base as FieldImpl>::Config: VecOps<P::Base>,
{
    unsafe {
        let input_flat = reinterpret_slice::<P, P::Base>(input_polyvec);
        let mut result_flat = reinterpret_slice_mut::<P, P::Base>(result);
        let mut local_cfg = cfg.clone();
        local_cfg.batch_size = P::DEGREE as i32;
        local_cfg.columns_batch = true;
        sum_scalars(&input_flat, &mut result_flat, &local_cfg)
    }
}

#[macro_export]
macro_rules! impl_poly_vecops_tests {
    ($poly_type: ident) => {
        use icicle_runtime::test_utilities;

        /// Initializes devices before running tests.
        pub fn initialize() {
            test_utilities::test_load_and_init_devices();
            test_utilities::test_set_main_device();
        }

        #[test]
        fn test_polyvec_add_sub_mul() {
            initialize();
            test_utilities::test_set_main_device();
            check_poly_vecops_add_sub_mul::<$poly_type>();
            test_utilities::test_set_ref_device();
            check_poly_vecops_add_sub_mul::<$poly_type>();
        }

        #[test]
        fn test_polyvec_mul_by_scalar() {
            initialize();
            test_utilities::test_set_main_device();
            check_polyvec_mul_by_scalar::<$poly_type>();
            test_utilities::test_set_ref_device();
            check_polyvec_mul_by_scalar::<$poly_type>();
        }

        #[test]
        fn test_polyvec_sum_reduce() {
            initialize();
            test_utilities::test_set_main_device();
            check_polyvec_sum_reduce::<$poly_type>();
            test_utilities::test_set_ref_device();
            check_polyvec_sum_reduce::<$poly_type>();
        }
    };
}
