use super::{add_scalars, mul_scalars, scalar_mul, sub_scalars, sum_scalars, VecOps, VecOpsConfig};
use crate::{polynomial_ring::PolynomialRing, traits::FieldImpl};
use icicle_runtime::{
    errors::eIcicleError,
    memory::{reinterpret_slice, reinterpret_slice_mut, HostOrDeviceSlice},
};

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
