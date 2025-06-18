use super::{VecOps, VecOpsConfig};
use crate::{polynomial_ring::{PolynomialRing, flatt}, traits::FieldImpl};
use icicle_runtime::{errors::eIcicleError, memory::HostOrDeviceSlice};

// <Rq,Zq> vector-mul
pub fn vector_mul_by_scalar<P>(
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
    Ok(())
}

// <Rq,Rq> vector-mul
// <Rq,Rq> vector-add
// <Rq,Rq> vector-sub
// <Rq> vector-reduce
