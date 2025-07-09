use icicle_core::affine::Affine;
use icicle_core::bignum::BigNum;
use icicle_core::projective::Projective;
use icicle_core::{impl_curve, impl_field, impl_montgomery_convertible};
use icicle_runtime::{errors::eIcicleError, memory::HostOrDeviceSlice, stream::IcicleStream};

pub(crate) const SCALAR_LIMBS: usize = 8;
pub(crate) const BASE_LIMBS: usize = 8;

impl_field!(ScalarField, "grumpkin", SCALAR_LIMBS);
impl_montgomery_convertible!(ScalarField, "grumpkin_scalar_convert_montgomery");

impl_field!(BaseField, "grumpkin_base_field", BASE_LIMBS);
impl_curve!("grumpkin", CurveCfg, ScalarField, BaseField, G1Affine, G1Projective);

#[cfg(test)]
mod tests {
    use super::{CurveCfg, ScalarField};
    use icicle_core::tests::*;
    use icicle_core::{impl_curve_tests, impl_field_tests};
    use icicle_runtime::test_utilities;

    impl_field_tests!(ScalarField);
    impl_curve_tests!(BASE_LIMBS, CurveCfg);
}
