use icicle_core::{impl_bignum, impl_curve, impl_field, impl_generate_random_ffi, impl_montgomery_convertible_ffi};
use icicle_runtime::{errors::eIcicleError, memory::HostOrDeviceSlice, stream::IcicleStream};

pub(crate) const SCALAR_LIMBS: usize = 8;
pub(crate) const BASE_LIMBS: usize = 8;
#[cfg(feature = "g2")]
pub(crate) const G2_BASE_LIMBS: usize = 16;

impl_field!(ScalarField, "bn254", SCALAR_LIMBS, true, true);
impl_montgomery_convertible_ffi!(ScalarField, bn254_scalar_convert_montgomery);
impl_generate_random_ffi!(ScalarField, bn254_generate_scalars);

impl_bignum!(BaseField, "bn254", BASE_LIMBS, false);
impl_curve!("bn254", CurveCfg, ScalarField, BaseField, G1Affine, G1Projective);

#[cfg(feature = "g2")]
impl_bignum!(G2BaseField, "bn254_g2", G2_BASE_LIMBS, false);
#[cfg(feature = "g2")]
impl_curve!("bn254_g2", G2CurveCfg, ScalarField, G2BaseField, G2Affine, G2Projective);

#[cfg(test)]
mod tests {
    #[cfg(feature = "g2")]
    use super::G2CurveCfg;
    use super::{CurveCfg, ScalarField};
    use icicle_core::tests::*;
    use icicle_core::{impl_curve_tests, impl_field_tests};
    use icicle_runtime::test_utilities;

    impl_field_tests!(ScalarField);
    impl_curve_tests!(BASE_LIMBS, CurveCfg);
    #[cfg(feature = "g2")]
    mod g2 {
        use super::*;
        impl_curve_tests!(G2_BASE_LIMBS, G2CurveCfg);
    }
}
