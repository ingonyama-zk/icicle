use icicle_core::{impl_bignum, impl_curve, impl_field, impl_generate_random_ffi, impl_montgomery_convertible_ffi};
use icicle_runtime::{errors::eIcicleError, memory::HostOrDeviceSlice, stream::IcicleStream};

pub(crate) const SCALAR_LIMBS: usize = 8;
pub(crate) const BASE_LIMBS: usize = 12;
#[cfg(feature = "g2")]
pub(crate) const G2_BASE_LIMBS: usize = 24;

impl_field!(ScalarField, "bls12_381", SCALAR_LIMBS, true, true);
impl_montgomery_convertible_ffi!(ScalarField, "bls12_381_scalar_convert_montgomery");
impl_generate_random_ffi!(ScalarField, "bls12_381_generate_scalars");

impl_bignum!(BaseField, "bls12_381_base_field", BASE_LIMBS, false);
impl_curve!("bls12_381", CurveCfg, ScalarField, BaseField, G1Affine, G1Projective);

#[cfg(feature = "g2")]
impl_bignum!(G2BaseField, "bls12_381_g2_base_field", G2_BASE_LIMBS, false);
#[cfg(feature = "g2")]
impl_curve!(
    "bls12_381_g2",
    G2CurveCfg,
    ScalarField,
    G2BaseField,
    G2Affine,
    G2Projective
);

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
