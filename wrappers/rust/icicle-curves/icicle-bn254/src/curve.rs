use icicle_core::affine::Affine;
use icicle_core::projective::Projective;
use icicle_core::{impl_curve, impl_field, impl_montgomery_convertible};
use icicle_runtime::{errors::eIcicleError, memory::HostOrDeviceSlice, stream::IcicleStream};

pub(crate) const SCALAR_LIMBS: usize = 8;
pub(crate) const BASE_LIMBS: usize = 8;
#[cfg(feature = "g2")]
pub(crate) const G2_BASE_LIMBS: usize = 16;

impl_field!(ScalarField, "bn254", SCALAR_LIMBS);
impl_montgomery_convertible!(ScalarField, "bn254_scalar_convert_montgomery");

impl_field!(BaseField, "bn254_base_field", BASE_LIMBS);
impl_curve!("bn254", ScalarField, BaseField, G1Affine, G1Projective);

#[cfg(feature = "g2")]
impl_field!(G2BaseField, "bn254_g2_base_field", G2_BASE_LIMBS);
#[cfg(feature = "g2")]
impl_curve!("bn254_g2", ScalarField, G2BaseField, G2Affine, G2Projective);

#[cfg(test)]
mod tests {
    use super::{G1Projective, ScalarField};
    use icicle_core::tests::*;
    use icicle_core::{impl_curve_tests, impl_field_tests};
    use icicle_runtime::test_utilities;

    impl_field_tests!(ScalarField);
    impl_curve_tests!(BASE_LIMBS, G1Projective);
    #[cfg(feature = "g2")]
    mod g2 {
        use super::*;
        use crate::curve::G2Projective;
        impl_curve_tests!(G2_BASE_LIMBS, G2Projective);
    }
}
