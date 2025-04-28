use icicle_bls12_377::curve::BaseField as Bls12_377BaseField;
use icicle_core::traits::{Arithmetic, GenerateRandom, MontgomeryConvertible};
use icicle_core::{
    curve::{Affine, Curve, Projective},
    field::PrimeField,
    impl_curve, impl_field, impl_field_arithmetic, impl_generate_random, impl_montgomery_convertible,
    vec_ops::VecOpsConfig,
};
use icicle_runtime::{eIcicleError, memory::HostOrDeviceSlice, stream::IcicleStream};
use std::fmt::{Debug, Display};
use std::ops::{Add, Mul, Sub};

pub(crate) const SCALAR_LIMBS: usize = 12;
pub(crate) const BASE_LIMBS: usize = 24;

impl_field!(BaseField, "bw6_761_g2_base_field", BASE_LIMBS, false);
impl_field_arithmetic!(BaseField, "bw6_761_g2_base_field", bw6_761_g2_base_field);
impl_montgomery_convertible!(BaseField, bw6_761_g2_base_field_convert_montgomery);
impl_generate_random!(BaseField, bw6_761_g2_base_field_generate_random);

// NOTE: Even though both G1 and G2 use the same base field, we define two different field types
//       to avoid using incorrect FFI functions.
impl_field!("bw6_761_base_field", BASE_LIMBS, BaseField, false);
#[cfg(feature = "g2")]
impl_field!("bw6_761_g2_base_field", BASE_LIMBS, G2BaseField, false);

impl_curve!(
    "bw6_761",
    bw6_761,
    CurveCfg,
    ScalarField,
    BaseField,
    G1Affine,
    G1Projective
);

#[cfg(feature = "g2")]
impl_curve!(
    "bw6_761_g2",
    bw6_761_g2,
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
    use super::{Bw6761ScalarField, CurveCfg};
    use icicle_core::curve::Curve;
    use icicle_core::tests::*;
    use icicle_core::{impl_curve_tests, impl_field_tests};
    use icicle_runtime::test_utilities;

    impl_field_tests!(Bw6761ScalarField);
    impl_curve_tests!(BASE_LIMBS, CurveCfg);
    #[cfg(feature = "g2")]
    mod g2 {
        use super::*;
        impl_curve_tests!(BASE_LIMBS, G2CurveCfg);
    }
}
