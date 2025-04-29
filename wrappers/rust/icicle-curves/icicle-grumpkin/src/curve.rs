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

pub(crate) const SCALAR_LIMBS: usize = 8;
pub(crate) const BASE_LIMBS: usize = 8;

impl_field!(GrumpkinScalarField, "grumpkin", SCALAR_LIMBS, true);
impl_field_arithmetic!(GrumpkinScalarField, "grumpkin", grumpkin_sf);
impl_montgomery_convertible!(GrumpkinScalarField, grumpkin_scalar_convert_montgomery);
impl_generate_random!(GrumpkinScalarField, grumpkin_generate_scalars);

impl_field!(GrumpkinBaseField, "grumpkin_base_field", BASE_LIMBS, false);

impl_curve!(
    "grumpkin",
    grumpkin,
    CurveCfg,
    GrumpkinScalarField,
    GrumpkinBaseField,
    G1Affine,
    G1Projective
);

#[cfg(test)]
mod tests {
    use super::{CurveCfg, GrumpkinScalarField};
    use icicle_core::curve::Curve;
    use icicle_core::tests::*;
    use icicle_core::{impl_curve_tests, impl_field_tests};
    use icicle_runtime::test_utilities;

    impl_field_tests!(GrumpkinScalarField);
    impl_curve_tests!(BASE_LIMBS, CurveCfg);
}
