use hex::FromHex;
use std::fmt::{Debug, Display};
use std::ops::{Mul, Add, Sub};
use icicle_core::{
    curve::{Affine, Curve, Projective},
    traits::{FieldImpl, ScalarImpl, MontgomeryConvertible, GenerateRandom},
    impl_curve, impl_field, impl_scalar_field,
    vec_ops::VecOpsConfig,
};
use icicle_runtime::{
    stream::{IcicleStream},
    memory::HostOrDeviceSlice,
    errors::eIcicleError,
};

pub(crate) const SCALAR_LIMBS: usize = 8;
pub(crate) const BASE_LIMBS: usize = 8;

impl_scalar_field!("grumpkin", grumpkin_f, grumpkin_sf, ScalarField, SCALAR_LIMBS);
impl_field!("grumpkin_point_field", grumpkin_bf, BaseField, BASE_LIMBS);
impl_curve!(
    "grumpkin",
    grumpkin,
    GrumpkinCurve,
    ScalarField,
    BaseField,
    G1Affine,
    G1Projective
);

#[cfg(test)]
mod tests {
    use super::{ScalarField, BASE_LIMBS, GrumpkinCurve};
    use icicle_core::tests::*;
    use icicle_core::{impl_curve_tests, impl_field_tests};
    use icicle_runtime::test_utilities;
    use icicle_core::curve::Curve;

    impl_field_tests!(ScalarField);
    impl_curve_tests!(BASE_LIMBS, GrumpkinCurve);
}
