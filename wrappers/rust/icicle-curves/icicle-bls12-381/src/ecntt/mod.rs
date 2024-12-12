use crate::curve::{Bls12381Curve, ScalarField};
use icicle_core::ecntt::Projective;
use icicle_core::impl_ecntt;

impl_ecntt!("bls12_381", bls12_381, ScalarField, Bls12381Curve);

#[cfg(test)]
pub(crate) mod tests {
    use crate::curve::{Bls12381Curve, ScalarField};

    use icicle_core::ecntt::tests::*;
    use icicle_core::impl_ecntt_tests;

    impl_ecntt_tests!(ScalarField, Bls12381Curve);
}
