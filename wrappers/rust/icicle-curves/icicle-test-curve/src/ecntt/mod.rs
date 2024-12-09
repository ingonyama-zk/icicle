use crate::curve::{Bn254Curve, ScalarField};
use icicle_core::ecntt::Projective;
use icicle_core::impl_ecntt;

impl_ecntt!("bn254", bn254, ScalarField, Bn254Curve);

#[cfg(test)]
pub(crate) mod tests {
    use crate::curve::{Bn254Curve, ScalarField};

    use icicle_core::ecntt::tests::*;
    use icicle_core::impl_ecntt_tests;

    impl_ecntt_tests!(ScalarField, Bn254Curve);
}
