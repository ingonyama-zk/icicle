use crate::curve::{G1Projective, ScalarField};
use icicle_core::impl_ecntt;

impl_ecntt!("bls12_381", bls12_381, ScalarField, G1Projective);

#[cfg(test)]
pub(crate) mod tests {
    use crate::curve::G1Projective;
    use icicle_core::ecntt::tests::*;
    use icicle_core::impl_ecntt_tests;

    impl_ecntt_tests!(G1Projective);
}
