use crate::curve::{G1Projective, ScalarField};
use icicle_core::impl_ecntt;

impl_ecntt!("bls12_377", bls12_377, ScalarField, G1Projective);

#[cfg(test)]
pub(crate) mod tests {
    use icicle_core::ecntt::tests::*;
    use icicle_core::impl_ecntt_tests;

    use crate::curve::G1Projective;

    impl_ecntt_tests!(G1Projective);
}
