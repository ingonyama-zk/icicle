use crate::curve::{G1Projective, ScalarField};
use icicle_core::impl_ecntt;

impl_ecntt!("bn254", bn254, ScalarField, G1Projective);

#[cfg(test)]
pub(crate) mod tests {
    use crate::curve::G1Projective;

    use icicle_core::ecntt::tests::*;
    use icicle_core::impl_ecntt_tests;

    impl_ecntt_tests!(G1Projective);
}
