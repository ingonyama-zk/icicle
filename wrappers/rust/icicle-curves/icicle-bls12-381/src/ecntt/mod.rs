use crate::curve::{Bls12_381ScalarField, CurveCfg};
use icicle_core::ecntt::Projective;
use icicle_core::impl_ecntt;

impl_ecntt!("bls12_381", bls12_381, Bls12_381ScalarField, CurveCfg);

#[cfg(test)]
pub(crate) mod tests {
    use crate::curve::{Bls12_381ScalarField, CurveCfg};

    use icicle_core::ecntt::tests::*;
    use icicle_core::impl_ecntt_tests;

    impl_ecntt_tests!(Bls12_381ScalarField, CurveCfg);
}
