use crate::curve::{Bls12_377ScalarField, CurveCfg};
use icicle_core::ecntt::Projective;
use icicle_core::impl_ecntt;

impl_ecntt!("bls12_377", bls12_377, Bls12_377ScalarField, CurveCfg);

#[cfg(test)]
pub(crate) mod tests {
    use crate::curve::{Bls12_377ScalarField, CurveCfg};

    use icicle_core::ecntt::tests::*;
    use icicle_core::impl_ecntt_tests;

    impl_ecntt_tests!(Bls12_377ScalarField, CurveCfg);
}
