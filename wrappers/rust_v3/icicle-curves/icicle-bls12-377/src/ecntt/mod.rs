#![cfg(feature = "ec_ntt")]
use crate::curve::{CurveCfg, ScalarCfg, ScalarField};
use icicle_core::ecntt::Projective;
use icicle_core::impl_ecntt;

impl_ecntt!("bls12_377", bls12_377, ScalarField, ScalarCfg, CurveCfg);

#[cfg(test)]
pub(crate) mod tests {
    use crate::curve::{CurveCfg, ScalarField};

    use icicle_core::ecntt::tests::*;
    use icicle_core::impl_ecntt_tests;

    impl_ecntt_tests!(ScalarField, CurveCfg);
}
