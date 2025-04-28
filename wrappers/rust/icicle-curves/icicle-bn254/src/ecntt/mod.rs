use crate::curve::{Bn254ScalarField, CurveCfg};
use icicle_core::ecntt::Projective;
use icicle_core::impl_ecntt;

impl_ecntt!("bn254", bn254, Bn254ScalarField, CurveCfg);

#[cfg(test)]
pub(crate) mod tests {
    use crate::curve::{Bn254ScalarField, CurveCfg};

    use icicle_core::ecntt::tests::*;
    use icicle_core::impl_ecntt_tests;

    impl_ecntt_tests!(Bn254ScalarField, CurveCfg);
}
