use crate::curve::{CurveCfg, ScalarField};
use icicle_core::impl_ecntt;

impl_ecntt!("bls12_381", bls12_381, ScalarField, CurveCfg);

#[cfg(test)]
pub(crate) mod tests {
    use crate::curve::CurveCfg;

    use icicle_core::ecntt::tests::*;
    use icicle_core::impl_ecntt_tests;

    impl_ecntt_tests!(CurveCfg);
}
