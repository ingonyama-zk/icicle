use icicle_bls12_381::curve::{BaseField, CurveCfg, ScalarField};

use icicle_core::impl_ecntt_bench;

impl_ecntt_bench!("BLS12_381", ScalarField, BaseField, CurveCfg);
