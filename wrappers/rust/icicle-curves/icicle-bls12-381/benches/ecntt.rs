
use icicle_bls12_381::curve::{CurveCfg, BaseField, ScalarField};

use icicle_core::impl_ecntt_bench;
use std::sync::OnceLock;

impl_ecntt_bench!(ScalarField, BaseField, CurveCfg);