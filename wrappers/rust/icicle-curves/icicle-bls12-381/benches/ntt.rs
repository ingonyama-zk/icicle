use icicle_bls12_381::curve::ScalarField;

use icicle_core::impl_ntt_bench;

impl_ntt_bench!("BLS12_381", ScalarField);
