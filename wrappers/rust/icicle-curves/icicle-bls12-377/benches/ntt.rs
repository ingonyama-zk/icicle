use icicle_bls12_377::curve::Bls12_377ScalarField;

use icicle_core::impl_ntt_bench;

impl_ntt_bench!("bls12_377", Bls12_377ScalarField);
