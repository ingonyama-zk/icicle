use icicle_bn254::curve::Bn254ScalarField;

use icicle_core::impl_ntt_bench;

impl_ntt_bench!("bn254", Bn254ScalarField);
