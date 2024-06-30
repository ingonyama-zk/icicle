use icicle_bn254::curve::ScalarField;

use icicle_core::impl_ntt_bench;

impl_ntt_bench!("bn254", ScalarField);
