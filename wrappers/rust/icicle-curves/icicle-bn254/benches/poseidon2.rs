use icicle_bn254::curve::ScalarField;

use icicle_core::impl_poseidon2_bench;

impl_poseidon2_bench!("bn254", ScalarField);
