use icicle_bw6_761::curve::Bw6761ScalarField;

use icicle_core::impl_ntt_bench;

impl_ntt_bench!("bw6_761", Bw6761ScalarField);
