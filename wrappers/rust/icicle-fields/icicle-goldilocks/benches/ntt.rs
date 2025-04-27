use icicle_goldilocks::field::GoldilocksField;

use icicle_core::impl_ntt_bench;

impl_ntt_bench!("goldilocks", GoldilocksField);
