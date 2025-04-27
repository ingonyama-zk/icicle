use crate::field::GoldilocksField;
use icicle_core::impl_poseidon2;

impl_poseidon2!("goldilocks", goldilocks, GoldilocksField);

#[cfg(test)]
pub(crate) mod tests {
    use crate::field::GoldilocksField;
    use icicle_core::impl_poseidon2_tests;
    use icicle_core::poseidon2::tests::*;

    impl_poseidon2_tests!(GoldilocksField);
}
