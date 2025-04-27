use crate::field::GoldilocksField;
use icicle_core::impl_poseidon;

impl_poseidon!("goldilocks", goldilocks, GoldilocksField);

#[cfg(test)]
pub(crate) mod tests {
    use crate::field::GoldilocksField;
    use icicle_core::impl_poseidon_tests;
    use icicle_core::poseidon::tests::*;

    impl_poseidon_tests!(GoldilocksField);
}
