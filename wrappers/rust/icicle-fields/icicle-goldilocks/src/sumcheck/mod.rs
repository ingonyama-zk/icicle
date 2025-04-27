use crate::field::GoldilocksField;
use icicle_core::impl_sumcheck;

impl_sumcheck!("goldilocks", goldilocks, GoldilocksField);

#[cfg(test)]
pub(crate) mod tests {
    use crate::field::GoldilocksField;
    use icicle_core::impl_sumcheck_tests;

    impl_sumcheck_tests!(goldilocks, GoldilocksField);
}
