use icicle_core::impl_fri;

use crate::field::GoldilocksField;

impl_fri!("goldilocks", goldilocks_fri, GoldilocksField);

#[cfg(test)]
mod tests {
    use icicle_core::impl_fri_tests;

    use crate::field::GoldilocksField;

    impl_fri_tests!(goldilocks_fri_test, GoldilocksField, GoldilocksField);
}
