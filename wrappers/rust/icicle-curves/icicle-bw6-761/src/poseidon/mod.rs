#[cfg(test)]
pub(crate) mod tests {
    use crate::curve::Bw6761ScalarField;
    use icicle_core::impl_poseidon_tests;
    use icicle_core::poseidon::tests::*;

    impl_poseidon_tests!(Bw6761ScalarField);
}
