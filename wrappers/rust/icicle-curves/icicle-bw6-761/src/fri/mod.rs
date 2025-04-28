#[cfg(test)]
mod tests {
    use icicle_core::impl_fri_tests;

    use crate::curve::Bw6761ScalarField;

    impl_fri_tests!(bw6_761_fri_test, Bw6761ScalarField, Bw6761ScalarField);
}
