#[cfg(test)]
mod tests {
    use icicle_core::impl_fri_tests;

    use crate::curve::ScalarField;

    impl_fri_tests!(bw6_761_fri_test, ScalarField, ScalarField);
}
