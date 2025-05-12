#[cfg(test)]
mod tests {
    mod bw6_761_fri_test {
        use icicle_core::{impl_fri_test_with_poseidon, impl_fri_tests};
        use crate::curve::ScalarField;
        impl_fri_tests!(ScalarField, ScalarField);
        impl_fri_test_with_poseidon!(ScalarField, ScalarField);
    }
}
