use icicle_core::impl_fri;

use crate::curve::ScalarField;

impl_fri!("bn254", bn254_fri, ScalarField);

#[cfg(test)]
mod tests {
    use crate::curve::ScalarField;
    use icicle_core::{impl_fri_test_with_poseidon, impl_fri_tests};
    impl_fri_tests!(ScalarField, ScalarField);
    impl_fri_test_with_poseidon!(ScalarField, ScalarField);
}
