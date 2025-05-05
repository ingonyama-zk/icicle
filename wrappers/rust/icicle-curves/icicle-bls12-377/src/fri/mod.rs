use crate::curve::ScalarField;

use icicle_core::impl_fri;

impl_fri!("bls12_377", bls12_377_fri, ScalarField);

#[cfg(test)]
mod tests {
    use crate::curve::ScalarField;
    use icicle_core::{impl_fri_test_with_poseidon, impl_fri_tests};
    impl_fri_tests!(ScalarField, ScalarField);
    impl_fri_test_with_poseidon!(ScalarField, ScalarField);
}
