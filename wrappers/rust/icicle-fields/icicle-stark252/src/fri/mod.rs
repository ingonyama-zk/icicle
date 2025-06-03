use icicle_core::impl_fri;

use crate::field::{ScalarCfg, ScalarField};

impl_fri!("stark252", stark252_fri, ScalarField, ScalarCfg);

#[cfg(test)]
mod tests {
    use crate::field::ScalarField;
    use icicle_core::{impl_fri_test_with_poseidon, impl_fri_tests};
    impl_fri_tests!(ScalarField, ScalarField);
    impl_fri_test_with_poseidon!(ScalarField, ScalarField);
}
