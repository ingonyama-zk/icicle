use icicle_core::impl_fri;

use crate::field::{ScalarCfg, ScalarField};

impl_fri!("stark252", stark252_fri, ScalarField, ScalarCfg);

#[cfg(test)]
mod tests {
    use icicle_core::impl_fri_tests;

    use crate::field::ScalarField;

    impl_fri_tests!(stark252_fri_test, ScalarField, ScalarField);
}
