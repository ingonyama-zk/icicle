#[cfg(test)]
pub(crate) mod tests {
    use crate::curve::ScalarField;
    use icicle_core::poseidon::tests::*;
    use icicle_core::{impl_poseidon_custom_config_test, impl_poseidon_tests};

    impl_poseidon_tests!(ScalarField);
    impl_poseidon_custom_config_test!(ScalarField, 48, "bw6-761", 56);
}
