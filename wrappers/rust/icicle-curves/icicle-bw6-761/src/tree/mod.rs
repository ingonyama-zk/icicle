#[cfg(test)]
pub(crate) mod tests {
    use crate::curve::ScalarField;
    use icicle_core::impl_tree_builder_tests;
    use icicle_core::tree::tests::*;

    impl_tree_builder_tests!(ScalarField);
}
