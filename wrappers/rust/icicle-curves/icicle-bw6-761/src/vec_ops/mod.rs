#[cfg(test)]
pub(crate) mod tests {
    use crate::curve::Bw6761ScalarField;
    use icicle_core::impl_vec_ops_tests;
    use icicle_core::vec_ops::tests::*;

    impl_vec_ops_tests!(bw6_761, Bw6761ScalarField);
}
