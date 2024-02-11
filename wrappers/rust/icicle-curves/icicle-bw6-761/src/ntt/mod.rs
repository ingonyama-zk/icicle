#[cfg(test)]
pub(crate) mod tests {
    use crate::curve::ScalarField;
    use icicle_cuda_runtime::device_context::DEFAULT_DEVICE_ID;
    use icicle_core::impl_ntt_tests;
    use icicle_core::ntt::tests::*;
    use std::sync::OnceLock;

    impl_ntt_tests!(ScalarField);
}
