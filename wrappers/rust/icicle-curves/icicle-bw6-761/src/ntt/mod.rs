#[cfg(test)]
pub(crate) mod tests {
    use crate::curve::ScalarField;
    use icicle_core::impl_ntt_tests;
    use icicle_core::ntt::tests::*;
    use serial_test::{parallel, serial};

    impl_ntt_tests!(ScalarField);
}
