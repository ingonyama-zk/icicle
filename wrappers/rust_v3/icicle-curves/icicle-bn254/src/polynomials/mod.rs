use crate::curve::{ScalarCfg, ScalarField};
use icicle_core::impl_univariate_polynomial_api;

impl_univariate_polynomial_api!("bn254", bn254, ScalarField, ScalarCfg);

#[cfg(test)]
mod tests {
    use icicle_core::impl_polynomial_tests;
    impl_polynomial_tests!(bn254, ScalarField);
}
