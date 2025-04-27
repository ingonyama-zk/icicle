use icicle_core::impl_univariate_polynomial_api;

use crate::field::KoalabearField;

impl_univariate_polynomial_api!("koalabear", koalabear, KoalabearField);

#[cfg(test)]
mod tests {
    use icicle_core::impl_polynomial_tests;
    impl_polynomial_tests!(koalabear, KoalabearField);
}
