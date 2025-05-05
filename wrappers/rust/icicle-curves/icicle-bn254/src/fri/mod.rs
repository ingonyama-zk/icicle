use icicle_core::impl_fri;

use crate::curve::ScalarField;

impl_fri!("bn254", bn254_fri, ScalarField);

#[cfg(test)]
mod tests {
    use icicle_core::impl_fri_tests;

    use crate::curve::ScalarField;

    impl_fri_tests!(bn254_fri_test, ScalarField, ScalarField);
}
