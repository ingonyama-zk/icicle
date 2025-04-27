use icicle_core::impl_fri;

use crate::field::ScalarField;

impl_fri!("goldilocks", goldilocks_fri, ScalarField);

#[cfg(test)]
mod tests {
    use crate::field::ScalarField;
    use icicle_core::impl_fri_tests;
    impl_fri_tests!(ScalarField, ScalarField);
}

