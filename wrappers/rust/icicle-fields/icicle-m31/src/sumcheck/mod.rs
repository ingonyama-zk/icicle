use icicle_core::impl_sumcheck;

use crate::field::ScalarField;

impl_sumcheck!("m31", m31, ScalarField);

#[cfg(test)]
pub(crate) mod tests {
    use icicle_core::impl_sumcheck_tests;

    use crate::field::ScalarField;

    impl_sumcheck_tests!(m31, ScalarField);
}
