use icicle_core::impl_sumcheck;

use crate::field::ScalarField;

impl_sumcheck!("koalabear", koalabear, ScalarField);

#[cfg(test)]
pub(crate) mod tests {
    use icicle_core::impl_sumcheck_tests;

    use crate::field::ScalarField;

    impl_sumcheck_tests!(koalabear, ScalarField);
}
