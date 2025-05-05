use crate::curve::ScalarField;
use icicle_core::impl_sumcheck;

impl_sumcheck!("grumpkin", grumpkin, ScalarField);

#[cfg(test)]
pub(crate) mod tests {
    use crate::curve::ScalarField;
    use icicle_core::impl_sumcheck_tests;

    impl_sumcheck_tests!(grumpkin, ScalarField);
}
