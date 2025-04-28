use crate::curve::GrumpkinScalarField;
use icicle_core::impl_sumcheck;

impl_sumcheck!("grumpkin", grumpkin, GrumpkinScalarField);

#[cfg(test)]
pub(crate) mod tests {
    use crate::curve::GrumpkinScalarField;
    use icicle_core::impl_sumcheck_tests;

    impl_sumcheck_tests!(grumpkin, GrumpkinScalarField);
}
