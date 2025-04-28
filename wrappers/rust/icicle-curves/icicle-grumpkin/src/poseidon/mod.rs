use crate::curve::GrumpkinScalarField;
use icicle_core::impl_poseidon;

impl_poseidon!("grumpkin", grumpkin, GrumpkinScalarField);

#[cfg(test)]
pub(crate) mod tests {
    use crate::curve::GrumpkinScalarField;
    use icicle_core::impl_poseidon_tests;
    use icicle_core::poseidon::tests::*;

    impl_poseidon_tests!(GrumpkinScalarField);
}
