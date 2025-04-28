use crate::curve::GrumpkinScalarField;
use icicle_core::impl_poseidon2;

impl_poseidon2!("grumpkin", grumpkin, GrumpkinScalarField);

#[cfg(test)]
pub(crate) mod tests {
    use crate::curve::GrumpkinScalarField;
    use icicle_core::impl_poseidon2_tests;
    use icicle_core::poseidon2::tests::*;

    impl_poseidon2_tests!(GrumpkinScalarField);
}
