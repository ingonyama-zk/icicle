use crate::field::M31Field;
use icicle_core::impl_poseidon2;

impl_poseidon2!("m31", m31, M31Field);

#[cfg(test)]
pub(crate) mod tests {
    use crate::field::M31Field;
    use icicle_core::impl_poseidon2_tests;
    use icicle_core::poseidon2::tests::*;

    impl_poseidon2_tests!(M31Field);
}
