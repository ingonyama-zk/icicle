use crate::field::M31Field;
use icicle_core::impl_poseidon;

impl_poseidon!("m31", m31, M31Field);

#[cfg(test)]
pub(crate) mod tests {
    use crate::field::M31Field;
    use icicle_core::impl_poseidon_tests;
    use icicle_core::poseidon::tests::*;

    impl_poseidon_tests!(M31Field);
}
