use crate::field::KoalabearField;
use icicle_core::impl_poseidon2;

impl_poseidon2!("koalabear", koalabear, KoalabearField);

#[cfg(test)]
pub(crate) mod tests {
    use crate::field::KoalabearField;
    use icicle_core::impl_poseidon2_tests;
    use icicle_core::poseidon2::tests::*;

    impl_poseidon2_tests!(KoalabearField);
}
