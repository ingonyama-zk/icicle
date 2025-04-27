use crate::field::KoalabearField;
use icicle_core::impl_poseidon;

impl_poseidon!("koalabear", koalabear, KoalabearField);

#[cfg(test)]
pub(crate) mod tests {
    use crate::field::KoalabearField;
    use icicle_core::impl_poseidon_tests;
    use icicle_core::poseidon::tests::*;

    impl_poseidon_tests!(KoalabearField);
}
