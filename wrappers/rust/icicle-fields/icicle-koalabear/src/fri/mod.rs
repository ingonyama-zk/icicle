use icicle_core::impl_fri;

use crate::field::{KoalabearExtensionField, KoalabearField};

impl_fri!("koalabear", koalabear_fri, KoalabearField);
impl_fri!("koalabear_extension", koalabear_extension_fri, KoalabearExtensionField);

#[cfg(test)]
mod tests {
    use icicle_core::impl_fri_tests;

    use crate::field::{KoalabearExtensionField, KoalabearField};

    impl_fri_tests!(koalabear_fri_test, KoalabearField, KoalabearField);
    impl_fri_tests!(koalabear_extension_fri_test, KoalabearField, KoalabearExtensionField);
}
