use crate::field::{KoalabearExtensionField, KoalabearField};

use icicle_core::impl_symbol_field;

impl_symbol_field!("koalabear", koalabear, KoalabearField);
impl_symbol_field!("koalabear_extension", koalabear_extension, KoalabearExtensionField);
