use crate::field::{KoalabearExtensionField, KoalabearField};

use icicle_core::impl_program_field;

impl_program_field!("koalabear", koalabear, KoalabearField);
impl_program_field!("koalabear_extension", koalabear_extension, KoalabearExtensionField);
