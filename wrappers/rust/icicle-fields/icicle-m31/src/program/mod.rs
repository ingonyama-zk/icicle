use crate::field::{ExtensionField, ScalarField};

use icicle_core::impl_program_field;

impl_program_field!("m31", m31, ScalarField);
impl_program_field!("m31_extension", m31_extension, ExtensionField);
