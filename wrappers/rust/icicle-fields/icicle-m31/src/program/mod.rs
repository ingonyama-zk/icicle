use crate::field::{M31ExtensionField, M31Field};

use icicle_core::impl_program_field;

impl_program_field!("m31", m31, M31Field);
impl_program_field!("m31_extension", m31_extension, M31ExtensionField);
