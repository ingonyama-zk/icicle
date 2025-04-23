use crate::field::{BabybearExtensionField, BabybearField};

use icicle_core::impl_program_field;

impl_program_field!("babybear", babybear, BabybearField);
impl_program_field!("babybear_extension", babybear_extension, BabybearExtensionField);
