use crate::field::{BabybearExtensionField, BabybearField};

use icicle_core::impl_symbol_field;

impl_symbol_field!("babybear", babybear, BabybearField, ScalarCfg);
impl_symbol_field!(
    "babybear_extension",
    babybear_extension,
    BabybearExtensionField,
    ExtensionCfg
);
