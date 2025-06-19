use crate::field::{ExtensionField, ScalarField};

use icicle_core::impl_symbol_field;

impl_symbol_field!("goldilocks", goldilocks, ScalarField, ScalarCfg);
impl_symbol_field!(
    "goldilocks_extension",
    goldilocks_extension,
    ExtensionField,
    ExtensionCfg
);
