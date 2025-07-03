use crate::field::{ExtensionField, ScalarField};

use icicle_core::impl_program_field;

impl_program_field!("goldilocks", goldilocks, ScalarField, ScalarCfg);
impl_program_field!(
    "goldilocks_extension",
    goldilocks_extension,
    ExtensionField,
    ExtensionCfg
);

#[cfg(test)]
use icicle_core::impl_program_tests;

#[cfg(test)]
impl_program_tests!(goldilocks, ScalarField);
