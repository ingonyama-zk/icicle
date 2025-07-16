use crate::field::{ExtensionField, ScalarField};

use icicle_core::impl_program_ring;

impl_program_ring!("goldilocks", goldilocks, ScalarField);
impl_program_ring!("goldilocks_extension", goldilocks_extension, ExtensionField);

#[cfg(test)]
use icicle_core::impl_program_tests;

#[cfg(test)]
impl_program_tests!(goldilocks, ScalarField);
