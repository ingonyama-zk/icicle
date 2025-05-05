use crate::ring::{ScalarRing, ScalarRingRns};

use icicle_core::impl_program_field;

impl_program_field!("labrador", labrador, ScalarRing);
impl_program_field!("labrador_rns", labrador_rns, ScalarRingRns);
