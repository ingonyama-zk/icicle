use crate::ring::{LabradorScalarRing, LabradorScalarRingRns};

use icicle_core::impl_program_field;

impl_program_field!("labrador", labrador, LabradorScalarRing);
impl_program_field!("labrador_rns", labrador_rns, LabradorScalarRingRns);
