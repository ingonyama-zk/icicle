use crate::ring::{LabradorScalarRing, LabradorScalarRingRns};

use icicle_core::impl_symbol_field;

impl_symbol_field!("labrador", labrador, LabradorScalarRing);
impl_symbol_field!("labrador_rns", labrador_rns, LabradorScalarRingRns);
