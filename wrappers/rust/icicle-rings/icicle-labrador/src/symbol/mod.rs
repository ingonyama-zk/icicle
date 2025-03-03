use crate::ring::{ScalarRing, ScalarRingRns};

use icicle_core::impl_symbol_field;

impl_symbol_field!("labrador", labrador, ScalarRing, ScalarCfg);
impl_symbol_field!("labrador_rns", labrador_rns, ScalarRingRns, ScalarCfgRns);
