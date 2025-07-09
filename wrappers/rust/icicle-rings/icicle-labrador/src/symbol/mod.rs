pub use crate::ring::{ScalarRing, ScalarRingRns};
use icicle_core::impl_symbol_ring;

impl_symbol_ring!("labrador", labrador, ScalarRing);
impl_symbol_ring!("labrador_rns", labrador_rns, ScalarRingRns);
