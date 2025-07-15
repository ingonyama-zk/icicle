pub use crate::ring::{ScalarRing, ScalarRingRns};
use icicle_core::impl_symbol_ring;

impl_symbol_ring!("babykoala", babykoala, ScalarRing);
impl_symbol_ring!("babykoala_rns", babykoala_rns, ScalarRingRns);
