pub use crate::ring::{ScalarRing, ScalarRingRns};
use icicle_core::impl_program_ring;

impl_program_ring!("labrador", labrador, ScalarRing);
impl_program_ring!("labrador_rns", labrador_rns, ScalarRingRns);

// TODO: add tests that do not require Inverse
