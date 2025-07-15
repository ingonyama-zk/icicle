pub use crate::ring::{ScalarRing, ScalarRingRns};
use icicle_core::impl_program_ring;

impl_program_ring!("babykoala", babykoala, ScalarRing);
impl_program_ring!("babykoala_rns", babykoala_rns, ScalarRingRns);

// TODO: add tests that do not require Inverse
