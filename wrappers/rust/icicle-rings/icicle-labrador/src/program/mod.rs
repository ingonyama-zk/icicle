pub use crate::ring::{ScalarRing, ScalarRingRns};
use icicle_core::impl_program_ring;

impl_program_ring!("labrador", labrador, ScalarRing);
impl_program_ring!("labrador_rns", labrador_rns, ScalarRingRns);

#[cfg(test)]
pub(crate) mod tests {
    use crate::ring::{ScalarRing, ScalarRingRns};
    use icicle_core::impl_program_tests;

    impl_program_tests!(labrador, ScalarRing);

    mod rns {
        use super::*;
        impl_program_tests!(labrador_rns, ScalarRingRns);
    }
}
