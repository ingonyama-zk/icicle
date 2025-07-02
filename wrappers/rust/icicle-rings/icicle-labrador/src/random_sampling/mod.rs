use crate::ring::ScalarRing;
use icicle_core::impl_random_sampling;

impl_random_sampling!("labrador", ScalarRing, ScalarRing);

#[cfg(test)]
pub(crate) mod tests {
    use crate::ring::ScalarRing;
    use icicle_core::impl_random_sampling_tests;

    impl_random_sampling_tests!(ScalarRing);
}
