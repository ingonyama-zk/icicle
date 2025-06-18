use crate::{
    polynomial_ring::PolyRing,
    ring::{ScalarCfg, ScalarRing},
};
use icicle_core::{impl_random_sampling, impl_random_sampling_polyring};

impl_random_sampling!("labrador", ScalarRing, ScalarCfg);
impl_random_sampling_polyring!("labrador", PolyRing);

#[cfg(test)]
pub(crate) mod tests {
    use crate::polynomial_ring::PolyRing;
    use crate::ring::ScalarRing;
    use icicle_core::{impl_random_sampling_polyring_tests, impl_random_sampling_tests};

    impl_random_sampling_tests!(ScalarRing);
    impl_random_sampling_polyring_tests!(PolyRing);
}
