use crate::polynomial_ring::PolyRing;
use crate::ring::ScalarRing;
use icicle_core::{impl_challenge_space_polynomials_sampling, impl_random_sampling};

impl_random_sampling!("babykoala", ScalarRing);
impl_challenge_space_polynomials_sampling!("babykoala", PolyRing);

#[cfg(test)]
pub(crate) mod tests {
    use crate::polynomial_ring::PolyRing;
    use crate::ring::ScalarRing;
    use icicle_core::{impl_challenge_space_polynomials_sampling_tests, impl_random_sampling_tests};

    impl_random_sampling_tests!(ScalarRing);

    impl_challenge_space_polynomials_sampling_tests!(PolyRing);
}
