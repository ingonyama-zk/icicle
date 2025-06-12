use crate::ring::ScalarRing;
use icicle_core::impl_polynomial_ring;
use icicle_core::traits::FieldImpl;

// Define the Polynomial Ring Zq[X]/X^64+1 with Zq=ScalarRing type
pub(crate) const RQ_DEGREE: usize = 64;
impl_polynomial_ring!(Rq, ScalarRing, RQ_DEGREE, -1);

#[cfg(test)]
mod tests {
    use crate::polynomial_ring::Rq;
    use icicle_core::test_polynomial_ring;

    test_polynomial_ring!(Rq);
}
