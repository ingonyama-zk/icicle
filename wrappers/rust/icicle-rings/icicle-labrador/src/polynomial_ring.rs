use crate::ring::ScalarRing;
use icicle_core::impl_polynomial_ring;


// Define the Polynomial Ring Zq[X]/X^d+1
impl_polynomial_ring!(PolyRing, ScalarRing, 64, -1);


#[cfg(test)]
mod tests {
    use crate::polynomial_ring::PolyRing;
    use icicle_core::test_polynomial_ring;

    test_polynomial_ring!(PolyRing);
}
