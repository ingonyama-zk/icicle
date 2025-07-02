use crate::polynomial_ring::PolyRing;
use icicle_core::impl_matrix_ops;

impl_matrix_ops!("labrador_poly_ring", labrador_poly_ring, PolyRing, PolyRing);

#[cfg(test)]
mod tests {
    use crate::polynomial_ring::PolyRing;
    use icicle_core::impl_matrix_ops_tests;
    
    impl_matrix_ops_tests!(PolyRing);
}
