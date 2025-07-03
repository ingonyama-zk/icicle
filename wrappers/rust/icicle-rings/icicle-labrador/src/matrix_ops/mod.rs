use icicle_core::impl_matrix_ops;

impl_matrix_ops!("labrador", labrador, crate::ring::ScalarRing);
impl_matrix_ops!("labrador_rns", labrador_rns, crate::ring::ScalarRingRns);
impl_matrix_ops!("labrador_poly_ring", labrador_poly_ring, crate::polynomial_ring::PolyRing);

#[cfg(test)]
mod tests {
    use icicle_core::impl_matrix_ops_tests;

    mod ring {
        use super::*;
        impl_matrix_ops_tests!(crate::ring::ScalarRing);
    }
    // mod rns { // TODO: add rns matrix tests ?
    //     use super::*;
    //     impl_matrix_ops_tests!(crate::ring::ScalarRingRns);
    // }
    // mod poly_ring { // TODO: add poly ring matrix tests ?
    //     use super::*;
    //     impl_matrix_ops_tests!(crate::polynomial_ring::PolyRing);
    // }
}
