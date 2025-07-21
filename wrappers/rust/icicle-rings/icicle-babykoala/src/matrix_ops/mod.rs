use icicle_core::impl_matrix_ops;

impl_matrix_ops!("babykoala", babykoala, crate::ring::ScalarRing);
impl_matrix_ops!("babykoala_rns", babykoala_rns, crate::ring::ScalarRingRns);
impl_matrix_ops!(
    "babykoala_poly_ring",
    babykoala_poly_ring,
    crate::polynomial_ring::PolyRing
);

#[cfg(test)]
mod tests {
    use icicle_core::impl_matrix_ops_tests;

    mod ring {
        use super::*;
        impl_matrix_ops_tests!(crate::ring::ScalarRing);
    }
    // mod rns { // TODO: add rns matrix tests? ffi bindings missing
    //     use icicle_core::impl_matrix_ops_tests;
    //     impl_matrix_ops_tests!(crate::ring::ScalarRingRns);
    // }
    mod poly_ring {
        use icicle_core::impl_matrix_ops_tests;
        impl_matrix_ops_tests!(crate::polynomial_ring::PolyRing);
    }
}
