use icicle_core::impl_matrix_ops;

impl_matrix_ops!("labrador_poly_ring", crate::polynomial_ring::PolyRing);

#[cfg(test)]
mod test_polyring_matrix_ops {
    use icicle_core::impl_matrix_ops_tests;

    impl_matrix_ops_tests!(test_polyring_matrix_ops, crate::polynomial_ring::PolyRing);
}
