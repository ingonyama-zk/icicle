use icicle_core::{impl_matmul, impl_matmul_device_tests, impl_matrix_transpose, impl_matrix_transpose_device_tests};

impl_matmul!("labrador_poly_ring", crate::polynomial_ring::PolyRing);
impl_matmul_device_tests!(crate::polynomial_ring::PolyRing);

impl_matrix_transpose!("labrador_poly_ring", crate::polynomial_ring::PolyRing);
impl_matrix_transpose_device_tests!(crate::polynomial_ring::PolyRing);
