use icicle_core::{impl_matmul, impl_matmul_device_tests};

impl_matmul!("labrador", crate::polynomial_ring::PolyRing);
impl_matmul_device_tests!(crate::polynomial_ring::PolyRing);
