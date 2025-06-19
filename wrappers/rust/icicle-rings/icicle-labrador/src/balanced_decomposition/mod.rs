use crate::polynomial_ring::PolyRing;
use icicle_core::{impl_balanced_decomposition, vec_ops::VecOpsConfig};
use icicle_runtime::errors::eIcicleError;
use icicle_runtime::memory::HostOrDeviceSlice;

impl_balanced_decomposition!("labrador_poly_ring", PolyRing);

#[cfg(test)]
pub(crate) mod tests {
    use crate::polynomial_ring::PolyRing;
    use icicle_core::impl_balanced_decomposition_tests;

    impl_balanced_decomposition_tests!(PolyRing);
}
