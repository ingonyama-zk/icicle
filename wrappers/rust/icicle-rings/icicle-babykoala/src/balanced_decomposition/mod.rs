use crate::polynomial_ring::PolyRing;
use icicle_core::{balanced_decomposition::BalancedDecomposition, impl_balanced_decomposition, vec_ops::VecOpsConfig};
use icicle_runtime::{eIcicleError, memory::HostOrDeviceSlice, IcicleError};

impl_balanced_decomposition!("babykoala_poly_ring", PolyRing);

#[cfg(test)]
pub(crate) mod tests {
    use crate::polynomial_ring::PolyRing;
    use icicle_core::impl_balanced_decomposition_tests;

    impl_balanced_decomposition_tests!(PolyRing);
}
