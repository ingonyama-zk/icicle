use crate::ring::{ScalarRing, ScalarRingRns};

use icicle_core::impl_vec_ops_field;
use icicle_core::vec_ops::{VecOps, VecOpsConfig};
use icicle_runtime::errors::eIcicleError;
use icicle_runtime::memory::HostOrDeviceSlice;

use icicle_core::traits::PrimeField;


impl_vec_ops_field!("labrador", labrador, ScalarRing);
impl_vec_ops_field!("labrador_rns", labrador_rns, ScalarRingRns);

#[cfg(test)]
pub(crate) mod tests {
    use crate::polynomial_ring::PolyRing;
    use icicle_core::vec_ops::tests::*;
    use icicle_core::{impl_poly_vecops_tests, impl_vec_ops_tests};

    impl_vec_ops_tests!(labrador, ScalarRing);
    mod rns {
        use super::*;
        impl_vec_ops_tests!(labrador_rns, ScalarRingRns);
    }

    mod poly {
        use super::*;
        impl_poly_vecops_tests!(PolyRing);
    }
}
