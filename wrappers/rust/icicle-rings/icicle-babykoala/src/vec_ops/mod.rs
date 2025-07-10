use crate::ring::{ScalarRing, ScalarRingRns};

use icicle_core::impl_vec_ops_field;
use icicle_core::vec_ops::{VecOps, VecOpsConfig};
use icicle_runtime::errors::IcicleError;
use icicle_runtime::memory::HostOrDeviceSlice;

impl_vec_ops_field!("babykoala", babykoala, ScalarRing);
impl_vec_ops_field!("babykoala_rns", babykoala_rns, ScalarRingRns);

#[cfg(test)]
pub(crate) mod tests {
    use crate::polynomial_ring::PolyRing;
    use crate::ring::{ScalarRing, ScalarRingRns};
    use icicle_core::vec_ops::tests::*;
    use icicle_core::{impl_poly_vecops_tests, impl_vec_ops_tests};

    impl_vec_ops_tests!(babykoala, ScalarRing);
    mod rns {
        use super::*;
        impl_vec_ops_tests!(babykoala_rns, ScalarRingRns);
    }

    mod poly {
        use super::*;
        impl_poly_vecops_tests!(PolyRing);
    }
}
