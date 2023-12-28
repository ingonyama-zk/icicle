use crate::curve::{ScalarCfg, ScalarField};

use icicle_core::{error::*, impl_ntt, ntt::*, traits::*};

use icicle_cuda_runtime::{device_context::DeviceContext, error::CudaError};

impl_ntt!("bls12_377", ScalarField, ScalarCfg);

#[cfg(test)]
pub(crate) mod tests {
    use crate::curve::{ScalarCfg, ScalarField};
    use icicle_core::impl_ntt_tests;
    use icicle_core::ntt::tests::{check_ntt, check_ntt_coset_from_subgroup};

    impl_ntt_tests!(ScalarField, ScalarCfg);
}
