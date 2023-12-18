use crate::curve::{ScalarCfg, ScalarField};

use icicle_core::impl_ntt;
use icicle_core::ntt::{NTTConfig, NTT};
use icicle_cuda_runtime::device_context::DeviceContext;
use icicle_cuda_runtime::error::{CudaError, CudaResult, CudaResultWrap};

impl_ntt!("bn254", ScalarField, ScalarCfg);

#[cfg(test)]
pub(crate) mod tests {
    use crate::curve::{ScalarCfg, ScalarField};
    use icicle_core::impl_ntt_tests;
    use icicle_core::ntt::tests::{check_ntt, check_ntt_coset_from_subgroup};

    impl_ntt_tests!(ScalarField, ScalarCfg);
}
