use crate::curve::{ScalarCfg, ScalarField};

use icicle_core::impl_ntt;
use icicle_core::ntt::{NTTConfig, NTT};
use icicle_cuda_runtime::device_context::DeviceContext;
use icicle_cuda_runtime::error::{CudaError, CudaResult, CudaResultWrap};

impl_ntt!("bn254", ScalarField, ScalarCfg);

#[cfg(test)]
pub(crate) mod tests {
    use icicle_core::traits::{ArkConvertible, FieldImpl};
    use icicle_core::{
        impl_ntt_tests,
        ntt::{Ordering, NTT},
    };
    use icicle_cuda_runtime::device_context::get_default_device_context;

    use crate::curve::{generate_random_scalars, ScalarCfg, ScalarField};

    use ark_bn254::Fr;
    use ark_ff::FftField;
    use ark_poly::{EvaluationDomain, GeneralEvaluationDomain};

    impl_ntt_tests!(ScalarField, ScalarCfg);
}
