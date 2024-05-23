use crate::curve::{ScalarCfg, ScalarField};

use icicle_core::error::IcicleResult;
use icicle_core::impl_poseidon2;
use icicle_core::poseidon2::{DiffusionStrategy, MdsType, Poseidon2Handle, Poseidon2Impl};
use icicle_core::traits::IcicleResultWrap;
use icicle_cuda_runtime::device_context::DeviceContext;
use icicle_cuda_runtime::error::CudaError;
use icicle_cuda_runtime::memory::HostOrDeviceSlice;

use core::mem::MaybeUninit;

impl_poseidon2!("bn254", bn254, ScalarField, ScalarCfg);

#[cfg(test)]
pub(crate) mod tests {
    use crate::curve::ScalarField;
    use icicle_core::impl_poseidon2_tests;
    use icicle_core::ntt::FieldImpl;
    use icicle_core::poseidon2::{tests::*, DiffusionStrategy, MdsType};

    impl_poseidon2_tests!(ScalarField);

    #[test]
    fn test_poseidon2_kats() {
        let kats = [
            ScalarField::from_hex("0x0bb61d24daca55eebcb1929a82650f328134334da98ea4f847f760054f4a3033"),
            ScalarField::from_hex("0x303b6f7c86d043bfcbcc80214f26a30277a15d3f74ca654992defe7ff8d03570"),
            ScalarField::from_hex("0x1ed25194542b12eef8617361c3ba7c52e660b145994427cc86296242cf766ec8"),
        ];

        let poseidon = init_poseidon::<ScalarField>(3, MdsType::Default, DiffusionStrategy::Default);
        check_poseidon_kats(3, &kats, &poseidon);
    }
}
