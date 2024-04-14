use crate::field::{ExtensionField, ScalarCfg, ScalarField};

use icicle_core::error::IcicleResult;
use icicle_core::ntt::{NTTConfig, NTTDir, NTTDomain, NTT};
use icicle_core::traits::IcicleResultWrap;
use icicle_core::{impl_ntt, impl_ntt_without_domain};
use icicle_cuda_runtime::device_context::DeviceContext;
use icicle_cuda_runtime::error::CudaError;
use icicle_cuda_runtime::memory::HostOrDeviceSlice;

impl_ntt!("babybear", babybear, ScalarField, ScalarCfg);
impl_ntt_without_domain!("babybearExtension", ExtensionField, ScalarField, ScalarCfg);

#[cfg(test)]
pub(crate) mod tests {
    use super::{ExtensionField, ScalarField};
    use icicle_core::{
        ntt::{initialize_domain, ntt_inplace, NTTConfig, NTTDir},
        traits::{FieldImpl, GenerateRandom},
    };
    use icicle_cuda_runtime::{device_context::DeviceContext, memory::HostSlice};
    use risc0_core::field::{
        baby_bear::{Elem, ExtElem},
        Elem as FieldElem, RootsOfUnity,
    };

    #[test]
    fn test_against_risc0() {
        let log_sizes = [15, 20];
        let ctx = DeviceContext::default();
        let risc0_rou = Elem::ROU_FWD[log_sizes[1]];
        initialize_domain(ScalarField::from([risc0_rou.as_u32()]), &ctx, false).unwrap();
        for log_size in log_sizes {
            let ntt_size = 1 << log_size;

            let mut scalars: Vec<ScalarField> = <ScalarField as FieldImpl>::Config::generate_random(ntt_size);
            let mut scalars_risc0: Vec<Elem> = scalars
                .iter()
                .map(|x| Elem::new(Into::<[u32; 1]>::into(*x)[0]))
                .collect();

            let ntt_cfg: NTTConfig<'_, ScalarField> = NTTConfig::default();
            ntt_inplace(HostSlice::from_mut_slice(&mut scalars[..]), NTTDir::kForward, &ntt_cfg).unwrap();

            risc0_zkp::core::ntt::bit_reverse(&mut scalars_risc0[..]);
            risc0_zkp::core::ntt::evaluate_ntt::<Elem, Elem>(&mut scalars_risc0[..], ntt_size);

            for (s1, s2) in scalars
                .iter()
                .zip(scalars_risc0)
            {
                assert_eq!(Into::<[u32; 1]>::into(*s1)[0], s2.as_u32());
            }

            let mut ext_scalars: Vec<ExtensionField> = <ExtensionField as FieldImpl>::Config::generate_random(ntt_size);
            let mut ext_scalars_risc0: Vec<ExtElem> = ext_scalars
                .iter()
                .map(|x| ExtElem::from_u32_words(&Into::<[u32; 4]>::into(*x)[..]))
                .collect();

            ntt_inplace(
                HostSlice::from_mut_slice(&mut ext_scalars[..]),
                NTTDir::kForward,
                &ntt_cfg,
            )
            .unwrap();

            risc0_zkp::core::ntt::bit_reverse(&mut ext_scalars_risc0[..]);
            risc0_zkp::core::ntt::evaluate_ntt::<Elem, ExtElem>(&mut ext_scalars_risc0[..], ntt_size);

            for (s1, s2) in ext_scalars
                .iter()
                .zip(ext_scalars_risc0)
            {
                assert_eq!(Into::<[u32; 4]>::into(*s1)[..], s2.to_u32_words()[..]);
            }
        }
    }
}
