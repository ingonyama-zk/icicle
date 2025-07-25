use crate::field::{ExtensionField, ScalarField};
use icicle_core::ntt::{NTTConfig, NTTDir, NTTDomain, NTTInitDomainConfig, NTT};
use icicle_core::{impl_ntt, impl_ntt_without_domain};
use icicle_runtime::{eIcicleError, memory::HostOrDeviceSlice, IcicleError};

impl_ntt!("babybear", babybear, ScalarField);
impl_ntt_without_domain!("babybear_extension", ScalarField, NTT, "_ntt", ExtensionField);

#[cfg(test)]
pub(crate) mod tests {
    use crate::field::{ExtensionField, ScalarField};
    use icicle_core::impl_ntt_tests;
    use icicle_core::ntt::tests::*;
    use serial_test::{parallel, serial};
    impl_ntt_tests!(ScalarField);

    // Tests against risc0 and plonky3
    use icicle_core::{
        ntt::{initialize_domain, ntt_inplace, release_domain, NTTConfig, NTTDir, NTTInitDomainConfig},
        traits::GenerateRandom,
    };
    use icicle_runtime::memory::IntoIcicleSliceMut;
    use risc0_core::field::{
        baby_bear::{Elem, ExtElem},
        Elem as FieldElem, RootsOfUnity,
    };

    // Note that risc0 and plonky3 tests shouldn't be ran simultaneously in parallel to other ntt tests as they use different roots of unity.
    #[test]
    #[serial]
    fn phase2_test_ntt_against_risc0() {
        test_utilities::test_load_and_init_devices();
        test_utilities::test_set_main_device();

        release_domain::<ScalarField>().unwrap(); // release domain from previous tests, if exists

        let log_sizes = [15, 20];
        let risc0_rou = Elem::ROU_FWD[log_sizes[1]];
        initialize_domain(ScalarField::from([risc0_rou.as_u32()]), &NTTInitDomainConfig::default()).unwrap();
        for log_size in log_sizes {
            let ntt_size = 1 << log_size;

            let mut scalars: Vec<ScalarField> = <ScalarField as GenerateRandom>::generate_random(ntt_size);
            let mut scalars_risc0: Vec<Elem> = scalars
                .iter()
                .map(|x| Elem::new(Into::<[u32; 1]>::into(*x)[0]))
                .collect();

            let ntt_cfg: NTTConfig<ScalarField> = NTTConfig::default();
            ntt_inplace(scalars.into_slice_mut(), NTTDir::kForward, &ntt_cfg).unwrap();

            risc0_zkp::core::ntt::bit_reverse(scalars_risc0.as_mut_slice());
            risc0_zkp::core::ntt::evaluate_ntt::<Elem, Elem>(scalars_risc0.as_mut_slice(), ntt_size);

            for (s1, s2) in scalars
                .iter()
                .zip(scalars_risc0)
            {
                assert_eq!(Into::<[u32; 1]>::into(*s1)[0], s2.as_u32());
            }

            let mut ext_scalars: Vec<ExtensionField> = <ExtensionField as GenerateRandom>::generate_random(ntt_size);
            let mut ext_scalars_risc0: Vec<ExtElem> = ext_scalars
                .iter()
                .map(|x| ExtElem::from_u32_words(&Into::<[u32; 4]>::into(*x)[..]))
                .collect();

            ntt_inplace(ext_scalars.into_slice_mut(), NTTDir::kForward, &ntt_cfg).unwrap();

            risc0_zkp::core::ntt::bit_reverse(ext_scalars_risc0.as_mut_slice());
            risc0_zkp::core::ntt::evaluate_ntt::<Elem, ExtElem>(ext_scalars_risc0.as_mut_slice(), ntt_size);

            for (s1, s2) in ext_scalars
                .iter()
                .zip(ext_scalars_risc0)
            {
                assert_eq!(Into::<[u32; 4]>::into(*s1)[..], s2.to_u32_words()[..]);
            }
        }

        release_domain::<ScalarField>().unwrap();
    }

    // TODO test from V2. For some reason importing plonky3 Babybear cause an error
    // #[test]
    // #[serial]
    // fn test_against_plonky3() {
    // }
}
