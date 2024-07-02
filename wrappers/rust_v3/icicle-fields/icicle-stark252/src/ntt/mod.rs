use crate::field::{ScalarCfg, ScalarField};
use icicle_core::ntt::{NTTConfig, NTTDir, NTTDomain, NTTInitDomainConfig, NTT};
use icicle_core::{impl_ntt, impl_ntt_without_domain};
use icicle_runtime::errors::eIcicleError;
use icicle_runtime::memory::HostOrDeviceSlice;

impl_ntt!("stark252", stark252, ScalarField, ScalarCfg);

#[cfg(test)]
pub(crate) mod tests {
    use crate::field::ScalarField;
    use icicle_core::impl_ntt_tests;
    use icicle_core::ntt::tests::*;
    use serial_test::{parallel, serial};

    impl_ntt_tests!(ScalarField);


    use icicle_core::{
        ntt::{initialize_domain, ntt_inplace, release_domain, NTTConfig, NTTInitDomainConfig, NTTDir},
        traits::{FieldImpl, GenerateRandom},
    };
    use icicle_runtime::memory::HostSlice;
    use lambdaworks_math::{
        field::{
            element::FieldElement, fields::fft_friendly::stark_252_prime_field::Stark252PrimeField, traits::IsFFTField,
        },
        polynomial::Polynomial,
        traits::ByteConversion,
    };

    pub type FE = FieldElement<Stark252PrimeField>;

    #[test]
    #[serial]
    fn phase2_test_ntt_against_lambdaworks() {
        test_utilities::test_load_and_init_devices();
        test_utilities::test_set_main_device();

        release_domain::<ScalarField>().unwrap(); // release domain from previous tests, if exists

        let log_sizes = [15, 20];        
        let lw_root_of_unity = Stark252PrimeField::get_primitive_root_of_unity(log_sizes[log_sizes.len() - 1]).unwrap();
        initialize_domain(ScalarField::from_bytes_le(&lw_root_of_unity.to_bytes_le()), &NTTInitDomainConfig::default()).unwrap();
        for log_size in log_sizes {
            let ntt_size = 1 << log_size;

            let mut scalars: Vec<ScalarField> = <ScalarField as FieldImpl>::Config::generate_random(ntt_size);
            let scalars_lw: Vec<FE> = scalars
                .iter()
                .map(|x| FieldElement::from_bytes_le(&x.to_bytes_le()).unwrap())
                .collect();

            let ntt_cfg: NTTConfig<ScalarField> = NTTConfig::default();
            ntt_inplace(HostSlice::from_mut_slice(&mut scalars[..]), NTTDir::kForward, &ntt_cfg).unwrap();

            let poly = Polynomial::new(&scalars_lw[..]);
            let evaluations = Polynomial::evaluate_fft::<Stark252PrimeField>(&poly, 1, None).unwrap();

            for (s1, s2) in scalars
                .iter()
                .zip(evaluations.iter())
            {
                assert_eq!(s1.to_bytes_le(), s2.to_bytes_le());
            }
        }
        release_domain::<ScalarField>().unwrap();
    }
}
