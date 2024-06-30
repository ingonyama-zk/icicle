use crate::field::{ScalarCfg, ScalarField};

use icicle_core::error::IcicleResult;
use icicle_core::ntt::{NTTConfig, NTTDir, NTTDomain, NTT};
use icicle_core::traits::IcicleResultWrap;
use icicle_core::{impl_ntt, impl_ntt_without_domain};
use icicle_cuda_runtime::device_context::DeviceContext;
use icicle_cuda_runtime::error::CudaError;
use icicle_cuda_runtime::memory::HostOrDeviceSlice;

impl_ntt!("stark252", stark252, ScalarField, ScalarCfg);

#[cfg(test)]
pub(crate) mod tests {
    use super::ScalarField;
    use icicle_core::{
        ntt::{initialize_domain, ntt_inplace, NTTConfig, NTTDir},
        traits::{FieldImpl, GenerateRandom},
    };
    use icicle_cuda_runtime::{device_context::DeviceContext, memory::HostSlice};
    use lambdaworks_math::{
        field::{
            element::FieldElement, fields::fft_friendly::stark_252_prime_field::Stark252PrimeField, traits::IsFFTField,
        },
        polynomial::Polynomial,
        traits::ByteConversion,
    };

    pub type FE = FieldElement<Stark252PrimeField>;

    #[test]
    fn test_against_lambdaworks() {
        let log_sizes = [15, 20];
        let ctx = DeviceContext::default();
        let lw_root_of_unity = Stark252PrimeField::get_primitive_root_of_unity(log_sizes[log_sizes.len() - 1]).unwrap();
        initialize_domain(ScalarField::from_bytes_le(&lw_root_of_unity.to_bytes_le()), &ctx, false).unwrap();
        for log_size in log_sizes {
            let ntt_size = 1 << log_size;

            let mut scalars: Vec<ScalarField> = <ScalarField as FieldImpl>::Config::generate_random(ntt_size);
            let scalars_lw: Vec<FE> = scalars
                .iter()
                .map(|x| FieldElement::from_bytes_le(&x.to_bytes_le()).unwrap())
                .collect();

            let ntt_cfg: NTTConfig<'_, ScalarField> = NTTConfig::default();
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
    }
}
