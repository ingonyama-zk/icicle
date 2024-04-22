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
        ntt::{initialize_domain, release_domain, ntt_inplace, NTTConfig, NTTDir},
        traits::{FieldImpl, GenerateRandom},
    };
    use icicle_cuda_runtime::{device_context::DeviceContext, memory::HostSlice};
    use p3_baby_bear::BabyBear;
    use p3_dft::{Radix2Dit, TwoAdicSubgroupDft};
    use p3_field::{
        extension::BinomialExtensionField, AbstractExtensionField, AbstractField, PrimeField32,
        TwoAdicField,
    };
    use p3_matrix::dense::RowMajorMatrix;
    use risc0_core::field::{
        baby_bear::{Elem, ExtElem},
        Elem as FieldElem, RootsOfUnity,
    };
    use serial_test::serial;

    // Note that risc0 and plonky3 tests shouldn't be ran simultaneously in parallel as they use different roots of unity
    #[test]
    #[serial]
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

        release_domain::<ScalarField>(&ctx).unwrap();
    }

    #[test]
    #[serial]
    fn test_against_plonky3() {
        let log_ncols = [15, 18];
        let nrows = 4;
        let ctx = DeviceContext::default();
        let plonky3_rou = BabyBear::two_adic_generator(log_ncols[1]);
        // To compute FFTs using icicle, we first need to initialize it using plonky3's "two adic generator"
        initialize_domain(ScalarField::from([plonky3_rou.as_canonical_u32()]), &ctx, false).unwrap();
        for log_ncol in log_ncols {
            let ntt_size = 1 << log_ncol;

            let mut scalars: Vec<ScalarField> = <ScalarField as FieldImpl>::Config::generate_random(nrows * ntt_size);
            let scalars_p3: Vec<BabyBear> = scalars
                .iter()
                .map(|x| BabyBear::from_wrapped_u32(Into::<[u32; 1]>::into(*x)[0]))
                .collect();
            let matrix_p3 = RowMajorMatrix::new(scalars_p3, nrows);

            let mut ntt_cfg: NTTConfig<'_, ScalarField> = NTTConfig::default();
            // Next two lines signalize that we want to compute `nrows` FFTs in column-ordered fashion
            ntt_cfg.batch_size = nrows as i32;
            ntt_cfg.columns_batch = true;
            ntt_inplace(HostSlice::from_mut_slice(&mut scalars[..]), NTTDir::kForward, &ntt_cfg).unwrap();

            let result_p3 = Radix2Dit.dft_batch(matrix_p3);

            for i in 0..nrows {
                for j in 0..ntt_size {
                    assert_eq!(
                        Into::<[u32; 1]>::into(scalars[i + j * nrows])[0],
                        result_p3.values[i + j * nrows].as_canonical_u32()
                    );
                }
            }

            type Plonky3Extension = BinomialExtensionField<BabyBear, 4>;

            let mut ext_scalars: Vec<ExtensionField> =
                <ExtensionField as FieldImpl>::Config::generate_random(nrows * ntt_size);
            let ext_scalars_p3: Vec<Plonky3Extension> = ext_scalars
                .iter()
                .map(|x| {
                    let arr: [u32; 4] = (*x).into();
                    Plonky3Extension::from_base_slice(
                        &(arr
                            .iter()
                            .map(|y| BabyBear::from_wrapped_u32(*y))
                            .collect::<Vec<BabyBear>>())[..],
                    )
                })
                .collect();
            let ext_matrix_p3 = RowMajorMatrix::new(ext_scalars_p3, nrows);

            ntt_inplace(
                HostSlice::from_mut_slice(&mut ext_scalars[..]),
                NTTDir::kForward,
                &ntt_cfg,
            )
            .unwrap();

            let ext_result_p3 = Radix2Dit.dft_batch(ext_matrix_p3);

            for i in 0..nrows {
                for j in 0..ntt_size {
                    let arr: [u32; 4] = ext_scalars[i + j * nrows].into();
                    let base_slice: &[BabyBear] = ext_result_p3.values[i + j * nrows].as_base_slice();
                    for k in 0..4 {
                        assert_eq!(arr[k], base_slice[k].as_canonical_u32());
                    }
                }
            }
        }

        release_domain::<ScalarField>(&ctx).unwrap();
    }
}
