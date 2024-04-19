use ark_ff::FftField;
use icicle_cuda_runtime::memory::HostSlice;

use crate::curve::Curve;
use crate::curve::*;
use crate::{
    ecntt::*,
    ntt::{NTTDir, NttAlgorithm, Ordering},
    traits::{ArkConvertible, FieldImpl},
};

use crate::ntt::NTTConfig;

pub fn check_ecntt<C: Curve>()
where
    <<C as Curve>::ScalarField as FieldImpl>::Config: ECNTTC<C>,
{
    let test_sizes = [1 << 4];
    for test_size in test_sizes {
        // let ark_domain = GeneralEvaluationDomain::<F::ArkEquivalent>::new(test_size).unwrap();

        let points = C::generate_random_projective_points(test_size);
        // let ark_points = points
        //     .iter()
        //     .map(|v| v.to_ark())
        //     .collect::<Vec<_>>();
        // if we simply transmute arkworks types, we'll get scalars in Montgomery format
        //let points_mont = unsafe { &*(&ark_points[..] as *const _ as *const [_]) };
        //let points_mont_h = HostSlice::from_slice(points_mont.to_vec());
        let slice = &points.clone();
        let config: NTTConfig<'_, C::ScalarField> = NTTConfig::default();
        let mut out_p = vec![Projective::<C>::zero(); test_size];
        let ecntt_result = HostSlice::from_mut_slice(&mut out_p);
        let input = HostSlice::from_slice(slice);
        ecntta(input, NTTDir::kForward, &config, ecntt_result).unwrap();
        assert_ne!(ecntt_result.as_slice(), points);

        // let mut ark_ntt_result = ark_points.clone();
        // ark_domain.fft_in_place(&mut ark_ntt_result);
        // assert_ne!(ark_ntt_result, ark_points);

        // let ntt_result_as_ark =
        //     unsafe { &*(ecntt_result.as_slice() as *const _ as *const [<F as ArkConvertible>::ArkEquivalent]) };
        // assert_eq!(ark_ntt_result, ntt_result_as_ark);

        let mut slice = vec![Projective::<C>::zero(); test_size];
        let iecntt_result = HostSlice::from_mut_slice(&mut slice);
        ecntta(ecntt_result, NTTDir::kInverse, &config, iecntt_result).unwrap();

        assert_eq!(iecntt_result.as_slice(), points);
    }
}

pub fn check_ecntt_batch<C: Curve>()
where
    <C::ScalarField as FieldImpl>::Config: ECNTTC<C>,
{
    let test_sizes = [1 << 4, 1 << 10];
    let batch_sizes = [1];
    for test_size in test_sizes {
        // let coset_generators = [F::one(), F::Config::generate_random(1)[0]];
        let mut config: NTTConfig<'_, C::ScalarField> = NTTConfig::default();
        for batch_size in batch_sizes {
            let slice = &C::generate_random_projective_points(test_size * batch_size);
            let points = HostSlice::from_slice(slice);

            for is_inverse in [NTTDir::kInverse, NTTDir::kForward] {
                for ordering in [
                    Ordering::kNN, // ~same performance
                                   // Ordering::kNR,
                                   // Ordering::kRN,
                                   // Ordering::kRR,
                                   // Ordering::kNM, // no mixed radix ecntt
                                   // Ordering::kMN,
                ] {
                    config.ordering = ordering;
                    let mut slice = vec![Projective::zero(); batch_size * test_size];
                    let batch_ntt_result = HostSlice::from_mut_slice(&mut slice);
                    for alg in [NttAlgorithm::Radix2] {
                        config.batch_size = batch_size as i32;
                        config.ntt_algorithm = alg;
                        ecntta(points, is_inverse, &config, batch_ntt_result).unwrap();
                        config.batch_size = 1;
                        let mut slice = vec![Projective::zero(); test_size];
                        let one_ntt_result = HostSlice::from_mut_slice(&mut slice);
                        for i in 0..batch_size {
                            ecntta(
                                HostSlice::from_slice(
                                    &points[i * test_size..(i + 1) * test_size]
                                        .as_slice()
                                        .to_vec(),
                                ),
                                is_inverse,
                                &config,
                                one_ntt_result,
                            )
                            .unwrap();
                            assert_eq!(
                                batch_ntt_result[i * test_size..(i + 1) * test_size].as_slice(),
                                one_ntt_result.as_slice()
                            );
                        }
                    }
                }
            }
        }
    }
}
