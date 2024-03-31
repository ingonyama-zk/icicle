#[cfg(feature = "arkworks")]
use ark_ec::models::CurveConfig as ArkCurveConfig;
use ark_ff::FftField;
use ark_poly::{EvaluationDomain, GeneralEvaluationDomain};
use icicle_cuda_runtime::device::{get_device_count, set_device};
use icicle_cuda_runtime::memory::HostOrDeviceSlice;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::curve::Curve;
use crate::curve::*;
use crate::ecntt::ecntt;
use crate::ntt::tests::{init_domain, transpose_flattened_matrix};
use crate::{
    ntt::{ntt, NTTDir, NttAlgorithm, Ordering},
    traits::{ArkConvertible, FieldImpl, GenerateRandom},
};

use crate::ntt::{NTTConfig, NTT};

use super::ECNTT;

pub fn check_ecntt<F: FieldImpl + ArkConvertible, B: FieldImpl + ArkConvertible, C: Curve>()
where
    F::ArkEquivalent: FftField,
    <C::BaseField as FieldImpl>::Config: ECNTT<C>,
    C::BaseField: ArkConvertible<ArkEquivalent = <C::ArkSWConfig as ArkCurveConfig>::BaseField>,
{
    let test_sizes = [1 << 4, 1 << 17];
    for test_size in test_sizes {
        // let ark_domain = GeneralEvaluationDomain::<F::ArkEquivalent>::new(test_size).unwrap();

        let points = C::generate_random_projective_points(test_size);
        let ark_points = points
            .iter()
            .map(|v| v.to_ark())
            .collect::<Vec<_>>();
        // if we simply transmute arkworks types, we'll get scalars in Montgomery format
        let points_mont = unsafe { &*(&ark_points[..] as *const _ as *const [_]) };
        let points_mont_h = HostOrDeviceSlice::on_host(points_mont.to_vec());

        let config = NTTConfig::default();

        let mut ecntt_result = HostOrDeviceSlice::on_host(vec![Projective::<C>::zero(); test_size]);
        ecntt(&points_mont_h, NTTDir::kForward, &config, &mut ecntt_result).unwrap();
        assert_ne!(ecntt_result.as_slice(), points_mont);

        // let mut ark_ntt_result = ark_points.clone();
        // ark_domain.fft_in_place(&mut ark_ntt_result);
        // assert_ne!(ark_ntt_result, ark_points);

        // let ntt_result_as_ark =
        //     unsafe { &*(ecntt_result.as_slice() as *const _ as *const [<F as ArkConvertible>::ArkEquivalent]) };
        // assert_eq!(ark_ntt_result, ntt_result_as_ark);

        let mut intt_result = HostOrDeviceSlice::on_host(vec![Projective::<C>::zero(); test_size]);
        ecntt(&ecntt_result, NTTDir::kInverse, &config, &mut intt_result).unwrap();

        assert_eq!(intt_result.as_slice(), points_mont);
    }
}

pub fn check_ecntt_batch<F: FieldImpl>()
where
    <F as FieldImpl>::Config: NTT<F> + GenerateRandom<F>,
{
    let test_sizes = [1 << 4, 1 << 12];
    let batch_sizes = [1, 1 << 4, 100];
    for test_size in test_sizes {
        let coset_generators = [F::one(), F::Config::generate_random(1)[0]];
        let mut config = NTTConfig::default();
        for batch_size in batch_sizes {
            let scalars = HostOrDeviceSlice::on_host(F::Config::generate_random(test_size * batch_size));

            for coset_gen in coset_generators {
                for is_inverse in [NTTDir::kInverse, NTTDir::kForward] {
                    for ordering in [
                        Ordering::kNN,
                        Ordering::kNR,
                        Ordering::kRN,
                        Ordering::kRR,
                        Ordering::kNM,
                        Ordering::kMN,
                    ] {
                        config.coset_gen = coset_gen;
                        config.ordering = ordering;
                        let mut batch_ntt_result = HostOrDeviceSlice::on_host(vec![F::zero(); batch_size * test_size]);
                        for alg in [NttAlgorithm::Radix2, NttAlgorithm::MixedRadix] {
                            config.batch_size = batch_size as i32;
                            config.ntt_algorithm = alg;
                            ntt(&scalars, is_inverse, &config, &mut batch_ntt_result).unwrap();
                            config.batch_size = 1;
                            let mut one_ntt_result = HostOrDeviceSlice::on_host(vec![F::one(); test_size]);
                            for i in 0..batch_size {
                                ntt(
                                    &HostOrDeviceSlice::on_host(scalars[i * test_size..(i + 1) * test_size].to_vec()),
                                    is_inverse,
                                    &config,
                                    &mut one_ntt_result,
                                )
                                .unwrap();
                                assert_eq!(
                                    batch_ntt_result[i * test_size..(i + 1) * test_size],
                                    *one_ntt_result.as_slice()
                                );
                            }
                        }

                        // for now, columns batching only works with MixedRadix NTT
                        config.batch_size = batch_size as i32;
                        config.columns_batch = true;
                        let transposed_input =
                            HostOrDeviceSlice::on_host(transpose_flattened_matrix(&scalars[..], batch_size));
                        let mut col_batch_ntt_result =
                            HostOrDeviceSlice::on_host(vec![F::zero(); batch_size * test_size]);
                        ntt(&transposed_input, is_inverse, &config, &mut col_batch_ntt_result).unwrap();
                        assert_eq!(
                            batch_ntt_result[..],
                            transpose_flattened_matrix(&col_batch_ntt_result[..], test_size)
                        );
                        config.columns_batch = false;
                    }
                }
            }
        }
    }
}

pub fn check_ecntt_device_async<F: FieldImpl + ArkConvertible>()
where
    F::ArkEquivalent: FftField,
    <F as FieldImpl>::Config: NTT<F> + GenerateRandom<F>,
{
    let device_count = get_device_count().unwrap();

    (0..device_count)
        .into_par_iter()
        .for_each(move |device_id| {
            set_device(device_id).unwrap();
            // if have more than one device, it will use fast-twiddles-mode (note that domain is reused per device if not released)
            init_domain::<F>(1 << 16, device_id, true /*=fast twiddles mode*/); // init domain per device
            let test_sizes = [1 << 4, 1 << 12];
            let batch_sizes = [1, 1 << 4, 100];
            for test_size in test_sizes {
                let coset_generators = [F::one(), F::Config::generate_random(1)[0]];
                let mut config = NTTConfig::default_for_device(device_id);
                let stream = config
                    .ctx
                    .stream;
                for batch_size in batch_sizes {
                    let scalars_h: Vec<F> = F::Config::generate_random(test_size * batch_size);
                    let sum_of_coeffs: F::ArkEquivalent = scalars_h[..test_size]
                        .iter()
                        .map(|x| x.to_ark())
                        .sum();
                    let mut scalars_d = HostOrDeviceSlice::cuda_malloc(test_size * batch_size).unwrap();
                    scalars_d
                        .copy_from_host(&scalars_h)
                        .unwrap();
                    let mut ntt_out_d = HostOrDeviceSlice::cuda_malloc_async(test_size * batch_size, &stream).unwrap();

                    for coset_gen in coset_generators {
                        for ordering in [Ordering::kNN, Ordering::kRR] {
                            config.coset_gen = coset_gen;
                            config.ordering = ordering;
                            config.batch_size = batch_size as i32;
                            config.is_async = true;
                            config
                                .ctx
                                .stream = &stream;
                            for alg in [NttAlgorithm::Radix2, NttAlgorithm::MixedRadix] {
                                config.ntt_algorithm = alg;
                                ntt(&scalars_d, NTTDir::kForward, &config, &mut ntt_out_d).unwrap();
                                ntt(&ntt_out_d, NTTDir::kInverse, &config, &mut scalars_d).unwrap();
                                let mut intt_result_h = vec![F::zero(); test_size * batch_size];
                                scalars_d
                                    .copy_to_host_async(&mut intt_result_h, &stream)
                                    .unwrap();
                                stream
                                    .synchronize()
                                    .unwrap();
                                assert_eq!(scalars_h, intt_result_h);
                                if coset_gen == F::one() {
                                    let mut ntt_result_h = vec![F::zero(); test_size * batch_size];
                                    ntt_out_d
                                        .copy_to_host(&mut ntt_result_h)
                                        .unwrap();
                                    assert_eq!(sum_of_coeffs, ntt_result_h[0].to_ark());
                                }
                            }
                        }
                    }
                }
            }
        });
}
