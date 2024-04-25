use ark_ff::{FftField, Field as ArkField, One};
use ark_poly::{EvaluationDomain, GeneralEvaluationDomain};
use ark_std::{ops::Neg, test_rng, UniformRand};
use icicle_cuda_runtime::device::{get_device_count, set_device};
use icicle_cuda_runtime::device_context::DeviceContext;
use icicle_cuda_runtime::memory::{DeviceVec, HostSlice};
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::error::IcicleResult;
use crate::{
    ntt::{
        initialize_domain, ntt, ntt_inplace, release_domain, NTTConfig, NTTDir, NTTDomain, NttAlgorithm, Ordering, NTT,
    },
    traits::{ArkConvertible, FieldImpl, GenerateRandom},
    vec_ops::{transpose_matrix, VecOps},
};

pub fn init_domain<F: FieldImpl + ArkConvertible>(max_size: u64, device_id: usize, fast_twiddles_mode: bool)
where
    F::ArkEquivalent: FftField,
    <F as FieldImpl>::Config: NTTDomain<F>,
{
    let ctx = DeviceContext::default_for_device(device_id);
    let ark_rou = F::ArkEquivalent::get_root_of_unity(max_size).unwrap();
    initialize_domain(F::from_ark(ark_rou), &ctx, fast_twiddles_mode).unwrap();
}

pub fn rel_domain<F: FieldImpl>(ctx: &DeviceContext) -> IcicleResult<()>
where
    <F as FieldImpl>::Config: NTTDomain<F>,
{
    release_domain::<F>(&ctx)
}

pub fn reverse_bit_order(n: u32, order: u32) -> u32 {
    fn is_power_of_two(n: u32) -> bool {
        n != 0 && n & (n - 1) == 0
    }
    assert!(is_power_of_two(order));
    let mask = order - 1;
    let binary = format!("{:0width$b}", n, width = (32 - mask.leading_zeros()) as usize);
    let reversed = binary
        .chars()
        .rev()
        .collect::<String>();
    u32::from_str_radix(&reversed, 2).unwrap()
}

pub fn list_to_reverse_bit_order<T: Copy>(l: &[T]) -> Vec<T> {
    l.iter()
        .enumerate()
        .map(|(i, _)| l[reverse_bit_order(i as u32, l.len() as u32) as usize])
        .collect()
}

pub fn check_ntt<F: FieldImpl + ArkConvertible>()
where
    F::ArkEquivalent: FftField,
    <F as FieldImpl>::Config: NTT<F, F> + GenerateRandom<F>,
{
    let test_sizes = [1 << 4, 1 << 17];
    for test_size in test_sizes {
        let ark_domain = GeneralEvaluationDomain::<F::ArkEquivalent>::new(test_size).unwrap();

        let scalars: Vec<F> = F::Config::generate_random(test_size);
        let ark_scalars = scalars
            .iter()
            .map(|v| v.to_ark())
            .collect::<Vec<F::ArkEquivalent>>();
        // if we simply transmute arkworks types, we'll get scalars in Montgomery format
        let scalars_mont = unsafe { &*(&ark_scalars[..] as *const _ as *const [F]) };
        let scalars_mont_h = HostSlice::from_slice(&scalars_mont);

        let mut config: NTTConfig<'_, F> = NTTConfig::default();
        for alg in [NttAlgorithm::Radix2, NttAlgorithm::MixedRadix] {
            config.ntt_algorithm = alg;
            let mut ntt_result = vec![F::zero(); test_size];
            let ntt_result = HostSlice::from_mut_slice(&mut ntt_result);
            ntt(scalars_mont_h, NTTDir::kForward, &config, ntt_result).unwrap();
            assert_ne!(ntt_result.as_slice(), scalars_mont);

            let mut ark_ntt_result = ark_scalars.clone();
            ark_domain.fft_in_place(&mut ark_ntt_result);
            assert_ne!(ark_ntt_result, ark_scalars);

            let ntt_result_as_ark =
                unsafe { &*(ntt_result.as_slice() as *const _ as *const [<F as ArkConvertible>::ArkEquivalent]) };
            assert_eq!(ark_ntt_result, ntt_result_as_ark);

            let mut intt_result = vec![F::zero(); test_size];
            ntt(
                ntt_result,
                NTTDir::kInverse,
                &config,
                HostSlice::from_mut_slice(&mut intt_result),
            )
            .unwrap();

            assert_eq!(intt_result.as_slice(), scalars_mont);
        }
    }
}

pub fn check_ntt_coset_from_subgroup<F: FieldImpl + ArkConvertible>()
where
    F::ArkEquivalent: FftField,
    <F as FieldImpl>::Config: NTT<F, F> + GenerateRandom<F>,
{
    let test_sizes = [1 << 4, 1 << 16];
    for test_size in test_sizes {
        let small_size = test_size >> 1;
        let test_size_rou = F::ArkEquivalent::get_root_of_unity(test_size as u64).unwrap();
        let ark_small_domain = GeneralEvaluationDomain::<F::ArkEquivalent>::new(small_size)
            .unwrap()
            .get_coset(test_size_rou)
            .unwrap();
        let ark_large_domain = GeneralEvaluationDomain::<F::ArkEquivalent>::new(test_size).unwrap();

        let mut scalars: Vec<F> = F::Config::generate_random(small_size);
        let mut ark_scalars = scalars
            .iter()
            .map(|v| v.to_ark())
            .collect::<Vec<F::ArkEquivalent>>();

        for alg in [NttAlgorithm::Radix2, NttAlgorithm::MixedRadix] {
            let mut config = NTTConfig::default();
            config.ordering = Ordering::kNR;
            config.ntt_algorithm = alg;
            let mut ntt_result_1 = vec![F::zero(); small_size];
            let mut ntt_result_2 = vec![F::zero(); small_size];
            let ntt_result_2 = HostSlice::from_mut_slice(&mut ntt_result_2);
            let scalars_h = HostSlice::from_slice(&scalars[..small_size]);
            ntt(
                scalars_h,
                NTTDir::kForward,
                &config,
                HostSlice::from_mut_slice(&mut ntt_result_1),
            )
            .unwrap();
            assert_ne!(*ntt_result_1.as_slice(), scalars);
            config.coset_gen = F::from_ark(test_size_rou);
            ntt(scalars_h, NTTDir::kForward, &config, ntt_result_2).unwrap();
            let mut ntt_large_result = vec![F::zero(); test_size];
            // back to non-coset NTT
            config.coset_gen = F::one();
            scalars.resize(test_size, F::zero());
            ntt(
                HostSlice::from_slice(&scalars),
                NTTDir::kForward,
                &config,
                HostSlice::from_mut_slice(&mut ntt_large_result),
            )
            .unwrap();
            assert_eq!(*ntt_result_1.as_slice(), ntt_large_result.as_slice()[..small_size]);
            assert_eq!(*ntt_result_2.as_slice(), ntt_large_result.as_slice()[small_size..]);

            let mut ark_large_scalars = ark_scalars.clone();
            ark_small_domain.fft_in_place(&mut ark_scalars);
            let ntt_result_as_ark = ntt_large_result
                .as_slice()
                .iter()
                .map(|p| p.to_ark())
                .collect::<Vec<F::ArkEquivalent>>();
            assert_eq!(
                ark_scalars[..small_size],
                list_to_reverse_bit_order(&ntt_result_as_ark[small_size..])
            );
            ark_large_domain.fft_in_place(&mut ark_large_scalars);
            assert_eq!(ark_large_scalars, list_to_reverse_bit_order(&ntt_result_as_ark));

            config.coset_gen = F::from_ark(test_size_rou);
            config.ordering = Ordering::kRN;
            let mut intt_result = vec![F::zero(); small_size];
            ntt(
                ntt_result_2,
                NTTDir::kInverse,
                &config,
                HostSlice::from_mut_slice(&mut intt_result),
            )
            .unwrap();
            assert_eq!(*intt_result.as_slice(), scalars[..small_size]);

            ark_small_domain.ifft_in_place(&mut ark_scalars);
            let intt_result_as_ark = intt_result
                .iter()
                .map(|p| p.to_ark())
                .collect::<Vec<F::ArkEquivalent>>();
            assert_eq!(ark_scalars[..small_size], intt_result_as_ark);
        }
    }
}

pub fn check_ntt_coset_nm<F: FieldImpl + ArkConvertible>()
where
    F::ArkEquivalent: FftField,
    <F as FieldImpl>::Config: NTT<F, F> + GenerateRandom<F>,
{
    let test_sizes = [1 << 9, 1 << 10, 1 << 11, 1 << 13, 1 << 14, 1 << 16];
    for test_size in test_sizes {
        let test_size_rou = F::ArkEquivalent::get_root_of_unity((test_size << 1) as u64).unwrap();
        let coset_generators = [F::from_ark(test_size_rou), F::Config::generate_random(1)[0]];

        let scalars: Vec<F> = F::Config::generate_random(test_size);
        let mut ark_scalars = scalars
            .iter()
            .map(|v| v.to_ark())
            .collect::<Vec<F::ArkEquivalent>>();
        let ark_domain = GeneralEvaluationDomain::<F::ArkEquivalent>::new(test_size).unwrap();

        for coset_gen in coset_generators {
            let mut config = NTTConfig::default();
            config.ordering = Ordering::kNM;
            config.ntt_algorithm = NttAlgorithm::MixedRadix;
            config.coset_gen = coset_gen;

            let mut intt_result = vec![F::zero(); test_size];
            let intt_result = HostSlice::from_mut_slice(&mut intt_result);
            ntt(HostSlice::from_slice(&scalars), NTTDir::kInverse, &config, intt_result).unwrap();

            // Compare sum of coeffs to ark since don't know how it is mixed
            let ark_coset_domain = ark_domain
                .get_coset(coset_gen.to_ark())
                .unwrap();
            ark_coset_domain.ifft_in_place(&mut ark_scalars);
            let ark_from_coset_sum = ark_scalars
                .iter()
                .fold(F::zero().to_ark(), |acc, x| acc + x);
            let icicle_from_coset_sum = intt_result
                .iter()
                .fold(F::zero().to_ark(), |acc, x| acc + x.to_ark());
            assert_eq!(ark_from_coset_sum, icicle_from_coset_sum);

            config.ordering = Ordering::kMN;
            let mut ntt_result = vec![F::zero(); test_size];
            ntt(
                intt_result,
                NTTDir::kForward,
                &config,
                HostSlice::from_mut_slice(&mut ntt_result),
            )
            .unwrap();
            ark_coset_domain.fft_in_place(&mut ark_scalars); // to reuse in next iteration
            assert_eq!(ntt_result, scalars);
        }
    }
}

pub fn check_ntt_arbitrary_coset<F: FieldImpl + ArkConvertible>()
where
    F::ArkEquivalent: FftField + ArkField,
    <F as FieldImpl>::Config: NTT<F, F> + GenerateRandom<F>,
{
    let mut seed = test_rng();
    let test_sizes = [1 << 4, 1 << 17];
    for test_size in test_sizes {
        let coset_generators = [
            F::ArkEquivalent::rand(&mut seed),
            F::ArkEquivalent::neg(F::ArkEquivalent::one()),
            F::ArkEquivalent::get_root_of_unity(test_size as u64).unwrap(),
        ];
        for coset_gen in coset_generators {
            let ark_domain = GeneralEvaluationDomain::<F::ArkEquivalent>::new(test_size)
                .unwrap()
                .get_coset(coset_gen)
                .unwrap();

            let mut scalars = F::Config::generate_random(test_size);
            let scalars = HostSlice::from_mut_slice(&mut scalars);
            // here you can see how arkworks type can be easily created without any purpose-built conversions
            let mut ark_scalars = scalars
                .iter()
                .map(|v| F::ArkEquivalent::from_random_bytes(&v.to_bytes_le()).unwrap())
                .collect::<Vec<F::ArkEquivalent>>();

            let mut config = NTTConfig::default();
            config.coset_gen = F::from_ark(coset_gen);
            for alg in [NttAlgorithm::Radix2, NttAlgorithm::MixedRadix] {
                config.ordering = Ordering::kNR;
                config.ntt_algorithm = alg;
                ntt_inplace(scalars, NTTDir::kForward, &config).unwrap();

                let ark_scalars_copy = ark_scalars.clone();
                ark_domain.fft_in_place(&mut ark_scalars);
                let ntt_result_as_ark = scalars
                    .as_slice()
                    .iter()
                    .map(|p| p.to_ark())
                    .collect::<Vec<F::ArkEquivalent>>();
                assert_eq!(ark_scalars, list_to_reverse_bit_order(&ntt_result_as_ark));
                ark_domain.ifft_in_place(&mut ark_scalars);
                assert_eq!(ark_scalars, ark_scalars_copy);

                config.ordering = Ordering::kRN;
                ntt_inplace(scalars, NTTDir::kInverse, &config).unwrap();
                let ntt_result_as_ark = scalars
                    .iter()
                    .map(|p| p.to_ark())
                    .collect::<Vec<F::ArkEquivalent>>();
                assert_eq!(ark_scalars, ntt_result_as_ark);
            }
        }
    }
}

pub fn check_ntt_batch<F: FieldImpl>()
where
    <F as FieldImpl>::Config: NTT<F, F> + GenerateRandom<F>,
    <F as FieldImpl>::Config: VecOps<F>,
{
    let test_sizes = [1 << 4, 1 << 12];
    let batch_sizes = [1, 1 << 4, 100];
    for test_size in test_sizes {
        let coset_generators = [F::one(), F::Config::generate_random(1)[0]];
        let mut config = NTTConfig::default();
        for batch_size in batch_sizes {
            let scalars = F::Config::generate_random(test_size * batch_size);
            let scalars = HostSlice::from_slice(&scalars);

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
                        let mut batch_ntt_result = vec![F::zero(); batch_size * test_size];
                        for alg in [NttAlgorithm::Radix2, NttAlgorithm::MixedRadix] {
                            config.batch_size = batch_size as i32;
                            config.ntt_algorithm = alg;
                            ntt(
                                scalars,
                                is_inverse,
                                &config,
                                HostSlice::from_mut_slice(&mut batch_ntt_result),
                            )
                            .unwrap();
                            config.batch_size = 1;
                            let mut one_ntt_result = vec![F::one(); test_size];
                            for i in 0..batch_size {
                                ntt(
                                    &scalars[i * test_size..(i + 1) * test_size],
                                    is_inverse,
                                    &config,
                                    HostSlice::from_mut_slice(&mut one_ntt_result),
                                )
                                .unwrap();
                                assert_eq!(
                                    batch_ntt_result[i * test_size..(i + 1) * test_size],
                                    *one_ntt_result.as_slice()
                                );
                            }
                        }

                        let row_size = test_size as u32;
                        let column_size = batch_size as u32;
                        let on_device = false;
                        let is_async = false;
                        // for now, columns batching only works with MixedRadix NTT
                        config.batch_size = batch_size as i32;
                        config.columns_batch = true;
                        let mut transposed_input = vec![F::zero(); batch_size * test_size];
                        transpose_matrix(
                            scalars,
                            row_size,
                            column_size,
                            HostSlice::from_mut_slice(&mut transposed_input),
                            &config.ctx,
                            on_device,
                            is_async,
                        )
                        .unwrap();
                        let mut col_batch_ntt_result = vec![F::zero(); batch_size * test_size];
                        ntt(
                            HostSlice::from_slice(&transposed_input),
                            is_inverse,
                            &config,
                            HostSlice::from_mut_slice(&mut col_batch_ntt_result),
                        )
                        .unwrap();
                        transpose_matrix(
                            HostSlice::from_slice(&col_batch_ntt_result),
                            column_size,
                            row_size,
                            HostSlice::from_mut_slice(&mut transposed_input),
                            &config.ctx,
                            on_device,
                            is_async,
                        )
                        .unwrap();
                        assert_eq!(batch_ntt_result[..], *transposed_input.as_slice());
                        config.columns_batch = false;
                    }
                }
            }
        }
    }
}

pub fn check_ntt_device_async<F: FieldImpl + ArkConvertible>()
where
    F::ArkEquivalent: FftField,
    <F as FieldImpl>::Config: NTT<F, F> + GenerateRandom<F>,
{
    let device_count = get_device_count().unwrap();

    (0..device_count)
        .into_par_iter()
        .for_each(move |device_id| {
            set_device(device_id).unwrap();
            // if have more than one device, it will use fast-twiddles-mode (note that domain is reused per device if not released)
            init_domain::<F>(1 << 16, device_id, true /*=fast twiddles mode*/); // init domain per device
            let mut config: NTTConfig<'static, F> = NTTConfig::default_for_device(device_id);
            let test_sizes = [1 << 4, 1 << 12];
            let batch_sizes = [1, 1 << 4, 100];
            for test_size in test_sizes {
                let coset_generators = [F::one(), F::Config::generate_random(1)[0]];
                let stream = config
                    .ctx
                    .stream;
                for batch_size in batch_sizes {
                    let scalars: Vec<F> = F::Config::generate_random(test_size * batch_size);
                    let sum_of_coeffs: F::ArkEquivalent = scalars[..test_size]
                        .iter()
                        .map(|x| x.to_ark())
                        .sum();
                    let mut scalars_d = DeviceVec::<F>::cuda_malloc(test_size * batch_size).unwrap();
                    scalars_d
                        .copy_from_host(HostSlice::from_slice(&scalars))
                        .unwrap();

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
                                let mut ntt_result_h = vec![F::zero(); test_size * batch_size];
                                let mut ntt_result_slice = HostSlice::from_mut_slice(&mut ntt_result_h);
                                ntt_inplace(&mut *scalars_d, NTTDir::kForward, &config).unwrap();
                                if coset_gen == F::one() {
                                    scalars_d
                                        .copy_to_host(ntt_result_slice)
                                        .unwrap();
                                    assert_eq!(sum_of_coeffs, ntt_result_slice[0].to_ark());
                                }
                                ntt_inplace(&mut *scalars_d, NTTDir::kInverse, &config).unwrap();
                                scalars_d
                                    .copy_to_host_async(&mut ntt_result_slice, &stream)
                                    .unwrap();
                                stream
                                    .synchronize()
                                    .unwrap();
                                assert_eq!(scalars, *ntt_result_h.as_slice());
                            }
                        }
                    }
                }
            }
        });
}

pub fn check_release_domain<F: FieldImpl>()
where
    <F as FieldImpl>::Config: NTTDomain<F>,
{
    let config: NTTConfig<'static, F> = NTTConfig::default();
    let err = rel_domain::<F>(&config.ctx);
    assert!(err.is_ok())
}
