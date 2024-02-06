use ark_ff::{FftField, Field as ArkField, One};
use ark_poly::{EvaluationDomain, GeneralEvaluationDomain};
use ark_std::{ops::Neg, test_rng, UniformRand};
use icicle_cuda_runtime::device_context::get_default_device_context;
use icicle_cuda_runtime::memory::HostOrDeviceSlice;
use icicle_cuda_runtime::stream::CudaStream;

use crate::{
    ntt::{get_default_ntt_config, initialize_domain, ntt, NTTDir, Ordering},
    traits::{ArkConvertible, FieldImpl, GenerateRandom},
};

use super::NTT;

pub fn init_domain<F: FieldImpl + ArkConvertible>(max_size: u64)
where
    F::ArkEquivalent: FftField,
    <F as FieldImpl>::Config: NTT<F>,
{
    let ctx = get_default_device_context();
    let ark_rou = F::ArkEquivalent::get_root_of_unity(max_size).unwrap();
    initialize_domain(F::from_ark(ark_rou), &ctx).unwrap();
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
    <F as FieldImpl>::Config: NTT<F> + GenerateRandom<F>,
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
        let scalars_mont_h = HostOrDeviceSlice::on_host(scalars_mont.to_vec());

        let config = get_default_ntt_config();
        let mut ntt_result = HostOrDeviceSlice::on_host(vec![F::zero(); test_size]);
        ntt(&scalars_mont_h, NTTDir::kForward, &config, &mut ntt_result).unwrap();
        assert_ne!(ntt_result.as_slice(), scalars_mont);

        let mut ark_ntt_result = ark_scalars.clone();
        ark_domain.fft_in_place(&mut ark_ntt_result);
        assert_ne!(ark_ntt_result, ark_scalars);

        let ntt_result_as_ark =
            unsafe { &*(ntt_result.as_slice() as *const _ as *const [<F as ArkConvertible>::ArkEquivalent]) };
        assert_eq!(ark_ntt_result, ntt_result_as_ark);

        let mut intt_result = HostOrDeviceSlice::on_host(vec![F::zero(); test_size]);
        ntt(&ntt_result, NTTDir::kInverse, &config, &mut intt_result).unwrap();

        assert_eq!(intt_result.as_slice(), scalars_mont);
    }
}

pub fn check_ntt_coset_from_subgroup<F: FieldImpl + ArkConvertible>()
where
    F::ArkEquivalent: FftField,
    <F as FieldImpl>::Config: NTT<F> + GenerateRandom<F>,
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
        let scalars_h = HostOrDeviceSlice::on_host(scalars.clone());
        let mut ark_scalars = scalars
            .iter()
            .map(|v| v.to_ark())
            .collect::<Vec<F::ArkEquivalent>>();

        let mut config = get_default_ntt_config();
        config.ordering = Ordering::kNR;
        let mut ntt_result_1 = HostOrDeviceSlice::on_host(vec![F::zero(); small_size]);
        let mut ntt_result_2 = HostOrDeviceSlice::on_host(vec![F::zero(); small_size]);
        ntt(&scalars_h, NTTDir::kForward, &config, &mut ntt_result_1).unwrap();
        assert_ne!(*ntt_result_1.as_slice(), scalars);
        config.coset_gen = F::from_ark(test_size_rou);
        ntt(&scalars_h, NTTDir::kForward, &config, &mut ntt_result_2).unwrap();
        let mut ntt_large_result = HostOrDeviceSlice::on_host(vec![F::zero(); test_size]);
        // back to non-coset NTT
        config.coset_gen = F::one();
        scalars.resize(test_size, F::zero());
        ntt(
            &HostOrDeviceSlice::on_host(scalars.clone()),
            NTTDir::kForward,
            &config,
            &mut ntt_large_result,
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
        let mut intt_result = HostOrDeviceSlice::on_host(vec![F::zero(); small_size]);
        ntt(&ntt_result_2, NTTDir::kInverse, &config, &mut intt_result).unwrap();
        assert_eq!(*intt_result.as_slice(), scalars[..small_size]);

        ark_small_domain.ifft_in_place(&mut ark_scalars);
        let intt_result_as_ark = intt_result
            .as_slice()
            .iter()
            .map(|p| p.to_ark())
            .collect::<Vec<F::ArkEquivalent>>();
        assert_eq!(ark_scalars[..small_size], intt_result_as_ark);
    }
}

pub fn check_ntt_arbitrary_coset<F: FieldImpl + ArkConvertible>()
where
    F::ArkEquivalent: FftField + ArkField,
    <F as FieldImpl>::Config: NTT<F> + GenerateRandom<F>,
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

            let mut scalars = HostOrDeviceSlice::on_host(F::Config::generate_random(test_size));
            // here you can see how arkworks type can be easily created without any purpose-built conversions
            let mut ark_scalars = scalars
                .as_slice()
                .iter()
                .map(|v| F::ArkEquivalent::from_random_bytes(&v.to_bytes_le()).unwrap())
                .collect::<Vec<F::ArkEquivalent>>();

            let mut config = get_default_ntt_config();
            config.ordering = Ordering::kNR;
            config.coset_gen = F::from_ark(coset_gen);
            let mut ntt_result = HostOrDeviceSlice::on_host(vec![F::zero(); test_size]);
            ntt(&scalars, NTTDir::kForward, &config, &mut ntt_result).unwrap();
            assert_ne!(scalars.as_slice(), ntt_result.as_slice());

            let ark_scalars_copy = ark_scalars.clone();
            ark_domain.fft_in_place(&mut ark_scalars);
            let ntt_result_as_ark = ntt_result
                .as_slice()
                .iter()
                .map(|p| p.to_ark())
                .collect::<Vec<F::ArkEquivalent>>();
            assert_eq!(ark_scalars, list_to_reverse_bit_order(&ntt_result_as_ark));
            ark_domain.ifft_in_place(&mut ark_scalars);
            assert_eq!(ark_scalars, ark_scalars_copy);

            config.ordering = Ordering::kRN;
            ntt(&ntt_result, NTTDir::kInverse, &config, &mut scalars).unwrap();
            let ntt_result_as_ark = scalars
                .as_slice()
                .iter()
                .map(|p| p.to_ark())
                .collect::<Vec<F::ArkEquivalent>>();
            assert_eq!(ark_scalars, ntt_result_as_ark);
        }
    }
}

pub fn check_ntt_batch<F: FieldImpl>()
where
    <F as FieldImpl>::Config: NTT<F> + GenerateRandom<F>,
{
    let test_sizes = [1 << 4, 1 << 12];
    let batch_sizes = [1, 1 << 4, 100];
    for test_size in test_sizes {
        let coset_generators = [F::one(), F::Config::generate_random(1)[0]];
        let mut config = get_default_ntt_config();
        for batch_size in batch_sizes {
            let scalars = HostOrDeviceSlice::on_host(F::Config::generate_random(test_size * batch_size));

            for coset_gen in coset_generators {
                for is_inverse in [NTTDir::kInverse, NTTDir::kForward] {
                    for ordering in [Ordering::kNN, Ordering::kNR, Ordering::kRN, Ordering::kRR] {
                        config.coset_gen = coset_gen;
                        config.ordering = ordering;
                        config.batch_size = batch_size as i32;
                        let mut batch_ntt_result = HostOrDeviceSlice::on_host(vec![F::zero(); batch_size * test_size]);
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
                }
            }
        }
    }
}

pub fn check_ntt_device_async<F: FieldImpl + ArkConvertible>()
where
    F::ArkEquivalent: FftField,
    <F as FieldImpl>::Config: NTT<F> + GenerateRandom<F>,
{
    let test_sizes = [1 << 4, 1 << 12];
    let batch_sizes = [1, 1 << 4, 100];
    for test_size in test_sizes {
        let coset_generators = [F::one(), F::Config::generate_random(1)[0]];
        let stream = CudaStream::create().unwrap();
        let mut config = get_default_ntt_config();
        for batch_size in batch_sizes {
            let scalars_h: Vec<F> = F::Config::generate_random(test_size * batch_size);
            let sum_of_coeffs: F::ArkEquivalent = scalars_h[..test_size]
                .iter()
                .map(|x| x.to_ark())
                .sum();
            let mut scalars_d = HostOrDeviceSlice::cuda_malloc_async(test_size * batch_size, &stream).unwrap();
            scalars_d
                .copy_from_host_async(&scalars_h, &stream)
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
