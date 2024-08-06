use crate::ntt::{NttAlgorithm, Ordering, CUDA_NTT_ALGORITHM, CUDA_NTT_FAST_TWIDDLES_MODE};
use crate::test_utilities;
use crate::vec_ops::{transpose_matrix, VecOps, VecOpsConfig};
use icicle_runtime::{
    memory::{DeviceVec, HostSlice},
    runtime,
    stream::IcicleStream,
};
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::{
    ntt::{
        get_root_of_unity, initialize_domain, ntt, ntt_inplace, release_domain, NTTConfig, NTTDir, NTTDomain,
        NTTInitDomainConfig, NTT,
    },
    traits::{FieldImpl, GenerateRandom},
};

pub fn init_domain<F: FieldImpl>(max_size: u64, fast_twiddles_mode: bool)
where
    <F as FieldImpl>::Config: NTTDomain<F>,
{
    let config = NTTInitDomainConfig::default();
    config
        .ext
        .set_bool(CUDA_NTT_FAST_TWIDDLES_MODE, fast_twiddles_mode);
    let rou = get_root_of_unity::<F>(max_size);
    initialize_domain(rou, &config).unwrap();
}

pub fn rel_domain<F: FieldImpl>()
where
    <F as FieldImpl>::Config: NTTDomain<F>,
{
    release_domain::<F>().unwrap()
}

// This test is comparing main and reference devices (typically CUDA and CPU) for NTT and inplace-INTT
pub fn check_ntt<F: FieldImpl>()
where
    <F as FieldImpl>::Config: NTT<F, F> + GenerateRandom<F>,
{
    let test_sizes = [1 << 4, 1 << 17];
    for test_size in test_sizes {
        let scalars: Vec<F> = F::Config::generate_random(test_size);
        let mut ntt_result_main = vec![F::zero(); test_size];
        let mut ntt_result_ref = vec![F::zero(); test_size];

        let config: NTTConfig<F> = NTTConfig::default();
        for alg in [NttAlgorithm::Radix2, NttAlgorithm::MixedRadix] {
            config
                .ext
                .set_int(CUDA_NTT_ALGORITHM, alg as i32);

            // compute NTT on main and reference devices and compare
            test_utilities::test_set_main_device();
            ntt(
                HostSlice::from_slice(&scalars),
                NTTDir::kForward,
                &config,
                HostSlice::from_mut_slice(&mut ntt_result_main),
            )
            .unwrap();

            test_utilities::test_set_ref_device();
            ntt(
                HostSlice::from_slice(&scalars),
                NTTDir::kForward,
                &config,
                HostSlice::from_mut_slice(&mut ntt_result_ref),
            )
            .unwrap();

            assert_eq!(ntt_result_main, ntt_result_ref);

            // compute INTT on main and reference devices, inplace, and compare

            test_utilities::test_set_main_device();
            ntt_inplace(
                HostSlice::from_mut_slice(&mut ntt_result_main),
                NTTDir::kForward,
                &config,
            )
            .unwrap();

            test_utilities::test_set_ref_device();
            ntt_inplace(
                HostSlice::from_mut_slice(&mut ntt_result_ref),
                NTTDir::kForward,
                &config,
            )
            .unwrap();

            assert_eq!(ntt_result_main, ntt_result_ref);
        }
    }
}

// This test is testing computation of 2N reversed NTT via 2 reversed NTTs of size N, one regular, the other on coset.
pub fn check_ntt_coset_from_subgroup<F: FieldImpl>()
where
    <F as FieldImpl>::Config: NTT<F, F> + GenerateRandom<F>,
{
    let test_sizes = [1 << 4, 1 << 16];
    for test_size in test_sizes {
        test_utilities::test_set_main_device();
        let small_size: usize = test_size >> 1;
        let test_size_rou = get_root_of_unity::<F>(test_size as u64);
        let mut scalars: Vec<F> = F::Config::generate_random(small_size);

        for alg in [NttAlgorithm::Radix2, NttAlgorithm::MixedRadix] {
            let mut config = NTTConfig::<F>::default();
            config.ordering = Ordering::kNR;
            config
                .ext
                .set_int(CUDA_NTT_ALGORITHM, alg as i32);

            // compute 2 NTTs: (1) NR ntt of size N/2 (=small_size) and (2) NR coset ntt of size N/2
            // test this is equivalent to a size N ntt
            let mut ntt_result_half = vec![F::zero(); small_size];
            let ntt_result_half = HostSlice::from_mut_slice(&mut ntt_result_half);
            let mut ntt_result_coset = vec![F::zero(); small_size];
            let ntt_result_coset = HostSlice::from_mut_slice(&mut ntt_result_coset);
            let scalars_h = HostSlice::from_slice(&scalars[..small_size]);

            ntt(scalars_h, NTTDir::kForward, &config, ntt_result_half).unwrap();
            assert_ne!(*ntt_result_half.as_slice(), scalars);
            config.coset_gen = test_size_rou;
            ntt(scalars_h, NTTDir::kForward, &config, ntt_result_coset).unwrap();

            // compare coset ntt to ref-device
            test_utilities::test_set_ref_device();
            let mut ntt_coset_ref = vec![F::zero(); small_size];
            let ntt_coset_ref = HostSlice::from_mut_slice(&mut ntt_coset_ref);
            ntt(scalars_h, NTTDir::kForward, &config, ntt_coset_ref).unwrap();
            assert_eq!(*ntt_result_coset.as_slice(), *ntt_coset_ref.as_slice());

            test_utilities::test_set_main_device();
            // compute N size NTT and compare
            let mut ntt_large_result = vec![F::zero(); test_size];
            config.coset_gen = F::one();
            scalars.resize(test_size, F::zero());
            ntt(
                HostSlice::from_slice(&scalars),
                NTTDir::kForward,
                &config,
                HostSlice::from_mut_slice(&mut ntt_large_result),
            )
            .unwrap();
            assert_eq!(*ntt_result_half.as_slice(), ntt_large_result.as_slice()[..small_size]);
            assert_eq!(*ntt_result_coset.as_slice(), ntt_large_result.as_slice()[small_size..]);

            // intt back from coset
            config.coset_gen = test_size_rou;
            config.ordering = Ordering::kRN;
            let mut intt_result = vec![F::zero(); small_size];
            ntt(
                ntt_result_coset,
                NTTDir::kInverse,
                &config,
                HostSlice::from_mut_slice(&mut intt_result),
            )
            .unwrap();
            assert_eq!(*intt_result.as_slice(), scalars[..small_size]);
        }
    }
}

// This test is interpolating a coset, given evaluations on rou, and compares main to ref device.
//TODO SHANIE - mixed radix not implemented in icicle-cpu-ntt
pub fn check_ntt_coset_interpolation_nm<F: FieldImpl>()
where
    <F as FieldImpl>::Config: NTT<F, F> + GenerateRandom<F>,
{
    let test_sizes = [1 << 9, 1 << 10, 1 << 11, 1 << 13, 1 << 14, 1 << 16];
    for test_size in test_sizes {
        let test_size_rou = get_root_of_unity::<F>((test_size << 1) as u64);
        let coset_generators = [test_size_rou, F::Config::generate_random(1)[0]];
        let scalars: Vec<F> = F::Config::generate_random(test_size);

        for coset_gen in coset_generators {
            // (1) intt from evals to coeffs
            let mut config = NTTConfig::<F>::default();
            config.ordering = Ordering::kNM;
            config
                .ext
                .set_int(CUDA_NTT_ALGORITHM, NttAlgorithm::MixedRadix as i32);

            test_utilities::test_set_main_device();
            let mut intt_result = vec![F::zero(); test_size];
            let intt_result = HostSlice::from_mut_slice(&mut intt_result);
            ntt(HostSlice::from_slice(&scalars), NTTDir::kInverse, &config, intt_result).unwrap();

            test_utilities::test_set_ref_device();
            let mut intt_result_ref = vec![F::zero(); test_size];
            let intt_result_ref = HostSlice::from_mut_slice(&mut intt_result_ref);
            ntt(
                HostSlice::from_slice(&scalars),
                NTTDir::kInverse,
                &config,
                intt_result_ref,
            )
            .unwrap();

            // (2) coset-ntt (compute coset evals)
            config.coset_gen = coset_gen;
            config.ordering = Ordering::kMN;
            test_utilities::test_set_main_device();
            let mut coset_evals = vec![F::zero(); test_size];
            ntt(
                intt_result,
                NTTDir::kForward,
                &config,
                HostSlice::from_mut_slice(&mut coset_evals),
            )
            .unwrap();

            test_utilities::test_set_ref_device();
            let mut coset_evals_ref = vec![F::zero(); test_size];
            ntt(
                intt_result_ref,
                NTTDir::kForward,
                &config,
                HostSlice::from_mut_slice(&mut coset_evals_ref),
            )
            .unwrap();

            assert_eq!(coset_evals, coset_evals_ref);
        }
    }
}

pub fn check_ntt_arbitrary_coset<F: FieldImpl>()
where
    <F as FieldImpl>::Config: NTT<F, F> + GenerateRandom<F>,
{
    let test_sizes = [1 << 4, 1 << 17];
    for test_size in test_sizes {
        let coset_generators = [
            F::Config::generate_random(1)[0],
            get_root_of_unity::<F>(test_size as u64),
            F::one(),
        ];
        for coset_gen in coset_generators {
            let mut scalars = F::Config::generate_random(test_size);
            let mut scalars_ref = scalars.clone();
            let scalars = HostSlice::from_mut_slice(&mut scalars);
            let scalars_ref = HostSlice::from_mut_slice(&mut scalars_ref);

            let mut config = NTTConfig::<F>::default();
            config.coset_gen = coset_gen;
            for alg in [NttAlgorithm::Radix2, NttAlgorithm::MixedRadix] {
                config.ordering = Ordering::kNR;
                config
                    .ext
                    .set_int(CUDA_NTT_ALGORITHM, alg as i32);

                // coset NTT for main and ref devices
                test_utilities::test_set_main_device();
                ntt_inplace(scalars, NTTDir::kForward, &config).unwrap();
                test_utilities::test_set_ref_device();
                ntt_inplace(scalars_ref, NTTDir::kForward, &config).unwrap();
                assert_eq!(*scalars.as_slice(), *scalars_ref.as_slice());

                // coset INTT for both devices
                config.ordering = Ordering::kRN;
                test_utilities::test_set_main_device();
                ntt_inplace(scalars, NTTDir::kInverse, &config).unwrap();
                test_utilities::test_set_ref_device();
                ntt_inplace(scalars_ref, NTTDir::kInverse, &config).unwrap();
                assert_eq!(*scalars.as_slice(), *scalars_ref.as_slice());
            }
        }
    }
}

// self test, comparing batch ntt to multiple single ntts
// also testing column batch with transpose against row batch
pub fn check_ntt_batch<F: FieldImpl>()
where
    <F as FieldImpl>::Config: NTT<F, F> + GenerateRandom<F>,
    <F as FieldImpl>::Config: VecOps<F>,
{
    test_utilities::test_set_main_device();
    let test_sizes = [1 << 4, 1 << 12];
    let batch_sizes = [1, 1 << 4, 100];
    for test_size in test_sizes {
        let coset_generators = [F::one(), F::Config::generate_random(1)[0]];
        let mut config = NTTConfig::<F>::default();
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
                            config
                                .ext
                                .set_int(CUDA_NTT_ALGORITHM, alg as i32);
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

                        let nof_rows = batch_size as u32;
                        let nof_cols = test_size as u32;
                        // for now, columns batching only works with MixedRadix NTT
                        config.batch_size = batch_size as i32;
                        config.columns_batch = true;
                        let mut transposed_input = vec![F::zero(); batch_size * test_size];
                        transpose_matrix(
                            scalars,
                            nof_rows,
                            nof_cols,
                            HostSlice::from_mut_slice(&mut transposed_input),
                            &VecOpsConfig::default(),
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
                            nof_cols, // inverted since it was transposed above
                            nof_rows,
                            HostSlice::from_mut_slice(&mut transposed_input),
                            &VecOpsConfig::default(),
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

pub fn check_ntt_device_async<F: FieldImpl>()
where
    <F as FieldImpl>::Config: NTT<F, F> + GenerateRandom<F>,
{
    test_utilities::test_set_main_device();
    let device_count = runtime::get_device_count().unwrap();

    (0..device_count)
        .into_par_iter()
        .for_each(move |device_id| {
            test_utilities::test_set_main_device_with_id(device_id);
            let mut stream = IcicleStream::create().unwrap();

            // if have more than one device, it will use fast-twiddles-mode (note that domain is reused per device if not released)
            init_domain::<F>(1 << 16, true /*=fast twiddles mode*/);
            // init domain per device
            let mut config = NTTConfig::<F>::default();
            let test_sizes = [1 << 4, 1 << 12];
            let batch_sizes = [1, 1 << 4, 100];
            for test_size in test_sizes {
                let coset_generators = [F::one(), F::Config::generate_random(1)[0]];

                for batch_size in batch_sizes {
                    let scalars: Vec<F> = F::Config::generate_random(test_size * batch_size);
                    let mut scalars_d = DeviceVec::<F>::device_malloc(test_size * batch_size).unwrap();
                    scalars_d
                        .copy_from_host(HostSlice::from_slice(&scalars))
                        .unwrap();

                    for coset_gen in coset_generators {
                        for ordering in [Ordering::kNN, Ordering::kRR] {
                            config.coset_gen = coset_gen;
                            config.ordering = ordering;
                            config.batch_size = batch_size as i32;
                            config.is_async = false;
                            config.stream_handle = std::ptr::null_mut();

                            // compute the ntt on ref device
                            let mut scalars_clone = scalars.clone();
                            test_utilities::test_set_ref_device();
                            ntt_inplace(HostSlice::from_mut_slice(&mut scalars_clone), NTTDir::kForward, &config)
                                .unwrap();

                            test_utilities::test_set_main_device_with_id(device_id);
                            config.is_async = true;
                            config.stream_handle = *stream;
                            for alg in [NttAlgorithm::Radix2, NttAlgorithm::MixedRadix] {
                                config
                                    .ext
                                    .set_int(CUDA_NTT_ALGORITHM, alg as i32);
                                let mut ntt_result_h = vec![F::zero(); test_size * batch_size];
                                let mut ntt_result_slice = HostSlice::from_mut_slice(&mut ntt_result_h);
                                ntt_inplace(&mut *scalars_d, NTTDir::kForward, &config).unwrap();

                                scalars_d
                                    .copy_to_host_async(ntt_result_slice, &stream)
                                    .unwrap();
                                stream
                                    .synchronize()
                                    .unwrap();
                                assert_eq!(&scalars_clone, ntt_result_slice.as_slice());

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
            stream
                .destroy()
                .unwrap();
        });
}

pub fn check_release_domain<F: FieldImpl>()
where
    <F as FieldImpl>::Config: NTTDomain<F>,
{
    test_utilities::test_set_main_device();
    rel_domain::<F>();

    test_utilities::test_set_ref_device();
    rel_domain::<F>();
}
