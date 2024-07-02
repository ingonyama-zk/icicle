use crate::curve::{Affine, Curve, Projective};
use crate::msm::{msm, precompute_bases, MSMConfig, LARGE_BUCKET_FACTOR, MSM};
use crate::test_utilities;
use crate::traits::{FieldImpl, GenerateRandom, MontgomeryConvertible};
use icicle_runtime::{
    memory::{DeviceVec, HostSlice},
    runtime,
    stream::IcicleStream,
};
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use rand::thread_rng;
use rand::Rng;

pub fn generate_random_affine_points_with_zeroes<C: Curve>(size: usize, num_zeroes: usize) -> Vec<Affine<C>> {
    let mut rng = thread_rng();
    let mut points = C::generate_random_affine_points(size);
    for _ in 0..num_zeroes {
        points[rng.gen_range(0..size)] = Affine::<C>::zero();
    }
    points
}

pub fn check_msm<C: Curve + MSM<C>>()
where
    <C::ScalarField as FieldImpl>::Config: GenerateRandom<C::ScalarField>,
    C::ScalarField: MontgomeryConvertible,
{
    test_utilities::test_set_main_device();
    let device_count = runtime::get_device_count().unwrap();
    (0..device_count) // TODO: this is proto-loadbalancer
        .into_par_iter()
        .for_each(move |device_id| {
            //TODO: currently supported multi-GPU workflow:
            //      1) User starts child host thread from parent host thread
            //      2) Calls set_device once with selected device_id (0 is default device .. < device_count)
            //      3) Perform all operations (without changing device on the thread)
            //      4) If necessary - export results to parent host thread

            let test_sizes = [4, 8, 16, 32, 64, 128, 256, 1000, 1 << 18];
            test_utilities::test_set_main_device_with_id(device_id);
            let mut stream = IcicleStream::create().unwrap();
            let mut msm_results = DeviceVec::<Projective<C>>::device_malloc_async(1, &stream).unwrap();
            for test_size in test_sizes {
                let points = generate_random_affine_points_with_zeroes::<C>(test_size, 2);
                let scalars = <C::ScalarField as FieldImpl>::Config::generate_random(test_size);

                // (1) async msm on main device
                test_utilities::test_set_main_device_with_id(device_id);
                let mut scalars_d = DeviceVec::<C::ScalarField>::device_malloc_async(test_size, &stream).unwrap();
                scalars_d
                    .copy_from_host_async(HostSlice::from_slice(&scalars), &stream)
                    .unwrap();

                let mut cfg = MSMConfig::default();
                cfg.stream_handle = *stream;
                cfg.is_async = true;
                msm(
                    &scalars_d[..],
                    HostSlice::from_slice(&points),
                    &cfg,
                    &mut msm_results[..],
                )
                .unwrap();

                let mut msm_host_result = vec![Projective::<C>::zero(); 1];
                msm_results
                    .copy_to_host(HostSlice::from_mut_slice(&mut msm_host_result[..]))
                    .unwrap();
                stream
                    .synchronize()
                    .unwrap();

                // (2) compute on ref device and compare
                test_utilities::test_set_main_device_with_id(device_id);
                let mut ref_msm_host_result = vec![Projective::<C>::zero(); 1];
                msm(
                    HostSlice::from_slice(&scalars),
                    HostSlice::from_slice(&points),
                    &MSMConfig::default(),
                    HostSlice::from_mut_slice(&mut ref_msm_host_result),
                )
                .unwrap();

                assert_eq!(ref_msm_host_result, msm_host_result);
            }
            stream
                .destroy()
                .unwrap();
        });
}

pub fn check_msm_batch<C: Curve + MSM<C>>()
where
    <C::ScalarField as FieldImpl>::Config: GenerateRandom<C::ScalarField>,
{
    let test_sizes = [1000, 1 << 16];
    let batch_sizes = [1, 3, 1 << 4];
    let mut stream = IcicleStream::create().unwrap();
    let precompute_factor = 8;
    let mut cfg = MSMConfig::default();
    cfg.stream_handle = *stream;
    cfg.is_async = true;
    cfg.ext
        .set_int(LARGE_BUCKET_FACTOR, 5);
    cfg.c = 4;
    runtime::warmup(&stream).unwrap();
    stream
        .synchronize()
        .unwrap();
    for test_size in test_sizes {
        // (1) compute MSM with and w/o precompute on main device
        test_utilities::test_set_main_device();
        cfg.precompute_factor = precompute_factor;
        let points = generate_random_affine_points_with_zeroes::<C>(test_size, 10);
        let mut precomputed_points_d =
            DeviceVec::<Affine<C>>::device_malloc(cfg.precompute_factor as usize * test_size).unwrap();
        precompute_bases(HostSlice::from_slice(&points), &cfg, &mut precomputed_points_d).unwrap();
        for batch_size in batch_sizes {
            let scalars = <C::ScalarField as FieldImpl>::Config::generate_random(test_size * batch_size);
            // a version of batched msm without using `cfg.points_size`, requires copying bases
            let points_cloned: Vec<Affine<C>> = std::iter::repeat(points.clone())
                .take(batch_size)
                .flatten()
                .collect();
            let scalars_h = HostSlice::from_slice(&scalars);

            let mut msm_results_1 = DeviceVec::<Projective<C>>::device_malloc(batch_size).unwrap();
            let mut msm_results_2 = DeviceVec::<Projective<C>>::device_malloc(batch_size).unwrap();
            let mut points_d = DeviceVec::<Affine<C>>::device_malloc(test_size * batch_size).unwrap();
            points_d
                .copy_from_host_async(HostSlice::from_slice(&points_cloned), &stream)
                .unwrap();

            cfg.precompute_factor = precompute_factor;
            msm(scalars_h, &precomputed_points_d[..], &cfg, &mut msm_results_1[..]).unwrap();
            cfg.precompute_factor = 1;
            msm(scalars_h, &points_d[..], &cfg, &mut msm_results_2[..]).unwrap();

            let mut msm_host_result_1 = vec![Projective::<C>::zero(); batch_size];
            let mut msm_host_result_2 = vec![Projective::<C>::zero(); batch_size];
            msm_results_1
                .copy_to_host_async(HostSlice::from_mut_slice(&mut msm_host_result_1), &stream)
                .unwrap();
            msm_results_2
                .copy_to_host_async(HostSlice::from_mut_slice(&mut msm_host_result_2), &stream)
                .unwrap();
            stream
                .synchronize()
                .unwrap();

            // (2) compute on ref device and compare to both cases (with or w/o precompute)
            test_utilities::test_set_ref_device();
            let mut msm_ref_result = vec![Projective::<C>::zero(); batch_size];
            let mut ref_msm_config = MSMConfig::default();
            ref_msm_config.c = 4;
            msm(
                scalars_h,
                HostSlice::from_slice(&points),
                &MSMConfig::default(),
                HostSlice::from_mut_slice(&mut msm_ref_result),
            )
            .unwrap();

            assert_eq!(msm_host_result_1, msm_ref_result);
            assert_eq!(msm_host_result_2, msm_ref_result);
        }
    }
    stream
        .destroy()
        .unwrap();
}

pub fn check_msm_skewed_distributions<C: Curve + MSM<C>>()
where
    <C::ScalarField as FieldImpl>::Config: GenerateRandom<C::ScalarField>,
{
    let test_sizes = [1 << 10, 10000];
    let test_threshold = 1 << 11;
    let batch_sizes = [1, 3, 1 << 4];
    let rng = &mut thread_rng();
    for test_size in test_sizes {
        for batch_size in batch_sizes {
            let points = generate_random_affine_points_with_zeroes::<C>(test_size * batch_size, 100);
            let mut scalars = vec![C::ScalarField::zero(); test_size * batch_size];

            for _ in 0..(test_size * batch_size) {
                scalars[rng.gen_range(0..test_size * batch_size)] = C::ScalarField::one();
            }
            for _ in test_threshold..test_size {
                scalars[rng.gen_range(0..test_size * batch_size)] =
                    <<C::ScalarField as FieldImpl>::Config as GenerateRandom<C::ScalarField>>::generate_random(1)[0];
            }

            let mut cfg = MSMConfig::default();
            if test_size < test_threshold {
                cfg.bitsize = 1;
            }
            test_utilities::test_set_main_device();
            let mut msm_results = vec![Projective::<C>::zero(); batch_size];
            msm(
                HostSlice::from_slice(&scalars),
                HostSlice::from_slice(&points),
                &cfg,
                HostSlice::from_mut_slice(&mut msm_results),
            )
            .unwrap();

            test_utilities::test_set_ref_device();
            let mut msm_results_ref = vec![Projective::<C>::zero(); batch_size];
            msm(
                HostSlice::from_slice(&scalars),
                HostSlice::from_slice(&points),
                &cfg,
                HostSlice::from_mut_slice(&mut msm_results_ref),
            )
            .unwrap();

            assert_eq!(msm_results, msm_results_ref);
        }
    }
}
