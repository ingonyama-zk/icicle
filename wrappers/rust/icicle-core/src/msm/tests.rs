use crate::curve::{Affine, Curve, Projective};
use crate::field::PrimeField;
use crate::msm::{msm, precompute_bases, MSMConfig, CUDA_MSM_LARGE_BUCKET_FACTOR, MSM};
use crate::traits::{GenerateRandom, MontgomeryConvertible};
use icicle_runtime::{
    memory::{DeviceVec, HostOrDeviceSlice, HostSlice, IntoIcicleSlice, IntoIcicleSliceMut},
    runtime,
    stream::IcicleStream,
    test_utilities,
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

pub fn check_msm<C: Curve + MSM<C>>() {
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

            let test_sizes = [1, 5, 16, 32, 64, 128, 256, 1000, 1 << 18];
            test_utilities::test_set_main_device_with_id(device_id);
            let mut stream = IcicleStream::create().unwrap();
            let mut msm_results = DeviceVec::<Projective<C>>::device_malloc_async(1, &stream).unwrap();
            for test_size in test_sizes {
                let points = generate_random_affine_points_with_zeroes::<C>(test_size, 2);
                let scalars = C::ScalarField::generate_random(test_size);

                // (1) async msm on main device
                test_utilities::test_set_main_device_with_id(device_id);
                let mut scalars_d = DeviceVec::<C::ScalarField>::device_malloc_async(test_size, &stream).unwrap();
                scalars_d
                    .copy_from_host_async(scalars.into_slice(), &stream)
                    .unwrap();

                // convert to mont for testing MSM in this case
                C::ScalarField::to_mont(scalars_d.into_slice_mut(), &stream);

                let mut cfg = MSMConfig::default();
                cfg.stream_handle = *stream;
                cfg.are_scalars_montgomery_form = true;
                cfg.is_async = true;
                msm(
                    scalars_d.into_slice(),
                    points.into_slice(),
                    &cfg,
                    msm_results.into_slice_mut(),
                )
                .unwrap();

                let msm_host_result = msm_results.to_host_vec();

                stream
                    .synchronize()
                    .unwrap();

                // (2) compute on ref device and compare
                test_utilities::test_set_ref_device();
                let mut ref_msm_host_result = vec![Projective::<C>::zero(); 1];
                msm(
                    scalars.into_slice(),
                    points.into_slice(),
                    &MSMConfig::default(),
                    ref_msm_host_result.into_slice_mut(),
                )
                .unwrap();

                assert_eq!(ref_msm_host_result, msm_host_result);
            }
            stream
                .destroy()
                .unwrap();
        });
}

pub fn check_msm_batch_shared<C: Curve + MSM<C>>() {
    let test_sizes = [1000, 1 << 14];
    let batch_sizes = [1, 3, 1 << 4];
    let mut stream = IcicleStream::create().unwrap();
    let precompute_factor = 8;
    let mut cfg = MSMConfig::default();
    cfg.stream_handle = *stream;
    cfg.is_async = true;
    cfg.ext
        .set_int(CUDA_MSM_LARGE_BUCKET_FACTOR, 5);
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
        let mut precomputed_points_d = DeviceVec::<Affine<C>>::malloc(cfg.precompute_factor as usize * test_size);
        precompute_bases(points.into_slice(), &cfg, precomputed_points_d.into_slice_mut()).unwrap();
        for batch_size in batch_sizes {
            let scalars_h = C::ScalarField::generate_random(test_size * batch_size);
            let scalars_h = scalars_h.into_slice();

            let mut msm_results_1 = DeviceVec::<Projective<C>>::malloc(batch_size);
            let mut msm_results_2 = DeviceVec::<Projective<C>>::malloc(batch_size);
            let mut points_d = DeviceVec::<Affine<C>>::malloc(test_size);
            points_d
                .copy_from_host_async(points.into_slice(), &stream)
                .unwrap();

            cfg.precompute_factor = precompute_factor;
            msm(
                scalars_h,
                precomputed_points_d.into_slice(),
                &cfg,
                msm_results_1.into_slice_mut(),
            )
            .unwrap();
            cfg.precompute_factor = 1;
            msm(
                scalars_h,
                points_d.into_slice(),
                &cfg,
                msm_results_2.into_slice_mut(),
            )
            .unwrap();

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
                points.into_slice(),
                &MSMConfig::default(),
                msm_ref_result.into_slice_mut(),
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

pub fn check_msm_batch_not_shared<C: Curve + MSM<C>>() {
    let test_sizes = [1000, 1 << 14];
    let batch_sizes = [1, 3, 1 << 4];
    let mut stream = IcicleStream::create().unwrap();
    let precompute_factor = 8;
    let mut cfg = MSMConfig::default();
    cfg.stream_handle = *stream;
    cfg.is_async = true;
    cfg.ext
        .set_int(CUDA_MSM_LARGE_BUCKET_FACTOR, 5);
    cfg.c = 4;
    runtime::warmup(&stream).unwrap();
    stream
        .synchronize()
        .unwrap();
    for test_size in test_sizes {
        // (1) compute MSM with and w/o precompute on main device
        test_utilities::test_set_main_device();
        for batch_size in batch_sizes {
            cfg.precompute_factor = precompute_factor;
            let scalars = C::ScalarField::generate_random(test_size * batch_size);
            let scalars_h = scalars.into_slice();

            let points = generate_random_affine_points_with_zeroes::<C>(test_size * batch_size, 10);
            println!("points len: {}", points.len());
            let mut precomputed_points_d =
                DeviceVec::<Affine<C>>::malloc(cfg.precompute_factor as usize * test_size * batch_size);
            cfg.batch_size = batch_size as i32;
            cfg.are_points_shared_in_batch = false;
            precompute_bases(points.into_slice(), &cfg, precomputed_points_d.into_slice_mut()).unwrap();
            println!("precomputed points len: {}", (precomputed_points_d).len());

            let mut msm_results_1 = DeviceVec::<Projective<C>>::malloc(batch_size);
            let mut msm_results_2 = DeviceVec::<Projective<C>>::malloc(batch_size);
            let mut points_d = DeviceVec::<Affine<C>>::malloc(test_size * batch_size);
            points_d
                .copy_from_host_async(points.into_slice(), &stream)
                .unwrap();

            cfg.precompute_factor = precompute_factor;
            msm(
                scalars_h,
                precomputed_points_d.into_slice(),
                &cfg,
                msm_results_1.into_slice_mut(),
            )
            .unwrap();
            cfg.precompute_factor = 1;
            msm(scalars_h, points_d.into_slice(), &cfg, msm_results_2.into_slice_mut()).unwrap();

            let msm_host_result_1 = msm_results_1.to_host_vec();
            let msm_host_result_2 = msm_results_2.to_host_vec();
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
                points.into_slice(),
                &MSMConfig::default(),
                msm_ref_result.into_slice_mut(),
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

pub fn check_msm_skewed_distributions<C: Curve + MSM<C>>() {
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
                scalars[rng.gen_range(0..test_size * batch_size)] = C::ScalarField::generate_random(1)[0];
            }

            let mut cfg = MSMConfig::default();
            if test_size < test_threshold {
                cfg.bitsize = 1;
            }
            test_utilities::test_set_main_device();
            let mut msm_results = vec![Projective::<C>::zero(); batch_size];
            msm(
                scalars.into_slice(),
                points.into_slice(),
                &cfg,
                msm_results.into_slice_mut(),
            )
            .unwrap();

            test_utilities::test_set_ref_device();
            let mut msm_results_ref = vec![Projective::<C>::zero(); batch_size];
            msm(
                scalars.into_slice(),
                points.into_slice(),
                &cfg,
                msm_results_ref.into_slice_mut(),
            )
            .unwrap();

            assert_eq!(msm_results, msm_results_ref);
        }
    }
}
