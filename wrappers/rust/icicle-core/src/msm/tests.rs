use crate::curve::{Affine, Curve, Projective};
use crate::msm::{msm, precompute_points, MSMConfig, MSM};
use crate::traits::{FieldImpl, GenerateRandom};
use icicle_cuda_runtime::device::{get_device_count, set_device, warmup};
use icicle_cuda_runtime::memory::{CudaHostAllocFlags, CudaHostRegisterFlags, DeviceVec, HostOrDeviceSlice, HostSlice};
use icicle_cuda_runtime::stream::CudaStream;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

#[cfg(feature = "arkworks")]
use crate::traits::ArkConvertible;
#[cfg(feature = "arkworks")]
use ark_ec::models::CurveConfig as ArkCurveConfig;
#[cfg(feature = "arkworks")]
use ark_ec::VariableBaseMSM;
#[cfg(feature = "arkworks")]
use ark_std::{rand::Rng, test_rng, UniformRand};

pub fn generate_random_affine_points_with_zeroes<C: Curve>(size: usize, num_zeroes: usize) -> Vec<Affine<C>> {
    let rng = &mut test_rng();
    let mut points = C::generate_random_affine_points(size);
    for _ in 0..num_zeroes {
        points[rng.gen_range(0..size)] = Affine::<C>::zero();
    }
    points
}

pub fn check_msm<C: Curve + MSM<C>>()
where
    <C::ScalarField as FieldImpl>::Config: GenerateRandom<C::ScalarField>,
    C::ScalarField: ArkConvertible<ArkEquivalent = <C::ArkSWConfig as ArkCurveConfig>::ScalarField>,
    C::BaseField: ArkConvertible<ArkEquivalent = <C::ArkSWConfig as ArkCurveConfig>::BaseField>,
{
    let device_count = get_device_count().unwrap();
    (0..device_count) // TODO: this is proto-loadbalancer
        .into_par_iter()
        .for_each(move |device_id| {
            //TODO: currently supported multi-GPU workflow:
            //      1) User starts child host thread from parent host thread
            //      2) Calls set_device once with selected device_id (0 is default device .. < device_count)
            //      3) Perform all operations (without changing device on the thread)
            //      4) If necessary - export results to parent host thread

            set_device(device_id).unwrap();
            let test_sizes = [4, 8, 16, 32, 64, 128, 256, 1000, 1 << 18];
            let mut msm_results = DeviceVec::<Projective<C>>::cuda_malloc_for_device(1, device_id).unwrap();
            for test_size in test_sizes {
                let points = generate_random_affine_points_with_zeroes(test_size, 2);
                let scalars = <C::ScalarField as FieldImpl>::Config::generate_random(test_size);
                let points_ark: Vec<_> = points
                    .iter()
                    .map(|x| x.to_ark())
                    .collect();
                let scalars_ark: Vec<_> = scalars
                    .iter()
                    .map(|x| x.to_ark())
                    .collect();
                // if we simply transmute arkworks types, we'll get scalars or points in Montgomery format
                // (just beware the possible extra flag in affine point types, can't transmute ark Affine because of that)
                let scalars_mont = unsafe { &*(&scalars_ark[..] as *const _ as *const [C::ScalarField]) };

                let mut scalars_d = DeviceVec::<C::ScalarField>::cuda_malloc(test_size).unwrap();
                let stream = CudaStream::create().unwrap();
                scalars_d
                    .copy_from_host_async(HostSlice::from_slice(&scalars_mont), &stream)
                    .unwrap();

                let mut cfg = MSMConfig::default_for_device(device_id);
                cfg.ctx
                    .stream = &stream;
                cfg.is_async = true;
                cfg.are_scalars_montgomery_form = true;
                msm(
                    &scalars_d[..],
                    HostSlice::from_slice(&points),
                    &cfg,
                    &mut msm_results[..],
                )
                .unwrap();
                // need to make sure that scalars_d weren't mutated by the previous call
                let mut scalars_mont_after = vec![C::ScalarField::zero(); test_size];
                scalars_d
                    .copy_to_host_async(HostSlice::from_mut_slice(&mut scalars_mont_after), &stream)
                    .unwrap();
                assert_eq!(scalars_mont, scalars_mont_after);

                let mut msm_host_result = vec![Projective::<C>::zero(); 1];
                msm_results
                    .copy_to_host(HostSlice::from_mut_slice(&mut msm_host_result[..]))
                    .unwrap();
                stream
                    .synchronize()
                    .unwrap();
                stream
                    .destroy()
                    .unwrap();

                let msm_result_ark: ark_ec::models::short_weierstrass::Projective<C::ArkSWConfig> =
                    VariableBaseMSM::msm(&points_ark, &scalars_ark).unwrap();
                let msm_res_affine: ark_ec::short_weierstrass::Affine<C::ArkSWConfig> = msm_host_result[0]
                    .to_ark()
                    .into();
                assert!(msm_res_affine.is_on_curve());
                assert_eq!(msm_host_result[0].to_ark(), msm_result_ark);
            }
        });
}

pub fn check_msm_pinned_memory<C: Curve + MSM<C> + 'static>()
where
    <C::ScalarField as FieldImpl>::Config: GenerateRandom<C::ScalarField>,
    C::ScalarField: ArkConvertible<ArkEquivalent = <C::ArkSWConfig as ArkCurveConfig>::ScalarField>,
    C::BaseField: ArkConvertible<ArkEquivalent = <C::ArkSWConfig as ArkCurveConfig>::BaseField>,
{
    let largest_size = 1 << 16;
    // let test_sizes = [1 << 10, largest_size];
    let test_size = largest_size;
    let mut msm_results = DeviceVec::<Projective<C>>::cuda_malloc(1).unwrap();
    let random_points = generate_random_affine_points_with_zeroes(largest_size, 2);
    let points: &HostSlice<Affine<C>> = HostSlice::from_slice(&random_points);

    // TODO: implement cudaHostRegister via HostSlice.pin
    // let pin = points.is_pinnable();
    // if pin {
    //     points.pin(CudaHostRegisterFlags::DEFAULT).unwrap();
    //     let flags = points.get_memory_flags().unwrap();
    //     println!("Flags of registered pin: {:?}", flags);
    //     unsafe {
    //         println!("points address Rust after pin: {:?}", points.as_ptr());
    //     }
    //     points.unpin();
    //     unsafe {
    //         println!("points address Rust after unpin: {:?}", points.as_ptr());
    //     }
    // }

    let scalars = <<C as Curve>::ScalarField as FieldImpl>::Config::generate_random(largest_size);

    let mut scalars_d = DeviceVec::<<C as Curve>::ScalarField>::cuda_malloc(test_size).unwrap();
    let stream = CudaStream::create().unwrap();
    scalars_d
        .copy_from_host_async(HostSlice::from_slice(&scalars[..test_size]), &stream)
        .unwrap();

    let mut cfg = MSMConfig::default();
    cfg.ctx
        .stream = &stream;
    cfg.is_async = true;
    msm(&scalars_d[..], points, &cfg, &mut msm_results[..]).unwrap();

    let mut msm_host_result = vec![Projective::<C>::zero(); 1];
    msm_results
        .copy_to_host(HostSlice::from_mut_slice(&mut msm_host_result[..]))
        .unwrap();
    stream
        .synchronize()
        .unwrap();

    let msm_res_affine: ark_ec::short_weierstrass::Affine<<C as Curve>::ArkSWConfig> = msm_host_result[0]
        .to_ark()
        .into();
    assert!(msm_res_affine.is_on_curve());

    points.allocate_pinned(points.len(), CudaHostAllocFlags::DEFAULT).unwrap();
    // let allocated_pinned_points = HostSlice::allocate_pinned(points.len(), CudaHostAllocFlags::DEFAULT).unwrap();
    // allocated_pinned_points
    //     .as_mut_slice()
    //     .clone_from_slice(points.as_slice());

    msm(&scalars_d[..], points, &cfg, &mut msm_results[..]).unwrap();

    let mut msm_host_result = vec![Projective::<C>::zero(); 1];
    msm_results
        .copy_to_host(HostSlice::from_mut_slice(&mut msm_host_result[..]))
        .unwrap();
    stream
        .synchronize()
        .unwrap();

    let msm_res_affine: ark_ec::short_weierstrass::Affine<<C as Curve>::ArkSWConfig> = msm_host_result[0]
        .to_ark()
        .into();
    assert!(msm_res_affine.is_on_curve());
    points.free_pinned();

    stream
        .destroy()
        .unwrap();
}

pub fn check_msm_batch<C: Curve + MSM<C>>()
where
    <C::ScalarField as FieldImpl>::Config: GenerateRandom<C::ScalarField>,
    C::ScalarField: ArkConvertible<ArkEquivalent = <C::ArkSWConfig as ArkCurveConfig>::ScalarField>,
    C::BaseField: ArkConvertible<ArkEquivalent = <C::ArkSWConfig as ArkCurveConfig>::BaseField>,
{
    let test_sizes = [1000, 1 << 16];
    let batch_sizes = [1, 3, 1 << 4];
    let stream = CudaStream::create().unwrap();
    let mut cfg = MSMConfig::default();
    cfg.ctx
        .stream = &stream;
    cfg.is_async = true;
    cfg.large_bucket_factor = 5;
    cfg.c = 4;
    let precompute_factor = 8;
    warmup(&stream).unwrap();
    for test_size in test_sizes {
        let points = generate_random_affine_points_with_zeroes(test_size, 10);
        let mut precomputed_points_d = DeviceVec::cuda_malloc(precompute_factor * test_size).unwrap();
        cfg.precompute_factor = precompute_factor as i32;
        precompute_points(
            HostSlice::from_slice(&points),
            test_size as i32,
            &cfg,
            &mut precomputed_points_d,
        )
        .unwrap();
        for batch_size in batch_sizes {
            let scalars = <C::ScalarField as FieldImpl>::Config::generate_random(test_size * batch_size);
            // a version of batched msm without using `cfg.points_size`, requires copying bases
            let points_cloned: Vec<Affine<C>> = std::iter::repeat(points.clone())
                .take(batch_size)
                .flatten()
                .collect();
            let scalars_h = HostSlice::from_slice(&scalars);

            let mut msm_results_1 = DeviceVec::<Projective<C>>::cuda_malloc(batch_size).unwrap();
            let mut msm_results_2 = DeviceVec::<Projective<C>>::cuda_malloc(batch_size).unwrap();
            let mut points_d = DeviceVec::<Affine<C>>::cuda_malloc(test_size * batch_size).unwrap();
            points_d
                .copy_from_host_async(HostSlice::from_slice(&points_cloned), &stream)
                .unwrap();

            cfg.precompute_factor = precompute_factor as i32;
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

            let points_ark: Vec<_> = points
                .iter()
                .map(|x| x.to_ark())
                .collect();
            let scalars_ark: Vec<_> = scalars_h
                .iter()
                .map(|x| x.to_ark())
                .collect();
            for (i, scalars_chunk) in scalars_ark
                .chunks(test_size)
                .enumerate()
            {
                let msm_result_ark: ark_ec::models::short_weierstrass::Projective<C::ArkSWConfig> =
                    VariableBaseMSM::msm(&points_ark, &scalars_chunk).unwrap();
                assert_eq!(msm_host_result_1[i].to_ark(), msm_result_ark);
                assert_eq!(msm_host_result_2[i].to_ark(), msm_result_ark);
            }
        }
    }
    stream
        .destroy()
        .unwrap();
}

pub fn check_msm_skewed_distributions<C: Curve + MSM<C>>()
where
    <C::ScalarField as FieldImpl>::Config: GenerateRandom<C::ScalarField>,
    C::ScalarField: ArkConvertible<ArkEquivalent = <C::ArkSWConfig as ArkCurveConfig>::ScalarField>,
    C::BaseField: ArkConvertible<ArkEquivalent = <C::ArkSWConfig as ArkCurveConfig>::BaseField>,
{
    let test_sizes = [1 << 10, 10000];
    let test_threshold = 1 << 11;
    let batch_sizes = [1, 3, 1 << 4];
    let rng = &mut test_rng();
    for test_size in test_sizes {
        for batch_size in batch_sizes {
            let points = generate_random_affine_points_with_zeroes(test_size * batch_size, 100);
            let mut scalars = vec![C::ScalarField::zero(); test_size * batch_size];

            for _ in 0..(test_size * batch_size) {
                scalars[rng.gen_range(0..test_size * batch_size)] = C::ScalarField::one();
            }
            for _ in test_threshold..test_size {
                scalars[rng.gen_range(0..test_size * batch_size)] =
                    C::ScalarField::from_ark(<C::ScalarField as ArkConvertible>::ArkEquivalent::rand(rng));
            }
            let points_ark: Vec<_> = points
                .iter()
                .map(|x| x.to_ark())
                .collect();
            let scalars_ark: Vec<_> = scalars
                .iter()
                .map(|x| x.to_ark())
                .collect();

            let mut msm_results = vec![Projective::<C>::zero(); batch_size];

            let mut cfg = MSMConfig::default();
            if test_size < test_threshold {
                cfg.bitsize = 1;
            }
            msm(
                HostSlice::from_slice(&scalars),
                HostSlice::from_slice(&points),
                &cfg,
                HostSlice::from_mut_slice(&mut msm_results),
            )
            .unwrap();

            for (i, (scalars_chunk, points_chunk)) in scalars_ark
                .chunks(test_size)
                .zip(points_ark.chunks(test_size))
                .enumerate()
            {
                let msm_result_ark: ark_ec::models::short_weierstrass::Projective<C::ArkSWConfig> =
                    VariableBaseMSM::msm(&points_chunk, &scalars_chunk).unwrap();
                assert_eq!(msm_results[i].to_ark(), msm_result_ark);
            }
        }
    }
}
