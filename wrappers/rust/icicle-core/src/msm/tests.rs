use super::{msm, MSM};
use crate::curve::{Affine, Curve, Projective};
use crate::msm::MSMConfig;
use crate::traits::{FieldImpl, GenerateRandom};
use icicle_cuda_runtime::device::{get_device_count, set_device};
use icicle_cuda_runtime::memory::HostOrDeviceSlice;
use icicle_cuda_runtime::stream::CudaStream;
use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;

#[cfg(feature = "arkworks")]
use crate::traits::ArkConvertible;
#[cfg(feature = "arkworks")]
use ark_ec::models::CurveConfig as ArkCurveConfig;
#[cfg(feature = "arkworks")]
use ark_ec::VariableBaseMSM;
#[cfg(feature = "arkworks")]
use ark_std::{rand::Rng, test_rng, UniformRand};

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
            let mut msm_results = HostOrDeviceSlice::cuda_malloc(1).unwrap();
            for test_size in test_sizes {
                let points = C::generate_random_affine_points(test_size);
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

                let mut scalars_d = HostOrDeviceSlice::cuda_malloc(test_size).unwrap();
                let stream = CudaStream::create().unwrap();
                scalars_d
                    .copy_from_host_async(&scalars_mont, &stream)
                    .unwrap();

                let mut cfg = MSMConfig::default_for_device(device_id);
                cfg.ctx
                    .stream = &stream;
                cfg.is_async = true;
                cfg.are_scalars_montgomery_form = true;
                msm(&scalars_d, &HostOrDeviceSlice::on_host(points), &cfg, &mut msm_results).unwrap();
                // need to make sure that scalars_d weren't mutated by the previous call
                let mut scalars_mont_after = vec![C::ScalarField::zero(); test_size];
                scalars_d
                    .copy_to_host_async(&mut scalars_mont_after, &stream)
                    .unwrap();
                assert_eq!(scalars_mont, scalars_mont_after);

                let mut msm_host_result = vec![Projective::<C>::zero(); 1];
                msm_results
                    .copy_to_host(&mut msm_host_result[..])
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

pub fn check_msm_batch<C: Curve + MSM<C>>()
where
    <C::ScalarField as FieldImpl>::Config: GenerateRandom<C::ScalarField>,
    C::ScalarField: ArkConvertible<ArkEquivalent = <C::ArkSWConfig as ArkCurveConfig>::ScalarField>,
    C::BaseField: ArkConvertible<ArkEquivalent = <C::ArkSWConfig as ArkCurveConfig>::BaseField>,
{
    let test_sizes = [1000, 1 << 16];
    let batch_sizes = [1, 3, 1 << 4];
    for test_size in test_sizes {
        for batch_size in batch_sizes {
            let points = C::generate_random_affine_points(test_size);
            let scalars = <C::ScalarField as FieldImpl>::Config::generate_random(test_size * batch_size);
            // a version of batched msm without using `cfg.points_size`, requires copying bases
            let points_cloned: Vec<Affine<C>> = std::iter::repeat(points.clone())
                .take(batch_size)
                .flatten()
                .collect();
            let points_h = HostOrDeviceSlice::on_host(points);
            let scalars_h = HostOrDeviceSlice::on_host(scalars);

            let mut msm_results_1 = HostOrDeviceSlice::cuda_malloc(batch_size).unwrap();
            let mut msm_results_2 = HostOrDeviceSlice::cuda_malloc(batch_size).unwrap();
            let mut points_d = HostOrDeviceSlice::cuda_malloc(test_size * batch_size).unwrap();
            let stream = CudaStream::create().unwrap();
            points_d
                .copy_from_host_async(&points_cloned, &stream)
                .unwrap();

            let mut cfg = MSMConfig::default();
            cfg.ctx
                .stream = &stream;
            cfg.is_async = true;
            msm(&scalars_h, &points_h, &cfg, &mut msm_results_1).unwrap();
            msm(&scalars_h, &points_d, &cfg, &mut msm_results_2).unwrap();

            let mut msm_host_result_1 = vec![Projective::<C>::zero(); batch_size];
            let mut msm_host_result_2 = vec![Projective::<C>::zero(); batch_size];
            msm_results_1
                .copy_to_host_async(&mut msm_host_result_1[..], &stream)
                .unwrap();
            msm_results_2
                .copy_to_host_async(&mut msm_host_result_2[..], &stream)
                .unwrap();
            stream
                .synchronize()
                .unwrap();
            stream
                .destroy()
                .unwrap();

            let points_ark: Vec<_> = points_h
                .as_slice()
                .iter()
                .map(|x| x.to_ark())
                .collect();
            let scalars_ark: Vec<_> = scalars_h
                .as_slice()
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
}

pub fn check_msm_skewed_distributions<C: Curve + MSM<C>>()
where
    <C::ScalarField as FieldImpl>::Config: GenerateRandom<C::ScalarField>,
    C::ScalarField: ArkConvertible<ArkEquivalent = <C::ArkSWConfig as ArkCurveConfig>::ScalarField>,
    C::BaseField: ArkConvertible<ArkEquivalent = <C::ArkSWConfig as ArkCurveConfig>::BaseField>,
{
    let test_sizes = [1 << 6, 1000];
    let test_threshold = 1 << 8;
    let batch_sizes = [1, 3, 1 << 8];
    let rng = &mut test_rng();
    for test_size in test_sizes {
        for batch_size in batch_sizes {
            let points = C::generate_random_affine_points(test_size * batch_size);
            let mut scalars = vec![C::ScalarField::zero(); test_size * batch_size];
            for _ in 0..(test_size * batch_size / 2) {
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

            let mut msm_results = HostOrDeviceSlice::on_host(vec![Projective::<C>::zero(); batch_size]);

            let mut cfg = MSMConfig::default();
            if test_size < test_threshold {
                cfg.bitsize = 1;
            }
            msm(
                &HostOrDeviceSlice::on_host(scalars),
                &HostOrDeviceSlice::on_host(points),
                &cfg,
                &mut msm_results,
            )
            .unwrap();

            for (i, (scalars_chunk, points_chunk)) in scalars_ark
                .chunks(test_size)
                .zip(points_ark.chunks(test_size))
                .enumerate()
            {
                let msm_result_ark: ark_ec::models::short_weierstrass::Projective<C::ArkSWConfig> =
                    VariableBaseMSM::msm(&points_chunk, &scalars_chunk).unwrap();
                assert_eq!(msm_results.as_slice()[i].to_ark(), msm_result_ark);
            }
        }
    }
}
