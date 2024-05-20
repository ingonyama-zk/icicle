use crate::curve::CurveCfg;
#[cfg(feature = "g2")]
use crate::curve::G2CurveCfg;
use icicle_core::{
    curve::{Affine, Curve, Projective},
    error::IcicleResult,
    impl_msm,
    msm::{MSMConfig, MSM},
    traits::IcicleResultWrap,
};
use icicle_cuda_runtime::{
    device_context::DeviceContext,
    error::CudaError,
    memory::{DeviceSlice, HostOrDeviceSlice},
};

impl_msm!("bn254", bn254, CurveCfg);
#[cfg(feature = "g2")]
impl_msm!("bn254_g2", bn254_g2, G2CurveCfg);

#[cfg(test)]
pub(crate) mod tests {
    use crate::curve::CurveCfg;
    #[cfg(feature = "g2")]
    use crate::curve::G2CurveCfg;
    use icicle_core::impl_msm_tests;
    use icicle_core::msm::tests::*;

    use icicle_core::curve::{Affine, Curve, Projective};
    use icicle_core::msm::{msm, precompute_bases, MSMConfig, MSM};
    use icicle_core::traits::{FieldImpl, GenerateRandom, ArkConvertible};
    use icicle_cuda_runtime::device::{get_device_count, set_device, warmup};
    use icicle_cuda_runtime::memory::{CudaHostRegisterFlags, DeviceVec, HostOrDeviceSlice, HostSlice};
    use icicle_cuda_runtime::stream::CudaStream;

    #[test]
    fn test_pinned_memory() {
        let largest_size = 1 << 16;
        // let test_sizes = [1 << 10, largest_size];
        let test_size = largest_size;
        let mut msm_results = DeviceVec::<Projective<CurveCfg>>::cuda_malloc(1).unwrap();
        let random_points = generate_random_affine_points_with_zeroes(largest_size, 2);
        let points = HostSlice::from_slice(&random_points);
        points.pin(CudaHostRegisterFlags::DEFAULT).unwrap();
        points.unpin();
        let flags = points.get_memory_flags().unwrap();

        let scalars = <<CurveCfg as Curve>::ScalarField as FieldImpl>::Config::generate_random(largest_size);
        
        // for test_size in test_sizes {
            let mut scalars_d = DeviceVec::<<CurveCfg as Curve>::ScalarField>::cuda_malloc(test_size).unwrap();
            // let mut points_d = DeviceVec::<<CurveCfg as Curve>::Affine>::cuda_malloc(test_size).unwrap();
            let stream = CudaStream::create().unwrap();
            scalars_d
                .copy_from_host_async(HostSlice::from_slice(&scalars[..test_size]), &stream)
                .unwrap();

            let mut cfg = MSMConfig::default();
            cfg.ctx
                .stream = &stream;
            cfg.is_async = true;
            msm(
                &scalars_d[..],
                points,
                &cfg,
                &mut msm_results[..],
            )
            .unwrap();

            let mut msm_host_result = vec![Projective::<CurveCfg>::zero(); 1];
            msm_results
                .copy_to_host(HostSlice::from_mut_slice(&mut msm_host_result[..]))
                .unwrap();
            stream
                .synchronize()
                .unwrap();
            stream
                .destroy()
                .unwrap();

            let msm_res_affine: ark_ec::short_weierstrass::Affine<<CurveCfg as Curve>::ArkSWConfig> = msm_host_result[0]
                .to_ark()
                .into();
            assert!(msm_res_affine.is_on_curve());
        // }
    }

    impl_msm_tests!(CurveCfg);
    #[cfg(feature = "g2")]
    mod g2 {
        use super::*;
        impl_msm_tests!(G2CurveCfg);
    }
}
