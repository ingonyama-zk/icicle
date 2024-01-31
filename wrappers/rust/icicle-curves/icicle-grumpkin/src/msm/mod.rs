use crate::curve::CurveCfg;
use icicle_core::{
    curve::{Affine, Curve, Projective},
    error::IcicleResult,
    msm::{MSMConfig, MSM},
};
use icicle_cuda_runtime::memory::HostOrDeviceSlice;

impl MSM<CurveCfg> for CurveCfg {
    fn msm_unchecked(
        _scalars: &HostOrDeviceSlice<<CurveCfg as Curve>::ScalarField>,
        _points: &HostOrDeviceSlice<Affine<CurveCfg>>,
        _cfg: &MSMConfig,
        _results: &mut HostOrDeviceSlice<Projective<CurveCfg>>,
    ) -> IcicleResult<()> {
        todo!()
    }

    fn get_default_msm_config() -> MSMConfig<'static> {
        todo!()
    }
}
// impl_msm!("grumpkin", grumpkin, CurveCfg);

// #[cfg(test)]
// pub(crate) mod tests {
//     use icicle_core::impl_msm_tests;
//     use icicle_core::msm::tests::*;

//     use crate::curve::CurveCfg;

//     impl_msm_tests!(CurveCfg);
// }
