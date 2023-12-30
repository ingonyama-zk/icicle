#[macro_use(impl_curve)]
extern crate icicle_core;

pub mod curve;
pub mod msm;
pub mod ntt;

impl icicle_core::SNARKCurve for curve::CurveCfg {}
