pub mod curve;
pub mod msm;
pub mod ntt;
pub mod poseidon;
pub mod vec_ops;

impl icicle_core::SNARKCurve for curve::CurveCfg {}
