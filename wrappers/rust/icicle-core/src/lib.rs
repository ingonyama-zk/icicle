pub mod curve;
pub mod ecntt;
pub mod error;
pub mod field;
pub mod msm;
pub mod ntt;
pub mod polynomials;
pub mod poseidon;
pub mod poseidon2;
#[doc(hidden)]
pub mod tests;
pub mod traits;
pub mod tree;
pub mod vec_ops;

pub trait SNARKCurve: curve::Curve + msm::MSM<Self>
where
    <Self::ScalarField as traits::FieldImpl>::Config: ntt::NTT<Self::ScalarField, Self::ScalarField>,
{
}
