pub mod curve;
pub mod error;
pub mod field;
pub mod msm;
pub mod ntt;
pub mod poseidon;
pub mod tree;
#[cfg(feature = "arkworks")]
#[doc(hidden)]
pub mod tests;
pub mod traits;

pub trait SNARKCurve: curve::Curve + msm::MSM<Self>
where
    <Self::ScalarField as traits::FieldImpl>::Config: ntt::NTT<Self::ScalarField>,
{
}
