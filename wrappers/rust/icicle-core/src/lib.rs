pub mod curve;
pub mod error;
pub mod field;
pub mod msm;
pub mod ntt;
pub mod poseidon;
#[cfg(feature = "arkworks")]
#[doc(hidden)]
pub mod tests;
pub mod traits;
pub mod tree;

pub trait SNARKCurve: curve::Curve + msm::MSM<Self>
where
    <Self::ScalarField as traits::FieldImpl>::Config: ntt::NTT<Self::ScalarField>,
{
}
