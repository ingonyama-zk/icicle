use std::ffi::c_void;

pub mod curve;
pub mod ecntt;
pub mod error;
pub mod field;
pub mod hash;
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

#[repr(C)]
#[derive(Debug)]
pub struct Matrix {
    pub values: *const c_void,
    pub width: usize,
    pub height: usize,
}

impl Matrix {
    pub fn from_slice<T>(slice: &[T], width: usize, height: usize) -> Self {
        Matrix {
            values: slice.as_ptr() as *const c_void,
            width,
            height,
        }
    }
}
