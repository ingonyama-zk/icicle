use std::ffi::{c_int, c_uint};
use rand::{rngs::StdRng, RngCore, SeedableRng};
use rustacuda_derive::DeviceCopy;
use std::mem::transmute;
use rustacuda::prelude::*;
use rustacuda_core::DevicePointer;
use rustacuda::memory::{DeviceBox, CopyDestination, DeviceCopy};

use std::marker::PhantomData;
use std::convert::TryInto;

use crate::basic_structs::point::{PointT, PointAffineNoInfinityT};
use crate::basic_structs::scalar::ScalarT;
use crate::basic_structs::field::Field;


#[derive(Debug, PartialEq, Clone, Copy,DeviceCopy)]
#[repr(C)]
pub struct ScalarField;
impl Field<8> for ScalarField {
    const MODOLUS: [u32; 8] = [0x0;8];
}

#[derive(Debug, PartialEq, Clone, Copy,DeviceCopy)]
#[repr(C)]
pub struct BaseField;
impl Field<12> for BaseField {
    const MODOLUS: [u32; 12] = [0x0;12];
}


pub type Scalar = ScalarT<ScalarField,8>;
impl Default for Scalar {
    fn default() -> Self {
        Self{value: [0x0;ScalarField::LIMBS], phantom: PhantomData }
    }
}

unsafe impl DeviceCopy for Scalar{}


pub type Base = ScalarT<BaseField,12>;
impl Default for Base {
    fn default() -> Self {
        Self{value: [0x0;BaseField::LIMBS], phantom: PhantomData }
    }
}

unsafe impl DeviceCopy for Base{}

pub type Point = PointT<Base>;
pub type PointAffineNoInfinity = PointAffineNoInfinityT<Base>;

extern "C" {
    fn eq_bls12_381(point1: *const Point, point2: *const Point) -> c_uint;
}

impl PartialEq for Point {
    fn eq(&self, other: &Self) -> bool {
        unsafe { eq_bls12_381(self, other) != 0 }
    }
}