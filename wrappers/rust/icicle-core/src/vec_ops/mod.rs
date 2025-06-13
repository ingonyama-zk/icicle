//! Vector operations module for field elements
//! 
//! This module provides vector operations for field elements, including:
//! - Basic arithmetic operations (add, sub, mul, div)
//! - Scalar operations (scalar_add, scalar_sub, scalar_mul)
//! - Reduction operations (sum, product)
//! - Specialized operations (transpose, bit_reverse, slice)
//! 
//! All operations are implemented for both host and device memory.

use crate::traits::FieldImpl;
use icicle_runtime::{
    errors::eIcicleError,
    memory::HostOrDeviceSlice,
};

mod helpers;
mod ops;

pub use ops::*;

/// Configuration for vector operations
#[derive(Clone, Debug)]
pub struct VecOpsConfig {
    pub ctx: icicle_runtime::Context,
    pub is_async: bool,
}

/// Trait for vector operations on field elements
pub trait VecOps<F: FieldImpl> {
    fn add(
        a: &(impl HostOrDeviceSlice<F> + ?Sized),
        b: &(impl HostOrDeviceSlice<F> + ?Sized),
        result: &mut (impl HostOrDeviceSlice<F> + ?Sized),
        cfg: &VecOpsConfig,
    ) -> Result<(), eIcicleError>;

    fn accumulate(
        a: &mut (impl HostOrDeviceSlice<F> + ?Sized),
        b: &(impl HostOrDeviceSlice<F> + ?Sized),
        cfg: &VecOpsConfig,
    ) -> Result<(), eIcicleError>;

    fn sub(
        a: &(impl HostOrDeviceSlice<F> + ?Sized),
        b: &(impl HostOrDeviceSlice<F> + ?Sized),
        result: &mut (impl HostOrDeviceSlice<F> + ?Sized),
        cfg: &VecOpsConfig,
    ) -> Result<(), eIcicleError>;

    fn mul(
        a: &(impl HostOrDeviceSlice<F> + ?Sized),
        b: &(impl HostOrDeviceSlice<F> + ?Sized),
        result: &mut (impl HostOrDeviceSlice<F> + ?Sized),
        cfg: &VecOpsConfig,
    ) -> Result<(), eIcicleError>;

    fn div(
        a: &(impl HostOrDeviceSlice<F> + ?Sized),
        b: &(impl HostOrDeviceSlice<F> + ?Sized),
        result: &mut (impl HostOrDeviceSlice<F> + ?Sized),
        cfg: &VecOpsConfig,
    ) -> Result<(), eIcicleError>;

    fn inv(
        input: &(impl HostOrDeviceSlice<F> + ?Sized),
        output: &mut (impl HostOrDeviceSlice<F> + ?Sized),
        cfg: &VecOpsConfig,
    ) -> Result<(), eIcicleError>;

    fn sum(
        a: &(impl HostOrDeviceSlice<F> + ?Sized),
        result: &mut (impl HostOrDeviceSlice<F> + ?Sized),
        cfg: &VecOpsConfig,
    ) -> Result<(), eIcicleError>;

    fn product(
        a: &(impl HostOrDeviceSlice<F> + ?Sized),
        result: &mut (impl HostOrDeviceSlice<F> + ?Sized),
        cfg: &VecOpsConfig,
    ) -> Result<(), eIcicleError>;

    fn scalar_add(
        a: &(impl HostOrDeviceSlice<F> + ?Sized),
        b: &(impl HostOrDeviceSlice<F> + ?Sized),
        result: &mut (impl HostOrDeviceSlice<F> + ?Sized),
        cfg: &VecOpsConfig,
    ) -> Result<(), eIcicleError>;

    fn scalar_sub(
        a: &(impl HostOrDeviceSlice<F> + ?Sized),
        b: &(impl HostOrDeviceSlice<F> + ?Sized),
        result: &mut (impl HostOrDeviceSlice<F> + ?Sized),
        cfg: &VecOpsConfig,
    ) -> Result<(), eIcicleError>;

    fn scalar_mul(
        a: &(impl HostOrDeviceSlice<F> + ?Sized),
        b: &(impl HostOrDeviceSlice<F> + ?Sized),
        result: &mut (impl HostOrDeviceSlice<F> + ?Sized),
        cfg: &VecOpsConfig,
    ) -> Result<(), eIcicleError>;

    fn transpose(
        input: &(impl HostOrDeviceSlice<F> + ?Sized),
        nof_rows: u32,
        nof_cols: u32,
        output: &mut (impl HostOrDeviceSlice<F> + ?Sized),
        cfg: &VecOpsConfig,
    ) -> Result<(), eIcicleError>;

    fn bit_reverse(
        input: &(impl HostOrDeviceSlice<F> + ?Sized),
        cfg: &VecOpsConfig,
        output: &mut (impl HostOrDeviceSlice<F> + ?Sized),
    ) -> Result<(), eIcicleError>;

    fn bit_reverse_inplace(
        input: &mut (impl HostOrDeviceSlice<F> + ?Sized),
        cfg: &VecOpsConfig,
    ) -> Result<(), eIcicleError>;

    fn slice(
        input: &(impl HostOrDeviceSlice<F> + ?Sized),
        offset: u64,
        stride: u64,
        size_in: u64,
        size_out: u64,
        cfg: &VecOpsConfig,
        output: &mut (impl HostOrDeviceSlice<F> + ?Sized),
    ) -> Result<(), eIcicleError>;
}

/// Trait for mixed vector operations on different field element types
pub trait MixedVecOps<F: FieldImpl, T: FieldImpl> {
    fn mul(
        a: &(impl HostOrDeviceSlice<F> + ?Sized),
        b: &(impl HostOrDeviceSlice<T> + ?Sized),
        result: &mut (impl HostOrDeviceSlice<F> + ?Sized),
        cfg: &VecOpsConfig,
    ) -> Result<(), eIcicleError>;
}
