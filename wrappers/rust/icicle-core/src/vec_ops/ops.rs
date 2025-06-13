use crate::traits::FieldImpl;
use icicle_runtime::{
    errors::eIcicleError,
    memory::HostOrDeviceSlice,
};

use super::{VecOpsConfig, VecOps, MixedVecOps};

/// Adds two vectors of scalars
pub fn add_scalars<F>(
    a: &(impl HostOrDeviceSlice<F> + ?Sized),
    b: &(impl HostOrDeviceSlice<F> + ?Sized),
    result: &mut (impl HostOrDeviceSlice<F> + ?Sized),
    cfg: &VecOpsConfig,
) -> Result<(), eIcicleError>
where
    F: FieldImpl,
    <F as FieldImpl>::Config: VecOps<F>,
{
    let cfg = super::helpers::check_vec_ops_args(a, b, result, cfg);
    <<F as FieldImpl>::Config as VecOps<F>>::add(a, b, result, &cfg)
}

/// Accumulates values from one vector into another
pub fn accumulate_scalars<F>(
    a: &mut (impl HostOrDeviceSlice<F> + ?Sized),
    b: &(impl HostOrDeviceSlice<F> + ?Sized),
    cfg: &VecOpsConfig,
) -> Result<(), eIcicleError>
where
    F: FieldImpl,
    <F as FieldImpl>::Config: VecOps<F>,
{
    let cfg = super::helpers::check_vec_ops_args(a, b, a, cfg);
    <<F as FieldImpl>::Config as VecOps<F>>::accumulate(a, b, &cfg)
}

/// Subtracts two vectors of scalars
pub fn sub_scalars<F>(
    a: &(impl HostOrDeviceSlice<F> + ?Sized),
    b: &(impl HostOrDeviceSlice<F> + ?Sized),
    result: &mut (impl HostOrDeviceSlice<F> + ?Sized),
    cfg: &VecOpsConfig,
) -> Result<(), eIcicleError>
where
    F: FieldImpl,
    <F as FieldImpl>::Config: VecOps<F>,
{
    let cfg = super::helpers::check_vec_ops_args(a, b, result, cfg);
    <<F as FieldImpl>::Config as VecOps<F>>::sub(a, b, result, &cfg)
}

/// Multiplies two vectors of scalars
pub fn mul_scalars<F>(
    a: &(impl HostOrDeviceSlice<F> + ?Sized),
    b: &(impl HostOrDeviceSlice<F> + ?Sized),
    result: &mut (impl HostOrDeviceSlice<F> + ?Sized),
    cfg: &VecOpsConfig,
) -> Result<(), eIcicleError>
where
    F: FieldImpl,
    <F as FieldImpl>::Config: VecOps<F>,
{
    let cfg = super::helpers::check_vec_ops_args(a, b, result, cfg);
    <<F as FieldImpl>::Config as VecOps<F>>::mul(a, b, result, &cfg)
}

/// Multiplies two vectors of different scalar types
pub fn mixed_mul_scalars<F, T>(
    a: &(impl HostOrDeviceSlice<F> + ?Sized),
    b: &(impl HostOrDeviceSlice<T> + ?Sized),
    result: &mut (impl HostOrDeviceSlice<F> + ?Sized),
    cfg: &VecOpsConfig,
) -> Result<(), eIcicleError>
where
    F: FieldImpl,
    <F as FieldImpl>::Config: MixedVecOps<F, T>,
{
    let cfg = super::helpers::check_vec_ops_args(a, b, result, cfg);
    <<F as FieldImpl>::Config as MixedVecOps<F, T>>::mul(a, b, result, &cfg)
}

/// Divides two vectors of scalars
pub fn div_scalars<F>(
    a: &(impl HostOrDeviceSlice<F> + ?Sized),
    b: &(impl HostOrDeviceSlice<F> + ?Sized),
    result: &mut (impl HostOrDeviceSlice<F> + ?Sized),
    cfg: &VecOpsConfig,
) -> Result<(), eIcicleError>
where
    F: FieldImpl,
    <F as FieldImpl>::Config: VecOps<F>,
{
    let cfg = super::helpers::check_vec_ops_args(a, b, result, cfg);
    <<F as FieldImpl>::Config as VecOps<F>>::div(a, b, result, &cfg)
}

/// Computes the inverse of each element in a vector
pub fn inv_scalars<F>(
    input: &(impl HostOrDeviceSlice<F> + ?Sized),
    output: &mut (impl HostOrDeviceSlice<F> + ?Sized),
    cfg: &VecOpsConfig,
) -> Result<(), eIcicleError>
where
    F: FieldImpl,
    <F as FieldImpl>::Config: VecOps<F>,
{
    let cfg = super::helpers::check_vec_ops_args(input, input, output, cfg);
    <<F as FieldImpl>::Config as VecOps<F>>::inv(input, output, &cfg)
}

/// Computes the sum of elements in a vector
pub fn sum_scalars<F>(
    a: &(impl HostOrDeviceSlice<F> + ?Sized),
    result: &mut (impl HostOrDeviceSlice<F> + ?Sized),
    cfg: &VecOpsConfig,
) -> Result<(), eIcicleError>
where
    F: FieldImpl,
    <F as FieldImpl>::Config: VecOps<F>,
{
    let cfg = super::helpers::check_vec_ops_args_reduction_ops(a, result, cfg);
    <<F as FieldImpl>::Config as VecOps<F>>::sum(a, result, &cfg)
}

/// Computes the product of elements in a vector
pub fn product_scalars<F>(
    a: &(impl HostOrDeviceSlice<F> + ?Sized),
    result: &mut (impl HostOrDeviceSlice<F> + ?Sized),
    cfg: &VecOpsConfig,
) -> Result<(), eIcicleError>
where
    F: FieldImpl,
    <F as FieldImpl>::Config: VecOps<F>,
{
    let cfg = super::helpers::check_vec_ops_args_reduction_ops(a, result, cfg);
    <<F as FieldImpl>::Config as VecOps<F>>::product(a, result, &cfg)
}

/// Adds a scalar to each element of a vector
pub fn scalar_add<F>(
    a: &(impl HostOrDeviceSlice<F> + ?Sized),
    b: &(impl HostOrDeviceSlice<F> + ?Sized),
    result: &mut (impl HostOrDeviceSlice<F> + ?Sized),
    cfg: &VecOpsConfig,
) -> Result<(), eIcicleError>
where
    F: FieldImpl,
    <F as FieldImpl>::Config: VecOps<F>,
{
    let cfg = super::helpers::check_vec_ops_args_scalar_ops(a, b, result, cfg);
    <<F as FieldImpl>::Config as VecOps<F>>::scalar_add(a, b, result, &cfg)
}

/// Subtracts a scalar from each element of a vector
pub fn scalar_sub<F>(
    a: &(impl HostOrDeviceSlice<F> + ?Sized),
    b: &(impl HostOrDeviceSlice<F> + ?Sized),
    result: &mut (impl HostOrDeviceSlice<F> + ?Sized),
    cfg: &VecOpsConfig,
) -> Result<(), eIcicleError>
where
    F: FieldImpl,
    <F as FieldImpl>::Config: VecOps<F>,
{
    let cfg = super::helpers::check_vec_ops_args_scalar_ops(a, b, result, cfg);
    <<F as FieldImpl>::Config as VecOps<F>>::scalar_sub(a, b, result, &cfg)
}

/// Multiplies each element of a vector by a scalar
pub fn scalar_mul<F>(
    a: &(impl HostOrDeviceSlice<F> + ?Sized),
    b: &(impl HostOrDeviceSlice<F> + ?Sized),
    result: &mut (impl HostOrDeviceSlice<F> + ?Sized),
    cfg: &VecOpsConfig,
) -> Result<(), eIcicleError>
where
    F: FieldImpl,
    <F as FieldImpl>::Config: VecOps<F>,
{
    let cfg = super::helpers::check_vec_ops_args_scalar_ops(a, b, result, cfg);
    <<F as FieldImpl>::Config as VecOps<F>>::scalar_mul(a, b, result, &cfg)
}

/// Transposes a matrix
pub fn transpose_matrix<F>(
    input: &(impl HostOrDeviceSlice<F> + ?Sized),
    nof_rows: u32,
    nof_cols: u32,
    output: &mut (impl HostOrDeviceSlice<F> + ?Sized),
    cfg: &VecOpsConfig,
) -> Result<(), eIcicleError>
where
    F: FieldImpl,
    <F as FieldImpl>::Config: VecOps<F>,
{
    let cfg = super::helpers::check_vec_ops_args_transpose(input, nof_rows, nof_cols, output, cfg);
    <<F as FieldImpl>::Config as VecOps<F>>::transpose(input, nof_rows, nof_cols, output, &cfg)
}

/// Performs bit reversal on a vector
pub fn bit_reverse<F>(
    input: &(impl HostOrDeviceSlice<F> + ?Sized),
    cfg: &VecOpsConfig,
    output: &mut (impl HostOrDeviceSlice<F> + ?Sized),
) -> Result<(), eIcicleError>
where
    F: FieldImpl,
    <F as FieldImpl>::Config: VecOps<F>,
{
    let cfg = super::helpers::check_vec_ops_args(input, input /*dummy*/, output, cfg);
    <<F as FieldImpl>::Config as VecOps<F>>::bit_reverse(input, &cfg, output)
}

/// Performs bit reversal on a vector in-place
pub fn bit_reverse_inplace<F>(
    input: &mut (impl HostOrDeviceSlice<F> + ?Sized),
    cfg: &VecOpsConfig,
) -> Result<(), eIcicleError>
where
    F: FieldImpl,
    <F as FieldImpl>::Config: VecOps<F>,
{
    let cfg = super::helpers::check_vec_ops_args(input, input /*dummy*/, input, cfg);
    <<F as FieldImpl>::Config as VecOps<F>>::bit_reverse_inplace(input, &cfg)
}

/// Extracts a slice from a vector
pub fn slice<F>(
    input: &(impl HostOrDeviceSlice<F> + ?Sized),
    offset: u64,
    stride: u64,
    size_in: u64,
    size_out: u64,
    cfg: &VecOpsConfig,
    output: &mut (impl HostOrDeviceSlice<F> + ?Sized),
) -> Result<(), eIcicleError>
where
    F: FieldImpl,
    <F as FieldImpl>::Config: VecOps<F>,
{
    let cfg = super::helpers::check_vec_ops_args_slice(input, offset, stride, size_in, size_out, output, cfg);
    <<F as FieldImpl>::Config as VecOps<F>>::slice(input, offset, stride, size_in, size_out, &cfg, output)
} 