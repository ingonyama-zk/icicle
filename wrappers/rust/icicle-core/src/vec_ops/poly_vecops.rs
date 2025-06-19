//! # Polynomial Ring Vector Operations
//!
//! This module defines vectorized operations over slices of types implementing the [`PolynomialRing`] trait.
//!
//! Each function operates on slices of polynomials (e.g., `[Rq]`) that internally store their coefficients in a base field (e.g., `Zq`).
//! These polynomials are "flattened" at runtime to perform efficient scalar-wise operations over the coefficient field.
//!
//! The implementation supports both host and device memory via the [`HostOrDeviceSlice`] abstraction.
//!
//! ## Supported Operations
//! - `polyvec_add`: Element-wise addition of two polynomial vectors
//! - `polyvec_sub`: Element-wise subtraction of two polynomial vectors
//! - `polyvec_mul`: Element-wise multiplication of two polynomial vectors
//! - `polyvec_mul_by_scalar`: Multiply each polynomial by a scalar field element
//! - `polyvec_sum_reduce`: Reduce a vector of polynomials into a single polynomial by summing them coefficient-wise
//!
//! ## Safety
//! All vector operations rely on a guaranteed memory layout where a polynomial is backed by `[P::Base; DEGREE]`.
//! This is enforced by the [`PolynomialRing`] trait and the `reinterpret_slice` utility.

use super::{add_scalars, mul_scalars, scalar_mul, sub_scalars, sum_scalars, VecOps, VecOpsConfig};
use crate::{polynomial_ring::PolynomialRing, traits::FieldImpl};
use icicle_runtime::{
    errors::eIcicleError,
    memory::{
        reinterpret::{reinterpret_slice, reinterpret_slice_mut},
        HostOrDeviceSlice,
    },
};

/// Multiplies each polynomial in the input vector by a corresponding scalar field element.
///
/// Each `P` is a polynomial over `P::Base`, and this function scales all coefficients of `P[i]`
/// by `scalar[i]`.
///
/// # Constraints
/// - `input_scalarvec.len() == input_polyvec.len()`
/// - All memory regions must be properly allocated on the same memory space (host or device).
///
/// # Semantics
/// ```text
/// result[i] = input_polyvec[i] * input_scalarvec[i]
/// ```
///
/// # Example
/// ```ignore
/// polyvec_mul_by_scalar(&[Rq; N], &[Zq; N], &mut [Rq; N], &cfg)
/// ```
pub fn polyvec_mul_by_scalar<P>(
    input_polyvec: &(impl HostOrDeviceSlice<P> + ?Sized),
    input_scalarvec: &(impl HostOrDeviceSlice<P::Base> + ?Sized),
    result: &mut (impl HostOrDeviceSlice<P> + ?Sized),
    cfg: &VecOpsConfig,
) -> Result<(), eIcicleError>
where
    P: PolynomialRing,
    P::Base: FieldImpl,
    <P::Base as FieldImpl>::Config: VecOps<P::Base>,
{
    unsafe {
        let vec_flat = reinterpret_slice::<P, P::Base>(input_polyvec);
        let mut result_flat = reinterpret_slice_mut::<P, P::Base>(result);
        let mut local_cfg = cfg.clone();
        local_cfg.batch_size = input_scalarvec.len() as i32;
        scalar_mul(input_scalarvec, &vec_flat, &mut result_flat, &local_cfg)
    }
}

/// Computes element-wise multiplication of two polynomial vectors.
///
/// Each pair of polynomials is multiplied coefficient-wise.
///
/// # Semantics
/// ```text
/// result[i] = input_polyvec_a[i] * input_polyvec_b[i]
/// ```
pub fn polyvec_mul<P>(
    input_polyvec_a: &(impl HostOrDeviceSlice<P> + ?Sized),
    input_polyvec_b: &(impl HostOrDeviceSlice<P> + ?Sized),
    result: &mut (impl HostOrDeviceSlice<P> + ?Sized),
    cfg: &VecOpsConfig,
) -> Result<(), eIcicleError>
where
    P: PolynomialRing,
    P::Base: FieldImpl,
    <P::Base as FieldImpl>::Config: VecOps<P::Base>,
{
    unsafe {
        let vec_a_flat = reinterpret_slice::<P, P::Base>(input_polyvec_a);
        let vec_b_flat = reinterpret_slice::<P, P::Base>(input_polyvec_b);
        let mut result_flat = reinterpret_slice_mut::<P, P::Base>(result);
        mul_scalars(&vec_a_flat, &vec_b_flat, &mut result_flat, cfg)
    }
}

/// Computes element-wise addition of two polynomial vectors.
///
/// # Semantics
/// ```text
/// result[i] = input_polyvec_a[i] + input_polyvec_b[i]
/// ```
pub fn polyvec_add<P>(
    input_polyvec_a: &(impl HostOrDeviceSlice<P> + ?Sized),
    input_polyvec_b: &(impl HostOrDeviceSlice<P> + ?Sized),
    result: &mut (impl HostOrDeviceSlice<P> + ?Sized),
    cfg: &VecOpsConfig,
) -> Result<(), eIcicleError>
where
    P: PolynomialRing,
    P::Base: FieldImpl,
    <P::Base as FieldImpl>::Config: VecOps<P::Base>,
{
    unsafe {
        let vec_a_flat = reinterpret_slice::<P, P::Base>(input_polyvec_a);
        let vec_b_flat = reinterpret_slice::<P, P::Base>(input_polyvec_b);
        let mut result_flat = reinterpret_slice_mut::<P, P::Base>(result);
        add_scalars(&vec_a_flat, &vec_b_flat, &mut result_flat, cfg)
    }
}

/// Computes element-wise subtraction of two polynomial vectors.
///
/// # Semantics
/// ```text
/// result[i] = input_polyvec_a[i] - input_polyvec_b[i]
/// ```
pub fn polyvec_sub<P>(
    input_polyvec_a: &(impl HostOrDeviceSlice<P> + ?Sized),
    input_polyvec_b: &(impl HostOrDeviceSlice<P> + ?Sized),
    result: &mut (impl HostOrDeviceSlice<P> + ?Sized),
    cfg: &VecOpsConfig,
) -> Result<(), eIcicleError>
where
    P: PolynomialRing,
    P::Base: FieldImpl,
    <P::Base as FieldImpl>::Config: VecOps<P::Base>,
{
    unsafe {
        let vec_a_flat = reinterpret_slice::<P, P::Base>(input_polyvec_a);
        let vec_b_flat = reinterpret_slice::<P, P::Base>(input_polyvec_b);
        let mut result_flat = reinterpret_slice_mut::<P, P::Base>(result);
        sub_scalars(&vec_a_flat, &vec_b_flat, &mut result_flat, cfg)
    }
}

/// Reduces a vector of polynomials into a single polynomial by summing them coefficient-wise.
///
/// The output is expected to be a slice of length 1.
///
/// # Semantics
/// ```text
/// result[0] = input_polyvec.iter().sum()
/// ```
///
/// # Notes
/// - Each coefficient is reduced independently (column-wise sum).
/// - The output slice must have length 1 polynomial (`[P]`).
pub fn polyvec_sum_reduce<P>(
    input_polyvec: &(impl HostOrDeviceSlice<P> + ?Sized),
    result: &mut (impl HostOrDeviceSlice<P> + ?Sized),
    cfg: &VecOpsConfig,
) -> Result<(), eIcicleError>
where
    P: PolynomialRing,
    P::Base: FieldImpl,
    <P::Base as FieldImpl>::Config: VecOps<P::Base>,
{
    unsafe {
        let input_flat = reinterpret_slice::<P, P::Base>(input_polyvec);
        let mut result_flat = reinterpret_slice_mut::<P, P::Base>(result);
        let mut local_cfg = cfg.clone();
        local_cfg.batch_size = P::DEGREE as i32;
        local_cfg.columns_batch = true;
        sum_scalars(&input_flat, &mut result_flat, &local_cfg)
    }
}

#[macro_export]
macro_rules! impl_poly_vecops_tests {
    ($poly_type: ident) => {
        use icicle_runtime::test_utilities;

        /// Initializes devices before running tests.
        pub fn initialize() {
            test_utilities::test_load_and_init_devices();
            test_utilities::test_set_main_device();
        }

        #[test]
        fn test_polyvec_add_sub_mul() {
            initialize();
            test_utilities::test_set_main_device();
            check_poly_vecops_add_sub_mul::<$poly_type>();
            test_utilities::test_set_ref_device();
            check_poly_vecops_add_sub_mul::<$poly_type>();
        }

        #[test]
        fn test_polyvec_mul_by_scalar() {
            initialize();
            test_utilities::test_set_main_device();
            check_polyvec_mul_by_scalar::<$poly_type>();
            test_utilities::test_set_ref_device();
            check_polyvec_mul_by_scalar::<$poly_type>();
        }

        #[test]
        fn test_polyvec_sum_reduce() {
            initialize();
            test_utilities::test_set_main_device();
            check_polyvec_sum_reduce::<$poly_type>();
            test_utilities::test_set_ref_device();
            check_polyvec_sum_reduce::<$poly_type>();
        }
    };
}
