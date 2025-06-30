use crate::vec_ops::VecOpsConfig;
use crate::{polynomial_ring::PolynomialRing, traits::FieldImpl};
use icicle_runtime::{errors::eIcicleError, memory::HostOrDeviceSlice};

pub mod tests;

/// Trait for performing Johnson–Lindenstrauss (JL) projection operations
/// on scalar field elements (e.g., Zq).
///
/// This trait defines two core methods:
/// - `jl_projection` projects an input vector using a pseudo-random matrix
/// - `get_jl_matrix_rows` generates the raw matrix rows deterministically
pub trait JLProjection<T: FieldImpl> {
    /// Projects the input vector into a lower-dimensional space using a
    /// deterministic JL matrix with values in {-1, 0, 1}.
    fn jl_projection(
        input: &(impl HostOrDeviceSlice<T> + ?Sized),
        seed: &[u8],
        cfg: &VecOpsConfig,
        output_projection: &mut (impl HostOrDeviceSlice<T> + ?Sized),
    ) -> Result<(), eIcicleError>;

    /// Generates raw JL matrix rows over the scalar ring `T`.
    ///
    /// The output is written in row-major order: row 0 | row 1 | ...
    fn get_jl_matrix_rows(
        seed: &[u8],
        row_size: usize,
        start_row: usize,
        num_rows: usize,
        cfg: &VecOpsConfig,
        output_rows: &mut (impl HostOrDeviceSlice<T> + ?Sized),
    ) -> Result<(), eIcicleError>;
}

/// Trait for JL matrix generation in polynomial ring form (e.g., Rq),
/// with optional polynomial conjugation.
pub trait JLProjectionPolyRing<P: PolynomialRing> {
    /// Generates JL matrix rows grouped as polynomials.
    ///
    /// Each row contains `row_size` polynomials of degree `P::DEGREE`.
    /// If `conjugate = true`, applies the transformation:
    /// a(X) ↦ a(X⁻¹) mod X^d + 1.
    fn get_jl_matrix_rows_as_polyring(
        seed: &[u8],
        row_size: usize,
        start_row: usize,
        num_rows: usize,
        conjugate: bool,
        cfg: &VecOpsConfig,
        output_rows: &mut (impl HostOrDeviceSlice<P> + ?Sized),
    ) -> Result<(), eIcicleError>;
}

/// Projects an input vector using a Johnson–Lindenstrauss (JL) random projection matrix.
///
/// This is a generic wrapper over the `JLProjection` trait implemented for `T::Config`,
/// where `T` is a `FieldImpl`. The input vector is projected into a lower-dimensional
/// space defined by the shape of `output_projection`. The projection matrix is derived
/// deterministically from the given `seed`.
///
/// # Type Parameters
/// - `T`: The scalar field element type implementing `FieldImpl`.
///
/// # Parameters
/// - `input`: The input vector to be projected. Must implement `HostOrDeviceSlice<T>`.
/// - `seed`: Seed for deterministic JL matrix generation.
/// - `cfg`: Configuration for vector operations (e.g., backend, device).
/// - `output_projection`: Mutable output vector where projected values are written.
///
/// # Returns
/// - `Ok(())` if the projection succeeded.
/// - `Err(eIcicleError)` if an error occurred (e.g., invalid dimensions or device mismatch).
pub fn jl_projection<T>(
    input: &(impl HostOrDeviceSlice<T> + ?Sized),
    seed: &[u8],
    cfg: &VecOpsConfig,
    output_projection: &mut (impl HostOrDeviceSlice<T> + ?Sized),
) -> Result<(), eIcicleError>
where
    T: FieldImpl,
    T::Config: JLProjection<T>,
{
    T::Config::jl_projection(input, seed, cfg, output_projection)
}

/// Generates raw rows from a JL projection matrix over the field type `T`.
///
/// This function uses the `JLProjection` trait implemented by `T::Config`, where `T` is a scalar field.
/// The generated matrix rows are pseudo-random, derived from a cryptographic hash of the seed
/// and row indices. Each row contains `row_size` entries, and `num_rows` rows are generated,
/// starting from `start_row`.
///
/// # Type Parameters
/// - `T`: The scalar field type implementing `FieldImpl`.
///
/// # Parameters
/// - `seed`: Seed used to deterministically generate matrix content.
/// - `row_size`: Number of elements in each row (input dimensionality).
/// - `start_row`: Index of the first row to generate.
/// - `num_rows`: Number of rows to generate.
/// - `cfg`: Vector operations configuration (e.g., backend and device preferences).
/// - `output_rows`: Output buffer where matrix rows are written, in row-major order.
///
/// # Returns
/// - `Ok(())` if the rows were generated successfully.
/// - `Err(eIcicleError)` if an error occurred (e.g., dimension mismatch or backend failure).
pub fn get_jl_matrix_rows<T>(
    seed: &[u8],
    row_size: usize,
    start_row: usize,
    num_rows: usize,
    cfg: &VecOpsConfig,
    output_rows: &mut (impl HostOrDeviceSlice<T> + ?Sized),
) -> Result<(), eIcicleError>
where
    T: FieldImpl,
    T::Config: JLProjection<T>,
{
    T::Config::get_jl_matrix_rows(seed, row_size, start_row, num_rows, cfg, output_rows)
}

/// Generates JL projection matrix rows in polynomial ring form.
///
/// This is a generic wrapper over the `JLProjectionPolyRing` trait implementation
/// for the polynomial type `P`. The projection matrix is generated deterministically
/// from the provided seed, and each row contains `row_size` polynomials of degree `P::DEGREE`.
/// The matrix is laid out in row-major order.
///
/// If `conjugate` is `true`, each polynomial is transformed via:
/// `a(X) ↦ a(X⁻¹) mod X^d + 1`
///
/// # Type Parameters
/// - `P`: A type implementing `PolynomialRing` and `JLProjectionPolyRing`.
///
/// # Parameters
/// - `seed`: Seed used to deterministically generate JL matrix content.
/// - `row_size`: Number of polynomials per row.
/// - `start_row`: Index of the first JL matrix row to generate.
/// - `num_rows`: Total number of JL matrix rows to generate.
/// - `conjugate`: Whether to apply the polynomial conjugation transformation.
/// - `cfg`: Vector operations configuration (e.g., backend settings, device targeting).
/// - `output_rows`: Output buffer to hold the resulting polynomials, in row-major order.
///
/// # Returns
/// - `Ok(())` if successful.
/// - `Err(eIcicleError)` if an error occurs during generation (e.g., backend/device issues).
pub fn get_jl_matrix_rows_as_polyring<P>(
    seed: &[u8],
    row_size: usize,
    start_row: usize,
    num_rows: usize,
    conjugate: bool,
    cfg: &VecOpsConfig,
    output_rows: &mut (impl HostOrDeviceSlice<P> + ?Sized),
) -> Result<(), eIcicleError>
where
    P: PolynomialRing + JLProjectionPolyRing<P>,
{
    P::get_jl_matrix_rows_as_polyring(seed, row_size, start_row, num_rows, conjugate, cfg, output_rows)
}

/// Implements JLProjection for a scalar ring type using FFI.
#[macro_export]
macro_rules! impl_jl_projection {
    ($prefix:literal, $scalar_type:ty, $implement_for:ty) => {
        use icicle_core::jl_projection::JLProjection;
        use icicle_core::vec_ops::VecOpsConfig;
        use icicle_runtime::eIcicleError;
        use icicle_runtime::memory::HostOrDeviceSlice;

        extern "C" {
            #[link_name = concat!($prefix, "_jl_projection")]
            fn jl_projection_ffi(
                input: *const $scalar_type,
                size: usize,
                seed: *const u8,
                seed_len: usize,
                cfg: *const VecOpsConfig,
                output: *mut $scalar_type,
                output_size: usize,
            ) -> eIcicleError;

            #[link_name = concat!($prefix, "_jl_projection_get_rows")]
            fn jl_projection_get_rows(
                seed: *const u8,
                seed_len: usize,
                row_size: usize,
                start_row: usize,
                num_rows: usize,
                cfg: *const VecOpsConfig,
                output: *mut $scalar_type,
            ) -> eIcicleError;
        }

        impl JLProjection<$scalar_type> for $implement_for {
            fn jl_projection(
                input: &(impl HostOrDeviceSlice<$scalar_type> + ?Sized),
                seed: &[u8],
                cfg: &VecOpsConfig,
                output: &mut (impl HostOrDeviceSlice<$scalar_type> + ?Sized),
            ) -> Result<(), eIcicleError> {
                if input.is_on_device() && !input.is_on_active_device() {
                    eprintln!("Input is on an inactive device");
                    return Err(eIcicleError::InvalidArgument);
                }

                if output.is_on_device() && !output.is_on_active_device() {
                    eprintln!("Output is on an inactive device");
                    return Err(eIcicleError::InvalidArgument);
                }

                let mut cfg_clone = cfg.clone();
                cfg_clone.is_a_on_device = input.is_on_device();
                cfg_clone.is_result_on_device = output.is_on_device();

                unsafe {
                    jl_projection_ffi(
                        input.as_ptr(),
                        input.len(),
                        seed.as_ptr(),
                        seed.len(),
                        &cfg_clone,
                        output.as_mut_ptr(),
                        output.len(),
                    )
                    .wrap()
                }
            }

            fn get_jl_matrix_rows(
                seed: &[u8],
                row_size: usize,
                start_row: usize,
                num_rows: usize,
                cfg: &VecOpsConfig,
                output_rows: &mut (impl HostOrDeviceSlice<$scalar_type> + ?Sized),
            ) -> Result<(), eIcicleError> {
                if output_rows.is_on_device() && !output_rows.is_on_active_device() {
                    eprintln!("Output is on an inactive device");
                    return Err(eIcicleError::InvalidArgument);
                }

                let mut cfg_clone = cfg.clone();
                cfg_clone.is_result_on_device = output_rows.is_on_device();

                unsafe {
                    jl_projection_get_rows(
                        seed.as_ptr(),
                        seed.len(),
                        row_size,
                        start_row,
                        num_rows,
                        &cfg_clone,
                        output_rows.as_mut_ptr(),
                    )
                    .wrap()
                }
            }
        }
    };
}

/// Implements JLProjectionPolyRing for a polynomial ring type using FFI.
#[macro_export]
macro_rules! impl_jl_projection_as_polyring {
    ($prefix: literal, $poly_type: ty) => {
        use icicle_core::jl_projection::JLProjectionPolyRing;

        extern "C" {
            #[link_name = concat!($prefix, "_jl_projection_get_rows_polyring")]
            fn jl_projection_get_rows_as_polyring_ffi(
                seed: *const u8,
                seed_len: usize,
                row_size: usize,
                start_row: usize,
                num_rows: usize,
                conjugate: bool,
                cfg: *const VecOpsConfig,
                output: *mut $poly_type,
            ) -> eIcicleError;
        }

        impl JLProjectionPolyRing<$poly_type> for $poly_type {
            fn get_jl_matrix_rows_as_polyring(
                seed: &[u8],
                row_size: usize,
                start_row: usize,
                num_rows: usize,
                conjugate: bool,
                cfg: &VecOpsConfig,
                output_rows: &mut (impl HostOrDeviceSlice<$poly_type> + ?Sized),
            ) -> Result<(), eIcicleError> {
                if output_rows.is_on_device() && !output_rows.is_on_active_device() {
                    eprintln!("Output is on an inactive device");
                    return Err(eIcicleError::InvalidArgument);
                }

                let mut cfg_clone = cfg.clone();
                cfg_clone.is_result_on_device = output_rows.is_on_device();

                unsafe {
                    jl_projection_get_rows_as_polyring_ffi(
                        seed.as_ptr(),
                        seed.len(),
                        row_size,
                        start_row,
                        num_rows,
                        conjugate,
                        &cfg_clone,
                        output_rows.as_mut_ptr(),
                    )
                    .wrap()
                }
            }
        }
    };
}

/// Implements unit tests for JLProjection on scalar ring types.
#[macro_export]
macro_rules! impl_jl_projection_tests {
    ($scalar_type: ident) => {
        mod test_scalar {
            use super::*;
            use icicle_core::jl_projection::tests::*;
            use icicle_runtime::test_utilities;

            /// Initializes devices before running tests.
            pub fn initialize() {
                test_utilities::test_load_and_init_devices();
                test_utilities::test_set_main_device();
            }

            #[test]
            fn test_jl_projection() {
                initialize();
                test_utilities::test_set_main_device();
                // TODO uncomment when implemented for CUDA
                // check_jl_projection::<$implemented_for, $scalar_type>();
                test_utilities::test_set_ref_device();
                check_jl_projection::<$scalar_type>();
            }
        }
    };
}

/// Implements unit tests for JLProjectionPolyRing on polynomial ring types.
#[macro_export]
macro_rules! impl_jl_projection_polyring_tests {
    ($poly_type: ident) => {
        mod test_poly {
            use super::*;
            use icicle_core::jl_projection::tests::*;
            use icicle_runtime::test_utilities;

            /// Initializes devices before running tests.
            pub fn initialize() {
                test_utilities::test_load_and_init_devices();
                test_utilities::test_set_main_device();
            }

            #[test]
            fn test_jl_projection() {
                initialize();
                test_utilities::test_set_main_device();
                check_jl_projection_polyring::<$poly_type>();
                test_utilities::test_set_ref_device();
                check_jl_projection_polyring::<$poly_type>();
            }

            #[test]
            fn test_polynomial_projection() {
                initialize();
                test_utilities::test_set_main_device();
                check_polynomial_projection::<$poly_type>();
                test_utilities::test_set_ref_device();
                check_polynomial_projection::<$poly_type>();
            }
        }
    };
}
