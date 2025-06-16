use crate::vec_ops::VecOpsConfig;
use crate::{polynomial_ring::PolynomialRing, traits::FieldImpl};
use icicle_runtime::{errors::eIcicleError, memory::HostOrDeviceSlice};

/// JL projection traits and tests
pub mod tests;

/// Trait for performing Johnson–Lindenstrauss (JL) projection operations
/// on scalar field elements (e.g., Zq).
///
/// This trait provides two core methods:
/// - `jl_projection`: projects an input vector to a lower-dimensional output
///   using a pseudo-random {-1, 0, 1} matrix generated from a seed.
/// - `get_jl_matrix_rows`: directly generates the raw JL matrix rows
///   encoded in the scalar ring.
///
/// JL projection is deterministic given the same seed and supports host or device memory
/// representations via the `HostOrDeviceSlice` trait.
pub trait JLProjection<T: FieldImpl> {
    /// Projects the input vector into a lower-dimensional space using a
    /// pseudo-random JL matrix with entries in {-1, 0, 1}.
    ///
    /// # Parameters
    /// - `input`: Input slice over field elements (`T`), e.g., a Zq vector.
    /// - `seed`: Cryptographic seed used to generate the matrix deterministically.
    /// - `cfg`: Configuration for vector operations (e.g., CPU/GPU backend, parallelism).
    /// - `output_projection`: Output slice to hold the projected result.
    ///
    /// # Returns
    /// - `Ok(())` on success.
    /// - `Err(eIcicleError)` if an error occurs (e.g., invalid size or memory layout).
    fn jl_projection(
        input: &(impl HostOrDeviceSlice<T> + ?Sized),
        seed: &[u8],
        cfg: &VecOpsConfig,
        output_projection: &mut (impl HostOrDeviceSlice<T> + ?Sized),
    ) -> Result<(), eIcicleError>;

    /// Generates raw JL matrix rows encoded in the scalar ring `T` (e.g., Zq).
    ///
    /// Each row has `row_size` elements, and values are generated using 2-bit decoding
    /// of a hash function seeded with the row index and user-provided seed.
    ///
    /// # Parameters
    /// - `seed`: Seed buffer for deterministic row generation.
    /// - `row_size`: Number of elements per row.
    /// - `start_row`: Index of the first JL row to generate.
    /// - `num_rows`: Total number of rows to generate.
    /// - `cfg`: Vector operation configuration.
    /// - `output_rows`: Output buffer to hold the generated matrix rows (row-major layout).
    fn get_jl_matrix_rows(
        seed: &[u8],
        row_size: usize,
        start_row: usize,
        num_rows: usize,
        cfg: &VecOpsConfig,
        output_rows: &mut (impl HostOrDeviceSlice<T> + ?Sized),
    ) -> Result<(), eIcicleError>;
}

/// Trait for generating JL matrix rows in polynomial ring form (e.g., Rq),
/// optionally applying a conjugation transformation on each polynomial.
///
/// The resulting matrix can be interpreted as rows of structured polynomials,
/// useful for lattice-based cryptographic protocols or SNARKs.
pub trait JLProjectionPolyRing<P: PolynomialRing> {
    /// Generates JL matrix rows grouped as polynomials over `P` (e.g., Rq),
    /// optionally conjugated as `a(X) ↦ a(X⁻¹) mod (X^d + 1)`.
    ///
    /// # Parameters
    /// - `seed`: Cryptographic seed for matrix generation.
    /// - `row_size`: Number of polynomials per row.
    /// - `start_row`: First row index.
    /// - `num_rows`: Number of rows to generate.
    /// - `conjugate`: If true, applies polynomial conjugation.
    /// - `cfg`: Configuration for vector operations.
    /// - `output_rows`: Output buffer for polynomial rows (row-major layout).
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

// TODO Yuval: floating functions

#[macro_export]
macro_rules! impl_jl_projection {
    // implement_for is the type for which we implement the trait.
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

            // TODO Yuval: jl_projection_get_rows_polyring
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

#[macro_export]
macro_rules! impl_jl_projection_tests {
    ($scalar_type: ident, $implemented_for: ident) => {
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
            check_jl_projection::<$implemented_for, $scalar_type>();
        }
    };
}
