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
    );
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
    );
}
