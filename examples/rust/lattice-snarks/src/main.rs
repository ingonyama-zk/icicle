use clap::Parser;
use rand::{Rng, RngCore};
use std::time::Instant;

use icicle_core::{
    balanced_decomposition, jl_projection, matrix_ops,
    matrix_ops::MatMulConfig,
    negacyclic_ntt, norm,
    ntt::NTTDir,
    polynomial_ring::{flatten_polyring_slice, flatten_polyring_slice_mut, PolynomialRing},
    random_sampling,
    random_sampling::{challenge_space_polynomials_sampling, ChallengeSpacePolynomialsSampling},
    ring::IntegerRing,
    traits::GenerateRandom,
    vec_ops,
    vec_ops::{poly_vecops, VecOpsConfig},
};
use icicle_babykoala::{
    polynomial_ring::PolyRing, // polynomial ring type Zq[X]/X^64+1
    ring::ScalarRing as Zq,    // the scalar integer ring Zq (q~64b)
};
use icicle_runtime::{
    memory::{DeviceVec, HostSlice},
    IcicleError,
};

/// Command-line args
#[derive(Parser, Debug)]
struct Args {
    /// Device type (e.g., "CPU", "CUDA")
    #[arg(short, long, default_value = "CPU")]
    device_type: String,
}

/// Load runtime backend and select the device
fn try_load_and_set_backend_device(args: &Args) {
    if args.device_type != "CPU" {
        icicle_runtime::runtime::load_backend_from_env_or_default().unwrap();
    }
    println!("Setting device: {}", args.device_type);
    let device = icicle_runtime::Device::new(&args.device_type, 0);
    icicle_runtime::set_device(&device).unwrap();
}

/// Demonstrates in-place and out-of-place NTT for a polynomial ring.
fn negacyclic_ntt_example<P>(size: usize)
where
    P: PolynomialRing + negacyclic_ntt::NegacyclicNtt<P>,
{
    // Generate random input on the host
    let input = P::generate_random(size);

    // Allocate and transfer to device memory
    let mut device_input = DeviceVec::from_host_slice(&input);

    let cfg = negacyclic_ntt::NegacyclicNttConfig::default();

    println!("----------------------------------------------------------------------");

    // ----------------------------------------------------------------------
    // In-place NTT on device memory (timed)
    // ----------------------------------------------------------------------
    let start = Instant::now();
    negacyclic_ntt::ntt_inplace(&mut device_input, NTTDir::kForward, &cfg).unwrap();
    let duration = start.elapsed();
    println!(
        "[NTT] In-place forward NTT completed in {:.2?} for {} polynomials",
        duration, size
    );

    // ----------------------------------------------------------------------
    // Out-of-place NTT into host buffer (can compute to device or host memory)
    // ----------------------------------------------------------------------
    let mut output = vec![P::zero(); size];
    negacyclic_ntt::ntt(
        &device_input,
        NTTDir::kForward,
        &cfg,
        output.into_slice_mut(),
    )
    .unwrap();

    println!("[NTT] Output vector contains {} transformed elements", output.len());
}

/// Demonstrates matrix multiplication over `PolyRing` elements using device-accelerated APIs.
///
/// - Inputs A and B are interpreted as row-major matrices:
///     - A has shape `[n × m]`
///     - B has shape `[m × k]`
/// - The output C is also stored in row-major layout with shape `[n × k]`
///
/// - A and/or B can be logically transposed using configuration flags (`a_transposed`, `b_transposed`).
///   This allows computing `Aᵗ × B`, `A × Bᵗ`, or `Aᵗ × Bᵗ` without modifying the data in memory.
///
/// This API is useful for matrix-matrix multiplication, vector dot products,
/// Ajtai-style commitments, and other linear algebra primitives over polynomial rings.
fn matmul_example<P>(n: u32, m: u32, k: u32)
where
    P: PolynomialRing + matrix_ops::MatrixOps<P>,
{
    println!("----------------------------------------------------------------------");
    println!(
        "[Matmul] Computing matrix product: A ({}×{}) × B ({}×{}) = C ({}×{})",
        n, m, m, k, n, k
    );

    let a_len = (n * m) as usize;
    let b_len = (m * k) as usize;
    let c_len = (n * k) as usize;

    // Generate random host-side input matrices
    let host_a: Vec<P> = P::generate_random(a_len);
    let host_b: Vec<P> = P::generate_random(b_len);

    // Allocate device memory for inputs and output
    let device_a = DeviceVec::from_host_slice(&host_a);
    let device_b = DeviceVec::from_host_slice(&host_b);
    let mut device_c = DeviceVec::<P>::device_malloc(c_len).expect("Allocation failed");

    // Perform matrix multiplication on device: C = A × B
    let start = std::time::Instant::now();
    matrix_ops::matmul::<P>(
        &device_a,
        n,
        m,
        &device_b,
        m,
        k,
        &MatMulConfig::default(), // Can define Aᵗ or Bᵗ here assuming dims match
        &mut device_c,
    )
    .expect("Matmul failed");
    let elapsed = start.elapsed();

    println!("[Matmul] Completed in {:.2?}", elapsed);
}

/// Demonstrates matrix transpose over `PolyRing` elements using device-accelerated APIs.
///
/// - Input matrix is row-major with shape `[rows × cols]`
/// - Output matrix is also row-major and will have shape `[cols × rows]`
///
/// Useful for transposing matrices before/after multiplication, or for
/// column-major operations.
fn transpose_example<P>(rows: u32, cols: u32)
where
    P: PolynomialRing + matrix_ops::MatrixOps<P>,
{
    println!("----------------------------------------------------------------------");
    println!(
        "[Transpose] Transposing matrix of shape {}×{} → {}×{}",
        rows, cols, cols, rows
    );

    let len = (rows * cols) as usize;

    // Generate random input matrix (row-major)
    let host_input: Vec<P> = P::generate_random(len);

    // Allocate device memory
    let device_input = DeviceVec::from_host_slice(&host_input);
    let mut device_output = DeviceVec::<P>::device_malloc(len).expect("Allocation failed");

    // Transpose
    let start = std::time::Instant::now();
    matrix_ops::matrix_transpose::<P>(&device_input, rows, cols, &VecOpsConfig::default(), &mut device_output)
        .expect("Transpose failed");
    let elapsed = start.elapsed();

    println!("[Transpose] Completed in {:.2?}", elapsed);
}

/// Computes the modulus of a field element type `T` as `usize`.
fn modulus<T>() -> usize
where
    T: IntegerRing,
{
    let q_bytes = icicle_core::ring::modulus::<T>();

    assert!(
        q_bytes.len() <= std::mem::size_of::<usize>(),
        "Modulus minus one byte array should fit in 8 bytes"
    );

    // Convert the byte array to u64 (pad with zeros if needed)
    let mut padded = [0u8; std::mem::size_of::<usize>()];
    padded[..q_bytes.len()].copy_from_slice(&q_bytes);
    usize::from_le_bytes(padded)
}

/// Demonstrates balanced base decomposition and recomposition for polynomial ring elements.
/// Uses dynamic bases q^(1/t) for t ∈ {2, 4, 6}, and verifies correctness.
/// - The output vector has length `n * d`.
/// - Digits are grouped **by digit index**, not by element:
///     - The first `n` entries are the **first digit** of all elements.
///     - The next `n` entries are the **second digit** of all elements.
///     - And so on, until all `d` digits are emitted.
fn balanced_decomposition_example<P>(size: usize)
where
    P: PolynomialRing + balanced_decomposition::BalancedDecomposition<P>,
{
    println!("----------------------------------------------------------------------");
    let q = modulus::<P::Base>();
    println!("[Balanced Decomposition] q: {:#x}", q);

    let ts = [2, 4, 6];
    let bases: Vec<u32> = ts
        .iter()
        .map(|t| {
            (q as f64)
                .powf(1.0 / *t as f64)
                .floor() as u32
        })
        .collect();

    // Generate input data
    let input = P::generate_random(size);
    let mut recomposed = vec![P::zero(); size];

    let cfg = VecOpsConfig::default();

    for (i, base) in bases
        .iter()
        .enumerate()
    {
        println!("[Balanced Decomposition] Using [t = {}] Base = {}", ts[i], base);

        let digits_per_elem = balanced_decomposition::count_digits::<P>(*base);
        let decomposed_len = size * digits_per_elem as usize;

        println!(
            "[Balanced Decomposition] Elements: {} ({} digits per element → total = {})",
            size, digits_per_elem, decomposed_len
        );

        let mut decomposed = DeviceVec::<P>::device_malloc(decomposed_len).expect("Failed to allocate device memory");

        // ------------------------------
        // ⏱️ Decompose
        // ------------------------------
        let t0 = std::time::Instant::now();
        balanced_decomposition::decompose::<P>(input.into_slice(), &mut decomposed[..], *base, &cfg)
            .expect("Decomposition failed");
        let decompose_time = t0.elapsed();
        println!(
            "[Balanced Decomposition] Decomposition completed in {:.2?}",
            decompose_time
        );

        // ------------------------------
        // ⏱️ Recompose
        // ------------------------------
        let t1 = std::time::Instant::now();
        balanced_decomposition::recompose::<P>(
            &decomposed[..],
            recomposed.into_slice_mut(),
            *base,
            &cfg,
        )
        .expect("Recomposition failed");
        let recompose_time = t1.elapsed();
        println!(
            "[Balanced Decomposition] Recomposition completed in {:.2?}",
            recompose_time
        );

        // ------------------------------
        // ✅ Verification
        // ------------------------------
        assert_eq!(input, recomposed);
    }
}

/// Demonstrates JL projection for Zq vectors and polynomial ring vectors by reinterpretation.
/// - Projects a `PolyRing` vector (flattened as scalars) on the device
/// - Times the JL projection
/// - Retrieves conjugated JL matrix rows as `PolyRing` polynomials
fn jl_projection_example<P>(size: usize, projection_dim: usize)
where
    P: PolynomialRing,
    P::Base: jl_projection::JLProjection<P::Base>,
    P: jl_projection::JLProjectionPolyRing<P>,
{
    println!("----------------------------------------------------------------------");
    println!(
        "[JL Projection] Projecting {} PolyRing elements ({} scalars), output dim = {}",
        size,
        size * P::DEGREE,
        projection_dim
    );

    let cfg = VecOpsConfig::default();
    let mut seed = [0u8; 32];
    rand::thread_rng().fill(&mut seed);

    // ----------------------------------------------------------------------
    // (1) Generate input and copy to device
    // ----------------------------------------------------------------------

    let host_input: Vec<P> = P::generate_random(size);
    let device_input = DeviceVec::from_host_slice(&host_input);

    // ----------------------------------------------------------------------
    // (2) JL projection on flattened device memory
    // ----------------------------------------------------------------------

    // Reinterpret `PolyRing` as a flat slice of base field elements (`Zq`),
    // since the projection operates on individual scalars rather than polynomials.
    let zq_device_slice = flatten_polyring_slice(&device_input);
    let mut device_output = DeviceVec::<P::Base>::device_malloc(projection_dim).unwrap();

    let t_start = std::time::Instant::now();
    jl_projection::jl_projection(&zq_device_slice, &seed, &cfg, &mut device_output)
        .expect("JL projection failed on device");
    let t_elapsed = t_start.elapsed();

    println!(
        "[JL Projection] succeeded in {:.2?} ({} → {} scalars)",
        t_elapsed,
        size * P::DEGREE,
        projection_dim
    );

    // ----------------------------------------------------------------------
    // (3) Retrieve conjugated JL matrix rows as PolyRing polynomials
    // ----------------------------------------------------------------------

    let row_size = size; // number of input polynomials per row
    let num_rows = 1; // how many rows to extract (can be increased)
    let mut jl_rows = DeviceVec::<P>::device_malloc(num_rows * row_size).unwrap();

    let t_start = std::time::Instant::now();
    jl_projection::get_jl_matrix_rows_as_polyring(
        &seed,
        row_size, // Rq polynomials per row
        0,
        num_rows,
        true, // conjugated = true
        &cfg,
        &mut jl_rows,
    )
    .expect("Failed to retrieve JL matrix rows as conjugated PolyRing");
    let t_elapsed = t_start.elapsed();

    println!(
        "[JL Projection] Retrieved {} conjugated JL matrix row(s) as PolyRing in {:.2?}",
        num_rows, t_elapsed
    );

    // Note: to access raw {0, ±1} scalar matrix rows, use `get_jl_matrix_rows`.
}

/// Demonstrates and benchmarks norm bound checks for random `Zq` vectors on the device.
///
/// - Generates a random `Zq` vector with values in `[0, 2³⁰]`
/// - Computes estimated ℓ₂ and ℓ∞ norms
/// - Uploads the vector to the device and verifies:
///   - ℓ₂ norm bound passes when set just above the true norm
///   - ℓ₂ norm bound fails when set tightly to the computed norm
///   - ℓ∞ norm bound passes when interpreted as a batch of vectors
/// - Times each norm check call individually
fn norm_checking_example<T>(size: usize)
where
    T: IntegerRing + norm::Norm<T>,
{
    use std::time::Instant;

    println!("----------------------------------------------------------------------");
    println!(
        "[Norm Bound check] Generating {} random elements in [0, 2^30] and checking norm bounds...",
        size
    );

    let max_val: u32 = 1 << 30;
    let mut l2_squared: u128 = 0;
    let mut l_infinity_norm: u32 = 0;

    // Generate random input and compute norms
    let input: Vec<T> = (0..size)
        .map(|_| {
            let rand_val: u32 = rand::thread_rng().gen_range(0..=max_val);
            l2_squared += (rand_val as u128) * (rand_val as u128);
            l_infinity_norm = l_infinity_norm.max(rand_val);
            T::from_u32(rand_val)
        })
        .collect();

    use std::convert::TryInto;
    let isqrt = |n: u128| -> Result<u64, &'static str> {
        let mut x = n;
        let mut y = (x + 1) / 2;
        while y < x {
            x = y;
            y = (x + n / x) / 2;
        }
        x.try_into()
            .map_err(|_| "sqrt result does not fit in u64")
    };
    let l2_norm: u64 = isqrt(l2_squared).unwrap();
    println!("[Norm Bound check] Estimated ℓ₂ norm bound: {}", l2_norm);
    println!("[Norm Bound check] Computed ℓ∞ (max) norm: {}", l_infinity_norm);

    // Upload to device
    let device_input = DeviceVec::from_host_slice(&input);

    let cfg = VecOpsConfig::default();

    // ℓ₂ norm check — upper bound (should pass)
    let mut output = vec![false; 1];
    let upper_bound = l2_norm + 1;
    println!(
        "[Norm Bound check] Checking ℓ₂ norm with upper bound {} (should pass)",
        upper_bound
    );
    let start = Instant::now();
    norm::check_norm_bound(
        &device_input,
        norm::NormType::L2,
        upper_bound,
        &cfg,
        output.into_slice_mut(),
    )
    .expect("ℓ₂ norm bound check failed");
    println!("[Norm Bound check] ℓ₂ norm (pass case) took {:?}", start.elapsed());
    assert!(output[0], "ℓ₂ norm check failed unexpectedly");

    // ℓ₂ norm check — tight bound (should fail)
    let lower_bound = l2_norm;
    println!(
        "[Norm Bound check] Checking ℓ₂ norm with tight bound {} (should fail)",
        lower_bound
    );
    let start = Instant::now();
    norm::check_norm_bound(
        &device_input,
        norm::NormType::L2,
        lower_bound,
        &cfg,
        output.into_slice_mut(),
    )
    .expect("ℓ₂ norm bound check failed");
    println!("[Norm Bound check] ℓ₂ norm (fail case) took {:?}", start.elapsed());
    assert!(!output[0], "ℓ₂ norm check unexpectedly passed");

    // ℓ∞ norm check for batch vectors
    let batch = 4;
    let mut output = vec![false; batch];
    println!(
        "[Norm Bound check] Checking ℓ∞ norm with bound {}, batch-size {}",
        l_infinity_norm, batch
    );
    let start = Instant::now();
    norm::check_norm_bound(
        &device_input,
        norm::NormType::LInfinity,
        l_infinity_norm as u64 + 1, // upper bound
        &cfg,
        HostSlice::from_mut_slice(&mut output),
    )
    .expect("ℓ∞ norm bound check failed");
    println!("[Norm Bound check] ℓ∞ norm check took {:?}", start.elapsed());
    assert!(
        output
            .iter()
            .all(|&x| x),
        "ℓ∞ norm check failed for one or more vectors in the batch"
    );
}

/// Demonstrates pseudorandom sampling of `Zq` and `Rq` elements on the device.
///
/// - Allocates device buffers for `Zq` scalars and `Rq` polynomials (as `P`)
/// - Generates a random 32-byte seed for reproducible sampling
/// - Performs fast-mode pseudorandom sampling into a flat `Zq` buffer
/// - Reinterprets an `Rq` buffer as `Zq` coefficients and samples into it
/// - Measures and prints execution time for both cases
fn random_sampling_example<P>(size: usize)
where
    P: PolynomialRing,
    P::Base: random_sampling::RandomSampling<P::Base>,
{
    println!("----------------------------------------------------------------------");
    println!(
        "[Random Sampling] Generating {} pseudorandom elements in Zq and Rq using device sampling...",
        size
    );

    // Generate a non-zero 32-byte random seed
    let mut seed = [0u8; 32];
    rand::thread_rng().fill_bytes(&mut seed);

    let fast_mode = true;
    let cfg = VecOpsConfig::default();

    // Sample Zq elements
    let mut output_zq =
        DeviceVec::<P::Base>::device_malloc(size).expect("Failed to allocate device memory for Zq elements");

    let start = Instant::now();
    random_sampling::random_sampling(fast_mode, &seed, &cfg, &mut output_zq).expect("Zq sampling failed");
    let duration = start.elapsed();
    println!("[Random Sampling] Zq sampling completed in {:?}", duration);

    // Sample Rq polynomials by reinterpreting as Zq elements
    let mut output_rq =
        DeviceVec::<P>::device_malloc(size).expect("Failed to allocate device memory for Rq polynomials");
    let mut output_rq_coeffs = flatten_polyring_slice_mut(&mut output_rq);

    let start = Instant::now();
    random_sampling::random_sampling(fast_mode, &seed, &cfg, &mut output_rq_coeffs).expect("Rq sampling failed");
    let duration = start.elapsed();
    println!(
        "[Random Sampling] Rq sampling (as Zq coefficients) completed in {:?}",
        duration
    );
}

/// Demonstrates challenge space sampling for Rq polynomials.
/// Uses the Labrador protocol parameters: τ₁ (±1s), τ₂ (±2s), and an operator norm bound.
fn challenge_space_sampling_example<P>(size: usize)
where
    P: PolynomialRing + ChallengeSpacePolynomialsSampling<P>,
{
    println!("----------------------------------------------------------------------");
    println!("[Challenge Space Sampling] Generating {} challenge polynomials", size);

    // Labrador protocol parameters
    let tau1 = 31; // Number of ±1 coefficients
    let tau2 = 10; // Number of ±2 coefficients
    let opnorm_bound = 15; // Operator norm bound

    // Generate a non-zero 60-byte random seed
    let mut seed = [0u8; 60];
    rand::thread_rng().fill_bytes(&mut seed);

    // Allocate memory for the output polynomials
    let mut polys_from_challenge_space = DeviceVec::<P>::device_malloc(size).expect("Device allocation failed");

    // Perform the sampling
    let start = Instant::now();
    challenge_space_polynomials_sampling(
        &seed,
        &VecOpsConfig::default(),
        tau1,
        tau2,
        opnorm_bound, // Set to 0 to skip norm filtering
        &mut polys_from_challenge_space,
    )
    .expect("Challenge space sampling failed");
    let elapsed = start.elapsed();

    println!("[Challenge Space Sampling] Completed in {:.2?}", elapsed);
}

/// Demonstrates vectorized polynomial ring operations over device memory.
///
/// This example shows how to compute a random aggregation of polynomials:
/// - Randomize a vector of polynomials (`P`) and a vector of scalar field elements (`P::Base`)
/// - Perform pointwise multiplication of polynomials and scalars using `polyvec_mul_by_scalar`
/// - Reduce the resulting vector into a single polynomial with `polyvec_sum_reduce`
/// - Time both operations individually
///
/// Supported operations in `vecops` also include:
/// - `polyvec_add`, `polyvec_sub`, `polyvec_mul` for `<P, P>` element-wise operations
/// - `polyvec_mul_by_scalar` for `<P, P::Base>`
/// - `polyvec_sum_reduce` for reducing a polynomial vector into a single `P`
pub fn polynomial_vecops_example<P>(size: usize)
where
    P: PolynomialRing,
    P::Base: IntegerRing,
    P::Base: vec_ops::VecOps<P::Base> + random_sampling::RandomSampling<P::Base>,
{
    use rand::RngCore;
    use std::time::Instant;

    println!("----------------------------------------------------------------------");
    println!(
        "[Vector Ops] Demonstrating vectorized polynomial operations for {} elements",
        size
    );

    let cfg = VecOpsConfig::default();
    let fast_mode = true;

    // Generate a random seed
    let mut seed = [0u8; 32];
    rand::thread_rng().fill_bytes(&mut seed);

    // Allocate and sample a vector of polynomials
    let mut polyvec = DeviceVec::<P>::device_malloc(size).expect("Failed to allocate polyvec");
    {
        // Temporarily flatten polyvec to sample its base field coefficients
        let mut polyvec_flat = flatten_polyring_slice_mut(&mut polyvec);
        random_sampling::random_sampling(fast_mode, &seed, &cfg, &mut polyvec_flat)
            .expect("Random sampling for polyvec failed");
    }

    // Allocate and sample a vector of scalars
    let mut scalarvec = DeviceVec::<P::Base>::device_malloc(size).expect("Failed to allocate scalarvec");
    random_sampling::random_sampling(fast_mode, &seed, &cfg, &mut scalarvec)
        .expect("Random sampling for scalarvec failed");

    // Allocate result buffer for the pointwise multiplication
    let mut mul_result = DeviceVec::<P>::device_malloc(size).expect("Failed to allocate result buffer");

    println!("[Vector Ops] Performing polyvec_mul_by_scalar...");
    let start = Instant::now();
    poly_vecops::polyvec_mul_by_scalar(&polyvec, &scalarvec, &mut mul_result, &cfg)
        .expect("polyvec_mul_by_scalar failed");
    println!("[Vector Ops] polyvec_mul_by_scalar completed in {:?}", start.elapsed());

    // Allocate output for sum-reduction into a single polynomial
    let mut reduced = DeviceVec::<P>::device_malloc(1).expect("Failed to allocate reduction output");

    println!("[Vector Ops] Reducing with polyvec_sum_reduce...");
    let start = Instant::now();
    poly_vecops::polyvec_sum_reduce(&mul_result, &mut reduced, &cfg).expect("polyvec_sum_reduce failed");
    println!("[Vector Ops] polyvec_sum_reduce completed in {:?}", start.elapsed());

    println!("\nOther supported operations in `vecops` include:");
    println!("- polyvec_add, polyvec_sub, polyvec_mul for <P, P>");
    println!("- polyvec_mul_by_scalar for <P, P::Base>");
    println!("- polyvec_sum_reduce to collapse a polyvec into a single polynomial");
}

fn main() {
    println!("==================== Lattice SNARK Example ====================");

    let args = Args::parse();
    println!("Parsed arguments: {:?}", args);

    try_load_and_set_backend_device(&args);

    let size = 1 << 10; // Adjustable test size

    // ----------------------------------------------------------------------
    // (1) Integer ring: Zq
    // ----------------------------------------------------------------------
    println!("----------------------------------------------------------------------");
    let _zq_zeros: Vec<Zq> = vec![Zq::default(); size];
    let zq_random: Vec<Zq> = Zq::generate_random(size);
    println!("[Integer ring Zq] Generated {} random Zq elements", zq_random.len());

    // ----------------------------------------------------------------------
    // (2) Polynomial ring: PolyRing = Zq[X]/(X^n + 1)
    // ----------------------------------------------------------------------

    println!("----------------------------------------------------------------------");
    // Note: `PolyRing` is used for both coefficient-domain (Rq) and NTT-domain (Tq) representations.
    let _rq_zeros: Vec<PolyRing> = vec![PolyRing::zero(); size];
    let _rq_random: Vec<PolyRing> = PolyRing::generate_random(size);

    let unwrap_poly_from_slice = |result: Result<PolyRing, IcicleError>| result.unwrap();

    let rq_from_slice: Vec<PolyRing> = zq_random
        .chunks(PolyRing::DEGREE)
        .map(PolyRing::from_slice)
        .map(unwrap_poly_from_slice)
        .collect();
    println!(
        "[Polynomial Ring Rq] Converted {} Zq chunks into Rq polynomials",
        rq_from_slice.len()
    );

    // ----------------------------------------------------------------------
    // (3) Negacyclic NTT for polynomial rings
    // ----------------------------------------------------------------------

    negacyclic_ntt_example::<PolyRing>(size);

    // ----------------------------------------------------------------------
    // (4) Polynomial Ring Matrix Multiplication
    //     - Ajtai-style commitments
    //     - Dot products and vector-matrix ops
    // ----------------------------------------------------------------------

    matmul_example::<PolyRing>(size as u32 >> 3, size as u32, size as u32 >> 2);

    // ----------------------------------------------------------------------
    // (5) Balanced base decomposition for polynomial rings
    //     - Decompose and recompose using base b ≤ 2³²
    // ----------------------------------------------------------------------

    balanced_decomposition_example::<PolyRing>(size);

    // ----------------------------------------------------------------------
    // (6) Norm Checking for Integer Ring (Zq)
    //     - ℓ₂ and ℓ∞ norms over Zq vectors
    // ----------------------------------------------------------------------

    norm_checking_example::<Zq>(size);

    // ----------------------------------------------------------------------
    // (7) Johnson–Lindenstrauss Projection for Zq
    //     - Reinterpret Polynomial rings as Zq slices and project
    //     - Retrieve conjugated JL matrix rows as (conjugated) PolyRing polynomials
    //     - This is useful when proving the projection is computed correctly
    // ----------------------------------------------------------------------

    jl_projection_example::<PolyRing>(size, 256 /*projection dimension */);

    // ----------------------------------------------------------------------
    // (8) Vector APIs for Polynomial Rings
    //     - Aggregation, weighted sum, and dot-product ops
    // ----------------------------------------------------------------------

    polynomial_vecops_example::<PolyRing>(size);

    // ----------------------------------------------------------------------
    // (9) Matrix Transpose for Polynomial Rings
    // ----------------------------------------------------------------------

    transpose_example::<PolyRing>(size as u32, size as u32 >> 2);

    // ----------------------------------------------------------------------
    // (10) Random Sampling for Zq and PolyRing
    //      - Pseudorandom seeded generation
    // ----------------------------------------------------------------------

    random_sampling_example::<PolyRing>(size);

    // ----------------------------------------------------------------------
    // (11) Challenge Polynomial Sampling in PolyRing
    //      - Sample sparse polynomials (e.g., ±1, ±2, 0)
    //      - Sample polynomials with bound OpNorm (for Labrdaor and similar usecases)
    // ----------------------------------------------------------------------

    challenge_space_sampling_example::<PolyRing>(size);
}
