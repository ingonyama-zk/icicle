use clap::Parser;
use std::time::Instant;

use icicle_core::{
    balanced_decomposition,
    negacyclic_ntt::{ntt, ntt_inplace, NegacyclicNtt, NegacyclicNttConfig},
    ntt::NTTDir,
    polynomial_ring::PolynomialRing,
    traits::{FieldImpl, GenerateRandom},
    vec_ops::VecOpsConfig,
};
use icicle_labrador::{
    polynomial_ring::PolyRing,
    ring::{ScalarCfg as ZqCfg, ScalarRing as Zq},
};
use icicle_runtime::memory::{DeviceVec, HostSlice};

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
fn negacyclic_ntt<P>(size: usize)
where
    P: PolynomialRing + NegacyclicNtt<P> + GenerateRandom<P>,
    P::Base: FieldImpl,
{
    // Generate random input on the host
    let input = P::generate_random(size);

    // Allocate and transfer to device memory
    let mut device_input = DeviceVec::<P>::device_malloc(size).unwrap();
    device_input
        .copy_from_host(HostSlice::from_slice(&input))
        .unwrap();

    let cfg = NegacyclicNttConfig::default();

    println!("----------------------------------------------------------------------");

    // ----------------------------------------------------------------------
    // In-place NTT on device memory (timed)
    // ----------------------------------------------------------------------
    let start = Instant::now();
    ntt_inplace(&mut device_input, NTTDir::kForward, &cfg).unwrap();
    let duration = start.elapsed();
    println!(
        "[NTT] In-place forward NTT completed in {:.2?} for {} polynomials",
        duration, size
    );

    // ----------------------------------------------------------------------
    // Out-of-place NTT into host buffer (can compute to device or host memory)
    // ----------------------------------------------------------------------
    let mut output = vec![P::zero(); size];
    ntt(
        &device_input,
        NTTDir::kForward,
        &cfg,
        HostSlice::from_mut_slice(&mut output),
    )
    .unwrap();

    println!("[NTT] Output vector contains {} transformed elements", output.len());
}

/// Demonstrates balanced base decomposition and recomposition for polynomial ring elements.
/// Uses dynamic bases q^(1/t) for t ∈ {2, 4, 8}, and verifies correctness.
fn balanced_decomposition<P>(size: usize)
where
    P: PolynomialRing + balanced_decomposition::BalancedDecomposition<P> + GenerateRandom<P>,
{
    let q: usize = 4_289_678_649_214_369_793; // TODO: expose q from P::Base::MODULUS

    // Compute bases: q^(1/t) for t = 2, 4, 8
    let ts = [2, 4, 8];
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
        println!("----------------------------------------------------------------------");
        println!("[Balanced Decomposition] Using [t = {}] Base = {}", ts[i], base);

        let digits_per_elem = balanced_decomposition::count_digits::<P>(*base);
        let decomposed_len = size * digits_per_elem as usize;

        println!(
            "Elements: {} ({} digits per element → total = {})",
            size, digits_per_elem, decomposed_len
        );

        let mut decomposed = DeviceVec::<P>::device_malloc(decomposed_len).expect("Failed to allocate device memory");

        // ------------------------------
        // ⏱️ Decompose
        // ------------------------------
        let t0 = std::time::Instant::now();
        balanced_decomposition::decompose::<P>(HostSlice::from_slice(&input), &mut decomposed[..], *base, &cfg)
            .expect("Decomposition failed");
        let decompose_time = t0.elapsed();
        println!("Decomposition completed in {:.2?}", decompose_time);

        // ------------------------------
        // ⏱️ Recompose
        // ------------------------------
        let t1 = std::time::Instant::now();
        balanced_decomposition::recompose::<P>(
            &decomposed[..],
            HostSlice::from_mut_slice(&mut recomposed),
            *base,
            &cfg,
        )
        .expect("Recomposition failed");
        let recompose_time = t1.elapsed();
        println!("Recomposition completed in {:.2?}", recompose_time);

        // ------------------------------
        // ✅ Verification
        // ------------------------------
        assert_eq!(input, recomposed);
    }
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

    let _zq_zeros: Vec<Zq> = vec![Zq::zero(); size];
    let zq_random: Vec<Zq> = ZqCfg::generate_random(size);
    println!("Generated {} random Zq elements", zq_random.len());

    // ----------------------------------------------------------------------
    // (2) Polynomial ring: PolyRing = Zq[X]/(X^n + 1)
    // ----------------------------------------------------------------------

    // Note: `PolyRing` is used for both coefficient-domain (Rq) and NTT-domain (Tq) representations.
    let _rq_zeros: Vec<PolyRing> = vec![PolyRing::zero(); size];
    let _rq_random: Vec<PolyRing> = PolyRing::generate_random(size);

    let rq_from_slice: Vec<PolyRing> = zq_random
        .chunks(PolyRing::DEGREE)
        .map(PolyRing::from_slice)
        .collect();
    println!("Converted {} Zq chunks into Rq polynomials", rq_from_slice.len());

    // ----------------------------------------------------------------------
    // (3) Negacyclic NTT for polynomial rings
    // ----------------------------------------------------------------------

    negacyclic_ntt::<PolyRing>(size);

    // ----------------------------------------------------------------------
    // (4) Polynomial Ring Matrix Multiplication
    //     - Ajtai-style commitments
    //     - Dot products and vector-matrix ops
    // ----------------------------------------------------------------------

    // TODO

    // ----------------------------------------------------------------------
    // (5) Balanced base decomposition for polynomial rings
    //     - Decompose and recompose using base b ≤ 2³²
    // ----------------------------------------------------------------------

    balanced_decomposition::<PolyRing>(size);

    // ----------------------------------------------------------------------
    // (6) Norm Checking for Integer Ring (Zq)
    //     - ℓ₂ and ℓ∞ norms over Zq vectors
    // ----------------------------------------------------------------------

    // TODO

    // ----------------------------------------------------------------------
    // (7) Johnson–Lindenstrauss Projection for Zq
    //     - Reinterpret Polynomial rings as Zq slices and project
    // ----------------------------------------------------------------------

    // TODO

    // ----------------------------------------------------------------------
    // (8) Vector APIs for Polynomial Rings
    //     - Aggregation, weighted sum, and dot-product ops
    // ----------------------------------------------------------------------

    // TODO

    // ----------------------------------------------------------------------
    // (9) Matrix Transpose for Polynomial Rings
    // ----------------------------------------------------------------------

    // TODO

    // ----------------------------------------------------------------------
    // (10) Random Sampling for Zq and PolyRing
    //      - Pseudorandom seeded generation
    // ----------------------------------------------------------------------

    // TODO

    // ----------------------------------------------------------------------
    // (11) Challenge Polynomial Sampling in PolyRing
    //      - Sample sparse polynomials (e.g., ±1, ±2, 0)
    // ----------------------------------------------------------------------

    // TODO

    // ----------------------------------------------------------------------
    // (12) Operator Norm Testing for Polynomial Rings
    // ----------------------------------------------------------------------

    // TODO
}
