use clap::Parser;

use icicle_core::polynomial_ring::PolynomialRing;
use icicle_core::traits::{FieldImpl, GenerateRandom};
use icicle_labrador::{polynomial_ring::PolyRing, ring::ScalarCfg as ZqCfg, ring::ScalarRing as Zq};
use icicle_runtime::memory::{DeviceVec, HostSlice};
use std::time::Instant;

#[derive(Parser, Debug)]
struct Args {
    /// Device type (e.g., "CPU", "CUDA")
    #[arg(short, long, default_value = "CPU")]
    device_type: String,
}

// Load backend and set device
fn try_load_and_set_backend_device(args: &Args) {
    if args.device_type != "CPU" {
        icicle_runtime::runtime::load_backend_from_env_or_default().unwrap();
    }
    println!("Setting device {}", args.device_type);
    let device = icicle_runtime::Device::new(&args.device_type, 0 /* =device_id*/);
    icicle_runtime::set_device(&device).unwrap();
}

/// Runs negacyclic NTT on device memory and verifies host-side output.
/// This function demonstrates both in-place and out-of-place NTT APIs.
use icicle_core::negacyclic_ntt::{ntt, ntt_inplace, NegacyclicNtt, NegacyclicNttConfig};
use icicle_core::ntt::NTTDir;

fn negacyclic_ntt<P>(size: usize)
where
    P: PolynomialRing + NegacyclicNtt<P> + GenerateRandom<P>,
    P::Base: FieldImpl,
{
    // Generate random polynomials on the host
    let input = P::generate_random(size);

    // Allocate and copy to device memory
    let mut device_input = DeviceVec::<P>::device_malloc(size).unwrap();
    device_input
        .copy_from_host(HostSlice::from_slice(&input))
        .unwrap();

    // Configure NTT
    let cfg = NegacyclicNttConfig::default();

    // ------------------------------
    // In-place forward NTT (timed)
    // ------------------------------
    let start = Instant::now();
    ntt_inplace(&mut device_input, NTTDir::kForward, &cfg).unwrap();
    let duration = start.elapsed();
    println!(
        "In-place NTT (device) completed in {:.2?} for {} polynomials",
        duration, size
    );

    // ------------------------------
    // Out-of-place NTT to host buffer (or a device buffer)
    // ------------------------------
    let mut output = vec![P::zero(); size];
    ntt(
        &device_input,
        NTTDir::kForward,
        &cfg,
        HostSlice::from_mut_slice(&mut output),
    )
    .unwrap();

    println!("Computed forward NTT of {} polynomial elements", size);
}

fn main() {
    println!("---------------------- Lattice Snarks Example ------------------------");
    let args = Args::parse();
    println!("{:?}", args);

    try_load_and_set_backend_device(&args);

    let size = 1 << 10; // Example vector size (adjustable)

    // -----------------------------------------------------------------------------
    // (1) Integer ring elements: Zq
    // -----------------------------------------------------------------------------

    // Initialize with zeros
    let _zq_zeros: Vec<Zq> = vec![Zq::zero(); size];

    // Generate random elements using the configured field
    let zq_random: Vec<Zq> = ZqCfg::generate_random(size);
    println!("Generated {} random Zq elements", zq_random.len());

    // Zq elements can also be constructed from u32 values, byte arrays, hex strings, etc.
    // (See `FieldImpl` trait for full construction options)

    // -----------------------------------------------------------------------------
    // (2) Polynomial ring elements: PolyRing = Zq[X]/(X^n + 1)
    // -----------------------------------------------------------------------------

    // Note: `PolyRing` is used for both coefficient-domain (Rq) and NTT-domain (Tq) representations.

    // Initialize with zeros
    let _rq_zeros: Vec<PolyRing> = vec![PolyRing::zero(); size];

    // Generate random polynomials with random coefficients
    let _rq_random: Vec<PolyRing> = PolyRing::generate_random(size);

    // Convert a flat vector of Zq elements into PolyRing polynomials
    let rq_from_slice: Vec<PolyRing> = zq_random
        .chunks(PolyRing::DEGREE)
        .map(PolyRing::from_slice)
        .collect();
    println!("Generated {} Rq elements from Zq elements", rq_from_slice.len());

    // APIs to demonstrate:
    // (1) Negacyclic NTT for polynomial rings
    negacyclic_ntt::<PolyRing>(size);
    // (2) Matmul for polynomial rings (Ajtai, dot-products, etc.)
    // (3) Balanced-decomposition for polynomial rings, with base b (up to 32bits)
    // (4) Norm check for Integer rings (Zq)
    // (5) JL-projection for Integer rings (Zq) - including reintepretation of slices
    // (6) vector-apis for polynomial rings (PolyRing) - show Zq*PolyRing vectors for aggregation and vector-sum
    // (7) Matrix transpose for polynomial rings (PolyRing)
    // (8) Random-Sampling of Integer rings (Zq) and Polynomial rings (PolyRing)
    // (9) Challenge space sampling for polynomial rings (PolyRing) - show how to sample from a challenge space
    // (10) OpNorm testing for Polynomial rings (PolyRing) - show how to test OpNorms for polynomial rings
}
