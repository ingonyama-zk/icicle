use clap::Parser;

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

fn main() {
    println!("---------------------- Lattice Snarks Example ------------------------");
    let args = Args::parse();
    println!("{:?}", args);

    try_load_and_set_backend_device(&args);

    // Types:
    // (1) Integer ring: Zq
    // (2) Polynomial ring: Rq (Zq[X]/(X^n + 1)) - same type for Tq (NTT domain)

    // APIs to demonstrate:
    // (1) Negacyclic NTT for polynomial rings
    // (2) Matmul for polynomial rings (Ajtai, dot-products, etc.)
    // (3) Balanced-decomposition for polynomial rings, with base b (up to 32bits)
    // (4) Norm check for Integer rings (Zq)
    // (5) JL-projection for Integer rings (Zq) - including reintepretation of slices
    // (6) vector-apis for polynomial rings (Rq) - show Zq*Rq vectors for aggregation and vector-sum
    // (7) Matrix transpose for polynomial rings (Rq)
    // (8) Random-Sampling of Integer rings (Zq) and Polynomial rings (Rq)
    // (9) Challenge space sampling for polynomial rings (Rq) - show how to sample from a challenge space
    // (10) OpNorm testing for Polynomial rings (Rq) - show how to test OpNorms for polynomial rings
}
