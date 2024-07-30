use icicle_babybear::field::ScalarField as babybearScalar;
use icicle_babybear::polynomials::DensePolynomial as PolynomialBabyBear;
use icicle_bn254::curve::ScalarField as bn254Scalar;
use icicle_bn254::polynomials::DensePolynomial as PolynomialBn254;

use icicle_runtime::memory::{DeviceVec, HostSlice};

use icicle_core::{
    ntt::{get_root_of_unity, initialize_domain, NTTInitDomainConfig},
    polynomials::UnivariatePolynomial,
    traits::{FieldImpl, GenerateRandom},
};

use clap::Parser;
use std::time::Instant;

#[derive(Parser, Debug)]
struct Args {
    /// Size of NTT to run (20 for 2^20)
    #[arg(short, long, default_value_t = 20)]
    max_ntt_log_size: u8,
    #[arg(short, long, default_value_t = 15)]
    poly_log_size: u8,

    /// Device type (e.g., "CPU", "CUDA")
    #[arg(short, long, default_value = "CPU")]
    device_type: String,

    /// Backend installation directory
    #[arg(short, long, default_value = "/opt/icicle/backend")]
    backend_install_dir: String,
}

// Load backend and set device
fn try_load_and_set_backend_device(args: &Args) {
    if !args
        .backend_install_dir
        .is_empty()
    {
        println!("Trying to load backend from {}", &args.backend_install_dir);
        icicle_runtime::runtime::load_backend(&args.backend_install_dir).unwrap();
    }
    println!("Setting device {}", args.device_type);
    let device = icicle_runtime::Device::new(&args.device_type, 0 /* =device_id*/);
    icicle_runtime::set_device(&device).unwrap();
}

fn init_ntt_domain(max_ntt_size: u64) {
    // Initialize NTT domain for all fields. Polynomial operations rely on NTT.
    println!(
        "Initializing NTT domain for max size 2^{}",
        max_ntt_size.trailing_zeros()
    );
    let rou_bn254: bn254Scalar = get_root_of_unity(max_ntt_size);
    initialize_domain(rou_bn254, &NTTInitDomainConfig::default()).unwrap();

    let rou_babybear: babybearScalar = get_root_of_unity(max_ntt_size);
    initialize_domain(rou_babybear, &NTTInitDomainConfig::default()).unwrap();
}

fn randomize_poly<P>(size: usize, from_coeffs: bool) -> P
where
    P: UnivariatePolynomial,
    P::Field: FieldImpl,
    P::FieldConfig: GenerateRandom<P::Field>,
{
    println!("Randomizing polynomial of size {} (from_coeffs: {})", size, from_coeffs);
    let coeffs_or_evals = P::FieldConfig::generate_random(size);
    let p = if from_coeffs {
        P::from_coeffs(HostSlice::from_slice(&coeffs_or_evals), size)
    } else {
        P::from_rou_evals(HostSlice::from_slice(&coeffs_or_evals), size)
    };
    p
}

fn main() {
    let args = Args::parse();
    println!("{:?}", args);

    try_load_and_set_backend_device(&args);

    init_ntt_domain(1 << args.max_ntt_log_size);

    let poly_size = 1 << args.poly_log_size;

    println!("Randomizing polynomials [f(x),g(x),h(x)] over bn254 scalar field...");
    let f = randomize_poly::<PolynomialBn254>(poly_size, true /*from random coeffs*/);
    let g = randomize_poly::<PolynomialBn254>(poly_size / 2, true /*from random coeffs*/);
    let h = randomize_poly::<PolynomialBn254>(poly_size / 4, false /*from random evaluations on rou*/);

    println!("Randomizing polynomials [f_babybear(x), g_babyber(x)] over babybear field...");
    let f_babybear = randomize_poly::<PolynomialBabyBear>(poly_size, true /*from random coeffs*/);
    let g_babybear = randomize_poly::<PolynomialBabyBear>(poly_size / 2, true /*from random coeffs*/);

    let start = Instant::now();
    // Arithmetic
    println!("Computing t0(x) = f(x) + g(x)");
    let t0 = &f + &g;
    println!("Computing t1(x) f(x) * h(x)");
    let t1 = &f * &h;
    println!("Computing q(x),r(x) = t1(x)/t0(x) (where t1(x) = q(x) * t0(x) + r(x))");
    let (q, r) = t1.divide(&t0);

    println!("Computing f_babybear(x) * g_babybear(x)");
    let _r_babybear = &f_babybear * &g_babybear;

    // Check degree
    println!("Degree of r(x): {}", r.degree());

    // Evaluate in single domain point
    let five = bn254Scalar::from_u32(5);
    println!("Evaluating q(5)");
    let q_at_five = q.eval(&five);

    // Evaluate on domain
    let host_domain = [five, bn254Scalar::from_u32(30)];
    let mut device_image = DeviceVec::<bn254Scalar>::device_malloc(host_domain.len()).unwrap();
    println!("Evaluating t1(x) on domain {:?}", host_domain);
    t1.eval_on_domain(HostSlice::from_slice(&host_domain), &mut device_image[..]); // for NTT use eval_on_rou_domain()

    // Slicing
    println!("Performing slicing operations on h");
    let o = h.odd();
    let e = h.even();
    let fold = &e + &(&o * &q_at_five); // e(x) + o(x) * scalar

    let _coeff = fold.get_coeff(2); // Coeff of x^2

    println!(
        "Polynomial computation on selected device took: {} ms",
        start
            .elapsed()
            .as_millis()
    );
}
