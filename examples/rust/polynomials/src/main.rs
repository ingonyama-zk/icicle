use icicle_babybear::polynomials::DensePolynomial as PolynomialBabyBear;
use icicle_bn254::curve::ScalarField;
use icicle_bn254::polynomials::DensePolynomial as PolynomialBn254;

use icicle_cuda_runtime::{
    device_context::DeviceContext,
    memory::{DeviceVec, HostSlice},
};

use icicle_core::{
    ntt::{get_root_of_unity, initialize_domain},
    polynomials::UnivariatePolynomial,
    traits::{FieldImpl, GenerateRandom},
};

#[cfg(feature = "profile")]
use std::time::Instant;

use clap::Parser;

#[derive(Parser, Debug)]
struct Args {
    /// Size of NTT to run (20 for 2^20)
    #[arg(short, long, default_value_t = 20)]
    max_ntt_log_size: u8,
    #[arg(short, long, default_value_t = 15)]
    poly_log_size: u8,
}

fn init(max_ntt_size: u64) {
    // initialize NTT domain. Polynomials ops relies on NTT.
    let rou: ScalarField = get_root_of_unity(max_ntt_size);
    let ctx = DeviceContext::default();
    initialize_domain(rou, &ctx, false /*=fast twiddles mode*/).unwrap();

    // initialize the cuda backend for polynomials
    // make sure to initialize it per field
    PolynomialBn254::init_cuda_backend();
    PolynomialBabyBear::init_cuda_backend();
}

fn randomize_poly<P>(size: usize, from_coeffs: bool) -> P
where
    P: UnivariatePolynomial,
    P::Field: FieldImpl,
    P::FieldConfig: GenerateRandom<P::Field>,
{
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
    init(1 << args.max_ntt_log_size);

    // randomize three polynomials f,g,h over bn254 scalar field
    let poly_size = 1 << args.poly_log_size;
    let f = randomize_poly::<PolynomialBn254>(poly_size, true /*from random coeffs*/);
    let g = randomize_poly::<PolynomialBn254>(poly_size / 2, true /*from random coeffs*/);
    let h = randomize_poly::<PolynomialBn254>(poly_size / 4, false /*from random evaluations on rou*/);

    // randomize two polynomials over babybear field
    let f_babybear = randomize_poly::<PolynomialBabyBear>(poly_size, true /*from random coeffs*/);
    let g_babybear = randomize_poly::<PolynomialBabyBear>(poly_size / 2, true /*from random coeffs*/);

    // Arithmetic
    let t0 = &f + &g;
    let t1 = &f * &h;
    let (q, r) = t1.divide(&t0); // computes q,r for t1(x)=q(x)*t0(x)+r(x)

    let _r_babybear = &f_babybear - &g_babybear;

    // check degree
    let _r_degree = r.degree();

    // evaluate in single domain point
    let five = ScalarField::from_u32(5);
    let q_at_five = q.eval(&five);

    // evaluate on domain. Note: domain and image can be either Host or Device slice.
    // in this example domain in on host and evals on device.
    let host_domain = [five, ScalarField::from_u32(30)];
    let mut device_image = DeviceVec::<ScalarField>::cuda_malloc(host_domain.len()).unwrap();
    t1.eval_on_domain(HostSlice::from_slice(&host_domain), &mut device_image[..]);

    // slicing
    let o = h.odd();
    let e = h.even();
    let fold = &e + &(&o * &q_at_five); // e(x) + o(x)*scalar

    let _coeff = fold.get_coeff(2); // coeff of x^2
}
