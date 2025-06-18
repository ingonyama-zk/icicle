use icicle_bls12_377::curve::ScalarField as BLS12377ScalarField;
use icicle_bn254::curve::ScalarField;
use icicle_core::field::PrimeField;
use icicle_runtime::memory::{DeviceVec, HostOrDeviceSlice, HostSlice, DeviceSlice};

use clap::Parser;
use icicle_core::{
    ntt::{self, initialize_domain},
    traits::GenerateRandom,
};
use std::convert::TryInto;
use std::time::Instant;

// Add Arkworks imports for comparison
use ark_bn254::Fr as ArkBn254Fr;
use ark_bls12_377::Fr as ArkBls12377Fr;
use ark_ff::{BigInteger, PrimeField as ArkPrimeField};
use ark_poly::domain::Radix2EvaluationDomain;
use ark_poly::EvaluationDomain;

#[derive(Parser, Debug)]
struct Args {
    /// Size of NTT to run (20 for 2^20)
    #[arg(short, long, default_value_t = 20)]
    size: u8,

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

// Add a small extension trait that exposes a convenient one-liner for moving data back to the host.
trait DeviceVecExt<T: PrimeField + Copy> {
    /// Convenience helper that copies the `DeviceVec` contents back to the host and returns them
    /// as a `Vec`.
    fn to_host_vec(&self) -> Vec<T>;
}

impl<T: PrimeField + Copy> DeviceVecExt<T> for DeviceVec<T> {
    fn to_host_vec(&self) -> Vec<T> {
        let mut host_vec = vec![T::zero(); self.len()];
        self
            .copy_to_host(HostSlice::from_mut_slice(&mut host_vec[..]))
            .unwrap();
        host_vec
    }
}

// -----------------------------------------------------------------------------
// Adapter traits that allow calling `ntt/ntt_inplace` without manual slice casts
// -----------------------------------------------------------------------------

/// Immutable adapter – turns common buffer types into something that implements
/// `HostOrDeviceSlice`, which is what the low-level API expects.
pub trait IntoIcicleSlice<'a, T: PrimeField + 'a> {
    type Out: HostOrDeviceSlice<T> + ?Sized + 'a;
    fn into_icicle_slice(self) -> &'a Self::Out;
}

/// Mutable adapter counterpart.
pub trait IntoIcicleSliceMut<'a, T: PrimeField + 'a> {
    type Out: HostOrDeviceSlice<T> + ?Sized + 'a;
    fn into_icicle_slice_mut(self) -> &'a mut Self::Out;
}

// ------------------------ host-side buffers ----------------------------------

impl<'a, T: PrimeField> IntoIcicleSlice<'a, T> for &'a [T] {
    type Out = HostSlice<T>;
    fn into_icicle_slice(self) -> &'a HostSlice<T> {
        HostSlice::from_slice(self)
    }
}

impl<'a, T: PrimeField> IntoIcicleSlice<'a, T> for &'a Vec<T> {
    type Out = HostSlice<T>;
    fn into_icicle_slice(self) -> &'a HostSlice<T> {
        HostSlice::from_slice(self.as_slice())
    }
}

impl<'a, T: PrimeField> IntoIcicleSliceMut<'a, T> for &'a mut [T] {
    type Out = HostSlice<T>;
    fn into_icicle_slice_mut(self) -> &'a mut HostSlice<T> {
        HostSlice::from_mut_slice(self)
    }
}

impl<'a, T: PrimeField> IntoIcicleSliceMut<'a, T> for &'a mut Vec<T> {
    type Out = HostSlice<T>;
    fn into_icicle_slice_mut(self) -> &'a mut HostSlice<T> {
        HostSlice::from_mut_slice(self.as_mut_slice())
    }
}

// ------------------------ device-side buffers --------------------------------

impl<'a, T: PrimeField> IntoIcicleSlice<'a, T> for &'a DeviceVec<T> {
    type Out = DeviceSlice<T>;
    fn into_icicle_slice(self) -> &'a DeviceSlice<T> {
        &**self // `Deref` to DeviceSlice
    }
}

impl<'a, T: PrimeField> IntoIcicleSliceMut<'a, T> for &'a mut DeviceVec<T> {
    type Out = DeviceSlice<T>;
    fn into_icicle_slice_mut(self) -> &'a mut DeviceSlice<T> {
        &mut **self // `DerefMut`
    }
}

fn main() {
    let args = Args::parse();
    println!("{:?}", args);

    try_load_and_set_backend_device(&args);

    println!("Running Icicle Examples: Rust NTT");
    let log_size = args.size;
    let size = 1 << log_size;
    println!(
        "---------------------- NTT size 2^{} = {} ------------------------",
        log_size, size
    );

    // Setting Bn254 points and scalars
    println!("Generating random inputs on host for bn254...");
    let mut scalars = ScalarField::generate_random(size);
    let scalars_orig = scalars.clone();
    let mut ntt_results = DeviceVec::<ScalarField>::malloc(size);

    // Setting bls12377 points and scalars
    println!("Generating random inputs on host for bls12377...");
    let mut scalars_bls12377 = BLS12377ScalarField::generate_random(size);
    let scalars_bls12377_orig = scalars_bls12377.clone();
    let mut ntt_results_bls12377 = DeviceVec::<BLS12377ScalarField>::malloc(size);

    println!("Setting up bn254 Domain...");
    initialize_domain(
        ntt::get_root_of_unity::<ScalarField>(
            size.try_into()
                .unwrap(),
        ),
        &ntt::NTTInitDomainConfig::default(),
    )
    .unwrap();

    println!("Configuring bn254 NTT...");
    let cfg = ntt::NTTConfig::<ScalarField>::default();

    println!("Setting up bls12377 Domain...");
    initialize_domain(
        ntt::get_root_of_unity::<BLS12377ScalarField>(
            size.try_into()
                .unwrap(),
        ),
        &ntt::NTTInitDomainConfig::default(),
    )
    .unwrap();

    println!("Configuring bls12377 NTT...");
    let cfg_bls12377 = ntt::NTTConfig::<BLS12377ScalarField>::default();

    println!("Executing bn254 NTT on device...");
    let start = Instant::now();
    ntt::ntt(
        scalars.into_icicle_slice(),
        ntt::NTTDir::kForward,
        &cfg,
        ntt_results.into_icicle_slice_mut(),
    )
    .unwrap();
    println!(
        "ICICLE BN254 NTT on size 2^{log_size} took: {} μs",
        start
            .elapsed()
            .as_micros()
    );

    println!("Moving results to host...");
    let host_bn254_results = ntt_results.to_host_vec();
    assert_ne!(host_bn254_results, scalars); // check that the results are not the same as the inputs

    ntt::ntt_inplace(
        scalars.into_icicle_slice_mut(),
        ntt::NTTDir::kForward,
        &cfg,
    )
    .unwrap();

    assert_eq!(host_bn254_results, scalars); // check that the results are the same as the inputs after inplace

    //==================== BLS12-377 ====================
    println!("Executing bls12377 NTT on device...");
    let start = Instant::now();
    ntt::ntt(
        scalars_bls12377.into_icicle_slice(),
        ntt::NTTDir::kForward,
        &cfg_bls12377,
        ntt_results_bls12377.into_icicle_slice_mut(),
    )
    .unwrap();
    println!(
        "ICICLE Bls12377 NTT on size 2^{log_size} took: {} μs",
        start
            .elapsed()
            .as_micros()
    );

    let host_bls12377_results = ntt_results_bls12377.to_host_vec();
    assert_ne!(host_bls12377_results, scalars_bls12377); // check that the results are not the same as the inputs

    ntt::ntt_inplace(
        scalars_bls12377.into_icicle_slice_mut(),
        ntt::NTTDir::kForward,
        &cfg_bls12377,
    )
    .unwrap();

    assert_eq!(host_bls12377_results, scalars_bls12377);

    //==================== Arkworks comparison ====================
    println!("Comparing results with Arkworks...");
    // BN254
    let mut ark_bn_data: Vec<ArkBn254Fr> = scalars_orig
        .iter()
        .map(|x| {
            let bytes = x.to_bytes_le();
            ArkBn254Fr::from_le_bytes_mod_order(bytes.as_slice())
        })
        .collect();
    let domain_bn = Radix2EvaluationDomain::<ArkBn254Fr>::new(size).unwrap();
    domain_bn.fft_in_place(&mut ark_bn_data);
    let ark_bn_as_icicle: Vec<ScalarField> = ark_bn_data
        .iter()
        .map(|x| {
            let bytes = x.into_bigint().to_bytes_le();
            ScalarField::from_bytes_le(bytes.as_slice())
        })
        .collect();
    assert_eq!(ark_bn_as_icicle, host_bn254_results);

    // BLS12-377
    let mut ark_bls_data: Vec<ArkBls12377Fr> = scalars_bls12377_orig
        .iter()
        .map(|x| {
            let bytes = x.to_bytes_le();
            ArkBls12377Fr::from_le_bytes_mod_order(bytes.as_slice())
        })
        .collect();
    let domain_bls = Radix2EvaluationDomain::<ArkBls12377Fr>::new(size).unwrap();
    domain_bls.fft_in_place(&mut ark_bls_data);
    let ark_bls_as_icicle: Vec<BLS12377ScalarField> = ark_bls_data
        .iter()
        .map(|x| {
            let bytes = x.into_bigint().to_bytes_le();
            BLS12377ScalarField::from_bytes_le(bytes.as_slice())
        })
        .collect();
    assert_eq!(ark_bls_as_icicle, host_bls12377_results);

    println!("All comparisons with Arkworks passed ✅");

    println!("Done!");
}
