use ark_bn254::{Fr, G1Affine as ArkAffine, G1Projective as ArkProjective};
use ark_ec::{AffineRepr, CurveGroup};
use ark_ff::{BigInteger, PrimeField};
use std::ops::Mul;
use std::time::Instant;

use clap::Parser;
use icicle_bn254::curve::{G1Affine as IcicleAffine, G1Projective as IcicleProjective, ScalarField as IcicleScalar};
use icicle_core::traits::{FieldImpl, MontgomeryConvertible};
use icicle_runtime::{
    memory::{DeviceVec, HostSlice},
    stream::IcicleStream,
};
use rayon::prelude::*;

#[derive(Parser, Debug)]
struct Args {
    #[arg(short, long, default_value_t = 1<<15)]
    size: usize,

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

//// Generate scalars and ec points

fn incremental_ark_scalars(size: usize) -> Vec<Fr> {
    let ark_scalars = (0..size)
        .map(|i| Fr::from(i as u64))
        .collect();
    ark_scalars
}

// use ark_ff::UniformRand;
// use rand::thread_rng;
// fn randomize_ark_scalars(size: usize) -> Vec<Fr> {
//     let mut rng = thread_rng();
//     let ark_scalars = (0..size)
//         .map(|_| Fr::rand(&mut rng))
//         .collect();
//     ark_scalars
// }

fn incremental_ark_affine_points(size: usize) -> Vec<ArkAffine> {
    let ark_affine_points = (1..=size)
        .map(|i| {
            // Incremental scalars times the generator
            let scalar = Fr::from(i as u64);
            ArkAffine::generator()
                .mul(&scalar)
                .into_affine()
        })
        .collect();
    ark_affine_points
}

fn incremental_ark_projective_points(size: usize) -> Vec<ArkProjective> {
    let ark_projective_points = (1..=size)
        .map(|i| {
            // Incremental scalars times the generator
            let scalar = Fr::from(i as u64);
            ArkAffine::generator().mul(&scalar)
        })
        .collect();
    ark_projective_points
}

//// convert single field element
fn from_ark<T, I>(ark: &T) -> I
where
    T: PrimeField,
    I: FieldImpl,
{
    // Pre-allocate the memory based on the number of base field elements and their size
    let element_size = T::BigInt::NUM_LIMBS * 8; // Each limb is 8 bytes (u64)
    let num_elements = T::extension_degree() as usize; // Number of base prime field elements

    let mut ark_bytes = Vec::with_capacity(element_size * num_elements);

    // Directly iterate over the base prime field elements and append the bytes
    for base_elem in ark.to_base_prime_field_elements() {
        let bigint = base_elem.into_bigint();
        ark_bytes.extend_from_slice(&bigint.to_bytes_le()); // Append bytes directly to the buffer
    }

    // Convert the bytes to the Icicle field element
    I::from_bytes_le(&ark_bytes)
}

//// Transmute or copy scalars

// Generic function to transmute Arkworks field elements to Icicle format and return a mutable slice
fn transmute_ark_scalars_slice<T, I>(ark_scalars: &mut [T]) -> &mut [I]
where
    T: PrimeField,
    I: FieldImpl + MontgomeryConvertible,
{
    // SAFETY: Reinterpreting Arkworks field elements as Icicle-specific scalars
    let icicle_scalars = unsafe { &mut *(&mut ark_scalars[..] as *mut _ as *mut [I]) };

    // Create a HostSlice from the mutable slice
    let icicle_host_slice = HostSlice::from_mut_slice(&mut icicle_scalars[..]);

    // Convert from Montgomery representation using the Icicle type's conversion method
    I::from_mont(icicle_host_slice, &IcicleStream::default());

    // Return the mutable slice of Icicle scalars
    icicle_scalars
}

// Function to copy a slice of Arkworks scalar elements to Icicle scalar elements.
fn copy_ark_scalars_slice_async<T, I>(ark_scalars: &[T], stream: &IcicleStream) -> DeviceVec<I>
where
    T: PrimeField,
    I: FieldImpl + MontgomeryConvertible,
{
    // SAFETY: Reinterpreting Arkworks field elements as Icicle-specific scalars
    let icicle_scalars = unsafe { &*(&ark_scalars[..] as *const _ as *const [I]) };

    // Create a HostSlice from the mutable slice
    let icicle_host_slice = HostSlice::from_slice(&icicle_scalars[..]);

    let mut icicle_scalars = DeviceVec::<I>::device_malloc_async(ark_scalars.len(), &stream).unwrap();
    icicle_scalars
        .copy_from_host(&icicle_host_slice)
        .unwrap();

    // Convert from Montgomery representation using the Icicle type's conversion method
    I::from_mont(&mut icicle_scalars[..], &stream);

    // Return the mutable slice of Icicle scalars
    icicle_scalars
}

//// Copy ec points: note that this is a quite expensive operation due to different internal memory layout
/// Arkworks represents affine points as affine:{x,y,is_infinity}, projective is twisted edwards
/// ICICLE represents affine points as affine:{x,y}, projective is standard projective

fn copy_ark_scalars_slice<T, I>(ark_scalars: &[T]) -> DeviceVec<I>
where
    T: PrimeField,
    I: FieldImpl + MontgomeryConvertible,
{
    copy_ark_scalars_slice_async(ark_scalars, &IcicleStream::default())
}

fn copy_ark_affine_points(ark_affine: &[ArkAffine]) -> Vec<IcicleAffine> {
    ark_affine
        .par_iter() // parallel
        .map(|ark| IcicleAffine {
            x: from_ark(&ark.x),
            y: from_ark(&ark.y),
        })
        .collect()
}

// conversion between Jacobian used in arkworks and projective used in icicle
fn copy_ark_projective_points(ark_affine: &[ArkProjective]) -> Vec<IcicleProjective> {
    // Note: this can be accelerated on device (e.g. GPU) if we implement conversion from twisted-edwards
    ark_affine
        .par_iter() // parallel
        .map(|ark| {
            let proj_x = ark.x * ark.z;
            let proj_z = ark.z * ark.z * ark.z;
            IcicleProjective {
                x: from_ark(&proj_x),
                y: from_ark(&ark.y),
                z: from_ark(&proj_z),
            }
        })
        .collect()
}

fn main() {
    let args = Args::parse();
    println!("{:?}", args);

    try_load_and_set_backend_device(&args);

    let size = args.size;
    let print = size <= 10;
    let mut ark_scalars = incremental_ark_scalars(size);
    if print {
        println!("Ark scalars (incremental): {:?}\n", ark_scalars);
    }
    //============================================================================================//
    //================================ Part 1: copy ark scalars ==================================//
    //============================================================================================//
    let start_copy = Instant::now(); // Start timing
    let _icicle_scalars: DeviceVec<IcicleScalar> = copy_ark_scalars_slice(&ark_scalars);
    let duration_copy = start_copy.elapsed(); // End timing
    println!("Time taken for copying {} scalars: {:?}", size, duration_copy);

    // Can dispatch the copy and convert to a stream without blocking the CPU thread (for supported device such as CUDA)
    let start_copy = Instant::now(); // Start timing
    let mut stream = IcicleStream::create().unwrap();
    let icicle_scalars: DeviceVec<IcicleScalar> = copy_ark_scalars_slice_async(&ark_scalars, &stream);
    let duration_copy = start_copy.elapsed(); // End timing
    println!("Time taken for dispatching async copy: {:?}", duration_copy);

    if print {
        let mut h_icicle_scalars = vec![IcicleScalar::zero(); size];
        stream
            .synchronize()
            .unwrap();
        icicle_scalars
            .copy_to_host(HostSlice::from_mut_slice(&mut h_icicle_scalars))
            .unwrap();
        println!("ICICLE scalar (copied): {:?}\n", &h_icicle_scalars[..]);
    }

    stream
        .synchronize()
        .unwrap();
    stream
        .destroy()
        .unwrap();

    //============================================================================================//
    //========================== Part 2: transmute ark scalars in-place ==========================//
    //============================================================================================//
    // Note that this is reusing the ark-works owned memory and mutates in places (convert from Montgomery representation
    let start_transmute = Instant::now();
    let icicle_scalars: &mut [IcicleScalar] = transmute_ark_scalars_slice(&mut ark_scalars);
    let duration_transmute = start_transmute.elapsed(); // End timing
    println!("Time taken for transmuting {} scalars: {:?}", size, duration_transmute);

    if print {
        println!("ICICLE elements (transmuted): {:?}\n", icicle_scalars);
    }

    //============================================================================================//
    //================================ Part 3: copy ark affine ===================================//
    //============================================================================================//
    let ark_affine_points = incremental_ark_affine_points(size);
    if print {
        println!("Ark affine ec points (incremental): {:?}\n", ark_affine_points);
    }

    let start_copy_points = Instant::now();
    let icicle_affine_points = copy_ark_affine_points(&ark_affine_points);
    let duration_copy_points = start_copy_points.elapsed(); // End timing
    println!("Time taken for copy {} affine points: {:?}", size, duration_copy_points);
    if print {
        println!("ICICLE affine ec points (incremental): {:?}\n", icicle_affine_points);
    }

    //============================================================================================//
    //================================ Part 4: copy ark projective ===============================//
    //============================================================================================//

    // Note that arkworks is using twisted-edwards representation, which is different from ICICLE's projective.
    let ark_projective_points = incremental_ark_projective_points(size);
    if print {
        println!(
            "Ark twisted-edwards ec points (incremental): {:?}\n",
            ark_projective_points
        );
    }

    let start_copy_points = Instant::now();
    let icicle_projective_points = copy_ark_projective_points(&ark_projective_points);
    let duration_copy_points = start_copy_points.elapsed(); // End timing
    println!(
        "Time taken for copy {} projective points: {:?}",
        size, duration_copy_points
    );
    if print {
        println!(
            "ICICLE projective ec points (incremental): {:?}\n",
            icicle_projective_points
        );
    }
}
