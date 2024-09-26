use ark_bn254::{Fq, Fr, G1Affine as ArkAffine, G1Projective as ArkProjective};
use ark_ec::{AffineRepr, CurveGroup, VariableBaseMSM};
use ark_ff::{BigInteger, PrimeField};
use std::ops::Mul;
use std::time::Instant;

use clap::Parser;
use icicle_bn254::curve::{G1Affine as IcicleAffine, G1Projective as IcicleProjective, ScalarField as IcicleScalar};
use icicle_core::{
    msm::{msm, MSMConfig},
    traits::{FieldImpl, MontgomeryConvertible},
};
use icicle_runtime::{
    memory::{DeviceVec, HostSlice},
    stream::IcicleStream,
};
use rayon::prelude::*;

#[derive(Parser, Debug)]
struct Args {
    #[arg(short, long, default_value_t = 1<<12)]
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

//============================================================================================//
//============================ Generate ark scalars and ec points ============================//
//============================================================================================//

fn incremental_ark_scalars<T: PrimeField>(size: usize) -> Vec<T> {
    let ark_scalars = (0..size)
        .map(|i| T::from(i as u64))
        .collect();
    ark_scalars
}

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

//============================================================================================//
//========================= Convert single field element ark<->ICICLE ========================//
//============================================================================================//
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

fn to_ark<T, I>(icicle: &I) -> T
where
    T: PrimeField,
    I: FieldImpl,
{
    T::from_random_bytes(&icicle.to_bytes_le()).unwrap()
}

//============================================================================================//
//============================ Transmute or copy ark scalars =================================//
//============================================================================================//

// Generic function to transmute Arkworks field elements to Icicle format and return a mutable slice
fn transmute_ark_to_icicle_scalars<T, I>(ark_scalars: &mut [T]) -> &mut [I]
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
fn ark_to_icicle_scalars_async<T, I>(ark_scalars: &[T], stream: &IcicleStream) -> DeviceVec<I>
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

fn ark_to_icicle_scalars<T, I>(ark_scalars: &[T]) -> DeviceVec<I>
where
    T: PrimeField,
    I: FieldImpl + MontgomeryConvertible,
{
    ark_to_icicle_scalars_async(ark_scalars, &IcicleStream::default())
}

//============================================================================================//
//============================ Convert EC points ark<->ICICLE ================================//
//============================================================================================//

//// Note that this is a quite expensive operation due to different internal memory layout
/// Affine: Arkworks represents affine points as {x:Fq ,y:Fq ,is_infinity:bool } while ICICLE is {x,y}
/// Projective: Arkworks is using Jacobian representation while ICICLE is using Projective

fn ark_to_icicle_affine_points(ark_affine: &[ArkAffine]) -> Vec<IcicleAffine> {
    ark_affine
        .par_iter() // parallel
        .map(|ark| IcicleAffine { x: from_ark(&ark.x),y: from_ark(&ark.y)})
        .collect()
}

// conversion between Jacobian used in arkworks and projective used in icicle
fn ark_to_icicle_projective_points(ark_projective: &[ArkProjective]) -> Vec<IcicleProjective> {
    // Note: this can be accelerated on device (e.g. GPU) if we implement conversion from Jacobian
    ark_projective
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

#[allow(unused)]
fn icicle_to_ark_affine_points(icicle_projective: &[IcicleAffine]) -> Vec<ArkAffine> {
    icicle_projective
        .par_iter()
        .map(|icicle| ArkAffine::new_unchecked(to_ark(&icicle.x), to_ark(&icicle.y)))
        .collect()
}

fn icicle_to_ark_projective_points(icicle_projective: &[IcicleProjective]) -> Vec<ArkProjective> {
    icicle_projective
        .par_iter()
        .map(|icicle| {
            let proj_x: Fq = to_ark(&icicle.x);
            let proj_y: Fq = to_ark(&icicle.y);
            let proj_z: Fq = to_ark(&icicle.z);

            // conversion between projective used in icicle and Jacobian used in arkworks
            let proj_x = proj_x * proj_z;
            let proj_y = proj_y * proj_z * proj_z;
            ArkProjective::new_unchecked(proj_x, proj_y, proj_z)
        })
        .collect()
}
//============================================================================================//
//============================================================================================//
//============================================================================================//

fn main() {
    let args = Args::parse();
    println!("{:?}", args);

    try_load_and_set_backend_device(&args);

    let size = args.size;
    let print = size <= 10;
    let ark_scalars = incremental_ark_scalars(size);
    if print {
        println!("Ark scalars (incremental): {:?}\n", ark_scalars);
    }
    //============================================================================================//
    //================================ Part 1: copy ark scalars ==================================//
    //============================================================================================//
    let start_copy = Instant::now(); // Start timing
    let _icicle_scalars: DeviceVec<IcicleScalar> = ark_to_icicle_scalars(&ark_scalars);
    let duration_copy = start_copy.elapsed(); // End timing
    println!("Time taken for copying {} scalars: {:?}", size, duration_copy);

    // Can dispatch the copy and convert to a stream without blocking the CPU thread (for supported device such as CUDA)
    let start_copy = Instant::now(); // Start timing
    let mut stream = IcicleStream::create().unwrap();
    let icicle_scalars: DeviceVec<IcicleScalar> = ark_to_icicle_scalars_async(&ark_scalars, &stream);
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
    let mut ark_scalars_copy = ark_scalars.clone();
    let start_transmute = Instant::now();
    let icicle_transumated_scalars: &mut [IcicleScalar] = transmute_ark_to_icicle_scalars(&mut ark_scalars_copy);
    let duration_transmute = start_transmute.elapsed(); // End timing
    println!("Time taken for transmuting {} scalars: {:?}", size, duration_transmute);

    if print {
        println!("ICICLE elements (transmuted): {:?}\n", icicle_transumated_scalars);
    }

    //============================================================================================//
    //================================ Part 3: copy ark affine ===================================//
    //============================================================================================//
    let ark_affine_points = incremental_ark_affine_points(size);
    if print {
        println!("Ark affine ec points (incremental): {:?}\n", ark_affine_points);
    }

    let start_copy_points = Instant::now();
    let icicle_affine_points = ark_to_icicle_affine_points(&ark_affine_points);
    let duration_copy_points = start_copy_points.elapsed(); // End timing
    println!("Time taken for copy {} affine points: {:?}", size, duration_copy_points);
    if print {
        println!("ICICLE affine ec points (incremental): {:?}\n", icicle_affine_points);
    }

    //============================================================================================//
    //================================ Part 4: copy ark projective ===============================//
    //============================================================================================//

    // Note that arkworks is using Jacobian representation, which is different from ICICLE's projective.
    let ark_projective_points = incremental_ark_projective_points(size);
    if print {
        println!("Ark Jacobian ec points (incremental): {:?}\n", ark_projective_points);
    }

    let start_copy_points = Instant::now();
    let icicle_projective_points = ark_to_icicle_projective_points(&ark_projective_points);
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

    //============================================================================================//
    //================================ Part 5: compute MSM  ======================================//
    //============================================================================================//
    let start_ark_msm = Instant::now();
    let ark_msm_result = ArkProjective::msm(&ark_affine_points, &ark_scalars).unwrap();
    let ark_msm_duration = start_ark_msm.elapsed(); // End timing
    println!("Time taken for ark msm : {:?}", ark_msm_duration);

    let mut icicle_msm_result = vec![IcicleProjective::zero()];
    let start_icicle_msm = Instant::now();
    msm(
        &icicle_scalars[..],
        HostSlice::from_slice(&icicle_affine_points),
        &MSMConfig::default(),
        HostSlice::from_mut_slice(&mut icicle_msm_result),
    )
    .unwrap();
    let icicle_msm_duration = start_icicle_msm.elapsed();
    println!("Time taken for ICICLE msm : {:?}", icicle_msm_duration);

    let ark_res_from_icicle = icicle_to_ark_projective_points(&icicle_msm_result);
    assert_eq!(ark_res_from_icicle[0], ark_msm_result);
}
