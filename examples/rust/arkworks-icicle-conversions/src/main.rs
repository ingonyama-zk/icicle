use clap::Parser;
use std::ops::Mul;
use std::time::Instant;

use ark_bn254::{Fq, Fr, G1Affine as ArkAffine, G1Projective as ArkProjective};
use ark_ec::{AffineRepr, CurveGroup, VariableBaseMSM};
use ark_ff::{BigInteger, Field, PrimeField as ArkPrimeField};

use icicle_bn254::curve::{G1Affine as IcicleAffine, G1Projective as IcicleProjective, ScalarField};
use icicle_core::{
    field::Field as IcicleField,
    msm::{msm, MSMConfig},
    projective::Projective,
    traits::MontgomeryConvertible,
};
use icicle_runtime::{
    memory::{DeviceVec, HostSlice, IntoIcicleSlice, IntoIcicleSliceMut},
    stream::IcicleStream,
};
use rayon::prelude::*;

#[derive(Parser, Debug)]
struct Args {
    #[arg(short, long, default_value_t = 1 << 18)]
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

fn random_ark_scalars<T: ArkPrimeField>(size: usize) -> Vec<T> {
    let mut rng = rand::thread_rng();
    (0..size)
        .map(|_| T::rand(&mut rng))
        .collect()
}

fn incremental_ark_affine_points(size: usize) -> Vec<ArkAffine> {
    (1..=size)
        .map(|i| {
            ArkAffine::generator()
                .mul(&Fr::from(i as u64))
                .into_affine()
        })
        .collect()
}

fn incremental_ark_projective_points(size: usize) -> Vec<ArkProjective> {
    (1..=size)
        .map(|i| ArkAffine::generator().mul(&Fr::from(i as u64)))
        .collect()
}

//============================================================================================//
//========================= Convert single field element ark<->ICICLE ========================//
//============================================================================================//
fn from_ark<T, I>(ark: &T) -> I
where
    T: Field,
    I: IcicleField,
{
    let mut ark_bytes = vec![];
    for base_elem in ark.to_base_prime_field_elements() {
        ark_bytes.extend_from_slice(
            &base_elem
                .into_bigint()
                .to_bytes_le(),
        );
    }
    I::from_bytes_le(&ark_bytes)
}

fn to_ark<T, I>(icicle: &I) -> T
where
    T: Field,
    I: IcicleField,
{
    T::from_random_bytes(&icicle.to_bytes_le()).unwrap()
}

//============================================================================================//
//============================ Transmute or copy ark scalars =================================//
//============================================================================================//

// Generic function to transmute Arkworks field elements to Icicle format and return a mutable slice
fn transmute_ark_to_icicle_scalars<T, I>(ark_scalars: &mut [T]) -> &mut [I]
where
    T: ArkPrimeField,
    I: IcicleField + MontgomeryConvertible,
{
    // SAFETY: Reinterpreting Arkworks field elements as Icicle-specific scalars
    let icicle_scalars = unsafe { &mut *(ark_scalars as *mut _ as *mut [I]) };

    let icicle_host_slice = icicle_scalars.into_slice_mut();

    // Convert from Montgomery representation using the Icicle type's conversion method
    I::from_mont(icicle_host_slice, &IcicleStream::default()).unwrap();

    icicle_scalars
}

fn ark_to_icicle_scalars_async<T, I>(ark_scalars: &[T], stream: &IcicleStream) -> DeviceVec<I>
where
    T: ArkPrimeField,
    I: IcicleField + MontgomeryConvertible,
{
    // SAFETY: Reinterpreting Arkworks field elements as Icicle-specific scalars
    let icicle_scalars = unsafe { &*(ark_scalars as *const _ as *const [I]) };

    // Create a HostSlice from the mutable slice
    let icicle_host_slice = icicle_scalars.into_slice();

    let mut icicle_scalars = DeviceVec::<I>::device_malloc_async(ark_scalars.len(), &stream).unwrap();
    icicle_scalars
        .copy_from_host_async(&icicle_host_slice, &stream)
        .unwrap();

    // Convert from Montgomery representation using the Icicle type's conversion method
    I::from_mont(&mut icicle_scalars, &stream).unwrap();
    icicle_scalars
}

fn ark_to_icicle_scalars<T, I>(ark_scalars: &[T]) -> DeviceVec<I>
where
    T: ArkPrimeField,
    I: IcicleField + MontgomeryConvertible,
{
    ark_to_icicle_scalars_async(ark_scalars, &IcicleStream::default()) // default stream is sync
}

// Note that you can also do the following but it's slower and we prefer the result in device memory
// fn ark_to_icicle_scalars<T, I>(ark_scalars: &[T]) -> Vec<I>
// where
//     T: PrimeField,
//     I: FieldImpl,
// {
//     ark_scalars
//         .par_iter()
//         .map(|ark| from_ark(ark))
//         .collect()
// }

// Note: can convert scalars back to Ark if need to by from_mont() or to_ark()

//============================================================================================//
//============================ Convert EC points ark<->ICICLE ================================//
//============================================================================================//

fn ark_to_icicle_affine_points(ark_affine: &[ArkAffine]) -> Vec<IcicleAffine> {
    ark_affine
        .par_iter()
        .map(|ark| IcicleAffine {
            x: from_ark(&ark.x),
            y: from_ark(&ark.y),
        })
        .collect()
}

fn ark_to_icicle_projective_points(ark_projective: &[ArkProjective]) -> Vec<IcicleProjective> {
    ark_projective
        .par_iter()
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

fn main() {
    let args = Args::parse();
    println!("{:?}", args);
    try_load_and_set_backend_device(&args);

    println!(
        "Randomizing {} scalars, affine and ark projective (actually Jacobian) points",
        args.size
    );
    let ark_scalars = random_ark_scalars(args.size);
    let ark_projective_points = incremental_ark_projective_points(args.size);
    let ark_affine_points = incremental_ark_affine_points(args.size);

    //============================================================================================//
    //================================ Part 1: copy ark scalars ==================================//
    //============================================================================================//
    let start = Instant::now();
    let icicle_scalars_dev: DeviceVec<ScalarField> = ark_to_icicle_scalars(&ark_scalars);
    let duration = start.elapsed();
    println!("Time taken for copying {} scalars: {:?}", args.size, duration);

    // Can also do it async using a stream
    let mut stream = IcicleStream::create().unwrap();
    let start = Instant::now();
    let _icicle_scalars_dev: DeviceVec<ScalarField> = ark_to_icicle_scalars_async(&ark_scalars, &stream);
    let duration = start.elapsed();
    println!("Time taken for dispatching async copy: {:?}", duration);

    stream
        .synchronize()
        .unwrap();
    stream
        .destroy()
        .unwrap();

    //============================================================================================//
    //========================= Part 2: or transmute ark scalars in-place ========================//
    //============================================================================================//
    let mut ark_scalars_copy = ark_scalars.clone(); // copy since transmute modifies the scalars in-place
    let start = Instant::now();
    let _icicle_transumated_scalars: &mut [ScalarField] = transmute_ark_to_icicle_scalars(&mut ark_scalars_copy);
    let duration = start.elapsed();
    println!("Time taken for transmuting {} scalars: {:?}", args.size, duration);

    //============================================================================================//
    //================================ Part 3: copy ark affine ===================================//
    //============================================================================================//
    let start = Instant::now();
    let icicle_affine_points = ark_to_icicle_affine_points(&ark_affine_points);
    let duration = start.elapsed();
    println!("Time taken for copying {} affine points: {:?}", args.size, duration);

    //============================================================================================//
    //================================ Part 4: copy ark projective ===============================//
    //============================================================================================//
    let start = Instant::now();
    let _icicle_projective_points = ark_to_icicle_projective_points(&ark_projective_points);
    let duration = start.elapsed();
    println!("Time taken for copying {} projective points: {:?}", args.size, duration);

    //============================================================================================//
    //================================ Part 5: compute MSM  ======================================//
    //============================================================================================//
    let start = Instant::now();
    let ark_msm_result = ArkProjective::msm(&ark_affine_points, &ark_scalars).unwrap();
    let duration = start.elapsed();
    println!("Time taken for Ark MSM: {:?}", duration);

    let mut icicle_msm_result = vec![IcicleProjective::zero()];
    let start = Instant::now();
    msm(
        &icicle_scalars_dev,
        icicle_affine_points.into_slice(),
        &MSMConfig::default(),
        icicle_msm_result.into_slice_mut(),
    )
    .unwrap();
    let duration = start.elapsed();
    let device = icicle_runtime::runtime::get_active_device().unwrap();
    println!(
        "Time taken for ICICLE ({:?}) MSM (scalars on device, points on host): {:?}",
        device, duration
    );

    // convert the ICICLE result back to Ark projective and compare
    let ark_res_from_icicle = icicle_to_ark_projective_points(&icicle_msm_result);
    assert_eq!(ark_res_from_icicle[0], ark_msm_result);

    //============================================================================================//
    //======================= Part 5b: compute MSM on device memory  =============================//
    //============================================================================================//

    let is_device_sharing_host_mem = icicle_runtime::runtime::get_device_properties()
        .unwrap()
        .using_host_memory;

    if is_device_sharing_host_mem {
        println!("Skipping MSM from device memory since device doesn't have global memory");
        return;
    }

    // transfer points to device
    let d_icicle_affine_points = DeviceVec::<IcicleAffine>::from_host_slice(&icicle_affine_points);

    let start = Instant::now();
    msm(
        &icicle_scalars_dev,
        &d_icicle_affine_points,
        &MSMConfig::default(),
        icicle_msm_result.into_slice_mut(),
    )
    .unwrap();
    let duration = start.elapsed();
    let device = icicle_runtime::runtime::get_active_device().unwrap();
    println!(
        "Time taken for ICICLE ({:?}) MSM (scalars on device, points on device): {:?}",
        device, duration
    );

    assert_eq!(ark_res_from_icicle[0], ark_msm_result);
}
