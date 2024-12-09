use clap::Parser;
use std::ops::Mul;
use std::time::Instant;

use ark_bn254::{Fq, Fr, G1Affine as ArkAffine, G1Projective as ArkJacobian};
use ark_ec::{AffineRepr, CurveGroup, VariableBaseMSM};
use ark_ff::PrimeField;

use icicle_bn254::curve::{G1Affine as IcicleAffine, G1Projective as IcicleProjective, ScalarField as IcicleScalar};
use icicle_core::{
    msm::{msm, MSMConfig},
    traits::FieldImpl,
};
use icicle_runtime::{
    memory::{DeviceVec, HostSlice},
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

fn random_ark_scalars<T: PrimeField>(size: usize) -> Vec<T> {
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

fn incremental_ark_projective_points(size: usize) -> Vec<ArkJacobian> {
    (1..=size)
        .map(|i| ArkAffine::generator().mul(&Fr::from(i as u64)))
        .collect()
}

//============================================================================================//
//========================= Convert single field element ark<->ICICLE ========================//
//============================================================================================//

// Since both arkworks and ICICLE use montgomery format, we simply copy the underlying data
// Note: We can also transmute and avoid the copy entirely (unsafe)

fn from_ark<T, I>(ark: &T) -> I
where
    T: PrimeField,
    I: FieldImpl,
{
    // Ensure the size of the output type matches the input representation
    assert_eq!(
        std::mem::size_of::<T>(),
        std::mem::size_of::<I>(),
        "Size mismatch between input and output types"
    );

    // Transmute the element and copy as is
    let raw_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(ark as *const T as *const u8, std::mem::size_of::<T>()) };
    I::from_bytes_le(raw_bytes)
}

fn to_ark<T, I>(icicle: &I) -> T
where
    T: PrimeField,
    I: FieldImpl,
{
    unsafe {
        // Ensure sizes match between `I` and `T`
        assert_eq!(
            std::mem::size_of::<I>(),
            std::mem::size_of::<T>(),
            "Size mismatch between source and target field elements"
        );

        // Transmute the element and copy as is
        let raw_icicle_bytes = icicle as *const I as *const T;
        std::ptr::read(raw_icicle_bytes)
    }
}

//============================================================================================//
//============================ Transmute or copy ark scalars =================================//
//============================================================================================//

// Generic function to transmute Arkworks field elements to Icicle format and return a mutable slice
fn transmute_ark_to_icicle_scalars<T, I>(ark_scalars: &mut [T]) -> &mut [I]
where
    T: PrimeField,
    I: FieldImpl,
{
    // SAFETY: Reinterpreting Arkworks field elements as Icicle-specific scalars
    // NOTE: both are assumed to be in Montgomery form
    let icicle_scalars = unsafe { &mut *(ark_scalars as *mut _ as *mut [I]) };
    icicle_scalars
}

// Copying to device-memory since it is faster for ICICLE backend to access

fn ark_to_icicle_scalars_async<T, I>(ark_scalars: &[T], stream: &IcicleStream) -> DeviceVec<I>
where
    T: PrimeField,
    I: FieldImpl,
{
    // SAFETY: Reinterpreting Arkworks field elements as Icicle-specific scalars
    let icicle_scalars = unsafe { &*(ark_scalars as *const _ as *const [I]) };

    // Create a HostSlice from the mutable slice
    let icicle_host_slice = HostSlice::from_slice(&icicle_scalars[..]);

    let mut icicle_scalars = DeviceVec::<I>::device_malloc_async(ark_scalars.len(), &stream).unwrap();
    icicle_scalars
        .copy_from_host_async(&icicle_host_slice, &stream)
        .unwrap();

    icicle_scalars
}

fn ark_to_icicle_scalars<T, I>(ark_scalars: &[T]) -> DeviceVec<I>
where
    T: PrimeField,
    I: FieldImpl,
{
    let icicle_scalars = ark_to_icicle_scalars_async(ark_scalars, &IcicleStream::default());
    IcicleStream::default()
        .synchronize()
        .unwrap();
    icicle_scalars
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

fn ark_to_icicle_projective_points(ark_projective: &[ArkJacobian]) -> Vec<IcicleProjective> {
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

fn icicle_to_ark_projective_points(icicle_projective: &[IcicleProjective]) -> Vec<ArkJacobian> {
    icicle_projective
        .par_iter()
        .map(|icicle| {
            let proj_x: Fq = to_ark(&icicle.x);
            let proj_y: Fq = to_ark(&icicle.y);
            let proj_z: Fq = to_ark(&icicle.z);

            // conversion between projective used in icicle and Jacobian used in arkworks
            let proj_x = proj_x * proj_z;
            let proj_y = proj_y * proj_z * proj_z;
            ArkJacobian::new_unchecked(proj_x, proj_y, proj_z)
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
    let ark_scalars = random_ark_scalars::<Fr /*=scalar field*/>(args.size);
    let ark_projective_points = incremental_ark_projective_points(args.size);
    let ark_affine_points = incremental_ark_affine_points(args.size);

    //============================================================================================//
    //================================ Part 1: copy ark scalars ==================================//
    //============================================================================================//
    let start = Instant::now();
    let icicle_scalars_dev: DeviceVec<IcicleScalar> = ark_to_icicle_scalars(&ark_scalars);
    let duration = start.elapsed();
    println!("Time taken for copying {} scalars: {:?}", args.size, duration);

    // Can also do it async using a stream
    let mut stream = IcicleStream::create().unwrap();
    let start = Instant::now();
    let _icicle_scalars_dev: DeviceVec<IcicleScalar> = ark_to_icicle_scalars_async(&ark_scalars, &stream);
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
    let _icicle_transumated_scalars: &mut [IcicleScalar] = transmute_ark_to_icicle_scalars(&mut ark_scalars_copy);
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
    let ark_msm_result = ArkJacobian::msm(&ark_affine_points, &ark_scalars).unwrap();
    let duration = start.elapsed();
    println!("Time taken for Ark MSM: {:?}", duration);

    let mut icicle_msm_result = vec![IcicleProjective::zero()];
    let start = Instant::now();
    msm(
        &icicle_scalars_dev,
        HostSlice::from_slice(&icicle_affine_points),
        &MSMConfig::default(),
        HostSlice::from_mut_slice(&mut icicle_msm_result),
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
    let mut d_icicle_affine_points = DeviceVec::<IcicleAffine>::device_malloc(icicle_affine_points.len()).unwrap();
    d_icicle_affine_points
        .copy_from_host(HostSlice::from_slice(&icicle_affine_points[..]))
        .unwrap();

    let start = Instant::now();
    msm(
        &icicle_scalars_dev, // or HostSlice::from_slice(&_icicle_transumated_scalars)
        &d_icicle_affine_points,
        &MSMConfig::default(),
        HostSlice::from_mut_slice(&mut icicle_msm_result),
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
