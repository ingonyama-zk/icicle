use ark_bn254::Fr; // Fr is the scalar field of BN254
use ark_ff::{PrimeField, UniformRand}; // Added BigInteger trait import
use rand::thread_rng;
use std::time::Instant;

use icicle_bn254::curve::ScalarField as IcicleBn254Scalar;
use icicle_core::traits::{FieldImpl, MontgomeryConvertible};
use icicle_runtime::{
    memory::{DeviceVec, HostSlice},
    stream::IcicleStream,
};

fn randomize_ark_scalars(size: usize) -> Vec<Fr> {
    let mut rng = thread_rng();
    let ark_scalars = (0..size)
        .map(|_| Fr::rand(&mut rng))
        .collect();
    ark_scalars
}

fn incremental_ark_scalars(size: usize) -> Vec<Fr> {
    let ark_scalars = (0..size)
        .map(|i| Fr::from(i as u64))
        .collect();
    ark_scalars
}

// Generic function to convert Arkworks field elements to Icicle format and return a mutable slice
fn transmute_ark_scalars_slice<T, I>(ark_scalars: &mut [T]) -> &mut [I]
where
    T: ark_ff::PrimeField,                // Arkworks field element must implement PrimeField
    I: FieldImpl + MontgomeryConvertible, // Icicle scalar must implement a trait to convert from Arkworks type
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

fn copy_ark_scalars_slice<T, I>(ark_scalars: &[T]) -> DeviceVec<I>
where
    T: PrimeField,
    I: FieldImpl + MontgomeryConvertible,
{
    copy_ark_scalars_slice_async(ark_scalars, &IcicleStream::default())
}

fn main() {
    //============================================================================================//
    //================================ Part 1: transmute ark scalars =============================//
    //============================================================================================//
    let mut ark_scalars = incremental_ark_scalars(1 << 20);
    // println!("Ark scalars (incremental): {:?}\n", ark_scalars);

    // Note that this is reusing the ark-works owned memory and mutates in places (convert from Montgomery representation
    let start_transmute = Instant::now(); // Start timing                                          )
    let icicle_scalars: &mut [IcicleBn254Scalar] = transmute_ark_scalars_slice(&mut ark_scalars);
    let duration_transmute = start_transmute.elapsed(); // End timing
    println!("Time taken for transmuting: {:?}", duration_transmute);

    // println!("ICICLE elements (transmuted): {:?}\n", icicle_scalars);

    //============================================================================================//
    //================================ Part 2: copy ark scalars ==================================//
    //============================================================================================//
    let start_copy = Instant::now(); // Start timing
    let icicle_scalars: DeviceVec<IcicleBn254Scalar> = copy_ark_scalars_slice(&ark_scalars);
    let duration_copy = start_copy.elapsed(); // End timing
    println!("Time taken for copying: {:?}", duration_copy);

    // Can dispatch the copy and convert to a stream without blocking the CPU thread (for supported device such as CUDA)
    let start_copy = Instant::now(); // Start timing
    let mut stream = IcicleStream::create().unwrap();
    let icicle_scalars: DeviceVec<IcicleBn254Scalar> = copy_ark_scalars_slice_async(&ark_scalars, &stream);
    let duration_copy = start_copy.elapsed(); // End timing
    println!("Time taken for dispatching async copy: {:?}", duration_copy);

    // println!("ICICLE scalar (copied): {:?}\n", icicle_scalars);

    stream
        .destroy()
        .unwrap();
}
