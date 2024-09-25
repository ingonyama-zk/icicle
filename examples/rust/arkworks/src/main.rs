use ark_bn254::Fr; // Fr is the scalar field of BN254
use ark_ff::UniformRand; // Trait that provides the `rand` method
use rand::thread_rng;

use icicle_bn254::curve::ScalarField as IcicleBn254Scalar;
use icicle_core::traits::FieldImpl;
use icicle_core::traits::MontgomeryConvertible;
use icicle_runtime::{memory::HostSlice, stream::IcicleStream};

fn randomize_ark_scalars(size: usize) -> Vec<Fr> {
    let mut rng = thread_rng();
    let ark_scalars = (0..size)
        .map(|_| Fr::rand(&mut rng))
        .collect();
    return ark_scalars;
}

fn increment_ark_scalars(size: usize) -> Vec<Fr> {
    let ark_scalars = (0..size)
        .map(|i| Fr::from(i as u64))
        .collect();
    return ark_scalars;
}

// Generic function to convert Arkworks field elements to Icicle format and return a mutable slice
fn transmute_ark_to_icicle_scalars<T, I>(ark_scalars: &mut [T]) -> &mut [I]
where
    T: ark_ff::PrimeField,                // Arkworks field element must implement PrimeField
    I: FieldImpl + MontgomeryConvertible, // Icicle scalar must implement a trait to convert from Arkworks type
{
    // SAFETY: Reinterpreting Arkworks field elements as Icicle-specific scalars
    let icicle_scalars = unsafe { &mut *(&mut ark_scalars[..] as *mut _ as *mut [I]) };

    // Create a HostSlice from the mutable slice
    let icicle_host_slice = HostSlice::from_mut_slice(&mut icicle_scalars[..]);

    // Optional: Convert from Montgomery representation using the Icicle type's conversion method
    I::from_mont(icicle_host_slice, &IcicleStream::default());

    // Return the mutable slice of Icicle scalars
    icicle_scalars
}

fn main() {
    //============================================================================================//
    //================================ Part 1: transmute ark scalars =============================//
    //============================================================================================//
    // let ark_elements = randomize_ark_field_elements(2);
    let mut ark_scalars = increment_ark_scalars(10);
    println!("ark elements: {:?}\n", ark_scalars);

    let icicle_scalars = transmute_ark_to_icicle_scalars::<Fr, IcicleBn254Scalar>(&mut ark_scalars);
    println!("ICICLE elements: {:?}\n", icicle_scalars);
}
