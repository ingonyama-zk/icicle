use ark_bn254::Fr; // Fr is the scalar field of BN254
use ark_ec::short_weierstrass::{Affine as ArkAffine, Projective as ArkProjective, SWCurveConfig};
use ark_ec::AffineRepr;
use ark_ff::{BigInteger, PrimeField, UniformRand}; // Trait that provides the `rand` method
use rand::thread_rng;

use icicle_bn254::curve::{G1Affine as IcicleBn254G1, ScalarField as IcicleBn254Scalar};
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

fn incremental_ark_scalars(size: usize) -> Vec<Fr> {
    let ark_scalars = (0..size)
        .map(|i| Fr::from(i as u64))
        .collect();
    return ark_scalars;
}

// Generic function to convert Arkworks field elements to Icicle format and return a mutable slice
// This is a zero-copy method where the memory owned by arkworks can be modified read/write in icicle
fn transmute_ark_scalars_slice<T, I>(ark_scalars: &mut [T]) -> &mut [I]
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

// Alternatively can copy and own the memory in icicle
// Function to copy a slice of Arkworks scalar elements to Icicle scalar elements.
fn copy_ark_scalars_slice<T, I>(ark_scalars: &[T]) -> Vec<I>
where
    T: PrimeField,
    I: FieldImpl + MontgomeryConvertible,
{
    ark_scalars
        .iter()
        .map(|ark_scalar| {
            // Convert each scalar element to bytes in little-endian form
            let ark_bytes: Vec<u8> = ark_scalar
                .to_base_prime_field_elements()
                .map(|x| x.into_bigint().to_bytes_le()) // Get little-endian byte representation
                .flatten()
                .collect();

            // Convert the bytes into Icicle scalar type
            I::from_bytes_le(&ark_bytes)
        })
        .collect() // Collect all Icicle scalars into a Vec
}

// fn convert_ark_to_icicle_affine<T>(ark_affine: &mut [T]) {
//     // IcicleBn254G1
// }

fn main() {
    //============================================================================================//
    //================================ Part 1: transmute ark scalars =============================//
    //============================================================================================//
    // let ark_elements = randomize_ark_field_elements(2);
    let mut ark_scalars = incremental_ark_scalars(10);
    println!("ark elements: {:?}\n", ark_scalars);

    let icicle_scalars: &mut [IcicleBn254Scalar] = transmute_ark_scalars_slice(&mut ark_scalars);
    println!("ICICLE elements: {:?}\n", icicle_scalars);

    //============================================================================================//
    //================================ Part 2: copy ark scalars =============================//
    //============================================================================================//
    let ark_scalars = [Fr::from(5 as u64)];
    println!("ark scalar=: {:?}\n", ark_scalars);
    let icicle_scalars: Vec<IcicleBn254Scalar> = copy_ark_scalars_slice(&ark_scalars);
    println!("ICICLE scalar=: {:?}\n", icicle_scalars);
}
