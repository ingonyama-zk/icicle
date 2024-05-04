use crate::ntt::IcicleResult;
use crate::traits::FieldImpl;
use icicle_cuda_runtime::device_context::DeviceContext;
use icicle_cuda_runtime::memory::{HostOrDeviceSlice, HostSlice};

use std::io::Read;
use std::path::PathBuf;
use std::{env, fs::File};

use super::{
    create_optimized_poseidon2_constants, load_optimized_poseidon2_constants, poseidon_hash_many, Poseidon2,
    Poseidon2Config, Poseidon2Constants,
};

pub fn init_poseidon<'a, F: FieldImpl>(width: u32) -> Poseidon2Constants<'a, F>
where
    <F as FieldImpl>::Config: Poseidon2<F>,
{
    let ctx = DeviceContext::default();
    let res = load_optimized_poseidon2_constants::<F>(width, &ctx);

    println!("wtf, {:?}", res.is_err());
    match res {
        Ok(t) => return t,
        Err(e) => {
            println!("EC {:?}", e.icicle_error_code);
            println!("CE {:?}", e.cuda_error.is_none());
            println!("R {:?}", e.reason);
        },
    }
    let res = res.unwrap();
    println!("wtf2");
    res
}

fn _check_poseidon_hash_many<F: FieldImpl>(width: u32, constants: Poseidon2Constants<F>) -> (F, F)
where
    <F as FieldImpl>::Config: Poseidon2<F>,
{
    let test_size = 1 << 10;
    let mut inputs = vec![F::one(); test_size * width as usize];
    let mut outputs = vec![F::zero(); test_size];

    let input_slice = HostSlice::from_mut_slice(&mut inputs);
    let output_slice = HostSlice::from_mut_slice(&mut outputs);

    let config = Poseidon2Config::default();
    poseidon_hash_many::<F>(
        input_slice,
        output_slice,
        test_size as u32,
        width as u32,
        &constants,
        &config,
    )
    .unwrap();

    let a1 = output_slice[0];
    let a2 = output_slice[output_slice.len() - 2];

    println!("first: {:?}, last: {:?}", a1, a2);
    assert_eq!(a1, a2);

    (a1, a2)
}

pub fn check_poseidon_hash_many<'a, F: FieldImpl + 'a>()
where
    <F as FieldImpl>::Config: Poseidon2<F>,
{
    let widths = [2, 3, 4, 8, 12, 16, 20, 24];
    for width in widths {
        let constants = init_poseidon::<'a, F>(width as u32);

        _check_poseidon_hash_many(width, constants);
    }
    
}

pub fn check_poseidon_kats<F: FieldImpl>(width: usize, kats: &[F])
where
    <F as FieldImpl>::Config: Poseidon2<F>,
{
    assert_eq!(width, kats.len());
    let constants = init_poseidon::<F>(width as u32);

    let batch_size = 1024;
    let mut input = vec![F::one(); width];
    let mut outputs = vec![F::zero(); width * batch_size];

    for i in 0..width {
        input[i] = F::from_u32(i as u32);
    }

    let mut inputs: Vec<F> = std::iter::repeat(input.clone())
                            .take(batch_size)
                            .flatten()
                            .collect();

    let input_slice = HostSlice::from_mut_slice(&mut inputs);
    let output_slice = HostSlice::from_mut_slice(&mut outputs);

    let config = Poseidon2Config::default();
    poseidon_hash_many::<F>(
        input_slice,
        output_slice,
        batch_size as u32,
        width as u32,
        &constants,
        &config,
    )
    .unwrap();

    for (i, val) in output_slice.iter().enumerate() {
        assert_eq!(*val, kats[i % width]);
    }
}

// pub fn check_poseidon_custom_config<F: FieldImpl>(field_bytes: usize, field_prefix: &str, partial_rounds: u32)
// where
//     <F as FieldImpl>::Config: Poseidon2<F>,
// {
//     let width = 2u32;
//     let alpha = 5u32;
//     let constants = init_poseidon::<F>(width as u32);

//     let external_rounds = 8;

//     let ctx = DeviceContext::default();
//     let cargo_manifest_dir = env!("CARGO_MANIFEST_DIR");
//     let constants_file = PathBuf::from(cargo_manifest_dir)
//         .join("tests")
//         .join(format!("{}_constants.bin", field_prefix));
//     let mut constants_buf = vec![];
//     File::open(constants_file)
//         .unwrap()
//         .read_to_end(&mut constants_buf)
//         .unwrap();

//     let mut custom_constants = vec![];
//     for chunk in constants_buf.chunks(field_bytes) {
//         custom_constants.push(F::from_bytes_le(chunk));
//     }

//     let custom_constants = create_optimized_poseidon2_constants::<F>(
//         width as u32,
//         alpha as u32,
//         &ctx,
//         external_rounds,
//         internal_rounds,
//         &mut custom_constants,
//     )
//     .unwrap();

//     let (a1, a2) = _check_poseidon_hash_many(constants);
//     let (b1, b2) = _check_poseidon_hash_many(custom_constants);

//     assert_eq!(a1, b1);
//     assert_eq!(a2, b2);
// }
