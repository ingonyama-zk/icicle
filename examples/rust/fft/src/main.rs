use icicle_bn254::curve::ScalarField as F;
use icicle_core::traits::FieldImpl;
use icicle_core::traits::MontgomeryConvertible;
use icicle_cuda_runtime::memory::HostOrDeviceSlice;
use std::fs;
use std::os::unix::fs::FileExt;

use ark_ff::BigInteger;
use ark_ff::Fp;
use ark_ff::{BigInt, Field, PrimeField};
use num_bigint::BigUint;

use icicle_core::fft::{fft_evaluate, fft_interpolate};

pub type FrBN254 = ark_bn254::Fr;

#[cfg(feature = "profile")]
use std::time::Instant;

fn read_ws(name: &str, n: usize, size: usize) -> Vec<F> {
    let file = fs::OpenOptions::new()
        .read(true)
        .open(name)
        .unwrap();

    let mut ret: Vec<F> = Vec::with_capacity(n - 1);

    for i in 0..n - 1 {
        let mut buf: Vec<u8> = vec![0; size];
        let index = (i * size) as u64;

        file.read_exact_at(&mut buf, index)
            .unwrap();

        let x = F::from_bytes_le(buf.as_slice());
        ret.push(x);
    }

    ret
}

fn main() {
    let n = 8usize;

    let a: Vec<u128> = vec![3, 1, 4, 1, 5, 9, 2, 6];

    let mut inout: Vec<F> = Vec::with_capacity(n);

    for i in 0..a.len() {
        let x_ark = FrBN254::from(a[i]);

        println!(
            "x_ark = {:?}",
            x_ark
                .0
                .to_string()
        );

        let bz = x_ark
            .0
            .to_bytes_le();

        let num = F::from_bytes_le(bz.as_slice());

        inout.push(num);

        // F::from_ark(x_ark);
    }

    println!("AAAAAAAA 000000");

    let ws = read_ws("ws.bin", n, 32);
    let ws_inv = read_ws("ws_inv.bin", n, 32);

    let mut inout_slice = HostOrDeviceSlice::on_host(inout);
    let mut ws_slice = HostOrDeviceSlice::on_host(ws);
    let mut ws_inv_slice = HostOrDeviceSlice::on_host(ws_inv);

    let is_mont = true;
    // let convert_result = F::to_mont(&mut inout_slice);
    // println!("AAAAAAAA 111111, convert_result = {:?}", convert_result);

    #[cfg(feature = "profile")]
    let start = Instant::now();
    fft_evaluate::<F>(&mut inout_slice, &mut ws_slice, n as u32, is_mont).unwrap();

    println!("AAAAAAAA 22222 ============================================================");

    fft_interpolate::<F>(&mut inout_slice, &mut ws_inv_slice, n as u32, is_mont).unwrap();

    let result = inout_slice.as_slice();
    for x in result.iter() {
        let x_big: FrBN254 = Fp(
            BigInt::try_from(BigUint::from_bytes_le(&x.to_bytes_le())).unwrap(),
            std::marker::PhantomData,
        );

        println!("x = {:?}", x_big.to_string());
    }

    #[cfg(feature = "profile")]
    println!(
        "FFT evaluation with size {n} took: {} μs",
        start
            .elapsed()
            .as_micros()
    );
}
