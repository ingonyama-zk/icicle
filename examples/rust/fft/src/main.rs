use icicle_bn254::curve::ScalarField as F;
use icicle_core::traits::FieldImpl;
use icicle_core::traits::MontgomeryConvertible;
use icicle_cuda_runtime::memory::HostOrDeviceSlice;
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::fs;
use std::os::unix::fs::FileExt;
use std::time::Instant;

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

pub fn run_fft(a: Vec<FrBN254>) {
    let n = a.len();

    let start = Instant::now();

    let mut inout: Vec<F> = Vec::with_capacity(n);
    for i in 0..a.len() {
        let x_ark = a[i];

        let bz = x_ark
            .0
            .to_bytes_le();

        let num = F::from_bytes_le(bz.as_slice());

        inout.push(num);
    }

    println!("Finish conversion, loading time = {:.2?}", start.elapsed());
    let start = Instant::now();

    let ws = read_ws("ws.bin", n, 32);
    let ws_inv = read_ws("ws_inv.bin", n, 32);

    println!("Finish loading ws! time = {:.2?}", start.elapsed());
    let start = Instant::now();

    let mut inout_slice = HostOrDeviceSlice::on_host(inout);
    let mut ws_slice = HostOrDeviceSlice::on_host(ws);
    let mut ws_inv_slice = HostOrDeviceSlice::on_host(ws_inv);

    let is_mont = true;

    println!("Done preparing. Start running on GPU... time = {:.2?}", start.elapsed());
    let start = Instant::now();

    fft_evaluate::<F>(&mut inout_slice, &mut ws_slice, n as u32, is_mont).unwrap();

    fft_interpolate::<F>(&mut inout_slice, &mut ws_inv_slice, n as u32, is_mont).unwrap();

    println!("Done Running on GPU, time = {:.2?}", start.elapsed());
    let start = Instant::now();

    let result = inout_slice.as_slice();
    // for x in result.iter() {
    for i in 0..result.len() {
        let x = result[i];
        let x_big: FrBN254 = Fp(
            BigInt::try_from(BigUint::from_bytes_le(&x.to_bytes_le())).unwrap(),
            std::marker::PhantomData,
        );

        if i < 8 {
            println!("x = {:?}", x_big.to_string());
        }
    }

    println!("Done Convert back to Arkwork, time = {:.2?}", start.elapsed());
    let start = Instant::now();
}

fn main() {
    let mut rng = StdRng::seed_from_u64(42);

    let n = (1 << 23) as usize;

    // let a: Vec<u128> = vec![3, 1, 4, 1, 5, 9, 2, 6];
    let mut a: Vec<FrBN254> = Vec::with_capacity(n);
    for i in 0..n {
        let num = rng.gen_range(0..100);
        a.push(FrBN254::from(num));

        if i < 8 {
            println!("num = {:?}", num);
        }
    }

    run_fft(a);
}
