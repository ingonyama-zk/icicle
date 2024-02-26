use icicle_bn254::curve::ScalarField as F;
// use icicle_bls12_381::curve::ScalarField as F;
use icicle_core::traits::FieldImpl;
use icicle_cuda_runtime::memory::HostOrDeviceSlice;
use std::fs;
use std::os::unix::fs::FileExt;

use icicle_core::fft::fft_evaluate;

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

    let a: Vec<u8> = vec![3, 1, 4, 1, 5, 9, 2, 6];

    let mut inout: Vec<F> = Vec::with_capacity(n);

    for x in a.iter() {
        let mut bz: Vec<u8> = vec![0u8; 32];
        bz[0] = *x;
        inout.push(F::from_bytes_le(bz.as_slice()));
    }

    let ws = read_ws("ws.bin", n, 32);
    let ws_inv = read_ws("ws_inv.bin", n, 32);

    let mut inout_slice = HostOrDeviceSlice::on_host(inout);
    let mut ws_slice = HostOrDeviceSlice::on_host(ws);

    #[cfg(feature = "profile")]
    let start = Instant::now();
    fft_evaluate::<F>(&mut inout_slice, &mut ws_slice, n as u32).unwrap();

    let result = inout_slice.as_slice();

    #[cfg(feature = "profile")]
    println!(
        "FFT evaluation with size {n} took: {} μs",
        start
            .elapsed()
            .as_micros()
    );
}
