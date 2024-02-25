use icicle_bn254::curve::ScalarField as F;
// use icicle_bls12_381::curve::ScalarField as F;
use icicle_core::traits::FieldImpl;
use icicle_cuda_runtime::memory::HostOrDeviceSlice;

use icicle_core::fft::fft_evaluate;

#[cfg(feature = "profile")]
use std::time::Instant;

fn main() {
    let n = 16usize;

    let inout = vec![F::one(); n];
    let ws = vec![F::one(); n];

    let mut inout_slice = HostOrDeviceSlice::on_host(inout);
    let mut ws_slice = HostOrDeviceSlice::on_host(ws);

    #[cfg(feature = "profile")]
    let start = Instant::now();
    fft_evaluate::<F>(&mut inout_slice, &mut ws_slice, n as u32).unwrap();

    #[cfg(feature = "profile")]
    println!(
        "FFT evaluation with size {n} took: {} μs",
        start
            .elapsed()
            .as_micros()
    );
}
