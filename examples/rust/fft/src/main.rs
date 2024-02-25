use icicle_bn254::curve::ScalarField as F;
use icicle_cuda_runtime::device_context::DeviceContext;
use icicle_cuda_runtime::memory::HostOrDeviceSlice;

use icicle_core::fft::fft_evaluate;

#[cfg(feature = "profile")]
use std::time::Instant;

use clap::Parser;

fn main() {
    let n = 16usize;

    let inout = vec![F::one(); n];
    let ws = vec![F::one(); n];

    let mut inout_slice = HostOrDeviceSlice::on_host(inout);
    let mut ws_slice = HostOrDeviceSlice::on_host(inout);

    #[cfg(feature = "profile")]
    let start = Instant::now();
    fft_evaluate::<F>(&mut inout, &mut ws, n).unwrap();

    #[cfg(feature = "profile")]
    println!(
        "FFT evaluation with size {n} took: {} μs",
        start
            .elapsed()
            .as_micros()
    );
}
