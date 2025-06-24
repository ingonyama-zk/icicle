use icicle_babybear::field::ScalarField;
use icicle_core::{
    ntt::{self, get_root_of_unity, initialize_domain, ntt, NTTConfig},
    traits::GenerateRandom,
    field::PrimeField
};
use icicle_runtime::memory::{DeviceVec, IntoIcicleSlice, IntoIcicleSliceMut};
use icicle_runtime::{self, Device};

fn main() {
    // Load installed backends
    let _ = icicle_runtime::load_backend_from_env_or_default();

    // Check if GPU is available
    let device_cpu = Device::new("CPU", 0);
    let mut device_gpu = Device::new("CUDA", 0);
    let is_cuda_device_available = icicle_runtime::is_device_available(&device_gpu);

    if is_cuda_device_available {
        println!("GPU is available");
    } else {
        println!("GPU is not available, falling back to CPU only");
        device_gpu = device_cpu.clone();
    }

    // Example input (on host memory) for NTT
    let log_ntt_size = 2;
    let ntt_size = 1 << log_ntt_size;
    let input_cpu = ScalarField::generate_random(ntt_size);

    // Allocate output on host memory
    let mut output_cpu = vec![ScalarField::zero(); ntt_size];
    let root_of_unity = get_root_of_unity::<ScalarField>(ntt_size as u64);
    let ntt_config = NTTConfig::<ScalarField>::default();

    // Part 1: Running NTT on CPU
    println!("Part 1: compute on CPU: ");
    icicle_runtime::set_device(&device_cpu).expect("Failed to set device to CPU");
    initialize_domain(root_of_unity, &ntt::NTTInitDomainConfig::default()).expect("Failed to initialize NTT domain");
    ntt(
        input_cpu.into_slice(),
        ntt::NTTDir::kForward,
        &ntt_config,
        output_cpu.into_slice_mut(),
    )
    .expect("NTT computation failed on CPU");
    println!("{:?}", output_cpu);

    // Part 2: Running NTT on GPU (from/to CPU memory)
    println!("Part 2: compute on GPU (from/to CPU memory): ");
    icicle_runtime::set_device(&device_gpu).expect("Failed to set device to GPU");
    initialize_domain(root_of_unity, &ntt::NTTInitDomainConfig::default()).expect("Failed to initialize NTT domain");
    ntt(
        input_cpu.into_slice(),
        ntt::NTTDir::kForward,
        &ntt_config,
        output_cpu.into_slice_mut(),
    )
    .expect("NTT computation failed on GPU");
    println!("{:?}", output_cpu);

    // Part 2 (cont.): Compute on GPU (from/to GPU memory)
    println!("Part 2: compute on GPU (from/to GPU memory): ");
    let mut output_gpu = DeviceVec::<ScalarField>::malloc(ntt_size);
    let input_gpu = DeviceVec::<ScalarField>::from_host_vec(&input_cpu);
    ntt(
        input_gpu.into_slice(),
        ntt::NTTDir::kForward,
        &ntt_config,
        output_gpu.into_slice_mut(),
    )
    .expect("NTT computation failed on GPU memory");
    let output_cpu = output_gpu.to_host_vec();

    println!("{:?}", output_cpu);

    // Part 3: Using both CPU and GPU to compute NTT (GPU) and inverse INTT (CPU)
    let mut output_intt_cpu = vec![ScalarField::zero(); ntt_size];

    // Step 1: Compute NTT on GPU
    println!("Part 3: compute NTT on GPU (NTT input): ");
    icicle_runtime::set_device(&device_gpu).expect("Failed to set device to GPU");
    ntt(
        input_cpu.into_slice(),
        ntt::NTTDir::kForward,
        &ntt_config,
        output_cpu.into_slice_mut(),
    )
    .expect("NTT computation failed on GPU");
    println!("{:?}", input_cpu);

    // Step 2: Compute INTT on CPU
    println!("Part 3: compute INTT on CPU (INTT output): ");
    icicle_runtime::set_device(&device_cpu).expect("Failed to set device to CPU");
    ntt(
        output_cpu.into_slice(),
        ntt::NTTDir::kInverse,
        &ntt_config,
        output_intt_cpu.into_slice_mut(),
    )
    .expect("INTT computation failed on CPU");
    println!("{:?}", output_intt_cpu);
}
