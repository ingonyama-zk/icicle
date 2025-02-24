use clap::Parser;
use icicle_core::traits::{FieldImpl, GenerateRandom};
use icicle_runtime::{runtime, Device};

#[derive(Parser, Debug)]
pub struct Args {
    /// Device type (e.g., "CPU", "CUDA")
    #[arg(short, long, default_value = "CPU")]
    device_type: String,
}

// Load backend and set device
pub fn try_load_and_set_backend_device(args: &Args) {
    if args.device_type != "CPU" {
        icicle_runtime::runtime::load_backend_from_env_or_default().unwrap();
    }
    println!("Setting device {}", args.device_type);
    let device = icicle_runtime::Device::new(&args.device_type, 0 /* =device_id*/);
    icicle_runtime::set_device(&device).unwrap();
}
//we want verifier to run in CPU
pub fn set_backend_cpu() {
    let device_cpu = Device::new("CPU", 0);
    icicle_runtime::set_device(&device_cpu).unwrap();
}

pub fn generate_random_vector<F: FieldImpl>(size: usize) -> Vec<F>
where
    <F as FieldImpl>::Config: GenerateRandom<F>,
{
    F::Config::generate_random(size)
}
