use icicle_runtime::{Device,runtime};
use icicle_core::traits::{FieldImpl,GenerateRandom};

pub fn set_backend_cpu() {
    let device_cpu = Device::new("CPU", 0);
    icicle_runtime::set_device(&device_cpu).unwrap();
}


pub fn try_load_and_set_backend_gpu() {
    runtime::load_backend("../../../icicle/backend/cuda").unwrap();
    let device_gpu = Device::new("CUDA", 0);
    let is_cuda_device_available = icicle_runtime::is_device_available(&device_gpu);
    if is_cuda_device_available {
        icicle_runtime::set_device(&device_gpu).unwrap();
    } else {
        set_backend_cpu();
}
}
pub fn generate_random_vector<F:FieldImpl> (size:usize) -> Vec<F> 
    where 
    <F as FieldImpl>::Config: GenerateRandom<F>,
    {
    F::Config::generate_random(size)
    }
