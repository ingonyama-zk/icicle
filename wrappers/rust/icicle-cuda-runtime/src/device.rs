use crate::{
    bindings::{cudaGetDevice, cudaGetDeviceCount, cudaSetDevice},
    error::{CudaResult, CudaResultWrap},
};

pub fn set_device(device_id: usize) -> CudaResult<()> { 
    unsafe { cudaSetDevice(device_id as i32) }.wrap()
}

pub fn get_device_count() -> CudaResult<usize> {
    let mut count = 0;
    unsafe { cudaGetDeviceCount(&mut count) }.wrap_value(count as usize)
}

pub fn get_device() -> CudaResult<usize> {
    let mut device_id = 0;
    unsafe { cudaGetDevice(&mut device_id) }.wrap_value(device_id as usize)
}
