use crate::{
    bindings::{cudaGetDevice, cudaGetDeviceCount, cudaPointerAttributes, cudaPointerGetAttributes, cudaSetDevice},
    error::{CudaResult, CudaResultWrap},
};
use std::mem::MaybeUninit;

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

pub fn get_device_from_pointer(ptr: *const ::std::os::raw::c_void) -> CudaResult<usize> {
    let mut ptr_attributes = MaybeUninit::<cudaPointerAttributes>::uninit();
    unsafe {
        cudaPointerGetAttributes(ptr_attributes.as_mut_ptr(), ptr).wrap()?;
        Ok(ptr_attributes
            .assume_init()
            .device as usize)
    }
}

pub fn check_device(device_id: usize) {
    match device_id == get_device().unwrap() {
        true => (),
        false => panic!("Attempt to use on a different device"),
    }
}
