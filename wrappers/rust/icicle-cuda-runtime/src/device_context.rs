use crate::memory::CudaMemPool;
use crate::stream::CudaStream;

pub const DEFAULT_DEVICE_ID: usize = 0;

use crate::device::get_device;
use crate::device::set_device;

/// Properties of the device used in Icicle functions.
#[repr(C)]
#[derive(Debug, Clone)]
pub struct DeviceContext<'a> {
    /// Stream to use. Default value: 0. //TODO: multiple streams per device ?
    pub stream: &'a CudaStream, // Assuming the type is provided by a CUDA binding crate

    /// Index of the currently used GPU. Default value: 0.
    pub device_id: usize,

    /// Mempool to use. Default value: 0. //TODO: multiple mempools per device ?
    pub mempool: CudaMemPool, // Assuming the type is provided by a CUDA binding crate
}

pub fn get_default_device_context() -> DeviceContext<'static> {
    get_default_context_for_device(DEFAULT_DEVICE_ID)
}

pub fn get_default_context_for_device(device_id: usize) -> DeviceContext<'static> {
    check_device(device_id as i32);
    // TODO: default stream? on what device? or create one after set_device
    let stream: &'static CudaStream  = Box::leak(Box::new(CudaStream::create().unwrap()));
    Box::leak(Box::new(stream)); // TODO: leaky abstraction
    // TODO: default mempool? on what device? or create one after set_device
    DeviceContext {
        stream,
        device_id,
        mempool: std::ptr::null_mut(),
    }
}

pub fn check_device(device_id: i32) {
    match device_id == get_device().unwrap() as i32 {
        true => (),
        false => panic!("Attempt to use on a different device"),
    }
}