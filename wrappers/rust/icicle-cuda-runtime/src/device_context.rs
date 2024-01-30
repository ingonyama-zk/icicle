use crate::device::set_device;
use crate::memory::CudaMemPool;
use crate::stream::CudaStream;

pub const DEFAULT_DEVICE_ID: usize = 0;

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

// TODO: CudaResult
pub fn get_default_context_for_device(device_id: usize) -> DeviceContext<'static> {
    set_device(device_id).unwrap();
    // TODO: default stream? on what device? or create one after set_device
    static default_stream: CudaStream = CudaStream {
        handle: std::ptr::null_mut(),
    };

    // TODO: default mempool? on what device? or create one after set_device
    DeviceContext {
        stream: &default_stream,
        device_id,
        mempool: std::ptr::null_mut(),
    }
}
