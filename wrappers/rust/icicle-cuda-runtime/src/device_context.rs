use crate::memory::CudaMemPool;
use crate::stream::CudaStream;

/// Properties of the device used in icicle functions.
#[repr(C)]
#[derive(Debug)]
pub struct DeviceContext<'a> {
    /// Index of the currently used GPU. Default value: 0.
    pub device_id: usize,

    /// Stream to use. Default value: 0.
    pub stream: &'a CudaStream, // Assuming the type is provided by a CUDA binding crate

    /// Mempool to use. Default value: 0.
    pub mempool: CudaMemPool, // Assuming the type is provided by a CUDA binding crate
}

pub fn get_default_device_context() -> DeviceContext<'static> {
    static default_stream: CudaStream = CudaStream { handle: std::ptr::null_mut() };
    DeviceContext {
        device_id: 0,
        stream: &default_stream,
        mempool: 0,
    }
}