use crate::memory::CudaMemPool;
use crate::stream::CudaStream;

/// Properties of the device used in icicle functions.
#[repr(C)]
#[derive(Debug)]
pub struct DeviceContext {
    /// Index of the currently used GPU. Default value: 0.
    pub device_id: usize,

    /// Stream to use. Default value: 0.
    pub stream: CudaStream, // Assuming the type is provided by a CUDA binding crate

    /// Mempool to use. Default value: 0.
    pub mempool: CudaMemPool, // Assuming the type is provided by a CUDA binding crate
}

pub fn get_default_device_context() -> DeviceContext {
    DeviceContext {
        device_id: 0,
        stream: CudaStream::from_handle(std::ptr::null_mut()),
        mempool: 0,
    }
}