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

impl Default for DeviceContext<'_> {
    fn default() -> Self {
        Self::default_for_device(DEFAULT_DEVICE_ID)
    }
}

impl DeviceContext<'_> {
    /// Default for device_id
    pub fn default_for_device(device_id: usize) -> DeviceContext<'static> {
        static default_stream: CudaStream = CudaStream {
            handle: std::ptr::null_mut(),
        };
        DeviceContext {
            stream: &default_stream,
            device_id,
            mempool: std::ptr::null_mut(),
        }
    }
}
