use std::ffi::c_uint;

// Assuming that the CUDA types can be defined as follows.
// The exact type might depend on the specifics of the Rust CUDA bindings, if available.
#[allow(non_camel_case_types)]
pub(crate) type cudaStream_t = usize; // This might be a placeholder, check your binding crate for the exact type
#[allow(non_camel_case_types)]
pub(crate) type cudaMemPool_t = usize; // This might be a placeholder, check your binding crate for the exact type

/// Properties of the device used in icicle functions.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeviceContext {
    /// Index of the currently used GPU. Default value: 0.
    pub device_id: usize,

    /// Stream to use. Default value: 0.
    pub stream: cudaStream_t, // Assuming the type is provided by a CUDA binding crate

    /// Mempool to use. Default value: 0.
    pub mempool: cudaMemPool_t, // Assuming the type is provided by a CUDA binding crate
}

pub struct DevicePointer<E> {
    // Placeholder for a type representing a device pointer.
    raw: *mut E,
}

#[allow(non_camel_case_types)]
pub(crate) type cudaError_t = c_uint;

fn get_device_context(device_id: usize) -> DeviceContext {
    //TODO: on cuda side
    DeviceContext {
        device_id,
        stream: 0,
        mempool: 0,
    }
}

pub(crate) fn get_default_device_context() -> DeviceContext {
    get_device_context(0)
}
