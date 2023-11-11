pub struct DevicePointer<E> {
    // Placeholder for a type representing a device pointer.
    raw: *mut E,
}

#[allow(non_camel_case_types)]
pub type CudaMemPool = usize; // This is a placeholder, TODO: actually make this into a proper CUDA wrapper
