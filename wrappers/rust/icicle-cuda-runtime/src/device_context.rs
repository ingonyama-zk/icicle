use crate::memory::CudaMemPool;
use crate::stream::CudaStream;

pub const DEFAULT_DEVICE_ID: usize = 0;

use crate::device::get_device;

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
        let default_stream = Box::new(CudaStream {
            handle: std::ptr::null_mut(),
        });

        let leaked_stream = Box::leak(default_stream);

        DeviceContext {
            stream: leaked_stream,
            device_id,
            mempool: std::ptr::null_mut(),
        }
    }
}

// impl Drop for DeviceContext<'_> {
//     fn drop(&mut self) {
//         unsafe {
//             // SAFETY: This is safe only if we're sure that self.stream is valid
//             // and not used after being dropped.
//             let raw_stream_pointer = self.stream as *const _ as *mut CudaStream;
//             let _ = Box::from_raw(raw_stream_pointer); // Reconstruct the Box to drop it
//         }
//         // After this block, the Box goes out of scope and its destructor is run, freeing the memory.
//     }
// }

pub fn check_device(device_id: i32) {
    match device_id == get_device().unwrap() as i32 {
        true => (),
        false => panic!("Attempt to use on a different device"),
    }
}
