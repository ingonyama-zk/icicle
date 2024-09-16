use icicle_runtime::{config::ConfigExtension, errors::eIcicleError, stream::IcicleStreamHandle};
use std::ffi::c_void;

pub mod tests;

/// Configuration structure for hash operations.
///
/// The `HashConfig` structure holds various configuration options that control how hash operations
/// are executed. It supports features such as specifying the execution stream, input/output locations
/// (device or host), batch sizes, and backend-specific extensions. Additionally, it allows
/// synchronous and asynchronous execution modes.
#[repr(C)]
pub struct HashConfig {
    pub stream_handle: IcicleStreamHandle,
    pub batch: u64,
    pub are_inputs_on_device: bool,
    pub are_outputs_on_device: bool,
    pub is_async: bool,
    pub ext: ConfigExtension,
}

impl HashConfig {
    /// Create a default configuration (same as the C++ struct's defaults)
    pub fn default() -> Self {
        Self {
            stream_handle: std::ptr::null_mut(),
            batch: 1,
            are_inputs_on_device: false,
            are_outputs_on_device: false,
            is_async: false,
            ext: ConfigExtension::new(),
        }
    }
}

type HasherHandle = *const c_void;

pub struct Hasher {
    handle: HasherHandle,
}

// External C functions for hashing and deleting hash objects
extern "C" {
    fn hasher_hash(
        hash_ptr: HasherHandle,
        input_ptr: *const u8,
        input_len: u64,
        config: *const HashConfig,
        output_ptr: *mut u8,
    ) -> eIcicleError;
    fn hasher_output_size(hash_ptr: HasherHandle) -> u64;
    fn hasher_delete(hash_ptr: HasherHandle) -> eIcicleError;
}

impl Hasher {
    pub fn hash(&self, input: &[u8], config: &HashConfig, output: &mut [u8]) -> Result<(), eIcicleError> {
        unsafe {
            hasher_hash(
                self.handle,
                input.as_ptr(),
                input.len() as u64,
                config,
                output.as_mut_ptr(),
            )
            .wrap()
        }
    }

    pub fn output_size(&self) -> u64 {
        unsafe { hasher_output_size(self.handle) }
    }
}

impl Drop for Hasher {
    fn drop(&mut self) {
        unsafe {
            hasher_delete(self.handle);
        }
    }
}
