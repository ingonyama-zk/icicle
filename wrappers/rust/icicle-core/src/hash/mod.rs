use icicle_runtime::{
    config::ConfigExtension, errors::eIcicleError, memory::HostOrDeviceSlice, stream::IcicleStreamHandle,
};
use std::ffi::c_void;

/// Configuration structure for hash operations.
///
/// The `HashConfig` structure holds various configuration options that control how hash operations
/// are executed. It supports features such as specifying the execution stream, input/output locations
/// (device or host), batch sizes, and backend-specific extensions. Additionally, it allows
/// synchronous and asynchronous execution modes.
#[repr(C)]
#[derive(Clone)]
pub struct HashConfig {
    pub stream_handle: IcicleStreamHandle,
    batch: u64,
    are_inputs_on_device: bool,
    are_outputs_on_device: bool,
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

pub type HasherHandle = *const c_void;

pub struct Hasher {
    pub handle: HasherHandle,
}

// External C functions for hashing and deleting hash objects
extern "C" {
    fn icicle_hasher_hash(
        hash_ptr: HasherHandle,
        input_ptr: *const u8,
        input_len: u64,
        config: *const HashConfig,
        output_ptr: *mut u8,
    ) -> eIcicleError;
    fn icicle_hasher_output_size(hash_ptr: HasherHandle) -> u64;
    fn icicle_hasher_delete(hash_ptr: HasherHandle) -> eIcicleError;
}

impl Hasher {
    pub fn from_handle(_handle: HasherHandle) -> Self {
        Hasher { handle: _handle }
    }

    pub fn hash<TInput, TOutput>(
        &self,
        input: &(impl HostOrDeviceSlice<TInput> + ?Sized),
        cfg: &HashConfig,
        output: &mut (impl HostOrDeviceSlice<TOutput> + ?Sized),
    ) -> Result<(), eIcicleError> {
        // Check if input is on active device
        if input.is_on_device() && !input.is_on_active_device() {
            eprintln!("input not allocated on the active device");
            return Err(eIcicleError::InvalidPointer);
        }
        // Check if output is on active device
        if output.is_on_device() && !output.is_on_active_device() {
            eprintln!("output not allocated on the active device");
            return Err(eIcicleError::InvalidPointer);
        }

        let mut local_cfg = cfg.clone();
        local_cfg.are_inputs_on_device = input.is_on_device();
        local_cfg.are_outputs_on_device = output.is_on_device();

        // Calculate the byte lengths for input and output
        let input_byte_len = (input.len() * std::mem::size_of::<TInput>()) as u64;
        let output_byte_len = (output.len() * std::mem::size_of::<TOutput>()) as u64;

        // Ensure output size is divisible by single hash output size
        if output_byte_len % self.output_size() != 0 {
            eprintln!(
                "output size (={}Bytes) must divide single hash output size (={}Bytes)",
                output_byte_len,
                self.output_size()
            );
            return Err(eIcicleError::InvalidArgument);
        }
        local_cfg.batch = output_byte_len / self.output_size();

        // Ensure input size is divisible by batch size inferred from output
        if input_byte_len % local_cfg.batch != 0 {
            eprintln!(
                "input size (={}Bytes) must divide batch-size={} (inferred from output size)",
                input_byte_len, local_cfg.batch,
            );
            return Err(eIcicleError::InvalidArgument);
        }

        // Unsafe block for pointer conversion and calling the hashing function
        unsafe {
            let input_ptr = input.as_ptr() as *const u8;
            let output_ptr = output.as_mut_ptr() as *mut u8;

            icicle_hasher_hash(
                self.handle,
                input_ptr, // Cast to *const u8
                input_byte_len / local_cfg.batch,
                &local_cfg,
                output_ptr, // Cast to *mut u8
            )
            .wrap()
        }
    }

    pub fn output_size(&self) -> u64 {
        unsafe { icicle_hasher_output_size(self.handle) }
    }
}

impl Drop for Hasher {
    fn drop(&mut self) {
        unsafe {
            icicle_hasher_delete(self.handle);
        }
    }
}
