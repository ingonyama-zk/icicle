use icicle_core::hash::{Hasher, HasherHandle};
use icicle_runtime::{config::ConfigExtension, errors::eIcicleError, memory::HostOrDeviceSlice, IcicleStreamHandle};

extern "C" {
    fn icicle_create_blake3(default_input_chunk_size: u64) -> HasherHandle;
}

pub struct Blake3;

impl Blake3 {
    pub fn new(default_input_chunk_size: u64) -> Result<Hasher, eIcicleError> {
        let handle: HasherHandle = unsafe { icicle_create_blake3(default_input_chunk_size) };
        if handle.is_null() {
            return Err(eIcicleError::UnknownError);
        }
        Ok(Hasher::from_handle(handle))
    }
}

#[repr(C)]
pub struct PowConfig {
    pub stream: IcicleStreamHandle, // `icicleStreamHandle` is represented as a raw pointer.
    pub is_challenge_on_device: bool,
    pub is_result_on_device: bool,
    pub is_async: bool,
    pub ext: ConfigExtension, // `ConfigExtension*` is represented as a raw pointer.
}

impl Default for PowConfig {
    fn default() -> Self {
        PowConfig {
            stream: std::ptr::null_mut(),
            is_challenge_on_device: false,
            is_result_on_device: false,
            is_async: false,
            ext: ConfigExtension::new(),
        }
    }
}

extern "C" {
    pub fn some_pow_blake3(
        challenge: *const u8,
        bits: u8,
        config: *const PowConfig,
        found: *mut bool,
        nonce: *mut u64,
        mined_hash: *mut u64,
    ) -> eIcicleError;
}

pub fn rust_some_pow_blake3(
    challenge: &(impl HostOrDeviceSlice<u8> + ?Sized),
    bits: u8,
    config: &PowConfig,
    found: &mut bool,
    nonce: &mut u64,
    mined_hash: &mut u64,
) -> eIcicleError {
    // Ensure the challenge length matches what the C function expects

    let result = unsafe {
        some_pow_blake3(
            challenge.as_ptr(),
            bits,
            config as *const PowConfig,
            found as *mut bool,
            nonce as *mut u64,
            mined_hash as *mut u64,
        )
    };
    result
}
