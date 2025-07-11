use icicle_core::hash::{Hasher, HasherHandle};
use icicle_runtime::{eIcicleError, IcicleError};

extern "C" {
    fn icicle_create_keccak_256(default_input_chunk_size: u64) -> HasherHandle;
    fn icicle_create_keccak_512(default_input_chunk_size: u64) -> HasherHandle;
}

pub struct Keccak256;
pub struct Keccak512;

impl Keccak256 {
    pub fn new(default_input_chunk_size: u64) -> Result<Hasher, IcicleError> {
        let handle: HasherHandle = unsafe { icicle_create_keccak_256(default_input_chunk_size) };
        if handle.is_null() {
            return Err(IcicleError::new(
                eIcicleError::UnknownError,
                "Failed to create Keccak256 hasher",
            ));
        }
        Ok(Hasher::from_handle(handle))
    }
}

impl Keccak512 {
    pub fn new(default_input_chunk_size: u64) -> Result<Hasher, IcicleError> {
        let handle: HasherHandle = unsafe { icicle_create_keccak_512(default_input_chunk_size) };
        if handle.is_null() {
            return Err(IcicleError::new(
                eIcicleError::UnknownError,
                "Failed to create Keccak512 hasher",
            ));
        }
        Ok(Hasher::from_handle(handle))
    }
}
