use icicle_core::hash::{Hasher, HasherHandle};
use icicle_runtime::errors::eIcicleError;

extern "C" {
    fn icicle_create_sha3_256(default_input_chunk_size: u64) -> HasherHandle;
    fn icicle_create_sha3_512(default_input_chunk_size: u64) -> HasherHandle;
}

pub struct Sha3_256;
pub struct Sha3_512;

impl Sha3_256 {
    pub fn new(default_input_chunk_size: u64) -> Result<Hasher, eIcicleError> {
        let handle: HasherHandle = unsafe { icicle_create_sha3_256(default_input_chunk_size) };
        if handle.is_null() {
            return Err(eIcicleError::UnknownError);
        }
        Ok(Hasher::from_handle(handle))
    }
}

impl Sha3_512 {
    pub fn new(default_input_chunk_size: u64) -> Result<Hasher, eIcicleError> {
        let handle: HasherHandle = unsafe { icicle_create_sha3_512(default_input_chunk_size) };
        if handle.is_null() {
            return Err(eIcicleError::UnknownError);
        }
        Ok(Hasher::from_handle(handle))
    }
}
