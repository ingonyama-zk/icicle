use icicle_core::hash::{Hasher, HasherHandle};
use icicle_runtime::errors::eIcicleError;

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
