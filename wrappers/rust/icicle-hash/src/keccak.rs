use icicle_core::hash::{Hasher, HasherHandle};
use icicle_runtime::errors::eIcicleError;

extern "C" {
    fn icicle_create_keccak_256(input_chunk_size: u64) -> HasherHandle;
    fn icicle_create_keccak_512(input_chunk_size: u64) -> HasherHandle;
}

pub fn create_keccak_256_hasher(input_chunk_size: u64) -> Result<Hasher, eIcicleError> {
    let handle: HasherHandle = unsafe { icicle_create_keccak_256(input_chunk_size) };
    if handle.is_null() {
        return Err(eIcicleError::UnknownError);
    }
    Ok(Hasher::from_handle(handle))
}

pub fn create_keccak_512_hasher(input_chunk_size: u64) -> Result<Hasher, eIcicleError> {
    let handle: HasherHandle = unsafe { icicle_create_keccak_512(input_chunk_size) };
    if handle.is_null() {
        return Err(eIcicleError::UnknownError);
    }
    Ok(Hasher::from_handle(handle))
}
