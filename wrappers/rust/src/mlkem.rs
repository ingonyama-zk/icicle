use std::ffi::{c_void, CString};
use std::slice;
use crate::error::Error;
use crate::result::Result;

#[repr(C)]
pub struct MLKEMKeyPair {
    public_key: *mut u8,
    public_key_len: usize,
    secret_key: *mut u8,
    secret_key_len: usize,
}

#[repr(C)]
pub struct MLKEMCiphertext {
    ciphertext: *mut u8,
    ciphertext_len: usize,
    shared_secret: *mut u8,
    shared_secret_len: usize,
}

extern "C" {
    fn mlkem_init(backend_type: u32) -> u32;
    fn mlkem_keygen() -> *mut MLKEMKeyPair;
    fn mlkem_encaps(public_key: *const u8, public_key_len: usize) -> *mut MLKEMCiphertext;
    fn mlkem_decaps(ciphertext: *const u8, ciphertext_len: usize, secret_key: *const u8, secret_key_len: usize) -> *mut u8;
    fn mlkem_free_keypair(keypair: *mut MLKEMKeyPair);
    fn mlkem_free_ciphertext(ciphertext: *mut MLKEMCiphertext);
    fn mlkem_free_shared_secret(shared_secret: *mut u8);
}

/// Initialize the ML-KEM backend
pub fn init(backend_type: u32) -> Result<()> {
    unsafe {
        let result = mlkem_init(backend_type);
        if result == 0 {
            Ok(())
        } else {
            Err(Error::BackendError("Failed to initialize ML-KEM backend".to_string()))
        }
    }
}

/// Generate a new ML-KEM key pair
pub fn keygen() -> Result<(Vec<u8>, Vec<u8>)> {
    unsafe {
        let keypair = mlkem_keygen();
        if keypair.is_null() {
            return Err(Error::BackendError("Failed to generate ML-KEM key pair".to_string()));
        }

        let public_key = slice::from_raw_parts(
            (*keypair).public_key,
            (*keypair).public_key_len,
        ).to_vec();

        let secret_key = slice::from_raw_parts(
            (*keypair).secret_key,
            (*keypair).secret_key_len,
        ).to_vec();

        mlkem_free_keypair(keypair);
        Ok((public_key, secret_key))
    }
}

/// Encapsulate a shared secret using a public key
pub fn encaps(public_key: &[u8]) -> Result<(Vec<u8>, Vec<u8>)> {
    unsafe {
        let ciphertext = mlkem_encaps(
            public_key.as_ptr(),
            public_key.len(),
        );

        if ciphertext.is_null() {
            return Err(Error::BackendError("Failed to encapsulate shared secret".to_string()));
        }

        let ciphertext_bytes = slice::from_raw_parts(
            (*ciphertext).ciphertext,
            (*ciphertext).ciphertext_len,
        ).to_vec();

        let shared_secret = slice::from_raw_parts(
            (*ciphertext).shared_secret,
            (*ciphertext).shared_secret_len,
        ).to_vec();

        mlkem_free_ciphertext(ciphertext);
        Ok((ciphertext_bytes, shared_secret))
    }
}

/// Decapsulate a shared secret using a secret key
pub fn decaps(ciphertext: &[u8], secret_key: &[u8]) -> Result<Vec<u8>> {
    unsafe {
        let shared_secret = mlkem_decaps(
            ciphertext.as_ptr(),
            ciphertext.len(),
            secret_key.as_ptr(),
            secret_key.len(),
        );

        if shared_secret.is_null() {
            return Err(Error::BackendError("Failed to decapsulate shared secret".to_string()));
        }

        let result = slice::from_raw_parts(
            shared_secret,
            (*ciphertext).shared_secret_len,
        ).to_vec();

        mlkem_free_shared_secret(shared_secret);
        Ok(result)
    }
} 