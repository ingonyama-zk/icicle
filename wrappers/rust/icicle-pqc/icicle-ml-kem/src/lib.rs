use icicle_runtime::{errors::eIcicleError, memory::HostOrDeviceSlice};

use crate::{config::MlKemConfig, kyber_params::KyberParams};

pub mod config;
mod ffi;
pub mod kyber_params;
mod tests;

/// Generates Kyber keypairs in batch mode using the specified parameters.
///
/// # Type Parameters
/// - `P`: The Kyber parameter set implementing the `KyberParams` trait.
///
/// # Parameters
/// - `entropy`: A slice containing random entropy input (length = batch_size × 64 bytes).
/// - `config`: A configuration object.
/// - `public_keys`: Output buffer to be filled with generated public keys (length = batch_size × PUBLIC_KEY_BYTES).
/// - `secret_keys`: Output buffer to be filled with generated secret keys (length = batch_size × SECRET_KEY_BYTES).
///
/// # Returns
/// - `Ok(())` on success.
/// - `Err(eIcicleError)` if the operation fails.
pub fn keygen<P: KyberParams>(
    entropy: &(impl HostOrDeviceSlice<u8> + ?Sized), // batch_size × 64 bytes
    config: &MlKemConfig,
    public_keys: &mut (impl HostOrDeviceSlice<u8> + ?Sized), // batch_size × PUBLIC_KEY_BYTES
    secret_keys: &mut (impl HostOrDeviceSlice<u8> + ?Sized), // batch_size × SECRET_KEY_BYTES
) -> Result<(), eIcicleError> {
    P::keygen(entropy, config, public_keys, secret_keys)
}

/// Performs Kyber encapsulation in batch mode to produce ciphertexts and shared secrets.
///
/// # Type Parameters
/// - `P`: The Kyber parameter set implementing the `KyberParams` trait.
///
/// # Parameters
/// - `message`: Input buffer of random 32-byte messages (length = batch_size × 32 bytes).
/// - `public_keys`: Input buffer of public keys (length = batch_size × PUBLIC_KEY_BYTES).
/// - `config`: A configuration object.
/// - `ciphertexts`: Output buffer to be filled with ciphertexts (length = batch_size × CIPHERTEXT_BYTES).
/// - `shared_secrets`: Output buffer to be filled with shared secrets (length = batch_size × SHARED_SECRET_BYTES).
///
/// # Returns
/// - `Ok(())` on success.
/// - `Err(eIcicleError)` if the operation fails.
pub fn encapsulate<P: KyberParams>(
    message: &(impl HostOrDeviceSlice<u8> + ?Sized), // batch_size × 32 bytes
    public_keys: &(impl HostOrDeviceSlice<u8> + ?Sized), // batch_size × PUBLIC_KEY_BYTES
    config: &MlKemConfig,
    ciphertexts: &mut (impl HostOrDeviceSlice<u8> + ?Sized), // batch_size × CIPHERTEXT_BYTES
    shared_secrets: &mut (impl HostOrDeviceSlice<u8> + ?Sized), // batch_size × SHARED_SECRET_BYTES
) -> Result<(), eIcicleError> {
    P::encapsulate(message, public_keys, config, ciphertexts, shared_secrets)
}

/// Performs Kyber decapsulation in batch mode to derive shared secrets from ciphertexts and secret keys.
///
/// # Type Parameters
/// - `P`: The Kyber parameter set implementing the `KyberParams` trait.
///
/// # Parameters
/// - `secret_keys`: Input buffer of secret keys (length = batch_size × SECRET_KEY_BYTES).
/// - `ciphertexts`: Input buffer of ciphertexts (length = batch_size × CIPHERTEXT_BYTES).
/// - `config`: A configuration object.
/// - `shared_secrets`: Output buffer to be filled with shared secrets (length = batch_size × SHARED_SECRET_BYTES).
pub fn decapsulate<P: KyberParams>(
    secret_keys: &(impl HostOrDeviceSlice<u8> + ?Sized), // batch_size × SECRET_KEY_BYTES
    ciphertexts: &(impl HostOrDeviceSlice<u8> + ?Sized), // batch_size × CIPHERTEXT_BYTES
    config: &MlKemConfig,
    shared_secrets: &mut (impl HostOrDeviceSlice<u8> + ?Sized), // batch_size × SHARED_SECRET_BYTES
) -> Result<(), eIcicleError> {
    P::decapsulate(secret_keys, ciphertexts, config, shared_secrets)
}
