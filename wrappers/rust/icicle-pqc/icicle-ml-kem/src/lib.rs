use icicle_runtime::{errors::eIcicleError, memory::HostOrDeviceSlice};

use crate::{config::MlKemConfig, kyber_params::KyberParams};

pub mod config;
mod ffi;
pub mod kyber_params;
mod tests;

pub fn keygen<P: KyberParams>(
    entropy: &(impl HostOrDeviceSlice<u8> + ?Sized), // batch_size × 64 bytes
    config: &MlKemConfig,
    public_keys: &mut (impl HostOrDeviceSlice<u8> + ?Sized), // batch_size × PUBLIC_KEY_BYTES
    secret_keys: &mut (impl HostOrDeviceSlice<u8> + ?Sized), // batch_size × SECRET_KEY_BYTES
) -> Result<(), eIcicleError> {
    P::keygen(entropy, config, public_keys, secret_keys)
}

pub fn encapsulate<P: KyberParams>(
    message: &(impl HostOrDeviceSlice<u8> + ?Sized), // batch_size × 32 bytes
    public_keys: &(impl HostOrDeviceSlice<u8> + ?Sized), // batch_size × PUBLIC_KEY_BYTES
    config: &MlKemConfig,
    ciphertexts: &mut (impl HostOrDeviceSlice<u8> + ?Sized), // batch_size × CIPHERTEXT_BYTES
    shared_secrets: &mut (impl HostOrDeviceSlice<u8> + ?Sized), // batch_size × SHARED_SECRET_BYTES
) -> Result<(), eIcicleError> {
    P::encapsulate(message, public_keys, config, ciphertexts, shared_secrets)
}

pub fn decapsulate<P: KyberParams>(
    secret_keys: &(impl HostOrDeviceSlice<u8> + ?Sized), // batch_size × SECRET_KEY_BYTES
    ciphertexts: &(impl HostOrDeviceSlice<u8> + ?Sized), // batch_size × CIPHERTEXT_BYTES
    config: &MlKemConfig,
    shared_secrets: &mut (impl HostOrDeviceSlice<u8> + ?Sized), // batch_size × SHARED_SECRET_BYTES
) -> Result<(), eIcicleError> {
    P::decapsulate(secret_keys, ciphertexts, config, shared_secrets)
}
