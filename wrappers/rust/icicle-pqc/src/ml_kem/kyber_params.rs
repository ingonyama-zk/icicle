use icicle_runtime::{
    errors::{eIcicleError, IcicleError},
    memory::HostOrDeviceSlice,
};

use super::{config::MlKemConfig, ffi::*};

// Constants for general configuration
pub const ENTROPY_BYTES: usize = 64;
pub const MESSAGE_BYTES: usize = 32;

pub trait KyberParams {
    const PUBLIC_KEY_BYTES: usize;
    const SECRET_KEY_BYTES: usize;
    const CIPHERTEXT_BYTES: usize;
    const SHARED_SECRET_BYTES: usize;

    const K: u8;
    const ETA1: u8;
    const ETA2: u8;
    const DU: u8;
    const DV: u8;

    unsafe fn keygen_ffi(
        entropy: *const u8, // batch_size × 64 bytes
        config: *const MlKemConfig,
        public_keys: *mut u8, // batch_size × PUBLIC_KEY_BYTES
        secret_keys: *mut u8, // batch_size × SECRET_KEY_BYTES
    ) -> Result<(), IcicleError>;

    unsafe fn encapsulate_ffi(
        message: *const u8,     // batch_size × 32 bytes
        public_keys: *const u8, // batch_size × PUBLIC_KEY_BYTES
        config: *const MlKemConfig,
        ciphertexts: *mut u8,    // batch_size × CIPHERTEXT_BYTES
        shared_secrets: *mut u8, // batch_size × SHARED_SECRET_BYTES
    ) -> Result<(), IcicleError>;

    unsafe fn decapsulate_ffi(
        secret_keys: *const u8, // batch_size × SECRET_KEY_BYTES
        ciphertexts: *const u8, // batch_size × CIPHERTEXT_BYTES
        config: *const MlKemConfig,
        shared_secrets: *mut u8, // batch_size × SHARED_SECRET_BYTES
    ) -> Result<(), IcicleError>;

    fn keygen(
        entropy: &(impl HostOrDeviceSlice<u8> + ?Sized), // batch_size × 64 bytes
        config: &MlKemConfig,
        public_keys: &mut (impl HostOrDeviceSlice<u8> + ?Sized), // batch_size × PUBLIC_KEY_BYTES
        secret_keys: &mut (impl HostOrDeviceSlice<u8> + ?Sized), // batch_size × SECRET_KEY_BYTES
    ) -> Result<(), IcicleError> {
        let mut config = config.clone();
        if entropy.len() != config.batch_size * ENTROPY_BYTES {
            return Err(IcicleError::new(
                eIcicleError::InvalidArgument,
                format!(
                    "entropy length does not match batch size * entropy bytes: {} != {}",
                    entropy.len(),
                    config.batch_size * ENTROPY_BYTES
                ),
            ));
        }
        if public_keys.len() != config.batch_size * Self::PUBLIC_KEY_BYTES {
            return Err(IcicleError::new(
                eIcicleError::InvalidArgument,
                format!(
                    "public keys length does not match batch size * public key bytes: {} != {}",
                    public_keys.len(),
                    config.batch_size * Self::PUBLIC_KEY_BYTES
                ),
            ));
        }
        if secret_keys.len() != config.batch_size * Self::SECRET_KEY_BYTES {
            return Err(IcicleError::new(
                eIcicleError::InvalidArgument,
                format!(
                    "secret keys length does not match batch size * secret key bytes: {} != {}",
                    secret_keys.len(),
                    config.batch_size * Self::SECRET_KEY_BYTES
                ),
            ));
        }
        if entropy.is_on_device() && !entropy.is_on_active_device() {
            return Err(IcicleError::new(
                eIcicleError::InvalidArgument,
                "entropy is allocated on an inactive device",
            ));
        }
        if public_keys.is_on_device() && !public_keys.is_on_active_device() {
            return Err(IcicleError::new(
                eIcicleError::InvalidArgument,
                "public keys is allocated on an inactive device",
            ));
        }
        if secret_keys.is_on_device() && !secret_keys.is_on_active_device() {
            return Err(IcicleError::new(
                eIcicleError::InvalidArgument,
                "secret keys is allocated on an inactive device",
            ));
        }

        config.entropy_on_device = entropy.is_on_device();
        config.public_keys_on_device = public_keys.is_on_device();
        config.secret_keys_on_device = secret_keys.is_on_device();

        unsafe {
            Self::keygen_ffi(
                entropy.as_ptr(),
                &config,
                public_keys.as_mut_ptr(),
                secret_keys.as_mut_ptr(),
            )
        }
    }

    fn encapsulate(
        message: &(impl HostOrDeviceSlice<u8> + ?Sized), // batch_size × 32 bytes
        public_keys: &(impl HostOrDeviceSlice<u8> + ?Sized), // batch_size × PUBLIC_KEY_BYTES
        config: &MlKemConfig,
        ciphertexts: &mut (impl HostOrDeviceSlice<u8> + ?Sized), // batch_size × CIPHERTEXT_BYTES
        shared_secrets: &mut (impl HostOrDeviceSlice<u8> + ?Sized), // batch_size × SHARED_SECRET_BYTES
    ) -> Result<(), IcicleError> {
        let mut config = config.clone();
        if message.len() != config.batch_size * MESSAGE_BYTES {
            return Err(IcicleError::new(
                eIcicleError::InvalidArgument,
                format!(
                    "message length does not match batch size * message bytes: {} != {}",
                    message.len(),
                    config.batch_size * MESSAGE_BYTES
                ),
            ));
        }
        if public_keys.len() != config.batch_size * Self::PUBLIC_KEY_BYTES {
            return Err(IcicleError::new(
                eIcicleError::InvalidArgument,
                format!(
                    "public keys length does not match batch size * public key bytes: {} != {}",
                    public_keys.len(),
                    config.batch_size * Self::PUBLIC_KEY_BYTES
                ),
            ));
        }
        if ciphertexts.len() != config.batch_size * Self::CIPHERTEXT_BYTES {
            return Err(IcicleError::new(
                eIcicleError::InvalidArgument,
                format!(
                    "ciphertexts length does not match batch size * ciphertext bytes: {} != {}",
                    ciphertexts.len(),
                    config.batch_size * Self::CIPHERTEXT_BYTES
                ),
            ));
        }
        if shared_secrets.len() != config.batch_size * Self::SHARED_SECRET_BYTES {
            return Err(IcicleError::new(
                eIcicleError::InvalidArgument,
                format!(
                    "shared secrets length does not match batch size * shared secret bytes: {} != {}",
                    shared_secrets.len(),
                    config.batch_size * Self::SHARED_SECRET_BYTES
                ),
            ));
        }

        if message.is_on_device() && !message.is_on_active_device() {
            return Err(IcicleError::new(
                eIcicleError::InvalidArgument,
                "message is allocated on an inactive device",
            ));
        }
        if public_keys.is_on_device() && !public_keys.is_on_active_device() {
            return Err(IcicleError::new(
                eIcicleError::InvalidArgument,
                "public keys is allocated on an inactive device",
            ));
        }
        if ciphertexts.is_on_device() && !ciphertexts.is_on_active_device() {
            return Err(IcicleError::new(
                eIcicleError::InvalidArgument,
                "ciphertexts is allocated on an inactive device",
            ));
        }
        if shared_secrets.is_on_device() && !shared_secrets.is_on_active_device() {
            return Err(IcicleError::new(
                eIcicleError::InvalidArgument,
                "shared secrets is allocated on an inactive device",
            ));
        }

        config.messages_on_device = message.is_on_device();
        config.public_keys_on_device = public_keys.is_on_device();
        config.ciphertexts_on_device = ciphertexts.is_on_device();
        config.shared_secrets_on_device = shared_secrets.is_on_device();

        unsafe {
            Self::encapsulate_ffi(
                message.as_ptr(),
                public_keys.as_ptr(),
                &config,
                ciphertexts.as_mut_ptr(),
                shared_secrets.as_mut_ptr(),
            )
        }
    }

    fn decapsulate(
        secret_keys: &(impl HostOrDeviceSlice<u8> + ?Sized), // batch_size × SECRET_KEY_BYTES
        ciphertexts: &(impl HostOrDeviceSlice<u8> + ?Sized), // batch_size × CIPHERTEXT_BYTES
        config: &MlKemConfig,
        shared_secrets: &mut (impl HostOrDeviceSlice<u8> + ?Sized), // batch_size × SHARED_SECRET_BYTES
    ) -> Result<(), IcicleError> {
        let mut config = config.clone();
        if secret_keys.len() != config.batch_size * Self::SECRET_KEY_BYTES {
            return Err(IcicleError::new(
                eIcicleError::InvalidArgument,
                format!(
                    "secret keys length does not match batch size * secret key bytes: {} != {}",
                    secret_keys.len(),
                    config.batch_size * Self::SECRET_KEY_BYTES
                ),
            ));
        }
        if ciphertexts.len() != config.batch_size * Self::CIPHERTEXT_BYTES {
            return Err(IcicleError::new(
                eIcicleError::InvalidArgument,
                format!(
                    "ciphertexts length does not match batch size * ciphertext bytes: {} != {}",
                    ciphertexts.len(),
                    config.batch_size * Self::CIPHERTEXT_BYTES
                ),
            ));
        }
        if shared_secrets.len() != config.batch_size * Self::SHARED_SECRET_BYTES {
            return Err(IcicleError::new(
                eIcicleError::InvalidArgument,
                format!(
                    "shared secrets length does not match batch size * shared secret bytes: {} != {}",
                    shared_secrets.len(),
                    config.batch_size * Self::SHARED_SECRET_BYTES
                ),
            ));
        }

        if secret_keys.is_on_device() && !secret_keys.is_on_active_device() {
            return Err(IcicleError::new(
                eIcicleError::InvalidArgument,
                "secret keys is allocated on an inactive device",
            ));
        }
        if ciphertexts.is_on_device() && !ciphertexts.is_on_active_device() {
            return Err(IcicleError::new(
                eIcicleError::InvalidArgument,
                "ciphertexts is allocated on an inactive device",
            ));
        }
        if shared_secrets.is_on_device() && !shared_secrets.is_on_active_device() {
            return Err(IcicleError::new(
                eIcicleError::InvalidArgument,
                "shared secrets is allocated on an inactive device",
            ));
        }

        config.secret_keys_on_device = secret_keys.is_on_device();
        config.ciphertexts_on_device = ciphertexts.is_on_device();
        config.shared_secrets_on_device = shared_secrets.is_on_device();

        unsafe {
            Self::decapsulate_ffi(
                secret_keys.as_ptr(),
                ciphertexts.as_ptr(),
                &config,
                shared_secrets.as_mut_ptr(),
            )
        }
    }
}

// Parameter sets for Kyber variants
#[derive(Debug, Clone, Copy)]
pub struct Kyber512Params;

impl KyberParams for Kyber512Params {
    const PUBLIC_KEY_BYTES: usize = 800;
    const SECRET_KEY_BYTES: usize = 1632;
    const CIPHERTEXT_BYTES: usize = 768;
    const SHARED_SECRET_BYTES: usize = 32;
    const K: u8 = 2;
    const ETA1: u8 = 3;
    const ETA2: u8 = 2;
    const DU: u8 = 10;
    const DV: u8 = 4;

    unsafe fn keygen_ffi(
        entropy: *const u8, // batch_size × 64 bytes
        config: *const MlKemConfig,
        public_keys: *mut u8, // batch_size × PUBLIC_KEY_BYTES
        secret_keys: *mut u8, // batch_size × SECRET_KEY_BYTES
    ) -> Result<(), IcicleError> {
        keygen_ffi512(entropy, config, public_keys, secret_keys).wrap()
    }

    unsafe fn encapsulate_ffi(
        message: *const u8,
        public_keys: *const u8,
        config: *const MlKemConfig,
        ciphertexts: *mut u8,
        shared_secrets: *mut u8,
    ) -> Result<(), IcicleError> {
        encapsulate_ffi512(message, public_keys, config, ciphertexts, shared_secrets).wrap()
    }

    unsafe fn decapsulate_ffi(
        secret_keys: *const u8,
        ciphertexts: *const u8,
        config: *const MlKemConfig,
        shared_secrets: *mut u8,
    ) -> Result<(), IcicleError> {
        decapsulate_ffi512(secret_keys, ciphertexts, config, shared_secrets).wrap()
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Kyber768Params;

impl KyberParams for Kyber768Params {
    const PUBLIC_KEY_BYTES: usize = 1184;
    const SECRET_KEY_BYTES: usize = 2400;
    const CIPHERTEXT_BYTES: usize = 1088;
    const SHARED_SECRET_BYTES: usize = 32;
    const K: u8 = 3;
    const ETA1: u8 = 2;
    const ETA2: u8 = 2;
    const DU: u8 = 10;
    const DV: u8 = 4;

    unsafe fn keygen_ffi(
        entropy: *const u8,
        config: *const MlKemConfig,
        public_keys: *mut u8,
        secret_keys: *mut u8,
    ) -> Result<(), IcicleError> {
        keygen_ffi768(entropy, config, public_keys, secret_keys).wrap()
    }

    unsafe fn encapsulate_ffi(
        message: *const u8,
        public_keys: *const u8,
        config: *const MlKemConfig,
        ciphertexts: *mut u8,
        shared_secrets: *mut u8,
    ) -> Result<(), IcicleError> {
        encapsulate_ffi768(message, public_keys, config, ciphertexts, shared_secrets).wrap()
    }

    unsafe fn decapsulate_ffi(
        secret_keys: *const u8,
        ciphertexts: *const u8,
        config: *const MlKemConfig,
        shared_secrets: *mut u8,
    ) -> Result<(), IcicleError> {
        decapsulate_ffi768(secret_keys, ciphertexts, config, shared_secrets).wrap()
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Kyber1024Params;

impl KyberParams for Kyber1024Params {
    const PUBLIC_KEY_BYTES: usize = 1568;
    const SECRET_KEY_BYTES: usize = 3168;
    const CIPHERTEXT_BYTES: usize = 1568;
    const SHARED_SECRET_BYTES: usize = 32;
    const K: u8 = 4;
    const ETA1: u8 = 2;
    const ETA2: u8 = 2;
    const DU: u8 = 11;
    const DV: u8 = 5;

    unsafe fn keygen_ffi(
        entropy: *const u8,
        config: *const MlKemConfig,
        public_keys: *mut u8,
        secret_keys: *mut u8,
    ) -> Result<(), IcicleError> {
        keygen_ffi1024(entropy, config, public_keys, secret_keys).wrap()
    }

    unsafe fn encapsulate_ffi(
        message: *const u8,
        public_keys: *const u8,
        config: *const MlKemConfig,
        ciphertexts: *mut u8,
        shared_secrets: *mut u8,
    ) -> Result<(), IcicleError> {
        encapsulate_ffi1024(message, public_keys, config, ciphertexts, shared_secrets).wrap()
    }

    unsafe fn decapsulate_ffi(
        secret_keys: *const u8,
        ciphertexts: *const u8,
        config: *const MlKemConfig,
        shared_secrets: *mut u8,
    ) -> Result<(), IcicleError> {
        decapsulate_ffi1024(secret_keys, ciphertexts, config, shared_secrets).wrap()
    }
}
