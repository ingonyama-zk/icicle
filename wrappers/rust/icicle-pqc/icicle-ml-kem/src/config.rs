use icicle_runtime::{config::ConfigExtension, stream::IcicleStreamHandle};

#[repr(C)]
#[derive(Debug, Clone)]
pub struct MlKemConfig {
    pub stream: IcicleStreamHandle, // Corresponds to `icicleStreamHandle stream = nullptr;`
    pub is_async: bool,

    // Host/device location hints for each buffer type
    pub messages_on_device: bool,
    pub entropy_on_device: bool,
    pub public_keys_on_device: bool,
    pub secret_keys_on_device: bool,
    pub ciphertexts_on_device: bool,
    pub shared_secrets_on_device: bool,

    pub batch_size: i32,

    pub ext: ConfigExtension, // Optional backend-specific settings
}

impl Default for MlKemConfig {
    fn default() -> Self {
        Self {
            stream: std::ptr::null_mut(),
            is_async: false,
            messages_on_device: false,
            entropy_on_device: false,
            public_keys_on_device: false,
            secret_keys_on_device: false,
            ciphertexts_on_device: false,
            shared_secrets_on_device: false,
            batch_size: 1,
            ext: ConfigExtension::new(),
        }
    }
}
