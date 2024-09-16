use icicle_runtime::{config::ConfigExtension, stream::IcicleStreamHandle};

pub mod tests;

/// Configuration structure for hash operations.
///
/// The `HashConfig` structure holds various configuration options that control how hash operations
/// are executed. It supports features such as specifying the execution stream, input/output locations
/// (device or host), batch sizes, and backend-specific extensions. Additionally, it allows
/// synchronous and asynchronous execution modes.
#[repr(C)]
pub struct HashConfig {
    pub stream_handle: IcicleStreamHandle,
    pub batch: u64,
    pub are_inputs_on_device: bool,
    pub are_outputs_on_device: bool,
    pub is_async: bool,
    pub ext: ConfigExtension,
}

impl HashConfig {
    /// Create a default configuration (same as the C++ struct's defaults)
    pub fn default() -> Self {
        Self {
            stream_handle: std::ptr::null_mut(),
            batch: 1,
            are_inputs_on_device: false,
            are_outputs_on_device: false,
            is_async: false,
            ext: ConfigExtension::new(),
        }
    }
}
