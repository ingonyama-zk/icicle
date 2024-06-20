use icicle_runtime::{config::ConfigExtension, stream::IcicleStream};

pub mod tests;

#[repr(C)]
#[derive(Debug)]
pub struct VecOpsConfig {
    pub stream: IcicleStream,
    pub is_a_on_device: bool,
    pub is_b_on_device: bool,
    pub is_result_on_device: bool,
    pub is_async: bool,
    pub ext: ConfigExtension,
}

impl VecOpsConfig {
    pub fn default() -> Self {
        Self {
            stream: IcicleStream::default(),
            is_a_on_device: false,
            is_b_on_device: false,
            is_result_on_device: false,
            is_async: false,
            ext: ConfigExtension::new(),
        }
    }
}

impl Drop for VecOpsConfig {
    fn drop(&mut self) {
        // ConfigExtension will be automatically dropped
    }
}
