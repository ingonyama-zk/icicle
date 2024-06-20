use icicle_runtime::{config::ConfigExtension, stream::IcicleStreamHandle};

pub mod tests;

#[repr(C)]
#[derive(Debug)]
pub struct VecOpsConfig {
    pub stream_handle: IcicleStreamHandle,
    pub is_a_on_device: bool,
    pub is_b_on_device: bool,
    pub is_result_on_device: bool,
    pub is_async: bool,
    pub ext: ConfigExtension,
}

impl VecOpsConfig {
    pub fn default() -> Self {
        Self {
            stream_handle: std::ptr::null_mut(),
            is_a_on_device: false,
            is_b_on_device: false,
            is_result_on_device: false,
            is_async: false,
            ext: ConfigExtension::new(),
        }
    }
}
