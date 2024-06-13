use std::ffi::{CStr, CString};
use std::os::raw::c_char;

#[repr(C)]
pub struct Device {
    pub device_type: *const std::os::raw::c_char,
    pub id: i32,
}

#[repr(C)]
pub struct DeviceProperties {
    pub using_host_memory: bool,
    pub num_memory_regions: i32,
    pub supports_pinned_memory: bool,
}

impl Device {
    pub fn new(device_type: &str, id: i32) -> Device {
        let c_string = CString::new(device_type).expect("CString::new failed");
        Device {
            device_type: c_string.into_raw(),
            id,
        }
    }
}
