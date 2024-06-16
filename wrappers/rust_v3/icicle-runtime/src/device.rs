use std::ffi::CString;
use std::os::raw::c_char;

#[repr(C)]
pub struct Device {
    device_type: *const c_char,
    id: i32,
}

#[repr(C)]
pub struct DeviceProperties {
    pub using_host_memory: bool,
    pub num_memory_regions: i32,
    pub supports_pinned_memory: bool,
}

impl Device {
    pub fn new(device_type: &str, id: i32) -> Device {
        // Note that the CString is released when device is dropped.
        let c_string = CString::new(device_type).expect("CString::new failed");
        Device {
            device_type: c_string.into_raw(),
            id,
        }
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        if !self
            .device_type
            .is_null()
        {
            unsafe {
                // Convert back to CString so it is released
                let _ = CString::from_raw(self.device_type as *mut c_char);
                self.device_type = std::ptr::null();
            }
        }
    }
}
