use std::ffi::{CStr, CString};
use std::fmt;
use std::os::raw::c_char;

#[repr(C)]
pub struct Device {
    device_type: *const c_char,
    id: i32,
}

#[derive(Debug)]
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

impl fmt::Debug for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let device_type_str = unsafe {
            if self
                .device_type
                .is_null()
            {
                "null"
            } else {
                CStr::from_ptr(self.device_type)
                    .to_str()
                    .unwrap_or("Invalid UTF-8")
            }
        };
        f.debug_struct("Device")
            .field("device_type", &device_type_str)
            .field("id", &self.id)
            .finish()
    }
}
