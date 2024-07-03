use std::ffi::{CStr, CString};
use std::fmt;
use std::os::raw::c_char;

const MAX_TYPE_SIZE: usize = 64;

#[derive(Clone)]
#[repr(C)]
pub struct Device {
    device_type: [c_char; MAX_TYPE_SIZE],
    pub id: i32,
}

#[derive(Debug, Clone)]
#[repr(C)]
pub struct DeviceProperties {
    pub using_host_memory: bool,
    pub num_memory_regions: i32,
    pub supports_pinned_memory: bool,
}

impl Device {
    pub fn new(type_: &str, id: i32) -> Device {
        // copy the string to a c-string for C++ side
        let mut device_type = [0 as c_char; MAX_TYPE_SIZE];
        let c_string = CString::new(type_).expect("CString::new failed");
        let bytes = c_string.as_bytes_with_nul();

        for (i, &byte) in bytes
            .iter()
            .enumerate()
        {
            if i < MAX_TYPE_SIZE {
                device_type[i] = byte as c_char;
            } else {
                break;
            }
        }

        // Ensure the last character is null if the source string is too long
        if bytes.len() > MAX_TYPE_SIZE {
            device_type[MAX_TYPE_SIZE - 1] = 0;
        }

        Device { device_type, id }
    }

    /// Returns the device_type as a Rust String.
    pub fn get_device_type(&self) -> String {
        // Find the first null byte in the array to handle C strings
        let c_str = unsafe {
            CStr::from_ptr(
                self.device_type
                    .as_ptr(),
            )
        };
        c_str
            .to_string_lossy()
            .into_owned()
    }
}

impl fmt::Debug for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let device_type_str = unsafe {
            CStr::from_ptr(
                self.device_type
                    .as_ptr(),
            )
            .to_str()
            .unwrap_or("Invalid UTF-8")
        };
        f.debug_struct("Device")
            .field("device_type", &device_type_str)
            .field("id", &self.id)
            .finish()
    }
}

