use std::ffi::CString;
use std::os::raw::{c_char, c_int, c_void};

#[repr(C)]
#[derive(Debug)] // TODO: implement Debug trait correctly
pub struct ConfigExtension {
    ext: *mut c_void,
}

extern "C" {
    fn create_config_extension() -> *mut c_void;
    fn destroy_config_extension(ext: *mut c_void);
    fn config_extension_set_int(ext: *mut c_void, key: *const c_char, value: c_int);
    fn config_extension_set_bool(ext: *mut c_void, key: *const c_char, value: bool);
    fn config_extension_get_int(ext: *const c_void, key: *const c_char) -> c_int;
    fn config_extension_get_bool(ext: *const c_void, key: *const c_char) -> bool;
    fn clone_config_extension(ext: *const c_void) -> *mut c_void;
}

impl Default for ConfigExtension {
    fn default() -> Self {
        Self::new()
    }
}

impl ConfigExtension {
    pub fn new() -> Self {
        unsafe {
            Self {
                ext: create_config_extension(),
            }
        }
    }

    pub fn set_int(&self, key: &str, value: i32) {
        let key_c = CString::new(key).unwrap();
        unsafe {
            config_extension_set_int(self.ext, key_c.as_ptr(), value);
        }
    }

    pub fn set_bool(&self, key: &str, value: bool) {
        let key_c = CString::new(key).unwrap();
        unsafe {
            config_extension_set_bool(self.ext, key_c.as_ptr(), value);
        }
    }

    pub fn get_int(&self, key: &str) -> i32 {
        let key_c = CString::new(key).unwrap();
        unsafe { config_extension_get_int(self.ext, key_c.as_ptr()) }
    }

    pub fn get_bool(&self, key: &str) -> bool {
        let key_c = CString::new(key).unwrap();
        unsafe { config_extension_get_bool(self.ext, key_c.as_ptr()) }
    }
}

impl Drop for ConfigExtension {
    fn drop(&mut self) {
        unsafe {
            destroy_config_extension(self.ext);
        }
    }
}

impl Clone for ConfigExtension {
    fn clone(&self) -> Self {
        unsafe {
            ConfigExtension {
                ext: clone_config_extension(self.ext),
            }
        }
    }
}
