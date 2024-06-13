pub mod device;
pub mod errors;
pub mod runtime;

// Re-export the types for easier access
pub use device::{Device, DeviceProperties};
pub use errors::eIcicleError;
pub use runtime::*;

use std::ffi::CString;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_set_device() {
        // load backends to process
        let backend_install_dir = env!("DEFAULT_BACKEND_INSTALL_DIR");
        assert_eq!(load_backend(&backend_install_dir, true), eIcicleError::Success);

        let devtype = String::from("CPU");
        let dev = Device::new(&devtype, 0);
        assert_eq!(set_device(&dev), eIcicleError::Success);
    }
}
