use icicle_runtime::{device::Device, runtime};
use once_cell::sync::Lazy;
use std::sync::{Mutex, OnceLock};

// This module is used to load backends and choose a main and reference devices for tests

static INIT: OnceLock<()> = OnceLock::new();
pub static TEST_MAIN_DEVICE: Lazy<Mutex<Device>> = Lazy::new(|| Mutex::new(Device::new("", 0)));
pub static TEST_REF_DEVICE: Lazy<Mutex<Device>> = Lazy::new(|| Mutex::new(Device::new("", 0)));

pub fn test_load_and_init_devices() {
    INIT.get_or_init(move || {
        runtime::load_backend(&env!("DEFAULT_BACKEND_INSTALL_DIR"), true).unwrap();
        let registered_devices = runtime::get_registered_devices().unwrap();
        assert!(registered_devices.len() >= 2);
        // select main and ref devices
        let mut main_device = TEST_MAIN_DEVICE
            .lock()
            .unwrap();
        *main_device = Device::new(&registered_devices[0], 0);
        let mut ref_device = TEST_REF_DEVICE
            .lock()
            .unwrap();
        *ref_device = Device::new(&registered_devices[0] /* TODO YUVAL SHOULD BE 1 */, 0);
    });
}

pub fn test_set_main_device() {
    let main_device = TEST_MAIN_DEVICE
        .lock()
        .unwrap();
    runtime::set_device(&main_device).unwrap();
}

pub fn test_set_ref_device() {
    let ref_device = TEST_REF_DEVICE
        .lock()
        .unwrap();
    runtime::set_device(&ref_device).unwrap();
}
