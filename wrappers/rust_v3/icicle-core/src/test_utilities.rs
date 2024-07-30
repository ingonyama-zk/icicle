use icicle_runtime::{device::Device, runtime};
use once_cell::sync::Lazy;
use std::sync::{Mutex, OnceLock};

// This module is used to load backends and choose a main and reference devices for tests

static INIT: OnceLock<()> = OnceLock::new();
pub static TEST_MAIN_DEVICE: Lazy<Mutex<Device>> = Lazy::new(|| Mutex::new(Device::new("CUDA", 0)));
pub static TEST_REF_DEVICE: Lazy<Mutex<Device>> = Lazy::new(|| Mutex::new(Device::new("CPU", 0)));

pub fn test_load_and_init_devices() {
    INIT.get_or_init(move || {
        runtime::load_backend(env!("DEFAULT_BACKEND_INSTALL_DIR")).unwrap();
        let registered_devices = runtime::get_registered_devices().unwrap();
        assert!(registered_devices.len() >= 2);
        // select main and ref devices

        let mut main_device = TEST_MAIN_DEVICE
            .lock()
            .unwrap();
        if !runtime::is_device_available(&main_device) {
            *main_device = Device::new(&registered_devices[0], 0);
        }

        let mut ref_device = TEST_REF_DEVICE
            .lock()
            .unwrap();
        if !runtime::is_device_available(&ref_device) {
            *ref_device = Device::new(&registered_devices[1], 0);
        }
    });
}

pub fn test_set_main_device() {
    let main_device = TEST_MAIN_DEVICE
        .lock()
        .unwrap();
    runtime::set_device(&main_device).unwrap();
}

pub fn test_set_main_device_with_id(device_id: i32) {
    let main_device = TEST_MAIN_DEVICE
        .lock()
        .unwrap();
    let mut device = main_device.clone();
    device.id = device_id;
    runtime::set_device(&device).unwrap();
}

pub fn test_set_ref_device() {
    let ref_device = TEST_REF_DEVICE
        .lock()
        .unwrap();
    runtime::set_device(&ref_device).unwrap();
}

pub fn test_set_ref_device_with_id(device_id: i32) {
    let ref_device = TEST_REF_DEVICE
        .lock()
        .unwrap();
    let mut device = ref_device.clone();
    device.id = device_id;
    runtime::set_device(&device).unwrap();
}
