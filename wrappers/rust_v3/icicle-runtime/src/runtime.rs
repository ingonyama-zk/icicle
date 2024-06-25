use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_void};

use crate::device::{Device, DeviceProperties};
use crate::errors::eIcicleError;

pub type IcicleStreamHandle = *mut c_void;

extern "C" {
    fn icicle_load_backend(path: *const c_char, is_recursive: bool) -> eIcicleError;
    fn icicle_set_device(device: &Device) -> eIcicleError;
    fn icicle_get_active_device(device: &mut Device) -> eIcicleError;
    fn icicle_get_device_count(device_count: &i32) -> eIcicleError;
    fn icicle_is_device_avialable(device: &Device) -> eIcicleError;
    pub fn icicle_malloc(ptr: *mut *mut c_void, size: usize) -> eIcicleError;
    pub fn icicle_malloc_async(ptr: *mut *mut c_void, size: usize, stream: IcicleStreamHandle) -> eIcicleError;
    pub fn icicle_free(ptr: *mut c_void) -> eIcicleError;
    pub fn icicle_free_async(ptr: *mut c_void, stream: IcicleStreamHandle) -> eIcicleError;
    fn icicle_get_available_memory(total: *mut usize, free: *mut usize) -> eIcicleError;
    pub fn icicle_copy_to_host(dst: *mut c_void, src: *const c_void, size: usize) -> eIcicleError;
    pub fn icicle_copy_to_host_async(
        dst: *mut c_void,
        src: *const c_void,
        size: usize,
        stream: IcicleStreamHandle,
    ) -> eIcicleError;
    pub fn icicle_copy_to_device(dst: *mut c_void, src: *const c_void, size: usize) -> eIcicleError;
    pub fn icicle_copy_to_device_async(
        dst: *mut c_void,
        src: *const c_void,
        size: usize,
        stream: IcicleStreamHandle,
    ) -> eIcicleError;
    pub fn icicle_create_stream(stream: *mut IcicleStreamHandle) -> eIcicleError;
    pub fn icicle_destroy_stream(stream: IcicleStreamHandle) -> eIcicleError;
    pub fn icicle_stream_synchronize(stream: IcicleStreamHandle) -> eIcicleError;
    fn icicle_device_synchronize() -> eIcicleError;
    fn icicle_get_device_properties(properties: *mut DeviceProperties) -> eIcicleError;
    fn icicle_get_registered_devices(output: *mut c_char, output_size: usize) -> eIcicleError;
}

pub fn load_backend(path: &str, is_recursive: bool) -> Result<(), eIcicleError> {
    let c_path = CString::new(path).unwrap();
    unsafe { icicle_load_backend(c_path.as_ptr(), is_recursive).wrap() }
}

pub fn set_device(device: &Device) -> Result<(), eIcicleError> {
    let result = unsafe { icicle_set_device(device) };
    if result == eIcicleError::Success {
        Ok(())
    } else {
        Err(result)
    }
}

pub fn get_active_device() -> Result<Device, eIcicleError> {
    let mut device: Device = Device::new("invalid", -1);
    unsafe { icicle_get_active_device(&mut device).wrap_value::<Device>(device) }
}

pub fn get_device_count() -> Result<i32, eIcicleError> {
    let mut device_count = 0;
    unsafe { icicle_get_device_count(&mut device_count).wrap_value::<i32>(device_count) }
}

pub fn is_device_available(device: &Device) -> bool {
    let err = unsafe { icicle_is_device_avialable(device) };
    err == eIcicleError::Success
}

pub fn get_available_memory() -> Result<(usize, usize), eIcicleError> {
    let mut total: usize = 0;
    let mut free: usize = 0;
    let result = unsafe { icicle_get_available_memory(&mut total, &mut free) };
    if result == eIcicleError::Success {
        Ok((total, free))
    } else {
        Err(result)
    }
}

pub fn device_synchronize() -> Result<(), eIcicleError> {
    unsafe { icicle_device_synchronize().wrap() }
}

pub fn get_device_properties() -> Result<DeviceProperties, eIcicleError> {
    let mut properties = DeviceProperties {
        using_host_memory: false,
        num_memory_regions: 0,
        supports_pinned_memory: false,
    };
    let result = unsafe { icicle_get_device_properties(&mut properties) };
    if result == eIcicleError::Success {
        Ok(properties)
    } else {
        Err(result)
    }
}

pub fn get_registered_devices() -> Result<Vec<String>, eIcicleError> {
    const BUFFER_SIZE: usize = 256;
    let mut buffer = vec![0 as c_char; BUFFER_SIZE];

    unsafe {
        let result = icicle_get_registered_devices(buffer.as_mut_ptr(), BUFFER_SIZE);
        if result != eIcicleError::Success {
            return Err(result);
        }

        let c_str = CStr::from_ptr(buffer.as_ptr());
        let str_slice: &str = c_str
            .to_str()
            .unwrap();
        let devices: Vec<String> = str_slice
            .split(',')
            .map(|s| s.to_string())
            .collect();
        Ok(devices)
    }
}
