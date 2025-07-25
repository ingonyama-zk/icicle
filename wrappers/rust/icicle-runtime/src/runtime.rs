use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_void};

use crate::device::{Device, DeviceProperties};
use crate::errors::{eIcicleError, IcicleError};
use crate::stream::IcicleStream;

pub type IcicleStreamHandle = *mut c_void;

extern "C" {
    fn icicle_load_backend(path: *const c_char, is_recursive: bool) -> eIcicleError;
    fn icicle_load_backend_from_env_or_default() -> eIcicleError;
    fn icicle_set_device(device: &Device) -> eIcicleError;
    fn icicle_set_default_device(device: &Device) -> eIcicleError;
    fn icicle_get_active_device(device: &mut Device) -> eIcicleError;
    fn icicle_is_host_memory(ptr: *const c_void) -> eIcicleError;
    fn icicle_is_active_device_memory(ptr: *const c_void) -> eIcicleError;
    fn icicle_get_device_count(device_count: &mut i32) -> eIcicleError;
    fn icicle_is_device_available(device: &Device) -> eIcicleError;
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
    pub fn icicle_copy(dst: *mut c_void, src: *const c_void, size: usize) -> eIcicleError;
    pub fn icicle_copy_async(
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
    fn icicle_memset(ptr: *mut c_void, value: i32, size: usize) -> eIcicleError;
    fn icicle_memset_async(ptr: *mut c_void, value: i32, size: usize, stream: IcicleStreamHandle) -> eIcicleError;
}

pub fn load_backend_from_env_or_default() -> Result<(), IcicleError> {
    unsafe { icicle_load_backend_from_env_or_default().wrap() }
}

pub fn load_backend(path: &str) -> Result<(), IcicleError> {
    let c_path = CString::new(path).unwrap();
    unsafe { icicle_load_backend(c_path.as_ptr(), true).wrap() }
}

pub fn load_backend_non_recursive(path: &str) -> Result<(), IcicleError> {
    let c_path = CString::new(path).unwrap();
    unsafe { icicle_load_backend(c_path.as_ptr(), false).wrap() }
}

pub fn set_device(device: &Device) -> Result<(), IcicleError> {
    unsafe { icicle_set_device(device).wrap() }
}

pub fn set_default_device(device: &Device) -> Result<(), IcicleError> {
    unsafe { icicle_set_default_device(device).wrap() }
}

pub fn get_active_device() -> Result<Device, IcicleError> {
    let mut device: Device = Device::new("invalid", -1);
    unsafe { icicle_get_active_device(&mut device).wrap_value(device) }
}

pub fn is_host_memory(ptr: *const c_void) -> bool {
    unsafe { eIcicleError::Success == icicle_is_host_memory(ptr) }
}

pub fn is_active_device_memory(ptr: *const c_void) -> bool {
    unsafe { eIcicleError::Success == icicle_is_active_device_memory(ptr) }
}

pub fn get_device_count() -> Result<i32, IcicleError> {
    let mut device_count = 0;
    unsafe { icicle_get_device_count(&mut device_count).wrap_value(device_count) }
}

pub fn is_device_available(device: &Device) -> bool {
    let err = unsafe { icicle_is_device_available(device) };
    err == eIcicleError::Success
}

pub fn get_available_memory() -> Result<(usize, usize), IcicleError> {
    let mut total: usize = 0;
    let mut free: usize = 0;
    unsafe { icicle_get_available_memory(&mut total, &mut free).wrap_value((total, free)) }
}

pub fn device_synchronize() -> Result<(), IcicleError> {
    unsafe { icicle_device_synchronize().wrap() }
}

pub fn get_device_properties() -> Result<DeviceProperties, IcicleError> {
    let mut properties = DeviceProperties {
        using_host_memory: false,
        num_memory_regions: 0,
        supports_pinned_memory: false,
    };
    unsafe { icicle_get_device_properties(&mut properties).wrap_value(properties) }
}

pub fn get_registered_devices() -> Result<Vec<String>, IcicleError> {
    const BUFFER_SIZE: usize = 256;
    let mut buffer = vec![0 as c_char; BUFFER_SIZE];

    unsafe {
        let result = icicle_get_registered_devices(buffer.as_mut_ptr(), BUFFER_SIZE);
        result.wrap()?;

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

// This function pre-allocates default memory pool and warms the GPU up
// so that subsequent memory allocations and other calls are not slowed down
pub fn warmup(stream: &IcicleStream) -> Result<(), IcicleError> {
    let mut device_ptr: *mut c_void = std::ptr::null_mut();
    let free_memory: usize = 1 << 28;
    unsafe {
        icicle_malloc_async(&mut device_ptr, free_memory >> 1, stream.handle).wrap()?;
        icicle_free_async(device_ptr, stream.handle).wrap()
    }
}

pub fn memset(ptr: *mut c_void, value: i32, size: usize) -> Result<(), IcicleError> {
    unsafe { icicle_memset(ptr, value, size).wrap() }
}

pub fn memset_async(ptr: *mut c_void, value: i32, size: usize, stream: *mut c_void) -> Result<(), IcicleError> {
    unsafe { icicle_memset_async(ptr, value, size, stream).wrap() }
}
