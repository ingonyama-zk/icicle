use crate::errors::eIcicleError;
use crate::runtime;
use std::os::raw::c_void;

// Define the type alias for IcicleStreamHandle
type IcicleStreamHandle = *mut c_void;

#[repr(transparent)]
#[derive(Debug)]
pub struct IcicleStream {
    pub(crate) handle: IcicleStreamHandle,
}

unsafe impl Sync for IcicleStream {}

impl IcicleStream {
    pub(crate) fn from_handle(handle: IcicleStreamHandle) -> Self {
        Self { handle }
    }

    pub fn create() -> Result<Self, eIcicleError> {
        let mut handle = std::ptr::null_mut();
        unsafe {
            let error = runtime::icicle_create_stream(&mut handle);
            if error != eIcicleError::Success {
                return Err(error);
            }
        }

        Ok(Self::from_handle(handle))
    }

    pub fn synchronize(&self) -> eIcicleError {
        unsafe { runtime::icicle_stream_synchronize(self.handle) }
    }
}

impl Default for IcicleStream {
    fn default() -> Self {
        Self {
            handle: std::ptr::null_mut(),
        }
    }
}

impl Drop for IcicleStream {
    fn drop(&mut self) {
        if !self
            .handle
            .is_null()
        {
            let _ = unsafe { runtime::icicle_destroy_stream(self.handle) };
        }
    }
}

impl From<&IcicleStream> for IcicleStreamHandle {
    fn from(stream: &IcicleStream) -> Self {
        stream.handle
    }
}
