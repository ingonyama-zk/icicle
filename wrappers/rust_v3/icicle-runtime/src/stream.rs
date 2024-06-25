use crate::errors::eIcicleError;
use crate::runtime;
use std::{ops::Deref, os::raw::c_void};

// Define the type alias for IcicleStreamHandle
pub type IcicleStreamHandle = *mut c_void;

#[repr(transparent)]
#[derive(Debug)]
pub struct IcicleStream {
    pub handle: IcicleStreamHandle,
}

unsafe impl Sync for IcicleStream {}

impl IcicleStream {
    pub(crate) fn from_handle(handle: IcicleStreamHandle) -> Self {
        Self { handle }
    }

    pub fn create() -> Result<Self, eIcicleError> {
        let mut handle = std::ptr::null_mut();
        unsafe { runtime::icicle_create_stream(&mut handle).wrap_value(Self::from_handle(handle)) }
    }

    pub fn synchronize(&self) -> Result<(), eIcicleError> {
        unsafe { runtime::icicle_stream_synchronize(self.handle).wrap() }
    }

    pub fn is_null(&self) -> bool {
        self.handle
            .is_null()
    }

    pub fn destroy(&mut self) -> Result<(), eIcicleError> {
        if !self
            .handle
            .is_null()
        {
            let err = unsafe { runtime::icicle_destroy_stream(self.handle) };
            self.handle = std::ptr::null_mut();
            return err.wrap();
        }
        Ok(())
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
            eprintln!("Warning: IcicleStream was not explicitly destroyed. Make sure to call stream.destroy() to release the stream resource.");
            // let _ = unsafe { runtime::icicle_destroy_stream(self.handle) };
        }
    }
}

impl Deref for IcicleStream {
    type Target = IcicleStreamHandle;

    fn deref(&self) -> &Self::Target {
        &self.handle
    }
}

impl From<&IcicleStream> for IcicleStreamHandle {
    fn from(stream: &IcicleStream) -> Self {
        stream.handle
    }
}
