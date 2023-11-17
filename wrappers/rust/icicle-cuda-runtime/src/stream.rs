use crate::bindings::{cudaStream_t, cudaStreamDefault, cudaStreamNonBlocking, cudaStreamDestroy, cudaStreamCreate, cudaStreamSynchronize};
use crate::error::{CudaResult, CudaResultWrap};
use std::mem::{MaybeUninit, forget};
use bitflags::bitflags;

#[repr(transparent)]
#[derive(Debug)]
pub struct CudaStream {
    pub(crate) handle: cudaStream_t,
}

unsafe impl Sync for CudaStream {}

bitflags! {
    pub struct CudaStreamCreateFlags: u32 {
        const DEFAULT = cudaStreamDefault;
        const NON_BLOCKING = cudaStreamNonBlocking;
    }
}

impl CudaStream {
    pub(crate) fn from_handle(handle: cudaStream_t) -> Self {
        Self { handle }
    }

    pub fn create() -> CudaResult<Self> {
        let mut handle = MaybeUninit::<cudaStream_t>::uninit();
        unsafe {
            cudaStreamCreate(handle.as_mut_ptr())
                .wrap_maybe_uninit(handle)
                .map(CudaStream::from_handle)
        }
    }

    pub fn destroy(self) -> CudaResult<()> {
        let handle = self.handle;
        forget(self);
        if handle.is_null() {
            Ok(())
        } else {
            unsafe { cudaStreamDestroy(handle).wrap() }
        }
    }

    pub fn synchronize(&self) -> CudaResult<()> {
        unsafe { cudaStreamSynchronize(self.handle).wrap() }
    }
}

impl Default for CudaStream {
    fn default() -> Self {
        Self { handle: std::ptr::null_mut(), }
    }
}

impl Drop for CudaStream {
    fn drop(&mut self) {
        let handle = self.handle;
        if handle.is_null() {
            return;
        }
        let _ = unsafe { cudaStreamDestroy(handle) };
    }
}

impl From<&CudaStream> for cudaStream_t {
    fn from(stream: &CudaStream) -> Self {
        stream.handle
    }
}
