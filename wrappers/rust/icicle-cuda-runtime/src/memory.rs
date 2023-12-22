use crate::bindings::{cudaMalloc, cudaMallocAsync, cudaMemcpy, cudaMemcpyAsync, cudaMemcpyKind};
use crate::error::{CudaError, CudaResult, CudaResultWrap};
use crate::stream::CudaStream;
use std::mem::{size_of, MaybeUninit};
use std::os::raw::c_void;
use std::slice;

/// Fixed-size device-side slice.
#[derive(Debug)]
#[repr(C)]
pub struct DeviceSlice<'a, T>(&'a mut [T]);

impl<'a, T> DeviceSlice<'a, T> {
    pub fn len(&self) -> usize {
        self.0
            .len()
    }

    pub fn is_empty(&self) -> bool {
        self.0
            .is_empty()
    }

    pub fn as_slice(&mut self) -> &mut [T] {
        self.0
    }

    pub fn as_ptr(&self) -> *const T {
        self.0
            .as_ptr()
    }

    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.0
            .as_mut_ptr()
    }

    pub fn cuda_malloc(count: usize) -> CudaResult<Self> {
        let size = count
            .checked_mul(size_of::<T>())
            .unwrap_or(0);
        if size == 0 {
            return Err(CudaError::cudaErrorMemoryAllocation);
        }

        let mut device_ptr = MaybeUninit::<*mut c_void>::uninit();
        unsafe {
            cudaMalloc(device_ptr.as_mut_ptr(), size).wrap()?;
            Ok(DeviceSlice {
                0: slice::from_raw_parts_mut(device_ptr.assume_init() as *mut T, count),
            })
        }
    }

    pub fn cuda_malloc_async(count: usize, stream: &CudaStream) -> CudaResult<Self> {
        let size = count
            .checked_mul(size_of::<T>())
            .unwrap_or(0);
        if size == 0 {
            return Err(CudaError::cudaErrorMemoryAllocation);
        }

        let mut device_ptr = MaybeUninit::<*mut c_void>::uninit();
        unsafe {
            cudaMallocAsync(device_ptr.as_mut_ptr(), size, stream.handle as *mut _ as *mut _).wrap()?;
            Ok(DeviceSlice {
                0: slice::from_raw_parts_mut(device_ptr.assume_init() as *mut T, count),
            })
        }
    }

    pub fn copy_from_host(&mut self, val: &[T]) -> CudaResult<()> {
        assert!(
            self.len() == val.len(),
            "destination and source slices have different lengths"
        );
        let size = size_of::<T>() * self.len();
        if size != 0 {
            unsafe {
                cudaMemcpy(
                    self.as_mut_ptr() as *mut c_void,
                    val.as_ptr() as *const c_void,
                    size,
                    cudaMemcpyKind::cudaMemcpyHostToDevice,
                )
                .wrap()?
            }
        }
        Ok(())
    }

    pub fn copy_to_host(&self, val: &mut [T]) -> CudaResult<()> {
        assert!(
            self.len() == val.len(),
            "destination and source slices have different lengths"
        );
        let size = size_of::<T>() * self.len();
        if size != 0 {
            unsafe {
                cudaMemcpy(
                    val.as_mut_ptr() as *mut c_void,
                    self.as_ptr() as *const c_void,
                    size,
                    cudaMemcpyKind::cudaMemcpyDeviceToHost,
                )
                .wrap()?
            }
        }
        Ok(())
    }

    pub fn copy_from_host_async(&mut self, val: &[T], stream: &CudaStream) -> CudaResult<()> {
        assert!(
            self.len() == val.len(),
            "destination and source slices have different lengths"
        );
        let size = size_of::<T>() * self.len();
        if size != 0 {
            unsafe {
                cudaMemcpyAsync(
                    self.as_mut_ptr() as *mut c_void,
                    val.as_ptr() as *const c_void,
                    size,
                    cudaMemcpyKind::cudaMemcpyHostToDevice,
                    stream.handle as *mut _ as *mut _,
                )
                .wrap()?
            }
        }
        Ok(())
    }

    pub fn copy_to_host_async(&self, val: &mut [T], stream: &CudaStream) -> CudaResult<()> {
        assert!(
            self.len() == val.len(),
            "destination and source slices have different lengths"
        );
        let size = size_of::<T>() * self.len();
        if size != 0 {
            unsafe {
                cudaMemcpyAsync(
                    val.as_mut_ptr() as *mut c_void,
                    self.as_ptr() as *const c_void,
                    size,
                    cudaMemcpyKind::cudaMemcpyDeviceToHost,
                    stream.handle as *mut _ as *mut _,
                )
                .wrap()?
            }
        }
        Ok(())
    }
}

#[allow(non_camel_case_types)]
pub type CudaMemPool = usize; // This is a placeholder, TODO: actually make this into a proper CUDA wrapper
