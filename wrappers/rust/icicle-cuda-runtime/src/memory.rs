use crate::bindings::{
    cudaFree, cudaMalloc, cudaMallocAsync, cudaMemPool_t, cudaMemcpy, cudaMemcpyAsync, cudaMemcpyKind,
};
use crate::device::get_device;
use crate::error::{CudaError, CudaResult, CudaResultWrap};
use crate::stream::CudaStream;
use std::mem::{size_of, MaybeUninit};
use std::ops::{Index, IndexMut, Range, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive};
use std::os::raw::c_void;
use std::slice::from_raw_parts_mut;

pub enum HostOrDeviceSlice<'a, T> {
    Host(Vec<T>),
    Device(&'a mut [T], i32),
}

impl<'a, T> HostOrDeviceSlice<'a, T> {
    // Function to get the device_id for Device variant
    pub fn get_device_id(&self) -> Option<i32> {
        match self {
            HostOrDeviceSlice::Device(_, device_id) => Some(*device_id),
            HostOrDeviceSlice::Host(_) => None,
        }
    }

    fn check_device(device_id: i32) {
        match device_id == get_device().unwrap() as i32 {
            true => (),
            false => panic!("Attempt to use on a different device"),
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Self::Device(s, _) => s.len(),
            Self::Host(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        match self {
            Self::Device(s, _) => s.is_empty(),
            Self::Host(v) => v.is_empty(),
        }
    }

    pub fn is_on_device(&self) -> bool {
        match self {
            Self::Device(_, _) => true,
            Self::Host(_) => false,
        }
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        match self {
            Self::Device(_, _) => panic!("Use copy_to_host and copy_to_host_async to move device data to a slice"),
            Self::Host(v) => v.as_mut_slice(),
        }
    }

    pub fn as_slice(&self) -> &[T] {
        match self {
            Self::Device(_, _) => panic!("Use copy_to_host and copy_to_host_async to move device data to a slice"),
            Self::Host(v) => v.as_slice(),
        }
    }

    pub fn as_ptr(&self) -> *const T {
        match self {
            Self::Device(s, _) => s.as_ptr(),
            Self::Host(v) => v.as_ptr(),
        }
    }

    pub fn as_mut_ptr(&mut self) -> *mut T {
        match self {
            Self::Device(s, _) => s.as_mut_ptr(),
            Self::Host(v) => v.as_mut_ptr(),
        }
    }

    pub fn on_host(src: Vec<T>) -> Self {
        //TODO: HostOrDeviceSlice on_host() with slice input without actually copying the data
        Self::Host(src)
    }

    pub fn cuda_malloc(count: usize) -> CudaResult<Self> {
        let size = count
            .checked_mul(size_of::<T>())
            .unwrap_or(0);
        if size == 0 {
            return Err(CudaError::cudaErrorMemoryAllocation); //TODO: only CUDA backend should return CudaError
        }

        let mut device_ptr = MaybeUninit::<*mut c_void>::uninit();
        unsafe {
            cudaMalloc(device_ptr.as_mut_ptr(), size).wrap()?;
            Ok(Self::Device(
                from_raw_parts_mut(device_ptr.assume_init() as *mut T, count),
                get_device().unwrap() as i32,
            ))
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
            Ok(Self::Device(
                from_raw_parts_mut(device_ptr.assume_init() as *mut T, count),
                get_device().unwrap() as i32,
            ))
        }
    }

    pub fn copy_from_host(&mut self, val: &[T]) -> CudaResult<()> {
        match self {
            Self::Device(_, device_id) => Self::check_device(*device_id),
            Self::Host(_) => panic!("Need device memory to copy into, and not host"),
        };
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
        match self {
            Self::Device(_, device_id) => Self::check_device(*device_id),
            Self::Host(_) => panic!("Need device memory to copy from, and not host"),
        };
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
        match self {
            Self::Device(_, device_id) => Self::check_device(*device_id),
            Self::Host(_) => panic!("Need device memory to copy into, and not host"),
        };
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
        match self {
            Self::Device(_, device_id) => Self::check_device(*device_id),
            Self::Host(_) => panic!("Need device memory to copy from, and not host"),
        };
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

macro_rules! impl_index {
    ($($t:ty)*) => {
        $(
            impl<'a, T> Index<$t> for HostOrDeviceSlice<'a, T>
            {
                type Output = [T];

                fn index(&self, index: $t) -> &Self::Output {
                    match self {
                        Self::Device(s, _) => s.index(index),
                        Self::Host(v) => v.index(index),
                    }
                }
            }

            impl<'a, T> IndexMut<$t> for HostOrDeviceSlice<'a, T>
            {
                fn index_mut(&mut self, index: $t) -> &mut Self::Output {
                    match self {
                        Self::Device(s,_) => s.index_mut(index),
                        Self::Host(v) => v.index_mut(index),
                    }
                }
            }
        )*
    }
}
impl_index! {
    Range<usize>
    RangeFull
    RangeFrom<usize>
    RangeInclusive<usize>
    RangeTo<usize>
    RangeToInclusive<usize>
}

impl<'a, T> Drop for HostOrDeviceSlice<'a, T> {
    fn drop(&mut self) {
        match self {
            Self::Device(s, device_id) => {
                Self::check_device(*device_id);
                if s.is_empty() {
                    return;
                }

                unsafe {
                    cudaFree(s.as_mut_ptr() as *mut c_void)
                        .wrap()
                        .unwrap();
                }
            }
            Self::Host(_) => {}
        }
    }
}

#[allow(non_camel_case_types)]
pub type CudaMemPool = cudaMemPool_t;
