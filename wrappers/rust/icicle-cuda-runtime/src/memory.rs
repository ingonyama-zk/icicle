use crate::bindings::{
    cudaDeviceAttr, cudaFree, cudaFreeHost, cudaHostAlloc, cudaHostAllocDefault, cudaHostAllocPortable,
    cudaHostGetFlags, cudaHostRegister, cudaHostRegisterDefault, cudaHostRegisterPortable, cudaHostUnregister,
    cudaMalloc, cudaMallocAsync, cudaMemPool_t, cudaMemcpy, cudaMemcpyAsync, cudaMemcpyKind,
};
use crate::device::{check_device, get_device_attribute, get_device_from_pointer};
use crate::error::{CudaError, CudaResult, CudaResultWrap};
use crate::stream::CudaStream;
use bitflags::bitflags;
use std::mem::{size_of, ManuallyDrop, MaybeUninit};
use std::ops::{
    Deref, DerefMut, Index, IndexMut, Range, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive,
};
use std::os::raw::{c_uint, c_void};
use std::slice::from_raw_parts_mut;
use std::slice::SliceIndex;

bitflags! {
    pub struct CudaHostAllocFlags: u32 {
        const DEFAULT = cudaHostAllocDefault;
        const PORTABLE = cudaHostAllocPortable;
    }
}

bitflags! {
    pub struct CudaHostRegisterFlags: u32 {
        const DEFAULT = cudaHostRegisterDefault;
        const PORTABLE = cudaHostRegisterPortable;
    }
}

#[derive(Debug)]
pub struct HostSlice<T: Sized>([T]);
pub struct DeviceVec<T>(ManuallyDrop<Box<[T]>>);
pub struct DeviceSlice<T>([T]);

pub trait HostOrDeviceSlice<T> {
    fn is_on_device(&self) -> bool;
    fn device_id(&self) -> Option<usize>;
    unsafe fn as_ptr(&self) -> *const T;
    unsafe fn as_mut_ptr(&mut self) -> *mut T;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool;
}

impl<T> HostOrDeviceSlice<T> for HostSlice<T> {
    fn is_on_device(&self) -> bool {
        false
    }

    fn device_id(&self) -> Option<usize> {
        None
    }

    unsafe fn as_ptr(&self) -> *const T {
        self.0
            .as_ptr()
    }

    unsafe fn as_mut_ptr(&mut self) -> *mut T {
        self.0
            .as_mut_ptr()
    }

    fn len(&self) -> usize {
        self.0
            .len()
    }

    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<T> HostOrDeviceSlice<T> for DeviceSlice<T> {
    fn is_on_device(&self) -> bool {
        true
    }

    fn device_id(&self) -> Option<usize> {
        Some(
            get_device_from_pointer(unsafe { self.as_ptr() as *const ::std::os::raw::c_void })
                .expect("Invalid pointer. Maybe host pointer was used here?"),
        )
    }

    unsafe fn as_ptr(&self) -> *const T {
        self.0
            .as_ptr()
    }

    unsafe fn as_mut_ptr(&mut self) -> *mut T {
        self.0
            .as_mut_ptr()
    }

    fn len(&self) -> usize {
        self.0
            .len()
    }

    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<T> HostSlice<T> {
    // Currently this function just transmutes types. However it is not guaranteed that this function
    // will always be cheap as it might at some point e.g. pin the memory which takes some time.
    pub fn from_slice(slice: &[T]) -> &Self {
        unsafe { &*(slice as *const [T] as *const Self) }
    }

    // Currently this function just transmutes types. However it is not guaranteed that this function
    // will always be cheap as it might at some point e.g. pin the memory which takes some time.
    pub fn from_mut_slice(slice: &mut [T]) -> &mut Self {
        unsafe { &mut *(slice as *mut [T] as *mut Self) }
    }

    pub fn as_slice(&self) -> &[T] {
        &self.0
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.0
    }

    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.0
            .iter()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut T> {
        self.0
            .iter_mut()
    }

    pub fn is_pinnable(&self) -> bool {
        let pinnable = get_device_attribute(cudaDeviceAttr::cudaDevAttrHostRegisterSupported, 0).unwrap();
        let lockable =
            get_device_attribute(cudaDeviceAttr::cudaDevAttrPageableMemoryAccessUsesHostPageTables, 0).unwrap();

        pinnable == 1 && lockable == 0
    }

    pub fn pin(&self, flags: CudaHostRegisterFlags) -> CudaResult<()> {
        unsafe {
            let ptr = self.as_ptr() as *mut c_void;
            let flags_to_set = flags.bits();
            cudaHostRegister(ptr, self.len(), flags_to_set as c_uint).wrap()
        }
    }

    pub fn unpin(&self) -> CudaResult<()> {
        unsafe {
            let mut flags = 0;
            let ptr = self.as_ptr() as *mut c_void;
            cudaHostGetFlags(&mut flags, ptr).wrap()?;
            cudaHostUnregister(ptr).wrap()
        }
    }

    pub fn allocate_pinned(count: usize, flags: CudaHostAllocFlags) -> CudaResult<&'static mut Self> {
        let size = count
            .checked_mul(size_of::<T>())
            .unwrap_or(0);
        if size == 0 {
            return Err(CudaError::cudaErrorMemoryAllocation); //TODO: only CUDA backend should return CudaError
        }

        let mut pinned_host_ptr = MaybeUninit::<*mut c_void>::uninit();

        unsafe {
            cudaHostAlloc(pinned_host_ptr.as_mut_ptr(), size, flags.bits).wrap()?;
            let pinned_host_slice = from_raw_parts_mut(pinned_host_ptr.assume_init() as *mut T, count);
            Ok(Self::from_mut_slice(pinned_host_slice))
        }
    }

    pub fn free_pinned(&self) -> CudaResult<()> {
        unsafe {
            let mut flags: u32 = 0;
            let ptr = self.as_ptr() as *mut c_void;
            cudaHostGetFlags(&mut flags, ptr).wrap()?;
            cudaFreeHost(ptr).wrap()
        }
    }

    pub fn get_memory_flags(&self) -> CudaResult<u32> {
        unsafe {
            let mut flags: u32 = 1234;
            let ptr = self.as_ptr() as *mut c_void;
            cudaHostGetFlags(&mut flags, ptr).wrap()?;
            Ok(flags)
        }
    }
}

impl<T> DeviceSlice<T> {
    pub unsafe fn from_slice(slice: &[T]) -> &Self {
        &*(slice as *const [T] as *const Self)
    }

    pub unsafe fn from_mut_slice(slice: &mut [T]) -> &mut Self {
        &mut *(slice as *mut [T] as *mut Self)
    }

    pub fn copy_from_host(&mut self, val: &HostSlice<T>) -> CudaResult<()> {
        assert!(
            self.len() == val.len(),
            "In copy from host, destination and source slices have different lengths"
        );
        check_device(
            self.device_id()
                .unwrap(),
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

    pub fn copy_to_host(&self, val: &mut HostSlice<T>) -> CudaResult<()> {
        assert!(
            self.len() == val.len(),
            "In copy to host, destination and source slices have different lengths"
        );
        check_device(
            self.device_id()
                .unwrap(),
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

    pub fn copy_from_host_async(&mut self, val: &HostSlice<T>, stream: &CudaStream) -> CudaResult<()> {
        assert!(
            self.len() == val.len(),
            "In copy from host async, destination and source slices have different lengths"
        );
        check_device(
            self.device_id()
                .unwrap(),
        );
        let size = size_of::<T>() * self.len();
        if size != 0 {
            unsafe {
                cudaMemcpyAsync(
                    self.as_mut_ptr() as *mut c_void,
                    val.as_ptr() as *const c_void,
                    size,
                    cudaMemcpyKind::cudaMemcpyHostToDevice,
                    stream.handle,
                )
                .wrap()?
            }
        }
        Ok(())
    }

    pub fn copy_to_host_async(&self, val: &mut HostSlice<T>, stream: &CudaStream) -> CudaResult<()> {
        assert!(
            self.len() == val.len(),
            "In copy to host async, destination and source slices have different lengths"
        );
        check_device(
            self.device_id()
                .unwrap(),
        );
        let size = size_of::<T>() * self.len();
        if size != 0 {
            unsafe {
                cudaMemcpyAsync(
                    val.as_mut_ptr() as *mut c_void,
                    self.as_ptr() as *const c_void,
                    size,
                    cudaMemcpyKind::cudaMemcpyDeviceToHost,
                    stream.handle,
                )
                .wrap()?
            }
        }
        Ok(())
    }
}

impl<T> DeviceVec<T> {
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
            let res = Self(ManuallyDrop::new(Box::from_raw(from_raw_parts_mut(
                device_ptr.assume_init() as *mut T,
                count,
            ))));
            Ok(res)
        }
    }

    pub fn cuda_malloc_async(count: usize, stream: &CudaStream) -> CudaResult<Self> {
        let size = count
            .checked_mul(size_of::<T>())
            .unwrap_or(0);
        if size == 0 {
            return Err(CudaError::cudaErrorMemoryAllocation); //TODO: only CUDA backend should return CudaError
        }

        let mut device_ptr = MaybeUninit::<*mut c_void>::uninit();
        unsafe {
            cudaMallocAsync(device_ptr.as_mut_ptr(), size, stream.handle).wrap()?;
            Ok(Self(ManuallyDrop::new(Box::from_raw(from_raw_parts_mut(
                device_ptr.assume_init() as *mut T,
                count,
            )))))
        }
    }

    pub fn cuda_malloc_for_device(count: usize, device_id: usize) -> CudaResult<Self> {
        check_device(device_id);
        Self::cuda_malloc(count)
    }

    pub fn cuda_malloc_async_for_device(count: usize, stream: &CudaStream, device_id: usize) -> CudaResult<Self> {
        check_device(device_id);
        Self::cuda_malloc_async(count, stream)
    }
}

macro_rules! impl_host_index {
    ($($t:ty)*) => {
        $(
            impl<T> Index<$t> for HostSlice<T>
            {
                type Output = Self;

                fn index(&self, index: $t) -> &Self::Output {
                    Self::from_slice(
                        self.0
                            .index(index),
                    )
                }
            }

            impl<T> IndexMut<$t> for HostSlice<T>
            {
                fn index_mut(&mut self, index: $t) -> &mut Self::Output {
                    Self::from_mut_slice(
                        self.0
                            .index_mut(index),
                    )
                }
            }
        )*
    }
}
impl_host_index! {
    Range<usize>
    RangeFull
    RangeFrom<usize>
    RangeInclusive<usize>
    RangeTo<usize>
    RangeToInclusive<usize>
}

impl<T> Index<usize> for HostSlice<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        self.0
            .index(index)
    }
}

impl<T> IndexMut<usize> for HostSlice<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.0
            .index_mut(index)
    }
}

impl<Idx, T> Index<Idx> for DeviceVec<T>
where
    Idx: SliceIndex<[T], Output = [T]>,
{
    type Output = DeviceSlice<T>;

    fn index(&self, index: Idx) -> &Self::Output {
        unsafe {
            Self::Output::from_slice(
                self.0
                    .index(index),
            )
        }
    }
}

impl<Idx, T> IndexMut<Idx> for DeviceVec<T>
where
    Idx: SliceIndex<[T], Output = [T]>,
{
    fn index_mut(&mut self, index: Idx) -> &mut Self::Output {
        unsafe {
            Self::Output::from_mut_slice(
                self.0
                    .index_mut(index),
            )
        }
    }
}

impl<Idx, T> Index<Idx> for DeviceSlice<T>
where
    Idx: SliceIndex<[T], Output = [T]>,
{
    type Output = Self;

    fn index(&self, index: Idx) -> &Self::Output {
        unsafe {
            Self::from_slice(
                self.0
                    .index(index),
            )
        }
    }
}

impl<Idx, T> IndexMut<Idx> for DeviceSlice<T>
where
    Idx: SliceIndex<[T], Output = [T]>,
{
    fn index_mut(&mut self, index: Idx) -> &mut Self::Output {
        unsafe {
            Self::from_mut_slice(
                self.0
                    .index_mut(index),
            )
        }
    }
}

impl<T> Deref for DeviceVec<T> {
    type Target = DeviceSlice<T>;

    fn deref(&self) -> &Self::Target {
        &self[..]
    }
}

impl<T> DerefMut for DeviceVec<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self[..]
    }
}

impl<T> Drop for DeviceVec<T> {
    fn drop(&mut self) {
        if self
            .0
            .is_empty()
        {
            return;
        }

        unsafe {
            let ptr = self
                .0
                .as_mut_ptr() as *mut c_void;
            cudaFree(ptr)
                .wrap()
                .unwrap();
        }
    }
}

#[allow(non_camel_case_types)]
pub type CudaMemPool = cudaMemPool_t;

pub(crate) mod tests {
    use super::{CudaHostRegisterFlags, HostSlice};
    use crate::bindings::cudaError;
    use crate::memory::{CudaHostAllocFlags, HostOrDeviceSlice};

    #[test]
    fn test_pin_memory() {
        let data = vec![1, 2, 3, 4, 5, 7, 8, 9];
        let data_host_slice = HostSlice::from_slice(&data);

        data_host_slice
            .pin(CudaHostRegisterFlags::DEFAULT)
            .expect("Registering host mem failed");
        let err = data_host_slice
            .pin(CudaHostRegisterFlags::DEFAULT)
            .expect_err("Registering already registered memory succeeded");
        assert_eq!(err, cudaError::cudaErrorHostMemoryAlreadyRegistered);

        data_host_slice
            .unpin()
            .expect("Unregistering pinned memory failed");
        let err = data_host_slice
            .unpin()
            .expect_err("Unregistering non-registered pinned memory succeeded");
        assert_eq!(err, cudaError::cudaErrorInvalidValue);
    }

    #[test]
    fn test_allocated_pinned_memory() {
        let data = vec![1, 2, 3, 4, 5, 7, 8, 9];
        let data_host_slice = HostSlice::from_slice(&data);
        let newly_allocated_pinned_host_slice: &HostSlice<i32> =
            HostSlice::allocate_pinned(data_host_slice.len(), CudaHostAllocFlags::DEFAULT)
                .expect("Allocating new pinned memory failed");
        newly_allocated_pinned_host_slice
            .free_pinned()
            .expect("Freeing pinned memory failed");
        let err = newly_allocated_pinned_host_slice
            .free_pinned()
            .expect_err("Freeing non-pinned memory succeeded");
        assert_eq!(err, cudaError::cudaErrorInvalidValue);
    }
}
