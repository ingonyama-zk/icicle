use crate::errors::eIcicleError;
use crate::runtime;
use crate::stream::IcicleStream;
use std::mem::{size_of, ManuallyDrop};
use std::ops::{
    Deref, DerefMut, Index, IndexMut, Range, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive,
};
use std::os::raw::c_void;
use std::slice::{from_raw_parts_mut, SliceIndex};

#[derive(Debug)]
pub struct HostSlice<T>([T]);
pub struct DeviceVec<T>(ManuallyDrop<Box<[T]>>);
pub struct DeviceSlice<T>([T]);

pub trait HostOrDeviceSlice<T> {
    fn is_on_device(&self) -> bool;
    fn is_on_active_device(&self) -> bool;
    unsafe fn as_ptr(&self) -> *const T;
    unsafe fn as_mut_ptr(&mut self) -> *mut T;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool;
}

impl<T> HostOrDeviceSlice<T> for HostSlice<T> {
    fn is_on_device(&self) -> bool {
        false
    }

    fn is_on_active_device(&self) -> bool {
        false
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

    fn is_on_active_device(&self) -> bool {
        runtime::is_active_device_memory(
            self.0
                .as_ptr() as *const c_void,
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
}

impl<T> DeviceSlice<T> {
    pub unsafe fn from_slice(slice: &[T]) -> &Self {
        &*(slice as *const [T] as *const Self)
    }

    pub unsafe fn from_mut_slice(slice: &mut [T]) -> &mut Self {
        &mut *(slice as *mut [T] as *mut Self)
    }

    pub fn copy_from_host(&mut self, val: &HostSlice<T>) -> Result<(), eIcicleError> {
        assert!(
            self.len() == val.len(),
            "In copy from host, destination and source slices have different lengths"
        );

        if self.is_empty() {
            return Ok(());
        }
        if !self.is_on_active_device() {
            panic!("not allocated on an inactive device");
        }

        let size = size_of::<T>() * self.len();
        unsafe {
            runtime::icicle_copy_to_device(self.as_mut_ptr() as *mut c_void, val.as_ptr() as *const c_void, size).wrap()
        }
    }

    pub fn copy_to_host(&self, val: &mut HostSlice<T>) -> Result<(), eIcicleError> {
        assert!(
            self.len() == val.len(),
            "In copy to host, destination and source slices have different lengths"
        );

        if self.is_empty() {
            return Ok(());
        }
        if !self.is_on_active_device() {
            panic!("not allocated on an inactive device");
        }

        let size = size_of::<T>() * self.len();
        unsafe {
            runtime::icicle_copy_to_host(val.as_mut_ptr() as *mut c_void, self.as_ptr() as *const c_void, size).wrap()
        }
    }

    pub fn copy_from_host_async(&mut self, val: &HostSlice<T>, stream: &IcicleStream) -> Result<(), eIcicleError> {
        assert!(
            self.len() == val.len(),
            "In copy from host, destination and source slices have different lengths"
        );
        if self.is_empty() {
            return Ok(());
        }
        if !self.is_on_active_device() {
            panic!("not allocated on an inactive device");
        }

        let size = size_of::<T>() * self.len();
        unsafe {
            runtime::icicle_copy_to_device_async(
                self.as_mut_ptr() as *mut c_void,
                val.as_ptr() as *const c_void,
                size,
                stream.handle,
            )
            .wrap()
        }
    }

    pub fn copy_to_host_async(&self, val: &mut HostSlice<T>, stream: &IcicleStream) -> Result<(), eIcicleError> {
        assert!(
            self.len() == val.len(),
            "In copy to host, destination and source slices have different lengths"
        );
        if self.is_empty() {
            return Ok(());
        }
        if !self.is_on_active_device() {
            panic!("not allocated on an inactive device");
        }

        let size = size_of::<T>() * self.len();
        unsafe {
            runtime::icicle_copy_to_host_async(
                val.as_mut_ptr() as *mut c_void,
                self.as_ptr() as *const c_void,
                size,
                stream.handle,
            )
            .wrap()
        }
    }
}

impl<T> DeviceVec<T> {
    pub fn device_malloc(count: usize) -> Result<Self, eIcicleError> {
        let size = count
            .checked_mul(size_of::<T>())
            .unwrap_or(0);
        if size == 0 {
            return Err(eIcicleError::AllocationFailed);
        }

        let mut device_ptr: *mut c_void = std::ptr::null_mut();
        let error = unsafe { runtime::icicle_malloc(&mut device_ptr, size) };
        if error != eIcicleError::Success {
            return Err(error);
        }

        unsafe {
            Ok(Self(ManuallyDrop::new(Box::from_raw(from_raw_parts_mut(
                device_ptr as *mut T,
                count,
            )))))
        }
    }

    pub fn device_malloc_async(count: usize, stream: &IcicleStream) -> Result<Self, eIcicleError> {
        let size = count
            .checked_mul(size_of::<T>())
            .unwrap_or(0);
        if size == 0 {
            return Err(eIcicleError::AllocationFailed);
        }

        let mut device_ptr: *mut c_void = std::ptr::null_mut();
        unsafe { runtime::icicle_malloc_async(&mut device_ptr, size, stream.handle).wrap()? };

        unsafe {
            Ok(Self(ManuallyDrop::new(Box::from_raw(from_raw_parts_mut(
                device_ptr as *mut T,
                count,
            )))))
        }
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
            let err = runtime::icicle_free(ptr);
            if err != eIcicleError::Success {
                panic!("releasing memory failed due to invalid active device");
            }
        }
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
