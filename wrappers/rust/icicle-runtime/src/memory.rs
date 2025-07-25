use crate::errors::{eIcicleError, IcicleError};
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

    fn copy(&mut self, src: &(impl HostOrDeviceSlice<T> + ?Sized)) -> Result<(), IcicleError> {
        if self.len() < src.len() {
            return Err(IcicleError::new(
                eIcicleError::CopyFailed,
                "In copy, destination has shorter length than source",
            ));
        }

        let size = size_of::<T>() * src.len();
        unsafe { runtime::icicle_copy(self.as_mut_ptr() as *mut c_void, src.as_ptr() as *const c_void, size).wrap() }
    }

    fn copy_async(
        &mut self,
        src: &(impl HostOrDeviceSlice<T> + ?Sized),
        stream: &IcicleStream,
    ) -> Result<(), IcicleError> {
        if self.len() < src.len() {
            return Err(IcicleError::new(
                eIcicleError::CopyFailed,
                "In copy, destination has shorter length than source",
            ));
        }

        let size = size_of::<T>() * src.len();
        unsafe {
            runtime::icicle_copy_async(
                self.as_mut_ptr() as *mut c_void,
                src.as_ptr() as *const c_void,
                size,
                stream.handle,
            )
            .wrap()
        }
    }

    fn memset(&mut self, value: u8, size: usize) -> Result<(), IcicleError>;
    fn memset_async(&mut self, value: u8, size: usize, stream: &IcicleStream) -> Result<(), IcicleError>;
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

    fn memset(&mut self, value: u8, size: usize) -> Result<(), IcicleError> {
        if size == 0 || self.is_empty() {
            return Ok(());
        }
        if size > self.len() {
            return Err(IcicleError::new(eIcicleError::CopyFailed, "size exceeds slice length"));
        }
        unsafe { runtime::memset(self.as_mut_ptr() as *mut c_void, value as i32, size) }
    }

    fn memset_async(&mut self, value: u8, size: usize, stream: &IcicleStream) -> Result<(), IcicleError> {
        if size == 0 || self.is_empty() {
            return Ok(());
        }
        if size > self.len() {
            return Err(IcicleError::new(eIcicleError::CopyFailed, "size exceeds slice length"));
        }
        unsafe { runtime::memset_async(self.as_mut_ptr() as *mut c_void, value as i32, size, stream.handle) }
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

    fn memset(&mut self, value: u8, size: usize) -> Result<(), IcicleError> {
        if size == 0 || self.is_empty() {
            return Ok(());
        }
        if size > self.len() {
            return Err(IcicleError::new(eIcicleError::CopyFailed, "size exceeds slice length"));
        }
        if !self.is_on_active_device() {
            return Err(IcicleError::new(
                eIcicleError::CopyFailed,
                "not allocated on an active device",
            ));
        }

        let byte_size = size_of::<T>() * size;
        unsafe { runtime::memset(self.as_mut_ptr() as *mut c_void, value as i32, byte_size) }
    }

    fn memset_async(&mut self, value: u8, size: usize, stream: &IcicleStream) -> Result<(), IcicleError> {
        if size == 0 || self.is_empty() {
            return Ok(());
        }
        if size > self.len() {
            return Err(IcicleError::new(eIcicleError::CopyFailed, "size exceeds slice length"));
        }
        if !self.is_on_active_device() {
            return Err(IcicleError::new(
                eIcicleError::CopyFailed,
                "not allocated on an active device",
            ));
        }

        let byte_size = size_of::<T>() * size;
        unsafe { runtime::memset_async(self.as_mut_ptr() as *mut c_void, value as i32, byte_size, stream.handle) }
    }
}

// Note: Implementing the trait for DeviceVec such that functions expecting DeviceSlice reference can take a HostOrDeviceSlice reference without being a breaking change.
// Otherwise the syntax would be &device_vec[..] rather than &device_vec which would be a breaking change.
impl<T> HostOrDeviceSlice<T> for DeviceVec<T> {
    fn is_on_device(&self) -> bool {
        // Forward to the dereferenced DeviceSlice
        (**self).is_on_device()
    }

    fn is_on_active_device(&self) -> bool {
        // Forward to the dereferenced DeviceSlice
        (**self).is_on_active_device()
    }

    unsafe fn as_ptr(&self) -> *const T {
        // Forward to the dereferenced DeviceSlice
        (**self).as_ptr()
    }

    unsafe fn as_mut_ptr(&mut self) -> *mut T {
        // Forward to the dereferenced DeviceSlice
        (**self).as_mut_ptr()
    }

    fn len(&self) -> usize {
        // Forward to the dereferenced DeviceSlice
        (**self).len()
    }

    fn is_empty(&self) -> bool {
        // Forward to the dereferenced DeviceSlice
        (**self).is_empty()
    }

    fn memset(&mut self, value: u8, size: usize) -> Result<(), IcicleError> {
        self.as_mut_slice()
            .memset(value, size)
    }

    fn memset_async(&mut self, value: u8, size: usize, stream: &IcicleStream) -> Result<(), IcicleError> {
        self.as_mut_slice()
            .memset_async(value, size, stream)
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

    pub unsafe fn from_raw_parts<'a>(ptr: *const T, len: usize) -> &'a HostSlice<T> {
        &*(core::ptr::slice_from_raw_parts(ptr, len) as *const HostSlice<T>)
    }

    pub unsafe fn from_raw_parts_mut<'a>(ptr: *mut T, len: usize) -> &'a mut HostSlice<T> {
        &mut *(core::ptr::slice_from_raw_parts_mut(ptr, len) as *mut HostSlice<T>)
    }
}

impl<T> DeviceSlice<T> {
    pub unsafe fn from_slice(slice: &[T]) -> &Self {
        &*(slice as *const [T] as *const Self)
    }

    pub unsafe fn from_mut_slice(slice: &mut [T]) -> &mut Self {
        &mut *(slice as *mut [T] as *mut Self)
    }

    pub fn copy_from_host(&mut self, val: &HostSlice<T>) -> Result<(), IcicleError> {
        if self.len() != val.len() {
            return Err(IcicleError::new(
                eIcicleError::CopyFailed,
                "In copy from host, destination and source slices have different lengths",
            ));
        }

        if self.is_empty() {
            return Ok(());
        }
        if !self.is_on_active_device() {
            return Err(IcicleError::new(
                eIcicleError::CopyFailed,
                "not allocated on an active device",
            ));
        }

        let size = size_of::<T>() * self.len();
        unsafe {
            runtime::icicle_copy_to_device(self.as_mut_ptr() as *mut c_void, val.as_ptr() as *const c_void, size).wrap()
        }
    }

    pub fn copy_to_host(&self, val: &mut HostSlice<T>) -> Result<(), IcicleError> {
        if self.len() != val.len() {
            return Err(IcicleError::new(
                eIcicleError::CopyFailed,
                "In copy to host, destination and source slices have different lengths",
            ));
        }

        if self.is_empty() {
            return Ok(());
        }
        if !self.is_on_active_device() {
            return Err(IcicleError::new(
                eIcicleError::CopyFailed,
                "not allocated on an active device",
            ));
        }

        let size = size_of::<T>() * self.len();
        unsafe {
            runtime::icicle_copy_to_host(val.as_mut_ptr() as *mut c_void, self.as_ptr() as *const c_void, size).wrap()
        }
    }

    pub fn copy_from_host_async(&mut self, val: &HostSlice<T>, stream: &IcicleStream) -> Result<(), IcicleError> {
        if self.len() != val.len() {
            return Err(IcicleError::new(
                eIcicleError::CopyFailed,
                "In copy from host, destination and source slices have different lengths",
            ));
        }
        if self.is_empty() {
            return Ok(());
        }
        if !self.is_on_active_device() {
            return Err(IcicleError::new(
                eIcicleError::CopyFailed,
                "not allocated on an active device",
            ));
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

    pub fn copy_to_host_async(&self, val: &mut HostSlice<T>, stream: &IcicleStream) -> Result<(), IcicleError> {
        if self.len() != val.len() {
            return Err(IcicleError::new(
                eIcicleError::CopyFailed,
                "In copy to host, destination and source slices have different lengths",
            ));
        }
        if self.is_empty() {
            return Ok(());
        }
        if !self.is_on_active_device() {
            return Err(IcicleError::new(
                eIcicleError::CopyFailed,
                "not allocated on an active device",
            ));
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

    /// Copy the contents of the device slice back to the host and return them as `Vec<T>`.
    /// Convenience method to avoid the boilerplate of allocating a host buffer and calling
    /// `copy_to_host` manually. Requires `T: Copy + Default` so that we can cheaply create a
    /// zero-initialised host vector.
    pub fn to_host_vec(&self) -> Vec<T>
    where
        T: Copy + Default,
    {
        let mut host_vec = vec![T::default(); self.len()];
        let host_slice = host_vec.into_slice_mut();
        self.copy_to_host(host_slice)
            .unwrap();
        host_vec
    }

    /// # Safety
    /// `ptr` must point to `len` contiguous elements in device memory.
    /// The caller must ensure the memory is valid for the lifetime `'a` and not aliased.
    pub unsafe fn from_raw_parts<'a>(ptr: *const T, len: usize) -> &'a DeviceSlice<T> {
        &*(core::ptr::slice_from_raw_parts(ptr, len) as *const DeviceSlice<T>)
    }

    /// # Safety
    /// `ptr` must point to `len` contiguous elements in device memory and be uniquely owned.
    /// The caller must ensure the memory is valid for the lifetime `'a` and not aliased.
    pub unsafe fn from_raw_parts_mut<'a>(ptr: *mut T, len: usize) -> &'a mut DeviceSlice<T> {
        &mut *(core::ptr::slice_from_raw_parts_mut(ptr, len) as *mut DeviceSlice<T>)
    }
}

impl<T> DeviceVec<T> {
    pub fn device_malloc(count: usize) -> Result<Self, IcicleError> {
        let size = count
            .checked_mul(size_of::<T>())
            .unwrap_or(0);
        if size == 0 {
            return Err(IcicleError::new(eIcicleError::AllocationFailed, "invalid size"));
        }

        let mut device_ptr: *mut c_void = std::ptr::null_mut();
        unsafe { runtime::icicle_malloc(&mut device_ptr, size).wrap_err_msg("device malloc failed")? };

        unsafe {
            Ok(Self(ManuallyDrop::new(Box::from_raw(from_raw_parts_mut(
                device_ptr as *mut T,
                count,
            )))))
        }
    }

    /// Fallible allocation that panics on failure, yielding slightly cleaner call sites:
    /// `let mut buf = DeviceVec::<F>::malloc(size);`
    pub fn malloc(count: usize) -> Self {
        Self::device_malloc(count).expect("device allocation failed")
    }

    /// Allocate and zero-initialise the memory in a single call. Useful when the caller
    /// needs the buffer to be in a well-defined state before copying or computing into it.
    pub fn zeros(count: usize) -> Self
    where
        T: Copy,
    {
        let mut v = Self::malloc(count);
        // Ignore potential failure because we just allocated the buffer on the active device.
        let _ = v.memset(0, count);
        v
    }

    pub fn device_malloc_async(count: usize, stream: &IcicleStream) -> Result<Self, IcicleError> {
        let size = count
            .checked_mul(size_of::<T>())
            .unwrap_or(0);
        if size == 0 {
            return Err(IcicleError::new(eIcicleError::AllocationFailed, "invalid size"));
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

    pub fn as_mut_slice(&mut self) -> &mut DeviceSlice<T> {
        &mut self[..]
    }

    /// Copy the contents of the device vector back to the host and return them as `Vec<T>`.
    /// Convenience method to avoid the boilerplate of allocating a host buffer and calling
    /// `copy_to_host` manually. Requires `T: Copy + Default` so that we can cheaply create a
    /// zero-initialised host vector.
    pub fn to_host_vec(&self) -> Vec<T>
    where
        T: Copy + Default,
    {
        let mut host_vec = vec![T::default(); self.len()];
        let host_slice = host_vec.into_slice_mut();
        self.copy_to_host(host_slice)
            .unwrap();
        host_vec
    }

    /// Convenience constructor: allocate a new `DeviceVec` on the active device, copy the
    /// contents of a host slice into it, and return the populated vector.
    ///
    /// Example:
    /// ```
    /// use icicle_runtime::memory::{DeviceVec, HostSlice};
    /// let host_data = vec![1u32, 2, 3, 4];
    /// let device_buf = DeviceVec::<u32>::from_host_slice(&host_data);
    /// ```
    pub fn from_host_slice(src: &[T]) -> Self
    where
        T: Copy,
    {
        let mut device_vec = Self::malloc(src.len());
        device_vec
            .copy_from_host(src.into_slice())
            .unwrap();
        device_vec
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

// =============================================================================================
// Adapter traits to turn common buffer types into `HostOrDeviceSlice`s with minimal boilerplate.
// =============================================================================================

pub trait IntoIcicleSlice<'a, T: 'a> {
    type Out: HostOrDeviceSlice<T> + ?Sized + 'a;
    fn into_slice(&'a self) -> &'a Self::Out;
}

pub trait IntoIcicleSliceMut<'a, T: 'a> {
    type Out: HostOrDeviceSlice<T> + ?Sized + 'a;
    fn into_slice_mut(&'a mut self) -> &'a mut Self::Out;
}

// Host buffer implementations
impl<'a, T> IntoIcicleSlice<'a, T> for &'a [T] {
    type Out = HostSlice<T>;
    fn into_slice(&'a self) -> &'a HostSlice<T> {
        HostSlice::from_slice(self)
    }
}

impl<'a, T> IntoIcicleSlice<'a, T> for &'a Vec<T> {
    type Out = HostSlice<T>;
    fn into_slice(&'a self) -> &'a HostSlice<T> {
        HostSlice::from_slice(self.as_slice())
    }
}

impl<'a, T: 'a> IntoIcicleSlice<'a, T> for Vec<T> {
    type Out = HostSlice<T>;
    fn into_slice(&'a self) -> &'a HostSlice<T> {
        HostSlice::from_slice(self.as_slice())
    }
}

impl<'a, T: 'a, const N: usize> IntoIcicleSlice<'a, T> for [T; N] {
    type Out = HostSlice<T>;
    fn into_slice(&'a self) -> &'a HostSlice<T> {
        HostSlice::from_slice(self)
    }
}

impl<'a, T> IntoIcicleSliceMut<'a, T> for &'a mut [T] {
    type Out = HostSlice<T>;
    fn into_slice_mut(&'a mut self) -> &'a mut HostSlice<T> {
        HostSlice::from_mut_slice(self)
    }
}

impl<'a, T> IntoIcicleSliceMut<'a, T> for &'a mut Vec<T> {
    type Out = HostSlice<T>;
    fn into_slice_mut(&'a mut self) -> &'a mut HostSlice<T> {
        HostSlice::from_mut_slice(self.as_mut_slice())
    }
}

impl<'a, T: 'a> IntoIcicleSliceMut<'a, T> for Vec<T> {
    type Out = HostSlice<T>;
    fn into_slice_mut(&'a mut self) -> &'a mut HostSlice<T> {
        HostSlice::from_mut_slice(self.as_mut_slice())
    }
}

impl<'a, T: 'a, const N: usize> IntoIcicleSliceMut<'a, T> for [T; N] {
    type Out = HostSlice<T>;
    fn into_slice_mut(&'a mut self) -> &'a mut HostSlice<T> {
        HostSlice::from_mut_slice(self)
    }
}

// Device buffer implementations
impl<'a, T> IntoIcicleSlice<'a, T> for &'a DeviceVec<T> {
    type Out = DeviceSlice<T>;
    fn into_slice(&'a self) -> &'a DeviceSlice<T> {
        &**self
    }
}

impl<'a, T> IntoIcicleSliceMut<'a, T> for &'a mut DeviceVec<T> {
    type Out = DeviceSlice<T>;
    fn into_slice_mut(&'a mut self) -> &'a mut DeviceSlice<T> {
        &mut **self
    }
}

impl<'a, T: 'a> IntoIcicleSlice<'a, T> for DeviceVec<T> {
    type Out = DeviceSlice<T>;
    fn into_slice(&'a self) -> &'a DeviceSlice<T> {
        &**self
    }
}

impl<'a, T: 'a> IntoIcicleSliceMut<'a, T> for DeviceVec<T> {
    type Out = DeviceSlice<T>;
    fn into_slice_mut(&'a mut self) -> &'a mut DeviceSlice<T> {
        &mut **self
    }
}
// Utility to reinterpret HostOrDeviceSlice via a UnifiedSlice or UnifiedSliceMut that also implement HostOrDeviceSlice.
pub mod reinterpret {
    use super::*;

    pub enum UnifiedSlice<'a, T: 'a> {
        Host(&'a HostSlice<T>),
        Device(&'a DeviceSlice<T>),
    }

    pub enum UnifiedSliceMut<'a, T: 'a> {
        Host(&'a mut HostSlice<T>),
        Device(&'a mut DeviceSlice<T>),
    }

    /// SAFETY: Caller must ensure layout compatibility between `From` and `To`.
    fn compute_output_len<From, To>(len: usize) -> Result<usize, IcicleError> {
        let from_size = size_of::<From>();
        let to_size = size_of::<To>();

        if from_size == 0 || to_size == 0 {
            return Err(IcicleError::new(eIcicleError::InvalidArgument, "invalid size"));
        }

        let total_bytes = from_size
            .checked_mul(len)
            .ok_or(IcicleError::new(eIcicleError::InvalidArgument, "size overflow"))?;

        if total_bytes % to_size != 0 {
            return Err(IcicleError::new(eIcicleError::InvalidArgument, "size not aligned"));
        }

        Ok(total_bytes / to_size)
    }

    /// SAFETY: Caller must ensure layout of P as [P::Base; DEGREE]
    pub unsafe fn reinterpret_slice<From, To>(
        input: &(impl HostOrDeviceSlice<From> + ?Sized),
    ) -> Result<UnifiedSlice<'_, To>, IcicleError>
    where
        From: Sized,
        To: Sized,
    {
        let len = input.len();
        let flat_len = compute_output_len::<From, To>(len)?;
        let ptr = input.as_ptr() as *const To;

        if input.is_on_device() {
            Ok(UnifiedSlice::Device(DeviceSlice::from_raw_parts(ptr, flat_len)))
        } else {
            Ok(UnifiedSlice::Host(HostSlice::from_raw_parts(ptr, flat_len)))
        }
    }

    pub unsafe fn reinterpret_slice_mut<From, To>(
        input: &mut (impl HostOrDeviceSlice<From> + ?Sized),
    ) -> Result<UnifiedSliceMut<'_, To>, IcicleError>
    where
        From: Sized,
        To: Sized,
    {
        let len = input.len();
        let flat_len = compute_output_len::<From, To>(len)?;
        let ptr = input.as_mut_ptr() as *mut To;

        if input.is_on_device() {
            Ok(UnifiedSliceMut::Device(DeviceSlice::from_raw_parts_mut(ptr, flat_len)))
        } else {
            Ok(UnifiedSliceMut::Host(HostSlice::from_raw_parts_mut(ptr, flat_len)))
        }
    }

    impl<T> HostOrDeviceSlice<T> for UnifiedSlice<'_, T> {
        fn is_on_device(&self) -> bool {
            match self {
                UnifiedSlice::Device(d) => d.is_on_device(),
                UnifiedSlice::Host(h) => h.is_on_device(),
            }
        }

        fn is_on_active_device(&self) -> bool {
            match self {
                UnifiedSlice::Device(d) => d.is_on_active_device(),
                UnifiedSlice::Host(h) => h.is_on_active_device(),
            }
        }

        unsafe fn as_ptr(&self) -> *const T {
            match self {
                UnifiedSlice::Device(d) => d.as_ptr(),
                UnifiedSlice::Host(h) => h.as_ptr(),
            }
        }

        unsafe fn as_mut_ptr(&mut self) -> *mut T {
            panic!("Cannot get mutable pointer from immutable UnifiedSlice")
        }

        fn len(&self) -> usize {
            match self {
                UnifiedSlice::Device(d) => d.len(),
                UnifiedSlice::Host(h) => h.len(),
            }
        }

        fn is_empty(&self) -> bool {
            match self {
                UnifiedSlice::Device(d) => d.is_empty(),
                UnifiedSlice::Host(h) => h.is_empty(),
            }
        }

        fn memset(&mut self, _: u8, _: usize) -> Result<(), IcicleError> {
            Err(IcicleError::new(
                eIcicleError::CopyFailed,
                "Cannot memset immutable UnifiedSlice",
            ))
        }

        fn memset_async(&mut self, _: u8, _: usize, _: &IcicleStream) -> Result<(), IcicleError> {
            Err(IcicleError::new(
                eIcicleError::CopyFailed,
                "Cannot memset_async immutable UnifiedSlice",
            ))
        }
    }

    impl<T> HostOrDeviceSlice<T> for UnifiedSliceMut<'_, T> {
        fn is_on_device(&self) -> bool {
            match self {
                UnifiedSliceMut::Device(d) => d.is_on_device(),
                UnifiedSliceMut::Host(h) => h.is_on_device(),
            }
        }

        fn is_on_active_device(&self) -> bool {
            match self {
                UnifiedSliceMut::Device(d) => d.is_on_active_device(),
                UnifiedSliceMut::Host(h) => h.is_on_active_device(),
            }
        }

        unsafe fn as_ptr(&self) -> *const T {
            match self {
                UnifiedSliceMut::Device(d) => d.as_ptr(),
                UnifiedSliceMut::Host(h) => h.as_ptr(),
            }
        }

        unsafe fn as_mut_ptr(&mut self) -> *mut T {
            match self {
                UnifiedSliceMut::Device(d) => d.as_mut_ptr(),
                UnifiedSliceMut::Host(h) => h.as_mut_ptr(),
            }
        }

        fn len(&self) -> usize {
            match self {
                UnifiedSliceMut::Device(d) => d.len(),
                UnifiedSliceMut::Host(h) => h.len(),
            }
        }

        fn is_empty(&self) -> bool {
            match self {
                UnifiedSliceMut::Device(d) => d.is_empty(),
                UnifiedSliceMut::Host(h) => h.is_empty(),
            }
        }

        fn memset(&mut self, value: u8, size: usize) -> Result<(), IcicleError> {
            match self {
                UnifiedSliceMut::Device(d) => d.memset(value, size),
                UnifiedSliceMut::Host(h) => h.memset(value, size),
            }
        }

        fn memset_async(&mut self, value: u8, size: usize, stream: &IcicleStream) -> Result<(), IcicleError> {
            match self {
                UnifiedSliceMut::Device(d) => d.memset_async(value, size, stream),
                UnifiedSliceMut::Host(h) => h.memset_async(value, size, stream),
            }
        }
    }
}
