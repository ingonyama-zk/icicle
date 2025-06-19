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

    fn copy(&mut self, src: &(impl HostOrDeviceSlice<T> + ?Sized)) -> Result<(), eIcicleError> {
        assert!(
            self.len() >= src.len(),
            "In copy, destination has shorter length than source"
        );

        let size = size_of::<T>() * src.len();
        unsafe { runtime::icicle_copy(self.as_mut_ptr() as *mut c_void, src.as_ptr() as *const c_void, size).wrap() }
    }

    fn copy_async(
        &mut self,
        src: &(impl HostOrDeviceSlice<T> + ?Sized),
        stream: &IcicleStream,
    ) -> Result<(), eIcicleError> {
        assert!(
            self.len() >= src.len(),
            "In copy, destination has shorter length than source"
        );

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

    fn memset(&mut self, value: u8, size: usize) -> Result<(), eIcicleError>;
    fn memset_async(&mut self, value: u8, size: usize, stream: &IcicleStream) -> Result<(), eIcicleError>;
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

    fn memset(&mut self, value: u8, size: usize) -> Result<(), eIcicleError> {
        if size == 0 || self.is_empty() {
            return Ok(());
        }
        if size > self.len() {
            panic!("size exceeds slice length");
        }
        unsafe {
            runtime::memset(self.as_mut_ptr() as *mut c_void, value as i32, size).wrap()?;
        }
        Ok(())
    }

    fn memset_async(&mut self, value: u8, size: usize, stream: &IcicleStream) -> Result<(), eIcicleError> {
        if size == 0 || self.is_empty() {
            return Ok(());
        }
        if size > self.len() {
            panic!("size exceeds slice length");
        }
        unsafe {
            runtime::memset_async(self.as_mut_ptr() as *mut c_void, value as i32, size, stream.handle).wrap()?;
        }
        Ok(())
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

    fn memset(&mut self, value: u8, size: usize) -> Result<(), eIcicleError> {
        if size == 0 || self.is_empty() {
            return Ok(());
        }
        if size > self.len() {
            panic!("size exceeds slice length");
        }
        if !self.is_on_active_device() {
            panic!("not allocated on an active device");
        }

        let byte_size = size_of::<T>() * size;
        unsafe { runtime::memset(self.as_mut_ptr() as *mut c_void, value as i32, byte_size).wrap() }
    }

    fn memset_async(&mut self, value: u8, size: usize, stream: &IcicleStream) -> Result<(), eIcicleError> {
        if size == 0 || self.is_empty() {
            return Ok(());
        }
        if size > self.len() {
            panic!("size exceeds slice length");
        }
        if !self.is_on_active_device() {
            panic!("not allocated on an active device");
        }

        let byte_size = size_of::<T>() * size;
        unsafe {
            runtime::memset_async(self.as_mut_ptr() as *mut c_void, value as i32, byte_size, stream.handle).wrap()
        }
    }
}

// Note: Implementing the trait for DeviceVec such that functions expecting DeviceSlice reference can take a HostOrDeviceSlice reference without being a breaking change.
// Otherwise the syntax would be &device_vec[..] rather than &device_vec which would be a breaking change.
impl<T> HostOrDeviceSlice<T> for DeviceVec<T> {
    fn is_on_device(&self) -> bool {
        // Forward to the dereferenced DeviceSlice
        (&**self).is_on_device()
    }

    fn is_on_active_device(&self) -> bool {
        // Forward to the dereferenced DeviceSlice
        (&**self).is_on_active_device()
    }

    unsafe fn as_ptr(&self) -> *const T {
        // Forward to the dereferenced DeviceSlice
        (&**self).as_ptr()
    }

    unsafe fn as_mut_ptr(&mut self) -> *mut T {
        // Forward to the dereferenced DeviceSlice
        (&mut **self).as_mut_ptr()
    }

    fn len(&self) -> usize {
        // Forward to the dereferenced DeviceSlice
        (&**self).len()
    }

    fn is_empty(&self) -> bool {
        // Forward to the dereferenced DeviceSlice
        (&**self).is_empty()
    }

    fn memset(&mut self, value: u8, size: usize) -> Result<(), eIcicleError> {
        self.as_mut_slice()
            .memset(value, size)
    }

    fn memset_async(&mut self, value: u8, size: usize, stream: &IcicleStream) -> Result<(), eIcicleError> {
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
        &*(std::slice::from_raw_parts(ptr, len) as *const [T] as *const HostSlice<T>)
    }

    pub unsafe fn from_raw_parts_mut<'a>(ptr: *mut T, len: usize) -> &'a mut HostSlice<T> {
        &mut *(std::slice::from_raw_parts_mut(ptr, len) as *mut [T] as *mut HostSlice<T>)
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
            panic!("not allocated on an active device");
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
            panic!("not allocated on an active device");
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
            panic!("not allocated on an active device");
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
            panic!("not allocated on an active device");
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

    /// # Safety
    /// `ptr` must point to `len` contiguous elements in device memory.
    /// The caller must ensure the memory is valid for the lifetime `'a` and not aliased.
    pub unsafe fn from_raw_parts<'a>(ptr: *const T, len: usize) -> &'a DeviceSlice<T> {
        &*(std::slice::from_raw_parts(ptr, len) as *const [T] as *const DeviceSlice<T>)
    }

    /// # Safety
    /// `ptr` must point to `len` contiguous elements in device memory and be uniquely owned.
    /// The caller must ensure the memory is valid for the lifetime `'a` and not aliased.
    pub unsafe fn from_raw_parts_mut<'a>(ptr: *mut T, len: usize) -> &'a mut DeviceSlice<T> {
        &mut *(std::slice::from_raw_parts_mut(ptr, len) as *mut [T] as *mut DeviceSlice<T>)
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

    pub fn as_mut_slice(&mut self) -> &mut DeviceSlice<T> {
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
    fn compute_output_len<From, To>(len: usize) -> Result<usize, eIcicleError> {
        let from_size = size_of::<From>();
        let to_size = size_of::<To>();

        if from_size == 0 || to_size == 0 {
            return Err(eIcicleError::InvalidArgument);
        }

        let total_bytes = from_size
            .checked_mul(len)
            .ok_or(eIcicleError::InvalidArgument)?;

        if total_bytes % to_size != 0 {
            return Err(eIcicleError::InvalidArgument);
        }

        Ok(total_bytes / to_size)
    }

    /// SAFETY: Caller must ensure layout of P as [P::Base; DEGREE]
    pub unsafe fn reinterpret_slice<From, To>(
        input: &(impl HostOrDeviceSlice<From> + ?Sized),
    ) -> Result<UnifiedSlice<'_, To>, eIcicleError>
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
    ) -> Result<UnifiedSliceMut<'_, To>, eIcicleError>
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

    impl<'a, T> HostOrDeviceSlice<T> for UnifiedSlice<'a, T> {
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

        fn memset(&mut self, _: u8, _: usize) -> Result<(), eIcicleError> {
            panic!("Cannot memset immutable UnifiedSlice");
        }

        fn memset_async(&mut self, _: u8, _: usize, _: &IcicleStream) -> Result<(), eIcicleError> {
            panic!("Cannot memset_async immutable UnifiedSlice");
        }
    }

    impl<'a, T> HostOrDeviceSlice<T> for UnifiedSliceMut<'a, T> {
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

        fn memset(&mut self, value: u8, size: usize) -> Result<(), eIcicleError> {
            match self {
                UnifiedSliceMut::Device(d) => d.memset(value, size),
                UnifiedSliceMut::Host(h) => h.memset(value, size),
            }
        }

        fn memset_async(&mut self, value: u8, size: usize, stream: &IcicleStream) -> Result<(), eIcicleError> {
            match self {
                UnifiedSliceMut::Device(d) => d.memset_async(value, size, stream),
                UnifiedSliceMut::Host(h) => h.memset_async(value, size, stream),
            }
        }
    }
}
