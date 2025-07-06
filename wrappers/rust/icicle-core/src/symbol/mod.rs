use crate::ring::IntegerRing;
use crate::traits::Handle;
use icicle_runtime::errors::eIcicleError;
use std::ffi::c_void;
use std::marker::Copy;
use std::ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign};

pub type SymbolHandle = *const c_void;
#[doc(hidden)]
pub trait Symbol<T: IntegerRing>:
    Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Add<T, Output = Self>
    + Sub<T, Output = Self>
    + Mul<T, Output = Self>
    + AddAssign
    + SubAssign
    + MulAssign
    + AddAssign<T>
    + SubAssign<T>
    + MulAssign<T>
    + for<'a> Add<&'a Self, Output = Self>
    + for<'a> Sub<&'a Self, Output = Self>
    + for<'a> Mul<&'a Self, Output = Self>
    + for<'a> AddAssign<&'a Self>
    + for<'a> SubAssign<&'a Self>
    + for<'a> MulAssign<&'a Self>
    + Clone
    + Copy
    + Sized
    + Handle
{
    fn new_input(in_idx: u32) -> Result<Self, eIcicleError>; // New input symbol for the execution function
    fn from_constant(constant: T) -> Result<Self, eIcicleError>; // New symbol from a ring element
}

#[macro_export]
macro_rules! impl_symbol_ring {
  (
    $ring_prefix:literal,
    $ring_prefix_ident:ident,
    $ring:ident
  ) => {
    pub mod $ring_prefix_ident {
      use crate::symbol::$ring;
      use icicle_core::ring::IntegerRing;
      use icicle_core::traits::{Arithmetic, Handle};
      use icicle_core::symbol::{Symbol, SymbolHandle};
      use icicle_runtime::errors::eIcicleError;
      use std::ops::{Add, Sub, Mul, AddAssign, SubAssign, MulAssign};
      use std::ffi::c_void;
      use std::fmt;

      #[repr(C)]
      #[derive(Copy)]
      pub struct RingSymbol {
        pub(crate) m_handle: SymbolHandle,
      }

      // Symbol Operations
      extern "C" {
        #[link_name = concat!($ring_prefix, "_create_input_symbol")]
        pub(crate) fn ffi_input_symbol(in_idx: u32) -> SymbolHandle;

        #[link_name = concat!($ring_prefix, "_create_scalar_symbol")]
        pub(crate) fn ffi_symbol_from_const(constant: *const $ring) -> SymbolHandle;

        #[link_name = concat!($ring_prefix, "_copy_symbol")]
        pub(crate) fn ffi_copy_symbol(other: SymbolHandle) -> SymbolHandle;

        #[link_name = concat!($ring_prefix, "_add_symbols")]
        pub(crate) fn ffi_add_symbols(op_a: SymbolHandle, op_b: SymbolHandle, res: *mut SymbolHandle) -> eIcicleError;

        #[link_name = concat!($ring_prefix, "_sub_symbols")]
        pub(crate) fn ffi_sub_symbols(op_a: SymbolHandle, op_b: SymbolHandle, res: *mut SymbolHandle) -> eIcicleError;

        #[link_name = concat!($ring_prefix, "_multiply_symbols")]
        pub(crate) fn ffi_multiply_symbols(op_a: SymbolHandle, op_b: SymbolHandle, res: *mut SymbolHandle) -> eIcicleError;

      }

      // Implement Symbol UI
      impl Symbol<$ring> for RingSymbol {
        fn new_input(in_idx: u32) -> Result<Self, eIcicleError> {
          unsafe {
            let handle = ffi_input_symbol(in_idx);
            if handle.is_null() {
              Err(eIcicleError::AllocationFailed)
            } else {
              Ok(Self { m_handle: handle })
            }
          }
        }

        fn from_constant(constant: $ring) -> Result<Self, eIcicleError> {
          unsafe {
            let handle = ffi_symbol_from_const(&constant as *const $ring);
            if handle.is_null() {
              Err(eIcicleError::AllocationFailed)
            } else {
              Ok(Self { m_handle: handle })
            }
          }
        }
      }

      // Implement useful functions for the implementation of the above UI
      impl RingSymbol {
        fn add_handles(op_a: SymbolHandle, op_b: SymbolHandle) -> Result<SymbolHandle, eIcicleError> {
          unsafe {
            let mut handle = std::ptr::null();
            let ffi_status = ffi_add_symbols(op_a, op_b, &mut handle);
            if ffi_status != eIcicleError::Success {
              Err(ffi_status)
            } else if handle.is_null() {
              Err(eIcicleError::AllocationFailed)
            } else {
              Ok(handle)
            }
          }
        }

        fn sub_handles(op_a: SymbolHandle, op_b: SymbolHandle) -> Result<SymbolHandle, eIcicleError> {
          unsafe {
            let mut handle = std::ptr::null();
            let ffi_status = ffi_sub_symbols(op_a, op_b, &mut handle);
            if ffi_status != eIcicleError::Success {
              Err(ffi_status)
            } else if handle.is_null() {
              Err(eIcicleError::AllocationFailed)
            } else {
              Ok(handle)
            }
          }
        }

        fn mul_handles(op_a: SymbolHandle, op_b: SymbolHandle) -> Result<SymbolHandle, eIcicleError> {
          unsafe {
            let mut handle = std::ptr::null();
            let ffi_status = ffi_multiply_symbols(op_a, op_b, &mut handle);
            if ffi_status != eIcicleError::Success {
              Err(ffi_status)
            } else if handle.is_null() {
              Err(eIcicleError::AllocationFailed)
            } else {
              Ok(handle)
            }
          }
        }

        fn add_ring(self, other: $ring) -> Result<Self, eIcicleError> {
          let other_symbol = Self::from_constant(other)?;
          let res_handle = Self::add_handles(self.m_handle, other_symbol.m_handle)?;
          Ok(Self { m_handle: res_handle })
        }

        fn sub_ring(self, other: $ring) -> Result<Self, eIcicleError> {
          let other_symbol = Self::from_constant(other)?;
          let res_handle = Self::sub_handles(self.m_handle, other_symbol.m_handle)?;
          Ok(Self { m_handle: res_handle })
        }

        fn mul_ring(self, other: $ring) -> Result<Self, eIcicleError> {
          let other_symbol = Self::from_constant(other)?;
          let res_handle = Self::mul_handles(self.m_handle, other_symbol.m_handle)?;
          Ok(Self { m_handle: res_handle })
        }
      }

      impl Handle for RingSymbol {
        fn handle(&self) -> SymbolHandle { self.m_handle }
      }

      // Implement other traits required by Symbol<F>
      macro_rules! impl_op {
        ($op_token: tt, $op:ident, $assign_op:ident, $method:ident, $assign_method:ident, $handles_method:ident) => {
          // Owned op Owned
          impl $op<RingSymbol> for RingSymbol
          {
            type Output = RingSymbol;

            fn $method(self, other: RingSymbol) -> RingSymbol {
              let res_handle = Self::$handles_method(self.m_handle, other.m_handle)
                .expect(concat!("Allocation failed during ", stringify!($op), " operation"));
              RingSymbol { m_handle: res_handle }
            }
          }

          // Owned op &Reference
          impl $op<&RingSymbol> for RingSymbol
          {
            type Output = RingSymbol;

            fn $method(self, other: &RingSymbol) -> RingSymbol {
              let res_handle = Self::$handles_method(self.m_handle, other.m_handle)
                .expect(concat!("Allocation failed during ", stringify!($op), " operation"));
              RingSymbol { m_handle: res_handle }
            }
          }

          // &Reference op &Reference
          impl $op<&RingSymbol> for &RingSymbol
          {
            type Output = RingSymbol;

            fn $method(self, other: &RingSymbol) -> RingSymbol {
              let res_handle = RingSymbol::$handles_method(self.m_handle, other.m_handle)
                .expect(concat!("Allocation failed during ", stringify!($op), " operation"));
              RingSymbol { m_handle: res_handle }
            }
          }

          // &Reference op Owned
          impl $op<RingSymbol> for &RingSymbol
          {
            type Output = RingSymbol;

            fn $method(self, other: RingSymbol) -> RingSymbol {
              let res_handle = RingSymbol::$handles_method(self.m_handle, other.m_handle)
                .expect(concat!("Allocation failed during ", stringify!($op), " operation"));
              RingSymbol { m_handle: res_handle }
            }
          }

          // Owned op Field
          impl $op<$ring> for RingSymbol {
            type Output = RingSymbol;

            fn $method(self, other: $ring) -> Self {
              let other_symbol = RingSymbol::from_constant(other)
                .expect(concat!("Allocation failed during ", stringify!($op), " operation"));
              let res_handle = RingSymbol::$handles_method(self.m_handle, other_symbol.m_handle)
                .expect(concat!("Allocation failed during ", stringify!($op), " operation"));
              RingSymbol { m_handle: res_handle }
            }
          }

          // &Reference op Field
          impl $op<$ring> for &RingSymbol {
            type Output = RingSymbol;

            fn $method(self, other: $ring) -> RingSymbol {
              let other_symbol = RingSymbol::from_constant(other)
                .expect(concat!("Allocation failed during ", stringify!($op), " operation"));
              let res_handle = RingSymbol::$handles_method(self.m_handle, other_symbol.m_handle)
                .expect(concat!("Allocation failed during ", stringify!($op), " operation"));
              RingSymbol { m_handle: res_handle }
            }
          }

          // Field op Owned
          impl $op<RingSymbol> for $ring {
            type Output = RingSymbol;

            fn $method(self, other: RingSymbol) -> RingSymbol {
              other $op_token self
            }
          }

          // Field op &Reference
          impl $op<&RingSymbol> for $ring {
            type Output = RingSymbol;

            fn $method(self, other: &RingSymbol) -> RingSymbol {
              other $op_token self
            }
          }

          // Owned opAssign Owned
          impl $assign_op<RingSymbol> for RingSymbol
          {
            fn $assign_method(&mut self, other: RingSymbol) {
              let res_handle = Self::$handles_method(self.m_handle, other.m_handle)
                .expect(concat!("Allocation failed during ", stringify!($op), " operation"));
              self.m_handle = res_handle;
            }
          }

          // Owned opAssign &Reference
          impl $assign_op<&RingSymbol> for RingSymbol
          {
            fn $assign_method(&mut self, other: &RingSymbol) {
              let res_handle = Self::$handles_method(self.m_handle, other.m_handle)
                .expect(concat!("Allocation failed during ", stringify!($op), " operation"));
              self.m_handle = res_handle;
            }
          }

          // Owned opAssign Field
          impl $assign_op<$ring> for RingSymbol
          {
            fn $assign_method(&mut self, other: $ring) {
              let other_symbol = RingSymbol::from_constant(other)
                .expect(concat!("Allocation failed during ", stringify!($op), " operation"));
              let res_handle = Self::$handles_method(self.m_handle, other_symbol.m_handle)
                .expect(concat!("Allocation failed during ", stringify!($op), " operation"));
              self.m_handle = res_handle;
            }
          }
        };
      }

      impl_op!(+, Add, AddAssign, add, add_assign, add_handles);
      impl_op!(-, Sub, SubAssign, sub, sub_assign, sub_handles);
      impl_op!(*, Mul, MulAssign, mul, mul_assign, mul_handles);

      impl Clone for RingSymbol where RingSymbol: Symbol<$ring> {
        fn clone(&self) -> Self {
          unsafe {
            let handle = ffi_copy_symbol(self.m_handle);
            if handle.is_null() {
              panic!("Failed to clone Symbol: backend returned a null handle.");
            }
            Self { m_handle: handle }
          }
        }
      }

      // Implement Arithmetic trait for RingSymbol
      impl Arithmetic for RingSymbol {
        fn sqr(&self) -> Self {
          self * self
        }

        fn pow(&self, exp: usize) -> Self {
          let mut result = Self::from_constant($ring::one()).expect("Failed to create constant one");
          let mut base = *self;
          let mut exp_val = exp;

          while exp_val > 0 {
            if exp_val & 1 == 1 {
              result = result * base;
            }
            base = base * base;
            exp_val >>= 1;
          }

          result
        }
      }
    }
  };
}

#[macro_export]
macro_rules! impl_invertible_symbol_ring {
    ($ring_prefix:literal, $ring_prefix_ident:ident, $ring:ident) => {
        icicle_core::impl_symbol_ring!($ring_prefix, $ring_prefix_ident, $ring);

        impl icicle_core::traits::Invertible for $ring_prefix_ident::RingSymbol {
            fn inv(&self) -> Self {
                extern "C" {
                    #[link_name = concat!($ring_prefix, "_inverse_symbol")]
                    pub(crate) fn ffi_inverse_symbol(
                        op_a: icicle_core::symbol::SymbolHandle,
                        res: *mut icicle_core::symbol::SymbolHandle,
                    ) -> icicle_runtime::errors::eIcicleError;
                }
                unsafe {
                    let mut handle = std::ptr::null();
                    let ffi_status = ffi_inverse_symbol(self.m_handle, &mut handle);
                    if ffi_status != icicle_runtime::errors::eIcicleError::Success {
                        panic!("Couldn't invert symbol, due to {:?}", ffi_status);
                    } else if handle.is_null() {
                        panic!("Inverse allocation failed!");
                    } else {
                        Self { m_handle: handle }
                    }
                }
            }
        }
    };
}
