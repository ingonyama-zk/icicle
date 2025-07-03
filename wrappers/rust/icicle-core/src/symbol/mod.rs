use crate::field::PrimeField;
use crate::traits::Handle;
use icicle_runtime::errors::eIcicleError;
use std::ffi::c_void;
use std::marker::Copy;
use std::ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign};

pub type SymbolHandle = *const c_void;
#[doc(hidden)]
pub trait Symbol<F: PrimeField>:
    Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Add<F, Output = Self>
    + Sub<F, Output = Self>
    + Mul<F, Output = Self>
    + AddAssign
    + SubAssign
    + MulAssign
    + AddAssign<F>
    + SubAssign<F>
    + MulAssign<F>
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
    fn new_input(in_idx: u32) -> Result<Self, IcicleError>; // New input symbol for the execution function
    fn from_constant(constant: F) -> Result<Self, IcicleError>; // New symbol from a field element

    fn inverse(&self) -> Self; // Field inverse of the symbol
}

#[macro_export]
macro_rules! impl_symbol_field {
  (
    $field_prefix:literal,
    $field_prefix_ident:ident,
    $field:ident
  ) => {
    pub mod $field_prefix_ident {
      use crate::symbol::$field;
      use icicle_core::field::PrimeField;
      use icicle_core::traits::{Arithmetic, Handle};
      use icicle_core::symbol::{Symbol, SymbolHandle};
      use icicle_runtime::{eIcicleError, IcicleError};
      use std::ops::{Add, Sub, Mul, AddAssign, SubAssign, MulAssign};
      use std::ffi::c_void;
      use std::fmt;

      #[repr(C)]
      #[derive(Copy)]
      pub struct FieldSymbol {
        m_handle: SymbolHandle,
      }

      // Symbol Operations
      extern "C" {
        #[link_name = concat!($field_prefix, "_create_input_symbol")]
        pub(crate) fn ffi_input_symbol(in_idx: u32) -> SymbolHandle;

        #[link_name = concat!($field_prefix, "_create_scalar_symbol")]
        pub(crate) fn ffi_symbol_from_const(constant: *const $field) -> SymbolHandle;

        #[link_name = concat!($field_prefix, "_copy_symbol")]
        pub(crate) fn ffi_copy_symbol(other: SymbolHandle) -> SymbolHandle;

        #[link_name = concat!($field_prefix, "_add_symbols")]
        pub(crate) fn ffi_add_symbols(op_a: SymbolHandle, op_b: SymbolHandle, res: *mut SymbolHandle) -> eIcicleError;

        #[link_name = concat!($field_prefix, "_sub_symbols")]
        pub(crate) fn ffi_sub_symbols(op_a: SymbolHandle, op_b: SymbolHandle, res: *mut SymbolHandle) -> eIcicleError;

        #[link_name = concat!($field_prefix, "_multiply_symbols")]
        pub(crate) fn ffi_multiply_symbols(op_a: SymbolHandle, op_b: SymbolHandle, res: *mut SymbolHandle) -> eIcicleError;

        #[link_name = concat!($field_prefix, "_inverse_symbol")]
        pub(crate) fn ffi_inverse_symbol(op_a: SymbolHandle, res: *mut SymbolHandle) -> eIcicleError;
      }

      // Implement Symbol UI
      impl Symbol<$field> for FieldSymbol {
        fn new_input(in_idx: u32) -> Result<Self, IcicleError> {
          unsafe {
            let handle = ffi_input_symbol(in_idx);
            if handle.is_null() {
              Err(IcicleError::new(eIcicleError::AllocationFailed, "Failed to create input symbol"))
            } else {
              Ok(Self { m_handle: handle })
            }
          }
        }

        fn from_constant(constant: $field) -> Result<Self, IcicleError> {
          unsafe {
            let handle = ffi_symbol_from_const(&constant as *const $field);
            if handle.is_null() {
              Err(IcicleError::new(eIcicleError::AllocationFailed, "Failed to create from constant"))
            } else {
              Ok(Self { m_handle: handle })
            }
          }
        }

        fn inverse(&self) -> Self{
          unsafe {
            let mut handle = std::ptr::null();
            let ffi_status = ffi_inverse_symbol(self.m_handle, &mut handle);
            if ffi_status != eIcicleError::Success {
              panic!("Couldn't invert symbol, due to {:?}", ffi_status);
            } else if handle.is_null() {
              panic!("Inverse allocation failed!");
            } else {
              Self { m_handle: handle }
            }
          }
        }
      }

      // Implement useful functions for the implementation of the above UI
      impl FieldSymbol {
        fn add_handles(op_a: SymbolHandle, op_b: SymbolHandle) -> Result<SymbolHandle, IcicleError> {
          unsafe {
            let mut handle = std::ptr::null();
            ffi_add_symbols(op_a, op_b, &mut handle).wrap()?;
            if handle.is_null() {
              Err(IcicleError::new(eIcicleError::AllocationFailed, "Failed to add symbols"))
            } else {
              Ok(handle)
            }
          }
        }

        fn sub_handles(op_a: SymbolHandle, op_b: SymbolHandle) -> Result<SymbolHandle, IcicleError> {
          unsafe {
            let mut handle = std::ptr::null();
            ffi_sub_symbols(op_a, op_b, &mut handle).wrap()?;
            if handle.is_null() {
              Err(IcicleError::new(eIcicleError::AllocationFailed, "Failed to subtract symbols"))
            } else {
              Ok(handle)
            }
          }
        }

        fn mul_handles(op_a: SymbolHandle, op_b: SymbolHandle) -> Result<SymbolHandle, IcicleError> {
          unsafe {
            let mut handle = std::ptr::null();
            ffi_multiply_symbols(op_a, op_b, &mut handle).wrap()?;
            if handle.is_null() {
              Err(IcicleError::new(eIcicleError::AllocationFailed, "Failed to multiply symbols"))
            } else {
              Ok(handle)
            }
          }
        }

        fn add_field(self, other: $field) -> Result<Self, IcicleError> {
          let other_symbol = Self::from_constant(other)?;
          let res_handle = Self::add_handles(self.m_handle, other_symbol.m_handle)?;
          Ok(Self { m_handle: res_handle })
        }

        fn sub_field(self, other: $field) -> Result<Self, IcicleError> {
          let other_symbol = Self::from_constant(other)?;
          let res_handle = Self::sub_handles(self.m_handle, other_symbol.m_handle)?;
          Ok(Self { m_handle: res_handle })
        }

        fn mul_field(self, other: $field) -> Result<Self, IcicleError> {
          let other_symbol = Self::from_constant(other)?;
          let res_handle = Self::mul_handles(self.m_handle, other_symbol.m_handle)?;
          Ok(Self { m_handle: res_handle })
        }
      }

      impl Handle for FieldSymbol {
        fn handle(&self) -> SymbolHandle { self.m_handle }
      }

      // Implement other traits required by Symbol<F>
      macro_rules! impl_op {
        ($op_token: tt, $op:ident, $assign_op:ident, $method:ident, $assign_method:ident, $handles_method:ident) => {
          // Owned op Owned
          impl $op<FieldSymbol> for FieldSymbol
          {
            type Output = FieldSymbol;

            fn $method(self, other: FieldSymbol) -> FieldSymbol {
              let res_handle = Self::$handles_method(self.m_handle, other.m_handle)
                .expect(concat!("Allocation failed during ", stringify!($op), " operation"));
              FieldSymbol { m_handle: res_handle }
            }
          }

          // Owned op &Reference
          impl $op<&FieldSymbol> for FieldSymbol
          {
            type Output = FieldSymbol;

            fn $method(self, other: &FieldSymbol) -> FieldSymbol {
              let res_handle = Self::$handles_method(self.m_handle, other.m_handle)
                .expect(concat!("Allocation failed during ", stringify!($op), " operation"));
              FieldSymbol { m_handle: res_handle }
            }
          }

          // &Reference op &Reference
          impl $op<&FieldSymbol> for &FieldSymbol
          {
            type Output = FieldSymbol;

            fn $method(self, other: &FieldSymbol) -> FieldSymbol {
              let res_handle = FieldSymbol::$handles_method(self.m_handle, other.m_handle)
                .expect(concat!("Allocation failed during ", stringify!($op), " operation"));
              FieldSymbol { m_handle: res_handle }
            }
          }

          // &Reference op Owned
          impl $op<FieldSymbol> for &FieldSymbol
          {
            type Output = FieldSymbol;

            fn $method(self, other: FieldSymbol) -> FieldSymbol {
              let res_handle = FieldSymbol::$handles_method(self.m_handle, other.m_handle)
                .expect(concat!("Allocation failed during ", stringify!($op), " operation"));
              FieldSymbol { m_handle: res_handle }
            }
          }

          // Owned op Field
          impl $op<$field> for FieldSymbol {
            type Output = FieldSymbol;

            fn $method(self, other: $field) -> Self {
              let other_symbol = FieldSymbol::from_constant(other)
                .expect(concat!("Allocation failed during ", stringify!($op), " operation"));
              let res_handle = FieldSymbol::$handles_method(self.m_handle, other_symbol.m_handle)
                .expect(concat!("Allocation failed during ", stringify!($op), " operation"));
              FieldSymbol { m_handle: res_handle }
            }
          }

          // &Reference op Field
          impl $op<$field> for &FieldSymbol {
            type Output = FieldSymbol;

            fn $method(self, other: $field) -> FieldSymbol {
              let other_symbol = FieldSymbol::from_constant(other)
                .expect(concat!("Allocation failed during ", stringify!($op), " operation"));
              let res_handle = FieldSymbol::$handles_method(self.m_handle, other_symbol.m_handle)
                .expect(concat!("Allocation failed during ", stringify!($op), " operation"));
              FieldSymbol { m_handle: res_handle }
            }
          }

          // Field op Owned
          impl $op<FieldSymbol> for $field {
            type Output = FieldSymbol;

            fn $method(self, other: FieldSymbol) -> FieldSymbol {
              other $op_token self
            }
          }

          // Field op &Reference
          impl $op<&FieldSymbol> for $field {
            type Output = FieldSymbol;

            fn $method(self, other: &FieldSymbol) -> FieldSymbol {
              other $op_token self
            }
          }

          // Owned opAssign Owned
          impl $assign_op<FieldSymbol> for FieldSymbol
          {
            fn $assign_method(&mut self, other: FieldSymbol) {
              let res_handle = Self::$handles_method(self.m_handle, other.m_handle)
                .expect(concat!("Allocation failed during ", stringify!($op), " operation"));
              self.m_handle = res_handle;
            }
          }

          // Owned opAssign &Reference
          impl $assign_op<&FieldSymbol> for FieldSymbol
          {
            fn $assign_method(&mut self, other: &FieldSymbol) {
              let res_handle = Self::$handles_method(self.m_handle, other.m_handle)
                .expect(concat!("Allocation failed during ", stringify!($op), " operation"));
              self.m_handle = res_handle;
            }
          }

          // Owned opAssign Field
          impl $assign_op<$field> for FieldSymbol
          {
            fn $assign_method(&mut self, other: $field) {
              let other_symbol = FieldSymbol::from_constant(other)
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

      impl Clone for FieldSymbol where FieldSymbol: Symbol<$field> {
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

      // Implement Arithmetic trait for FieldSymbol
      impl Arithmetic for FieldSymbol {
        fn sqr(&self) -> Self {
          self * self
        }

        fn inv(&self) -> Self {
          self.inverse()
        }

        fn pow(&self, exp: usize) -> Self {
          let mut result = Self::from_constant($field::one()).expect("Failed to create constant one");
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
