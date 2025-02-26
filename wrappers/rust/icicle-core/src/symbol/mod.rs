use icicle_runtime::errors::eIcicleError;
use std::marker::Copy;
use std::ops::{Add, Sub, Mul, AddAssign, SubAssign, MulAssign};
use crate::traits::{FieldImpl, DeletableHandle};

#[doc(hidden)]
pub trait Symbol<F: FieldImpl>:
  Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self> +
  Add<F, Output = Self> + Sub<F, Output = Self> + Mul<F, Output = Self> +
  AddAssign + SubAssign + MulAssign + AddAssign<F> + SubAssign<F> + MulAssign<F> +
  for<'a> Add<&'a Self, Output = Self> +
  for<'a> Sub<&'a Self, Output = Self> +
  for<'a> Mul<&'a Self, Output = Self> +
  for<'a> AddAssign<&'a Self> +
  for<'a> SubAssign<&'a Self> +
  for<'a> MulAssign<&'a Self> +
  Clone + Copy + Sized + DeletableHandle
{
  fn new_input(in_idx: u32) -> Result<Self, eIcicleError>;      // New input symbol for the execution function
  fn from_constant(constant: F) -> Result<Self, eIcicleError>;  // New symbol from a field element

  fn inverse(&self) -> Self; // Field inverse of the symbol
}

#[macro_export]
macro_rules! impl_symbol_field {
  (
    $field_prefix:literal,
    $field_prefix_ident:ident,
    $field:ident,
    $field_config:ident
  ) => {
    pub mod $field_prefix_ident {
      use crate::symbol::$field;
      use icicle_core::traits::{DeletableHandle, Handle, HandleCPP};
      use icicle_core::symbol::Symbol;
      use icicle_runtime::errors::eIcicleError;
      use std::ops::{Add, Sub, Mul, AddAssign, SubAssign, MulAssign};
      use std::ffi::c_void;
      use std::fmt;

      #[repr(C)]
      #[derive(Copy)]
      pub struct FieldSymbol {
        m_handle: HandleCPP,
      }

      // Symbol Operations
      extern "C" {
        #[link_name = concat!($field_prefix, "_create_input_symbol")]
        pub(crate) fn ffi__input_symbol(in_idx: u32) -> HandleCPP;

        #[link_name = concat!($field_prefix, "_create_scalar_symbol")]
        pub(crate) fn ffi__symbol_from_const(constant: *const $field) -> HandleCPP;

        #[link_name = concat!($field_prefix, "_copy_symbol")]
        pub(crate) fn ffi_copy_symbol(other: HandleCPP) -> HandleCPP;

        #[link_name = concat!($field_prefix, "_add_symbols")]
        pub(crate) fn ffi_add_symbols(op_a: HandleCPP, op_b: HandleCPP, res: *mut HandleCPP) -> eIcicleError;

        #[link_name = concat!($field_prefix, "_sub_symbols")]
        pub(crate) fn ffi_sub_symbols(op_a: HandleCPP, op_b: HandleCPP, res: *mut HandleCPP) -> eIcicleError;

        #[link_name = concat!($field_prefix, "_multiply_symbols")]
        pub(crate) fn ffi_multiply_symbols(op_a: HandleCPP, op_b: HandleCPP, res: *mut HandleCPP) -> eIcicleError;

        #[link_name = concat!($field_prefix, "_inverse_symbol")]
        pub(crate) fn ffi_inverse_symbol(op_a: HandleCPP, res: *mut HandleCPP) -> eIcicleError;

        #[link_name = "delete_symbol"]
        pub(crate) fn ffi_delete_symbol(symbol: HandleCPP) -> eIcicleError;
      }

      // Implement Symbol UI
      impl Symbol<$field> for FieldSymbol {
        fn new_input(in_idx: u32) -> Result<Self, eIcicleError> {
          unsafe {
            let handle = ffi__input_symbol(in_idx);
            if handle.is_null() {
              Err(eIcicleError::AllocationFailed)
            } else {
              Ok(Self { m_handle: handle })
            }
          }
        }

        fn from_constant(constant: $field) -> Result<Self, eIcicleError> {
          unsafe {
            let handle = ffi__symbol_from_const(&constant as *const $field);
            if handle.is_null() {
              Err(eIcicleError::AllocationFailed)
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
        fn add_handles(op_a: HandleCPP, op_b: HandleCPP) -> Result<HandleCPP, eIcicleError> {
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

        fn sub_handles(op_a: HandleCPP, op_b: HandleCPP) -> Result<HandleCPP, eIcicleError> {
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

        fn mul_handles(op_a: HandleCPP, op_b: HandleCPP) -> Result<HandleCPP, eIcicleError> {
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

        fn add_field(self, other: $field) -> Result<Self, eIcicleError> {
          let other_symbol = Self::from_constant(other)?;
          let res_handle = Self::add_handles(self.m_handle, other_symbol.m_handle)?;
          Ok(Self { m_handle: res_handle })
        }

        fn sub_field(self, other: $field) -> Result<Self, eIcicleError> {
          let other_symbol = Self::from_constant(other)?;
          let res_handle = Self::sub_handles(self.m_handle, other_symbol.m_handle)?;
          Ok(Self { m_handle: res_handle })
        }

        fn mul_field(self, other: $field) -> Result<Self, eIcicleError> {
          let other_symbol = Self::from_constant(other)?;
          let res_handle = Self::mul_handles(self.m_handle, other_symbol.m_handle)?;
          Ok(Self { m_handle: res_handle })
        }
      }

      impl Handle for FieldSymbol {
        fn handle(&self) -> HandleCPP { self.m_handle }
      }

      impl DeletableHandle for FieldSymbol {
        fn delete_handle(handle: HandleCPP) {
          unsafe {
            if !handle.is_null()
            {
              unsafe {
                let ffi_status = ffi_delete_symbol(handle);
                if ffi_status != eIcicleError::Success {
                  panic!("Couldn't delete symbol, due to {:?}", ffi_status);
                }
              }
            }
          }
        }
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
              Self::delete_handle(self.m_handle);
              self.m_handle = res_handle;
            }
          }
      
          // Owned opAssign &Reference
          impl $assign_op<&FieldSymbol> for FieldSymbol
          {
            fn $assign_method(&mut self, other: &FieldSymbol) {
              let res_handle = Self::$handles_method(self.m_handle, other.m_handle)
                .expect(concat!("Allocation failed during ", stringify!($op), " operation"));
              Self::delete_handle(self.m_handle);
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
              Self::delete_handle(self.m_handle);
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
    }
  };
}
