use icicle_runtime::errors::eIcicleError;
use std::ffi::c_void;
use std::ops::{Add, Sub, Mul, AddAssign, SubAssign, MulAssign};
use crate::traits::FieldImpl;

type Handle = *const c_void;
pub type Symbol<F> = <<F as FieldImpl>::Config as FieldHasSymbol<F>>::Symbol;

#[doc(hidden)]
pub trait SymbolTrait<F: FieldImpl>:
  SymbolBackendAPI<F> +
  Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self> +
  Add<F, Output = Self> + Sub<F, Output = Self> + Mul<F, Output = Self> +
  AddAssign + SubAssign + MulAssign + AddAssign<F> + SubAssign<F> + MulAssign<F> +
  for<'a> Add<&'a Self, Output = Self> +
  for<'a> Sub<&'a Self, Output = Self> +
  for<'a> Mul<&'a Self, Output = Self> +
  for<'a> AddAssign<&'a Self> +
  for<'a> SubAssign<&'a Self> +
  for<'a> MulAssign<&'a Self> +
  Clone + Drop
{
  fn new_empty() -> Result<Self, eIcicleError>;
  fn new_constant(constant: F) -> Result<Self, eIcicleError>;
  fn copy_symbol(other: &Self) -> Result<Self, eIcicleError>;

  fn inverse(&self) -> Self;

  // To ensure operator overloading for &Symbol this Ref type is defined and will be assigned &Self upon implementation
  type Ref: Add<Self::Ref, Output=Self> + Add<Self, Output=Self> +
            Sub<Self::Ref, Output=Self> + Sub<Self, Output=Self> +
            Mul<Self::Ref, Output=Self> + Mul<Self, Output=Self>;
}

/// Functions necessary for the Symbol implementation the user shouldn't have access to
trait SymbolBackendAPI<F: FieldImpl>: Sized {
  fn handle(&self) -> Handle;
  fn delete_handle(handle: Handle);

  fn set_as_input(&self, in_index: u32);

  fn add_handles(op_a: Handle, op_b: Handle) -> Result<Handle, eIcicleError>;
  fn sub_handles(op_a: Handle, op_b: Handle) -> Result<Handle, eIcicleError>;
  fn mul_handles(op_a: Handle, op_b: Handle) -> Result<Handle, eIcicleError>;

  fn add_field(self, other: F) -> Result<Self, eIcicleError>;
  fn sub_field(self, other: F) -> Result<Self, eIcicleError>;
  fn mul_field(self, other: F) -> Result<Self, eIcicleError>;
}

pub trait FieldHasSymbol<F: FieldImpl> {
  type Symbol: SymbolTrait<F>;
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
      // use crate::field::{ScalarCfg, ScalarField};
      use crate::program::{$field, HostOrDeviceSlice};
      use icicle_core::program::{Handle, SymbolTrait, SymbolBackendAPI, ProgramTrait, Instruction};
      use icicle_runtime::errors::eIcicleError;
      use std::ops::{Add, Sub, Mul, AddAssign, SubAssign, MulAssign};
      use std::ffi::c_void;
      use std::fmt;

      #[repr(C)]
      pub struct FieldSymbol {
        m_handle: Handle
      }
      
      // Symbol Operations
      extern "C" {
        #[link_name = concat!($field_prefix, "_program_create_empty_symbol")]
        pub(crate) fn ffi_create_empty_symbol() -> Handle;

        #[link_name = concat!($field_prefix, "_program_create_scalar_symbol")]
        pub(crate) fn ffi_create_symbol(constant: $field) -> Handle;

        #[link_name = concat!($field_prefix, "_program_copy_symbol")]
        pub(crate) fn ffi_copy_symbol(other: Handle) -> Handle;

        #[link_name = concat!($field_prefix, "_program_set_symbol_as_input")]
        pub(crate) fn ffi_set_symbol_as_input(symbol: Handle, in_index: u32) -> eIcicleError;

        #[link_name = concat!($field_prefix, "_program_add_symbols")]
        pub(crate) fn ffi_add_symbols(op_a: Handle, op_b: Handle) -> Handle;

        #[link_name = concat!($field_prefix, "_program_sub_symbols")]
        pub(crate) fn ffi_sub_symbols(op_a: Handle, op_b: Handle) -> Handle;

        #[link_name = concat!($field_prefix, "_program_multiply_symbols")]
        pub(crate) fn ffi_multiply_symbols(op_a: Handle, op_b: Handle) -> Handle;

        #[link_name = concat!($field_prefix, "_program_inverse_symbol")]
        pub(crate) fn ffi_inverse_symbol(op_a: Handle) -> Handle;

        #[link_name = "program_delete_symbol"]
        pub(crate) fn ffi_delete_symbol(symbol: Handle) -> eIcicleError;
      }

      // Implement Symbol Operations
      impl SymbolBackendAPI<$field> for FieldSymbol {
        fn handle(&self) -> Handle { self.m_handle }

        fn delete_handle(handle: Handle) {
          unsafe {
            if !handle.is_null()
            {
              unsafe { ffi_delete_symbol(handle); }
            }
          }
        }

        fn set_as_input(&self, in_index: u32) {
          unsafe { ffi_set_symbol_as_input(self.m_handle, in_index); }
        }

        fn add_handles(op_a: Handle, op_b: Handle) -> Result<Handle, eIcicleError> {
          unsafe {
            let handle = ffi_add_symbols(op_a, op_b);
            if handle.is_null() {
              Err(eIcicleError::AllocationFailed)
            } else {
              Ok(handle)
            }
          }
        }

        fn sub_handles(op_a: Handle, op_b: Handle) -> Result<Handle, eIcicleError> {
          unsafe {
            let handle = ffi_sub_symbols(op_a, op_b);
            if handle.is_null() {
              Err(eIcicleError::AllocationFailed)
            } else {
              Ok(handle)
            }
          }
        }

        fn mul_handles(op_a: Handle, op_b: Handle) -> Result<Handle, eIcicleError> {
          unsafe {
            let handle = ffi_multiply_symbols(op_a, op_b);
            if handle.is_null() {
              Err(eIcicleError::AllocationFailed)
            } else {
              Ok(handle)
            }
          }
        }

        fn add_field(self, other: $field) -> Result<Self, eIcicleError> {
          let other_symbol = FieldSymbol::new_constant(other)?;
          let res_handle = Self::add_handles(self.m_handle, other_symbol.m_handle)?;
          Ok(Self { m_handle: res_handle })
        }

        fn sub_field(self, other: $field) -> Result<Self, eIcicleError> {
          let other_symbol = FieldSymbol::new_constant(other)?;
          let res_handle = Self::sub_handles(self.m_handle, other_symbol.m_handle)?;
          Ok(Self { m_handle: res_handle })
        }

        fn mul_field(self, other: $field) -> Result<Self, eIcicleError> {
          let other_symbol = FieldSymbol::new_constant(other)?;
          let res_handle = Self::mul_handles(self.m_handle, other_symbol.m_handle)?;
          Ok(Self { m_handle: res_handle })
        }
      }

      impl SymbolTrait<$field> for FieldSymbol {
        type Ref = &FieldSymbol;
        fn new_empty() -> Result<Self, eIcicleError> {
          unsafe {
            let handle = ffi_create_empty_symbol();
            if handle.is_null() {
              Err(eIcicleError::AllocationFailed)
            } else {
              Ok(FieldSymbol { m_handle: handle })
            }
          }
        }

        fn new_constant(constant: $field) -> Result<Self, eIcicleError> {
          unsafe {
            let handle = ffi_create_symbol(constant);
            if handle.is_null() {
              Err(eIcicleError::AllocationFailed)
            } else {
              Ok(FieldSymbol { m_handle: handle })
            }
          }
        }

        fn copy_symbol(other: &Self) -> Result<Self, eIcicleError> {
          unsafe {
            let handle = ffi_copy_symbol(other.m_handle);
            if handle.is_null() {
              Err(eIcicleError::AllocationFailed)
            } else {
              Ok(FieldSymbol { m_handle: handle })
            }
          }
        }
      }

      macro_rules! impl_op {
        // For owned types: FieldSymbol op FieldSymbol
        ($op:ident, $method:ident, $safe_method:ident) => {
          impl $op<FieldSymbol> for FieldSymbol
          where
            FieldSymbol: SymbolBackendAPI<$field>,
          {
            type Output = FieldSymbol;
      
            fn $method(self, other: FieldSymbol) -> FieldSymbol {
              let res_handle = Self::$safe_method(self.m_handle, other.m_handle)
                .expect(concat!("Allocation failed during ", stringify!($op), " operation"));
              FieldSymbol { m_handle: res_handle }
            }
          }
        };
      
        // For owned type with reference: FieldSymbol op &FieldSymbol
        ($op:ident, $method:ident, $safe_method:ident, ref) => {
          impl $op<&FieldSymbol> for FieldSymbol
          where
            FieldSymbol: SymbolBackendAPI<$field>,
          {
            type Output = FieldSymbol;
      
            fn $method(self, other: &'a FieldSymbol) -> FieldSymbol {
              let res_handle = Self::$safe_method(self.m_handle, other.m_handle)
                .expect(concat!("Allocation failed during ", stringify!($op), " operation"));
              FieldSymbol { m_handle: res_handle }
            }
          }
        };
      
        // For reference op owned: &FieldSymbol op FieldSymbol
        ($op:ident, $method:ident, $safe_method:ident, ref2) => {
          impl $op<FieldSymbol> for &FieldSymbol
          where
            FieldSymbol: SymbolBackendAPI<$field>,
          {
            type Output = FieldSymbol;
      
            fn $method(self, other: FieldSymbol) -> FieldSymbol {
              let res_handle = Self::$safe_method(self.m_handle, other.m_handle)
                .expect(concat!("Allocation failed during ", stringify!($op), " operation"));
              FieldSymbol { m_handle: res_handle }
            }
          }
        };
      
        // For reference op reference: &FieldSymbol op &FieldSymbol
        ($op:ident, $method:ident, $safe_method:ident, ref3) => {
          impl $op<&FieldSymbol> for &FieldSymbol
          where
            FieldSymbol: SymbolBackendAPI<$field>,
          {
            type Output = FieldSymbol;
      
            fn $method(self, other: &'a FieldSymbol) -> FieldSymbol {
              let res_handle = Self::$safe_method(self.m_handle, other.m_handle)
                .expect(concat!("Allocation failed during ", stringify!($op), " operation"));
              FieldSymbol { m_handle: res_handle }
            }
          }
        };
      }
      
      // Use the macro for the operations you need
      impl_op!(Add, add, add_handles);
      impl_op!(Sub, sub, sub_handles);
      impl_op!(Mul, mul, mul_handles);

      fn inverse(&self) -> Self{
        unsafe {
          let handle = ffi_inverse_symbol(self.m_handle);
          if handle.is_null() {
            panic!("Inverse allocation failed!");
          } else {
            FieldSymbol { m_handle: handle }
          }
        }
      }

      impl Drop for FieldSymbol where FieldSymbol: SymbolBackendAPI<$field> {
        fn drop(&mut self) {
          FieldSymbol::delete_handle(self.m_handle);
        }
      }

      impl Clone for FieldSymbol where FieldSymbol: SymbolBackendAPI<$field> {
        fn clone(&self) -> Self {
          Self::copy_symbol(self).unwrap()
        }
      }
    }
  };
}
