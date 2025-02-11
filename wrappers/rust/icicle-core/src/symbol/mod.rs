use icicle_runtime::errors::eIcicleError;
use std::ffi::c_void;
use std::ops::{Add, Sub, Mul, AddAssign, SubAssign, MulAssign, Deref};
use crate::traits::FieldImpl;

pub type Handle = *const c_void;
pub type Symbol<F> = <<F as FieldImpl>::Config as FieldHasSymbol<F>>::Symbol;

#[doc(hidden)]
pub trait SymbolTrait<F: FieldImpl>:
  Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self> +
  Add<F, Output = Self> + Sub<F, Output = Self> + Mul<F, Output = Self> +
  AddAssign + SubAssign + MulAssign + AddAssign<F> + SubAssign<F> + MulAssign<F> +
  for<'a> Add<&'a Self, Output = Self> +
  for<'a> Sub<&'a Self, Output = Self> +
  for<'a> Mul<&'a Self, Output = Self> +
  for<'a> AddAssign<&'a Self> +
  for<'a> SubAssign<&'a Self> +
  for<'a> MulAssign<&'a Self> +
  Clone + Drop + Sized
{
  fn new_empty() -> Result<Self, eIcicleError>;
  fn new_constant(constant: F) -> Result<Self, eIcicleError>;
  fn copy_symbol(other: &Self) -> Result<Self, eIcicleError>;

  fn inverse(&self) -> Self;

  // Functions necessary for the Symbol backend implementation (irrelevant to the user)
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

pub trait SymbolRefOps<F: FieldImpl, Symbol: SymbolTrait<F>>:
  Deref<Target=Symbol> +
  Add<Output=Symbol> + Sub<Output=Symbol> + Mul<Output=Symbol> +
  Add<Symbol, Output=Symbol> + Sub<Symbol, Output=Symbol> + Mul<Symbol, Output=Symbol> +
  Add<F, Output=Symbol> + Sub<F, Output=Symbol> + Mul<F, Output=Symbol> + Sized {}

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
      use crate::symbol::$field;
      use icicle_core::symbol::{Symbol, SymbolTrait, SymbolRefOps, FieldHasSymbol, Handle};
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
        #[link_name = concat!($field_prefix, "_create_empty_symbol")]
        pub(crate) fn ffi_create_empty_symbol() -> Handle;

        #[link_name = concat!($field_prefix, "_create_scalar_symbol")]
        pub(crate) fn ffi_create_symbol(constant: $field) -> Handle;

        #[link_name = concat!($field_prefix, "_copy_symbol")]
        pub(crate) fn ffi_copy_symbol(other: Handle) -> Handle;

        #[link_name = concat!($field_prefix, "_set_symbol_as_input")]
        pub(crate) fn ffi_set_symbol_as_input(symbol: Handle, in_index: u32) -> eIcicleError;

        #[link_name = concat!($field_prefix, "_add_symbols")]
        pub(crate) fn ffi_add_symbols(op_a: Handle, op_b: Handle) -> Handle;

        #[link_name = concat!($field_prefix, "_sub_symbols")]
        pub(crate) fn ffi_sub_symbols(op_a: Handle, op_b: Handle) -> Handle;

        #[link_name = concat!($field_prefix, "_multiply_symbols")]
        pub(crate) fn ffi_multiply_symbols(op_a: Handle, op_b: Handle) -> Handle;

        #[link_name = concat!($field_prefix, "_inverse_symbol")]
        pub(crate) fn ffi_inverse_symbol(op_a: Handle) -> Handle;

        #[link_name = "delete_symbol"]
        pub(crate) fn ffi_delete_symbol(symbol: Handle) -> eIcicleError;
      }

      // Implement Symbol Operations
      impl SymbolTrait<$field> for FieldSymbol {
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

      macro_rules! impl_op {
        ($op:ident, $assign_op:ident, $method:ident, $assign_method:ident, $handles_method:ident) => {
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
              let other_symbol = FieldSymbol::new_constant(other)
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
              let other_symbol = FieldSymbol::new_constant(other)
                .expect(concat!("Allocation failed during ", stringify!($op), " operation"));
              let res_handle = FieldSymbol::$handles_method(self.m_handle, other_symbol.m_handle)
                .expect(concat!("Allocation failed during ", stringify!($op), " operation"));
              FieldSymbol { m_handle: res_handle }
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
              let other_symbol = FieldSymbol::new_constant(other)
                .expect(concat!("Allocation failed during ", stringify!($op), " operation"));
              let res_handle = Self::$handles_method(self.m_handle, other_symbol.m_handle)
                .expect(concat!("Allocation failed during ", stringify!($op), " operation"));
              Self::delete_handle(self.m_handle);
              self.m_handle = res_handle;
            }
          }
        };
      }
      
      impl_op!(Add, AddAssign, add, add_assign, add_handles);
      impl_op!(Sub, SubAssign, sub, sub_assign, sub_handles);
      impl_op!(Mul, MulAssign, mul, mul_assign, mul_handles);

      impl Drop for FieldSymbol where FieldSymbol: SymbolTrait<$field> { // TODO test without drop and with copy
        fn drop(&mut self) {
          FieldSymbol::delete_handle(self.m_handle);
        }
      }

      impl Clone for FieldSymbol where FieldSymbol: SymbolTrait<$field> {
        fn clone(&self) -> Self {
          Self::copy_symbol(self).unwrap()
        }
      }

      impl SymbolRefOps<$field, FieldSymbol> for &FieldSymbol {}
    }

    use icicle_core::symbol::FieldHasSymbol;

    impl FieldHasSymbol<$field> for $field_config {
      type Symbol = $field_prefix_ident::FieldSymbol;
    }
  };
}
