use icicle_runtime::{
  errors::eIcicleError, memorr::HostOrDeviceSlice, stream::IcicleStreamHandle,
};
use std::{
  ffi::c_void, mem, ptr, slice,
  ops::{Add, Sub, Mul, AddAssign, SubAssign, MulAssign}
};

type SymbolHandle = *const c_void;

/// Symbol struct only holding a pointer to the memory allocated by the backend
pub struct Symbol<F> {
  m_handle: SymbolHandle
}

/// Symbol interface
pub trait SymbolTrait<F> {
  fn new_empty() -> Result<Self, eIcicleError>
  where
    Self: Sized;

  fn new_constant(constant: F) -> Result<Self, eIcicleError>
  where
    Self: Sized;

  fn copy_symbol(other: &Symbol<F>) -> Result<Self, eIcicleError>
  where
    Self: Sized;

  fn set_as_input(&self, in_index: u32);
  fn inverse(&self) -> Result<Self, eIcicleError>
  where
    Self: Sized;

  // Safe wrappers for base C arithmetic functions
  fn add_safe(self, other: Symbol<F>) -> Result<Self, eIcicleError>;
  fn sub_safe(self, other: Symbol<F>) -> Result<Self, eIcicleError>;
  fn mul_safe(self, other: Symbol<F>) -> Result<Self, eIcicleError>;

  fn add_field(self, other: F) -> Result<Self, eIcicleError>;
  fn sub_field(self, other: F) -> Result<Self, eIcicleError>;
  fn mul_field(self, other: F) -> Result<Self, eIcicleError>;

  fn add_assign_field(&mut self, other: F);
  fn sub_assign_field(&mut self, other: F);
  fn mul_assign_field(&mut self, other: F);
}

macro_rules! impl_program_field {
  ($field_prefix:literal, $field_prefix_ident:ident, $field:ty) => {
      mod $field_prefix_ident {
          use super::{eIcicleError, SymbolHandle, Symbol, SymbolTrait};

          extern "C" {
              #[link_name = concat!($field_prefix, "_program_create_empty_symbol")]
              pub(crate) fn ffi_create_empty_symbol() -> SymbolHandle;

              #[link_name = concat!($field_prefix, "_program_create_scalar_symbol")]
              pub(crate) fn ffi_create_symbol(constant: $field) -> SymbolHandle;

              #[link_name = concat!($field_prefix, "_program_copy_symbol")]
              pub(crate) fn ffi_copy_symbol(other: SymbolHandle) -> SymbolHandle;

              #[link_name = concat!($field_prefix, "_program_add_symbols")]
              pub(crate) fn ffi_add_symbols(op_a: SymbolHandle, op_b: SymbolHandle) -> SymbolHandle;

              #[link_name = concat!($field_prefix, "_program_sub_symbols")]
              pub(crate) fn ffi_sub_symbols(op_a: SymbolHandle, op_b: SymbolHandle) -> SymbolHandle;

              #[link_name = concat!($field_prefix, "_program_multiply_symbols")]
              pub(crate) fn ffi_multiply_symbols(op_a: SymbolHandle, op_b: SymbolHandle) -> SymbolHandle;
          }

          impl SymbolTrait<$field> for Symbol<$field> {
              pub fn new_empty() -> Result<Self, eIcicleError> {
                  unsafe {
                      let handle = ffi_create_empty_symbol();
                      if handle.is_null() {
                          Err(eIcicleError::AllocationFailed)
                      } else {
                          Ok(Symbol { handle })
                      }
                  }
              }

              pub fn new_constant(constant: $field) -> Result<Self, eIcicleError> {
                  unsafe {
                      let handle = ffi_create_symbol(constant);
                      if handle.is_null() {
                          Err(eIcicleError::AllocationFailed)
                      } else {
                          Ok(Symbol { handle })
                      }
                  }
              }

              pub fn copy_symbol(other: &Symbol<$field>) -> Result<Self, eIcicleError> {
                  unsafe {
                      let handle = ffi_copy_symbol(other.m_handle);
                      if handle.is_null() {
                          Err(eIcicleError::AllocationFailed)
                      } else {
                          Ok(Symbol { handle })
                      }
                  }
              }

              fn add_safe(self, other: Symbol<$field>) -> Result<Symbol<$field>, eIcicleError> {
                  unsafe {
                      let handle = ffi_add_symbols(self.m_handle, other.m_handle);
                      if handle.is_null() {
                          Err(eIcicleError::AllocationFailed)
                      } else {
                          Ok(Symbol { handle })
                      }
                  }
              }

              fn sub_safe(self, other: Symbol<$field>) -> Result<Symbol<$field>, eIcicleError> {
                unsafe {
                  let handle = ffi_sub_symbols(self.m_handle, other.m_handle);
                  if handle.is_null() {
                      Err(eIcicleError::AllocationFailed)
                  } else {
                      Ok(Symbol { handle })
                  }
                }
              }

              fn mul_safe(self, other: Symbol<$field>) -> Result<Symbol<$field>, eIcicleError> {
                unsafe {
                  let handle = ffi_multiply_symbols(self.m_handle, other.m_handle);
                  if handle.is_null() {
                      Err(eIcicleError::AllocationFailed)
                  } else {
                      Ok(Symbol { handle })
                  }
                }
              }

              pub fn add_field(self, other: $field) -> Result<Symbol<$field>, eIcicleError> {
                let other_symbol = Symbol::new_constant(other)?;
                self.add_safe(other_symbol)
              }
              
              pub fn sub_field(self, other: $field) -> Result<Symbol<$field>, eIcicleError> {
                let other_symbol = Symbol::new_constant(other)?;
                self.sub_safe(other_symbol)
              }
              
              pub fn mul_field(self, other: $field) -> Result<Symbol<$field>, eIcicleError> {
                let other_symbol = Symbol::new_constant(other)?;
                self.mul_safe(other_symbol)
              }
              
              pub fn add_assign_field(&mut self, other: $field) {
                let other_symbol = Symbol::new_constant(other).expect("Allocation failed during add operation");
                let result = self.add_safe(other_symbol).expect("Allocation failed during add operation");
                self.handle = result.handle;
              }
              
              pub fn sub_assign_field(&mut self, other: $field) {
                let other_symbol = Symbol::new_constant(other).expect("Allocation failed during sub operation");
                let result = self.sub_safe(other_symbol).expect("Allocation failed during sub operation");
                self.handle = result.handle;
              }
              
              pub fn mul_assign_field(&mut self, other: $field) {
                let other_symbol = Symbol::new_constant(other).expect("Allocation failed during mul operation");
                let result = self.mul_safe(other_symbol).expect("Allocation failed during mul operation");
                self.handle = result.handle;
              }

              fn inverse(&self) -> Symbol<$field, eIcicleError> {
                unsafe {
                  let handle = ffi_inverse(self.m_handle);
                  if handle.is_null() {
                    Err(eIcicleError::AllocationFailed)
                  } else {
                      Ok(Symbol { handle })
                  }
                }
              }
          }
      }
  };
}

// Overload arithmetic operators for symbol
impl<F: SymbolTrait<F>> Add for Symbol<F> {
  type Output = Result<Self, eIcicleError>;

  fn add(self, other: Self) -> Result<Self, eIcicleError> {
    SymbolTrait::add_safe(self, other)
  }
}

impl<F: SymbolTrait<F>> Sub for Symbol<F> {
  type Output = Result<Self, eIcicleError>;

  fn sub(self, other: Self) -> Result<Self, eIcicleError> {
    SymbolTrait::sub_safe(self, other)
  }
}

impl<F: SymbolTrait<F>> Mul for Symbol<F> {
  type Output = Result<Self, eIcicleError>;

  fn mul(self, other: Self) -> Result<Self, eIcicleError> {
    SymbolTrait::mul_safe(self, other)
  }
}

impl<F: SymbolTrait<F>> Add<F> for Symbol<F> {
  type Output = Result<Self, eIcicleError>;

  fn add(self, other: F) -> Result<Self, eIcicleError> {
    SymbolTrait::add_field(self, other)
  }
}

impl<F: SymbolTrait<F>> Sub<F> for Symbol<F> {
  type Output = Result<Self, eIcicleError>;

  fn sub(self, other: F) -> Result<Self, eIcicleError> {
    SymbolTrait::sub_field(self, other)
  }
}

impl<F: SymbolTrait<F>> Mul<F> for Symbol<F> {
  type Output = Result<Self, eIcicleError>;

  fn mul(self, other: F) -> Result<Self, eIcicleError> {
    SymbolTrait::mul_field(self, other)
  }
}

impl<F: SymbolTrait<F>> AddAssign for Symbol<F> {
  fn add_assign(&mut self, other: Self) {
    let result = SymbolTrait::add_safe(self, other).expect("Allocation failed during add operation");
    self.handle = result.handle;
  }
}

impl<F: SymbolTrait<F>> SubAssign for Symbol<F> {
  fn sub_assign(&mut self, other: Self) {
    let result = SymbolTrait::sub_safe(self, other).expect("Allocation failed during sub operation");
    self.handle = result.handle;
  }
}

impl<F: SymbolTrait<F>> MulAssign for Symbol<F> {
  fn mul_assign(&mut self, other: Self) {
    let result = SymbolTrait::mul_safe(self, other).expect("Allocation failed during mul operation");
    self.handle = result.handle;
  }
}

impl<F: SymbolTrait<F>> AddAssign<F> for Symbol<F> {
  fn add_assign(&mut self, other: F) {
    let other_symbol = Symbol::new_constant(other).expect("Allocation failed during add operation");
    let result = SymbolTrait::add_safe(self, other_symbol).expect("Allocation failed during add operation");
    self.handle = result.handle;
  }
}

impl<F: SymbolTrait<F>> SubAssign<F> for Symbol<F> {
  fn sub_assign(&mut self, other: F) {
    let other_symbol = Symbol::new_constant(other).expect("Allocation failed during sub operation");
    let result = SymbolTrait::sub_safe(self, other_symbol).expect("Allocation failed during sub operation");
    self.handle = result.handle;
  }
}

impl<F: SymbolTrait<F>> MulAssign<F> for Symbol<F> {
  fn mul_assign(&mut self, other: F) {
    let other_symbol = Symbol::new_constant(other).expect("Allocation failed during mul operation");
    let result = SymbolTrait::mul_safe(self, other_symbol).expect("Allocation failed during mul operation");
    self.handle = result.handle;
  }
}

// Arithmetics that return Result<Symbol, eIcicleError> instead of panicking in the case of the +,-,* operators
impl<F: SymbolTrait<F>> Symbol<F> {
  pub fn add(self, other: Symbol<F>) -> Result<Symbol<F>, eIcicleError> {
      SymbolTrait::add_safe(self, other)
  }

  pub fn sub(self, other: Symbol<F>) -> Result<Symbol<F>, eIcicleError> {
      SymbolTrait::sub_safe(self, other)
  }

  pub fn mul(self, other: Symbol<F>) -> Result<Symbol<F>, eIcicleError> {
      SymbolTrait::mul_safe(self, other)
  }
}
