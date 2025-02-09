use icicle_runtime::errors::eIcicleError;
use std::ffi::c_void;
use std::ops::{Add, Sub, Mul, AddAssign, SubAssign, MulAssign};
use crate::traits::FieldImpl;

pub type Handle = *const c_void;
pub type Instruction = u32;
pub type Symbol<F> = <<F as FieldImpl>::Config as FieldHasSymbol<F>>::Symbol;
pub type Program<F> = <<F as FieldImpl>::Config as FieldHasSymbol<F>>::Program;

#[repr(C)]
pub enum ProgramOpcode { // TODO remove
  OpCopy = 0,
  OpAdd,
  OpMult,
  OpSub,
  OpInv,

  NofOperation,

  OpInput,
  OpConst,
}

#[repr(C)]
pub enum PreDefinedPrograms {
  ABminusC = 0,
  EQtimesABminusC
}

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
  fn add_field(self, other: F) -> Result<Self, eIcicleError>;
  fn sub_field(self, other: F) -> Result<Self, eIcicleError>;
  fn mul_field(self, other: F) -> Result<Self, eIcicleError>;

  fn inverse(&self) -> Self;
  // TODO add assign
}

pub trait SymbolBackendAPI<F: FieldImpl>: Sized { // TODO try to remerge them
  fn new_empty() -> Result<Self, eIcicleError>;
  fn new_constant(constant: F) -> Result<Self, eIcicleError>;
  fn copy_symbol(other: &Self) -> Result<Self, eIcicleError>;

  fn handle(&self) -> Handle;
  fn delete_handle(handle: Handle);

  fn set_as_input(&self, in_index: u32); // TODO incorporate into program 

  fn add_safe(op_a: Handle, op_b: Handle) -> Result<Handle, eIcicleError>;
  fn sub_safe(op_a: Handle, op_b: Handle) -> Result<Handle, eIcicleError>;
  fn mul_safe(op_a: Handle, op_b: Handle) -> Result<Handle, eIcicleError>;
}

#[doc(hidden)]
pub trait ProgramTrait<F, S> where F:FieldImpl, S: SymbolTrait<F> {
  fn new(program_func: impl FnOnce(&mut Vec<S>), nof_parameters: u32) -> Result<Self, eIcicleError>
    where Self: Sized;

  fn handle(&self) -> Handle;

  fn get_nof_vars(&self) -> i32; // TODO do I need it?
}

pub trait FieldHasSymbol<F: FieldImpl> { // TODO split into field and program modules
  type Symbol: SymbolTrait<F>; // Associated type for Symbol
  type Program: ProgramTrait<F, Self::Symbol>;
}

#[macro_export]
macro_rules! impl_program_field {
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

        fn add_safe(op_a: Handle, op_b: Handle) -> Result<Handle, eIcicleError> {
          unsafe {
            let handle = ffi_add_symbols(op_a, op_b);
            if handle.is_null() {
              Err(eIcicleError::AllocationFailed)
            } else {
              Ok(handle)
            }
          }
        }

        fn sub_safe(op_a: Handle, op_b: Handle) -> Result<Handle, eIcicleError> {
          unsafe {
            let handle = ffi_sub_symbols(op_a, op_b);
            if handle.is_null() {
              Err(eIcicleError::AllocationFailed)
            } else {
              Ok(handle)
            }
          }
        }

        fn mul_safe(op_a: Handle, op_b: Handle) -> Result<Handle, eIcicleError> {
          unsafe {
            let handle = ffi_multiply_symbols(op_a, op_b);
            if handle.is_null() {
              Err(eIcicleError::AllocationFailed)
            } else {
              Ok(handle)
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
      }

      impl SymbolTrait<$field> for FieldSymbol {
        fn add_field(self, other: $field) -> Result<Self, eIcicleError> {
          let other_symbol = FieldSymbol::new_constant(other)?;
          let res_handle = Self::add_safe(self.m_handle, other_symbol.m_handle)?;
          Ok(Self { m_handle: res_handle })
        }

        fn sub_field(self, other: $field) -> Result<Self, eIcicleError> {
          let other_symbol = FieldSymbol::new_constant(other)?;
          let res_handle = Self::sub_safe(self.m_handle, other_symbol.m_handle)?;
          Ok(Self { m_handle: res_handle })
        }

        fn mul_field(self, other: $field) -> Result<Self, eIcicleError> {
          let other_symbol = FieldSymbol::new_constant(other)?;
          let res_handle = Self::mul_safe(self.m_handle, other_symbol.m_handle)?;
          Ok(Self { m_handle: res_handle })
        }
      }

      // Operations overloading
      impl Add for FieldSymbol where FieldSymbol: SymbolBackendAPI<$field> {
        type Output = Self;
        fn add(self, other: Self) -> Self {
          let res_handle = Self::add_safe(self.m_handle, other.m_handle)
                                .expect("Allocation failed during add operation");
          Self { m_handle: res_handle }
        }
      }

      impl Sub for FieldSymbol where FieldSymbol: SymbolBackendAPI<$field> {
        type Output = Self;
        fn sub(self, other: Self) -> Self {
          let res_handle = Self::sub_safe(self.m_handle, other.m_handle)
                                .expect("Allocation failed during sub operation");
          Self { m_handle: res_handle }
        }
      }

      impl Mul for FieldSymbol where FieldSymbol: SymbolBackendAPI<$field> {
        type Output = Self;
        fn mul(self, other: Self) -> Self {
          let res_handle = Self::mul_safe(self.m_handle, other.m_handle)
                                .expect("Allocation failed during mul operation");
          Self { m_handle: res_handle }
        }
      }

      impl Add<$field> for FieldSymbol where FieldSymbol: SymbolBackendAPI<$field> {
        type Output = Self;
        fn add(self, other: $field) -> Self {
          self.add_field(other).expect("Allocation failed during add operation")
        }
      }

      impl Sub<$field> for FieldSymbol where FieldSymbol: SymbolBackendAPI<$field> {
        type Output = Self;
        fn sub(self, other: $field) -> Self {
          self.sub_field(other).expect("Allocation failed during sub operation")
        }
      }

      impl Mul<$field> for FieldSymbol where FieldSymbol: SymbolBackendAPI<$field> {
        type Output = Self;
        fn mul(self, other: $field) -> Self {
          self.mul_field(other).expect("Allocation failed during mul operation")
        }
      }

      impl AddAssign for FieldSymbol where FieldSymbol: SymbolBackendAPI<$field> {
        fn add_assign(&mut self, other: Self) {
          let res_handle = Self::add_safe(self.m_handle, other.m_handle)
                                .expect("Allocation failed during add operation");
          Self::delete_handle(self.m_handle);
          self.m_handle = res_handle;
        }
      }

      impl SubAssign for FieldSymbol where FieldSymbol: SymbolBackendAPI<$field> {
        fn sub_assign(&mut self, other: Self) {
          let res_handle = Self::sub_safe(self.m_handle, other.m_handle)
                                .expect("Allocation failed during sub operation");
          Self::delete_handle(self.m_handle);
          self.m_handle = res_handle;
        }
      }

      impl MulAssign for FieldSymbol where FieldSymbol: SymbolBackendAPI<$field> {
        fn mul_assign(&mut self, other: Self) {
          let res_handle = Self::mul_safe(self.m_handle, other.m_handle)
                                .expect("Allocation failed during mul operation");
          Self::delete_handle(self.m_handle);
          self.m_handle = res_handle;
        }
      }

      impl AddAssign<$field> for FieldSymbol where FieldSymbol: SymbolBackendAPI<$field> {
        fn add_assign(&mut self, other: $field) {
          let other_symbol = FieldSymbol::new_constant(other).expect("Allocation failed during add operation");
          let res_handle = Self::add_safe(self.m_handle, other_symbol.m_handle)
                                .expect("Allocation failed during add operation");
          Self::delete_handle(self.m_handle);
          self.m_handle = res_handle;
        }
      }

      impl SubAssign<$field> for FieldSymbol where FieldSymbol: SymbolBackendAPI<$field> {
        fn sub_assign(&mut self, other: $field) {
          let other_symbol = FieldSymbol::new_constant(other).expect("Allocation failed during sub operation");
          let res_handle = Self::sub_safe(self.m_handle, other_symbol.m_handle)
                                .expect("Allocation failed during sub operation");
          Self::delete_handle(self.m_handle);
          self.m_handle = res_handle;
        }
      }

      impl MulAssign<$field> for FieldSymbol where FieldSymbol: SymbolBackendAPI<$field> {
        fn mul_assign(&mut self, other: $field) {
          let other_symbol = FieldSymbol::new_constant(other).expect("Allocation failed during mul operation");
          let res_handle = Self::mul_safe(self.m_handle, other_symbol.m_handle)
                                .expect("Allocation failed during mul operation");
          Self::delete_handle(self.m_handle);
          self.m_handle = res_handle;
        }
      }

      impl<'a> Add<&'a FieldSymbol> for FieldSymbol where FieldSymbol: SymbolBackendAPI<$field> {
          type Output = Self;
          fn add(self, other: &'a Self) -> Self {
              let res_handle = Self::add_safe(self.m_handle, other.m_handle)
                  .expect("Allocation failed during add operation");
              Self { m_handle: res_handle }
          }
      }

      impl<'a> Sub<&'a FieldSymbol> for FieldSymbol where FieldSymbol: SymbolBackendAPI<$field> {
          type Output = Self;
          fn sub(self, other: &'a Self) -> Self {
              let res_handle = Self::sub_safe(self.m_handle, other.m_handle)
                  .expect("Allocation failed during sub operation");
              Self { m_handle: res_handle }
          }
      }

      impl<'a> Mul<&'a FieldSymbol> for FieldSymbol where FieldSymbol: SymbolBackendAPI<$field> {
          type Output = Self;
          fn mul(self, other: &'a Self) -> Self {
              let res_handle = Self::mul_safe(self.m_handle, other.m_handle)
                  .expect("Allocation failed during mul operation");
              Self { m_handle: res_handle }
          }
      }

      impl<'a> AddAssign<&'a FieldSymbol> for FieldSymbol where FieldSymbol: SymbolBackendAPI<$field> {
        fn add_assign(&mut self, other: &'a Self) {
          let res_handle = Self::add_safe(self.m_handle, other.m_handle)
                                .expect("Allocation failed during add operation");
          Self::delete_handle(self.m_handle);
          self.m_handle = res_handle;
        }
      }

      impl<'a> SubAssign<&'a FieldSymbol> for FieldSymbol where FieldSymbol: SymbolBackendAPI<$field> {
        fn sub_assign(&mut self, other: &'a Self) {
          let res_handle = Self::sub_safe(self.m_handle, other.m_handle)
                                .expect("Allocation failed during sub operation");
          Self::delete_handle(self.m_handle);
          self.m_handle = res_handle;
        }
      }

      impl<'a> MulAssign<&'a FieldSymbol> for FieldSymbol where FieldSymbol: SymbolBackendAPI<$field> {
        fn mul_assign(&mut self, other: &'a Self) {
          let res_handle = Self::mul_safe(self.m_handle, other.m_handle)
                                .expect("Allocation failed during mul operation");
          Self::delete_handle(self.m_handle);
          self.m_handle = res_handle;
        }
      }

      impl Add for &FieldSymbol where FieldSymbol: SymbolBackendAPI<$field> {
        type Output = FieldSymbol;
        fn add(self, other: Self) -> FieldSymbol {
            let res_handle = FieldSymbol::add_safe(self.m_handle, other.m_handle)
                .expect("Allocation failed during add operation");
            FieldSymbol { m_handle: res_handle }
        }
      }

      impl Sub for &FieldSymbol where FieldSymbol: SymbolBackendAPI<$field> {
          type Output = FieldSymbol;
          fn sub(self, other: Self) -> FieldSymbol {
              let res_handle = FieldSymbol::sub_safe(self.m_handle, other.m_handle)
                  .expect("Allocation failed during sub operation");
              FieldSymbol { m_handle: res_handle }
          }
      }

      impl Mul for &FieldSymbol where FieldSymbol: SymbolBackendAPI<$field> {
          type Output = FieldSymbol;
          fn mul(self, other: Self) -> FieldSymbol {
              let res_handle = FieldSymbol::mul_safe(self.m_handle, other.m_handle)
                  .expect("Allocation failed during mul operation");
              FieldSymbol { m_handle: res_handle }
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

      // Program
      #[repr(C)]
      pub struct FieldProgram {
        m_handle: Handle
      }
       // Program Operations
      extern "C" {
        #[link_name = concat!($field_prefix, "_program_create_empty_program")]
        pub(crate) fn ffi_create_empty_program() -> Handle;

        #[link_name = concat!($field_prefix, "_program_generate_program")]
        pub(crate) fn ffi_generate_program(program: Handle, parameters_ptr: *const Handle, nof_parameter: u32);

        #[link_name = "program_get_nof_vars"]
        pub(crate) fn ffi_program_get_nof_vars(program: Handle) -> i32;

        #[link_name = "program_print_program"]
        pub(crate) fn ffi_print_program(program_handle: Handle);

        #[link_name = "program_delete_program"]
        pub(crate) fn ffi_delete_program(program: Handle) -> eIcicleError;
      }

      // Program trait implementation
      impl ProgramTrait<$field, FieldSymbol> for FieldProgram {
        // TODO add new with predfined program
        fn new(program_func: impl FnOnce(&mut Vec<FieldSymbol>), nof_parameters: u32) 
          -> Result<Self, eIcicleError> where Self: Sized
        {
          unsafe {
            let prog_handle = ffi_create_empty_program();
            if prog_handle.is_null() {
              return Err(eIcicleError::AllocationFailed);
            }

            let mut program_parameters: Vec<FieldSymbol> = (0..nof_parameters)
              .map(|_| FieldSymbol::new_empty().unwrap())
              .collect();

            for (i, param) in program_parameters.iter_mut().enumerate() { // Call program set as input instead of a for loop
              param.set_as_input(i as u32);
            }

            program_func(&mut program_parameters);

            let handles: Vec<*const c_void> = program_parameters.iter().map(|s| s.m_handle).collect();
            ffi_generate_program(prog_handle, handles.as_ptr(), program_parameters.len() as u32);

            Ok(Self { m_handle: prog_handle })
          }
        }

        fn handle(&self) -> Handle { self.m_handle }

        fn get_nof_vars(&self) -> i32 { // TODO remove
          unsafe { ffi_program_get_nof_vars(self.m_handle) }
        }
      }

      impl fmt::Debug for FieldProgram {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
          unsafe { ffi_print_program(self.m_handle); }
          write!(f, "")
        }
      }

      impl Drop for FieldProgram {
        fn drop(&mut self) {
          unsafe {
            if !self.m_handle.is_null()
            {
              unsafe { ffi_delete_symbol(self.m_handle); }
            }
          }
        }
      }
    }

    use icicle_core::program::FieldHasSymbol;
    use crate::program::$field_prefix_ident::{FieldSymbol, FieldProgram};

    impl FieldHasSymbol<$field> for $field_config {
      type Symbol = FieldSymbol;
      type Program = FieldProgram;
    }
  };
}

#[macro_export]
macro_rules! impl_program_tests {
  {
    $field:ident
  } => {
    pub(crate) mod test_program {
      use super::*;
      use icicle_runtime::test_utilities;
      use icicle_runtime::{device::Device, runtime};
      use std::sync_once;

      fn initialize() {
        test_utilities::test_load_and_init_devices();
        test_utilities::test_set_main_device();
      }

      
    }
  }
}
