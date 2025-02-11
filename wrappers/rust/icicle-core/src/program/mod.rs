use icicle_runtime::errors::eIcicleError;
use std::ffi::c_void;
use crate::traits::FieldImpl;
use crate::symbol::{Symbol, SymbolTrait, FieldHasSymbol};

pub type Handle = *const c_void;
pub type Instruction = u32;
pub type Program<F> = <<F as FieldImpl>::Config as FieldHasProgram<F>>::Program;

#[repr(C)]
pub enum PreDefinedProgram {
  ABminusC = 0,
  EQtimesABminusC
}

#[doc(hidden)]
pub trait ProgramTrait<F, S>: Sized
where
  F:FieldImpl,
  S: SymbolTrait<F>
{
  fn new(program_func: impl FnOnce(&mut Vec<S>), nof_parameters: u32) -> Result<Self, eIcicleError>;

  fn new_predefined(pre_def: PreDefinedProgram) -> Result<Self, eIcicleError>;

  fn handle(&self) -> Handle;
}

pub trait ReturningValueProgramTrait<F, S>: Sized
where
  F:FieldImpl,
  S: SymbolTrait<F>
{
  fn new(program_func: impl FnOnce(&mut Vec<S>) -> S, nof_parameters: u32) -> Result<Self, eIcicleError>;

  fn new_predefined(pre_def: PreDefinedProgram) -> Result<Self, eIcicleError>;

  fn handle(&self) -> Handle;

  fn get_polynomial_degree(&self) -> i32; // COMMENT do we need this on the rust side?
}

pub trait FieldHasProgram<F>
where
  F: FieldImpl,
  <F as FieldImpl>::Config: FieldHasSymbol<F>
{
  type Program: ProgramTrait<F, Symbol<F>>;
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
      use crate::program::$field;
      use icicle_core::symbol::{Symbol, SymbolTrait, FieldHasSymbol};
      use icicle_core::program::{Handle, ProgramTrait, ReturningValueProgramTrait, PreDefinedProgram, Instruction};
      use icicle_runtime::errors::eIcicleError;
      use std::ops::{Add, Sub, Mul, AddAssign, SubAssign, MulAssign};
      use std::ffi::c_void;
      use std::fmt;

      // Programs structs
      #[repr(C)]
      pub struct FieldProgram {
        m_handle: Handle
      }

      pub struct FieldReturningValueProgram {
        m_handle: Handle
      }

       // Program Operations
      extern "C" {
        #[link_name = concat!($field_prefix, "_create_empty_program")]
        pub(crate) fn ffi_create_empty_program() -> Handle;

        #[link_name = concat!($field_prefix, "_create_predefined_program")]
        pub(crate) fn ffi_create_predefined_program(pre_def: PreDefinedProgram) -> Handle;

        #[link_name = concat!($field_prefix, "_generate_program")]
        pub(crate) fn ffi_generate_program(program: Handle, parameters_ptr: *const Handle, nof_parameter: u32);

        #[link_name = "delete_program"]
        pub(crate) fn ffi_delete_program(program: Handle) -> eIcicleError;

        // ReturningValueProgram
        #[link_name = concat!($field_prefix, "_get_program_polynomial_degree")]
        pub(crate) fn ffi_get_program_polynomial_degree(program: Handle) -> i32;
      }

      // Program trait implementation
      impl ProgramTrait<$field, Symbol<$field>> for FieldProgram {
        fn new(program_func: impl FnOnce(&mut Vec<Symbol<$field>>), nof_parameters: u32) -> Result<Self, eIcicleError>
        {
          unsafe {
            let prog_handle = ffi_create_empty_program();
            if prog_handle.is_null() {
              return Err(eIcicleError::AllocationFailed);
            }

            let mut program_parameters: Vec<Symbol<$field>> = (0..nof_parameters)
              .map(|_| Symbol::<$field>::new_empty().unwrap())
              .collect();

            for (i, param) in program_parameters.iter_mut().enumerate() { // Call program set as input instead of a for loop
              param.set_as_input(i as u32);
            }

            program_func(&mut program_parameters);

            let handles: Vec<*const c_void> = program_parameters.iter().map(|s| s.handle()).collect();
            ffi_generate_program(prog_handle, handles.as_ptr(), program_parameters.len() as u32);

            Ok(Self { m_handle: prog_handle })
          }
        }

        fn new_predefined(pre_def: PreDefinedProgram) -> Result<Self, eIcicleError> {
          unsafe {
            let prog_handle = ffi_create_predefined_program(pre_def);
            if prog_handle.is_null() {
              return Err(eIcicleError::AllocationFailed);
            } else {
              Ok(Self { m_handle: prog_handle }) 
            }
          }
        }

        fn handle(&self) -> Handle { self.m_handle }
      }

      impl Drop for FieldProgram {
        fn drop(&mut self) {
          unsafe {
            if !self.m_handle.is_null()
            {
              unsafe { ffi_delete_program(self.m_handle); }
            }
          }
        }
      }

      // Returning Value Program trait implementation
      impl ReturningValueProgramTrait<$field, Symbol<$field>> for FieldReturningValueProgram {
        fn new(
          program_func: impl FnOnce(&mut Vec<Symbol<$field>>) -> Symbol<$field>, 
          nof_parameters: u32
        ) -> Result<Self, eIcicleError>
        {
          unsafe {
            let prog_handle = ffi_create_empty_program();
            if prog_handle.is_null() {
              return Err(eIcicleError::AllocationFailed);
            }

            let mut program_parameters: Vec<Symbol<$field>> = (0..nof_parameters)
              .map(|_| Symbol::<$field>::new_empty().unwrap())
              .collect();

            for (i, param) in program_parameters.iter_mut().enumerate() { // Call program set as input instead of a for loop
              param.set_as_input(i as u32);
            }

            let res_symbol = program_func(&mut program_parameters);
            program_parameters.push(res_symbol);

            let handles: Vec<*const c_void> = program_parameters.iter().map(|s| s.handle()).collect();
            ffi_generate_program(prog_handle, handles.as_ptr(), program_parameters.len() as u32);

            Ok(Self { m_handle: prog_handle })
          }
        }

        fn new_predefined(pre_def: PreDefinedProgram) -> Result<Self, eIcicleError> {
          unsafe {
            let prog_handle = ffi_create_predefined_program(pre_def);
            if prog_handle.is_null() {
              return Err(eIcicleError::AllocationFailed);
            } else {
              Ok(Self { m_handle: prog_handle }) 
            }
          }
        }

        fn handle(&self) -> Handle { self.m_handle }
        
        fn get_polynomial_degree(&self) -> i32 { unsafe { ffi_get_program_polynomial_degree(self.m_handle) } }
      }

      impl Drop for FieldReturningValueProgram {
        fn drop(&mut self) {
          unsafe {
            if !self.m_handle.is_null()
            {
              unsafe { ffi_delete_program(self.m_handle); }
            }
          }
        }
      }
    }

    use icicle_core::program::FieldHasProgram;

    impl FieldHasProgram<$field> for $field_config {
      type Program = $field_prefix_ident::FieldProgram;
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
