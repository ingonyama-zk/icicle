use icicle_runtime::errors::eIcicleError;
use std::ffi::c_void;
use crate::traits::FieldImpl;
use crate::symbol::SymbolTrait;

pub type Handle = *const c_void;
pub type Instruction = u32;

#[repr(C)]
pub enum PreDefinedProgram {
  ABminusC = 0,
  EQtimesABminusC
}

pub trait ProgramBaseTrait<F, S>: Sized
where
  F: FieldImpl,
  S: SymbolTrait<F>
{
  fn new_predefined(pre_def: PreDefinedProgram) -> Result<Self, eIcicleError>;

  fn handle(&self) -> Handle;
}

pub trait ProgramTrait<F, S>: 
  ProgramBaseTrait<F, S> +
  Sized
where
  F:FieldImpl,
  S: SymbolTrait<F>
{
  fn new(program_func: impl FnOnce(&mut Vec<S>), nof_parameters: u32) -> Result<Self, eIcicleError>;
}

pub trait ReturningValueProgramTrait<F, S>:
  ProgramBaseTrait<F, S> +
  Sized
where
  F:FieldImpl,
  S: SymbolTrait<F>
{
  fn new(program_func: impl FnOnce(&mut Vec<S>) -> S, nof_parameters: u32) -> Result<Self, eIcicleError>;

  fn get_polynomial_degree(&self) -> i32; // COMMENT do we need this on the rust side?
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
      use crate::program::$field;
      use icicle_core::traits::FieldImpl;
      use icicle_core::symbol::SymbolTrait;
      use crate::symbol::$field_prefix_ident::Symbol;
      use icicle_core::program::{Handle, ProgramBaseTrait, ProgramTrait, ReturningValueProgramTrait, PreDefinedProgram, Instruction};
      use icicle_runtime::errors::eIcicleError;
      use std::ops::{Add, Sub, Mul, AddAssign, SubAssign, MulAssign};
      use std::ffi::c_void;

      // Programs structs
      #[repr(C)]
      pub struct Program
      {
        m_handle: Handle
      }

      #[repr(C)]
      pub struct ReturningValueProgram {
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
      impl ProgramBaseTrait<$field, Symbol> for Program {
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

      impl ProgramTrait<$field, Symbol> for Program {
        fn new(program_func: impl FnOnce(&mut Vec<Symbol>), nof_parameters: u32) -> Result<Self, eIcicleError>
        {
          unsafe {
            let prog_handle = ffi_create_empty_program();
            if prog_handle.is_null() {
              return Err(eIcicleError::AllocationFailed);
            }

            let mut program_parameters: Vec<Symbol> = (0..nof_parameters)
              .map(|_| Symbol::new_empty().unwrap())
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
      }

      impl Drop for Program {
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
      impl ProgramBaseTrait<$field, Symbol> for ReturningValueProgram {
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

      impl ReturningValueProgramTrait<$field, Symbol> for ReturningValueProgram {
        fn new(
          program_func: impl FnOnce(&mut Vec<Symbol>) -> Symbol, 
          nof_parameters: u32
        ) -> Result<Self, eIcicleError>
        {
          unsafe {
            let prog_handle = ffi_create_empty_program();
            if prog_handle.is_null() {
              return Err(eIcicleError::AllocationFailed);
            }

            let mut program_parameters: Vec<Symbol> = (0..nof_parameters)
              .map(|_| Symbol::new_empty().unwrap())
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
        
        fn get_polynomial_degree(&self) -> i32 { unsafe { ffi_get_program_polynomial_degree(self.m_handle) } }
      }

      impl Drop for ReturningValueProgram {
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
