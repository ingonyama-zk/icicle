use icicle_runtime::errors::eIcicleError;
use crate::traits::{FieldImpl, Handle};
use crate::symbol::Symbol;
use std::ffi::c_void;

pub type Instruction = u32;
pub type ProgramHandle = *const c_void;

#[repr(C)]
pub enum PreDefinedProgram {
  ABminusC = 0,
  EQtimesABminusC
}

pub trait Program<F>: 
  Sized + Handle
where
  F:FieldImpl,
{
  type ProgSymbol: Symbol<F>;

  fn new(program_func: impl FnOnce(&mut Vec<Self::ProgSymbol>), nof_parameters: u32) -> Result<Self, eIcicleError>;
  
  fn new_predefined(pre_def: PreDefinedProgram) -> Result<Self, eIcicleError>;
}

pub trait ReturningValueProgram:
  Sized + Handle
{
  type Field: FieldImpl;
  type ProgSymbol: Symbol<Self::Field>;

  fn new(program_func: impl FnOnce(&mut Vec<Self::ProgSymbol>) -> Self::ProgSymbol, nof_parameters: u32) -> Result<Self, eIcicleError>;

  fn new_predefined(pre_def: PreDefinedProgram) -> Result<Self, eIcicleError>;
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
      use icicle_core::traits::{FieldImpl, Handle};
      use icicle_core::symbol::{Symbol, SymbolHandle};
      use crate::symbol::$field_prefix_ident::FieldSymbol;
      use icicle_core::program::{ Program, ReturningValueProgram, ProgramHandle, PreDefinedProgram, Instruction};
      use icicle_runtime::errors::eIcicleError;
      use std::ops::{Add, Sub, Mul, AddAssign, SubAssign, MulAssign};
      use std::ffi::c_void;

      // Programs structs
      #[repr(C)]
      pub struct FieldProgram
      {
        m_handle: ProgramHandle
      }

      #[repr(C)]
      pub struct FieldReturningValueProgram {
        m_handle: ProgramHandle
      }

       // Program Operations
      extern "C" {
        #[link_name = concat!($field_prefix, "_create_predefined_program")]
        pub(crate) fn ffi_create_predefined_program(pre_def: PreDefinedProgram) -> ProgramHandle;

        #[link_name = concat!($field_prefix, "_create_predefined_returning_value_program")]
        pub(crate) fn ffi_create_predefined_returning_value_program(pre_def: PreDefinedProgram) -> ProgramHandle;

        #[link_name = concat!($field_prefix, "_generate_program")]
        pub(crate) fn ffi_generate_program(parameters_ptr: *const SymbolHandle, nof_parameter: u32, program: *mut ProgramHandle) -> eIcicleError;

        #[link_name = concat!($field_prefix, "_generate_returning_value_program")]
        pub(crate) fn ffi_generate_returning_value_program(parameters_ptr: *const SymbolHandle, nof_parameter: u32, program: *mut ProgramHandle) -> eIcicleError;

        #[link_name = "delete_program"]
        pub(crate) fn ffi_delete_program(program: ProgramHandle) -> eIcicleError;
      }

      // Program trait implementation
      impl Program<$field> for FieldProgram {
        type ProgSymbol = FieldSymbol;

        fn new(program_func: impl FnOnce(&mut Vec<FieldSymbol>), nof_parameters: u32) -> Result<Self, eIcicleError>
        {
          let mut program_parameters: Vec<FieldSymbol> = (0..nof_parameters)
                                                      .enumerate()
                                                      .map(|(i, _)| FieldSymbol::new_input(i as u32).unwrap())
                                                      .collect();
          program_func(&mut program_parameters);
          let handles: Vec<*const c_void> = program_parameters.iter().map(|s| s.handle()).collect();
          
          let mut prog_handle = std::ptr::null();
          let ffi_status;
          unsafe {
            ffi_status = ffi_generate_program(handles.as_ptr(), program_parameters.len() as u32, &mut prog_handle);
          }
          if ffi_status != eIcicleError::Success {
            Err(ffi_status)
          } else if prog_handle.is_null() {
            Err(eIcicleError::AllocationFailed)
          } else {
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
      }

      impl Handle for FieldProgram {
        fn handle(&self) -> ProgramHandle { self.m_handle }
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
      impl ReturningValueProgram for FieldReturningValueProgram {
        type Field = $field;
        type ProgSymbol = FieldSymbol;

        fn new(
          program_func: impl FnOnce(&mut Vec<FieldSymbol>) -> FieldSymbol, 
          nof_parameters: u32
        ) -> Result<Self, eIcicleError>
        {
          let mut program_parameters: Vec<FieldSymbol> = (0..nof_parameters)
                                                      .enumerate()
                                                      .map(|(i, _)| FieldSymbol::new_input(i as u32).unwrap())
                                                      .collect();
          let res_symbol = program_func(&mut program_parameters);
          program_parameters.push(res_symbol);
          let handles: Vec<*const c_void> = program_parameters.iter().map(|s| s.handle()).collect();

          let mut prog_handle = std::ptr::null();
          let ffi_status;
          unsafe {
            ffi_status = ffi_generate_returning_value_program(handles.as_ptr(), program_parameters.len() as u32, &mut prog_handle);
          }
          if ffi_status != eIcicleError::Success {
            Err(ffi_status)
          } else if prog_handle.is_null() {
            Err(eIcicleError::AllocationFailed)
          } else {
            Ok(Self { m_handle: prog_handle })
          }
        }

        fn new_predefined(pre_def: PreDefinedProgram) -> Result<Self, eIcicleError> {
          unsafe {
            let prog_handle = ffi_create_predefined_returning_value_program(pre_def);
            if prog_handle.is_null() {
              return Err(eIcicleError::AllocationFailed);
            } else {
              Ok(Self { m_handle: prog_handle }) 
            }
          }
        }
      }

      impl Handle for FieldReturningValueProgram {
        fn handle(&self) -> ProgramHandle { self.m_handle }
      }

      impl Drop for FieldReturningValueProgram {
        fn drop(&mut self) {
          if !self.m_handle.is_null()
          {
            unsafe { ffi_delete_program(self.m_handle); }
          }
        }
      }
    }
  };
}
