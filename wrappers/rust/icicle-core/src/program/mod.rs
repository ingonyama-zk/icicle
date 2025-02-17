use icicle_runtime::errors::eIcicleError;
use crate::traits::{FieldImpl, Handle};
use crate::symbol::Symbol;

pub type Instruction = u32;

#[repr(C)]
pub enum PreDefinedProgram {
  ABminusC = 0,
  EQtimesABminusC
}

pub trait Program<F, S>: 
  Sized + Handle
where
  F:FieldImpl,
  S: Symbol<F>,
{
  fn new(program_func: impl FnOnce(&mut Vec<S>), nof_parameters: u32) -> Result<Self, eIcicleError>;
  
  fn new_predefined(pre_def: PreDefinedProgram) -> Result<Self, eIcicleError>;
}

pub trait ReturningValueProgram<F, S>:
  Sized + Handle
where
  F:FieldImpl,
  S: Symbol<F>,
{
  fn new(program_func: impl FnOnce(&mut Vec<S>) -> S, nof_parameters: u32) -> Result<Self, eIcicleError>;

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
      use icicle_core::traits::{FieldImpl, Handle, HandleCPP};
      use icicle_core::symbol::Symbol as SymbolTrait;
      use crate::symbol::$field_prefix_ident::Symbol;
      use icicle_core::program::{ Program as ProgramTrait,
                                  ReturningValueProgram as ReturningValueProgramTrait,
                                  PreDefinedProgram, Instruction};
      use icicle_runtime::errors::eIcicleError;
      use std::ops::{Add, Sub, Mul, AddAssign, SubAssign, MulAssign};
      use std::ffi::c_void;

      // Programs structs
      #[repr(C)]
      pub struct Program
      {
        m_handle: HandleCPP
      }

      #[repr(C)]
      pub struct ReturningValueProgram {
        m_handle: HandleCPP
      }

       // Program Operations
      extern "C" {
        #[link_name = concat!($field_prefix, "_create_empty_program")]
        pub(crate) fn ffi_create_empty_program() -> HandleCPP;

        #[link_name = concat!($field_prefix, "_create_predefined_program")]
        pub(crate) fn ffi_create_predefined_program(pre_def: PreDefinedProgram) -> HandleCPP;

        #[link_name = concat!($field_prefix, "_create_empty_returning_value_program")]
        pub(crate) fn ffi_create_empty_returning_value_program() -> HandleCPP;

        #[link_name = concat!($field_prefix, "_create_predefined_returning_value_program")]
        pub(crate) fn ffi_create_predefined_returning_value_program(pre_def: PreDefinedProgram) -> HandleCPP;

        #[link_name = concat!($field_prefix, "_generate_program")]
        pub(crate) fn ffi_generate_program(program: HandleCPP, parameters_ptr: *const HandleCPP, nof_parameter: u32);

        #[link_name = "delete_program"]
        pub(crate) fn ffi_delete_program(program: HandleCPP) -> eIcicleError;

        // ReturningValueProgram
        #[link_name = concat!($field_prefix, "_get_program_polynomial_degree")]
        pub(crate) fn ffi_get_program_polynomial_degree(program: HandleCPP) -> i32;
      }

      // Program trait implementation
      impl ProgramTrait<$field, Symbol> for Program {
        fn new(program_func: impl FnOnce(&mut Vec<Symbol>), nof_parameters: u32) -> Result<Self, eIcicleError>
        {
          unsafe {
            let prog_handle = ffi_create_empty_program();
            if prog_handle.is_null() {
              return Err(eIcicleError::AllocationFailed);
            }
            let mut program_parameters: Vec<Symbol> = (0..nof_parameters)
                                                      .enumerate()
                                                      .map(|(i, _)| Symbol::new_input(i as u32).unwrap())
                                                      .collect();

            // RELEASE POOL FOR SYMBOLS WITH COPY AND NO DROP! (Stat with memory leak and then set the release pool)

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
      }

      impl Handle for Program {
        fn handle(&self) -> HandleCPP { self.m_handle }
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
      impl ReturningValueProgramTrait<$field, Symbol> for ReturningValueProgram {
        fn new(
          program_func: impl FnOnce(&mut Vec<Symbol>) -> Symbol, 
          nof_parameters: u32
        ) -> Result<Self, eIcicleError>
        {
          unsafe {
            let prog_handle = ffi_create_empty_returning_value_program();
            if prog_handle.is_null() {
              return Err(eIcicleError::AllocationFailed);
            }

            let mut program_parameters: Vec<Symbol> = (0..nof_parameters)
                                                      .enumerate()
                                                      .map(|(i, _)| Symbol::new_input(i as u32).unwrap())
                                                      .collect();

            let res_symbol = program_func(&mut program_parameters);
            program_parameters.push(res_symbol);

            let handles: Vec<*const c_void> = program_parameters.iter().map(|s| s.handle()).collect();
            ffi_generate_program(prog_handle, handles.as_ptr(), program_parameters.len() as u32);

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

      impl Handle for ReturningValueProgram {
        fn handle(&self) -> HandleCPP { self.m_handle }
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
