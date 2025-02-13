use icicle_runtime::errors::eIcicleError;
use std::ffi::c_void;
use crate::traits::FieldImpl;

pub type Handle = *const c_void;
pub type Instruction = u32;

#[repr(C)]
pub enum PreDefinedProgram {
  ABminusC = 0,
  EQtimesABminusC
}

pub trait ProgramBaseTrait<F>: Sized
where
  F: FieldImpl,
{
  fn new_predefined(pre_def: PreDefinedProgram) -> Result<Self, eIcicleError>;
  fn handle(&self) -> Handle;
}

// COMMENT is there even a difference between returning and non returning value when it is predefined?
pub trait ProgramTrait<F>: 
  ProgramBaseTrait<F> +
  Sized
where
  F:FieldImpl
{}

pub trait ReturningValueProgramTrait<F>:
  ProgramBaseTrait<F> +
  Sized
where
  F:FieldImpl,
{}

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
        #[link_name = concat!($field_prefix, "_create_predefined_program")]
        pub(crate) fn ffi_create_predefined_program(pre_def: PreDefinedProgram) -> Handle;

        #[link_name = "delete_program"]
        pub(crate) fn ffi_delete_program(program: Handle) -> eIcicleError;
      }

      // Program trait implementation
      impl ProgramBaseTrait<$field> for Program {
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

      impl ProgramTrait<$field> for Program {}

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
      impl ProgramBaseTrait<$field> for ReturningValueProgram {
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

      impl ReturningValueProgramTrait<$field> for ReturningValueProgram {}

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
