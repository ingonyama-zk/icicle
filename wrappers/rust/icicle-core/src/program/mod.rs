use icicle_runtime::errors::eIcicleError;
use std::ffi::c_void;
use crate::traits::FieldImpl;
use crate::symbol::{Symbol, SymbolTrait, FieldHasSymbol};

pub type Handle = *const c_void;
pub type Instruction = u32;
pub type Program<F> = <<F as FieldImpl>::Config as FieldHasProgram<F>>::Program;

#[repr(C)]
pub enum PreDefinedPrograms {
  ABminusC = 0,
  EQtimesABminusC
}

#[doc(hidden)]
pub trait ProgramTrait<F, S> where F:FieldImpl, S: SymbolTrait<F> {
  fn new(program_func: impl FnOnce(&mut Vec<S>), nof_parameters: u32) -> Result<Self, eIcicleError>
    where Self: Sized;

  fn handle(&self) -> Handle;
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
      use crate::symbol::{Symbol, SymbolTrait, FieldHasSymbol};
      use crate::program::{$field, HostOrDeviceSlice};
      use icicle_core::program::{Handle, SymbolTrait, SymbolBackendAPI, ProgramTrait, Instruction};
      use icicle_runtime::errors::eIcicleError;
      use std::ops::{Add, Sub, Mul, AddAssign, SubAssign, MulAssign};
      use std::ffi::c_void;
      use std::fmt;

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

        #[link_name = "program_delete_program"]
        pub(crate) fn ffi_delete_program(program: Handle) -> eIcicleError;
      }

      // Program trait implementation
      impl ProgramTrait<$field, Symbol<$field>> for FieldProgram {
        // TODO add new with predfined program
        fn new(program_func: impl FnOnce(&mut Vec<Symbol<$field>>), nof_parameters: u32) 
          -> Result<Self, eIcicleError> where Self: Sized
        {
          unsafe {
            let prog_handle = ffi_create_empty_program();
            if prog_handle.is_null() {
              return Err(eIcicleError::AllocationFailed);
            }

            let mut program_parameters: Vec<Symbol<$field>> = (0..nof_parameters)
              .map(|_| Symbol<$field>::new_empty().unwrap())
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

    use icicle_core::program::FieldHasProgram;

    impl FieldHasProgram<$field> for $field_config {
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
