use icicle_runtime::eIcicleError;
use crate::traits::Handle;
use std::ffi::c_void;

#[repr(C)]
pub enum PreDefinedProgram {
    ABminusC = 0,
    EQtimesABminusC,
}

pub type ReturningValueProgramHandle = *const c_void;

pub trait ReturningValueProgram: Sized {
    fn new_predefined(pre_def: PreDefinedProgram) -> Result<Self, eIcicleError>;
}

/// Macro to implement Sumcheck functionality for a specific field.
#[macro_export]
macro_rules! impl_program {
  ($field_prefix:literal) => {
    use icicle_core::program::{
      ReturningValueProgramHandle, PreDefinedProgram, ReturningValueProgram,
    };
    use icicle_core::traits::Handle;
    use icicle_runtime::eIcicleError;
    use std::ffi::c_void;

    extern "C" {
      #[link_name = concat!($field_prefix, "_create_predefined_returning_value_program")]
      fn icicle_create_predefined_returning_value_program(pre_def: PreDefinedProgram) -> ReturningValueProgramHandle;
    }

    pub struct FieldReturningValueProgram {
        m_handle: ReturningValueProgramHandle,
    }

    // Returning Value Program trait implementation
    impl ReturningValueProgram for FieldReturningValueProgram {
        fn new_predefined(pre_def: PreDefinedProgram) -> Result<Self, eIcicleError> {
            unsafe {
                let prog_handle = icicle_create_predefined_returning_value_program(pre_def);
                if prog_handle.is_null() {
                    return Err(eIcicleError::AllocationFailed);
                } else {
                    Ok(Self {
                        m_handle: prog_handle,
                    })
                }
            }
        }
    }

    impl Handle for FieldReturningValueProgram {
      fn handle(&self) -> ReturningValueProgramHandle {
          self.m_handle
      }
    }
  }
}