use crate::symbol::Symbol;
use crate::traits::Handle;
use crate::vec_ops::VecOpsConfig;
use icicle_runtime::{errors::eIcicleError, memory::HostOrDeviceSlice};
use std::ffi::c_void;

pub type Instruction = u32;
pub type ProgramHandle = *const c_void;

#[repr(C)]
pub enum PreDefinedProgram {
    ABminusC = 0,
    EQtimesABminusC,
}

pub trait Program<F>: Sized + Handle
where
    F: FieldImpl,
{
    type ProgSymbol: Symbol<F>;

    fn new(program_func: impl FnOnce(&mut Vec<Self::ProgSymbol>), nof_parameters: u32) -> Result<Self, eIcicleError>;

    fn new_predefined(pre_def: PreDefinedProgram) -> Result<Self, eIcicleError>;

    fn execute_program<Data>(&self, data: &mut Vec<&Data>, cfg: &VecOpsConfig) -> Result<(), eIcicleError>
    where
        F: FieldImpl,
        Data: HostOrDeviceSlice<F> + ?Sized;
}

pub trait ReturningValueProgram: Sized + Handle {
    type Field: FieldImpl;
    type ProgSymbol: Symbol<Self::Field>;

    fn new(
        program_func: impl FnOnce(&mut Vec<Self::ProgSymbol>) -> Self::ProgSymbol,
        nof_parameters: u32,
    ) -> Result<Self, eIcicleError>;

    fn new_predefined(pre_def: PreDefinedProgram) -> Result<Self, eIcicleError>;
}

#[macro_export]
macro_rules! impl_program_field {
    (
    $field_prefix:literal,
    $field_prefix_ident:ident,
    $field:ident,
  ) => {
        pub mod $field_prefix_ident {
            use crate::program::$field;
            use crate::symbol::$field_prefix_ident::FieldSymbol;
            use icicle_core::program::{Instruction, PreDefinedProgram, Program, ProgramHandle, ReturningValueProgram};
            use icicle_core::symbol::{Symbol, SymbolHandle};
            use icicle_core::traits::{FieldImpl, Handle};
            use icicle_core::vec_ops::{execute_program_ffi, VecOpsConfig};
            use icicle_runtime::errors::eIcicleError;
            use icicle_runtime::memory::HostOrDeviceSlice;
            use std::ffi::c_void;
            use std::ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign};

            // Programs structs
            #[repr(C)]
            pub struct FieldProgram {
                m_handle: ProgramHandle,
            }

            #[repr(C)]
            pub struct FieldReturningValueProgram {
                m_handle: ProgramHandle,
            }

            // Program Operations
            extern "C" {
                #[link_name = concat!($field_prefix, "_create_predefined_program")]
                pub(crate) fn ffi_create_predefined_program(pre_def: PreDefinedProgram) -> ProgramHandle;

                #[link_name = concat!($field_prefix, "_create_predefined_returning_value_program")]
                pub(crate) fn ffi_create_predefined_returning_value_program(
                    pre_def: PreDefinedProgram,
                ) -> ProgramHandle;

                #[link_name = concat!($field_prefix, "_generate_program")]
                pub(crate) fn ffi_generate_program(
                    parameters_ptr: *const SymbolHandle,
                    nof_parameter: u32,
                    program: *mut ProgramHandle,
                ) -> eIcicleError;

                #[link_name = concat!($field_prefix, "_generate_returning_value_program")]
                pub(crate) fn ffi_generate_returning_value_program(
                    parameters_ptr: *const SymbolHandle,
                    nof_parameter: u32,
                    program: *mut ProgramHandle,
                ) -> eIcicleError;

                #[link_name = "delete_program"]
                pub(crate) fn ffi_delete_program(program: ProgramHandle) -> eIcicleError;

                #[link_name = concat!($field_prefix, "_execute_program")]
                pub(crate) fn execute_program_ffi(
                    data_ptr: *const *const $field,
                    nof_params: u64,
                    program: ProgramHandle,
                    nof_iterations: u64,
                    cfg: *const VecOpsConfig,
                ) -> eIcicleError;
            }

            // Program trait implementation
            impl Program<$field> for FieldProgram {
                type ProgSymbol = FieldSymbol;

                fn new(
                    program_func: impl FnOnce(&mut Vec<FieldSymbol>),
                    nof_parameters: u32,
                ) -> Result<Self, eIcicleError> {
                    let mut program_parameters: Vec<FieldSymbol> = (0..nof_parameters)
                        .enumerate()
                        .map(|(i, _)| FieldSymbol::new_input(i as u32).unwrap())
                        .collect();
                    program_func(&mut program_parameters);
                    let handles: Vec<*const c_void> = program_parameters
                        .iter()
                        .map(|s| s.handle())
                        .collect();

                    let mut prog_handle = std::ptr::null();
                    let ffi_status;
                    unsafe {
                        ffi_status = ffi_generate_program(
                            handles.as_ptr(),
                            program_parameters.len() as u32,
                            &mut prog_handle,
                        );
                    }
                    if ffi_status != eIcicleError::Success {
                        Err(ffi_status)
                    } else if prog_handle.is_null() {
                        Err(eIcicleError::AllocationFailed)
                    } else {
                        Ok(Self {
                            m_handle: prog_handle,
                        })
                    }
                }

                fn new_predefined(pre_def: PreDefinedProgram) -> Result<Self, eIcicleError> {
                    unsafe {
                        let prog_handle = ffi_create_predefined_program(pre_def);
                        if prog_handle.is_null() {
                            return Err(eIcicleError::AllocationFailed);
                        } else {
                            Ok(Self {
                                m_handle: prog_handle,
                            })
                        }
                    }
                }

                fn execute_program<Data>(&self, data: &mut Vec<&Data>, cfg: &VecOpsConfig) -> Result<(), eIcicleError>
                where
                    $field: FieldImpl,
                    Data: HostOrDeviceSlice<$field> + ?Sized,
                {
                    unsafe {
                        let data_vec: Vec<*const $field> = data
                            .iter()
                            .map(|s| s.as_ptr())
                            .collect();
                        execute_program_ffi(
                            data_vec.as_ptr(),
                            data.len() as u64,
                            self.handle(),
                            data[0].len() as u64,
                            cfg as *const VecOpsConfig,
                        )
                        .wrap()
                    }
                }
            }

            impl Handle for FieldProgram {
                fn handle(&self) -> ProgramHandle {
                    self.m_handle
                }
            }

            impl Drop for FieldProgram {
                fn drop(&mut self) {
                    unsafe {
                        if !self
                            .m_handle
                            .is_null()
                        {
                            unsafe {
                                ffi_delete_program(self.m_handle);
                            }
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
                    nof_parameters: u32,
                ) -> Result<Self, eIcicleError> {
                    let mut program_parameters: Vec<FieldSymbol> = (0..nof_parameters)
                        .enumerate()
                        .map(|(i, _)| FieldSymbol::new_input(i as u32).unwrap())
                        .collect();
                    let res_symbol = program_func(&mut program_parameters);
                    program_parameters.push(res_symbol);
                    let handles: Vec<*const c_void> = program_parameters
                        .iter()
                        .map(|s| s.handle())
                        .collect();

                    let mut prog_handle = std::ptr::null();
                    let ffi_status;
                    unsafe {
                        ffi_status = ffi_generate_returning_value_program(
                            handles.as_ptr(),
                            program_parameters.len() as u32,
                            &mut prog_handle,
                        );
                    }
                    if ffi_status != eIcicleError::Success {
                        Err(ffi_status)
                    } else if prog_handle.is_null() {
                        Err(eIcicleError::AllocationFailed)
                    } else {
                        Ok(Self {
                            m_handle: prog_handle,
                        })
                    }
                }

                fn new_predefined(pre_def: PreDefinedProgram) -> Result<Self, eIcicleError> {
                    unsafe {
                        let prog_handle = ffi_create_predefined_returning_value_program(pre_def);
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
                fn handle(&self) -> ProgramHandle {
                    self.m_handle
                }
            }

            impl Drop for FieldReturningValueProgram {
                fn drop(&mut self) {
                    if !self
                        .m_handle
                        .is_null()
                    {
                        unsafe {
                            ffi_delete_program(self.m_handle);
                        }
                    }
                }
            }
        }
    };
}

#[macro_export]
macro_rules! impl_program_tests {
    (
      $field_prefix_ident: ident,
      $field:ident
    ) => {
        pub(crate) mod test_program {
            use super::*;
            use crate::program::$field_prefix_ident::{FieldProgram, FieldReturningValueProgram};
            use icicle_runtime::test_utilities;
            use icicle_runtime::{device::Device, runtime};
            use std::sync::Once;

            fn initialize() {
                test_utilities::test_load_and_init_devices();
                test_utilities::test_set_main_device();
            }

            #[test]
            pub fn test_program() {
                initialize();
                test_utilities::test_set_main_device();
                icicle_core::program::tests::check_program::<$field, FieldProgram>();
                test_utilities::test_set_ref_device();
                icicle_core::program::tests::check_program::<$field, FieldProgram>()
            }

            #[test]
            pub fn test_predefined_program() {
                initialize();
                test_utilities::test_set_main_device();
                icicle_core::program::tests::check_predefined_program::<$field, FieldProgram>();
                test_utilities::test_set_ref_device();
                icicle_core::program::tests::check_predefined_program::<$field, FieldProgram>()
            }
        }
    };
}

pub mod tests;