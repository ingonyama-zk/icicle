use crate::symbol::Symbol;
use crate::traits::{FieldImpl, Handle};
use icicle_runtime::errors::eIcicleError;
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
    $field_config:ident
  ) => {
        pub mod $field_prefix_ident {
            use crate::program::$field;
            use crate::symbol::$field_prefix_ident::FieldSymbol;
            use icicle_core::program::{Instruction, PreDefinedProgram, Program, ProgramHandle, ReturningValueProgram};
            use icicle_core::symbol::{Symbol, SymbolHandle};
            use icicle_core::traits::{FieldImpl, Handle};
            use icicle_runtime::errors::eIcicleError;
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

fn check_execute_program<F, Data>(data: &Vec<&Data>, cfg: &crate::vec_ops::VecOpsConfig) -> crate::vec_ops::VecOpsConfig
where
    F: crate::traits::FieldImpl,
    <F as crate::traits::FieldImpl>::Config: crate::vec_ops::VecOps<F>,
    Data: icicle_runtime::memory::HostOrDeviceSlice<F> + ?Sized,
{
    let nof_iterations = data[0].len();
    let is_on_device = data[0].is_on_device();
    for i in 1..data.len() {
        if data[i].len() != nof_iterations {
            panic!(
                "First parameter length ({}) and parameter[{}] length do not match",
                nof_iterations,
                data[i].len()
            );
        }
        if data[i].is_on_device() != is_on_device {
            panic!(
                "First parameter length ({}) and parameter[{}] length ({}) do not match",
                nof_iterations,
                i,
                data[i].len()
            );
        }
    }
    crate::vec_ops::VecOpsConfig {
        ..cfg.clone()
    }
}

pub fn execute_program<F, Prog, Data>(
    data: &mut Vec<&Data>,
    program: &Prog,
    cfg: &crate::vec_ops::VecOpsConfig,
) -> Result<(), icicle_runtime::errors::eIcicleError>
where
    F: crate::traits::FieldImpl,
    <F as crate::traits::FieldImpl>::Config: crate::vec_ops::VecOps<F>,
    Data: icicle_runtime::memory::HostOrDeviceSlice<F> + ?Sized,
    Prog: Program<F>,
{
    let cfg = check_execute_program::<F, Data>(&data, cfg);
    <<F as crate::traits::FieldImpl>::Config as crate::vec_ops::VecOps<F>>::execute_program(data, program, &cfg)
}

// FFI for executing a program
extern "C" {
    #[link_name = "execute_program_ffi"]
    pub(crate) fn execute_program_ffi(
        data_ptr: *const *const std::ffi::c_void,
        nof_params: u64,
        program: *const std::ffi::c_void,
        nof_iterations: u64,
        cfg: *const crate::vec_ops::VecOpsConfig,
    ) -> icicle_runtime::errors::eIcicleError;
}

#[macro_export]
macro_rules! execute_program_ffi {
    ($($arg:tt)*) => { execute_program_ffi($($arg)*) };
}
