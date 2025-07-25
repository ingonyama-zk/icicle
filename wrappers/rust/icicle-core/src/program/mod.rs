use crate::ring::IntegerRing;
use crate::symbol::Symbol;
use crate::traits::Handle;
use crate::vec_ops::VecOpsConfig;
use icicle_runtime::{errors::IcicleError, memory::HostOrDeviceSlice};
use std::ffi::c_void;

pub type Instruction = u32;
pub type ProgramHandle = *const c_void;

#[repr(C)]
pub enum PreDefinedProgram {
    ABminusC = 0,
    EQtimesABminusC,
}

pub trait ProgramImpl<T>: Sized + Handle
where
    T: IntegerRing,
{
    type ProgSymbol: Symbol<T>;

    fn new(program_func: impl FnOnce(&mut Vec<Self::ProgSymbol>), nof_parameters: u32) -> Result<Self, IcicleError>;

    fn new_predefined(pre_def: PreDefinedProgram) -> Result<Self, IcicleError>;

    fn execute_program<Data>(&self, data: &mut Vec<&Data>, cfg: &VecOpsConfig) -> Result<(), IcicleError>
    where
        T: IntegerRing,
        Data: HostOrDeviceSlice<T> + ?Sized;
}

pub trait ReturningValueProgramImpl: Sized + Handle {
    type Ring: IntegerRing;
    type ProgSymbol: Symbol<Self::Ring>;

    fn new(
        program_func: impl FnOnce(&mut Vec<Self::ProgSymbol>) -> Self::ProgSymbol,
        nof_parameters: u32,
    ) -> Result<Self, IcicleError>;

    fn new_predefined(pre_def: PreDefinedProgram) -> Result<Self, IcicleError>;
}

#[macro_export]
#[allow(clippy::crate_in_macro_def)]
macro_rules! impl_program_ring {
    (
    $ring_prefix:literal,
    $ring_prefix_ident:ident,
    $ring:ident
  ) => {
        pub mod $ring_prefix_ident {
            use crate::program::$ring;
            use crate::symbol::$ring_prefix_ident::RingSymbol;
            use icicle_core::program::{
                Instruction, PreDefinedProgram, ProgramHandle, ProgramImpl, ReturningValueProgramImpl,
            };
            use icicle_core::ring::IntegerRing;
            use icicle_core::symbol::{Symbol, SymbolHandle};
            use icicle_core::traits::Handle;
            use icicle_core::vec_ops::VecOpsConfig;
            use icicle_runtime::errors::{eIcicleError, IcicleError};
            use icicle_runtime::memory::HostOrDeviceSlice;
            use std::ffi::c_void;
            use std::ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign};

            // Programs structs
            #[repr(C)]
            pub struct Program {
                m_handle: ProgramHandle,
            }

            #[repr(C)]
            pub struct ReturningValueProgram {
                m_handle: ProgramHandle,
            }

            // Program Operations
            extern "C" {
                #[link_name = concat!($ring_prefix, "_create_predefined_program")]
                pub(crate) fn ffi_create_predefined_program(pre_def: PreDefinedProgram) -> ProgramHandle;

                #[link_name = concat!($ring_prefix, "_create_predefined_returning_value_program")]
                pub(crate) fn ffi_create_predefined_returning_value_program(
                    pre_def: PreDefinedProgram,
                ) -> ProgramHandle;

                #[link_name = concat!($ring_prefix, "_generate_program")]
                pub(crate) fn ffi_generate_program(
                    parameters_ptr: *const SymbolHandle,
                    nof_parameter: u32,
                    program: *mut ProgramHandle,
                ) -> eIcicleError;

                #[link_name = concat!($ring_prefix, "_generate_returning_value_program")]
                pub(crate) fn ffi_generate_returning_value_program(
                    parameters_ptr: *const SymbolHandle,
                    nof_parameter: u32,
                    program: *mut ProgramHandle,
                ) -> eIcicleError;

                #[link_name = "delete_program"]
                pub(crate) fn ffi_delete_program(program: ProgramHandle) -> eIcicleError;

                #[link_name = concat!($ring_prefix, "_execute_program")]
                pub(crate) fn execute_program_ffi(
                    data_ptr: *const *const $ring,
                    nof_params: u64,
                    program: ProgramHandle,
                    nof_iterations: u64,
                    cfg: *const VecOpsConfig,
                ) -> eIcicleError;
            }

            // Program trait implementation
            impl ProgramImpl<$ring> for Program {
                type ProgSymbol = RingSymbol;

                fn new(
                    program_func: impl FnOnce(&mut Vec<RingSymbol>),
                    nof_parameters: u32,
                ) -> Result<Self, IcicleError> {
                    let mut program_parameters: Vec<RingSymbol> = (0..nof_parameters)
                        .enumerate()
                        .map(|(i, _)| RingSymbol::new_input(i as u32).unwrap())
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
                        )
                        .wrap()?;
                    }
                    if prog_handle.is_null() {
                        Err(IcicleError::new(
                            eIcicleError::AllocationFailed,
                            "program handle is null",
                        ))
                    } else {
                        Ok(Self {
                            m_handle: prog_handle,
                        })
                    }
                }

                fn new_predefined(pre_def: PreDefinedProgram) -> Result<Self, IcicleError> {
                    unsafe {
                        let prog_handle = ffi_create_predefined_program(pre_def);
                        if prog_handle.is_null() {
                            return Err(IcicleError::new(
                                eIcicleError::AllocationFailed,
                                "program handle is null",
                            ));
                        } else {
                            Ok(Self {
                                m_handle: prog_handle,
                            })
                        }
                    }
                }

                fn execute_program<Data>(&self, data: &mut Vec<&Data>, cfg: &VecOpsConfig) -> Result<(), IcicleError>
                where
                    $ring: IntegerRing,
                    Data: HostOrDeviceSlice<$ring> + ?Sized,
                {
                    unsafe {
                        let data_vec: Vec<*const $ring> = data
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

            impl Handle for Program {
                fn handle(&self) -> ProgramHandle {
                    self.m_handle
                }
            }

            impl Drop for Program {
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
            impl ReturningValueProgramImpl for ReturningValueProgram {
                type Ring = $ring;
                type ProgSymbol = RingSymbol;

                fn new(
                    program_func: impl FnOnce(&mut Vec<RingSymbol>) -> RingSymbol,
                    nof_parameters: u32,
                ) -> Result<Self, IcicleError> {
                    let mut program_parameters: Vec<RingSymbol> = (0..nof_parameters)
                        .enumerate()
                        .map(|(i, _)| RingSymbol::new_input(i as u32).unwrap())
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
                        )
                        .wrap()?;
                    }
                    if prog_handle.is_null() {
                        Err(IcicleError::new(
                            eIcicleError::AllocationFailed,
                            "program handle is null",
                        ))
                    } else {
                        Ok(Self {
                            m_handle: prog_handle,
                        })
                    }
                }

                fn new_predefined(pre_def: PreDefinedProgram) -> Result<Self, IcicleError> {
                    unsafe {
                        let prog_handle = ffi_create_predefined_returning_value_program(pre_def);
                        if prog_handle.is_null() {
                            return Err(IcicleError::new(
                                eIcicleError::AllocationFailed,
                                "program handle is null",
                            ));
                        } else {
                            Ok(Self {
                                m_handle: prog_handle,
                            })
                        }
                    }
                }
            }

            impl Handle for ReturningValueProgram {
                fn handle(&self) -> ProgramHandle {
                    self.m_handle
                }
            }

            impl Drop for ReturningValueProgram {
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
      $ring_prefix_ident: ident,
      $ring:ident
    ) => {
        pub(crate) mod test_program {
            use super::*;
            use crate::program::$ring_prefix_ident::{Program, ReturningValueProgram};
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
                icicle_core::program::tests::check_program::<$ring, Program>();
                test_utilities::test_set_ref_device();
                icicle_core::program::tests::check_program::<$ring, Program>()
            }

            #[test]
            pub fn test_predefined_program() {
                initialize();
                test_utilities::test_set_main_device();
                icicle_core::program::tests::check_predefined_program::<$ring, Program>();
                test_utilities::test_set_ref_device();
                icicle_core::program::tests::check_predefined_program::<$ring, Program>()
            }
        }
    };
}

pub mod tests;
