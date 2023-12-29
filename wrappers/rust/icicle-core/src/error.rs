use std::mem::MaybeUninit;

use icicle_cuda_runtime::error::CudaError;

use crate::traits::ResultWrap;
use crate::traits::IcicleResultWrap;

#[repr(u32)]
#[must_use]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum IcicleErrorCode {
    IcicleSuccess = 0,
    InvalidArgument = 1,
    MemoryAllocationError = 2,
    InternalCudaError = 199999999,
    UndefinedError = 999999999, // Assigning 0 as the value for UndefinedError
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct IcicleError {
    icicle_error_code: IcicleErrorCode,
    cuda_error: Option<CudaError>,
    reason: Option< &'static str>,
}

pub type IcicleResult<T> = Result<T, IcicleError>;
pub type OptionCudaError = Option<CudaError>;


impl IcicleError {
    pub fn from_cuda_error(cuda_error: CudaError) -> Self {
        let icicle_error_code = match cuda_error {
            CudaError::cudaSuccess => IcicleErrorCode::IcicleSuccess,
            _ => IcicleErrorCode::InternalCudaError,
        };

        IcicleError {
            icicle_error_code,
            cuda_error: Some(cuda_error),
            reason: Some("Runtime CUDA error."),
        }
    }

    pub fn from_code_and_reason(icicle_error_code: IcicleErrorCode, reason: &'static str) -> Self {
        IcicleError {
            icicle_error_code,
            reason: Some(reason),
            cuda_error: None,
        }
    }
    
    pub fn get_icicle_error_code(&self) -> IcicleErrorCode {
        self.icicle_error_code
    }

    pub fn get_cuda_error(&self) -> Option<CudaError> {
        self.cuda_error
    }
}

impl IcicleResultWrap for CudaError {
    fn wrap(self) -> IcicleResult<()> {
        self.wrap_value(())
    }

    fn wrap_value<T>(self, value: T) -> IcicleResult<T> {
        if self == CudaError::cudaSuccess {
            Ok(value)
        } else {
            Err(IcicleError::from_cuda_error(self))
        }
    }

    fn wrap_maybe_uninit<T>(self, value: MaybeUninit<T>) -> IcicleResult<T> {
        self.wrap_value(unsafe { value.assume_init() })
    }
}
