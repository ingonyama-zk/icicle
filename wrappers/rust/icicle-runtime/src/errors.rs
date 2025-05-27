use std::fmt::{self, Display};

#[repr(C)]
#[derive(Debug, PartialEq)]
#[allow(non_camel_case_types)]
pub enum eIcicleError {
    Success = 0,             // Operation completed successfully
    InvalidDevice,           // The specified device is invalid
    OutOfMemory,             // Memory allocation failed due to insufficient memory
    InvalidPointer,          // The specified pointer is invalid
    AllocationFailed,        // Memory allocation failed
    DeallocationFailed,      // Memory deallocation failed
    CopyFailed,              // Data copy operation failed
    SynchronizationFailed,   // Device synchronization failed
    StreamCreationFailed,    // Stream creation failed
    StreamDestructionFailed, // Stream destruction failed
    ApiNotImplemented,       // The API is not implemented for a device
    InvalidArgument,         // Invalid argument passed
    UnknownError,            // An unknown error occurred
}

impl eIcicleError {
    pub fn wrap(self) -> Result<(), eIcicleError> {
        match self {
            eIcicleError::Success => Ok(()),
            _ => Err(self),
        }
    }

    pub fn wrap_value<T>(self, val: T) -> Result<T, eIcicleError> {
        match self {
            eIcicleError::Success => Ok(val),
            _ => Err(self),
        }
    }
}

impl Display for eIcicleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            eIcicleError::Success => write!(f, "eIcicleError::SUCCESS"),
            eIcicleError::InvalidDevice => write!(f, "eIcicleError::INVALID_DEVICE"),
            eIcicleError::OutOfMemory => write!(f, "eIcicleError::OUT_OF_MEMORY"),
            eIcicleError::InvalidPointer => write!(f, "eIcicleError::INVALID_POINTER"),
            eIcicleError::AllocationFailed => write!(f, "eIcicleError::ALLOCATION_FAILED"),
            eIcicleError::DeallocationFailed => write!(f, "eIcicleError::DEALLOCATION_FAILED"),
            eIcicleError::CopyFailed => write!(f, "eIcicleError::COPY_FAILED"),
            eIcicleError::SynchronizationFailed => write!(f, "eIcicleError::SYNCHRONIZATION_FAILED"),
            eIcicleError::StreamCreationFailed => write!(f, "eIcicleError::STREAM_CREATION_FAILED"),
            eIcicleError::StreamDestructionFailed => write!(f, "eIcicleError::STREAM_DESTRUCTION_FAILED"),
            eIcicleError::ApiNotImplemented => write!(f, "eIcicleError::API_NOT_IMPLEMENTED"),
            eIcicleError::InvalidArgument => write!(f, "eIcicleError::INVALID_ARGUMENT"),
            eIcicleError::UnknownError => write!(f, "eIcicleError::UNKNOWN_ERROR"),
        }
    }
}
