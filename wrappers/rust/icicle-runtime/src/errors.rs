#[repr(C)]
#[derive(Debug, PartialEq)]
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
