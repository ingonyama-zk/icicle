package runtime

// eIcicleError represents the error codes
type EIcicleError int

const (
	Success                 EIcicleError = iota // Operation completed successfully
	InvalidDevice                               // The specified device is invalid
	OutOfMemory                                 // Memory allocation failed due to insufficient memory
	InvalidPointer                              // The specified pointer is invalid
	AllocationFailed                            // Memory allocation failed
	DeallocationFailed                          // Memory deallocation failed
	CopyFailed                                  // Data copy operation failed
	SynchronizationFailed                       // Device synchronization failed
	StreamCreationFailed                        // Stream creation failed
	StreamDestructionFailed                     // Stream destruction failed
	ApiNotImplemented                           // The API is not implemented for a device
	InvalidArgument                             // Invalid argument passed
	BackendLoadFailed                           // Failed to load the backend
	LicenseCheckError                           // Failed to check license or invalid license
	UnknownError                                // An unknown error occurred
)

func (e EIcicleError) AsString() string {
	switch e {
	case Success:
		return "EIcicleError.Success"
	case InvalidDevice:
		return "EIcicleError.InvalidDevice"
	case OutOfMemory:
		return "EIcicleError.OutOfMemory"
	case InvalidPointer:
		return "EIcicleError.InvalidPointer"
	case AllocationFailed:
		return "EIcicleError.AllocationFailed"
	case DeallocationFailed:
		return "EIcicleError.DeallocationFailed"
	case CopyFailed:
		return "EIcicleError.CopyFailed"
	case SynchronizationFailed:
		return "EIcicleError.SynchronizationFailed"
	case StreamCreationFailed:
		return "EIcicleError.StreamCreationFailed"
	case StreamDestructionFailed:
		return "EIcicleError.StreamDestructionFailed"
	case ApiNotImplemented:
		return "EIcicleError.ApiNotImplemented"
	case InvalidArgument:
		return "EIcicleError.InvalidArgument"
	case BackendLoadFailed:
		return "EIcicleError.BackendLoadFailed"
	case LicenseCheckError:
		return "EIcicleError.LicenseCheckError"
	case UnknownError:
	default:
	}

	return "EIcicleError.UnknownError"
}
