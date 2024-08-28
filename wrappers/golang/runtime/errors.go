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
