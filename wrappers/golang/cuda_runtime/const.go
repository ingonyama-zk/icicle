package cuda_runtime

/*
#cgo LDFLAGS: -L/usr/local/cuda/lib64 -lcudart
#cgo CFLAGS: -I /usr/local/cuda/include
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdlib.h>
*/
import "C"

type CudaStreamCreateFlags C.uint

const (
	// CudaStreamDefault as defined in include/driver_types.h:98
	CudaStreamDefault 			CudaStreamCreateFlags = iota
	// CudaStreamNonBlocking as defined in include/driver_types.h:99
	CudaStreamNonBlocking 	CudaStreamCreateFlags = 1
)

type CudaStreamWaitFlags C.uint

const (
	// CudaEventWaitDefault as defined in include/driver_types.h:129
	CudaEventWaitDefault 		CudaStreamWaitFlags = iota
	// CudaEventWaitExternal as defined in include/driver_types.h:130
	CudaEventWaitExternal 	CudaStreamWaitFlags = 1
)

// CudaErrorT as declared in include/driver_types.h:2868
type CudaError int32

// CudaErrorT enumeration from include/driver_types.h:2868
const (
	CudaSuccess                             CudaError = iota
	CudaErrorInvalidValue                   CudaError = 1
	CudaErrorMemoryAllocation               CudaError = 2
	CudaErrorInitializationError            CudaError = 3
	CudaErrorCudartUnloading                CudaError = 4
	CudaErrorProfilerDisabled               CudaError = 5
	CudaErrorProfilerNotInitialized         CudaError = 6
	CudaErrorProfilerAlreadyStarted         CudaError = 7
	CudaErrorProfilerAlreadyStopped         CudaError = 8
	CudaErrorInvalidConfiguration           CudaError = 9
	CudaErrorInvalidPitchValue              CudaError = 12
	CudaErrorInvalidSymbol                  CudaError = 13
	CudaErrorInvalidHostPointer             CudaError = 16
	CudaErrorInvalidDevicePointer           CudaError = 17
	CudaErrorInvalidTexture                 CudaError = 18
	CudaErrorInvalidTextureBinding          CudaError = 19
	CudaErrorInvalidChannelDescriptor       CudaError = 20
	CudaErrorInvalidMemcpyDirection         CudaError = 21
	CudaErrorAddressOfConstant              CudaError = 22
	CudaErrorTextureFetchFailed             CudaError = 23
	CudaErrorTextureNotBound                CudaError = 24
	CudaErrorSynchronizationError           CudaError = 25
	CudaErrorInvalidFilterSetting           CudaError = 26
	CudaErrorInvalidNormSetting             CudaError = 27
	CudaErrorMixedDeviceExecution           CudaError = 28
	CudaErrorNotYetImplemented              CudaError = 31
	CudaErrorMemoryValueTooLarge            CudaError = 32
	CudaErrorStubLibrary                    CudaError = 34
	CudaErrorInsufficientDriver             CudaError = 35
	CudaErrorCallRequiresNewerDriver        CudaError = 36
	CudaErrorInvalidSurface                 CudaError = 37
	CudaErrorDuplicateVariableName          CudaError = 43
	CudaErrorDuplicateTextureName           CudaError = 44
	CudaErrorDuplicateSurfaceName           CudaError = 45
	CudaErrorDevicesUnavailable             CudaError = 46
	CudaErrorIncompatibleDriverContext      CudaError = 49
	CudaErrorMissingConfiguration           CudaError = 52
	CudaErrorPriorLaunchFailure             CudaError = 53
	CudaErrorLaunchMaxDepthExceeded         CudaError = 65
	CudaErrorLaunchFileScopedTex            CudaError = 66
	CudaErrorLaunchFileScopedSurf           CudaError = 67
	CudaErrorSyncDepthExceeded              CudaError = 68
	CudaErrorLaunchPendingCountExceeded     CudaError = 69
	CudaErrorInvalidDeviceFunction          CudaError = 98
	CudaErrorNoDevice                       CudaError = 100
	CudaErrorInvalidDevice                  CudaError = 101
	CudaErrorDeviceNotLicensed              CudaError = 102
	CudaErrorSoftwareValidityNotEstablished CudaError = 103
	CudaErrorStartupFailure                 CudaError = 127
	CudaErrorInvalidKernelImage             CudaError = 200
	CudaErrorDeviceUninitialized            CudaError = 201
	CudaErrorMapBufferObjectFailed          CudaError = 205
	CudaErrorUnmapBufferObjectFailed        CudaError = 206
	CudaErrorArrayIsMapped                  CudaError = 207
	CudaErrorAlreadyMapped                  CudaError = 208
	CudaErrorNoKernelImageForDevice         CudaError = 209
	CudaErrorAlreadyAcquired                CudaError = 210
	CudaErrorNotMapped                      CudaError = 211
	CudaErrorNotMappedAsArray               CudaError = 212
	CudaErrorNotMappedAsPointer             CudaError = 213
	CudaErrorECCUncorrectable               CudaError = 214
	CudaErrorUnsupportedLimit               CudaError = 215
	CudaErrorDeviceAlreadyInUse             CudaError = 216
	CudaErrorPeerAccessUnsupported          CudaError = 217
	CudaErrorInvalidPtx                     CudaError = 218
	CudaErrorInvalidGraphicsContext         CudaError = 219
	CudaErrorNvlinkUncorrectable            CudaError = 220
	CudaErrorJitCompilerNotFound            CudaError = 221
	CudaErrorUnsupportedPtxVersion          CudaError = 222
	CudaErrorJitCompilationDisabled         CudaError = 223
	CudaErrorUnsupportedExecAffinity        CudaError = 224
	CudaErrorUnsupportedDevSideSync         CudaError = 225
	CudaErrorInvalidSource                  CudaError = 300
	CudaErrorFileNotFound                   CudaError = 301
	CudaErrorSharedObjectSymbolNotFound     CudaError = 302
	CudaErrorSharedObjectInitFailed         CudaError = 303
	CudaErrorOperatingSystem                CudaError = 304
	CudaErrorInvalidResourceHandle          CudaError = 400
	CudaErrorIllegalState                   CudaError = 401
	CudaErrorLossyQuery                     CudaError = 402
	CudaErrorSymbolNotFound                 CudaError = 500
	CudaErrorNotReady                       CudaError = 600
	CudaErrorIllegalAddress                 CudaError = 700
	CudaErrorLaunchOutOfResources           CudaError = 701
	CudaErrorLaunchTimeout                  CudaError = 702
	CudaErrorLaunchIncompatibleTexturing    CudaError = 703
	CudaErrorPeerAccessAlreadyEnabled       CudaError = 704
	CudaErrorPeerAccessNotEnabled           CudaError = 705
	CudaErrorSetOnActiveProcess             CudaError = 708
	CudaErrorContextIsDestroyed             CudaError = 709
	CudaErrorAssert                         CudaError = 710
	CudaErrorTooManyPeers                   CudaError = 711
	CudaErrorHostMemoryAlreadyRegistered    CudaError = 712
	CudaErrorHostMemoryNotRegistered        CudaError = 713
	CudaErrorHardwareStackError             CudaError = 714
	CudaErrorIllegalInstruction             CudaError = 715
	CudaErrorMisalignedAddress              CudaError = 716
	CudaErrorInvalidAddressSpace            CudaError = 717
	CudaErrorInvalidPc                      CudaError = 718
	CudaErrorLaunchFailure                  CudaError = 719
	CudaErrorCooperativeLaunchTooLarge      CudaError = 720
	CudaErrorNotPermitted                   CudaError = 800
	CudaErrorNotSupported                   CudaError = 801
	CudaErrorSystemNotReady                 CudaError = 802
	CudaErrorSystemDriverMismatch           CudaError = 803
	CudaErrorCompatNotSupportedOnDevice     CudaError = 804
	CudaErrorMpsConnectionFailed            CudaError = 805
	CudaErrorMpsRpcFailure                  CudaError = 806
	CudaErrorMpsServerNotReady              CudaError = 807
	CudaErrorMpsMaxClientsReached           CudaError = 808
	CudaErrorMpsMaxConnectionsReached       CudaError = 809
	CudaErrorMpsClientTerminated            CudaError = 810
	CudaErrorCdpNotSupported                CudaError = 811
	CudaErrorCdpVersionMismatch             CudaError = 812
	CudaErrorStreamCaptureUnsupported       CudaError = 900
	CudaErrorStreamCaptureInvalidated       CudaError = 901
	CudaErrorStreamCaptureMerge             CudaError = 902
	CudaErrorStreamCaptureUnmatched         CudaError = 903
	CudaErrorStreamCaptureUnjoined          CudaError = 904
	CudaErrorStreamCaptureIsolation         CudaError = 905
	CudaErrorStreamCaptureImplicit          CudaError = 906
	CudaErrorCapturedEvent                  CudaError = 907
	CudaErrorStreamCaptureWrongThread       CudaError = 908
	CudaErrorTimeout                        CudaError = 909
	CudaErrorGraphExecUpdateFailure         CudaError = 910
	CudaErrorExternalDevice                 CudaError = 911
	CudaErrorInvalidClusterSize             CudaError = 912
	CudaErrorUnknown                        CudaError = 999
	CudaErrorApiFailureBase                 CudaError = 10000
)

type CudaMemcpyKind C.uint

const (
	// CudaMemcpyHostToHost as declared in include/driver_types.h:1219
	CudaMemcpyHostToHost 			CudaMemcpyKind = iota
	// CudaMemcpyHostToDevice as declared in include/driver_types.h:1220
	CudaMemcpyHostToDevice 		CudaMemcpyKind = 1
	// CudaMemcpyDeviceToHost as declared in include/driver_types.h:1221
	CudaMemcpyDeviceToHost 		CudaMemcpyKind = 2
	// CudaMemcpyDeviceToDevice as declared in include/driver_types.h:1222
	CudaMemcpyDeviceToDevice 	CudaMemcpyKind = 3
	// CudaMemcpyDefault as declared in include/driver_types.h:1223
	CudaMemcpyDefault 				CudaMemcpyKind = 4
)
