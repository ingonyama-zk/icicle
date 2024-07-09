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
	CudaStreamDefault CudaStreamCreateFlags = iota
	// CudaStreamNonBlocking as defined in include/driver_types.h:99
	CudaStreamNonBlocking CudaStreamCreateFlags = 1
)

type CudaStreamWaitFlags C.uint

const (
	// CudaEventWaitDefault as defined in include/driver_types.h:129
	CudaEventWaitDefault CudaStreamWaitFlags = iota
	// CudaEventWaitExternal as defined in include/driver_types.h:130
	CudaEventWaitExternal CudaStreamWaitFlags = 1
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
	CudaMemcpyHostToHost CudaMemcpyKind = iota
	// CudaMemcpyHostToDevice as declared in include/driver_types.h:1220
	CudaMemcpyHostToDevice CudaMemcpyKind = 1
	// CudaMemcpyDeviceToHost as declared in include/driver_types.h:1221
	CudaMemcpyDeviceToHost CudaMemcpyKind = 2
	// CudaMemcpyDeviceToDevice as declared in include/driver_types.h:1222
	CudaMemcpyDeviceToDevice CudaMemcpyKind = 3
	// CudaMemcpyDefault as declared in include/driver_types.h:1223
	CudaMemcpyDefault CudaMemcpyKind = 4
)

type AllocPinnedFlags = C.uint

// CudaErrorT enumeration from include/driver_types.h:85-88
const (
	CudaHostAllocDefault  AllocPinnedFlags = 0x00 /**< Default page-locked allocation flag */
	CudaHostAllocPortable AllocPinnedFlags = 0x01 /**< Pinned memory accessible by all CUDA contexts */
	// Currently not supported
	// CudaHostAllocMapped AllocPinnedFlags 				= 0x02  /**< Map allocation into device space */
	// CudaHostAllocWriteCombined AllocPinnedFlags = 0x04  /**< Write-combined memory */
)

type RegisterPinnedFlags = C.uint

// CudaErrorT enumeration from include/driver_types.h:90-94
const (
	CudaHostRegisterDefault  RegisterPinnedFlags = 0x00 /**< Default host memory registration flag */
	CudaHostRegisterPortable RegisterPinnedFlags = 0x01 /**< Pinned memory accessible by all CUDA contexts */
	// Currently not supported
	// CudaHostRegisterMapped RegisterPinnedFlags    = 0x02  /**< Map registered memory into device space */
	// cudaHostRegisterIoMemory RegisterPinnedFlags  = 0x04  /**< Memory-mapped I/O space */
	// cudaHostRegisterReadOnly RegisterPinnedFlags  = 0x08  /**< Memory-mapped read-only */
)

type DeviceAttribute = uint32

const (
	CudaDevAttrMaxThreadsPerBlock                     DeviceAttribute = 1  /**< Maximum number of threads per block */
	CudaDevAttrMaxBlockDimX                           DeviceAttribute = 2  /**< Maximum block dimension X */
	CudaDevAttrMaxBlockDimY                           DeviceAttribute = 3  /**< Maximum block dimension Y */
	CudaDevAttrMaxBlockDimZ                           DeviceAttribute = 4  /**< Maximum block dimension Z */
	CudaDevAttrMaxGridDimX                            DeviceAttribute = 5  /**< Maximum grid dimension X */
	CudaDevAttrMaxGridDimY                            DeviceAttribute = 6  /**< Maximum grid dimension Y */
	CudaDevAttrMaxGridDimZ                            DeviceAttribute = 7  /**< Maximum grid dimension Z */
	CudaDevAttrMaxSharedMemoryPerBlock                DeviceAttribute = 8  /**< Maximum shared memory available per block in bytes */
	CudaDevAttrTotalConstantMemory                    DeviceAttribute = 9  /**< Memory available on device for __constant__ variables in a CUDA C kernel in bytes */
	CudaDevAttrWarpSize                               DeviceAttribute = 10 /**< Warp size in threads */
	CudaDevAttrMaxPitch                               DeviceAttribute = 11 /**< Maximum pitch in bytes allowed by memory copies */
	CudaDevAttrMaxRegistersPerBlock                   DeviceAttribute = 12 /**< Maximum number of 32-bit registers available per block */
	CudaDevAttrClockRate                              DeviceAttribute = 13 /**< Peak clock frequency in kilohertz */
	CudaDevAttrTextureAlignment                       DeviceAttribute = 14 /**< Alignment requirement for textures */
	CudaDevAttrGpuOverlap                             DeviceAttribute = 15 /**< Device can possibly copy memory and execute a kernel concurrently */
	CudaDevAttrMultiProcessorCount                    DeviceAttribute = 16 /**< Number of multiprocessors on device */
	CudaDevAttrKernelExecTimeout                      DeviceAttribute = 17 /**< Specifies whether there is a run time limit on kernels */
	CudaDevAttrIntegrated                             DeviceAttribute = 18 /**< Device is integrated with host memory */
	CudaDevAttrCanMapHostMemory                       DeviceAttribute = 19 /**< Device can map host memory into CUDA address space */
	CudaDevAttrComputeMode                            DeviceAttribute = 20 /**< Compute mode (See ::cudaComputeMode for details) */
	CudaDevAttrMaxTexture1DWidth                      DeviceAttribute = 21 /**< Maximum 1D texture width */
	CudaDevAttrMaxTexture2DWidth                      DeviceAttribute = 22 /**< Maximum 2D texture width */
	CudaDevAttrMaxTexture2DHeight                     DeviceAttribute = 23 /**< Maximum 2D texture height */
	CudaDevAttrMaxTexture3DWidth                      DeviceAttribute = 24 /**< Maximum 3D texture width */
	CudaDevAttrMaxTexture3DHeight                     DeviceAttribute = 25 /**< Maximum 3D texture height */
	CudaDevAttrMaxTexture3DDepth                      DeviceAttribute = 26 /**< Maximum 3D texture depth */
	CudaDevAttrMaxTexture2DLayeredWidth               DeviceAttribute = 27 /**< Maximum 2D layered texture width */
	CudaDevAttrMaxTexture2DLayeredHeight              DeviceAttribute = 28 /**< Maximum 2D layered texture height */
	CudaDevAttrMaxTexture2DLayeredLayers              DeviceAttribute = 29 /**< Maximum layers in a 2D layered texture */
	CudaDevAttrSurfaceAlignment                       DeviceAttribute = 30 /**< Alignment requirement for surfaces */
	CudaDevAttrConcurrentKernels                      DeviceAttribute = 31 /**< Device can possibly execute multiple kernels concurrently */
	CudaDevAttrEccEnabled                             DeviceAttribute = 32 /**< Device has ECC support enabled */
	CudaDevAttrPciBusId                               DeviceAttribute = 33 /**< PCI bus ID of the device */
	CudaDevAttrPciDeviceId                            DeviceAttribute = 34 /**< PCI device ID of the device */
	CudaDevAttrTccDriver                              DeviceAttribute = 35 /**< Device is using TCC driver model */
	CudaDevAttrMemoryClockRate                        DeviceAttribute = 36 /**< Peak memory clock frequency in kilohertz */
	CudaDevAttrGlobalMemoryBusWidth                   DeviceAttribute = 37 /**< Global memory bus width in bits */
	CudaDevAttrL2CacheSize                            DeviceAttribute = 38 /**< Size of L2 cache in bytes */
	CudaDevAttrMaxThreadsPerMultiProcessor            DeviceAttribute = 39 /**< Maximum resident threads per multiprocessor */
	CudaDevAttrAsyncEngineCount                       DeviceAttribute = 40 /**< Number of asynchronous engines */
	CudaDevAttrUnifiedAddressing                      DeviceAttribute = 41 /**< Device shares a unified address space with the host */
	CudaDevAttrMaxTexture1DLayeredWidth               DeviceAttribute = 42 /**< Maximum 1D layered texture width */
	CudaDevAttrMaxTexture1DLayeredLayers              DeviceAttribute = 43 /**< Maximum layers in a 1D layered texture */
	CudaDevAttrMaxTexture2DGatherWidth                DeviceAttribute = 45 /**< Maximum 2D texture width if cudaArrayTextureGather is set */
	CudaDevAttrMaxTexture2DGatherHeight               DeviceAttribute = 46 /**< Maximum 2D texture height if cudaArrayTextureGather is set */
	CudaDevAttrMaxTexture3DWidthAlt                   DeviceAttribute = 47 /**< Alternate maximum 3D texture width */
	CudaDevAttrMaxTexture3DHeightAlt                  DeviceAttribute = 48 /**< Alternate maximum 3D texture height */
	CudaDevAttrMaxTexture3DDepthAlt                   DeviceAttribute = 49 /**< Alternate maximum 3D texture depth */
	CudaDevAttrPciDomainId                            DeviceAttribute = 50 /**< PCI domain ID of the device */
	CudaDevAttrTexturePitchAlignment                  DeviceAttribute = 51 /**< Pitch alignment requirement for textures */
	CudaDevAttrMaxTextureCubemapWidth                 DeviceAttribute = 52 /**< Maximum cubemap texture width/height */
	CudaDevAttrMaxTextureCubemapLayeredWidth          DeviceAttribute = 53 /**< Maximum cubemap layered texture width/height */
	CudaDevAttrMaxTextureCubemapLayeredLayers         DeviceAttribute = 54 /**< Maximum layers in a cubemap layered texture */
	CudaDevAttrMaxSurface1DWidth                      DeviceAttribute = 55 /**< Maximum 1D surface width */
	CudaDevAttrMaxSurface2DWidth                      DeviceAttribute = 56 /**< Maximum 2D surface width */
	CudaDevAttrMaxSurface2DHeight                     DeviceAttribute = 57 /**< Maximum 2D surface height */
	CudaDevAttrMaxSurface3DWidth                      DeviceAttribute = 58 /**< Maximum 3D surface width */
	CudaDevAttrMaxSurface3DHeight                     DeviceAttribute = 59 /**< Maximum 3D surface height */
	CudaDevAttrMaxSurface3DDepth                      DeviceAttribute = 60 /**< Maximum 3D surface depth */
	CudaDevAttrMaxSurface1DLayeredWidth               DeviceAttribute = 61 /**< Maximum 1D layered surface width */
	CudaDevAttrMaxSurface1DLayeredLayers              DeviceAttribute = 62 /**< Maximum layers in a 1D layered surface */
	CudaDevAttrMaxSurface2DLayeredWidth               DeviceAttribute = 63 /**< Maximum 2D layered surface width */
	CudaDevAttrMaxSurface2DLayeredHeight              DeviceAttribute = 64 /**< Maximum 2D layered surface height */
	CudaDevAttrMaxSurface2DLayeredLayers              DeviceAttribute = 65 /**< Maximum layers in a 2D layered surface */
	CudaDevAttrMaxSurfaceCubemapWidth                 DeviceAttribute = 66 /**< Maximum cubemap surface width */
	CudaDevAttrMaxSurfaceCubemapLayeredWidth          DeviceAttribute = 67 /**< Maximum cubemap layered surface width */
	CudaDevAttrMaxSurfaceCubemapLayeredLayers         DeviceAttribute = 68 /**< Maximum layers in a cubemap layered surface */
	CudaDevAttrMaxTexture1DLinearWidth                DeviceAttribute = 69 /**< Maximum 1D linear texture width */
	CudaDevAttrMaxTexture2DLinearWidth                DeviceAttribute = 70 /**< Maximum 2D linear texture width */
	CudaDevAttrMaxTexture2DLinearHeight               DeviceAttribute = 71 /**< Maximum 2D linear texture height */
	CudaDevAttrMaxTexture2DLinearPitch                DeviceAttribute = 72 /**< Maximum 2D linear texture pitch in bytes */
	CudaDevAttrMaxTexture2DMipmappedWidth             DeviceAttribute = 73 /**< Maximum mipmapped 2D texture width */
	CudaDevAttrMaxTexture2DMipmappedHeight            DeviceAttribute = 74 /**< Maximum mipmapped 2D texture height */
	CudaDevAttrComputeCapabilityMajor                 DeviceAttribute = 75 /**< Major compute capability version number */
	CudaDevAttrComputeCapabilityMinor                 DeviceAttribute = 76 /**< Minor compute capability version number */
	CudaDevAttrMaxTexture1DMipmappedWidth             DeviceAttribute = 77 /**< Maximum mipmapped 1D texture width */
	CudaDevAttrStreamPrioritiesSupported              DeviceAttribute = 78 /**< Device supports stream priorities */
	CudaDevAttrGlobalL1CacheSupported                 DeviceAttribute = 79 /**< Device supports caching globals in L1 */
	CudaDevAttrLocalL1CacheSupported                  DeviceAttribute = 80 /**< Device supports caching locals in L1 */
	CudaDevAttrMaxSharedMemoryPerMultiprocessor       DeviceAttribute = 81 /**< Maximum shared memory available per multiprocessor in bytes */
	CudaDevAttrMaxRegistersPerMultiprocessor          DeviceAttribute = 82 /**< Maximum number of 32-bit registers available per multiprocessor */
	CudaDevAttrManagedMemory                          DeviceAttribute = 83 /**< Device can allocate managed memory on this system */
	CudaDevAttrIsMultiGpuBoard                        DeviceAttribute = 84 /**< Device is on a multi-GPU board */
	CudaDevAttrMultiGpuBoardGroupID                   DeviceAttribute = 85 /**< Unique identifier for a group of devices on the same multi-GPU board */
	CudaDevAttrHostNativeAtomicSupported              DeviceAttribute = 86 /**< Link between the device and the host supports native atomic operations */
	CudaDevAttrSingleToDoublePrecisionPerfRatio       DeviceAttribute = 87 /**< Ratio of single precision performance (in floating-point operations per second) to double precision performance */
	CudaDevAttrPageableMemoryAccess                   DeviceAttribute = 88 /**< Device supports coherently accessing pageable memory without calling cudaHostRegister on it */
	CudaDevAttrConcurrentManagedAccess                DeviceAttribute = 89 /**< Device can coherently access managed memory concurrently with the CPU */
	CudaDevAttrComputePreemptionSupported             DeviceAttribute = 90 /**< Device supports Compute Preemption */
	CudaDevAttrCanUseHostPointerForRegisteredMem      DeviceAttribute = 91 /**< Device can access host registered memory at the same virtual address as the CPU */
	CudaDevAttrReserved92                             DeviceAttribute = 92
	CudaDevAttrReserved93                             DeviceAttribute = 93
	CudaDevAttrReserved94                             DeviceAttribute = 94
	CudaDevAttrCooperativeLaunch                      DeviceAttribute = 95  /**< Device supports launching cooperative kernels via ::cudaLaunchCooperativeKernel*/
	CudaDevAttrCooperativeMultiDeviceLaunch           DeviceAttribute = 96  /**< Deprecated cudaLaunchCooperativeKernelMultiDevice is deprecated. */
	CudaDevAttrMaxSharedMemoryPerBlockOptin           DeviceAttribute = 97  /**< The maximum opt-in shared memory per block. This value may vary by chip. See ::cudaFuncSetAttribute */
	CudaDevAttrCanFlushRemoteWrites                   DeviceAttribute = 98  /**< Device supports flushing of outstanding remote writes. */
	CudaDevAttrHostRegisterSupported                  DeviceAttribute = 99  /**< Device supports host memory registration via ::cudaHostRegister. */
	CudaDevAttrPageableMemoryAccessUsesHostPageTables DeviceAttribute = 100 /**< Device accesses pageable memory via the host's page tables. */
	CudaDevAttrDirectManagedMemAccessFromHost         DeviceAttribute = 101 /**< Host can directly access managed memory on the device without migration. */
	CudaDevAttrMaxBlocksPerMultiprocessor             DeviceAttribute = 106 /**< Maximum number of blocks per multiprocessor */
	CudaDevAttrMaxPersistingL2CacheSize               DeviceAttribute = 108 /**< Maximum L2 persisting lines capacity setting in bytes. */
	CudaDevAttrMaxAccessPolicyWindowSize              DeviceAttribute = 109 /**< Maximum value of cudaAccessPolicyWindow::num_bytes. */
	CudaDevAttrReservedSharedMemoryPerBlock           DeviceAttribute = 111 /**< Shared memory reserved by CUDA driver per block in bytes */
	CudaDevAttrSparseCudaArraySupported               DeviceAttribute = 112 /**< Device supports sparse CUDA arrays and sparse CUDA mipmapped arrays */
	CudaDevAttrHostRegisterReadOnlySupported          DeviceAttribute = 113 /**< Device supports using the ::cudaHostRegister flag cudaHostRegisterReadOnly to register memory that must be mapped as read-only to the GPU */
	CudaDevAttrTimelineSemaphoreInteropSupported      DeviceAttribute = 114 /**< External timeline semaphore interop is supported on the device */
	CudaDevAttrMaxTimelineSemaphoreInteropSupported   DeviceAttribute = 114 /**< Deprecated External timeline semaphore interop is supported on the device */
	CudaDevAttrMemoryPoolsSupported                   DeviceAttribute = 115 /**< Device supports using the ::cudaMallocAsync and ::cudaMemPool family of APIs */
	CudaDevAttrGPUDirectRDMASupported                 DeviceAttribute = 116 /**< Device supports GPUDirect RDMA APIs like nvidia_p2p_get_pages (see https://docs.nvidia.com/cuda/gpudirect-rdma for more information) */
	CudaDevAttrGPUDirectRDMAFlushWritesOptions        DeviceAttribute = 117 /**< The returned attribute shall be interpreted as a bitmask where the individual bits are listed in the ::cudaFlushGPUDirectRDMAWritesOptions enum */
	CudaDevAttrGPUDirectRDMAWritesOrdering            DeviceAttribute = 118 /**< GPUDirect RDMA writes to the device do not need to be flushed for consumers within the scope indicated by the returned attribute. See ::cudaGPUDirectRDMAWritesOrdering for the numerical values returned here. */
	CudaDevAttrMemoryPoolSupportedHandleTypes         DeviceAttribute = 119 /**< Handle types supported with mempool based IPC */
	CudaDevAttrClusterLaunch                          DeviceAttribute = 120 /**< Indicates device supports cluster launch */
	CudaDevAttrDeferredMappingCudaArraySupported      DeviceAttribute = 121 /**< Device supports deferred mapping CUDA arrays and CUDA mipmapped arrays */
	CudaDevAttrReserved122                            DeviceAttribute = 122
	CudaDevAttrReserved123                            DeviceAttribute = 123
	CudaDevAttrReserved124                            DeviceAttribute = 124
	CudaDevAttrIpcEventSupport                        DeviceAttribute = 125 /**< Device supports IPC Events. */
	CudaDevAttrMemSyncDomainCount                     DeviceAttribute = 126 /**< Number of memory synchronization domains the device supports. */
	CudaDevAttrReserved127                            DeviceAttribute = 127
	CudaDevAttrReserved128                            DeviceAttribute = 128
	CudaDevAttrReserved129                            DeviceAttribute = 129
	CudaDevAttrNumaConfig                             DeviceAttribute = 130 /**< NUMA configuration of a device: value is of type cudaDeviceNumaConfig enum */
	CudaDevAttrNumaId                                 DeviceAttribute = 131 /**< NUMA node ID of the GPU memory */
	CudaDevAttrReserved132                            DeviceAttribute = 132
	CudaDevAttrMpsEnabled                             DeviceAttribute = 133 /**< Contexts created on this device will be shared via MPS */
	CudaDevAttrHostNumaId                             DeviceAttribute = 134 /**< NUMA ID of the host node closest to the device. Returns -1 when system does not support NUMA. */
	CudaDevAttrMax                                    DeviceAttribute = 135
)
