package cuda_runtime

/*
#cgo LDFLAGS: -L/usr/local/cuda/lib64 -lcudart
#cgo CFLAGS: -I /usr/local/cuda/include
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdlib.h>
*/
import "C"

// CudaStream as declared in include/driver_types.h:2873
type CudaStream C.cudaStream_t

// CudaEvent as declared in include/driver_types.h:2878
type CudaEvent C.cudaEvent_t

// CudaMemPool as declared in include/driver_types.h:2928
type CudaMemPool C.cudaMemPool_t
