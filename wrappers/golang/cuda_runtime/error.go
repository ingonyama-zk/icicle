package cuda_runtime

// #cgo CFLAGS: -I /usr/local/cuda/include
// #cgo LDFLAGS: -L/usr/local/cuda/lib64 -lcudart
/*
#include <cuda.h>
#include <cuda_runtime.h>
*/
import "C"

func GetLastError() CudaError {
	ret := C.cudaGetLastError()
	err := (CudaError)(ret)
	return err
}
