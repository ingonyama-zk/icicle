package cuda_runtime

// #cgo CFLAGS: -I /usr/local/cuda/include
// #cgo LDFLAGS: -L/usr/local/cuda/lib64 -lcudart
/*
#include <cuda.h>
#include <cuda_runtime.h>
*/
import "C"

type Stream = CudaStream

func CreateStream() (Stream, CudaError) {
	var stream Stream
	cPStream := (*C.cudaStream_t)(&stream)
	ret := C.cudaStreamCreate(cPStream)
	err := (CudaError)(ret)
	return stream, err
}

func CreateStreamWithFlags(flags CudaStreamCreateFlags) (Stream, CudaError) {
	var stream CudaStream
	cPStream := (*C.cudaStream_t)(&stream)
	cFlags := (C.uint)(flags)
	ret := C.cudaStreamCreateWithFlags(cPStream, cFlags)
	err := (CudaError)(ret)
	return stream, err
}

func DestroyStream(stream *Stream) CudaError {
	cStream := (C.cudaStream_t)(*stream)
	ret := C.cudaStreamDestroy(cStream)
	err := (CudaError)(ret)
	if err == CudaSuccess {
		stream = nil
	}
	return err
}

func SynchronizeStream(stream *Stream) CudaError {
	cStream := (C.cudaStream_t)(*stream)
	ret := C.cudaStreamSynchronize(cStream)
	err := (CudaError)(ret)
	return err
}
