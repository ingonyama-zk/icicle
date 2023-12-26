package cuda_runtime

type Stream = CudaStream

func CreateStream() (Stream, CudaError) {
	var stream CudaStream
	error := cudaStreamCreate(&stream)
	return stream, error
}

func CreateStreamWithFlags(flags CudaStreamCreateFlags) (Stream, CudaError) {
	var stream CudaStream
	error := cudaStreamCreateWithFlags(&stream, flags)
	return stream, error
}

func DestroyStream(s *Stream) CudaError {
	err := cudaStreamDestroy(*s)
	if err == CudaSuccess {
		s = nil
	}
	return err
}

func SynchronizeStream(s *Stream) CudaError {
	return cudaStreamSynchronize(*s)
}
