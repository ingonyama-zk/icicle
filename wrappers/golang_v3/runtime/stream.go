package runtime

// #cgo CFLAGS: -I./include/
// #include "runtime.h"
import "C"
import "unsafe"

type Stream = unsafe.Pointer

func CreateStream() (Stream, EIcicleError) {
	var stream Stream
	ret := C.icicle_create_stream(&stream)
	err := (EIcicleError)(ret)
	return stream, err
}

func DestroyStream(stream Stream) EIcicleError {
	ret := C.icicle_destroy_stream(stream)
	err := (EIcicleError)(ret)
	if err == Success {
		stream = nil
	}
	return err
}

func SynchronizeStream(stream Stream) EIcicleError {
	ret := C.icicle_stream_synchronize(stream)
	err := (EIcicleError)(ret)
	return err
}
