package hash

// #cgo CFLAGS: -I./include/
// #include "blake3.h"
import "C"
import (
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/runtime"
)

func NewBlake3Hasher(inputChunkSize uint64) (Hasher, runtime.EIcicleError) {
	h := C.icicle_create_blake3((C.uint64_t)(inputChunkSize))
	if h == nil {
		return Hasher{handle: nil}, runtime.UnknownError
	}

	return Hasher{
		handle: h,
	}, runtime.Success
}
