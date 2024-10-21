package hash

// #cgo CFLAGS: -I./include/
// #include "blake2s.h"
import "C"
import (
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/runtime"
)

func NewBlake2sHasher(inputChunkSize uint64) (Hasher, runtime.EIcicleError) {
	h := C.icicle_create_blake2s((C.ulong)(inputChunkSize))
	if h == nil {
		return Hasher{handle: nil}, runtime.UnknownError
	}

	return Hasher{
		handle: h,
	}, runtime.Success
}
