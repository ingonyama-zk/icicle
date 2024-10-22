package hash

// #cgo CFLAGS: -I./include/
// #include "sha3.h"
import "C"
import (
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/runtime"
)

func NewSha3256Hasher(inputChunkSize uint64) (Hasher, runtime.EIcicleError) {
	h := C.icicle_create_sha3_256((C.ulong)(inputChunkSize))
	if h == nil {
		return Hasher{handle: nil}, runtime.UnknownError
	}

	return Hasher{
		handle: h,
	}, runtime.Success
}

func NewSha3512Hasher(inputChunkSize uint64) (Hasher, runtime.EIcicleError) {
	h := C.icicle_create_sha3_512((C.ulong)(inputChunkSize))
	if h == nil {
		return Hasher{handle: nil}, runtime.UnknownError
	}

	return Hasher{
		handle: h,
	}, runtime.Success
}
