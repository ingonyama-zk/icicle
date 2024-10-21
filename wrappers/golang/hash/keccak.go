package hash

// #cgo CFLAGS: -I./include/
// #include "keccak.h"
import "C"
import (
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/runtime"
)

func NewKeccak256Hasher(inputChunkSize uint64) (Hasher, runtime.EIcicleError) {
	h := C.icicle_create_keccak_256((C.ulong)(inputChunkSize))
	if h == nil {
		return Hasher{handle: nil}, runtime.UnknownError
	}

	return Hasher{
		handle: h,
	}, runtime.Success
}

func NewKeccak512Hasher(inputChunkSize uint64) (Hasher, runtime.EIcicleError) {
	h := C.icicle_create_keccak_512((C.ulong)(inputChunkSize))
	if h == nil {
		return Hasher{handle: nil}, runtime.UnknownError
	}

	return Hasher{
		handle: h,
	}, runtime.Success
}
