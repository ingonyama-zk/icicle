package poseidon

// #cgo CFLAGS: -I./include/
// #include "poseidon.h"
import "C"

import (
	"unsafe"

	bn254 "github.com/ingonyama-zk/icicle/v3/wrappers/golang/curves/bn254"
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/hash"
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/runtime"
)

func NewHasher(t uint64, domainTag *bn254.ScalarField) (hash.Hasher, runtime.EIcicleError) {
	cT := (C.uint)(t)
	var cDomainTag *C.scalar_t // This is set to nil as zero value
	if domainTag != nil {
		cDomainTag = (*C.scalar_t)(unsafe.Pointer(domainTag.AsPointer()))
	}

	handle := C.bn254_create_poseidon_hasher(cT, cDomainTag)

	if handle == nil {
		return hash.Hasher{}, runtime.UnknownError
	}

	return hash.FromHandle(unsafe.Pointer(handle)), runtime.Success
}
