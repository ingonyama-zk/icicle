package hash

// #cgo LDFLAGS: -L/usr/local/lib  -licicle_hash -lstdc++ -Wl,-rpath=/usr/local/lib
// #cgo CFLAGS: -I./include/
// #include "hash.h"
import "C"
import (
	"unsafe"

	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/core"
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/runtime"
)

type HasherHandle = *C.struct_Hash

type Hasher struct {
	handle HasherHandle
}

// Hash computes the hash of the input data using the specified configuration and stores the result in the output slice.
//
// Parameters:
//
//   - input: The input data to be hashed; assumed to be in bytes.
//   - output: The output slice to store the computed hash; assumed to be in bytes.
//   - cfg: The configuration options for the hashing operation.
//
// Returns:
//
//   - An error if the hashing operation fails, otherwise runtime.Success.
func (h *Hasher) Hash(input, output core.HostOrDeviceSlice, cfg core.HashConfig) runtime.EIcicleError {
	inputPtr, outputPtr, length, err := core.HashCheck(input, output, h.OutputSize(), &cfg)
	if err != runtime.Success {
		return err
	}

	cInputPtr := (*C.uint8_t)(inputPtr)
	cLength := (C.uint64_t)(length)
	cCfg := (*C.HashConfig)(unsafe.Pointer(&cfg))
	cOutputPtr := (*C.uint8_t)(outputPtr)

	__err := C.icicle_hasher_hash(h.handle, cInputPtr, cLength, cCfg, cOutputPtr)
	return runtime.EIcicleError(__err)
}

func (h *Hasher) OutputSize() uint64 {
	return uint64(C.icicle_hasher_output_size(h.handle))
}

func (h *Hasher) GetHandle() HasherHandle {
	return h.handle
}

func (h *Hasher) Delete() runtime.EIcicleError {
	__err := C.icicle_hasher_delete(h.handle)
	return runtime.EIcicleError(__err)
}
