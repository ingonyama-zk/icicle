package bls12377

// #cgo CFLAGS: -I./include/
// #include "vec_ops.h"
import "C"

import (
	"unsafe"

	"github.com/ingonyama-zk/icicle/wrappers/golang/core"
	cr "github.com/ingonyama-zk/icicle/wrappers/golang/cuda_runtime"
)

func VecOp(a, b, out core.HostOrDeviceSlice, config core.VecOpsConfig, op core.VecOps) (ret cr.CudaError) {
	core.VecOpCheck(a, b, out, &config)
	var cA, cB, cOut *C.scalar_t

	if a.IsOnDevice() {
		cA = (*C.scalar_t)(a.(core.DeviceSlice).AsPointer())
	} else {
		cA = (*C.scalar_t)(unsafe.Pointer(&a.(core.HostSlice[ScalarField])[0]))
	}

	if b.IsOnDevice() {
		cB = (*C.scalar_t)(b.(core.DeviceSlice).AsPointer())
	} else {
		cB = (*C.scalar_t)(unsafe.Pointer(&b.(core.HostSlice[ScalarField])[0]))
	}

	if out.IsOnDevice() {
		cOut = (*C.scalar_t)(out.(core.DeviceSlice).AsPointer())
	} else {
		cOut = (*C.scalar_t)(unsafe.Pointer(&out.(core.HostSlice[ScalarField])[0]))
	}

	cConfig := (*C.VecOpsConfig)(unsafe.Pointer(&config))
	cSize := (C.int)(a.Len())

	switch op {
	case core.Sub:
		ret = (cr.CudaError)(C.bls12_377SubCuda(cA, cB, cSize, cConfig, cOut))
	case core.Add:
		ret = (cr.CudaError)(C.bls12_377AddCuda(cA, cB, cSize, cConfig, cOut))
	case core.Mul:
		ret = (cr.CudaError)(C.bls12_377MulCuda(cA, cB, cSize, cConfig, cOut))
	}

	return ret
}
