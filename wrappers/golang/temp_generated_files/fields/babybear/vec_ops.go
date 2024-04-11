package babybear

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
		aDevice := a.(core.DeviceSlice)
		aDevice.CheckDevice()
		cA = (*C.scalar_t)(aDevice.AsPointer())
	} else {
		cA = (*C.scalar_t)(unsafe.Pointer(&a.(core.HostSlice[ScalarField])[0]))
	}

	if b.IsOnDevice() {
		bDevice := b.(core.DeviceSlice)
		bDevice.CheckDevice()
		cB = (*C.scalar_t)(bDevice.AsPointer())
	} else {
		cB = (*C.scalar_t)(unsafe.Pointer(&b.(core.HostSlice[ScalarField])[0]))
	}

	if out.IsOnDevice() {
		outDevice := out.(core.DeviceSlice)
		outDevice.CheckDevice()
		cOut = (*C.scalar_t)(outDevice.AsPointer())
	} else {
		cOut = (*C.scalar_t)(unsafe.Pointer(&out.(core.HostSlice[ScalarField])[0]))
	}

	cConfig := (*C.VecOpsConfig)(unsafe.Pointer(&config))
	cSize := (C.int)(a.Len())

	switch op {
	case core.Sub:
		ret = (cr.CudaError)(C.babybearSubCuda(cA, cB, cSize, cConfig, cOut))
	case core.Add:
		ret = (cr.CudaError)(C.babybearAddCuda(cA, cB, cSize, cConfig, cOut))
	case core.Mul:
		ret = (cr.CudaError)(C.babybearMulCuda(cA, cB, cSize, cConfig, cOut))
	}

	return ret
}
