package bw6761

// #cgo CFLAGS: -I./include/
// #include "vec_ops.h"
import "C"

import (
	"unsafe"

	"github.com/ingonyama-zk/icicle/wrappers/golang/core"
	cr "github.com/ingonyama-zk/icicle/wrappers/golang/cuda_runtime"
)

func VecOp[S any](a, b, out core.HostOrDeviceSlice, config core.VecOpsConfig, op core.VecOps) (ret cr.CudaError) {
	core.VecOpCheck(a, b, out, &config)
	var cA, cB, cOut *C.scalar_t

	if a.IsOnDevice() {
		aDevice := a.(core.DeviceSlice)
		aDevice.CheckDevice()
		cA = (*C.scalar_t)(aDevice.AsUnsafePointer())
	} else {
		cA = (*C.scalar_t)(a.AsUnsafePointer())
	}

	if b.IsOnDevice() {
		bDevice := b.(core.DeviceSlice)
		bDevice.CheckDevice()
		cB = (*C.scalar_t)(bDevice.AsUnsafePointer())
	} else {
		cB = (*C.scalar_t)(b.AsUnsafePointer())
	}

	if out.IsOnDevice() {
		outDevice := out.(core.DeviceSlice)
		outDevice.CheckDevice()
		cOut = (*C.scalar_t)(outDevice.AsUnsafePointer())
	} else {
		cOut = (*C.scalar_t)(out.AsUnsafePointer())
	}

	cConfig := (*C.VecOpsConfig)(unsafe.Pointer(&config))
	cSize := (C.int)(a.Len())

	switch op {
	case core.Sub:
		ret = (cr.CudaError)(C.bw6_761SubCuda(cA, cB, cSize, cConfig, cOut))
	case core.Add:
		ret = (cr.CudaError)(C.bw6_761AddCuda(cA, cB, cSize, cConfig, cOut))
	case core.Mul:
		ret = (cr.CudaError)(C.bw6_761MulCuda(cA, cB, cSize, cConfig, cOut))
	}

	return ret
}

func TransposeMatrix(in, out core.HostOrDeviceSlice, columnSize, rowSize int, ctx cr.DeviceContext, onDevice, isAsync bool) (ret core.IcicleError) {
	core.TransposeCheck(in, out, onDevice)
	inPointer := in.AsUnsafePointer()
	outPointer := out.AsUnsafePointer()

	cIn := (*C.scalar_t)(inPointer)
	cOut := (*C.scalar_t)(outPointer)

	cCtx := (*C.DeviceContext)(unsafe.Pointer(&ctx))
	cRowSize := (C.int)(rowSize)
	cColumnSize := (C.int)(columnSize)
	cOnDevice := (C._Bool)(onDevice)
	cIsAsync := (C._Bool)(isAsync)

	err := (cr.CudaError)(C.bw6_761TransposeMatrix(cIn, cRowSize, cColumnSize, cOut, cCtx, cOnDevice, cIsAsync))
	return core.FromCudaError(err)
}
