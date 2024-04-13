package bls12377

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
		cA = (*C.scalar_t)(aDevice.AsPointer())
	} else {
		cA = (*C.scalar_t)(unsafe.Pointer(&a.(core.HostSlice[S])[0]))
	}

	if b.IsOnDevice() {
		bDevice := b.(core.DeviceSlice)
		bDevice.CheckDevice()
		cB = (*C.scalar_t)(bDevice.AsPointer())
	} else {
		cB = (*C.scalar_t)(unsafe.Pointer(&b.(core.HostSlice[S])[0]))
	}

	if out.IsOnDevice() {
		outDevice := out.(core.DeviceSlice)
		outDevice.CheckDevice()
		cOut = (*C.scalar_t)(outDevice.AsPointer())
	} else {
		cOut = (*C.scalar_t)(unsafe.Pointer(&out.(core.HostSlice[S])[0]))
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

func TransposeMatrix(in, out core.HostOrDeviceSlice, columnSize, rowSize int, ctx cr.DeviceContext, onDevice, isAsync bool) (ret core.IcicleError){

	var inPointer, outPointer unsafe.Pointer
	if onDevice {
		inPointer = in.(core.DeviceSlice).AsPointer()
		outPointer = out.(core.DeviceSlice).AsPointer()
	} else {
		inPointer = unsafe.Pointer(&in.(core.HostSlice[ScalarField])[0])
		outPointer = unsafe.Pointer(&out.(core.HostSlice[ScalarField])[0])
	}
	cIn := (*C.scalar_t)(inPointer)
	cOut := (*C.scalar_t)(outPointer)

	cCtx := (*C.DeviceContext)(unsafe.Pointer(&ctx))
	cRowSize := (C.int)(rowSize)
	cColumnSize := (C.int)(columnSize)
	cOnDevice := (C._Bool)(onDevice)
	cIsAsync := (C._Bool)(isAsync)

	err := (cr.CudaError)(C.bls12_377TransposeMatrix( cIn, cRowSize, cColumnSize, cOut, cCtx, cOnDevice, cIsAsync))
	return core.FromCudaError(err)
} 
