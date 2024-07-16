package vecOps

// #cgo CFLAGS: -I./include/
// #include "vec_ops.h"
import "C"

import (
	"github.com/ingonyama-zk/icicle/v2/wrappers/golang/core"
	cr "github.com/ingonyama-zk/icicle/v2/wrappers/golang/cuda_runtime"

	"unsafe"
)

func VecOp(a, b, out core.HostOrDeviceSlice, config core.VecOpsConfig, op core.VecOps) (ret cr.CudaError) {
	aPointer, bPointer, outPointer, cfgPointer, size := core.VecOpCheck(a, b, out, &config)

	cA := (*C.scalar_t)(aPointer)
	cB := (*C.scalar_t)(bPointer)
	cOut := (*C.scalar_t)(outPointer)
	cConfig := (*C.VecOpsConfig)(cfgPointer)
	cSize := (C.int)(size)

	switch op {
	case core.Sub:
		ret = (cr.CudaError)(C.bls12_377_sub_cuda(cA, cB, cSize, cConfig, cOut))
	case core.Add:
		ret = (cr.CudaError)(C.bls12_377_add_cuda(cA, cB, cSize, cConfig, cOut))
	case core.Mul:
		ret = (cr.CudaError)(C.bls12_377_mul_cuda(cA, cB, cSize, cConfig, cOut))
	}

	return ret
}

func TransposeMatrix(in, out core.HostOrDeviceSlice, columnSize, rowSize int, ctx cr.DeviceContext, onDevice, isAsync bool) (ret core.IcicleError) {
	core.TransposeCheck(in, out, onDevice)

	cIn := (*C.scalar_t)(in.AsUnsafePointer())
	cOut := (*C.scalar_t)(out.AsUnsafePointer())
	cCtx := (*C.DeviceContext)(unsafe.Pointer(&ctx))
	cRowSize := (C.int)(rowSize)
	cColumnSize := (C.int)(columnSize)
	cOnDevice := (C._Bool)(onDevice)
	cIsAsync := (C._Bool)(isAsync)

	err := (cr.CudaError)(C.bls12_377_transpose_matrix_cuda(cIn, cRowSize, cColumnSize, cOut, cCtx, cOnDevice, cIsAsync))
	return core.FromCudaError(err)
}
