package vecOps

// #cgo CFLAGS: -I./include/
// #include "vec_ops.h"
import "C"

import (
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/core"
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/runtime"
)

func VecOp(a, b, out core.HostOrDeviceSlice, config core.VecOpsConfig, op core.VecOps) (ret runtime.EIcicleError) {
	aPointer, bPointer, outPointer, cfgPointer, size := core.VecOpCheck(a, b, out, &config)

	cA := (*C.scalar_t)(aPointer)
	cB := (*C.scalar_t)(bPointer)
	cOut := (*C.scalar_t)(outPointer)
	cConfig := (*C.VecOpsConfig)(cfgPointer)
	cSize := (C.int)(size)

	switch op {
	case core.Sub:
		ret = (runtime.EIcicleError)(C.babybear_extension_vector_sub(cA, cB, cSize, cConfig, cOut))
	case core.Add:
		ret = (runtime.EIcicleError)(C.babybear_extension_vector_add(cA, cB, cSize, cConfig, cOut))
	case core.Mul:
		ret = (runtime.EIcicleError)(C.babybear_extension_vector_mul(cA, cB, cSize, cConfig, cOut))
	}

	return ret
}

func TransposeMatrix(in, out core.HostOrDeviceSlice, columnSize, rowSize int, config core.VecOpsConfig) runtime.EIcicleError {
	inPointer, _, outPointer, cfgPointer, _ := core.VecOpCheck(in, in, out, &config)

	cIn := (*C.scalar_t)(inPointer)
	cRowSize := (C.int)(rowSize)
	cColumnSize := (C.int)(columnSize)
	cConfig := (*C.VecOpsConfig)(cfgPointer)
	cOut := (*C.scalar_t)(outPointer)

	err := (C.babybear_extension_matrix_transpose(cIn, cRowSize, cColumnSize, cConfig, cOut))
	return runtime.EIcicleError(err)
}
