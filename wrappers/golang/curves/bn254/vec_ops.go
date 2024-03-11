package bn254

// #cgo CFLAGS: -I./include/
// #include "vec_ops.h"
import "C"

import (
	"github.com/ingonyama-zk/icicle/wrappers/golang/core"
	cr "github.com/ingonyama-zk/icicle/wrappers/golang/cuda_runtime"
)

func VecOp(a, b, out core.HostOrDeviceSlice, config core.VecOpsConfig, op core.VecOps) (ret cr.CudaError) {
	aPointer, bPointer, outPointer, cfgPointer, size := core.VecOpCheck[ScalarField](a, b, out, &config)

	cA := (*C.scalar_t)(aPointer)
	cB := (*C.scalar_t)(bPointer)
	cOut := (*C.scalar_t)(outPointer)
	cConfig := (*C.VecOpsConfig)(cfgPointer)
	cSize := (C.int)(size)

	switch op {
	case core.Sub:
		ret = (cr.CudaError)(C.bn254SubCuda(cA, cB, cSize, cConfig, cOut))
	case core.Add:
		ret = (cr.CudaError)(C.bn254AddCuda(cA, cB, cSize, cConfig, cOut))
	case core.Mul:
		ret = (cr.CudaError)(C.bn254MulCuda(cA, cB, cSize, cConfig, cOut))
	}

	return ret
}
