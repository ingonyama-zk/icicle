package poseidon

// #cgo CFLAGS: -I./include/
// #include "poseidon.h"
import "C"

import (
	"unsafe"

	"github.com/ingonyama-zk/icicle/v2/wrappers/golang/core"
	cr "github.com/ingonyama-zk/icicle/v2/wrappers/golang/cuda_runtime"
)

func GetDefaultPoseidonConfig() core.PoseidonConfig {
	return core.GetDefaultPoseidonConfig()
}

func PoseidonHash[T any](scalars, results core.HostOrDeviceSlice, numberOfStates int, cfg *core.PoseidonConfig, constants *core.PoseidonConstants[T]) core.IcicleError {
	scalarsPointer, resultsPointer, cfgPointer := core.PoseidonCheck(scalars, results, cfg, constants, numberOfStates)

	cScalars := (*C.scalar_t)(scalarsPointer)
	cResults := (*C.scalar_t)(resultsPointer)
	cNumberOfStates := (C.int)(numberOfStates)
	cArity := (C.int)(constants.Arity)
	cConstants := (*C.PoseidonConstants)(unsafe.Pointer(constants))
	cCfg := (*C.PoseidonConfig)(cfgPointer)

	__ret := C.bw6_761_poseidon_hash_cuda(cScalars, cResults, cNumberOfStates, cArity, cConstants, cCfg)

	err := (cr.CudaError)(__ret)
	return core.FromCudaError(err)
}

func CreateOptimizedPoseidonConstants[T any](arity, fullRoundsHalfs, partialRounds int, constants core.HostOrDeviceSlice, ctx cr.DeviceContext, poseidonConstants *core.PoseidonConstants[T]) core.IcicleError {

	cArity := (C.int)(arity)
	cFullRoundsHalfs := (C.int)(fullRoundsHalfs)
	cPartialRounds := (C.int)(partialRounds)
	cConstants := (*C.scalar_t)(constants.AsUnsafePointer())
	cCtx := (*C.DeviceContext)(unsafe.Pointer(&ctx))
	cPoseidonConstants := (*C.PoseidonConstants)(unsafe.Pointer(poseidonConstants))

	__ret := C.bw6_761_create_optimized_poseidon_constants_cuda(cArity, cFullRoundsHalfs, cPartialRounds, cConstants, cCtx, cPoseidonConstants)
	err := (cr.CudaError)(__ret)
	return core.FromCudaError(err)
}

func InitOptimizedPoseidonConstantsCuda[T any](arity int, ctx cr.DeviceContext, constants *core.PoseidonConstants[T]) core.IcicleError {

	cArity := (C.int)(arity)
	cCtx := (*C.DeviceContext)(unsafe.Pointer(&ctx))
	cConstants := (*C.PoseidonConstants)(unsafe.Pointer(constants))

	__ret := C.bw6_761_init_optimized_poseidon_constants_cuda(cArity, cCtx, cConstants)
	err := (cr.CudaError)(__ret)
	return core.FromCudaError(err)
}
