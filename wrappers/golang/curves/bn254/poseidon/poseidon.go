package poseidon

// #cgo CFLAGS: -I./include/
// #include "poseidon.h"
import "C"
import (
	"unsafe"

	"github.com/ingonyama-zk/icicle/v2/wrappers/golang/core"
	cr "github.com/ingonyama-zk/icicle/v2/wrappers/golang/cuda_runtime"
	"github.com/ingonyama-zk/icicle/v2/wrappers/golang/curves/bn254"
)

type PoseidonHandler = C.struct_PoseidonInst
type Poseidon struct {
	width  uint32
	handle *PoseidonHandler
}

func Create(arity uint32, alpha uint32, fullRoundsHalf uint32, partialRounds uint32, scalars core.HostOrDeviceSlice, mdsMatrix core.HostOrDeviceSlice, nonSparseMatrix core.HostOrDeviceSlice, sparseMatrices core.HostOrDeviceSlice, domainTag bn254.ScalarField, ctx *cr.DeviceContext) Poseidon {
	var poseidon *PoseidonHandler
	cArity := (C.uint)(arity)
	cAlpha := (C.uint)(alpha)
	cFullRoundsHalf := (C.uint)(fullRoundsHalf)
	cPartialRounds := (C.uint)(partialRounds)
	cScalars := (*C.scalar_t)(scalars.AsUnsafePointer())
	cMdsMatrix := (*C.scalar_t)(mdsMatrix.AsUnsafePointer())
	cNonSparseMatrix := (*C.scalar_t)(nonSparseMatrix.AsUnsafePointer())
	cSparseMatrices := (*C.scalar_t)(sparseMatrices.AsUnsafePointer())
	cDomainTag := (*C.scalar_t)(unsafe.Pointer(&domainTag))
	cCtx := (*C.DeviceContext)(unsafe.Pointer(ctx))
	C.bn254_poseidon_create_cuda(&poseidon, cArity, cAlpha, cFullRoundsHalf, cPartialRounds, cScalars, cMdsMatrix, cNonSparseMatrix, cSparseMatrices, cDomainTag, cCtx)
	return Poseidon{handle: poseidon, width: arity + 1}
}

func Load(arity uint32, ctx *cr.DeviceContext) Poseidon {
	var poseidon *PoseidonHandler
	cArity := (C.uint)(arity)
	cCtx := (*C.DeviceContext)(unsafe.Pointer(ctx))
	C.bn254_poseidon_load_cuda(&poseidon, cArity, cCtx)
	return Poseidon{handle: poseidon, width: arity + 1}
}

func (poseidon *Poseidon) AbsorbMany(inputs core.HostOrDeviceSlice, states core.DeviceSlice, numberOfStates uint32, inputBlockLen uint32, cfg *core.SpongeConfig) core.IcicleError {
	cInputs := (*C.scalar_t)(inputs.AsUnsafePointer())
	cStates := (*C.scalar_t)(states.AsUnsafePointer())
	cNumberOfStates := (C.uint)(numberOfStates)
	cInputBlockLen := (C.uint)(inputBlockLen)
	cCfg := (*C.SpongeConfig)(unsafe.Pointer(cfg))
	__ret := C.bn254_poseidon_absorb_many_cuda(poseidon.handle, cInputs, cStates, cNumberOfStates, cInputBlockLen, cCfg)
	err := (cr.CudaError)(__ret)
	return core.FromCudaError(err)
}

func (poseidon *Poseidon) SqueezeMany(states core.DeviceSlice, output core.HostOrDeviceSlice, numberOfStates uint32, outputLen uint32, cfg *core.SpongeConfig) core.IcicleError {
	cStates := (*C.scalar_t)(states.AsUnsafePointer())
	cOutput := (*C.scalar_t)(output.AsUnsafePointer())
	cNumberOfStates := (C.uint)(numberOfStates)
	cOutputLen := (C.uint)(outputLen)
	cCfg := (*C.SpongeConfig)(unsafe.Pointer(cfg))
	__ret := C.bn254_poseidon_absorb_many_cuda(poseidon.handle, cStates, cOutput, cNumberOfStates, cOutputLen, cCfg)
	err := (cr.CudaError)(__ret)
	return core.FromCudaError(err)
}

func (poseidon *Poseidon) HashMany(inputs core.HostOrDeviceSlice, output core.HostOrDeviceSlice, numberOfStates uint32, inputBlockLen uint32, outputLen uint32, cfg *core.SpongeConfig) core.IcicleError {
	cInputs := (*C.scalar_t)(inputs.AsUnsafePointer())
	cOutput := (*C.scalar_t)(output.AsUnsafePointer())
	cNumberOfStates := (C.uint)(numberOfStates)
	cInputBlockLen := (C.uint)(inputBlockLen)
	cOutputLen := (C.uint)(outputLen)
	cCfg := (*C.SpongeConfig)(unsafe.Pointer(cfg))
	__ret := C.bn254_poseidon_hash_many_cuda(poseidon.handle, cInputs, cOutput, cNumberOfStates, cInputBlockLen, cOutputLen, cCfg)
	err := (cr.CudaError)(__ret)
	return core.FromCudaError(err)
}

func (poseidon *Poseidon) Delete(ctx *cr.DeviceContext) core.IcicleError {
	cCtx := (*C.DeviceContext)(unsafe.Pointer(ctx))
	__ret := C.bn254_poseidon_delete_cuda(poseidon.handle, cCtx)
	err := (cr.CudaError)(__ret)
	return core.FromCudaError(err)
}

func (poseidon *Poseidon) GetDefaultSpongeConfig() core.SpongeConfig {
	cfg := core.GetDefaultSpongeConfig()
	cfg.InputRate = poseidon.width - 1
	cfg.OutputRate = poseidon.width
	return cfg
}

// func GetDefaultPoseidonConfig() core.PoseidonConfig {
// 	return core.GetDefaultPoseidonConfig()
// }

// func PoseidonHash[T any](scalars, results core.HostOrDeviceSlice, numberOfStates int, cfg *core.PoseidonConfig, constants *core.PoseidonConstants[T]) core.IcicleError {
// 	scalarsPointer, resultsPointer, cfgPointer := core.PoseidonCheck(scalars, results, cfg, constants, numberOfStates)

// 	cScalars := (*C.scalar_t)(scalarsPointer)
// 	cResults := (*C.scalar_t)(resultsPointer)
// 	cNumberOfStates := (C.int)(numberOfStates)
// 	cArity := (C.int)(constants.Arity)
// 	cConstants := (*C.PoseidonConstants)(unsafe.Pointer(constants))
// 	cCfg := (*C.PoseidonConfig)(cfgPointer)

// 	__ret := C.bn254_poseidon_hash_cuda(cScalars, cResults, cNumberOfStates, cArity, cConstants, cCfg)

// 	err := (cr.CudaError)(__ret)
// 	return core.FromCudaError(err)
// }

// func CreateOptimizedPoseidonConstants[T any](arity, fullRoundsHalfs, partialRounds int, constants core.HostOrDeviceSlice, ctx cr.DeviceContext, poseidonConstants *core.PoseidonConstants[T]) core.IcicleError {

// 	cArity := (C.int)(arity)
// 	cFullRoundsHalfs := (C.int)(fullRoundsHalfs)
// 	cPartialRounds := (C.int)(partialRounds)
// 	cConstants := (*C.scalar_t)(constants.AsUnsafePointer())
// 	cCtx := (*C.DeviceContext)(unsafe.Pointer(&ctx))
// 	cPoseidonConstants := (*C.PoseidonConstants)(unsafe.Pointer(poseidonConstants))

// 	__ret := C.bn254_create_optimized_poseidon_constants_cuda(cArity, cFullRoundsHalfs, cPartialRounds, cConstants, cCtx, cPoseidonConstants)
// 	err := (cr.CudaError)(__ret)
// 	return core.FromCudaError(err)
// }

// func InitOptimizedPoseidonConstantsCuda[T any](arity int, ctx cr.DeviceContext, constants *core.PoseidonConstants[T]) core.IcicleError {

// 	cArity := (C.int)(arity)
// 	cCtx := (*C.DeviceContext)(unsafe.Pointer(&ctx))
// 	cConstants := (*C.PoseidonConstants)(unsafe.Pointer(constants))

// 	__ret := C.bn254_init_optimized_poseidon_constants_cuda(cArity, cCtx, cConstants)
// 	err := (cr.CudaError)(__ret)
// 	return core.FromCudaError(err)
// }
