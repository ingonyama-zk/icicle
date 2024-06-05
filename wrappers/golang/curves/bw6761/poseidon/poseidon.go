package poseidon

// #cgo CFLAGS: -I./include/
// #include "poseidon.h"
import "C"
import (
	"runtime"
	"unsafe"

	"github.com/ingonyama-zk/icicle/v2/wrappers/golang/core"
	cr "github.com/ingonyama-zk/icicle/v2/wrappers/golang/cuda_runtime"
	bw6_761 "github.com/ingonyama-zk/icicle/v2/wrappers/golang/curves/bw6761"
)

type PoseidonHandler = C.struct_PoseidonInst
type Poseidon struct {
	width  uint32
	handle *PoseidonHandler
}

func Create(arity uint32, alpha uint32, fullRoundsHalf uint32, partialRounds uint32, scalars core.HostOrDeviceSlice, mdsMatrix core.HostOrDeviceSlice, nonSparseMatrix core.HostOrDeviceSlice, sparseMatrices core.HostOrDeviceSlice, domainTag bw6_761.ScalarField, ctx *cr.DeviceContext) (*Poseidon, core.IcicleError) {
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
	__ret := C.bw6_761_poseidon_create_cuda(&poseidon, cArity, cAlpha, cFullRoundsHalf, cPartialRounds, cScalars, cMdsMatrix, cNonSparseMatrix, cSparseMatrices, cDomainTag, cCtx)
	err := core.FromCudaError((cr.CudaError)(__ret))
	if err.IcicleErrorCode != core.IcicleSuccess {
		return nil, err
	}
	p := Poseidon{handle: poseidon, width: arity + 1}
	runtime.SetFinalizer(ctx, p.Delete)
	return &p, err
}

func Load(arity uint32, ctx *cr.DeviceContext) (*Poseidon, core.IcicleError) {
	var poseidon *PoseidonHandler
	cArity := (C.uint)(arity)
	cCtx := (*C.DeviceContext)(unsafe.Pointer(ctx))
	__ret := C.bw6_761_poseidon_load_cuda(&poseidon, cArity, cCtx)
	err := core.FromCudaError((cr.CudaError)(__ret))
	if err.IcicleErrorCode != core.IcicleSuccess {
		return nil, err
	}
	p := Poseidon{handle: poseidon, width: arity + 1}
	runtime.SetFinalizer(ctx, p.Delete)
	return &p, err
}

func (poseidon *Poseidon) AbsorbMany(inputs core.HostOrDeviceSlice, states core.DeviceSlice, numberOfStates uint32, inputBlockLen uint32, cfg *core.SpongeConfig) core.IcicleError {
	core.SpongeInputCheck(inputs, numberOfStates, inputBlockLen, cfg.InputRate, &cfg.Ctx)
	core.SpongeStatesCheck(states, numberOfStates, poseidon.width, &cfg.Ctx)

	cInputs := (*C.scalar_t)(inputs.AsUnsafePointer())
	cStates := (*C.scalar_t)(states.AsUnsafePointer())
	cNumberOfStates := (C.uint)(numberOfStates)
	cInputBlockLen := (C.uint)(inputBlockLen)
	cCfg := (*C.SpongeConfig)(unsafe.Pointer(cfg))
	__ret := C.bw6_761_poseidon_absorb_many_cuda(poseidon.handle, cInputs, cStates, cNumberOfStates, cInputBlockLen, cCfg)
	err := (cr.CudaError)(__ret)
	return core.FromCudaError(err)
}

func (poseidon *Poseidon) SqueezeMany(states core.DeviceSlice, output core.HostOrDeviceSlice, numberOfStates uint32, outputLen uint32, cfg *core.SpongeConfig) core.IcicleError {
	core.SpongeOutputsCheck(output, numberOfStates, outputLen, poseidon.width, false, &cfg.Ctx)
	core.SpongeStatesCheck(states, numberOfStates, poseidon.width, &cfg.Ctx)

	cStates := (*C.scalar_t)(states.AsUnsafePointer())
	cOutput := (*C.scalar_t)(output.AsUnsafePointer())
	cNumberOfStates := (C.uint)(numberOfStates)
	cOutputLen := (C.uint)(outputLen)
	cCfg := (*C.SpongeConfig)(unsafe.Pointer(cfg))
	__ret := C.bw6_761_poseidon_absorb_many_cuda(poseidon.handle, cStates, cOutput, cNumberOfStates, cOutputLen, cCfg)
	err := (cr.CudaError)(__ret)
	return core.FromCudaError(err)
}

func (poseidon *Poseidon) HashMany(inputs core.HostOrDeviceSlice, output core.HostOrDeviceSlice, numberOfStates uint32, inputBlockLen uint32, outputLen uint32, cfg *core.SpongeConfig) core.IcicleError {
	core.SpongeInputCheck(inputs, numberOfStates, inputBlockLen, cfg.InputRate, &cfg.Ctx)
	core.SpongeOutputsCheck(output, numberOfStates, outputLen, poseidon.width, false, &cfg.Ctx)

	cInputs := (*C.scalar_t)(inputs.AsUnsafePointer())
	cOutput := (*C.scalar_t)(output.AsUnsafePointer())
	cNumberOfStates := (C.uint)(numberOfStates)
	cInputBlockLen := (C.uint)(inputBlockLen)
	cOutputLen := (C.uint)(outputLen)
	cCfg := (*C.SpongeConfig)(unsafe.Pointer(cfg))
	__ret := C.bw6_761_poseidon_hash_many_cuda(poseidon.handle, cInputs, cOutput, cNumberOfStates, cInputBlockLen, cOutputLen, cCfg)
	err := (cr.CudaError)(__ret)
	return core.FromCudaError(err)
}

func (poseidon *Poseidon) Delete(ctx *cr.DeviceContext) core.IcicleError {
	cCtx := (*C.DeviceContext)(unsafe.Pointer(ctx))
	__ret := C.bw6_761_poseidon_delete_cuda(poseidon.handle, cCtx)
	err := (cr.CudaError)(__ret)
	return core.FromCudaError(err)
}

func (poseidon *Poseidon) GetDefaultSpongeConfig() core.SpongeConfig {
	cfg := core.GetDefaultSpongeConfig()
	cfg.InputRate = poseidon.width - 1
	cfg.OutputRate = poseidon.width
	return cfg
}
