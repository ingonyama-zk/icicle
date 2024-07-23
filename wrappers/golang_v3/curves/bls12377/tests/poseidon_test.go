package tests

import (
	"testing"

	core "github.com/ingonyama-zk/icicle/v2/wrappers/golang_v3/core"
	cr "github.com/ingonyama-zk/icicle/v2/wrappers/golang_v3/cuda_runtime"
	bls12_377 "github.com/ingonyama-zk/icicle/v2/wrappers/golang_v3/curves/bls12377"
	poseidon "github.com/ingonyama-zk/icicle/v2/wrappers/golang_v3/curves/bls12377/poseidon"
)

func TestPoseidon(t *testing.T) {

	arity := 2
	numberOfStates := 1

	cfg := poseidon.GetDefaultPoseidonConfig()
	cfg.IsAsync = true
	stream, _ := cr.CreateStream()
	cfg.Ctx.Stream = &stream

	var constants core.PoseidonConstants[bls12_377.ScalarField]

	poseidon.InitOptimizedPoseidonConstantsCuda(arity, cfg.Ctx, &constants) //generate constants

	scalars := bls12_377.GenerateScalars(numberOfStates * arity)
	scalars[0] = scalars[0].Zero()
	scalars[1] = scalars[0].Zero()

	scalarsCopy := core.HostSliceFromElements(scalars[:numberOfStates*arity])

	var deviceInput core.DeviceSlice
	scalarsCopy.CopyToDeviceAsync(&deviceInput, stream, true)
	var deviceOutput core.DeviceSlice
	deviceOutput.MallocAsync(numberOfStates*scalarsCopy.SizeOfElement(), scalarsCopy.SizeOfElement(), stream)

	poseidon.PoseidonHash(deviceInput, deviceOutput, numberOfStates, &cfg, &constants) //run Hash function

	output := make(core.HostSlice[bls12_377.ScalarField], numberOfStates)
	output.CopyFromDeviceAsync(&deviceOutput, stream)

}
