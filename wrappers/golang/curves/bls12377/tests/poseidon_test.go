package tests

import (
	"testing"

	core "github.com/ingonyama-zk/icicle/v2/wrappers/golang/core"
	cr "github.com/ingonyama-zk/icicle/v2/wrappers/golang/cuda_runtime"
	bls12_377 "github.com/ingonyama-zk/icicle/v2/wrappers/golang/curves/bls12377"
	poseidon "github.com/ingonyama-zk/icicle/v2/wrappers/golang/curves/bls12377/poseidon"
)

func TestPoseidon(t *testing.T) {

	arity := 2
	numberOfStates := 1

	cfg := poseidon.GetDefaultPoseidonConfig()

	var constants core.PoseidonConstants[bls12_377.ScalarField]
	ctx, _ := cr.GetDefaultDeviceContext()

	poseidon.InitOptimizedPoseidonConstantsCuda(arity, ctx, &constants) //generate constants

	scalars := bls12_377.GenerateScalars(numberOfStates * arity)
	scalars[0] = scalars[0].Zero()
	scalars[1] = scalars[0].Zero()

	scalarsCopy := core.HostSliceFromElements(scalars[:numberOfStates*arity])

	stream, _ := cr.CreateStream()

	var deviceInput core.DeviceSlice
	scalarsCopy.CopyToDeviceAsync(&deviceInput, stream, true)
	var deviceOutput core.DeviceSlice
	deviceOutput.MallocAsync(numberOfStates*scalarsCopy.SizeOfElement(), scalarsCopy.SizeOfElement(), stream)

	poseidon.PoseidonHash(deviceInput, deviceOutput, numberOfStates, arity, &cfg, &constants) //run Hash function

	output := make(core.HostSlice[bls12_377.ScalarField], numberOfStates)
	output.CopyFromDeviceAsync(&deviceOutput, stream)

}
