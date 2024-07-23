package tests

import (
	"testing"

	core "github.com/ingonyama-zk/icicle/v2/wrappers/golang_v3/core"
	cr "github.com/ingonyama-zk/icicle/v2/wrappers/golang_v3/cuda_runtime"
	bls12_381 "github.com/ingonyama-zk/icicle/v2/wrappers/golang_v3/curves/bls12381"
	poseidon "github.com/ingonyama-zk/icicle/v2/wrappers/golang_v3/curves/bls12381/poseidon"

	"fmt"

	"github.com/stretchr/testify/assert"
)

func formatOutput(x bls12_381.ScalarField) string {
	r := x.GetLimbs()
	return fmt.Sprintf("%08x%08x%08x%08x%08x%08x%08x%08x", r[7], r[6], r[5], r[4], r[3], r[2], r[1], r[0])
}

func TestPoseidon(t *testing.T) {

	arity := 2
	numberOfStates := 1

	cfg := poseidon.GetDefaultPoseidonConfig()
	cfg.IsAsync = true
	stream, _ := cr.CreateStream()
	cfg.Ctx.Stream = &stream

	var constants core.PoseidonConstants[bls12_381.ScalarField]

	poseidon.InitOptimizedPoseidonConstantsCuda(arity, cfg.Ctx, &constants) //generate constants

	scalars := bls12_381.GenerateScalars(numberOfStates * arity)
	scalars[0] = scalars[0].Zero()
	scalars[1] = scalars[0].Zero()

	scalarsCopy := core.HostSliceFromElements(scalars[:numberOfStates*arity])

	var deviceInput core.DeviceSlice
	scalarsCopy.CopyToDeviceAsync(&deviceInput, stream, true)
	var deviceOutput core.DeviceSlice
	deviceOutput.MallocAsync(numberOfStates*scalarsCopy.SizeOfElement(), scalarsCopy.SizeOfElement(), stream)

	poseidon.PoseidonHash(deviceInput, deviceOutput, numberOfStates, &cfg, &constants) //run Hash function

	output := make(core.HostSlice[bls12_381.ScalarField], numberOfStates)
	output.CopyFromDeviceAsync(&deviceOutput, stream)

	expectedString := "48fe0b1331196f6cdb33a7c6e5af61b76fd388e1ef1d3d418be5147f0e4613d4" //This result is from https://github.com/triplewz/poseidon
	outputString := formatOutput(output[0])

	assert.Equal(t, outputString, expectedString, "Poseidon hash does not match expected result")

}
