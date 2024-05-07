package tests

import (
	"fmt"
	"testing"

	core "github.com/ingonyama-zk/icicle/v2/wrappers/golang/core"
	cr "github.com/ingonyama-zk/icicle/v2/wrappers/golang/cuda_runtime"
	bls12_377 "github.com/ingonyama-zk/icicle/v2/wrappers/golang/curves/bls12377"
	poseidon "github.com/ingonyama-zk/icicle/v2/wrappers/golang/curves/bls12377/poseidon"
	//"github.com/iden3/go-iden3-crypto/poseidon"
)

func displayConst(con core.PoseidonConstants[bls12_377.ScalarField]) {

	fmt.Println("arity: ", con.Arity)
	fmt.Println("partial round", con.PartialRounds)
	fmt.Println("full round half", con.FullRoundsHalf)
	fmt.Println("round constants", con.RoundConstants)
	fmt.Println("mds matrix", con.MdsMatrix)
	fmt.Println("non sparse matrix", con.NonSparseMatrix)
	fmt.Println("sparse matrices", con.SparseMatrices)
	fmt.Println("domaintag", con.DomainTag)

}

func TestPoseidon(t *testing.T) {
	arity := 8

	//testSize := 1 << (largestTestSize - 15)
	//scalars := bls12_377.GenerateScalars(testSize * arity)
	//scalarsCopy := core.HostSliceFromElements[bls12_377.ScalarField](scalars[:testSize])
	numberOfStates := 1024

	cfg := poseidon.GetDefaultPoseidonConfig()

	var constants core.PoseidonConstants[bls12_377.ScalarField]

	displayConst(constants)
	ctx, _ := cr.GetDefaultDeviceContext()

	poseidon.InitOptimizedPoseidonConstantsCuda(arity, ctx, &constants)
	cr.SynchronizeStream(cfg.Ctx.Stream)
	fmt.Println("___________________________________")
	displayConst(constants)

	// input := make(core.HostSlice[bls12_377.ScalarField], numberOfStates * arity)
	// output := make(core.HostSlice[bls12_377.ScalarField], numberOfStates)
	// poseidon.PoseidonHash[bls12_377.ScalarField](input, output, numberOfStates, arity, &cfg, &constants)

	// cfg.IsAsync = false
	// cfg.InputIsAState = false

	scalars := bls12_377.GenerateScalars(numberOfStates * arity)
	for i := 0; i < numberOfStates*arity; i++ {
		scalars[i] = scalars[0].One()
	}
	scalarsCopy := core.HostSliceFromElements(scalars[:numberOfStates*arity])
	stream, _ := cr.CreateStream()
	var deviceInput core.DeviceSlice
	scalarsCopy.CopyToDeviceAsync(&deviceInput, stream, true)
	var deviceOutput core.DeviceSlice
	deviceOutput.MallocAsync(numberOfStates*scalarsCopy.SizeOfElement(), scalarsCopy.SizeOfElement(), stream)

	poseidon.PoseidonHash(deviceInput, deviceOutput, numberOfStates, arity, &cfg, &constants)

	output := make(core.HostSlice[bls12_377.ScalarField], numberOfStates)
	output.CopyFromDeviceAsync(&deviceOutput, stream)

}
