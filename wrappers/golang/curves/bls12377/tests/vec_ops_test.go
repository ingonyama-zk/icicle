package tests

import (
	"testing"

	"github.com/ingonyama-zk/icicle/v2/wrappers/golang/core"
	cr "github.com/ingonyama-zk/icicle/v2/wrappers/golang/cuda_runtime"
	bls12_377 "github.com/ingonyama-zk/icicle/v2/wrappers/golang/curves/bls12377"
	"github.com/ingonyama-zk/icicle/v2/wrappers/golang/curves/bls12377/vecOps"
	"github.com/stretchr/testify/assert"
)

func TestBls12_377VecOps(t *testing.T) {
	testSize := 1 << 14

	a := bls12_377.GenerateScalars(testSize)
	b := bls12_377.GenerateScalars(testSize)
	var scalar bls12_377.ScalarField
	scalar.One()
	ones := core.HostSliceWithValue(scalar, testSize)

	out := make(core.HostSlice[bls12_377.ScalarField], testSize)
	out2 := make(core.HostSlice[bls12_377.ScalarField], testSize)
	out3 := make(core.HostSlice[bls12_377.ScalarField], testSize)

	cfg := core.DefaultVecOpsConfig()

	vecOps.VecOp(a, b, out, cfg, core.Add)
	vecOps.VecOp(out, b, out2, cfg, core.Sub)

	assert.Equal(t, a, out2)

	vecOps.VecOp(a, ones, out3, cfg, core.Mul)

	assert.Equal(t, a, out3)
}

func TestBls12_377Transpose(t *testing.T) {
	rowSize := 1 << 6
	columnSize := 1 << 8
	onDevice := false
	isAsync := false

	matrix := bls12_377.GenerateScalars(rowSize * columnSize)

	out := make(core.HostSlice[bls12_377.ScalarField], rowSize*columnSize)
	out2 := make(core.HostSlice[bls12_377.ScalarField], rowSize*columnSize)

	ctx, _ := cr.GetDefaultDeviceContext()

	vecOps.TransposeMatrix(matrix, out, columnSize, rowSize, ctx, onDevice, isAsync)
	vecOps.TransposeMatrix(out, out2, rowSize, columnSize, ctx, onDevice, isAsync)

	assert.Equal(t, matrix, out2)

	var dMatrix, dOut, dOut2 core.DeviceSlice
	onDevice = true

	matrix.CopyToDevice(&dMatrix, true)
	dOut.Malloc(columnSize*rowSize*matrix.SizeOfElement(), matrix.SizeOfElement())
	dOut2.Malloc(columnSize*rowSize*matrix.SizeOfElement(), matrix.SizeOfElement())

	vecOps.TransposeMatrix(dMatrix, dOut, columnSize, rowSize, ctx, onDevice, isAsync)
	vecOps.TransposeMatrix(dOut, dOut2, rowSize, columnSize, ctx, onDevice, isAsync)
	output := make(core.HostSlice[bls12_377.ScalarField], rowSize*columnSize)
	output.CopyFromDevice(&dOut2)

	assert.Equal(t, matrix, output)
}
