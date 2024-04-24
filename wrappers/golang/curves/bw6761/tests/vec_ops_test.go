package tests

import (
	"testing"

	"github.com/ingonyama-zk/icicle/v2/wrappers/golang/core"
	cr "github.com/ingonyama-zk/icicle/v2/wrappers/golang/cuda_runtime"
	bw6_761 "github.com/ingonyama-zk/icicle/v2/wrappers/golang/curves/bw6761"
	"github.com/ingonyama-zk/icicle/v2/wrappers/golang/curves/bw6761/vecOps"
	"github.com/stretchr/testify/assert"
)

func TestBw6_761VecOps(t *testing.T) {
	testSize := 1 << 14

	a := bw6_761.GenerateScalars(testSize)
	b := bw6_761.GenerateScalars(testSize)
	var scalar bw6_761.ScalarField
	scalar.One()
	ones := core.HostSliceWithValue(scalar, testSize)

	out := make(core.HostSlice[bw6_761.ScalarField], testSize)
	out2 := make(core.HostSlice[bw6_761.ScalarField], testSize)
	out3 := make(core.HostSlice[bw6_761.ScalarField], testSize)

	cfg := core.DefaultVecOpsConfig()

	vecOps.VecOp(a, b, out, cfg, core.Add)
	vecOps.VecOp(out, b, out2, cfg, core.Sub)

	assert.Equal(t, a, out2)

	vecOps.VecOp(a, ones, out3, cfg, core.Mul)

	assert.Equal(t, a, out3)
}

func TestBw6_761Transpose(t *testing.T) {
	rowSize := 1 << 6
	columnSize := 1 << 8
	onDevice := false
	isAsync := false

	matrix := bw6_761.GenerateScalars(rowSize * columnSize)

	out := make(core.HostSlice[bw6_761.ScalarField], rowSize*columnSize)
	out2 := make(core.HostSlice[bw6_761.ScalarField], rowSize*columnSize)

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
	output := make(core.HostSlice[bw6_761.ScalarField], rowSize*columnSize)
	output.CopyFromDevice(&dOut2)

	assert.Equal(t, matrix, output)
}
