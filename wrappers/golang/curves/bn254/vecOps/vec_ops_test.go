package vecOps

import (
	"testing"

	"github.com/ingonyama-zk/icicle/wrappers/golang/core"
	cr "github.com/ingonyama-zk/icicle/wrappers/golang/cuda_runtime"
	bn254 "github.com/ingonyama-zk/icicle/wrappers/golang/curves/bn254"
	"github.com/stretchr/testify/assert"
)

func TestVecOps(t *testing.T) {
	testSize := 1 << 14

	a := bn254.GenerateScalars(testSize)
	b := bn254.GenerateScalars(testSize)
	var scalar bn254.ScalarField
	scalar.One()
	ones := core.HostSliceWithValue(scalar, testSize)

	out := make(core.HostSlice[bn254.ScalarField], testSize)
	out2 := make(core.HostSlice[bn254.ScalarField], testSize)
	out3 := make(core.HostSlice[bn254.ScalarField], testSize)

	cfg := core.DefaultVecOpsConfig()

	VecOp(a, b, out, cfg, core.Add)
	VecOp(out, b, out2, cfg, core.Sub)

	assert.Equal(t, a, out2)

	VecOp(a, ones, out3, cfg, core.Mul)

	assert.Equal(t, a, out3)
}

func TestTranspose(t *testing.T) {
	rowSize := 1 << 6
	columnSize := 1 << 8
	onDevice := false
	isAsync := false

	matrix := bn254.GenerateScalars(rowSize * columnSize)

	out := make(core.HostSlice[bn254.ScalarField], rowSize*columnSize)
	out2 := make(core.HostSlice[bn254.ScalarField], rowSize*columnSize)

	ctx, _ := cr.GetDefaultDeviceContext()

	TransposeMatrix(matrix, out, columnSize, rowSize, ctx, onDevice, isAsync)
	TransposeMatrix(out, out2, rowSize, columnSize, ctx, onDevice, isAsync)

	assert.Equal(t, matrix, out2)

	var dMatrix, dOut, dOut2 core.DeviceSlice
	onDevice = true

	matrix.CopyToDevice(&dMatrix, true)
	dOut.Malloc(columnSize*rowSize*matrix.SizeOfElement(), matrix.SizeOfElement())
	dOut2.Malloc(columnSize*rowSize*matrix.SizeOfElement(), matrix.SizeOfElement())

	TransposeMatrix(dMatrix, dOut, columnSize, rowSize, ctx, onDevice, isAsync)
	TransposeMatrix(dOut, dOut2, rowSize, columnSize, ctx, onDevice, isAsync)
	output := make(core.HostSlice[bn254.ScalarField], rowSize*columnSize)
	output.CopyFromDevice(&dOut2)

	assert.Equal(t, matrix, output)
}
