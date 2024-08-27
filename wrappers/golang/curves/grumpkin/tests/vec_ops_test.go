package tests

import (
	"testing"

	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/core"
	grumpkin "github.com/ingonyama-zk/icicle/v3/wrappers/golang/curves/grumpkin"
	"github.com/ingonyama-zk/icicle/v3/wrappers/golang/curves/grumpkin/vecOps"
	"github.com/stretchr/testify/assert"
)

func TestGrumpkinVecOps(t *testing.T) {
	testSize := 1 << 14

	a := grumpkin.GenerateScalars(testSize)
	b := grumpkin.GenerateScalars(testSize)
	var scalar grumpkin.ScalarField
	scalar.One()
	ones := core.HostSliceWithValue(scalar, testSize)

	out := make(core.HostSlice[grumpkin.ScalarField], testSize)
	out2 := make(core.HostSlice[grumpkin.ScalarField], testSize)
	out3 := make(core.HostSlice[grumpkin.ScalarField], testSize)

	cfg := core.DefaultVecOpsConfig()

	vecOps.VecOp(a, b, out, cfg, core.Add)
	vecOps.VecOp(out, b, out2, cfg, core.Sub)

	assert.Equal(t, a, out2)

	vecOps.VecOp(a, ones, out3, cfg, core.Mul)

	assert.Equal(t, a, out3)
}

func TestGrumpkinTranspose(t *testing.T) {
	rowSize := 1 << 6
	columnSize := 1 << 8

	matrix := grumpkin.GenerateScalars(rowSize * columnSize)

	out := make(core.HostSlice[grumpkin.ScalarField], rowSize*columnSize)
	out2 := make(core.HostSlice[grumpkin.ScalarField], rowSize*columnSize)

	cfg := core.DefaultVecOpsConfig()

	vecOps.TransposeMatrix(matrix, out, columnSize, rowSize, cfg)
	vecOps.TransposeMatrix(out, out2, rowSize, columnSize, cfg)

	assert.Equal(t, matrix, out2)

	var dMatrix, dOut, dOut2 core.DeviceSlice

	matrix.CopyToDevice(&dMatrix, true)
	dOut.Malloc(columnSize*rowSize*matrix.SizeOfElement(), matrix.SizeOfElement())
	dOut2.Malloc(columnSize*rowSize*matrix.SizeOfElement(), matrix.SizeOfElement())

	vecOps.TransposeMatrix(dMatrix, dOut, columnSize, rowSize, cfg)
	vecOps.TransposeMatrix(dOut, dOut2, rowSize, columnSize, cfg)
	output := make(core.HostSlice[grumpkin.ScalarField], rowSize*columnSize)
	output.CopyFromDevice(&dOut2)

	assert.Equal(t, matrix, output)
}
